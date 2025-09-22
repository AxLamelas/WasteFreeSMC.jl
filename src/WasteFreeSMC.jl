module WasteFreeSMC

export waste_free_smc

using CovarianceEstimation
using StatsBase
using LogExpFunctions
using LinearAlgebra
using Distributions
using Random
using SimpleUnPack
using Distances
using Primes
using ProgressMeter
using ConcreteStructs


import LogDensityProblems as LD

struct InterpolatingDensity{P,L}
  ref::P
  mul::L
  β::Float64
  dim::Int
end

LD.dimension(ℓ::InterpolatingDensity) = ℓ.dim

_order(_::LD.LogDensityOrder{K}) where K = K
function _lowest_capability(ℓ1,ℓs...)
  o = mapreduce(_order,(a,b) -> a < b,ℓs,init=_order(ℓ1))
  return LD.LogDensityOrder{o}()
end

LD.capabilities(ℓ::InterpolatingDensity) = _lowest_capability(ℓ.ref,ℓ.mul)

function LD.logdensity(ℓ::InterpolatingDensity,θ)
  ℓ.β * LD.logdensity(ℓ.mul,θ) + LD.logdensity(ℓ.ref,θ) 
end

function LD.logdensity_and_gradient(ℓ::InterpolatingDensity,θ)
  ref,refgrad = LD.logdensity_and_gradient(ℓ.ref,θ)
  mul,mulgrad = LD.logdensity_and_gradient(ℓ.mul,θ)
  ℓ.β * mul + ref, ℓ.β * mulgrad + refgrad
end

norm2(v::AbstractVector) = dot(v,v)

abstract type AbstractMCMCKernel{G <: Val} end

function (_::AbstractMCMCKernel{Val{false}})(target,x,logp_x,state) end
function (_::AbstractMCMCKernel{Val{true}})(target,x,logp_x,gradlogp_x,state) end

target_acceptance_rate(_::AbstractMCMCKernel) = 0.237

function logpΔ(logps,bot=1,top=length(logps),rev=false)
    l,u = rev ? (top,bot) : (bot,top)
    if top-bot == 0
        return logps[l]
    elseif top-bot == 1
        return logsubexp(logps[l],logps[u])
    end
    return logsubexp(
        logpΔ(logps,bot,top-1,rev),
        logpΔ(logps,bot+1,top,!rev)
    )
end

function logα(forward_logps,reverse_logps,level)
    logpΔ(reverse_logps,1,level,false) - logpΔ(forward_logps,1,level,false)
end

function (k::SymmDelayedRejection)(target,x,target_x,C::Cholesky)
  @unpack levels,factor,proposal_dist = k
  forward_logps = Vector{eltype(x)}(undef,levels)
  backward_logps = Vector{eltype(x)}(undef,levels)
  us = Vector{Vector{eltype(x)}}(undef,levels)
  n = length(x)
  us[1] = rand(proposal_dist,n) 
  y = x + C.L * us[1]

  forward_logps[1] = target_x
  backward_logps[1] = target(y)

  α = min(1.,exp(logα(forward_logps,backward_logps,1)))

  if rand() < α
    return y, backward_logps[1], true
  end


  for l in 2:levels
    us[l] = rand(proposal_dist,n)
    y = x + factor^(l-1)*C.L * us[l]
    forward_logps[l] = backward_logps[1]
    backward_logps[1] = target(y)
    for j in 1:l-1
      rev_y = y - factor^(j-1)*C.L * us[j]
      backward_logps[j+1] = target(rev_y)
    end

    α = min(1.,exp(logα(forward_logps,backward_logps,l)))

    if rand() < α
      # Return false so that for the DelayedRejection the acceptance rate
      # calculated is the one at the first stage
      return y, backward_logps[1], false
    end
  end

  return x,target_x, false 
end

Base.@kwdef struct SymmRWMH <: AbstractMCMCKernel
  proposal_dist::ContinuousUnivariateDistribution = Normal()
end

function (k::SymmRWMH)(target,x,target_x,C::Cholesky)
  @unpack proposal_dist = k
  y = x + C.L*rand(proposal_dist,length(x))
  target_y = target(y)

  α = min(1.,exp.(target_y-target_x))
  if rand() < α
    return y, target_y, true
  end

  return x, target_x, false
end


function mcmc_chain(mcmc_kernel::AbstractMCMCKernel{Val{true}},target,x,state,n_samples::Int)
  n_accepts = 0

  ref_lp, ref_grad = LD.logdensity_and_gradient(target,x)
  T = eltype(x)
  samples = Vector{Vector{T}}(undef,n_samples)
  lps = Vector{typeof(ref_lp)}(undef,n_samples)
  gradlps = Vector{Vector{T}}(undef,n_samples)

  samples[1] = x
  lps[1] = ref_lp
  gradlps[1] = ref_grad

  for i in 1:n_samples-1
    samples[i+1],lps[i+1],gradlps[i+1],acc,state =
      mcmc_kernel(target,samples[i],lps[i],gradlps[i],state)
    n_accepts += acc
  end

  return (;n_accepts,samples,lps,state)
end

function mcmc_chain(mcmc_kernel::AbstractMCMCKernel{Val{false}},target,x,state,n_samples::Int)
  n_accepts = 0

  ref_lp = LD.logdensity(target,x)
  T = eltype(x)
  samples = Vector{Vector{T}}(undef,n_samples)
  lps = Vector{typeof(ref_lp)}(undef,n_samples)

  samples[1] = x
  lps[1] = ref_lp

  for i in 1:n_samples-1
    samples[i+1],lps[i+1],acc,state = mcmc_kernel(target,samples[i],lps[i],state)
    n_accepts += acc
  end

  return (;n_accepts,samples,lps,state)
end

"""
    _beta_and_weights(β, likelihood, target)

Compute the next value for `β` and the nominal weights `w` using bisection.
"""
function _beta_and_weights(β::Real, adjusted_likelihood::AbstractVector{<:Real}, target)
  low = β
  high = 2one(β)

  local x # Declare variables so they are visible outside the loop

  w = similar(adjusted_likelihood)

  while (high - low) / ((high + low) / 2) > 1e-6 && high > eps()
    x = (high + low) / 2
    w .= exp.((x - β) .* adjusted_likelihood)
    w ./= sum(w)
    ess = 1/sum(abs2,w)
    if ess == target
      break
    end

    if ess < target
      high = x # Reduce high
    else
      low = x # Increase low
    end
  end

  return min(1, x), w
end

function divisors(n)

    d = Int64[1]

    for (p,e) in factor(n)
        t = Int64[]
        r = 1

        for i in 1:e
            r *= p
            for u in d
                push!(t, u*r)
            end
        end

        append!(d, t)
    end

    return sort(d)
end

function _guess_n_starting(n_samples)
  estimate = 2log10(n_samples)^2
  best = 1
  diff = Inf
  for d in divisors(n_samples)
    if abs(d-estimate) < diff
      best = d
      diff = abs(d-estimate)
    end
  end

  return best
end

function waste_free_smc(ref_logdensity,mul_logdensity,initial_samples;
                        mcmc_kernel::AbstractMCMCKernel = SymmDelayedRejection(),
                        covariance_method = LinearShrinkage(DiagonalUnequalVariance()),
                        # Should be much smaller than the number of samples
                        n_starting = _guess_n_starting(size(initial_samples,2)),
                        target_acceptance_rate = 0.234,
                        α = 0.5,
                        init_scale = 2.38^2/size(initial_samples,1),
                        γ = 10.,
                        ϵ = 1e-8,
                        parallel_map = map,
                        maxiter = 200,
                        callback=(_) -> false,
                        )

  trace= NamedTuple[]
  β = 0.
  log_evidence = 0.
  acceptance_rate = 0.
  samples = copy(initial_samples)
  n_dims, n_samples = size(samples)

  loop_prog = ProgressUnknown(desc="Tempering:",showspeed=true,dt=1e-9)

  # For Kernel estimate of the covariance
  H = I - ones(n_samples,n_samples)/n_samples
  M = similar(H,n_dims,n_samples)
  cov_scale = init_scale

  chain_length = div(n_samples, n_starting)

  ℓ = parallel_map(eachcol(samples)) do c
    mul_logdensity(c)
  end
  ℓ_adjust  = maximum(ℓ)
  ℓ .-= ℓ_adjust

  ProgressMeter.update!(loop_prog,0)
  iter = 0
  while β < 1 && iter < maxiter
    iter += 1
    push!(trace,(;iter,samples=copy(samples),ℓ,ℓ_adjust,cov_scale,β,acceptance_rate,log_evidence))

    if callback(trace)
      @warn "Stopped by callback"
      return trace
    end

    new_β, w = _beta_and_weights(β,ℓ,α*n_samples)
    ProgressMeter.next!(loop_prog,showvalues=[("β",new_β),("Maximum ℓ",ℓ_adjust)])

    log_evidence += log(mean(w)) + (new_β - β) * ℓ_adjust 

    wn = w ./ sum(w)

    cw = FrequencyWeights(wn)
    indices = sample(1:n_samples,cw,n_starting,replace=true)

    # Covarience estimate with all samples
    Σ0 = cov(covariance_method,samples,cw,dims=2) 
    refdists = pairwise(Euclidean(),samples)
    lengthscale = median(refdists) / sqrt(2) + sqrt(ϵ)

    chains = let scale = cov_scale
      parallel_map([view(samples,:,i) for i in indices]) do x 
        interp_density = InterpolatingDensity(ref_logdensity,mul_logdensity,new_β,n_dims)
        # Kernel Adaptive Metropolis-Hastings
        xzdists = pairwise(SqEuclidean(),[x],eachcol(samples)) 
        k = @. exp(-0.5 * xzdists / lengthscale^2)
        for (j,z) in enumerate(eachcol(samples))
          @. M[:,j] = 2/lengthscale^2 * k[j] * (z - x)
        end
        K = scale * 0.5 * (Σ0 + M * H * M') + ϵ*I
        mcmc_chain(mcmc_kernel,interp_density,x,cholesky(Symmetric(K)),chain_length)
      end
    end

    # Setup for next iteration
    β = new_β
    offset = 0
    for c in chains
      s = view(samples,:,(offset+1):(offset+chain_length))
      copyto!(s,c.samples)
      # Assumes that calculating the prior is much faster than the likelihood
      # so it gets the valus from chain
      for j in 1:chain_length
        ℓ[offset+j] = (c.lps[j]-ref_logdensity(view(c.samples,:,j))) / β
      end
      offset += chain_length
    end

    acceptance_rate = sum(c.n_accepts for c in chains) / (n_starting * (chain_length-1))

    cov_scale = exp(log(cov_scale) + γ * (acceptance_rate - target_acceptance_rate))

    ℓ_adjust = maximum(ℓ)
    ℓ .-= ℓ_adjust

  end

  ProgressMeter.finish!(loop_prog)

  iter += 1
  push!(trace,(;iter,samples,ℓ,ℓ_adjust,cov_scale,β,acceptance_rate,log_evidence))

  if !isone(β)
    @warn "Did not reach β=1 in the give limit of iterations"
  end

  return trace
end

end # module WasteFreeSMC
