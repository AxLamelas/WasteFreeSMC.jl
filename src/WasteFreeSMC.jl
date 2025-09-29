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

using Metadata


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
  mul = LD.logdensity(ℓ.mul,θ) 
  ref = LD.logdensity(ℓ.ref,θ) 
  attach_metadata((ℓ.β * mul + ref),(;mul,ref))
end

function LD.logdensity_and_gradient(ℓ::InterpolatingDensity,θ)
  ref,refgrad = LD.logdensity_and_gradient(ℓ.ref,θ)
  mul,mulgrad = LD.logdensity_and_gradient(ℓ.mul,θ)
  attach_metadata(ℓ.β * mul + ref,(;mul,mulgrad,ref,refgrad)), ℓ.β * mulgrad + refgrad
end

norm2(v::AbstractVector) = dot(v,v)

abstract type AbstractMCMCKernel{G <: Val} end

function (_::AbstractMCMCKernel{Val{false}})(target,x,logp_x,state) end
function (_::AbstractMCMCKernel{Val{true}})(target,x,logp_x,gradlogp_x,state) end

# Default kernel initialization
function init_kernel_state(_::AbstractMCMCKernel,x,scale,Σ) 
  cholesky(Symmetric(scale*Σ))
end

usesgrad(_::AbstractMCMCKernel{Val{V}}) where {V} = V

Base.@kwdef @concrete struct FisherMALA <: AbstractMCMCKernel{Val{true}}
  λ = 10.
  ρ = 0.015
  αstar = 0.574
end

function init_kernel_state(_::FisherMALA,x,scale,Σ)
  (;iter=1,σ2 = scale*tr(Σ),R = sqrt(Σ))
end

function (k::FisherMALA)(target,x,logp_x,gradlogp_x,state)
  @unpack iter,σ2,R = state
  @unpack λ,ρ,αstar = k

  σ2_rel = σ2/(sum(abs2,R)/length(x))

  u = randn(length(x))
  y = x + σ2_rel/2*R*(R'*gradlogp_x) + sqrt(σ2_rel)*R*u

  logp_y,gradlogp_y = LD.logdensity_and_gradient(target,y)
  
  α = min(1.,exp(logp_y-logp_x +
                  1/2*(x-y-σ2_rel/4*R*(R'*gradlogp_y))'*gradlogp_y -
                  1/2*(y-x-σ2_rel/4*R*(R'*gradlogp_x))'*gradlogp_x 
                  ))

  s = sqrt(α)*(gradlogp_y-gradlogp_x)

  if iter == 1
    ϕ = R'*s
    n = λ + ϕ'*ϕ
    r = 1/(1+sqrt(λ/n))
    nextR = 1/sqrt(λ) * (R - r/n * (R*ϕ)*ϕ')
  else
    ϕ = R'*s
    n = 1 + ϕ'*ϕ
    r = 1/(1+sqrt(1/n))
    nextR = R - r/n * (R*ϕ)*ϕ' 
  end

  nextσ2 = exp(log(σ2) + ρ*(α-αstar))

  next_state = (;iter = iter+1,σ2 = nextσ2, R = nextR)

  if rand() < α
    return y, logp_y, gradlogp_y, true, α, next_state
  end

  return x, logp_x, gradlogp_x, false, α, next_state
end

Base.@kwdef @concrete struct PathDelayedRejection <: AbstractMCMCKernel{Val{false}}
  proposal_dist = Normal()
  n_stages = 4
  factor = 0.25
end

_scaled_logpdf(dist,u,scale) = sum(logpdf(dist,u/scale)) - length(u)*log(scale)

function _scaled_logΔ(proposal_dist,logps,us,factor,seq)
  stage = length(seq)-1
  if stage == 1
    return logps[seq[1]] + sum(logpdf(proposal_dist,us[seq[2]]-us[seq[1]]))
  end

  q = _scaled_logpdf(proposal_dist,us[seq[end]]-us[seq[1]],factor^(stage-1))

  next_seq = seq[1:end-1]
  a = _scaled_logΔ(proposal_dist,logps,us,factor,next_seq)

  next_seq[1], next_seq[end] = next_seq[end], next_seq[1]

  b = _scaled_logΔ(proposal_dist,logps,us,factor,next_seq)

  return  q + logsubexp(a,b)
end

function _logα(proposal_dist,logps,us,factor,stage)
  seq = collect(1:stage+1)
  forward_Δ = _scaled_logΔ(proposal_dist,logps,us,factor,seq)
  seq[1],seq[end] = seq[end],seq[1]
  backward_Δ = _scaled_logΔ(proposal_dist,logps,us,factor,seq)

  return backward_Δ - forward_Δ
end

function (k::PathDelayedRejection)(target,x,logp_x,C::Cholesky)
  @unpack proposal_dist,factor,n_stages = k
  n = length(x)

  us = Vector{Vector{eltype(x)}}(undef,n_stages+1)
  logps = Vector{typeof(logp_x)}(undef,n_stages+1)
  us[1] = zeros(n)
  logps[1] = logp_x
  
  local α
  for i in 1:n_stages
    us[i+1] = factor^(i-1) * rand(proposal_dist,n)
    y = x + C.L*us[i+1]
    logps[i+1] =  LD.logdensity(target,y)

    α = min(1.,exp(_logα(
      proposal_dist,
      logps,us,
      factor,i
    )))

    if rand() < α
      return y,logps[i+1], true, α,C
    end
  end

  return x, logp_x, false, α, C
end

Base.@kwdef @concrete struct RWMH <: AbstractMCMCKernel{Val{false}}
  proposal_dist = Normal()
end

function (k::RWMH)(target,x,logp_x,C::Cholesky)
  @unpack proposal_dist = k
  u = rand(proposal_dist,length(x))
  y = x + C.L*u
  logp_y = LD.logdensity(target,y)

  α = min(1.,exp.(logp_y + sum(logpdf(proposal_dist,-u)) - logp_x - sum(logpdf(proposal_dist,u))))
  if rand() < α
    return y, logp_y, true, α, C
  end

  return x, logp_x, false, α, C
end


function mcmc_chain(mcmc_kernel::AbstractMCMCKernel{Val{true}},target,x,state,n_samples::Int)
  n_accepts = 0

  ref_lp, ref_grad = LD.logdensity_and_gradient(target,x)
  T = eltype(x)
  samples = Vector{Vector{T}}(undef,n_samples)
  lps = Vector{typeof(ref_lp)}(undef,n_samples)
  gradlps = Vector{Vector{T}}(undef,n_samples)
  α = Vector{typeof(ref_lp)}(undef,n_samples-1)

  samples[1] = x
  lps[1] = ref_lp
  gradlps[1] = ref_grad

  for i in 1:n_samples-1
    samples[i+1],lps[i+1],gradlps[i+1],acc,α[i],state =
      mcmc_kernel(target,samples[i],lps[i],gradlps[i],state)
    n_accepts += acc
  end


  return (;n_accepts,samples,lps,state,α)
end

function mcmc_chain(mcmc_kernel::AbstractMCMCKernel{Val{false}},target,x,state,n_samples::Int)
  n_accepts = 0

  ref_lp = LD.logdensity(target,x)
  T = eltype(x)
  samples = Vector{Vector{T}}(undef,n_samples)
  lps = Vector{typeof(ref_lp)}(undef,n_samples)
  α = Vector{typeof(ref_lp)}(undef,n_samples-1)

  samples[1] = x
  lps[1] = ref_lp

  for i in 1:n_samples-1
    samples[i+1],lps[i+1],acc,α[i],state = mcmc_kernel(target,samples[i],lps[i],state)
    n_accepts += acc
  end

  return (;n_accepts,samples,lps,state,α)
end


mutable struct SMCState{T}
  iter::Int
  samples::Matrix{T}
  W::Vector{T}
  ℓ::Vector{T}
  ℓ_adjust::T
  scales::Vector{T}
  scale_weights::Vector{T}
  β::Float64
  acceptance_rate::Float64
  log_evidence::T
end

function SMCState(samples,W,ℓ,ℓ_adjust,scales,scale_weights)
  T = promote_type(eltype(samples),eltype(ℓ),eltype(scales),eltype(scale_weights))
  return SMCState{T}(0,samples,W,ℓ,ℓ_adjust,scales,scale_weights,0.,0.,zero(T))
end


"""
  _next_β(state::SMCstate, metric_target)

Compute the next value for `β` and the nominal weights `w` using bisection.
Uses the conditional effective sample size (https://www.jstor.org/stable/44861887) as a metric.
"""
function _next_β(state::SMCState,metric_target)
  low = state.β
  high = 2one(state.β)

  local x # Declare variables so they are visible outside the loop

  w = similar(state.ℓ)

  while (high - low) / ((high + low) / 2) > 1e-6 && high > eps()
    x = (high + low) / 2
    w .= exp.((x - state.β) .* state.ℓ)
    cess = sum(state.W[i]*w[i] for i in eachindex(w))^2/
      mean(state.W[i]*(w[i])^2 for i in eachindex(w))

    if cess == metric_target
      break
    end

    if cess < metric_target
      high = x # Reduce high
    else
      low = x # Increase low
    end
  end

  return min(1, x)
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

function resample_systematic(
    weights::AbstractVector{<:Real}, n::Integer=length(weights)
)
    # check input
    m = length(weights)
    m > 0 || error("weight vector is empty")

    # pre-calculations
    @inbounds v = n * weights[1]
    u = oftype(v, rand())

    # find all samples
    samples = Array{Int}(undef, n)
    sample = 1
    @inbounds for i in 1:n
        # as long as we have not found the next sample
        while v < u
            # increase and check the sample
            sample += 1
            sample > m &&
                error("sample could not be selected (are the weights normalized?)")

            # update the cumulative sum of weights (scaled by `n`)
            v += n * weights[sample]
        end

        # save the next sample
        samples[i] = sample

        # update `u`
        u += one(u)
    end

    return samples
end

function _default_sampler(ref_logdensity,mul_logdensity)
  lc = _lowest_capability(ref_logdensity,mul_logdensity)
  if lc isa LD.LogDensityOrder{0}()
    return PathDelayedRejection()
  end
  return FisherMALA()
end

abstract type AbstractCovEstimator end

function estimate_cov(_::AbstractCovEstimator,samples,weights) end

struct IdentityCov <: AbstractCovEstimator end

estimate_cov(_::IdentityCov,samples,weights,xs) = fill(Matrix(one(eltype(samples)) * I, size(samples,1),size(samples,1)),length(xs))

Base.@kwdef @concrete struct ParticleCov <: AbstractCovEstimator 
  method = LinearShrinkage(DiagonalUnequalVariance())
end

function estimate_cov(c::ParticleCov,samples,weights,xs) 
  fill(
    cov(
      c.method,samples,FrequencyWeights(weights),dims=2
    )
    ,length(xs))
end

Base.@kwdef @concrete struct KernelCov <: AbstractCovEstimator 
  max_samples = 1000
  γ = 0.05
end

function estimate_cov(c::KernelCov,samples,weights,xs)
  n_samples, ref_samples,wfun = if size(samples,2) > c.max_samples
    inds = resample_systematic(weights,c.max_samples)
    c.max_samples, samples[:,inds], i -> 1
  else
    n_samples = size(samples,2)
    n_samples,samples, i -> sqrt(n_samples * weights[i])
  end

  n_dims = size(samples,1)
  H = I - ones(n_samples,n_samples)/n_samples
  M = similar(H,n_dims,n_samples)

  refdists = pairwise(Euclidean(),ref_samples)
  lengthscale = median(refdists) / sqrt(2) + 1e-8

  # Covariance in kernel space from Kernel Adaptive Metropolis-Hastings
  xzdists = pairwise(SqEuclidean(),xs,eachcol(ref_samples)) 
  return map(eachindex(xs)) do i
    for (j,z) in enumerate(eachcol(ref_samples))
      @. M[:,j] = wfun(j) * 2/lengthscale^2 * exp(-0.5*xzdists[i,j]/lengthscale^2) * (z - xs[i])
    end
    Symmetric(c.γ*I + M * H * M')
  end
end

"""
  stabilized_map(f,x,map_func)

  Uses the `Base.map` infrastructure to infer the return type of the map, using 
  a type assertion to enforce it.
"""
function stabilized_map(f,x,map_func)
  gen = Base.Generator(f,x)
  T = Base.@default_eltype gen
  return map_func(identity,gen)::Vector{T}
end


function waste_free_smc(ref_logdensity,mul_logdensity,initial_samples;
                        mcmc_kernel::AbstractMCMCKernel = _default_sampler(ref_logdensity,mul_logdensity),
                        cov_estimator::AbstractCovEstimator = IdentityCov(),
                        # Should be much smaller than the number of samples
                        n_starting = _guess_n_starting(size(initial_samples,2)),
                        # Normalized weights of the samples according to the
                        # reference distribution
                        initial_weights = fill(1/size(initial_samples,2),size(initial_samples,2)),
                        # Reference scale
                        ref_cov_scale = 2.38^2/size(initial_samples,1),
                        # Search scales up to `ϵ` orders of magnitude lower than 
                        # the ref scale
                        ϵ = 6,
                        # Magnitude of the perturbation of the scale estimate
                        perturb_scale = 0.015,
                        α = 0.5,
                        map_func = map,
                        maxiter = 200,
                        callback=(_) -> false,
                        store_trace = true
                        )


  samples = copy(initial_samples)
  n_dims, n_samples = size(samples)

  loop_prog = ProgressUnknown(desc="Tempering:",showspeed=true,dt=1e-9)

  chain_length = div(n_samples, n_starting)

  ℓ = stabilized_map(eachcol(samples),map_func) do c
    LD.logdensity(mul_logdensity,c)
  end
  ℓ_adjust  = maximum(ℓ)
  ℓ .-= ℓ_adjust

  state = SMCState(
    samples,initial_weights,ℓ,ℓ_adjust,
    ref_cov_scale * 10 .^ (range(-ϵ,0,length=n_starting)),
    ones(n_starting)
  )
  trace = typeof(state)[]

  ProgressMeter.update!(loop_prog,0)
  while state.β < 1 && state.iter < maxiter
    # `state` contains information regarding the previous step in the sequence

    if store_trace
      push!(trace,deepcopy(state))
    end

    if callback(trace)
      @warn "Stopped by callback"
      return store_trace ? trace : state
    end


    # Determines the current distribution in the sequence
    new_β = _next_β(state,α*n_samples)

    # Resample 
    indices = resample_systematic(state.W,n_starting) 

    starting_x = [view(samples,:,i) for i in indices]
    cov_estimate = estimate_cov(cov_estimator, samples,state.W,starting_x)
    chains = stabilized_map(zip(starting_x,state.scales,cov_estimate),map_func) do (x,scale,Σ)
      interp_density = InterpolatingDensity(ref_logdensity,mul_logdensity,new_β,n_dims)
      kernel_state = init_kernel_state(mcmc_kernel,x,scale,Σ)
      mcmc_chain(mcmc_kernel,interp_density,x,kernel_state,chain_length)
    end

    # Update the state
    
    offset = 0
    for c in chains
      for j in 1:chain_length
        state.samples[:,offset+j] .= c.samples[j]
        state.ℓ[offset+j] = c.lps[j].mul 
      end
      offset += chain_length
    end
    state.ℓ_adjust = maximum(ℓ)
    state.ℓ .-= state.ℓ_adjust
    
    # New weights do no depend on the previous weights due to resampling
    # at every iteration
    state.W = exp.((new_β-state.β) .* state.ℓ)
    state.log_evidence += log(mean(state.W)) + (new_β - state.β) * ℓ_adjust 
    
    # Normalize weights
    state.W ./= sum(state.W)

    state.β = new_β

    # Average acceptance rate of the chains
    state.acceptance_rate = sum(c.n_accepts for c in chains) / ((chain_length-1)*n_starting)

    # Resample scale following 10.1214/13-BA814
    for j in 1:n_starting
      c = chains[j]
      Σ = cov_estimate[j]
      # Rao-Blackwellized estimator of the Expected squared jump distance 
      w = 0.
      for i in 1:chain_length-1
        δ = c.samples[i+1]-c.samples[i]
        w += c.α[i] * δ'*(Σ\δ)
      end
      w /= chain_length - 1
      state.scale_weights[j] = w
    end
    state.scale_weights ./= sum(state.scale_weights)
    scale_inds = resample_systematic(state.scale_weights)
    state.scales = state.scales[scale_inds]
    for i in eachindex(state.scales)
      # Instead of just the Mixture model from the paper
      # Do also a mixture with the initial uniform distribution
      # so that if the scale changes abruptly between steps 
      # the distribution of scale parameters is not stuck on the old scale
      if rand() < 0.5 + 0.5*state.β # it is also tempered
        state.scales[i] = exp(log(state.scales[i]) + perturb_scale*randn())
      else
        state.scales[i] = ref_cov_scale * 10 .^ (-ϵ*rand())
      end
    end
    state.iter += 1

    ProgressMeter.next!(loop_prog,
                    showvalues=[
                    ("β",state.β),
                    ("Maximum ℓ",state.ℓ_adjust),
                    ("Log evidence",state.log_evidence),
                    ("Acceptance rate",state.acceptance_rate)
                    ])

  end

  if store_trace
    push!(trace,state)
  end

  ProgressMeter.finish!(loop_prog)

  if !isone(state.β)
    @warn "Did not reach β=1 in the give limit of iterations"
  end

  return store_trace ? trace : state
end

end # module WasteFreeSMC
