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

struct MALA <: AbstractMCMCKernel{Val{true}} end

function (k::MALA)(target,x,logp_x,gradlogp_x,C::Cholesky)
  u = randn(length(x))
  y = x + 1/2*C.L*(C.L'*gradlogp_x) + C.L*u

  logp_y,gradlogp_y = LD.logdensity_and_gradient(target,y)
  
  α = min(1.,exp(logp_y-logp_x +
                  1/2*(x-y-1/4*C.L*(C.L'*gradlogp_y))'*gradlogp_y -
                  1/2*(y-x-1/4*C.L*(C.L'*gradlogp_x))'*gradlogp_x 
                  ))

  if rand() < α
    return y, logp_y, gradlogp_y, true, α, C
  end

  return x, logp_x, gradlogp_x, false, α, C
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


