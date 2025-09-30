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

norm2(v::AbstractVector) = dot(v,v)

function _default_sampler(ref_logdensity,mul_logdensity)
  lc = _lowest_capability(ref_logdensity,mul_logdensity)
  if lc isa LD.LogDensityOrder{0}()
    return PathDelayedRejection()
  end
  return FisherMALA()
end

"""
  stabilized_map(f,x,map_func)

  Uses the `Base.map` infrastructure to infer the return type of the map, using 
  a type assertion to enforce it.
"""
function stabilized_map(f,x,map_func)
  gen = Base.Generator(f,x)
  T = Base.@default_eltype gen
  return map_func(f,x)::Vector{T}
end

