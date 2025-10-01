"""
  Number that has some metadata, but promotes to val
  so that if an operation is performed the metadata is dropped
"""
struct MetaNumber{T<:Real,I} <: Real
  val::T
  info::I
end

Base.promote(a::MetaNumber,b,cs...) = Base.promote(a.val,b,cs...)
Base.promote(a,b::MetaNumber,cs...) = Base.promote(a,b.val,cs...)
Base.promote(a::MetaNumber,b::MetaNumber,cs...) = Base.promote(a.val,b.val,cs...)

Base.:(+)(x::T, y::T) where {T<:MetaNumber} = x.val+y.val
Base.:(*)(x::T, y::T) where {T<:MetaNumber} = x.val*y.val
Base.:(-)(x::T, y::T) where {T<:MetaNumber} = x.val-y.val
Base.:(/)(x::T, y::T) where {T<:MetaNumber} = x.val/y.val
Base.:(^)(x::T, y::T) where {T<:MetaNumber} = x.val^y.val

struct TemperedLogDensity{P,L}
  ref::P
  mul::L
  β::Float64
  dim::Int
end

LD.dimension(ℓ::TemperedLogDensity) = ℓ.dim

_order(_::LD.LogDensityOrder{K}) where K = K
function _lowest_capability(ℓ1,ℓs...)
  o = mapreduce(_order,(a,b) -> a < b,ℓs,init=_order(ℓ1))
  return LD.LogDensityOrder{o}()
end

LD.capabilities(ℓ::TemperedLogDensity) = _lowest_capability(ℓ.ref,ℓ.mul)

function LD.logdensity(ℓ::TemperedLogDensity,θ)
  mul = LD.logdensity(ℓ.mul,θ) 
  ref = LD.logdensity(ℓ.ref,θ) 
  MetaNumber(ℓ.β * mul + ref,(;mul,ref))
end

function LD.logdensity_and_gradient(ℓ::TemperedLogDensity,θ)
  ref,refgrad = LD.logdensity_and_gradient(ℓ.ref,θ)
  mul,mulgrad = LD.logdensity_and_gradient(ℓ.mul,θ)
  MetaNumber(ℓ.β * mul + ref,(;mul,mulgrad,ref,refgrad)), ℓ.β * mulgrad + refgrad
end

