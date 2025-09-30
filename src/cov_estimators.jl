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

