function mcmc_chain(mcmc_kernel::AbstractMCMCKernel{Val{true}},target,x,state,n_samples::Int)
  n_accepts = 0

  ref_lp, ref_grad = LD.logdensity_and_gradient(target,x)
  T = eltype(x)
  samples = Vector{Vector{T}}(undef,n_samples)
  lps = Vector{typeof(ref_lp)}(undef,n_samples)
  gradlps = Vector{Vector{T}}(undef,n_samples)
  α = Vector{T}(undef,n_samples-1)

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
  α = Vector{T}(undef,n_samples-1)

  samples[1] = x
  lps[1] = ref_lp

  for i in 1:n_samples-1
    samples[i+1],lps[i+1],acc,α[i],state = mcmc_kernel(target,samples[i],lps[i],state)
    n_accepts += acc
  end

  return (;n_accepts,samples,lps,state,α)
end

