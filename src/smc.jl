function smc(ref_logdensity,mul_logdensity,initial_samples;
                        mcmc_kernel::AbstractMCMCKernel = _default_sampler(ref_logdensity,mul_logdensity),
                        cov_estimator::AbstractCovEstimator = IdentityCov(),
                        resampler::AbstractResampler = ResidualResampler(),
                        α = 0.95,
                        resampling_α = 0.5,
                        mcmc_steps = 1,
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
                        map_func = map,
                        maxiter = 200,
                        callback=(_) -> false,
                        store_trace = true
                        )


  samples = copy(initial_samples)
  n_dims, n_samples = size(samples)

  loop_prog = ProgressUnknown(desc="Tempering:",showspeed=true,dt=1e-9)

  ℓ = stabilized_map(eachcol(samples),map_func) do c
    LD.logdensity(mul_logdensity,c)
  end
  ℓ_adjust  = maximum(ℓ)
  ℓ .-= ℓ_adjust

  state = SMCState(
    samples,initial_weights,ℓ,ℓ_adjust,
    ref_cov_scale * 10 .^ (range(-ϵ,0,length=n_samples)),
    ones(n_samples)
  )
  trace = typeof(state)[]

  indices_no_resampling = collect(1:n_samples)
  did_resampling = false

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

    ess = 1/sum(abs2,state.W)
    # Resample 
    indices = if ess < resampling_α*n_samples
      v = resampler(state.W) 
      state.W .= 1/n_samples
      did_resampling = true
      v
    else
      did_resampling = false
      indices_no_resampling
    end

    starting_x = [view(samples,:,i) for i in indices]
    cov_estimate = estimate_cov(cov_estimator, samples,state.W,starting_x)
    chains = stabilized_map(collect(zip(starting_x,state.scales,cov_estimate)),map_func) do (x,scale,Σ)
      interp_density = TemperedLogDensity(ref_logdensity,mul_logdensity,new_β,n_dims)
      kernel_state = init_kernel_state(mcmc_kernel,x,scale,Σ)
      mcmc_chain(mcmc_kernel,interp_density,x,kernel_state,mcmc_steps+1)
    end

    # Update the state
    for (i,c) in enumerate(chains)
      state.samples[:,i] .= c.samples[end]
      state.ℓ[i] = c.lps[end].info.mul 
    end
    state.ℓ_adjust = maximum(ℓ)
    state.ℓ .-= state.ℓ_adjust
    
    state.W .*= exp.((new_β-state.β) .* state.ℓ)
    state.log_evidence += log(sum(state.W)) + (new_β - state.β) * ℓ_adjust 
    
    # Normalize weights
    state.W ./= sum(state.W)

    state.β = new_β

    # Average acceptance rate of the chains
    state.acceptance_rate = sum(c.n_accepts for c in chains) / (mcmc_steps*n_samples)

    # Resample scale following 10.1214/13-BA814
    for j in 1:n_samples
      c = chains[j]
      Σ = cov_estimate[j]
      # Rao-Blackwellized estimator of the Expected squared jump distance 
      w = 0.
      for i in 1:mcmc_steps
        δ = c.samples[i+1]-c.samples[i]
        w += c.α[i] * δ'*(Σ\δ)
      end
      w /= mcmc_steps
      state.scale_weights[j] = w
    end
    state.scale_weights ./= sum(state.scale_weights)
    scale_inds = resampler(state.scale_weights)
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
                    ("Resampled?",did_resampling),
                    ("Maximum ℓ",state.ℓ_adjust),
                    ("Log evidence",state.log_evidence),
                    ("Acceptance rate",state.acceptance_rate),
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
