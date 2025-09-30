# Resampling methods adapted from AdvancedPS.jl

using Random, Distributions

abstract type AbstractResampler end

function (::AbstractResampler)(w::AbstractVector,num_particles=length(w)) end

struct MultinomialResampler <: AbstractResampler end

function (::MultinomialResampler)(
    w::AbstractVector{<:Real}, num_particles::Integer=length(w)
)
    return rand(Distributions.sampler(Distributions.Categorical(w)), num_particles)
end

struct ResidualResampler <: AbstractResampler end

function (::ResidualResampler)(
    w::AbstractVector{<:Real},
    num_particles::Integer=length(w),
)
    # Pre-allocate array for resampled particles
    indices = Vector{Int}(undef, num_particles)

    # deterministic assignment
    residuals = similar(w)
    i = 1
    @inbounds for j in 1:length(w)
        x = num_particles * w[j]
        floor_x = floor(Int, x)
        for k in 1:floor_x
            indices[i] = j
            i += 1
        end
        residuals[j] = x - floor_x
    end

    # sampling from residuals
    if i <= num_particles
        residuals ./= sum(residuals)
        rand!(Distributions.Categorical(residuals), view(indices, i:num_particles))
    end

    return indices
end

struct StratifiedResampler <: AbstractResampler end

function (::StratifiedResampler)(
    weights::AbstractVector{<:Real}, n::Integer=length(weights)
)
    # check input
    m = length(weights)
    m > 0 || error("weight vector is empty")

    # pre-calculations
    @inbounds v = n * weights[1]

    # generate all samples
    samples = Array{Int}(undef, n)
    sample = 1
    @inbounds for i in 1:n
        # sample next `u` (scaled by `n`)
        u = oftype(v, i - 1 + rand())

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
    end

    return samples
end

struct SystematicResampler <: AbstractResampler end

function (::SystematicResampler)(
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


