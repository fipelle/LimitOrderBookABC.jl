mutable struct ParticleSystem

    # Specifics
    markov_order::Int64
    
    # Dimensions
    num_particles::Int64

    # Densities
    priors::Vector{ContinuousDistribution}
    likelihood::Function
    gradient::Function
    
    # Particles and weights
    particles::Matrix{Float64}
    weights::Matrix{Float64}
    particles_history::Vector{Matrix{Float64}}
    weights_history::Vector{Matrix{Float64}}
end
