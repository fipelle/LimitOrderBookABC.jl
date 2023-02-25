"""
    ParticleSystem(...)

`ParticleSystem` type. The particles are structured to have rows equal to the number of parameters.
"""
mutable struct ParticleSystem

    # Specifics
    markov_order      :: Int64
    
    # Dimensions
    num_parameters    :: Int64
    num_particles     :: Int64
    
    # Densities
    priors            :: Vector{Distribution{Univariate, Continuous}}
    log_objective     :: Function
    log_gradient      :: Function
    
    # Particles and weights
    particles         :: Matrix{Float64}
    log_weights       :: Vector{Float64}
    weights           :: Vector{Float64}
    particles_history :: Vector{Matrix{Float64}}
    weights_history   :: Vector{Vector{Float64}}
end
