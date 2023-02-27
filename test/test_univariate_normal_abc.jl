include("../src/StaticSMC.jl");
using Main.StaticSMC;
using Distances, Distributions, MessyTimeSeries, Random;

"""
    log_objective!(
        batch        :: AbstractArray{Float64}, 
        batch_length :: Int64, 
        parameters   :: AbstractVector{Float64}; 
        no_sim       :: Int64 = 1000
    )

Compute the log-objective.
"""
function log_objective!(
    batch           :: AbstractArray{Float64}, 
    batch_length    :: Int64, 
    parameters      :: AbstractVector{Float64},
    trial_tolerance :: Vector{Float64}, # this is updated in-place within the function
    system          :: ParticleSystem; 
    no_sim          :: Int64 = 1000
)
    
    # Retrieve current parameters configuration
    μ = parameters[1];
    σ² = get_bounded_logit(parameters[2], 0.0, 1000.0);
    
    # Compute summary statistics
    summary = 0.0; 
    distances = similar(system.tolerance);

    for i=1:no_sim

        # Simulate data
        batch_simulated = μ .+ sqrt(σ²)*randn(batch_length);

        # Compute distances
        distances[1] = euclidean(mean(batch), mean(batch_simulated));
        distances[2] = euclidean(var(batch), var(batch_simulated));

        # Update summary
        summary += prod(distances .<= system.tolerance);

        # Compute new tolerance
        trial_tolerance .= max.(distances, trial_tolerance); # this works since `trial_tolerance` is initialised to zero
    end

    # Finalise `summary` calculation
    summary /= no_sim;

    # Return `summary`
    return summary;
end

"""
    log_gradient(
        observation :: Float64, 
        parameters  :: AbstractVector{Float64}
    )

Compute the log-gradient.
"""
function log_gradient(
    observation :: Float64, 
    parameters  :: AbstractVector{Float64}
)
    
    μ = parameters[1];
    σ² = get_bounded_logit(parameters[2], 0.0, 1000.0);

    return [0.0; 0.0;
        #-2*(μ-observation);
        #-2*σ²
    ]
end

"""
    update_weights!(
        batch        :: AbstractArray{Float64}, 
        batch_length :: Int64, 
        system       :: ParticleSystem
    )

Update weights within ibis iteration as in Chopin (2002).
"""
function update_weights!(
    batch        :: AbstractArray{Float64}, 
    batch_length :: Int64, 
    system       :: ParticleSystem
)

    # Initialise `trial_tolerance`
    trial_tolerance = zeros(size(system.tolerance));

    # Loop over each particle
    for i=1:system.num_particles
        system.log_weights[i] += system.log_objective(batch, batch_length, view(system.particles, :, i), trial_tolerance, system);
    end

    # Update tolerance
    system.tolerance .= min.(0.95*trial_tolerance, 0.95*system.tolerance);
    println(system.tolerance);
end

"""
This test estimates the mean and standard deviation of normally distributed data with a static SMC.

# Data
The data is such that each y_{i} ~ N(μ, σ^2) for i=1, ..., T.

# Priors
- μ ~ Normal(0, λ^2)
- σ² ~ InverseGamma(3, 1)
"""
function test_univariate_normal_smc(N::Int64, M::Int64, num_particles::Int64; μ0::Float64=0.0, σ0::Float64=1.0, λ::Float64=10.0)

    # Set seed for reproducibility
    Random.seed!(1);

    # Memory pre-allocation for output
    simulations_output = Vector{ParticleSystem}();

    @info("Looping over replications");

    for i=1:M

        # Current simulated data
        y_i = μ0 .+ σ0*randn(N);
        
        # Setup particle system
        system = ParticleSystem(
            0,
            2,
            num_particles,
            
            # Functions
            [Normal(0, λ^2); InverseGamma(3, 1)],
            log_objective!,
            log_gradient,
            update_weights!, 

            # Particles and weights
            [rand(Normal(0, λ^2), 1, num_particles); rand(InverseGamma(3, 1), 1, num_particles)],
            log.(ones(num_particles) / num_particles),
            ones(num_particles) / num_particles,
            Vector{Matrix{Float64}}(),
            Vector{Matrix{Float64}}(),

            # Optional parameters
            [Inf; Inf]
        );
        
        StaticSMC.sample!(y_i, fld(N, 10), system);
        push!(simulations_output, system);
    end

    return simulations_output;
end

simulation_output = test_univariate_normal_smc(1000, 1, 1000);

using Plots