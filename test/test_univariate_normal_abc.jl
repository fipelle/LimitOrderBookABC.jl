include("../src/StaticSMC.jl");
using Main.StaticSMC;
using Distances, Distributions, MessyTimeSeries, Random, StaticArrays;
using Infiltrator;

"""
    log_objective!(
        batch        :: AbstractArray{Float64}, 
        batch_length :: Int64, 
        parameters   :: AbstractVector{Float64},
        accuracy     :: AbstractVector{Float64};
        no_sim       :: Int64 = 1000
    )

Compute the log-objective.
"""
function log_objective!(
    batch        :: AbstractArray{Float64}, 
    batch_length :: Int64, 
    parameters   :: AbstractVector{Float64},
    accuracy     :: AbstractVector{Float64};
    no_sim       :: Int64 = 1000
)
    
    # Retrieve current parameters configuration
    μ = parameters[1];
    σ² = get_bounded_logit(parameters[2], 0.0, 1000.0);
    
    for i=1:no_sim

        # Simulate data
        batch_simulated = μ .+ sqrt(σ²)*randn(batch_length);
        
        # Compute accuracy
        accuracy[1] += -euclidean(batch, batch_simulated);
        #accuracy[1] += -euclidean(mean(batch), mean(batch_simulated));
        #accuracy[2] += -euclidean(std(batch), std(batch_simulated));
    end

    # Take average across simulations
    accuracy ./= no_sim;
end

"""
    update_weights!(
        batch        :: AbstractArray{Float64}, 
        batch_length :: Int64, 
        system       :: ParticleSystem
    )

Update weights within ibis iteration.
"""
function update_weights!(
    batch        :: AbstractArray{Float64}, 
    batch_length :: Int64, 
    system       :: ParticleSystem
)

    # Initialise `accuracy`
    accuracy = zeros(length(system.tolerance_abc), system.num_particles);

    # Loop over each particle
    for i=1:system.num_particles
        system.log_objective(batch, batch_length, view(system.particles, :, i), view(accuracy, :, i));
    end

    @infiltrate

    # Aggregate accuracy
    aggregate_accuracy = accuracy[:];

    system.tolerance_abc = StaticSMC._find_best_tuning(
        system.num_particles / 2 + 1,
        system.num_particles / 10,
        MVector(500.0, 250.5, 1.0),
        StaticSMC._effective_sample_size_abc_scaling,
        (aggregate_accuracy,)
    )

    @infiltrate
    
    # Compute log weights
    system.log_weights .= aggregate_accuracy[:] / system.tolerance_abc;
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
            update_weights!, 

            # Particles and weights
            [rand(Normal(0, λ^2), 1, num_particles); rand(InverseGamma(3, 1), 1, num_particles)],
            log.(ones(num_particles) / num_particles),
            ones(num_particles) / num_particles,
            Vector{Matrix{Float64}}(),
            Vector{Matrix{Float64}}(),

            # Tolerances
            [NaN],
            0.1
        );
        
        StaticSMC.sample!(y_i, fld(N, 10), system);
        push!(simulations_output, system);
    end

    return simulations_output;
end

simulation_output = test_univariate_normal_smc(1000, 1, 1000);

using Plots