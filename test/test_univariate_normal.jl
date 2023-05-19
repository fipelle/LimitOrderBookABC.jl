include("../src/StaticSMC.jl");
using Main.StaticSMC;
using Distributions, MessyTimeSeries, Random;
using Infiltrator;

"""
    log_likelihood(
        observation :: Float64, 
        parameters  :: AbstractVector{Float64}
    )

Compute log-likelihood.
"""
function log_likelihood(
    observation :: Float64, 
    parameters  :: AbstractVector{Float64}
)

    μ = parameters[1];
    σ² = get_bounded_logit(parameters[2], 0.0, 1000.0);
    return logpdf(Normal(μ, sqrt(σ²)), observation);
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

    # Loop over each particle
    for i=1:system.num_particles

        # Loop over each observation in `batch`
        for (j, observation) in enumerate(batch)
            
            @infiltrate

            # i.i.d.
            if system.markov_order == 0
                system.log_weights[i] += system.log_objective(observation, view(system.particles, :, i));
            
            # Markov of order `system.markov_order` > 0
            else
                batch_lags = @view data[end-batch_length+j-system.markov_order:end-batch_length+j-1];
                system.log_weights[i] += system.log_objective(observation, batch_lags, view(system.particles, :, i));
            end
        end
    end
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
            log_likelihood,
            update_weights!,

            # Particles and weights
            [rand(Normal(0, λ^2), 1, num_particles); rand(InverseGamma(3, 1), 1, num_particles)],
            log.(ones(num_particles) / num_particles),
            ones(num_particles) / num_particles,
            Vector{Matrix{Float64}}(),
            Vector{Matrix{Float64}}(),

            # Tolerances
            nothing,
            0.1
        );
        
        StaticSMC.sample!(y_i, 1, system);
        push!(simulations_output, system);
    end

    return simulations_output;
end

simulation_output = test_univariate_normal_smc(100, 1, 1000);
