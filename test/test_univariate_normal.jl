include("../src/StaticSMC.jl");
using Main.StaticSMC;
using Distributions, Random;

test_univariate_normal_smc_likelihood(observation::Float64, parameters::AbstractVector{Float64}) = pdf(Normal(parameters[1], parameters[2]^2), observation);
test_univariate_normal_smc_gradient(observation::Float64, parameters::AbstractVector{Float64}) = zeros(length(parameters));

"""
This test estimates the mean and standard deviation of normally distributed data with a static SMC.

# Data
The data is such that each y_{i} ~ N(μ, σ^2) for i=1, ..., T.

# Priors
- μ ~ Normal(0, λ^2)
- σ ~ InverseGamma(3, 1)
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
            
            # Densities
            [Normal(0, λ^2); InverseGamma(3, 1)],
            test_univariate_normal_smc_likelihood,
            test_univariate_normal_smc_gradient,
            
            # Particles and weights
            [rand(Normal(0, λ^2), 1, num_particles); rand(InverseGamma(3, 1), 1, num_particles)],
            ones(num_particles) / num_particles,
            Vector{Matrix{Float64}}(),
            Vector{Matrix{Float64}}()
        );
        
        StaticSMC.sample!(y_i, 1, system);
        push!(simulations_output, system);
    end
end
