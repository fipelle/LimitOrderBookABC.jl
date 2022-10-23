using Distances, LinearAlgebra, Random, Statistics, StatsPlots, Turing;
include("../src/TuringABC.jl"); using Main.TuringABC;

"""
This test estimates the mean and standard deviation of normally distributed data.

# Data
The data is such that each y_{i} ~ N(μ, σ^2).

# Priors
- μ ~ Normal(0, λ^2)
- σ ~ InverseGamma(3, 1)
"""

# Declare Turing model
@model function univariate_normal_model(y::Vector{Float64}, λ::Float64; no_simulations::Int64=1, percentile::Float64=0.25)
    
    # Priors
    μ ~ Normal(0, λ);
    σ ~ InverseGamma(3, 1);

    # Data percentiles
    percentiles_y = quantile(y, percentile:percentile:1);

    # Initialise summary statistics
    summary_statistics_value = 0.0;

    # Loop over no_simulations
    for i=1:no_simulations
    
        # Simulated data
        y_star = μ .+ σ*randn(length(y));

        # Simulated data percentiles
        percentiles_y_star = quantile(y_star, percentile:percentile:1);

        # Update summary_statistics_value
        summary_statistics_value += nrmsd(percentiles_y, percentiles_y_star)/no_simulations;
    end

    # Loop over the measurements
    for i in axes(y, 1)
        y[i] ~ UnknownContinuousDistribution(-summary_statistics_value, -Inf, 0.0);
    end
end

# Test function
function univariate_normal_test(N::Int64, M::Int64; μ0::Float64=0.0, σ0::Float64=1.0, λ::Float64=100.0, noise::Float64=0.0)

    # Set seed for reproducibility
    Random.seed!(1);

    # Memory pre-allocation for output
    simulations_output = Vector{Chains}();
    simulations_output_avgs = zeros(2, M);

    @info("Looping over replications");

    for i=1:M
        
        # Current simulated data
        y_i = μ0 .+ σ0*randn(N);
        
        # Initial value for the model parameters
        init_θ = [0; 1/2] + noise*[randn(); abs(randn())];
        
        # Current simulation output
        current_simulation_output = sample(
            univariate_normal_model(y_i, λ), 
            MH(0.01*Matrix(I, 2, 2)), 
            50000, discard_initial=25000, 
            init_params=init_θ # somewhere close to the priors' mean
        );

        # Update `simulations_output`
        push!(simulations_output, current_simulation_output);

        # Update `simulations_output_avgs`
        simulations_output_avgs[:, i] = mean(current_simulation_output).nt[:mean];
    end

    @info("Done!");

    # Return output
    return simulations_output, simulations_output_avgs;
end

# Get output
output_1, output_avg_1 = univariate_normal_test(200, 100);
output_2, output_avg_2 = univariate_normal_test(200, 100, noise=1.0);
output_3, output_avg_3 = univariate_normal_test(200, 100, noise=2.5);
output_4, output_avg_4 = univariate_normal_test(200, 100, noise=5.0);

# Select simulation
p1 = density(output_avg_1', labels=["μ" "σ"]);
p2 = density(output_avg_2', labels=["μ" "σ"]);
p3 = density(output_avg_3', labels=["μ" "σ"]);
p4 = density(output_avg_4', labels=["μ" "σ"]);
plot(p1, p2, p3, p4)
