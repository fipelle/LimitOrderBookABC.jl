using Random, Turing;

"""
This test estimates the mean and standard deviation of normally distributed data.

# Data
The data is such that each y_{i} ~ N(μ, σ^2).

# Priors
- μ ~ Normal(0, λ^2)
- σ ~ TruncatedNormal(0, λ^2, 0, Inf)
"""

# Declare Turing model
@model function univariate_normal_model(y::Vector{Float64}, λ::Float64)
    μ ~ Normal(0, λ);
    σ ~ TruncatedNormal(0, λ, 0, Inf);
    for i in axes(y, 1)
        y[i] ~ Normal(μ, σ);
    end
end

# Test function
function univariate_normal_test(N::Int64, M::Int64; μ0::Float64=0.0, σ0::Float64=1.0, λ::Float64=100.0)

    # Set seed for reproducibility
    Random.seed!(1);

    # Memory pre-allocation for output
    simulations_output = Vector{Chains}();
    
    # Loop over replications
    for i=1:M
        
        # Current simulated data
        y_i = μ0 .+ σ0*randn(N);
        
        # Update `simulations_output`
        push!(simulations_output, sample(univariate_normal_model(y_i, λ), SMC(), 1000));
    end

    # Return output
    return simulations_output;
end

"""
This test uses a simple particle system to estimate a linear regression model.

# Data
The model of interest is s.t. the data y_{i} ~ N(β*x_{i}, σ^2) where for some predictor x_{i}.
The predictors are simulated s.t. x_{i} ~ N(0, γ^2) for some γ ≥ 0.

# Priors
The coefficients have the following priors:
- β ~ Normal(0, λ^2)
- σ ~ TruncatedNormal(0, λ^2, 0, Inf)
"""

# Declare Turing model
@model function univariate_regression_model(y::Vector{Float64}, x::Vector{Float64}, λ::Float64)
    β ~ Normal(0, λ);
    σ ~ TruncatedNormal(0, λ, 0, Inf);
    for i in axes(y, 1)
        y[i] ~ Normal(β*x[i], σ);
    end
end

# Test function
function univariate_regression_test(N::Int64, M::Int64; β0::Float64=0.0, σ0::Float64=1.0, γ::Float64=10.0, λ::Float64=100.0)

    # Set seed for reproducibility
    Random.seed!(1);

    # Memory pre-allocation for output
    simulations_output = Vector{Chains}();

    # Loop over replications
    for i=1:M
        
        # Current simulated data
        x_i = γ*randn(N);
        y_i = β0 .* x_i;
        y_i .+= σ0*randn(N);
        
        # Update `simulations_output`
        push!(simulations_output, sample(univariate_regression_model(y_i, x_i, λ), SMC(), 1000));
    end

    # Return output
    return simulations_output;
end
