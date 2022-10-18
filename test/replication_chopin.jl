using Turing;

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

"""
This test replicates the example in Section 5.2 (Mixtures) of Nicolas Chopin (2002)

# Data
The data y_{i} | {z_{i} = l} ~ N(μ_{l}, σ_{l}^2) where:
- the latent variables z_{i} ~ MN(k; p_1, ..., p_k)
- the identifiability constraint is s.t. μ_{1} < ... < μ_{k}
- the standard errors σ_{j} were reparameterised to s_{j} = log(σ_{j})

# Priors
Uniform priors over the compact set corresponding to the following constraints:
- p_{1}, ..., p_{k-1} ≥ 0, p_{1} + ... + p_{k-1} ≤ 1
- μ̄_{L} < μ_{1} < ... < μ_{k} < μ̄_{U}
- s_{1}, ..., s_{k} ∈ [s̄_{L}, s̄_{U}]

where:
- μ̄_{L} = min(y_{i}), μ̄_{U} = max(y_{i})
- s̄_{L} = log[min(|y_{i}-y_{i-1}|)/2], s̄_{U} = log[max(μ̄_{U}-μ̄_{L})/(k+1)]
[the last k+1 may be a typo in the paper -> k+2 seems more reasonable]

# Further settings
- N=1000 observations drawn from a Gaussian mixture model with k=5 components
- The parameter θ=(p_{1}, ..., p_{4}, μ_{1}, ..., μ_{5}, s_{1}, ... s_{5}) is a vector of 14 dimensions
- The number of particles is set to H=10,000
"""
function simulatate_chopin_data()

end