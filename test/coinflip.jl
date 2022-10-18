using Random, Turing;

# Unconditioned coinflip model with `N` observations.
@model function coinflip(; N::Int)
    # Our prior belief about the probability of heads in a coin toss.
    p ~ Beta(1, 1)

    # Heads or tails of a coin are drawn from `N` independent and identically
    # distributed Bernoulli distributions with success rate `p`.
    y ~ filldist(Bernoulli(p), N)

    return y
end;

# Conditioned coinflip model with `N` observations.
coinflip(y::AbstractVector{<:Real}) = coinflip(; N=length(y)) | (; y)

# Unconditioned coinflip model with `N` observations.
@model function coinflip_2(y::Vector{Bool})

    # Our prior belief about the probability of heads in a coin toss.
    p ~ Beta(1, 1)

    # Heads or tails of a coin are drawn from `N` independent and identically
    # distributed Bernoulli distributions with success rate `p`.
    y ~ filldist(Bernoulli(p), length(y))

    return y
end;

# Unconditioned coinflip model with `N` observations.
@model function coinflip_3(y::Vector{Bool})
    
    # Our prior belief about the probability of heads in a coin toss.
    p ~ Beta(1, 1)

    # Heads or tails of a coin are drawn from `N` independent and identically
    # distributed Bernoulli distributions with success rate `p`.
    for i in axes(y,1)
        y[i] ~ Bernoulli(p)
    end

    return y
end;

# Unconditioned coinflip model with `N` observations.
@model function coinflip_4(y::Vector{Bool})
    
    # Our prior belief about the probability of heads in a coin toss.
    p ~ Beta(1, 1)

    # Heads or tails of a coin are drawn from `N` independent and identically
    # distributed Bernoulli distributions with success rate `p`.
    for i in axes(y,1)
        y[i] ~ Bernoulli(p)
    end
end;

simulated_data_25 = rand(Bernoulli(0.25), 100);
simulated_data_50 = rand(Bernoulli(0.50), 100);

chain_1 = sample(coinflip(N=100), HMC(0.05, 10), 1_000; progress=false);
chain_2 = sample(coinflip(simulated_data_25), HMC(0.05, 10), 1_000; progress=false);

Random.seed!(1);
chain_3 = sample(coinflip(simulated_data_50), HMC(0.05, 10), 1_000; progress=false);

Random.seed!(1);
chain_3_SMC = sample(coinflip(simulated_data_50), SMC(), 1_000; progress=false);

Random.seed!(1);
chain_3_v2 = sample(coinflip_2(simulated_data_50), HMC(0.05, 10), 1_000; progress=false);

Random.seed!(1);
chain_3_v3 = sample(coinflip_3(simulated_data_50), HMC(0.05, 10), 1_000; progress=false);

Random.seed!(1);
chain_3_v4 = sample(coinflip_4(simulated_data_50), HMC(0.05, 10), 1_000; progress=false);
