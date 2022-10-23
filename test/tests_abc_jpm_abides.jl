using MessyTimeSeries;
using Distances, LinearAlgebra, Random, Statistics, StatsPlots, Turing;
include("../src/TuringABC.jl"); using Main.TuringABC;
include("../../AbidesMarkets.jl/src/AbidesMarkets.jl"); using Main.AbidesMarkets;

"""
Estimate the coefficients used for simulating data through ABIDES-Markets.
"""

# Shortcut to simulate through ABIDES
function generate_abides_simulation(build_config_kwargs::NamedTuple; nlevels::Int64=10)

    # Build runnable configuration
    config = AbidesMarkets.build_config("rmsc04", build_config_kwargs);

    # Run simulation
    end_state = AbidesMarkets.run(config);

    # Retrieving results from `end_state`
    order_book = end_state["agents"][1].order_books["ABM"]; # Julia starts indexing from 1, not 0

    # Return L2 snapshots
    return AbidesMarkets.get_L2_snapshots(order_book, nlevels);
end

# Declare Turing model
@model function abides_model(seed::Int64, L2_data::JMatrix{Float64})
    
    # Priors
    μ ~ Normal(0, 10_000);  # r_bar
    κ ~ InverseGamma(3, 1); # kappa
    σ ~ InverseGamma(3, 1); # fund_vol
    
    # Simulate data
    L2_simulated = generate_abides_simulation((seed=seed, r_bar=μ, kappa=κ, fund_vol=σ));

    # Aggregate data
    L2_data_per_minute = aggregate_L2_eop(L2_data, Minute(1));
    L2_simulated_per_minute = aggregate_L2_eop(L2_simulated, Minute(1));

    # Compute summary statistics
    # [TBA]

    # Target
    for i in axes(y, 1)
        y[i] ~ UnknownContinuousDistribution(-summary_statistics_value, -Inf, 0.0);
    end
end

# Declare test function
function abides_test(M::Int64; build_config_specifics::NamedTuple = NamedTuple())

    # Set seed for reproducibility
    Random.seed!(1);

    # Memory pre-allocation for output
    simulations_output = Vector{Chains}();
    simulations_output_avgs = zeros(2, M);

    @info("Looping over replications");

    for i=1:M

        # Generate L2 through ABIDES-Markets
        L2_data = generate_abides_simulation((; seed=i, build_config_specifics...));

        # Current simulation output
        current_simulation_output = sample(
            abides_model(i, L2_data),
            MH(0.01*Matrix(I, 2, 2)), 
            50000, 
            discard_initial=25000
        );

    end
    
    return simulations_output, simulations_output_avgs;
end
