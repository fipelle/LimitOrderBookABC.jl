using MessyTimeSeries;
using Dates, Distances, LinearAlgebra, Random, Statistics, StatsBase, StatsPlots;
using AdvancedMH, Turing;
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

# Shortcut to compute weighted quantiles
function L2_weighted_quantiles(X::SnapshotL2)

    # Compute summary statistics
    summary_per_period = zeros(4, size(X.bids, 1));
    for index in axes(X.bids, 1)
        
        # Collect data incl. missings
        p_index_raw = vcat(X.bids[index, :, 1], X.asks[index, :, 1]);
        q_index_raw = vcat(X.bids[index, :, 2], X.asks[index, :, 2]);
        not_missing = .~ismissing.(p_index_raw);

        # Remove missings
        p_index = @view p_index_raw[not_missing];
        q_index = @view q_index_raw[not_missing];

        # Compute weighted quantiles
        summary_per_period[:, index] = StatsBase.quantile(p_index, Weights(q_index/sum(q_index)), 0.25:0.25:1.0);
    end

    # Return output
    return summary_per_period;
end

# Declare Turing model
@model function abides_model(seed::Int64, y::Vector{Float64}, L2_data::SnapshotL2, ::Type{T}=NamedTuple{(:num_momentum_agents, :num_value_agents, :num_noise_agents), Tuple{Int64, Int64, Int64}}) where {T}
    
    # Priors
    num_momentum_agents ~ DiscreteUniform(1, 200);
    num_value_agents ~ DiscreteUniform(1, 200);
    num_noise_agents ~ DiscreteUniform(1, 2_000);

    # Print current trial
    println("");
    println("num_momentum_agents=$(num_momentum_agents)")
    println("num_value_agents=$(num_value_agents)")
    println("num_noise_agents=$(num_noise_agents)")

    # Get simulated data
    build_config_kwargs = T( (num_momentum_agents, num_value_agents, num_noise_agents) );
    L2_simulated = generate_abides_simulation(build_config_kwargs);

    # Aggregate data
    L2_data_per_minute = aggregate_L2_snapshot_eop(L2_data, Minute(1));
    L2_simulated_per_minute = aggregate_L2_snapshot_eop(L2_simulated, Minute(1));

    # Compute summary statistics
    L2_data_summary_per_period = L2_weighted_quantiles(L2_data_per_minute);
    L2_simulated_summary_per_period = L2_weighted_quantiles(L2_simulated_per_minute);
    summary_statistics_value = median(abs.(L2_data_summary_per_period .- L2_simulated_summary_per_period));
    println("summary=$(summary_statistics_value)");
    
    # Target
    y[1] ~ UnknownContinuousDistribution(-summary_statistics_value, -Inf, 0.0);
end

# Declare test function
function abides_test(M::Int64; build_config_specifics::NamedTuple = NamedTuple())

    # Set seed for reproducibility
    Random.seed!(1);

    # Memory pre-allocation for output
    simulations_output = Vector{Chains}();
    simulations_output_avgs = zeros(3, M);

    # MH tuning parameter
    MH_const = 10;

    @info("Looping over replications");

    for i=1:M

        # Generate L2 through ABIDES-Markets
        L2_data = generate_abides_simulation((; seed=i, build_config_specifics...));

        # Current simulation output
        current_simulation_output = sample(
            abides_model(i, ones(1), L2_data),
            #MH(0.1*Matrix(I, 3, 3)), 
            #=
            MH(
                :num_momentum_agents => AdvancedMH.RandomWalkProposal(DiscreteUniform(-MH_const, MH_const)),
                :num_value_agents => AdvancedMH.RandomWalkProposal(DiscreteUniform(-MH_const, MH_const)),
                :num_noise_agents => AdvancedMH.RandomWalkProposal(DiscreteUniform(-MH_const, MH_const))
            ),
            =#
            SMC(), 
            1000, 
            #discard_initial=1500
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

output_1, output_avg_1 = abides_test(1);
