include("../../AbidesMarkets.jl/src/AbidesMarkets.jl");
using Main.AbidesMarkets;

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

"""
    abides_test(M::Int64; build_config_specifics::NamedTuple = NamedTuple())

Estimate the coefficients used for simulating data through ABIDES-Markets.
"""
function abides_test(M::Int64; build_config_specifics::NamedTuple = NamedTuple())

    # AbidesMarkets.build_config relevant keyword arguments
    build_config_kwargs = (
        seed = 0,
        build_config_specifics...,
    );
    
    # Generate two simulations
    L2_1 = generate_abides_simulation(build_config_kwargs);
    L2_2 = generate_abides_simulation(build_config_kwargs);

    return L2_1, L2_2;
end
