include("../../AbidesMarkets.jl/src/AbidesMarkets.jl");
using Main.AbidesMarkets;

function generate_abides_simulation(build_config_kwargs::NamedTuple; nlevels::Int64=10)

    # Build runnable configuration
    config = AbidesMarkets.build_config("rmsc03", build_config_kwargs);

    # Run simulation
    end_state = AbidesMarkets.run(config);

    # Retrieving results from `end_state`
    order_book = end_state["agents"][1].order_books["ABM"]; # Julia starts indexing from 1, not 0

    # Return L2 snapshots
    return AbidesMarkets.get_L2_snapshots(order_book, nlevels);
end

"""
    abides_test(

        # Number of simulations
        M::Int64; 

        # DGP mean and volatility
        r_bar::Float64 = 100000.0,
        vol::Float64 = 1e-8,
        
        # Further keyword arguments for `AbidesMarkets.build_config`
        build_config_additional_kwargs::NamedTuple = ()
    )

Estimate the coefficients used for simulating data through ABIDES-Markets.
"""
function abides_test(

    # Number of simulations
    M::Int64; 

    # DGP mean and volatility
    r_bar::Float64 = 100000.0,
    vol::Float64 = 1e-8,
    
    # Further keyword arguments for `AbidesMarkets.build_config`
    build_config_additional_kwargs::NamedTuple = ()
)

    # AbidesMarkets.build_config relevant keyword arguments
    build_config_kwargs = NamedTuple(

        # Fix random seed
        seed = 0,

        # Value traders have an exact expectation for both the mean and variance of the DGP
        # -> this identifies them as traders with priviledged (potentially imperfect) information
        fund_r_bar = r_bar, 
        val_r_bar = r_bar, 
        fund_vol = vol, 
        val_vol = vol, 
        
        # Further arguments (if any)
        build_config_additional_kwargs...,
    );
    
    L2_1 = generate_abides_simulation(build_config_kwargs);
    L2_2 = generate_abides_simulation(build_config_kwargs);

    return L2_1, L2_2;
end

#=
    # Further DGP parameters [true values]
    fund_kappa::Float64 = 1.67e-16,
    fund_sigma_s::Float64 = 0.0,
    fund_megashock_lambda_a::Float64 = 2.77778e-18,
    fund_megashock_mean::Float64 = 1000.0,
    fund_megashock_var::Float64 = 50000.0,
    
    # Value agents' appraisal of DGP [true values]
    val_kappa::Float64 = 1.67e-15,
    val_lambda_a::Float64 = 7e-11,

    # Number of agents per group [true values]
    num_momentum_agents::Int64 = 25,
    num_noise_agents::Int64 = 5000,
    num_value_agents::Int64 = 100,
=#