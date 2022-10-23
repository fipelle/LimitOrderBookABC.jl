include("../../AbidesMarkets.jl/src/AbidesMarkets.jl");
using Main.AbidesMarkets;

function get_abides_simulation(build_config_kwargs::NamedTuple; nlevels::Int64=10)

    # Build runnable configuration
    config = AbidesMarkets.build_config("rmsc03", build_config_kwargs);

    # Run simulation
    end_state = AbidesMarkets.run(config);

    # Retrieving results from `end_state`
    order_book = end_state["agents"][1].order_books["ABM"]; # Julia starts indexing from 1, not 0

    # Return L2 snapshots
    return AbidesMarkets.get_L2_snapshots(order_book, nlevels);
end

build_config_kwargs = (
    
    # Common settings
    seed=0,
    ticker="ABM",
    historical_date="20200603",
    start_time="09:30:00",
    end_time="16:00:00",
    
    # DGP
    fund_r_bar=100_000,
    fund_kappa=1.67e-16,
    fund_sigma_s=0,
    fund_vol=1e-8,
    fund_megashock_lambda_a=2.77778e-18,
    fund_megashock_mean=1000,
    fund_megashock_var=50_000,
    
    # Number of agents per group
    num_momentum_agents=25,
    num_noise_agents=5000,
    num_value_agents=100,
    
    # Value agents' appraisal of DGP
    val_r_bar=100_000,
    val_kappa=1.67e-15,
    val_vol=1e-8,
    val_lambda_a=7e-11
);