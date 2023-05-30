include("../src/StaticSMC.jl");
using AbidesMarkets, Dates, Main.StaticSMC, Suppressor;
using Distributions, FileIO, Logging, MessyTimeSeries, Random, StaticArrays;
using Infiltrator;

"""
    generate_abides_simulation(build_config_kwargs::NamedTuple; nlevels::Int64=10)

Shortcut to simulate through ABIDES.
"""
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
    get_batch_best(batch::SnapshotL2)

Get best price from L2 snapshot.
"""
function get_batch_best(batch::SnapshotL2)
    
    # Aggregate batch
    batch_per_minute = aggregate_L2_snapshot_eop(batch, Minute(1));

    # Best price per minute
    batch_L1_bids = view(batch_per_minute.bids, :, 1, :);
    batch_L1_asks = view(batch_per_minute.asks, :, 1, :);
    batch_best = prod(batch_L1_bids, dims=2) + prod(batch_L1_asks, dims=2);
    batch_best ./= view(batch_L1_bids, :, 2) + view(batch_L1_asks, :, 2);

    # Return best price per minute
    return batch_best;
end

"""
    log_objective(
        batch        :: SnapshotL2,
        batch_length :: Int64, 
        parameters   :: AbstractVector{Float64};
        no_sim       :: Int64 = 1
    )

Compute the log-objective.
"""
function log_objective(
    batch        :: SnapshotL2,
    batch_length :: Int64, 
    parameters   :: AbstractVector{Float64};
    no_sim       :: Int64 = 1
)

    # Number of agents
    num_value_agents    = floor(Int64, get_bounded_logit(parameters[1], 0.0, 201.0));
    num_momentum_agents = floor(Int64, get_bounded_logit(parameters[2], 0.0, 51.0));
    num_noise_agents    = floor(Int64, get_bounded_logit(parameters[3], 249.0, 2001.0));
    
    # Kwargs for AbidesMarkets
    build_config_kwargs = (
        # Number of agents
        num_value_agents    = num_value_agents,
        num_momentum_agents = num_momentum_agents, 
        num_noise_agents    = num_noise_agents,
        # Fundamental/oracle
        fund_r_bar          = 100_000,
        fund_vol            = 1e-3,
        fund_megashock_mean = 1000,
        fund_megashock_var  = 50_000,
        # Value agent
        val_r_bar           = 100_000,
        val_vol             = 1e-8,
        end_time            = Dates.format(batch.times[end], "HH:MM:SS") # end at the same time of the current batch
    );

    # Best price per minute
    batch_best = get_batch_best(batch);

    # Initialise multi-threaded loop output
    accuracy_sim = zeros(no_sim);

    # Loop over `no_sim`
    for i=1:no_sim

        # Simulate data
        local simulated_data;
        #@suppress begin
        @infiltrate
            simulated_data = generate_abides_simulation(merge((seed=1000+i,), build_config_kwargs));
        #end

        # Get coordinates to align `simulated_batch` with `batch`
        is_simulated_batch = simulated_data.times .>= batch.times[1];

        # Get `simulated_batch`
        simulated_batch = SnapshotL2(
            simulated_data.times[is_simulated_batch],
            simulated_data.bids[is_simulated_batch, :, :],
            simulated_data.asks[is_simulated_batch, :, :]
        );
        
        # Best price per minute
        simulated_batch_best = get_batch_best(simulated_batch);

        # Compute accuracy
        accuracy_sim[i] = mean(skipmissing((batch_best - simulated_batch_best).^2));
    end

    # Take average across simulations
    return mean(accuracy_sim);
end

"""
    update_weights!(
        batch        :: SnapshotL2,
        batch_length :: Int64, 
        system       :: ParticleSystem
    )

Update weights within ibis iteration.
"""
function update_weights!(
    batch        :: SnapshotL2,
    batch_length :: Int64, 
    system       :: ParticleSystem
)

    # Initialise Logger
    io = open("log_$(now()).txt", "w+");
    global_logger(ConsoleLogger(io));

    # Initialise `accuracy`
    accuracy = zeros(length(system.tolerance_abc), system.num_particles);
    @info("started updating weights at $(now())")
    flush(io)

    # Loop over each particle
    accuracy = for i=1:system.num_particles #Threads.@threads
        @info(i)
        flush(io)
        accuracy[i] = system.log_objective(batch, batch_length, view(system.particles, :, i));
    end
    accuracy /= system.num_particles;
    @info("finished evaluating log_objective at $(now())")
    flush(io)
    #@infiltrate

    # Aggregate accuracy
    aggregate_accuracy = accuracy[:];

    system.tolerance_abc = StaticSMC._find_best_tuning(
        system.num_particles / 2 + 1,
        system.num_particles / 10,
        MVector(500.0, 250.5, 1.0),
        StaticSMC._effective_sample_size_abc_scaling,
        (system.log_weights, aggregate_accuracy)
    )
    @info("abc tolerance updated at $(now())")
    flush(io)

    # Compute log weights
    system.log_weights .+= aggregate_accuracy / system.tolerance_abc;
    @info("Current update completed at $(now())")
    flush(io)
    close(io)
end

"""
This test estimates the number of momentum, value and noise agents via SMC and building on ABIDES.
"""
function test_abides_basic(
    M                   :: Int64, 
    num_particles       :: Int64; 
    num_value_agents    :: Int64 = 102,
    num_momentum_agents :: Int64 = 12,
    num_noise_agents    :: Int64 = 1000
)

    # Kwargs for AbidesMarkets
    build_config_kwargs = (
        # Number of agents
        num_value_agents    = num_value_agents,
        num_momentum_agents = num_momentum_agents, 
        num_noise_agents    = num_noise_agents,
        # Fundamental/oracle
        fund_r_bar          = 100_000,
        fund_vol            = 1e-3,
        fund_megashock_mean = 1000,
        fund_megashock_var  = 50_000,
        # Value agent
        val_r_bar           = 100_000,
        val_vol             = 1e-8,
        end_time            = "16:00:00"
    );

    # Priors
    priors = [
        DiscreteUniform(1,    200); # no. of value agents
        DiscreteUniform(1,    50);  # no. of momentum agents
        DiscreteUniform(250, 2000); # no. of noise agents
    ];

    # Set seed for reproducibility
    Random.seed!(1);

    # Memory pre-allocation for output
    simulations_output = Vector{ParticleSystem}();

    @info("Looping over replications");

    for i=1:M

        @info("Replication $(i) out of $(M)");

        # Current simulated data
        local y_i;
        @suppress begin
            y_i = generate_abides_simulation(merge((seed=i,), build_config_kwargs));
        end
        N_i = size(y_i.asks, 1);
        
        # Setup particle system
        system = ParticleSystem(
            0,
            2,
            num_particles,
            
            # Functions
            priors,
            log_objective,
            update_weights!, 

            # Particles and weights
            [
                [get_unbounded_logit(Float64(x), 0.0, 201.0) for x in rand(priors[1], num_particles)]'    # no. of value agents
                [get_unbounded_logit(Float64(x), 0.0, 51.0) for x in rand(priors[2], num_particles)]'     # no. of momentum agents
                [get_unbounded_logit(Float64(x), 249.0, 2001.0) for x in rand(priors[3], num_particles)]' # no. of noise agents
            ],
            log.(ones(num_particles) / num_particles),
            ones(num_particles) / num_particles,
            Vector{Matrix{Float64}}(),
            Vector{Matrix{Float64}}(),

            # Tolerances
            [NaN],
            0.1
        );
        
        StaticSMC.sample!(y_i, fld(N_i, 5), system);
        push!(simulations_output, system);
    end

    return simulations_output;
end

# Run simulations
simulation_output = test_abides_basic(1, 1000);

# Store output in JLD2
FileIO.save("./test/res_test_abides_basic.jld2", Dict("simulation_output" => simulation_output));

# Explore one simulation at the time
#=
using Plots
vv = 1; # simulation id
system = simulation_output[vv];
fig1 = histogram(system.particles[1, :]);
fig2 = histogram([get_bounded_logit(system.particles[2, i], 0.0, 1000.0) for i=1:1000]);
=#
