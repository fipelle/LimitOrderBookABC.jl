include("../src/StaticSMC.jl");
using Main.StaticSMC;
using AbidesMarkets, Distances, Distributions, FileIO, MessyTimeSeries, Random, StaticArrays;
using Infiltrator;

"""
    generate_abides_simulation(build_config_kwargs::NamedTuple; nlevels::Int64=10)

Shortcut to simulate through ABIDES.
"""
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
    log_objective!(
        batch        :: SnapshotL2,
        batch_length :: Int64, 
        parameters   :: AbstractVector{Float64},
        accuracy     :: AbstractVector{Float64};
        no_sim       :: Int64 = 1000
    )

Compute the log-objective.
"""
function log_objective!(
    batch        :: SnapshotL2,
    batch_length :: Int64, 
    parameters   :: AbstractVector{Float64},
    accuracy     :: AbstractVector{Float64};
    no_sim       :: Int64 = 1000
)
    
    # Noise agents relative to the cumulative number of value and momentum agents
    noise_agents_scale = get_bounded_logit(parameters[3], 0.0, 100.0);

    # Number of agents
    num_value_agents    = fld(get_bounded_logit(parameters[1], 0.0, 200.0), 1);
    num_momentum_agents = fld(get_bounded_logit(parameters[2], 0.0, 200.0), 1);
    num_noise_agents    = fld(noise_agents_scale*(num_value_agents+num_momentum_agents), 1);

    # Kwargs for AbidesMarkets
    build_config_kwargs = (
        num_value_agents    = num_value_agents,
        num_momentum_agents = num_momentum_agents, 
        num_noise_agents    = num_noise_agents,
        end_time            = batch.times[end] # end at the same time of the current batch
    );

    # Aggregate L2 data
    batch_per_minute = aggregate_L2_snapshot_eop(batch, Minute(1));

    for i=1:no_sim

        # Simulate data
        simulated_L2 = generate_abides_simulation(build_config_kwargs);
        simulated_batch_per_minute = aggregate_L2_snapshot_eop(L2_simulated, Minute(1));
        
        # Compute accuracy
        accuracy[1] += -euclidean(batch, batch_simulated);
        #accuracy[1] += -euclidean(mean(batch), mean(batch_simulated));
        #accuracy[2] += -euclidean(std(batch), std(batch_simulated));
    end

    # Take average across simulations
    accuracy ./= no_sim;
end

"""
    update_weights!(
        batch        :: AbstractArray{Float64}, 
        batch_length :: Int64, 
        system       :: ParticleSystem
    )

Update weights within ibis iteration.
"""
function update_weights!(
    batch        :: AbstractArray{Float64}, 
    batch_length :: Int64, 
    system       :: ParticleSystem
)

    # Initialise `accuracy`
    accuracy = zeros(length(system.tolerance_abc), system.num_particles);

    # Loop over each particle
    for i=1:system.num_particles
        system.log_objective(batch, batch_length, view(system.particles, :, i), view(accuracy, :, i));
    end

    # Aggregate accuracy
    aggregate_accuracy = accuracy[:];

    system.tolerance_abc = StaticSMC._find_best_tuning(
        system.num_particles / 2 + 1,
        system.num_particles / 10,
        MVector(500.0, 250.5, 1.0),
        StaticSMC._effective_sample_size_abc_scaling,
        (system.log_weights, aggregate_accuracy)
    )
        
    # Compute log weights
    system.log_weights .+= aggregate_accuracy / system.tolerance_abc;
end

"""
This test estimates the number of momentum, value and noise agents via SMC and building on ABIDES.

# Priors
- μ ~ Normal(0, λ^2)
- σ² ~ InverseGamma(3, 1)
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
        num_value_agents    = num_value_agents,
        num_momentum_agents = num_momentum_agents, 
        num_noise_agents    = num_noise_agents,
        end_time            = "16:00:00"
    );

    # Set seed for reproducibility
    Random.seed!(1);

    # Memory pre-allocation for output
    simulations_output = Vector{ParticleSystem}();

    @info("Looping over replications");

    for i=1:M

        @info("Replication $(i) out of $(M)");

        # Current simulated data
        y_i = generate_abides_simulation(merge((seed=i,), build_config_kwargs));
        N_i = size(y_i.asks, 1);

        # Setup particle system
        system = ParticleSystem(
            0,
            2,
            num_particles,
            
            # Functions
            [
                DiscreteUniform(1, 200); # value agents
                DiscreteUniform(1, 200); # momentum agents
                Gamma(10, 1)             # noise agents as no. times the sum of value and momentum agents
            ]
            log_objective!,
            update_weights!, 

            # Particles and weights
            [
                [get_unbounded_logit(x, 0.0, 200.0) for x in rand(DiscreteUniform(1, 200), num_particles)]' # value agents
                [get_unbounded_logit(x, 0.0, 200.0) for x in rand(DiscreteUniform(1, 200), num_particles)]' # momentum agents
                [get_unbounded_logit(x, 0.0, 100.0) for x in rand(Gamma(10, 1), num_particles)]'            # noise agents as no. times the sum of value and momentum agents
            ],
            log.(ones(num_particles) / num_particles),
            ones(num_particles) / num_particles,
            Vector{Matrix{Float64}}(),
            Vector{Matrix{Float64}}(),

            # Tolerances
            [NaN],
            0.1
        );
        
        StaticSMC.sample!(y_i, fld(N_i, 10), system);
        push!(simulations_output, system);
    end

    return simulations_output;
end

# Run simulations
simulation_output = test_univariate_normal_smc(1000, 500, 1000);

# Store output in JLD2
FileIO.save("./test/res_test_univariate_normal_abc.jld2", Dict("simulation_output" => simulation_output));

# Explore one simulation at the time
#=
using Plots
vv = 1; # simulation id
system = simulation_output[vv];
fig1 = histogram(system.particles[1, :]);
fig2 = histogram([get_bounded_logit(system.particles[2, i], 0.0, 1000.0) for i=1:1000]);
=#
