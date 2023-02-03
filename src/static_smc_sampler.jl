"""
    effective_sample_size(system::ParticleSystem)

Return ESS of the weights.
"""
effective_sample_size(system::ParticleSystem) = 1/sum(system.weights.^2);

"""
    resample!(system::ParticleSystem)

Resampling step.
"""
function resample!(system::ParticleSystem)

    # Create a Multinomial distribution using the weights as probabilities
    mn = Multinomial(system.num_particles, system.weights);

    # Draw a sample from `mn`
    mn_draw = rand(mn);

    # Resample the particles
    counter = 1; # 1 is correct for this counter
    old_particles = copy(system.particles);
    for (index, value) in enumerate(mn_draw)
        for k=1:value
            system.particles[:, counter] = old_particles[:, index];
            counter += 1;
        end
    end

    # Reset weights
    system.weights = ones(system.num_particles) / system.num_particles;
    system.log_weights = log.(system.weights);
end

"""
    move!(
        last_datapoint::Float64,
        system::ParticleSystem; 
        ε::Float64=0.1
    )

Rejuvinating step based on the unadjusted Langevin algorithm (ULA).

# References
- Roberts and Tweedie (1996, 1.4.1)
"""
function move!(
    last_datapoint::Float64,
    system::ParticleSystem; 
    ε::Float64=0.1
)

    for i=1:system.num_particles
        system.particles[:, i] .+= ε/2*system.log_gradient(last_datapoint, view(system.particles, :, i));
        system.particles[:, i] .+= sqrt(ε)*randn(system.num_parameters);
    end
end

"""
    ibis_iteration!(
        batch::SubArray{Float64},
        system::ParticleSystem;
        batch_lags::Union{Vector{SubArray{Float64}}, Nothing}=nothing
    )

Iterated batch importance sampling algorithms: iteration for new batch of data.

# References
- Chopin (2002, Section 4.1)
- Roberts and Tweedie (1996, 1.4.1)
"""
function ibis_iteration!(
    data::Vector{Float64},
    batch_length::Int64,
    system::ParticleSystem;
)

    # Generate view on current batch
    batch = @view data[end-batch_length+1:end];

    # Update the weights
    for i=1:system.num_particles

        # Loop over each observation in `batch`
        for (j, observation) in enumerate(batch)
            
            # i.i.d.
            if system.markov_order == 0
                system.log_weights[i] += system.log_likelihood(observation, view(system.particles, :, i));
            
            # Markov of order `system.markov_order` > 0
            else
                @infiltrate
                batch_lags = @view data[end-batch_length+j-system.markov_order:end-batch_length+j-1];
                system.log_weights[i] += system.log_likelihood(observation, batch_lags, view(system.particles, :, i));
                @infiltrate
            end
        end
    end

    # Normalise the weights
    offset = maximum(system.log_weights);
    system.weights = exp.(system.log_weights .- offset);
    system.weights ./= sum(system.weights);

    # Resample and move
    if effective_sample_size(system) < system.num_particles/2

        # Resample the particles and reset the weights
        resample!(system);
        
        # Rejuvenate the particles
        move!(data[end], system);
    
    # Update `system.log_weights` to reflect normalisation
    else
        system.log_weights = log.(system.weights);
    end

    # Update history
    push!(system.particles_history, copy(system.particles));
    push!(system.weights_history, copy(system.weights));
end

"""
    sample!(
        full_data::Vector{Float64},
        batch_length::Int64,
        system::ParticleSystem;
    )


"""
function sample!(
    full_data::Vector{Float64},
    batch_length::Int64,
    system::ParticleSystem;
)
    
    # Initialise counter
    counter = batch_length;

    # Loop over time
    for t=1:fld(length(full_data), batch_length)

        # Current data
        data = full_data[1:counter];
        
        # Iterated batch importance sampling round
        ibis_iteration!(data, batch_length, system);
        
        # Update counter
        counter += batch_length;
    end
end
