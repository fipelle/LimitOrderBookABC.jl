"""
    effective_sample_size(system::ParticleSystem)

Return ESS of the weights.
"""
effective_sample_size(system::ParticleSystem) = 1/sum(system.weights.^2);

"""
    _resample!(system::ParticleSystem)

Resampling step.
"""
function _resample!(system::ParticleSystem)

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
    _move!(
        batch  :: AbstractArray{Float64}, 
        system :: ParticleSystem; 
        ε      :: Float64=0.1
    )

Rejuvinating step based on the unadjusted Langevin algorithm (ULA).

# References
- Roberts and Tweedie (1996, 1.4.1)
"""
function _move!(
    batch  :: AbstractArray{Float64}, 
    system :: ParticleSystem; 
    ε      :: Float64=0.1
)

    for i=1:system.num_particles
        
        # Loop over batch
        for observation in batch
            system.particles[:, i] .+= ε/2*system.log_gradient(observation, view(system.particles, :, i));
        end
        
        # Random perturbation
        system.particles[:, i] .+= sqrt(ε)*randn(system.num_parameters);
    end
end

"""
    _ibis_iteration!(
        data         :: Vector{Float64},
        batch_length :: Int64,
        system       :: ParticleSystem
    )

Iterated batch importance sampling algorithms: iteration for new batch of data.

# References
- Chopin (2002, Section 4.1)
- Roberts and Tweedie (1996, 1.4.1)
"""
function _ibis_iteration!(
    data         :: Vector{Float64},
    batch_length :: Int64,
    system       :: ParticleSystem
)

    # Generate view on current batch
    batch = @view data[end-batch_length+1:end];

    # Update the weights
    system.update_weights!(batch, batch_length, system);

    # Normalise the weights
    offset = maximum(system.log_weights);
    system.weights = exp.(system.log_weights .- offset);
    system.weights ./= sum(system.weights);

    # Resample and move
    println(effective_sample_size(system))
    if effective_sample_size(system) < system.num_particles/2

        # Resample the particles and reset the weights
        _resample!(system);

        # Rejuvenate the particles
        _move!(batch, system);
    
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
        full_data    :: Vector{Float64},
        batch_length :: Int64,
        system       :: ParticleSystem;
    )

"""
function sample!(
    full_data    :: Vector{Float64},
    batch_length :: Int64,
    system       :: ParticleSystem;
)
    
    # Initialise counter
    counter = batch_length;

    # Loop over time
    for t=1:fld(length(full_data), batch_length)

        # Current data
        data = full_data[1:counter];
        
        # Iterated batch importance sampling round
        _ibis_iteration!(data, batch_length, system);
        
        # Update counter
        counter += batch_length;
    end
end
