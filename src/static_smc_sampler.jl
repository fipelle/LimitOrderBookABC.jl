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
    mn = Multinomial(length(system.weights), system.weights);

    # Draw a sample from `mn`
    mn_draw = rand(mn);
    
    # Resample the particles
    resampled_particles = [];
    for (index, value) in enumerate(mn_draw)
        for k=1:value
            push!(resampled_particles, system.particles[index]);
        end
    end

    # Update system
    copyto!(system.particles, resampled_particles);
    copyto!(system.weights, ones(system.num_particles) / system.num_particles);
end

"""
    move!(
        data::Vector{Float64},
        system::ParticleSystem; 
        ε::Float64=0.1
    )

Rejuvinating step based on the unadjusted Langevin algorithm (ULA).

# References
- Roberts and Tweedie (1996, 1.4.1)
"""
function move!(
    data::Vector{Float64},
    system::ParticleSystem; 
    ε::Float64=0.1
)

    for i=1:system.num_particles
        system.particles[i] += ε/2*system.gradient(data, system.particles[i]);
        system.particles[i] += sqrt(ε)*randn();
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
                system.weights[i] *= system.likelihood(observation, view(system.particles, :, i));
            
            # Markov of order > 0
            else
                batch_lags = @view data[end-batch_length+j-system.markov_order:end-batch_length+j-1];
                system.weights[i] *= system.likelihood(observation, batch_lags, view(system.particles, :, i));
            end
        end
    end

    # Normalise the weights
    system.weights ./= sum(system.weights);

    if effective_sample_size(system) < 0.5

        # Resample the particles and reset the weights
        resample!(system);
        
        # Rejuvenate the particles
        move!(data, system);
    end
end
