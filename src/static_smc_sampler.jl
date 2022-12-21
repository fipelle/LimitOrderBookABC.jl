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
    move!(system::ParticleSystem)

Rejuvinating step.
"""
function move!(system::ParticleSystem)

end

"""
    ibis_iteration(
        batch::SubArray{Float64},
        system::ParticleSystem;
        batch_lags::Union{Vector{SubArray{Float64}}, Nothing}=nothing
    )

Iterated batch importance sampling algorithms: iteration for new batch of data.

# References
- Steps 1-3 in Chopin (2002, Section 4.1).
"""
function ibis_iteration(
    batch::SubArray{Float64},
    system::ParticleSystem;
    batch_lags::Union{Vector{SubArray{Float64}}, Nothing}=nothing
)
    
    # Step 1: Update the weights
    for i=1:system.num_particles

        # Loop over each observation in `batch`
        for (j, observation) in enumerate(batch)
                
            # i.i.d.
            if system.is_iid
                system.weights[i] *= system.likelihood(observation, view(system.particles, :, i));
            
            # Markov of order `m`
            else
                system.weights[i] *= system.likelihood(observation, batch_lags[j], view(system.particles, :, i));
            end
        end
    end

    # Normalise the weights
    system.weights ./= sum(system.weights);
    
    # Step 2: resample
    resample!(system);

    # Step 3: move
    move!(system);
end
