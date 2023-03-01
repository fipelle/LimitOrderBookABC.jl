"""
    _effective_sample_size(weights :: AbstractVector{Float64})    
    _effective_sample_size(system  :: ParticleSystem)

Return ESS of the weights.
"""
_effective_sample_size(weights :: AbstractVector{Float64}) = 1/sum(weights.^2);
_effective_sample_size(system  :: ParticleSystem) = _effective_sample_size(system.weights);

"""
    _effective_sample_size_abc_scaling(
        candidate_scaling  :: Float64,
        aggregate_accuracy :: AbstractVector{Float64},
    )

Return ESS associated to `candidate_scaling` having se the log_weights as a function of the `aggregate_accuracy`.
"""
function _effective_sample_size_abc_scaling(
    candidate_scaling  :: Float64,
    aggregate_accuracy :: AbstractVector{Float64},
)

    # Weights in log-scale
    log_weights = aggregate_accuracy / candidate_scaling;

    # Convert in lin-scale
    offset = maximum(log_weights);
    weights = exp.(log_weights .- offset);
    weights ./= sum(weights);

    # Return ess
    return _effective_sample_size(weights);
end

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
        system :: ParticleSystem
    )

Rejuvinating step based on the unadjusted Langevin algorithm (ULA).

# References
- Roberts and Tweedie (1996, 1.4.1)
"""
function _move!(
    batch  :: AbstractArray{Float64}, 
    system :: ParticleSystem
)

    for i=1:system.num_particles
        
        # Loop over batch
        for observation in batch
            system.particles[:, i] .+= system.tolerance_move/2*system.log_gradient(observation, view(system.particles, :, i));
        end
        
        # Random perturbation
        system.particles[:, i] .+= sqrt(system.tolerance_move)*randn(system.num_parameters);
    end
end

"""
    _find_best_tuning(
        target_ess           :: Float64,
        target_ess_tolerance :: Float64,
        search_region        :: MVector{3, Float64}, # default: [max, mid, min]
        candidate_ess_fun    :: Function,
        candidate_ess_args   :: Tuple
    )

Find best tuning parameter.
"""
function _find_best_tuning(
    target_ess           :: Float64,
    target_ess_tolerance :: Float64,
    search_region        :: MVector{3, Float64}, # default: [max, mid, min]
    candidate_ess_fun    :: Function,
    candidate_ess_args   :: Tuple
)

    # Compute effective sample sizes
    ess = [candidate_ess_fun(search_region[1], candidate_ess_args...);
           candidate_ess_fun(search_region[2], candidate_ess_args...);
           candidate_ess_fun(search_region[3], candidate_ess_args...)];

    # Compute its distance from `target_ess`
    distance = euclidean.(ess, target_ess);

    # First best
    first_best_ind = findfirst(distance .<= target_ess_tolerance);
    if ~isnothing(first_best_ind)
        return search_region[first_best_ind];
    
    # Second best or re-try
    else

        # Second best (i.e., early stopping)
        if ess[1] < target_ess_tolerance
            return search_region[1];

        # Re-try
        else

            # Reset `search_region`
            if distance[1] <= distance[3]
                search_region[1] = search_region[1]/2 + search_region[2]/2;
                search_region[3] = search_region[2];
            else
                search_region[1] = search_region[2];
                search_region[3] = search_region[2]/2 + search_region[3]/2;
            end
            search_region[2] = search_region[1]/2 + search_region[3]/2;

            # Recursive call
            return _find_best_tuning(
                target_ess,
                target_ess_tolerance,
                search_region,
                candidate_ess_fun,
                candidate_ess_args
            );
        end
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
    @infiltrate

    # Resample and move
    println(_effective_sample_size(system))
    if _effective_sample_size(system) < system.num_particles/2

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
