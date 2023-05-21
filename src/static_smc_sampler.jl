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
        old_log_weights    :: AbstractVector{Float64}, 
        aggregate_accuracy :: AbstractVector{Float64}
    )

Return ESS associated to `candidate_scaling` having se the log_weights as a function of the `aggregate_accuracy`.
"""
function _effective_sample_size_abc_scaling(
    candidate_scaling  :: Float64,
    old_log_weights    :: AbstractVector{Float64}, 
    aggregate_accuracy :: AbstractVector{Float64}
)

    # Weights in log-scale
    log_weights = old_log_weights + aggregate_accuracy / candidate_scaling;

    # Convert in lin-scale
    offset = maximum(log_weights);
    weights = exp.(log_weights .- offset);
    weights ./= sum(weights);

    # Return ess
    return _effective_sample_size(weights);
end

#=
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

Rejuvinating step based on a Metropolis-Hasting kernel.
"""
function _move!(
    batch  :: AbstractArray{Float64}, 
    system :: ParticleSystem
)

    for i=1:system.num_particles
        # TBA -> New routine to move system.particles[:, i]
    end
end
=#

"""
    _resample_and_move!(
        system :: ParticleSystem
    )

Resample and move in the direction of the region with higher weight.
"""
function _resample_and_move!(
    system :: ParticleSystem
)

    # Estimate first two moments [note: the weights are normalised at this step]
    exp_particles = mean(system.particles, weights(system.weights), dims=2);
    var_particles = cov(system.particles, weights(system.weights), 2);

    # Resample particles
    system.particles .= rand(MvNormal(view(exp_particles, :, 1), var_particles), system.num_particles);
    
    # Reset weights
    system.weights = ones(system.num_particles) / system.num_particles;
    system.log_weights = log.(system.weights);
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
    
    # Coordinates of the best option
    coordinate_best_option = argmin(distance);
    
    # Stopping criterion 1
    if (distance[coordinate_best_option] <= target_ess_tolerance) || (search_region[1]-search_region[3] <= target_ess_tolerance/10)
        return search_region[coordinate_best_option];
    
    # Second best
    else

        # Stopping criterion 2
        if ess[1] < target_ess
            return search_region[1];

        # Re-try
        else

            # Compute high-to-mid ratio
            high_to_mid_ratio = search_region[1]/search_region[2];

            # Update mid-point
            search_region[2] *= 0.5;
            if ess[1] <= target_ess <= ess[2]
                search_region[2] += 0.5*search_region[1];
            else
                search_region[2] += 0.5*search_region[3];
            end

            # Update high and low
            search_region[1] = high_to_mid_ratio*search_region[2];
            search_region[3] = (2-high_to_mid_ratio)*search_region[2];

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
    if _effective_sample_size(system) < system.num_particles/2
        
        #=
        # Resample the particles and reset the weights
        _resample!(system);

        # Rejuvenate the particles
        _move!(batch, system);
        =#

        _resample_and_move!(system);

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
