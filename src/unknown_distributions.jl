#=
--------------------------------------------------------------------------------------------------------------------------------
Datatypes
--------------------------------------------------------------------------------------------------------------------------------
=#

"""
    UnknownContinuousDistribution(...)

Datatype representing an unknown continuous distribution (bounded or unbounded).
"""
struct UnknownContinuousDistribution <: ContinuousUnivariateDistribution
    summary_statistics_value::Real
    summary_statistics_lower_bound::Real
    summary_statistics_upper_bound::Real
end

"""
    UnknownContinuousDistribution(summary_statistics_value::Real)

`UnknownContinuousDistribution` constructor for unbounded continuous distributions.
"""
UnknownContinuousDistribution(summary_statistics_value::Real) = UnknownContinuousDistribution(summary_statistics_value, -Inf, Inf);



#=
--------------------------------------------------------------------------------------------------------------------------------
Unknown distributions methods and properties
--------------------------------------------------------------------------------------------------------------------------------
=#

# Shortcut for unknown distributions
const UnknownDistribution = Union{UnknownContinuousDistribution, UnknownDiscreteDistribution};

# Julia cannot sample from an unknown distribution
Distributions.rand(rng::AbstractRNG, d::UnknownDistribution) = nothing;

# While the pdf is also unknown, a good summary statistics should be able to proxy it to some extent - the latter is computed externally (e.g., within a Turing model), and stored in `d`
Distributions.logpdf(d::UnknownDistribution, x::Real) = d.summary_statistics_value;

# Bounds
Distributions.minimum(d::UnknownDistribution) = d.lower_bound;
Distributions.maximum(d::UnknownDistribution) = d.upper_bound;
