#=
--------------------------------------------------------------------------------------------------------------------------------
Unknown distributions' datatypes
--------------------------------------------------------------------------------------------------------------------------------
=#

"""
    UnknownContinuousDistribution(...)

Datatype representing an unknown continuous distribution (bounded or unbounded).
"""
struct UnknownContinuousDistribution <: ContinuousUnivariateDistribution
    summary_statistics_value::Real
    summary_statistics_minimum::Real
    summary_statistics_maximum::Real
end

"""
    UnknownContinuousDistribution(summary_statistics_value::Real)

`UnknownContinuousDistribution` constructor for unbounded continuous distributions.
"""
UnknownContinuousDistribution(summary_statistics_value::Real) = UnknownContinuousDistribution(summary_statistics_value, -Inf, Inf);

"""
    UnknownDiscreteDistribution(...)

Datatype representing an unknown discrete distribution (bounded or unbounded).
"""
struct UnknownDiscreteDistribution <: DiscreteUnivariateDistribution
    summary_statistics_value::Real
    summary_statistics_minimum::Real
    summary_statistics_maximum::Real
end

"""
    UnknownDiscreteDistribution(summary_statistics_value::Real)

`UnknownDiscreteDistribution` constructor for unbounded discrete distributions.
"""
UnknownDiscreteDistribution(summary_statistics_value::Real) = UnknownDiscreteDistribution(summary_statistics_value, -Inf, Inf);


#=
--------------------------------------------------------------------------------------------------------------------------------
Unknown distributions' methods and properties
--------------------------------------------------------------------------------------------------------------------------------
=#

# Shortcut for unknown distributions
const UnknownDistribution = Union{UnknownContinuousDistribution, UnknownDiscreteDistribution};

# Julia cannot sample from an unknown distribution
Distributions.rand(rng::AbstractRNG, d::UnknownDistribution) = nothing;

# While the pdf is also unknown, a good summary statistics should be able to proxy it to some extent - the latter is computed externally (e.g., within a Turing model), and stored in `d`
Distributions.logpdf(d::UnknownDistribution, x::Real) = d.summary_statistics_value;

# Bounds
Distributions.minimum(d::UnknownDistribution) = d.summary_statistics_minimum;
Distributions.maximum(d::UnknownDistribution) = d.summary_statistics_maximum;
