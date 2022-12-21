module StaticSMC

    # Dependencies
    using Distributions, Distances, Random, Statistics;

    # Local Dependencies
    include("static_smc_types.jl");
    include("static_smc_sampler.jl");

    # Exports
    export sample;
end