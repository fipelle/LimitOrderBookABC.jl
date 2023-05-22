module StaticSMC

    # Dependencies
    using Distributions, Distances, LinearAlgebra, Random, StaticArrays;
    using AbidesMarkets: SnapshotL2;
    using StatsBase: mean, cov, weights;
    using Infiltrator;
    
    # Local Dependencies
    include("static_smc_types.jl");
    include("static_smc_sampler.jl");

    # Exports
    export sample!, ParticleSystem;
end
