# LimitOrderBookABC.jl

Julia package for estimating the parameters for [ABIDES-Markets](https://github.com/jpmorganchase/abides-jpmc-public) with a SMC-ABC sampler.

## Installation

This package uses [AbidesMarkets.jl](https://github.com/fipelle/AbidesMarkets.jl) to load [ABIDES-Markets](https://github.com/jpmorganchase/abides-jpmc-public) within Julia. Therefore, its preliminary setup should be followed.

Having done that, LimitOrderBookABC.jl can then be installed with the Julia package manager via:

```julia
using Pkg
Pkg.add(url="https://github.com/fipelle/LimitOrderBookABC.jl")
```
