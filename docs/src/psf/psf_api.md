```@meta
CurrentModule = CrowdPhot.PSF
```

# PSF API

## Abstract Interface

All PSF models in CrowdPhot are subtypes of `AbstractPSFModel{T}`. The
abstract interface defines the operations that every model must (or should)
support.

```@docs
AbstractPSFModel
```

### Core Evaluation

```@docs
evaluate
evaluate_fg
```

!!! note
    Note that by default it is assumed that PSF models have `evaluate` methods that are
    SIMD vectorizable via [LoopVectorization.@turbo](https://github.com/JuliaSIMD/LoopVectorization.jl).
    Specific model types can opt-out by declaring `CrowdPhot.PSF._turbo_safe(::Type{MyType}) = false`;
    this is presently used by the [`ImagePSF`](@ref) type as an example, due to its array gathers
    currently failing on architectures with narrow lane widths (e.g., 128 bit on Apple silicon)
    when used with `@turbo`.
    The functions [`add_star!`](@ref), [`subtract_star!`](@ref), and [`render`](@ref) internally use
    `_turbo_safe` to choose between executing loops declared with `@turbo` and basic `@simd` loops.
    The most common cases of such failures are PSF models that use special functions that do not have
    vectorized versions that LoopVectorization.jl can substitute in.
    For an example of how to support such PSFs, see [`AiryPSF`](@ref) which uses bessel functions.
    In this case, we took pure-Julia bessel function implementations from
    [Bessels.jl](https://github.com/JuliaMath/Bessels.jl) and adapted them to make them branchless,
    making them safe for SIMD vectorization through `@turbo`.

### Model Properties

```@docs
centroid
integral
background
peak
amplitude
effective_area
fwhm
theta
```

### Spatial Utilities

```@docs
extent
ellipse_bounds
render
add_star!
subtract_star!
pixel_response_kernel
```

## API Internals

```@docs
free_params
model_from_vector
_turbo_safe
```
