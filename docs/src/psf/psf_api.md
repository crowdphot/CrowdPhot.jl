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
```

## API Internals

```@docs
free_params
model_from_vector
```
