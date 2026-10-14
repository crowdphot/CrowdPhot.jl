```@meta
CurrentModule = CrowdPhot
```

# Simulation

CrowdPhot provides utilities for generating synthetic images. The
simulation workflow is: generate random source positions and fluxes with
[`simulate_sources`](@ref), then render them through a PSF model and add noise
with [`simulate_image`](@ref).

## Source Generation

```@docs
simulate_sources
```

## Image Rendering

```@docs
simulate_image
```

## Flux from SNR

```@docs
flux_for_snr
```

## Noise Injection

```@docs
add_noise!
```
