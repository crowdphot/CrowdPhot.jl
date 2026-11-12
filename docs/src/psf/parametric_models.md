```@meta
CurrentModule = CrowdPhot
```

# Parametric PSF Models

Parametric PSF models represent the point-spread function with analytic
formulas. Each model implements [`evaluate`](@ref) and [`evaluate_fg`](@ref)
(value and gradient), making them efficient for
[Levenberg-Marquardt fitting](@ref lm_fitter).

Models suffixed with "PSF" (e.g. [`GaussianPSF`](@ref)) sample the continuous PSF.
Models suffixed with "PRF" (pixel response function, e.g.
[`GaussianPRF`](@ref)) integrate the PSF over each pixel, modeling the
detector pixel response, which can be important for undersampled imaging.

## Gaussian Models

```@docs
CircularGaussianPSF
GaussianPSF
CircularGaussianPRF
GaussianPRF
```

## Moffat Models

Moffat profiles have heavier wings than Gaussians, often providing a better
match to ground-based imaging.

```@docs
CircularMoffatPSF
MoffatPSF
```

## Airy Model

```@docs
AiryPSF
```

## API Internals

```@docs
CrowdPhot.PSF.GAUSS_PRE
CrowdPhot.PSF.AIRY_RZ
```
