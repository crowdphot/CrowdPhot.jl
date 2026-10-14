```@meta
CurrentModule = CrowdPhot
```

# CrowdPhot.jl

CrowdPhot is a Julia package for crowded-field stellar photometry. It provides
tools for point-spread-function (PSF) modeling and fitting, robust
Levenberg-Marquardt optimization, source detection and centroiding, background
estimation, and synthetic-image simulation.

The package is organized around a few composable pieces:

- **PSF models and fitting:** analytic PSF/PRF models, image-backed empirical
  PSFs, rendering utilities, and specialized methods for fast
  single-star least-squares fits.
- **Robust nonlinear least squares:** a Levenberg-Marquardt implementation with
  configurable damping, iteratively reweighted least squares (IRLS), robust
  scale estimators, and covariance estimation.
- **Background estimation:** scalar background/RMS estimates and spatially
  varying 2-D background maps with masking, sigma clipping, robust location
  estimators, and bicubic interpolation.
- **Detection and centroiding:** matched-filter source detection, local-maximum
  finding, and polynomial centroid refinement.
- **Simulation:** source-list generation, PSF rendering, Poisson/read-noise
  injection, and convenience routines for Gaussian-star test images.

## PSF Modeling

CrowdPhot's PSF interface is defined by [`AbstractPSFModel`](@ref). Models
support common operations such as evaluation, gradients, centroids, fluxes,
effective areas, extents, rendering, and image addition/subtraction.

Available parametric models include:

| Model | Description |
|-------|-------------|
| [`CircularGaussianPSF`](@ref) | Circular sampled Gaussian PSF |
| [`GaussianPSF`](@ref) | Elliptical sampled Gaussian PSF with rotation |
| [`CircularGaussianPRF`](@ref) | Circular Gaussian pixel-response function integrated over pixels |
| [`GaussianPRF`](@ref) | Elliptical Gaussian pixel-response function integrated over pixels |
| [`CircularMoffatPSF`](@ref) | Circular Moffat PSF |
| [`MoffatPSF`](@ref) | Elliptical Moffat PSF with rotation |
| [`AiryPSF`](@ref) | Airy-disk PSF parameterized by first-dark-ring radius |
| [`ImagePSF`](@ref) | Empirical image-backed PSF with bicubic interpolation |

All current PSF models that implement analytic gradients have specialized
`fit_star` dispatches. These preserve the generic `fixed`-parameter interface
while using scalar model-specific accumulation kernels for the normal equations.

Empirical PSF support includes [`ImagePSF`](@ref) for tabulated PSFs and
[`fit_psf`](@ref) for building image-backed ePSFs from stellar cutouts.

## Fitting

[`fit_star`](@ref) fits the free parameters of a PSF model to an image cutout.
Parameters can be frozen with the `fixed` keyword, and per-pixel inverse
variance weights can be supplied with `inv_var`.

The underlying LM engine supports:

- Marquardt, Levenberg, and no-damping strategies.
- Optional IRLS reweighting with losses such as [`TukeyLoss`](@ref).
- Robust scale estimators including [`MADScale`](@ref), [`FixedScale`](@ref),
  and [`MScale`](@ref).
- Known-weight and reweighted covariance estimators.

## Image Analysis Tools

CrowdPhot includes image-processing utilities needed before and after PSF
fitting:

- [`estimate_background`](@ref) for scalar background and RMS estimates.
- [`Background2D`](@ref) for mesh-based spatial background models.
- [`matched_filter`](@ref) for point-source detection with uniform or
  inverse-variance weighting.
- [`centroid_poly`](@ref) and [`choose_centroid`](@ref) for subpixel centroid
  refinement.

## Simulation

Simulation utilities are available for tests, examples, and benchmarking:

- [`simulate_sources`](@ref) generates source positions and fluxes with optional
  separation constraints.
- [`simulate_image`](@ref) renders sources through a PSF model and adds noise.
- [`flux_for_snr`](@ref) converts target SNR values into model fluxes.
- `make_gaussians_image` provides a compact convenience interface for
  Gaussian-star images.

## Documentation Map

| Page | Description |
|------|-------------|
| [Background Estimation](background.md) | Scalar and 2-D background/RMS estimation |
| [Detection and Centroiding](detection.md) | Matched-filter detection and centroid refinement |
| [PSF API](psf/psf_api.md) | Abstract PSF interface and shared utilities |
| [Parametric PSF Models](psf/parametric_models.md) | Gaussian, PRF, Moffat, and Airy models |
| [Effective PSF Models](psf/empirical/epsf_overview.md) | Image-backed empirical PSF modeling |
| [ImagePSF](psf/empirical/image_psf.md) | Bicubic interpolation for tabulated PSFs |
| [Levenberg-Marquardt Fitter](lm_fitter.md) | LM optimizer with IRLS robust fitting |
| [Simulation](simulation.md) | Synthetic image and source generation |
| [API Index](doc_index.md) | Complete documented symbol index |
