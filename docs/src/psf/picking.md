```@meta
CurrentModule = CrowdPhot
```

# Picking PSF Stars

[`CrowdPhot.PSF.pick_psf_stars`](@ref) selects stars suitable for empirical PSF
construction from the morphological measurements returned by
[`measure_star_shapes`](@ref).

## Algorithm

The selection proceeds in three stages:

1. **Magnitude clipping.** The `mag_quantiles` keyword argument will filter
   the list of stars and include only stars in the quantile range between
   `mag_quantiles[1]` and `mag_quantiles[2]`.  In the default configuration,
   this does nothing as `mag_quantiles = (0.0, 1.0)`, but it is provided
   in the case you want to guarantee you do not draw the brightest or
   faintest stars.

2. **Core curvature constraint.** Stars with `core.normalized_curvature`
   outside `[curvature_min, curvature_max]` are rejected.  The normalized
   curvature is the negated Laplacian of the 3×3 core divided by the fitted
   amplitude above the background -- it is independent of flux for an
   unsaturated PSF and depends only on the PSF shape.

3. **Sigma clipping by magnitude bin.** Stars are partitioned into `nbins`
   instrumental-magnitude bins, and within each bin sequential sigma-clipping
   is applied to `fwhm.y`, `fwhm.x`, `ellipticity1_aperture`,
   `ellipticity2_aperture`, and `normalized_curvature`.  Only stars within
   `σ_low`--`σ_high` standard deviations of the clipped median for each
   parameter are retained.

## Why curvature detects saturation

For an unsaturated star, the core of the PSF is peaked — the second
derivative is negative, making the negated Laplacian positive.  A saturated
star has a flat-topped core where the Laplacian is approximately zero.
The normalized curvature cleanly separates the two
regimes independent of the star's brightness or the detector gain.

The table below shows `normalized_curvature` for a Gaussian PSF
(FWHM = 2 pix) at various degrees of saturation.  Clipping is applied as a
hard ceiling on the pixel values to simulate saturation.

| Saturation level | `normalized_curvature` |
|---|---|
| Unsaturated (any peak) | 1.5 |
| Mild saturation (50% of peak) | 1.09 |
| Moderate saturation (20% of peak) | ~0.0 |
| Heavy saturation (≤ 10% of peak) | ~0.0 |
| Cosmic ray (single bright pixel) | 2.4 |

On the HST ACS/WFC F814W DRC image used in the tutorials, the curvature
distribution has median 1.25 and IQR 1.06–1.40.  Only 0.37% of sources have
curvature ≤ 0, and 0.77% have curvature ≤ 0.1.  A bright-end magnitude
percentile clip (5th percentile) excludes 3,609 stars, of which none
have curvature ≤ 0 -- the curvature cut alone correctly identifies saturated
stars without discarding good bright stars that are valuable for measuring
the PSF wings.

## API

```@docs
CrowdPhot.PSF.pick_psf_stars
```
