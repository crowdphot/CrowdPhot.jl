```@meta
CurrentModule = CrowdPhot
```

# Centroid Refinement and Morphology

After matched-filter detection identifies candidate sources, two
post-detection steps are needed before PSF fitting: refining sub-pixel
centroids and measuring morphological diagnostics that distinguish
stellar sources from cosmic rays, hot pixels, and extended objects.

## Centroiding

CrowdPhot provides [`centroid_poly`](@ref), a fast polynomial centroiding algorithm
based on [Vakili2016](@citet).  It fits a quadratic 2-D
polynomial to the 3×3 patch surrounding the brightest pixel of a
PSF-correlated image.  Under the well-sampled,
approximately Gaussian assumptions studied by [Vakili2016](@citet), the polynomial
centroid can approach the Cramér-Rao lower bound and takes ``\sim\!200`` ns per
source.  Correlating the observed image with the PSF broadens
source profiles helps to improve centroid estimation for undersampled
images.

Both the polynomial centroid and inverse-variance-weighted
center-of-mass (COM) centroid are returned, along with their full
covariance matrices and 1-σ uncertainties.  The COM centroid is
effectively free -- all required moments are already computed during
the normal-equation assembly for the polynomial fit.

[`choose_centroid`](@ref) selects between the two estimators
automatically: the polynomial centroid is used for well-sampled data,
while the COM centroid is preferred when the polynomial's curvature
matrix is nearly singular (e.g. for very broad PSFs).

In addition to the centroid and its covariance,
[`centroid_poly`](@ref) also returns morphological diagnostics for the
core of the source within the central 3x3 pixel box at near-zero additional
cost (see [Morphological Measurements](#Morphological-Measurements) below):

- `normalized_curvature` -- negated Laplacian divided by the fitted
  amplitude above `background`; ``\approx 16\log(2)/\mathrm{FWHM}^2`` for a
  circular Gaussian, flux-independent.
- `compactness_core` -- inverse of the total second central moment of the light
  distribution; larger for more compact light distributions.
- `ellipticity1_core`, `ellipticity2_core` -- the two normalized quadrupole
  (ellipticity) components of the 3×3 second central moments; both 0 for a
  circular core, ``e_1 > 0`` extended in y, ``e_2 > 0`` extended along the
  ``+45^\circ`` diagonal.

```@docs
centroid_poly
CrowdPhot._centroid_poly3
choose_centroid
```

## Morphological Measurements

[`measure_star_shape`](@ref) computes aperture-scale morphological
diagnostics from inverse-variance-weighted second moments over the full
cutout.  It complements the core diagnostics from
[`centroid_poly`](@ref) with measurements that are integrated over the
entire stellar profile rather than just the 3×3 core. Combining the core
and full-aperture diagnostics provides a multi-scale view of the source's
morphology.

### Shape Components

Two useful shape metrics are the normalized quadrupole components
of the second-moment tensor:

| Statistic | CrowdPhot (core) | CrowdPhot (aperture) | Definition | Meaning |
|-----------|-----------------|---------------------|------------|---------|
| ``e_1`` | `ellipticity1_core` | `ellipticity1_aperture` | ``(\sigma^2_{yy}-\sigma^2_{xx})/(\sigma^2_{yy}+\sigma^2_{xx})`` | 0 = circular, negative = extended in x (columns), positive = extended in y (rows) |
| ``e_2`` | `ellipticity2_core` | `ellipticity2_aperture` | ``2\sigma^2_{xy}/(\sigma^2_{yy}+\sigma^2_{xx})`` | 0 = no diagonal power, positive = extended along ``+45^\circ`` |

Both are 0 for an ideal circular source and both lie in ``[-1, 1]``.

``e_1`` and ``e_2`` are the complete linear description of the shape at
quadrupole order.  They are not two independent scalars that happen to be
numbered: they are the components of a single spin-2 object, so a
rotation of the source by ``\varphi`` rotates the vector
``(e_1, e_2)`` by ``2\varphi``.  Elongation at ``45^\circ`` therefore
moves entirely out of ``e_1`` and into ``e_2``, and only the magnitude ``|e|``
is rotationally invariant. **They should therefore be interpreted together.**
Note that they are equivalent to an `ellipticity` plus a position angle:

```math
|e| = \sqrt{e_1^2 + e_2^2}, \qquad
\text{ellipticity} = 1 - \sqrt{\frac{1-|e|}{1+|e|}}, \qquad
\theta = \tfrac{1}{2}\arctan(e_2, -e_1).
```

Note that ``|e| = (a^2-b^2)/(a^2+b^2)`` is the *distortion*, not
the scalar ellipticity ``1 - b/a``.

Because these are ratios of *linear* moment sums taken about the center of
mass, they are also the shape statistics least sensitive to where the
source falls inside its pixel; see
[Sub-pixel phase](#Sub-pixel-phase) below.

For elongation along a row or column, or a one-sided spike, ``e_1``
responds and its sign gives the direction of the stretch.  A source
elongated at 45° keeps the marginal widths
``\sigma^2_{xx} \approx \sigma^2_{yy}`` equal, so ``e_1`` stays near zero
while the cross moment ``\sigma^2_{xy}``, and therefore ``e_2``, becomes
large.  Isolated off-axis hot pixels behave the same way: they inflate
``\sigma^2_{xx}`` and ``\sigma^2_{yy}`` about equally but register in
``e_2`` unless they happen to lie on a row or column through the center.
The two components together register elongation at any position angle;
their magnitude ``|e|`` is discussed below.

Departures from a Gaussian that *preserve* fourfold symmetry, e.g. a
faint ring, leave ``e_1``, ``e_2``, and ellipticity all near zero, so
none of these statistics flags them.  Such issues are better identified
by a PSF-fit residual.

!!! note "Correspondence with DAOPHOT and photutils"
    The ellipticity components `e_1` and `e_2` are closely related to
    the DAOPHOT SROUND and GROUND statistics (photutils `roundness1` and
    `roundness2`), but they are not numerically equivalent:

    | CrowdPhot | DAOPHOT | photutils | Shape component |
    |---|---|---|---|
    | ``e_1`` (`ellipticity1_*`) | GROUND | `roundness2` | axis-aligned elongation |
    | ``e_2`` (`ellipticity2_*`) | SROUND | `roundness1` | diagonal (``45^\circ``) elongation |

    **The numbering is crossed.** SROUND (`roundness1`) measures diagonal
    asymmetry and therefore corresponds most closely to ``e_2``, whereas
    GROUND (`roundness2`) compares the axis-aligned widths and therefore
    corresponds most closely to ``e_1``.

    There are several important differences between the statistics:

    - **Range.** The DAOPHOT roundness statistics use a normalization with
      values spanning approximately ``[-2, 2]``. The ellipticity components
      ``e_1`` and ``e_2`` lie in ``[-1, 1]``, with
      ``|e| = \sqrt{e_1^2 + e_2^2}`` giving the rotationally invariant
      distortion.
    - **Parameterization.** GROUND compares axis-aligned widths,
      approximately
      ``2(\sqrt{\sigma^2_{yy}}-\sqrt{\sigma^2_{xx}})/
      (\sqrt{\sigma^2_{yy}}+\sqrt{\sigma^2_{xx}})``, whereas ``e_1``
      compares the corresponding second moments directly. The two are
      monotonically related for an axis-aligned elliptical source but are
      not numerically equal.
    - **Estimator.** SROUND is based on a sign-quadrant comparison of the
      image around the source center, while GROUND is based on fitted
      curvature along the coordinate axes. In contrast, ``e_1`` and
      ``e_2`` are components of the second-moment tensor. They therefore
      provide a unified spin-2 description of quadrupole shape rather than
      two separately defined roundness statistics.

### Normalized Curvature

The `normalized_curvature` field returned by [`centroid_poly`](@ref)
is the negated Laplacian of the quadratic fit divided by the fitted
amplitude above the background.  For a Gaussian this approximates
``16\log(2)/\mathrm{FWHM}^2`` and is independent of flux, making it a fast
discriminator: cosmic rays (single bright pixels) produce larger values
than stellar PSFs of the expected width. Saturated stars produce lower values
because their cores are flat.

### Compactness

`compactness_core` (from [`centroid_poly`](@ref)) and
`compactness_aperture` (from [`measure_star_shape`](@ref)) are the
inverse of the total weighted second central moment of the light
distribution:

```math
\text{compactness} = \frac{1}{\sigma_x^2 + \sigma_y^2},
```

where ``\sigma_x^2`` and ``\sigma_y^2`` are the inverse‑variance‑weighted
second central moments of the pixel values about the estimated centroid,
taken over the 3×3 core (`compactness_core`) or the full cutout
(`compactness_aperture`).  For a Gaussian profile this quantity is
proportional to the inverse of the FWHM squared, similar to the
normalized curvature above, but using a moment statistic rather than
the Laplacian. Larger values indicate more compact, sharper profiles.
Cosmic rays and hot pixels, with negligible spatial extent, produce very large values.

!!! note "Subtract background or pass `background` keyword"
    The compactness moments are weighted by the (background-subtracted)
    pixel values, so a sky pedestal that is *not* removed biases them.  A
    flat background has second central moment ``2/3`` per axis on the 3×3
    core, so an un‑subtracted `compactness_core` is a convex combination of
    the true source value and ``2/3``, weighted by the sky-to-source count
    ratio inside the box; for a faint source on a bright sky it collapses
    to ``\approx 0.75`` regardless of the true width.  `normalized_curvature`
    is likewise suppressed by an un‑subtracted pedestal through its
    division by the fitted peak, though its curvature numerator is
    pedestal-independent.  Pass the sky level via the `background` keyword
    of [`centroid_poly`](@ref) (or of [`measure_star_shapes`](@ref), which
    forwards it), or subtract it from the image beforehand.  The polynomial
    centroid itself is unaffected either way.  `compactness_aperture` is
    computed on ``\max(0,\,\text{image} - \text{background})`` and needs the
    same correct `background`.

`compactness_core` returns `NaN` when the weighted moment sum is
non‑positive (`R00 ≤ 0` or ``\sigma_x^2 + \sigma_y^2 \le 0``), which
happens for pure-noise peaks; `compactness_aperture` returns `NaN` under
the analogous degenerate conditions.  This lets non‑stellar detections
be rejected with a simple `isfinite` check.

### Sharpness

[`measure_star_shapes`](@ref) reports the DAOPHOT `SHARP` statistic
[Stetson1987](@citet) (photutils `sharpness`) as `sharpness`:

```math
\mathtt{sharpness} = \frac{D_\mathrm{peak} - \bar{D}_\mathrm{neighbors}}{H},
\qquad H = \mathtt{Flux} \times \max(P)
```

The numerator is measured on the **raw** image over the kernel footprint
with the center pixel excluded; ``H`` is the source's fitted central
height, obtained by scaling the flux estimate by the peak
pixel fraction of the PSF template ``P``.

It measures how much of the flux sits in one pixel relative to what the
PSF would put there.  Large values indicate a cosmic ray or hot pixel,
small values a blend or a resolved source, and stars cluster tightly in
between (``\approx 0.9`` for a source matched to the template).

Two practical notes.  A spatially flat background cancels from the
numerator, so `sharpness` needs no `background` argument and is unbiased
by an un-subtracted sky, unlike `normalized_curvature`, which is
suppressed by a pedestal through its division by the fitted peak.  And
`sharpness` requires no quadratic fit, so it remains available where that fit
degenerates.

The `compactness_core` and `normalized_curvature` fields remain
complementary to it: they are available immediately during centroid
refinement from [`centroid_poly`](@ref) alone, with no reference to the
detection kernel.

### Sub-pixel phase

A statistic computed from pixel samples of a star can depend on where the
star happens to fall inside its peak pixel.  This is a systematic, not a
noise term: it does not shrink as sources get brighter, and it sets a
floor on how tight a stellar locus can be.

The aperture statistics are ratios of *linear* moment sums taken about
the center of mass.  The reference point ``(y_0, x_0)`` therefore cancels
identically -- passing the integer peak pixel costs nothing -- and Poisson
summation bounds the residual phase dependence at roughly
``e^{-2\pi^2\sigma^2}`` with ``\sigma`` the profile width in pixels.
Measured on noiseless round sources scanned over many sub-pixel phases:

| statistic | FWHM 1.2 px | FWHM 1.6 px | FWHM 2.0 px | FWHM 3.5 px |
|---|---|---|---|---|
| `fwhm`, `compactness_aperture`, `ellipticity1_aperture`, `ellipticity2_aperture` (fractional scatter) | ``10^{-2}`` | ``2\times10^{-4}`` | ``<10^{-6}`` | ``<10^{-6}`` |
| `normalized_curvature` (fractional scatter) | 0.092 | 0.079 | 0.070 | 0.036 |
| ``\|e\|`` over the 3×3 core (*mean*, for a circular source) | 0.011 | 0.024 | 0.017 | 0.003 |
| `poly` centroid (px) | 0.079 | 0.051 | 0.035 | 0.013 |

The split is structural.  Everything derived from the 3×3 *quadratic fit*
-- `normalized_curvature` and the `poly` centroid -- is affected, because a
parabola fit over ``\pm1`` pixel recovers the profile curvature
at the sampling phase ``\phi`` rather than at the peak.  Both the
Laplacian and the fitted amplitude vary with ``\phi``, so renormalizing
does not help.

Practical guidance:

- Prefer the aperture statistics for any absolute threshold.
- Use the quadratic-fit core diagnostics as relative, sigma-clipped
  quantities within a magnitude bin -- which is what
  [`CrowdPhot.PSF.pick_psf_stars`](@ref) does == never against a fixed cut.
- Treat all core morphology as unreliable on undersampled data.
- `poly.y_err` / `poly.x_err` propagate pixel noise only and do not
  include the phase term in the table above, so for bright stars the
  quoted uncertainty understates the true error. Internally CrowdPhot uses
  the quadratic centroid only as an initial guess for PSF-fitting photometry,
  so the error is never used for anything. See [`choose_centroid`](@ref).

### Core vs. Aperture Measurements

| Aspect | Core (`centroid_poly`) | Aperture (`measure_star_shape`) |
|--------|----------------------|--------------------------------|
| Scale | 3×3 central patch | Full cutout |
| Cost | ~200 ns (free with centroid) | ~1.5 μs (21×21), scales with pixel count |
| Shape accuracy | Limited by 3×3 sampling; ``e_1``/``e_2`` saturate for broad PSFs | Integrates over full profile |
| FWHM | Not available (use `compactness_core`) | Marginal moment widths (Gaussian approx.) |
| Compactness / ellipticity | From the 3×3 moment tensor | From the full-profile moment covariance |
| Background | `background` kwarg (centroid unaffected; needed for the diagnostics) | `background` kwarg (subtracted; residuals kept signed) |
| Best use | Fast pre-filter at detection time | Candidate evaluation before PSF fitting |

The returned `fwhm.y` and `fwhm.x` are row/column marginal widths, not
principal-axis widths; for a rotated source, use `fwhm.theta` only as the
covariance major-axis orientation.

### Windowed aperture moments

An unweighted second moment over a fixed box is dominated by sky noise
which grows with the box area. This can be mitigated by tapering the moments
by a Gaussian window, which bounds the noise growth.

The taper is matched to the PSF, which is the width that maximizes sensitivity
to a small departure from the PSF. [`measure_star_shapes`](@ref) derives it from
the detection kernel automatically, and
`fit_all_stars_simultaneous_multipass` from its PSF model;
[`measure_star_shape`](@ref) defaults to [`FlatWindow`](@ref) (no taper) since it
has no PSF to measure against. Pass `window` to override.

Two practical consequences:

- **`half_width` no longer needs to be kept tight.** It only has to be wide
  enough to contain the source; the window handles the sky pixels. Sizing the
  box to exclude a *neighbor* is a separate problem, and still unsolved -- see
  below.
- **Use background-only inverse-variance weights.** A map that includes the
  source's own Poisson noise down-weights the bright core, and the reported
  `fwhm` then grows with flux. Total inverse variance is right for
  `aperture_sum_err` and for PSF fitting, not for anything built from weighted
  moments.

The window compresses the moments by a known factor, which is divided back out
internally, so `fwhm`, `fwhm.theta`, the ellipticity components and
`compactness_aperture` are absolute either way. For a non-Gaussian profile the
recovered moments are Gaussian-equivalent rather than exact, which is a fixed
offset rather than a flux-dependent bias.

```@docs
AbstractMomentWindow
FlatWindow
GaussianWindow
```

### Crowded fields

Every aperture statistic above is a sum over an aperture, so in a crowded field it
sums light from more than one star.  At a separation of 1.5 FWHM a circular
Gaussian measured with `half_width = 7` reads back an ``x`` FWHM twice its true
value and ``e_1 = -0.60``; the shape being reported is the pair's, not the
star's.  No choice of `half_width` fixes this, because shrinking the box to
exclude the neighbor also truncates the profile being measured.

The fix is to measure on the residual.  After PSF-fitting photometry the
residual image has every fitted source subtracted, so adding one source's model
back reconstructs that source alone:

```math
\mathtt{clean} = \mathtt{residual} + \mathtt{render}(\mathtt{model}_j)
```

[`measure_star_shape_ref`](@ref) takes that reconstructed cutout together with
the render it was built from, and returns the same fields as the detection-time
method.  `fit_all_stars_simultaneous_multipass` does this for its whole catalog
in its finalization pass, filling the `morphology` field of the returned
[`MultiPassPhotResult`](@ref).  On the pair above it recovers the true FWHM and
``e_1 \approx 0`` for both stars.

The inverse variance passed alongside should remain the variance of the **data**,
not of the isolated star.  Subtracting a neighbor's model removes its signal but
not the Poisson noise of the photons it contributed, so the original variance is
still correct for `clean`.

### PSF-normalized statistics

Every statistic on this page carries two nuisance dependences: the source's
sub-pixel phase, and the width of the PSF itself. When a PSF model is available,
both can be removed.  [`measure_star_shape_ref`](@ref) recomputes every
statistic on the noiseless render, at the same sub-pixel phase, with the same
weights, anchor pixel, `background` and height normalization, and returns it as
`psf_ref`.  Both halves come from that one function so that the measurement and
its reference cannot be given different weights, anchors or normalizations:

```julia
r = res.phot.morphology[i]
r.sharpness / r.psf_ref.sharpness # 1 for a PSF-like source
r.core.normalized_curvature / r.psf_ref.core.normalized_curvature
r.aperture.compactness_aperture / r.psf_ref.aperture.compactness_aperture
r.aperture.ellipticity1_aperture - r.psf_ref.aperture.ellipticity1_aperture
```

For a noiseless PSF-like source **every ratio is exactly 1 and every difference
exactly 0**, by construction.  On a source that is *not* the PSF, the
normalization suppresses phase-induced scatter by a factor of 2 to 16, largest
on undersampled data, and compresses the PSF-width dependence by roughly 2.5x,
Such PSF-corrected quantities make star/galaxy separation cuts on the morphological
parameters portable across images with different PSFs.

They have additional value for quantifying spatial PSF irregularities. A spatial trend
across an image in a quantity like
`r.aperture.compactness_aperture / r.psf_ref.aperture.compactness_aperture`
indicates there is spatial variation in the image PSF that your PSF model is not
taking into account.

The right comparison differs by quantity, so the components are returned rather
than pre-formed combinations:

| statistic | comparison | PSF-like value |
|---|---|---|
| `sharpness`, `normalized_curvature` | ratio | 1 |
| `compactness_core`, `compactness_aperture` | ratio | 1 |
| `fwhm.y`, `fwhm.x` | ratio | 1 |
| `ellipticity1_*`, `ellipticity2_*` | difference | 0 |
| `ellipticity_*` | rebuild from the differenced components | 0 |
| positions, `aperture_sum`, `fwhm.theta`, metadata | none | -- |

Ellipticity components take a difference, not a ratio: they are signed and pass
through zero, so a ratio is unbounded for a round PSF.

Note that `psf_ref` is `nothing` on the `MatchedFilterResult` method, where no PSF model
exists.

These comparisons **nearly always** what you want to use for distinguishing between
sources that are star-like and those that are not, as they have been corrected for the true
PSF morphology at the location of the source. We return raw values mainly to enable spot-checking
of the PSF model: a coherent pattern across the field in raw `ellipticity1_aperture` that
disappears in the differenced version means the model is capturing that variation, while
structure remaining in the differenced version means it is not.

### Moment Normalization and Aperture Sums

The aperture result includes `moment_norm`, the weighted zeroth moment
``M_{00}`` used internally to normalize centroids, FWHM estimates, and
aperture shape components.  This value follows the same weighting and masking as
the shape moments.  If `inv_var` is non-uniform, `moment_norm` is not a
physical source flux and should not be used for magnitude calibration or
as a photometric prior.

The result also includes `aperture_sum`, `aperture_area`, and
`aperture_sum_err`.  These are quick rectangular-cutout diagnostics over
unmasked pixels (`inv_var > 0`). `aperture_sum` is the unweighted sum of
`image - background`, `aperture_area` is the number of contributing
pixels, and `aperture_sum_err` is the formal propagated uncertainty
``\sqrt{\sum 1/\mathtt{inv\_var}}`` under the assumption of independent
pixel errors.  Mask invalid pixels by setting their inverse variance to
zero.  They are useful flux proxies, but are not aperture-corrected and
should not be used for anything requiring precision.

```@docs
measure_star_shape
measure_star_shape_ref
measure_star_shapes
```

## References
This page cites the following references:

```@bibliography
Pages = ["morphology.md"]
Canonical = false
```
