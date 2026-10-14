```@meta
CurrentModule = CrowdPhot
```

# Detection and Centroiding

## Matched Filter Detection

CrowdPhot implements the formally correct matched filter for point-source
detection under stationary (uncorrelated) Gaussian noise.

### Mathematical Derivation

Consider an image ``D(x,y)`` consisting of a uniform background ``B``, a
point source with flux ``F`` and PSF shape ``P(x,y)`` (normalized such that
``\sum P = 1``), and independent Gaussian noise
``N(x,y) \sim \mathcal{N}(0, \sigma^2)``:

```math
D(x,y) = B + F \cdot P(x - x_0, y - y_0) + N(x,y)
```

At each position we test the hypothesis ``\mathcal{H}_1`` (source present
with flux ``F``) versus ``\mathcal{H}_0`` (no source).  The log-likelihood
ratio for Gaussian noise is

```math
\ln\Lambda = \frac{F}{\sigma^2} \sum_i P_i (D_i - B)
           - \frac{F^2}{2\sigma^2} \sum_i P_i^2
```

Maximizing with respect to ``F`` gives the optimal (matched-filter) flux
estimator:

```math
\hat{F} = \frac{\sum_i P_i (D_i - B)}{\sum_i P_i^2}
```

This expression requires the background ``B`` to be known and subtracted
beforehand.

In practice the image may not be background-subtracted.  Rather than
estimating ``B`` explicitly, we can fold the cancellation of a
**constant** background into the kernel itself by enforcing
``\sum K_i = 0``.  This removes any offset that is **uniform** across the
kernel footprint.  It does *not* remove gradients, curvature, or other
small-scale background structure; a proper background subtraction
should still be performed before detection.  Define the **zero-sum
kernel**

```math
K_i = \frac{P_i - \bar{P}}{\mathrm{denom}}, \qquad
\bar{P} = \frac{1}{N}\sum_j P_j, \qquad
\mathrm{denom} = \sum_j P_j^2 - \frac{(\sum_j P_j)^2}{N}
```

This kernel satisfies ``\sum K_i = 0`` and
``E[\sum K_i \cdot (F P_i)] = F``.  Consequently the correlation
``S = K \star D``, evaluated at a source position, is the matched-filter
flux estimate after profiling out an unknown constant background over the
kernel footprint.  This is not algebraically identical to the known-background
estimator above, but it has the same expected response for an isolated source
with template ``P``, albeit with a different noise realization and a larger
variance (see below for additional information).

The **detection statistic** evaluated at every pixel ``(x,y)`` is the
correlation response divided by its noise standard deviation:

```math
z(x,y) = \frac{\sum_{i,j} K_{i,j} \, D_{x+i,\,y+j}}
               {\sigma \;/\; \sqrt{\mathrm{denom}}}
       = \frac{S(x,y) \, \sqrt{\mathrm{denom}}}{\sigma},
```

where ``S = K \star D`` is the correlation of the image with the
normalized kernel.  This ``z(x,y)`` is the true matched-filter SNR when
the pixel noise ``\sigma`` is known.

In the uniform-weight code path (`inv_var = nothing`), [`matched_filter`](@ref)
does not take a scalar noise estimate.  It stores and thresholds
``S\sqrt{\mathrm{denom}}``, which equals ``\sigma z``.  Divide that map by
your pixel-noise estimate, or provide `inv_var = fill(1/sigma^2, size(image))`,
to obtain a map in standard-deviation units.

**Why this statistic is meaningful:**  Under the null hypothesis
``\mathcal{H}_0`` (no source present), ``D = B + N`` and the
zero-sum kernel cancels the background, leaving only the noise term
``\sum K_{i,j} N_{x+i,y+j}``.  Since each ``N \sim \mathcal{N}(0,\sigma^2)``
and the kernel weights are fixed,

```math
z \mid \mathcal{H}_0 \sim \mathcal{N}(0, 1).
```

A threshold of ``z = 5`` in the true SNR map therefore corresponds to a false-positive
probability of ``\sim 3\times10^{-7}`` per independent resolution
element (one-sided Gaussian tail).

Under ``\mathcal{H}_1`` (source with flux ``F`` present at that pixel),
the expectation shifts to
``E[z \mid \mathcal{H}_1] = F \sqrt{\mathrm{denom}} \,/\, \sigma``,
which is the true source signal-to-noise ratio.  Sources with
``F \gg \sigma / \sqrt{\mathrm{denom}}`` produce large ``z`` values
and are detected; fainter sources fall below the threshold.

For **spatially varying noise** described by an inverse-variance map
``w_i = 1/\sigma_i^2``, two correlations are required:

```math
\mathrm{num}(x,y) = \sum_{i,j} K_{i,j} \, w_{x+i,\,y+j} \, D_{x+i,\,y+j}
```
```math
\mathrm{den}(x,y) = \sum_{i,j} K_{i,j}^2 \, w_{x+i,\,y+j}
```
```math
\mathrm{SNR}(x,y) = \frac{\mathrm{num}(x,y)}
                          {\sqrt{\mathrm{den}(x,y)}}
```

This is the statistic computed by the current fixed-kernel weighted path.
It is calibrated under the null hypothesis when the image is already
background-subtracted.

!!! note "Kernel Reweighting for Spatially-Variable Weights"
    If an unknown constant background is also being
    profiled out while the weights vary across
    the footprint, the background projection should be weighted locally:

    ```math
    \bar{P}_w(x,y) =
    \frac{\sum_{i,j} w_{x+i,\,y+j} P_{i,j}}
        {\sum_{i,j} w_{x+i,\,y+j}},
    \qquad
    \mathrm{denom}_w(x,y) =
    \sum_{i,j} w_{x+i,\,y+j}
    \left(P_{i,j} - \bar{P}_w(x,y)\right)^2.
    ```

    The corresponding profiled-background statistic is

    ```math
    z_w(x,y) =
    \frac{\sum_{i,j} w_{x+i,\,y+j}
        \left(P_{i,j} - \bar{P}_w(x,y)\right)
        D_{x+i,\,y+j}}
        {\sqrt{\mathrm{denom}_w(x,y)}}.
    ```

    When the sky background dominates the noise and is approximately constant,
    ``w`` is approximately constant over the kernel footprint, so this reduces to
    the zero-sum kernel above.  If ``w`` varies strongly, ``\sum K_i = 0`` alone
    does not cancel a constant background inside ``\sum K_i w_i D_i``. For
    simplicity and speed we **do not** apply this kernel reweighting, which means
    that if you have image with significant spatial variation in the noise properties,
    you should **always** model and subtract the background before running detection.

The zero-sum kernel is the default and the recommended choice: even on
a background-subtracted image, ``\sum K_i = 0`` cancels any residual
uniform offset.  For an isolated source matched by ``P``, the expected flux
response remains unbiased, although the form of the estimator changes slightly
and the variance is slightly larger.  The
`normalize_zerosum` parameter can be set to `false` to skip the
zero-sum correction, which yields marginally lower noise variance at
the cost of sensitivity to imperfect background subtraction.

The zero-sum constraint carries a small noise penalty.  The variance
of the correlation response is ``\sigma^2/\mathrm{denom}`` rather than
the ``\sigma^2/\sum P^2`` that would be achieved without the zero-sum
correction.  The SNR ratio is

```math
\frac{\mathrm{SNR}_\mathrm{zs}}{\mathrm{SNR}_\mathrm{nzs}}
 = \sqrt{1 - \frac{(\sum P)^2}{N\sum P^2}}.
```

The penalty depends on the PSF width relative to the kernel truncation
radius, not on the pixel count alone.  For a Gaussian PSF with width
``\sigma`` truncated at radius ``R``, the ratio is approximately
``\sqrt{1 - 4\sigma^2/R^2}``.  At ``R = 4\sigma`` (typical) the
penalty is ~13%; at ``R = 3\sigma`` it is ~25%.  Only for very
generously padded kernels (``R \gtrsim 10\sigma``) does it drop
below a few percent.

### Recommended configurations

| Image state | `inv_var` | `normalize_zerosum` | Rationale |
|---|---|---|---|
| Not background-subtracted, noise level unknown | `nothing` | `true` (default) | Zero-sum kernel cancels a uniform background automatically.  The significance map is in units of the (unknown) pixel-noise σ (RMS).  To renormalize it after calculation, divide by your σ estimate to obtain true SNR in standard deviations. |
| Not background-subtracted, uniform noise σ known | `fill(1/σ², size(image))` | `true` (default) | Zero-sum kernel cancels a uniform background automatically.  Significance map is in standard-deviation units (∼N(0,1) under the null). |
| Not background-subtracted, varying noise | — | — | Background-subtract the image first.  With an un-subtracted background and varying `inv_var`, neither kernel normalization is correct: `true` leaks background through ΣKᵢwᵢ ≠ 0, and `false` leaks through ΣKᵢ ≠ 0.  After subtraction, use the "Background-subtracted, varying noise" row below. |
| Background-subtracted, uniform noise σ known | `fill(1/σ², size(image))` | `false` | No residual background remains, so the zero-sum correction carries a variance penalty with no benefit.  Significance map is in standard-deviation units (∼N(0,1) under the null).  `true` also works but is noisier. |
| Background-subtracted, varying noise | `Matrix` | `false` | The known-background estimator (K = P/ΣP²) is correct for any weight map.  `true` also works but carries the zero-sum variance penalty with no benefit.  Significance map is in standard-deviation units (∼N(0,1) under the null). |

Background and RMS error estimates can be measured by tools described in the [`Background`](@ref bkg)
section.

### The `rel_err` term

The DAOPHOT family of codes [Stetson1987](@cite) -- and
[photutils'](https://github.com/astropy/photutils)
`DAOStarFinder`, which follows the same formalism -- define

```math
\mathrm{rel\_err} = \frac{1}{\sqrt{\mathrm{denom}}},
\qquad
\mathrm{denom} = \sum_i P_i^2 - \frac{(\sum_i P_i)^2}{N}.
```

This quantity is formally correct for matched-filter detection.
It arises directly from the variance of the convolved image under
``\mathcal{H}_0``.  With the zero-sum kernel
``K_i = (P_i - \bar{P})/\mathrm{denom}``,

```math
\operatorname{Var}[K \star D \mid \mathcal{H}_0]
   = \sigma^2 \sum_i K_i^2
   = \frac{\sigma^2}{\mathrm{denom}}
   = (\sigma \cdot \mathrm{rel\_err})^2.
```

The photutils detection threshold is therefore
``\mathtt{threshold} = n_\sigma \cdot \sigma \cdot \mathrm{rel\_err}``,
which equals ``n_\sigma`` times the standard deviation of the convolved
noise.

CrowdPhot does **not** expose `rel_err` as a separate quantity.
Instead, the ``\mathrm{rel\_err}`` factor is folded into the
`significance_map` computation.  In the uniform-weight path
(`inv_var = nothing`), the map is formed as

```math
\mathtt{significance\_map} = (K \star D) \cdot \sqrt{\mathrm{denom}}
                            = \frac{K \star D}{\mathrm{rel\_err}},
```

which equals the matched-filter SNR up to the unknown factor of
``1/\sigma`` (divide by your noise estimate to obtain the true SNR).
In the weighted path (`inv_var` supplied), the division by
``\sqrt{\mathrm{den}}`` in the expression
``\mathrm{SNR} = \mathrm{num} / \sqrt{\mathrm{den}}`` plays the
same role, with the noise variance coming from the data rather than
from a single scalar ``\sigma``.  In both cases the `rel_err` is
present — it simply appears as ``1/\sqrt{\mathrm{denom}}`` inside
the kernel normalization rather than as a separately stored
parameter.

### API

[`matched_filter`](@ref) is the entry point for matched filter source
detection. The input for the `kernel` argument can be
- a generic `AbstractMatrix` following the conventions expected by [`CrowdPhot.correlate`](@ref);
- an instance of an `AbstractPSFModel` such as a [`MoffatPSF`](@ref);
  the [`render`](@ref) function will be called on the model and the 
  resulting matrix will be used as the kernel.
- an `Int` giving the FWHM of a [`CircularGaussianPRF`](@ref) model to use for the
  correlation;
- a `Tuple{Int, Int}` giving the x- and y-direction FWHMs of a general [`GaussianPRF`](@ref)
  model to use for the correlation.

```@docs
matched_filter
MatchedFilterResult
CrowdPhot.findlocalmaxima
CrowdPhot.correlate
CrowdPhot.correlate!
```

## Centroiding

CrowdPhot provides a fast polynomial centroiding algorithm based on
[Vakili2016](@citet).  The method fits a quadratic 2-D
polynomial to the 3×3 patch surrounding the brightest pixel of a
PSF-correlated (matched-filtered) image.  Under the well-sampled,
approximately Gaussian assumptions studied by Vakili & Hogg, the polynomial
centroid can approach the Cramér-Rao lower bound and is ``\sim\!200`` ns per
source.

Both the polynomial centroid and an inverse-variance-weighted
center-of-mass (COM) centroid are returned, along with their full
covariance matrices and 1-σ uncertainties.  The COM centroid is
effectively free -- all required moments are already computed during
the normal-equation assembly for the polynomial fit.

[`choose_centroid`](@ref) selects between the two estimators
automatically: the polynomial centroid is used for well-sampled data,
while the COM centroid is preferred when the polynomial's curvature
matrix is nearly singular (e.g. for very broad PSFs).

```@docs
centroid_poly
CrowdPhot._centroid_poly3
choose_centroid
```

## References
This page cites the following references:

```@bibliography
Pages = ["detection.md"]
Canonical = false
```
