# morphology.jl — morphological measurements for stellar image cutouts.
#
# Provides `measure_star_shape` for aperture-based FWHM, ellipticity, and
# position angle from inverse-variance-weighted second moments.  The
# 3×3 core sharpness and ellipticity are available from `centroid_poly`
# (see centroids.jl).

# ---------------------------------------------------------------------------
# Moment windows
# ---------------------------------------------------------------------------

"""
    AbstractMomentWindow

A multiplicative weight ``g(y, x)`` applied to the moment accumulators in
`_moments2`, on top of `inv_var`.

Unweighted second moments over a fixed box are unusable in the presence of
noise: the noise contribution accumulates as ``\\sum r^2``, so it grows with
the box area and swamps the source.  A smooth taper bounds it.  See
[`GaussianWindow`](@ref) for the width that follows from that, and
[`FlatWindow`](@ref) for the untapered limit.

Concrete subtypes implement

- `yfactor(w, dy)`, `xfactor(w, dx)`: the separable factors of ``g`` at integer
  offsets from the anchor pixel, returning zero outside the window's support.
- `inv_window_var(w)`: ``1/\\sigma_w^2``, zero for an infinitely wide window.

`deconvolve_moments` needs no per-type method: it is written once in terms of
`inv_window_var`.

Dispatch is the point of the hierarchy: `_moments2` is specialized on the
window type, so [`FlatWindow`](@ref) folds its factors to `one` at compile
time and reduces to an untapered loop with no per-pixel branch.
"""
abstract type AbstractMomentWindow end

"""
    FlatWindow()

The untapered limit, ``g \\equiv 1``: moments are weighted by `inv_var` alone.

Formally the ``\\sigma_w \\to \\infty`` limit of [`GaussianWindow`](@ref), so
`inv_window_var` is zero, which makes `deconvolve_moments` the identity.
The factors return `Bool` `true` so that multiplying
through preserves the caller's float type exactly.

Correct only where the source fills the box: the noise term the window exists
to bound is unbounded here, growing with box area.  Retained because it is the
historical behaviour, because the rectangular `aperture_sum` diagnostics are
defined on it, and because it is the right choice when the moments are wanted
over a deliberately compact cutout.
"""
struct FlatWindow <: AbstractMomentWindow end
@inline yfactor(::FlatWindow, ::Integer) = true
@inline xfactor(::FlatWindow, ::Integer) = true

"""
    inv_window_var(w::AbstractMomentWindow)

``1/\\sigma_w^2`` for the window, zero for [`FlatWindow`](@ref).
"""
@inline inv_window_var(::FlatWindow) = false


@doc raw"""
    GaussianWindow(fwhm::Real; hw::Integer = ...) -> GaussianWindow

A circular Gaussian taper of the given FWHM, centered on the anchor pixel,
tabulated separably as `gy`/`gx` over integer offsets `-hw:hw`.
Non-finite or non-positive `fwhm` throws an `ArgumentError`.

# Width

For a Gaussian source ``\sigma_s`` and Gaussian window ``\sigma_w`` the
windowed second moment is
``\sigma^{-2}_\mathrm{meas} = \sigma^{-2}_s + \sigma^{-2}_w``, so the window
compresses the response by ``[\sigma_w^2/(\sigma_s^2+\sigma_w^2)]^2`` while
suppressing the noise faster.  Maximizing the signal-to-noise on a *small*
departure in size, evaluated at the source scale, gives a figure of merit
``\propto u^{3/2}/[(1+u)^2\sqrt{u^2+1}]`` with ``u = \sigma_w^2/\sigma_s^2``.
That function is invariant under ``u \to 1/u``, so its unique interior maximum
is the fixed point ``u = 1``:

**The sensitivity-optimal window is matched to the scale being measured**,
``\sigma_w = \sigma_s``.  For star-likeness that is the PSF scale, hence
`GaussianWindow(psf_fwhm)`.  The optimum is flat -- a factor ``\sqrt 2`` either
way costs 18% -- so the width needs no per-field tuning.  Weak-lensing
adaptive-moments schemes are the generalization of this, for when each
object has a different optimum scale.

# Support

`hw` defaults to ``\lceil 5\sigma_w \rceil``, where the taper has fallen to
``< 4\times10^{-6}``.  Offsets beyond the table return exactly zero.
"""
struct GaussianWindow{T} <: AbstractMomentWindow
    gy::Vector{T}
    gx::Vector{T}
    hw::Int
    inv_var_w::T
end

function GaussianWindow{T}(fwhm, hw) where {T}
    (isfinite(fwhm) && fwhm > 0) ||
        throw(ArgumentError("`fwhm` must be finite and positive; got $fwhm"))
    hw >= 1 || throw(ArgumentError("`hw` must be at least 1; got $hw"))
    sigma = T(fwhm) / T(2 * sqrt(2 * log(2)))
    n = 2 * Int(hw) + 1
    g = Vector{T}(undef, n)
    for (k, d) in enumerate(-Int(hw):Int(hw))
        g[k] = exp(-T(d)^2 / (2 * sigma * sigma))
    end
    # `gy` and `gx` are equal for a circular window, but kept as separate
    # vectors so an anisotropic window can be added without touching callers.
    return GaussianWindow{T}(g, copy(g), Int(hw), inv(sigma * sigma))
end

function GaussianWindow(fwhm::Real; hw::Union{Nothing, Integer} = nothing)
    (isfinite(fwhm) && fwhm > 0) ||
        throw(ArgumentError("`fwhm` must be finite and positive; got $fwhm"))
    T = float(typeof(fwhm))
    sigma = T(fwhm) / T(2 * sqrt(2 * log(2)))
    h = hw === nothing ? max(1, ceil(Int, 5 * sigma)) : Int(hw)
    return GaussianWindow{T}(fwhm, h)
end

@inline function yfactor(w::GaussianWindow{T}, d::Integer) where {T}
    k = Int(d) + w.hw + 1
    return (1 <= k <= length(w.gy)) ? (@inbounds w.gy[k]) : zero(T)
end
@inline function xfactor(w::GaussianWindow{T}, d::Integer) where {T}
    k = Int(d) + w.hw + 1
    return (1 <= k <= length(w.gx)) ? (@inbounds w.gx[k]) : zero(T)
end

@inline inv_window_var(w::GaussianWindow) = w.inv_var_w

@doc raw"""
    deconvolve_moments(w, sig2_yy, sig2_xx, sig2_xy) -> (; yy, xx, xy)

Undo a window's known bias on a measured second-moment tensor, recovering the
source's own moments:

```math
\Sigma_s = \left(\Sigma_\mathrm{meas}^{-1} - \sigma_w^{-2} I\right)^{-1}
```

Exact for a Gaussian source and a circular Gaussian window.

This has to be done on the **whole tensor**.  The relation is diagonal in the
eigenbasis, not in the pixel axes, so deconvolving ``\sigma^2_{yy}`` and
``\sigma^2_{xx}`` separately is only correct when ``\sigma^2_{xy} = 0``; for a
rotated elliptical source it is wrong by 2% at a moment correlation of 0.25 and
11% at 0.53.  For the same reason there is no shortcut on the trace, so
`compactness_aperture` cannot be corrected without the cross moment either.

Returns `NaN` for all three when the deconvolution has no solution: a measured
tensor that is not positive definite, or a source at least as broad as the
window in some direction, where the window gives no purchase on the width.

Needs no per-window method.  An infinitely wide window has
`inv_window_var == 0`, which short-circuits to the identity -- returned
unchanged rather than round-tripped through two matrix inversions, so
[`FlatWindow`](@ref) stays bitwise exact.
"""
function deconvolve_moments(w::AbstractMomentWindow, σ²_yy::Real, σ²_xx::Real,
                            σ²_xy::Real)
    FT = float(promote_type(typeof(σ²_yy), typeof(σ²_xx), typeof(σ²_xy)))
    a, b, c = FT(σ²_yy), FT(σ²_xx), FT(σ²_xy)
    k = inv_window_var(w)
    iszero(k) && return (; yy = a, xx = b, xy = c)

    # Closed form of the 2x2 inverse above.  With `D = det(Sigma_meas)` and
    # `Q = D * det(Sigma_meas^-1 - k I)`, the result is
    # `[(a - kD)/Q  c/Q; c/Q  (b - kD)/Q]`; `Q = 1` when `k = 0`, which is the
    # identity the short circuit returns exactly.
    kf = FT(k)
    nan = FT(NaN)
    D = a * b - c * c
    D > 0 || return (; yy = nan, xx = nan, xy = nan)
    Q = 1 - kf * (a + b) + kf * kf * D
    Q > 0 || return (; yy = nan, xx = nan, xy = nan)
    yy = (a - kf * D) / Q
    xx = (b - kf * D) / Q
    # `Q > 0` and a positive diagonal are together enough for positive
    # definiteness, since `det(Sigma_s) = D / Q`.
    (yy > 0 && xx > 0) || return (; yy = nan, xx = nan, xy = nan)
    return (; yy, xx, xy = c / Q)
end

# ---------------------------------------------------------------------------
# Internal: weighted second moments about a reference point
# ---------------------------------------------------------------------------

# TODO: accept two inverse-variance maps -- background-only for the shape and
# size statistics, which have to stay flux-independent, and total for
# `aperture_sum_err`.  `aperture_sum` is the only unweighted value computed
# here, so its variance really is `sum(1/w)` and the total map is exactly right
# for it.  Every other quantity is a weighted estimator whose uncertainty is
# tied to its own weights, so correcting those needs a sandwich covariance
# rather than a second map.

"""
    _moments2(image, inv_var, background, y0, x0 [, window]) -> NamedTuple

Compute inverse-variance-weighted second moments of `image .- background`
about the reference point `(y0, x0)`, optionally tapered
by a multiplicative `window` (see [`AbstractMomentWindow`](@ref); defaults to
[`FlatWindow`](@ref), i.e. no taper).
Mask invalid image pixels by setting their inverse variance to zero.

The returned moments are **not** centralized -- the caller must compute
the centroid offset `μ_y = M10 / M00`, `μ_x = M01 / M00` and subtract
to obtain central moments.

# Returns
`(; M00, M10, M01, M20, M02, M11, W00, W10, W01, W20, W02, W11,
    aperture_sum, aperture_area, aperture_sum_err)`
where each flux moment is
```math
M_{pq} = \\sum_{y,x} w_{y,x} \\; g_{y,x} \\; z_{y,x} \\; (y - y_0)^p \\; (x - x_0)^q
```
with ``w = \\mathtt{inv\\_var}``, ``z = \\mathtt{image} - \\mathtt{background}``
and ``g`` the window, so the effective moment weight is ``w g``.
The ``W_{pq}`` fields are the matching *variance* moments
``\\sum w_{y,x} g^2 (y-y_0)^p(x-x_0)^q`` over the same included pixels, used
for delta-method covariance propagation: with ``u = wg`` and
``\\mathrm{Var}(z) = 1/w``, ``\\mathrm{Var}(\\sum u z \\cdots) = \\sum w g^2 \\cdots``.
The extra factor of ``g`` is why ``W`` is not simply ``\\sum w``; the two
coincide only for [`FlatWindow`](@ref).
Pixels with ``w \\le 0`` are skipped, as are pixels outside the window's
support.  If ``M_{00} \\le 0`` (a non-detection, or fully masked), `M00 = 0`
and higher moments are meaningless; the caller should guard against this.

Signed residuals are kept: there is **no** positivity clip.  Dropping
``z \\le 0`` would retain only the upward noise excursions in the wings, each
with its full ``r^2`` lever arm, biasing every second moment by an amount that
grows with the box area.

Callers centralize these moments, and central moments are
translation-invariant, so ``(y_0, x_0)`` cancels out of the lever arms
algebraically, for any source.  It does **not** cancel out of the window anchor,
which ``(y_0, x_0)`` also sets: moving it 2 px shifts `compactness_aperture` by
~14% for a Moffat profile.  A Gaussian source is the exception, being
offset-invariant against a Gaussian window to ~1e-5.  So pass the peak pixel!

`aperture_sum` is the unweighted, **unwindowed** rectangular-cutout sum of
``z = \\mathtt{image} - \\mathtt{background}`` over pixels with positive
inverse variance (a tapered sum would be a matched-filter flux, a different
quantity).  `aperture_area` is the number of pixels in that sum,
and `aperture_sum_err` is the formal propagated uncertainty
``\\sqrt{\\sum 1/w}`` assuming independent pixel errors.
"""
function _moments2(
        image::AbstractMatrix{T},
        inv_var::AbstractMatrix,
        background::Real,
        y0::Real,
        x0::Real,
        window::AbstractMomentWindow = FlatWindow(),
    ) where {T}
    FT = float(T)
    M00 = zero(FT)
    M10 = zero(FT)
    M01 = zero(FT)
    M20 = zero(FT)
    M02 = zero(FT)
    M11 = zero(FT)
    # Weight-only moments for delta-method centroid covariance.
    # Delta method takes ~ 20% longer than using flux-weighted
    # second moments, but should be more accurate for faint sources
    # where the flux-weighted second moments can be noisy and even
    # negative. This is not a significant bottleneck for full-pipeline
    # runs so we can afford the extra computation.
    W00 = zero(FT)
    W10 = zero(FT)
    W01 = zero(FT)
    W20 = zero(FT)
    W02 = zero(FT)
    W11 = zero(FT)
    # Unweighted rectangular aperture diagnostics over valid pixels.
    aperture_sum = zero(FT)
    aperture_area = 0
    aperture_var = zero(FT)
    bg = FT(background)
    fy0 = FT(y0)
    fx0 = FT(x0)
    # The window is anchored on the integer pixel nearest `(y0, x0)`
    iy0 = round(Int, y0)
    jx0 = round(Int, x0)

    @inbounds for j in axes(image, 2)
        dx = FT(j) - fx0
        gxj = FT(xfactor(window, j - jx0))
        for i in axes(image, 1)
            w = inv_var[i, j]
            w > 0 || continue
            fw = FT(w)
            z = FT(image[i, j]) - bg

            # Aperture sums keep signed residuals over the same unmasked cutout,
            # and are deliberately *unwindowed*: a tapered sum is a
            # matched-filter flux, which is a different quantity.
            aperture_sum += z
            aperture_area += 1
            aperture_var += inv(fw)

            g = gxj * FT(yfactor(window, i - iy0))
            iszero(g) && continue
            dy = FT(i) - fy0
            # Moments carry w*g; their variances carry w*g^2, because
            # Var(sum u z) = sum u^2 / w with u = w*g.
            u = fw * g
            uv = u * g
            wz = u * z
            M00 += wz
            M10 += wz * dy
            M01 += wz * dx
            M20 += wz * dy * dy
            M02 += wz * dx * dx
            M11 += wz * dx * dy
            W00 += uv
            W10 += uv * dy
            W01 += uv * dx
            W20 += uv * dy * dy
            W02 += uv * dx * dx
            W11 += uv * dx * dy
        end
    end
    return (; M00, M10, M01, M20, M02, M11,
             W00, W10, W01, W20, W02, W11,
             aperture_sum, aperture_area, aperture_sum_err = sqrt(aperture_var))
end

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@doc raw"""
    measure_star_shape(image, y0, x0; inv_var, background, fwhm_factor) -> NamedTuple

Compute aperture-based morphological measurements for a stellar image
cutout using inverse-variance-weighted second central moments.

# Arguments
- `image::AbstractMatrix`: image cutout of a single star.
- `y0::Real, x0::Real`: approximate centroid about which the moments
  are accumulated.  Integer pixel coordinates (e.g. the peak pixel) are
  usually sufficient; the function computes the precise center-of-mass
  from the moments themselves.
- `inv_var::AbstractMatrix`: per-pixel inverse variance, same size as
  `image`.  Defaults to `Fill(one(float(eltype(image))), size(image))`
  (uniform weighting).  Set entries to zero to mask bad pixels.
  Note that the morphological statistics are typically most informative when
  `inv_var` includes source variance (Poisson noise) in addition to background variance.
- `background::Real`: scalar background level subtracted before computing
  moments.  Defaults to `0`.  Pixels with ``\mathtt{image} - \mathtt{background} \le 0``
  are excluded from the moment sum.
- `fwhm_factor::Real`: scale factor from Gaussian σ to FWHM.
  Defaults to ``2\sqrt{2\log 2} \approx 2.35482``.
- `window::AbstractMomentWindow = FlatWindow()`: multiplicative taper applied
  to the moment accumulators on top of `inv_var`, to bound the wing noise that
  otherwise grows with the box area.  Its known compression is divided back out
  through deconvolution, so `fwhm`, `theta`, the ellipticities and
  `compactness_aperture` stay absolute. For a non-Gaussian profile
  the recovered moments are Gaussian-equivalent rather than exact.
- `y_offset::Real = 0`, `x_offset::Real = 0`: origin of `image` in the
  caller's coordinate frame.  Added to the returned `centroid.y` and
  `centroid.x`, so a caller working on a cutout extracted at
  `image[y_start:y_end, x_start:x_end]` passes `y_offset = y_start - 1`,
  `x_offset = x_start - 1` and gets results in the original image's
  coordinates.  Uncertainties, covariances, and all shape statistics are
  translation-invariant and unaffected.

# Returns
`(; fwhm, ellipticity1_aperture, ellipticity2_aperture,
    compactness_aperture, moment_norm, aperture_sum, aperture_area,
    aperture_sum_err, centroid)` where

- `fwhm::NamedTuple (; y, x, theta)`: moment-based, axis-aligned marginal
  full width at half maximum along the ``y`` (row) and ``x`` (column)
  axes.  These are not principal-axis widths for a rotated source.
  `theta` is the position angle of the covariance major axis in degrees,
  measured counter-clockwise from the ``+x``-axis (column direction).
  ``\theta = 0`` means the major axis is aligned with columns;
  positive ``\theta`` rotates toward rows.
- `ellipticity1_aperture::T`, `ellipticity2_aperture::T`: the two
  normalized quadrupole (ellipticity) components of the second-moment
  covariance,
  ``e_1 = (\sigma^2_{yy} - \sigma^2_{xx})/(\sigma^2_{yy} + \sigma^2_{xx})``
  and ``e_2 = 2\sigma^2_{xy}/(\sigma^2_{yy} + \sigma^2_{xx})``.
  Both are ``0`` for a circular source.  ``e_1 > 0`` is extended in ``y``
  (rows) and ``e_1 < 0`` extended in ``x`` (columns); ``e_2 > 0`` is
  extended along the
  ``+45°`` diagonal.  `NaN` when
  ``\sigma^2_{yy} + \sigma^2_{xx} \le 0``.
- `compactness_aperture::T`: inverse total second central moment
  ``1/(\sigma^2_{yy} + \sigma^2_{xx})``.  Proportional to
  ``1/\mathrm{FWHM}^2`` for a Gaussian; larger for more compact profiles.
  `NaN` when ``\sigma^2_{yy} + \sigma^2_{xx} \le 0``.
- `moment_norm::T`: weighted zeroth moment ``M_{00}`` used to normalize
  the shape moments.  When `inv_var` is not uniform this is not a physical
  source flux and should not be used for photometric calibration.
- `aperture_sum::T`: unweighted sum of `image - background` over unmasked
  pixels in the rectangular cutout.
- `aperture_area::Int`: number of pixels contributing to `aperture_sum`.
- `aperture_sum_err::T`: formal inverse-variance propagated uncertainty
  on `aperture_sum`, assuming independent pixel errors.
- `centroid::NamedTuple (; y, x, y_err, x_err, cov)`: center-of-mass
  centroid, 1-σ uncertainties, and 2×2 `SMatrix` covariance.

!!! note "Ellipticity"
    The ellipticity ``1 - b/a = 1 - \sqrt{(1-|e|)/(1+|e|)}`` with ``|e| = \sqrt{e_1^2 + e_2^2}``,
    with position angle
    ``\theta = \tfrac{1}{2}\arctan(e_2, -e_1)``, is a one-liner from the
    pair above and is deliberately not returned.  It rectifies: component
    scatter cannot cancel, so noise and residual
    sub-pixel phase both push it up and never down, and a round source has a positive
    expectation of order ``\sigma\sqrt{\pi/2}`` in the component
    error.  It is also the one
    shape statistic that cannot be corrected against a PSF model from its
    own value, because the magnitude does not commute with the subtraction:
    the correction has to be applied to ``e_1`` and ``e_2`` *before*
    taking the magnitude.

If ``M_{00} \le 0`` (all pixels at or below background), shape and
centroid fields are `NaN`; aperture-sum diagnostics are still reported.
If ``\sigma^2_{yy} \le 0`` or ``\sigma^2_{xx} \le 0``
(the distribution has no measurable width, e.g. a single bright pixel),
`fwhm.y` and `fwhm.x` are `NaN`, and `ellipticity1_aperture`,
`ellipticity2_aperture` and `compactness_aperture` are `NaN`.

!!! note "Robustness to sub-pixel phase"
    Every shape statistic here is a ratio of linear moment sums taken
    about the center of mass, so the integer reference point `(y0, x0)`
    cancels identically and a sub-pixel shift of the source is suppressed
    exponentially in the sampling ratio (roughly ``e^{-2\pi^2\sigma^2}``
    with ``\sigma`` the profile width in pixels).  Measured on noiseless
    round sources scanned over all sub-pixel phases, the fractional
    scatter of `fwhm`, `compactness_aperture`, `ellipticity1_aperture`
    and `ellipticity2_aperture` is ``\sim 1\%`` at
    FWHM 1.2 px, ``2\times10^{-4}`` at FWHM 1.6 px, and below ``10^{-6}``
    at FWHM 2 px and above.  The residual comes from the hard rectangular
    cutout, whose edges move relative to the source; it is larger for a
    PSF with power-law wings than for a Gaussian.

# Examples
```jldoctest
julia> using CrowdPhot: measure_star_shape

julia> img = [0.1 0.3 0.1; 0.3 1.0 0.3; 0.1 0.3 0.1];

julia> result = measure_star_shape(img; background=0);

julia> result.fwhm.y ≈ result.fwhm.x ≈ 1.4603973964538084
true

julia> result.centroid.y ≈ 2.0
true

julia> result.centroid.x ≈ 2.0
true

julia> result.centroid.y_err > 0
true
```
"""
function measure_star_shape(
        image::AbstractMatrix{T},
        y0::Real,
        x0::Real;
        inv_var::AbstractMatrix = Fill(one(float(T)), size(image)),
        background::Real = zero(float(T)),
        fwhm_factor::Real = 2.3548200450309493,
        y_offset::Real = 0,
        x_offset::Real = 0,
        window::AbstractMomentWindow = FlatWindow(),
    ) where {T}
    FT = float(T)

    mom = _moments2(image, inv_var, background, y0, x0, window)
    FT_M00 = FT(mom.M00)

    if FT_M00 <= zero(FT)
        n = FT(NaN)
        return (; fwhm = (; y = n, x = n, theta = n),
                 ellipticity1_aperture = n, ellipticity2_aperture = n,
                 compactness_aperture = n,
                 moment_norm = FT_M00,
                 aperture_sum = FT(mom.aperture_sum),
                 aperture_area = mom.aperture_area,
                 aperture_sum_err = FT(mom.aperture_sum_err),
                 centroid = (; y = n, x = n, y_err = n, x_err = n,
                              cov = @SMatrix [n n; n n]))
    end

    inv_M00 = inv(FT_M00)
    μ_y = FT(mom.M10) * inv_M00
    μ_x = FT(mom.M01) * inv_M00

    # Centralised second moments: σ²_pq = M_pq / M00 - μ_p μ_q.
    σ²_yy = FT(mom.M20) * inv_M00 - μ_y * μ_y
    σ²_xx = FT(mom.M02) * inv_M00 - μ_x * μ_x
    σ²_xy = FT(mom.M11) * inv_M00 - μ_x * μ_y

    # Clamp negative variances (possible from noise on faint sources).
    σ²_yy = max(zero(FT), σ²_yy)
    σ²_xx = max(zero(FT), σ²_xx)

    # Divide the window's known compression back out, so every size and shape
    # statistic below is absolute rather than window-relative.
    # Deconvolution is exact only for a Gaussian
    # source, so these will not be exact for non-Gaussian profiles. The solution
    # is to measure the same quantities on the windowed PSF render, deconvolve,
    # and then use a psf-relative quantity for downstream analysis
    # (see measure_star_shape_ref)
    dm = deconvolve_moments(window, σ²_yy, σ²_xx, σ²_xy)
    σ²_yy = FT(dm.yy)
    σ²_xx = FT(dm.xx)
    σ²_xy = FT(dm.xy)

    ff = FT(fwhm_factor)
    fwhm_y = σ²_yy > 0 ? ff * sqrt(σ²_yy) : FT(NaN)
    fwhm_x = σ²_xx > 0 ? ff * sqrt(σ²_xx) : FT(NaN)

    # Position angle from the 2×2 covariance.
    # theta = 0.5 * atan(2σ²_xy, σ²_xx - σ²_yy), converted to degrees.
    num = 2 * σ²_xy
    den = σ²_xx - σ²_yy
    # For a nearly isotropic profile, both num and den are ~0 and the
    # angle is undefined.  Use a tolerance rather than exact zero because
    # floating-point summation order can leave σ²_yy ≠ σ²_xx at ~1e-16.
    if abs(num) + abs(den) <= eps(FT) * (σ²_yy + σ²_xx)
        theta = zero(FT)
    else
        theta = FT(rad2deg(atan(num, den) / 2))
    end

    # `*_offset` shifts the cutout's origin into the caller's frame so no
    # caller has to translate the result.
    fwhm_cen = FT(y0) + FT(y_offset) + μ_y
    fwhm_cen_x = FT(x0) + FT(x_offset) + μ_x

    # compactness_aperture is the inverse total second central moment,
    # ∝ 1/FWHM² for a Gaussian.
    total_moment = σ²_yy + σ²_xx
    compactness_aperture = total_moment > 0 ? inv(total_moment) : FT(NaN)

    # Normalized quadrupole (ellipticity) components.  These are ratios of
    # linear moment sums about the center of mass, so a sub-pixel shift of
    # the source cancels out of them (see the docstring); that is why they
    # replace the DAOPHOT SROUND/GROUND statistics: SROUND's quadrant
    # boundaries were anchored to the integer peak pixel and responded to
    # phase at first order, and GROUND's sqrt-of-variance marginals are a
    # less natural parameterization of the same axis e1 covers.  e2 carries
    # the 45-degree information SROUND was meant to supply, with none of its
    # cross-talk from axis-aligned elongation.
    ellipticity1_aperture = total_moment > 0 ? (σ²_yy - σ²_xx) / total_moment : FT(NaN)
    ellipticity2_aperture = total_moment > 0 ? 2 * σ²_xy / total_moment : FT(NaN)

    # Centroid covariance from the delta method for the ratio estimator.
    inv_M00_sq = inv_M00 * inv_M00
    cent_cov_yy = (FT(mom.W20) - 2 * μ_y * FT(mom.W10) +
                   μ_y * μ_y * FT(mom.W00)) * inv_M00_sq
    cent_cov_xx = (FT(mom.W02) - 2 * μ_x * FT(mom.W01) +
                   μ_x * μ_x * FT(mom.W00)) * inv_M00_sq
    cent_cov_xy = (FT(mom.W11) - μ_y * FT(mom.W01) - μ_x * FT(mom.W10) +
                   μ_y * μ_x * FT(mom.W00)) * inv_M00_sq

    return (; fwhm = (; y = fwhm_y, x = fwhm_x, theta),
             ellipticity1_aperture, ellipticity2_aperture, compactness_aperture,
             moment_norm = FT_M00,
             aperture_sum = FT(mom.aperture_sum),
             aperture_area = mom.aperture_area,
             aperture_sum_err = FT(mom.aperture_sum_err),
             centroid = (; y = fwhm_cen, x = fwhm_cen_x,
                          y_err = sqrt(max(zero(FT), cent_cov_yy)),
                          x_err = sqrt(max(zero(FT), cent_cov_xx)),
                          cov = @SMatrix [cent_cov_yy cent_cov_xy;
                                          cent_cov_xy cent_cov_xx]))
end

"""
    measure_star_shape(image; kws...) -> NamedTuple

Convenience method that finds the brightest pixel in `image` via
`findmax` and calls [`measure_star_shape`](@ref measure_star_shape)
with those integer coordinates.  See the core method for keyword
arguments and return fields.
"""
function measure_star_shape(image::AbstractMatrix; kws...)
    _, maxidx = findmax(image)
    i0, j0 = Tuple(maxidx)
    return measure_star_shape(image, Int(i0), Int(j0); kws...)
end

# ---------------------------------------------------------------------------
# One source, measured against its own PSF reference
# ---------------------------------------------------------------------------

"""
    measure_star_shape_ref(clean, rend, i0, j0, height; kws...) -> NamedTuple

Every shape statistic for one source, measured on the isolated cutout `clean`,
together with the same statistics measured on the noiseless render `rend` of
that source's PSF model.

Callers supply the cutout and its render: a fitter's finalization pass already
holds both, and renders once for the pair rather than a second time here.  See
`finalize_multipass` in `psf_photometry_simultaneous_multipass.jl` for how
`clean`, `rend`, `i0`, `j0` and `height` are built per source.

# Arguments

- `clean`: the isolated cutout -- the residual with every source removed and
  this one added back, so it holds this source's light plus noise but not its
  neighbors'.
- `rend`: this source's model rendered over **the same pixels** as `clean`,
  with the model's own `bkg` already subtracted.
- `i0`, `j0`: the anchor pixel, in `clean`'s own (1-based) indices.  Must be a
  local maximum: [`centroid_poly`](@ref) fits a quadratic to the 3x3
  neighborhood and degenerates otherwise.
- `height`: the central height used to normalize `sharpness`, normally
  `maximum(rend)`.

# Keyword arguments

- `inv_var`: per-pixel inverse variance over the cutout, or `nothing`
  (default) for unit weights.
- `sharp_half_width::Integer = 2`: half-width of the `sharpness` footprint.
  See `_sharp_half_width`.
- `background::Real = 0`, `fwhm_factor::Real`, `window`: as for
  [`measure_star_shape`](@ref).  `window` is applied to **both** halves, which
  is what keeps the `psf_ref` comparison exact; see the note below.
- `y_offset`, `x_offset`: added to every returned coordinate, to lift cutout
  indices into the frame `clean` was cut from.

# Returns

`(; sharpness, sharpness_err, core, centroid, aperture, psf_ref)`, where
`psf_ref` is `(; sharpness, core, aperture)` measured on `rend`.  See
[`measure_star_shapes`](@ref) for the individual fields and for how to combine
a statistic with its `psf_ref` counterpart.  `psf_ref` mirrors values only: the
render is noiseless, so it carries no `sharpness_err`.

!!! note "Why both halves live in one function"
    The measurement and its reference must share `inv_var`, the anchor pixel,
    `background`, `height` and `window`, or the comparison stops being exact: with
    `clean == rend` every ratio below must come out at exactly 1 and every
    difference at exactly 0, which is what cancels the sub-pixel phase and
    PSF-width dependence instead of modeling it away.  Computing the two halves
    in one place is what enforces that; splitting them across call sites would
    let them drift apart silently, since a violation changes no types and
    throws no error.
"""
function measure_star_shape_ref(clean::AbstractMatrix, rend::AbstractMatrix,
        i0::Integer, j0::Integer, height::Real;
        inv_var::Union{Nothing, AbstractMatrix{<:Real}} = nothing,
        sharp_half_width::Integer = 2,
        background::Real = 0,
        fwhm_factor::Real = 2.3548200450309493,
        y_offset::Real = 0,
        x_offset::Real = 0,
        window::AbstractMomentWindow = FlatWindow(),
    )
    axes(clean) == axes(rend) ||
        throw(DimensionMismatch("`clean` and `rend` must have the same `axes`; " *
                                "got $(axes(clean)) and $(axes(rend))"))
    FT = float(eltype(clean))
    ivar = inv_var === nothing ? Fill(one(FT), size(clean)) : inv_var
    i, j, shw, h = Int(i0), Int(j0), Int(sharp_half_width), FT(height)

    core = centroid_poly(clean, i, j, ivar; background, y_offset, x_offset)
    centroid = choose_centroid(core)
    aperture = measure_star_shape(clean, i, j;
                                  inv_var = ivar, background, fwhm_factor,
                                  y_offset, x_offset, window)
    sharpness, sharpness_err = _sharpness(clean, i, j, shw, shw, h, ivar)

    # The same measurement on the noiseless render: what each statistic would
    # read for a source that *is* the PSF, at this source's sub-pixel phase and
    # with this source's weights, anchor, background and height normalization.
    psf_ref = (;
        sharpness = first(_sharpness(rend, i, j, shw, shw, h, ivar)),
        core = centroid_poly(rend, i, j, ivar; background, y_offset, x_offset),
        aperture = measure_star_shape(rend, i, j;
                                      inv_var = ivar, background, fwhm_factor,
                                      y_offset, x_offset, window),
    )
    return (; sharpness, sharpness_err, core, centroid, aperture, psf_ref)
end

# ---------------------------------------------------------------------------
# Batch measurement from matched-filter results
# ---------------------------------------------------------------------------

"""
    _default_half_width(result::MatchedFilterResult) -> Int

Return a sensible default half-width for cutout extraction around each
detected peak.  The kernel radius (half the kernel size) is used as a
proxy for the PSF extent, with a 2-pixel margin added to capture the
wings.  The minimum half-width is 3, guaranteeing at least a 7×7 cutout.
"""
function _default_half_width(result::MatchedFilterResult)
    kr = size(result.kernel, 1) ÷ 2
    kc = size(result.kernel, 2) ÷ 2
    return max(3, max(kr, kc) + 2)
end

"""
    _kernel_template(K, kernel_norm) -> Matrix

The sum-normalized PSF template `P` behind a `matched_filter` kernel `K`,
recovered from the stored kernel and `kernel_norm`.
"""
function _kernel_template(K::AbstractMatrix, kernel_norm::Real)
    d = kernel_norm^2
    c = (1 - sum(K) * d) / length(K)
    return @. K * d + c
end

@doc raw"""
    _sharp_half_width(fwhm) -> Int

DAOPHOT's `SHARP` neighborhood half-width,
``\max(2, \lfloor 0.72\,\mathrm{FWHM} \rfloor)``.  The scale matters
because the neighbor mean has to be measured where the profile is still
falling steeply,
so a footprint much larger than this dilutes it toward sky and drives
`sharpness` to 1 for every source. The floor of 2 keeps it from
collapsing onto the 3x3 core, where it would duplicate
`normalized_curvature` and inherit that statistic's sub-pixel phase
sensitivity.  Non-finite or non-positive `fwhm` falls back to the floor.

Approximate `fwhm` are okay here; see e.g.
[`effective_fwhm`](@ref CrowdPhot.PSF.effective_fwhm).
"""
_sharp_half_width(fwhm::Real) =
    (isfinite(fwhm) && fwhm > 0) ? max(2, floor(Int, 0.72 * fwhm)) : 2

@doc raw"""
    _sharpness(image, i0, j0, hy, hx, height [, inv_var]) -> (Real, Real)

DAOPHOT `SHARP` for the peak at `image[i0, j0]`:

```math
\mathtt{sharpness} = \frac{D_{i_0 j_0} - \bar{D}_\mathrm{neighbors}}{H}
```

the raw peak pixel minus the mean of its neighbors over the
``(2h_y+1) \times (2h_x+1)`` footprint (center excluded), divided by
the source's fitted central `height`.

A spatially flat background cancels from the numerator, so no background
argument is needed.  Non-finite neighbors are skipped; the footprint is
clipped at the image border.

Returns the statistic and its 1-σ uncertainty, the latter propagated from
`inv_var` (unit weights when omitted, giving a formal error) over exactly the
pixels that entered the value:

```math
\sigma(\mathtt{sharpness}) = \frac{1}{H}
    \sqrt{\sigma^2_{i_0 j_0} + \frac{1}{n^2}\sum_\mathrm{neighbors} \sigma^2_i}
```

!!! note "`height` is treated as exact"
    No uncertainty is propagated for `H`.  In [`measure_star_shape_ref`](@ref)
    it comes from a noiseless render, so it genuinely has none.  In
    [`measure_star_shapes`](@ref) it is the matched-filter flux times the
    template peak fraction, which *is* uncertain (exactly
    `peak_fluxes[i] / peak_significances[i]`, since the matched filter's
    weighted flux error reduces to that ratio), but it is built from pixels
    that overlap the numerator's footprint.  Propagating it would need
    ``\mathrm{Cov}(\text{numerator}, H)`` rather than a variance, and keeping
    `H` fixed is also what keeps `sharpness` a pure concentration statistic
    instead of folding flux uncertainty into it.  Both call paths therefore
    agree on what `sharpness_err` means.

Both returned values are `NaN` if `height` is non-positive or no valid neighbor
exists; the error alone is `NaN` if any contributing pixel has a non-positive
or non-finite weight.
"""
function _sharpness(image::AbstractMatrix{T}, i0::Int, j0::Int,
                    hy::Int, hx::Int, height::Real,
                    inv_var::Union{Nothing, AbstractMatrix} = nothing) where {T}
    FT = float(T)
    nan = FT(NaN)
    (isfinite(height) && height > 0) || return (nan, nan)
    ny, nx = size(image)
    s = zero(FT)
    n = 0
    # Variance of the neighbor sum.  Accumulated over exactly the pixels that
    # enter `s`, so a masked or non-finite weight anywhere in that set makes the
    # error undefined (NaN) without perturbing the value.
    svar = zero(FT)
    @inbounds for i in max(1, i0 - hy):min(ny, i0 + hy),
                  j in max(1, j0 - hx):min(nx, j0 + hx)
        (i == i0 && j == j0) && continue
        v = FT(image[i, j])
        isfinite(v) || continue
        s += v
        n += 1
        w = inv_var === nothing ? one(FT) : FT(inv_var[i, j])
        svar += (isfinite(w) && w > 0) ? inv(w) : nan
    end
    n == 0 && return (nan, nan)
    h = FT(height)
    sharpness = (FT(image[i0, j0]) - s / n) / h
    # `height` is taken as exact: in the reference path it comes from a noiseless
    # render, and in the batch path it shares pixels with the numerator, so
    # propagating its error would need a covariance rather than a variance.
    wc = inv_var === nothing ? one(FT) : FT(inv_var[i0, j0])
    var_c = (isfinite(wc) && wc > 0) ? inv(wc) : nan
    err = sqrt(var_c + svar / (n * n)) / h
    return (sharpness, err)
end

"""
    measure_star_shapes(result::MatchedFilterResult; kws...) -> Vector{NamedTuple}

Measure centroid, shape, and morphological properties for every peak
detected by [`matched_filter`](@ref).

For each peak in `result.peaks`, this function:

1. Extracts a square cutout of size ``(2 \\times \\mathtt{half\\_width} + 1)^2``
   centered on the peak pixel from the original image.
2. Calls [`centroid_poly`](@ref) on the 3×3 core (passing `background`) to
   obtain a sub-pixel centroid (polynomial and center-of-mass) and core
   diagnostics (normalized curvature, compactness, and the two ellipticity
   components).
3. Calls [`choose_centroid`](@ref) to select the best centroid estimate.
4. Calls [`measure_star_shape`](@ref) on the full cutout to compute
   aperture-based morphology and rectangular aperture sums.

All coordinate fields in the returned NamedTuples are in **global**
pixel coordinates of the original image.

# Arguments

- `result::MatchedFilterResult`: result from [`matched_filter`](@ref).

# Keyword Arguments

- `inv_var::Union{Nothing, AbstractMatrix{<:Real}}`:
  inverse-variance map corresponding to `result.image`. Defaults to `result.inv_var`,
  the same inverse variance map used for the `matched_filter` call.
  Note that the morphological statistics are typically most informative when
  `inv_var` includes source variance (Poisson noise)
  in addition to background variance.  However, it is recommended that source variance *not* be
  included in the `inv_var` passed to `matched_filter`, so we provide the option here to
  pass in a separate, more complete `inv_var` for the morphology measurements.
  If `nothing`, the morphology measurements will be computed with uniform weighting.
- `half_width::Int`: half-width of the square cutout extracted around
  each peak.  The cutout size is ``(2 \\times \\mathtt{half\\_width} + 1)
  \\times (2 \\times \\mathtt{half\\_width} + 1)`` pixels.  Defaults to the
  kernel radius plus 2, with a minimum of 3.
- `background::Real`: scalar background level subtracted before computing
  image moments.  Defaults to `0`.  Passed to both
  [`measure_star_shape`](@ref) and [`centroid_poly`](@ref); the core
  diagnostics `normalized_curvature`, `compactness_core`,
  `ellipticity1_core`, `ellipticity2_core`, and the center-of-mass are
  only meaningful when this matches the true sky level.
- `fwhm_factor::Real`: scale factor from Gaussian σ to FWHM.  Defaults to
  ``2\\sqrt{2\\log 2} \\approx 2.35482``.  Passed to [`measure_star_shape`](@ref).
- `peaks::Union{AbstractVector{Int}, Nothing}`: optional vector of integer
  indices into `result.peaks` specifying which peaks to measure.  When
  `nothing` (default), all peaks are measured (subject to
  `min_significance`).
- `min_significance::Union{Real, Nothing}`: optional significance threshold;
  only peaks with `result.peak_significances[i] >= min_significance` are
  measured.  Ignored if `peaks` is also provided.

# Returns

A `Vector` of `NamedTuple`s, one per measured peak.  Each `NamedTuple`
has the following fields:

- `peak_index::Int`: index into the original `result.peaks` array.
- `pixel::CartesianIndex{2}`: the peak pixel `(row, column)` in the
  original image.
- `significance`: detection significance at this peak.
- `flux`: matched-filter flux estimate at this peak.
- `sharpness`: DAOPHOT `SHARP` [Stetson1987](@citet),
  ``(D_\\mathrm{peak} - \\bar{D}_\\mathrm{neighbors}) / H``.  The numerator
  is taken on the **raw** image over DAOPHOT's
  ``\\max(2, \\lfloor 0.72\\,\\mathrm{FWHM}\\rfloor)`` footprint, with the
  PSF width recovered from the detection kernel. The denominator
  ``H = \\mathtt{flux} \\times \\max(P)`` is the source's fitted central
  height, matching DAOPHOT's normalization.  This statistic is large for a
  cosmic ray or hot pixel whose flux is confined to a single pixel, small for a
  blend or a resolved source, and tightly clustered for stars.
  Unlike `normalized_curvature` it needs no quadratic fit, so it survives
  where that fit degenerates.  `NaN` if the fitted height is non-positive.
- `sharpness_err`: 1-σ uncertainty on `sharpness`, propagated from `inv_var`
  over the same footprint.  The central height ``H`` is treated as exact: it
  shares pixels with the numerator, so propagating its error would need a
  covariance rather than a variance, and holding it fixed keeps `sharpness` a
  pure concentration statistic.
- `core`: the full [`centroid_poly`](@ref) result — `(; poly, com,
  normalized_curvature, normalized_curvature_err, compactness_core,
  ellipticity1_core, ellipticity2_core)` with coordinates in global pixels.
- `centroid`: the chosen centroid `(; y, x, source)` from
  [`choose_centroid`](@ref) in global pixels.  `source` is `:poly` or `:com`.
- `aperture`: the full [`measure_star_shape`](@ref) result — `(; fwhm,
  ellipticity1_aperture, ellipticity2_aperture, compactness_aperture,
  moment_norm, aperture_sum, aperture_area, aperture_sum_err, centroid)`
  with coordinates in global pixels.
- `psf_ref`: `nothing` for this method.  There is no PSF model at detection
  time, so there is nothing to normalize against.
  [`measure_star_shape_ref`](@ref) fills this in, which is what the
  post-fit morphology pass calls; see it for the contract.

!!! note
    If a peak is so close to the image border that no full 3×3
    neighborhood exists, all fields in `core` are `NaN` and
    `centroid.source` is `:poly` (degenerate).  The `aperture` fields
    are computed from the available (clipped) cutout and may still be
    valid.

# Examples

```jldoctest
julia> using CrowdPhot: matched_filter, measure_star_shapes

julia> mf_result = matched_filter(zeros(50, 50), 3.0; sigma=0.0);

julia> results = measure_star_shapes(mf_result);

julia> isempty(results)
true
```

# References
See [Vakili2016](@citet) for the polynomial centroid method.
"""
function measure_star_shapes(
        result::MatchedFilterResult{T};
        inv_var::Union{Nothing, AbstractMatrix{<:Real}} = result.inv_var,
        half_width::Int = _default_half_width(result),
        background::Real = zero(T),
        fwhm_factor::Real = 2.3548200450309493,
        peaks::Union{AbstractVector{Int}, Nothing} = nothing,
        min_significance::Union{Real, Nothing} = nothing,
        window::Union{Nothing, AbstractMomentWindow} = nothing,
    ) where {T}

    if !isnothing(inv_var)
        @assert axes(result.image) == axes(inv_var) "image and inv_var must have the same `axes`"
    end

    # Determine which peaks to measure.
    all_peak_idx = if peaks !== nothing
        collect(Int, peaks)
    elseif min_significance !== nothing
        FT = float(T)
        findall(sig -> sig >= FT(min_significance), result.peak_significances)
    else
        collect(1:length(result.peaks))
    end

    n = length(all_peak_idx)
    n == 0 && return NamedTuple[]

    H, W = size(result.image)
    FT = float(T)
    # SHARP normalization: the matched-filter flux times the PSF peak pixel
    # fraction is the source's central height, which is DAOPHOT's denominator.
    # Both are constant over the frame for a spatially constant kernel.
    template = _kernel_template(result.kernel, result.kernel_norm)
    peak_fraction = maximum(template)
    # DAOPHOT's 0.72*FWHM footprint, not the kernel size
    khy = khx = _sharp_half_width(PSF.effective_fwhm(template))
    # Taper the aperture moments at the detection kernel's own width.  That is
    # the sensitivity-optimal choice for a PSF-like source (see
    # `GaussianWindow`), and the kernel is picked to be PSF-scale, which is
    # close enough given how flat the optimum is. Because
    # the width is then known, `measure_star_shape` divides it back out of
    # `fwhm`, so tapering costs no absolute meaning there.
    kfwhm = PSF.effective_fwhm(template)
    win = if window !== nothing
        window
    elseif isfinite(kfwhm) && kfwhm > 0
        GaussianWindow(FT(kfwhm))
    else
        FlatWindow()
    end
    return map(all_peak_idx) do pidx
        pixel = result.peaks[pidx]
        i0, j0 = Tuple(pixel)  # row, column

        # Extract cutout with boundary clipping.
        y_start = max(1, i0 - half_width)
        y_end   = min(H, i0 + half_width)
        x_start = max(1, j0 - half_width)
        x_end   = min(W, j0 + half_width)

        cutout = @view result.image[y_start:y_end, x_start:x_end]

        ivar_cutout = if !isnothing(inv_var)
            @view inv_var[y_start:y_end, x_start:x_end]
        else
            Fill(one(T), size(cutout))
        end

        # Peak pixel within the cutout (1-indexed).
        i0_cut = Int(i0 - y_start + 1)
        j0_cut = Int(j0 - x_start + 1)

        # Global offset for coordinate conversion.
        dy_global = FT(y_start - 1)
        dx_global = FT(x_start - 1)

        # 1. Polynomial centroid on the 3×3 core.  `*_offset` puts every
        #    returned coordinate straight into the original image's frame.
        core = centroid_poly(cutout, i0_cut, j0_cut, ivar_cutout;
                             background, y_offset = dy_global, x_offset = dx_global)

        # 2. Choose best centroid (offset-invariant: it only compares
        #    covariances and passes the chosen coordinates through).
        centroid = choose_centroid(core)

        # 3. Aperture morphology.
        aperture = measure_star_shape(cutout, i0_cut, j0_cut;
                                      inv_var = ivar_cutout, background, fwhm_factor,
                                      y_offset = dy_global, x_offset = dx_global,
                                      window = win)

        # 4. DAOPHOT SHARP from the raw cutout and the matched-filter flux.
        #    Measured on the full frame, so the full-frame weights go with it.
        sharpness, sharpness_err = _sharpness(result.image, Int(i0), Int(j0), khy, khx,
                                              result.peak_fluxes[pidx] * peak_fraction,
                                              inv_var)

        return (;
            peak_index = pidx,
            pixel = pixel,
            significance = result.peak_significances[pidx],
            flux = result.peak_fluxes[pidx],
            sharpness,
            sharpness_err,
            core,
            centroid,
            aperture,
            psf_ref = nothing,
        )
    end
end
