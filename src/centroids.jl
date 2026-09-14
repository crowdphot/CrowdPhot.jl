# ---------------------------------------------------------------------------
# 3x3 polynomial centroiding  —  Vakili & Hogg (2016), arXiv:1610.05873
#
# The 3x3 patch is indexed in row-major order.  Local coordinates (y, x)
# follow the global convention: y is the row offset (first index), x is the
# column offset (second index), each in {-1, 0, 1} relative to the center pixel:
#
#   (-1,-1)  (-1, 0)  (-1, 1)
#   ( 0,-1)  ( 0, 0)  ( 0, 1)
#   ( 1,-1)  ( 1, 0)  ( 1, 1)
#
# Design-matrix row for pixel i: [1, x_i, y_i, x_i², x_i*y_i, y_i²]
# (the polynomial is P(x,y) = a + bx + cy + dx² + exy + fy² where x tracks
# column offset and y tracks row offset).
#
# The normal-equation matrix N = Aᵀ W A and right-hand side Aᵀ W z are built
# in closed form from weighted geometric moments S_pq = Σ w_i x_i^p y_i^q,
# exploiting that x,y ∈ {-1,0,1} collapses higher powers (x³=x, x⁴=x², …).
# ---------------------------------------------------------------------------

"""
    _centroid_poly3(image, inv_var; background=0) -> NamedTuple

Fit a quadratic 2-D polynomial ``P(x,y) = a + bx + cy + dx² + exy + fy²``
to a 3×3 patch using weighted least squares with inverse-variance weights
`inv_var`.

`background` is a scalar subtracted from every pixel before the fit.
Because the constant design column absorbs a uniform offset exactly, the
polynomial coefficients ``b`` through ``f`` — and therefore `poly.y`,
`poly.x`, and `poly.cov` — are unaffected.  It does change every
moment-weighted diagnostic (`com`, `compactness_core`,
`ellipticity1_core`, `ellipticity2_core`) and the
fitted amplitude used to normalize `normalized_curvature`; those
quantities are only meaningful on background-subtracted data.
`poly.peak` is returned as the fitted image value *including*
`background`.

# Returns
A `NamedTuple` with keys `(; poly, com, normalized_curvature,
normalized_curvature_err, compactness_core, compactness_core_err,
ellipticity1_core, ellipticity2_core)`:

- `poly`: `NamedTuple` `(; y, x, peak, y_err, x_err, peak_err, cov)` with
  the polynomial centroid (row, column) relative to the patch center, the
  fitted image value at the centroid (including `background`), 1-σ
  uncertainties, and 3×3 `SMatrix` covariance of `(y, x, peak)`.
- `com`: `NamedTuple` `(; y, x, y_err, x_err, cov)` with the
  inverse-variance-weighted center-of-mass centroid and its 2×2
  covariance on the same 3×3 patch.  All fields are `NaN` when the
  background-subtracted weighted sum ``\\sum w_i (z_i - \\mathrm{background})``
  is non-positive.
- `normalized_curvature`: negated Laplacian divided by the fitted
  amplitude above `background`,
  ``-(2d + 2f)/\\mathrm{amplitude} \\approx 16\\log(2)/\\mathrm{FWHM}^2``
  for a circular Gaussian.  Flux-independent; broad stellar PSFs have
  lower values than cosmic rays and hot pixels.
- `normalized_curvature_err`: 1-σ uncertainty on `normalized_curvature`.
- `compactness_core`: inverse of the total second central moment,
  defined as `1 / (σ_x² + σ_y²)`, where `σ_x²` and `σ_y²` are
  the inverse‑variance‑weighted second central moments of the
  background‑subtracted pixel values.  Larger values indicate more
  compact (sharper) profiles.  Returns `NaN` if the weighted sum or the
  estimated variance sum is non‑positive, which occurs for spurious noise
  peaks, hot pixels, severely saturated detections, or an over‑subtracted
  background.
- `compactness_core_err`: 1-σ uncertainty on `compactness_core`, from the
  delta method on the five flux moments behind ``\\sigma_x^2 + \\sigma_y^2``.
  Costs no extra accumulators: on the 3×3 the third- and fourth-order weight
  moments it needs collapse onto ones already formed, since
  ``x^3 = x`` and ``x^4 = x^2`` for ``x \\in \\{-1,0,1\\}``.
- `ellipticity1_core`, `ellipticity2_core`: the two normalized quadrupole
  (ellipticity) components of the inverse-variance-weighted 3×3 second
  central moments,
  ``e_1 = (\\sigma^2_{yy} - \\sigma^2_{xx})/(\\sigma^2_{yy} + \\sigma^2_{xx})``
  and ``e_2 = 2\\sigma^2_{xy}/(\\sigma^2_{yy} + \\sigma^2_{xx})``.
  ``e_1 > 0`` means extended in ``y`` (rows), ``e_1 < 0`` extended in
  ``x`` (columns); ``e_2 > 0`` means extended along the ``+45°``
  diagonal.  Both are `0` for a circular core.  `NaN` when the weighted
  second-moment sum is non-positive.

  !!! note "Ellipticity"
      The ellipticity ``1 - b/a = 1 - \\sqrt{(1-|e|)/(1+|e|)}`` with
      ``|e| = \\sqrt{e_1^2 + e_2^2}`` is a one-liner from the two
      components, and is deliberately not returned.  It rectifies:
      component scatter cannot cancel, so noise and residual phase error
      both push it up and never down, and a round source has a positive
      expectation of order ``\\sigma\\sqrt{\\pi/2}`` in the component
      error.  It is also the one shape statistic that cannot be corrected
      against a PSF model from its own value, because the magnitude does not
      commute with the subtraction: the correction has to be applied to
      ``e_1`` and ``e_2`` *before* taking the magnitude.

The design matrix is fixed (local coordinates `{-1,0,1}²`), so the
only free inputs are the 9 pixel values and 9 inverse-variance weights.

If the curvature matrix `D = [2d  e;  e  2f]` is near-singular,
a small Tikhonov-style regularization is added to its diagonal
before computing the centroid.  If the data are
so noisy that the regularized determinant is still effectively zero,
the covariance will be large but the centroid estimates remain finite.

`ellipticity1_core` / `ellipticity2_core` and their
aperture counterparts in [`measure_star_shape`](@ref) are the same
estimator applied at two scales: the difference is the 3×3 truncation, not
the basis.  The core versions therefore agree with the aperture ones only
when the profile is well contained in the 3×3 box.  Once the source is much
broader than the box the core moment tensor saturates toward the box's own
moment and compresses the measured elongation -- for a 2:1 axis ratio at
6 px FWHM, `ellipticity1_core` reads ``\\approx -0.04`` where
`ellipticity1_aperture` reads ``< -0.5``.  The sign stays right; the
dynamic range does not.  Prefer the aperture values whenever the PSF is
broad, and use the core values for locality (a blend, a neighbor, a defect
inside the 3×3) rather than for magnitude.

!!! note "Sensitivity to sub-pixel phase"
    The quantities derived from the *quadratic fit* — `normalized_curvature`
    and the `poly` centroid — vary systematically with where the source falls
    inside its peak pixel, because a parabola fit over ``\\pm 1`` pixel
    recovers the profile curvature at the sampling phase rather than at the
    peak.  For a noiseless, perfectly round source scanned over all sub-pixel
    phases, `normalized_curvature` has a scatter of 9% (FWHM 1.2 px) to 4%
    (FWHM 3.5 px), and the polynomial centroid an error of 0.079 px to 0.013
    px.  Use them as relative, sigma-clipped quantities within a magnitude
    bin, not against absolute values.

    The moment-based `ellipticity1_core` and `ellipticity2_core` are far less
    phase-sensitive (for a separable profile the cross moment behind ``e_2``
    cancels exactly at any offset), but the 3×3 box truncation still leaves
    them well short of the aperture-scale statistics from
    [`measure_star_shape`](@ref), and it compresses their response to a real
    elongation once the source is much broader than the box.  Treat all core
    morphology as unreliable below Nyquist sampling.

!!! note
    This function assumes the inputs are valid 3×3 matrices.  Border
    checking (whether a full 3×3 neighborhood exists around the peak
    pixel) is the caller's responsibility — see [`centroid_poly`](@ref).
    `NaN` and `Inf` pixel values are not checked; they should be handled by
    a higher-level function.  Bad or saturated pixels can be masked by
    setting the corresponding entries in `inv_var` to zero.

# References
See [Vakili2016](@citet) for details.
"""
function _centroid_poly3(image::AbstractMatrix, inv_var::AbstractMatrix; background::Real = 0)
    FT = float(promote_type(eltype(image), eltype(inv_var)))
    bg = FT(background)

    # Coordinates:
    #
    #   image[1,1] image[1,2] image[1,3]     y = -1
    #   image[2,1] image[2,2] image[2,3]     y =  0
    #   image[3,1] image[3,2] image[3,3]     y =  1
    #
    #      x=-1       x=0       x=1
    #
    # Design row is (1, x, y, x^2, x*y, y^2).

    @inbounds begin
        z11 = FT(image[1,1]) - bg; z12 = FT(image[1,2]) - bg; z13 = FT(image[1,3]) - bg
        z21 = FT(image[2,1]) - bg; z22 = FT(image[2,2]) - bg; z23 = FT(image[2,3]) - bg
        z31 = FT(image[3,1]) - bg; z32 = FT(image[3,2]) - bg; z33 = FT(image[3,3]) - bg

        w11 = FT(inv_var[1,1]); w12 = FT(inv_var[1,2]); w13 = FT(inv_var[1,3])
        w21 = FT(inv_var[2,1]); w22 = FT(inv_var[2,2]); w23 = FT(inv_var[2,3])
        w31 = FT(inv_var[3,1]); w32 = FT(inv_var[3,2]); w33 = FT(inv_var[3,3])
    end

    wz11 = w11 * z11; wz12 = w12 * z12; wz13 = w13 * z13
    wz21 = w21 * z21; wz22 = w22 * z22; wz23 = w23 * z23
    wz31 = w31 * z31; wz32 = w32 * z32; wz33 = w33 * z33

    # Weighted geometric moments S_pq = Σ w_i x_i^p y_i^q.
    S00 = w11 + w12 + w13 + w21 + w22 + w23 + w31 + w32 + w33

    S10 = (w13 + w23 + w33) - (w11 + w21 + w31)
    S01 = (w31 + w32 + w33) - (w11 + w12 + w13)

    S20 = w11 + w13 + w21 + w23 + w31 + w33
    S02 = w11 + w12 + w13 + w31 + w32 + w33

    S11 = w11 - w13 - w31 + w33

    S21 = (w31 + w33) - (w11 + w13)
    S12 = (w13 + w33) - (w11 + w31)
    S22 = w11 + w13 + w31 + w33

    # Right-hand side r = A'Wz.
    R00 = wz11 + wz12 + wz13 + wz21 + wz22 + wz23 + wz31 + wz32 + wz33

    R10 = (wz13 + wz23 + wz33) - (wz11 + wz21 + wz31)
    R01 = (wz31 + wz32 + wz33) - (wz11 + wz12 + wz13)

    R20 = wz11 + wz13 + wz21 + wz23 + wz31 + wz33
    R02 = wz11 + wz12 + wz13 + wz31 + wz32 + wz33

    R11 = wz11 - wz13 - wz31 + wz33

    # Normal matrix: N = A'WA.  Aliases exploit x³=x, x⁴=x² (and same for y).
    Nmat = @SMatrix [
        S00 S10 S01 S20 S11 S02
        S10 S20 S11 S10 S21 S12
        S01 S11 S02 S21 S12 S01
        S20 S10 S21 S20 S11 S22
        S11 S21 S12 S11 S22 S11
        S02 S12 S01 S22 S11 S02
    ]

    rvec = @SVector [R00, R10, R01, R20, R11, R02]

    # The propagated covariance needs N⁻¹, so form it once and reuse it for
    # both the fitted coefficients and the output covariance.
    # TODO: Make this robust to PosDefException from singular or nearly
    # singular weighted 3×3 designs instead of relying on a pseudoinverse.
    Ninv = try
        C = cholesky(Symmetric(Nmat))
        SMatrix{6,6,FT}(inv(C))
    catch err
        err isa PosDefException || rethrow()
        SMatrix{6,6,FT}(pinv(Nmat))
    end

    X = Ninv * rvec
    a, b, c, d, e, f = X[1], X[2], X[3], X[4], X[5], X[6]

    # Centroid from the quadratic coefficients:
    #   D = [2d  e;  e  2f]
    #   x_c = (c e - 2 b f) / Δ,   y_c = (b e - 2 c d) / Δ,   Δ = 4 d f - e²
    two_d = 2 * d
    two_f = 2 * f
    Δ = two_d * two_f - e * e

    # TODO: Make polynomial-centroid failure recognition more robust: use a
    # scale-relative determinant test for D, reject non-maximum curvatures
    # before regularizing, apply sign-aware scale-relative damping only to
    # weak local maxima, and return an invalid polynomial estimate when the
    # damped curvature still fails validation so choose_centroid can use COM.
    # For now we just add Tikhonov-style regularization.
    if abs(Δ) < FT(1e-10)
        ε = FT(1e-8)
        two_d += ε
        two_f += ε
        Δ = two_d * two_f - e * e
    end

    invΔ = inv(Δ)

    xc = (c * e - b * two_f) * invΔ
    yc = (b * e - c * two_d) * invΔ

    xc2 = xc * xc
    yc2 = yc * yc
    xcyc = xc * yc

    amplitude = a + b * xc + c * yc + d * xc2 + e * xcyc + f * yc2
    peak = amplitude + bg

    # Normalized curvature — negated Laplacian divided by the fitted
    # amplitude above `background`.  For a circular Gaussian this is
    # ≈ 16log(2)/FWHM² and independent of flux.
    normalized_curvature = -(two_d + two_f) / max(abs(amplitude), eps(FT))

    # Jacobian of (yc, xc, peak) w.r.t. (a, b, c, d, e, f).
    # At the stationary point ∂P/∂x = ∂P/∂y = 0, so d(peak)/dθ simplifies
    # to the basis vector evaluated at the centroid. Note that the third row
    # is exact only if ε = 0 in the Tikhonov regularization above, but
    # the approximation is very good for small ε.
    J = @SMatrix [
        zero(FT)   e * invΔ         -two_d * invΔ   -(2 * c + 2 * two_f * yc) * invΔ   (b + 2 * e * yc) * invΔ   -2 * two_d * yc * invΔ
        zero(FT)  -two_f * invΔ     e * invΔ        -2 * two_f * xc * invΔ             (c + 2 * e * xc) * invΔ   -(2 * b + 2 * two_d * xc) * invΔ
        one(FT)    xc               yc              xc2                               xcyc                      yc2
    ]

    cov = J * Ninv * transpose(J)

    y_err = sqrt(max(zero(FT), cov[1,1]))
    x_err = sqrt(max(zero(FT), cov[2,2]))
    peak_err = sqrt(max(zero(FT), cov[3,3]))

    # σ(normalized_curvature).  normalized_curvature = -L/D with L = 2d + 2f
    # the Laplacian and D = max(|amplitude|, eps), so the Jacobian row combines
    # the Laplacian's own row, ∂L/∂θ = (0,0,0,2,0,2), with the amplitude row
    # (1, xc, yc, xc², xc·yc, yc²). This is `J`'s third row, since `peak` and `amplitude`
    # differ only by the constant `bg`.  Under the `eps` clamp the denominator
    # stops varying with the coefficients, so `dD` drops its derivative.
    L = two_d + two_f
    D = max(abs(amplitude), eps(FT))
    dD = abs(amplitude) > eps(FT) ? sign(amplitude) : zero(FT)
    invD = inv(D)
    g = L * dD * invD * invD
    j_nc = @SVector [g, g * xc, g * yc,
                     -2 * invD + g * xc2, g * xcyc, -2 * invD + g * yc2]
    normalized_curvature_err = sqrt(max(zero(FT), dot(j_nc, Ninv * j_nc)))

    # Inverse-variance-weighted center-of-mass on the 3×3 patch, and the
    # inverse of its total second central moment (compactness_core).  After
    # background subtraction the weighted sum R00 = Σ wᵢ(zᵢ - bg) can be
    # non-positive for a spurious peak or an over-subtracted background, in
    # which case the COM and compactness are undefined and returned as NaN;
    # choose_centroid then falls back to the polynomial centroid.
    if R00 > 0
        invR00 = inv(R00)
        com_x = R10 * invR00
        com_y = R01 * invR00

        # Delta-method covariance of the ratio estimator com = R / S00.
        # Var(R_pq) = S_{2p,2q}, Cov(R_pq, R_rs) = S_{p+r, q+s}.
        invR00_sq = invR00 * invR00
        var_com_x = (S20 - 2 * com_x * S10 + com_x * com_x * S00) * invR00_sq
        var_com_y = (S02 - 2 * com_y * S01 + com_y * com_y * S00) * invR00_sq
        cov_com_xy = (S11 - com_x * S01 - com_y * S10 + com_x * com_y * S00) * invR00_sq
        com_y_err = sqrt(max(zero(FT), var_com_y))
        com_x_err = sqrt(max(zero(FT), var_com_x))

        var_x = (R20 - 2 * com_x * R10 + com_x * com_x * R00) / R00
        var_y = (R02 - 2 * com_y * R01 + com_y * com_y * R00) / R00
        cov_xy = (R11 - com_x * R01 - com_y * R10 + com_x * com_y * R00) / R00
        var_sum = var_x + var_y
        if var_sum > 0
            compactness_core = 1 / var_sum
            ellipticity1_core = (var_y - var_x) / var_sum
            ellipticity2_core = 2 * cov_xy / var_sum

            # σ(compactness_core).  `var_sum` is a smooth function of the five
            # flux moments (R00, R10, R01, R20, R02), whose covariance is
            # Cov(R_pq, R_rs) = S_{p+r, q+s}.  On the 3x3 every third- and
            # fourth-order term collapses onto a sum already formed above,
            # because x, y ∈ {-1,0,1} gives x³ = x and x⁴ = x²: S30 -> S10,
            # S03 -> S01, S40 -> S20, S04 -> S02.  So this costs no new
            # accumulators, only algebra.
            Cm = @SMatrix [S00 S10 S01 S20 S02
                           S10 S20 S11 S10 S12
                           S01 S11 S02 S21 S01
                           S20 S10 S21 S20 S22
                           S02 S12 S01 S22 S02]
            # ∂var_sum/∂(R00, R10, R01, R20, R02).  The R00 row collapses to
            # -(var_sum - com_x² - com_y²)/R00 once the ratio terms are folded in.
            jv = @SVector [-(var_sum - com_x * com_x - com_y * com_y) * invR00,
                           -2 * com_x * invR00, -2 * com_y * invR00,
                           invR00, invR00]
            # compactness = 1/var_sum, so σ(compactness) = σ(var_sum)/var_sum².
            compactness_core_err = sqrt(max(zero(FT), dot(jv, Cm * jv))) *
                compactness_core * compactness_core
        else
            compactness_core = ellipticity1_core = ellipticity2_core = FT(NaN)
            compactness_core_err = FT(NaN)
        end
    else
        nan = FT(NaN)
        com_x = com_y = nan
        var_com_x = var_com_y = cov_com_xy = nan
        com_x_err = com_y_err = nan
        compactness_core = ellipticity1_core = ellipticity2_core = nan
        compactness_core_err = nan
    end
    com_cov = @SMatrix [var_com_y cov_com_xy; cov_com_xy var_com_x]

    return (; poly = (; y = yc, x = xc, peak,
                      y_err, x_err, peak_err, cov),
             com = (; y = com_y, x = com_x,
                     cov = com_cov,
                     y_err = com_y_err, x_err = com_x_err),
             normalized_curvature, normalized_curvature_err,
             compactness_core, compactness_core_err,
             ellipticity1_core, ellipticity2_core)
end

"""
    centroid_poly(image::AbstractMatrix [, inv_var]) -> NamedTuple

Polynomial centroid of a point source in `image`. This version is
designed to work with image cutouts containing a single point source,
where the brightest pixel is expected to be near the source centroid.

Finds the brightest pixel via `findmax`, extracts the surrounding 3×3
patch, fits a 2nd-order 2-D polynomial via weighted least squares (see
[`_centroid_poly3`](@ref)), and returns both the polynomial centroid and
the inverse-variance-weighted center-of-mass (COM) centroid in *global*
pixel coordinates.

# Arguments
- `image::AbstractMatrix`: image cutout containing a point source.
- `inv_var::AbstractMatrix`: per-pixel inverse variance (same size as
  `image`).  If omitted, a uniform inverse variance `Fill(1, size(image))`
  is used (equivalent to ordinary least squares).  Set entries to zero to
  mask bad or saturated pixels.
- `background::Real = 0`: scalar background subtracted from the 3×3 core
  before the fit.  Does not affect the polynomial centroid (`poly.y`,
  `poly.x`, `poly.cov`), but is required for `normalized_curvature`,
  `compactness_core`, `ellipticity1_core`, `ellipticity2_core`,
  and `com` to be meaningful on data with a nonzero
  sky level.  `poly.peak` is returned including `background`.
- `y_offset::Real = 0`, `x_offset::Real = 0`: origin of `image` in the
  caller's coordinate frame.  Added to every returned coordinate
  (`poly.y`, `poly.x`, `com.y`, `com.x`), so a caller working on a cutout
  extracted at `image[y_start:y_end, x_start:x_end]` passes
  `y_offset = y_start - 1`, `x_offset = x_start - 1` and gets results in
  the original image's coordinates.  Uncertainties, covariances, and all
  shape statistics are translation-invariant and unaffected.

# Returns
A `NamedTuple` with keys `(; poly, com, normalized_curvature,
normalized_curvature_err, compactness_core, compactness_core_err,
ellipticity1_core, ellipticity2_core)` where

- `poly` — `NamedTuple` `(; y, x, peak, y_err, x_err, peak_err, cov)`
  with the polynomial centroid in global pixel coordinates (row, column),
  the fitted image value at the centroid (including `background`), 1-σ
  uncertainties, and 3×3 `SMatrix` covariance of `(y, x, peak)`.  Access
  as `result.poly.y`, `result.poly.x`, etc.
- `com` — `NamedTuple` `(; y, x, y_err, x_err, cov)` with the
  inverse-variance-weighted center-of-mass centroid, its 1-σ
  uncertainties, and its 2×2 `SMatrix` covariance.  Access as
  `result.com.y`, `result.com.x`, etc.  All fields are `NaN` when the
  background-subtracted weighted sum is non-positive.
- `normalized_curvature` — negated Laplacian divided by the fitted
  amplitude above `background`; ``\\approx 16\\log(2)/\\mathrm{FWHM}^2``
  for a circular Gaussian.  This flux-independent statistic is useful for
  distinguishing stars from cosmic rays and hot pixels.
- `normalized_curvature_err` — 1-σ uncertainty on `normalized_curvature`.
- `compactness_core` — inverse of the total second central moment,
  `1 / (σ_x² + σ_y²)`, where `σ_x²` and `σ_y²` are the
  inverse‑variance‑weighted second central moments of the
  background‑subtracted pixel values.  Larger values indicate more
  compact (sharper) profiles.  Returns `NaN` if the weighted sum or the
  estimated variance sum is non‑positive.
- `compactness_core_err` — 1-σ uncertainty on `compactness_core`, by the
  delta method over the flux moments it is built from.
- `ellipticity1_core`, `ellipticity2_core` — normalized quadrupole
  components of the 3×3 second central moments,
  ``e_1 = (\\sigma^2_{yy}-\\sigma^2_{xx})/(\\sigma^2_{yy}+\\sigma^2_{xx})``
  and ``e_2 = 2\\sigma^2_{xy}/(\\sigma^2_{yy}+\\sigma^2_{xx})``.
  ``e_1 > 0`` is extended in ``y`` (rows), ``e_2 > 0`` is extended along
  the ``+45°`` diagonal; both are 0 for a circular core.  Unlike the
  quadratic-fit diagnostics, these are ratios of linear moment sums and
  are much less sensitive to sub-pixel phase.
If the brightest pixel lies on the image border (no full 3×3
neighborhood), every field is `NaN`:

```julia
(; poly = (; y = NaN, x = NaN, peak = NaN,
            y_err = NaN, x_err = NaN, peak_err = NaN,
            cov = @SMatrix [NaN NaN NaN; NaN NaN NaN; NaN NaN NaN]),
   com = (; y = NaN, x = NaN, y_err = NaN, x_err = NaN,
          cov = @SMatrix [NaN NaN; NaN NaN]),
   normalized_curvature = NaN, normalized_curvature_err = NaN,
   compactness_core = NaN, compactness_core_err = NaN,
   ellipticity1_core = NaN, ellipticity2_core = NaN)
```

# Examples
```jldoctest
julia> using CrowdPhot: centroid_poly

julia> img = [0.1 0.3 0.1; 0.3 1.0 0.3; 0.1 0.3 0.1];

julia> result = centroid_poly(img);

julia> round(result.poly.x; digits=1), round(result.poly.y; digits=1)
(2.0, 2.0)

julia> round(result.com.x; digits=1), round(result.com.y; digits=1)
(2.0, 2.0)
```

# References
See [Vakili2016](@citet) for details.
"""
function centroid_poly(
        image::AbstractMatrix{T},
        inv_var::AbstractMatrix = Fill(one(float(T)), size(image));
        background::Real = 0,
        y_offset::Real = 0,
        x_offset::Real = 0,
    ) where {T <: Real}
    _, maxidx = findmax(image)
    i0, j0 = Tuple(maxidx)  # row, column
    return centroid_poly(image, Int(i0), Int(j0), inv_var; background, y_offset, x_offset)
end
"""
    centroid_poly(image, i0::Int, j0::Int, inv_var; background=0) -> NamedTuple

Variant of [`centroid_poly`](@ref) that accepts pre-computed brightest-pixel
coordinates `i0, j0` (corresponding to pixel `image[i0, j0]`) instead of
calling `findmax` internally. Useful when the caller has already identified
the peak pixel (e.g. from a correlation map). `i0` is the row index
(y-coordinate) and `j0` is the column index (x-coordinate).

Returns the same `NamedTuple` as the two-argument form.
"""
function centroid_poly(
        image::AbstractMatrix{T},
        i0::Int,
        j0::Int,
        inv_var::AbstractMatrix = Fill(one(float(T)), size(image));
        background::Real = 0,
        y_offset::Real = 0,
        x_offset::Real = 0,
    ) where {T <: Real}
    # check that a full 3×3 neighborhood exists
    FT = float(promote_type(T, eltype(inv_var)))
    if i0 < 2 || i0 > size(image, 1) - 1 || j0 < 2 || j0 > size(image, 2) - 1
        nan = FT(NaN)
        nan3 = @SMatrix [nan nan nan; nan nan nan; nan nan nan]
        nan2 = @SMatrix [nan nan; nan nan]
        nancom = (; y = nan, x = nan, y_err = nan, x_err = nan, cov = nan2)
        return (; poly = (; y = nan, x = nan, peak = nan,
                          y_err = nan, x_err = nan, peak_err = nan,
                          cov = nan3),
                 com = nancom,
                 normalized_curvature = nan, normalized_curvature_err = nan,
                 compactness_core = nan, compactness_core_err = nan,
                 ellipticity1_core = nan, ellipticity2_core = nan)
    end

    # extract 3×3 views
    patch = view(image, i0-1:i0+1, j0-1:j0+1)
    wpatch = view(inv_var, i0-1:i0+1, j0-1:j0+1)

    # delegate to the 3×3 solver
    local_result = _centroid_poly3(patch, wpatch; background)

    # convert local → global coordinates
    # i0 is row (y), j0 is column (x); `*_offset` shifts the cutout's origin
    # into the caller's frame so no caller has to translate the result.
    oy = FT(i0) + FT(y_offset)
    ox = FT(j0) + FT(x_offset)
    return (; poly = (; y = oy + local_result.poly.y,
                       x = ox + local_result.poly.x,
                       peak = local_result.poly.peak,
                       y_err = local_result.poly.y_err,
                       x_err = local_result.poly.x_err,
                       peak_err = local_result.poly.peak_err,
                       cov = local_result.poly.cov),
             com = (; y = oy + local_result.com.y,
                     x = ox + local_result.com.x,
                     y_err = local_result.com.y_err,
                     x_err = local_result.com.x_err,
                     cov = local_result.com.cov),
             normalized_curvature = local_result.normalized_curvature,
             normalized_curvature_err = local_result.normalized_curvature_err,
             compactness_core = local_result.compactness_core,
             compactness_core_err = local_result.compactness_core_err,
             ellipticity1_core = local_result.ellipticity1_core,
             ellipticity2_core = local_result.ellipticity2_core)
end

"""
    choose_centroid(result) -> NamedTuple

Given the `NamedTuple` returned by [`centroid_poly`](@ref) or
[`_centroid_poly3`](@ref), choose between the polynomial centroid
(`result.poly.y`, `result.poly.x`) and the center-of-mass centroid
(`result.com.y`, `result.com.x`).

The polynomial centroid is preferred for well-sampled data where the
3×3 patch has enough curvature for a reliable quadratic fit.  The COM
centroid is chosen when the polynomial result is non-finite, has invalid
covariance, or has a polynomial-vs-COM variance ratio exceeding 10² in
either coordinate.

Returns `(; y, x, source)` where `source` is `:poly` or `:com`.

!!! note
    This heuristic detects curvature degeneracy but cannot detect
    quadratic model bias on undersampled data.  For undersampled
    images the matched-filter step in the detection pipeline broadens
    the PSF enough that the polynomial centroid is usually reliable.
    If you are centroiding raw (un-convolved) undersampled data,
    prefer the COM centroid directly.
"""
function choose_centroid(result)
    # Validate both estimators before comparing their covariance scales.
    poly_y_var = result.poly.cov[1,1]
    poly_x_var = result.poly.cov[2,2]
    com_y_var  = result.com.cov[1,1]
    com_x_var  = result.com.cov[2,2]

    poly_ok = isfinite(result.poly.y) && isfinite(result.poly.x) &&
              isfinite(poly_y_var) && isfinite(poly_x_var) &&
              poly_y_var > 0 && poly_x_var > 0

    com_ok = isfinite(result.com.y) && isfinite(result.com.x) &&
             isfinite(com_y_var) && isfinite(com_x_var) &&
             com_y_var >= 0 && com_x_var >= 0

    # Only meaningful when COM variances are finite/valid, but safe to compute here.
    y_scale = max(com_y_var, eps(float(typeof(com_y_var))))
    x_scale = max(com_x_var, eps(float(typeof(com_x_var))))

    poly_degenerate = poly_ok && com_ok &&
                      (poly_y_var > 100 * y_scale ||
                       poly_x_var > 100 * x_scale)

    if !poly_ok && com_ok
        # Polynomial failed, COM is valid.
        return (; y = result.com.y, x = result.com.x, source = :com)

    elseif poly_degenerate
        # Both are valid, but polynomial covariance is much worse than COM.
        return (; y = result.com.y, x = result.com.x, source = :com)

    else
        # Prefer polynomial in all other cases:
        # - polynomial valid, COM invalid
        # - both valid and polynomial is not degenerate
        # - both invalid, preserving the behavior of the original code
        return (; y = result.poly.y, x = result.poly.x, source = :poly)
    end
end
