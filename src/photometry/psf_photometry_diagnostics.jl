# Shared per-star goodness-of-fit diagnostics for PSF photometry.
#
# `fit_all_stars` (sequential) and `fit_all_stars_simultaneous` compute the
# same qfit / qfit_expected / qfit_z / crowding / spread_model statistics so
# their results compare field-for-field.  The math below is extracted from the
# sequential path (psf_photometry_single.jl) so both arms share one
# implementation.  All inputs are stamp-local: `(length(yr), length(xr))` blocks
# aligned to the star's fitting box, so the only per-arm difference is how those
# blocks are produced.

# Mixture-of-Gaussians approximation of a unit-integral circular exponential
# profile exp(-r/h), from David Hogg & Dustin Lang's "The Tractor"
# (via crowdsource/galconv.py `ExpGalaxy`).  `_EXPGAL_VAR` is in units of the
# exponential effective (half-light) radius squared; `re = 1.67834699 * h`.
const _EXPGAL_AMP = (1.99485977e-4, 2.61612679e-3, 1.89726655e-2, 1.00186544e-1, 3.68534484e-1, 5.09490694e-1)
const _EXPGAL_VAR = (1.20078965e-3, 8.84526493e-3, 3.91463084e-2, 1.39976817e-1, 4.60962500e-1, 1.50159566e0)

"""
    _exp_disk_kernel(fwhm::Real, ::Type{FT}) -> Matrix{FT}

Fixed 5x5, native-pixel-scale convolution kernel for the SExtractor
`spread_model` reference: a unit-integral circular exponential disk with
scalelength `h = fwhm / 16`, integrated exactly over each detector pixel.

The raw `exp(-r/h)` does not separate in `x`/`y`, so it has no closed-form
pixel integral; instead the profile is represented as a sum of 6 concentric
circular Gaussians ([`_EXPGAL_AMP`](@ref)/[`_EXPGAL_VAR`](@ref)) and each
Gaussian component is pixel-integrated with the same `erf` formula
[`CircularGaussianPRF`](@ref) uses.  The kernel is renormalized to sum 1
(the mass outside the 5x5 support is `< 1e-3` for `fwhm` up to ~6 px).
"""
function _exp_disk_kernel(fwhm::Real, ::Type{FT}) where {FT}
    re = FT(fwhm) / 16 * FT(1.67834699)
    fwhm_per_sigma = FT(2) * sqrt(FT(2) * log(FT(2)))
    half = 2
    K = zeros(FT, 2half + 1, 2half + 1)
    for (a, v) in zip(_EXPGAL_AMP, _EXPGAL_VAR)
        sigma_k = sqrt(FT(v)) * re
        comp = PSF.CircularGaussianPRF(y = zero(FT), x = zero(FT),
            fwhm = sigma_k * fwhm_per_sigma, flux = FT(a), bkg = zero(FT))
        for (jj, j) in enumerate(-half:half), (ii, i) in enumerate(-half:half)
            K[ii, jj] += PSF.evaluate(comp, i, j)
        end
    end
    K ./= sum(K)
    return K
end

"""
    _exp_disk_kernel_bandlimited(fwhm::Real, ::Type{FT}; half::Integer = ...) -> Matrix{FT}

Alternative `spread_model` reference-disk kernel: the same Gaussian-mixture
circular exponential disk as [`_exp_disk_kernel`](@ref) (`h = fwhm / 16`), but
**band-limited to the pixel Nyquist frequency** instead of pixel-integrated.

This is the effective spatial kernel of the Fourier-space convolution used by
SExtractor and ``\\texttt{crowdsource}`` (`irfft` of the analytic disk
transform), and it is the correct kernel when the PSF stamp is treated as
band-limited / sinc-interpolated -- the right assumption for a well-sampled PSF.
Against the analytic PSF-convolved-with-disk truth for a Gaussian PSF it agrees
to well under 1% for `fwhm <= 4` px, whereas the pixel-integrated
[`_exp_disk_kernel`](@ref) is 15-50% off with a `fwhm`-dependent sign of bias.

Each mixture component `k` contributes a separable outer product
`I(mx; var_k) * I(my; var_k)`, with the 1-D band-limited profile

    I(m; v) = ∫_{-1/2}^{1/2} exp(-a k^2) cos(2π k m) dk
            = exp(-π^2 m^2 / a) * sqrt(π / a) * real(erf(sqrt(a)/2 + im*π*m/sqrt(a))),
    a = 2 π^2 re^2 v,   re = 1.67834699 * h.

The kernel has slowly decaying negative side-lobes (sinc ringing), so its
support (`2*half + 1` per axis) grows with the disk size; `half` defaults
accordingly and may be overridden.  Renormalized to sum 1.

The `I(m; v)` closed form is evaluated in `Float64` regardless of `FT`: the
`erf` argument has a large imaginary part for the wide components at large `m`,
so `erf` itself is `O(exp(π^2 m^2 / a))` and overflows in `Float32` before the
`exp(-π^2 m^2 / a)` prefactor cancels it.  The finished kernel is converted to
`FT`.
"""
function _exp_disk_kernel_bandlimited(fwhm::Real, ::Type{FT};
        half::Integer = ceil(Int, 2.5 * fwhm)) where {FT}
    re = Float64(fwhm) / 16 * 1.67834699
    twopi2 = 2 * π^2
    Iband(m::Integer, v) = begin
        a = twopi2 * re^2 * Float64(v)
        π^2 * m^2 / a > 690 && return 0.0
        sa = sqrt(a)
        real(exp(-π^2 * m^2 / a) * sqrt(π / a) * PSF.erf(sa / 2 + im * π * m / sa))
    end
    n = 2half + 1
    K = zeros(Float64, n, n)
    for (a, v) in zip(_EXPGAL_AMP, _EXPGAL_VAR)
        Iv = Float64[Iband(m, v) for m in -half:half]
        for jj in 1:n, ii in 1:n
            K[ii, jj] += a * Iv[ii] * Iv[jj]
        end
    end
    K ./= sum(K)
    return convert(Matrix{FT}, K)
end

"""
    _diagnostic_sinks(::Type{FT}, n) -> NamedTuple of `n`-element vectors

The per-source fit-quality columns [`_star_diagnostics!`](@ref) writes into:
`chisq`, `qfit`, `qfit_expected`, `qfit_z`, `crowding`, `spread_model`,
`spread_model_err`.

Every column starts at `NaN`, so a source whose diagnostics were never computed
(non-positive flux, too few unmasked pixels) is distinguishable from one that
was measured.  `chisq` included: a zero there would read as a perfect fit.
"""
function _diagnostic_sinks(::Type{FT}, n::Integer) where {FT}
    nanv() = fill(convert(FT, NaN), n)
    return (; chisq = nanv(), qfit = nanv(), qfit_expected = nanv(), qfit_z = nanv(),
              crowding = nanv(), spread_model = nanv(), spread_model_err = nanv())
end

"""
    _star_diagnostics!(diag, idx, model, image, resid, star_model, g_model,
                       inv_var, n_free)

Compute the per-star `chisq`, `qfit`, `qfit_expected`, `qfit_z`, `crowding`, and
`spread_model` diagnostics for star `idx` and write them into `diag`, the
[`_diagnostic_sinks`](@ref) NamedTuple.

Nothing is written for a source with non-positive `model.flux`: every statistic
here is normalized by the flux, so none of them means anything in that case.
All three fitters gate or drop such sources before reaching this point.

`image`, `resid`, `star_model`, `g_model`, and `inv_var` are all *stamp-local*
matrices of the same shape, covering the star's fitting box:

- `resid` is the residual of this star's fit: the observed data with every
  other star's best-fit model removed *and* this star's own model subtracted.
- `star_model` is this star's best-fit model rendered over the box
  (`evaluate(model, y, x)`).  The neighbor-subtracted "clean" value the
  crowding statistic needs is reconstructed pixel-wise as
  `resid[I] + star_model[I]`.
- `g_model` is `star_model` convolved with the [`_exp_disk_kernel`](@ref)
  reference disk (SExtractor `spread_model`), or `nothing` to skip
  `spread_model` / `spread_model_err` (they are left untouched, i.e. `NaN`).
- `image` is the original data (used for the `crowding` "dirty" flux).
- `inv_var` is the per-pixel inverse variance, or `nothing` for unweighted.
  `spread_model` uses `inv_var` weighting when present (unit weights otherwise);
  `spread_model_err` requires `inv_var` and is left `NaN` without it.

Only the scalar `flux`/`bkg` of `model` are read here.  `n_free` is the number
of free parameters, used both to correct `qfit_z` for fitting leverage and as
the `chisq` degrees-of-freedom subtraction.
"""
function _star_diagnostics!(
        diag::NamedTuple, idx::Int, model, image, resid, star_model, g_model,
        inv_var, n_free::Int
    )
    FT = float(eltype(image))
    flux = FT(model.flux)
    bkg = FT(model.bkg)
    flux > 0 || return nothing

    inv_flux = inv(flux)
    qfit_val = zero(FT)
    chisq_num = zero(FT)
    n_pix_good = 0
    num_clean = zero(FT)
    num_dirty = zero(FT)
    den_crowd = zero(FT)
    do_spread = g_model !== nothing
    Sp = zero(FT); Sq = zero(FT); Sp0 = zero(FT); Sq0 = zero(FT); SqG = zero(FT)
    for I in CartesianIndices(star_model)
        model_val = star_model[I]
        wp = inv_var !== nothing ? inv_var[I] : one(FT)
        if isfinite(wp) && wp > 0
            qfit_val += abs(resid[I])
            chisq_num += wp * resid[I]^2
            n_pix_good += 1
            # Unit-flux PSF kernel for the crowding calculation.
            Pp = (model_val - bkg) * inv_flux
            wP = wp * Pp
            p_i = resid[I] + model_val - bkg
            num_clean += wP * p_i
            num_dirty += wP * (image[I] - bkg)
            den_crowd += wP * Pp
            if do_spread
                g_i = (g_model[I] - bkg) * inv_flux
                Sp += wP * p_i          # sum w * phi * p
                Sq += wp * g_i * p_i    # sum w * G   * p
                Sp0 += wP * Pp          # sum w * phi * phi
                Sq0 += wp * Pp * g_i    # sum w * phi * G
                SqG += wp * g_i * g_i   # sum w * G   * G
            end
        end
    end
    diag.qfit[idx] = qfit_val * inv_flux
    if n_pix_good > n_free
        diag.chisq[idx] = chisq_num / (n_pix_good - n_free)
    end
    if den_crowd > 0 && num_clean > 0 && num_dirty > 0
        diag.crowding[idx] = FT(2.5) * log10(num_dirty / num_clean)
    end

    # spread_model: SExtractor / crowdsource star-galaxy discriminant.  Zero for
    # a point source (p proportional to phi), positive for extended sources.
    if do_spread && Sp > 0
        diag.spread_model[idx] = Sq / Sp - Sq0 / Sp0
        if inv_var !== nothing
            v = Sp^2 * SqG + Sq^2 * Sp0 - 2 * Sq * Sp * Sq0
            diag.spread_model_err[idx] = sqrt(max(zero(FT), v)) / Sp^2
        end
    end

    # qfit_expected and qfit_z depend only on inv_var and the box.
    if !isnothing(inv_var)
        sigma_sum = zero(FT)
        sigma2_sum = zero(FT)
        for I in CartesianIndices(inv_var)
            iv = inv_var[I]
            if isfinite(iv) && iv > 0
                sigma_i = inv(sqrt(iv))
                sigma_sum += sigma_i
                sigma2_sum += sigma_i^2
            end
        end
        diag.qfit_expected[idx] = FT(sqrt(2 / FT(π))) * sigma_sum * inv_flux
        if sigma2_sum > 0 && n_pix_good > n_free
            dof_factor = FT(sqrt(1 - n_free / n_pix_good))
            num = qfit_val - FT(sqrt(2 / FT(π))) * dof_factor * sigma_sum
            den = FT(sqrt((1 - 2 / FT(π)) * sigma2_sum)) * dof_factor
            diag.qfit_z[idx] = num / den
        end
    end
    return nothing
end
