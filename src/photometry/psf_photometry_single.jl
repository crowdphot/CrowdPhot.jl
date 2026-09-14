# PSF-fitting photometry for a single image.
#
# Implements a DOLPHOT-style multi-pass algorithm: stars are sorted by
# brightness and fitted one at a time against a residual image from which
# all brighter (and, on later passes, all other) stars have been subtracted.
# On passes 2+, each star is added back before re-fitting so it sees the
# original data minus only its neighbors.

# ==============================================================================
# Result type
# ==============================================================================

"""
    MultiPassPhotResult{T}

Result of [`fit_all_stars`](@ref).  All per-star vectors have the same
length (the number of input sources).

# Fields

- `y`, `x`: fitted centroid positions.
- `y_err`, `x_err`: 1-sigma centroid errors (`zero(T)` if positions were fixed).
- `flux`: fitted flux.
- `flux_err`: 1-sigma flux error (`zero(T)` if flux was fixed).
- `bkg`: fitted local background.
- `bkg_err`: 1-sigma background error (`zero(T)` if background was fixed).
- `converged::BitVector`: whether the final LM fit converged.
- `valid::BitVector`: whether the source survived between-pass validation.
- `n_iter::Vector{Int}`: total LM iterations accumulated across all passes for each source.
- `n_passes::Int`: number of passes actually performed.
- `n_failed::Int`: number of stars whose fits threw exceptions and were marked
  invalid.  A non-zero count indicates systematic problems (e.g. shape
  mismatches in `inv_var`) and a warning is emitted.
- `failure_msgs::Vector{String}`: first few exception messages from failed fits,
  for diagnosis.  Empty when `n_failed == 0`.
- `residual::Matrix{T}`: final residual image after all subtractions.

# Goodness-of-fit diagnostics
- `chisq::Vector{T}`: final reduced χ² for each source.  Uses squared residuals (L2 norm), so a
  single bad pixel (cosmic ray, hot pixel) can dominate.  Best for catching
  single-pixel excursions.
- `qfit::Vector{T}`: sum of absolute residuals over valid-weight pixels in the
  fitting aperture divided by flux.  Uses L1 norm, so it is robust to single-pixel
  outliers and sensitive to persistent misfit across many pixels (PSF mismatch,
  extended sources).  Computed on the final pass from the best-fit model;
  check `valid` and `converged` before interpreting.  Same statistic as in
  ``\\texttt{hst1pass}``, see [Anderson2022hst1pass](@citet) page 10.
- `qfit_expected::Vector{T}`: expected `qfit` under the noise model
  ``\\left(\\sqrt{2/\\pi} \\sum \\sigma_i \\,/\\, \\mathrm{F}\\right)``.
  The difference and/or ratio between `qfit` and `qfit_expected` can be used to
  identify stars whose fits are worse than the noise model predicts, though `qfit_z`
  is likely to be more informative. `NaN` when `inv_var` was not provided.
- `qfit_z::Vector{T}`: noise-standardized excess absolute residual:
  ```math
  q_{\\rm fit,z} =
  \\frac{\\sum |r_i| - \\sqrt{\\frac{2}{\\pi}(1 - p/N)} \\sum \\sigma_i}
  {\\sqrt{(1 - \\frac{2}{\\pi})(1 - p/N) \\sum \\sigma_i^2}},
  \\qquad
  r_i = P_i - s - F\\psi_i .
  ```
  where ``p`` is the number of free parameters and ``N`` is the number of
  valid-weight pixels in the fitting aperture.  The ``\\sqrt{1 - p/N}``
  factors correct for the reduction in residual variance due to fitting:
  with ``p`` parameters estimated from ``N`` data points, each residual's
  variance is reduced by approximately ``(1 - p/N)`` (average leverage
  ``h_{ii} \\approx p/N``).  The expectation of ``|r_i|`` for a half-normal
  scales with the standard deviation, so both the numerator expectation and
  the denominator standard deviation are multiplied by ``\\sqrt{1 - p/N}``.
  Under the null hypothesis, ``q_{\\rm fit,z} \\sim \\mathcal{N}(0,1)`` to
  within ``\\mathcal{O}(p^2/N^2)``.  Large positive values indicate
  structured residuals inconsistent with pure noise.  `NaN` when `inv_var`
  was not provided or when the aperture has fewer valid-weight pixels than
  free parameters.
- `crowding::Vector{T}`: DOLPHOT-like blend-contamination diagnostic:
  ``2.5 \\log_{10}(F_{\\rm dirty} / F_{\\rm clean})`` where ``F_{\\rm clean}``
  is measured on the neighbor-subtracted image and ``F_{\\rm dirty}`` is
  measured on the original image with all neighbors present.  Values near zero
  indicate an isolated source; positive values indicate contamination by
  neighbor light.  Computed on the final pass; check `valid` and `converged`
  before interpreting.
- `spread_model::Vector{T}`: SExtractor star/galaxy discriminant:
  ```math
  \\mathrm{spread\\_model} = \\frac{\\boldsymbol{G}^\\mathsf{T} \\mathbf{W} \\boldsymbol{p}}
  {\\boldsymbol{\\phi}^\\mathsf{T} \\mathbf{W} \\boldsymbol{p}}
  - \\frac{\\boldsymbol{\\phi}^\\mathsf{T} \\mathbf{W} \\boldsymbol{G}}
  {\\boldsymbol{\\phi}^\\mathsf{T} \\mathbf{W} \\boldsymbol{\\phi}}
  ```
  where ``\\boldsymbol{p}`` is the neighbor-subtracted background-subtracted image,
  ``\\boldsymbol{\\phi}`` the unit-flux local PSF, ``\\boldsymbol{G}`` the PSF
  convolved with a circular exponential disk of scalelength ``\\rm FWHM/16``,
  and ``\\mathbf{W}`` the inverse-variance weights (unit weights when `inv_var`
  is not supplied, which changes the value).  Near zero for point sources,
  positive for extended sources, negative for detections sharper than the PSF
  (e.g. cosmic rays).
- `spread_model_err::Vector{T}`: 1-sigma uncertainty on `spread_model` from
  pixel-noise propagation of the weighted estimator.  `NaN` when `inv_var` was
  not provided.
"""
struct MultiPassPhotResult{T}
    y::Vector{T}
    x::Vector{T}
    y_err::Vector{T}
    x_err::Vector{T}
    flux::Vector{T}
    flux_err::Vector{T}
    bkg::Vector{T}
    bkg_err::Vector{T}
    converged::BitVector
    valid::BitVector
    chisq::Vector{T}
    qfit::Vector{T}
    qfit_expected::Vector{T}
    qfit_z::Vector{T}
    crowding::Vector{T}
    spread_model::Vector{T}
    spread_model_err::Vector{T}
    n_iter::Vector{Int}
    n_passes::Int
    n_failed::Int
    failure_msgs::Vector{String}
    residual::Matrix{T}
end

# ==============================================================================
# Source catalog extraction
# ==============================================================================

# Shared helper: build the internal params matrix from per-star initial values.
# `overrides` is a NamedTuple mapping field-name → Vector of per-star values
# for the fields we want to set from the source catalog.  Rows for other PSF
# fields are filled from the PSF model defaults and are identical across stars.
function _build_params_matrix(psf, overrides::NamedTuple, T::Type)
    defaults = ConstructionBase.getproperties(psf)
    prop_names = collect(keys(defaults))
    n_params = length(prop_names)
    n_stars = length(first(values(overrides)))
    params = Matrix{T}(undef, n_params, n_stars)
    errors = Matrix{T}(undef, n_params, n_stars)
    for (k, name) in enumerate(prop_names)
        vals = if haskey(overrides, name)
            T.(getfield(overrides, name))
        else
            fill(T(getfield(defaults, name)), n_stars)
        end
        params[k, :] .= vals
        errors[k, :] .= T(NaN)
    end
    return params, errors
end

"""
    _extract_source_catalog(sources::Vector{<:NamedTuple}, psf, T)

Extract initial source parameters from the output of
[`measure_star_shapes`](@ref).  Uses `centroid.y`, `centroid.x`,
`matched_filter_flux` as the initial flux guess; `bkg` defaults to zero.
"""
function _extract_source_catalog(sources::Vector{<:NamedTuple}, psf, ::Type{T}) where {T}
    n = length(sources)
    y  = T[s.centroid.y for s in sources]
    x  = T[s.centroid.x for s in sources]
    flux = T[s.matched_filter_flux for s in sources]
    bkg = zeros(T, n)
    return _build_params_matrix(psf, (; y, x, flux, bkg), T)
end

"""
    _extract_source_catalog(sources::NamedTuple, psf, T)

Extract initial source parameters from a `NamedTuple` with fields `:y` and
`:x` (required).  Optional fields `:flux` (defaults to 1) and `:bkg`
(defaults to 0).
"""
function _extract_source_catalog(sources::NamedTuple, psf, ::Type{T}) where {T}
    n = length(sources.y)
    y  = T.(sources.y)
    x  = T.(sources.x)
    flux = haskey(sources, :flux) ? T.(sources.flux) : fill(T(1), n)
    bkg  = haskey(sources, :bkg)  ? T.(sources.bkg)  : zeros(T, n)
    return _build_params_matrix(psf, (; y, x, flux, bkg), T)
end

"""
    _extract_source_catalog(mf::MatchedFilterResult, psf, T)

Extract initial source parameters from a [`MatchedFilterResult`](@ref).
Uses `Tuple.(peaks)` for pixel-center `(y, x)`, `peak_fluxes` for initial
flux, and zero background.
"""
function _extract_source_catalog(mf::MatchedFilterResult, psf, ::Type{T}) where {T}
    n = length(mf.peaks)
    y  = T[Tuple(p)[1] for p in mf.peaks]
    x  = T[Tuple(p)[2] for p in mf.peaks]
    flux = T.(mf.peak_fluxes)
    bkg = zeros(T, n)
    return _build_params_matrix(psf, (; y, x, flux, bkg), T)
end

# ==============================================================================
# Error extraction
# ==============================================================================

"""
    _extract_errors!(errors, cov, free_idx, is_fixed, i::Int)

Extract 1-sigma parameter errors from the `(n_free × n_free)` covariance
matrix `cov` into column `i` of `errors`.  Free-parameter errors are
`sqrt(cov[j, j])`; fixed-parameter errors are set to `zero(T)`.
"""
function _extract_errors!(errors::Matrix{T}, cov, free_idx, is_fixed, i::Int) where {T}
    for (j, k) in enumerate(free_idx)
        errors[k, i] = T(sqrt(max(zero(T), cov[j, j])))
    end
    for k in is_fixed
        errors[k, i] = zero(T)
    end
    return errors
end

# ==============================================================================
# Core algorithm
# ==============================================================================

"""
    fit_all_stars(image, psf, sources, fit_rad; kws...) -> MultiPassPhotResult

Perform multi-pass PSF-fitting photometry on all sources in `image`.

Stars are sorted by brightness and fitted one at a time against a
progressive residual image.  Each fitted model is subtracted before the
next star is processed, so fainter neighbors are measured after brighter
stars have been removed.  On subsequent passes each star is added back,
re-fitted, and re-subtracted, progressively refining all measurements.

# Arguments

- `image::AbstractMatrix`: the background-subtracted image.
- `psf::AbstractPSFModel`: PSF model shared by all stars.  Its structural
  parameters (FWHM, shape, etc.) are fixed; `y`, `x`, `flux`, `bkg` are
  fitted per star (unless frozen via `fixed`).
- `sources`: source catalog.  Accepted formats:
  - `Vector{<:NamedTuple}` — output of [`measure_star_shapes`](@ref).
  - `NamedTuple` with `(:y, :x)` fields and optional `(:flux, :bkg)`.
  - [`MatchedFilterResult`](@ref).
- `fit_rad::Real`: fitting radius in detector pixels.  A ±`fit_rad`
  rectangular cutout is extracted around each star's current position.

# Keyword arguments

- `n_passes::Integer = 3`: number of full subtract/add-back/re-fit cycles.
- `fixed::NamedTuple = (;)`: parameters frozen for ALL stars, e.g.
  `(; bkg)` or `(; x, y)`.
- `inv_var`: per-pixel inverse variance for weighted fitting.  Must be the same
  shape as `image`.  Forwarded to [`fit_star`](@ref CrowdPhot.PSF.fit_star),
  where non-positive or non-finite values are treated as masked pixels.  Stars
  with too few valid pixels are marked invalid and skipped.

  Interpreted as the **total** inverse variance, including the Poisson
  contribution of the source flux itself -- what a reduction pipeline's
  `err`/`weight` array normally provides.  The map is used exactly as given;
  supplying one also selects [`KnownWeightsCovarianceEstimator`](@ref) (absent
  IRLS reweighting), which trusts it, so a background-only map yields position
  and flux errors that are too small for bright sources.
- All other keyword arguments (`max_iter`, `x_tol`, `f_tol`,
  `g_tol`, `show_trace`, `reweight`, `covariance_estimator`,
  `scale_estimator`, `damping`, etc.) are forwarded to
  [`fit_star`](@ref CrowdPhot.PSF.fit_star).

# Returns

A [`MultiPassPhotResult`](@ref) containing fitted parameters, errors,
convergence flags, and the final residual image.
"""
function fit_all_stars(
        image::AbstractMatrix{T},
        psf::AbstractPSFModel,
        sources,
        fit_rad::Real;
        n_passes::Integer = 3,
        fixed::NamedTuple = (;),
        inv_var = nothing,
        spread_model_fwhm::Union{Nothing, Real} = nothing,
        kws...,
    ) where {T}
    FT = float(T)
    n_passes > 0 || throw(ArgumentError("n_passes must be positive, got $n_passes"))

    # -------------------------------------------------------------------
    # 1. Extract source catalog into contiguous params/errors matrices
    # -------------------------------------------------------------------
    params, errors = _extract_source_catalog(sources, psf, FT)
    n_params, n_stars = size(params)
    n_stars == 0 && return MultiPassPhotResult(
        FT[], FT[], FT[], FT[], FT[], FT[], FT[], FT[],
        falses(0), falses(0), FT[], FT[], FT[], FT[], FT[], FT[], FT[], Int[], Int(0), Int(0), String[], Matrix{FT}(undef, 0, 0),
    )

    # Map PSF property names to matrix row indices.
    prop_names = collect(keys(ConstructionBase.getproperties(psf)))
    @assert length(prop_names) == n_params

    # Identify which rows hold y, x, flux, bkg for sorting and validation.
    row_y = findfirst(==(:y), prop_names)
    row_x = findfirst(==(:x), prop_names)
    row_flux = findfirst(==(:flux), prop_names)
    row_bkg = findfirst(==(:bkg), prop_names)

    # -------------------------------------------------------------------
    # 2. Free-parameter metadata (computed once)
    # -------------------------------------------------------------------
    free_names, free_idx, _ = PSF.free_params(psf, fixed)
    isempty(free_idx) && throw(ArgumentError("all model parameters are fixed; nothing to fit"))
    is_fixed = Tuple(setdiff(1:n_params, free_idx))

    # Set fixed-parameter errors to zero immediately (won't change).
    for k in is_fixed
        errors[k, :] .= zero(FT)
    end

    # -------------------------------------------------------------------
    # 3. Copy image for progressive subtraction
    # -------------------------------------------------------------------
    residual = Matrix{FT}(image)
    # Small per-star model-render buffer for the final-pass diagnostics, reused
    # across stars.  A ±fit_rad box spans at most 2*ceil(fit_rad)+2 pixels per
    # axis (worst case: a star centered on a half-integer coordinate).
    S_max = 2 * ceil(Int, fit_rad) + 2
    model_stamp = Matrix{FT}(undef, S_max, S_max)
    # spread_model reference: one field-constant exponential-disk kernel, plus a
    # reused buffer for the per-star PSF-convolved-with-disk stamp.
    spread_fwhm = spread_model_fwhm === nothing ?
        _spread_fwhm(psf, FT(params[row_y, 1]), FT(params[row_x, 1])) : FT(spread_model_fwhm)
    # Resolved to tuple form once, which stops `correlate!` re-running its
    # separability test -- an SVD for a matrix kernel -- on every source.
    spread_kernel = isfinite(spread_fwhm) && spread_fwhm > 0 ?
        _canonicalize(_exp_disk_kernel_bandlimited(spread_fwhm, FT; half = ceil(Int, fit_rad))) : nothing
    g_stamp = Matrix{FT}(undef, S_max, S_max)

    # -------------------------------------------------------------------
    # 4. Per-star state vectors
    # -------------------------------------------------------------------
    valid = trues(n_stars)
    converged = falses(n_stars)
    chisq = zeros(FT, n_stars)
    qfit = fill(convert(FT, NaN), n_stars)
    qfit_expected = fill(convert(FT, NaN), n_stars)
    qfit_z = fill(convert(FT, NaN), n_stars)
    crowding = fill(convert(FT, NaN), n_stars)
    spread_model = fill(convert(FT, NaN), n_stars)
    spread_model_err = fill(convert(FT, NaN), n_stars)
    n_iter = zeros(Int, n_stars)
    n_failed = 0
    failure_msgs = String[]

    # -------------------------------------------------------------------
    # 5. Multi-pass loop
    # -------------------------------------------------------------------
    for pass in 1:n_passes
        # Sort by flux descending (brightest first) so brighter stars
        # are subtracted before fainter neighbors are fitted.
        order = sortperm(view(params, row_flux, :); rev = true)

        for idx in order
            valid[idx] || continue

            # Build model from all parameters in the params column.
            # Fixed parameters retain their per-star initial values.
            all_vals = NamedTuple{Tuple(prop_names)}(ntuple(k -> params[k, idx], Val(n_params)))
            m = ConstructionBase.setproperties(psf, all_vals)

            # Pixel footprint of ±fit_rad around the star center, clamped
            # to image bounds.
            FT_fit = FT(fit_rad)
            yr = floor(Int, m.y - FT_fit):ceil(Int, m.y + FT_fit)
            xr = floor(Int, m.x - FT_fit):ceil(Int, m.x + FT_fit)
            yr, xr = _clamp_inds(yr, xr, residual)
            length(yr) * length(xr) < 3 && (valid[idx] = false; continue)
            inds = CartesianIndices((yr, xr))

            # Fit the star on the residual image.  A failed fit (e.g. too
            # few valid inv_var pixels) marks the star invalid and continues
            # to the next source without crashing the full run.
            local best, result
            try
                # On passes 2+, add this star's previous model back so it is
                # fitted against data containing only its own signal (plus
                # noise); all neighbors are already subtracted.
                if pass > 1
                    PSF.add_star!(residual, m, yr, xr)
                end

                best, result = PSF.fit_star(
                    m, residual, inds;
                    fixed, inv_var, kws...,
                )

                # Update all parameter rows from the fitted model.
                best_props = ConstructionBase.getproperties(best)
                for k in 1:n_params
                    params[k, idx] = FT(getfield(best_props, prop_names[k]))
                end

                # Extract errors from the covariance matrix.
                if ~isnothing(result.cov)
                    _extract_errors!(errors, result.cov, free_idx, is_fixed, idx)
                end

                converged[idx] = result.converged
                chisq[idx] = result.chisq
                n_iter[idx] += result.iterations

                # Subtract the updated best-fit model.
                PSF.subtract_star!(residual, best, yr, xr)

                # Compute qfit and crowding on the final pass, over exactly the
                # same pixels as the fit.  `residual` is now this star's fit
                # residual (neighbors and own model both removed); render its
                # own model into the stamp buffer and pass stamp-local views.
                if pass == n_passes
                    ms = PSF.render!(model_stamp, best, yr, xr)
                    ivv = inv_var === nothing ? nothing : view(inv_var, yr, xr)
                    gs = spread_kernel === nothing ? nothing :
                        correlate!(view(g_stamp, axes(ms)...), ms, spread_kernel, :zero)
                    _star_diagnostics!(qfit, qfit_expected, qfit_z, crowding,
                        spread_model, spread_model_err, idx, best,
                        view(image, yr, xr), view(residual, yr, xr), ms, gs, ivv, length(free_idx))
                end
            catch e
                # Undo the add-back so the residual stays consistent.
                if pass > 1
                    PSF.subtract_star!(residual, m, yr, xr)
                end
                valid[idx] = false
                n_failed += 1
                if length(failure_msgs) < 5
                    push!(failure_msgs, "star $idx (pass $pass): " * sprint(showerror, e))
                end
                continue
            end
        end

        # Between-pass validation: reject non-finite positions,
        # non-positive / NaN flux, non-converged fits.
        for idx in 1:n_stars
            valid[idx] &= converged[idx] &&
                          isfinite(params[row_y, idx]) &&
                          isfinite(params[row_x, idx]) &&
                          isfinite(params[row_flux, idx]) &&
                          params[row_flux, idx] > 0
        end
    end

    if n_failed > 0
        msg = "$n_failed / $n_stars star fits threw exceptions and were marked invalid"
        if !isempty(failure_msgs)
            msg *= ". Examples: " * join(failure_msgs, "; ")
        end
        @warn msg
    end

    # -------------------------------------------------------------------
    # 6. Assemble result
    # -------------------------------------------------------------------
    y = params[row_y, :]
    x = params[row_x, :]
    flux = params[row_flux, :]
    bkg = params[row_bkg, :]

    y_err = errors[row_y, :]
    x_err = errors[row_x, :]
    flux_err = errors[row_flux, :]
    bkg_err = errors[row_bkg, :]

    return MultiPassPhotResult(
        y, x, y_err, x_err, flux, flux_err, bkg, bkg_err,
        converged, valid, chisq, qfit, qfit_expected, qfit_z, crowding,
        spread_model, spread_model_err, n_iter, Int(n_passes), n_failed, failure_msgs, residual,
    )
end
