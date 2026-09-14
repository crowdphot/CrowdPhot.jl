"""
    _sigma_clip_bounds(values, σ_low, σ_high; maxiters=10) -> (lo, hi)

Return the lower and upper sigma-clip bounds for `values` after iterative
clipping via [`sigma_clip!`](@ref).  Returns `(NaN, NaN)` if fewer than 4
finite values remain after clipping.
"""
function _sigma_clip_bounds(values, σ_low::Real, σ_high::Real; maxiters::Integer = 10)
    T = float(eltype(values))
    work = T[x for x in values if isfinite(x)]
    n = sigma_clip!(work, σ_low, σ_high; maxiters)
    n < 4 && return (T(NaN), T(NaN))
    active = view(vec(work), 1:n)
    med = median(active)
    s = std(active; corrected = false)
    return (med - σ_low * s, med + σ_high * s)
end

"""
    pick_psf_stars(results; kws...) -> Vector{Int}

Select stars suitable for PSF fitting from the morphological measurements
returned by [`measure_star_shapes`](@ref CrowdPhot.measure_star_shapes).

The selection proceeds in three stages:

1. **Faint-end magnitude clipping**: instrumental magnitudes are computed
   from `aperture.aperture_sum` and the faintest stars (above
   `mag_quantiles[2]`) are excluded.  There is no bright-end clip by
   default -- saturation is detected via the core curvature constraint
   (stage 2), which is independent of brightness and does not discard
   good bright stars that are valuable for measuring the PSF wings.

2. **Hard curvature constraint**: stars with `core.normalized_curvature`
   outside `[curvature_min, curvature_max]` are rejected before sigma
   clipping.  Saturated stars have flat cores with curvature near zero;
   unsaturated PSF cores have positive curvature (e.g. ≈ 1.5 for a
   Gaussian with FWHM = 2 pix).  Cosmic rays produce anomalously high
   curvature.

3. **Sigma clipping by magnitude bin**: stars are partitioned into `nbins`
   magnitude bins via quantiles, and within each bin sequential
   sigma-clipping is applied to `fwhm.y`, `fwhm.x`,
   `ellipticity1_aperture`, `ellipticity2_aperture`, and
   `normalized_curvature`.  Only stars within `σ_low`--`σ_high` standard
   deviations of the clipped median for each parameter are retained.

# Arguments

- `results::AbstractVector{<:NamedTuple}`: output of
  [`measure_star_shapes`](@ref CrowdPhot.measure_star_shapes).

# Keyword Arguments

- `mag_quantiles::NTuple{2,<:Number} = (0.0, 0.95)`: lower and upper
  quantiles for instrumental-magnitude clipping.  The default lower bound
  of `0.0` disables the bright-end clip; saturation is detected via the
  curvature constraint instead.
- `nbins::Int = 5`: number of instrumental-magnitude bins.
- `σ_low::Real = 3.0, σ_high::Real = σ_low`: lower and upper sigma
  thresholds for clipping.
- `maxiters::Integer = 10`: maximum iterations for sigma clipping in each
  magnitude bin.
- `curvature_min::Real = 0.1`: hard lower bound on
  `core.normalized_curvature`.  Values ≤ 0.1 indicate a flat or
  saturated core.
- `curvature_max::Real = 5.0`: hard upper bound on
  `core.normalized_curvature`.  Values ≫ 5 indicate a cosmic ray or
  other sharp spike.

# Returns

A `Vector{Int}` of indices into `results` for the stars that passed all
cuts, sorted by instrumental magnitude so brighter stars appear first.
"""
function pick_psf_stars(results; mag_quantiles::NTuple{2,<:Number}=(0.0, 0.95), nbins::Int=5,
        σ_low::Real = 3.0, σ_high::Real = σ_low, maxiters::Integer = 10,
        curvature_min::Real = 0.1, curvature_max::Real = 5.0)

    # -----------------------------------------------------------------------
    # Stage 1: Instrumental magnitude clipping
    # -----------------------------------------------------------------------

    inst_mags = [r.aperture.aperture_sum > 0 ?
        -2.5 * log10(r.aperture.aperture_sum) : NaN for r in results]

    # Keep only finite magnitudes.
    idxs = collect(eachindex(results))
    valid_mag = isfinite.(inst_mags)
    idxs = idxs[valid_mag]
    inst_mags = inst_mags[idxs]
    isempty(idxs) && return Int[]

    # Clip to magnitude percentile range.
    q = quantile(inst_mags, (mag_quantiles[1], mag_quantiles[2]))
    mag_mask = q[1] .<= inst_mags .<= q[2]
    idxs = idxs[mag_mask]
    inst_mags = inst_mags[mag_mask]
    isempty(idxs) && return Int[]

    # -----------------------------------------------------------------------
    # Stage 2: Hard curvature constraint
    # -----------------------------------------------------------------------

    curvatures = [results[i].core.normalized_curvature for i in idxs]
    curv_fin = isfinite.(curvatures)
    curv_mask = curv_fin .& (curvature_min .<= curvatures .<= curvature_max)
    idxs = idxs[curv_mask]
    inst_mags = inst_mags[curv_mask]
    isempty(idxs) && return Int[]

    # -----------------------------------------------------------------------
    # Stage 3: Sigma clipping by magnitude bin
    # -----------------------------------------------------------------------

    mag_edges = quantile(inst_mags, range(0, 1, length = nbins + 1))

    # Morphological parameters to sigma-clip, in extraction order.
    param_extractors = (
        r -> r.aperture.fwhm.y,
        r -> r.aperture.fwhm.x,
        r -> r.aperture.ellipticity1_aperture,
        r -> r.aperture.ellipticity2_aperture,
        r -> r.core.normalized_curvature,
    )

    clipped = Int[]
    for b in 1:nbins
        lo, hi = mag_edges[b], mag_edges[b + 1]
        bin_mask = if b == nbins
            lo .<= inst_mags .<= hi
        else
            lo .<= inst_mags .< hi
        end
        bin_idxs = idxs[bin_mask]
        length(bin_idxs) < 4 && continue

        # Sequential sigma-clip: each parameter narrows the candidate set.
        current = collect(bin_idxs)
        for extract in param_extractors
            values = Float64[extract(results[i]) for i in current]
            fin = isfinite.(values)
            count(fin) < 4 && (empty!(current); break)

            fin_vals = values[fin]
            lo_bound, hi_bound = _sigma_clip_bounds(fin_vals, σ_low, σ_high; maxiters)
            isnan(lo_bound) && (empty!(current); break)

            current = current[fin][lo_bound .<= fin_vals .<= hi_bound]
            isempty(current) && break
        end
        append!(clipped, current)
    end

    # Sort by instrumental magnitude so brighter stars appear first.
    sort!(clipped; by = i -> -2.5 * log10(results[i].aperture.aperture_sum))
    return clipped
end

"""
    pick_psf_stars(results, Nstars::Int; kws...) -> Vector{Int}

Return the brightest `Nstars` stars from the quality-filtered set produced
by [`pick_psf_stars`](@ref).  `kws...` are forwarded to that method.
"""
function pick_psf_stars(results, Nstars::Int; kws...)
    candidates = pick_psf_stars(results; kws...)
    return candidates[1:min(Nstars, length(candidates))]
end
