# The mad, mad!, mean_and_std functions are reproduced from StatsBase.jl and are licensed under the MIT License.

Base.@irrational mad_constant 1.4826022185056018 big"""
1.482602218505601860547076529360423431326703202590312896536266275245674447622701
"""

"""
    mad(x; center=median(x), normalize=true)

Compute the median absolute deviation (MAD) of collection `x` around `center`
(by default, around the median).

If `normalize` is set to `true`, the MAD is multiplied by
`1 / quantile(Normal(), 3/4) ≈ 1.4826`, in order to obtain a consistent estimator
of the standard deviation under the assumption that the data is normally distributed.
"""
function mad(x; center=nothing, normalize::Union{Bool, Nothing}=nothing, constant=nothing)
    mad!(Base.copymutable(x); center=center, normalize=normalize, constant=constant)
end
mad(data, ::Colon; kwargs...) = mad(data; kwargs...)
mad(data, dims; kwargs...) = mapslices(x -> mad(x; kwargs...), data; dims)

"""
    StatsBase.mad!(x; center=median!(x), normalize=true)

Compute the median absolute deviation (MAD) of array `x` around `center`
(by default, around the median), overwriting `x` in the process.

If `normalize` is set to `true`, the MAD is multiplied by
`1 / quantile(Normal(), 3/4) ≈ 1.4826`, in order to obtain a consistent estimator
of the standard deviation under the assumption that the data is normally distributed.
"""
function mad!(x::AbstractArray;
              center=median!(x),
              normalize::Union{Bool,Nothing}=true,
              constant=nothing)
    isempty(x) && throw(ArgumentError("mad is not defined for empty arrays"))
    c = center === nothing ? median!(x) : center
    T = promote_type(typeof(c), eltype(x))
    U = eltype(x)
    x2 = U == T ? x : isconcretetype(U) && isconcretetype(T) && sizeof(U) == sizeof(T) ? reinterpret(T, x) : similar(x, T)
    x2 .= abs.(x .- c)
    m = median!(x2)
    if normalize isa Nothing
        Base.depwarn("the `normalize` keyword argument will be false by default in future releases: set it explicitly to silence this deprecation", :mad)
        normalize = true
    end
    if !isa(constant, Nothing)
        Base.depwarn("keyword argument `constant` is deprecated, use `normalize` instead or apply the multiplication directly", :mad)
        m * constant
    elseif normalize
        m * mad_constant
    else
        m
    end
end

# """
#     mean_and_std(x, [dim]; corrected=true) -> (mean, std)

# Return the mean and standard deviation of collection `x`. If `x` is an `AbstractArray`,
# `dim` can be specified as a tuple to compute statistics over these dimensions.
# A weighting vector `w` can be specified to weight the estimates.
# Finally, bias correction is applied to the
# standard deviation calculation if `corrected=true`.
# See [`std`](@ref) documentation for more details.
# """
# function mean_and_std(x; corrected::Bool=true)
#     m = mean(x)
#     s = std(x, mean=m, corrected=corrected)
#     m, s
# end
# function mean_and_std(x::AbstractArray{<:Real}, dims; corrected::Bool=true)
#     m = mean(x, dims=dims)
#     s = std(x, dims=dims, mean=m, corrected=corrected)
#     m, s
# end
# # mean_and_std(x::AbstractArray{<:Real}, ::Colon; corrected::Bool=true) =
# #     mean_and_std(x; corrected=corrected)

# ==============================================================================
# Sigma clipping
# ==============================================================================

"""
    _compact_finite!(data::AbstractArray{T}) where {T <: AbstractFloat} -> Int

Move all finite elements of `data` to the front of its linear storage and
return the count.  Non-finite elements beyond the returned prefix are
left in an unspecified order.
"""
function _compact_finite!(data::AbstractArray{T}) where {T <: AbstractFloat}
    flat = vec(data)
    n = 0
    @inbounds for i in eachindex(flat)
        v = flat[i]
        if isfinite(v)
            n += 1
            flat[n] = v
        end
    end
    return n
end

"""
    sigma_clip!(data, sigma; maxiters=10)
    sigma_clip!(data, sigma_low, sigma_high; maxiters=10)

Iteratively sigma-clip `data` in place and return the number of retained
finite samples.

Rejected samples are removed from the active prefix of `vec(data)`.
"""
function sigma_clip!(
        data::AbstractArray{T}, σ_low::Real, σ_high::Real = σ_low;
        maxiters::Integer = 10
    ) where {T <: AbstractFloat}
    # vec shares storage with data (no copy); mutations through flat
    # affect data in place, and vice versa.
    flat = vec(data)
    n = _compact_finite!(data)
    for _ in 1:maxiters
        n == 0 && break

        # Estimate clipping bounds from the current active finite prefix.
        active = view(flat, 1:n)
        m = median!(active)
        s = std(active; corrected = false)
        s == 0 && break
        lo, hi = m - σ_low * s, m + σ_high * s

        # Compact retained values in place so each iteration reuses storage.
        n_new = 0
        @inbounds for i in 1:n
            x = flat[i]
            if lo ≤ x ≤ hi
                n_new += 1
                flat[n_new] = x
            end
        end
        n_new == n && break
        n = n_new
    end
    return n
end

"""
    sigma_clip(data, sigma; maxiters=10)
    sigma_clip(data, sigma_low, sigma_high; maxiters=10)

Return a mutable floating-point copy of `data` after iterative sigma clipping.
The returned array has the **same shape and total number of elements** as `data`.
Non-finite input values are replaced with `NaN`.

Retained (finite, not clipped) samples are compacted to the front of the
linear storage (`vec(result)[1:n]`) and are **partially sorted** — their
original order is not preserved because `median!` rearranges elements during
clipping.  Elements beyond the active prefix contain displaced values from
the compaction process and should be ignored.

Use [`sigma_clip!`](@ref) when the number of retained samples (`n`) is needed,
or when operating in-place on a pre-allocated float array.
"""
function sigma_clip(
        data::AbstractArray, σ_low::Real, σ_high::Real = σ_low;
        maxiters::Integer = 10
    )
    F = float(eltype(data))
    work = similar(data, F)
    @inbounds for i in eachindex(work, data)
        v = data[i]
        work[i] = isfinite(v) ? F(v) : F(NaN)
    end
    sigma_clip!(work, σ_low, σ_high; maxiters)
    return work
end

# ==============================================================================
# Total error calculation
# ==============================================================================

"""
    calc_total_error(data, bkg_error, effective_gain)

Calculate the total 1-sigma error by combining a background-only error
with the Poisson noise of sources.

The total error in countable units (e.g., electrons) is

```math
\\sigma_{\\mathrm{tot, counts}}
= \\sqrt{g_{\\mathrm{eff}}^2 \\sigma_{\\mathrm{bkg}}^2
    + g_{\\mathrm{eff}} I}
```

where ``\\sigma_{\\mathrm{bkg}}`` is the background-only error,
``I`` is the background-subtracted data, and ``g_{\\mathrm{eff}}`` is the
effective gain (converting data units to counts).

Converting back to data units gives

```math
\\sigma_{\\mathrm{tot}}
= \\sqrt{\\sigma_{\\mathrm{bkg}}^2 + \\frac{I}{g_{\\mathrm{eff}}}}
```

If ``I \\leq 0`` or ``g_{\\mathrm{eff}} \\leq 0``, the source Poisson
noise term is omitted and the total error is ``\\sigma_{\\mathrm{bkg}}``.

This function operates on scalars; broadcast over arrays with
`calc_total_error.(data, bkg_error, effective_gain)`.

# Arguments

- `data::Real`: background-subtracted pixel value.
- `bkg_error::Real`: 1-sigma background-only error for that pixel.
- `effective_gain::Real`: ratio of counts (e.g., electrons) to data units.
  Must be non-negative.

# Returns

- `::Real`: the total 1-sigma error in the same units as ``\\sigma_{\\mathrm{bkg}}``.

# Examples

```jldoctest
julia> calc_total_error(2.0, 1.0, 2.0) ≈ sqrt(1.0^2 + 2.0 / 2.0)
true

julia> calc_total_error(-1.0, 0.5, 2.0) ≈ 0.5
true

julia> calc_total_error(10.0, 2.0, 0.0) ≈ 2.0
true
```
"""
function calc_total_error(data::Real, bkg_error::Real, effective_gain::Real)
    effective_gain >= 0 ||
        throw(ArgumentError("effective_gain must be non-negative, got $effective_gain"))
    if effective_gain > 0 && data > 0
        return sqrt(bkg_error^2 + data / effective_gain)
    else
        return sqrt(bkg_error^2)  # abs(bkg_error), skips source Poisson term
    end
end

# ==============================================================================
# Index utilities
# ==============================================================================

"""
    _clamp_inds(yr::AbstractUnitRange{<:Integer}, xr::AbstractUnitRange{<:Integer}, image::AbstractMatrix) -> (yr, xr)

Clip the row range `yr` and column range `xr` to the axes of `image`,
returning `(yr, xr)` ranges that lie entirely within the image bounds. If
`yr`/`xr` do not overlap the corresponding axis of `image` at all, the
returned range is empty (`isempty` is `true`, with `last < first`).
"""
function _clamp_inds(yr::AbstractUnitRange{<:Integer}, xr::AbstractUnitRange{<:Integer}, image::AbstractMatrix)
    ay, ax = axes(image)
    yr = max(first(ay), first(yr)):min(last(ay), last(yr))
    xr = max(first(ax), first(xr)):min(last(ax), last(xr))
    return yr, xr
end
_clamp_inds(inds::CartesianIndices{2}, image::AbstractMatrix) = _clamp_inds(inds.indices..., image)

# ==============================================================================
# Photometric calibration
# ==============================================================================

"""
    abmag(flux_jy)

AB magnitude of a flux density `flux_jy` given in janskys, using the AB zero
point of 3631 Jy.

Returns `NaN` for non-positive flux rather than throwing, so that a bad
measurement in a large catalog does not abort the conversion.

# Examples
```jldoctest
julia> using CrowdPhot: abmag

julia> abmag(3631.0)
0.0

julia> abmag(36.31)     # 100x fainter is exactly 5 magnitudes
5.0

julia> isnan(abmag(-1.0))
true
```
"""
function abmag(flux_jy::Real)
    FT = float(typeof(flux_jy))
    # Written as log10(3631 / f) rather than -log10(f / 3631): bit-identical
    # across the range, but returns +0.0 instead of -0.0 at the zero point.
    return flux_jy > 0 ? FT(2.5 * log10(3631 / flux_jy)) : FT(NaN)
end

"""
    abmag_err(flux, flux_err)

1-sigma magnitude uncertainty from a flux and its 1-sigma uncertainty,
`2.5 / log(10) * flux_err / flux`.

The result is scale invariant, so `flux` and `flux_err` may be in any units as
long as they share them; there is no need to calibrate to janskys first.
Returns `NaN` for non-positive `flux`, matching [`abmag`](@ref), so the two
columns are `NaN` together.

# Examples
```jldoctest
julia> using CrowdPhot: abmag_err

julia> round(abmag_err(100.0, 10.0); digits = 4)
0.1086
```
"""
function abmag_err(flux::Real, flux_err::Real)
    FT = float(promote_type(typeof(flux), typeof(flux_err)))
    return flux > 0 ? FT(2.5 / log(10) * flux_err / flux) : FT(NaN)
end