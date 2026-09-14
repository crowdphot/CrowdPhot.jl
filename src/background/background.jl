# Portions of this file are adapted from BackgroundMeshes.jl and
# the background-estimation functionality in astropy/photutils and astropy/astropy,
# which are licensed under a BSD 3-clause license. Full licenses are acknowledged
# our main license file.

module Background

using ..CrowdPhot: mad, mad!
using FillArrays: Fill
import Random
using Statistics: mean, median, median!, std

export AbstractBackgroundEstimator, AbstractBackgroundRMSEstimator
export MeanBackground, MedianBackground, SExtractorBackground,
    MMMBackground, BiweightLocationBackground
export StdRMS, MADStdRMS, BiweightScaleRMS
export Background2D, estimate_background, sigma_clip, sigma_clip!

###############################################################################
# Abstract types

"""
    AbstractBackgroundEstimator

Supertype for all scalar background location estimators.

Concrete subtypes must implement `(::MyEstimator)(data::AbstractArray)`,
returning a scalar background estimate for the pixel values in `data`.
"""
abstract type AbstractBackgroundEstimator end

"""
    AbstractBackgroundRMSEstimator

Supertype for all background RMS (scatter) estimators.

Concrete subtypes must implement `(::MyEstimator)(data::AbstractArray)`,
returning a scalar RMS estimate for the pixel values in `data`.
"""
abstract type AbstractBackgroundRMSEstimator end

###############################################################################
# Internal utilities

# Copy input data into mutable floating-point storage without changing shape.
function _float_copy(data::AbstractArray{T}) where {T}
    F = float(T)
    work = Base.copymutable(data)
    return eltype(work) === F ? work : F.(work)
end

function _compact_finite!(data::AbstractArray{T}) where {T <: AbstractFloat}
    flat = vec(data)
    n = 0
    @inbounds for i in eachindex(flat)
        # Move finite samples into the active prefix while preserving storage.
        v = flat[i]
        if isfinite(v)
            n += 1
            flat[n] = v
        end
    end
    return n
end

# Fused float-copy, non-finite rejection, and mask application in a single pass.
function _prepare_work(image::AbstractArray{<:Real}, ::Nothing)
    F = float(eltype(image))
    work = similar(image, F)
    @inbounds for i in eachindex(work, image)
        v = image[i]
        work[i] = isfinite(v) ? F(v) : F(NaN)
    end
    return work
end

function _prepare_work(image::AbstractArray{<:Real}, mask::AbstractArray{Bool})
    size(image) == size(mask) ||
        throw(DimensionMismatch("image and mask must have the same size"))
    F = float(eltype(image))
    work = similar(image, F)
    mask_work = axes(mask) == axes(image) ? mask : Base.copymutable(mask)
    @inbounds for i in eachindex(work, image, mask_work)
        v = image[i]
        work[i] = (mask_work[i] || !isfinite(v)) ? F(NaN) : F(v)
    end
    return work
end

@inline _active_data(data::AbstractArray, n::Integer) =
    n == length(data) ? data : view(vec(data), 1:n)

###############################################################################
# Sigma clipping

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
    work = _prepare_work(data, nothing)
    sigma_clip!(work, σ_low, σ_high; maxiters)
    return work
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

# Biweight location (Tukey 1977; Beers, Flynn & Gebhardt 1990).
function _biweight_location(data::AbstractArray{T}, c::Real = 6.0) where {T}
    FT = float(T)
    M = median(data)
    S = mad(data; center = M, normalize = false)
    S == 0 && return FT(M)
    inv_cS = inv(FT(c * S))
    sw = zero(FT)
    sdw = zero(FT)
    @inbounds for d in data
        # Accumulate Tukey biweight terms without materializing weights.
        u = (d - M) * inv_cS
        if abs(u) < 1
            w = (1 - u^2)^2
            sw += w
            sdw += (d - M) * w
        end
    end
    sw == 0 && return FT(M)
    return FT(M) + sdw / sw
end

# Biweight scale (square-root of biweight midvariance; Beers et al. 1990).
function _biweight_scale(data::AbstractArray{T}, c::Real = 9.0; center = nothing) where {T}
    FT = float(T)
    n = length(data)
    n == 0 && return zero(FT)
    M = something(center, median(data))
    S = mad(data; center = M, normalize = false)
    S == 0 && return zero(FT)
    inv_cS = inv(FT(c * S))
    num = zero(FT)
    den_sum = zero(FT)
    @inbounds for d in data
        # Accumulate accepted residual terms without boolean or u arrays.
        u = (d - M) * inv_cS
        if abs(u) < 1
            one_minus_u2 = 1 - u^2
            num += (d - M)^2 * one_minus_u2^4
            den_sum += 1 - 5 * u^2
        end
    end
    den = den_sum^2
    den == 0 && return zero(FT)
    return sqrt(n * num / den)
end

###############################################################################
# Location estimators

"""
    MeanBackground()

Estimate the background as the unweighted arithmetic mean of the pixel values.

# Examples
```jldoctest
julia> using CrowdPhot.Background

julia> MeanBackground()(fill(3.0, 10))
3.0
```
"""
struct MeanBackground <: AbstractBackgroundEstimator end
(::MeanBackground)(data::AbstractArray; dims = :) = mean(data; dims)

"""
    MedianBackground()

Estimate the background as the sample median of the pixel values.

# Examples
```jldoctest
julia> using CrowdPhot.Background

julia> MedianBackground()(fill(3.0, 10))
3.0
```
"""
struct MedianBackground <: AbstractBackgroundEstimator end
(::MedianBackground)(data::AbstractArray; dims = :) = median(data; dims)

"""
    SExtractorBackground()

Background estimator modelled on the mode statistic used in SExtractor
(Bertin & Arnouts 1996).

When the pixel distribution is roughly symmetric the background is estimated
as `2.5 × median − 1.5 × mean`.  If the distribution is strongly skewed
(`|mean − median| / σ > 0.3`) the median is returned instead, and when the
standard deviation is zero the mean is returned.  This rule makes the estimator
more resistant to source contamination than the mean while being roughly 30 %
noisier than the clipped mean in clean sky regions.

# Examples
```jldoctest
julia> using CrowdPhot.Background

julia> SExtractorBackground()(fill(5.0, 20))
5.0

julia> SExtractorBackground()([5.0, 5.1, 4.9, 5.0, 5.2, 4.8])
5.0
```
"""
struct SExtractorBackground <: AbstractBackgroundEstimator end

function (::SExtractorBackground)(data::AbstractArray; dims = :)
    function validate_se(background, m, med, s)
        return ifelse(s ≈ 0, m, ifelse(abs(m - med) / s > 0.3, med, background))
    end
    med = median(data; dims)
    m = mean(data; dims)
    s = std(data; mean = m, corrected = false, dims)
    return @. validate_se(2.5 * med - 1.5 * m, m, med, s)
end

"""
    MMMBackground(; median_factor=3, mean_factor=2)

Background estimator based on the MMM algorithm from DAOPHOT
(Stetson 1987).

Returns `median_factor × median − mean_factor × mean`.  With the default
coefficients this equals `3 × median − 2 × mean`.  The estimator assumes that
source contamination introduces a positive skew to the pixel distribution, so
it down-weights the mean relative to the median.

# Examples
```jldoctest
julia> using CrowdPhot.Background

julia> MMMBackground()(fill(7.0, 15))
7.0

julia> MMMBackground(; median_factor=4, mean_factor=3)(fill(7.0, 15))
7.0
```
"""
Base.@kwdef struct MMMBackground{T} <: AbstractBackgroundEstimator
    median_factor::T = 3
    mean_factor::T = 2
    function MMMBackground(median_factor, mean_factor)
        T = promote_type(typeof(median_factor), typeof(mean_factor))
        T = float(T)
        return new{T}(T(median_factor), T(mean_factor))
    end
end

function (alg::MMMBackground)(data; dims = :)
    med = median(data; dims)
    m = mean(data; dims)
    return @. alg.median_factor * med - alg.mean_factor * m
end

"""
    BiweightLocationBackground(; c=6.0)

Estimate the background using the robust biweight location statistic
(Tukey 1977; Beers, Flynn & Gebhardt 1990).

The tuning constant `c` controls the rejection threshold: pixels whose
standardised residual (normalized by the median absolute deviation)
exceeds `c` receive zero weight.  `c = 6` is the conventional default.

# Examples
```jldoctest
julia> using CrowdPhot.Background

julia> BiweightLocationBackground()(fill(4.0, 20))
4.0

julia> BiweightLocationBackground(; c=4.0)(fill(4.0, 20))
4.0
```
"""
Base.@kwdef struct BiweightLocationBackground{T} <: AbstractBackgroundEstimator
    c::T = 6
    function BiweightLocationBackground(c)
        T = float(typeof(c))
        return new{T}(T(c))
    end
end
(alg::BiweightLocationBackground)(data; dims = :) =
    _biweight_stat(_biweight_location, data, dims, alg.c)

###############################################################################
# RMS estimators

"""
    StdRMS()

Estimate the background RMS as the (population) standard deviation.

# Examples
```jldoctest
julia> using CrowdPhot.Background

julia> StdRMS()(fill(1.0, 10))
0.0
```
"""
struct StdRMS <: AbstractBackgroundRMSEstimator end
(::StdRMS)(data::AbstractArray; dims = :) = std(data; corrected = false, dims)

"""
    MADStdRMS()

Estimate the background RMS via the normalized median absolute deviation:

```math
\\hat{\\sigma} \\approx 1.4826 \\times \\mathrm{MAD}
```

This is a robust scale estimate that is less sensitive to outliers
than the standard deviation.

# Examples
```jldoctest
julia> using CrowdPhot.Background

julia> MADStdRMS()(fill(1.0, 10))
0.0
```
"""
struct MADStdRMS <: AbstractBackgroundRMSEstimator end
(::MADStdRMS)(data::AbstractArray; dims = :) = mad(data, dims; normalize = true)

"""
    BiweightScaleRMS(; c=9.0)

Estimate the background RMS as the biweight scale (square-root of the
biweight midvariance; Beers, Flynn & Gebhardt 1990).

The tuning constant `c = 9` is the standard default.

# Examples
```jldoctest
julia> using CrowdPhot.Background

julia> BiweightScaleRMS()(fill(2.0, 10))
0.0
```
"""
Base.@kwdef struct BiweightScaleRMS{T} <: AbstractBackgroundRMSEstimator
    c::T = 9
    function BiweightScaleRMS(c)
        T = float(typeof(c))
        return new{T}(T(c))
    end
end
(alg::BiweightScaleRMS)(data; dims = :) =
    _biweight_stat(_biweight_scale, data, dims, alg.c)

_biweight_stat(f, data, ::Colon, c) = f(_float_copy(data), c)
_biweight_stat(f, data, dims, c) = mapslices(x -> f(_float_copy(x), c), data; dims)

_location_estimate!(::MeanBackground, data::AbstractArray) = mean(data)
_location_estimate!(::MedianBackground, data::AbstractArray) = median!(data)

function _location_estimate!(::SExtractorBackground, data::AbstractArray)
    # Measure moments before median! permutes the working copy.
    m = mean(data)
    s = std(data; mean = m, corrected = false)
    med = median!(data)
    s ≈ 0                  && return m
    abs(m - med) / s > 0.3 && return med
    return 2.5 * med - 1.5 * m
end

function _location_estimate!(alg::MMMBackground, data::AbstractArray)
    # Combine order-independent mean with in-place median on copied pixels.
    m = mean(data)
    med = median!(data)
    return alg.median_factor * med - alg.mean_factor * m
end

_location_estimate!(alg::BiweightLocationBackground, data::AbstractArray) =
    _biweight_location(data, alg.c)
_location_estimate!(estimator, data::AbstractArray) = estimator(data)

_rms_estimate!(::StdRMS, data::AbstractArray, location) = std(data; mean = location, corrected = false)
_rms_estimate!(::MADStdRMS, data::AbstractArray, location) = mad!(data; center = location, normalize = true)
_rms_estimate!(alg::BiweightScaleRMS, data::AbstractArray, location) = _biweight_scale(data, alg.c; center = location)
_rms_estimate!(rms_estimator, data::AbstractArray, location) = rms_estimator(data)

function _estimate_pair!(estimator, rms_estimator, data::AbstractArray, n_valid::Integer)
    active = _active_data(data, n_valid)
    # Estimate location first, then compute RMS around that location.
    bkg = _location_estimate!(estimator, active)
    bkg_rms = _rms_estimate!(rms_estimator, active, bkg)
    return (bkg = bkg, bkg_rms = bkg_rms)
end

###############################################################################
# Scalar estimation

"""
    estimate_background(image;
        estimator = SExtractorBackground(),
        rms_estimator = StdRMS(),
        mask = nothing,
        sigma = 3.0,
        maxiters = 10) -> NamedTuple{(:bkg, :bkg_rms)}

Estimate a global (scalar) background and background RMS for `image`.

Non-finite values are always excluded.  Where `mask` is provided, pixels
at `true` positions are excluded before any other processing.  After
masking, iterative sigma clipping is applied when `sigma` is not `nothing`.

Returns a `NamedTuple` `(bkg = …, bkg_rms = …)`.

# Arguments
- `estimator`: an [`AbstractBackgroundEstimator`](@ref) or any callable
  `f(v::AbstractArray) -> scalar`.
- `rms_estimator`: an [`AbstractBackgroundRMSEstimator`](@ref) or any callable.
- `mask`: `AbstractArray{Bool}` with the same shape as `image`; `true`
  excludes the pixel.  `nothing` (default) uses all finite pixels.
- `sigma`: sigma-clipping threshold (default 3.0).  Pass a scalar for symmetric
  clipping.  For asymmetric clipping, pass a length-2 tuple or vector,
  e.g. `sigma=(2.0, 5.0)`.  Set to `nothing` to disable clipping.
- `maxiters`: maximum number of sigma-clipping iterations.

# Notes
If you have a coverage mask of the type described by [`Background2D`](@ref),
you should simply combine it with your bad-pixel mask of the interior
points as this function only accepts a single `mask` keyword argument
(e.g., `mask .= mask .| coverage_mask`).
For scalar estimation (as this function performs) the distinction between
the coverage mask and the user-provided `mask` is moot -- pixels that are
`true` in either mask are excluded from the estimation, so separating the
two mask components would be redundant.

# Examples
```jldoctest
julia> using CrowdPhot.Background

julia> img = fill(100.0, 64, 64);

julia> r = estimate_background(img)
(bkg = 100.0, bkg_rms = 0.0)

julia> r = estimate_background(img; estimator=MMMBackground(), rms_estimator=MADStdRMS())
(bkg = 100.0, bkg_rms = 0.0)
```
"""
function estimate_background(
        image::AbstractArray{<:Real};
        estimator::Union{AbstractBackgroundEstimator, Function} = SExtractorBackground(),
        rms_estimator::Union{AbstractBackgroundRMSEstimator, Function} = StdRMS(),
        mask::Union{Nothing, AbstractArray{Bool}} = nothing,
        sigma = 3.0,
        maxiters::Integer = 10,
    )
    work = _prepare_work(image, mask)
    n_valid = if isnothing(sigma)
        _compact_finite!(work)
    else
        slo, shi = _to_pair(sigma)
        sigma_clip!(work, slo, shi; maxiters)
    end
    n_valid == 0 && throw(
        ArgumentError(
            isnothing(sigma) ?
                "no finite pixels available for background estimation" :
                "all pixels were removed by sigma clipping"
        )
    )
    return _estimate_pair!(estimator, rms_estimator, work, n_valid)
end

###############################################################################
# Bicubic Catmull-Rom zoom interpolator (no external dependencies)

@inline function _catmull_rom(p0, p1, p2, p3, t)
    return (
        (2 * p1) +
            (-p0 + p2) * t +
            (2 * p0 - 5 * p1 + 4 * p2 - p3) * t^2 +
            (-p0 + 3 * p1 - 3 * p2 + p3) * t^3
    ) / 2
end

# Precompute the four Catmull-Rom basis weights for a given t ∈ [0,1].
# This avoids recomputing the cubic polynomial for every pixel when the
# interpolation parameter is reused across an entire row or column.
@inline function _catmull_rom_weights(t::F) where {F <: AbstractFloat}
    t2 = t * t
    t3 = t2 * t
    w1 = (-t + 2 * t2 - t3) / 2
    w2 = (2 - 5 * t2 + 3 * t3) / 2
    w3 = (t + 4 * t2 - 3 * t3) / 2
    w4 = (-t2 + t3) / 2
    return (w1, w2, w3, w4)
end

# Map output pixel k (1-indexed) in a dimension of length K_out to the
# corresponding mesh coordinate (1-indexed) in a mesh of length K_mesh.
@inline function _zoom_coord(T::Type, k::Int, K_out::Int, K_mesh::Int)
    K_mesh == 1 && return one(T)
    K_out == 1 && return one(T)
    return one(T) + (k - 1) * (K_mesh - 1) / (K_out - 1)
end

@inline _clamp_idx(i::Int, n::Int) = max(1, min(n, i))

"""
    _bicubic_zoom(mesh, H, W, coord_H = H, coord_W = W)

Upsample `mesh` (size `M × N`) to a `H × W` array using bicubic
Catmull-Rom interpolation.  Corner samples are preserved exactly.

`coord_H` and `coord_W` specify the coordinate extent used for scaling
the mesh grid.  When `coord_H > H` (e.g., when the mesh covers a
virtual padded image larger than the true output), the output is
sampled from the sub-region `[1, H]` of the full `[1, coord_H]`
coordinate range.  This is used by `Background2D` when the image
dimensions are not exact multiples of `box_size`.
"""
function _bicubic_zoom(
        mesh::AbstractMatrix{T}, H::Integer, W::Integer,
        coord_H::Integer = H, coord_W::Integer = W
    ) where {T <: Real}
    M, N = size(mesh)
    F = float(T)
    out = similar(mesh, F, H, W)

    # Precompute column-dependent indices and y-direction cubic weights.
    col_jm1 = Vector{Int}(undef, W)
    col_jm2 = Vector{Int}(undef, W)
    col_jm3 = Vector{Int}(undef, W)
    col_jm4 = Vector{Int}(undef, W)
    col_wy1 = Vector{F}(undef, W)
    col_wy2 = Vector{F}(undef, W)
    col_wy3 = Vector{F}(undef, W)
    col_wy4 = Vector{F}(undef, W)
    for j in 1:W
        yf = _zoom_coord(F, j, coord_W, N)
        jm = floor(Int, yf)
        ty = F(yf - jm)
        col_jm1[j] = _clamp_idx(jm - 1, N)
        col_jm2[j] = _clamp_idx(jm, N)
        col_jm3[j] = _clamp_idx(jm + 1, N)
        col_jm4[j] = _clamp_idx(jm + 2, N)
        wy1, wy2, wy3, wy4 = _catmull_rom_weights(ty)
        col_wy1[j] = wy1; col_wy2[j] = wy2
        col_wy3[j] = wy3; col_wy4[j] = wy4
    end

    # Precompute row-dependent indices and x-direction cubic weights.
    row_r1 = Vector{Int}(undef, H)
    row_r2 = Vector{Int}(undef, H)
    row_r3 = Vector{Int}(undef, H)
    row_r4 = Vector{Int}(undef, H)
    row_wx1 = Vector{F}(undef, H)
    row_wx2 = Vector{F}(undef, H)
    row_wx3 = Vector{F}(undef, H)
    row_wx4 = Vector{F}(undef, H)
    for i in 1:H
        xf = _zoom_coord(F, i, coord_H, M)
        im_ = floor(Int, xf)
        tx = F(xf - im_)
        row_r1[i] = _clamp_idx(im_ - 1, M)
        row_r2[i] = _clamp_idx(im_, M)
        row_r3[i] = _clamp_idx(im_ + 1, M)
        row_r4[i] = _clamp_idx(im_ + 2, M)
        wx1, wx2, wx3, wx4 = _catmull_rom_weights(tx)
        row_wx1[i] = wx1; row_wx2[i] = wx2
        row_wx3[i] = wx3; row_wx4[i] = wx4
    end

    # Column-outer, row-inner traversal: writes `out[i, j]` with unit-stride
    # in `i` (contiguous in column-major storage) and reads the small `mesh`
    # from L1 cache regardless of access pattern.
    for j in 1:W
        jm1 = col_jm1[j]; jm2 = col_jm2[j]
        jm3 = col_jm3[j]; jm4 = col_jm4[j]
        wy1 = col_wy1[j]; wy2 = col_wy2[j]
        wy3 = col_wy3[j]; wy4 = col_wy4[j]
        @inbounds for i in 1:H
            r1 = row_r1[i]; r2 = row_r2[i]
            r3 = row_r3[i]; r4 = row_r4[i]
            wx1 = row_wx1[i]; wx2 = row_wx2[i]
            wx3 = row_wx3[i]; wx4 = row_wx4[i]

            # Bicubic interpolation: separable 4×4 kernel.
            # y-direction interpolation via muladd chains.
            q1 = muladd(wy4, mesh[r1, jm4],
                 muladd(wy3, mesh[r1, jm3],
                 muladd(wy2, mesh[r1, jm2],
                        wy1 * mesh[r1, jm1])))
            q2 = muladd(wy4, mesh[r2, jm4],
                 muladd(wy3, mesh[r2, jm3],
                 muladd(wy2, mesh[r2, jm2],
                        wy1 * mesh[r2, jm1])))
            q3 = muladd(wy4, mesh[r3, jm4],
                 muladd(wy3, mesh[r3, jm3],
                 muladd(wy2, mesh[r3, jm2],
                        wy1 * mesh[r3, jm1])))
            q4 = muladd(wy4, mesh[r4, jm4],
                 muladd(wy3, mesh[r4, jm3],
                 muladd(wy2, mesh[r4, jm2],
                        wy1 * mesh[r4, jm1])))
            # x-direction interpolation via muladd chain.
            out[i, j] = muladd(wx4, q4,
                         muladd(wx3, q3,
                         muladd(wx2, q2,
                                wx1 * q1)))
        end
    end
    return out
end

###############################################################################
# 2D median filter (no ImageFiltering dependency)

# In-place 2-D median filter with boundary replication.
function _median_filter2d!(
        dst::AbstractMatrix{T}, src::AbstractMatrix{T},
        fh::Int, fw::Int
    ) where {T <: AbstractFloat}
    M, N = size(src)
    hh, hw = fh ÷ 2, fw ÷ 2
    buf = Vector{T}(undef, fh * fw)
    for i in 1:M, j in 1:N
        k = 0
        for di in -hh:hh, dj in -hw:hw
            k += 1
            buf[k] = src[_clamp_idx(i + di, M), _clamp_idx(j + dj, N)]
        end
        # For odd fh, fw the integer division fh ÷ 2 rounds down so
        # -hh:hh spans exactly fh elements and k == fh * fw here.
        dst[i, j] = median(view(buf, 1:k))
    end
    return dst
end

function _median_filter2d(arr::AbstractMatrix{T}, fh::Int, fw::Int) where {T <: AbstractFloat}
    (fh == 1 && fw == 1) && return arr
    return _median_filter2d!(similar(arr), arr, fh, fw)
end

###############################################################################
# Mesh NaN filling by iterative nearest-neighbour propagation

function _fill_nans!(mesh::AbstractMatrix{T}) where {T <: AbstractFloat}
    any(isnan, mesh) || return mesh
    M, N = size(mesh)
    changed = true
    while changed
        changed = false
        for i in 1:M, j in 1:N
            isnan(mesh[i, j]) || continue
            s, k = zero(T), 0
            for di in -1:1, dj in -1:1
                (di == 0 && dj == 0) && continue
                v = mesh[_clamp_idx(i + di, M), _clamp_idx(j + dj, N)]
                if !isnan(v)
                    s += v; k += 1
                end
            end
            if k > 0
                mesh[i, j] = s / k
                changed = true
            end
        end
    end
    # Fallback: fill any remaining NaNs with the global mean of finite values.
    if any(isnan, mesh)
        s, n = zero(T), 0
        @inbounds for v in mesh
            # Accumulate finite mesh values without a temporary vector.
            if !isnan(v)
                s += v
                n += 1
            end
        end
        gm = iszero(n) ? zero(T) : s / n
        for i in eachindex(mesh)
            isnan(mesh[i]) && (mesh[i] = gm)
        end
    end
    return mesh
end

###############################################################################
# 2D background estimation

# Result of spatially varying background estimation over a mesh grid.
struct Background2D{T <: AbstractFloat, M <: AbstractMatrix{T}}
    background::M
    background_rms::M
    mesh_background::M
    mesh_background_rms::M
    box_size::Tuple{Int, Int}
end

"""
    Background2D(image, box_size;
        estimator = SExtractorBackground(),
        rms_estimator = StdRMS(),
        mask = nothing,
        coverage_mask = nothing,
        fill_value = 0,
        filter_size = (3, 3),
        exclude_percentile = 10.0,
        sigma = 3.0,
        maxiters = 10)::Background2D

Estimate a spatially varying background by tiling `image` into boxes,
estimating the background (and RMS) in each box, optionally median-filtering
the mesh, and upsampling the result back to the original image size via a
bicubic Catmull-Rom interpolator.

`box_size` may be an integer (square boxes) or a `(rows, cols)` tuple and
does not need to be a multiple of the image dimensions; partial edge boxes
are included in the mesh.

A box is excluded from the mesh (and filled by interpolation from neighbours)
when the fraction of valid (unmasked, finite, not sigma-clipped) pixels falls
below `exclude_percentile / 100 * box_npixels`.

Non-finite image values are automatically excluded.  A warning is emitted
when the image contains non-finite values that are not covered by `mask` or
`coverage_mask`.

# Arguments
- `estimator`: background location estimator (default [`SExtractorBackground`](@ref)).
- `rms_estimator`: background RMS estimator (default [`StdRMS`](@ref)).
- `mask`: `AbstractMatrix{Bool}` where `true` marks pixels to exclude
  (contaminated pixels — stars, cosmic rays, bad pixels).  Mesh cells excluded
  due to `mask` are filled by interpolation from neighbouring cells.
- `coverage_mask`: `AbstractMatrix{Bool}` where `true` marks pixels outside
  the data region (e.g., blank areas in a mosaic, corners of a rotated image).
  These pixels are excluded from estimation, and mesh cells with zero
  non-coverage pixels are set to `fill_value` rather than being interpolated.
- `fill_value`: value assigned to out-of-coverage regions in the output maps
  (default `0`, giving `0.0` for floating-point images).
- `filter_size`: window size of the 2-D median filter applied to the mesh
  (must be odd; default `(3, 3)`; use `1` or `(1,1)` to skip filtering).
- `exclude_percentile`: boxes with fewer than this percent of valid pixels
  are excluded and filled by interpolation.
- `sigma`: sigma-clipping threshold (default 3.0).  Pass a scalar for symmetric
  clipping.  For asymmetric clipping, pass a length-2 tuple or vector,
  e.g. `sigma=(2.0, 5.0)`, to preserve a faint extended source while rejecting
  bright stars.  Set to `nothing` to disable clipping.
- `maxiters`: maximum number of sigma-clipping iterations.

# Returns
A `Background2D` struct containing the following fields.
- `background`:  full-resolution background map (same size as the input image).
- `background_rms`: full-resolution background-RMS map.
- `mesh_background`: low-resolution per-box background estimates.
- `mesh_background_rms`: low-resolution per-box background-RMS estimates.
- `box_size`: the `(rows, cols)` box dimensions used.

# Examples
```jldoctest
julia> using CrowdPhot.Background

julia> img = fill(200.0, 128, 128);

julia> b = Background2D(img, 32);

julia> b.background ≈ fill(200.0, 128, 128)
true

julia> all(iszero, b.background_rms)
true
```
"""
function Background2D(
        image::AbstractMatrix{<:Real},
        box_size::Union{Integer, NTuple{2, <:Integer}} = 64;
        estimator::Union{AbstractBackgroundEstimator, Function} = SExtractorBackground(),
        rms_estimator::Union{AbstractBackgroundRMSEstimator, Function} = StdRMS(),
        mask::Union{Nothing, AbstractMatrix{Bool}} = nothing,
        coverage_mask::Union{Nothing, AbstractMatrix{Bool}} = nothing,
        fill_value::Real = 0,
        filter_size::Union{Integer, NTuple{2, <:Integer}} = (3, 3),
        exclude_percentile::Real = 10.0,
        sigma = 3.0,
        maxiters::Integer = 10,
    )
    H, W = size(image)
    bh, bw = _to_pair(Int, box_size)
    fh, fw = _to_pair(Int, filter_size)
    (isone(fh) || isodd(fh)) && (isone(fw) || isodd(fw)) ||
        throw(ArgumentError("filter_size must be odd; got ($fh, $fw)"))
    0 ≤ exclude_percentile ≤ 100 ||
        throw(ArgumentError("exclude_percentile must be in [0, 100]"))

    # Validate and normalise a user-supplied mask so downstream code always sees a
    # standard AbstractMatrix{Bool} with OneTo axes. `nothing` becomes a Fill of
    # `false` (O(1) storage, no copy). Non-standard axes (e.g. OffsetArrays) are
    # converted to a plain Matrix; this is the only path that allocates.
    function _normalize_mask(mask::Union{Nothing, AbstractMatrix{Bool}}, H::Int, W::Int,
                            name::String)
        if isnothing(mask)
            return Fill(false, H, W)
        elseif size(mask) != (H, W)
            throw(DimensionMismatch("$name must be the same size as image"))
        elseif axes(mask) != (Base.OneTo(H), Base.OneTo(W))
            return Matrix(mask)
        else
            return mask
        end
    end

    mask_work = _normalize_mask(mask, H, W, "mask")
    covmask_work = _normalize_mask(coverage_mask, H, W, "coverage_mask")

    # Warn about non-finite values not covered by either mask.
    for i in eachindex(image)
        if !isfinite(image[i]) && !mask_work[i] && !covmask_work[i]
            @warn "image contains non-finite values that are not covered by the mask or coverage_mask; they will be excluded automatically"
            break
        end
    end

    # ceil division: partial edge boxes are included in the mesh and get their
    # out-of-bounds pixels filled with NaN by _copy_box_data!.  pad_H / pad_W
    # are the virtual padded dimensions used only for _bicubic_zoom coordinate
    # scaling; the output is always sized to the original image.
    nmesh_rows = cld(H, bh)
    nmesh_cols = cld(W, bw)
    pad_H = nmesh_rows * bh
    pad_W = nmesh_cols * bw

    T = float(eltype(image))
    mesh_bkg = fill(T(NaN), nmesh_rows, nmesh_cols)
    mesh_rms = fill(T(NaN), nmesh_rows, nmesh_cols)

    min_valid = exclude_percentile / 100.0 * bh * bw
    box = Matrix{T}(undef, bh, bw)

    for mi in 1:nmesh_rows, mj in 1:nmesh_cols
        rows = ((mi - 1) * bh + 1):(mi * bh)
        cols = ((mj - 1) * bw + 1):(mj * bw)
        n_covered = _copy_box_data!(box, image, mask_work, covmask_work, rows, cols)
        n_covered == 0 && continue
        n_valid = if !isnothing(sigma)
            # sigma_clip! compacts finite values internally.
            slo, shi = _to_pair(T, sigma)
            sigma_clip!(box, slo, shi; maxiters)
        else
            count(isfinite, box)
        end
        n_valid < min_valid && continue
        pair = _estimate_pair!(estimator, rms_estimator, box, n_valid)
        mesh_bkg[mi, mj] = pair.bkg
        mesh_rms[mi, mj] = pair.bkg_rms
    end

    all(isnan, mesh_bkg) &&
        throw(
        ArgumentError(
            "all mesh boxes were excluded; try larger box_size, " *
                "smaller exclude_percentile, or looser sigma clipping"
        )
    )

    # Fill excluded (NaN) mesh cells by iterative nearest-neighbour propagation.
    _fill_nans!(mesh_bkg)
    _fill_nans!(mesh_rms)

    # Optional 2-D median filter on the mesh.
    fmesh_bkg = _median_filter2d(mesh_bkg, fh, fw)
    fmesh_rms = _median_filter2d(mesh_rms, fh, fw)

    # Upsample directly to the returned size.  pad_H ≥ H and pad_W ≥ W
    # always (ceil division), so out_H = H, out_W = W.
    full_bkg = _bicubic_zoom(fmesh_bkg, H, W, pad_H, pad_W)
    full_rms = _bicubic_zoom(fmesh_rms, H, W, pad_H, pad_W)

    # Overwrite coverage-masked pixels with the requested fill value.
    # The bicubic upsampling interpolates mesh values across coverage
    # boundaries, so we must enforce fill_value at the pixel level.
    fv = T(fill_value)
    for i in eachindex(full_bkg, full_rms)
        if covmask_work[i]
            full_bkg[i] = fv
            full_rms[i] = fv
        end
    end

    return Background2D{T, typeof(full_bkg)}(full_bkg, full_rms, mesh_bkg, mesh_rms, (bh, bw))
end

###############################################################################
# Internal helpers for Background2D

_to_pair(x) = (x, x)
_to_pair(x::NTuple{2, T}) where {T} = (x[1], x[2])
_to_pair(T::Type, x) = (T(x), T(x))
_to_pair(T::Type, x::NTuple{2, S}) where {S} = (T(x[1]), T(x[2]))
_to_pair(x::AbstractVector) = length(x) == 2 ? (x[1], x[2]) : throw(ArgumentError("expected a length-2 argument; got length $(length(x))"))
_to_pair(T::Type, x::AbstractVector) = length(x) == 2 ? (T(x[1]), T(x[2])) : throw(ArgumentError("expected a length-2 argument; got length $(length(x))"))

function _copy_box_data!(box::AbstractMatrix{T}, image, mask, coverage_mask,
                         rows, cols) where {T}
    H, W = size(image)
    n_covered = 0
    @inbounds for (bj, j) in enumerate(cols), (bi, i) in enumerate(rows)
        if i <= H && j <= W
            v = image[i, j]
            fin = isfinite(v)
            # Count pixels with real data (mask==true is irrelevant for coverage).
            if fin && !coverage_mask[i, j]
                n_covered += 1
            end
            # Exclude from estimation: non-finite, mask==true, or coverage_mask==true.
            box[bi, bj] = ifelse(fin && !mask[i, j] && !coverage_mask[i, j], T(v), T(NaN))
        else
            box[bi, bj] = T(NaN)
        end
    end
    return n_covered
end

end # module Background
