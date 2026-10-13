function _as_oversampling(oversampling)
    # Accept scalar oversampling as the same factor along both axes.
    if oversampling isa Integer
        oversampling > 0 || throw(ArgumentError("`oversampling` must be positive"))
        return (Int(oversampling), Int(oversampling))
    elseif oversampling isa Tuple || oversampling isa AbstractVector
        # Accept explicit axis factors for anisotropic sampling.
        length(oversampling) == 2 ||
            throw(ArgumentError("`oversampling` must be an integer or a length-2 tuple/vector of integers"))
        all(x -> x isa Integer, oversampling) ||
            throw(ArgumentError("length-2 `oversampling` values must be integers"))
        sx, sy = Int(oversampling[1]), Int(oversampling[2])
        sx > 0 && sy > 0 || throw(ArgumentError("`oversampling` values must be positive"))
        return (sx, sy)
    else
        throw(ArgumentError("`oversampling` must be an integer or a length-2 tuple/vector of integers"))
    end
end

"""
    ImagePSF(data; y=0, x=0, flux=1, bkg=0, origin=nothing,
             oversampling=1, fill_value=0, normalize=false)
    ImagePSF(data, y, x, flux, bkg; kwargs...)

Empirical image-backed PSF/ePSF model. `data` is a finite 2D sampled PSF image;
`y` and `x` are the source centroid in detector-pixel coordinates, `flux` is a
multiplicative scale, and `bkg` is an additive scalar background. `oversampling`
is the integer sampling factor of `data` relative to detector pixels.

The model evaluates the tabulated PSF with separable 4x4 bicubic interpolation.
Values outside the tabulated image are `flux * fill_value + bkg`.

# Notes
## Centroid and origin
`y` and `x` give the center of the star *in the image being modeled*. `origin` locates the
stellar centroid inside the tabulated PSF image, in 1-based array coordinates.
When evaluating at detector-pixel coordinates `(py, px)` for a star whose putative
center is at `(y, x)`, the tabulated PSF is sampled at

```julia
u = oversampling[1] * (px - x) + origin.x
v = oversampling[2] * (py - y) + origin.y
```

Here `oversampling[1]` is the x (column) oversampling factor and
`oversampling[2]` is the y (row) oversampling factor. `origin.x` and `origin.y`
are the 1-based column and row coordinates of the PSF center within `data`.
The tabulated grid stores `data[row, col]`, i.e. dimension 1 is y (rows) and
dimension 2 is x (columns).

Thus changing `y` and `x` moves the source through the detector image, while
changing `origin` changes which point in `data` is interpreted as the PSF
center. For ordinary fitted sources, `origin` should remain fixed and only
`y`, `x`, `flux`, and optionally `bkg` should vary.

## Oversampling
Oversampling is useful when many stars sample different subpixel phases of the
same effective PSF. Higher oversampling requires more well-centered,
isolated stars; otherwise many oversampled cells will be
weakly constrained and must be filled by interpolation. A few dozen stars can be
enough for modest factors such as 2x in clean simulations, but real undersampled
data generally benefit from hundreds of good stars when available. Very high
oversampling with too few stars can produce a smooth-looking but poorly
constrained model.

## Normalization
If `normalize=true`, the input `data` is rescaled so that `sum(data) ==
prod(oversampling)`. This is the convention used by the evaluator for a
unit-flux ePSF: a fitted `flux` remains on the detector-pixel flux scale. Use
`normalize=true` when constructing an `ImagePSF` from an arbitrary image stamp
or stack whose sum is not already in this convention. Use `normalize=false`
when `data` is already normalized, when preserving the exact tabulated values is
important, or when intentionally using `flux` as a pure amplitude multiplier
rather than a physical total flux.
"""
struct ImagePSF{T, A <: AbstractMatrix{T}} <: AbstractPSFModel{T}
    data::A
    x::T
    y::T
    flux::T
    bkg::T
    origin::NamedTuple{(:y, :x), Tuple{T, T}}
    oversampling::Tuple{Int, Int}
    fill_value::T
end

# `evaluate` gathers through `bicubic_interpolate`, whose branch-free `ifelse`
# guards `@turbo` cannot lower when both loop dimensions are unrolled.
_turbo_safe(::Type{<:ImagePSF}) = false

function ImagePSF(
        data::AbstractMatrix;
        y = 0,
        x = 0,
        flux = 1,
        bkg = 0,
        origin = nothing,
        oversampling = 1,
        fill_value = 0,
        normalize::Bool = false
    )
    # Validate the tabulated PSF before building interpolation metadata.
    ndims(data) == 2 || throw(ArgumentError("`data` must be a 2D matrix"))
    all(≥(4), size(data)) || throw(ArgumentError("each axis of `data` must have at least 4 samples"))
    all(isfinite, data) || throw(ArgumentError("all elements of `data` must be finite"))
    os = _as_oversampling(oversampling)

    # Promote model parameters and grid values to one floating-point type.
    T = promote_type(eltype(data), typeof(x), typeof(y), typeof(flux), typeof(bkg), typeof(fill_value))
    if !isnothing(origin)
        origin isa NamedTuple ||
            throw(ArgumentError("`origin` must be `nothing` or a NamedTuple with `y` and `x` fields"))
        length(propertynames(origin)) == 2 && haskey(origin, :y) && haskey(origin, :x) ||
            throw(ArgumentError("`origin` must include exactly `y` and `x` fields"))
        T = promote_type(T, typeof(origin.y), typeof(origin.x))
    end
    T = float(T)

    # Allocating new matrix here is not *strictly* necessary but it is safe;
    # external modifications to the input `data` after model
    # construction would be surprising and could cause hard-to-debug issues.
    dataT = Matrix{T}(data)
    # Optionally normalize so the oversampled grid sums to the sampling area.
    if normalize
        s = sum(dataT)
        isfinite(s) && s > zero(T) || throw(ArgumentError("cannot normalize PSF data with non-positive sum"))
        dataT .*= T(os[1] * os[2]) / s
    end

    # Default origin is the central grid sample; otherwise use the caller's origin.
    org = if isnothing(origin)
        (y = T((size(dataT, 1) + 1) / 2), x = T((size(dataT, 2) + 1) / 2))
    else
        oy, ox = T(origin.y), T(origin.x)
        isfinite(ox) && isfinite(oy) || throw(ArgumentError("`origin` must be finite"))
        (y = oy, x = ox)
    end

    return ImagePSF{T, typeof(dataT)}(dataT, T(x), T(y), T(flux), T(bkg), org, os, T(fill_value))
end

ImagePSF(data::AbstractMatrix, y, x, flux, bkg; kwargs...) =
    ImagePSF(data; x, y, flux, bkg, kwargs...)

ConstructionBase.getproperties(model::ImagePSF) = (y = model.y, x = model.x, flux = model.flux, bkg = model.bkg)

function ConstructionBase.setproperties(model::ImagePSF{T, S}, patch::NamedTuple) where {T, S}
    # Only fit parameters can change; PSF data stay fixed.
    x = haskey(patch, :x) ? T(patch.x) : model.x
    y = haskey(patch, :y) ? T(patch.y) : model.y
    flux = haskey(patch, :flux) ? T(patch.flux) : model.flux
    bkg = haskey(patch, :bkg) ? T(patch.bkg) : model.bkg
    return ImagePSF{T, S}(
        model.data, x, y, flux, bkg, model.origin,
        model.oversampling, model.fill_value
    )
end

# Accepts the concrete `ImagePSF{T, A}` as well as `ImagePSF{T}`: `convert(::GriddedPSFModel)`
# passes the node type it was asked for, which is concrete when taken from `typeof(model)`.
function Base.convert(to::Type{I}, from::ImagePSF) where {T, I <: ImagePSF{T}}
    from isa I && return from
    dataT = T.(from.data)
    isconcretetype(I) && (dataT = convert(fieldtype(I, :data), dataT))
    return ImagePSF{T, typeof(dataT)}(dataT, T(from.x), T(from.y), T(from.flux), T(from.bkg), map(x -> T(x), from.origin), from.oversampling, T(from.fill_value))
end

theta(model::ImagePSF{T}) where {T} = zero(T)

function extent(model::ImagePSF{T}) where {T}
    # Convert tabulated grid bounds back into detector-pixel coordinates.
    ny, nx = size(model.data)
    sx, sy = model.oversampling
    ox, oy = model.origin.x, model.origin.y
    return (
        (model.y - (oy - one(T)) / sy, model.y + (T(ny) - oy) / sy),
        (model.x - (ox - one(T)) / sx, model.x + (T(nx) - ox) / sx),
    )
end
extent(model::ImagePSF, _) = extent(model) # ignore extra argument (fwhm_factor) if provided.

function effective_area(model::ImagePSF{T}) where {T}
    # Approximate effective area from the normalized discrete ePSF samples.
    denom = sum(abs2, model.data)
    return denom == zero(T) ? T(Inf) : T(model.oversampling[1] * model.oversampling[2]) / denom
end

@inline function _cubic4(f1, f2, f3, f4, t)
    # Build cubic-convolution coefficients from four adjacent samples.
    c1 = (f3 - f1) / 2
    c4 = f3 - f2 - c1
    c2 = 3 * c4 - (f4 - f2) / 2 + c1
    c3 = c4 - c2
    # Evaluate both the interpolated value and its derivative at t.
    c4t = t * c3
    value = t * (t * (c4t + c2) + c1) + f2
    deriv = t * (3 * c4t + 2 * c2) + c1
    return value, deriv
end

@doc raw"""
    bicubic_interpolate(data, y, x; fill_value=0)

Evaluate `data` at 1-based fractional array coordinates `(y, x)`, where `y`
is the row (first array index) and `x` is the column (second array index),
using a separable 4x4 cubic-convolution interpolant. Returns `(value, dfdy, dfdx)`,
where `dfdy` and `dfdx` are derivatives with respect to the row and column
grid coordinates, respectively.

For a coordinate inside the array, let `ly = floor(y)`, `lx = floor(x)`,
`dy = y - ly`, and `dx = x - lx`. The interpolation point is treated as lying
between rows `ly` and `ly + 1`, and between grid columns `lx` and `lx + 1`.
For each of the four neighboring rows `ly-1:ly+2`, the four samples
`data[row, lx-1:lx+2]` define a cubic polynomial in `dx`:

```math
p(t) = a_0 + a_1 t + a_2 t^2 + a_3 t^3,\qquad 0 \le t \le 1.
```

The coefficients are chosen by the cubic-convolution constraints

```math
p(0) = f_2,\quad p(1) = f_3,\quad
p'(0) = \frac{f_3 - f_1}{2},\quad
p'(1) = \frac{f_4 - f_2}{2},
```

where `(f_1, f_2, f_3, f_4)` are the four row samples. This gives one
interpolated value and one x-derivative for each row. The four row values are
then interpolated with the same cubic-convolution rule in `dy` to produce the
final value and `dfdy`. The four row-wise x-derivatives are also interpolated in
`dy` to produce `dfdx`.

Boundary conditions and assumptions:
- `data` must have at least four samples along each axis.
- Coordinates outside the closed array bounds return `(fill_value, 0, 0)`.
- Inside the array, the 4x4 stencil is clamped at the nearest array edge. This
  is equivalent to constant extrapolation of edge samples for the stencil only.
- The interpolant is intended for regularly spaced grid samples. `y` and `x`
  are array coordinates, not detector-pixel coordinates.
"""
@inline function bicubic_interpolate(data::AbstractMatrix, y, x; fill_value = zero(eltype(data)))
    T = promote_type(eltype(data), typeof(y), typeof(x), typeof(fill_value))
    T = float(T)
    ny, nx = size(data)
    nx ≥ 4 && ny ≥ 4 || throw(ArgumentError("`data` must have at least four samples along each axis"))
    yy, xx = T(y), T(x)
    # Branch-free inbounds detection so this stays safe to call with SIMD
    # Means we do extra work if the point is out-of-bounds compared to scalar case / early return
    _inbound = isfinite(xx) & isfinite(yy) & (xx >= one(T)) & (xx <= T(nx)) & (yy >= one(T)) & (yy <= T(ny))
    xxc = ifelse(_inbound, xx, one(T))
    yyc = ifelse(_inbound, yy, one(T))

    # Locate the lower-left grid cell and fractional offset inside it.
    # `Int32` (not `Int`) roughly halves the SIMD gather count under `@turbo`.
    lx = clamp(floor(Int32, xxc), Int32(1), Int32(nx - 1))
    ly = clamp(floor(Int32, yyc), Int32(1), Int32(ny - 1))
    dx = xxc - T(lx)
    dy = yyc - T(ly)

    # Clamp the 4x4 stencil once so edge samples are reused directly below.
    ix1 = clamp(lx - Int32(1), Int32(1), Int32(nx))
    ix2 = lx
    ix3 = lx + Int32(1)
    ix4 = clamp(lx + Int32(2), Int32(1), Int32(nx))
    iy1 = clamp(ly - Int32(1), Int32(1), Int32(ny))
    iy2 = ly
    iy3 = ly + Int32(1)
    iy4 = clamp(ly + Int32(2), Int32(1), Int32(ny))

    # Interpolate each row in x and keep the row-wise x derivatives.
    @inbounds begin
        row1, drow1dx = _cubic4(
            T(data[iy1, ix1]), T(data[iy1, ix2]),
            T(data[iy1, ix3]), T(data[iy1, ix4]), dx
        )
        row2, drow2dx = _cubic4(
            T(data[iy2, ix1]), T(data[iy2, ix2]),
            T(data[iy2, ix3]), T(data[iy2, ix4]), dx
        )
        row3, drow3dx = _cubic4(
            T(data[iy3, ix1]), T(data[iy3, ix2]),
            T(data[iy3, ix3]), T(data[iy3, ix4]), dx
        )
        row4, drow4dx = _cubic4(
            T(data[iy4, ix1]), T(data[iy4, ix2]),
            T(data[iy4, ix3]), T(data[iy4, ix4]), dx
        )
    end

    # Interpolate those row values in y, including both first derivatives.
    value, dfdy = _cubic4(row1, row2, row3, row4, dy)
    dfdx, _ = _cubic4(drow1dx, drow2dx, drow3dx, drow4dx, dy)
    return ifelse(_inbound, value, T(fill_value)), ifelse(_inbound, dfdy, zero(T)), ifelse(_inbound, dfdx, zero(T))
end

function evaluate(model::ImagePSF{T}, py, px) where {T}
    # Map detector-pixel coordinates into the oversampled PSF grid.
    sx, sy = model.oversampling
    u = T(sx) * (T(px) - model.x) + model.origin.x
    v = T(sy) * (T(py) - model.y) + model.origin.y
    # Scale the unit-flux ePSF sample and add scalar background.
    p, _, _ = bicubic_interpolate(model.data, v, u; fill_value = model.fill_value)
    return muladd(model.flux, T(p), model.bkg)
end

function evaluate_fg(model::ImagePSF{T}, py, px) where {T}
    # Evaluate the ePSF and its grid-space derivatives at this detector pixel.
    sx, sy = model.oversampling
    u = T(sx) * (T(px) - model.x) + model.origin.x
    v = T(sy) * (T(py) - model.y) + model.origin.y
    p, dpdv, dpdu = bicubic_interpolate(model.data, v, u; fill_value = model.fill_value)
    # Apply the chain rule for centroid, flux, and background parameters.
    profile = T(p)
    f = muladd(model.flux, profile, model.bkg)
    df_dx = -model.flux * T(sx) * T(dpdu)
    df_dy = -model.flux * T(sy) * T(dpdv)
    df_dflux = profile
    df_dbkg = one(T)
    return f, SA[df_dy, df_dx, df_dflux, df_dbkg]
end

"""
    ImagePSFBuildResult

Metadata returned by `fit_psf(ImagePSF, ...)`.
"""
struct ImagePSFBuildResult{T}
    psf::ImagePSF{T}
    y::Vector{T}
    x::Vector{T}
    flux::Vector{T}
    bkg::Vector{T}
    used::BitVector
    converged::BitVector
    iterations::Int
    costs::Vector{T}
end
