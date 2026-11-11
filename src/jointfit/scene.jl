"""
    JointImage(data, weight, psf, to_sky; band, flux_scale = 1, psf_offset = (0, 0))

One exposure (or detector cutout) of a joint fit, on the host.

- `data`: the image (background is estimated and subtracted per pass).
- `weight`: the full per-pixel inverse variance, which must include the sources' Poisson
  variance as well as the background and detector noise (e.g. `1 / ERR²` from a pipeline error
  array); `0` masks a pixel, as do a non-finite weight or datum.  The fit's χ² and the coadd's outlier test rely on it.
- `psf`: unit-flux PRF in this image's pixel frame (`CircularGaussianPRF`, `ImagePSF`, or
  `GriddedPSFModel` of `ImagePSF` nodes).
- `to_sky(y, x) -> (ra, dec)`: degrees, for 1-based pixel centers; the only thing needed from
  the WCS (see [`PolyMap`](@ref)).
- `band`: filter identifier; images sharing it share one flux per source.
- `flux_scale`: data units per unit of the band's common flux (exposure time, zero point, ...),
  so a source's model in this image is `flux_scale` times its band flux; `1` when the images of a
  band are already in common units.  Not a detector gain: no noise model uses it.
- `psf_offset`: added to pixel positions for the PSF's spatial (node) lookup, for cutouts of a
  larger detector.
"""
struct JointImage{M <: AbstractMatrix, W <: AbstractMatrix, P, F}
    data::M
    weight::W
    psf::P
    to_sky::F
    band::Symbol
    flux_scale::Float64
    psf_offset::NTuple{2, Float64}
end

function JointImage(data::AbstractMatrix, weight::AbstractMatrix, psf, to_sky; band, flux_scale::Real = 1, psf_offset = (0, 0))
    size(data) == size(weight) || throw(DimensionMismatch("`data` and `weight` sizes differ: $(size(data)) vs $(size(weight))"))
    flux_scale > 0 || throw(ArgumentError("`flux_scale` must be positive"))
    return JointImage(data, weight, psf, to_sky, Symbol(band), Float64(flux_scale), Float64.(Tuple(psf_offset)))
end

"""
    ImageFootprint(dims, to_sky; band, name)
    ImageFootprint(im::JointImage, name)

What planning a tiled fit needs of an image, without its pixels or PSF: its size `dims = (ny, nx)`,
`to_sky(y, x) -> (ra, dec)` as a `JointImage`'s, its `band`, and the `name` a plan refers to it by.
"""
struct ImageFootprint{F}
    dims::NTuple{2, Int}
    to_sky::F
    band::Symbol
    name::String
end

function ImageFootprint(dims, to_sky; band, name)
    all(>(0), dims) || throw(ArgumentError("image dims must be positive, got $dims"))
    return ImageFootprint(NTuple{2, Int}(dims), to_sky, Symbol(band), String(name))
end
ImageFootprint(im::JointImage, name) = ImageFootprint(size(im.data), im.to_sky; im.band, name)

# The footprints of `JointImage`s, named by their index.
_footprints(images) = [ImageFootprint(im, string(e)) for (e, im) in enumerate(images)]

# Rows of the per-image parameter matrix `ip`.
const IP_U0, IP_V0, IP_INVS, IP_SCALE, IP_UMIN, IP_UMAX, IP_VMIN, IP_VMAX = 1:8
const N_IP = 8

"""
Everything kernels need about the images, as one argument (adapted to device arrays on a
GPU).  Images are concatenated column-major into flat vectors; image `e` occupies
`off[e] + 1 : off[e] + ny[e] nx[e]`.
"""
struct DeviceImages{M, MI, VI, PS}
    ip::M        # (N_IP, E): u0, v0, 1/s, flux_scale, fitted (u, v) bounding box
    cy::M        # (nmonomials, E) PolyMap coefficients
    cx::M
    dims::MI     # (3, E) Int32: ny, nx, flat offset
    band::VI     # (E,) Int32
    degree::Int32
    psf::PS
end
Adapt.@adapt_structure DeviceImages

"""
    JointScene(images, frame::TanGrid; backend = KernelAbstractions.CPU(), T = Float64,
               degree = 5, max_degree = 9, margin = 40, tol = 1e-4)

The images of a joint fit resident on `backend` with element type `T`: data, weights, one
[`PolyMap`](@ref) per image onto `frame`, device PSFs, and band indices (bands are numbered in
order of first appearance).  Source parameters are a `P x N` matrix on the same backend, rows
`(u, v, f_1 .. f_B)`, with `(u, v)` on `frame` and `P = 2 + B`.  A non-finite datum is stored as 0,
and a pixel without a finite datum and a finite positive weight with weight 0.
"""
struct JointScene{T, B, V <: AbstractVector{T}, D <: DeviceImages}
    backend::B
    frame::TanGrid
    bands::Vector{Symbol}
    dims::Vector{NTuple{2, Int}}
    offsets::Vector{Int}
    maps::Vector{PolyMap}
    footprints::Vector{NTuple{4, Float64}}   # (umin, umax, vmin, vmax) of each image's pixel area on `frame`
    data::V
    weight::V
    dev::D
end

nimages(sc::JointScene) = length(sc.dims)
nbands(sc::JointScene) = length(sc.bands)
nparams(sc::JointScene) = 2 + nbands(sc)
Base.eltype(::JointScene{T}) where {T} = T

function JointScene(images::AbstractVector{<:JointImage}, frame::TanGrid; backend = KA.CPU(), T::Type{<:AbstractFloat} = Float64,
        degree::Integer = 5, max_degree::Integer = 9, margin::Real = 40, tol::Real = 1e-4)
    g = _image_geometry(images, frame, backend, T; degree, max_degree, margin, tol)
    psf = device_psf(backend, T, [im.psf for im in images], [im.psf_offset for im in images])
    dev = DeviceImages(g.dev.ip, g.dev.cy, g.dev.cx, g.dev.dims, g.dev.band, g.dev.degree, psf)
    data = _flat(backend, T, im -> im.data, images); weight = _flat(backend, T, im -> im.weight, images)
    launch!(clean_kernel!, backend, length(data), data, weight)
    return JointScene(backend, frame, g.bands, g.dims, g.offsets, g.maps, g.footprints, data, weight, dev)
end

# The one guard on the fit's inputs, as `pass_weights` in the single-image fitter: a non-finite datum
# becomes 0, and a pixel without a finite datum and a finite positive weight gets weight 0 (after
# conversion to T, so Float32 overflow counts).  The kernels mask on `w > 0` alone, so one NaN datum or
# Inf weight (1 / ERR² with ERR = 0) left in would make the whole scene's cost non-finite and every step rejected.
@kernel function clean_kernel!(data, weight)
    i = @index(Global)
    @inbounds begin
        d = data[i]; w = weight[i]
        okd = isfinite(d)
        data[i] = ifelse(okd, d, zero(d)); weight[i] = ifelse(okd & isfinite(w) & (w > zero(w)), w, zero(w))
    end
end

# `f(x)` of every `x` concatenated into one flat array on `backend`, each copied straight into its
# slice (no host staging copy; a conversion to `T` copies one image at a time)
function _flat(backend, T, f, xs)
    a = KA.allocate(backend, T, sum(x -> length(f(x)), xs; init = 0))
    o = 0
    for x in xs
        v = f(x); n = length(v)
        copyto!(a, o + 1, convert(Array{T}, vec(v)), 1, n)
        o += n
    end
    return a
end

"""
The images' geometry on `frame` as kernels need it, without PSFs (shared by `JointScene` and
`coadd`): bands, dims, flat offsets, `PolyMap`s, frame footprints, and `dev`, a `DeviceImages`
with `psf = nothing`.
"""
function _image_geometry(images, frame::TanGrid, backend, ::Type{T}; degree, max_degree, margin, tol) where {T}
    E = length(images)
    E > 0 || throw(ArgumentError("no images"))
    bands = unique(im.band for im in images)
    dims = [size(im.data) for im in images]
    offsets = cumsum([0; [prod(d) for d in dims]])
    offsets[end] < typemax(Int32) || throw(ArgumentError("more than typemax(Int32) pixels in total"))
    maps = Vector{PolyMap}(undef, E)
    Threads.@threads for e in 1:E
        maps[e] = PolyMap(images[e].to_sky, dims[e], frame; degree, max_degree, margin, tol)
    end
    ip = Matrix{T}(undef, N_IP, E)
    for (e, m) in enumerate(maps)
        ip[:, e] .= (m.u0, m.v0, 1 / m.s, images[e].flux_scale, m.bbox...)
    end
    D = maximum(m.degree for m in maps)        # PolyMap may have escalated past `degree` for some images
    cy = reduce(hcat, [_lift_coefs(m.cy, m.degree, D) for m in maps]); cx = reduce(hcat, [_lift_coefs(m.cx, m.degree, D) for m in maps])
    dm = Int32[k == 3 ? offsets[e] : dims[e][k] for k in 1:3, e in 1:E]
    band = Int32[findfirst(==(im.band), bands) for im in images]
    dev = DeviceImages(to_device(backend, T, ip), to_device(backend, T, cy), to_device(backend, T, cx), to_device(backend, dm),
                       to_device(backend, band), Int32(D), nothing)
    footprints = [_footprint(im.to_sky, d, frame) for (im, d) in zip(images, dims)]
    return (; bands, dims, offsets, maps, footprints, dev)
end

# Points on the boundary of the pixel area [0.5, ny + 0.5] x [0.5, nx + 0.5].
_edge_points((ny, nx)) = [[(y, x) for y in range(0.5, ny + 0.5; length = 33) for x in (0.5, nx + 0.5)];
                          [(y, x) for x in range(0.5, nx + 0.5; length = 33) for y in (0.5, ny + 0.5)]]

# Frame bounding box of the pixel area, from its boundary.
function _footprint(to_sky, dims, frame)
    uv = [world_to_grid(frame, to_sky(p...)...) for p in _edge_points(dims)]
    return (minimum(first, uv), maximum(first, uv), minimum(last, uv), maximum(last, uv))
end

"""
    TanGrid(images; scale = nothing) -> TanGrid

The north-up `TanGrid` just covering the pixel areas of `images` (`JointImage`s, whose `psf` may be
`nothing`, or [`ImageFootprint`](@ref)s): tangent point at the mean direction of their boundaries,
extent from those boundaries.  `scale` (arcsec per pixel) defaults to the median of the images'
native scales at their centers; `oversample` refines the result.
"""
TanGrid(images::AbstractVector{<:JointImage}; scale = nothing) = TanGrid(_footprints(images); scale)

function TanGrid(footprints::AbstractVector{<:ImageFootprint}; scale = nothing)
    isempty(footprints) && throw(ArgumentError("no images"))
    edge = [fp.to_sky(p...) for fp in footprints for p in _edge_points(fp.dims)]
    m = reduce((a, b) -> a .+ b, ((cosd(d) * cosd(r), cosd(d) * sind(r), sind(d)) for (r, d) in edge))
    ra0 = mod(atand(m[2], m[1]), 360); dec0 = atand(m[3], hypot(m[1], m[2]))
    s = scale === nothing ? median(_native_scale(fp) for fp in footprints) : Float64(scale)
    s > 0 || throw(ArgumentError("scale must be positive"))
    uv = [world_to_grid(TanGrid(ra0, dec0, s, 1, 1, 0.0, 0.0), p...) for p in edge]
    umin, umax = extrema(first, uv); vmin, vmax = extrema(last, uv)
    return TanGrid(ra0, dec0, s, max(1, ceil(Int, umax - umin)), max(1, ceil(Int, vmax - vmin)), 0.5 - umin, 0.5 - vmin)
end

# Pixel scale (arcsec, square root of the pixel's area) at the image center.
function _native_scale(fp::ImageFootprint)
    y, x = (fp.dims .+ 1) ./ 2
    g1 = TanGrid(fp.to_sky(y, x)..., 1.0, 1, 1, 0.0, 0.0)   # arcsec offsets about the center
    a = world_to_grid(g1, fp.to_sky(y + 1, x)...); b = world_to_grid(g1, fp.to_sky(y, x + 1)...)
    return sqrt(abs(a[1] * b[2] - a[2] * b[1]))
end

"View of image `e` of a flat per-pixel vector `x` as a matrix."
image_view(sc::JointScene, x::AbstractVector, e) = reshape(view(x, sc.offsets[e] + 1:sc.offsets[e + 1]), sc.dims[e])

"Overwrite image `e` of the scene's data with `img` (e.g. raw minus the pass's background)."
set_data!(sc::JointScene, e, img::AbstractMatrix) = (copyto!(image_view(sc, sc.data, e), img); sc)

# Pixel position and Jacobian of frame position (u, v) in image e.  `ok` is false outside the
# PolyMap's fitted footprint, where the polynomial is not trusted.
Base.@propagate_inbounds function source_pixel(d::DeviceImages, e, u::T, v::T) where {T}
    ip = d.ip
    ok = (u >= ip[IP_UMIN, e]) & (u <= ip[IP_UMAX, e]) & (v >= ip[IP_VMIN, e]) & (v <= ip[IP_VMAX, e])
    is = ip[IP_INVS, e]
    y, x, ya, yb, xa, xb = poly_eval(d.cy, d.cx, e, d.degree, (u - ip[IP_U0, e]) * is, (v - ip[IP_V0, e]) * is)
    return ok, y, x, ya * is, yb * is, xa * is, xb * is
end
