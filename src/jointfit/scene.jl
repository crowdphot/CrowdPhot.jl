"""
    JointImage(data, weight, psf, to_sky; band, gain = 1, psf_offset = (0, 0))

One exposure (or detector cutout) of a joint fit, on the host.

- `data`: the image (background is estimated and subtracted per pass).
- `weight`: inverse variance; `0` masks a pixel.
- `psf`: unit-flux PRF in this image's pixel frame (`CircularGaussianPRF`, `ImagePSF`, or
  `GriddedPSFModel` of `ImagePSF` nodes).
- `to_sky(y, x) -> (ra, dec)`: degrees, for 1-based pixel centers; the only thing needed from
  the WCS (see [`PolyMap`](@ref)).
- `band`: filter identifier; images sharing it share one flux per source.
- `gain`: data units per unit of the band's common flux (exposure time, zero point, ...).
- `psf_offset`: added to pixel positions for the PSF's spatial (node) lookup, for cutouts of a
  larger detector.
"""
struct JointImage{M <: AbstractMatrix, W <: AbstractMatrix, P, F}
    data::M
    weight::W
    psf::P
    to_sky::F
    band::Symbol
    gain::Float64
    psf_offset::NTuple{2, Float64}
end

function JointImage(data::AbstractMatrix, weight::AbstractMatrix, psf, to_sky; band, gain::Real = 1, psf_offset = (0, 0))
    size(data) == size(weight) || throw(DimensionMismatch("`data` and `weight` sizes differ: $(size(data)) vs $(size(weight))"))
    gain > 0 || throw(ArgumentError("`gain` must be positive"))
    return JointImage(data, weight, psf, to_sky, Symbol(band), Float64(gain), Float64.(Tuple(psf_offset)))
end

# Rows of the per-image parameter matrix `ip`.
const IP_U0, IP_V0, IP_INVS, IP_GAIN, IP_UMIN, IP_UMAX, IP_VMIN, IP_VMAX = 1:8
const N_IP = 8

"""
Everything kernels need about the images, as one argument (adapted to device arrays on a
GPU).  Images are concatenated column-major into flat vectors; image `e` occupies
`off[e] + 1 : off[e] + ny[e] nx[e]`.
"""
struct DeviceImages{M, MI, VI, PS}
    ip::M        # (N_IP, E): u0, v0, 1/s, gain, fitted (u, v) bounding box
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
`(u, v, f_1 .. f_B)`, with `(u, v)` on `frame` and `P = 2 + B`.
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
        ip[:, e] .= (m.u0, m.v0, 1 / m.s, images[e].gain, m.bbox...)
    end
    D = maximum(m.degree for m in maps)        # PolyMap may have escalated past `degree` for some images
    cy = reduce(hcat, [_lift_coefs(m.cy, m.degree, D) for m in maps]); cx = reduce(hcat, [_lift_coefs(m.cx, m.degree, D) for m in maps])
    dm = Int32[k == 3 ? offsets[e] : dims[e][k] for k in 1:3, e in 1:E]
    band = Int32[findfirst(==(im.band), bands) for im in images]
    psf = device_psf(backend, T, [im.psf for im in images], [im.psf_offset for im in images])
    dev = DeviceImages(to_device(backend, T, ip), to_device(backend, T, cy), to_device(backend, T, cx), to_device(backend, dm),
                       to_device(backend, band), Int32(D), psf)
    # reduce(vcat, ::Vector) allocates once; over a generator it concatenates pairwise (quadratic in E)
    flat(f) = to_device(backend, T, reduce(vcat, [vec(f(im)) for im in images]))
    footprints = [_footprint(im.to_sky, d, frame) for (im, d) in zip(images, dims)]
    return JointScene(backend, frame, bands, dims, offsets, maps, footprints, flat(im -> im.data), flat(im -> im.weight), dev)
end

# Frame bounding box of the pixel area [0.5, ny + 0.5] x [0.5, nx + 0.5], from its boundary.
function _footprint(to_sky, (ny, nx), frame)
    edge = [[(y, x) for y in range(0.5, ny + 0.5; length = 33) for x in (0.5, nx + 0.5)];
            [(y, x) for x in range(0.5, nx + 0.5; length = 33) for y in (0.5, ny + 0.5)]]
    uv = [world_to_grid(frame, to_sky(p...)...) for p in edge]
    return (minimum(first, uv), maximum(first, uv), minimum(last, uv), maximum(last, uv))
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
