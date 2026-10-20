# Device representations of the PSF models the joint fitter supports, with kernel-safe
# evaluation.  Each family has
#   psf_context(p, e, Y, X)                  per (source, image) setup at source position (Y, X)
#   psf_value(p, c, i, j, Y, X)              unit-flux value at detector pixel (i, j)
#   psf_value_grad(p, c, i, j, Y, X)         (value, d/dY, d/dX), derivatives w.r.t. the source position
# and must reproduce `PSF.evaluate` / `PSF.evaluate_fg` of the host model (tested against them).

abstract type DevicePSF end

"`CircularGaussianPRF` per image: `α[e] = 2 sqrt(ln 2) / fwhm`."
struct DevGaussPRF{V} <: DevicePSF
    α::V
end
Adapt.@adapt_structure DevGaussPRF

@inline psf_context(p::DevGaussPRF, e, Y, X) = p.α[e]

@inline function psf_value(::DevGaussPRF, α, i, j, Y::T, X::T) where {T}
    dx = T(j) - X; dy = T(i) - Y
    h = T(0.5)
    Ex = erf(α * (dx + h)) - erf(α * (dx - h))
    Ey = erf(α * (dy + h)) - erf(α * (dy - h))
    return Ex * Ey / 4
end

@inline function psf_value_grad(::DevGaussPRF, α, i, j, Y::T, X::T) where {T}
    dx = T(j) - X; dy = T(i) - Y
    h = T(0.5); c = 2 / sqrt(T(π))
    up = α * (dx + h); um = α * (dx - h); vp = α * (dy + h); vm = α * (dy - h)
    Ex = erf(up) - erf(um); Ey = erf(vp) - erf(vm)
    Gx = c * (exp(-um * um) - exp(-up * up)); Gy = c * (exp(-vm * vm) - exp(-vp * vp))
    return Ex * Ey / 4, Ex * α * Gy / 4, Ey * α * Gx / 4
end

"""
Gridded ePSFs (and `ImagePSF`s as 1x1 grids) of every image.  All node images share one size
and all images share one node-grid shape.

- `nodes`: `(nd1, nd2, nnodes)` node data
- `par`: `(5, nnodes)`: `origin.x, origin.y, oversampling[1], oversampling[2], fill_value`
- `yg`, `xg`: `(ngy, E)`, `(ngx, E)` node grid coordinates; `ig`: `(ngy, ngx, E)` node index
- `off`: `(2, E)` offset added to a source's image position for the node-weight lookup
  (cutouts of a larger detector)
"""
struct DevGridded{A3, M, AI} <: DevicePSF
    nodes::A3
    par::M
    yg::M
    xg::M
    ig::AI
    off::M
    ngy::Int32
    ngx::Int32
end
Adapt.@adapt_structure DevGridded

# Bilinear node weights and their (Y, X) derivatives at PSF-frame position (Y, X), as
# `PSF._grid_corners_dw` (positions outside the grid clamp the weights, zero derivatives).
@inline function psf_context(p::DevGridded, e, Y::T, X::T) where {T}
    Yg = Y + p.off[1, e]; Xg = X + p.off[2, e]
    z = zero(T)
    if p.ngy * p.ngx == 1
        k = p.ig[1, 1, e]
        return (k, k, k, k), (one(T), z, z, z), (z, z, z, z), (z, z, z, z)
    end
    iy = Int32(0)
    for k in Int32(1):p.ngy
        p.yg[k, e] <= Yg && (iy = k)
    end
    ix = Int32(0)
    for k in Int32(1):p.ngx
        p.xg[k, e] <= Xg && (ix = k)
    end
    iy = clamp(iy, Int32(1), p.ngy - Int32(1)); ix = clamp(ix, Int32(1), p.ngx - Int32(1))
    y0 = p.yg[iy, e]; y1 = p.yg[iy + 1, e]; x0 = p.xg[ix, e]; x1 = p.xg[ix + 1, e]
    Yc = clamp(Yg, y0, y1); Xc = clamp(Xg, x0, x1)
    cy = ifelse((Yg > y0) & (Yg < y1), one(T), z)
    cx = ifelse((Xg > x0) & (Xg < x1), one(T), z)
    inv_ = inv((x1 - x0) * (y1 - y0))
    dy1 = y1 - Yc; dy0 = Yc - y0; dx1 = x1 - Xc; dx0 = Xc - x0
    ks = (p.ig[iy, ix, e], p.ig[iy, ix + 1, e], p.ig[iy + 1, ix, e], p.ig[iy + 1, ix + 1, e])
    ws = (dx1 * dy1 * inv_, dx0 * dy1 * inv_, dx1 * dy0 * inv_, dx0 * dy0 * inv_)
    dwy = (-dx1 * cy * inv_, -dx0 * cy * inv_, dx1 * cy * inv_, dx0 * cy * inv_)
    dwx = (-dy1 * cx * inv_, dy1 * cx * inv_, -dy0 * cx * inv_, dy0 * cx * inv_)
    return ks, ws, dwy, dwx
end

# Cubic convolution value and derivative in t (`PSF._cubic4`).
@inline function cubic_conv(a, b, c, d, t)
    c1 = (c - a) / 2
    c4 = c - b - c1
    c2 = 3 * c4 - (d - b) / 2 + c1
    c3 = c4 - c2
    return t * (t * (t * c3 + c2) + c1) + b, t * (3 * t * c3 + 2 * c2) + c1
end

# One node's (value, d/dv, d/du) at detector pixel (i, j) for a source at (Y, X): the
# bicubic of `PSF.bicubic_interpolate` (same clamps, Int32 stencil, out-of-bounds fill).
@inline function node_value_grad(nodes, par, k, i, j, Y::T, X::T) where {T}
    ox = par[1, k]; oy = par[2, k]; sx = par[3, k]; sy = par[4, k]; fv = par[5, k]
    nyd = Int32(size(nodes, 1)); nxd = Int32(size(nodes, 2))
    u = sx * (T(j) - X) + ox
    v = sy * (T(i) - Y) + oy
    inb = isfinite(u) & isfinite(v) & (u >= one(T)) & (u <= T(nxd)) & (v >= one(T)) & (v <= T(nyd))
    uc = ifelse(inb, u, one(T)); vc = ifelse(inb, v, one(T))
    lx = clamp(unsafe_trunc(Int32, floor(uc)), Int32(1), nxd - Int32(1))
    ly = clamp(unsafe_trunc(Int32, floor(vc)), Int32(1), nyd - Int32(1))
    du = uc - T(lx); dv = vc - T(ly)
    x1 = clamp(lx - Int32(1), Int32(1), nxd); x2 = lx; x3 = lx + Int32(1); x4 = clamp(lx + Int32(2), Int32(1), nxd)
    y1 = clamp(ly - Int32(1), Int32(1), nyd); y2 = ly; y3 = ly + Int32(1); y4 = clamp(ly + Int32(2), Int32(1), nyd)
    @inbounds begin
        r1, d1 = cubic_conv(nodes[y1, x1, k], nodes[y1, x2, k], nodes[y1, x3, k], nodes[y1, x4, k], du)
        r2, d2 = cubic_conv(nodes[y2, x1, k], nodes[y2, x2, k], nodes[y2, x3, k], nodes[y2, x4, k], du)
        r3, d3 = cubic_conv(nodes[y3, x1, k], nodes[y3, x2, k], nodes[y3, x3, k], nodes[y3, x4, k], du)
        r4, d4 = cubic_conv(nodes[y4, x1, k], nodes[y4, x2, k], nodes[y4, x3, k], nodes[y4, x4, k], du)
    end
    val, dpdv = cubic_conv(r1, r2, r3, r4, dv)
    dpdu, _ = cubic_conv(d1, d2, d3, d4, dv)
    return ifelse(inb, val, fv), ifelse(inb, dpdv, zero(T)), ifelse(inb, dpdu, zero(T)), sx, sy
end

@inline function psf_value(p::DevGridded, c, i, j, Y::T, X::T) where {T}
    ks, ws, _, _ = c
    s = zero(T)
    for q in 1:4
        iszero(ws[q]) && continue        # as `GriddedPSFModel`'s evaluate: zero-weight corners are skipped
        val, _, _, _, _ = node_value_grad(p.nodes, p.par, ks[q], i, j, Y, X)
        s += ws[q] * val
    end
    return s
end

@inline function psf_value_grad(p::DevGridded, c, i, j, Y::T, X::T) where {T}
    ks, ws, dwy, dwx = c
    s = dY = dX = zero(T)
    for q in 1:4
        iszero(ws[q]) && continue
        val, dv, du, sx, sy = node_value_grad(p.nodes, p.par, ks[q], i, j, Y, X)
        s += ws[q] * val
        dY += dwy[q] * val - ws[q] * sy * dv
        dX += dwx[q] * val - ws[q] * sx * du
    end
    return s, dY, dX
end

# A source's context in column j of `buf` (CTX_N rows of the scene's element type: a GPU render's
# local memory) and back, for `psf_value` (the derivative weights are not kept).
const CTX_N = 8
@inline ctx_store!(buf, j, ::DevGaussPRF, α) = (buf[1, j] = α; nothing)
@inline ctx_load(::DevGaussPRF, buf, j) = buf[1, j]
@inline function ctx_store!(buf, j, ::DevGridded, c)
    ks, ws, _, _ = c
    for q in 1:4
        buf[q, j] = ks[q]; buf[q + 4, j] = ws[q]          # node indices are exact in Float32 below 2^24
    end
    return nothing
end
@inline function ctx_load(::DevGridded, buf, j)
    z = zero(eltype(buf))
    ks = (unsafe_trunc(Int32, buf[1, j]), unsafe_trunc(Int32, buf[2, j]), unsafe_trunc(Int32, buf[3, j]), unsafe_trunc(Int32, buf[4, j]))
    return ks, (buf[5, j], buf[6, j], buf[7, j], buf[8, j]), (z, z, z, z), (z, z, z, z)
end

_as_gridded(p::ImagePSF) = GriddedPSFModel([p], [zero(eltype(p))], [zero(eltype(p))])
_as_gridded(p::GriddedPSFModel{<:Any, <:ImagePSF}) = p

"Unit-flux, zero-background copy of `psf` centered at PSF-frame `(y, x)`."
unit_psf(psf, y, x) = CrowdPhot.ConstructionBase.setproperties(psf, (; y = Float64(y), x = Float64(x), flux = 1.0, bkg = 0.0))

# Structural equality of PSF models: same type and recursively equal fields (arrays elementwise;
# isequal so NaN fill values match).  Base.hash is no shortcut: it samples large arrays.
function _same_content(a, b)
    typeof(a) === typeof(b) || return false
    a isa AbstractArray{<:Number} && return isequal(a, b)
    a isa AbstractArray && return size(a) == size(b) && all(_same_content(x, y) for (x, y) in zip(a, b))
    (a isa Number || a isa Symbol || a isa AbstractString || isbits(a)) && return isequal(a, b)
    return all(_same_content(getfield(a, i), getfield(b, i)) for i in 1:nfields(a))
end

# Groups of 1:n under `same(r, e)`, tested against each group's first member: the members that
# head a group (`reps`) and every item's group (`slot`).  Exposures of one detector and filter
# share device PSF nodes and detection banks this way.
function _groups(same, n)
    reps = Int[]; slot = zeros(Int, n)
    for e in 1:n
        k = findfirst(r -> same(r, e), reps)
        k === nothing && (push!(reps, e); k = length(reps))
        slot[e] = k
    end
    return reps, slot
end

"""
    device_psf(be, T, psfs, offsets) -> DevicePSF

One device PSF for every image's host model: all `CircularGaussianPRF`, or all `ImagePSF` /
`GriddedPSFModel` of `ImagePSF` nodes.  `offsets[e] = (oy, ox)` is added to image `e`'s pixel
position for the gridded node lookup.  Images whose models have identical content share one copy
of the nodes (`ig` points them at it); node grids and offsets stay per image.
"""
function device_psf(be, ::Type{T}, psfs::AbstractVector, offsets) where {T}
    if all(p -> p isa CircularGaussianPRF, psfs)
        return DevGaussPRF(to_device(be, T[2 * sqrt(log(2)) / p.fwhm for p in psfs]))
    end
    all(p -> p isa ImagePSF || p isa GriddedPSFModel{<:Any, <:ImagePSF}, psfs) ||
        throw(ArgumentError("joint fitting supports CircularGaussianPRF, ImagePSF, or GriddedPSFModel of ImagePSF nodes, " *
                            "one family for all images; got $(unique(typeof.(psfs)))"))
    gs = map(_as_gridded, psfs)
    nd = size(first(gs).psfs[1].data)
    ngy, ngx = length(gs[1].ygrid), length(gs[1].xgrid)
    all(g -> size(g.index_grid) == (ngy, ngx), gs) || throw(ArgumentError("every image's PSF grid must have the same shape"))
    all(g -> all(n -> size(n.data) == nd, g.psfs), gs) || throw(ArgumentError("every PSF node must have the same data size"))
    units = [unit_psf(g, 0, 0) for g in gs]
    reps, slot = _groups((r, e) -> _same_content(units[r], units[e]), length(gs))
    base = cumsum([0; [length(gs[r].psfs) for r in reps]])
    nodes = Array{T}(undef, nd..., base[end]); par = Matrix{T}(undef, 5, base[end])
    for (i, r) in enumerate(reps), (j, n) in enumerate(gs[r].psfs)
        k = base[i] + j
        nodes[:, :, k] .= n.data
        par[:, k] .= (n.origin.x, n.origin.y, n.oversampling[1], n.oversampling[2], n.fill_value)
    end
    ig = Array{Int32}(undef, ngy, ngx, length(gs))
    for (e, g) in enumerate(gs)
        ig[:, :, e] .= g.index_grid .+ base[slot[e]]
    end
    yg = reduce(hcat, [T.(g.ygrid) for g in gs]); xg = reduce(hcat, [T.(g.xgrid) for g in gs])
    off = T[o[k] for k in 1:2, o in offsets]
    return DevGridded(to_device(be, nodes), to_device(be, par), to_device(be, yg), to_device(be, xg), to_device(be, ig),
                      to_device(be, off), Int32(ngy), Int32(ngx))
end
