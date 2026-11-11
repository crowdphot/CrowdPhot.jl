# Coaddition onto a TanGrid by drizzle (square kernel) in gather form: every output node maps
# its square into each covering image through the image's PolyMap (a parallelogram from the
# center and Jacobian) and intersects it with the input pixels shrunk by `pixfrac`.  The kernels
# read the base per-pixel arrays (data, weight, background, RMS, model) and derive the values,
# weights and validity of each input pixel themselves, so no derived copies are made; in
# `fit_joint_multipass` those arrays and the PolyMaps are the scene's own.

const _COADD_QUANTITIES = (:data, :background, :model, :residual, :rms, :rms_aperture, :weight, :nimg)
const _COADD_PRODUCTS = map(q -> Symbol(:coadd_, q), _COADD_QUANTITIES)   # their `fit_joint_multipass` product names
# what the `:coadd` product shorthand requests: all but :rms_aperture, which costs extra passes
const _COADD_DEFAULT = filter(!=(:coadd_rms_aperture), _COADD_PRODUCTS)

"""
    coadd(images, grid; quantities = (:data,), background = nothing, rms = nothing, model = nothing,
          weight = :rms, pixfrac = 1, units = :flux, clip = nothing, clip_grow = 1, clip_mu = 0.003,
          clip_min_images = 5, background_options = (;), backend = KernelAbstractions.CPU(), T = Float64) -> NamedTuple

Per-band coadds of `images` (`JointImage`s; `psf` may be `nothing`) on `grid` by drizzle with
the square kernel (Fruchter & Hook 2002).  Returns the requested `quantities`, each a
`NamedTuple` keyed by band (in order of first appearance) of `grid`-sized host matrices in `T`,
and `grid`:
- `:data`, `:background`, `:model`, `:residual` (data - background - model): weighted means of
  the inputs resampled to the grid, with the same weights, so the residual coadd is the data
  coadd minus the background and model coadds;
- `:rms`: the noise of each output pixel, `√(Σ a² w') / Σ a w'`, exact for independent inputs;
- `:rms_aperture`: the noise per pixel that applies to sums over areas much larger than a drop,
  as in aperture photometry on the coadd: the variance of such a sum is the sum of
  `:rms_aperture²` over its pixels.  Drizzle correlates neighboring pixels, so `:rms` understates
  the noise of such sums by the ratio of the two (Fruchter & Hook's `R`, e.g. 2.4 for
  `pixfrac = 1` onto a grid twice as fine as the inputs).  Each drop's contribution to any sum
  containing it, `w' G²` with `G = Σ a / W` over the output pixels it covers (`W = Σ a w'`), is
  shared out among those pixels by overlap, so `R` follows the actual dithers, masks, weights
  and scales rather than Fruchter & Hook's uniform-dither formula (which it reproduces for a
  uniform dither).  Sums over smaller areas fall between the two maps.  It costs two extra
  passes per band (the total weight, then `G` accumulated per input pixel) and an array of the
  largest band's input pixels, so `fit_joint_multipass`'s `:coadd` shorthand leaves it out;
- `:weight`: each pixel's total weight `Σ a w'` (drizzle's output weight, in the inverse-variance
  units of the coadd's values; `0` without data), e.g. for depth maps or for combining coadds;
- `:nimg` (`Int32`): the images contributing to each pixel.
All requested quantities but `:rms_aperture` come from one pass per band over the overlaps.
Pixels without data are `NaN` (`0` in `:nimg`).

Each input pixel is a square drop `pixfrac` times its side; `a` is its overlap with an output
pixel and `A` its area, both in output pixels.  Values are converted to flux per output pixel
(`units = :flux`: the value times the output pixel's area in input pixels, so a source's sum
over the coadd is its flux in the input units) or to flux per arcsec² (`:surface_brightness`).
`w'` is the inverse variance of the converted value: `w / f²`, `f` the output pixel's area in
input pixels and `w` = `1 / rms²` (`weight = :rms`) or the images' own weights (`weight =
:input`).  The images' `weight` arrays must be full inverse variances, including the sources'
Poisson variance (see [`JointImage`](@ref)); for the same reason they make poor averaging weights:
weights that depend on the signal bias bright sources and make the coadd's PSF depend on
brightness, which is why `weight = :rms`, background-only and smooth on the mesh scale, is the
default (as recommended for SWarp, AstroDrizzle's IVM weights and others).  With inputs of one
pixel scale this is drizzle's weighted mean; with mixed scales drizzle weights a drop by the
fraction of it in the output pixel, which favors finer inputs beyond their inverse variance.
Pixels with a non-positive or non-finite weight, or a non-finite value, are left out.

`background` and `rms` (one matrix per image) default to `Background2D` of each image (minus
`model`, if given) with `background_options` (the `JointLoopOptions.background` keywords).
`model` is needed for `:model`; without it `:residual` is data - background.

`clip = k` rejects outliers per pixel and band where at least `clip_min_images` images
contribute: an image is rejected where its resampled residual differs from the band's
inverse-variance-weighted median residual by more than `k` times the standard deviation of the
difference.  That uses the images' known variances, which unlike the scatter of a handful of
values is well determined: those of the images' `weight` arrays (whichever `weight` the coadd
averages with), which include the sources' Poisson variance as required, so the test does not
reject the cores of bright sources; plus `clip_mu` times the square of the image's resampled
model (with no `model`, of data - background clipped at zero), which covers imperfect models of
bright sources.  (Background-only `weight` arrays would leave only `clip_mu` to protect sources.)
The median tolerates outliers in up to half the images; testing residuals rather than data keeps
sources out of the test.  An image rejected at a pixel is also rejected within `clip_grow`
pixels of it.  With fewer images no outlier test is reliable (with two there is no telling which
is the outlier), so typical 2-4 exposure stacks are not clipped at all.  `:rms_aperture` of a
clipped coadd uses the unclipped weights.
"""
function coadd(images::AbstractVector{<:JointImage}, grid::TanGrid; quantities = (:data,), background = nothing, rms = nothing,
        model = nothing, weight::Symbol = :rms, pixfrac::Real = 1.0, units::Symbol = :flux, clip = nothing, clip_grow::Integer = 1,
        clip_mu::Real = 0.003, clip_min_images::Integer = 5, background_options::NamedTuple = (;), backend = KA.CPU(), T::Type{<:AbstractFloat} = Float64,
        degree::Integer = 5, max_degree::Integer = 9, margin::Real = 40, tol::Real = 1e-4)
    bad = setdiff(quantities, _COADD_QUANTITIES)
    isempty(bad) || throw(ArgumentError("unknown coadd quantities $(join(bad, ", ")); available: $(join(_COADD_QUANTITIES, ", "))"))
    model === nothing && :model in quantities && throw(ArgumentError("quantity :model needs `model`"))
    E = length(images)
    for (name, x) in ((:background, background), (:rms, rms), (:model, model))
        x === nothing || (length(x) == E && all(e -> size(x[e]) == size(images[e].data), 1:E)) ||
            throw(DimensionMismatch("`$name` must hold one matrix per image, of that image's size"))
    end
    need_bkg = any(in(quantities), (:background, :residual)) || clip !== nothing
    need_rms = weight === :rms
    if (need_bkg && background === nothing) || (need_rms && rms === nothing)
        bset = _background_settings(background_options)
        est = map(1:E) do e
            im = images[e]
            B2 = CrowdPhot.Background2D(model === nothing ? im.data : im.data .- model[e], bset.box; estimator = bset.estimator,
                                        rms_estimator = bset.rms_estimator, mask = .!(isfinite.(im.data) .& isfinite.(im.weight) .& (im.weight .> 0)),
                                        bset.kws...)
            (B2.background, B2.background_rms)
        end
        background === nothing && (background = first.(est)); rms === nothing && (rms = last.(est))
    end
    g = _image_geometry(images, grid, backend, T; degree, max_degree, margin, tol)
    flat(xs) = xs === nothing ? nothing : _flat(backend, T, identity, xs)
    inp = (; data = _flat(backend, T, im -> im.data, images), weight = _flat(backend, T, im -> im.weight, images),
             bkg = need_bkg ? flat(background) : nothing, rms = need_rms ? flat(rms) : nothing, model = flat(model))
    return _coadd(g.dev, g.footprints, g.bands, grid, 1, inp, quantities; weight, pixfrac, units, clip, clip_grow, clip_mu, clip_min_images, backend, T)
end

"""
The coadd of the images described by `dev` (a `DeviceImages` whose PolyMaps map the frame
`grid` is `K` times finer than; `footprints` on that frame) from flat device arrays in its
layout, `inp = (; data, weight, bkg, rms, model)` (`nothing` for inputs not needed).
"""
function _coadd(dev, footprints, bands, grid::TanGrid, K, inp, quantities; weight::Symbol = :rms, pixfrac::Real = 1.0, units::Symbol = :flux,
        clip = nothing, clip_grow::Integer = 1, clip_mu::Real = 0.003, clip_min_images::Integer = 5, backend, T)
    weight in (:rms, :input) || throw(ArgumentError("weight must be :rms or :input, got $(repr(weight))"))
    units in (:flux, :surface_brightness) || throw(ArgumentError("units must be :flux or :surface_brightness, got $(repr(units))"))
    0 < pixfrac <= 1 || throw(ArgumentError("pixfrac must be in (0, 1], got $pixfrac"))
    clip === nothing || clip > 0 || throw(ArgumentError("clip must be nothing or a positive threshold, got $clip"))
    clip_grow >= 0 || throw(ArgumentError("clip_grow must be non-negative"))
    clip_min_images >= 3 || throw(ArgumentError("clip_min_images must be at least 3"))
    ny, nx = grid.ny, grid.nx; nd = (ny, nx)
    bandh = Array(dev.band)
    tile = 64
    tptr, timg, ntr = _tile_lists(footprints, bandh, grid, K, 1.0, tile)
    tptr = to_device(backend, tptr); timg = to_device(backend, timg)
    vals = Tuple(q for q in (:data, :background, :model, :residual) if q in quantities)
    Kt = T(K); p = T(pixfrac); wm = Val(weight)
    ω = units === :flux ? nothing : (T(grid.cy), T(grid.cx), T((grid.scale / ARCSEC_PER_RAD)^2), T(grid.scale^2))
    gs = is_cpu(backend) ? (64, 1) : (16, 16)
    alloc(want, S) = want ? KA.allocate(backend, S, ny, nx) : nothing
    # :rms_aperture: each band's total weight map, and G per input pixel of the band (offsets within it)
    want_ap = :rms_aperture in quantities
    Wm = alloc(want_ap, T)
    dims = Array(dev.dims)
    goff = zeros(Int, length(bandh)); nb = zeros(Int, length(bands))
    for e in eachindex(bandh)
        goff[e] = nb[bandh[e]]; nb[bandh[e]] += Int(dims[1, e]) * Int(dims[2, e])
    end
    G = want_ap ? KA.allocate(backend, T, maximum(nb)) : nothing
    goff = to_device(backend, goff)
    cl = nothing
    if clip !== nothing
        # rejections per pixel as bit masks over each band's images (by their rank in the band)
        rank = Int32[count(==(bandh[e]), view(bandh, 1:(e - 1))) for e in eachindex(bandh)]
        nw = cld(maximum(count(==(b), bandh) for b in eachindex(bands)), 64)
        cl = (; k2 = T(clip^2), grow = Int(clip_grow), mu = T(clip_mu), nmin = Int(clip_min_images), rank = to_device(backend, rank),
                M = KA.allocate(backend, UInt64, nw, ny, nx))
    end
    per_band = map(eachindex(bands)) do b
        outs = ntuple(_ -> KA.allocate(backend, T, ny, nx), length(vals))
        rm = alloc(:rms in quantities, T); ra = alloc(want_ap, T); wt = alloc(:weight in quantities, T); ni = alloc(:nimg in quantities, Int32)
        if want_ap
            coadd_weight_kernel!(backend, gs)(Wm, dev, inp, Int32(b), Kt, p, tptr, timg, tile, ntr, wm; ndrange = nd)
            fill!(G, zero(T))
            coadd_g_kernel!(backend, gs)(G, dev, inp, Int32(b), Kt, p, tptr, timg, tile, ntr, Wm, goff, wm; ndrange = nd)
        end
        cl === nothing || coadd_reject_kernel!(backend, gs)(cl.M, dev, inp, cl.rank, Int32(b), Kt, p, tptr, timg, tile, ntr, cl.k2, cl.mu,
                                                           cl.nmin, wm, Val(_clip_fast(backend)); ndrange = nd)
        coadd_kernel!(backend, gs)(outs, rm, ra, wt, ni, dev, inp, Int32(b), Kt, p, tptr, timg, tile, ntr, ω, cl, G, goff, wm, Val(vals); ndrange = nd)
        host_out = Dict{Symbol, Any}(zip(vals, map(Array, outs)))
        rm === nothing || (host_out[:rms] = Array(rm)); ra === nothing || (host_out[:rms_aperture] = Array(ra))
        wt === nothing || (host_out[:weight] = Array(wt)); ni === nothing || (host_out[:nimg] = Array(ni))
        host_out
    end
    names = Tuple(q for q in _COADD_QUANTITIES if q in quantities)
    bnames = Tuple(bands)
    return NamedTuple{(names..., :grid)}((map(q -> NamedTuple{bnames}(Tuple(pb[q] for pb in per_band)), names)..., grid))
end

# Tuple `t` with entry `m` replaced (no bounds-check throw, for GPU kernels).
@inline _tset(t::NTuple{8, T}, v::T, m) where {T} = ntuple(i -> i == m ? v : t[i], Val(8))

# Sutherland-Hodgman: polygon (py, px)[1:n] (at most 8 vertices) clipped to the side of the line
# `coordinate = bound` (`on_y`: the y coordinate) where `sign * (coordinate - bound) >= 0`.
@inline function _clip_side(py::NTuple{8, T}, px::NTuple{8, T}, n, on_y::Bool, bound::T, sign::T) where {T}
    qy = py; qx = px; m = 0
    @inbounds for k in 1:n
        k2 = k == n ? 1 : k + 1
        ay = py[k]; ax = px[k]; by = py[k2]; bx = px[k2]
        sa = sign * ((on_y ? ay : ax) - bound); sb = sign * ((on_y ? by : bx) - bound)
        if sa >= 0
            m += 1; qy = _tset(qy, ay, m); qx = _tset(qx, ax, m)
        end
        if (sa >= 0) != (sb >= 0)
            t = sa / (sa - sb)
            m += 1; qy = _tset(qy, ay + t * (by - ay), m); qx = _tset(qx, ax + t * (bx - ax), m)
        end
    end
    return qy, qx, m
end

# Area of the parallelogram centered at (y, x) with edge vectors (ya, xa) and (yb, xb), clipped to
# the rectangle [y0, y1] x [x0, x1].  Computed relative to the center: at absolute pixel
# coordinates the shoelace products (~4000²) leave no Float32 digits for areas of ~0.1.
@inline function _overlap(y::T, x::T, ya::T, xa::T, yb::T, xb::T, y0::T, y1::T, x0::T, x1::T) where {T}
    h = T(0.5); z = zero(T)
    y0 -= y; y1 -= y; x0 -= x; x1 -= x
    py = (-h * ya - h * yb, h * ya - h * yb, h * ya + h * yb, -h * ya + h * yb, z, z, z, z)
    px = (-h * xa - h * xb, h * xa - h * xb, h * xa + h * xb, -h * xa + h * xb, z, z, z, z)
    n = 4
    py, px, n = _clip_side(py, px, n, true, y0, one(T))
    py, px, n = _clip_side(py, px, n, true, y1, -one(T))
    py, px, n = _clip_side(py, px, n, false, x0, one(T))
    py, px, n = _clip_side(py, px, n, false, x1, -one(T))
    s = z
    @inbounds for k in 1:n
        k2 = k == n ? 1 : k + 1
        s += py[k] * px[k2] - py[k2] * px[k]
    end
    return abs(s) / 2
end

@inline _at(::Nothing, k, ::Type{T}) where {T} = zero(T)
@inline _at(a, k, ::Type{T}) where {T} = T(@inbounds a[k])

# Input pixel k: whether it is usable (finite data, background and model; a finite positive input
# weight and, with weight = :rms, a finite positive 1 / rms²), its weight (the coadd's, or with
# CW the input inverse variance, for clipping), and the values Q: :data, :background, :model,
# :residual (data - background - model) and :signal (the model, or data - background clipped
# at zero without one).
@inline function _in_pixel(inp, k, ::Val{WM}, ::Val{Q}, ::Val{CW}) where {WM, Q, CW}
    d = @inbounds inp.data[k]; wi = @inbounds inp.weight[k]
    T = typeof(d)
    b = _at(inp.bkg, k, T); m = _at(inp.model, k, T)
    w = WM === :rms ? inv(_at(inp.rms, k, T)^2) : wi
    ok = isfinite(d) & isfinite(wi) & (wi > zero(T)) & isfinite(w) & (w > zero(T)) & isfinite(b) & isfinite(m)
    vals = map(q -> q === :data ? d : q === :background ? b : q === :model ? m : q === :residual ? d - b - m :
                    inp.model === nothing ? max(d - b, zero(T)) : m, Q)
    return ok, CW ? wi : w, vals
end

# Image e's drops at node (r, c) of a grid K times finer than its PolyMap's frame (node r is frame
# (r - 1/2) / K + 1/2): the node's parallelogram in image pixels (y, x, ya, xa, yb, xb) (per grid
# unit), f = its area in input pixels, the image's rows, flat offset, the drops' half side and
# the range of input pixels whose drops can overlap (empty if the node is off the image's map).
@inline function _node_drops(d, e, r::T, c::T, K::T, p::T) where {T}
    h = T(0.5)
    ok, y, x, ya, yb, xa, xb = source_pixel(d, e, (r - h) / K + h, (c - h) / K + h)
    ya /= K; yb /= K; xa /= K; xb /= K
    f = abs(ya * xb - yb * xa)
    ok &= f > 0
    ny = Int(d.dims[1, e]); nx = Int(d.dims[2, e])
    hp = p / 2
    hy = (abs(ya) + abs(yb)) / 2; hx = (abs(xa) + abs(xb)) / 2
    i0 = ok ? max(1, ceil(Int, y - hy - hp)) : 1; i1 = ok ? min(ny, floor(Int, y + hy + hp)) : 0
    j0 = ok ? max(1, ceil(Int, x - hx - hp)) : 1; j1 = ok ? min(nx, floor(Int, x + hx + hp)) : 0
    return (y, x, ya, xa, yb, xb), f, ny, Int(d.dims[3, e]), hp, i0, i1, j0, j1
end

# One image's drizzle sums at node (r, c): W = Σ a w', S_q = Σ a w' v_q (v = value times f, flux per
# output pixel), V = Σ a² w', with a the overlap and A the drop area in output pixels, f the
# output pixel's area in input pixels and w' = w / f²; and, given the band's G (per input pixel
# of the band, at offset goff[e]), VA = Σ (a / A) w' G²: each drop's contribution to the variance
# of any sum containing all of it, w' G², G = Σ a / W over its output pixels, shared out by overlap.
@inline function _drop_sums(d, e, r::T, c::T, K::T, p::T, inp, wm, q::Val{Q}, cw::Val, G, goff) where {T, Q}
    z = zero(T); S = map(_ -> z, Q)
    gm, f, ny, off, hp, i0, i1, j0, j1 = _node_drops(d, e, r, c, K, p)
    A = p * p / f
    go = G === nothing ? 0 : Int(@inbounds goff[e])
    W = V = VA = z
    for j in j0:j1, i in i0:i1
        kl = (j - 1) * ny + i
        ok, wk, vals = _in_pixel(inp, off + kl, wm, q, cw)
        ok || continue
        ai = _overlap(gm..., T(i) - hp, T(i) + hp, T(j) - hp, T(j) + hp)
        ai > 0 || continue
        a = ai / f
        aw = a * wk / (f * f)
        W += aw; V += a * aw
        # a closure over the reassigned S would box it, which GPU kernels cannot compile
        S = map((s, v) -> s + aw * f * v, S, vals)
        G === nothing || (VA += aw / A * (@inbounds G[go + kl])^2)
    end
    return W, S, V, VA
end

# Images per band at one pixel whose clipping statistics fit the rejection kernel's per-item
# arrays and are sorted; deeper pixels find the median by bisection (~10x slower, unbounded).
# On GPUs the size cost nothing measurable up to 1024; on the CPU the time grows ~10% per 4x (the
# larger stack frame), so it stops at 256 (experiments/coadd/clip_fast_*.jl).
_clip_fast(backend) = is_cpu(backend) ? 256 : 1024

# Image e's resampled residual at node (r, c), with its weight and its variance from the input
# weights plus the model term: (W, x, v), W = 0 if the image has no data there.
@inline function _clip_stat(d, e, r::T, c::T, K::T, p::T, inp, wm, mu::T) where {T}
    We, Se, Ve, _ = _drop_sums(d, e, r, c, K, p, inp, wm, Val((:residual, :signal)), Val(true), nothing, nothing)
    We > 0 || return We, zero(T), zero(T)
    return We, Se[1] / We, Ve / (We * We) + mu * max(Se[2] / We, zero(T))^2
end

@inline _set_bit!(M, rk, r, c) = (@inbounds M[rk >> 6 + 1, r, c] |= UInt64(1) << (rk & 63); nothing)

# Whether image e (rank `rk` in its band) was rejected at a node within `grow` of (r, c).
@inline function _masked(M, rk, r, c, grow)
    wd = rk >> 6 + 1; bit = UInt64(1) << (rk & 63)
    for cc in max(1, c - grow):min(size(M, 3), c + grow), rr in max(1, r - grow):min(size(M, 2), r + grow)
        (@inbounds M[wd, rr, cc]) & bit != 0 && return true
    end
    return false
end

# Per node, the band's total weight W (for :rms_aperture's G).
@kernel function coadd_weight_kernel!(Wt, d, inp, band, K, p, @Const(tptr), @Const(timg), tile, ntr, wm)
    r, c = @index(Global, NTuple)
    T = eltype(Wt)
    t = cld(r, tile) + (cld(c, tile) - 1) * ntr
    W = zero(T)
    @inbounds for k in tptr[t]:(tptr[t + 1] - 1)
        e = timg[k]
        d.band[e] == band || continue
        W += _drop_sums(d, e, T(r), T(c), K, p, inp, wm, Val(()), Val(false), nothing, nothing)[1]
    end
    @inbounds Wt[r, c] = W
end

# Per node, G[i] += a / W for every drop i overlapping it: each input pixel's Σ a / W over the
# output pixels its drop covers, from the same overlaps as W (atomic: nodes share drops).
@kernel function coadd_g_kernel!(G, d, inp, band, K, p, @Const(tptr), @Const(timg), tile, ntr, @Const(Wm), @Const(goff), wm)
    r, c = @index(Global, NTuple)
    T = eltype(Wm)
    t = cld(r, tile) + (cld(c, tile) - 1) * ntr
    W = @inbounds Wm[r, c]
    if W > 0
        @inbounds for k in tptr[t]:(tptr[t + 1] - 1)
            e = timg[k]
            d.band[e] == band || continue
            go = Int(goff[e])
            gm, f, ny, off, hp, i0, i1, j0, j1 = _node_drops(d, e, T(r), T(c), K, p)
            for j in j0:j1, i in i0:i1
                kl = (j - 1) * ny + i
                ok, _, _ = _in_pixel(inp, off + kl, wm, Val(()), Val(false))
                ok || continue
                a = _overlap(gm..., T(i) - hp, T(i) + hp, T(j) - hp, T(j) + hp) / f
                a > 0 && (Atomix.@atomic G[go + kl] += a / W)
            end
        end
    end
end

# Per node, the band's rejections as bits (by rank in the band) in M[:, r, c]: every image whose
# residual x differs from the inverse-variance-weighted median of the band's images by more than
# √k2 (v + (π/2) / Σ 1/v)^(1/2), when at least `nmin` images contribute.  Up to NF images the
# statistics sit in private arrays and are sorted; beyond, the median is found by bisection to
# 0.05 of its standard deviation, recomputing the statistics each step.
@kernel function coadd_reject_kernel!(M, d, inp, @Const(rank), band, K, p, @Const(tptr), @Const(timg), tile, ntr, k2, mu, nmin, wm,
                                      ::Val{NF}) where {NF}
    r, c = @index(Global, NTuple)
    T = typeof(p)
    # per-item scratch: KernelAbstractions' CPU `@private` is one heap MArray of NF x groupsize per
    # workgroup per call (for persistence across @synchronize, unused here); a non-escaping MVector
    # stays on the stack (CPU) or in private memory (GPU)
    xs = MVector{NF, T}(undef); vs = MVector{NF, T}(undef); rks = MVector{NF, Int32}(undef)
    @inbounds for wd in axes(M, 1)
        M[wd, r, c] = zero(UInt64)
    end
    t = cld(r, tile) + (cld(c, tile) - 1) * ntr
    rr = T(r); cc = T(c)
    n = 0; Wsum = zero(T); lo = T(Inf); hi = T(-Inf)
    @inbounds for k in tptr[t]:(tptr[t + 1] - 1)
        e = timg[k]
        d.band[e] == band || continue
        We, x, v = _clip_stat(d, e, rr, cc, K, p, inp, wm, mu)
        We > 0 || continue
        n += 1
        if n <= NF
            xs[n] = x; vs[n] = v; rks[n] = rank[e]
        end
        Wsum += 1 / v; lo = min(lo, x); hi = max(hi, x)
    end
    vm = T(π / 2) / Wsum                       # variance of the weighted median
    # @inbounds: a bounds check's BoundsError would capture the MVectors, so they would escape to the
    # heap, which GPU kernels cannot do
    @inbounds if n >= nmin && n <= NF
        for i in 2:n                          # insertion sort by residual
            xi = xs[i]; vi = vs[i]; ri = rks[i]; j = i - 1
            while j >= 1 && xs[j] > xi
                xs[j + 1] = xs[j]; vs[j + 1] = vs[j]; rks[j + 1] = rks[j]; j -= 1
            end
            xs[j + 1] = xi; vs[j + 1] = vi; rks[j + 1] = ri
        end
        med = xs[n]; acc = zero(T)
        for i in 1:n                          # lower weighted median
            acc += 1 / vs[i]
            if acc >= Wsum / 2
                med = xs[i]; break
            end
        end
        for i in 1:n
            (xs[i] - med)^2 > k2 * (vs[i] + vm) && _set_bit!(M, rks[i], r, c)
        end
    elseif n > NF
        tol = sqrt(vm) / 20
        while hi - lo > tol                   # smallest x with weight at or below it >= Wsum / 2
            mid = (lo + hi) / 2; below = zero(T)
            lo < mid < hi || break            # adjacent floats: tol is below hi's spacing, so the loop would not end
            @inbounds for k in tptr[t]:(tptr[t + 1] - 1)
                e = timg[k]
                d.band[e] == band || continue
                We, x, v = _clip_stat(d, e, rr, cc, K, p, inp, wm, mu)
                (We > 0 && x <= mid) && (below += 1 / v)
            end
            below >= Wsum / 2 ? (hi = mid) : (lo = mid)
        end
        @inbounds for k in tptr[t]:(tptr[t + 1] - 1)
            e = timg[k]
            d.band[e] == band || continue
            We, x, v = _clip_stat(d, e, rr, cc, K, p, inp, wm, mu)
            (We > 0 && (x - hi)^2 > k2 * (v + vm)) && _set_bit!(M, rank[e], r, c)
        end
    end
end

# Per node, every requested quantity of the band from one pass over the overlaps.
@kernel function coadd_kernel!(outs, rms, rmsap, wtout, nimg, d, inp, band, K, p, @Const(tptr), @Const(timg), tile, ntr, ω, cl, G, goff, wm, q)
    r, c = @index(Global, NTuple)
    T = typeof(p)
    t = cld(r, tile) + (cld(c, tile) - 1) * ntr
    W = V = VA = zero(T); S = map(_ -> zero(T), _qnames(q)); n = Int32(0)
    @inbounds for k in tptr[t]:(tptr[t + 1] - 1)
        e = timg[k]
        d.band[e] == band || continue
        We, Se, Ve, VAe = _drop_sums(d, e, T(r), T(c), K, p, inp, wm, q, Val(false), G, goff)
        We > 0 || continue
        cl === nothing || !_masked(cl.M, cl.rank[e], r, c, cl.grow) || continue
        W += We; V += Ve; VA += VAe; n += Int32(1)
        S = map(+, S, Se)
    end
    # flux per output pixel, or per arcsec²: a TAN pixel subtends scale² cos³θ
    u = ω === nothing ? one(T) : (1 + ω[3] * ((r - ω[1])^2 + (c - ω[2])^2))^T(1.5) / ω[4]
    ok = W > 0
    fac = ok ? u / W : T(NaN)                  # assigned once, so the closure below does not box it
    foreach((o, s) -> (@inbounds o[r, c] = s * fac; nothing), outs, S)
    @inbounds rms === nothing || (rms[r, c] = ok ? sqrt(V) / W * u : T(NaN))
    @inbounds rmsap === nothing || (rmsap[r, c] = ok ? sqrt(VA) * u : T(NaN))
    @inbounds wtout === nothing || (wtout[r, c] = W / (u * u))   # the values scale by u, their weights by 1/u²
    @inbounds nimg === nothing || (nimg[r, c] = n)
end
_qnames(::Val{Q}) where {Q} = Q
