# Tiled joint photometry: the common frame is split into tiles (a core plus an overlap ring), each
# fitted independently by `fit_joint_multipass` on cutouts of the images; core sources are merged into
# one catalog and the requested products assembled as tiles finish.  Design: tiling-design.tex.

"Default `tile_size` (frame px) of `fit_joint_tiled` and `plan_tiles`."
const TILE_SIZE = 2000

# Part k of 1:n split into `parts` ranges whose lengths differ by at most 1.
_part(n, parts, k) = (fld((k - 1) * n, parts) + 1):fld(k * n, parts)
# Range `r` grown by `o` on both sides, clipped to 1:n.
_grow(r, o, n) = max(1, first(r) - o):min(n, last(r) + o)
_padded_lengths(n, parts, o) = (length(_grow(_part(n, parts, k), o, n)) for k in 1:parts)

"""
`(N, M)`: the split of an `ny x nx` frame into `N x M` cores whose padded tiles all have area at most
`S²`, with the fewest tiles, then the least total padded area.
"""
function _partition(ny, nx, S, o)
    ny * nx <= S^2 && return (1, 1)
    S > 2o || throw(ArgumentError("tile_size = $S must exceed twice the overlap ($o frame px)"))
    hmax = Dict{Int, Int}(); wmax = Dict{Int, Int}()
    # cores of at most S - 2o per side always fit, which bounds the search
    N0, M0 = cld(ny, S - 2o), cld(nx, S - 2o)
    best = (N0 * M0, typemax(Int), N0, M0)          # (tiles, total padded area, N, M)
    for N in 1:min(ny, best[1])
        N > best[1] && break
        h = get!(() -> maximum(_padded_lengths(ny, N, o)), hmax, N)
        for M in 1:min(nx, fld(best[1], N))
            h * get!(() -> maximum(_padded_lengths(nx, M, o)), wmax, M) <= S^2 || continue
            best = min(best, (N * M, sum(_padded_lengths(ny, N, o)) * sum(_padded_lengths(nx, M, o)), N, M))
            break
        end
    end
    return best[3], best[4]
end

# Frame positions of an image's pixels on a lattice of every `step`th row and column (and the last),
# from `to_sky`.  Tied to the parent's pixels, so every tile sees the same position for a pixel.
function _pixel_lattice(im, frame; step = 32)
    ny, nx = size(im.data)
    ys = unique([1:step:ny; ny]); xs = unique([1:step:nx; nx])
    uv = [world_to_grid(frame, im.to_sky(y, x)...) for y in ys, x in xs]
    return (; ys, xs, u = first.(uv), v = last.(uv))
end

# Lattice interval of `y` in sorted `ys` and the interpolation weight.
function _bracket(ys, y)
    length(ys) == 1 && return 1, 1, 0.0
    i = clamp(searchsortedlast(ys, y), 1, length(ys) - 1)
    return i, i + 1, (y - ys[i]) / (ys[i + 1] - ys[i])
end

# Rows and columns of the image covering frame rectangle `(u1, u2, v1, v2)`: the lattice cells whose
# corners' bounding box meets it, snapped outward to the background box grid `(bh, bw)` (boxes
# from pixel 1, so the cutout's mesh boxes are the image's), or `nothing`.
function _cutout_box(lat, (u1, u2, v1, v2), (bh, bw), (ny, nx))
    r0 = c0 = typemax(Int); r1 = c1 = 0
    nr, nc = length(lat.ys), length(lat.xs)
    for j in 1:max(1, nc - 1), i in 1:max(1, nr - 1)
        i2 = min(i + 1, nr); j2 = min(j + 1, nc)
        us = (lat.u[i, j], lat.u[i2, j], lat.u[i, j2], lat.u[i2, j2]); vs = (lat.v[i, j], lat.v[i2, j], lat.v[i, j2], lat.v[i2, j2])
        (maximum(us) >= u1 && minimum(us) <= u2 && maximum(vs) >= v1 && minimum(vs) <= v2) || continue
        r0 = min(r0, lat.ys[i]); r1 = max(r1, lat.ys[i2]); c0 = min(c0, lat.xs[j]); c1 = max(c1, lat.xs[j2])
    end
    r1 == 0 && return nothing
    return (1 + bh * fld(r0 - 1, bh)):min(ny, bh * cld(r1, bh)), (1 + bw * fld(c0 - 1, bw)):min(nx, bw * cld(c1, bw))
end

"""
The overlap `:auto` gives (frame px): the reach of a core pixel's inputs in native px,
`max(R_cap + R_fit, 3 bkg_box_size) + bkg_box_size` (the model of a bright source reaching a fit
stamp; the background mesh, three boxes; one box of margin), times the largest native-to-frame
scale ratio of the images.
"""
function _auto_overlap(images, frame::TanGrid, o::JointLoopOptions)
    box = maximum(_box_dims(o))
    reach = max(o.fit.R_cap + o.fit.R_fit, 3box) + box
    return ceil(Int, round(reach * maximum(_native_scale(im) for im in images) / frame.scale; digits = 3))
end

_box_dims(o::JointLoopOptions) = (b = _background_settings(o.background).box; b isa Integer ? (b, b) : Tuple(b))

"""
    plan_tiles(images, frame; tile_size = $TILE_SIZE, overlap = :auto, options = JointLoopOptions()) -> NamedTuple

The tiles [`fit_joint_tiled`](@ref) runs, from the images' sizes and `to_sky` only (no pixel is
read): `(; N, M, tile_size, overlap, tiles)`.  The frame's rows are split into `N` and its
columns into `M` near-equal parts, the fewest tiles whose padded tile (core plus `overlap` on
every side that is not a frame edge) has area at most `tile_size²`, then the least total padded
area.  Each entry of `tiles` (column-major in the `N x M` grid; tiles no image reaches are left
out) has
- `core`, `padded`: `(rows, cols)` frame ranges; the cores partition the frame;
- `offset`: the tile frame's position on `frame` minus its own, `(first.(padded) .- 1)`;
- `frame`: the tile's `TanGrid` (same projection and scale, reference pixel shifted by `offset`);
- `cutouts`: per image reaching the padded tile, `(; image, rows, cols)`, its pixels covering
  the tile, snapped outward to the `bkg_box_size` grid.

`overlap = :auto` is the reach of a core pixel's inputs (the background mesh, three boxes, plus
one box: 80 native px with the defaults); a smaller integer (frame px) is accepted with a warning,
for validation studies of the boundary.
"""
plan_tiles(images::AbstractVector{<:JointImage}, frame::TanGrid; kw...) = first(_plan_tiles(images, frame; kw...))

function _plan_tiles(images, frame::TanGrid; tile_size::Integer = TILE_SIZE, overlap = :auto, options::JointLoopOptions = JointLoopOptions())
    isempty(images) && throw(ArgumentError("no images"))
    tile_size > 0 || throw(ArgumentError("tile_size must be positive"))
    oa = _auto_overlap(images, frame, options)
    if overlap === :auto
        o = oa
    elseif overlap isa Integer && overlap >= 0
        o = Int(overlap)
        o < oa && @warn "overlap = $o frame px is below the $oa px a core pixel's inputs can reach (overlap = :auto); sources near tile edges may differ from an untiled run"
    else
        throw(ArgumentError("overlap must be :auto or a non-negative integer, got $(repr(overlap))"))
    end
    N, M = _partition(frame.ny, frame.nx, tile_size, o)
    lats = Vector{Any}(undef, length(images))
    Threads.@threads for e in eachindex(images)
        lats[e] = _pixel_lattice(images[e], frame)
    end
    box = _box_dims(options)
    tiles = []
    for l in 1:M, k in 1:N
        cr = _part(frame.ny, N, k); cc = _part(frame.nx, M, l)
        pr = _grow(cr, o, frame.ny); pc = _grow(cc, o, frame.nx)
        rect = (first(pr) - 0.5, last(pr) + 0.5, first(pc) - 0.5, last(pc) + 0.5)
        cutouts = []
        for (e, im) in enumerate(images)
            b = _cutout_box(lats[e], rect, box, size(im.data))
            b === nothing || push!(cutouts, (; image = e, rows = b[1], cols = b[2]))
        end
        isempty(cutouts) && continue
        offset = (first(pr) - 1, first(pc) - 1)
        tf = TanGrid(frame.ra0, frame.dec0, frame.scale, length(pr), length(pc), frame.cy - offset[1], frame.cx - offset[2])
        push!(tiles, (; core = (cr, cc), padded = (pr, pc), offset, frame = tf, cutouts = [c for c in cutouts]))
    end
    return (; N, M, tile_size = Int(tile_size), overlap = o, tiles = [t for t in tiles]), lats
end

# The tile's cutouts as `JointImage`s and their parent indices: one range read of each parent's
# `weight` and `data` (the lazy-input contract), leaving out cutouts without a positive finite weight.
function _cutouts(images, t)
    ws = [images[c.image].weight[c.rows, c.cols] for c in t.cutouts]
    keep = findall(w -> any(x -> isfinite(x) && x > 0, w), ws)
    cuts = map(keep) do k
        c = t.cutouts[k]; im = images[c.image]
        r0, c0 = first(c.rows) - 1, first(c.cols) - 1
        sky = let f = im.to_sky; (i, j) -> f(i + r0, j + c0); end
        JointImage(im.data[c.rows, c.cols], ws[k], im.psf, sky; im.band, im.flux_scale, psf_offset = im.psf_offset .+ (r0, c0))
    end
    return cuts, [t.cutouts[k].image for k in keep]
end

# Element type and empty value of an assembled product.
_product_type(p, T) = p === :nimg ? UInt8 : p === :coadd_nimg ? Int32 : T
_empty_value(S) = S <: AbstractFloat ? S(NaN) : zero(S)

"""
    fit_joint_tiled(images, frame; tile_size = $TILE_SIZE, overlap = :auto, backend = KernelAbstractions.CPU(),
                    T = Float64, options = JointLoopOptions(), show_trace = false, products = (),
                    coadd_options = (;), on_error = :continue) -> NamedTuple

[`fit_joint_multipass`](@ref) on a frame of any size: the frame is split into tiles
([`plan_tiles`](@ref)), each fitted independently on cutouts of the images, one tile at a time,
and the sources each tile's core owns are merged into one catalog.  Peak memory is set by
`tile_size` (frame px, the side of a square of the largest padded tile's area) rather than by the
frame.  Crowded fields (more than about one source per 25 frame px) need a smaller `tile_size` for
the same memory, since the fit's matrix grows with the source density squared; every tile's
`history` carries `mem_est`, its estimated peak.

Every image overlapping a tile contributes the cutout covering it, read once as
`data[rows, cols]` and `weight[rows, cols]`: no other access is made to the images' arrays, so a
parent may be any `AbstractMatrix` that returns a dense array for a range read (e.g. a wrapper
reading a subset of a file).  Quantities summarizing whole images (the model-radius threshold, the
wing template) are each tile's own.

A source belongs to the tile whose core contains its final position.  Each tile reports its core
grown by `options.min_separation`, and pairs from different tiles closer than that are reduced to
the one deeper inside its tile's core.

Returns `(; catalog, bands, plan, tiles, converged, products, t_total)`:
- `catalog`: `fit_joint_multipass`'s columns, `u, v` on `frame` (the tile's position plus its
  offset), plus `tile` (the index into `tiles`).  Errors, significances, `nimg` and `pass_number` are
  the reporting tile's; a band the tile lacks is `NaN` in its rows.  Rows are in tile order;
- `bands`: `unique(im.band for im in images)`, the untiled order;
- `plan`: the `plan_tiles` result;
- `tiles`: per tile, `status` (`:ok`, `:failed`, or `:empty` when every cutout is masked),
  `exception` and `backtrace` (a failed tile's), `history`, `converged`, `n_passes`, `bands`,
  `images` (the parent indices of its cutouts), `n_reported`, `n_dedup` (its sources removed by the
  cross-tile dedup) and the timings `t_cutout`, `t_setup`, `t_final`, `t_total`;
- `converged`: whether every tile's stopping rule fired;
- `products`: those of `fit_joint_multipass`, assembled as tiles finish: detection-grid products
  on `products.grid` (`oversample(frame, K)`) from each tile's core nodes, per-image products on
  each parent image's grid from each tile's core pixels.  Where no tile reports (off the frame, a
  failed tile, a band a tile lacks) they are `NaN` (`0` in `:nimg` and `:coadd_nimg`).

A tile that throws (other than an interrupt) is recorded in `tiles` and the run continues
(`on_error = :continue`); its core is then empty, and a warning lists the failed tiles at the end.
`on_error = :throw` rethrows.
"""
function fit_joint_tiled(images::AbstractVector{<:JointImage}, frame::TanGrid; tile_size::Integer = TILE_SIZE, overlap = :auto,
        backend = KA.CPU(), T::Type{<:AbstractFloat} = Float64, options::JointLoopOptions = JointLoopOptions(), show_trace::Bool = false,
        products = (), coadd_options::NamedTuple = (;), on_error::Symbol = :continue)
    on_error in (:continue, :throw) || throw(ArgumentError("on_error must be :continue or :throw"))
    o = options
    # each tile would refine its own copy of a detector's PSF, discontinuous at tile edges
    o.psf_refine === nothing || throw(ArgumentError("psf_refine is not supported by fit_joint_tiled; refine with fit_joint_multipass"))
    t_total = @elapsed begin
        # arguments that would fail every tile throw before any runs
        products = _check_run_args(o, products, coadd_options)
        bands = unique(im.band for im in images)
        _check_bands(bands)
        plan, lats = _plan_tiles(images, frame; tile_size, overlap, options)
        K = o.detector.K; grid = oversample(frame, K)
        prods = _assembly(products, images, bands, grid, T)
        cats = []; tiles = []
        for (k, t) in enumerate(plan.tiles)
            entry = (; status = :ok, exception = nothing, backtrace = nothing, history = NamedTuple[], converged = false, n_passes = 0,
                       bands = Symbol[], images = Int[], n_reported = 0, t_cutout = 0.0, t_setup = 0.0, t_final = 0.0)
            R = nothing; idx = Int[]
            tk = @elapsed try
                tc = @elapsed (cuts, idx) = _cutouts(images, t)
                entry = merge(entry, (; images = idx, t_cutout = tc))
                if isempty(cuts)
                    entry = merge(entry, (; status = :empty, converged = true))
                else
                    R = fit_joint_multipass(cuts, t.frame; backend, T, options, show_trace, products, coadd_options)
                end
            catch err
                (err isa InterruptException || on_error === :throw) && rethrow()
                entry = merge(entry, (; status = :failed, exception = err, backtrace = catch_backtrace()))
                # a GPU array is released only when the GC finalizes it, and the failed fit's would
                # otherwise stay on the device into the next tiles
                GC.gc()
            end
            if R !== nothing
                c = _core_rows(R.catalog, t, bands, k, o.min_separation)
                push!(cats, c)
                _assemble!(prods, R.products, t, idx, lats, K)
                entry = merge(entry, (; R.history, R.converged, n_passes = length(R.history), R.bands, n_reported = length(c.u), R.t_setup,
                                        R.t_final))
            end
            push!(tiles, merge(entry, (; t_total = tk)))
        end
        catalog, n_dedup = _merge_catalogs(cats, bands, length(tiles), o.min_separation)
        tiles = [merge(t, (; n_reported = t.n_reported - n_dedup[k], n_dedup = n_dedup[k])) for (k, t) in enumerate(tiles)]
    end
    failed = findall(t -> t.status === :failed, tiles)
    isempty(failed) || @warn "$(length(failed)) of $(length(tiles)) tiles failed ($(join(failed, ", "))); their cores have no sources. See `tiles[k].exception`."
    names = Tuple(p for p in (_IMAGE_PRODUCTS..., _MAP_PRODUCTS..., _COADD_PRODUCTS...) if p in products)
    products = NamedTuple{(names..., :grid)}((map(p -> prods[p], names)..., grid))
    return (; catalog, bands, plan, tiles, converged = all(t -> t.converged, tiles), products, t_total)
end

# The tile's catalog rows its core grown by `ε` contains, in global columns (`bands` order, NaN for a
# band the tile lacks; `u, v` on the full frame), with `tile` and `depth` (distance inside the core).
function _core_rows(c, t, bands, k, ε)
    (cr, cc) = t.core
    u = c.u .+ t.offset[1]; v = c.v .+ t.offset[2]
    depth = min.(u .- (first(cr) - 0.5), last(cr) + 0.5 .- u, v .- (first(cc) - 0.5), last(cc) + 0.5 .- v)
    keep = [(first(cr) - 0.5 - ε <= u[i] < last(cr) + 0.5 + ε) & (first(cc) - 0.5 - ε <= v[i] < last(cc) + 0.5 + ε) for i in eachindex(u)]
    n = count(keep)
    col(name) = hasproperty(c, name) ? getproperty(c, name)[keep] : fill(NaN, n)
    perband(q) = NamedTuple{Tuple(Symbol(q, :_, b) for b in bands)}(Tuple(col(Symbol(q, :_, b)) for b in bands))
    return merge((; u = u[keep], v = v[keep], ra = c.ra[keep], dec = c.dec[keep], u_err = c.u_err[keep], v_err = c.v_err[keep]),
                 perband(:flux), perband(:flux_err), (; significance = c.significance[keep]), perband(:significance),
                 (; nimg = c.nimg[keep], pass_number = c.pass_number[keep], tile = fill(k, n), depth = depth[keep]))
end

# Concatenated tile rows less the cross-tile duplicates, and each tile's count of removed rows.
function _merge_catalogs(cats, bands, ntiles, sep)
    perband(q) = Tuple(Symbol(q, :_, b) for b in bands)
    names = (:u, :v, :ra, :dec, :u_err, :v_err, perband(:flux)..., perband(:flux_err)..., :significance, perband(:significance)..., :nimg,
             :pass_number, :tile)
    cols = NamedTuple{(names..., :depth)}(Tuple(reduce(vcat, (getproperty(c, n) for c in cats);
                                                       init = n === :tile || n === :nimg || n === :pass_number ? Int[] : Float64[])
                                                for n in (names..., :depth)))
    keep = _dedup_keep(cols.u, cols.v, cols.tile, cols.depth, sep)
    n_dedup = [count(i -> !keep[i] && cols.tile[i] == k, eachindex(keep)) for k in 1:ntiles]
    return StructArray(NamedTuple{names}(Tuple(getproperty(cols, n)[keep] for n in names))), n_dedup
end

# Keep mask removing cross-tile pairs closer than `sep`: greedy by depth, so of each pair the source
# deeper inside its own tile's core stays.
function _dedup_keep(u, v, tile, depth, sep)
    keep = trues(length(u))
    sep > 0 || return keep
    cells = Dict{NTuple{2, Int}, Vector{Int}}()
    for i in sortperm(depth; rev = true)
        cu = floor(Int, u[i] / sep); cv = floor(Int, v[i] / sep); hit = false
        for a in -1:1, b in -1:1, j in get(cells, (cu + a, cv + b), Int[])
            tile[j] != tile[i] && (u[j] - u[i])^2 + (v[j] - v[i])^2 < sep^2 && (hit = true; break)
        end
        hit ? (keep[i] = false) : push!(get!(() -> Int[], cells, (cu, cv)), i)
    end
    return keep
end

# The full-size product arrays, filled with their empty value: per image on its pixel grid, on `grid`,
# or per band on `grid`.
function _assembly(products, images, bands, grid, T)
    out = Dict{Symbol, Any}()
    for p in products
        S = _product_type(p, T)
        full() = fill(_empty_value(S), grid.ny, grid.nx)
        out[p] = p in _IMAGE_PRODUCTS ? [fill(_empty_value(S), size(im.data)) for im in images] :
                 p in _MAP_PRODUCTS ? full() : NamedTuple{Tuple(bands)}(Tuple(full() for _ in bands))
    end
    return out
end

# Copy a tile's core into the full-size products: detection-grid nodes through the tile's integer node
# offset, and each cutout's pixels whose frame position (on the parent's lattice) lies in the core.
function _assemble!(prods, tp, t, idx, lats, K)
    (cr, cc) = t.core; (pr, pc) = t.padded
    gr = (K * (first(cr) - 1) + 1):(K * last(cr)); gc = (K * (first(cc) - 1) + 1):(K * last(cc))
    tr = gr .- K * (first(pr) - 1); tcol = gc .- K * (first(pc) - 1)
    rc = [(c.rows, c.cols) for c in t.cutouts if c.image in idx]      # in `idx` order, as `_cutouts` kept them
    owned = nothing
    for (p, full) in prods
        if p in _MAP_PRODUCTS
            full[gr, gc] .= view(getproperty(tp, p), tr, tcol)
        elseif p in _COADD_PRODUCTS
            x = getproperty(tp, p)
            for b in keys(x)
                getproperty(full, b)[gr, gc] .= view(getproperty(x, b), tr, tcol)
            end
        else
            owned === nothing && (owned = [_owned(lats[e], r, cr, cc) for (e, r) in zip(idx, rc)])
            for (i, e) in enumerate(idx)
                dst = view(full[e], rc[i]...); src = getproperty(tp, p)[i]
                dst[owned[i]] .= src[owned[i]]
            end
        end
    end
    return prods
end

# Pixels of a cutout whose frame position, bilinear on the parent's lattice, lies in the core
# (half-open, so in exactly one core).
function _owned(lat, (rows, cols), cr, cc)
    by = [_bracket(lat.ys, y) for y in rows]; bx = [_bracket(lat.xs, x) for x in cols]
    m = falses(length(rows), length(cols))
    for (j, (j1, j2, b)) in enumerate(bx), (i, (i1, i2, a)) in enumerate(by)
        f(q) = (1 - a) * ((1 - b) * q[i1, j1] + b * q[i1, j2]) + a * ((1 - b) * q[i2, j1] + b * q[i2, j2])
        u = f(lat.u); v = f(lat.v)
        m[i, j] = (first(cr) - 0.5 <= u < last(cr) + 0.5) & (first(cc) - 0.5 <= v < last(cc) + 0.5)
    end
    return m
end
