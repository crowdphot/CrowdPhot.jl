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
function _pixel_lattice(fp::ImageFootprint, frame; step = 32)
    ny, nx = fp.dims
    ys = unique([1:step:ny; ny]); xs = unique([1:step:nx; nx])
    uv = [world_to_grid(frame, fp.to_sky(y, x)...) for y in ys, x in xs]
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
scale ratio of the footprints `fps`.  `nothing` without `psfs` when `R_cap` is `nothing`: the PSF
stamps' extent is then unknown.
"""
function _auto_overlap(fps, psfs, frame::TanGrid, o::JointLoopOptions)
    psfs === nothing && o.fit.R_cap === nothing && return nothing
    box = maximum(_box_dims(o))
    reach = max(_resolve_R_cap(o.fit.R_cap, psfs, o.fit.R_fit) + o.fit.R_fit, 3box) + box
    return ceil(Int, round(reach * maximum(_native_scale(fp) for fp in fps) / frame.scale; digits = 3))
end

_box_dims(o::JointLoopOptions) = (b = _background_settings(o.background).box; b isa Integer ? (b, b) : Tuple(b))

"""
    plan_tiles(images, frame; cores = nothing, tile_size = $TILE_SIZE, max_memory = Inf, density = nothing,
               overlap = :auto, options = JointLoopOptions(), products = (), coadd_options = (;),
               product_eltype = Float32, backend = KernelAbstractions.CPU(), T = Float64) -> NamedTuple

The tiles of a tiled fit ([`fit_joint_tiled`](@ref), or [`fit_tile`](@ref) per tile and
[`merge_tiles`](@ref) per cell), from the images' sizes and `to_sky` only (no pixel is read): `images`
are `JointImage`s or [`ImageFootprint`](@ref)s.  Returns
`(; N, M, tile_size, overlap, bands, options, products, coadd_options, product_eltype, frame, cells, uncovered, tiles)`.

The *cells* are the regions whose sources are reported, each split into tiles: a tile is a core,
whose sources it reports, padded by `overlap` frame px.
- `cores = nothing`: one cell, the whole frame, its rows split into `N` and its columns into `M`
  near-equal parts, the fewest tiles whose padded tile (core plus `overlap` on every side that is
  not a frame edge) has area at most `tile_size²`, then the least total padded area;
- `cores`, a list of disjoint `(rows, cols)` frame ranges (e.g. skycells): one cell each, padded on
  every side, past the frame's edges too (so `offset` may be negative).  A cell is split into the
  fewest near-equal parts (then the least padded area), none narrower than the overlap, whose
  tiles' `mem_est` are at most `max_memory` bytes (a cap no split can satisfy throws, early once the finest
  split is seen to exceed it); with the default `Inf` every cell is one tile.
  `N`, `M` and `tile_size` are then `nothing`.

`mem_est` is the tile's estimated peak memory on `backend` in `T` (as a pass's `history.mem_est`
counts it): its cutouts' pixels, plus `H` for the sources and pairs of `density` (sources per frame
px², a number or a function of frame `(u, v)` taken at the tile's center, e.g. from pre-pass
catalogs).  A finite `max_memory` needs `density`; without one `mem_est` counts the pixels alone.

`cells` are the cells' `(rows, cols)`, and `uncovered` the indices of those no image reaches, which
have no tiles (their merge is empty).  Each entry of `tiles` (in the order of the cells, each
cell's tiles column-major; tiles no image reaches are left out) has
- `core`, `padded`: `(rows, cols)` frame ranges; a cell's cores partition it;
- `offset`: the tile frame's position on `frame` minus its own, `(first.(padded) .- 1)`;
- `frame`: the tile's `TanGrid` (same projection and scale, reference pixel shifted by `offset`);
- `cell`: the index of its cell in `cells`;
- `mem_est`: as above;
- `cutouts`: per image reaching the padded tile, `(; image, name, rows, cols)`: its index in
  `images` and its name (`"1"`, `"2"`, ... for `JointImage`s), and its pixels covering the tile,
  snapped outward to the `bkg_box_size` grid.
`bands` are the images' bands in order of first appearance, the order of the catalog's columns.
`options`, `products` (as [`fit_joint_multipass`](@ref) takes them, with `:coadd` expanded),
`coadd_options` and `product_eltype` are the fit's: `fit_tile` and `merge_tiles` take them from the
plan, so every tile and cell has the same products in the same element type.  `backend` and `T` are
where and in what precision a worker runs, so they are not in the plan (here they only set `mem_est`).

`overlap = :auto` is the reach of a core pixel's inputs (the background mesh, three boxes, plus
one box: 80 native px with the defaults); from `ImageFootprint`s, which have no PSFs, it needs a
numeric `options.fit.R_cap`.  A smaller integer (frame px) is accepted with a warning, for
validation studies of the boundary.
"""
plan_tiles(images::AbstractVector{<:JointImage}, frame::TanGrid; kw...) = first(_plan_tiles(_footprints(images), frame, [im.psf for im in images]; kw...))
plan_tiles(footprints::AbstractVector{<:ImageFootprint}, frame::TanGrid; kw...) = first(_plan_tiles(footprints, frame, nothing; kw...))

function _plan_tiles(fps, frame::TanGrid, psfs; cores = nothing, tile_size::Integer = TILE_SIZE, max_memory::Real = Inf, density = nothing,
        overlap = :auto, options::JointLoopOptions = JointLoopOptions(), products = (), coadd_options::NamedTuple = (;),
        product_eltype::Type{<:AbstractFloat} = Float32, backend = KA.CPU(), T::Type{<:AbstractFloat} = Float64)
    isempty(fps) && throw(ArgumentError("no images"))
    prods = _check_run_args(options, products, coadd_options)
    # a worker finds each cutout's file by its name, so two images sharing one would fit the wrong pixels
    imnames = [fp.name for fp in fps]
    allunique(imnames) || throw(ArgumentError("image names must be unique; repeated: $(join(repr.(unique(filter(n -> count(==(n), imnames) > 1, imnames))), ", "))"))
    tile_size > 0 || throw(ArgumentError("tile_size must be positive"))
    max_memory > 0 || throw(ArgumentError("max_memory must be positive"))
    isfinite(max_memory) && cores === nothing && throw(ArgumentError("max_memory splits given cores; the frame's even split (cores = nothing) is set by tile_size"))
    isfinite(max_memory) && density === nothing &&
        throw(ArgumentError("max_memory needs density: sources per frame px², a number or a function of frame (u, v), e.g. from pre-pass catalogs"))
    # each tile would refine its own copy of a detector's PSF, discontinuous at tile edges
    options.psf_refine === nothing || throw(ArgumentError("psf_refine is not supported by tiled fits; refine with fit_joint_multipass"))
    (overlap === :auto || (overlap isa Integer && overlap >= 0)) ||
        throw(ArgumentError("overlap must be :auto or a non-negative integer, got $(repr(overlap))"))
    oa = _auto_overlap(fps, psfs, frame, options)
    overlap === :auto && oa === nothing &&
        throw(ArgumentError("overlap = :auto without PSFs (planning from ImageFootprints) needs a numeric options.fit.R_cap; or give an integer overlap"))
    o::Int = overlap === :auto ? oa : overlap
    oa !== nothing && o < oa &&
        @warn "overlap = $o frame px is below the $oa px a core pixel's inputs can reach (overlap = :auto); sources near tile edges may differ from an untiled run"
    N, M = cores === nothing ? _partition(frame.ny, frame.nx, tile_size, o) : (nothing, nothing)
    if cores === nothing
        cells = [(1:frame.ny, 1:frame.nx)]
    else
        cells = [(UnitRange{Int}(r), UnitRange{Int}(c)) for (r, c) in cores]
        (isempty(cells) || any(((r, c),) -> isempty(r) || isempty(c), cells)) &&
            throw(ArgumentError("cores must be a non-empty list of non-empty (rows, cols) ranges"))
        for a in eachindex(cells), b in (a + 1):lastindex(cells)
            isempty(intersect(cells[a][1], cells[b][1])) || isempty(intersect(cells[a][2], cells[b][2])) ||
                throw(ArgumentError("cores $a and $b overlap: a source must have one owner"))
        end
    end
    lats = Vector{Any}(undef, length(fps))
    Threads.@threads for e in eachindex(fps)
        lats[e] = _pixel_lattice(fps[e], frame)
    end
    box = _box_dims(options)
    # source pairs in H: positions within linearize's pair_reach, here at worst (45 deg) rotation
    reach = (2 * options.fit.R_fit + 1) * sqrt(2) * maximum(_native_scale(fp) for fp in fps) / frame.scale * 1.01
    # the tile of core (cr, cc) in cell ic from footprints `es`, or `nothing` where none reaches it
    function tile_at(cr, cc, ic, es = eachindex(fps))
        # the even split's padding stops at the frame; given cores' rings go past it wherever images reach,
        # since sources just outside a core affect those inside
        pr, pc = cores === nothing ? (_grow(cr, o, frame.ny), _grow(cc, o, frame.nx)) : ((first(cr) - o):(last(cr) + o), (first(cc) - o):(last(cc) + o))
        rect = (first(pr) - 0.5, last(pr) + 0.5, first(pc) - 0.5, last(pc) + 0.5)
        cutouts = []
        for e in es
            b = _cutout_box(lats[e], rect, box, fps[e].dims)
            b === nothing || push!(cutouts, (; image = e, fps[e].name, rows = b[1], cols = b[2]))
        end
        isempty(cutouts) && return nothing
        offset = (first(pr) - 1, first(pc) - 1)
        tf = TanGrid(frame.ra0, frame.dec0, frame.scale, length(pr), length(pc), frame.cy - offset[1], frame.cx - offset[2])
        ρ = density === nothing ? 0.0 : density isa Function ? density((first(pr) + last(pr)) / 2, (first(pc) + last(pc)) / 2) : density
        ns = ρ * length(pr) * length(pc)
        mem_est = _memory_estimate(sum(c -> length(c.rows) * length(c.cols), cutouts), length(cutouts), length(unique(fps[c.image].band for c in cutouts)),
                                   ns, ns * ρ * (2reach)^2 / 2, backend, T)
        return (; core = (cr, cc), padded = (pr, pc), offset, frame = tf, cell = ic, mem_est, cutouts = [c for c in cutouts])
    end
    # cell ic's tiles: its core split into the fewest near-equal parts, then the least padded area, whose
    # estimates fit max_memory; no part narrower than the overlap, where the ring would dominate
    function split_cell(cr, cc, ic)
        h, w = length(cr), length(cc)
        nmax = (max(1, fld(h, max(o, 1))), max(1, fld(w, max(o, 1))))
        # the footprints reaching the cell's padded box, which contains every part's
        es = [e for e in eachindex(fps) if _cutout_box(lats[e], (first(cr) - o - 0.5, last(cr) + o + 0.5, first(cc) - o - 0.5, last(cc) + o + 0.5), box, fps[e].dims) !== nothing]
        # the tiles of the cell's core split into n1 x n2 near-equal parts, less those no image reaches
        parts(n1, n2) = filter(!isnothing, [tile_at((first(cr) - 1) .+ _part(h, n1, k), (first(cc) - 1) .+ _part(w, n2, l), ic, es) for l in 1:n2 for k in 1:n1])
        fail(why) = throw(ArgumentError("cell $ic ($cr, $cc): no split into parts at least the overlap ($o px) across has every tile within max_memory = $max_memory bytes$why"))
        spent = 0; checked = false
        for n in 1:prod(nmax)
            best = nothing; area = typemax(Int)
            for n1 in 1:min(n, nmax[1])
                (n % n1 == 0 && n ÷ n1 <= nmax[2]) || continue
                ts = parts(n1, n ÷ n1); spent += n
                all(t -> t.mem_est <= max_memory, ts) || continue
                a = sum(t -> length(t.padded[1]) * length(t.padded[2]), ts; init = 0)
                a < area && ((best, area) = (ts, a))
            end
            best === nothing || return best
            # the search to n costs ~n² log n tiles: once it has spent the finest split's tiles, try that once, and
            # stop if even it exceeds the cap (mem_est shrinks with the parts, though not strictly: cutouts snap to boxes)
            if !checked && spent >= prod(nmax) && n < prod(nmax)
                checked = true
                m = maximum(t -> t.mem_est, parts(nmax...); init = 0)
                m <= max_memory || fail("; even the finest split's ($(nmax[1]) x $(nmax[2])) largest tile is $(round(Int, m)) bytes")
            end
        end
        fail("")
    end
    tiles = []
    if cores === nothing
        for l in 1:M, k in 1:N
            t = tile_at(_part(frame.ny, N, k), _part(frame.nx, M, l), 1)
            t === nothing || push!(tiles, t)
        end
    else
        for (ic, (cr, cc)) in enumerate(cells)
            append!(tiles, split_cell(cr, cc, ic))
        end
    end
    uncovered = [ic for ic in eachindex(cells) if !any(t -> t.cell == ic, tiles)]
    plan = (; N, M, tile_size = cores === nothing ? Int(tile_size) : nothing, overlap = o, bands = unique(fp.band for fp in fps), options,
            products = prods, coadd_options, product_eltype, frame, cells, uncovered, tiles = [t for t in tiles])
    return plan, lats
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

"""
    fit_joint_tiled(images, frame; tile_size = $TILE_SIZE, overlap = :auto, backend = KernelAbstractions.CPU(),
                    T = Float64, product_eltype = Float32, options = JointLoopOptions(), show_trace = false,
                    products = (), coadd_options = (;), on_error = :continue) -> NamedTuple

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

A source belongs to the tile whose core contains its final position (see [`merge_tiles`](@ref)): the
frame is one cell.  This is [`plan_tiles`](@ref), then [`fit_tile`](@ref) per tile, then
`merge_tiles`, in one process; those run the same tiles separately.

Returns `(; catalog, bands, plan, tiles, converged, products, t_total)`:
- `catalog`: `fit_joint_multipass`'s columns, `u, v` on `frame` (the tile's position plus its
  offset), plus `tile` (the index into `tiles`) and `depth` (the signed distance inside the frame,
  frame px).  Errors, significances, `nimg` and `pass_number` are the reporting tile's; a band the
  tile lacks is `NaN` in its rows.  Rows are in tile order;
- `bands`: `unique(im.band for im in images)`, the untiled order;
- `plan`: the `plan_tiles` result;
- `tiles`: per tile, `status` (`:ok`, `:failed`, or `:empty` when every cutout is masked),
  `exception` and `backtrace` (a failed tile's), `history`, `converged`, `n_passes`, `bands`,
  `images` (the parent indices of the cutouts it fitted), `n_reported`, `n_dedup` (its sources removed by the
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
        backend = KA.CPU(), T::Type{<:AbstractFloat} = Float64, product_eltype::Type{<:AbstractFloat} = Float32,
        options::JointLoopOptions = JointLoopOptions(), show_trace::Bool = false, products = (), coadd_options::NamedTuple = (;),
        on_error::Symbol = :continue)
    on_error in (:continue, :throw) || throw(ArgumentError("on_error must be :continue or :throw"))
    o = options
    t_total = @elapsed begin
        # arguments that would fail every tile throw before any runs (the plan checks products and coadd_options)
        plan, lats = _plan_tiles(_footprints(images), frame, [im.psf for im in images]; tile_size, overlap, options, products, coadd_options,
                                 product_eltype, backend, T)
        bands = plan.bands
        _check_bands(bands)
        K = o.detector.K; grid = oversample(frame, K)
        prods = _assembly(plan.products, images, bands, grid, plan.product_eltype)
        results = []; tiles = []
        for (k, t) in enumerate(plan.tiles)
            entry = (; status = :ok, exception = nothing, backtrace = nothing, history = NamedTuple[], converged = false, n_passes = 0,
                       bands = Symbol[], images = Int[], n_reported = 0, t_cutout = 0.0, t_setup = 0.0, t_final = 0.0)
            R = nothing
            tk = @elapsed try
                R = fit_tile(plan, k, images; backend, T, show_trace)
                R === nothing && (entry = merge(entry, (; status = :empty, converged = true)))
            catch err
                (err isa InterruptException || on_error === :throw) && rethrow()
                entry = merge(entry, (; status = :failed, exception = err, backtrace = catch_backtrace()))
                # a GPU array is released only when the GC finalizes it, and the failed fit's would
                # otherwise stay on the device into the next tiles
                GC.gc()
            end
            if R !== nothing
                # the catalog only: the products are copied into the full arrays now, so no tile's stay in memory
                push!(results, (; R.catalog, R.tile))
                _assemble!(prods, R.products, t, R.images, lats, K, (0, 0))
                entry = merge(entry, (; R.history, R.converged, n_passes = length(R.history), R.bands, R.images, R.t_cutout, R.t_setup, R.t_final))
            end
            # an empty tile's time is its cutout reads
            push!(tiles, merge(entry, (; t_total = tk), entry.status === :empty ? (; t_cutout = tk) : (;)))
        end
        # the catalog only: the products are already assembled, per-image ones included
        (; catalog, n_dedup) = _merge_cell(plan, 1, results)
        n_reported = zeros(Int, length(tiles))
        for k in catalog.tile; n_reported[k] += 1; end
        tiles = [merge(t, (; n_reported = n_reported[k], n_dedup = n_dedup[k])) for (k, t) in enumerate(tiles)]
    end
    failed = findall(t -> t.status === :failed, tiles)
    isempty(failed) || @warn "$(length(failed)) of $(length(tiles)) tiles failed ($(join(failed, ", "))); their cores have no sources. See `tiles[k].exception`."
    names = Tuple(p for p in (_IMAGE_PRODUCTS..., _MAP_PRODUCTS..., _COADD_PRODUCTS...) if p in plan.products)
    products = NamedTuple{(names..., :grid)}((map(p -> prods[p], names)..., grid))
    return (; catalog, bands, plan, tiles, converged = all(t -> t.converged, tiles), products, t_total)
end

"""
    fit_tile(plan, k, images; backend = KernelAbstractions.CPU(), T = Float64, show_trace = false) -> NamedTuple or nothing

Tile `k` of `plan` (a [`plan_tiles`](@ref) result) fitted by [`fit_joint_multipass`](@ref) with the
plan's `options`, `products`, `coadd_options` and `product_eltype`, on cutouts of its parent
`images`: the plan's images as `JointImage`s, a vector or a dictionary holding at least the tile's
parents, keyed by their index in the plan (a cutout entry's
`image`; its `name` says which file it is).  Each parent is read once as `data[rows, cols]` and
`weight[rows, cols]`, so a parent may be a lazy range reader of a file; cutouts with no positive
finite weight are left out, and with none left the result is `nothing`.  Tiles are independent, so
they may run in any order and in separate processes.

Returns `fit_joint_multipass`'s result with `tile` (`k`), `images` (the indices of the parents
fitted), `t_cutout` (the reads' time) and `catalog` replaced by the tile's whole catalog on the
plan's frame: every source the fit kept, over the padded tile, in `fit_joint_multipass`'s columns for
the plan's `bands` (`NaN` for a band the tile lacks), `u, v` on the frame, plus `tile` and `depth`,
the signed distance (frame px) of the source inside the tile's core, negative in the ring.  The
products are the tile's own, on its padded tile.  [`merge_tiles`](@ref) merges a cell's tiles.
"""
function fit_tile(plan, k::Integer, images::Union{AbstractVector{<:JointImage}, AbstractDict{<:Integer, <:JointImage}}; backend = KA.CPU(),
        T::Type{<:AbstractFloat} = Float64, show_trace::Bool = false)
    t = plan.tiles[k]
    for c in t.cutouts
        (images isa AbstractDict ? haskey(images, c.image) : checkbounds(Bool, images, c.image)) ||
            throw(ArgumentError("tile $k needs image $(c.image) ($(repr(c.name))), which `images` lacks"))
    end
    t_cutout = @elapsed (cuts, idx) = _cutouts(images, t)
    isempty(cuts) && return nothing
    R = fit_joint_multipass(cuts, t.frame; backend, T, plan.product_eltype, options = plan.options, show_trace, plan.products, plan.coadd_options)
    return merge(R, (; catalog = _tile_rows(R.catalog, t, plan.bands, k), tile = k, images = idx, t_cutout))
end

"""
    merge_tiles(plan, c, results) -> (; catalog, n_dedup, products, cell, tiles, absent)

Cell `c` of `plan` from its tiles' [`fit_tile`](@ref) `results`: any of them, in any order; `nothing`
(an empty tile's result) is skipped, and a result from another cell throws.  A source belongs to the
tile whose core contains its final position.  Each tile contributes
- the rows its core grown by `ε = plan.options.min_separation` contains, and
- of the rows outside every tile's grown core (around the cell, in the tiles' rings), those whose
  distance to its core is within `ε` of the distance to the nearest core (as in the cell, a source
  near the line between two tiles is kept by both and then deduplicated),
and of each pair of rows from different tiles closer than `ε`, the one deeper inside its own tile's
core stays.  `catalog` has `fit_tile`'s columns in tile order, with `depth` the signed distance inside
the cell (negative outside it), and `n_dedup[k]` counts tile `k`'s rows removed as duplicates.  A
cell of one tile is its `fit_tile` catalog.

`products`: the plan's detection maps and coadds on the cell's detection grid (`products.grid`),
each tile's core copied at its integer offset; `NaN` (`0` in the counts) where no tile reports, and
in the plan's `product_eltype`.  Per-image products are not merged.

`tiles` are the cell's planned tiles and `absent` those given no result, so the cell is complete when
`absent` is empty.  `absent` does not say why a tile is absent: empty (`fit_tile` returned `nothing`),
failed or not run.  The caller's records of its tiles tell, and a cell is final only when every
absent tile was empty; re-running the others and merging again completes it.  A cell no image
reaches (`plan.uncovered`) has no tiles and merges to an empty, complete cell.
"""
function merge_tiles(plan, c::Integer, results)
    rs = [r for r in results if r !== nothing]
    m = _merge_cell(plan, c, rs)
    # the products: the cell's detection grid, each tile's core nodes copied in
    cell = plan.cells[c]; f = plan.frame; K = plan.options.detector.K; off = first.(cell) .- 1
    grid = oversample(TanGrid(f.ra0, f.dec0, f.scale, length(cell[1]), length(cell[2]), f.cy - off[1], f.cx - off[2]), K)
    names = Tuple(p for p in plan.products if p in _MAP_PRODUCTS || p in _COADD_PRODUCTS)
    prods = _assembly(names, (), plan.bands, grid, plan.product_eltype)
    isempty(names) || foreach(r -> _assemble!(prods, r.products, plan.tiles[r.tile], nothing, nothing, K, K .* off), rs)
    return (; m.catalog, m.n_dedup, products = NamedTuple{(names..., :grid)}((map(p -> prods[p], names)..., grid)), m.cell, m.tiles, m.absent)
end

# merge_tiles' catalog of cell c from results `rs` (no `nothing`s): `(; catalog, n_dedup, cell, tiles, absent)`.
function _merge_cell(plan, c, rs)
    ε = plan.options.min_separation
    1 <= c <= length(plan.cells) || throw(ArgumentError("cell $c is not one of the plan's $(length(plan.cells))"))
    ks = [r.tile for r in rs]
    allunique(ks) || throw(ArgumentError("tiles $(unique(filter(k -> count(==(k), ks) > 1, ks))) appear more than once in `results`"))
    other = filter(k -> plan.tiles[k].cell != c, ks)
    isempty(other) || throw(ArgumentError("results for tiles $other belong to other cells than cell $c: merge each cell separately"))
    cell = plan.cells[c]
    tiles = [k for (k, t) in enumerate(plan.tiles) if t.cell == c]
    peers = [plan.tiles[k].core for k in tiles]   # every tile of the cell, fitted or not
    grown((cr, cc), u, v) = (first(cr) - 0.5 - ε <= u < last(cr) + 0.5 + ε) & (first(cc) - 0.5 - ε <= v < last(cc) + 0.5 + ε)
    cats = map(sortperm(ks)) do i
        r = rs[i].catalog; core = plan.tiles[ks[i]].core
        r[[grown(core, r.u[j], r.v[j]) ||
           (!any(p -> grown(p, r.u[j], r.v[j]), peers) && _depth(core, r.u[j], r.v[j]) >= maximum(p -> _depth(p, r.u[j], r.v[j]), peers) - ε)
           for j in eachindex(r.u)]]
    end
    cat, n_dedup = _merge_catalogs(cats, plan.bands, length(plan.tiles), ε)
    catalog = StructArray(merge(StructArrays.components(cat), (; depth = [_depth(cell, cat.u[j], cat.v[j]) for j in eachindex(cat.u)])))
    return (; catalog, n_dedup, cell = c, tiles, absent = setdiff(tiles, ks))
end

# Signed distance (frame px) of (u, v) inside the frame rectangle (rows, cols), negative outside.
_depth((cr, cc), u, v) = min(u - (first(cr) - 0.5), last(cr) + 0.5 - u, v - (first(cc) - 0.5), last(cc) + 0.5 - v)

# The tile's catalog in global columns (`bands` order, NaN for a band the tile lacks; `u, v` on the full
# frame), with `tile` and `depth` (signed distance inside the core).
function _tile_rows(c, t, bands, k)
    u = c.u .+ t.offset[1]; v = c.v .+ t.offset[2]; n = length(u)
    depth = [_depth(t.core, u[i], v[i]) for i in 1:n]
    col(name) = hasproperty(c, name) ? getproperty(c, name) : fill(NaN, n)
    perband(q) = NamedTuple{Tuple(Symbol(q, :_, b) for b in bands)}(Tuple(col(Symbol(q, :_, b)) for b in bands))
    return StructArray(merge((; u, v, c.ra, c.dec, c.u_err, c.v_err, c.uv_cov), perband(:flux), perband(:flux_err), (; c.significance),
                             perband(:significance), (; c.nimg, c.pass_number, tile = fill(k, n), depth)))
end

# Concatenated tile rows less the cross-tile duplicates, and each tile's count of removed rows.
function _merge_catalogs(cats, bands, ntiles, sep)
    perband(q) = Tuple(Symbol(q, :_, b) for b in bands)
    names = (:u, :v, :ra, :dec, :u_err, :v_err, :uv_cov, perband(:flux)..., perband(:flux_err)..., :significance, perband(:significance)..., :nimg,
             :pass_number, :tile)
    # one allocation per column (Base's reduce(vcat, ::Vector{<:Vector}); with `init` it concatenates
    # pairwise); without tile rows, empty columns of the catalog's types
    col(n) = isempty(cats) ? (n === :nimg ? Int32[] : n in (:tile, :pass_number) ? Int[] : Float64[]) : reduce(vcat, [getproperty(c, n) for c in cats])
    cols = NamedTuple{(names..., :depth)}(Tuple(col(n) for n in (names..., :depth)))
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
function _assembly(products, images, bands, grid, product_eltype)
    out = Dict{Symbol, Any}()
    for p in products
        # the image counts are Int32, empty as 0; the rest `product_eltype`, empty as NaN
        e = p === :nimg || p === :coadd_nimg ? Int32(0) : product_eltype(NaN)
        full() = fill(e, grid.ny, grid.nx)
        out[p] = p in _IMAGE_PRODUCTS ? [fill(e, size(im.data)) for im in images] :
                 p in _MAP_PRODUCTS ? full() : NamedTuple{Tuple(bands)}(Tuple(full() for _ in bands))
    end
    return out
end

# Copy a tile's core into the full-size products: detection-grid nodes through the tile's integer node
# offset (onto a grid starting `goff` nodes into the frame's: a cell's), and each cutout's pixels whose
# frame position (on the parent's lattice) lies in the core.
function _assemble!(prods, tp, t, idx, lats, K, goff)
    (cr, cc) = t.core; (pr, pc) = t.padded
    gr = (K * (first(cr) - 1) + 1):(K * last(cr)); gc = (K * (first(cc) - 1) + 1):(K * last(cc))
    tr = gr .- K * (first(pr) - 1); tcol = gc .- K * (first(pc) - 1)
    gr = gr .- goff[1]; gc = gc .- goff[2]                             # on the target grid, `goff` nodes into the frame's
    rc = owned = nothing
    for (p, full) in prods
        if p in _MAP_PRODUCTS
            full[gr, gc] .= view(getproperty(tp, p), tr, tcol)
        elseif p in _COADD_PRODUCTS
            x = getproperty(tp, p)
            for b in keys(x)
                getproperty(full, b)[gr, gc] .= view(getproperty(x, b), tr, tcol)
            end
        else
            rc === nothing && (rc = [(c.rows, c.cols) for c in t.cutouts if c.image in idx])   # in `idx` order, as `_cutouts` kept them
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
