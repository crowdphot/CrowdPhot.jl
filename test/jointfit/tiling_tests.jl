using CrowdPhot
using CrowdPhot.Joint
using CrowdPhot.PSF: CircularGaussianPRF
using StableRNGs: StableRNG
using Statistics: median, std
using Test

const J = CrowdPhot.Joint

function affine_sky(frame, ang, t)
    A = [cosd(ang) -sind(ang); sind(ang) cosd(ang)]
    return (y, x) -> (uv = A * [y, x] .+ t; grid_to_world(frame, uv[1], uv[2]))
end

"""
Sparse field of `n x n` images `specs = (angle, offset, band, fwhm)` on a frame just covering them
(`TanGrid(images)`, so the images lie inside it), sky 30 plus Poisson noise, weights 1 / (sky + model).
"""
function tiling_field(specs; n = 300, seed = 7, density = 1 / 150, sky = 30.0)
    F0 = TanGrid(80.0, -69.0, 0.11, 4n, 4n)
    imgs = [JointImage(zeros(n, n), ones(n, n), CircularGaussianPRF(y = 0, x = 0, fwhm = f, flux = 1, bkg = 0), affine_sky(F0, a, t); band = b)
            for (a, t, b, f) in specs]
    frame = TanGrid(imgs)
    rng = StableRNG(seed)
    N = round(Int, density * frame.ny * frame.nx)
    θ = vcat(rand(rng, N)' .* frame.ny .+ 0.5, rand(rng, N)' .* frame.nx .+ 0.5, exp.(randn(rng, 2, N) .* 0.8 .+ 6.5))
    sc = JointScene(imgs, frame)
    rt = J.RadiusTable(sc, [im.psf for im in imgs], [im.weight for im in imgs]; R_fit = 3, R_cap = 30)
    model = zeros(length(sc.data)); J.render!(model, sc, θ, J.footprint(sc, θ, rt))
    imgs = map(enumerate(imgs)) do (e, im)
        m = J.image_view(sc, model, e)
        JointImage(m .+ sky .+ sqrt.(sky .+ m) .* randn(rng, n, n), 1 ./ (sky .+ m), im.psf, im.to_sky; band = im.band)
    end
    return (; imgs, frame)
end

const DITHERS = [(0.0, [40.0, 45.0], :A, 2.0), (12.0, [50.0, 30.0], :A, 2.0), (35.0, [60.0, 40.0], :B, 2.6), (60.0, [45.0, 55.0], :B, 2.6)]

"A background estimator from outside CrowdPhot.Background, which a plan file can't name."
struct NotOursBackground <: CrowdPhot.Background.AbstractBackgroundEstimator end

"A parent array allowing only range reads (counted); read number `fail` throws."
struct RangeOnly{T, M <: AbstractMatrix{T}} <: AbstractMatrix{T}
    a::M
    reads::Base.RefValue{Int}
    fail::Int
end
RangeOnly(a, reads; fail = 0) = RangeOnly(a, reads, fail)
Base.size(x::RangeOnly) = size(x.a)
Base.getindex(::RangeOnly, ::Int, ::Int) = error("scalar getindex of a RangeOnly parent")
function Base.getindex(x::RangeOnly, r::UnitRange{Int}, c::UnitRange{Int})
    x.reads[] += 1
    x.reads[] == x.fail && error("simulated read failure")
    return x.a[r, c]
end

lazy(imgs, reads; fail = 0) = [JointImage(RangeOnly(im.data, reads; fail), RangeOnly(im.weight, reads; fail), im.psf, im.to_sky; im.band) for im in imgs]

@testset "partition" begin
    # the design's examples: (fewest tiles, then least padded area) with tile_size 2000, overlap 80
    @test J._partition(4100, 1500, 2000, 80) == (2, 1)
    @test J._partition(4100, 4100, 2000, 80) == (2, 3)
    @test J._partition(4000, 3000, 2000, 80) == (2, 2)
    @test J._partition(1500, 2600, 2000, 80) == (1, 1)
    @test J._part(10, 3, 1) == 1:3 && J._part(10, 3, 2) == 4:6 && J._part(10, 3, 3) == 7:10
    # brute force over every split
    rng = StableRNG(3)
    for _ in 1:200
        ny, nx, S, o = rand(rng, 50:3000), rand(rng, 50:3000), rand(rng, 300:1500), rand(rng, 0:80)
        best = (typemax(Int), typemax(Int), 0, 0)
        for N in 1:ny
            for M in 1:nx
                N * M > best[1] && break
                h = maximum(J._padded_lengths(ny, N, o)); w = maximum(J._padded_lengths(nx, M, o))
                h * w <= S^2 && (best = min(best, (N * M, sum(J._padded_lengths(ny, N, o)) * sum(J._padded_lengths(nx, M, o)), N, M)))
            end
        end
        @test J._partition(ny, nx, S, o) == best[3:4]
    end
    @test_throws "tile_size = 150 must exceed twice the overlap (80 frame px)" J._partition(1000, 1000, 150, 80)
end

@testset "merging tile catalogs" begin
    rows(k, u, v, depth) = (n = length(u); (; u, v, ra = u, dec = v, u_err = fill(0.1, n), v_err = fill(0.1, n), uv_cov = fill(0.001, n), flux_A = ones(n), flux_err_A = fill(0.1, n),
                                           significance = fill(9.0, n), significance_A = fill(9.0, n), nimg = fill(Int32(2), n), pass_number = fill(1, n),
                                           tile = fill(k, n), depth))
    # the second tile's first source duplicates the first tile's second one (within sep), shallower in its own tile
    cats = [rows(1, [10.0, 50.0], [10.0, 50.0], [5.0, 3.0]), rows(2, [50.1, 90.0], [50.0, 90.0], [1.0, 7.0])]
    cat, n_dedup = J._merge_catalogs(cats, [:A], 2, 0.5)
    @test length(cat) == 3 && n_dedup == [0, 1] && cat.tile == [1, 1, 2]
    @test eltype(cat.u) == Float64 && eltype(cat.nimg) == Int32 && eltype(cat.tile) == Int
    # no tile with rows: empty columns of the same types
    cat0, n0 = J._merge_catalogs([], [:A], 2, 0.5)
    @test isempty(cat0) && n0 == [0, 0] && eltype(cat0.u) == Float64 && eltype(cat0.nimg) == Int32 && eltype(cat0.pass_number) == Int
    # a source outside a two-tile cell, by its corner, fitted by each tile on the other's side of the line
    # u = 100.5 where their depths cross: one copy stays, whichever side each fit falls
    o = JointLoopOptions()
    plan = (; options = o, bands = [:A], products = (), product_eltype = Float32, frame = TanGrid(0.0, 0.0, 0.1, 100, 100), cells = [(1:100, 1:100)],
            tiles = [(; core = (1:50, 1:100), cell = 1), (; core = (51:100, 1:100), cell = 1)])
    # results without a `hash` field, as an in-memory plan's may be
    tr(k, u, v) = (; tile = k, catalog = J.StructArray(rows(k, [u], [v], [J._depth(plan.tiles[k].core, u, v)])))
    for (u1, u2) in ((100.6, 100.4), (100.4, 100.6), (100.6, 100.6))
        @test length(merge_tiles(plan, 1, [tr(1, u1, -49.5), tr(2, u2, -49.5)]).catalog) == 1
    end
    # below the cell, 4.5 px from tile 2's core and 54.5 from tile 1's: tile 1's copy, more than min_separation
    # from tile 2's (so no dedup), is dropped as more than min_separation farther from its core than the nearest
    m = merge_tiles(plan, 1, [tr(1, 105.0, 50.0), tr(2, 106.5, 50.0)])
    @test m.catalog.tile == [2] && m.catalog.u == [106.5]
end

@testset "fit_joint_tiled" begin
    f = tiling_field(DITHERS)
    o3 = JointLoopOptions(max_iter = 3, min_growth = 0)
    pr = (:background, :residual, :nu, :nimg, :coadd)
    R0 = fit_joint_multipass(f.imgs, f.frame; options = o3, products = pr)

    @testset "plan" begin
        p = plan_tiles(f.imgs, f.frame; tile_size = 330)
        @test (p.N, p.M, p.overlap, p.tile_size) == (2, 2, 80, 330)
        # the cores partition the frame; padded tiles are the cores grown by the overlap, within the bound
        @test sort(vcat([vec([(r, c) for r in t.core[1], c in t.core[2]]) for t in p.tiles]...)) == sort(vec([(r, c) for r in 1:f.frame.ny, c in 1:f.frame.nx]))
        @test all(t -> length(t.padded[1]) * length(t.padded[2]) <= 330^2, p.tiles)
        @test all(t -> t.offset == first.(t.padded) .- 1 && (t.frame.ny, t.frame.nx) == length.(t.padded), p.tiles)
        # same projection, shifted reference pixel
        t = p.tiles[4]
        @test all(isapprox.(grid_to_world(t.frame, 3.0, 7.0), grid_to_world(f.frame, 3.0 + t.offset[1], 7.0 + t.offset[2]); atol = 1e-12))
        # cutouts are snapped to the 20 px background box grid
        @test all(c -> (first(c.rows) - 1) % 20 == 0 && (first(c.cols) - 1) % 20 == 0 && (last(c.rows) % 20 == 0 || last(c.rows) == 300) &&
                       (last(c.cols) % 20 == 0 || last(c.cols) == 300), (c for t in p.tiles for c in t.cutouts))
        # :auto: max(R_cap + R_fit, 3 boxes) + a box, in frame px; a smaller value warns; bad values throw
        @test plan_tiles(f.imgs, f.frame; options = JointLoopOptions(background = (; bkg_box_size = 30))).overlap == 120
        @test (@test_logs (:warn, r"overlap = 40 frame px is below the 80 px") plan_tiles(f.imgs, f.frame; tile_size = 330, overlap = 40)).overlap == 40
        @test_throws "overlap must be :auto or a non-negative integer" plan_tiles(f.imgs, f.frame; overlap = -1)
        # every pixel of an image is in some tile's cutout, so the parents are covered
        for e in eachindex(f.imgs)
            m = falses(300, 300)
            for t in p.tiles, c in t.cutouts
                c.image == e && (m[c.rows, c.cols] .= true)
            end
            @test all(m)
        end
    end

    @testset "plan from footprints and given cores" begin
        p = plan_tiles(f.imgs, f.frame; tile_size = 330)
        fps = [ImageFootprint(im, "img$e") for (e, im) in enumerate(f.imgs)]
        @test fps[2].dims == (300, 300) && fps[2].band == :A && fps[2].name == "img2"
        @test_throws "image dims must be positive" ImageFootprint((0, 5), identity; band = :A, name = "x")
        @test TanGrid(fps) == f.frame
        # footprints give the same plan as the images, with the overlap their PSFs set
        pf = plan_tiles(fps, f.frame; tile_size = 330, overlap = p.overlap)
        unnamed(t) = merge(t, (; cutouts = [(; c.image, c.rows, c.cols) for c in t.cutouts]))
        @test unnamed.(pf.tiles) == unnamed.(p.tiles) && (pf.N, pf.M, pf.overlap) == (p.N, p.M, p.overlap) && pf.bands == p.bands == [:A, :B]
        # each cutout names its image: the footprint's name, or the JointImage's index
        @test all(c -> c.name == "img$(c.image)", (c for t in pf.tiles for c in t.cutouts)) && all(c -> c.name == string(c.image), (c for t in p.tiles for c in t.cutouts))
        # names must be unique: a worker finds each cutout's file by it
        dup = [ImageFootprint(im, e <= 2 ? "wfi02" : "img$e") for (e, im) in enumerate(f.imgs)]
        @test_throws "image names must be unique; repeated: \"wfi02\"" plan_tiles(dup, f.frame; overlap = 80)
        # without PSFs, :auto needs a numeric R_cap
        @test_throws "needs a numeric options.fit.R_cap" plan_tiles(fps, f.frame)
        oc = JointLoopOptions(fit = JointFitOptions(R_cap = 30))
        @test plan_tiles(fps, f.frame; options = oc).overlap == plan_tiles(f.imgs, f.frame; options = oc).overlap
        # given cores: one tile each in their order, padded on every side, past the frame's edges too
        ny, nx = f.frame.ny, f.frame.nx
        cores = [(1:(ny ÷ 2), 1:(nx ÷ 2)), ((ny ÷ 2 + 1):ny, 1:(nx ÷ 2)), ((ny ÷ 2 - 30):(ny ÷ 2 + 30), (3nx ÷ 4 - 30):(3nx ÷ 4 + 30))]
        pc = plan_tiles(f.imgs, f.frame; cores, overlap = 80)
        @test pc.N === pc.M === pc.tile_size === nothing && pc.overlap == 80 && [t.core for t in pc.tiles] == cores == pc.cells
        @test [t.cell for t in pc.tiles] == [1, 2, 3] && all(t -> t.cell == 1, p.tiles) && p.cells == [(1:ny, 1:nx)] && p.frame == f.frame
        # without a density, mem_est is the cutouts' pixels alone (87 B each on the CPU)
        @test all(t -> t.mem_est == 87 * sum(c -> length(c.rows) * length(c.cols), t.cutouts), pc.tiles)
        @test all(t -> t.padded == ((first(t.core[1]) - 80):(last(t.core[1]) + 80), (first(t.core[2]) - 80):(last(t.core[2]) + 80)), pc.tiles)
        t = pc.tiles[1]
        @test t.offset == (-80, -80) && (t.frame.ny, t.frame.nx) == length.(t.padded)
        @test all(isapprox.(grid_to_world(t.frame, 3.0, 7.0), grid_to_world(f.frame, 3.0 + t.offset[1], 7.0 + t.offset[2]); atol = 1e-12))
        # the even split's tile over the same core clips its padding to the frame
        @test p.tiles[1].core == pc.tiles[1].core && p.tiles[1].padded == (1:(ny ÷ 2 + 80), 1:(nx ÷ 2 + 80))
        @test_throws "cores 1 and 2 overlap" plan_tiles(f.imgs, f.frame; cores = [(1:150, 1:200), (150:300, 1:200)])
        @test_throws "non-empty list of non-empty" plan_tiles(f.imgs, f.frame; cores = [(1:0, 1:10)])
        # a cell over max_memory is split into near-equal parts, the fewest whose tiles all fit: with a cap
        # between the cell's estimate and its best split in two's, in two
        cell = (61:(ny - 60), 61:(nx - 60))
        p1 = plan_tiles(f.imgs, f.frame; cores = [cell], overlap = 80)
        h, w = length.(cell)
        halves = [[((first(cell[1]) - 1) .+ J._part(h, 2, k), cell[2]) for k in 1:2], [(cell[1], (first(cell[2]) - 1) .+ J._part(w, 2, k)) for k in 1:2]]
        m2 = minimum(cs -> maximum(t -> t.mem_est, plan_tiles(f.imgs, f.frame; cores = cs, overlap = 80).tiles), halves)
        @test m2 < only(p1.tiles).mem_est
        cap = (m2 + only(p1.tiles).mem_est) / 2
        ps = plan_tiles(f.imgs, f.frame; cores = [cell], overlap = 80, max_memory = cap, density = 0)
        @test length(ps.tiles) == 2 && all(t -> t.mem_est <= cap && t.cell == 1, ps.tiles)
        @test sort(vcat([vec([(r, c) for r in t.core[1], c in t.core[2]]) for t in ps.tiles]...)) == sort(vec([(r, c) for r in cell[1], c in cell[2]]))
        # a density adds the sources' H; a function of frame position is taken at the padded tile's center
        @test plan_tiles(f.imgs, f.frame; cores = [cell], overlap = 80, density = 0.01).tiles[1].mem_est > only(p1.tiles).mem_est
        at = NTuple{2, Float64}[]
        pd = plan_tiles(f.imgs, f.frame; cores = [(1:(ny ÷ 2), 1:nx), ((ny ÷ 2 + 1):ny, 1:nx)], overlap = 80,
                        density = (u, v) -> (push!(at, (u, v)); u < ny / 2 ? 0.0 : 0.01))
        # every tile's center is queried (a split search may query other points too)
        centers = [((first(t.padded[1]) + last(t.padded[1])) / 2, (first(t.padded[2]) + last(t.padded[2])) / 2) for t in pd.tiles]
        @test issubset(centers, at)
        npix(t) = 87 * sum(c -> length(c.rows) * length(c.cols), t.cutouts)
        @test pd.tiles[1].mem_est == npix(pd.tiles[1]) && pd.tiles[2].mem_est > npix(pd.tiles[2])
        @test_throws "max_memory needs density" plan_tiles(f.imgs, f.frame; cores = [cell], max_memory = cap)
        @test_throws "max_memory splits given cores" plan_tiles(f.imgs, f.frame; max_memory = cap, density = 0)
        # an impossible cap stops once the search has spent the finest split's tiles, rather than at the end
        @test_throws "no split into parts at least the overlap (80 px) across has every tile within max_memory = 1000.0 bytes; even the finest split's" plan_tiles(f.imgs, f.frame; cores = [cell], overlap = 80, max_memory = 1e3, density = 0)
        # a cell under twice the overlap across has one candidate, the whole cell, and fails after it
        err = try plan_tiles(f.imgs, f.frame; cores = [(61:150, 61:150)], overlap = 80, max_memory = 1e3, density = 0); nothing catch e; e end
        @test err isa ArgumentError && occursin("no split into parts at least the overlap", err.msg) && !occursin("even the finest", err.msg)
    end

    @testset "fit_tile and merge_tiles reproduce fit_joint_tiled" begin
        R = fit_joint_tiled(f.imgs, f.frame; tile_size = 330, options = o3)
        p = plan_tiles(f.imgs, f.frame; tile_size = 330, options = o3)
        @test p.tiles == R.plan.tiles && p.options === o3
        # tiles fitted separately and in another order merge to the same catalog
        order = [3, 1, 4, 2]
        res = [fit_tile(p, k, f.imgs) for k in order]
        m = merge_tiles(p, 1, res)
        @test isequal(m.catalog, R.catalog) && m.n_dedup == [t.n_dedup for t in R.tiles]
        @test (m.cell, m.tiles, m.absent) == (1, [1, 2, 3, 4], Int[]) && keys(m.products) == (:grid,)
        @test_throws "tiles [3] have more than one result" merge_tiles(p, 1, [res[1], res[1]])
        @test_throws "cell 2 is not one of the plan's 1" merge_tiles(p, 2, res)
        # `nothing` (an empty tile) is skipped, and tiles given no result are absent: their cores stay empty but
        # for the margin `min_separation` their neighbors report, though tile 3's ring reaches into core 1
        m2 = merge_tiles(p, 1, [res[1], nothing, res[3]])           # tiles 3 and 4
        @test m2.absent == [1, 2] && all(in((3, 4)), m2.catalog.tile)
        indepth(k, c) = [J._depth(p.tiles[k].core, c.u[i], c.v[i]) for i in eachindex(c.u)]
        @test count(>(5), indepth(1, res[1].catalog)) > 10
        @test all(k -> all(<=(o3.min_separation), indepth(k, m2.catalog)), (1, 2))
        @test isequal(m2.catalog, merge_tiles(p, 1, res[[1, 3]]).catalog)
        # the parents may be just the tile's, keyed by name (as a worker loads them); results still name them by plan index
        parents = Dict(c.name => f.imgs[c.image] for c in p.tiles[3].cutouts)
        r3 = fit_tile(p, 3, parents)
        @test isequal(r3.catalog, res[1].catalog) && r3.images == res[1].images == [c.image for c in p.tiles[3].cutouts] && r3.t_cutout > 0
        missing_one = Dict(c.name => f.imgs[c.image] for c in p.tiles[3].cutouts[2:end])
        @test_throws "tile 3 needs image $(p.tiles[3].cutouts[1].image)" fit_tile(p, 3, missing_one)
        # keyed by plan index, a dictionary is refused: an index is only meaningful against the plan's full list
        @test_throws "key `images` by image name" fit_tile(p, 3, Dict(c.image => f.imgs[c.image] for c in p.tiles[3].cutouts))
        # a vector must be the plan's whole list: one without an image would shift every later index
        @test_throws "pass every image it was made from" fit_tile(p, 3, f.imgs[2:end])
        # a tile whose every cutout is masked has no result
        masked = [JointImage(im.data, zero(im.weight), im.psf, im.to_sky; im.band) for im in f.imgs]
        @test fit_tile(p, 3, masked) === nothing
        # in fit_joint_tiled such a tile is :empty, its time all reads, and reports nothing
        wm = [copy(im.weight) for im in f.imgs]
        for c in p.tiles[3].cutouts; wm[c.image][c.rows, c.cols] .= 0; end
        Rm = fit_joint_tiled([JointImage(im.data, wm[e], im.psf, im.to_sky; im.band) for (e, im) in enumerate(f.imgs)], f.frame; tile_size = 330,
                             options = JointLoopOptions(max_iter = 1))
        tm = Rm.tiles[3]
        @test Rm.plan.tiles == p.tiles && tm.status === :empty && tm.converged && tm.images == Int[] && tm.t_cutout == tm.t_total > 0
        @test tm.n_reported == 0 && !any(==(3), Rm.catalog.tile) && all(k -> Rm.tiles[k].status !== :empty, (1, 2, 4))
        # a tile's catalog is its whole padded tile on the frame, with depth inside its core
        c = res[1].catalog; t = p.tiles[3]
        @test propertynames(c) == propertynames(R.catalog) && all(==(3), c.tile) && res[1].tile == 3
        @test count(<(-5), c.depth) > 10 && count(>(5), c.depth) > 10
        i = argmin(c.depth)
        @test c.depth[i] ≈ min(c.u[i] - first(t.core[1]) + 0.5, last(t.core[1]) + 0.5 - c.u[i], c.v[i] - first(t.core[2]) + 0.5, last(t.core[2]) + 0.5 - c.v[i])
        @test length(c) > count(==(3), R.catalog.tile)
    end

    @testset "a cell split into tiles" begin
        ny, nx = f.frame.ny, f.frame.nx
        cell = (61:(ny - 60), 61:(nx - 60)); other = (11:60, 61:(nx - 60))
        pr2 = (:nu, :nimg, :coadd_data, :coadd_nimg)
        p1 = plan_tiles(f.imgs, f.frame; cores = [cell, other], overlap = 80, options = o3, products = pr2)
        @test length(p1.tiles) == 2 && p1.products == pr2 && p1.coadd_options == (;)
        # split in two (the cap as in the plan tests)
        h = length(cell[1])
        m2 = maximum(t -> t.mem_est, plan_tiles(f.imgs, f.frame; cores = [((first(cell[1]) - 1) .+ J._part(h, 2, k), cell[2]) for k in 1:2], overlap = 80).tiles)
        ps = plan_tiles(f.imgs, f.frame; cores = [cell, other], overlap = 80, options = o3, products = pr2, max_memory = (m2 + p1.tiles[1].mem_est) / 2,
                        density = 0)
        ks = findall(t -> t.cell == 1, ps.tiles)
        @test length(ks) == 2
        fit(p, k) = fit_tile(p, k, f.imgs)
        # an unsplit cell is its tile's catalog
        r1 = fit(p1, 1)
        @test isequal(merge_tiles(p1, 1, [r1]).catalog, r1.catalog)
        @test_throws "results for tiles [2] belong to other cells than cell 1" merge_tiles(p1, 1, [r1, fit(p1, 2)])
        m = merge_tiles(ps, 1, [fit(ps, k) for k in reverse(ks)])
        @test m.cell == 1 && m.tiles == ks && isempty(m.absent)
        c = m.catalog
        @test propertynames(c) == propertynames(r1.catalog) && issorted(c.tile)
        # no cross-tile pair closer than min_separation; each row in its own tile's core grown by it, or in no tile's
        @test !any(i -> any(j -> c.tile[j] != c.tile[i] && hypot(c.u[j] - c.u[i], c.v[j] - c.v[i]) < 1.0, eachindex(c.u)), eachindex(c.u))
        @test all(i -> J._depth(ps.tiles[c.tile[i]].core, c.u[i], c.v[i]) >= -1 || all(k -> J._depth(ps.tiles[k].core, c.u[i], c.v[i]) < -1, ks), eachindex(c.u))
        # the ring around the cell is kept, depth measured inside the cell
        @test count(<(-5), c.depth) > 10 && all(i -> c.depth[i] ≈ J._depth(cell, c.u[i], c.v[i]), eachindex(c.u))
        # close to the unsplit cell's catalog
        c1 = r1.catalog
        d = [minimum(hypot.(c.u .- c1.u[i], c.v .- c1.v[i])) for i in eachindex(c1.u) if c1.depth[i] > 0]
        @test count(<(0.5), d) >= 0.99 * length(d) && abs(count(>(0), c.depth) - length(d)) <= 0.01 * length(d)
        # the products: each tile's core on the cell's detection grid
        K = JointDetector().K; g = m.products.grid
        @test keys(m.products) == (:nu, :nimg, :coadd_data, :coadd_nimg, :grid) && size(m.products.nu) == K .* length.(cell)
        @test all(isapprox.(grid_to_world(g, 1.0, 1.0), grid_to_world(J.oversample(f.frame, K), K * (first(cell[1]) - 1) + 1.0, K * (first(cell[2]) - 1) + 1.0); atol = 1e-12))
        for k in ks
            t = ps.tiles[k]; rk = fit(ps, k)
            gr = (K * (first(t.core[1]) - first(cell[1])) + 1):(K * (last(t.core[1]) - first(cell[1]) + 1))
            gc = (K * (first(t.core[2]) - first(cell[2])) + 1):(K * (last(t.core[2]) - first(cell[2]) + 1))
            tr = (K * (first(t.core[1]) - first(t.padded[1])) + 1):(K * (last(t.core[1]) - first(t.padded[1]) + 1))
            tc = (K * (first(t.core[2]) - first(t.padded[2])) + 1):(K * (last(t.core[2]) - first(t.padded[2]) + 1))
            @test isequal(m.products.nu[gr, gc], rk.products.nu[tr, tc]) && isequal(m.products.coadd_data.B[gr, gc], rk.products.coadd_data.B[tr, tc])
        end
        @test isnan.(m.products.nu) == (m.products.nimg .== 0) && count(>(0), m.products.nimg) > 0.9 * length(m.products.nimg)
        @test eltype(m.products.coadd_nimg.A) == Int32
        # no results (every tile failed or not run): an empty cell, every tile absent, the plan's products all NaN
        m0 = merge_tiles(ps, 1, [])
        @test isempty(m0.catalog) && propertynames(m0.catalog) == propertynames(c) && m0.absent == ks
        @test keys(m0.products) == keys(m.products) && all(isnan, m0.products.nu) && all(==(0), m0.products.coadd_nimg.B)
        # a cell no image reaches has no tiles: it merges to an empty, complete cell with the same fields
        pu = plan_tiles(f.imgs, f.frame; cores = [cell, other, ((ny + 500):(ny + 520), 1:20)], overlap = 80, options = o3, products = pr2)
        @test pu.uncovered == [3] && p1.uncovered == Int[] && all(t -> t.cell != 3, pu.tiles)
        mu = merge_tiles(pu, 3, [])
        @test isempty(mu.catalog) && propertynames(mu.catalog) == propertynames(c) && mu.tiles == mu.absent == Int[]
        @test keys(mu.products) == keys(m.products) && size(mu.products.nu) == 2 .* (21, 20) && all(isnan, mu.products.nu)
        @test pu.product_eltype === Float32 && eltype(m.products.nu) == eltype(mu.products.nu) == eltype(m0.products.coadd_data.A) == Float32
        # the plan's product_eltype sets every tile's and cell's, with results or without
        p64 = plan_tiles(f.imgs, f.frame; cores = [cell, other, ((ny + 500):(ny + 520), 1:20)], overlap = 80, options = o3, products = pr2,
                         product_eltype = Float64)
        r64 = fit_tile(p64, 2, f.imgs)
        @test eltype(r64.products.nu) == eltype(first(r64.products.coadd_data)) == Float64 && eltype(first(r64.products.coadd_nimg)) == Int32
        @test eltype(merge_tiles(p64, 2, [r64]).products.nu) == Float64
        @test eltype(merge_tiles(p64, 2, []).products.nu) == eltype(merge_tiles(p64, 3, []).products.coadd_data.B) == Float64
    end

    @testset "one tile reproduces the untiled run" begin
        R1 = fit_joint_tiled(f.imgs, f.frame; options = o3, products = pr)
        @test length(R1.tiles) == 1 && R1.plan.tiles[1].offset == (0, 0)
        @test all(n -> isequal(getproperty(R1.catalog, n), getproperty(R0.catalog, n)), propertynames(R0.catalog))
        @test propertynames(R1.catalog) == (propertynames(R0.catalog)..., :tile, :depth)
        @test R1.bands == R0.bands && [h.n_catalog for h in R1.tiles[1].history] == [h.n_catalog for h in R0.history]
        @test keys(R1.products) == keys(R0.products) && R1.products.grid == R0.products.grid
        @test isequal(R1.products.background, R0.products.background) && isequal(R1.products.nu, R0.products.nu)
        @test isequal(R1.products.coadd_data, R0.products.coadd_data) && R1.products.coadd_nimg == R0.products.coadd_nimg
    end

    @testset "2 x 2 tiles" begin
        R = fit_joint_tiled(f.imgs, f.frame; tile_size = 330, options = o3, products = pr)
        c = R.catalog; c0 = R0.catalog; p = R.plan
        @test all(t -> t.status === :ok, R.tiles) && R.converged == all(t -> t.converged, R.tiles)
        @test sum(t -> t.n_reported, R.tiles) == length(c) && all(k -> count(==(k), c.tile) == R.tiles[k].n_reported, 1:4)
        @test sum(t -> t.n_dedup, R.tiles) > 0
        # no cross-tile pair closer than min_separation; each row within its tile's core grown by it, or (fitted
        # past the frame's edge) in no tile's
        @test !any(i -> any(j -> c.tile[j] != c.tile[i] && hypot(c.u[j] - c.u[i], c.v[j] - c.v[i]) < 1.0, eachindex(c.u)), eachindex(c.u))
        @test all(i -> J._depth(p.tiles[c.tile[i]].core, c.u[i], c.v[i]) >= -1 || all(t -> J._depth(t.core, c.u[i], c.v[i]) < -1, p.tiles), eachindex(c.u))
        @test all(i -> c.depth[i] ≈ J._depth((1:f.frame.ny, 1:f.frame.nx), c.u[i], c.v[i]), eachindex(c.u))
        # close to the untiled catalog
        d = [minimum(hypot.(c.u .- c0.u[i], c.v .- c0.v[i])) for i in eachindex(c0.u)]
        @test count(<(0.5), d) >= 0.99 * length(c0) && abs(length(c) - length(c0)) <= 0.01 * length(c0)
        j = [argmin(hypot.(c.u .- c0.u[i], c.v .- c0.v[i])) for i in eachindex(c0.u)]
        k = findall((d .< 0.5) .& isfinite.(c0.flux_err_A))
        @test median(abs.((c.flux_A[j[k]] .- c0.flux_A[k]) ./ c0.flux_err_A[k])) < 0.01
        # the assembled products are each tile's own products over its core
        g = R.products.grid; K = JointDetector().K
        @test g == R0.products.grid && R.products.nimg == R0.products.nimg
        @test all(e -> !any(isnan, R.products.background[e]), 1:4)
        for (kt, t) in enumerate(p.tiles)
            cuts, idx = J._cutouts(f.imgs, t)
            Rk = fit_joint_multipass(cuts, t.frame; options = o3, products = pr)
            @test Rk.catalog.u .+ t.offset[1] ⊇ c.u[c.tile .== kt]
            gr = (K * (first(t.core[1]) - 1) + 1):(K * last(t.core[1])); gc = (K * (first(t.core[2]) - 1) + 1):(K * last(t.core[2]))
            tr = gr .- K * t.offset[1]; tc = gc .- K * t.offset[2]
            @test isequal(R.products.nu[gr, gc], Rk.products.nu[tr, tc])
            @test isequal(R.products.coadd_residual.B[gr, gc], Rk.products.coadd_residual.B[tr, tc])
            # a pixel is the owning tile's: compare where the frame position (through the cutout's map) is well inside the core
            for (i, e) in enumerate(idx)
                cu = t.cutouts[findfirst(x -> x.image == e, t.cutouts)]
                own = [begin uv = J.world_to_grid(f.frame, f.imgs[e].to_sky(y, x)...)
                           first(t.core[1]) + 0.5 <= uv[1] < last(t.core[1]) - 0.5 && first(t.core[2]) + 0.5 <= uv[2] < last(t.core[2]) - 0.5 end
                       for y in cu.rows, x in cu.cols]
                @test R.products.background[e][cu.rows, cu.cols][own] == Rk.products.background[i][own]
                @test R.products.residual[e][cu.rows, cu.cols][own] == Rk.products.residual[i][own]
            end
        end
    end

    @testset "lazy parents: one range read of data and weight per cutout" begin
        reads = Ref(0)
        R = fit_joint_tiled(lazy(f.imgs, reads), f.frame; tile_size = 330, options = o3)
        @test reads[] == 2 * sum(t -> length(t.cutouts), R.plan.tiles)
        @test isequal(R.catalog, fit_joint_tiled(f.imgs, f.frame; tile_size = 330, options = o3).catalog)
    end

    @testset "a failed tile leaves its core empty" begin
        p = plan_tiles(f.imgs, f.frame; tile_size = 330)
        reads = Ref(0)
        fail = 2 * length(p.tiles[1].cutouts) + 1          # the first read of tile 2
        R = @test_logs (:warn, r"1 of 4 tiles failed \(2\)") match_mode = :any fit_joint_tiled(lazy(f.imgs, reads; fail), f.frame; tile_size = 330,
                                                                                              options = o3, products = (:nu, :background))
        t = R.tiles[2]
        @test t.status === :failed && t.exception isa ErrorException && t.backtrace !== nothing && !R.converged
        @test all(k -> R.tiles[k].status === :ok, (1, 3, 4)) && !any(==(2), R.catalog.tile)
        cr, cc = p.tiles[2].core
        @test !any(i -> first(cr) + 0.5 <= R.catalog.u[i] < last(cr) - 0.5 && first(cc) + 0.5 <= R.catalog.v[i] < last(cc) - 0.5, eachindex(R.catalog.u))
        K = JointDetector().K
        core2 = falses(size(R.products.nu)); core2[(K * (first(cr) - 1) + 1):(K * last(cr)), (K * (first(cc) - 1) + 1):(K * last(cc))] .= true
        @test all(isnan, R.products.nu[core2]) && isnan.(R.products.nu)[.!core2] == isnan.(R0.products.nu)[.!core2]
        reads[] = 0
        @test_throws "simulated read failure" fit_joint_tiled(lazy(f.imgs, reads; fail), f.frame; tile_size = 330, options = o3, on_error = :throw)
    end
end

@testset "tiles with permuted and missing bands" begin
    # image order A1, B1, B2, A2 in three tiles: the left meets A1, A2 (no B), the right B1, B2, A2
    # (bands numbered B, A)
    specs = [(0.0, [40.0, 40.0], :A, 2.0), (0.0, [40.0, 520.0], :B, 2.6), (5.0, [45.0, 515.0], :B, 2.6), (0.0, [40.0, 240.0], :A, 2.0)]
    f = tiling_field(specs; seed = 11)
    o3 = JointLoopOptions(max_iter = 3, min_growth = 0)
    R = fit_joint_tiled(f.imgs, f.frame; tile_size = 380, options = o3, products = (:coadd_data,))
    @test (R.plan.N, R.plan.M) == (1, 3) && R.bands == [:A, :B]
    @test R.tiles[1].bands == [:A] && R.tiles[2].bands == [:A, :B] && R.tiles[3].bands == [:B, :A]
    @test R.tiles[1].images == [1, 4] && R.tiles[2].images == [1, 2, 3, 4] && R.tiles[3].images == [2, 3, 4]
    c = R.catalog
    @test propertynames(c)[8:11] == (:flux_A, :flux_B, :flux_err_A, :flux_err_B)
    l = c.tile .== 1; r = c.tile .== 3
    @test count(l) > 50 && count(r) > 50
    @test all(isnan, c.flux_B[l]) && all(isnan, c.significance_B[l]) && all(isfinite, c.flux_A[l])
    # columns are matched by band name: the right tile's B fluxes are its band-1 fluxes
    @test count(isfinite, c.flux_B[r]) > 0.8 * count(r) && median(filter(isfinite, c.flux_B[r])) > 0
    @test keys(R.products.coadd_data) == (:A, :B)
    K = JointDetector().K; cc = R.plan.tiles[3].core[2]
    @test count(isfinite, R.products.coadd_data.B[:, (K * (first(cc) - 1) + 1):(K * last(cc))]) > 0.5 * K^2 * 326 * length(cc)
    # a fully masked cutout is left out of its tile: A2 masked over its left-tile cutout only
    w = copy(f.imgs[4].weight); w[:, only(c.cols for c in R.plan.tiles[1].cutouts if c.image == 4)] .= 0
    imgs = [f.imgs[1:3]; JointImage(f.imgs[4].data, w, f.imgs[4].psf, f.imgs[4].to_sky; band = :A)]
    Rm = fit_joint_tiled(imgs, f.frame; tile_size = 380, options = JointLoopOptions(max_iter = 1))
    @test [t.images for t in Rm.tiles] == [[1], [1, 2, 3, 4], [2, 3, 4]]
end

@testset "plan folders" begin
    f = tiling_field(DITHERS)
    o3 = JointLoopOptions(max_iter = 3, min_growth = 0)
    fps = [ImageFootprint(im, "img$e") for (e, im) in enumerate(f.imgs)]
    ny, nx = f.frame.ny, f.frame.nx

    @testset "options as TOML" begin
        o = JointLoopOptions(detect_sigma = 4.0, blend_threshold = nothing, band_weights = Dict(:A => 2.0, :B => 0.5), discard_passes = 2,
                             detector = JointDetector(K = 4), fit = JointFitOptions(R_cap = 30, model_rad = 2.5),
                             background = (; bkg_estimator = CrowdPhot.MMMBackground(median_factor = 3.5, mean_factor = 2.0), bkg_box_size = (20, 30),
                                           bkg_rms_estimator = CrowdPhot.MADStdRMS()))
        d = J.TOML.parse(sprint(io -> J.TOML.print(io, J._toml(o); sorted = true)))
        # every field is written: nothing as "nothing", estimators by name (with their fields when they have some)
        @test Set(keys(d)) == Set(string.(fieldnames(JointLoopOptions))) && d["blend_threshold"] == "nothing" && d["psf_refine"] == "nothing"
        @test d["fit"]["max_flux_step"] == Inf && d["fit"]["R_cap"] == 30 && d["accept"] == "nu" && d["band_weights"] == Dict("A" => 2.0, "B" => 0.5)
        @test d["background"]["bkg_rms_estimator"] == "MADStdRMS" && d["background"]["bkg_estimator"] == Dict("type" => "MMMBackground", "median_factor" => 3.5, "mean_factor" => 2.0)
        r = J._options_from(JointLoopOptions, d, "options."; strict = true)
        @test J._toml(r) == J._toml(o) && r.blend_threshold === nothing && r.psf_refine === nothing && r.fit.model_rad === 2.5 && r.fit.R_cap === 30
        @test r.accept === :nu && r.detector.K == 4 && r.band_weights == o.band_weights && r.background.bkg_estimator == o.background.bkg_estimator
        @test keys(r.background) == (:bkg_box_size, :bkg_estimator, :bkg_rms_estimator) && r.background.bkg_box_size === (20, 30)
        # a plan records every field, so a missing one is an error; a user's configuration takes the default
        d2 = deepcopy(d); delete!(d2, "dedup"); delete!(d2["fit"], "R_fit")
        @test_throws "option options.dedup is missing" J._options_from(JointLoopOptions, d2, "options."; strict = true)
        r2 = J._options_from(JointLoopOptions, d2, "options."; strict = false)
        @test r2.dedup == JointLoopOptions().dedup && r2.fit.R_fit == JointFitOptions().R_fit && r2.detect_sigma == 4.0
        d3 = deepcopy(d); d3["fit"]["bogus"] = 1
        @test_throws "unknown option options.fit.bogus" J._options_from(JointLoopOptions, d3, "options."; strict = true)
        d4 = deepcopy(d); d4["detect_sigma"] = "five"
        @test_throws "option options.detect_sigma: \"five\" is not a Float64" J._options_from(JointLoopOptions, d4, "options."; strict = true)
        # estimators: only CrowdPhot.Background's, of the right kind
        for (v, msg) in (("NoSuchBackground", "\"NoSuchBackground\" is not one of CrowdPhot.Background's AbstractBackgroundEstimators"),
                         ("MADStdRMS", "\"MADStdRMS\" is not one of CrowdPhot.Background's AbstractBackgroundEstimators"),
                         (Dict("median_factor" => 3), "an estimator is a name, or a table with `type`"))
            d5 = deepcopy(d); d5["background"]["bkg_estimator"] = v
            @test_throws msg J._options_from(JointLoopOptions, d5, "options."; strict = true)
        end
        @test_throws "a plan can name only CrowdPhot.Background's estimators" J._toml(JointLoopOptions(background = (; bkg_estimator = NotOursBackground())))
        # background keywords and estimator fields read by their schema: `nothing` and tuples survive, estimator
        # fields are Float64, and plan_tiles normalizes the same way, so a plan matches its file
        bo = JointLoopOptions(background = (; bkg_estimator = CrowdPhot.MMMBackground(median_factor = 3.5f0, mean_factor = 2.0f0),
                                             bkg_kws = (; sigma = nothing, filter_size = (3, 5), maxiters = 4)))
        db = J.TOML.parse(sprint(io -> J.TOML.print(io, J._toml(bo); sorted = true)))
        rb = J._options_from(JointLoopOptions, db, "options."; strict = true).background
        @test rb.bkg_kws === (; filter_size = (3, 5), sigma = nothing, maxiters = 4)
        @test rb.bkg_estimator === CrowdPhot.MMMBackground(median_factor = 3.5, mean_factor = 2.0)
        @test plan_tiles(fps, f.frame; overlap = 80, options = bo).options.background === rb
        @test J._options_from(J._BKG_KWS, (; sigma = 3, filter_size = [3, 3]), "k."; strict = true) === (; filter_size = (3, 3), sigma = 3.0)
        @test J._options_from(J._BKG_KWS, (; sigma = (2, 5)), "k."; strict = true) === (; sigma = (2.0, 5.0))
        @test_throws "option k.sigma: (1, 2, 3) is not a" J._options_from(J._BKG_KWS, (; sigma = (1, 2, 3)), "k."; strict = true)
        @test_throws "unknown option options.background.bkg_kws.mask" plan_tiles(fps, f.frame; overlap = 80,
                                                                                 options = JointLoopOptions(background = (; bkg_kws = (; mask = trues(2, 2)))))
        # another module's estimator, or a function, is kept in memory (only write_plan refuses it)
        @test plan_tiles(fps, f.frame; overlap = 80, options = JointLoopOptions(background = (; bkg_estimator = NotOursBackground()))).options.background.bkg_estimator === NotOursBackground()
        pf = plan_tiles(fps, f.frame; overlap = 80, options = JointLoopOptions(background = (; bkg_estimator = median, bkg_rms_estimator = std)))
        @test pf.options.background.bkg_estimator === median && pf.options.background.bkg_rms_estimator === std
        @test_throws "a plan file can't hold a" write_plan(joinpath(mktempdir(), "fn"), pf)
        # the schema's keywords are ones Background2D and background2d! take
        @test any(m -> keys(J._BKG_KWS) ⊆ Base.kwarg_decl(m), methods(CrowdPhot.Background2D)) && keys(J._BKG_KWS) ⊆ Base.kwarg_decl(only(methods(J.background2d!)))
    end

    @testset "exclusions" begin
        p = plan_tiles(fps, f.frame; tile_size = 330, overlap = 80, exclude = ["img2" => "F146 exposure 1: bad"])
        @test [im.name for im in p.images] == ["img1", "img2", "img3", "img4"] && p.excluded == [(; name = "img2", reason = "F146 exposure 1: bad")]
        @test p.images[1] == (; name = "img1", dims = (300, 300), band = :A)
        # cutouts index the list given, and the excluded image is in none
        @test all(c -> c.name != "img2" && c.image != 2 && p.images[c.image].name == c.name, (c for t in p.tiles for c in t.cutouts))
        @test_throws "exclude names images not in the list: \"img9\"" plan_tiles(fps, f.frame; overlap = 80, exclude = ["img9" => "x"])
        @test_throws "every image is excluded" plan_tiles(fps, f.frame; overlap = 80, exclude = [fp.name => "x" for fp in fps])
        # a plan made from JointImages with an exclusion is fitted with the same list: the same tile as a plan
        # made without that image
        pe = plan_tiles(f.imgs, f.frame; tile_size = 330, options = o3, exclude = ["2" => "bad"])
        ps = plan_tiles(f.imgs[[1, 3, 4]], f.frame; tile_size = 330, options = o3)
        @test [c.image for c in pe.tiles[1].cutouts] == [1, 3, 4] && [c.name for c in pe.tiles[1].cutouts] == ["1", "3", "4"]
        @test [(t.core, t.padded) for t in pe.tiles] == [(t.core, t.padded) for t in ps.tiles] && pe.bands == ps.bands && pe.overlap == ps.overlap
        re = fit_tile(pe, 1, f.imgs)
        @test re.images == [1, 3, 4] && isequal(re.catalog, fit_tile(ps, 1, f.imgs[[1, 3, 4]]).catalog)
    end

    @testset "write, read back, refuse edits" begin
        root = mktempdir()
        cores = [(1:(ny ÷ 2), 1:nx), ((ny ÷ 2 + 1):ny, 1:nx), ((ny + 500):(ny + 520), 1:20)]
        p = plan_tiles(fps, f.frame; cores, overlap = 80, options = o3, products = (:nu, :coadd), coadd_options = (; clip = 3, pixfrac = 0.8),
                       product_eltype = Float64, exclude = ["img4" => "a test"])
        # coadd_options in their canonical order and types, as the file reads them back; a value of the wrong type
        # is an error naming the option
        @test p.coadd_options === (; pixfrac = 0.8, clip = 3.0) && p.uncovered == [3]
        @test_throws "option coadd_options.clip: \"3\" is not a" plan_tiles(fps, f.frame; overlap = 80, coadd_options = (; clip = "3"))
        @test_throws "option coadd_options.clip_grow: 1.5 is not a Int64" plan_tiles(fps, f.frame; overlap = 80, coadd_options = (; clip_grow = 1.5))
        dir = joinpath(root, "plan")
        h = write_plan(dir, p; shard_size = 1)
        @test readdir(dir) == ["jobs.toml", "plan.toml", "tiles"] && readdir(joinpath(dir, "tiles")) == ["shard_$k.toml" for k in eachindex(p.tiles)]
        @test readdir(root) == ["plan"]                    # the temporary folder was renamed into place
        # the folder's permissions are any folder's (the umask's), not a private temporary folder's
        plain = joinpath(mktempdir(), "plain"); mkdir(plain)
        @test stat(dir).mode & 0o777 == stat(plain).mode & 0o777
        rp = read_plan(dir)
        @test rp.hash == h && (rp.N, rp.M, rp.tile_size, rp.overlap) == (p.N, p.M, p.tile_size, p.overlap)
        @test (rp.bands, rp.products, rp.coadd_options, rp.product_eltype, rp.frame) == (p.bands, p.products, p.coadd_options, p.product_eltype, p.frame)
        @test (rp.images, rp.excluded, rp.cells, rp.uncovered) == (p.images, p.excluded, p.cells, p.uncovered) && J._toml(rp.options) == J._toml(p.options)
        # the index's tiles are the plan's without cutouts, frames exact, with each one's shard and hash
        @test [Base.structdiff(t, NamedTuple{(:shard, :hash)}) for t in rp.tiles] == [Base.structdiff(t, NamedTuple{(:cutouts,)}) for t in p.tiles]
        @test [t.shard for t in rp.tiles] == collect(eachindex(p.tiles)) && allunique(t.hash for t in rp.tiles)
        # one job per cell, read from its shards alone: the tiles exactly as planned, and the images they use
        jobs = [read_job(dir, j) for j in eachindex(p.cells)]
        @test [jb.cells for jb in jobs] == [[1], [2], [3]] && all(jb -> jb.plan_hash == h, jobs) && isempty(jobs[3].tiles)
        @test sort(vcat([collect(keys(jb.tiles)) for jb in jobs]...)) == collect(eachindex(p.tiles))
        @test all(jb -> all(k -> Base.structdiff(jb.tiles[k], NamedTuple{(:hash,)}) == p.tiles[k] && jb.tiles[k].hash == rp.tiles[k].hash &&
                                 p.tiles[k].cell == only(jb.cells), keys(jb.tiles)), jobs)
        @test all(jb -> all(e -> jb.images[e] == p.images[e], keys(jb.images)), jobs) && !isempty(jobs[1].images)
        @test J._toml(jobs[3].options) == J._toml(p.options) && jobs[3].product_eltype === Float64 && jobs[3].products == p.products
        @test_throws "has jobs 1 to 3, not 4" read_job(dir, 4)
        @test_throws "exists: a plan folder is written once" write_plan(dir, p)
        # what read_plan or read_job returns can't be written back: they don't keep every tile's cutouts
        @test_throws "write_plan needs plan_tiles' result" write_plan(joinpath(mktempdir(), "again"), rp)
        @test_throws "write_plan needs plan_tiles' result" write_plan(joinpath(mktempdir(), "again"), jobs[1])
        # a path with a trailing separator is the same folder, its temporary folder beside it and gone
        slashed = mktempdir()
        write_plan(joinpath(slashed, "plan") * "/", p; shard_size = 1)          # as `dir` was: the shards are in the hash
        @test readdir(slashed) == ["plan"] && read_plan(joinpath(slashed, "plan")).hash == h
        # any edit to a file is refused by its hash: the index's plan table or a tile's core, a job's tiles, a shard's
        # cutout, its image index (img1 and img2 have the same size and band), or its plan table
        copied(name) = (d = joinpath(root, name); cp(dir, d); d)
        edit!(file, pat) = write(file, replace(read(file, String), pat; count = 1))
        shard1(d) = joinpath(d, "tiles", "shard_1.toml")
        for (name, file, pat, reader, msg) in (("index_plan", "plan.toml", "overlap = 80" => "overlap = 81", read_plan, "plan.toml doesn't match its hash"),
                                               ("index_core", "plan.toml", r"core = \[\[\d+" => "core = [[41", read_plan, "plan.toml doesn't match its hash"),
                                               ("job_tiles", "jobs.toml", r"tiles = \[\d+\]" => "tiles = [2]", d -> read_job(d, 1), "jobs.toml doesn't match its hash"),
                                               ("cutout_rows", shard1, r"rows = \[\d+" => "rows = [2", d -> read_job(d, 1), "shard_1.toml doesn't match its hash"),
                                               ("cutout_image", shard1, r"^(\s*)image = 1$"m => s"\1image = 2", d -> read_job(d, 1), "shard_1.toml doesn't match its hash"),
                                               ("shard_plan", shard1, "max_iter = 3" => "max_iter = 4", d -> read_job(d, 1), "shard_1.toml doesn't match its hash"))
            bad = copied(name); edit!(file isa String ? joinpath(bad, file) : file(bad), pat)
            @test_throws msg reader(bad)
        end
        # a shard rewritten consistently (its hash recomputed) is still refused: the tile's identity hash, which takes
        # each cutout's image through the image table, or the plan table against the job file's
        rehash!(file) = (d = J.TOML.parsefile(file); d["hash"] = J._content_hash(d); open(io -> J.TOML.print(io, d; sorted = true), file, "w"))
        for (name, pat, msg) in (("rehashed_rows", r"rows = \[\d+" => "rows = [2", "tile 1 doesn't match its hash"),
                                 ("rehashed_image", r"^(\s*)image = 1$"m => s"\1image = 2", "tile 1 doesn't match its hash"),
                                 ("rehashed_plan", "max_iter = 3" => "max_iter = 4", "plan table differs from"))
            bad = copied(name); edit!(shard1(bad), pat); rehash!(shard1(bad))
            @test_throws msg read_job(bad, 1)
        end
        @test any(c -> c.image == 1, p.tiles[1].cutouts) && p.images[1].dims == p.images[2].dims && p.images[1].band == p.images[2].band
        bad = copied("other_version"); edit!(joinpath(bad, "plan.toml"), r"^version = 1$"m => "version = 2")
        @test_throws "has plan version 2; this CrowdPhot reads version 1" read_plan(bad)
        other = joinpath(root, "other_plan")
        write_plan(other, plan_tiles(fps, f.frame; cores, overlap = 80, options = JointLoopOptions(max_iter = 2)); shard_size = 1)
        bad = copied("foreign_shard"); cp(joinpath(other, "tiles", "shard_1.toml"), joinpath(bad, "tiles", "shard_1.toml"); force = true)
        @test_throws "belongs to plan" read_job(bad, 1)
    end

    @testset "a job's tiles through fit_tile and merge_tiles" begin
        p = plan_tiles(f.imgs, f.frame; tile_size = 330, options = o3)
        R = fit_joint_tiled(f.imgs, f.frame; tile_size = 330, options = o3)
        dir = joinpath(mktempdir(), "even")
        write_plan(dir, p; shard_size = 3)
        jb = read_job(dir, 1)
        @test sort(collect(keys(jb.tiles))) == 1:4 && sort(unique(rp_t.shard for rp_t in read_plan(dir).tiles)) == [1, 2]
        # a unit loads each image its cutouts use by name (here the JointImages, named by their index)
        parents = Dict(im.name => f.imgs[parse(Int, im.name)] for im in values(jb.images))
        res = [fit_tile(jb, k, parents) for k in sort(collect(keys(jb.tiles)))]
        # a job lists only its tiles' images, so a vector can't be checked against the plan's list: refused,
        # even the full list (a filtered one would shift every later index onto the wrong image, silently)
        @test_throws "pass them as a dictionary keyed by name" fit_tile(jb, first(keys(jb.tiles)), f.imgs)
        # names are matched exactly: the images of a dictionary keyed by other names are missing
        @test_throws "which `images` lacks" fit_tile(jb, first(keys(jb.tiles)), Dict("x" * n => im for (n, im) in parents))
        m = merge_tiles(read_plan(dir), 1, res)
        @test isequal(m.catalog, R.catalog) && m.n_dedup == [t.n_dedup for t in R.tiles] && isempty(m.absent)
        @test all(r -> r.hash == read_plan(dir).tiles[r.tile].hash, res)
        # results and plan must agree: a plan folder's results with read_plan's plan, in-memory ones with theirs
        @test_throws "was fitted from a plan folder: merge it with read_plan's plan" merge_tiles(p, 1, res)
        r0 = fit_tile(p, 1, f.imgs)
        @test r0.hash === nothing
        @test_throws "has no hash, but the plan was read from a plan folder" merge_tiles(read_plan(dir), 1, [r0])
        # a result saved without the field at all gets the same message
        @test_throws "has no hash, but the plan was read from a plan folder" merge_tiles(read_plan(dir), 1, [Base.structdiff(res[1], NamedTuple{(:hash,)})])
    end

    @testset "results found by their hash after a re-plan renumbers the tiles" begin
        root = mktempdir()
        top = (1:(ny ÷ 2), 1:nx); bottom = ((ny ÷ 2 + 1):ny, 1:nx)
        # the same two cells, in the other order: the top cell is tile 1 in one plan, tile 2 in the other
        d1 = joinpath(root, "top_first"); d2 = joinpath(root, "top_second")
        write_plan(d1, plan_tiles(fps, f.frame; cores = [top, bottom], overlap = 80, options = o3, products = (:nu,)))
        write_plan(d2, plan_tiles(fps, f.frame; cores = [bottom, top], overlap = 80, options = o3, products = (:nu,)))
        p1 = read_plan(d1); p2 = read_plan(d2)
        @test p1.tiles[1].hash == p2.tiles[2].hash && p1.tiles[1].core == p2.tiles[2].core
        fitjob(d, k) = (jb = read_job(d, k); fit_tile(jb, k, Dict(im.name => f.imgs[e] for (e, im) in jb.images)))
        old = fitjob(d1, 1); fresh = fitjob(d2, 2)
        @test old.tile == 1 && all(==(1), old.catalog.tile)
        # the old result merges into the new plan as its tile 2, its rows renumbered and its products in place
        m = merge_tiles(p2, 2, [old]); mf = merge_tiles(p2, 2, [fresh])
        @test isequal(m.catalog, mf.catalog) && all(==(2), m.catalog.tile) && isequal(m.products.nu, mf.products.nu) && isempty(m.absent)
        # a hash no tile of the cell has (the old top result in the new bottom cell), and two results for one tile
        @test_throws "no tile of cell 1 has the hash of the results fitted as tiles [1]" merge_tiles(p2, 1, [old])
        @test_throws "tiles [2] have more than one result" merge_tiles(p2, 2, [fresh, old])
    end

    @testset "re-planning without an image keeps the hashes of the tiles it doesn't touch" begin
        specs = [(0.0, [40.0, 40.0], :A, 2.0), (0.0, [40.0, 520.0], :B, 2.6), (5.0, [45.0, 515.0], :B, 2.6), (0.0, [40.0, 240.0], :A, 2.0)]
        g = tiling_field(specs; seed = 11)
        gfps = [ImageFootprint(im, "img$e") for (e, im) in enumerate(g.imgs)]
        pa = plan_tiles(gfps, g.frame; tile_size = 380, overlap = 80)
        pb = plan_tiles(gfps, g.frame; tile_size = 380, overlap = 80, exclude = ["img2" => "a test"])
        @test !any(c -> c.name == "img2", pa.tiles[1].cutouts) && any(c -> c.name == "img2", pa.tiles[3].cutouts)
        root = mktempdir()
        da = joinpath(root, "with_img2"); db = joinpath(root, "without_img2")
        write_plan(da, pa); write_plan(db, pb)
        ra = read_plan(da); rb = read_plan(db)
        # the left tile, which never used img2, is unchanged with the same hash; the right one changes
        @test pa.tiles[1] == pb.tiles[1] && pa.tiles[3] != pb.tiles[3]
        @test ra.tiles[1].hash == rb.tiles[1].hash && ra.tiles[3].hash != rb.tiles[3].hash && ra.hash != rb.hash
        # the identity hash ignores image indices: img2 removed from the list itself shifts img4 to index 3, and
        # the left tile keeps its hash (the bands, and so the plan table, are unchanged: img3 is still the first B)
        pc = plan_tiles(gfps[[1, 3, 4]], g.frame; tile_size = 380, overlap = 80)
        @test pc.bands == pa.bands && only(c.image for c in pc.tiles[1].cutouts if c.name == "img4") == 3
        @test only(c.image for c in pa.tiles[1].cutouts if c.name == "img4") == 4
        dc = joinpath(root, "list_without_img2"); write_plan(dc, pc)
        @test read_plan(dc).tiles[1].hash == ra.tiles[1].hash
    end
end
