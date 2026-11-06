using CrowdPhot
using CrowdPhot.Joint
using CrowdPhot.PSF: CircularGaussianPRF
using StableRNGs: StableRNG
using Statistics: median
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
        @test_throws "tiles [3] appear more than once" merge_tiles(p, 1, [res[1], res[1]])
        @test_throws "cell 2 is not one of the plan's 1" merge_tiles(p, 2, res)
        # `nothing` (an empty tile) is skipped, and tiles given no result are absent: their cores stay empty but
        # for the margin `min_separation` their neighbors report, though tile 3's ring reaches into core 1
        m2 = merge_tiles(p, 1, [res[1], nothing, res[3]])           # tiles 3 and 4
        @test m2.absent == [1, 2] && all(in((3, 4)), m2.catalog.tile)
        indepth(k, c) = [J._depth(p.tiles[k].core, c.u[i], c.v[i]) for i in eachindex(c.u)]
        @test count(>(5), indepth(1, res[1].catalog)) > 10
        @test all(k -> all(<=(o3.min_separation), indepth(k, m2.catalog)), (1, 2))
        @test isequal(m2.catalog, merge_tiles(p, 1, res[[1, 3]]).catalog)
        # the parents may be just the tile's, keyed by their index in the plan (as a worker loads them by name)
        parents = Dict(c.image => f.imgs[c.image] for c in p.tiles[3].cutouts)
        r3 = fit_tile(p, 3, parents)
        @test isequal(r3.catalog, res[1].catalog) && r3.images == res[1].images == [c.image for c in p.tiles[3].cutouts] && r3.t_cutout > 0
        missing_one = Dict(c.image => f.imgs[c.image] for c in p.tiles[3].cutouts[2:end])
        @test_throws "tile 3 needs image $(p.tiles[3].cutouts[1].image)" fit_tile(p, 3, missing_one)
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
