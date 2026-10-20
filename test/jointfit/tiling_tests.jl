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
    rows(k, u, v, depth) = (n = length(u); (; u, v, ra = u, dec = v, u_err = fill(0.1, n), v_err = fill(0.1, n), flux_A = ones(n), flux_err_A = fill(0.1, n),
                                           significance = fill(9.0, n), significance_A = fill(9.0, n), nimg = fill(2, n), pass_number = fill(1, n),
                                           tile = fill(k, n), depth))
    # the second tile's first source duplicates the first tile's second one (within sep), shallower in its own tile
    cats = [rows(1, [10.0, 50.0], [10.0, 50.0], [5.0, 3.0]), rows(2, [50.1, 90.0], [50.0, 90.0], [1.0, 7.0])]
    cat, n_dedup = J._merge_catalogs(cats, [:A], 2, 0.5)
    @test length(cat) == 3 && n_dedup == [0, 1] && cat.tile == [1, 1, 2]
    @test eltype(cat.u) == Float64 && eltype(cat.nimg) == Int && eltype(cat.tile) == Int
    # no tile with rows: empty columns of the same types
    cat0, n0 = J._merge_catalogs([], [:A], 2, 0.5)
    @test isempty(cat0) && n0 == [0, 0] && eltype(cat0.u) == Float64 && eltype(cat0.pass_number) == Int
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

    @testset "one tile reproduces the untiled run" begin
        R1 = fit_joint_tiled(f.imgs, f.frame; options = o3, products = pr)
        @test length(R1.tiles) == 1 && R1.plan.tiles[1].offset == (0, 0)
        @test all(n -> isequal(getproperty(R1.catalog, n), getproperty(R0.catalog, n)), propertynames(R0.catalog))
        @test propertynames(R1.catalog) == (propertynames(R0.catalog)..., :tile)
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
        # no cross-tile pair closer than min_separation; each row within its tile's core grown by it
        @test !any(i -> any(j -> c.tile[j] != c.tile[i] && hypot(c.u[j] - c.u[i], c.v[j] - c.v[i]) < 1.0, eachindex(c.u)), eachindex(c.u))
        @test all(i -> (t = p.tiles[c.tile[i]]; first(t.core[1]) - 1.5 <= c.u[i] < last(t.core[1]) + 1.5 && first(t.core[2]) - 1.5 <= c.v[i] < last(t.core[2]) + 1.5), eachindex(c.u))
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
    @test propertynames(c)[7:10] == (:flux_A, :flux_B, :flux_err_A, :flux_err_B)
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
