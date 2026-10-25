using CrowdPhot
using CrowdPhot.Joint
using CrowdPhot.PSF: CircularGaussianPRF, ImagePSF, GriddedPSFModel, evaluate, evaluate_fg
using StableRNGs: StableRNG
using LinearAlgebra: norm, diag
using Test

const J = CrowdPhot.Joint

"Image whose pixels map affinely onto `frame`: (u, v) = A(ang) (y, x) + t."
function affine_sky(frame, ang, t)
    A = [cosd(ang) -sind(ang); sind(ang) cosd(ang)]
    return (y, x) -> (uv = A * [y, x] .+ t; grid_to_world(frame, uv[1], uv[2]))
end

"Oversampled ePSF node from a wide Gaussian PRF (sums to the oversampled area, as `ImagePSF` expects)."
gauss_node(fwhm; os = 4, n = 41) = (c = (n + 1) / 2;
    ImagePSF([evaluate(CircularGaussianPRF(y = c, x = c, fwhm = fwhm * os, flux = os^2, bkg = 0), i, j) for i in 1:n, j in 1:n]; oversampling = os))

function two_image_scene(; gridded = false, model_rad = :auto, os = 4)
    frame = TanGrid(80.0, -69.0, 0.11, 120, 120)
    dims = ((70, 64), (60, 72))
    psfs = gridded ?
        [GriddedPSFModel([gauss_node(f; os), gauss_node(f + 0.2; os), gauss_node(f + 0.1; os), gauss_node(f + 0.4; os)], [0.0, 0.0, 80.0, 80.0], [0.0, 80.0, 0.0, 80.0]) for f in (2.0, 2.6)] :
        [CircularGaussianPRF(y = 0, x = 0, fwhm = f, flux = 1, bkg = 0) for f in (2.0, 2.6)]
    imgs = [JointImage(zeros(dims[1]), ones(dims[1]), psfs[1], affine_sky(frame, 0.0, [25.0, 28.0]); band = :A, flux_scale = 1.0),
            JointImage(zeros(dims[2]), ones(dims[2]), psfs[2], affine_sky(frame, 30.0, [30.0, 20.0]); band = :B, flux_scale = 2.0, psf_offset = (5, -3))]
    sc = JointScene(imgs, frame)
    rng = StableRNG(5); N = 25
    θ = vcat(rand(rng, 2, N) .* 50 .+ 35, exp.(randn(rng, 2, N) .+ 6))
    rt = J.RadiusTable(sc, psfs, [im.weight for im in imgs]; model_rad, R_fit = 3, R_cap = 30)
    return (; sc, imgs, psfs, θ, rt, frame)
end

@testset "TanGrid" begin
    g = TanGrid(80.0, -69.0, 0.11, 100, 120)
    ra, dec = grid_to_world(g, 17.3, 88.9)
    @test all(world_to_grid(g, ra, dec) .≈ (17.3, 88.9))
    # The oversampled grid covers the same sky: pixel edges and the tangent point line up.
    g2 = J.oversample(g, 4)
    @test all(world_to_grid(g2, grid_to_world(g, 0.5, 0.5)...) .≈ (0.5, 0.5))
    @test all(world_to_grid(g2, grid_to_world(g, 17.3, 88.9)...) .≈ (4 * (17.3 - 0.5) + 0.5, 4 * (88.9 - 0.5) + 0.5))
    # 90 deg or more from the tangent point the projection is undefined (it diverges, then mirrors)
    @test_throws "at least 90 deg from the TanGrid tangent point" world_to_grid(g, 80.0, 21.5)     # 90.5 deg away
    @test_throws "at least 90 deg from the TanGrid tangent point" world_to_grid(g, 260.0, 10.0)   # ~121 deg, over the pole
    @test all(isfinite, world_to_grid(g, 80.0, 20.5))                 # 89.5 deg away: still defined
end

@testset "PolyMap" begin
    frame = TanGrid(80.0, -69.0, 0.11, 200, 200)
    sky = affine_sky(frame, 25.0, [40.0, 60.0])
    pm = PolyMap(sky, (100, 90), frame)
    @test pm.maxerr < 1e-6
    # Reproduces the affine inverse and its Jacobian.
    A = [cosd(25.0) -sind(25.0); sind(25.0) cosd(25.0)]
    y, x, Jm = pm(77.0, 91.0)
    @test [y, x] ≈ A \ ([77.0, 91.0] .- [40.0, 60.0]) atol = 1e-6
    @test Jm ≈ inv(A) atol = 1e-6
    # A degree-7 distortion: degree 5 is not enough, so the fit escalates once (with a warning).
    seventh = (y, x) -> sky(y + 2 * ((x - 45) / 60)^7, x)
    pm7 = @test_logs (:warn, r"retrying at degree 7") PolyMap(seventh, (100, 90), frame)
    @test pm7.degree == 7
    @test pm7.maxerr < 1e-4
    # A scene mixing the two degrees evaluates every image's own map on the device.
    psf = CircularGaussianPRF(y = 0, x = 0, fwhm = 2.0, flux = 1, bkg = 0)
    imgs = [JointImage(zeros(100, 90), ones(100, 90), psf, s; band = :A) for s in (sky, seventh)]
    sc = @test_logs (:warn, r"retrying at degree 7") match_mode = :any JointScene(imgs, frame)
    @test sc.dev.degree == 7
    for e in 1:2, (u, v) in ((77.0, 91.0), (60.0, 130.0))
        _, y, x = J.source_pixel(sc.dev, e, u, v)
        yh, xh, _ = sc.maps[e](u, v)
        @test y ≈ yh atol = 1e-9
        @test x ≈ xh atol = 1e-9
    end
    # A distortion no polynomial up to max_degree follows is refused, not silently approximated.
    wavy = (y, x) -> sky(y + 0.5 * sin(x / 7), x)
    @test_throws "up to max_degree" PolyMap(wavy, (100, 90), frame; degree = 9)
    # A celestial pole inside the image: the map is fit in tangent-plane coordinates, so RA's
    # wrap and its undefined value at the pole do not matter.  The image's own TAN WCS is tangent
    # 3.6" from the north pole and rotated 37 deg; frames tangent at, near and 6' from the pole.
    img = TanGrid(123.0, 89.999, 0.11, 300, 300)
    A = [cosd(37.0) -sind(37.0); sind(37.0) cosd(37.0)]
    polar = (y, x) -> (p = A * ([y, x] .- 150.5) .+ 150.5; grid_to_world(img, p[1], p[2]))
    pix = map((TanGrid(0.0, 90.0, 0.11, 600, 600), TanGrid(10.0, 89.995, 0.11, 600, 600), TanGrid(200.0, 89.9, 0.11, 600, 600))) do fr
        pm = PolyMap(polar, (300, 300), fr)
        @test pm.degree == 5 && pm.maxerr < 1e-4
        @test all(all(world_to_grid(fr, ra, 90.0) .≈ world_to_grid(fr, 0.0, 90.0)) for ra in 15.0:30.0:345.0)
        yx = pm(world_to_grid(fr, 0.0, 90.0)...)[1:2]
        @test all(1 .<= yx .<= 300)                                     # the pole is inside the image
        yx
    end
    # every frame puts the pole at the same image pixel, where the image's own WCS has it
    @test all(p -> all(p .≈ pix[1]), pix)
    @test polar(pix[1]...)[2] ≈ 90.0 atol = 1e-8
end

@testset "device PSF evaluation matches PSF.evaluate_fg" begin
    for gridded in (false, true)
        s = two_image_scene(; gridded)
        d = s.sc.dev
        for e in 1:2, (Y, X) in ((10.3, 20.7), (35.5, 31.25), (-2.2, 50.1))
            host = s.psfs[e]
            ctx = J.psf_context(d.psf, e, Y, X)
            off = s.imgs[e].psf_offset
            m = gridded ? CrowdPhot.ConstructionBase.setproperties(host, (; y = Y + off[1], x = X + off[2], flux = 1.0, bkg = 0.0)) :
                CircularGaussianPRF(y = Y, x = X, fwhm = host.fwhm, flux = 1, bkg = 0)
            for i in round(Int, Y) .+ (-3:3), j in round(Int, X) .+ (-3:3)
                # gridded host models are evaluated in the PSF frame (pixel + offset)
                hi, hj = gridded ? (i + off[1], j + off[2]) : (i, j)
                v, gr = evaluate_fg(m, hi, hj)
                dv, dY, dX = J.psf_value_grad(d.psf, ctx, i, j, Y, X)
                @test dv ≈ v atol = 1e-14
                @test J.psf_value(d.psf, ctx, i, j, Y, X) ≈ v atol = 1e-14
                @test dY ≈ gr[1] atol = 1e-12
                @test dX ≈ gr[2] atol = 1e-12
            end
        end
    end
end

@testset "device PSF nodes are shared between identical models" begin
    g(f) = GriddedPSFModel([gauss_node(f), gauss_node(f + 0.2), gauss_node(f + 0.1), gauss_node(f + 0.4)], [0.0, 0.0, 80.0, 80.0], [0.0, 80.0, 0.0, 80.0])
    A = g(2.0); psfs = [A, deepcopy(A), g(2.6), A]; offs = [(0.0, 0.0), (5.0, -3.0), (0.0, 0.0), (-4.0, 7.0)]
    d = J.device_psf(J.KA.CPU(), Float64, psfs, offs)
    @test size(d.nodes, 3) == 8                        # A once (shared by images 1, 2, 4), the 2.6 model once
    @test d.ig[:, :, 1] == d.ig[:, :, 2] == d.ig[:, :, 4] && d.ig[:, :, 3] != d.ig[:, :, 1]
    # every image still evaluates its own host model, at its own offset
    for e in eachindex(psfs), (Y, X) in ((10.3, 20.7), (35.5, 31.25))
        ctx = J.psf_context(d, e, Y, X)
        m = CrowdPhot.ConstructionBase.setproperties(psfs[e], (; y = Y + offs[e][1], x = X + offs[e][2], flux = 1.0, bkg = 0.0))
        for i in round(Int, Y) .+ (-2:2), j in round(Int, X) .+ (-2:2)
            @test J.psf_value(d, ctx, i, j, Y, X) ≈ evaluate_fg(m, i + offs[e][1], j + offs[e][2])[1] atol = 1e-14
        end
    end
end

@testset "render" begin
    for gridded in (false, true)
        s = two_image_scene(; gridded, model_rad = 6)
        fp = J.footprint(s.sc, s.θ, s.rt)
        model = zeros(length(s.sc.data))
        J.render!(model, s.sc, s.θ, fp)
        for e in 1:2
            ref = zeros(s.sc.dims[e]); off = s.imgs[e].psf_offset
            for k in axes(s.θ, 2)
                y, x, _ = s.sc.maps[e](s.θ[1, k], s.θ[2, k])
                a, b = round(Int, y), round(Int, x)
                f = s.imgs[e].flux_scale * s.θ[2 + e, k]
                m = gridded ? CrowdPhot.ConstructionBase.setproperties(s.psfs[e], (; y = y + off[1], x = x + off[2], flux = f, bkg = 0.0)) :
                    CircularGaussianPRF(y = y, x = x, fwhm = s.psfs[e].fwhm, flux = f, bkg = 0)
                for j in max(1, b - 6):min(s.sc.dims[e][2], b + 6), i in max(1, a - 6):min(s.sc.dims[e][1], a + 6)
                    ref[i, j] += gridded ? evaluate(m, i + off[1], j + off[2]) : evaluate(m, i, j)
                end
            end
            @test J.image_view(s.sc, model, e) ≈ ref rtol = 1e-8
        end
        # the GPU render adds each pixel's sources in the CPU's order: bitwise equal, except that the
        # CPU's constant-weight filter for gridded PSFs rounds differently
        tiles = J.render_tiles!(zeros(length(s.sc.data)), J.RenderOp(s.θ, s.sc.dev), s.sc, fp)
        @test gridded ? isapprox(tiles, model; rtol = 1e-12) : tiles == model
    end
    # a crowded field: tiles with more sources than one batch (TILE_G)
    s = two_image_scene(; gridded = true, model_rad = 8)
    rng = StableRNG(11); θ = vcat(rand(rng, 2, 3000) .* 30 .+ 45, exp.(randn(rng, 2, 3000) .+ 4))
    fp = J.footprint(s.sc, θ, s.rt)
    @test J.tile_bins(s.sc, fp).nb > 1
    model = zeros(length(s.sc.data)); J.render!(model, s.sc, θ, fp)
    tiles = J.render_tiles!(zeros(length(s.sc.data)), J.RenderOp(θ, s.sc.dev), s.sc, fp)
    @test tiles ≈ model rtol = 1e-12
    # the stamp buffer filled band by band, one tile column per band: boxes cross every band edge
    @test J.render_tiles!(zeros(length(s.sc.data)), J.RenderOp(θ, s.sc.dev), s.sc, fp; words = 1) == tiles
    # sources over and beyond the whole frame: boxes clipped at the image edges or off the images
    θe = vcat(rand(rng, 2, 400) .* 130 .- 5, exp.(randn(rng, 2, 400) .+ 4))
    fpe = J.footprint(s.sc, θe, s.rt); me = zeros(length(s.sc.data)); J.render!(me, s.sc, θe, fpe)
    te = J.render_tiles!(zeros(length(s.sc.data)), J.RenderOp(θe, s.sc.dev), s.sc, fpe)
    @test te ≈ me rtol = 1e-12
    @test J.render_tiles!(zeros(length(s.sc.data)), J.RenderOp(θe, s.sc.dev), s.sc, fpe; words = 1) == te
    # the CPU's constant-weight filter at any integer oversampling, boxes within and beyond the nodes
    # (41 nodes: 10 pixels at os = 4), against the per-pixel evaluation
    for os in 1:5, model_rad in (3, 25)
        s = two_image_scene(; gridded = true, model_rad, os)
        fp = J.footprint(s.sc, θe, s.rt); m = zeros(length(s.sc.data)); J.render!(m, s.sc, θe, fp)
        @test m ≈ J.render_tiles!(zeros(length(s.sc.data)), J.RenderOp(θe, s.sc.dev), s.sc, fp) rtol = 1e-12
        # every row chunking (thread count) gives the same bits
        @test all(n -> J.render_rows_cw!(zeros(length(s.sc.data)), J.RenderOp(θe, s.sc.dev), s.sc, fp, n) == m, (1, 7, 64))
    end
end

# Dense equilibration-undone H from a Linearization (host arrays).
function dense_H(lin)
    P, N = size(lin.g)
    Hd = Array(lin.Hd); Ho = Array(lin.Ho); D = vec(Array(lin.D)); ptr = Array(lin.pat.ptr); nbr = Array(lin.pat.nbr)
    H = zeros(P * N, P * N)
    blk(s) = (s - 1) * P + 1:s * P
    for s in 1:N; H[blk(s), blk(s)] .= Hd[:, :, s]; end
    for i in 1:N, q in ptr[i]:(ptr[i + 1] - 1)
        H[blk(i), blk(nbr[q])] .= Ho[:, :, q]; H[blk(nbr[q]), blk(i)] .= Ho[:, :, q]'
    end
    return D .* H .* D'
end

@testset "linearization matches finite differences" begin
    # model_rad = R_fit: the Jacobian covers the whole rendered box, so J' J is the exact H.
    s = two_image_scene(; gridded = true, model_rad = 3)
    rng = StableRNG(9)
    s.sc.data .= randn(rng, length(s.sc.data)) .* 5
    work = J.LinearizeWork(s.sc)
    lin = J.linearize(s.sc, s.θ, s.rt, work; R_fit = 3)
    sw = sqrt.(s.sc.weight) .* work.mask
    wmodel(θ) = (m = zeros(length(s.sc.data)); J.render!(m, s.sc, θ, lin.fp); sw .* m)
    r0 = wmodel(s.θ) .- sw .* s.sc.data
    @test lin.cost ≈ sum(abs2, r0)
    Jd = zeros(length(s.sc.data), length(s.θ))
    for k in eachindex(s.θ)
        h = mod1(k, 4) <= 2 ? 1e-5 : 1e-3
        tp = copy(s.θ); tp[k] += h; tm = copy(s.θ); tm[k] -= h
        Jd[:, k] .= (wmodel(tp) .- wmodel(tm)) ./ 2h
    end
    @test vec(Array(lin.g)) ≈ Jd' * r0 rtol = 1e-7
    @test dense_H(lin) ≈ Jd' * Jd rtol = 1e-7
end

@testset "noiseless convergence" begin
    s = two_image_scene(; model_rad = 3)     # fixed radii: the data are rendered with the fit's own boxes
    rng = StableRNG(2)
    θt = s.θ .+ vcat(0.3 .* randn(rng, 2, size(s.θ, 2)), 0.2 .* s.θ[3:4, :] .* randn(rng, 2, size(s.θ, 2)))
    rt = s.rt
    J.render!(s.sc.data, s.sc, θt, J.footprint(s.sc, θt, rt))
    o = J.JointFitOptions()
    θ = s.θ; λ = o.λ_init; work = J.LinearizeWork(s.sc); r = nothing
    for _ in 1:15
        r = J.fit_pass(s.sc, θ, rt, o, λ; work)
        θ = r.θ; λ = r.λ
    end
    @test maximum(abs, θ[1:2, :] .- θt[1:2, :]) < 1e-8
    obs = Array(r.lin.hdiag)[3:4, :] .> 0            # bands a source has data in
    @test maximum(abs.(θ[3:4, :] ./ θt[3:4, :] .- 1)[obs]) < 1e-8
end

@testset "one image reproduces the single-image fitter" begin
    rng = StableRNG(11); ny, nx = 90, 100; N = 40
    frame = TanGrid(80.0, -69.0, 0.11, ny, nx)
    psf = CircularGaussianPRF(y = 0, x = 0, fwhm = 2.4, flux = 1, bkg = 0)
    yt = rand(rng, N) .* (ny - 10) .+ 5; xt = rand(rng, N) .* (nx - 10) .+ 5; ft = exp.(randn(rng, N) .+ 6.5)
    img = simulate_image((ny, nx), psf, (; y = yt, x = xt, flux = ft); background = 0.0, noise = :none, model_radius = 15)
    img .+= 3 .* randn(rng, ny, nx); w = fill(1 / 9, ny, nx)
    y0 = yt .+ 0.3 .* randn(rng, N); x0 = xt .+ 0.3 .* randn(rng, N); f0 = ft .* (1 .+ 0.2 .* randn(rng, N))
    fitter = CrowdPhot.SimultaneousFitter{Float64}(:lsqr, 1, 10, 1e-12, 8, 1e-3, 10.0, 10.0, 1e-12, 1e12, 8)
    plan = CrowdPhot.FitPlan(psf, (; bkg = 0.0, fwhm = 2.4))
    cat = CrowdPhot.Catalog{Float64}((; y = y0, x = x0, flux = f0), psf)
    o = (; R_fit = 3, R_cap = 30, model_rad = :auto, model_rad_nsigma = 1.0, max_step = 1.0)
    λ = 1e-3
    for pass in 1:4
        r = CrowdPhot.fit_pass(fitter, vec(img), vec(w), CrowdPhot.stamp_geometry(cat, vec(w), 3, ny, nx), cat, psf, plan, o, λ; ny, nx, pass)
        cat = r.catalog; λ = r.state
    end
    # identity sky map: frame position (u, v) is pixel (y, x)
    sc = JointScene([JointImage(img, w, psf, (y, x) -> grid_to_world(frame, y, x); band = :F)], frame)
    θ = permutedims(hcat(y0, x0, f0)); rt = J.RadiusTable(sc, [psf], [w]; R_fit = 3, R_cap = 30)
    oj = J.JointFitOptions(); λj = oj.λ_init; work = J.LinearizeWork(sc)
    for _ in 1:4
        r = J.fit_pass(sc, θ, rt, oj, λj; work); θ = r.θ; λj = r.λ
    end
    # the single-image catalog is spatially sorted; match by position
    k = [argmin((cat.y .- θ[1, s]) .^ 2 .+ (cat.x .- θ[2, s]) .^ 2) for s in 1:N]
    @test sort(k) == 1:N
    # LSQR there, CG here: equal in exact arithmetic
    @test maximum(hypot.(θ[1, :] .- cat.y[k], θ[2, :] .- cat.x[k])) < 1e-6
    @test maximum(abs, θ[3, :] ./ cat.flux[k] .- 1) < 1e-6
end

@testset "cost! matches the direct sum" begin
    s = two_image_scene()
    rng = StableRNG(9); n = length(s.sc.data)
    model = randn(rng, n); mask = rand(rng, Bool, n); copyto!(s.sc.data, randn(rng, n)); s.sc.weight .= rand(rng, n)
    buf = zeros(n)
    c = J.cost!(buf, s.sc, model, mask, J.cost_parts(J.KA.CPU(), Float64))
    ref = ifelse.(mask, sqrt.(s.sc.weight) .* (model .- s.sc.data), 0.0)
    @test buf == ref
    @test c ≈ sum(abs2, ref) rtol = 1e-12
    # with a background the fit data is data - bkg
    bkg = randn(rng, n)
    c = J.cost!(buf, s.sc, model, mask, J.cost_parts(J.KA.CPU(), Float64), bkg)
    ref = ifelse.(mask, sqrt.(s.sc.weight) .* (model .- (s.sc.data .- bkg)), 0.0)
    @test buf == ref
    @test c ≈ sum(abs2, ref) rtol = 1e-12
end

@testset "marginal_errors matches a dense reference" begin
    s = two_image_scene()
    h = J.HostH(J.linearize(s.sc, s.θ, s.rt, J.LinearizeWork(s.sc); R_fit = 3))
    P, N = size(h.D)
    act(b) = [p for p in 1:P if h.hdiag[p, b] > 0]
    blk(b, c) = (B = J.hblock(h, b, c); B === nothing ? zeros(length(act(b)), length(act(c))) : B[act(b), act(c)])
    for k in (0, 2, 8)
        σ = J.marginal_errors(h; k)
        for a in 1:N
            A = act(a)
            isempty(A) && continue
            # the k neighbors most strongly coupled over a's parameters, then the inverse of the dense submatrix
            cand = [[(h.nbr[q], sum(abs2, h.Ho[A, :, q])) for q in h.ptr[a]:(h.ptr[a + 1] - 1)];
                    [(h.row[h.cslot[t]], sum(abs2, h.Ho[:, A, h.cslot[t]])) for t in h.cptr[a]:(h.cptr[a + 1] - 1)]]
            nb = first.(sort(filter(c -> c[2] > 0, cand); by = c -> -c[2]))
            srcs = [nb[1:min(k, end)]; a]
            M = reduce(vcat, [reduce(hcat, [blk(b, c) for c in srcs]) for b in srcs])
            t = sum(diag(M)); for i in axes(M, 1); M[i, i] += 1e-12 * t; end
            ref = sqrt.(diag(inv(M))[(end - length(A) + 1):end]) ./ h.D[A, a]
            @test σ[A, a] ≈ ref rtol = 1e-8
        end
    end
    # a Float32 H (a GPU run's) is factored in Float64: a near-duplicate pair makes its blocks nearly
    # singular, where a Float32 factorization gave errors off by up to ~1e4
    sc32 = JointScene(s.imgs, s.frame; T = Float32)
    rt32 = J.RadiusTable(sc32, s.psfs, [im.weight for im in s.imgs]; R_fit = 3, R_cap = 30)
    θ = Float32.(hcat(s.θ, s.θ[:, 1] .+ [0.03, 0.02, 0, 0]))
    h32 = J.HostH(J.linearize(sc32, θ, rt32, J.LinearizeWork(sc32); R_fit = 3))
    h64 = J.HostH(Float64.(h32.Hd), Float64.(h32.Ho), Float64.(h32.D), Float64.(h32.hdiag), h32.ptr, h32.nbr, h32.cptr, h32.cslot, h32.row)
    σ32 = J.marginal_errors(h32; k = 8); σ64 = J.marginal_errors(h64; k = 8)
    k = isfinite.(σ64)
    @test eltype(σ32) == Float64 && isfinite.(σ32) == k
    @test σ32[k] ≈ σ64[k] rtol = 1e-10
end

@testset "CPU kernels do not allocate per pixel" begin
    # A dynamic workgroup size or a type-unstable kernel allocates in the hot loop; launches
    # themselves allocate only per task.
    s = two_image_scene()
    rng = StableRNG(4); N = 1000
    θ = vcat(rand(rng, 2, N) .* 50 .+ 35, exp.(randn(rng, 2, N) .+ 6))
    fp = J.footprint(s.sc, θ, s.rt)
    model = zeros(length(s.sc.data))
    J.render!(model, s.sc, θ, fp)
    @test (@allocated J.render!(model, s.sc, θ, fp)) < 200_000
    work = J.LinearizeWork(s.sc)
    lin = J.linearize(s.sc, θ, s.rt, work; R_fit = 3)
    # dominated by the H blocks it returns
    @test (@allocated J.linearize(s.sc, θ, s.rt, work; R_fit = 3)) < 3 * (sizeof(lin.Ho) + sizeof(lin.Hd)) + 5_000_000
end

@testset "joint detection" begin
    s = two_image_scene()
    rng = StableRNG(21)
    psfs = [im.psf for im in s.imgs]; offs = [im.psf_offset for im in s.imgs]
    det = JointDetector(K = 2)
    jd = J.JointDetection(s.sc, psfs, offs, det)
    # Pure noise with unit inverse variance: ν is N(0, 1) under the null.
    noise = randn(rng, length(s.sc.data))
    m = J.joint_maps(jd, s.sc, noise, s.sc.weight)
    ν = filter(isfinite, Array(m.ν))
    # neighboring nodes are correlated, so loose bounds
    @test abs(sum(ν) / length(ν)) < 0.15
    @test 0.8 < sqrt(sum(abs2, ν) / length(ν)) < 1.2
    # joint_point at grid nodes reproduces the map, and returns the per-band fluxes of a star
    θ1 = [60.0; 55.0; 3000.0; 1500.0;;]
    fp = J.footprint(s.sc, θ1, s.rt)
    data = zeros(length(s.sc.data)); J.render!(data, s.sc, θ1, fp)
    m = J.joint_maps(jd, s.sc, data .+ 0.01 .* noise, s.sc.weight)
    nodes = [(119, 109), (120, 110), (121, 111)]           # grid nodes around the star (K = 2)
    uv = reduce(hcat, [collect(J.node_frame(r, c, 2, Float64)) for (r, c) in nodes])
    jp = J.joint_point(jd, s.sc, data .+ 0.01 .* noise, s.sc.weight, uv)
    @test jp[1, :] ≈ [Array(m.ν)[r, c] for (r, c) in nodes]
    # at the star's exact position the seed fluxes are its band fluxes (noise is negligible)
    jp = J.joint_point(jd, s.sc, data, s.sc.weight, θ1[1:2, :])
    @test jp[4:5, 1] ≈ θ1[3:4, 1] rtol = 0.02
    # the peak of the map sits on the star
    pk = J.find_peaks((; ν = Array(m.ν), snm = Array(m.snm)); floor = 5.0, dedup = 3.0)
    @test length(pk.y) >= 1
    i = argmax(pk.ν)
    @test hypot((pk.y[i] - 0.5) / 2 + 0.5 - 60, (pk.x[i] - 0.5) / 2 + 0.5 - 55) < 0.2
    # the wing template peaks at 1 on the star and the gate keeps an isolated star
    tm = J.wing_template(jd, s.sc, ones(2))
    @test tm.W[tm.h + 1, tm.h + 1] ≈ 1
    @test J.wing_gate(pk, tm)[i]
    # no image covers the frame center: the star moves to the best-covered point (at the
    # center it would see no image, and the peak would be NaN; offsets off these small images are)
    imgs = [JointImage(zeros(40, 40), ones(40, 40), s.psfs[1], affine_sky(s.frame, 0.0, t); band = :A) for t in ([5.0, 5.0], [10.0, 10.0])]
    sc = JointScene(imgs, s.frame)
    tm = J.wing_template(J.JointDetection(sc, [im.psf for im in imgs], [im.psf_offset for im in imgs], det), sc, ones(2))
    @test tm.W[tm.h + 1, tm.h + 1] ≈ 1
    # a NaN template offset predicts nothing instead of gating the fainter peak
    W = fill(NaN, 5, 5); W[3, 3] = 1.0; W[3, 5] = 0.1
    two(y2, x2) = J.wing_gate((; y = [10.0, y2], x = [10.0, x2], ν = [100.0, 15.0]), (; W, h = 2, K = 1); rcore = 0.5)
    @test two(12.0, 10.0) == [true, true]           # offset (2, 0): NaN, kept
    @test two(10.0, 12.0) == [true, false]          # offset (0, 2): 15 < 2 * 100 * 0.1, gated
end

@testset "detection banks are shared between images with identical inputs" begin
    frame = TanGrid(80.0, -69.0, 0.11, 120, 120)
    g(f) = GriddedPSFModel([gauss_node(f), gauss_node(f + 0.2), gauss_node(f + 0.1), gauss_node(f + 0.4)], [0.0, 0.0, 80.0, 80.0], [0.0, 80.0, 0.0, 80.0])
    A = g(2.0); A2 = deepcopy(A); A2.psfs[3].data[20, 21] *= 1 + 1e-6
    @test J._same_content(A, deepcopy(A)) && !J._same_content(A, A2) && !J._same_content(A, g(2.6))
    # unit_psf drops position, flux and background; any shape parameter splits
    gp(; fwhm = 2.0, kw...) = CircularGaussianPRF(; y = 0, x = 0, fwhm, flux = 1, bkg = 0, kw...)
    @test J._same_content(J.unit_psf(gp(y = 5, x = 3, flux = 9, bkg = 1), 0, 0), J.unit_psf(gp(), 0, 0))
    @test !J._same_content(gp(), gp(fwhm = 2.0 + 1e-12))
    # (psf, dims, psf_offset): images 1 and 2 share; 3-6 each differ from 1 in one input
    spec = [(A, (60, 64), (0, 0)), (deepcopy(A), (60, 64), (0, 0)), (A, (60, 64), (5, -3)), (A2, (60, 64), (0, 0)), (g(2.6), (60, 64), (0, 0)), (A, (64, 60), (0, 0))]
    imgs = [JointImage(zeros(d), ones(d), p, affine_sky(frame, 0.0, [30.0, 30.0]); band = :A, flux_scale = 1.0, psf_offset = o) for (p, d, o) in spec]
    nr = 2; det = JointDetector(psf_regions = nr)
    bk = J.DeviceBanks(JointScene(imgs, frame), [im.psf for im in imgs], [im.psf_offset for im in imgs], det)
    @test length(bk.h) == 5 * nr^2
    @test bk.reg[1, 1] == bk.reg[1, 2] && allunique(bk.reg[1, 2:6])
    # every region of every image reads exactly the bank _bank builds for it
    for (e, im) in enumerate(imgs), j in 1:nr, i in 1:nr
        (ny, nx) = size(im.data); sy, sx = cld(ny, nr), cld(nx, nr)
        b = J._bank(im.psf, det, min((i - 0.5) * sy + 0.5, ny) + im.psf_offset[1], min((j - 0.5) * sx + 0.5, nx) + im.psf_offset[2])
        kb = J.bank_index(bk, e, (i - 1) * sy + 1, (j - 1) * sx + 1); r = bk.off[kb] .+ (1:length(b.P))
        @test bk.h[kb] == b.h && bk.P[r] == vec(b.P) && bk.c[r] == vec(b.c) && bk.cχ[r] == vec(b.cχ)
    end
end

# The two-pass form `image_terms` replaced (every pixel of the stamp square, sums about the
# aperture means from a first pass, weights divided per pixel): the reference for the one-pass form.
function reference_image_terms(data, iv, off, ny, nx, y::T, x::T, g::T, bk, kb, level::Bool, chi::Bool, μ::T) where {T}
    z = zero(T)
    py = unsafe_trunc(Int32, round(y)); px = unsafe_trunc(Int32, round(x))
    ((1 <= py <= ny) & (1 <= px <= nx)) || return z, z, z, z, z, z, z, false
    Q = bk.Q; ky = J.phase_index(y - T(py), Q); kx = J.phase_index(x - T(px), Q)
    h = bk.h[kb]; n = 2 * Int(h) + 1
    s0 = J.bank_entry(bk, kb, Int(h) + 1, Int(h) + 1, ky, kx)
    jr = max(Int32(1), px - h):min(Int32(nx), px + h); ir = max(Int32(1), py - h):min(Int32(ny), py + h)
    pbar = ebar = z
    if level
        F = Ep = G = z
        for j in jr, i in ir
            q = off + i + (j - 1) * ny; s = s0 + (i - py) + (j - px) * n
            w = iv[q]; c = T(bk.c[s]) / T(255)
            ((w > 0) & (c > 0)) || continue
            F += c * w; Ep += c * w * bk.P[s]; G += c * w * data[q]
        end
        F > 0 || return z, z, z, z, z, z, z, false
        pbar = Ep / F; ebar = G / F
    end
    N = Dd = V = A = Bx = Cc = dd = z
    for j in jr, i in ir
        q = off + i + (j - 1) * ny; w = iv[q]
        w > 0 || continue
        s = s0 + (i - py) + (j - px) * n; ev = data[q]
        pt = bk.P[s] - pbar; et = ev - ebar; c = T(bk.c[s]) / T(255)
        if c > 0
            N += c * w * pt * et; Dd += c * w * pt * pt; V += c * c * w * pt * pt
        end
        if chi
            cx = T(bk.cχ[s]) / T(255)
            if cx > 0
                wχ = cx / (1 / w + μ * max(ev, z)^2)
                A += wχ * et * et; Bx += wχ * et * pt; Cc += wχ * pt * pt; dd += cx
            end
        end
    end
    Dd > 0 || return z, z, z, z, z, z, z, false
    return g * N, g * g * Dd, g * g * V, A, -2g * Bx, g * g * Cc, dd - ifelse(level, one(T), z), true
end

@testset "image_terms matches the two-pass reference" begin
    s = two_image_scene()
    psfs = [im.psf for im in s.imgs]; offs = [im.psf_offset for im in s.imgs]
    rng = StableRNG(17); n = length(s.sc.data)
    # a 1e4 pedestal, noise, and masked pixels whose data are NaN
    data = 1e4 .+ 30 .* randn(rng, n); iv = rand(rng, n) .+ 0.5
    bad = rand(rng, n) .< 0.05; iv[bad] .= 0; data[bad] .= NaN
    for method in (:dolphot, :matched_filter), level in (true, false)
        jd = J.JointDetection(s.sc, psfs, offs, JointDetector(; method, level))
        bk = jd.banks; chi = J.uses_chi(jd.det); μ = jd.det.μ
        for e in 1:2, _ in 1:200
            ny, nx = s.sc.dims[e]
            y = 1 + (ny - 1) * rand(rng); x = 1 + (nx - 1) * rand(rng)
            kb = J.bank_index(bk, e, round(Int32, y), round(Int32, x))
            got = J.image_terms(data, iv, s.sc.offsets[e], ny, nx, y, x, 2.0, bk, kb, level, chi, μ)
            ref = reference_image_terms(data, iv, s.sc.offsets[e], ny, nx, y, x, 2.0, bk, kb, level, chi, μ)
            @test got[8] == ref[8]
            got[8] && @test all(isapprox(a, b; rtol = 1e-9, atol = 1e-9 * maximum(abs, ref[1:7])) for (a, b) in zip(got[1:7], ref[1:7]))
        end
    end
end

# The serial algorithm `find_peaks` replaced: CrowdPhot.findlocalmaxima, then refinement and dedup.
function serial_peaks(m; floor, dedup, border = 2)
    S = m.snm; ny, nx = size(S)
    Sf = map(v -> isfinite(v) ? Float64(v) : -Inf, S)
    pk = [p for p in CrowdPhot.findlocalmaxima(Sf; edges = false)
          if isfinite(m.ν[p]) && m.ν[p] >= floor && border < p[1] <= ny - border && border < p[2] <= nx - border]
    sort!(pk; by = p -> -Sf[p])
    y = Float64[]; x = Float64[]
    for p in pk
        i, j = Tuple(p); c = Sf[i, j]
        dy = (Sf[i + 1, j] - Sf[i - 1, j]) / (2 * (2c - Sf[i + 1, j] - Sf[i - 1, j]))
        dx = (Sf[i, j + 1] - Sf[i, j - 1]) / (2 * (2c - Sf[i, j + 1] - Sf[i, j - 1]))
        yy = i + clamp(isfinite(dy) ? dy : 0.0, -0.5, 0.5); xx = j + clamp(isfinite(dx) ? dx : 0.0, -0.5, 0.5)
        any(k -> (y[k] - yy)^2 + (x[k] - xx)^2 < dedup^2, eachindex(y)) && continue
        push!(y, yy); push!(x, xx)
    end
    return y, x
end

@testset "find_peaks matches the serial algorithm" begin
    rng = StableRNG(61)
    ν = randn(rng, 120, 90) .* 2
    for _ in 1:40                                    # bright blobs, some touching
        i, j = rand(rng, 3:118), rand(rng, 3:88)
        ν[max(1, i - 1):min(120, i + 1), max(1, j - 1):min(90, j + 1)] .+= 10 .+ 20rand(rng)
    end
    ν[rand(rng, 1:length(ν), 200)] .= NaN            # uncovered nodes
    snm = ν .* 0.8 .+ 0.3 .* randn(rng, size(ν))
    m = (; ν, snm)
    pk = J.find_peaks(m; floor = 5.0, dedup = 3.0)
    y, x = serial_peaks(m; floor = 5.0, dedup = 3.0)
    @test pk.y == y
    @test pk.x == x
    @test all(>=(5), pk.ν)
    @test isempty(J.find_peaks(m; floor = 1e9).y)
end

@testset "device pair pattern" begin
    rng = StableRNG(71); N = 400; h = 2.5
    θ = vcat(rand(rng, 2, N) .* 40, ones(1, N))
    θ[1, 7] = NaN; θ[2, 9] = Inf                      # sources with non-finite positions get no pairs
    pat = J.PairPattern(J.KA.CPU(), θ, h)
    # a frame-wide grid (as linearize uses) gives the same pattern, also with sources outside it
    for ext in ((40, 40), (25, 30))
        pf = J.PairPattern(J.KA.CPU(), θ, h; ext)
        @test pf.ptr == pat.ptr && pf.nbr == pat.nbr && pf.cptr == pat.cptr && pf.cslot == pat.cslot
    end
    ptr = pat.ptr; nbr = pat.nbr
    ref = [(i, j) for i in 1:N for j in (i + 1):N
           if all(isfinite, θ[1:2, i]) && all(isfinite, θ[1:2, j]) && abs(θ[1, j] - θ[1, i]) <= h && abs(θ[2, j] - θ[2, i]) <= h]
    got = [(i, Int(nbr[q])) for i in 1:N for q in ptr[i]:(ptr[i + 1] - 1)]
    @test got == ref                                   # same pairs, rows sorted
    @test pat.np == length(ref) && all(pat.row[q] == i for (q, (i, _)) in enumerate(got))
    # transposed index: column j lists, in increasing order, the slots whose neighbor is j
    for j in 1:N
        slots = pat.cslot[pat.cptr[j]:(pat.cptr[j + 1] - 1)]
        @test issorted(slots) && all(nbr[q] == j for q in slots)
        @test length(slots) == count(p -> p[2] == j, ref)
    end
end

@testset "fit_joint_multipass end to end" begin
    # sparse two-band, two-dither field: stars recovered, errors calibrated
    frame = TanGrid(80.0, -69.0, 0.11, 120, 120)
    rng = StableRNG(31)
    N = 60
    θt = vcat(rand(rng, 2, N) .* 70 .+ 25, exp.(randn(rng, 2, N) .* 0.5 .+ 7.5))
    specs = [(0.0, [22.0, 25.0], :A, 2.0), (17.0, [30.0, 18.0], :A, 2.0), (40.0, [35.0, 24.0], :B, 2.6), (63.0, [27.0, 31.0], :B, 2.6)]
    sky = 30.0
    imgs = map(specs) do (ang, t, band, fwhm)
        psf = CircularGaussianPRF(y = 0, x = 0, fwhm = fwhm, flux = 1, bkg = 0)
        JointImage(zeros(80, 80), fill(1 / sky, 80, 80), psf, affine_sky(frame, ang, t); band)
    end
    sc = JointScene(imgs, frame)
    rt = J.RadiusTable(sc, [im.psf for im in imgs], [im.weight for im in imgs]; R_fit = 3, R_cap = 30)
    model = zeros(length(sc.data)); J.render!(model, sc, θt, J.footprint(sc, θt, rt))
    imgs = map(enumerate(imgs)) do (e, im)
        d = J.image_view(sc, model, e) .+ sky .+ sqrt(sky) .* randn(rng, 80, 80)
        JointImage(d, im.weight, im.psf, im.to_sky; band = im.band)
    end
    R = fit_joint_multipass(imgs, frame)
    c = R.catalog
    @test R.bands == [:A, :B]
    # truth matching within 1 px
    idx = [argmin((θt[1, :] .- c.u[i]) .^ 2 .+ (θt[2, :] .- c.v[i]) .^ 2) for i in eachindex(c.u)]
    d = hypot.(θt[1, idx] .- c.u, θt[2, idx] .- c.v)
    tp = d .< 1
    covered = [any(e -> begin y, x, _ = sc.maps[e](θt[1, s], θt[2, s]); 3 < y < 78 && 3 < x < 78 end, 1:4) for s in 1:N]
    @test count(tp) >= 0.9 * count(covered)
    @test count(.!tp .& (c.significance .>= 5)) <= 1
    k = findall(tp .& isfinite.(c.flux_err_A))
    pull = (c.flux_A[k] .- θt[3, idx[k]]) ./ c.flux_err_A[k]
    # robust width: two truth pairs 0.1-0.3 px apart are fitted as one source with their summed flux
    @test 0.7 < 1.4826 * sort(abs.(pull))[cld(length(pull), 2)] < 1.3
    @test all(>(0), c.nimg)
    @test length(R.history) >= 1
    # the stop rule: from pass 3, at most min_growth x catalog surviving new sources; 0 runs max_iter
    h = R.history
    @test R.converged == (length(h) >= 3 && h[end].n_new_surviving <= 0.01 * h[end].n_catalog)
    @test all(x -> x.pass < 3 || x.n_new_surviving > 0.01 * x.n_catalog, h[1:(end - 1)])
    @test length(fit_joint_multipass(imgs, frame; options = JointLoopOptions(max_iter = 3, min_growth = 0)).history) == 3
    # discard_passes: prune_snr_min = 40 prunes real stars from pass 2 on, whose re-detections the log blocks.
    # 0 turns the log off; with 1 a pass-2 prune blocks pass 3 (the runs agree through pass 3) and expires at pass 4
    ndisc(K) = [get(x, :n_dup_disc, 0) for x in fit_joint_multipass(imgs, frame; options = JointLoopOptions(max_iter = 4, min_growth = 0, prune_snr_min = 40, discard_passes = K)).history]
    dn, d0, d1 = ndisc(nothing), ndisc(0), ndisc(1)
    @test dn[3] > 0 && dn[4] > 0 && all(==(0), d0)
    @test d1[3] == dn[3] && d1[4] < dn[4]
    @test_throws "discard_passes must be nothing or >= 0, got -1" fit_joint_multipass(imgs, frame; options = JointLoopOptions(discard_passes = -1))
    # flat per-band columns in `bands` order; a Tables.jl table with one row per source
    @test propertynames(c) == (:u, :v, :ra, :dec, :u_err, :v_err, :flux_A, :flux_B, :flux_err_A, :flux_err_B, :significance,
                               :significance_A, :significance_B, :nimg, :pass_number)
    @test c[1].flux_B == c.flux_B[1]
    @test all(in(1:length(h)), c.pass_number)
    # memory estimate: per-pixel term plus H (own and pair blocks) and its host copy, P = 4 in Float64
    @test J._memory_estimate(sc, 10, 5) == 87 * length(sc.data) + 2 * 16 * 8 * 15
    @test all(x -> x.mem_est >= 87 * length(sc.data), h)
    # bands b and err_b would both give a flux_err_b column
    clash = [JointImage(im.data, im.weight, im.psf, im.to_sky; band = im.band === :A ? :err_B : :B) for im in imgs]
    @test_throws "bands B and err_B give the same catalog column flux_err_B" fit_joint_multipass(clash, frame)
end

@testset "Float32 scenes agree with Float64" begin
    s = two_image_scene(; gridded = true)
    rng = StableRNG(41)
    data = randn(rng, length(s.sc.data)) .* 5
    lins = map((Float64, Float32)) do T
        sc = JointScene(s.imgs, s.frame; T)
        sc.data .= T.(data)
        rt = J.RadiusTable(sc, s.psfs, [im.weight for im in s.imgs]; R_fit = 3, R_cap = 30)
        J.linearize(sc, T.(s.θ), rt, J.LinearizeWork(sc); R_fit = 3)
    end
    rel(a, b) = maximum(abs, Float64.(a) .- b) / maximum(abs, b)
    @test lins[2].cost ≈ lins[1].cost rtol = 1e-5
    @test rel(lins[2].g, lins[1].g) < 1e-4
    @test rel(lins[2].Hd, lins[1].Hd) < 1e-4
    @test rel(lins[2].Ho, lins[1].Ho) < 1e-4
end

@testset "background2d! matches Background2D" begin
    B = CrowdPhot.Background
    function bkg_image(H, W; seed = 1, nan = false)
        rng = StableRNG(seed)
        img = [100 + 0.02i + 0.01j + 5randn(rng) for i in 1:H, j in 1:W]
        for _ in 1:(H * W ÷ 600)                                  # bright outliers for the clipping
            img[rand(rng, 1:H), rand(rng, 1:W)] += 2000rand(rng)
        end
        nan && (img[rand(rng, 1:H, 30) .+ H .* (rand(rng, 1:W, 30) .- 1)] .= NaN)
        return img
    end
    cases = [(; box = 20), (; box = 20, estimator = B.MedianBackground(), rms_estimator = B.StdRMS()),
             (; box = 20, estimator = B.MMMBackground()), (; box = 20, estimator = B.MeanBackground(), rms_estimator = B.StdRMS(), sigma = (2.0, 4.0), maxiters = 3),
             (; box = 20, sigma = nothing), (; box = 20, filter_size = 1), (; box = 20, filter_size = (5, 3)), (; box = (16, 24), exclude_percentile = 50.0),
             (; box = 7), (; box = 20, nan = true, sparse_mask = true), (; box = 20, sigma = nothing, nan = true, sparse_mask = true)]
    for (k, c) in enumerate(cases), T in (Float64, Float32)
        img = T.(bkg_image(137, 119; seed = k, nan = get(c, :nan, false)))
        mask = get(c, :sparse_mask, false) ? (StableRNG(k) |> r -> rand(r, size(img)...) .< 0.01) : nothing
        kw = merge((; estimator = B.SExtractorBackground(), rms_estimator = B.MADStdRMS()), Base.structdiff(c, NamedTuple{(:box, :nan, :sparse_mask)}))
        ref = get(c, :nan, false) ? (@test_logs (:warn,) B.Background2D(img, c.box; mask, kw...)) : B.Background2D(img, c.box; mask, kw...)
        bkg, rms = similar(img), similar(img)
        J.background2d!(bkg, rms, img, c.box; mask, kw...)
        # no mesh cell is excluded here, so the maps agree to rounding (Float32: the RMS is a
        # MAD about a location the CPU forms with Float64 literals)
        @test maximum(abs, bkg .- ref.background) / maximum(abs, ref.background) < (T == Float64 ? 1e-10 : 1e-5)
        @test maximum(abs, rms .- ref.background_rms) / maximum(abs, ref.background_rms) < (T == Float64 ? 1e-10 : 1e-3)
    end
    # Excluded cells are filled order-independently (the CPU fills in loop order): close, not equal.
    img = bkg_image(137, 119; seed = 3)
    mask = falses(size(img)); mask[30:80, 40:90] .= true
    ref = B.Background2D(img, 20; mask, rms_estimator = B.MADStdRMS())
    bkg, rms = similar(img), similar(img)
    J.background2d!(bkg, rms, img, 20; mask)
    @test maximum(abs, bkg .- ref.background) / maximum(abs, ref.background) < 0.02
    # CPU launches allocate per workgroup (one mesh box), never per pixel
    big = bkg_image(400, 400); bkg, rms = similar(big), similar(big)
    J.background2d!(bkg, rms, big, 20)
    @test (@allocated J.background2d!(bkg, rms, big, 20)) / 400 < 2048 * sizeof(Float64)
    @test J.device_background_ok(20, B.SExtractorBackground(), B.MADStdRMS())
    @test !J.device_background_ok(160, B.SExtractorBackground(), B.MADStdRMS())
    @test !J.device_background_ok(20, B.BiweightLocationBackground(), B.MADStdRMS())
end

@testset "device and host backgrounds give the same joint loop" begin
    s = two_image_scene()
    rng = StableRNG(51)
    J.render!(s.sc.data, s.sc, s.θ, J.footprint(s.sc, s.θ, s.rt))
    imgs = map(enumerate(s.imgs)) do (e, im)
        JointImage(J.image_view(s.sc, s.sc.data, e) .+ 30 .+ sqrt(30) .* randn(rng, s.sc.dims[e]), fill(1 / 30, s.sc.dims[e]), im.psf, im.to_sky;
                   band = im.band, flux_scale = im.flux_scale, psf_offset = im.psf_offset)
    end
    a = fit_joint_multipass(imgs, s.frame; options = JointLoopOptions(device_background = true)).catalog
    b = fit_joint_multipass(imgs, s.frame; options = JointLoopOptions(device_background = false)).catalog
    @test length(a.u) == length(b.u)
    @test a.u ≈ b.u rtol = 1e-8
    @test a.flux_A ≈ b.flux_A rtol = 1e-6
    @test a.flux_B ≈ b.flux_B rtol = 1e-6
    # an unsupported estimator is announced once (`@info` on the CPU)
    o = JointLoopOptions(max_iter = 1, background = (; bkg_estimator = CrowdPhot.Background.BiweightLocationBackground()))
    @test_logs (:info, r"falls back to the host Background2D: bkg_estimator") fit_joint_multipass(imgs, s.frame; options = o)
    @test_logs fit_joint_multipass(imgs, s.frame; options = JointLoopOptions(max_iter = 1, device_background = false))
    # single-image keywords the joint loop does not support, and typos, throw
    for kw in ((; bkg_coarse_passes = 1), (; bkg_rms_box_size = 40), (; bkg_box = 30))
        @test_throws "unsupported joint background keyword(s) $(only(keys(kw)))" fit_joint_multipass(imgs, s.frame; options = JointLoopOptions(background = kw))
    end
end

@testset "fit_joint_multipass products" begin
    s = two_image_scene()
    rng = StableRNG(52)
    J.render!(s.sc.data, s.sc, s.θ, J.footprint(s.sc, s.θ, s.rt))
    imgs = map(enumerate(s.imgs)) do (e, im)
        JointImage(J.image_view(s.sc, s.sc.data, e) .+ 30 .+ sqrt(30) .* randn(rng, s.sc.dims[e]), fill(1 / 30, s.sc.dims[e]), im.psf, im.to_sky;
                   band = im.band, flux_scale = im.flux_scale, psf_offset = im.psf_offset)
    end
    names = (:background, :rms, :model, :residual, :nu, :snm, :chi2, :nu_residual, :snm_residual, :chi2_residual, :nimg)
    R = fit_joint_multipass(imgs, s.frame; products = reverse(names))
    p = R.products
    @test keys(p) == (names..., :grid)
    @test p.grid == J.oversample(s.frame, JointDetector().K)
    @test all(e -> size(p.model[e]) == size(imgs[e].data), eachindex(imgs))
    @test all(e -> p.residual[e] ≈ imgs[e].data .- p.background[e] .- p.model[e], eachindex(imgs))
    # the model is the returned catalog's
    sc = JointScene(imgs, s.frame); c = R.catalog; f = JointFitOptions()
    psfs = [im.psf for im in imgs]
    rt = J.RadiusTable(sc, psfs, [im.weight for im in imgs]; model_rad = f.model_rad, nsigma = f.model_rad_nsigma, f.R_fit, f.R_cap)
    θ = permutedims(hcat(c.u, c.v, c.flux_A, c.flux_B))
    m = zeros(length(sc.data)); J.render!(m, sc, θ, J.footprint(sc, θ, rt))
    flat(x) = reduce(vcat, vec.(x))
    @test flat(p.model) ≈ m rtol = 1e-12
    # the maps are joint_maps of data - background and of the residual, with the loop's 1 / rms² weights
    jd = J.JointDetection(sc, psfs, [im.psf_offset for im in imgs], JointDetector())
    iv = ifelse.(flat([im.weight for im in imgs]) .> 0, 1 ./ flat(p.rms) .^ 2, 0.0)
    md = J.joint_maps(jd, sc, flat([im.data for im in imgs]) .- flat(p.background), iv)
    mr = J.joint_maps(jd, sc, flat(p.residual), iv)
    @test isequal(p.nu, md.ν) && isequal(p.snm, md.snm) && isequal(p.chi2, md.χ2)
    @test isequal(p.nu_residual, mr.ν) && isequal(p.snm_residual, mr.snm) && isequal(p.chi2_residual, mr.χ2)
    @test p.nimg == md.nimg
    @test maximum(filter(isfinite, p.nu_residual)) < maximum(filter(isfinite, p.nu)) / 2
    # nothing requested: only the grid; :nimg alone; unknown names and χ² without one throw
    o1 = JointLoopOptions(max_iter = 1)
    @test fit_joint_multipass(imgs, s.frame; options = o1).products == (; grid = p.grid)
    @test fit_joint_multipass(imgs, s.frame; options = o1, products = (:nimg,)).products.nimg == md.nimg
    @test_throws "unknown products coadd_flux" fit_joint_multipass(imgs, s.frame; products = (:coadd_flux, :nu))
    mf = JointLoopOptions(detector = JointDetector(method = :matched_filter))
    @test_throws "products :chi2 and :chi2_residual need" fit_joint_multipass(imgs, s.frame; options = mf, products = (:chi2_residual,))
    # :coadd requests every coadd product but :coadd_rms_aperture, which can be named: the coadd of
    # the images with the final background, RMS and model, as `coadd` gives it (which fits its own
    # PolyMaps on the grid rather than using the scene's, so agreement is to their fit tolerance)
    Rc = fit_joint_multipass(imgs, s.frame; options = o1, products = (:coadd, :background, :rms, :model, :coadd_rms_aperture),
                             coadd_options = (; pixfrac = 0.8))
    pc = Rc.products
    @test keys(pc) == (:background, :rms, :model, :coadd_data, :coadd_background, :coadd_model, :coadd_residual, :coadd_rms,
                       :coadd_rms_aperture, :coadd_weight, :coadd_nimg, :grid)
    @test !haskey(fit_joint_multipass(imgs, s.frame; options = o1, products = (:coadd,)).products, :coadd_rms_aperture)
    @test keys(pc.coadd_data) == (:A, :B) && size(pc.coadd_data.A) == (pc.grid.ny, pc.grid.nx)
    C = coadd(imgs, pc.grid; quantities = (:data, :residual, :rms_aperture, :nimg), background = pc.background, rms = pc.rms, model = pc.model,
              pixfrac = 0.8)
    for (a, b) in ((C.data.B, pc.coadd_data.B), (C.residual.A, pc.coadd_residual.A), (C.rms_aperture.A, pc.coadd_rms_aperture.A))
        k = isfinite.(a) .& isfinite.(b)
        @test count(xor.(isfinite.(a), isfinite.(b))) <= 2 && maximum(abs.(a[k] .- b[k])) < 1e-4 * maximum(abs, b[k])
    end
    @test_throws "unsupported coadd_options quantities" fit_joint_multipass(imgs, s.frame; products = (:coadd,), coadd_options = (; quantities = (:data,)))
end
