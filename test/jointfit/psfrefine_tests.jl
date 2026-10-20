using CrowdPhot
using CrowdPhot.Joint
using CrowdPhot.PSF: CircularGaussianPRF, ImagePSF, GriddedPSFModel, evaluate
using StableRNGs: StableRNG
using LinearAlgebra: dot
using Test

const J = CrowdPhot.Joint

affine_sky(frame, ang, t) = (A = [cosd(ang) -sind(ang); sind(ang) cosd(ang)]; (y, x) -> (uv = A * [y, x] .+ t; grid_to_world(frame, uv[1], uv[2])))
gauss_node(fwhm; os = 4, n = 61) = (c = (n + 1) / 2;
    ImagePSF([evaluate(CircularGaussianPRF(y = c, x = c, fwhm = fwhm * os, flux = os^2, bkg = 0), i, j) for i in 1:n, j in 1:n]; oversampling = os))
gridded(f; os = 4) = GriddedPSFModel([gauss_node(x; os) for x in (f, f + 0.2, f + 0.1, f + 0.3)], [0.0, 0.0, 90.0, 90.0], [0.0, 90.0, 0.0, 90.0])

"Two dithers of one band (one PSF group) and `N` stars, noiseless data rendered with `truth` PSFs (default the reference); `extra` sources are in the data only."
function refine_field(; ref = gridded(2.2), truth = ref, N = 40, seed = 3, T = Float64, sky = 0.0, noise = 0.0, extra = zeros(3, 0))
    frame = TanGrid(80.0, -69.0, 0.11, 110, 110)
    rng = StableRNG(seed)
    θ = vcat(rand(rng, 2, N) .* 70 .+ 20, exp.(randn(rng, 1, N) .* 0.5 .+ 9))
    specs = ((0.0, [18.0, 20.0]), (23.0, [24.0, 14.0]))
    mk(psf) = [JointImage(zeros(90, 90), ones(90, 90), psf, affine_sky(frame, a, t); band = :A) for (a, t) in specs]
    sct = JointScene(mk(truth), frame)
    rt = J.RadiusTable(sct, [truth, truth], [ones(90, 90), ones(90, 90)]; model_rad = 3, R_fit = 3, R_cap = 30)
    θt = hcat(θ, extra)
    model = zeros(length(sct.data)); J.render!(model, sct, θt, J.footprint(sct, θt, rt))
    imgs = map(enumerate(mk(ref))) do (e, im)
        d = J.image_view(sct, model, e) .+ sky .+ noise .* randn(rng, 90, 90)
        JointImage(d, fill(noise > 0 ? 1 / noise^2 : 1.0, 90, 90), ref, im.to_sky; band = :A)
    end
    return (; frame, imgs, θ)
end

"The pieces `_refine_problem` reads, at catalog θ (every star selected), and the problem of the one group."
function refine_state(f; T = Float64, o = PSFRefineOptions(window = 3), reserve = nothing)
    sc = JointScene(f.imgs, f.frame; T)
    psfs = [im.psf for im in f.imgs]
    rt = J.RadiusTable(sc, psfs, [im.weight for im in f.imgs]; model_rad = 3, R_fit = 3, R_cap = 30)
    st = (; sc, images = f.imgs, rt, bkg = J.KA.zeros(sc.backend, T, length(sc.data)), work = (; model = similar(sc.data)))
    θd = T.(f.θ); fp = J.footprint(sc, θd, rt); J.render!(st.work.model, sc, θd, fp)
    groups = J._refine_setup(sc, psfs, o)
    snr = fill(1e3, 1, size(f.θ, 2))
    p = J._refine_problem(st, groups[1], θd, snr, Array(fp.ay), Array(fp.ax), Array(fp.mR), o; reserve)
    buf = J._rf_buffers(p, sc.backend, T)
    return (; sc, st, g = groups[1], p, buf, o)
end

@testset "scan!" begin
    # tile (256·64 on the CPU) and recursion boundaries
    for n in (1, 255, 256, 16383, 16384, 16385, 300_000, 16384^2 ÷ 64 + 5)
        x = rand(StableRNG(n), Int32(0):Int32(3), n)
        @test J.scan!(copy(x)) == cumsum(x)
    end
end

@testset "PSF refinement operator, oversampling $os" for os in (2, 3, 4, 5)
    f = refine_field(; ref = gridded(2.2; os))
    s = refine_state(f)
    g, p, buf, sc = s.g, s.p, s.buf, s.sc
    np = g.N * g.nw^2; be = sc.backend
    rng = StableRNG(7)
    # J x is the change of the render when the window samples change by x (the model is linear in them)
    x = 1e-3 .* randn(rng, np)
    Jx = J._rf_J!(zeros(p.nr), x, nothing, p, g, buf, be)
    pert = deepcopy(g.hpsf)
    for (n, nd) in enumerate(pert.psfs), c in 1:g.nw, r in 1:g.nw
        nd.data[g.r0 + r - 1, g.c0 + c - 1] += x[(n - 1) * g.nw^2 + (c - 1) * g.nw + r]
    end
    sc2 = JointScene([JointImage(im.data, im.weight, pert, im.to_sky; band = im.band) for im in f.imgs], f.frame)
    m2 = zeros(length(sc.data)); J.render!(m2, sc2, f.θ, J.footprint(sc2, f.θ, s.st.rt))
    rows = Array(p.d.rowpix)
    @test Jx ≈ m2[rows] .- s.st.work.model[rows] rtol = 1e-9
    # the adjoint: <J x + Js y, w> = <x, Jᵀ w> + <y, Jsᵀ w>
    y = randn(rng, 3, p.ns); w = randn(rng, p.nr)
    fwd = J._rf_J!(zeros(p.nr), x, y, p, g, buf, be)
    gx = zeros(np); gy = zeros(3, p.ns); J._rf_Jt!(gx, gy, w, p, g, buf, be)
    @test dot(fwd, w) ≈ dot(x, gx) + dot(y, gy) rtol = 1e-12
    # the gauge projection leaves no sum or first moment on any phase sub-lattice
    v = randn(rng, np)
    J.launch!(J.rf_proj_kernel!, be, g.N * g.sy * g.sx, v, g.gq, g.ra, g.ca, g.nw, g.sy, g.sx)
    o0 = round(Int, (size(g.hpsf.psfs[1].data, 1) + 1) / 2)
    for n in 1:g.N, a in 0:(os - 1), b in 0:(os - 1)
        idx = [(r, c) for r in 1:g.nw, c in 1:g.nw if mod(g.r0 + r - 1 - o0, os) == a && mod(g.c0 + c - 1 - o0, os) == b]
        vals = [v[(n - 1) * g.nw^2 + (c - 1) * g.nw + r] for (r, c) in idx]
        @test abs(sum(vals)) < 1e-10
        @test abs(sum(vals .* [g.c0 + c - 1 - o0 for (r, c) in idx])) < 1e-9
        @test abs(sum(vals .* [g.r0 + r - 1 - o0 for (r, c) in idx])) < 1e-9
    end
    # Float32 builds the same problem
    s32 = refine_state(f; T = Float32)
    Jx32 = J._rf_J!(zeros(Float32, p.nr), Float32.(x), nothing, s32.p, s32.g, s32.buf, be)
    @test maximum(abs, Jx32 .- Jx) / maximum(abs, Jx) < 1e-4
end

@testset "PSF refinement step" begin
    o = PSFRefineOptions(window = 3)
    # null test: reference = truth and the exact catalog, so the step changes nothing
    s = refine_state(refine_field(); o)
    sol = J._refine_solve(s.p, s.g, o, s.sc.backend, Float64)
    peak = maximum(maximum, (nd.data for nd in s.g.hpsf.psfs))
    @test maximum(abs, sol.x) < 1e-8 * peak
    @test maximum(abs, sol.d) < 1e-6
    # a gross outlier (a corrupted pixel at 10⁴ times the peak) is rejected, not fitted
    f = refine_field()
    r = s.p.d.rowpix[findmax(Array(s.p.m))[2]]            # the brightest modeled pixel
    e = findlast(<(r), s.sc.offsets); i = r - s.sc.offsets[e]
    f.imgs[e].data[i] = -1e4 * maximum(f.imgs[e].data)
    s2 = refine_state(f; o)
    sol2 = J._refine_solve(s2.p, s2.g, o, s2.sc.backend, Float64)
    @test maximum(abs, sol2.x) < 1e-6 * peak
    # the correction toward a broader truth reduces the window mismatch
    ref = gridded(2.2); truth = gridded(2.3)
    s3 = refine_state(refine_field(; ref, truth); o)
    sol3 = J._refine_solve(s3.p, s3.g, o, s3.sc.backend, Float64)
    g = s3.g
    D(gp, n) = [Float64(gp.psfs[n].data[g.r0 + r - 1, g.c0 + c - 1]) for r in 1:g.nw, c in 1:g.nw]
    for n in 1:g.N
        x = reshape(sol3.x[(n - 1) * g.nw^2 + 1:n * g.nw^2], g.nw, g.nw)
        @test sum(abs2, D(ref, n) .+ x .- D(truth, n)) < 0.3 * sum(abs2, D(ref, n) .- D(truth, n))
    end
    # the CG stops on the device at cg_tol: the sweeps after the stop change nothing
    @test sol3.iters == [10, 10]
    ol = PSFRefineOptions(window = 3, cg_tol = 0.5); ol30 = PSFRefineOptions(window = 3, cg_tol = 0.5, cg_iterations = 30)
    sl = J._refine_solve(s3.p, s3.g, ol, s3.sc.backend, Float64); sl30 = J._refine_solve(s3.p, s3.g, ol30, s3.sc.backend, Float64)
    @test all(in(1:9), sl.iters) && sl30.iters == sl.iters
    @test sl30.x == sl.x && sl30.d == sl.d
end

@testset "reserved stars and the star-level test" begin
    # reserved stars never constrain the PSF: with every star reserved the correction stays zero
    f = refine_field(; ref = gridded(2.2), truth = gridded(2.3))
    o = PSFRefineOptions(window = 3)
    s = refine_state(f; o, reserve = trues(size(f.θ, 2)))
    sol = J._refine_solve(s.p, s.g, o, s.sc.backend, Float64)
    @test all(iszero, sol.x)
    @test maximum(abs, sol.d) > 0                         # their own positions and fluxes still move
    # an unmodeled companion beside the brightest star: the test rejects that star, and only it
    θ0 = refine_field(; noise = sqrt(20.0), sky = 20.0).θ; b = argmax(θ0[3, :])
    fc = refine_field(; noise = sqrt(20.0), sky = 20.0, extra = [θ0[1, b] + 1.3; θ0[2, b] - 0.8; 0.4 * θ0[3, b];;])
    ot = PSFRefineOptions(window = 3, star_nsigma = 5.5)
    sc_ = refine_state(fc; o = ot)
    solc = J._refine_solve(sc_.p, sc_.g, ot, sc_.sc.backend, Float64)
    @test solc.rejected == (sc_.p.stars .== b)
    # on the clean field none is rejected
    sn = refine_state(refine_field(; noise = sqrt(20.0), sky = 20.0); o = ot)
    @test !any(J._refine_solve(sn.p, sn.g, ot, sn.sc.backend, Float64).rejected)
    # options
    @test_throws "psf_refine needs irls >= 1" J._refine_setup(s.sc, [im.psf for im in f.imgs], PSFRefineOptions(irls = 0))
    @test_throws "reserve_frac must be in [0, 1)" J._refine_setup(s.sc, [im.psf for im in f.imgs], PSFRefineOptions(reserve_frac = 1.0))
end

@testset "fit_joint_multipass with psf_refine" begin
    ref = gridded(2.2); truth = gridded(2.3)
    f = refine_field(; ref, truth, N = 60, sky = 20.0, noise = sqrt(20.0))
    o = JointLoopOptions(psf_refine = PSFRefineOptions(window = 3))
    R = fit_joint_multipass(f.imgs, f.frame; options = o)
    R0 = fit_joint_multipass(f.imgs, f.frame)
    @test all(h -> h.t_refine > 0 && only(h.refine).nsel > 0, R.history)
    @test R0.psfs == [im.psf for im in f.imgs]
    @test R0.psf_stars === nothing
    # psf_stars: one row per star of the last step, pointing at catalog rows near the star
    ps = R.psf_stars
    @test !isempty(ps) && all(in(axes(R.catalog, 1)), ps.row) && all(>=(20), ps.snr) && ps.fit == .!(ps.reserve .| ps.rejected)
    # reserved stars, drawn at detection: reruns give the same rows and catalog, bit for bit
    orr = JointLoopOptions(psf_refine = PSFRefineOptions(window = 3, reserve_frac = 0.3, star_nsigma = 5.5))
    Ra = fit_joint_multipass(f.imgs, f.frame; options = orr); Rb = fit_joint_multipass(f.imgs, f.frame; options = orr)
    @test 0 < count(Ra.psf_stars.reserve) < length(Ra.psf_stars)
    @test Ra.psf_stars == Rb.psf_stars && Ra.catalog == Rb.catalog
    @test all(h -> only(h.refine).n_reserve > 0 && isfinite(only(h.refine).chi2_reserve), Ra.history)
    @test R.psfs[1] === R.psfs[2]                          # one refined model per group
    # the node the stars constrain (the others lie beyond the star field) moves toward the truth,
    # and the fluxes are less biased
    w = 17:45                                              # the refined window (window = 3 px at oversampling 4 about origin 31)
    mis(gp) = sum(abs2, Float64.(gp.psfs[1].data[w, w]) .- Float64.(truth.psfs[1].data[w, w]))
    @test mis(R.psfs[1]) < 0.3 * mis(ref)
    bias(R) = (c = R.catalog; k = [argmin(hypot.(f.θ[1, :] .- c.u[i], f.θ[2, :] .- c.v[i])) for i in eachindex(c.u)];
               ok = hypot.(f.θ[1, k] .- c.u, f.θ[2, k] .- c.v) .< 0.5; sort(c.flux_A[ok] ./ f.θ[3, k[ok]] .- 1)[cld(count(ok), 2)])
    @test abs(bias(R)) < 0.3 * abs(bias(R0))
    # unsupported: a Gaussian PRF has nothing tabulated; tiles would refine per tile
    g = CircularGaussianPRF(y = 0, x = 0, fwhm = 2.0, flux = 1, bkg = 0)
    gi = [JointImage(im.data, im.weight, g, im.to_sky; band = :A) for im in f.imgs]
    @test_throws "psf_refine needs ImagePSF or GriddedPSFModel PSFs" fit_joint_multipass(gi, f.frame; options = o)
    @test_throws "psf_refine is not supported by fit_joint_tiled" fit_joint_tiled(f.imgs, f.frame; options = o)
end
