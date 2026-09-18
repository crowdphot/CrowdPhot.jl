using CrowdPhot
using CrowdPhot: Catalog, FitPlan, SequentialFitter, SimultaneousFitter, _clamp_inds,
    _extract_source_catalog, fit_pass, pass_weights, stamp_geometry, estimate_background_multipass
using CrowdPhot.PSF: CircularGaussianPSF, CircularGaussianPRF, GaussianPRF, GriddedPSFModel, ImagePSF, render
using CrowdPhot: PSF
import ConstructionBase
using StableRNGs
using Statistics: median
using Test

const PSF_FWHM = 2.5
const TEST_PSF = CircularGaussianPSF(0.0, 0.0, PSF_FWHM, 1.0, 0.0)
const TEST_FIXED = (; fwhm = PSF_FWHM, bkg = 0.0)
# Small frames: a coarse mesh that still fits in the image.
const SMALL_BKG = (; bkg_box_size = 20, bkg_box_size_coarse = 60)

function crowded_field(; n = 150, seed = 11, shape = (120, 120))
    rng = StableRNG(seed)
    img, src = simulate_image(shape, TEST_PSF, n; background = 50.0, noise = :poisson_gaussian,
        read_noise = 2.0, gain = 1.0, flux = (300.0, 20000.0), flux_distribution = :powerlaw,
        flux_power = 2.0, border = 8, rng)
    iv = 1.0 ./ (simulate_image(shape, TEST_PSF, src; background = 50.0, noise = :none) .+ 4.0)
    return img, src, iv
end

@testset "_clamp_inds" begin
    img = zeros(10, 10)
    @test _clamp_inds(3:7, 3:7, img) == (3:7, 3:7)
    @test _clamp_inds(8:12, 8:12, img) == (8:10, 8:10)
    yr, xr = _clamp_inds(11:15, 3:7, img)
    @test length(yr) * length(xr) == 0
    @test _clamp_inds(CartesianIndices((8:12, 8:12)), img) == (8:10, 8:10)
end

@testset "_extract_source_catalog" begin
    psf = CircularGaussianPSF(y = 0.0, x = 0.0, fwhm = 2.0, flux = 1.0, bkg = 0.0)
    params, errors = _extract_source_catalog((; y = [10.0, 20.0], x = [30.0, 40.0], flux = [500.0, 600.0]), psf, Float64)
    @test size(params) == (5, 2)
    @test params[1, :] == [10.0, 20.0] && params[2, :] == [30.0, 40.0]
    @test params[3, :] == [2.0, 2.0]          # fwhm row from the psf
    @test params[4, :] == [500.0, 600.0] && params[5, :] == [0.0, 0.0]
    @test all(isnan, errors)
    shapes = [(; centroid = (; y = 10.0, x = 30.0), flux = 500.0)]
    p2, _ = _extract_source_catalog(shapes, psf, Float64)
    @test p2[[1, 2, 4], 1] == [10.0, 30.0, 500.0]
    p3, _ = _extract_source_catalog((; y = [5.0], x = [15.0]), psf, Float64)
    @test p3[4, 1] == 1.0 && p3[5, 1] == 0.0  # flux and bkg defaults
end

@testset "_exp_disk_kernel invariants" begin
    for fw in (1.5, 2.5, 4.0)
        K = CrowdPhot._exp_disk_kernel(fw, Float64)
        @test size(K) == (5, 5)
        @test sum(K) ≈ 1.0
        @test K ≈ reverse(K; dims = 1) && K ≈ reverse(K; dims = 2)
        @test argmax(K) == CartesianIndex(3, 3)
    end
    @test CrowdPhot._exp_disk_kernel(6.0, Float64)[3, 3] < CrowdPhot._exp_disk_kernel(2.0, Float64)[3, 3]
end

@testset "argument validation" begin
    img = fill(10.0, 40, 40)
    @test_throws "sweeps_per_pass must be positive" fit_all_stars_multipass(img, TEST_PSF, 3.0;
        fixed = TEST_FIXED, sweeps_per_pass = 0)
    @test_throws "lm_iterations must be positive" fit_all_stars_multipass(img, TEST_PSF, 3.0;
        fixed = TEST_FIXED, lm_iterations = 0)
    @test_throws "fits only (y, x, flux)" fit_all_stars_multipass(img, TEST_PSF, 3.0)  # fwhm free
    @test_throws "fixed.bkg must be zero" fit_all_stars_multipass(img, TEST_PSF, 3.0;
        fixed = (; fwhm = PSF_FWHM, bkg = 5.0))
    # Driver keywords reach the shared validation.
    @test_throws "max_iter must be positive" fit_all_stars_multipass(img, TEST_PSF, 3.0;
        fixed = TEST_FIXED, max_iter = 0)
    @test_throws MethodError fit_all_stars_multipass(img, TEST_PSF, 3.0; fixed = TEST_FIXED, not_a_keyword = 1)
end

@testset "fitter hooks" begin
    plan = FitPlan(TEST_PSF, TEST_FIXED)
    free = SequentialFitter(1, 3, true, 1e-4, 0.1, (;))
    pinned = SequentialFitter(1, 3, false, 1e-4, 0.1, (;))
    @test CrowdPhot.n_free_per_source(free, plan) == 4
    @test CrowdPhot.n_free_per_source(pinned, plan) == 3
    @test CrowdPhot.min_stamp_pixels(free, plan) == 5
    @test CrowdPhot.initial_fit_state(free, Float64) === nothing
    st = CrowdPhot.empty_pass_stats(free, nothing, Float64)
    @test (st.n_lin, st.n_trials, st.n_accepted) == (0, 0, 0) && isnan(st.gnorm)
end

@testset "move = false agrees with the simultaneous fitter" begin
    # Both fitters' non-moving rebuild renders the same model and computes the same
    # pruning statistic `flux * sqrt(H_ff)`; this pins the sequential scatter and
    # its curvature accumulation against the stamp-based implementation.
    img, src, iv = crowded_field(; n = 60)
    o = (; bkg_box_size = 20, bkg_box_size_coarse = 60, bkg_coarse_passes = 0,
           bkg_rms_box_size = 20, bkg_estimator = CrowdPhot.Background.SExtractorBackground(),
           bkg_rms_estimator = CrowdPhot.Background.MADStdRMS(), bkg_kws = (;), mask = nothing,
           coverage_mask = nothing, fixed_inv_var = iv, R_fit = 4, R_cap = 20,
           model_rad = :auto, model_rad_nsigma = 1.0, max_step = 1.0)
    ny, nx = size(img)
    catalog = Catalog{Float64}((; y = src.y, x = src.x, flux = src.flux), TEST_PSF)
    bkg = estimate_background_multipass(img, zeros(ny, nx), 2, o)
    data, w = pass_weights(img, bkg, Float64)
    plan = FitPlan(TEST_PSF, TEST_FIXED)
    geom = stamp_geometry(catalog, w, o.R_fit, ny, nx; min_pixels = 5)
    seq = fit_pass(SequentialFitter(1, 3, false, 1e-4, 0.1, (;)), data, w, geom, catalog, TEST_PSF, plan, o,
        nothing; ny, nx, move = false)
    sim = fit_pass(SimultaneousFitter{Float64}(:lsqr, 1, 10, 1e-4, 8, 1e-3, 10.0, 10.0, 1e-12, 1e12),
        data, w, geom, catalog, TEST_PSF, plan, o, 1e-3; ny, nx, move = false)
    @test seq.catalog.y == catalog.y && seq.catalog.flux == catalog.flux
    @test seq.catalog.flux_snr ≈ sim.catalog.flux_snr rtol = 1e-10
    @test seq.model ≈ sim.model rtol = 1e-10
    @test seq.stats.cost_end ≈ sim.stats.cost_start rtol = 1e-10
    @test seq.stats.n_lin == 0 && seq.stats.n_star_fits == 0

    # Both fitters report `gnorm` at the parameters they return: after an
    # accepted simultaneous step it must match an independent evaluation there.
    simfit = fit_pass(SimultaneousFitter{Float64}(:lsqr, 1, 10, 1e-4, 8, 1e-3, 10.0, 10.0, 1e-12, 1e12),
        data, w, geom, catalog, TEST_PSF, plan, o, 1e-3; ny, nx)
    @test simfit.stats.n_accepted == 1
    up = CrowdPhot._touched_pixels(geom.pixels, length(data))
    g_ref, c_ref = CrowdPhot._end_of_pass_gnorm(TEST_PSF, plan, simfit.theta, vec(simfit.model), data, w, geom, up, o.R_fit)
    @test simfit.stats.gnorm ≈ g_ref rtol = 1e-8
    @test simfit.stats.cost_end ≈ c_ref rtol = 1e-10
end

@testset "separable circular Gaussian kernels" begin
    # Reference: the per-pixel `evaluate_fg` normal equations, every parameter free.
    img0 = [0.3 * sin(i + 2j) + 5.0 for i in 1:40, j in 1:50]
    wts2 = [1.0 + 0.01 * (i + j) for i in 1:40, j in 1:50]
    inds = CartesianIndices((14:26, 25:37))
    for m in (CircularGaussianPRF(20.3, 30.6, 2.7, 1500.0, 3.0),
              CircularGaussianPSF(20.3, 30.6, 2.7, 1500.0, 3.0),
              GaussianPRF(20.3, 30.6, 2.4, 3.1, 0.0, 1500.0, 3.0),
              GaussianPRF(20.3, 30.6, 2.4, 3.1, 25.0, 1500.0, 3.0))   # rotated: per-pixel fallback
        img = img0 .+ [PSF.evaluate(ConstructionBase.setproperties(m, (; y = 20.0, x = 31.0)), i, j) for i in 1:40, j in 1:50]
        np = length(ConstructionBase.getproperties(m))
        wts = [wts2[I] for I in inds]
        Aref = zeros(np, np); bref = zeros(np); cref = 0.0
        for (k, I) in enumerate(inds)
            f, g = PSF.evaluate_fg(m, I)
            rr = f - img[I]
            cref += wts[k] * rr^2
            for q in 1:np
                bref[q] += wts[k] * rr * g[q]
                for p in 1:np
                    Aref[p, q] += wts[k] * g[p] * g[q]
                end
            end
        end
        A = zeros(np, np); b = zeros(np); r = zeros(length(inds))
        sep = m isa Union{CircularGaussianPSF, CircularGaussianPRF}
        acc = sep ? PSF._accum_separable_gaussian! : PSF._accum_gaussian_prf!
        c = sep ?
            acc(A, b, r, img, inds, m, Tuple(1:np), wts, PSF._separable_axis_buffers(13, 13, Float64)) :
            acc(A, b, r, img, inds, m, Tuple(1:np), wts)
        @test c ≈ cref rtol = 1e-10
        @test isapprox(A, Aref; rtol = 1e-9, atol = 1e-9 * maximum(abs, Aref))
        @test isapprox(b, bref; rtol = 1e-9, atol = 1e-9 * maximum(abs, bref))
        # The allocating convenience method agrees bitwise for the separable kernel.
        if sep
            A2 = zeros(np, np); b2 = zeros(np)
            @test acc(A2, b2, zeros(length(inds)), img, inds, m, Tuple(1:np), wts) == c && A2 == A
        end
        # Specialized rendering, when present, matches generic per-pixel rendering.
        v1 = PSF.render!(zeros(15, 15), m, 13:27, 24:38, PSF._render_scratch(m, 15, Float64))
        v2 = PSF.render!(zeros(15, 15), m, 13:27, 24:38, nothing)
        @test v1 ≈ v2 rtol = 1e-12
    end

    # The simultaneous separable stamp fill matches generic `evaluate_fg`, masking included.
    for prf in (CircularGaussianPRF(0.0, 0.0, 2.7, 1.0, 0.0), CircularGaussianPSF(0.0, 0.0, 2.7, 1.0, 0.0))
        fx = (; fwhm = 2.7, bkg = 0.0)
        plan = FitPlan(prf, fx)
        R, ny, nx = 4, 40, 40
        cat = Catalog{Float64}([12.3, 20.7], [15.4, 24.1], [900.0, 400.0], fill(NaN, 2), [1, 1])
        w = fill(0.5, ny * nx); w[12 + 14 * ny] = 0.0
        geom = stamp_geometry(cat, w, R, ny, nx)
        θ = CrowdPhot.theta_from_catalog(cat, plan)
        mk() = CrowdPhot.StampDerivatives{Float64, Int32}(zeros(plan.p, geom.S2, 2), geom.pixels, zeros(plan.p, 2), ny * nx, plan.p, geom.S2)
        s1, s2 = mk(), mk()
        args = (plan.free_names_val, plan.fixed, θ, w, plan.grad_col, geom.dy_off, geom.dx_off,
                geom.anchor_y, geom.anchor_x, plan.row_y, plan.row_x, plan.row_flux, trues(2))
        CrowdPhot._fill_stamps!(s1, prf, args..., CrowdPhot._fill_scratch(prf, 2R + 1, Float64))
        CrowdPhot._fill_stamps_generic!(s2, prf, args..., nothing)
        @test isapprox(s1.values, s2.values; rtol = 1e-10, atol = 1e-12)
        @test s1.colnorm ≈ s2.colnorm rtol = 1e-12
    end

    # `lm_irls` started from caller-supplied normal equations reproduces `fit_star` exactly.
    prf = CircularGaussianPRF(20.2, 30.4, 2.7, 1200.0, 0.0)
    img = [PSF.evaluate(CircularGaussianPRF(20.5, 30.1, 2.7, 1500.0, 0.0), i, j) + 0.01 * cos(3i + j) for i in 1:40, j in 1:50]
    inds = CartesianIndices((15:25, 25:35)); iv = fill(2.0, 40, 50); fx = (; fwhm = 2.7, bkg = 0.0)
    _, res_ref = PSF.fit_star(prf, img, inds; fixed = fx, inv_var = iv, max_iter = 4)
    prob = PSF._star_problem(prf, img, inds, fx, iv)
    A0 = zeros(3, 3); b0 = zeros(3)
    c0 = prob.accum!(A0, b0, zeros(prob.nobs), prob.x0, prob.base_weights)
    res = CrowdPhot.lm_irls(prob; max_iter = 4, initial_normal = (A0, b0, c0))
    @test res.minimizer == res_ref.minimizer
    @test res.iterations == res_ref.iterations && res.minimum == res_ref.minimum
end

@testset "convergence precheck" begin
    img, src, iv = crowded_field(; n = 150, seed = 21)
    kws = (; SMALL_BKG..., inv_var = iv, fixed = TEST_FIXED, max_iter = 4, min_iter = 4, sweeps_per_pass = 2)
    r0 = fit_all_stars_multipass(img, TEST_PSF, 4.0; kws..., skip_tol = 0.0)
    r1 = fit_all_stars_multipass(img, TEST_PSF, 4.0; kws...)
    @test all(h.n_skipped == 0 for h in r0.pass_history)
    n_skip = sum(h.n_skipped for h in r1.pass_history)
    @test n_skip > 0.3 * (n_skip + sum(h.n_star_fits for h in r1.pass_history))
    @test sum(h.n_star_fits for h in r1.pass_history) < sum(h.n_star_fits for h in r0.pass_history)
    # Skipping sources that are already converged changes nothing that matters.
    @test abs(length(r1.phot.flux) - length(r0.phot.flux)) <= 3
    js = [argmin(hypot.(r0.phot.y .- r1.phot.y[i], r0.phot.x .- r1.phot.x[i])) for i in eachindex(r1.phot.y)]
    d = [hypot(r0.phot.y[js[i]] - r1.phot.y[i], r0.phot.x[js[i]] - r1.phot.x[i]) for i in eachindex(js)]
    good = d .< 0.5
    @test count(good) > 0.95 * length(good)
    @test median(abs.(r1.phot.flux[good] ./ r0.phot.flux[js[good]] .- 1)) < 2e-3
    @test median(d[good]) < 5e-3
    @test_throws "skip_tol must be non-negative" fit_all_stars_multipass(img, TEST_PSF, 4.0; kws..., skip_tol = -1.0)
end

@testset "isolated star, noiseless" begin
    img = fill(20.0, (60, 60))
    CrowdPhot.PSF.add_star!(img, CircularGaussianPSF(y = 30.4, x = 29.7, fwhm = PSF_FWHM, flux = 5000.0, bkg = 0.0))
    iv = fill(1 / 20.0, size(img))
    catalog = (; y = [30.1], x = [30.0], flux = [3000.0])
    kws = (; SMALL_BKG..., inv_var = iv, max_iter = 1, min_iter = 1)
    r = fit_all_stars_multipass(img, TEST_PSF, catalog, 8; kws..., fixed = TEST_FIXED)
    # The simultaneous reference starts at the truth: one linearization per pass
    # does not converge this far-off start, and the errors depend on the position.
    rs = fit_all_stars_simultaneous_multipass(img, TEST_PSF, (; y = [30.4], x = [29.7], flux = [5000.0]),
        8; kws..., fixed = TEST_FIXED)
    p = r.phot
    @test length(p.flux) == 1
    @test p.flux[1] ≈ 5000.0 rtol = 1e-6
    @test p.y[1] ≈ 30.4 atol = 1e-6
    @test p.x[1] ≈ 29.7 atol = 1e-6
    @test p.flux_err[1] ≈ rs.phot.flux_err[1] rtol = 1e-3
    @test p.y_err[1] ≈ rs.phot.y_err[1] rtol = 1e-3
    @test p.bkg == [0.0] && p.bkg_err == [0.0]
    @test isapprox(p.spread_model[1], 0.0; atol = 3e-3)
    @test isfinite(p.spread_model_err[1]) && p.spread_model_err[1] > 0
    @test length(p.morphology) == 1
    # The terminal pass either refits the source or finds it already converged.
    @test r.pass_history[end].n_lin == 1 && r.pass_history[end].n_star_fits + r.pass_history[end].n_skipped == 1
    @test p.n_passes == sum(h.n_lin for h in r.pass_history)

    # A free local pedestal: estimated, with an error, and still zero here.
    rb = fit_all_stars_multipass(img, TEST_PSF, catalog, 8; kws..., fixed = (; fwhm = PSF_FWHM))
    @test rb.phot.flux[1] ≈ 5000.0 rtol = 1e-6
    @test abs(rb.phot.bkg[1]) < 1e-6
    @test rb.phot.bkg_err[1] > 0
    @test rb.phot.flux_err[1] > p.flux_err[1]   # one more free parameter
end

@testset "local pedestal absorbs a background step" begin
    # An offset confined to a region much smaller than the background mesh is
    # invisible to the mesh; a free pedestal takes it up and keeps the flux.
    img = fill(20.0, (80, 80))
    img[30:50, 30:50] .+= 3.0
    CrowdPhot.PSF.add_star!(img, CircularGaussianPSF(y = 40.2, x = 39.6, fwhm = PSF_FWHM, flux = 4000.0, bkg = 0.0))
    iv = fill(1 / 20.0, size(img))
    catalog = (; y = [40.0], x = [40.0], flux = [4000.0])
    kws = (; bkg_box_size = 80, bkg_box_size_coarse = 80, inv_var = iv, max_iter = 1, min_iter = 1,
             detect_sigma = 1e6, prune = false)
    rp = fit_all_stars_multipass(img, TEST_PSF, catalog, 4; kws..., fixed = TEST_FIXED)
    rf = fit_all_stars_multipass(img, TEST_PSF, catalog, 4; kws..., fixed = (; fwhm = PSF_FWHM))
    offset = 23.0 - rf.background.background[40, 40]   # what the mesh left behind
    @test rf.phot.bkg[1] ≈ offset atol = 0.05
    @test abs(rf.phot.flux[1] - 4000.0) < abs(rp.phot.flux[1] - 4000.0)
    @test abs(rf.phot.flux[1] - 4000.0) < 5.0
end

@testset "crowded field: agreement with the simultaneous fitter" begin
    img, src, iv = crowded_field(; n = 150)
    kws = (; SMALL_BKG..., inv_var = iv, fixed = TEST_FIXED, max_iter = 6, min_iter = 2, few_sources = 2)
    rq = fit_all_stars_multipass(img, TEST_PSF, 4.0; kws...)
    rs = fit_all_stars_simultaneous_multipass(img, TEST_PSF, 4.0; kws...)
    @test abs(length(rq.phot.flux) - length(rs.phot.flux)) <= 0.05 * length(rs.phot.flux)
    # Match each sequential source to the nearest simultaneous one.
    js = [argmin(hypot.(rs.phot.y .- rq.phot.y[i], rs.phot.x .- rq.phot.x[i])) for i in eachindex(rq.phot.y)]
    d = [hypot(rs.phot.y[js[i]] - rq.phot.y[i], rs.phot.x[js[i]] - rq.phot.x[i]) for i in eachindex(js)]
    good = d .< 0.5
    @test count(good) > 0.9 * length(good)
    rel = abs.(rq.phot.flux[good] ./ rs.phot.flux[js[good]] .- 1)
    @test median(rel) < 0.01
    @test median(d[good]) < 0.02
    # Every returned source was fit and passes the final gate.
    @test all(rq.phot.valid) && all(>(0), rq.phot.flux) && all(isfinite, rq.phot.flux_err)
    @test length(rq.phot.morphology) == length(rq.phot.flux)
    # Each pass leaves the model better than it found it.
    for h in rq.pass_history
        @test h.cost_end <= h.cost_start
        @test length(h.sweep_costs) == h.n_lin
        @test issorted(h.sweep_costs; rev = true)
    end
end

@testset "sweeps and iterations reduce the gradient" begin
    img, src, iv = crowded_field(; n = 150, seed = 5)
    kws = (; SMALL_BKG..., inv_var = iv, fixed = TEST_FIXED, max_iter = 3, min_iter = 3)
    r1 = fit_all_stars_multipass(img, TEST_PSF, 4.0; kws..., sweeps_per_pass = 1)
    r3 = fit_all_stars_multipass(img, TEST_PSF, 4.0; kws..., sweeps_per_pass = 3)
    @test all(h.n_lin == 3 for h in r3.pass_history)
    @test r3.pass_history[end].gnorm < r1.pass_history[end].gnorm
    @test r3.pass_history[end].cost_end <= r1.pass_history[end].cost_end * (1 + 1e-3)
end

@testset "freeze_positions_final" begin
    img, src, iv = crowded_field(; n = 40, seed = 3)
    kws = (; SMALL_BKG..., inv_var = iv, fixed = TEST_FIXED, max_iter = 2, min_iter = 2)
    r = fit_all_stars_multipass(img, TEST_PSF, 4.0; kws..., freeze_positions_final = true)
    # The terminal pass moved nothing but fluxes, so no position step can cap.
    @test r.pass_history[end].n_capped == 0
    @test all(isfinite, r.phot.x_err) && all(>(0), r.phot.x_err)
end

@testset "GriddedPSFModel" begin
    node = ImagePSF(render(TEST_PSF); y = 0.0, x = 0.0, flux = 1.0, bkg = 0.0, oversampling = 1, normalize = true)
    gpsf = GriddedPSFModel([node], [0.0], [0.0]; y = 0.0, x = 0.0, flux = 1.0, bkg = 0.0)
    img, src, iv = crowded_field(; n = 40, seed = 8)
    kws = (; SMALL_BKG..., inv_var = iv, fixed = (; bkg = 0.0), max_iter = 3, min_iter = 2)
    r = fit_all_stars_multipass(img, gpsf, 4.0; kws...)
    rs = fit_all_stars_simultaneous_multipass(img, gpsf, 4.0; kws...)
    @test length(r.phot.flux) > 0.8 * length(rs.phot.flux)
    @test all(isfinite, r.phot.flux_err)
end

@testset "spread_model" begin
    psf = TEST_PSF
    catalog = (; y = [30.4], x = [29.7], flux = [5000.0])
    kws = (; SMALL_BKG..., fixed = TEST_FIXED, max_iter = 1, min_iter = 1)
    function sm_for_width(wd; kw...)
        im = fill(20.0, (60, 60))
        CrowdPhot.PSF.add_star!(im, CircularGaussianPSF(y = 30.4, x = 29.7, fwhm = wd, flux = 5000.0, bkg = 0.0))
        return fit_all_stars_multipass(im, psf, catalog, 8; kws..., inv_var = fill(1 / 20.0, size(im)), kw...).phot
    end
    # Extended sources: positive and increasing with width.
    sm = [sm_for_width(wd).spread_model[1] for wd in (2.5, 3.0, 4.0, 6.0)]
    @test abs(sm[1]) < 3e-3
    @test all(sm[2:end] .> 5e-4) && issorted(sm)
    # A broader reference disk raises spread_model for an extended source.
    @test sm_for_width(4.0; spread_model_fwhm = 8.0).spread_model[1] > sm[3]
    # Without inv_var the diagnostics still get weights, from the background RMS
    # map, so the error is finite -- unlike the per-star fitter this replaced.
    # Those weights come from the noise, so this image needs some, and both
    # fitters must agree on it.
    im = 20.0 .+ sqrt(20.0) .* randn(StableRNG(2), 60, 60)
    CrowdPhot.PSF.add_star!(im, CircularGaussianPSF(y = 30.4, x = 29.7, fwhm = PSF_FWHM, flux = 5000.0, bkg = 0.0))
    p = fit_all_stars_multipass(im, psf, catalog, 8; kws...).phot
    q = fit_all_stars_simultaneous_multipass(im, psf, catalog, 8; kws...).phot
    @test isfinite(p.spread_model[1]) && isfinite(p.spread_model_err[1])
    @test abs(p.spread_model[1]) < 3 * p.spread_model_err[1]
    @test p.spread_model[1] ≈ q.spread_model[1] rtol = 1e-3
end
