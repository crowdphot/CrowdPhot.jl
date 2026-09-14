using CrowdPhot
using CrowdPhot: CircularGaussianPSF, StampDerivatives, apply_JT!, apply_J!,
    _jacobian_operator, _clamp_inds, _fill_stamps!, _model_radii
using CrowdPhot.PSF
using ConstructionBase
using Krylov: lsqr!, lsmr!, LsqrWorkspace, LsmrWorkspace, solution
using LinearAlgebra: dot, norm, I
using StableRNGs
using Statistics: median, quantile
using Test

# Build the full (npix x n) Jacobian from a StampDerivatives so the operator
# and LSQR step can be checked against dense linear algebra.
function dense_J(stamp::StampDerivatives)
    p, S2, n_active = size(stamp.values)
    J = zeros(stamp.npix, n_active * p)
    for a in 0:(n_active - 1)
        for m in 1:S2
            fi = stamp.pixels[m, a + 1]
            fi != 0 || continue
            for k in 1:p
                J[fi, a * p + k] = stamp.values[k, m, a + 1]
            end
        end
    end
    return J
end

# Column-equilibrate a stamp (as the real fill does) so H has unit diagonal.
function equilibrate!(stamp::StampDerivatives)
    p, S2, n_active = size(stamp.values)
    for a in 1:n_active
        for k in 1:p
            s = zero(eltype(stamp.values))
            for m in 1:S2
                s += stamp.values[k, m, a]^2
            end
            s = max(sqrt(s), eps())
            for m in 1:S2
                stamp.values[k, m, a] /= s
            end
        end
    end
    return stamp
end

@testset "simultaneous fitting" begin
    rng = StableRNG(1234)
    T = Float64

    @testset "_fill_stamps! preserves frozen colnorm/values" begin
        psf = CircularGaussianPSF(y=0.0, x=0.0, fwhm=2.0, flux=1.0, bkg=0.0)
        fixed = (; fwhm=2.0, bkg=0.0)
        free_names, free_idx, _ = PSF.free_params(psf, fixed)
        p = length(free_idx)
        prop_names = collect(keys(ConstructionBase.getproperties(psf)))
        row_y = findfirst(==(:y), prop_names)
        row_x = findfirst(==(:x), prop_names)
        row_flux = findfirst(==(:flux), prop_names)
        grad_col = [free_names[k] === :y ? 1 : (free_names[k] === :x ? 2 : 3) for k in 1:p]
        free_names_val = Val(free_names)

        R = 1
        dy_off = Int[]
        dx_off = Int[]
        for dx in -R:R, dy in -R:R
            push!(dy_off, dy)
            push!(dx_off, dx)
        end
        S2 = length(dy_off)
        n_active = 2
        ny = 30
        anchor_y = [10, 10]
        anchor_x = [10, 20]  # far apart: no shared pixels between the two stars
        pixels = zeros(Int32, S2, n_active)
        for a in 1:n_active, m in 1:S2
            gy = anchor_y[a] + dy_off[m]
            gx = anchor_x[a] + dx_off[m]
            pixels[m, a] = gy + (gx - 1) * ny
        end
        npix = ny * 30
        θ = zeros(p * n_active)
        for a in 1:n_active
            θ[(a - 1) * p .+ (1:p)] .= (Float64(anchor_y[a]), Float64(anchor_x[a]), 100.0)
        end
        w = ones(npix)
        stamp = StampDerivatives{Float64, Int32}(
            zeros(p, S2, n_active), pixels, zeros(p, n_active), npix, p, S2)

        live = trues(n_active)
        live[2] = false  # star 2 frozen from the start
        _fill_stamps!(stamp, psf, free_names_val, fixed, θ, w,
            grad_col, dy_off, dx_off, anchor_y, anchor_x, row_y, row_x, row_flux, live, nothing)
        values_before = copy(stamp.values)
        colnorm_before = copy(stamp.colnorm)
        @test all(isfinite, values_before)
        @test all(isfinite, colnorm_before)

        # Repeated fills with a perturbed θ for the still-live star only: the
        # frozen star's column must stay bitwise untouched, not drift or blow
        # up via a stale colnorm floor (the bug this test regresses).
        θ2 = copy(θ)
        for _ in 1:5
            θ2[1] += 0.1
            _fill_stamps!(stamp, psf, free_names_val, fixed, θ2, w,
                grad_col, dy_off, dx_off, anchor_y, anchor_x, row_y, row_x, row_flux, live, nothing)
        end
        @test stamp.values[:, :, 2] == values_before[:, :, 2]
        @test stamp.colnorm[:, 2] == colnorm_before[:, 2]
        @test all(isfinite, stamp.values)
        @test all(isfinite, stamp.colnorm)
    end

    @testset "_fill_stamps! zeros masked entries" begin
        # `apply_J!`/`apply_JT!` touch masked entries (the adjoint gathers the
        # clamped `u[1]`) and rely on the derivative being exactly zero.  The
        # multipass `StampStore` reuses one buffer across passes, so a skipped
        # slot would otherwise keep a derivative from a different source/mask.
        psf = CircularGaussianPSF(y = 0.0, x = 0.0, fwhm = 2.0, flux = 1.0, bkg = 0.0)
        fixed = (; fwhm = 2.0, bkg = 0.0)
        free_names, free_idx, _ = PSF.free_params(psf, fixed)
        p = length(free_idx)
        prop_names = collect(keys(ConstructionBase.getproperties(psf)))
        row_y, row_x, row_flux = findfirst(==(:y), prop_names), findfirst(==(:x), prop_names), findfirst(==(:flux), prop_names)
        grad_col = [free_names[k] === :y ? 1 : (free_names[k] === :x ? 2 : 3) for k in 1:p]
        free_names_val = Val(free_names)

        R, ny = 1, 30
        dy_off = Int[]; dx_off = Int[]
        for dx in -R:R, dy in -R:R
            push!(dy_off, dy); push!(dx_off, dx)
        end
        S2 = length(dy_off)
        npix = ny * 30
        anchor_y = [10]; anchor_x = [10]
        pixels = zeros(Int32, S2, 1)
        for m in 1:S2                      # every other stamp entry masked out
            isodd(m) || continue
            pixels[m, 1] = (anchor_y[1] + dy_off[m]) + (anchor_x[1] + dx_off[m] - 1) * ny
        end
        θ = Float64[anchor_y[1], anchor_x[1], 100.0]
        w = ones(npix)
        stamp = StampDerivatives{Float64, Int32}(
            fill(NaN, p, S2, 1), pixels, zeros(p, 1), npix, p, S2)   # poisoned buffer

        _fill_stamps!(stamp, psf, free_names_val, fixed, θ, w, grad_col, dy_off, dx_off,
            anchor_y, anchor_x, row_y, row_x, row_flux, trues(1), nothing)
        for m in 1:S2
            pixels[m, 1] == 0 || continue
            @test all(iszero, stamp.values[:, m, 1])
        end
        @test all(isfinite, stamp.values)
        # The adjoint must stay finite: a masked slot left at NaN poisons all of it.
        z = zeros(p)
        apply_JT!(z, stamp, ones(npix), trues(1), Vector{Float64}(undef, S2))
        @test all(isfinite, z)
    end

    @testset "adjoint identity" begin
        p = 3
        S2 = 9
        n_active = 4
        npix = 20
        pixels = zeros(Int32, S2, n_active)
        # Give stars overlapping footprints.
        for a in 1:n_active
            for m in 1:S2
                fi = mod(m + (a - 1) * 2 - 1, npix) + 1
                pixels[m, a] = fi
            end
        end
        stamp = StampDerivatives{Float64, Int32}(
            randn(rng, p, S2, n_active), pixels, zeros(p, n_active), npix, p, S2)
        u = randn(rng, npix)
        v = randn(rng, p * n_active)
        y = zeros(npix)
        z = zeros(p * n_active)
        apply_J!(y, stamp, v)
        apply_JT!(z, stamp, u, trues(n_active))
        @test dot(u, y) ≈ dot(z, v) rtol = 1e-12

        # Both products agree with the explicit dense Jacobian.
        J = dense_J(stamp)
        @test y ≈ J * v rtol = 1e-12
        @test z ≈ J' * u rtol = 1e-12

        # A frozen star drops its columns from J: `apply_J!` ignores that
        # star's slice of `v`, and `apply_JT!` leaves its slice of `z` at 0.
        live = trues(n_active); live[2] = false
        Jm = copy(J); Jm[:, (2 - 1) * p + 1:2 * p] .= 0
        apply_J!(y, stamp, v, live, zeros(S2))
        apply_JT!(z, stamp, u, live)
        @test y ≈ Jm * v rtol = 1e-12
        @test z ≈ Jm' * u rtol = 1e-12
        @test all(iszero, view(z, (2 - 1) * p + 1:2 * p))
    end

    @testset "matrix-free operator matches dense J" begin
        p = 3
        S2 = 9
        n_active = 6
        npix = 30
        pixels = zeros(Int32, S2, n_active)
        for a in 1:n_active, m in 1:S2
            pixels[m, a] = mod(m + (a - 1) * 2 - 1, npix) + 1
        end
        stamp = StampDerivatives{Float64, Int32}(
            randn(rng, p, S2, n_active), pixels, zeros(p, n_active), npix, p, S2)
        equilibrate!(stamp)
        J = dense_J(stamp)
        n = p * n_active

        live = trues(n_active)
        op = _jacobian_operator(stamp, live, zeros(S2), npix, n)
        v = randn(rng, n)
        u = randn(rng, npix)
        @test op * v ≈ J * v rtol = 1e-12
        @test op' * u ≈ J' * u rtol = 1e-12

        # `op` closes over `live`: freezing a star zeroes its columns in place.
        live[3] = false
        Jm = copy(J); Jm[:, (3 - 1) * p + 1:3 * p] .= 0
        @test op * v ≈ Jm * v rtol = 1e-12
        @test op' * u ≈ Jm' * u rtol = 1e-12
    end

    @testset "LSQR/LSMR step matches dense damped normal-equation solve" begin
        p = 3
        S2 = 9
        n_active = 6
        npix = 8 * n_active + 8
        pixels = zeros(Int32, S2, n_active)
        # Light overlap (1 shared pixel between adjacent stars) so JᵀJ is
        # well-conditioned and the Krylov solves converge tightly.
        for a in 1:n_active, m in 1:S2
            pixels[m, a] = mod(m + (a - 1) * 8 - 1, npix) + 1
        end
        stamp = StampDerivatives{Float64, Int32}(
            randn(rng, p, S2, n_active), pixels, zeros(p, n_active), npix, p, S2)
        equilibrate!(stamp)
        J = dense_J(stamp)
        n = p * n_active

        μ = 1.0e-3                       # Marquardt damping (the outer-loop `λ`)
        b = randn(rng, npix)             # weighted residual RHS
        δ_dense = (J' * J + μ * Matrix{Float64}(I, n, n)) \ (J' * b)

        for (wsT, solve!) in ((LsqrWorkspace, lsqr!), (LsmrWorkspace, lsmr!))
            live = trues(n_active)
            op = _jacobian_operator(stamp, live, zeros(S2), npix, n)
            ws = wsT(npix, n, Vector{Float64})
            solve!(ws, op, b; λ = sqrt(μ), itmax = 200, atol = 1e-12, btol = 1e-12)
            @test solution(ws) ≈ δ_dense rtol = 1e-7 atol = 1e-9

            # A frozen star: its slice of the step stays exactly 0 (its columns
            # of `op` are structurally zero and the solver starts from 0).
            live[2] = false
            solve!(ws, op, randn(rng, npix); λ = sqrt(μ), itmax = 200, atol = 1e-12, btol = 1e-12)
            @test all(iszero, view(solution(ws), (2 - 1) * p + 1:2 * p))
        end
    end


    @testset "isolated star matches fit_all_stars" begin
        circ_psf = CircularGaussianPSF(y=0.0, x=0.0, fwhm=2.0, flux=1.0, bkg=0.0)
        # Sample the circular Gaussian onto a unit-oversampled grid and wrap it
        # as a (single-node) GriddedPSFModel, so the loop below also exercises
        # `_fill_stamps!`/`_render_model!`'s specialized
        # `GriddedPSFModel{T,<:ImagePSF{T}}` methods against the same fixture.
        node_data = render(circ_psf)
        node = ImagePSF(node_data; y=0.0, x=0.0, flux=1.0, bkg=0.0, oversampling=1, normalize=true)
        gridded_psf = GriddedPSFModel([node], [0.0], [0.0]; y=0.0, x=0.0, flux=1.0, bkg=0.0)

        for (psf, fixed) in ((circ_psf, (; fwhm=2.0, bkg=0.0)), (gridded_psf, (; bkg=0.0)))
            rng2 = StableRNG(42)
            image, sources = simulate_image((128, 128), psf, 5;
                background=20.0, noise=:none, flux=(600.0, 900.0),
                min_separation=15, border=10, model_radius=5, rng=rng2)
            img_sub = image .- 20.0
            cat = (; y=sources.y, x=sources.x, flux=fill(400.0, 5))
            r_seq = fit_all_stars(img_sub, psf, cat, 5; fixed, n_passes=1, max_iter=200)
            # This fixture checks convergence to near machine precision in the
            # noiseless limit, which needs a tighter `x_tol` than
            # `fit_all_stars_simultaneous`'s default which is appropriate for noisy data.
            r_sim = fit_all_stars_simultaneous(img_sub, psf, cat, 5;
                fixed, max_iter=40, inner_iterations=10, x_tol=1e-8, model_rad=5)
            @test all(r_sim.valid)
            @test r_sim.flux ≈ r_seq.flux rtol = 1e-10
            @test r_sim.y ≈ r_seq.y atol = 1e-10
            @test r_sim.x ≈ r_seq.x atol = 1e-10
            @test r_sim.flux ≈ sources.flux rtol = 1e-10
            # LSMR agrees with the default LSQR solver.
            r_lsmr = fit_all_stars_simultaneous(img_sub, psf, cat, 5;
                fixed, solver = :lsmr, max_iter = 40, inner_iterations = 10, x_tol = 1e-8, model_rad=5)
            @test r_lsmr.flux ≈ r_sim.flux rtol = 1e-9
            @test r_lsmr.y ≈ r_sim.y atol = 1e-9
        end
    end

    @testset "LSQR/LSMR agree in a crowded fit that forces mid-fit freezing" begin
        # Unlike the isolated-star fixture above, this one is crowded enough
        # that most stars freeze well before max_iter -- it exercises the
        # matrix-free operator's structural projection of frozen stars, and
        # cross-checks the two Krylov solvers against each other on a
        # degenerate, heavily-blended field.
        rng7 = StableRNG(11)
        psf = CircularGaussianPSF(y=0.0, x=0.0, fwhm=2.0, flux=1.0, bkg=0.0)
        image, sources = simulate_image((150, 150), psf, 200;
            background=50.0, noise=:poisson_gaussian, read_noise=2.0, gain=1.0,
            flux=(100.0, 3000.0), flux_distribution=:powerlaw, flux_power=2.0,
            min_separation=2, border=8, model_radius=6, rng=rng7)
        img_sub = image .- 50.0
        cat = (; y=sources.y, x=sources.x, flux=copy(sources.flux))
        fixed = (; fwhm=2.0, bkg=0.0)
        # Disable f_tol, tight `x_tol` (see the isolated-star fixture above for why):
        # a looser `x_tol` lets the two solvers freeze the same star at
        # slightly different iterations in this degenerate fixture.
        r_lsqr = fit_all_stars_simultaneous(img_sub, psf, cat, 5; fixed, max_iter=25, inner_iterations=10, x_tol=1e-8, f_tol=0, model_rad=5)
        r_lsmr = fit_all_stars_simultaneous(img_sub, psf, cat, 5; fixed, max_iter=25, inner_iterations=10, solver=:lsmr, x_tol=1e-8, f_tol=0, model_rad=5)
        # This fixture must actually exercise freezing before the fit ends.
        @test sum(r_lsqr.n_iter[r_lsqr.valid] .< r_lsqr.n_passes) > 0.5 * sum(r_lsqr.valid)
        g = r_lsqr.valid .& r_lsmr.valid
        @test sum(g) > 0.8 * length(g)
        @test r_lsqr.flux[g] ≈ r_lsmr.flux[g] rtol = 1e-2
        # Positions are compared per-star rather than with `≈ atol=...` (which
        # for arrays compares 2-norms and fails as soon as *any* single star
        # converges one iteration apart between the two solvers). The bulk
        # must agree far below the centroid noise floor.
        dy = abs.(r_lsqr.y[g] .- r_lsmr.y[g])
        dx = abs.(r_lsqr.x[g] .- r_lsmr.x[g])
        @test median(dy) < 1e-5
        @test median(dx) < 1e-5
        @test quantile(dy, 0.9) < 1e-3
        @test quantile(dx, 0.9) < 1e-3
        @test maximum(dy) < 2e-2
        @test maximum(dx) < 2e-2
    end

    @testset "free-set contract" begin
        psf = CircularGaussianPSF(y=0.0, x=0.0, fwhm=2.0, flux=1.0, bkg=0.0)
        image, sources = simulate_image((64, 64), psf, 1;
            background=20.0, noise=:none, flux=(500.0, 500.0),
            border=8, model_radius=5, rng=StableRNG(1))
        cat = (; y=sources.y, x=sources.x, flux=copy(sources.flux))
        # bkg left free -> fwhm also free -> error.
        @test_throws ArgumentError fit_all_stars_simultaneous(image, psf, cat, 5)
        # Bad solver.
        @test_throws ArgumentError fit_all_stars_simultaneous(image, psf, cat, 5;
            fixed=(; fwhm=2.0, bkg=0.0), solver=:foo)
    end

    @testset "empty input" begin
        psf = CircularGaussianPSF(y=0.0, x=0.0, fwhm=2.0, flux=1.0, bkg=0.0)
        img = zeros(16, 16)
        cat = (; y=Float64[], x=Float64[], flux=Float64[])
        r = fit_all_stars_simultaneous(img, psf, cat, 5; fixed=(; fwhm=2.0, bkg=0.0))
        @test isempty(r.flux)
        @test r.n_passes == 0
    end

    @testset "failure paths" begin
        psf = CircularGaussianPSF(y=0.0, x=0.0, fwhm=2.0, flux=1.0, bkg=0.0)
        img = zeros(32, 32)
        # Non-finite image, no inv_var -> error.
        img_bad = copy(img); img_bad[5, 5] = NaN
        @test_throws ArgumentError fit_all_stars_simultaneous(img_bad, psf,
            (; y=[16.0], x=[16.0], flux=[100.0]), 5; fixed=(; fwhm=2.0, bkg=0.0))
        # Fully masked star -> excluded before the solve.
        inv_var = ones(32, 32)
        inv_var[10:22, 10:22] .= 0.0
        r = fit_all_stars_simultaneous(img, psf,
            (; y=[16.0], x=[16.0], flux=[100.0]), 5;
            fixed=(; fwhm=2.0, bkg=0.0), inv_var)
        @test r.n_failed == 1
        @test !r.valid[1]
    end

    @testset "qfit is global-residual based" begin
        # qfit equals sum(|global residual|) / flux
        rng4 = StableRNG(5)
        psf = CircularGaussianPSF(y=0.0, x=0.0, fwhm=2.0, flux=1.0, bkg=0.0)
        image, sources = simulate_image((64, 64), psf, 3;
            background=20.0, noise=:none, flux=(500.0, 800.0),
            min_separation=6, border=8, model_radius=5, rng=rng4)
        img_sub = image .- 20.0
        cat = (; y=sources.y, x=sources.x, flux=copy(sources.flux))
        fixed = (; fwhm=2.0, bkg=0.0)
        r = fit_all_stars_simultaneous(img_sub, psf, cat, 5; fixed, max_iter=40, model_rad=5)
        for i in 1:length(r.flux)
            r.valid[i] || continue
            # The diagnostics box is the fit's own `anchor +- R_fit`, and the
            # anchors were rounded from the *input* positions when the stamps
            # were built, not from the fitted ones.
            ay, ax = round(Int, catalog.y[i]), round(Int, catalog.x[i])
            yr, xr = _clamp_inds((ay - 5):(ay + 5), (ax - 5):(ax + 5), r.residual)
            @test r.qfit[i] ≈ sum(abs, view(r.residual, yr, xr)) / r.flux[i] rtol = 1e-10
        end
    end

    @testset "spread_model matches sequential path" begin
        rng8 = StableRNG(21)
        fwhm_psf = 2.5
        psf = CircularGaussianPSF(y = 0.0, x = 0.0, fwhm = fwhm_psf, flux = 1.0, bkg = 0.0)
        # isolated stars + one injected broadened Gaussian
        img = fill(20.0, (90, 90))
        pts = [(20.5, 20.5), (20.5, 68.5), (68.5, 20.5)]
        for (yy, xx) in pts
            CrowdPhot.PSF.add_star!(img, CircularGaussianPSF(y = yy, x = xx, fwhm = fwhm_psf, flux = 4000.0, bkg = 0.0))
        end
        CrowdPhot.PSF.add_star!(img, CircularGaussianPSF(y = 68.5, x = 68.5, fwhm = 5.0, flux = 4000.0, bkg = 0.0))
        img_sub = img .- 20.0
        cat = (; y = [first.(pts); 68.5], x = [last.(pts); 68.5], flux = fill(4000.0, 4))
        iv = fill(1 / 20.0, size(img))
        fixed = (; fwhm = fwhm_psf, bkg = 0.0)
        rseq = fit_all_stars(img_sub, psf, cat, 7; fixed, n_passes = 3, max_iter = 100, inv_var = iv)
        rsim = fit_all_stars_simultaneous(img_sub, psf, cat, 7; fixed, max_iter = 40, inv_var = iv)
        for i in 1:3  # the isolated point sources
            @test rseq.valid[i] && rsim.valid[i]
            @test isapprox(rseq.spread_model[i], 0.0; atol = 3e-3)
            @test isapprox(rsim.spread_model[i], 0.0; atol = 3e-3)
            @test isapprox(rseq.spread_model[i], rsim.spread_model[i]; atol = 2e-3)
        end
        # the extended source: positive in both, and consistent
        @test rseq.spread_model[4] > 5e-3
        @test rsim.spread_model[4] > 5e-3
        @test isapprox(rseq.spread_model[4], rsim.spread_model[4]; rtol = 0.25)
    end

    @testset "recovery vs truth" begin
        rng3 = StableRNG(7)
        psf = CircularGaussianPSF(y=0.0, x=0.0, fwhm=2.0, flux=1.0, bkg=0.0)
        image, sources = simulate_image((160, 160), psf, 120;
            background=100.0, noise=:poisson_gaussian, read_noise=2.0, gain=1.5,
            flux=(30.0, 3000.0), flux_distribution=:powerlaw, flux_power=2.0,
            min_separation=2, border=10, model_radius=8, rng=rng3)
        img_sub = image .- 100.0
        cat = (; y=sources.y, x=sources.x, flux=copy(sources.flux))
        fixed = (; fwhm=2.0, bkg=0.0)
        r = fit_all_stars_simultaneous(img_sub, psf, cat, 5;
            fixed, max_iter=15, inner_iterations=10, model_rad=5)
        g = r.valid
        @test sum(g) > 0.8 * length(g)
        # Bright sources recover flux more accurately than faint ones.
        f = sources.flux[g]
        bias = (r.flux[g] .- f) ./ f
        bright = f .> median(f)
        faint = .!bright
        @test median(abs.(bias[bright])) < median(abs.(bias[faint]))
    end

    @testset "per-star convergence is non-uniform" begin
        rng5 = StableRNG(99)
        psf = CircularGaussianPSF(y=0.0, x=0.0, fwhm=2.0, flux=1.0, bkg=0.0)
        image, sources = simulate_image((200, 200), psf, 60;
            background=50.0, noise=:poisson_gaussian, read_noise=2.0, gain=1.0,
            flux=(200.0, 3000.0), flux_distribution=:powerlaw, flux_power=2.0,
            min_separation=2, border=10, model_radius=6, rng=rng5)
        img_sub = image .- 50.0
        cat = (; y=sources.y, x=sources.x, flux=copy(sources.flux))
        fixed = (; fwhm=2.0, bkg=0.0)
        r = fit_all_stars_simultaneous(img_sub, psf, cat, 5; fixed, max_iter=30, model_rad=5)
        g = r.valid
        @test sum(g) > 0.5 * length(g)
        # Not every star takes the whole loop to freeze, and not every star
        # freezes at the same iteration.
        @test any(r.n_iter[g] .< r.n_passes)
        @test length(unique(r.n_iter[g])) > 1
    end

    @testset "frozen star parameters are stable under extra iterations" begin
        rng6 = StableRNG(99)
        psf = CircularGaussianPSF(y=0.0, x=0.0, fwhm=2.0, flux=1.0, bkg=0.0)
        image, sources = simulate_image((200, 200), psf, 60;
            background=50.0, noise=:poisson_gaussian, read_noise=2.0, gain=1.0,
            flux=(200.0, 3000.0), flux_distribution=:powerlaw, flux_power=2.0,
            min_separation=2, border=10, model_radius=6, rng=rng6)
        img_sub = image .- 50.0
        cat = (; y=sources.y, x=sources.x, flux=copy(sources.flux))
        fixed = (; fwhm=2.0, bkg=0.0)
        K1, K2 = 10, 25
        r1 = fit_all_stars_simultaneous(img_sub, psf, cat, 5; fixed, max_iter=K1, model_rad=5)
        r2 = fit_all_stars_simultaneous(img_sub, psf, cat, 5; fixed, max_iter=K2, model_rad=5)
        # Stars that froze at or before K1 in the longer run took an
        # identical path through the loop in the shorter run (nothing before
        # iteration K1 depends on max_iter), so their final values must
        # match exactly -- the direct operational check that freezing
        # actually stops a star's motion rather than merely reporting it as
        # converged.
        frozen_early = [i for i in eachindex(sources.y)
                        if r1.valid[i] && r2.valid[i] && 0 < r2.n_iter[i] <= K1]
        @test length(frozen_early) > 0  # fixture must actually exercise this
        for i in frozen_early
            @test r1.y[i] ≈ r2.y[i] atol = 1e-8
            @test r1.x[i] ≈ r2.x[i] atol = 1e-8
            @test r1.flux[i] ≈ r2.flux[i] rtol = 1e-8
        end
    end

    @testset "_model_radii" begin
        psf = CircularGaussianPSF(y = 0.0, x = 0.0, fwhm = 2.5, flux = 1.0, bkg = 0.0)
        R_fit, R_cap = 2, 20
        w = fill(1 / 125.0, 4000)               # sigma_bg ~ 11.2 ADU
        # Scalar path: one clamped value for all.
        @test _model_radii(psf, 7.0, 1.0, R_fit, R_cap, w, Float64[100, 5e4]) == fill(7, 2)
        @test _model_radii(psf, 1.0, 1.0, R_fit, R_cap, w, Float64[100]) == [R_fit]   # clamped up
        @test _model_radii(psf, 999.0, 1.0, R_fit, R_cap, w, Float64[100]) == [R_cap] # clamped down
        # :auto path: monotone non-decreasing in flux, faint -> R_fit, bright grows.
        fl = Float64[50, 500, 5_000, 50_000, 500_000]
        rr = _model_radii(psf, :auto, 1.0, R_fit, R_cap, w, fl)
        @test issorted(rr)
        @test all(R_fit .<= rr .<= R_cap)
        @test rr[1] == R_fit          # a faint source's wings are below the noise
        @test rr[end] > rr[1]         # a bright source needs a larger box
        # A larger nsigma (looser threshold) never needs a larger box.
        @test all(_model_radii(psf, :auto, 3.0, R_fit, R_cap, w, fl) .<= rr)
    end

    @testset "model_rad = :auto requires inv_var" begin
        psf = CircularGaussianPSF(y = 0.0, x = 0.0, fwhm = 2.0, flux = 1.0, bkg = 0.0)
        img = zeros(40, 40)
        cat = (; y = [20.0], x = [20.0], flux = [100.0])
        @test_throws ArgumentError fit_all_stars_simultaneous(img, psf, cat, 3;
            fixed = (; fwhm = 2.0, bkg = 0.0), model_rad = :auto)
        @test_throws "requires inv_var" fit_all_stars_simultaneous(img, psf, cat, 3;
            fixed = (; fwhm = 2.0, bkg = 0.0))                    # :auto is the default
        # An explicit scalar model_rad works without inv_var (OLS mode).
        r = fit_all_stars_simultaneous(img, psf, cat, 3;
            fixed = (; fwhm = 2.0, bkg = 0.0), model_rad = 3, max_iter = 5)
        @test length(r.flux) == 1
    end

    @testset "model_rad fixes the crowded-field wing-truncation bias" begin
        # A noiseless bright star ringed by faint neighbors: with a single
        # `fit_rad = 2` stamp (97% of the PSF) the neighbors' free flux absorbs
        # the bright star's truncated wings and biases it low ~1.5%.  A wide
        # `model_rad` (auto or scalar) removes it; the Jacobian box stays small.
        psf = CircularGaussianPSF(y = 0.0, x = 0.0, fwhm = 2.5, flux = 1.0, bkg = 0.0)
        fixed = (; fwhm = 2.5, bkg = 0.0)
        ffs(s) = CrowdPhot.flux_for_snr(psf, s; background = 100.0, read_noise = 5.0, gain = 1.0)
        rng = StableRNG(3)
        ys = [25.35]; xs = [25.65]; fs = [ffs(300.0)]
        for _ in 1:40
            push!(ys, 6 + 38rand(rng)); push!(xs, 6 + 38rand(rng)); push!(fs, ffs(3 + 9rand(rng)))
        end
        src = (; y = ys, x = xs, flux = fs)
        model_only = simulate_image((50, 50), psf, src; background = 0.0, noise = :none)
        iv = 1.0 ./ (100.0 .+ model_only .+ 25.0)
        img = Matrix(model_only)
        cat = (; y = copy(ys), x = copy(xs), flux = copy(fs))
        common = (; fixed, inv_var = iv, inner_iterations = 30, max_iter = 80, λ_down = 10.0)

        r_single = fit_all_stars_simultaneous(img, psf, cat, 2; common..., model_rad = 2)
        r_auto = fit_all_stars_simultaneous(img, psf, cat, 2; common..., model_rad = :auto)
        r_wide = fit_all_stars_simultaneous(img, psf, cat, 2; common..., model_rad = 8)

        bias(r) = (r.flux[1] - fs[1]) / fs[1]
        @test abs(bias(r_single)) > 5e-3            # the known single-radius bias
        @test abs(bias(r_auto)) < 1e-3             # auto model_rad removes it
        @test abs(bias(r_wide)) < 1e-3
        @test abs(r_auto.y[1] - ys[1]) < 2e-3 && abs(r_auto.x[1] - xs[1]) < 2e-3
    end

    @testset "model_rad = :auto works for GriddedPSFModel" begin
        # The :auto path renders a unit PSF and takes its curve of growth; for
        # an image/gridded PSF that uses the generic (pixel-integrated)
        # `curve_of_growth` and `ConstructionBase.setproperties`.
        circ = CircularGaussianPSF(y = 0.0, x = 0.0, fwhm = 2.0, flux = 1.0, bkg = 0.0)
        node = ImagePSF(render(circ); y = 0.0, x = 0.0, flux = 1.0, bkg = 0.0, oversampling = 1, normalize = true)
        gpsf = GriddedPSFModel([node], [0.0], [0.0]; y = 0.0, x = 0.0, flux = 1.0, bkg = 0.0)
        w = fill(1 / 125.0, 4000)
        rr = _model_radii(gpsf, :auto, 1.0, 2, 15, w, Float64[50, 500, 5_000, 50_000])
        @test issorted(rr) && all(2 .<= rr .<= 15) && rr[end] > rr[1]
        rng = StableRNG(42)
        img, src = simulate_image((128, 128), gpsf, 8; background = 20.0, noise = :none,
            flux = (600.0, 900.0), min_separation = 8, border = 10, model_radius = 6, rng)
        iv = fill(1 / 20.0, size(img))
        cat = (; y = src.y, x = src.x, flux = fill(500.0, length(src.y)))
        r = fit_all_stars_simultaneous(img .- 20.0, gpsf, cat, 3;
            fixed = (; bkg = 0.0), inv_var = iv, max_iter = 30, model_rad = :auto)
        @test all(r.valid)
        @test r.flux ≈ src.flux rtol = 1e-8
    end
end
