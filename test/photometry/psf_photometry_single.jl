using CrowdPhot
using CrowdPhot: CircularGaussianPSF, _clamp_inds
using CrowdPhot.PSF: free_params
using StableRNGs
using Statistics: median
using Test

# ---------------------------------------------------------------------------
# _clamp_inds
# ---------------------------------------------------------------------------

@testset "_clamp_inds" begin
    img = zeros(10, 10)

    @testset "fully inside" begin
        yr, xr = _clamp_inds(3:7, 3:7, img)
        @test yr == 3:7
        @test xr == 3:7
        @test length(yr) * length(xr) == 25
    end

    @testset "partially outside" begin
        yr, xr = _clamp_inds(8:12, 8:12, img)
        @test yr == 8:10
        @test xr == 8:10
        @test length(yr) * length(xr) == 9
    end

    @testset "completely outside" begin
        yr, xr = _clamp_inds(11:15, 3:7, img)
        @test length(yr) * length(xr) == 0
    end

    @testset "CartesianIndices forwarding" begin
        yr, xr = _clamp_inds(CartesianIndices((8:12, 8:12)), img)
        @test yr == 8:10
        @test xr == 8:10
    end
end

# ---------------------------------------------------------------------------
# _extract_source_catalog
# ---------------------------------------------------------------------------

@testset "_extract_source_catalog" begin
    psf = CircularGaussianPSF(y=0.0, x=0.0, fwhm=2.0, flux=1.0, bkg=0.0)

    @testset "NamedTuple with :y, :x" begin
        sources = (; y=[10.0, 20.0], x=[30.0, 40.0], flux=[500.0, 600.0])
        params, errors = CrowdPhot._extract_source_catalog(sources, psf, Float64)
        @test size(params) == (5, 2)
        @test params[1, :] == [10.0, 20.0]  # y
        @test params[2, :] == [30.0, 40.0]  # x
        @test params[4, :] == [500.0, 600.0] # flux
        @test params[5, :] == [0.0, 0.0]     # bkg (default)
    end

    @testset "measure_star_shapes output" begin
        # The `Vector{<:NamedTuple}` method reads `centroid.y`/`centroid.x` and
        # the `flux` field of each element.
        sources = [(; centroid = (; y = 10.0, x = 30.0), flux = 500.0),
                   (; centroid = (; y = 20.0, x = 40.0), flux = 600.0)]
        params, _ = CrowdPhot._extract_source_catalog(sources, psf, Float64)
        @test params[1, :] == [10.0, 20.0]
        @test params[2, :] == [30.0, 40.0]
        @test params[4, :] == [500.0, 600.0]
        @test params[5, :] == [0.0, 0.0]
    end

    @testset "NamedTuple without optional fields" begin
        sources = (; y=[5.0], x=[15.0])
        params, errors = CrowdPhot._extract_source_catalog(sources, psf, Float64)
        @test params[4, 1] == 1.0  # flux default
        @test params[5, 1] == 0.0  # bkg default
    end

    @testset "NamedTuple with bkg" begin
        sources = (; y=[5.0], x=[15.0], bkg=[3.0])
        params, _ = CrowdPhot._extract_source_catalog(sources, psf, Float64)
        @test params[5, 1] == 3.0
    end

    @testset "fwhm row filled from psf default" begin
        sources = (; y=[5.0], x=[15.0])
        params, _ = CrowdPhot._extract_source_catalog(sources, psf, Float64)
        @test params[3, 1] == 2.0  # fwhm from psf
    end

    @testset "errors initialized to NaN" begin
        sources = (; y=[5.0], x=[15.0])
        _, errors = CrowdPhot._extract_source_catalog(sources, psf, Float64)
        @test all(isnan, errors)
    end
end

# ---------------------------------------------------------------------------
# _extract_errors!
# ---------------------------------------------------------------------------

@testset "_extract_errors!" begin
    T = Float64
    errors = fill(T(NaN), 3, 1)
    cov = T[4.0 0.5; 0.5 1.0]
    free_idx = (1, 3)
    is_fixed = (2,)

    CrowdPhot._extract_errors!(errors, cov, free_idx, is_fixed, 1)
    @test errors[1, 1] ≈ 2.0         # sqrt(4.0)
    @test errors[2, 1] == 0.0         # fixed
    @test errors[3, 1] ≈ 1.0         # sqrt(1.0)
end

# ---------------------------------------------------------------------------
# fit_all_stars — integration
# ---------------------------------------------------------------------------

@testset "fit_all_stars" begin
    rng = StableRNG(42)
    T = Float64
    truth_psf = CircularGaussianPSF(y=0.0, x=0.0, fwhm=2.0, flux=1.0, bkg=0.0)

    @testset "noiseless recovery" begin
        image, sources = simulate_image((128, 128), truth_psf, 5;
            background = 20.0, noise = :none, flux = (600.0, 900.0),
            min_separation = 7, border = 8, model_radius = 30, rng)
        psf = CircularGaussianPSF(y=0.0, x=0.0, fwhm=2.0, flux=1.0, bkg=0.0)
        # Sequentially fitting blended stars leaves residual crosstalk even
        # with multiple passes; DAOPHOT handles this via simultaneous group
        # fits.  Tolerances reflect what the sequential algorithm can achieve.
        result = fit_all_stars(image, psf, sources, 5; n_passes = 3, max_iter = 100, fixed=(; bkg = 20.0))

        @test result.n_passes == 3
        @test result.n_failed == 0
        @test isempty(result.failure_msgs)
        @test sum(result.valid) == 5
        @test all(result.converged)
        for i in 1:5
            @test result.flux[i] ≈ sources.flux[i] rtol = 0.10
            @test result.y[i] ≈ sources.y[i] atol = 0.01
            @test result.x[i] ≈ sources.x[i] atol = 0.01
            @test isfinite(result.qfit[i])
            @test result.qfit[i] >= 0
        end
        # qfit_expected and qfit_z are NaN when inv_var is not provided.
        @test all(isnan, result.qfit_expected)
        @test all(isnan, result.qfit_z)
        # Crowding: finite, non-negative (may be positive due to blending).
        @test all(isfinite, result.crowding)
        @test all(x -> x >= 0, result.crowding)
    end

    @testset "single-pass runs without error" begin
        image, sources = simulate_image((128, 128), truth_psf, 5;
            background = 20.0, noise = :none, flux = (600.0, 900.0),
            min_separation = 10, border = 10, model_radius = 5, rng)
        psf = CircularGaussianPSF(y=0.0, x=0.0, fwhm=2.0, flux=1.0, bkg=0.0)
        inv_var = fill(1.0, size(image))
        result = fit_all_stars(image, psf, sources, 5; n_passes = 1, max_iter = 100, inv_var)
        @test result.n_passes == 1
        @test all(result.valid)
        @test all(result.converged)
        @test all(isfinite, result.qfit_expected)
        @test all(x -> x > 0, result.qfit_expected)
        @test all(isfinite, result.qfit_z)
    end

    @testset "fixed background" begin
        image, sources = simulate_image((64, 64), truth_psf, 5;
            background = 20.0, noise = :none, flux = (500.0, 800.0),
            min_separation = 10, border = 8, model_radius = 5, rng)
        psf = CircularGaussianPSF(y=0.0, x=0.0, fwhm=2.0, flux=1.0, bkg=0.0)
        result = fit_all_stars(image, psf, sources, 5;
            fixed = (; bkg = 20.0), n_passes = 1, max_iter = 100)

        @test all(x -> x ≈ 20.0, result.bkg)
        @test all(iszero, result.bkg_err)
        for i in 1:5
            @test result.flux[i] ≈ sources.flux[i] rtol = 0.02
        end
    end

    @testset "fixed positions" begin
        image, sources = simulate_image((64, 64), truth_psf, 5;
            background = 20.0, noise = :none, flux = (500.0, 800.0),
            min_separation = 10, border = 8, model_radius = 5, rng)
        psf = CircularGaussianPSF(y=0.0, x=0.0, fwhm=2.0, flux=1.0, bkg=0.0)
        result = fit_all_stars(image, psf, sources, 5;
            fixed = (; x = 32.0, y = 32.0), n_passes = 1, max_iter = 100)

        @test all(x -> x ≈ 32.0, result.x)
        @test all(y -> y ≈ 32.0, result.y)
        @test all(iszero, result.x_err)
        @test all(iszero, result.y_err)
        # Fixed-position errors are zero; flux and bkg errors may be NaN
        # (no covariance_estimator) or near-zero (noiseless fit).
        @test all(isfinite, result.y_err)
        @test all(isfinite, result.x_err)
    end

    @testset "edge stars" begin
        image, _ = simulate_image((64, 64), truth_psf, 2;
            background = 20.0, noise = :none, flux = (500.0, 600.0),
            min_separation = 30, border = 30, model_radius = 5, rng)
        psf = CircularGaussianPSF(y=0.0, x=0.0, fwhm=2.0, flux=1.0, bkg=0.0)
        # One star at the edge, one at center
        sources = (; y = [1.0, 32.0], x = [32.0, 32.0], flux = [500.0, 500.0])
        result = fit_all_stars(image, psf, sources, 5; n_passes = 1, max_iter = 50)
        # Edge star may or may not survive depending on cutout size
        @test result.valid[2]  # center star should be valid
    end

    @testset "multi-pass does not error" begin
        image, sources = simulate_image((64, 64), truth_psf, 5;
            background = 20.0, noise = :none, flux = (400.0, 700.0),
            min_separation = 5, border = 8, model_radius = 5, rng)
        psf = CircularGaussianPSF(y=0.0, x=0.0, fwhm=2.0, flux=1.0, bkg=0.0)
        result = fit_all_stars(image, psf, sources, 5; n_passes = 3, max_iter = 100)
        @test result.n_passes == 3
        @test sum(result.valid) == 5
        @test all(result.converged)
    end

    @testset "NamedTuple input with fluxes" begin
        image, sources = simulate_image((64, 64), truth_psf, 3;
            background = 20.0, noise = :none, flux = (500.0, 700.0),
            min_separation = 15, border = 10, model_radius = 5, rng)
        psf = CircularGaussianPSF(y=0.0, x=0.0, fwhm=2.0, flux=1.0, bkg=0.0)
        # Use the actual source positions from the simulation, with reasonable
        # initial flux guesses.
        cat = (; y = sources.y, x = sources.x, flux = fill(400.0, 3))
        result = fit_all_stars(image, psf, cat, 5; n_passes = 1, max_iter = 200)
        @test length(result.y) == 3
        @test all(result.converged)
        for i in 1:3
            @test result.flux[i] ≈ sources.flux[i] rtol = 0.05
        end
    end

    @testset "return struct fields have correct types" begin
        image, sources = simulate_image((64, 64), truth_psf, 3;
            background = 20.0, noise = :none, flux = (500.0, 700.0),
            min_separation = 15, border = 10, model_radius = 5, rng)
        psf = CircularGaussianPSF(y=0.0, x=0.0, fwhm=2.0, flux=1.0, bkg=0.0)
        result = fit_all_stars(image, psf, sources, 5; n_passes = 1, max_iter = 100)

        @test result.y isa Vector{Float64}
        @test result.x isa Vector{Float64}
        @test result.flux isa Vector{Float64}
        @test result.bkg isa Vector{Float64}
        @test result.y_err isa Vector{Float64}
        @test result.flux_err isa Vector{Float64}
        @test result.converged isa BitVector
        @test result.valid isa BitVector
        @test result.n_iter isa Vector{Int}
        @test result.qfit isa Vector{Float64}
        @test result.qfit_expected isa Vector{Float64}
        @test result.qfit_z isa Vector{Float64}
        @test result.crowding isa Vector{Float64}
        @test result.spread_model isa Vector{Float64}
        @test result.spread_model_err isa Vector{Float64}
        @test result.n_failed isa Int
        @test result.failure_msgs isa Vector{String}
        @test result.residual isa Matrix{Float64}
        @test size(result.residual) == size(image)
    end

    @testset "invalid flux rejected between passes" begin
        image, sources = simulate_image((64, 64), truth_psf, 5;
            background = 20.0, noise = :none, flux = (400.0, 700.0),
            min_separation = 10, border = 8, model_radius = 5, rng)
        psf = CircularGaussianPSF(y=0.0, x=0.0, fwhm=2.0, flux=1.0, bkg=0.0)
        # Include a source with negative flux that should be rejected
        sources_bad = (; y = [sources.y; 50.0],
                         x = [sources.x; 50.0],
                         flux = [sources.flux; -100.0])
        result = fit_all_stars(image, psf, sources_bad, 5; n_passes = 2, max_iter = 100)
        @test sum(result.valid) == 5  # 5 good + 1 bad rejected
        @test !result.valid[6]
    end

    @testset "spread_model" begin
        srng = StableRNG(7)
        fwhm_psf = 2.5
        psf = CircularGaussianPSF(y = 0.0, x = 0.0, fwhm = fwhm_psf, flux = 1.0, bkg = 0.0)

        # Point source: spread_model ~ 0 when fit with the matching PSF.
        img = fill(20.0, (60, 60))
        CrowdPhot.PSF.add_star!(img, CircularGaussianPSF(y = 30.4, x = 29.7, fwhm = fwhm_psf, flux = 5000.0, bkg = 0.0))
        iv = fill(1 / 20.0, size(img))
        cat = (; y = [30.4], x = [29.7], flux = [5000.0])
        r = fit_all_stars(img, psf, cat, 8; n_passes = 2, max_iter = 100, fixed = (; bkg = 20.0, fwhm = fwhm_psf), inv_var = iv)
        @test r.valid[1]
        @test isapprox(r.spread_model[1], 0.0; atol = 3e-3)
        @test isfinite(r.spread_model_err[1]) && r.spread_model_err[1] > 0

        # spread_model_err is NaN without inv_var; the value stays finite.
        r_noiv = fit_all_stars(img, psf, cat, 8; n_passes = 2, max_iter = 100, fixed = (; bkg = 20.0, fwhm = fwhm_psf))
        @test isfinite(r_noiv.spread_model[1])
        @test isnan(r_noiv.spread_model_err[1])

        # Extended sources: spread_model > 0 and increases with injected width.
        function spread_for_width(w)
            im = fill(20.0, (60, 60))
            CrowdPhot.PSF.add_star!(im, CircularGaussianPSF(y = 30.4, x = 29.7, fwhm = w, flux = 5000.0, bkg = 0.0))
            rr = fit_all_stars(im, psf, cat, 8; n_passes = 2, max_iter = 100,
                fixed = (; bkg = 20.0, fwhm = fwhm_psf), inv_var = fill(1 / 20.0, size(im)))
            rr.spread_model[1]
        end
        widths = [2.5, 3.0, 4.0, 6.0]
        sm = spread_for_width.(widths)
        @test all(sm[2:end] .> 5e-4)
        @test issorted(sm)

        # Sharper than the PSF (single hot pixel): spread_model < 0.
        spike = fill(20.0, (40, 40))
        spike[20, 20] += 3000.0
        rspk = fit_all_stars(spike, psf, (; y = [20.0], x = [20.0], flux = [3000.0]), 8;
            n_passes = 2, max_iter = 100, fixed = (; bkg = 20.0, fwhm = fwhm_psf), inv_var = fill(1 / 20.0, size(spike)))
        @test rspk.valid[1]
        @test rspk.spread_model[1] < -1e-3

        # Explicit spread_model_fwhm changes the reference-disk scale, hence the
        # spread_model magnitude for an extended source (a noiseless point
        # source stays at the structural zero regardless of the kernel).
        ext = fill(20.0, (60, 60))
        CrowdPhot.PSF.add_star!(ext, CircularGaussianPSF(y = 30.4, x = 29.7, fwhm = 4.0, flux = 5000.0, bkg = 0.0))
        r_auto = fit_all_stars(ext, psf, cat, 8; n_passes = 2, max_iter = 100,
            fixed = (; bkg = 20.0, fwhm = fwhm_psf), inv_var = fill(1 / 20.0, size(ext)))
        r_big = fit_all_stars(ext, psf, cat, 8; n_passes = 2, max_iter = 100,
            fixed = (; bkg = 20.0, fwhm = fwhm_psf), inv_var = fill(1 / 20.0, size(ext)), spread_model_fwhm = 8.0)
        @test r_auto.spread_model[1] > 0
        @test r_big.spread_model[1] > r_auto.spread_model[1]
    end

    @testset "_exp_disk_kernel invariants" begin
        for fw in (1.5, 2.5, 4.0)
            K = CrowdPhot._exp_disk_kernel(fw, Float64)
            @test size(K) == (5, 5)
            @test sum(K) ≈ 1.0
            @test K ≈ reverse(K; dims = 1)
            @test K ≈ reverse(K; dims = 2)
            @test argmax(K) == CartesianIndex(3, 3)
        end
        # Broader reference disk removes more from the kernel center.
        @test CrowdPhot._exp_disk_kernel(6.0, Float64)[3, 3] < CrowdPhot._exp_disk_kernel(2.0, Float64)[3, 3]
    end
end
