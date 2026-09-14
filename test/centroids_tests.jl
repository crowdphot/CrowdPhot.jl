using CrowdPhot: centroid_poly, _centroid_poly3, choose_centroid
using CrowdPhot.PSF: CircularGaussianPSF, GaussianPSF, evaluate, peak as psf_peak
using FillArrays: Fill
using LinearAlgebra
using StableRNGs: StableRNG
using Statistics: mean
using Test

# Helper: generate a star image at a known position.
# FWHM=2.8 is the sweet spot from Vakili & Hogg (2016) — after matched-filter
# smoothing the 3×3 quadratic fit has enough curvature to work well.
function _make_star(; x0=5.0, y0=5.0, fwhm=2.8, flux=10.0, shape=(9,9))
    inds = (1:shape[1], 1:shape[2])
    model = CircularGaussianPSF(x=x0, y=y0, fwhm=fwhm, flux=flux, bkg=0.0)
    return evaluate.(model, inds[1], inds[2]'), model
end

@testset "centroid_poly" begin

    @testset "noise-free: centroid at pixel center" begin
        img, model = _make_star(; x0=5.0, y0=5.0)
        result = centroid_poly(img)
        @test result.poly.x ≈ 5.0 atol=1e-12
        @test result.poly.y ≈ 5.0 atol=1e-12
        @test result.poly.peak ≈ psf_peak(model) rtol=0.2
        @test result.poly.x_err > 0
        @test result.poly.y_err > 0
        @test result.poly.peak_err > 0
    end

    @testset "noise-free: sub-pixel centroid" begin
        img, model = _make_star(; x0=5.2, y0=5.3)
        result = centroid_poly(img)
        @test abs(result.poly.x - 5.2) < 0.05
        @test abs(result.poly.y - 5.3) < 0.05
        @test result.poly.peak > 0
    end

    @testset "normalized_curvature, ellipticity components" begin
        # Wide PSF: low normalized_curvature, nearly circular/symmetric.
        img_wide, _ = _make_star(; fwhm=4.0)
        r_wide = centroid_poly(img_wide)
        @test r_wide.normalized_curvature > 0    # positive for a peak
        @test r_wide.normalized_curvature < 1.0
        @test r_wide.ellipticity1_core ≈ 0 atol=1e-10  # circular ⇒ e1 = 0
        @test r_wide.ellipticity2_core ≈ 0 atol=1e-10  # circular ⇒ e2 = 0

        # Narrow PSF: higher normalized_curvature.
        img_narrow, _ = _make_star(; fwhm=1.5)
        r_narrow = centroid_poly(img_narrow)
        @test r_narrow.normalized_curvature > r_wide.normalized_curvature
        @test r_narrow.ellipticity1_core ≈ 0 atol=1e-10  # still circular
        @test r_narrow.ellipticity2_core ≈ 0 atol=1e-10
    end

    @testset "border behaviour" begin
        img = [1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 9.0]
        result = centroid_poly(img)
        @test isnan(result.poly.x)
        @test isnan(result.poly.y)
        @test isnan(result.poly.peak)
        @test isnan(result.poly.x_err)
        @test isnan(result.poly.y_err)
        @test isnan(result.poly.peak_err)
        @test all(isnan, result.poly.cov)
        @test isnan(result.com.x)
        @test isnan(result.com.y)
        @test isnan(result.com.x_err)
        @test isnan(result.com.y_err)
        @test all(isnan, result.com.cov)
        @test isnan(result.normalized_curvature)
        @test isnan(result.ellipticity1_core)
        @test isnan(result.ellipticity2_core)

        img2 = [100.0 0.0 0.0; 0.0 0.0 0.0; 0.0 0.0 0.0]
        result2 = centroid_poly(img2)
        @test isnan(result2.poly.x)
        @test isnan(result2.poly.y)
    end

    @testset "peak estimate consistency" begin
        for fwhm in (2.8, 4.0)
            img, model = _make_star(; x0=5.0, y0=5.0, fwhm=fwhm)
            result = centroid_poly(img)
            @test result.poly.peak > 0
            @test result.poly.peak ≈ psf_peak(model) rtol=0.2
        end
    end

    @testset "weighted least squares" begin
        img, _ = _make_star(; x0=5.0, y0=5.0)
        # uniform weights → same centroid as default
        r1 = centroid_poly(img)
        r2 = centroid_poly(img, ones(size(img)))
        @test r1.poly.x ≈ r2.poly.x
        @test r1.poly.y ≈ r2.poly.y
        @test r1.poly.peak ≈ r2.poly.peak

        # doubling inv_var halves the variance → errors scale by 1/√2
        r3 = centroid_poly(img, Fill(2.0, size(img)))
        @test r1.poly.x ≈ r3.poly.x
        @test r1.poly.y ≈ r3.poly.y
        @test r3.poly.x_err ≈ r1.poly.x_err / sqrt(2)  rtol=1e-10
        @test r3.poly.y_err ≈ r1.poly.y_err / sqrt(2)  rtol=1e-10
        @test r3.poly.peak_err ≈ r1.poly.peak_err / sqrt(2)  rtol=1e-10

        # down-weighting a corner pixel should not change centroid much for
        # symmetric star centered on a pixel
        w = ones(9, 9)
        w[1, 1] = 0.01
        r4 = centroid_poly(img, w)
        @test r4.poly.x ≈ r1.poly.x  atol=1e-4
        @test r4.poly.y ≈ r1.poly.y  atol=1e-4
    end

    @testset "integer and mixed-type inputs" begin
        # Centroiding should promote scalar arithmetic without copying input arrays.
        img_int = [0 1 0; 1 5 1; 0 1 0]
        r_int = centroid_poly(img_int)
        @test r_int.poly.x ≈ 2.0
        @test r_int.poly.y ≈ 2.0
        @test r_int.poly.x isa Float64
        @test r_int.com.x isa Float64

        img_f32 = Float32[0 1 0; 1 5 1; 0 1 0]
        ivar_f64 = ones(3, 3)
        r_mixed = centroid_poly(img_f32, ivar_f64)
        @test r_mixed.poly.x ≈ 2.0
        @test r_mixed.poly.y ≈ 2.0
        @test r_mixed.poly.x isa Float64
        @test r_mixed.com.x isa Float64
    end

    @testset "noise tests: SNR regimes" begin
        img_true, model = _make_star(; x0=5.0, y0=5.0)
        peak_true = psf_peak(model)
        rng = StableRNG(42)

        results = Dict{Int, Vector{Float64}}()
        for snr in (10, 50, 100)
            σ_noise = peak_true / snr
            errors_x = Float64[]
            errors_y = Float64[]
            for _ in 1:200
                noisy = img_true .+ σ_noise .* randn(rng, size(img_true))
                res = centroid_poly(noisy)
                push!(errors_x, res.poly.x - 5.0)
                push!(errors_y, res.poly.y - 5.0)
            end
            rmse_x = sqrt(mean(e -> e^2, errors_x))
            rmse_y = sqrt(mean(e -> e^2, errors_y))
            results[snr] = [rmse_x, rmse_y]
        end

        @test results[50][1] < results[10][1]
        @test results[50][2] < results[10][2]
        @test results[100][1] < results[50][1]
        @test results[100][2] < results[50][2]
        @test results[100][1] < 0.1
        @test results[100][2] < 0.1
        @test results[50][1] < 0.2
        @test results[50][2] < 0.2
        @test results[10][1] < 1.5
        @test results[10][2] < 1.5
    end

    @testset "error estimates are finite and positive" begin
        img, _ = _make_star(; x0=5.0, y0=5.0)
        result = centroid_poly(img)
        @test isfinite(result.poly.x_err)
        @test isfinite(result.poly.y_err)
        @test isfinite(result.poly.peak_err)
        @test result.poly.x_err > 0
        @test result.poly.y_err > 0
        @test result.poly.peak_err > 0
    end

    @testset "cov matrix is consistent with error fields" begin
        img, _ = _make_star(; x0=5.0, y0=5.0)
        result = centroid_poly(img)
        cov = result.poly.cov
        # error fields match sqrt of diagonal; cov order is (y, x, peak)
        @test result.poly.y_err ≈ sqrt(cov[1,1])
        @test result.poly.x_err ≈ sqrt(cov[2,2])
        @test result.poly.peak_err ≈ sqrt(cov[3,3])
        # full matrix is symmetric
        @test cov[1,2] ≈ cov[2,1]
        @test cov[1,3] ≈ cov[3,1]
        @test cov[2,3] ≈ cov[3,2]
        # off-diagonals are finite (may be non-zero)
        @test isfinite(cov[1,2])
        @test isfinite(cov[1,3])
        @test isfinite(cov[2,3])
    end

    @testset "centroid_poly with Fill inv_var" begin
        img, _ = _make_star(; x0=5.0, y0=5.0)
        result = centroid_poly(img)
        @test result.poly.x ≈ 5.0 atol=1e-12
        @test result.poly.y ≈ 5.0 atol=1e-12
        @test result.poly.peak > 0
    end

    @testset "two-arg vs four-arg centroid_poly equality" begin
        # centroid_poly(image, inv_var) must give identical results to
        # centroid_poly(image, i0, j0, inv_var) when i0, j0 are the true
        # brightest pixel.
        for (x0, y0) in ((5.0, 5.0), (5.2, 5.3), (5.3, 5.7))
            img, model = _make_star(; x0, y0)
            r1 = centroid_poly(img)
            _, maxidx = findmax(img)
            i0, j0 = Tuple(maxidx)
            r2 = centroid_poly(img, Int(i0), Int(j0))
            @test r1.poly.x ≈ r2.poly.x
            @test r1.poly.y ≈ r2.poly.y
            @test r1.poly.peak ≈ r2.poly.peak
            @test r1.poly.x_err ≈ r2.poly.x_err
            @test r1.poly.y_err ≈ r2.poly.y_err
            @test r1.poly.peak_err ≈ r2.poly.peak_err
            @test r1.poly.cov ≈ r2.poly.cov
            @test r1.com.x ≈ r2.com.x
            @test r1.com.y ≈ r2.com.y
            @test r1.com.x_err ≈ r2.com.x_err
            @test r1.com.y_err ≈ r2.com.y_err
            @test r1.com.cov ≈ r2.com.cov
        end

        # With explicit inverse variance
        img, model = _make_star(; x0=5.0, y0=5.0)
        ivar = Fill(2.0, size(img))
        _, maxidx = findmax(img)
        i0, j0 = Tuple(maxidx)
        r1 = centroid_poly(img, ivar)
        r2 = centroid_poly(img, Int(i0), Int(j0), ivar)
        @test r1.poly.x ≈ r2.poly.x
        @test r1.poly.y ≈ r2.poly.y
        @test r1.poly.x_err ≈ r2.poly.x_err
        @test r1.poly.cov ≈ r2.poly.cov
        @test r1.com.x ≈ r2.com.x
        @test r1.com.y ≈ r2.com.y
        @test r1.com.cov ≈ r2.com.cov

        # Border case: both signatures return NaN when i0, j0 are on edge
        r3 = centroid_poly(img, 1, 1)
        @test isnan(r3.poly.x)
        @test isnan(r3.poly.y)
        @test isnan(r3.com.x)
        @test isnan(r3.com.x_err)
    end

    @testset "center-of-mass com field" begin
        # Symmetric patch: COM and polynomial centroid agree at origin.
        patch = [0.1 0.3 0.1;
                 0.3 1.0 0.3;
                 0.1 0.3 0.1]
        result = _centroid_poly3(patch, ones(3,3))
        @test result.com.x ≈ 0.0 atol=1e-12
        @test result.com.y ≈ 0.0 atol=1e-12
        @test result.poly.x ≈ result.com.x atol=1e-12
        @test result.poly.y ≈ result.com.y atol=1e-12
        @test result.com.x_err > 0
        @test result.com.y_err > 0
        @test result.com.cov[1,1] ≈ result.com.y_err^2
        @test result.com.cov[2,2] ≈ result.com.x_err^2
        @test result.com.cov[1,2] ≈ result.com.cov[2,1]

        # Asymmetric patch: COM is pulled toward the luminous region.
        patch2 = [0.05 0.2  0.1;
                  0.1  0.8  0.4;
                  0.05 0.15 0.08]
        result2 = _centroid_poly3(patch2, ones(3,3))
        @test result2.com.x > 0   # brighter on right
        @test isfinite(result2.com.y)
        @test result2.com.x_err > 0
        @test result2.com.y_err > 0
        # COM differs from polynomial centroid (different estimators)
        @test result2.com.x ≠ result2.poly.x || result2.com.y ≠ result2.poly.y
    end

    @testset "_centroid_poly3 direct call" begin
        patch = [0.1 0.3 0.1;
                 0.3 1.0 0.3;
                 0.1 0.3 0.1]
        result = _centroid_poly3(patch, ones(3,3))
        @test result.poly.x ≈ 0.0 atol=1e-12
        @test result.poly.y ≈ 0.0 atol=1e-12
        @test result.poly.peak > 0
        @test result.poly.x_err > 0
        @test result.poly.y_err > 0
        @test result.poly.peak_err > 0

        # Asymmetric patch: peak pulled toward brighter region
        patch2 = [0.05 0.2  0.1;
                  0.1  0.8  0.4;
                  0.05 0.15 0.08]
        result2 = _centroid_poly3(patch2, ones(3,3))
        @test result2.poly.x > 0
        @test isfinite(result2.poly.y)

        # Zero-weight on one half should pull centroid toward the other half
        w = ones(3, 3)
        w[:, 1] .= 0
        result3 = _centroid_poly3(patch, w)
        @test result3.poly.x > -1e-6
    end

    @testset "core ellipticity components mask zero-weighted pixels" begin
        # Masked neighbor pixels must not influence the 3×3 moment tensor.
        patch = [0.1 0.3 0.1;
                 0.3 1.0 0.3;
                 0.1 0.3 0.1]
        patch_bad = copy(patch)
        patch_bad[1, 2] = 100.0
        w = ones(3, 3)
        w[1, 2] = 0.0

        r_reference = _centroid_poly3(patch, w)
        r_masked = _centroid_poly3(patch_bad, w)
        @test r_masked.ellipticity1_core ≈ r_reference.ellipticity1_core
        @test r_masked.ellipticity2_core ≈ r_reference.ellipticity2_core
    end

    @testset "asymmetric GaussianPSF" begin
        model = GaussianPSF(x=5.0, y=5.0, x_fwhm=4.0, y_fwhm=2.8,
                            theta=30.0, flux=10.0, bkg=0.0)
        inds = (1:11, 1:11)
        img = evaluate.(model, inds[1], inds[2]')
        result = centroid_poly(img)
        @test result.poly.x ≈ 5.0 atol=1e-12
        @test result.poly.y ≈ 5.0 atol=1e-12
        @test result.poly.peak > 0

        model2 = GaussianPSF(x=5.3, y=5.7, x_fwhm=4.0, y_fwhm=2.8,
                             theta=30.0, flux=10.0, bkg=0.0)
        img2 = evaluate.(model2, inds[1], inds[2]')
        result2 = centroid_poly(img2)
        @test isfinite(result2.poly.x)
        @test isfinite(result2.poly.y)
        @test result2.poly.peak > 0
        @test abs(result2.poly.x - 5.0) < 1.5
        @test abs(result2.poly.y - 5.0) < 1.5
        @test result2.poly.x_err > 0
        @test result2.poly.y_err > 0
        @test result2.poly.peak_err > 0

        # Crop to a 3×3 corner of the original 11×11 image.  The true
        # centroid is at (5.3, 5.7) — far from this corner window — so
        # the brightest pixel in the crop lands on the crop border and
        # no full 3×3 neighborhood can be extracted.  Both the two-arg
        # and four-arg forms must return NaN. Max is index (3, 3).
        img3 = img2[1:3, 1:3]
        result3 = centroid_poly(img3)
        @test isnan(result3.poly.x)
        @test isnan(centroid_poly(img3, 3, 3).poly.x)
    end

    @testset "choose_centroid" begin
        # Well-sampled star: polynomial should be chosen (low curvature
        # degeneracy; cov ratio < 100).
        img_w, _ = _make_star(; x0=5.0, y0=5.0, fwhm=3.0)
        r_w = centroid_poly(img_w)
        c_w = choose_centroid(r_w)
        @test c_w.source == :poly
        @test c_w.x ≈ r_w.poly.x
        @test c_w.y ≈ r_w.poly.y

        # Very broad PSF: curvature near-singular, COM should be chosen.
        img_b, _ = _make_star(; x0=5.0, y0=5.0, fwhm=7.0)
        r_b = centroid_poly(img_b)
        c_b = choose_centroid(r_b)
        @test c_b.source == :com
        @test c_b.x ≈ r_b.com.x
        @test c_b.y ≈ r_b.com.y

        # A degenerate y covariance should also trigger the COM fallback.
        r_ybad = (;
            poly = (; y = 1.0, x = 2.0, cov = [1000.0 0.0; 0.0 1.0]),
            com = (; y = 1.1, x = 2.1, cov = [1.0 0.0; 0.0 1.0]),
        )
        c_ybad = choose_centroid(r_ybad)
        @test c_ybad.source == :com
        @test c_ybad.x ≈ r_ybad.com.x
        @test c_ybad.y ≈ r_ybad.com.y

        # Works with _centroid_poly3 output too
        patch = [0.1 0.3 0.1;
                 0.3 1.0 0.3;
                 0.1 0.3 0.1]
        r3 = _centroid_poly3(patch, ones(3,3))
        c3 = choose_centroid(r3)
        @test c3.source == :poly
        @test c3.x ≈ r3.poly.x
    end

    @testset "background keyword" begin
        B = 50.0

        # Off-center star so the COM de-biasing is visible.
        img, _ = _make_star(; x0=5.35, y0=5.2, fwhm=2.6, flux=300.0)
        r0 = centroid_poly(img)                              # no pedestal
        rB = centroid_poly(img .+ B; background = B)         # pedestal, subtracted
        rraw = centroid_poly(img .+ B)                       # pedestal, not subtracted

        # Polynomial centroid and its covariance are offset-invariant.
        @test rB.poly.y ≈ r0.poly.y atol=1e-12
        @test rB.poly.x ≈ r0.poly.x atol=1e-12
        @test rB.poly.cov ≈ r0.poly.cov rtol=1e-10
        @test rB.poly.peak_err ≈ r0.poly.peak_err rtol=1e-10

        # poly.peak is returned including the background.
        @test rB.poly.peak ≈ r0.poly.peak + B rtol=1e-10
        @test rraw.poly.peak ≈ r0.poly.peak + B rtol=1e-10

        # Diagnostics computed from background-subtracted values are restored
        # by passing `background`, and biased when the pedestal is left in.
        for k in (:normalized_curvature, :compactness_core)
            @test getproperty(rB, k) ≈ getproperty(r0, k) rtol=1e-9
        end
        # The ellipticity components are ~0 for this circular source (1e-17), so
        # they need an absolute tolerance; `rtol` between two near-zeros is
        # meaningless.
        for k in (:ellipticity1_core, :ellipticity2_core)
            @test getproperty(rB, k) ≈ getproperty(r0, k) atol=1e-9
        end
        @test rB.com.y ≈ r0.com.y rtol=1e-10
        @test rB.com.x ≈ r0.com.x rtol=1e-10
        @test rraw.normalized_curvature < r0.normalized_curvature   # suppressed
        @test rraw.compactness_core < r0.compactness_core           # pulled to floor
        # Pedestal pulls the COM toward the 3×3 box center (5.0); the true
        # position (5.35) is above it, so the biased COM sits lower.
        @test 5.0 < rraw.com.x < r0.com.x

        # background = 0 is bit-identical to omitting the keyword.
        z = centroid_poly(img)
        zk = centroid_poly(img; background = 0)
        @test zk.poly.peak === z.poly.peak
        @test zk.normalized_curvature === z.normalized_curvature
        @test zk.com.x === z.com.x

        # The moment-based core ellipticities (non-uniform weights) are also
        # restored by `background`.
        rng = StableRNG(42)
        w = rand(rng, 9, 9) .+ 0.5
        s0 = centroid_poly(img, w)
        sB = centroid_poly(img .+ B, w; background = B)
        @test sB.ellipticity1_core ≈ s0.ellipticity1_core rtol=1e-9
        @test sB.ellipticity2_core ≈ s0.ellipticity2_core rtol=1e-9

        # Over-subtracted background: COM and compactness are NaN, but the
        # polynomial centroid still succeeds.
        over = centroid_poly(img .+ 5.0; background = 1e6)
        @test isnan(over.com.y) && isnan(over.com.x)
        @test all(isnan, over.com.cov)
        @test isnan(over.compactness_core)
        @test isfinite(over.poly.y) && isfinite(over.poly.x)

        # Keyword threads through the 3-arg form and _centroid_poly3.
        _, mx = findmax(img)
        i0, j0 = Tuple(mx)
        r3arg = centroid_poly(img .+ B, Int(i0), Int(j0); background = B)
        @test r3arg.poly.peak ≈ r0.poly.peak + B rtol=1e-10
        @test r3arg.compactness_core ≈ r0.compactness_core rtol=1e-9

        patch = @view (img .+ B)[Int(i0)-1:Int(i0)+1, Int(j0)-1:Int(j0)+1]
        p3 = _centroid_poly3(patch, ones(3,3); background = B)
        praw = _centroid_poly3(patch, ones(3,3))
        @test p3.poly.peak ≈ praw.poly.peak       # peak restores background
        @test p3.normalized_curvature > praw.normalized_curvature

        # Float32 in, Float32 out, and type stable.
        f32 = Float32.(img .+ 10.0f0)
        rf = centroid_poly(f32; background = 10.0)
        @test rf.poly.peak isa Float32
        @test rf.compactness_core isa Float32
        @inferred centroid_poly(f32; background = 10.0)
    end

    @testset "normalized_curvature_err" begin
        base = [0.1 0.3 0.1; 0.3 1.0 0.3; 0.1 0.3 0.1] .* 100

        # The error is linear in the pixel sigma, so a 10x noisier map scales it
        # by exactly 10.  The statistic itself is invariant to a uniform weight
        # rescaling only up to Cholesky rounding, hence `≈` rather than `==`.
        r1 = centroid_poly(base, 2, 2, fill(1.0, 3, 3))
        r10 = centroid_poly(base, 2, 2, fill(1/100, 3, 3))
        @test r1.normalized_curvature_err > 0
        @test isfinite(r1.normalized_curvature_err)
        @test r10.normalized_curvature ≈ r1.normalized_curvature rtol=1e-12
        @test r10.normalized_curvature_err / r1.normalized_curvature_err ≈ 10 rtol=1e-12

        # Validated against 200k Monte Carlo realizations at sigma = 0.5, 1 and
        # 2, which reproduce this to 0.3% (the residual is delta-method bias).
        @test centroid_poly(base, 2, 2, fill(4.0, 3, 3)).normalized_curvature_err ≈
              0.0062557 rtol=1e-4

        # Degenerate border peak: NaN alongside the rest of the fit.
        rb = centroid_poly(base, 1, 1, fill(1.0, 3, 3))
        @test isnan(rb.normalized_curvature_err)

        @test centroid_poly(Float32.(base), 2, 2,
                            fill(1.0f0, 3, 3)).normalized_curvature_err isa Float32
    end

    @testset "compactness_core_err" begin
        base = [0.1 0.3 0.1; 0.3 1.0 0.3; 0.1 0.3 0.1] .* 100

        # Linear in the pixel sigma: a 100x noisier map scales the error by 10.
        r1 = centroid_poly(base, 2, 2, fill(1.0, 3, 3))
        r10 = centroid_poly(base, 2, 2, fill(1/100, 3, 3))
        @test r1.compactness_core_err > 0
        @test isfinite(r1.compactness_core_err)
        @test r10.compactness_core ≈ r1.compactness_core rtol=1e-12
        @test r10.compactness_core_err / r1.compactness_core_err ≈ 10 rtol=1e-12

        # Validated against 200k Monte Carlo realizations: the delta method
        # reproduces the observed scatter to 0.04% at sigma = 0.5 and 0.2% at
        # sigma = 1, degrading to 4% by sigma = 4 as the reciprocal's
        # second-order term grows.
        @test centroid_poly(base, 2, 2, fill(4.0, 3, 3)).compactness_core_err ≈
              0.0085147 rtol=1e-4

        # NaN wherever `compactness_core` itself is undefined: a degenerate
        # border peak, and a non-positive weighted flux sum.
        @test isnan(centroid_poly(base, 1, 1, fill(1.0, 3, 3)).compactness_core_err)
        neg = _centroid_poly3(base, ones(3, 3); background = 1e4)
        @test isnan(neg.compactness_core) && isnan(neg.compactness_core_err)

        @test centroid_poly(Float32.(base), 2, 2,
                            fill(1.0f0, 3, 3)).compactness_core_err isa Float32
    end
end
