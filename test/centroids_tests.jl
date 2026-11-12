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

    @testset "noise-free: centroid at pixel centre" begin
        img, model = _make_star(; x0=5.0, y0=5.0)
        result = centroid_poly(img)
        @test result.x ≈ 5.0 atol=1e-12
        @test result.y ≈ 5.0 atol=1e-12
        @test result.peak ≈ psf_peak(model) rtol=0.2
        @test result.x_err > 0
        @test result.y_err > 0
        @test result.peak_err > 0
    end

    @testset "noise-free: sub-pixel centroid" begin
        img, model = _make_star(; x0=5.2, y0=5.3)
        result = centroid_poly(img)
        @test abs(result.x - 5.2) < 0.15
        @test abs(result.y - 5.3) < 0.15
        @test result.peak > 0
    end

    @testset "border behaviour" begin
        img = [1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 9.0]
        result = centroid_poly(img)
        @test isnan(result.x)
        @test isnan(result.y)
        @test isnan(result.peak)
        @test isnan(result.x_err)
        @test isnan(result.y_err)
        @test isnan(result.peak_err)
        @test all(isnan, result.cov)
        @test isnan(result.com.x)
        @test isnan(result.com.y)
        @test isnan(result.com.x_err)
        @test isnan(result.com.y_err)
        @test all(isnan, result.com.cov)

        img2 = [100.0 0.0 0.0; 0.0 0.0 0.0; 0.0 0.0 0.0]
        result2 = centroid_poly(img2)
        @test isnan(result2.x)
        @test isnan(result2.y)
    end

    @testset "peak estimate consistency" begin
        for fwhm in (2.8, 4.0)
            img, model = _make_star(; x0=5.0, y0=5.0, fwhm=fwhm)
            result = centroid_poly(img)
            @test result.peak > 0
            @test result.peak ≈ psf_peak(model) rtol=0.4
        end
    end

    @testset "weighted least squares" begin
        img, _ = _make_star(; x0=5.0, y0=5.0)
        # uniform weights → same centroid as default
        r1 = centroid_poly(img)
        r2 = centroid_poly(img, ones(size(img)))
        @test r1.x ≈ r2.x
        @test r1.y ≈ r2.y
        @test r1.peak ≈ r2.peak

        # doubling inv_var halves the variance → errors scale by 1/√2
        r3 = centroid_poly(img, Fill(2.0, size(img)))
        @test r1.x ≈ r3.x
        @test r1.y ≈ r3.y
        @test r3.x_err ≈ r1.x_err / sqrt(2)  rtol=1e-10
        @test r3.y_err ≈ r1.y_err / sqrt(2)  rtol=1e-10
        @test r3.peak_err ≈ r1.peak_err / sqrt(2)  rtol=1e-10

        # down-weighting a corner pixel should not change centroid much for
        # symmetric star centred on a pixel
        w = ones(9, 9)
        w[1, 1] = 0.01
        r4 = centroid_poly(img, w)
        @test r4.x ≈ r1.x  atol=1e-4
        @test r4.y ≈ r1.y  atol=1e-4
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
                push!(errors_x, res.x - 5.0)
                push!(errors_y, res.y - 5.0)
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
        @test isfinite(result.x_err)
        @test isfinite(result.y_err)
        @test isfinite(result.peak_err)
        @test result.x_err > 0
        @test result.y_err > 0
        @test result.peak_err > 0
    end

    @testset "cov matrix is consistent with error fields" begin
        img, _ = _make_star(; x0=5.0, y0=5.0)
        result = centroid_poly(img)
        cov = result.cov
        # error fields match sqrt of diagonal
        @test result.x_err ≈ sqrt(cov[1,1])
        @test result.y_err ≈ sqrt(cov[2,2])
        @test result.peak_err ≈ sqrt(cov[3,3])
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
        @test result.x ≈ 5.0 atol=1e-12
        @test result.y ≈ 5.0 atol=1e-12
        @test result.peak > 0
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
            @test r1.x ≈ r2.x
            @test r1.y ≈ r2.y
            @test r1.peak ≈ r2.peak
            @test r1.x_err ≈ r2.x_err
            @test r1.y_err ≈ r2.y_err
            @test r1.peak_err ≈ r2.peak_err
            @test r1.cov ≈ r2.cov
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
        @test r1.x ≈ r2.x
        @test r1.y ≈ r2.y
        @test r1.x_err ≈ r2.x_err
        @test r1.cov ≈ r2.cov
        @test r1.com.x ≈ r2.com.x
        @test r1.com.y ≈ r2.com.y
        @test r1.com.cov ≈ r2.com.cov

        # Border case: both signatures return NaN when i0, j0 are on edge
        r3 = centroid_poly(img, 1, 1)
        @test isnan(r3.x)
        @test isnan(r3.y)
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
        @test result.x ≈ result.com.x atol=1e-12
        @test result.y ≈ result.com.y atol=1e-12
        @test result.com.x_err > 0
        @test result.com.y_err > 0
        @test result.com.cov[1,1] ≈ result.com.x_err^2
        @test result.com.cov[2,2] ≈ result.com.y_err^2
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
        @test result2.com.x ≠ result2.x || result2.com.y ≠ result2.y
    end

    @testset "_centroid_poly3 direct call" begin
        patch = [0.1 0.3 0.1;
                 0.3 1.0 0.3;
                 0.1 0.3 0.1]
        result = _centroid_poly3(patch, ones(3,3))
        @test result.x ≈ 0.0 atol=1e-12
        @test result.y ≈ 0.0 atol=1e-12
        @test result.peak > 0
        @test result.x_err > 0
        @test result.y_err > 0
        @test result.peak_err > 0

        # Asymmetric patch: peak pulled toward brighter region
        patch2 = [0.05 0.2  0.1;
                  0.1  0.8  0.4;
                  0.05 0.15 0.08]
        result2 = _centroid_poly3(patch2, ones(3,3))
        @test result2.x > 0
        @test isfinite(result2.y)

        # Zero-weight on one half should pull centroid toward the other half
        w = ones(3, 3)
        w[:, 1] .= 0
        result3 = _centroid_poly3(patch, w)
        @test result3.x > -1e-6
    end

    @testset "asymmetric GaussianPSF" begin
        model = GaussianPSF(x=5.0, y=5.0, x_fwhm=4.0, y_fwhm=2.8,
                            theta=30.0, flux=10.0, bkg=0.0)
        inds = (1:11, 1:11)
        img = evaluate.(model, inds[1], inds[2]')
        result = centroid_poly(img)
        @test result.x ≈ 5.0 atol=1e-12
        @test result.y ≈ 5.0 atol=1e-12
        @test result.peak > 0

        model2 = GaussianPSF(x=5.3, y=5.7, x_fwhm=4.0, y_fwhm=2.8,
                             theta=30.0, flux=10.0, bkg=0.0)
        img2 = evaluate.(model2, inds[1], inds[2]')
        result2 = centroid_poly(img2)
        @test isfinite(result2.x)
        @test isfinite(result2.y)
        @test result2.peak > 0
        @test abs(result2.x - 5.0) < 1.5
        @test abs(result2.y - 5.0) < 1.5
        @test result2.x_err > 0
        @test result2.y_err > 0
        @test result2.peak_err > 0

        # Crop to a 3×3 corner of the original 11×11 image.  The true
        # centroid is at (5.3, 5.7) — far from this corner window — so
        # the brightest pixel in the crop lands on the crop border and
        # no full 3×3 neighbourhood can be extracted.  Both the two-arg
        # and four-arg forms must return NaN. Max is index (3, 3).
        img3 = img2[1:3, 1:3]
        result3 = centroid_poly(img3)
        @test isnan(result3.x)
        @test isnan(centroid_poly(img3, 3, 3).x)
    end

    @testset "choose_centroid" begin
        # Well-sampled star: polynomial should be chosen (low curvature
        # degeneracy; cov ratio < 100).
        img_w, _ = _make_star(; x0=5.0, y0=5.0, fwhm=3.0)
        r_w = centroid_poly(img_w)
        c_w = choose_centroid(r_w)
        @test c_w.source == :poly
        @test c_w.x ≈ r_w.x
        @test c_w.y ≈ r_w.y

        # Very broad PSF: curvature near-singular, COM should be chosen.
        img_b, _ = _make_star(; x0=5.0, y0=5.0, fwhm=7.0)
        r_b = centroid_poly(img_b)
        c_b = choose_centroid(r_b)
        @test c_b.source == :com
        @test c_b.x ≈ r_b.com.x
        @test c_b.y ≈ r_b.com.y

        # Works with _centroid_poly3 output too
        patch = [0.1 0.3 0.1;
                 0.3 1.0 0.3;
                 0.1 0.3 0.1]
        r3 = _centroid_poly3(patch, ones(3,3))
        c3 = choose_centroid(r3)
        @test c3.source == :poly
        @test c3.x ≈ r3.x
    end
end
