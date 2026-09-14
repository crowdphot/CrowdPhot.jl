using CrowdPhot: measure_star_shape, measure_star_shape_ref, _moments2, matched_filter,
    MatchedFilterResult, centroid_poly, choose_centroid, measure_star_shapes,
    FlatWindow, GaussianWindow, deconvolve_moments, inv_window_var, yfactor, xfactor
using CrowdPhot.PSF: CircularGaussianPSF, GaussianPSF, CircularGaussianPRF, CircularMoffatPSF,
    evaluate, add_star!, fwhm as psf_fwhm
using FillArrays: Fill
using LinearAlgebra: I
using StableRNGs: StableRNG
using Statistics: median, std
using Test

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

function _make_gaussian_cutout(; x0=5.0, y0=5.0, fwhm=2.8, flux=100.0, shape=(9,9))
    inds = (1:shape[1], 1:shape[2])
    model = CircularGaussianPSF(x=x0, y=y0, fwhm=fwhm, flux=flux, bkg=0.0)
    img = evaluate.(model, inds[1], inds[2]')
    return img, model
end

function _make_elliptical_gaussian(; x0=5.0, y0=5.0, x_fwhm=3.0, y_fwhm=1.5, theta=30.0, flux=100.0, shape=(11,11))
    inds = (1:shape[1], 1:shape[2])
    model = GaussianPSF(x=x0, y=y0, x_fwhm=x_fwhm, y_fwhm=y_fwhm, theta=theta, flux=flux, bkg=0.0)
    img = evaluate.(model, inds[1], inds[2]')
    return img, model
end

# ---------------------------------------------------------------------------
# _moments2
# ---------------------------------------------------------------------------

@testset "_moments2" begin

    @testset "noise-free Gaussian — positive moments" begin
        img, _ = _make_gaussian_cutout()
        inv_var = Fill(1.0, size(img))
        mom = _moments2(img, inv_var, 0.0, 5.0, 5.0)
        @test mom.M00 > 0
        @test abs(mom.M10) < 1.0    # centroid near reference point
        @test abs(mom.M01) < 1.0
        @test mom.M20 > 0
        @test mom.M02 > 0
        @test mom.M11 > -1e-6       # circular => near zero
    end

    @testset "zero background — all flux captured" begin
        img, _ = _make_gaussian_cutout(; flux=200.0, fwhm=2.0)
        mom = _moments2(img, Fill(1.0, size(img)), 0.0, 5.0, 5.0)
        @test mom.M00 ≈ 200.0 rtol=0.01
    end

    @testset "rectangular aperture diagnostics" begin
        # Aperture sums are signed residual sums over pixels with inv_var > 0.
        img = [1.0 4.0; 3.0 5.0]
        inv_var = [1.0 4.0; 1.0 0.0]
        mom = _moments2(img, inv_var, 2.0, 1.0, 1.0)
        @test mom.aperture_sum == 2.0
        @test mom.aperture_area == 3
        @test mom.aperture_sum_err ≈ sqrt(1.0 + 0.25 + 1.0)
        # Signed: z = -1, 2, 1 over the three unmasked pixels, so
        # M00 = 1*(-1) + 4*2 + 1*1 = 8.  There is no positivity clip, which
        # would drop the z = -1 pixel and give 9.
        @test mom.M00 == 8.0
    end

    @testset "background subtraction" begin
        img, _ = _make_gaussian_cutout(; flux=100.0, fwhm=2.0)
        # All pixels > 0, so bg=0 and bg=50 should give different M00
        mom0 = _moments2(img, Fill(1.0, size(img)), 0.0, 5.0, 5.0)
        mom50 = _moments2(img, Fill(1.0, size(img)), 50.0, 5.0, 5.0)
        @test mom0.M00 > mom50.M00
    end

    @testset "zero-weight pixels are skipped" begin
        img, _ = _make_gaussian_cutout()
        w = ones(size(img))
        w[5, 5] = 0.0  # mask the peak
        mom_masked = _moments2(img, w, 0.0, 5.0, 5.0)
        mom_full = _moments2(img, Fill(1.0, size(img)), 0.0, 5.0, 5.0)
        @test mom_masked.M00 < mom_full.M00
    end

    @testset "all pixels below background — M00 non-positive" begin
        # Signed moments make this -225 rather than 0 (25 pixels at z = -9,
        # unit weight).  What matters is the invariant the callers rely on:
        # M00 <= 0 means no usable moments, and `measure_star_shape` must turn
        # that into NaN rather than a number.
        img = fill(1.0, 5, 5)
        mom = _moments2(img, Fill(1.0, size(img)), 10.0, 3.0, 3.0)
        @test mom.M00 == -225.0
        @test mom.M00 <= 0
        r = measure_star_shape(img, 3, 3; background = 10.0)
        @test isnan(r.compactness_aperture)
        @test isnan(r.fwhm.y) && isnan(r.fwhm.x)
        @test isnan(r.ellipticity1_aperture) && isnan(r.ellipticity2_aperture)
        @test isnan(r.centroid.y) && isnan(r.centroid.x)
    end

    @testset "single pixel" begin
        img = zeros(5, 5)
        img[3, 3] = 10.0
        mom = _moments2(img, Fill(1.0, size(img)), 0.0, 3.0, 3.0)
        @test mom.M00 == 10.0
        @test mom.M10 == 0.0
        @test mom.M01 == 0.0
        @test mom.M20 == 0.0
        @test mom.M02 == 0.0
        @test mom.M11 == 0.0
        # The W sums are the variance bookkeeping for the M sums, so they run
        # over exactly the pixels M runs over -- every unmasked pixel, not just
        # the bright one.  W00 = 25 unit weights; W20 = sum(dy^2) = 50 over the
        # 5x5.  Under the old positive-only clip only the single z > 0 pixel contributed.
        @test mom.W00 == 25.0
        @test mom.W20 == 50.0
        @test mom.W02 == 50.0
    end

    @testset "moment windows" begin
        img, _ = _make_gaussian_cutout(; flux = 500.0, fwhm = 2.5, shape = (15, 15))
        iv = [0.4 + 0.02 * (i + j) for i in 1:15, j in 1:15]

        # The default is the untapered limit, and passing it explicitly is a no-op.
        @test _moments2(img, iv, 0.0, 8, 8) === _moments2(img, iv, 0.0, 8, 8, FlatWindow())

        w = GaussianWindow(2.5)
        mf = _moments2(img, iv, 0.0, 8, 8, FlatWindow())
        mw = _moments2(img, iv, 0.0, 8, 8, w)

        # A taper strictly reduces every positive-definite accumulator.
        @test mw.M00 < mf.M00
        @test mw.W00 < mf.W00
        # `aperture_sum` is unwindowed by design, so it must not move at all.
        @test mw.aperture_sum == mf.aperture_sum
        @test mw.aperture_area == mf.aperture_area
        @test mw.aperture_sum_err == mf.aperture_sum_err

        # The subtle contract: moments carry w*g, their variances carry w*g^2.
        # Checked against the tabulated factors directly, since getting this
        # wrong mis-scales every propagated uncertainty while leaving the
        # values correct.
        expect_M00 = sum(iv[i, j] * yfactor(w, i - 8) * xfactor(w, j - 8) * img[i, j]
                         for i in 1:15, j in 1:15)
        expect_W00 = sum(iv[i, j] * (yfactor(w, i - 8) * xfactor(w, j - 8))^2
                         for i in 1:15, j in 1:15)
        @test mw.M00 ≈ expect_M00 rtol=1e-12
        @test mw.W00 ≈ expect_W00 rtol=1e-12
        @test !isapprox(mw.W00, sum(iv[i, j] * yfactor(w, i - 8) * xfactor(w, j - 8)
                                    for i in 1:15, j in 1:15); rtol=1e-3)

        # Support: unit at the anchor, negligible at the edge, exactly zero past it.
        @test yfactor(w, 0) == 1
        @test 0 < yfactor(w, w.hw) < 1e-5
        @test yfactor(w, w.hw + 1) == 0
        @test yfactor(w, -w.hw - 100) == 0
        # FlatWindow factors are type-preserving.
        @test 1.0f0 * yfactor(FlatWindow(), 3) isa Float32

        # Deconvolution: the bitwise identity for flat, exact for a Gaussian pair.
        a0, b0, c0 = 1.2345678901234567, 0.98765432109876, -0.5555555555555
        f0 = deconvolve_moments(FlatWindow(), a0, b0, c0)
        @test f0.yy === a0 && f0.xx === b0 && f0.xy === c0
        @test inv_window_var(FlatWindow()) == 0

        sw2 = 1.1^2
        wg = GaussianWindow(1.1 * 2 * sqrt(2 * log(2)))
        @test inv_window_var(wg) ≈ inv(sw2) rtol=1e-12
        # Recovered exactly for any source tensor, *including* a rotated one.
        # Deconvolving the marginals separately -- which is what this replaced --
        # is wrong by 2% at a moment correlation of 0.25 and 11% at 0.53.
        for Ss in ([3.0 0.0; 0.0 3.0], [4.0 0.0; 0.0 2.0],
                   [4.0 0.7; 0.7 2.0], [4.0 1.5; 1.5 2.0])
            Sm = inv(inv(Ss) + I / sw2)
            r = deconvolve_moments(wg, Sm[1, 1], Sm[2, 2], Sm[1, 2])
            @test r.yy ≈ Ss[1, 1] rtol=1e-10
            @test r.xx ≈ Ss[2, 2] rtol=1e-10
            @test r.xy ≈ Ss[1, 2] atol=1e-10
        end
        # No solution -> NaN for all three, never a partial answer.
        for args in ((1.0, 1.0, 2.0),      # measured tensor not positive definite
                     (50.0, 50.0, 0.0),    # source at least as broad as the window
                     (0.0, 0.0, 0.0))      # no signal
            @test all(isnan, values(deconvolve_moments(wg, args...)))
        end

        # Every size and shape statistic is absolute, window or not, because the
        # window is divided back out.  On a Gaussian source the deconvolution is
        # exact, so tapering changes the reported values hardly at all even
        # though it changes the moments underneath a lot (checked above).
        rf = measure_star_shape(img, 8, 8; inv_var = iv)
        rw = measure_star_shape(img, 8, 8; inv_var = iv, window = GaussianWindow(2.5))
        @test rw.fwhm.y ≈ rf.fwhm.y rtol=0.02
        @test rw.fwhm.x ≈ rf.fwhm.x rtol=0.02
        @test rw.compactness_aperture ≈ rf.compactness_aperture rtol=0.05
        @test rw.ellipticity1_aperture ≈ rf.ellipticity1_aperture atol=0.02
        @test rw.ellipticity2_aperture ≈ rf.ellipticity2_aperture atol=0.02

        # Degenerate widths are rejected rather than silently unwindowed.
        @test_throws "must be finite and positive" GaussianWindow(0.0)
        @test_throws "must be finite and positive" GaussianWindow(-1.0)
        @test_throws "must be finite and positive" GaussianWindow(NaN)

        @test GaussianWindow(2.5f0).inv_var_w isa Float32
        @test measure_star_shape(Float32.(img), 8, 8;
                                 inv_var = Float32.(iv),
                                 window = GaussianWindow(2.5f0)).fwhm.y isa Float32
    end
end

# ---------------------------------------------------------------------------
# measure_star_shape — core method
# ---------------------------------------------------------------------------

@testset "measure_star_shape" begin

    @testset "noise-free circular Gaussian" begin
        # Large cutout centered on the star to avoid edge truncation bias.
        img, model = _make_gaussian_cutout(; x0=8.0, y0=8.0, fwhm=2.8, shape=(17,17))
        result = measure_star_shape(img, 8, 8; background=0)
        y_fwhm_model, x_fwhm_model = psf_fwhm(model)
        @test result.fwhm.y ≈ y_fwhm_model rtol=0.03
        @test result.fwhm.x ≈ x_fwhm_model rtol=0.03
        # theta is undefined for a perfectly circular PSF; just check finite.
        @test isfinite(result.fwhm.theta)
        @test abs(result.ellipticity1_aperture) < 1e-4   # ~0 for symmetric
        @test abs(result.ellipticity2_aperture) < 1e-4
        @test result.moment_norm > 0
        @test result.centroid.y ≈ 8.0 atol=0.5
        @test result.centroid.x ≈ 8.0 atol=0.5
    end

    @testset "noise-free elliptical Gaussian" begin
        # Large cutout so Gaussian wings are not truncated.
        img, model = _make_elliptical_gaussian(;
            x0=10.0, y0=10.0, x_fwhm=3.0, y_fwhm=1.5, theta=30.0, shape=(21,21))
        result = measure_star_shape(img, 10, 10; background=0)
        # y_fwhm < x_fwhm for this model
        @test result.fwhm.y < result.fwhm.x
        # Moment-based FWHM approximates the Gaussian FWHM; tolerances
        # are loose because finite-aperture moments differ from the
        # analytic infinite-integral FWHM.
        @test result.fwhm.y ≈ 1.5 rtol=0.35
        @test result.fwhm.x ≈ 3.0 rtol=0.35
        @test abs(result.fwhm.theta - 30.0) < 15.0
        # Extended in x → e1 negative.
        @test result.ellipticity1_aperture < -0.1
    end

    @testset "constant image — NaN" begin
        # Shape fields are undefined, but aperture diagnostics still report the cutout sum.
        img = fill(5.0, 7, 7)
        result = measure_star_shape(img, 4, 4; background=10)
        @test isnan(result.fwhm.y)
        @test isnan(result.fwhm.x)
        @test isnan(result.fwhm.theta)
        @test isnan(result.ellipticity1_aperture)
        @test isnan(result.ellipticity2_aperture)
        @test result.aperture_sum == -245.0
        @test result.aperture_area == 49
        @test result.aperture_sum_err ≈ 7.0
        @test isnan(result.centroid.y)
        @test isnan(result.centroid.x)
    end

    @testset "single bright pixel — near-zero width" begin
        # A point-like source has zero moment width but a well-defined cutout sum.
        img = zeros(7, 7)
        img[4, 4] = 100.0
        result = measure_star_shape(img, 4, 4; background=0)
        # Single pixel has zero spatial extent → FWHM NaN.
        @test isnan(result.fwhm.y)
        @test isnan(result.fwhm.x)
        # Zero-width → degenerate: the moment sum vanishes, so the shape
        # components are undefined rather than isotropic.
        @test isnan(result.ellipticity1_aperture)
        @test isnan(result.ellipticity2_aperture)
        @test result.moment_norm == 100.0
        @test result.aperture_sum == 100.0
        @test result.aperture_area == 49
        @test result.aperture_sum_err ≈ 7.0
        @test result.centroid.y ≈ 4.0
        @test result.centroid.x ≈ 4.0
    end

    @testset "zero-weight pixels" begin
        # Masked pixels are excluded from both moment and aperture diagnostics.
        img, _ = _make_gaussian_cutout()
        w = ones(size(img))
        w[5, 5] = 0.0
        result_full = measure_star_shape(img, 5, 5)
        result_masked = measure_star_shape(img, 5, 5; inv_var=w)
        @test result_masked.moment_norm < result_full.moment_norm
        @test result_masked.aperture_sum < result_full.aperture_sum
        @test result_masked.aperture_area == result_full.aperture_area - 1
        @test result_masked.aperture_sum_err ≈ sqrt(result_full.aperture_area - 1)
    end

    @testset "sub-pixel centroid via measure_star_shape convenience" begin
        img, _ = _make_gaussian_cutout(; x0=5.2, y0=5.3, shape=(15,15))
        result = measure_star_shape(img)
        # Convenience method finds the peak then calls the core.
        @test result.fwhm.y > 0
        @test result.fwhm.x > 0
        @test abs(result.ellipticity1_aperture) < 1e-4
        @test abs(result.ellipticity2_aperture) < 1e-4
    end

    @testset "Float32 precision" begin
        # Public scalar diagnostics preserve the floating-point type of the image.
        img_f32 = Float32[0.1 0.3 0.1; 0.3 1.0 0.3; 0.1 0.3 0.1]
        result = measure_star_shape(img_f32; background=0)
        @test result.fwhm.y > 0
        @test result.fwhm.x > 0
        @test result.ellipticity1_aperture isa Float32
        @test result.ellipticity2_aperture isa Float32
        @test result.moment_norm isa Float32
        @test result.aperture_sum isa Float32
        @test result.aperture_sum_err isa Float32
    end

    @testset "shift invariance — integer pixel" begin
        # A shifted cutout of the same star should give the same FWHM.
        img1, _ = _make_gaussian_cutout(; x0=5.0, y0=5.0, shape=(15,15))
        img2, _ = _make_gaussian_cutout(; x0=7.0, y0=7.0, shape=(15,15))
        r1 = measure_star_shape(img1, 5, 5)
        r2 = measure_star_shape(img2, 7, 7)
        @test r1.fwhm.y ≈ r2.fwhm.y rtol=0.01
        @test r1.fwhm.x ≈ r2.fwhm.x rtol=0.01
    end

    @testset "theta is finite for circular PSF" begin
        img, _ = _make_gaussian_cutout(; x0=10.0, y0=10.0, shape=(21,21))
        result = measure_star_shape(img, 10, 10)
        @test isfinite(result.fwhm.theta)
        @test abs(result.ellipticity1_aperture) < 0.15   # ~0 for circular
    end

    @testset "cosmic ray — sharp star comparison" begin
        # A cosmic ray (single bright pixel) should have extreme sharpness
        # from centroid_poly.  Here we test that a normal star has moderate
        # sharpness and the single-pixel case is handled.
        img_star, _ = _make_gaussian_cutout(; fwhm=2.8, shape=(9,9))
        img_cr = zeros(9, 9)
        img_cr[5, 5] = 100.0

        # measure_star_shape reports NaN FWHM for single pixel (zero width)
        r_star = measure_star_shape(img_star, 5, 5; background=0)
        r_cr = measure_star_shape(img_cr, 5, 5; background=0)
        @test r_star.fwhm.y > 0
        @test isnan(r_cr.fwhm.y)   # zero-width → NaN
    end

    @testset "centroid covariance" begin
        img, _ = _make_gaussian_cutout(; x0=8.0, y0=8.0, shape=(17,17))
        result = measure_star_shape(img, 8, 8; background=0)
        @test result.centroid.y_err > 0
        @test result.centroid.x_err > 0
        @test result.centroid.cov[1,1] ≈ result.centroid.y_err^2
        @test result.centroid.cov[2,2] ≈ result.centroid.x_err^2
        @test result.centroid.cov[1,2] ≈ result.centroid.cov[2,1]
    end

    @testset "asymmetric — one-sided feature" begin
        # A spike along the +x axis is pure x-elongation, so it registers in
        # e1 and leaves e2 (the 45-degree component) alone.
        img, _ = _make_gaussian_cutout(; x0=5.0, y0=5.0, fwhm=2.0, shape=(9,9))
        r_sym = measure_star_shape(img, 5, 5; background=0)
        @test r_sym.ellipticity1_aperture ≈ 0 atol=1e-10
        @test r_sym.ellipticity2_aperture ≈ 0 atol=1e-10
        # Add a diffraction-spike-like feature to the right side.
        img[5, 7] += 50.0
        img[5, 8] += 30.0
        r_asym = measure_star_shape(img, 5, 5; background=0)
        @test r_asym.ellipticity1_aperture < -0.05   # extended in x
        @test r_asym.ellipticity2_aperture ≈ 0 atol=1e-10  # no diagonal power
    end

    @testset "e1/e2 divergence — symmetric diagonal pair" begin
        # Flux on the same diagonal (top-left + bottom-right) keeps
        # M20 ≈ M02, so e1 stays ~0 while the cross moment — and therefore
        # e2 — becomes positive.  This is where the two components provide
        # complementary information
        img, _ = _make_gaussian_cutout(; x0=5.0, y0=5.0, fwhm=2.0, shape=(9,9))
        img[3, 3] += 80.0  # top-left
        img[7, 7] += 80.0  # bottom-right
        r = measure_star_shape(img, 5, 5; background=0)
        @test r.ellipticity1_aperture ≈ 0 atol = 1e-10  # e1 ~0 (σ² balanced)
        @test r.ellipticity2_aperture > 0.5   # +45° diagonal power
    end

    @testset "asymmetric elliptical Gaussian (e1)" begin
        # Axis-aligned ellipse: e1 registers the elongation and its sign
        # gives the direction, while e2 stays zero because there is no power
        # on the diagonals.
        #
        # Extended in x → e1 negative.
        img, _ = _make_elliptical_gaussian(; x_fwhm=4.0, y_fwhm=2.0, theta=0.0,
            x0=10.0, y0=10.0, shape=(21,21))
        r = measure_star_shape(img, 10, 10; background=0)
        @test r.ellipticity1_aperture < -0.2  # e1: x-elongation
        @test r.ellipticity2_aperture ≈ 0 atol=1e-10  # axis-aligned ⇒ no e2
        @test r.fwhm.x > r.fwhm.y

        # Extended in y → e1 positive.
        img2, _ = _make_elliptical_gaussian(; x_fwhm=2.0, y_fwhm=4.0, theta=0.0,
            x0=10.0, y0=10.0, shape=(21,21))
        r2 = measure_star_shape(img2, 10, 10; background=0)
        @test r2.ellipticity1_aperture > 0.2
        @test r2.ellipticity2_aperture ≈ 0 atol=1e-10
        @test r2.fwhm.y > r2.fwhm.x
    end

    @testset "ellipticity sign agreement (core vs aperture)" begin
        # Core and aperture estimates of the same axis should agree in sign.
        using CrowdPhot: centroid_poly
        img, _ = _make_elliptical_gaussian(; x_fwhm=3.0, y_fwhm=1.5, theta=0.0,
            x0=10.0, y0=10.0, shape=(21,21))
        cent = centroid_poly(img)
        shape = measure_star_shape(img, 10, 10; background=0)
        @test sign(cent.ellipticity1_core) == sign(shape.ellipticity1_aperture)
        # The source is axis-aligned, so e2 is legitimately zero at both
        # scales; comparing signs of a zero is meaningless, check magnitude.
        @test abs(cent.ellipticity2_core) < 1e-10
        @test abs(shape.ellipticity2_aperture) < 1e-10
    end

    @testset "broad elliptical PSF — core vs aperture ellipticity" begin
        using CrowdPhot: centroid_poly
        # Elliptical Gaussian with broad FWHM: both core and aperture
        # detect the ellipticity, but the 3×3 core is strongly compressed.
        # A moment tensor confined to a ±1 px box saturates once the source
        # is much broader than the box, so `ellipticity1_core` retains only
        # a fraction of the aperture signal for a very broad PSF.  This is
        # the cost of the moment basis; the curvature-based statistic it
        # replaced kept more dynamic range here but was ~5x noisier in
        # sub-pixel phase.  Use the aperture value when the PSF is broad.
        model = GaussianPSF(x=16.0, y=16.0, x_fwhm=6.0, y_fwhm=3.0,
            theta=0.0, flux=1000.0, bkg=0.0)
        img = evaluate.(model, 1:31, (1:31)')
        cent = centroid_poly(img)
        shape = measure_star_shape(img, 16, 16; background=0)
        @test cent.ellipticity1_core < -0.02       # x-extended (compressed)
        @test shape.ellipticity1_aperture < -0.5   # x-extended
        @test shape.fwhm.x > shape.fwhm.y
    end

    @testset "background over-subtraction is caught, not absorbed" begin
        img, _ = _make_gaussian_cutout(; x0=5.0, y0=5.0, flux=200.0, fwhm=2.0, shape=(9,9))

        # Moments are signed, so over-subtracting the background drives
        # `moment_norm` negative instead of quietly zeroing the offending pixels.
        # That is the safer failure: the guard fires and every shape statistic
        # comes back NaN, where the old positivity clip would have built a
        # plausible-looking measurement out of only the surviving bright pixels.
        for bg in (5, 1000)
            r = measure_star_shape(img, 5, 5; background = bg)
            @test r.moment_norm < 0
            @test isnan(r.fwhm.y) && isnan(r.fwhm.x)
            @test isnan(r.compactness_aperture)
            @test isnan(r.ellipticity1_aperture)
        end

        # A correct background leaves the measurement intact: with unit weights
        # `moment_norm` is just the summed signal, so it recovers the input flux.
        # (Slightly *above* it here, since `evaluate` samples the analytic
        # profile rather than integrating over each pixel.)
        r_ok = measure_star_shape(img, 5, 5; background = 0)
        @test r_ok.moment_norm ≈ 200.0 rtol=1e-3
        @test isfinite(r_ok.fwhm.y)
    end

    @testset "noisy image — ellipticity bounded" begin
        using StableRNGs: StableRNG
        rng = StableRNG(42)
        img, _ = _make_gaussian_cutout(; x0=3.5, y0=3.5, fwhm=2.8, flux=200.0, shape=(7,7))
        noisy = img .+ 5.0 .* randn(rng, size(img))
        r = measure_star_shape(noisy, 4, 4; background=0)
        @test isfinite(r.ellipticity1_aperture)
        @test isfinite(r.ellipticity2_aperture)
        @test r.fwhm.y > 0
        @test r.fwhm.x > 0
        @test r.centroid.y_err > 0
        @test r.centroid.x_err > 0
    end

    @testset "shape uncertainties" begin
        img, _ = _make_elliptical_gaussian(; flux=3000.0, shape=(11,11))
        iv = Fill(1 / 20.0, size(img))

        @testset "finite and positive on a clean measurement" begin
            for win in (FlatWindow(), GaussianWindow(3.0))
                r = measure_star_shape(img, 6, 6; inv_var=iv, background=0, window=win)
                @test r.ellipticity1_aperture_err > 0
                @test r.ellipticity2_aperture_err > 0
                @test r.compactness_aperture_err > 0
                @test all(isfinite, (r.ellipticity1_aperture_err,
                                     r.ellipticity2_aperture_err,
                                     r.compactness_aperture_err))
            end
        end

        @testset "errors scale as 1/flux at fixed sky" begin
            # The statistics are ratios of moments, so at fixed noise the
            # fractional moment error, and hence every shape error, is ~1/flux.
            e_lo = measure_star_shape(img, 6, 6; inv_var=iv, background=0)
            bright, _ = _make_elliptical_gaussian(; flux=30000.0, shape=(11,11))
            e_hi = measure_star_shape(bright, 6, 6; inv_var=iv, background=0)
            for k in (:ellipticity1_aperture_err, :ellipticity2_aperture_err,
                      :compactness_aperture_err)
                @test getfield(e_lo, k) / getfield(e_hi, k) ≈ 10 rtol=0.05
            end
        end

        @testset "NaN wherever the statistic is NaN" begin
            r = measure_star_shape(fill(1.0, 5, 5), 3, 3; background=1.0)
            @test isnan(r.compactness_aperture) && isnan(r.compactness_aperture_err)
            @test isnan(r.ellipticity1_aperture) && isnan(r.ellipticity1_aperture_err)
            @test isnan(r.ellipticity2_aperture) && isnan(r.ellipticity2_aperture_err)
            single = zeros(5, 5); single[3, 3] = 10.0
            rs = measure_star_shape(single, 3, 3; background=0)
            @test isnan(rs.compactness_aperture) && isnan(rs.compactness_aperture_err)
        end

        @testset "element type follows the image" begin
            r = measure_star_shape(Float32.(img), 6, 6; background=0f0)
            @test r.ellipticity1_aperture_err isa Float32
            @test r.ellipticity2_aperture_err isa Float32
            @test r.compactness_aperture_err isa Float32
        end

        @testset "predicted errors match the Monte Carlo scatter" begin
            # A round-trip check on the whole chain: moment covariance from the
            # W accumulators, centralization, window deconvolution, and the
            # three scalar Jacobians.  The windowed case is the one that
            # exercises the deconvolution Jacobian (inv_window_var != 0).
            sky = 20.0
            truth, _ = _make_elliptical_gaussian(; y_fwhm=3.4, x_fwhm=2.2,
                                                 theta=30.0, flux=3.0e4, shape=(11,11))
            for (win, tol) in ((FlatWindow(), 0.06), (GaussianWindow(3.0), 0.06))
                rng = StableRNG(20250912)
                e1 = Float64[]; e2 = Float64[]; cc = Float64[]
                p1 = Float64[]; p2 = Float64[]; pc = Float64[]
                for _ in 1:3000
                    noisy = truth .+ sqrt(sky) .* randn(rng, size(truth))
                    r = measure_star_shape(noisy, 6, 6; inv_var=Fill(1 / sky, size(truth)),
                                           background=0, window=win)
                    push!(e1, r.ellipticity1_aperture); push!(p1, r.ellipticity1_aperture_err)
                    push!(e2, r.ellipticity2_aperture); push!(p2, r.ellipticity2_aperture_err)
                    push!(cc, r.compactness_aperture); push!(pc, r.compactness_aperture_err)
                end
                @test std(e1) ≈ median(p1) rtol=tol
                @test std(e2) ≈ median(p2) rtol=tol
                @test std(cc) ≈ median(pc) rtol=tol
            end
        end
    end
end

# ---------------------------------------------------------------------------
# measure_star_shapes — batch measurement from MatchedFilterResult
# ---------------------------------------------------------------------------

@testset "measure_star_shapes" begin
    rng = StableRNG(99)

    @testset "noise-free circular Gaussian" begin
        # Place a single source in a clean image, detect and measure.
        x0, y0 = 25.3, 25.7
        fwhm_val = 3.0
        flux_val = 500.0
        model = CircularGaussianPSF(; x=x0, y=y0, fwhm=fwhm_val, flux=flux_val, bkg=0.0)
        img = evaluate.(model, 1:50, (1:50)')
        mf = matched_filter(img, fwhm_val; sigma=4.0)
        @test length(mf.peaks) == 1

        results = measure_star_shapes(mf)
        @test length(results) == 1
        r = results[1]

        # Peak index and pixel
        @test r.peak_index == 1
        @test r.pixel == mf.peaks[1]

        # Centroid near true position
        @test abs(r.centroid.y - y0) < 1.0
        @test abs(r.centroid.x - x0) < 1.0

        # FWHM approximately recovered
        y_fwhm_true, x_fwhm_true = psf_fwhm(model)
        @test r.aperture.fwhm.y ≈ y_fwhm_true rtol=0.20
        @test r.aperture.fwhm.x ≈ x_fwhm_true rtol=0.20

        # Roundness near zero for circular PSF
        @test abs(r.core.ellipticity1_core) ≈ 0 atol=1e-10
        @test abs(r.core.ellipticity2_core) ≈ 0 atol=1e-10
        @test abs(r.aperture.ellipticity1_aperture) ≈ 0 atol=1e-10
        @test abs(r.aperture.ellipticity2_aperture) ≈ 0 atol=1e-10

        # Moment normalization is positive for a valid source.
        @test r.aperture.moment_norm > 1.0
        @test r.aperture.aperture_sum > 1.0
        @test r.aperture.aperture_area > 1.0
        @test r.aperture.aperture_sum_err > 1.0
        @test r.significance > 1.0
        @test r.flux > 1.0

        # Centroid errors are positive
        @test r.core.poly.y_err > 0
        @test r.core.poly.x_err > 0
        @test r.aperture.centroid.y_err > 0
        @test r.aperture.centroid.x_err > 0
    end

    @testset "noise-free elliptical Gaussian" begin
        # Elliptical PSF with known orientation.
        x0, y0 = 30.0, 30.0
        x_fwhm, y_fwhm = 4.0, 2.0
        theta = 30.0
        model = GaussianPSF(; x=x0, y=y0, x_fwhm, y_fwhm, theta, flux=500.0, bkg=0.0)
        img = evaluate.(model, 1:60, (1:60)')
        mf = matched_filter(img, (y_fwhm, x_fwhm); sigma=4.0)
        @test length(mf.peaks) == 1

        results = measure_star_shapes(mf)
        r = results[1]

        # y_FWHM < x_FWHM for this model
        @test r.aperture.fwhm.y < r.aperture.fwhm.x

        # Extended in x → e1 negative.  The core value is compressed by the
        # 3×3 box and further reduced by the 30° rotation moving power into
        # e2, so only the sign is meaningful there.
        @test r.core.ellipticity1_core < -0.02
        @test r.aperture.ellipticity1_aperture < -0.2

        # A rotated major axis puts power on a diagonal, which is exactly
        # what e2 measures; its sign follows the sense of the rotation.
        @test abs(r.aperture.ellipticity2_aperture) > 0.2
    end

    @testset "sharpness (DAOPHOT SHARP)" begin
        using CrowdPhot: _kernel_template, _sharp_half_width
        using CrowdPhot.PSF: effective_fwhm
        prf(y, x, f) = [1000 * evaluate(CircularGaussianPRF(; y, x, fwhm = f,
                        flux = 1.0, bkg = 0.0), i, j) for i in 1:49, j in 1:49]
        n = 9
        Praw = [evaluate(CircularGaussianPRF(y = (n + 1) / 2, x = (n + 1) / 2,
                fwhm = 2.5, flux = 1.0, bkg = 0.0), i, j) for i in 1:n, j in 1:n]
        P = Praw ./ sum(Praw)

        # The template must be recovered exactly on both kernel normalization
        # paths; max(P) is the flux -> central-height conversion and the
        # effective FWHM sets the SHARP footprint.
        for zs in (true, false)
            r = matched_filter(zeros(60, 60), P; normalize_zerosum = zs, sigma = 1e9)
            T = _kernel_template(r.kernel, r.kernel_norm)
            @test T ≈ P rtol=1e-12
            @test maximum(T) ≈ maximum(P) rtol=1e-12
            @test effective_fwhm(T) ≈ 2.5 rtol=0.05
        end

        # DAOPHOT's rule, and its floor: never the bare 3x3, and never the
        # kernel size (which for a 9x9 kernel here would be 4, not 2).
        @test _sharp_half_width(2.5) == 2
        @test _sharp_half_width(3.0) == 2
        @test _sharp_half_width(5.0) == 3
        @test _sharp_half_width(10.0) == 7
        @test _sharp_half_width(0.5) == 2
        @test _sharp_half_width(NaN) == 2

        sharp(img) = begin
            r = matched_filter(img, P; sigma = 3.0)
            res = measure_star_shapes(r)
            res[argmax([x.significance for x in res])].sharpness
        end

        s_star = sharp(prf(25.0, 25.0, 2.5))
        hot = zeros(49, 49); hot[25, 25] = 200.0
        s_hot = sharp(hot)
        s_broad = sharp(prf(25.0, 25.0, 5.0))

        # A single-pixel spike is sharper than the PSF; a resolved source is not.
        @test s_broad < s_star < s_hot
        @test 0.5 < s_star < 1.5
        @test s_hot > 2

        # A flat background cancels from the numerator, so sharpness is
        # unchanged by a sky pedestal even though no background is passed.
        @test sharp(prf(25.0, 25.0, 2.5) .+ 100.0) ≈ s_star rtol=1e-8
    end

    @testset "empty peaks (high sigma)" begin
        # Pure noise, high threshold → no detections.
        img = randn(rng, 50, 50)
        mf = matched_filter(img, 3.0; sigma=10.0)
        @test isempty(mf.peaks)

        results = measure_star_shapes(mf)
        @test results isa Vector
        @test isempty(results)
    end

    @testset "peaks keyword — subset selection" begin
        # Two sources, measure only the first.
        model1 = CircularGaussianPSF(; x=15.0, y=15.0, fwhm=2.5, flux=300.0, bkg=0.0)
        model2 = CircularGaussianPSF(; x=35.0, y=35.0, fwhm=2.5, flux=200.0, bkg=0.0)
        img = evaluate.(model1, 1:50, (1:50)') .+ evaluate.(model2, 1:50, (1:50)')
        mf = matched_filter(img, 3.0; sigma=4.0)
        @test length(mf.peaks) >= 2

        # Measure only the first peak.
        results = measure_star_shapes(mf; peaks=[1])
        @test length(results) == 1
        @test results[1].peak_index == 1

        # Measure only the second peak.
        results2 = measure_star_shapes(mf; peaks=[2])
        @test length(results2) == 1
        @test results2[1].peak_index == 2
    end

    @testset "min_significance filtering" begin
        # Two sources with different fluxes → different significances.
        model_bright = CircularGaussianPSF(; x=15.0, y=15.0, fwhm=2.5, flux=500.0, bkg=0.0)
        model_faint  = CircularGaussianPSF(; x=35.0, y=35.0, fwhm=2.5, flux=50.0, bkg=0.0)
        img = evaluate.(model_bright, 1:50, (1:50)') .+
              evaluate.(model_faint, 1:50, (1:50)') .+
              randn(rng, 50, 50) .* 0.5
        inv_var = fill(4.0, size(img))
        mf = matched_filter(img, 3.0; inv_var, sigma=3.0)

        # Bright source should have much higher significance than faint.
        sig_bright = maximum(mf.peak_significances)
        # Filter to only the brightest.
        results = measure_star_shapes(mf; min_significance=sig_bright * 0.8)
        @test length(results) >= 1
        # All returned peaks should satisfy the threshold.
        for r in results
            @test r.significance >= sig_bright * 0.8
        end
    end

    @testset "clipped cutout from large half_width" begin
        # Place a source near the image corner and use a large half_width
        # so the cutout is clipped on two sides by the image boundary.
        model = CircularGaussianPSF(; x=5.0, y=5.0, fwhm=2.5, flux=500.0, bkg=0.0)
        img = evaluate.(model, 1:50, (1:50)')
        mf = matched_filter(img, 3.0; sigma=4.0)
        @test length(mf.peaks) >= 1

        # Use a half_width larger than the distance to the image corner,
        # forcing clipping on the top and left edges.
        results = measure_star_shapes(mf; half_width=10)
        r = results[1]

        # The 3×3 core is complete since the peak is at least 2 px inside,
        # so centroid_poly should succeed.
        @test isfinite(r.core.poly.y)
        @test isfinite(r.core.poly.x)
        @test r.aperture.moment_norm > 0
        @test r.centroid.y > 0
        @test r.centroid.x > 0
    end

    @testset "inv_var=nothing (uniform) vs explicit inv_var" begin
        x0, y0 = 25.0, 25.0
        model = CircularGaussianPSF(; x=x0, y=y0, fwhm=3.0, flux=400.0, bkg=0.0)
        img = evaluate.(model, 1:50, (1:50)')

        # With explicit inv_var
        mf_ivar = matched_filter(img, 3.0; inv_var=fill(1.0, size(img)), sigma=4.0)
        r_ivar = measure_star_shapes(mf_ivar)[1]

        # Without inv_var (uniform assumed)
        mf_none = matched_filter(img, 3.0; sigma=4.0)
        r_none = measure_star_shapes(mf_none)[1]

        # Both should find the source at the same position.
        @test r_ivar.peak_index == r_none.peak_index
        # Moment normalizations should be similar because both paths use
        # uniform weights.
        @test r_ivar.aperture.moment_norm ≈ r_none.aperture.moment_norm rtol=0.01
    end

    @testset "coordinate consistency" begin
        # Verify that all global coordinates are consistent: the centroid
        # from core, chosen centroid, and morphology centroid should all
        # be near each other.
        x0, y0 = 25.3, 25.7
        model = CircularGaussianPSF(; x=x0, y=y0, fwhm=3.0, flux=500.0, bkg=0.0)
        img = evaluate.(model, 1:50, (1:50)')
        mf = matched_filter(img, 3.0; sigma=4.0)
        results = measure_star_shapes(mf)
        r = results[1]

        # All three centroid estimates should be near each other.
        @test abs(r.core.poly.y - r.centroid.y) < 1.0
        @test abs(r.core.poly.x - r.centroid.x) < 1.0
        @test abs(r.aperture.centroid.y - r.centroid.y) < 1.0
        @test abs(r.aperture.centroid.x - r.centroid.x) < 1.0

        # All should be near the true position.
        @test abs(r.centroid.y - y0) < 1.0
        @test abs(r.centroid.x - x0) < 1.0
    end

    @testset "custom half_width" begin
        x0, y0 = 30.0, 30.0
        model = CircularGaussianPSF(; x=x0, y=y0, fwhm=3.0, flux=500.0, bkg=0.0)
        img = evaluate.(model, 1:60, (1:60)')
        mf = matched_filter(img, 3.0; sigma=4.0)

        # Small half_width — cutout is tight around the peak.
        r_small = measure_star_shapes(mf; half_width=3)[1]
        # Large half_width — cutout captures more of the PSF wings.
        r_large = measure_star_shapes(mf; half_width=10)[1]

        # Larger cutout contributes more positive weighted signal.
        @test r_large.aperture.moment_norm > r_small.aperture.moment_norm
        # Both should have reasonable FWHM estimates.
        @test r_small.aperture.fwhm.y > 0
        @test r_large.aperture.fwhm.y > 0
    end

    @testset "noisy image — multiple sources" begin
        # Multiple sources with noise: verify we get one result per peak.
        img = randn(rng, 50, 50) .* 0.3
        σ_psf = 3.0 / 2.355
        kern_half = ceil(Int, 4 * σ_psf) |> k -> isodd(k) ? k : k + 1
        x = LinRange(-kern_half÷2, kern_half÷2, kern_half)
        g = exp.(-0.5 .* (x ./ σ_psf) .^ 2)
        g ./= sum(g)
        kernel = g .* g'

        # Place three sources.
        for (xx, yy, flux) in [(15.0, 15.0, 80.0), (35.0, 20.0, 60.0), (25.0, 40.0, 40.0)]
            kr = kern_half ÷ 2
            xr = max(1, round(Int, xx - kr)):min(50, round(Int, xx + kr))
            yr = max(1, round(Int, yy - kr)):min(50, round(Int, yy + kr))
            for j in xr, i in yr
                dx, dy = j - xx, i - yy
                img[i, j] += flux * exp(-(dx^2 + dy^2) / (2σ_psf^2)) / (2π * σ_psf^2)
            end
        end

        inv_var = fill(1 / 0.3^2, size(img))
        mf = matched_filter(img, kernel; inv_var, sigma=4.0)
        results = measure_star_shapes(mf)

        @test length(results) == length(mf.peaks)
        @test length(results) >= 2  # at least 2 of 3 detected at 4σ

        # Each result should have valid morphology.
        for r in results
            @test r.peak_index >= 1
            @test r.aperture.moment_norm > 0
            @test r.aperture.fwhm.y > 0
            @test r.aperture.fwhm.x > 0
            @test isfinite(r.core.normalized_curvature)
        end
    end

    @testset "background keyword threads to centroid_poly" begin
        x0, y0, fwhm_val, flux_val = 25.4, 25.3, 3.0, 500.0
        model = CircularGaussianPSF(; x=x0, y=y0, fwhm=fwhm_val, flux=flux_val, bkg=0.0)
        base = evaluate.(model, 1:50, (1:50)')
        B = 200.0   # well above the ~49-count peak amplitude
        mf0 = matched_filter(base, fwhm_val; sigma=4.0)
        mfB = matched_filter(base .+ B, fwhm_val; sigma=4.0)

        r0 = measure_star_shapes(mf0; half_width=3)[1]
        rB = measure_star_shapes(mfB; half_width=3, background=B)[1]
        rbiased = measure_star_shapes(mfB; half_width=3)[1]

        # Passing `background` restores the un-pedestaled core diagnostics.
        @test rB.core.normalized_curvature ≈ r0.core.normalized_curvature rtol=1e-6
        @test rB.core.compactness_core ≈ r0.core.compactness_core rtol=1e-6
        @test rB.core.poly.y ≈ r0.core.poly.y atol=1e-9
        @test rB.core.poly.x ≈ r0.core.poly.x atol=1e-9
        # poly.peak is reported including the background.
        @test rB.core.poly.peak ≈ r0.core.poly.peak + B rtol=1e-6

        # Leaving the pedestal in strongly suppresses the curvature diagnostic.
        @test rbiased.core.normalized_curvature < 0.5 * r0.core.normalized_curvature
    end
end

# ---------------------------------------------------------------------------
# measure_star_shape_ref -- the measurement/reference pairing invariant
# ---------------------------------------------------------------------------

@testset "measure_star_shape_ref" begin
    ny = nx = 15
    m = CircularGaussianPSF(; y = 8.3, x = 7.6, fwhm = 3.0, flux = 1000.0, bkg = 0.0)
    rend = zeros(Float64, ny, nx)
    add_star!(rend, m, 1:ny, 1:nx)

    @testset "clean == rend gives exact normalization" begin
        # The whole point of the function: with the data equal to the noiseless
        # render, every ratio must be exactly 1 and every difference exactly 0,
        # at any sub-pixel phase and under any weighting.  Anything else means
        # the two halves were not given identical inputs.
        for (ph, iv) in ((0.0, nothing), (0.3, nothing), (0.5, nothing),
                         (0.3, Fill(0.01, ny, nx)), (0.3, [1 / (1 + i + j) for i in 1:ny, j in 1:nx]))
            mm = CircularGaussianPSF(; y = 8.0 + ph, x = 7.0 + ph, fwhm = 3.0, flux = 1000.0, bkg = 0.0)
            r = zeros(Float64, ny, nx)
            add_star!(r, mm, 1:ny, 1:nx)
            i0, j0 = Tuple(argmax(r))
            s = measure_star_shape_ref(r, r, i0, j0, maximum(r); inv_var = iv)
            @test s.sharpness == s.psf_ref.sharpness
            @test s.core.normalized_curvature == s.psf_ref.core.normalized_curvature
            @test s.core.compactness_core == s.psf_ref.core.compactness_core
            @test s.aperture.compactness_aperture == s.psf_ref.aperture.compactness_aperture
            @test s.aperture.fwhm.y == s.psf_ref.aperture.fwhm.y
            @test s.aperture.fwhm.x == s.psf_ref.aperture.fwhm.x
            @test s.core.ellipticity1_core == s.psf_ref.core.ellipticity1_core
            @test s.aperture.ellipticity1_aperture == s.psf_ref.aperture.ellipticity1_aperture
            @test s.aperture.ellipticity2_aperture == s.psf_ref.aperture.ellipticity2_aperture
        end
    end

    @testset "an extended source separates from its reference" begin
        # A broader source than the model: concentration measures must drop
        # below the reference, confirming the ratio is actually sensitive.
        broad = CircularGaussianPSF(; y = 8.3, x = 7.6, fwhm = 5.0, flux = 1000.0, bkg = 0.0)
        clean = zeros(Float64, ny, nx)
        add_star!(clean, broad, 1:ny, 1:nx)
        i0, j0 = Tuple(argmax(rend))
        s = measure_star_shape_ref(clean, rend, i0, j0, maximum(rend))
        @test s.sharpness < s.psf_ref.sharpness
        @test s.aperture.fwhm.y > s.psf_ref.aperture.fwhm.y
        @test s.aperture.fwhm.x > s.psf_ref.aperture.fwhm.x
    end

    @testset "coordinate offsets lift both halves together" begin
        i0, j0 = Tuple(argmax(rend))
        a = measure_star_shape_ref(rend, rend, i0, j0, maximum(rend))
        b = measure_star_shape_ref(rend, rend, i0, j0, maximum(rend);
                                   y_offset = 100, x_offset = 200)
        @test b.centroid.y ≈ a.centroid.y + 100 rtol=1e-12
        @test b.centroid.x ≈ a.centroid.x + 200 rtol=1e-12
        @test b.psf_ref.core.poly.y ≈ a.psf_ref.core.poly.y + 100 rtol=1e-12
        @test b.aperture.centroid.x ≈ a.aperture.centroid.x + 200 rtol=1e-12
    end

    @testset "mismatched cutout and render are rejected" begin
        @test_throws "must have the same `axes`" measure_star_shape_ref(
            rend, zeros(Float64, ny, nx - 1), 8, 8, 1.0)
    end

    @testset "the window is shared by both halves" begin
        # The invariant the whole function exists to enforce: measurement and
        # reference must see the identical window, or the comparison stops
        # cancelling.  With `clean == rend` every ratio must still be exactly 1
        # and every difference exactly 0 *with* a window in play.
        #
        # Note this holds by identity of method, not by correctness of method:
        # window deconvolution is exact only for a Gaussian source, so the check
        # is repeated on a Moffat render, where it still has to be exact.  That
        # is why the reference half is windowed too rather than left untapered,
        # which would be individually more accurate and break the invariant.
        iv = fill(0.7, ny, nx)
        moff = zeros(Float64, ny, nx)
        add_star!(moff, CircularMoffatPSF(; y = 8.3, x = 7.6, α = 1.8, β = 2.5,
                                            flux = 1000.0, bkg = 0.0), 1:ny, 1:nx)
        for (lab, img) in (("Gaussian", rend), ("Moffat", moff))
            w = GaussianWindow(3.0)
            s = measure_star_shape_ref(img, img, 8, 8, maximum(img);
                                       inv_var = iv, window = w)
            @test s.aperture.compactness_aperture == s.psf_ref.aperture.compactness_aperture
            @test s.aperture.fwhm.y == s.psf_ref.aperture.fwhm.y
            @test s.aperture.fwhm.x == s.psf_ref.aperture.fwhm.x
            @test s.aperture.ellipticity1_aperture - s.psf_ref.aperture.ellipticity1_aperture == 0
            @test s.aperture.ellipticity2_aperture - s.psf_ref.aperture.ellipticity2_aperture == 0
        end

        # Non-vacuity: the window is genuinely in play.  A Gaussian source would
        # show nothing here, since deconvolving an exactly-Gaussian profile
        # returns the untapered answer; a Moffat is only approximately recovered,
        # so tapering moves the value.
        sw = measure_star_shape_ref(moff, moff, 8, 8, maximum(moff);
                                    inv_var = iv, window = GaussianWindow(3.0))
        sf = measure_star_shape_ref(moff, moff, 8, 8, maximum(moff); inv_var = iv)
        @test sw.aperture.compactness_aperture != sf.aperture.compactness_aperture
    end

    @testset "sharpness_err" begin
        rng = StableRNG(505)
        clean = rend .+ 2.0 .* randn(rng, ny, nx)
        h = maximum(rend)

        # Linear in the pixel sigma: a 100x smaller weight map scales the error
        # by exactly 10 and leaves `sharpness` itself bitwise unchanged, since
        # the value's footprint average is unweighted.
        s1 = measure_star_shape_ref(clean, rend, 8, 8, h; inv_var = fill(1.0, ny, nx))
        s10 = measure_star_shape_ref(clean, rend, 8, 8, h; inv_var = fill(1/100, ny, nx))
        @test s1.sharpness_err > 0
        @test isfinite(s1.sharpness_err)
        @test s10.sharpness === s1.sharpness
        @test s10.sharpness_err / s1.sharpness_err ≈ 10 rtol=1e-12

        # Closed form: sqrt(sigma_c^2 + sigma_n^2/n) / H over the (2*shw+1)^2
        # footprint less the center.  Validated against 200k Monte Carlo
        # realizations to within the MC error on a standard deviation.
        shw = 2
        n_nb = (2shw + 1)^2 - 1
        @test s1.sharpness_err ≈ sqrt(1 + 1 / n_nb) / h rtol=1e-12

        # A masked pixel anywhere in the footprint makes the error undefined
        # without disturbing the value.
        iv_masked = fill(1.0, ny, nx)
        iv_masked[7, 7] = 0.0
        sm = measure_star_shape_ref(clean, rend, 8, 8, h; inv_var = iv_masked)
        @test isnan(sm.sharpness_err)
        @test sm.sharpness === s1.sharpness

        # `psf_ref` mirrors values only; the render is noiseless.
        @test !haskey(s1.psf_ref, :sharpness_err)
        @test keys(s1.psf_ref) == (:sharpness, :core, :aperture)
    end
end
