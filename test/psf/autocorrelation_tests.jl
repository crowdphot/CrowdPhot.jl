import CrowdPhot
using CrowdPhot.PSF: CircularGaussianPSF, CircularGaussianPRF, GriddedPSFModel, ImagePSF, evaluate,
    _psf_acf, _psf_acf_at, _cross_corr
import ConstructionBase
using Test

# Phase-averaged autocorrelation of the blended model at `(y, x)`, sampled directly.
function direct_acf(m, y, x, L, R, os)
    mb = ConstructionBase.setproperties(m, (; y, x))
    f = [evaluate(mb, y + i / os, x + j / os) for i in (-os * R):(os * R), j in (-os * R):(os * R)]
    A = _cross_corr(f, f, L, os, os)
    return A ./ A[L + 1, L + 1]
end

@testset "Gaussian PSF: closed form" begin
    L = 6
    for σ in (0.8, 1.3, 2.0)
        g = CircularGaussianPSF(y = 0.0, x = 0.0, fwhm = σ * 2sqrt(2log(2)), flux = 3.0, bkg = 1.0)
        # ∫ p(u) p(u + Δ) du ∝ exp(-|Δ|² / 4σ²) for a Gaussian of width σ.
        ref = [exp(-(dy^2 + dx^2) / (4σ^2)) for dy in -L:L, dx in -L:L]
        A = _psf_acf_at(_psf_acf(g, L, ceil(Int, 6σ)), 10.0, 20.0)
        @test size(A) == (2L + 1, 2L + 1)
        @test A[L + 1, L + 1] == 1
        @test A ≈ ref atol = 1.0e-6
        @test A == reverse(A)   # A(Δ) = A(-Δ)
    end
end

@testset "Cropping" begin
    g = CircularGaussianPSF(y = 0.0, x = 0.0, fwhm = 2.5, flux = 1.0, bkg = 0.0)
    L = 6
    full = _psf_acf_at(_psf_acf(g, L, 15), 0, 0)
    # The squared profile is concentrated, so a crop at a few FWHM barely changes the shape.
    @test _psf_acf_at(_psf_acf(g, L, 4), 0, 0) ≈ full atol = 1.0e-3
    @test _psf_acf_at(_psf_acf(g, L, 8), 0, 0) ≈ full atol = 1.0e-10
end

@testset "GriddedPSFModel" begin
    L, R = 6, 10
    grid_y, grid_x = [0.0, 0.0, 100.0, 100.0], [0.0, 100.0, 0.0, 100.0]
    @testset "ImagePSF nodes (tabulated path)" begin
        mk(s) = ImagePSF([exp(-((i - 41)^2 + (j - 41)^2) / (2 * (4s)^2)) for i in 1:81, j in 1:81];
                         oversampling = 4, normalize = true)
        gm = GriddedPSFModel([mk(s) for s in (1.0, 1.2, 1.4, 1.6)], grid_y, grid_x)
        acf = _psf_acf(gm, L, R)
        @test acf isa CrowdPhot.PSF._GriddedACF
        for (y, x) in ((30.0, 70.0), (0.0, 0.0), (100.0, 50.0), (-20.0, 150.0))
            @test _psf_acf_at(acf, y, x) ≈ direct_acf(gm, y, x, L, R, 4) atol = 1.0e-12
        end
        # Mixed oversampling falls back to sampling each node like any other model.
        n2 = ImagePSF(mk(1.0).data[1:2:end, 1:2:end]; oversampling = 2, normalize = true)
        gmix = GriddedPSFModel([mk(1.0), mk(1.2), mk(1.4), n2], grid_y, grid_x)
        @test !CrowdPhot.PSF._acf_tabulated(gmix.psfs)
        @test _psf_acf_at(_psf_acf(gmix, L, R), 60.0, 60.0) ≈ direct_acf(gmix, 60.0, 60.0, L, R, 4) atol = 1.0e-3
    end
    @testset "parametric nodes (sampled path)" begin
        nodes = [CircularGaussianPRF(y = 0.0, x = 0.0, fwhm = 2.5 + k / 4, flux = 1.0, bkg = 0.0) for k in 0:3]
        gm = GriddedPSFModel(nodes, grid_y, grid_x)
        acf = _psf_acf(gm, L, R)
        for (y, x) in ((30.0, 70.0), (100.0, 100.0))
            @test _psf_acf_at(acf, y, x) ≈ direct_acf(gm, y, x, L, R, 4) atol = 1.0e-12
        end
    end
    @testset "single node" begin
        g = CircularGaussianPRF(y = 0.0, x = 0.0, fwhm = 3.0, flux = 1.0, bkg = 0.0)
        gm = GriddedPSFModel([g], [50.0], [50.0])
        @test _psf_acf_at(_psf_acf(gm, L, R), 10.0, 90.0) ≈ _psf_acf_at(_psf_acf(g, L, R), 0, 0) atol = 1.0e-12
    end
end
