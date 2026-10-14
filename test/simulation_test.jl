using CrowdPhot
using CrowdPhot.PSF
using Distributions: Poisson
using Statistics: mean, median
using StableRNGs: StableRNG
using Test

"""
    ks_2sample_statistic(x, y)

Return the two-sample Kolmogorov–Smirnov statistic between samples `x` and `y`.

This computes

    D = sup_z |F_x(z) - F_y(z)|

where `F_x` and `F_y` are empirical CDFs.
"""
function ks_2sample_statistic(x::AbstractVector, y::AbstractVector)
    n = length(x)
    m = length(y)
    n > 0 || throw(ArgumentError("x must be nonempty"))
    m > 0 || throw(ArgumentError("y must be nonempty"))

    xs = sort(x)
    ys = sort(y)

    i = j = 1
    cdf_x = 0.0
    cdf_y = 0.0
    d = 0.0

    while i <= n || j <= m
        if j > m || (i <= n && xs[i] < ys[j])
            z = xs[i]
        elseif i > n || ys[j] < xs[i]
            z = ys[j]
        else
            z = xs[i]
        end

        while i <= n && xs[i] == z
            i += 1
        end
        cdf_x = (i - 1) / n

        while j <= m && ys[j] == z
            j += 1
        end
        cdf_y = (j - 1) / m

        d = max(d, abs(cdf_x - cdf_y))
    end

    return d
end

@testset "simulation utilities" begin
    # Exercise source placement, deterministic rendering, SNR conversion, and noise.
    rng = StableRNG(10)
    psf = CircularGaussianPRF(x = 0, y = 0, fwhm = 2.5, flux = 1, bkg = 0)
    sources = simulate_sources((40, 45), 8; flux = (100.0, 120.0), min_separation = 4, border = 5, rng)
    @test length(sources.x) == 8
    @test all(6 .≤ sources.x .≤ 40)
    @test all(6 .≤ sources.y .≤ 35)
    for i in 1:length(sources.x), j in (i + 1):length(sources.x)
        @test hypot(sources.x[i] - sources.x[j], sources.y[i] - sources.y[j]) ≥ 4
    end

    image = simulate_image((40, 45), psf, sources; background = 12.0, noise = :none, model_radius = 7)
    @test size(image) == (40, 45)
    @test sum(image) ≈ 12.0 * 40 * 45 + sum(sources.flux) rtol = 1.0e-6
    @test CrowdPhot.flux_for_snr(psf, 20; background = 12, read_noise = 2) > 0

    @testset "flux_for_snr" begin
        # Verify that requested SNR inverts the data-unit noise model, including source shot noise.
        snr = 20.0
        background = 12.0
        read_noise = 2.0
        gain = 2.5
        flux = CrowdPhot.flux_for_snr(psf, snr; background, read_noise, gain)
        σb2 = background / gain + read_noise^2
        @test flux / sqrt(flux / gain + effective_area(psf) * σb2) ≈ snr

        # Source-only Poisson noise still requires nonzero flux in data units.
        @test CrowdPhot.flux_for_snr(psf, 5; background = 0, gain = 2) ≈ 12.5
    end

    @testset "snr_for_flux" begin
        # snr_for_flux must directly match the quadratic-formula noise model.
        flux = 317.0
        background = 12.0
        read_noise = 2.0
        gain = 2.5
        σb2 = background / gain + read_noise^2
        snr = CrowdPhot.snr_for_flux(psf, flux; background, read_noise, gain)
        @test snr ≈ flux / sqrt(flux / gain + effective_area(psf) * σb2)

        # snr_for_flux and flux_for_snr must be exact inverses of one another.
        for snr0 in (5.0, 20.0, 100.0)
            flux0 = CrowdPhot.flux_for_snr(psf, snr0; background, read_noise, gain)
            @test CrowdPhot.snr_for_flux(psf, flux0; background, read_noise, gain) ≈ snr0
        end
        for flux0 in (12.5, 100.0, 500.0)
            snr0 = CrowdPhot.snr_for_flux(psf, flux0; background, read_noise, gain)
            @test CrowdPhot.flux_for_snr(psf, snr0; background, read_noise, gain) ≈ flux0
        end

        # Zero flux with zero background/read-noise variance is a 0/0 indeterminate form.
        @test isnan(CrowdPhot.snr_for_flux(psf, 0.0; background = 0, read_noise = 0))
        # Zero flux with nonzero background variance gives zero SNR.
        @test CrowdPhot.snr_for_flux(psf, 0.0; background, read_noise, gain) == 0
    end

    @testset "simulate_sources snr branch" begin
        # A scalar `snr` should place every source at the flux required for that SNR.
        rng_snr = StableRNG(11)
        background = 12.0
        read_noise = 2.0
        gain = 1.0
        snr_sources = simulate_sources(
            (40, 45), 6; snr = 15.0, psf, background, read_noise, gain,
            min_separation = 4, border = 5, rng = rng_snr
        )
        expected_flux = CrowdPhot.flux_for_snr(psf, 15.0; background, read_noise, gain)
        @test all(f -> f ≈ expected_flux, snr_sources.flux)

        # A two-value `snr` range should sample each source's SNR uniformly before converting to flux.
        rng_range = StableRNG(12)
        range_sources = simulate_sources(
            (40, 45), 6; snr = (10.0, 30.0), psf, background, read_noise, gain,
            min_separation = 4, border = 5, rng = rng_range
        )
        @test length(range_sources.flux) == 6
        for f in range_sources.flux
            snr_f = CrowdPhot.snr_for_flux(psf, f; background, read_noise, gain)
            @test 10.0 ≤ snr_f ≤ 30.0
        end

        # A matrix background stands for its median: same sources as the scalar median.
        bgm = 10 .+ rand(StableRNG(3), 40, 45)
        kw = (; snr = (10.0, 30.0), psf, read_noise, gain, min_separation = 4, border = 5)
        a = simulate_sources((40, 45), 6; kw..., background = bgm, rng = StableRNG(13))
        b = simulate_sources((40, 45), 6; kw..., background = median(bgm), rng = StableRNG(13))
        @test a.flux == b.flux && a.x == b.x && a.y == b.y

        # The SNR spec is validated once, before any source is drawn.
        @test_throws "`psf` is required" simulate_sources((40, 45), 6; snr = 10.0)
        @test_throws "`snr` must be non-negative" simulate_sources((40, 45), 6; snr = (-5.0, 10.0), psf)
        @test_throws "`snr` range must have two values" simulate_sources((40, 45), 6; snr = (1.0, 2.0, 3.0), psf)
        @test_throws "`snr` must be a scalar or two-value range" simulate_sources((40, 45), 6; snr = "10", psf)
    end

    noisy = copy(image)
    CrowdPhot.add_noise!(noisy; noise = :poisson_gaussian, read_noise = 1.0, rng)
    @test noisy != image

    @testset "poisson" begin
        # Compare the internal Poisson sampler against Distributions.jl across small and large rates.
        for val in (0.5, 2.0, 10.0, 1000.0)
            rng = StableRNG(123)
            x = [CrowdPhot.rand_poisson(rng, val) for _ in 1:100_000]
            y = rand(rng, Poisson(val), 100_000)
            @test ks_2sample_statistic(x, y) < 0.01
        end
        # A zero or underflowing rate (exp(-λ) == 1) is exactly zero counts, not -1.
        rng = StableRNG(123)
        @test all(iszero, CrowdPhot.rand_poisson(rng, λ) for λ in (0.0, 1e-20, 1e-300) for _ in 1:100)
        zimg = CrowdPhot.add_noise!(zeros(64, 64); noise = :poisson, rng)
        @test all(iszero, zimg)
        @test abs(mean(CrowdPhot.add_noise!(fill(0.1, 256, 256); noise = :poisson, rng)) - 0.1) < 0.005
    end
end
