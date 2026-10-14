using CrowdPhot: matched_filter, MatchedFilterResult, findlocalmaxima
using CrowdPhot.PSF: CircularGaussianPRF, GaussianPRF, CircularGaussianPSF, render
using LinearAlgebra
using StableRNGs: StableRNG
using Statistics: mean, std, median
using Test

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

"Create a 2D Gaussian PSF kernel centred in an odd-sized array."
function _gaussian_kernel(k::Int, σ::Real)
    x = LinRange(-k÷2, k÷2, k)
    g = exp.(-0.5 .* (x ./ σ) .^ 2)
    g ./= sum(g)
    return g .* g'
end

"Place a Gaussian source with flux `F` at sub-pixel matrix position `(x, y)`."
function _place_source!(img, x::Real, y::Real, flux::Real, σ::Real)
    X, Y = size(img)
    k = ceil(Int, 8σ) |> x -> isodd(x) ? x : x + 1
    xr = max(1, round(Int, x - k÷2)):min(X, round(Int, x + k÷2))
    yr = max(1, round(Int, y - k÷2)):min(Y, round(Int, y + k÷2))
    for i in xr, j in yr
        dx, dy = i - x, j - y
        img[i, j] += flux * exp(-(dx^2 + dy^2) / (2σ^2)) / (2π * σ^2)
    end
    return img
end

"Return detected peak x-coordinates in matrix-index convention."
_peak_xs(result::MatchedFilterResult) = Float64[p[1] for p in result.peaks]

"Return detected peak y-coordinates in matrix-index convention."
_peak_ys(result::MatchedFilterResult) = Float64[p[2] for p in result.peaks]

"Return the index of the detected peak nearest the supplied x/y coordinates."
function _nearest_peak(result::MatchedFilterResult, x::Real, y::Real)
    return argmin([(p[1] - x)^2 + (p[2] - y)^2 for p in result.peaks])
end

# ---------------------------------------------------------------------------
# Matched filter — basic detection
# ---------------------------------------------------------------------------
@testset "matched_filter: basic detection" begin
    rng = StableRNG(101)

    @testset "recovers known sources in low-noise image" begin
        img = randn(rng, 80, 80) .* 0.5
        σ_psf = 1.5
        kern = _gaussian_kernel(9, σ_psf)
        # Place three well-separated sources
        _place_source!(img, 20.0, 20.0, 40.0, σ_psf)
        _place_source!(img, 50.0, 60.0, 25.0, σ_psf)
        _place_source!(img, 65.0, 30.0, 15.0, σ_psf)

        inv_var = fill(4.0, size(img))  # σ=0.5 → w=4
        result = matched_filter(img, kern; inv_var, sigma=5.0)

        # Should find the two brightest sources
        @test length(result.peaks) >= 2
        # Brightest source should be near (20, 20)
        brightest = argmax(result.peak_significances)
        @test abs(_peak_xs(result)[brightest] - 20) ≤ 2
        @test abs(_peak_ys(result)[brightest] - 20) ≤ 2
    end

    @testset "pure noise image yields few or no detections at high sigma" begin
        img = randn(rng, 100, 100)
        kern = _gaussian_kernel(11, 2.0)
        inv_var = fill(1.0, size(img))  # σ=1
        result = matched_filter(img, kern; inv_var, sigma=5.0)
        # At 5σ, expect ≪ 1 false positive in 10^4 pixels
        @test length(result.peaks) ≤ 3
    end

    @testset "sigma threshold controls detection count" begin
        img = randn(rng, 80, 80) .* 0.3
        kern = _gaussian_kernel(11, 2.0)
        _place_source!(img, 40.0, 40.0, 30.0, 2.0)
        inv_var = fill(1 / 0.3^2, size(img))

        r_high = matched_filter(img, kern; inv_var, sigma=8.0)
        r_low  = matched_filter(img, kern; inv_var, sigma=3.0)
        @test length(r_low.peaks) >= length(r_high.peaks)
    end
end

# ---------------------------------------------------------------------------
# Kernel normalization
# ---------------------------------------------------------------------------
@testset "matched_filter: kernel normalization" begin
    rng = StableRNG(202)

    @testset "zero-sum kernel has sum(K) ≈ 0" begin
        kern = _gaussian_kernel(11, 2.0)
        img = zeros(100, 100)
        result = matched_filter(img, kern; normalize_zerosum=true)
        # Cancellation of Σ(P - P̄) can incur N·eps roundoff.
        @test abs(sum(result.kernel)) < 1e-13
    end

    @testset "non-zero-sum kernel has sum(K) > 0 for positive PSF" begin
        kern = _gaussian_kernel(11, 2.0)
        img = zeros(100, 100)
        result = matched_filter(img, kern; normalize_zerosum=false)
        @test sum(result.kernel) > 0
    end

    @testset "flux estimator is unbiased (zero-sum, no noise)" begin
        σ_psf = 1.5
        kern = _gaussian_kernel(9, σ_psf)
        # Image with a single source of known flux, uniform background
        img = fill(10.0, 40, 40)  # background
        _place_source!(img, 20.0, 20.0, 50.0, σ_psf)
        # Zero-sum should cancel the background automatically
        result = matched_filter(img, kern; normalize_zerosum=true)
        # Find the peak nearest the true position
        peak_idx = _nearest_peak(result, 20, 20)
        @test result.peak_fluxes[peak_idx] ≈ 50.0 rtol = 0.05
    end

    @testset "flux estimator is unbiased (zero-sum, with noise)" begin
        σ_psf = 1.5
        kern = _gaussian_kernel(9, σ_psf)
        # Monte Carlo: many realisations, average flux estimate should → true flux
        fluxes = Float64[]
        for _ in 1:50
            img = fill(5.0, 40, 40) .+ randn(rng, 40, 40) .* 0.5
            _place_source!(img, 20.0, 20.0, 50.0, σ_psf)
            result = matched_filter(img, kern; normalize_zerosum=true)
            if !isempty(result.peaks)
                peak_idx = _nearest_peak(result, 20, 20)
                push!(fluxes, result.peak_fluxes[peak_idx])
            end
        end
        @test length(fluxes) ≥ 30  # most realisations should find it
        @test mean(fluxes) ≈ 50.0 rtol = 0.10
    end

    @testset "non-zero-sum kernel leaks background into significance map" begin
        kern = _gaussian_kernel(9, 1.5)
        img = fill(100.0, 40, 40)  # large uniform background, no sources
        r_zs  = matched_filter(img, kern; normalize_zerosum=true)
        r_nzs = matched_filter(img, kern; normalize_zerosum=false)
        # A constant image correlated with any kernel is also constant, so
        # findlocalmaxima finds no peaks in either case.  The right test:
        # the zero-sum significance map should be ~0 everywhere; the
        # non-zero-sum map should be dominated by B·∑K ≠ 0.
        @test maximum(abs, r_zs.significance_map) < 1e-12
        @test maximum(abs, r_nzs.significance_map) > 100
    end

    @testset "both normalizations agree on bg-subtracted image" begin
        kern = _gaussian_kernel(9, 1.5)
        bg = 10.0
        # Image with background, then subtract it
        img_raw = fill(bg, 40, 40) .+ randn(rng, 40, 40) .* 0.3
        _place_source!(img_raw, 20.0, 20.0, 40.0, 1.5)
        img_sub = img_raw .- bg
        inv_var = fill(1 / 0.3^2, size(img_sub))

        r_zs  = matched_filter(img_sub, kern; inv_var, normalize_zerosum=true)
        r_nzs = matched_filter(img_sub, kern; inv_var, normalize_zerosum=false)

        # Both should detect the source
        @test length(r_zs.peaks) >= 1
        @test length(r_nzs.peaks) >= 1

        # The zero-sum significance should be lower by the SNR penalty
        # factor √(1 - (∑P)²/(N·∑P²)).  Compute it from the kernel.
        P = kern
        N = length(P)
        penalty = sqrt(1 - sum(P)^2 / (N * sum(abs2, P)))
        idx_zs  = _nearest_peak(r_zs, 20, 20)
        idx_nzs = _nearest_peak(r_nzs, 20, 20)
        ratio = r_zs.peak_significances[idx_zs] / r_nzs.peak_significances[idx_nzs]
        @test ratio ≈ penalty rtol = 0.10
    end
end

# ---------------------------------------------------------------------------
# Significance calibration
# ---------------------------------------------------------------------------
@testset "matched_filter: significance calibration" begin
    rng = StableRNG(303)

    @testset "z ≈ N(0,1) under pure noise with inv_var" begin
        kern = _gaussian_kernel(11, 2.0)
        # Collect significance values from interior pixels across realisations
        sigs = Float64[]
        for _ in 1:20
            img = randn(rng, 50, 50)
            inv_var = fill(1.0, size(img))
            result = matched_filter(img, kern; inv_var, sigma=0.0)
            # Sample interior pixels (avoid border effects from replicate padding)
            s = result.significance_map[10:40, 10:40]
            append!(sigs, vec(s))
        end
        # Standard deviation should be close to 1
        @test abs(std(sigs) - 1.0) < 0.15
        # Mean should be close to 0
        @test abs(mean(sigs)) < 0.05
    end

    @testset "significance scales correctly with noise level" begin
        kern = _gaussian_kernel(11, 2.0)
        σ_psf = 2.0
        # Place same source in images with different noise levels
        sig_vals = Float64[]
        for σ_noise in [0.5, 1.0, 2.0]
            img = randn(rng, 60, 60) .* σ_noise
            _place_source!(img, 30.0, 30.0, 30.0, σ_psf)
            inv_var = fill(1 / σ_noise^2, size(img))
            result = matched_filter(img, kern; inv_var, sigma=0.0)
            peak_idx = _nearest_peak(result, 30, 30)
            push!(sig_vals, result.peak_significances[peak_idx])
        end
        # Peak significance should be roughly inversely proportional to noise
        @test sig_vals[1] > sig_vals[2] > sig_vals[3]
    end
end

# ---------------------------------------------------------------------------
# Inverse variance handling
# ---------------------------------------------------------------------------
@testset "matched_filter: inverse variance" begin
    rng = StableRNG(404)

    @testset "uniform inv_var vs nothing gives consistent relative significances" begin
        kern = _gaussian_kernel(9, 1.5)
        img = randn(rng, 50, 50)
        _place_source!(img, 25.0, 25.0, 30.0, 1.5)

        σ = 0.5
        r_ivar = matched_filter(img, kern; inv_var=fill(1/σ^2, size(img)), sigma=0.0)
        r_none = matched_filter(img, kern; sigma=0.0)

        # Significances should be related by the noise scaling
        peak_ivar = _nearest_peak(r_ivar, 25, 25)
        peak_none = _nearest_peak(r_none, 25, 25)
        ratio = r_ivar.peak_significances[peak_ivar] / r_none.peak_significances[peak_none]
        # Without inv_var, significance is in units of σ⁻¹; divide by σ to compare
        @test ratio ≈ 1 / σ rtol = 0.3
    end

    @testset "zero inv_var produces zero significance" begin
        # Use a larger image and a well-isolated zero-weight region so
        # that the kernel convolution does not smear non-zero weights
        # from the boundary into the test region.
        kern = _gaussian_kernel(9, 1.5)
        kr = 4  # kernel radius
        img = randn(rng, 50, 50)
        inv_var = ones(50, 50)
        # Zero-weight block with >kr margin on all sides from any non-zero weight.
        inv_var[15:35, 15:35] .= 0.0
        result = matched_filter(img, kern; inv_var, sigma=0.0)
        # Interior of the zero-weight block (outside the kernel's reach).
        @test all(iszero, result.significance_map[20:30, 20:30])
        # But should not error
        @test result isa MatchedFilterResult
    end

    @testset "negative inv_var is not rejected at input (user responsibility)" begin
        kern = _gaussian_kernel(5, 1.0)
        img = randn(rng, 20, 20)
        # Negative inverse variance is physically nonsensical but won't crash;
        # the significance calculation handles it via den > 0 guard.
        inv_var = fill(-1.0, size(img))
        @test_nowarn matched_filter(img, kern; inv_var, sigma=0.0)
    end

    @testset "spatially varying noise: high-noise region suppresses peaks" begin
        kern = _gaussian_kernel(9, 1.5)
        img = randn(rng, 50, 50)
        _place_source!(img, 20.0, 20.0, 20.0, 1.5)
        _place_source!(img, 40.0, 40.0, 20.0, 1.5)
        # Make the high-x half of the image very noisy.
        inv_var = ones(50, 50)
        inv_var[26:end, :] .= 0.01  # very low weight = very noisy
        result = matched_filter(img, kern; inv_var, sigma=3.0)
        # Source in quiet region should be detected; source in noisy region may not
        n_low_x = count(p -> p[1] < 25, result.peaks)
        n_high_x = count(p -> p[1] > 25, result.peaks)
        @test n_low_x ≥ n_high_x
    end

    @testset "inv_var size mismatch throws" begin
        kern = _gaussian_kernel(5, 1.0)
        img = randn(rng, 20, 20)
        @test_throws DimensionMismatch matched_filter(
            img, kern; inv_var=ones(10, 10))
    end

    @testset "SEP-style: inv_var weighting detects source that uniform misses" begin
        # Reproduction of the SEP documentation example:
        # https://sep.readthedocs.io/en/stable/filter.html
        # sep.extract(data, 3.0, err=error, filter_type='conv')    → 0
        # sep.extract(data, 3.0, err=error, filter_type='matched') → 1
        #
        # Image has spatially varying noise: σ=4 where the first index is
        # greater than the second and σ=1 elsewhere. A source at the center sits at the
        # boundary between quiet and noisy regions.  Uniform-weight correlation
        # (CrowdPhot's default, analogous to SEP filter_type='conv') gives
        # full weight to the σ=4 pixels and buries the source.  Inverse-variance
        # weighting (CrowdPhot with inv_var, analogous to SEP
        # filter_type='matched') downweights those pixels by 1/16 and recovers
        # the source.
        rng_sep = StableRNG(21)
        n = 16

        # σ map: 4 where x > y, 1 elsewhere
        err = ones(n, n)
        for i in 1:n, j in 1:n
            if i > j
                err[i, j] = 4.0
            end
        end

        data = err .* randn(rng_sep, n, n)

        # Source at center (matching Python data[7:10, 7:10] in 0-indexed)
        source = [1.0 2.0 1.0;
                  2.0 4.0 2.0;
                  1.0 2.0 1.0]
        data[8:10, 8:10] .+= source

        # SEP's default detection kernel
        kernel = [1.0 2.0 3.0 2.0 1.0;
                  2.0 3.0 5.0 3.0 2.0;
                  3.0 5.0 8.0 5.0 3.0;
                  2.0 3.0 5.0 3.0 2.0;
                  1.0 2.0 3.0 2.0 1.0]

        inv_var = 1.0 ./ (err .^ 2)
        thresh = 3.0

        r_uniform = matched_filter(data, kernel;
                                   normalize_zerosum=false, sigma=thresh)
        r_weighted = matched_filter(data, kernel; inv_var,
                                    normalize_zerosum=false, sigma=thresh)

        x0, y0 = 9, 9  # source center in matrix-index convention

        # Uniform-weight (SEP conv): source is not detected
        @test r_uniform.significance_map[x0, y0] < thresh

        # Inverse-variance weighted (SEP matched): source is detected
        @test r_weighted.significance_map[x0, y0] >= thresh
    end
end

# ---------------------------------------------------------------------------
# Peak finding (findlocalmaxima)
# ---------------------------------------------------------------------------
@testset "findlocalmaxima" begin

    @testset "finds single isolated peak" begin
        img = zeros(10, 10)
        img[5, 5] = 1.0
        peaks = findlocalmaxima(img)
        @test length(peaks) == 1
        @test peaks[1] == CartesianIndex(5, 5)
    end

    @testset "finds multiple separated peaks" begin
        img = zeros(20, 20)
        img[5, 5] = 1.0
        img[5, 15] = 2.0
        img[15, 10] = 3.0
        peaks = findlocalmaxima(img)
        @test length(peaks) == 3
        @test CartesianIndex(15, 10) in peaks
    end

    @testset "does not find peaks on flat regions" begin
        img = fill(1.0, 10, 10)
        peaks = findlocalmaxima(img)
        @test isempty(peaks)
    end

    @testset "adjacent maxima: only the highest qualifies" begin
        img = zeros(7, 7)
        img[2, 2] = 1.0
        img[2, 3] = 0.9  # not a maximum: neighbor (2,2) is higher
        img[5, 5] = 1.0  # well-separated from (2,2)
        peaks = findlocalmaxima(img)
        # (2,2) qualifies; (2,3) is suppressed by (2,2);
        # (5,5) is isolated and qualifies.
        @test length(peaks) == 2
        @test CartesianIndex(2, 2) in peaks
        @test CartesianIndex(5, 5) in peaks
        @test CartesianIndex(2, 3) ∉ peaks
    end

    @testset "edges=false excludes boundary pixels" begin
        img = zeros(10, 10)
        img[1, 5] = 10.0  # edge peak
        img[5, 5] = 1.0   # interior peak
        peaks_with    = findlocalmaxima(img; edges=true)
        peaks_without = findlocalmaxima(img; edges=false)
        @test CartesianIndex(1, 5) in peaks_with
        @test CartesianIndex(1, 5) ∉ peaks_without
        @test CartesianIndex(5, 5) in peaks_without
    end

    @testset "corners are handled" begin
        img = zeros(5, 5)
        img[1, 1] = 10.0
        img[1, 5] = 10.0
        img[5, 1] = 10.0
        img[5, 5] = 10.0
        peaks = findlocalmaxima(img)
        @test length(peaks) == 4
    end

    @testset "tiny images (1×1, 2×2) do not error" begin
        @test_nowarn findlocalmaxima(ones(1, 1))
        @test_nowarn findlocalmaxima(ones(2, 2))
        @test isempty(findlocalmaxima(ones(1, 1)))
        @test isempty(findlocalmaxima(ones(2, 2)))
    end

    @testset "strict inequality: equal-valued neighbors disqualify" begin
        img = zeros(5, 5)
        img[2, 2] = 1.0
        img[2, 3] = 1.0  # equal neighbor
        peaks = findlocalmaxima(img)
        @test CartesianIndex(2, 2) ∉ peaks
        @test CartesianIndex(2, 3) ∉ peaks
    end
end

# ---------------------------------------------------------------------------
# Convenience methods
# ---------------------------------------------------------------------------
@testset "matched_filter: convenience methods" begin
    rng = StableRNG(505)

    @testset "AbstractPSFModel dispatch" begin
        model = CircularGaussianPSF(; x=0.0, y=0.0, fwhm=4.0, flux=1.0, bkg=0.0)
        img = randn(rng, 50, 50) .* 0.3
        _place_source!(img, 25.0, 25.0, 30.0, 4.0 / 2.355)
        result = matched_filter(img, model; sigma=4.0)
        @test result isa MatchedFilterResult
        # Rendered kernel should be odd-sized
        @test isodd(size(result.kernel, 1))
        @test isodd(size(result.kernel, 2))
    end

    @testset "Int FWHM dispatch" begin
        img = randn(rng, 50, 50) .* 0.3
        _place_source!(img, 25.0, 25.0, 30.0, 3.0 / 2.355)
        result = matched_filter(img, 5; sigma=4.0)
        @test result isa MatchedFilterResult
        @test isodd(size(result.kernel, 1))
        @test length(result.peaks) >= 1
    end

    @testset "Tuple{Int,Int} FWHM dispatch" begin
        img = randn(rng, 50, 50) .* 0.3
        _place_source!(img, 25.0, 25.0, 30.0, 3.0 / 2.355)
        result = matched_filter(img, (5, 7); sigma=4.0)
        @test result isa MatchedFilterResult
        @test isodd(size(result.kernel, 1))
        @test isodd(size(result.kernel, 2))
    end
end

# ---------------------------------------------------------------------------
# MatchedFilterResult structure
# ---------------------------------------------------------------------------
@testset "MatchedFilterResult structure" begin
    rng = StableRNG(606)

    @testset "field sizes are consistent" begin
        kern = _gaussian_kernel(9, 1.5)
        img = randn(rng, 40, 40) .* 0.3
        _place_source!(img, 20.0, 20.0, 20.0, 1.5)
        inv_var = fill(4.0, size(img))
        result = matched_filter(img, kern; inv_var, sigma=3.0)

        @test size(result.image) == (40, 40)
        @test size(result.significance_map) == (40, 40)
        @test size(result.smoothed_image) == (40, 40)
        @test size(result.inv_var) == (40, 40)
        @test size(result.smoothed_inv_var) == (40, 40)
        @test length(result.peaks) == length(result.peak_significances)
        @test length(result.peaks) == length(result.peak_fluxes)
        @test eltype(result.peaks) == CartesianIndex{2}
        @test result.kernel_norm > 0
    end

    @testset "inv_var=nothing gives nothing fields" begin
        kern = _gaussian_kernel(5, 1.0)
        img = randn(rng, 20, 20)
        result = matched_filter(img, kern)
        @test result.inv_var === nothing
        @test result.smoothed_inv_var === nothing
    end

    @testset "peak coordinates are within image bounds" begin
        kern = _gaussian_kernel(7, 1.5)
        img = randn(rng, 30, 30) .* 0.3
        _place_source!(img, 15.0, 15.0, 25.0, 1.5)
        result = matched_filter(img, kern; sigma=3.0)
        for p in result.peaks
            @test 1 ≤ p[1] ≤ 30
            @test 1 ≤ p[2] ≤ 30
        end
    end
end

# ---------------------------------------------------------------------------
# Edge cases and pathological inputs
# ---------------------------------------------------------------------------
@testset "matched_filter: edge cases" begin
    rng = StableRNG(707)

    @testset "empty result when no peaks exceed threshold" begin
        kern = _gaussian_kernel(9, 1.5)
        img = randn(rng, 40, 40)  # pure noise
        inv_var = fill(1.0, size(img))
        result = matched_filter(img, kern; inv_var, sigma=10.0)
        @test isempty(result.peaks)
        @test isempty(result.peak_significances)
    end

    @testset "degenerate uniform kernel does not divide by zero" begin
        kern = fill(1.0, 5, 5)
        img = randn(rng, 20, 20)
        @test_nowarn matched_filter(img, kern; sigma=3.0)
    end

    @testset "degenerate zero kernel does not error" begin
        kern = zeros(5, 5)
        img = randn(rng, 20, 20)
        @test_nowarn matched_filter(img, kern; sigma=3.0)
    end

    @testset "kernel larger than image does not error" begin
        kern = _gaussian_kernel(31, 5.0)
        img = randn(rng, 20, 20)
        @test_nowarn matched_filter(img, kern; sigma=3.0)
    end

    @testset "1×1 kernel" begin
        kern = fill(1.0, 1, 1)
        img = randn(rng, 20, 20)
        @test_nowarn matched_filter(img, kern; sigma=3.0)
    end

    @testset "even-sized kernel throws (from correlate validation)" begin
        kern = rand(4, 4)
        img = randn(rng, 20, 20)
        @test_throws ArgumentError matched_filter(img, kern)
    end

    @testset "sigma = 0 returns all local maxima" begin
        kern = _gaussian_kernel(9, 1.5)
        img = randn(rng, 30, 30) .* 0.3
        _place_source!(img, 15.0, 15.0, 40.0, 1.5)
        inv_var = fill(4.0, size(img))
        result_all = matched_filter(img, kern; inv_var, sigma=0.0)
        result_5   = matched_filter(img, kern; inv_var, sigma=5.0)
        @test length(result_all.peaks) ≥ length(result_5.peaks)
    end

    @testset "sources near image border are detected" begin
        kern = _gaussian_kernel(11, 2.0)
        img = randn(rng, 60, 60) .* 0.2
        _place_source!(img, 3.0, 3.0, 50.0, 2.0)    # top-left corner
        _place_source!(img, 57.0, 57.0, 50.0, 2.0)  # bottom-right corner
        inv_var = fill(25.0, size(img))
        result = matched_filter(img, kern; inv_var, sigma=5.0)
        @test length(result.peaks) ≥ 1  # at least one should be found
    end
end
