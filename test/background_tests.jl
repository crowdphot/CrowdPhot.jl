using Test
using CrowdPhot
using CrowdPhot.Background
using CrowdPhot.Background: _catmull_rom, _catmull_rom_weights, _bicubic_zoom, _fill_nans!
using StableRNGs: StableRNG
using Statistics: mean, std

# ─── Shared fixture ────────────────────────────────────────────────────────────
const _FLAT100  = fill(100.0, 32, 32)
const _FLAT200  = fill(200.0, 64, 64)
# Realistic mock image used across multiple tests.
const _MOCK_IMG = make_gaussians_image(40, (128, 128); rng = StableRNG(7), background = 200.0,
                                        read_noise = 5.0, gain = 1.5)

@testset "Estimator constructor promotion" begin
    # Constructor inputs should share a promoted floating storage type.
    mmm_default = MMMBackground()
    @test mmm_default isa MMMBackground{Float64}
    @test mmm_default.median_factor === 3.0
    @test mmm_default.mean_factor === 2.0

    mmm_mixed = MMMBackground(Int8(3), Float32(2))
    @test mmm_mixed isa MMMBackground{Float32}
    @test mmm_mixed.median_factor === Float32(3)
    @test mmm_mixed.mean_factor === Float32(2)

    mmm_keyword = MMMBackground(; median_factor = 3, mean_factor = 2.0)
    @test mmm_keyword isa MMMBackground{Float64}
    @test mmm_keyword.median_factor === 3.0
    @test mmm_keyword.mean_factor === 2.0

    # Integers not allowed
    @test_throws MethodError MMMBackground{Int}(3, 2)

    loc32 = BiweightLocationBackground(Float32(6))
    @test loc32 isa BiweightLocationBackground{Float32}
    @test loc32.c === Float32(6)

    # Integers not allowed
    @test_throws MethodError BiweightLocationBackground{Int}(6)

    scale32 = BiweightScaleRMS(; c = Float32(9))
    @test scale32 isa BiweightScaleRMS{Float32}
    @test scale32.c === Float32(9)

    # Integers not allowed
    @test_throws MethodError BiweightScaleRMS{Int}(c = 9)
end

@testset "estimate_background — scalar" begin
    @testset "constant image" begin
        r = estimate_background(_FLAT100)
        @test r.bkg     ≈ 100.0
        @test r.bkg_rms ≈ 0.0
    end

    @testset "sigma=nothing disables clipping" begin
        data = fill(50.0, 20, 20)
        r = estimate_background(data; sigma = nothing)
        @test r.bkg ≈ 50.0
    end

    @testset "mask excludes pixels" begin
        img = fill(10.0, 8, 8)
        img[1:4, :] .= 9999.0   # large outlier region
        mask = falses(8, 8)
        mask[1:4, :] .= true
        r = estimate_background(img; mask, sigma = nothing)
        @test r.bkg ≈ 10.0
    end

    @testset "non-finite values are silently excluded" begin
        img = fill(5.0, 16, 16)
        img[1, 1] = NaN
        img[2, 2] = Inf
        img[3, 3] = -Inf
        r = estimate_background(img; sigma = nothing)
        @test r.bkg ≈ 5.0
    end

    @testset "all-NaN image throws" begin
        @test_throws ArgumentError estimate_background(fill(NaN, 10, 10))
    end

    @testset "all pixels masked throws" begin
        mask = trues(8, 8)
        @test_throws ArgumentError estimate_background(_FLAT100[1:8, 1:8]; mask)
    end

    @testset "sigma clipping removes all pixels throws" begin
        # sigma=0 retains only pixels exactly equal to the median.
        # A 2×2 array with all-distinct values has a fractional median (2.5),
        # so every pixel is rejected and the function must throw.
        data2 = [1.0 2.0; 3.0 4.0]
        @test_throws ArgumentError estimate_background(data2; sigma = 0.0, maxiters = 10)
    end

    @testset "asymmetric sigma clipping" begin
        # sigma_clip! accepts independent low/high thresholds.  Test each
        # direction in isolation so that the permissive side does not keep
        # the standard deviation inflated and prevent the tight side from
        # converging.

        # Data with only high-side outliers (5000).
        data_high = 100.0 .+ randn(StableRNG(2024), 20, 20) .* 2.0
        data_high[1, :] .= 5000.0

        work_high = float(copy(data_high))
        n_high = sigma_clip!(work_high, 100.0, 3.0; maxiters = 10)
        ret_high = view(vec(work_high), 1:n_high)
        @test maximum(ret_high) < 4000 # bright outliers removed by tight high side

        # Data with only low-side outliers (-10).
        data_low = 100.0 .+ randn(StableRNG(2024), 20, 20) .* 2.0
        data_low[1, :] .= -10.0

        work_low = float(copy(data_low))
        n_low = sigma_clip!(work_low, 3.0, 100.0; maxiters = 10)
        ret_low = view(vec(work_low), 1:n_low)
        @test minimum(ret_low) > -5 # dim outliers removed by tight low side

        # Symmetric clipping removes outliers on both sides.
        data_both = 100.0 .+ randn(StableRNG(2024), 20, 20) .* 2.0
        data_both[1, :] .= 5000.0
        data_both[end, :] .= -10.0

        work_both = float(copy(data_both))
        n_both = sigma_clip!(work_both, 3.0, 3.0; maxiters = 10)
        ret_both = view(vec(work_both), 1:n_both)
        @test maximum(ret_both) < 4000   # bright clipped
        @test minimum(ret_both) > -5     # dim clipped
    end

    @testset "array-shaped inputs are accepted" begin
        # Estimators should operate on multidimensional arrays without callers flattening data first.
        cube = fill(12.0, 3, 4, 2)
        r = estimate_background(cube; sigma = nothing)
        @test r.bkg ≈ 12.0
        @test MeanBackground()(cube) ≈ 12.0
        @test size(MMMBackground()(cube; dims = 1)) == (1, 4, 2)
        @test size(sigma_clip(cube, 3.0)) == size(cube)
    end
end

@testset "Location estimators on skewed data" begin
    # Build a positively skewed distribution: clean background plus bright sources.
    bkg_pixels = fill(100.0, 900)
    source_pixels = fill(1000.0, 100)   # 10 % contamination
    data = vcat(bkg_pixels, source_pixels)

    @testset "SExtractorBackground falls back to median for skewed input" begin
        # With strong positive skew the SE estimator should return roughly 100.
        val = SExtractorBackground()(data)
        @test val < 110.0   # clearly below the contaminated mean
    end

    @testset "MMMBackground also suppresses positive skew" begin
        val = MMMBackground()(data)
        @test val < 110.0
    end

    @testset "BiweightLocationBackground robust to outliers" begin
        d_clean  = randn(StableRNG(1), 200) .+ 5.0
        d_dirty  = vcat(d_clean, fill(1e4, 10))
        val_clean = BiweightLocationBackground()(d_clean)
        val_dirty = BiweightLocationBackground()(d_dirty)
        @test abs(val_clean - 5.0) < 0.5
        # Outliers should have minimal effect on the biweight location.
        @test abs(val_dirty - val_clean) < 1.0
    end
end

@testset "RMS estimators" begin
    data = fill(3.0, 50)

    @testset "all estimators return 0 for constant data" begin
        @test StdRMS()(data)             ≈ 0.0
        @test MADStdRMS()(data)          ≈ 0.0
        @test BiweightScaleRMS()(data)   ≈ 0.0
    end

    @testset "StdRMS matches std for random data" begin
        v = randn(StableRNG(5), 500)
        @test StdRMS()(v) ≈ std(v; corrected = false) atol = 1e-12
    end

    @testset "MADStdRMS is approximately equal to StdRMS for Gaussian data" begin
        v = randn(StableRNG(6), 5000)
        @test MADStdRMS()(v) ≈ StdRMS()(v) rtol = 0.10
    end

    @testset "BiweightScaleRMS robust to outliers" begin
        v_clean = randn(StableRNG(8), 300)
        v_dirty = vcat(v_clean, fill(1e5, 5))
        sc  = BiweightScaleRMS()(v_clean)
        sd  = BiweightScaleRMS()(v_dirty)
        @test abs(sd - sc) < 0.2
    end
end

@testset "Background2D" begin
    @testset "constant image recovers exact background" begin
        b = Background2D(_FLAT200, 16)
        @test b.background ≈ _FLAT200
        @test all(iszero, b.background_rms)
        @test b.box_size == (16, 16)
        @test size(b.mesh_background) == (4, 4)
        b2 = Background2D(_FLAT200, (16, 32))
        @test b2.background ≈ _FLAT200
        @test all(iszero, b2.background_rms)
        @test b2.box_size == (16, 32)
        @test size(b2.mesh_background) == (4, 2)
    end

    @testset "output size matches input" begin
        b = Background2D(_MOCK_IMG, 32)
        @test size(b.background)     == size(_MOCK_IMG)
        @test size(b.background_rms) == size(_MOCK_IMG)
    end

    @testset "non-multiple image dimensions handled (ceil division)" begin
        img = fill(7.0, 65, 65)
        b   = Background2D(img, 16)
        @test size(b.background) == (65, 65)   # ceil(65/16)=5 boxes → output matches input
    end

    @testset "mask excludes pixels in mesh computation" begin
        img  = fill(150.0, 64, 64)
        img[1:32, :] .= 9999.0   # large contaminant in the top half
        mask = falses(64, 64)
        mask[1:32, :] .= true
        b = Background2D(img, 16; mask, sigma = nothing)
        # All background values should be close to 150 despite the contaminated region.
        @test all(v -> isapprox(v, 150.0; atol = 1.0), b.background)
    end

    @testset "NaN pixels in image are excluded with warning" begin
        img      = fill(50.0, 32, 32)
        img[1,1] = NaN
        @test_logs (:warn,) Background2D(img, 16)
        b = (@test_logs (:warn,) Background2D(img, 16))
        @test b.background ≈ fill(50.0, 32, 32) atol = 1e-10
    end

    @testset "box with too few valid pixels is excluded and filled" begin
        img  = fill(100.0, 32, 32)
        # Mask almost an entire quadrant so that box hits exclude_percentile.
        mask = falses(32, 32)
        mask[1:16, 1:15] .= true   # 15 of 16 columns → > 90 % masked
        b = Background2D(img, 16; mask, exclude_percentile = 10.0, sigma = nothing)
        # The excluded box should be filled — background stays near 100.
        @test all(v -> isapprox(v, 100.0; atol = 1.0), b.background)
    end

    @testset "all boxes excluded throws" begin
        img  = fill(0.0, 16, 16)
        mask = trues(16, 16)     # mask everything
        @test_throws ArgumentError Background2D(img, 16; mask)
    end

    @testset "filter_size=1 is a no-op" begin
        b_filtered = Background2D(_MOCK_IMG, 32; filter_size = (3, 3))
        b_unfiltered = Background2D(_MOCK_IMG, 32; filter_size = 1)
        # Results should differ (filtering smooths the mesh).
        # This simply checks that the option is accepted without error.
        @test size(b_filtered.background)   == size(_MOCK_IMG)
        @test size(b_unfiltered.background) == size(_MOCK_IMG)
    end

    @testset "odd filter_size check" begin
        @test_throws ArgumentError Background2D(_FLAT200, 16; filter_size = 2)
    end

    @testset "single box covers entire image" begin
        b = Background2D(fill(7.0, 16, 16), 16)   # exactly one 16×16 box, no padding
        @test all(v -> isapprox(v, 7.0; atol = 1e-10), b.background)
    end

    @testset "gradient background is estimated without large bias" begin
        ramp = [float(i) for i in 1:64, _ in 1:64]   # background grows from 1 to 64 along rows
        b    = Background2D(ramp, 8; sigma = nothing, filter_size = 1)
        # The interpolated background should track the true ramp to within ±5.
        @test maximum(abs.(b.background .- ramp)) < 5.0
    end

    @testset "realistic image: background close to truth" begin
        b = Background2D(_MOCK_IMG, 32; sigma = 3.0)
        # True background was 200 ADU; expect the estimate within ±5 ADU everywhere.
        @test all(v -> abs(v - 200.0) < 5.0, b.background)
    end

    @testset "different estimators are accepted" begin
        for est in (MeanBackground(), MedianBackground(), SExtractorBackground(),
                    MMMBackground(), BiweightLocationBackground())
            for rms in (StdRMS(), MADStdRMS(), BiweightScaleRMS())
                b = Background2D(_FLAT200, 16; estimator = est, rms_estimator = rms, sigma = nothing)
                @test b.background ≈ _FLAT200 atol = 0.1
            end
        end
    end

    @testset "coverage_mask pixels are set to fill_value in output maps" begin
        img = fill(100.0, 64, 64)
        cov = falses(64, 64)
        cov[1:20, :] .= true   # left strip: no data coverage
        cov[:, 50:64] .= true  # right strip: no data coverage

        # Default fill_value = 0.0
        b = Background2D(img, 16; coverage_mask = cov, sigma = nothing, filter_size = 1)
        @test all(b.background[cov] .== 0.0)
        @test all(b.background_rms[cov] .== 0.0)

        # Interior (non-coverage) pixels should still be close to the true background.
        interior = .!cov
        @test mean(abs, b.background[interior] .- img[interior]) < 5.0

        # Custom fill_value.
        b2 = Background2D(img, 16; coverage_mask = cov, fill_value = -99.0,
                          sigma = nothing, filter_size = 1)
        @test all(b2.background[cov] .== -99.0)
        @test all(b2.background_rms[cov] .== -99.0)

        # With larger median filter
        b3 = Background2D(img, 16; coverage_mask = cov, fill_value = -99.0,
                          sigma = nothing, filter_size = 3)
        @test all(b3.background[cov] .== -99.0)
        @test all(b3.background_rms[cov] .== -99.0)
    end

    @testset "_fill_nans! propagates across multi-cell holes" begin
        # 5×5 mesh: 2×2 NaN block in the center
        mesh = fill(1.0, 5, 5)
        mesh[2:3, 2:3] .= NaN
        _fill_nans!(mesh)
        @test all(isfinite, mesh)
        # Filled values should be close to 1.0 (neighbour average).
        @test all(≈(1), mesh)
    end
end

@testset "bicubic zoom internals" begin
    @testset "_catmull_rom_weights consistency" begin
        # The separated weight form must agree with the direct Catmull-Rom evaluation.
        rng = StableRNG(99)
        for _ in 1:200
            p0, p1, p2, p3 = randn(rng, 4)
            t = rand(rng)
            direct = _catmull_rom(p0, p1, p2, p3, t)
            w1, w2, w3, w4 = _catmull_rom_weights(t)
            via_weights = w1 * p0 + w2 * p1 + w3 * p2 + w4 * p3
            @test direct ≈ via_weights
        end
    end

    @testset "weights sum to unity" begin
        # Catmull-Rom basis functions partition unity for any t ∈ [0,1].
        rng = StableRNG(123)
        for _ in 1:100
            t = rand(rng)
            w1, w2, w3, w4 = _catmull_rom_weights(t)
            @test w1 + w2 + w3 + w4 ≈ 1.0
        end
    end

    @testset "weights reproduce endpoints" begin
        # At t=0 the interpolation should be exactly p1.
        w1, w2, w3, w4 = _catmull_rom_weights(0.0)
        @test w2 ≈ 1.0
        @test w1 + w3 + w4 ≈ 0.0

        # At t=1 the interpolation should be exactly p2.
        w1, w2, w3, w4 = _catmull_rom_weights(1.0)
        @test w3 ≈ 1.0
        @test w1 + w2 + w4 ≈ 0.0
    end

    @testset "_bicubic_zoom output properties" begin
        # Output should be the requested size, of floating-point type, and contain only finite values.
        mesh = randn(StableRNG(1), 10, 12)
        result = _bicubic_zoom(mesh, 50, 70)
        @test size(result) == (50, 70)
        @test eltype(result) == Float64
        @test all(isfinite, result)

        # Integer-valued mesh should promote to floating output.
        int_mesh = fill(3, 4, 5)
        z = _bicubic_zoom(int_mesh, 20, 30)
        @test eltype(z) == Float64
        @test all(isfinite, z)

        # Float32 input should produce Float32 output.
        float32_mesh = fill(Float32(2.5), 6, 8)
        z32 = _bicubic_zoom(float32_mesh, 12, 16)
        @test eltype(z32) == Float32
    end

    @testset "constant mesh → constant output" begin
        for val in (0.0, 5.0, -3.0, 1e3)
            mesh = fill(val, 5, 7)
            zoomed = _bicubic_zoom(mesh, 20, 30)
            @test all(x -> x ≈ val, zoomed)
        end
    end

    @testset "single-cell mesh is reproduced" begin
        # A 1×1 mesh should tile the value across the output.
        mesh = fill(42.0, 1, 1)
        zoomed = _bicubic_zoom(mesh, 15, 15)
        @test all(x -> x ≈ 42.0, zoomed)
    end

    @testset "corners of the mesh are preserved at output corners" begin
        mesh = randn(StableRNG(5), 6, 8)
        result = _bicubic_zoom(mesh, 60, 80)
        @test result[1, 1] ≈ mesh[1, 1]
        @test result[end, 1] ≈ mesh[end, 1]
        @test result[1, end] ≈ mesh[1, end]
        @test result[end, end] ≈ mesh[end, end]
    end

    @testset "interpolation is contained within the convex hull" begin
        # Bicubic Catmull-Rom can overshoot slightly near sharp edges, but for
        # smooth data the output should not stray far from the input range.
        rng = StableRNG(2)
        mesh = randn(rng, 5, 5)
        # Smooth so the mesh has no sharp gradients.
        mesh_smooth = (mesh .+ circshift(mesh, (1, 0)) .+ circshift(mesh, (0, 1))) ./ 3
        lo, hi = extrema(mesh_smooth)
        result = _bicubic_zoom(mesh_smooth, 40, 50)
        @test minimum(result) ≥ lo - 0.1 * (hi - lo)
        @test maximum(result) ≤ hi + 0.1 * (hi - lo)
    end

    @testset "no-op zoom (H=M, W=N) preserves pixel centers" begin
        # When the output size equals the mesh size, interior pixels (away from
        # the clamped boundary) should approximately reproduce the mesh.
        mesh = randn(StableRNG(7), 10, 12)
        result = _bicubic_zoom(mesh, 10, 12)
        # Interior 6×8 region (avoid boundary clamping effects).
        @test result[3:8, 3:10] ≈ mesh[3:8, 3:10] rtol = 1e-6
    end
end

@testset "make_gaussians_image" begin
    rng = StableRNG(42)
    img = make_gaussians_image(30, (128, 128); rng)

    @testset "output shape" begin
        @test size(img) == (128, 128)
    end

    @testset "all pixels are finite" begin
        @test all(isfinite, img)
    end

    @testset "background level roughly preserved" begin
        @test 100.0 < mean(img) < 400.0   # rough sanity check around 200 ADU
    end

    @testset "zero stars produces only background noise" begin
        img0 = make_gaussians_image(0, (64, 64); rng = StableRNG(1), background = 100.0,
                                     read_noise = 2.0, gain = 1.0)
        @test mean(img0) ≈ 100.0 rtol = 0.05
    end
end
