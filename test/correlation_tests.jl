using CrowdPhot: correlate, correlate!
using LinearAlgebra
using OffsetArrays: OffsetArray
using StableRNGs: StableRNG
using Test

# Reference: direct full-image correlation with replicate padding.
# This is a straightforward O(H·W·kr·kc) implementation used as the ground
# truth for all correctness checks.
function _direct_corr(img::AbstractMatrix, kernel::AbstractMatrix)
    H, W = size(img)
    kr, kc = size(kernel)
    cr, cc = (kr + 1) ÷ 2, (kc + 1) ÷ 2
    pt, pb = cr - 1, kr - cr
    pl, pr = cc - 1, kc - cc
    padded = similar(img, H + pt + pb, W + pl + pr)
    padded[pt+1:pt+H, pl+1:pl+W] .= img
    # replicate edges
    for c in 1:size(padded, 2), r in 1:pt
        padded[r, c] = padded[pt+1, c]
    end
    for c in 1:size(padded, 2), r in pt+H+1:size(padded, 1)
        padded[r, c] = padded[pt+H, c]
    end
    for r in 1:size(padded, 1), c in 1:pl
        padded[r, c] = padded[r, pl+1]
    end
    for r in 1:size(padded, 1), c in pl+W+1:size(padded, 2)
        padded[r, c] = padded[r, pl+W]
    end
    out = similar(img)
    for col in 1:W, row in 1:H
        acc = zero(eltype(out))
        for kc_i in 1:kc, kr_i in 1:kr
            acc += padded[row+kr_i-1, col+kc_i-1] * kernel[kr_i, kc_i]
        end
        out[row, col] = acc
    end
    return out
end

function _direct_corr_zero(img::AbstractMatrix, kernel::AbstractMatrix)
    H, W = size(img)
    kr, kc = size(kernel)
    cr, cc = (kr + 1) ÷ 2, (kc + 1) ÷ 2
    pt, pb = cr - 1, kr - cr
    pl, pr = cc - 1, kc - cc
    padded = zeros(eltype(img), H + pt + pb, W + pl + pr)
    padded[pt+1:pt+H, pl+1:pl+W] .= img
    out = similar(img)
    for col in 1:W, row in 1:H
        acc = zero(eltype(out))
        for kc_i in 1:kc, kr_i in 1:kr
            acc += padded[row+kr_i-1, col+kc_i-1] * kernel[kr_i, kc_i]
        end
        out[row, col] = acc
    end
    return out
end

@testset "correlation" begin

    rng = StableRNG(42)
    img = rand(rng, 10, 10)

    @testset "inseparable 3×3, replicate padding" begin
        kern = rand(rng, 3, 3)
        out = correlate(img, kern, :replicate)
        ref = _direct_corr(img, kern)
        @test out ≈ ref
    end

    @testset "inseparable 3×3, zero padding" begin
        kern = rand(rng, 3, 3)
        out = correlate(img, kern, :zero)
        ref = _direct_corr_zero(img, kern)
        @test out ≈ ref
    end

    @testset "separable rank-1 kernel (outer product)" begin
        u = [0.2, 0.6, 0.2]
        v = [0.25, 0.5, 0.25]
        kern = u * v'
        out = correlate(img, kern, :replicate)
        ref = _direct_corr(img, kern)
        @test out ≈ ref
    end

    @testset "in-place" begin
        kern = rand(rng, 3, 3)
        out_ip = similar(img)
        correlate!(out_ip, img, kern, :replicate)
        ref = _direct_corr(img, kern)
        @test out_ip ≈ ref
    end

    @testset "pre-factored tuple of 1D kernels" begin
        col_vec = reshape(Float64[0.2, 0.6, 0.2], 3, 1)
        row_vec = reshape(Float64[0.25, 0.5, 0.25], 1, 3)
        out = correlate(img, (col_vec, row_vec), :replicate)
        ref = _direct_corr(img, col_vec * row_vec)
        @test out ≈ ref
    end

    @testset "1×1 kernel" begin
        kern = fill(0.5, 1, 1)
        out = correlate(img, kern, :replicate)
        ref = _direct_corr(img, kern)
        @test out ≈ ref
    end

    @testset "5×5 random kernel" begin
        kern = rand(rng, 5, 5)
        out = correlate(img, kern, :replicate)
        ref = _direct_corr(img, kern)
        @test out ≈ ref
    end

    @testset "15×15 separable Gaussian kernel" begin
        x = LinRange(-3, 3, 15)
        g1d = exp.(-0.5 .* x .^ 2)
        g1d ./= sum(g1d)
        kern = g1d * g1d'
        out = correlate(img, kern, :replicate)
        ref = _direct_corr(img, kern)
        @test out ≈ ref rtol = 1e-12
    end

    @testset "100×100 image, 7×7 kernel" begin
        img_large = rand(rng, 100, 100)
        kern = rand(rng, 7, 7)
        out = correlate(img_large, kern, :replicate)
        ref = _direct_corr(img_large, kern)
        @test out ≈ ref
    end

    @testset "output size matches input" begin
        kern = rand(rng, 5, 5)
        for sz in [(20, 20), (31, 47), (100, 100)]
            img_sz = rand(rng, sz...)
            out = correlate(img_sz, kern, :replicate)
            @test size(out) == sz
        end
    end

    @testset "known-value spot check: separable" begin
        # Filter a single-pixel spike.  The output should be the reversed
        # kernel positioned at the spike location.  Since correlation is
        # not convolution (no flip), the output is the kernel itself.
        kern = Float64[1 2 1; 2 4 2; 1 2 1] ./ 16
        H, W = 9, 9
        img = zeros(H, W)
        img[5, 5] = 1.0   # spike at centre

        out = correlate(img, kern, :zero)

        # The kernel should be reproduced centred at the spike.
        # For a 9×9 image with a 3×3 kernel: out[4:6, 4:6] ≈ kern
        @test out[4:6, 4:6] ≈ kern
        @test all(out[1:3, :] .== 0)   # top zero region
        @test all(out[7:9, :] .== 0)   # bottom zero region
    end

    @testset "known-value spot check: inseparable" begin
        # A non-rank-1 kernel (full-rank 3×3).
        kern = Float64[1 0 0; 0 1 0; 0 0 1] ./ 3
        H, W = 9, 9
        img = zeros(H, W)
        img[5, 5] = 1.0

        out = correlate(img, kern, :zero)
        @test out[4:6, 4:6] ≈ kern
    end

    @testset "error: mismatched output axes" begin
        img = rand(rng, 10, 10)
        kern = rand(rng, 3, 3)
        out_wrong = zeros(9, 9)
        @test_throws DimensionMismatch correlate!(out_wrong, img, kern, :replicate)
    end

    @testset "error: bad border mode" begin
        img = rand(rng, 5, 5)
        kern = rand(rng, 3, 3)
        @test_throws ArgumentError correlate(img, kern, :circular)
    end

    @testset "error: even-sized kernel" begin
        img = rand(rng, 9, 9)
        @test_throws ArgumentError correlate(img, rand(rng, 4, 4), :replicate)
        @test_throws ArgumentError correlate(img, rand(rng, 3, 4), :replicate)
        @test_throws ArgumentError correlate(img, rand(rng, 4, 3), :replicate)
        # tuple with an even factor
        @test_throws ArgumentError correlate(img, (rand(rng, 4, 1), rand(rng, 1, 3)),
                                             :replicate)
        # odd kernels should still work
        out = correlate(img, rand(rng, 5, 5), :replicate)
        @test size(out) == (9, 9)
    end

    @testset "error: non-1-based indexing" begin
        img = rand(rng, 9, 9)
        # OffsetArray image
        img_off = OffsetArray(img, 0:8, 0:8)
        kern_odd = rand(rng, 5, 5)
        @test_throws ArgumentError correlate(img_off, kern_odd, :replicate)
        # OffsetArray kernel
        kern_off = OffsetArray(kern_odd, -2:2, -2:2)
        @test_throws ArgumentError correlate(img, kern_off, :replicate)
        # OffsetArray in a tuple factor
        col_off = OffsetArray(rand(rng, 5, 1), -2:2, 0:0)
        @test_throws ArgumentError correlate(img, (col_off, rand(rng, 1, 5)),
                                             :replicate)
    end
end
