using FFTW
using BenchmarkTools
using Random
using Statistics
using Printf
using ImageFiltering
using OffsetArrays

# -----------------------------
# Utilities
# -----------------------------

round_to_multiple(n, k) = k * round(Int, n / k)

function gaussian_kernel(k::Integer; fwhm = k / 3)
    isodd(k) || throw(ArgumentError("kernel size must be odd"))

    σ = fwhm / (2sqrt(2log(2)))
    r = k ÷ 2

    ker = Matrix{Float64}(undef, k, k)

    @inbounds for j in 1:k, i in 1:k
        dy = j - r - 1
        dx = i - r - 1
        ker[j, i] = exp(-0.5 * (dx^2 + dy^2) / σ^2)
    end

    ker ./= sum(ker)
    return ker
end

# -----------------------------
# Direct valid correlation
# -----------------------------
#
# Computes:
#
#   out[y-r, x-r] = sum_{dy=-r:r, dx=-r:r} image[y+dy, x+dx] * kernel[dy, dx]
#
# for y, x away from the image boundary.
#
# This is correlation, not convolution. The kernel is not flipped.

function direct_corr_valid!(
    out::AbstractMatrix{T},
    image::AbstractMatrix{T},
    kernel::AbstractMatrix{T},
) where {T <: AbstractFloat}

    H, W = size(image)
    kH, kW = size(kernel)

    kH == kW || throw(ArgumentError("kernel must be square"))
    isodd(kH) || throw(ArgumentError("kernel size must be odd"))

    k = kH
    r = k ÷ 2

    size(out) == (H - 2r, W - 2r) ||
        throw(DimensionMismatch("out must have size $((H - 2r, W - 2r))"))

    @inbounds for x in (r + 1):(W - r)
        ox = x - r

        for y in (r + 1):(H - r)
            oy = y - r
            acc = zero(T)

            for kx in 1:k
                ix = x + kx - r - 1

                @simd for ky in 1:k
                    iy = y + ky - r - 1
                    acc += image[iy, ix] * kernel[ky, kx]
                end
            end

            out[oy, ox] = acc
        end
    end

    return out
end

# -----------------------------
# FFT circular correlation setup
# -----------------------------

struct FFTCorrWorkspace{T, PF, PI}
    a::Matrix{Complex{T}}
    tmp::Matrix{Complex{T}}
    kernel_hat::Matrix{Complex{T}}
    plan_fwd::PF
    plan_inv::PI
end

function embed_kernel_for_correlation(
    kernel::AbstractMatrix{T},
    H::Integer,
    W::Integer,
) where {T <: AbstractFloat}

    kH, kW = size(kernel)
    kH == kW || throw(ArgumentError("kernel must be square"))
    isodd(kH) || throw(ArgumentError("kernel size must be odd"))

    k = kH
    r = k ÷ 2

    padded = zeros(Complex{T}, H, W)

    # Put the zero-displacement kernel sample at [1, 1].
    # Negative offsets wrap to the end. This makes the FFT operation
    # implement circular correlation with the same coordinate convention
    # as direct_corr_valid!.
    @inbounds for kx in 1:k
        dx = kx - r - 1
        x = mod1(dx + 1, W)

        for ky in 1:k
            dy = ky - r - 1
            y = mod1(dy + 1, H)

            padded[y, x] = kernel[ky, kx]
        end
    end

    return padded
end

function FFTCorrWorkspace(
    image::AbstractMatrix{T},
    kernel::AbstractMatrix{T},
) where {T <: AbstractFloat}

    H, W = size(image)

    a = Matrix{Complex{T}}(undef, H, W)
    tmp = Matrix{Complex{T}}(undef, H, W)

    kernel_padded = embed_kernel_for_correlation(kernel, H, W)

    plan_fwd = plan_fft!(a; flags = FFTW.MEASURE)
    plan_inv = plan_ifft!(tmp; flags = FFTW.MEASURE)

    kernel_hat = copy(kernel_padded)
    plan_fwd * kernel_hat

    return FFTCorrWorkspace(a, tmp, kernel_hat, plan_fwd, plan_inv)
end

function fft_corr_circular!(
    work::FFTCorrWorkspace{T},
    image::AbstractMatrix{T},
) where {T <: AbstractFloat}

    H, W = size(image)

    @inbounds for j in 1:W, i in 1:H
        work.a[i, j] = Complex{T}(image[i, j], zero(T))
    end

    work.plan_fwd * work.a

    @inbounds @simd for idx in eachindex(work.tmp, work.a, work.kernel_hat)
        work.tmp[idx] = work.a[idx] * conj(work.kernel_hat[idx])
    end

    work.plan_inv * work.tmp

    return work.tmp
end

# -----------------------------
# ImageFiltering.jl helper
# -----------------------------

function imagefiltering_workspace(image::AbstractMatrix{T}, kernel::AbstractMatrix{T}) where {T}
    out = similar(image)
    kern_centered = centered(kernel)
    return out, kern_centered
end

# -----------------------------
# Benchmark driver
# -----------------------------

function check_correctness(image, kernel; rtol = 1e-10, atol = 1e-10)
    H, W = size(image)
    k = size(kernel, 1)
    r = k ÷ 2

    out_direct = Matrix{Float64}(undef, H - 2r, W - 2r)
    direct_corr_valid!(out_direct, image, kernel)

    work = FFTCorrWorkspace(image, kernel)
    out_fft = fft_corr_circular!(work, image)

    maxerr = maximum(abs.(
        out_direct .- real.(view(out_fft, (r + 1):(H - r), (r + 1):(W - r)))
    ))

    return maxerr
end

function run_benchmarks(;
    base_sizes = (500, 1000, 2000),
    kernel_sizes = (5, 11, 21),
    rng = Xoshiro(1234),
)

    rows = NamedTuple[]

    println()
    println("Matched-filter correlation benchmark")
    println("Direct method computes valid interior output.")
    println("FFT method computes full circular correlation; timing includes full image FFT.")
    println("FFT planning time is excluded.")
    println()

    for k in kernel_sizes
        kernel = gaussian_kernel(k)
        r = k ÷ 2

        for n0 in base_sizes
            n = round_to_multiple(n0, k)

            image = randn(rng, Float64, n, n)

        # Workspaces
        out_direct = Matrix{Float64}(undef, n - 2r, n - 2r)
        work = FFTCorrWorkspace(image, kernel)

        out_imfilter_fir, kern_centered = imagefiltering_workspace(image, kernel)
        out_imfilter_fft = similar(image)
        out_imfilter_auto = similar(image)
        out_imfilter_auto_factored = similar(image)
        out_imfilter_auto_nofactored = similar(image)

        # Warmup
        direct_corr_valid!(out_direct, image, kernel)
        out_fft = fft_corr_circular!(work, image)

        imfilter!(out_imfilter_fir, image, kern_centered, "circular", Algorithm.FIR())
        imfilter!(out_imfilter_fft, image, kern_centered, "circular", Algorithm.FFT())
        imfilter!(out_imfilter_auto, image, kern_centered, "circular")

        # Correctness checks on the same valid interior.
        #
        # ImageFiltering returns full-size output, so compare its interior against
        # direct_corr_valid!.
        direct_region = out_direct
        fft_region = real.(view(out_fft, (r + 1):(n - r), (r + 1):(n - r)))
        imfilter_fir_region = view(out_imfilter_fir, (r + 1):(n - r), (r + 1):(n - r))
        imfilter_fft_region = view(out_imfilter_fft, (r + 1):(n - r), (r + 1):(n - r))
        imfilter_auto_region = view(out_imfilter_auto, (r + 1):(n - r), (r + 1):(n - r))

        maxerr_fft = maximum(abs.(direct_region .- fft_region))
        maxerr_imfilter_fir = maximum(abs.(direct_region .- imfilter_fir_region))
        maxerr_imfilter_fft = maximum(abs.(direct_region .- imfilter_fft_region))
        maxerr_imfilter_auto = maximum(abs.(direct_region .- imfilter_auto_region))

        # Benchmarks
        b_direct = @benchmark direct_corr_valid!($out_direct, $image, $kernel) evals=1 samples=10
        b_fft = @benchmark fft_corr_circular!($work, $image) evals=1 samples=10

        b_imfilter_fir = @benchmark imfilter!(
            $out_imfilter_fir,
            $image,
            $kern_centered,
            "circular",
            $(Algorithm.FIR()),
        ) evals=1 samples=10

        b_imfilter_fft = @benchmark imfilter!(
            $out_imfilter_fft,
            $image,
            $kern_centered,
            "circular",
            $(Algorithm.FFT()),
        ) evals=1 samples=10

        b_imfilter_auto = @benchmark imfilter!(
            $out_imfilter_auto,
            $image,
            $kern_centered,
            "circular",
        ) evals=1 samples=10

        b_imfilter_auto_factored = @benchmark imfilter!(
            $out_imfilter_auto_factored,
            $image,
            $(ImageFiltering.factorkernel(kern_centered)),
            "circular",
        ) evals=1 samples=10

        b_imfilter_auto_nofactored = @benchmark imfilter!(
            $out_imfilter_auto_nofactored,
            $image,
            $((kernel,)),
            "circular",
        ) evals=1 samples=10

        t_direct = median(b_direct).time / 1e6
        t_fft = median(b_fft).time / 1e6
        t_imfilter_fir = median(b_imfilter_fir).time / 1e6
        t_imfilter_fft = median(b_imfilter_fft).time / 1e6
        t_imfilter_auto = median(b_imfilter_auto).time / 1e6
        t_imfilter_auto_factored = median(b_imfilter_auto_factored).time / 1e6
        t_imfilter_auto_nofactored = median(b_imfilter_auto_nofactored).time / 1e6
        push!(rows, (
            image_size = n,
            kernel_size = k,
            direct_valid_ms = t_direct,
            custom_fft_circular_ms = t_fft,
            imfilter_fir_circular_ms = t_imfilter_fir,
            imfilter_fft_circular_ms = t_imfilter_fft,
            imfilter_auto_circular_ms = t_imfilter_auto,
            imfilter_auto_factored_circular_ms = t_imfilter_auto_factored,
            imfilter_auto_nofactored_circular_ms = t_imfilter_auto_nofactored,
            custom_fft_over_direct = t_fft / t_direct,
            imfilter_fir_over_direct = t_imfilter_fir / t_direct,
            imfilter_fft_over_direct = t_imfilter_fft / t_direct,
            imfilter_auto_over_direct = t_imfilter_auto / t_direct,
            maxerr_custom_fft = maxerr_fft,
            maxerr_imfilter_fir = maxerr_imfilter_fir,
            maxerr_imfilter_fft = maxerr_imfilter_fft,
            maxerr_imfilter_auto = maxerr_imfilter_auto,
        ))

        @printf(
            "N=%4d | k=%2d | direct=%9.3f ms | customFFT=%9.3f ms | imFIR=%9.3f ms  | imFFT=%9.3f ms | imAuto=%9.3f ms | imAutoFactored=%9.3f ms | imAutoNoFactored=%9.3f ms\n",
            n, k, t_direct, t_fft, t_imfilter_fir, t_imfilter_fft, t_imfilter_auto, t_imfilter_auto_factored, t_imfilter_auto_nofactored
        )

        @printf(
            "                    err customFFT=%.3e  imFIR=%.3e  imFFT=%.3e  imAuto=%.3e\n",
            maxerr_fft, maxerr_imfilter_fir, maxerr_imfilter_fft, maxerr_imfilter_auto,
        )
        end

        println()
    end

    return rows
end

results = run_benchmarks()
