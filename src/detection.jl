# detection.jl — Matched-filter source detection for CrowdPhot.jl
#
# Implements the formally correct matched filter for point-source detection
# under stationary (uncorrelated) Gaussian noise, following the same
# mathematics as photutils (Bradley et al.).

"""
    MatchedFilterResult{T}

Result of [`matched_filter`](@ref) source detection.

# Fields

- `image::AbstractMatrix{T}` — the input image (not copied; caller retains ownership).
- `inv_var::Union{AbstractMatrix{T}, Nothing}` — the inverse variance map (or `nothing`
  if uniform weights were assumed) with any non-finite entries set to 0.
- `smoothed_image::Matrix{T}` — the image correlated with the detection kernel.
- `smoothed_inv_var::Union{Matrix{T}, Nothing}` — the inverse variance map
  correlated with the squared kernel; used in the denominator of the
  significance calculation when `inv_var` is provided.
- `significance_map::Matrix{T}` — the detection statistic used for thresholding.
  When `inv_var` is provided, each pixel is the matched-filter SNR in standard
  deviation units.  When `inv_var = nothing`, the map is scaled per unit
  pixel noise; divide by the pixel-noise estimate to obtain true SNR.
- `kernel::Matrix{T}` — the normalized kernel used for the correlation.
- `kernel_norm::T` — the square root of the template normalization denominator
  before applying the final flux-normalizing scale: `√(Σ(P - P̄)^2)` for the
  zero-sum path, or `√(ΣP²)` for the non-zero-sum path.  It converts the raw
  flux estimate to SNR per unit pixel noise in the uniform-weight path.
- `peaks::Vector{CartesianIndex{2}}` — detected peak pixels as `(row, column)`
  Cartesian indices; for a peak `p`, `p[1]` is the row (y coordinate) and
  `p[2]` is the column (x coordinate).
- `peak_significances::Vector{T}` — significance values at each peak.
- `peak_fluxes::Vector{T}` — matched-filter flux estimate at each peak.
"""
struct MatchedFilterResult{T}
    image::AbstractMatrix{T}
    inv_var::Union{AbstractMatrix{T}, Nothing}
    smoothed_image::Matrix{T}
    smoothed_inv_var::Union{Matrix{T}, Nothing}
    significance_map::Matrix{T}
    kernel::Matrix{T}
    kernel_norm::T
    peaks::Vector{CartesianIndex{2}}
    peak_significances::Vector{T}
    peak_fluxes::Vector{T}
end

"""
    matched_filter(image::AbstractMatrix, kernel::AbstractMatrix;
                   inv_var::Union{AbstractMatrix, Nothing}=nothing,
                   normalize_zerosum::Bool=true,
                   sigma::Real=5.0,
                   border::Symbol=:replicate) → MatchedFilterResult

Perform matched-filter source detection on `image` using `kernel` as the
PSF template.

# Arguments

- `image` is the input image on which to perform detection.
- `kernel` will be correlated with the image using [`CrowdPhot.correlate`](@ref)
  and must follow that function's requirements (namely, odd dimensions).
  This is typically a rendered PSF model.
- `inv_var::Union{AbstractMatrix{<:AbstractFloat}, Nothing}`: optional
  inverse-variance map with the same shape as `image`.  Pixels with zero,
  non-finite, or `nothing`-suppressed weights are excluded from the significance
  calculation.  **The weights should represent background-only error** (e.g.
  ``1 / \\sigma_{\\mathrm{bkg}}^2``); do not include source Poisson noise,
  which would inflate the noise estimate at source positions and reduce
  detection sensitivity.
- `normalize_zerosum` controls whether the kernel is renormalized to have
  zero sum, which cancels any uniform background offset in the correlation.
  The default (`true`) is the safe choice and should be used whenever the
  image may contain an un-subtracted background.  Set to `false` when the
  image has been reliably background-subtracted.
- `sigma` is the detection threshold in units of the significance map.
- `border` controls edge handling: `:replicate` (default) or `:zero`.

# Returns

A [`MatchedFilterResult`](@ref) containing the significance map, peak
positions, and flux estimates.
"""
function matched_filter(image::AbstractMatrix{T}, kernel::AbstractMatrix;
                        inv_var::Union{AbstractMatrix{<:AbstractFloat}, Nothing}=nothing,
                        normalize_zerosum::Bool=true,
                        sigma::Real=5.0,
                        border::Symbol=:replicate) where {T}

    # Validate inverse variance
    if inv_var !== nothing
        size(inv_var) == size(image) ||
            throw(DimensionMismatch("inv_var size $(size(inv_var)) must match " *
                                    "image size $(size(image))"))
        # Replace non-finite weights with zero so those pixels are ignored rather
        # than propagating NaN/Inf through the correlations. Copy first: the map is
        # stored in the result and the docstring promises the caller's array is
        # left untouched (and `inv_var` may be a read-only `Fill`).
        if any(!isfinite, inv_var)
            inv_var = replace(x -> isfinite(x) ? x : zero(x), inv_var)
        end
    end

    # ----- 1. normalize the kernel -----
    P = float.(kernel)
    N = length(P)
    sumP = sum(P)
    sumP2 = sum(abs2, P)

    if normalize_zerosum
        # Zero-sum kernel: K = (P - P̄) / denom,  denom = ΣP² - (ΣP)²/N.
        # This makes sum(K) = 0 and E[correlate(F·P, K)] = F.
        denom = sumP2 - sumP^2 / N
        if denom <= 0
            K = P
            kernel_norm = one(float(T))
        else
            K = (P .- sumP / N) ./ denom
            kernel_norm = sqrt(denom)
        end
    else
        # Non-zero-sum kernel: K = P / ΣP².
        # Use when the image is confidently background-subtracted.
        if sumP2 <= 0
            K = P
            kernel_norm = one(float(T))
        else
            K = P ./ sumP2
            kernel_norm = sqrt(sumP2)
        end
    end

    # ----- 2. Compute correlations -----
    smoothed = correlate(image, K, border)

    if inv_var !== nothing
        # Guard the product so a non-finite image value at a zero-weight pixel
        # (a masked bad pixel is typically NaN in both the image and the weight
        # map) does not leak through as 0 * NaN = NaN.
        weighted_img = map((w, v) -> iszero(w) ? zero(w * v) : w * v, inv_var, image)
        num = correlate(weighted_img, K, border)

        K_sq = K .^ 2
        # weighted_img is dead after num; reuse its buffer for den.
        den = weighted_img
        correlate!(den, inv_var, K_sq, border)

        # Significance = num ./ sqrt(den).
        # num's eltype is already wide enough (promoted with Float64 kernel),
        # so reuse its buffer for the significance map.
        significance = num
        LV.@turbo for i in eachindex(den)
            significance[i] = ifelse(den[i] > 0, significance[i] / sqrt(den[i]), zero(eltype(significance)))
        end
        smoothed_inv_var = den
    else
        # Uniform weights: significance = smoothed * kernel_norm.
        # (kernel_norm converts the raw correlation to SNR per unit σ.)
        S = promote_type(T, typeof(kernel_norm))
        significance = similar(image, S)
        LV.@turbo for i in eachindex(smoothed)
            significance[i] = S(smoothed[i]) * kernel_norm
        end
        smoothed_inv_var = nothing
    end

    # ----- 3. Find local maxima -----
    peaks = findlocalmaxima(significance; edges=false)
    threshold_peaks(peaks, significance, sigma) = [p for p in peaks if significance[p] >= sigma]
    sig_peaks = threshold_peaks(peaks, significance, sigma)

    # ----- 4. Extract peak information -----
    peak_sigs = T[significance[p] for p in sig_peaks]
    # Flux estimate.  Without inv_var the normalized kernel makes smoothed[p] an
    # unbiased flux estimate directly (E[corr] = F).  With inv_var, use the
    # weighted estimate num[p]/den[p]/kernel_norm^2, equivalently
    # significance[p] / (sqrt(den[p]) * kernel_norm^2) since num was overwritten
    # by significance.  Zero-weight pixels drop out of num and den, so a masked
    # pixel in the footprint does not contaminate the estimate.
    if smoothed_inv_var === nothing
        peak_fluxes = T[smoothed[p] for p in sig_peaks]
    else
        kn2 = T(kernel_norm)^2
        peak_fluxes = T[smoothed_inv_var[p] > 0 ?
                        significance[p] / (sqrt(smoothed_inv_var[p]) * kn2) : zero(T)
                        for p in sig_peaks]
    end

    return MatchedFilterResult(
        image,
        inv_var,
        smoothed,
        smoothed_inv_var,
        significance,
        K,
        T(kernel_norm),
        sig_peaks,
        peak_sigs,
        peak_fluxes,
    )
end

"""
    matched_filter(image::AbstractMatrix, model::AbstractPSFModel; kws...)

Renders the provided `model` using [`render`](@ref) and performs
matched-filter detection on `image` using the rendered kernel.
"""
matched_filter(image::AbstractMatrix, model::AbstractPSFModel; kws...) = 
    matched_filter(image, render(model); kws...)

"""
    matched_filter(image::AbstractMatrix, fwhm::Number; kws...)

Convenience method for matched-filter detection using a
[`CircularGaussianPRF`](@ref) with the specified `fwhm`.
"""
matched_filter(image::AbstractMatrix, fwhm::Number; kws...) = 
    matched_filter(image, CircularGaussianPRF(; fwhm, x=0, y=0, bkg=0, flux=1); kws...)

"""
    matched_filter(image::AbstractMatrix, fwhm::NTuple{2, Number}; kws...)

Convenience method for matched-filter detection using a
[`GaussianPRF`](@ref) with y- and x-FWHM specified by the `fwhm` tuple as
`(y_fwhm, x_fwhm)`.
"""
matched_filter(image::AbstractMatrix, fwhm::Tuple{<:Number, <:Number}; kws...) = 
    matched_filter(image, GaussianPRF(; x=0, y=0, bkg=0, flux=1, x_fwhm=fwhm[2],
        y_fwhm=fwhm[1], theta=0); kws...)

# ---------------------------------------------------------------------------
# Local maximum finding
# ---------------------------------------------------------------------------

"""
    findlocalmaxima(img::AbstractMatrix; edges::Bool=true) -> Vector{CartesianIndex}

Return the coordinates of all local maxima in `img`.  A pixel is a local
maximum if its value is strictly greater than all 8 of its immediate
neighbors (3×3 window).  If `edges=false`, pixels on the image boundary
are excluded from consideration.  Each returned `CartesianIndex(row, col)`
has `row` corresponding to the `y` (first array dimension) coordinate and
`col` corresponding to the `x` (second array dimension) coordinate.

See also: [`matched_filter`](@ref).
"""
function findlocalmaxima(img::AbstractMatrix{T}; edges::Bool=true) where {T}
    H, W = size(img)
    maxima = Vector{CartesianIndex{2}}(undef, 0)

    H < 3 && W < 3 && return maxima

    # Interior pixels — no bounds checking needed, fast path.
    for col in 2:W-1, row in 2:H-1
        v = img[row, col]
        @inbounds if v > img[row-1, col-1] && v > img[row-1, col] &&
                     v > img[row-1, col+1] && v > img[row,   col-1] &&
                     v > img[row,   col+1] && v > img[row+1, col-1] &&
                     v > img[row+1, col]   && v > img[row+1, col+1]
            push!(maxima, CartesianIndex(row, col))
        end
    end

    edges || return maxima

    # Edge pixels: top and bottom rows (excluding corners already covered
    # by interior).
    for col in 2:W-1
        # top row
        v = img[1, col]
        if v > img[1, col-1] && v > img[1, col+1] &&
           v > img[2, col-1] && v > img[2, col] && v > img[2, col+1]
            push!(maxima, CartesianIndex(1, col))
        end
        # bottom row
        v = img[H, col]
        if v > img[H-1, col-1] && v > img[H-1, col] && v > img[H-1, col+1] &&
           v > img[H,   col-1] && v > img[H,   col+1]
            push!(maxima, CartesianIndex(H, col))
        end
    end

    # Left and right columns (excluding corners)
    for row in 2:H-1
        v = img[row, 1]
        if v > img[row-1, 1] && v > img[row-1, 2] &&
           v > img[row,   2] &&
           v > img[row+1, 1] && v > img[row+1, 2]
            push!(maxima, CartesianIndex(row, 1))
        end
        v = img[row, W]
        if v > img[row-1, W-1] && v > img[row-1, W] &&
           v > img[row,   W-1] &&
           v > img[row+1, W-1] && v > img[row+1, W]
            push!(maxima, CartesianIndex(row, W))
        end
    end

    # Four corners
    if H >= 2 && W >= 2
        # top-left
        v = img[1, 1]
        v > img[1, 2] && v > img[2, 1] && v > img[2, 2] &&
            push!(maxima, CartesianIndex(1, 1))
        # top-right
        v = img[1, W]
        v > img[1, W-1] && v > img[2, W-1] && v > img[2, W] &&
            push!(maxima, CartesianIndex(1, W))
        # bottom-left
        v = img[H, 1]
        v > img[H-1, 1] && v > img[H-1, 2] && v > img[H, 2] &&
            push!(maxima, CartesianIndex(H, 1))
        # bottom-right
        v = img[H, W]
        v > img[H-1, W-1] && v > img[H-1, W] && v > img[H, W-1] &&
            push!(maxima, CartesianIndex(H, W))
    end

    return maxima
end
