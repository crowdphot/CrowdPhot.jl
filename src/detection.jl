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
  if uniform weights were assumed; not copied).
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
  Cartesian indices.
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
- `normalize_zerosum` controls whether the kernel is renormalized to have
  zero sum, which cancels any uniform background offset in the correlation.
  The default (`true`) is the safe choice and should be used whenever the
  image may contain an un-subtracted background.  Set to `false` when the
  image has been reliably background-subtracted: the zero-sum/non-zero-sum
  SNR ratio is approximately ``\\sqrt{1 - 4/(R/\\sigma)^2}``, where ``R``
  is the kernel truncation radius in units of the PSF width ``\\sigma``.
  Thus the zero-sum penalty is ~13% at ``R=4\\sigma`` and ~25% at
  ``R=3\\sigma``.
- `sigma` is the detection threshold in units of the significance map.
- `border` controls edge handling: `:replicate` (default) or `:zero`.

# Returns

A [`MatchedFilterResult`](@ref) containing the significance map, peak
positions, and flux estimates.

# Mathematical background

By default (`normalize_zerosum = true`) the kernel is normalized to zero
sum, ``\\sum K_i = 0``, so that any uniform background offset cancels
automatically in the correlation.  This is valid regardless of whether
the image has been background-subtracted.  On a subtracted image the expected
flux response for an isolated source matched by ``P`` is unchanged, but the
formal estimator differs and the zero-sum constraint carries an additional
variance penalty.

When `normalize_zerosum = false` the kernel is instead normalized by
``\\sum P^2``, which yields marginally lower noise variance but requires
the image to be reliably background-subtracted.

The kernel is

```math
K_i = \\frac{P_i - \\bar{P}}{\\mathrm{denom}}, \\qquad
\\bar{P} = \\frac{1}{N}\\sum_j P_j, \\qquad
\\mathrm{denom} = \\sum_j P_j^2 - \\frac{(\\sum_j P_j)^2}{N}.
```

The detection significance (SNR) at each pixel is

```math
\\mathrm{SNR}(x,y) = \\frac{\\sum_{i,j} K_{i,j} \\, D_{x+i,\\,y+j}}
                          {\\sigma \\; / \\sqrt{\\mathrm{denom}}}
```

and the matched-filter flux estimate at a peak is ``\\hat{F} = \\sum K \\cdot D``
evaluated at the peak position.  With `normalize_zerosum = true`, this is the
profiled-background flux estimator: it fits the source amplitude after
removing the best constant offset over the kernel footprint.  It is unbiased
for an isolated source matching ``P``, but it is not algebraically identical
to the known-background estimator ``\\sum P(D-B)/\\sum P^2``.

For **spatially varying noise** described by an inverse-variance map
``w_i = 1/\\sigma_i^2``, two correlations are required:

```math
\\mathrm{num}(x,y) = \\sum K_{i,j} \\, w_{x+i,\\,y+j} \\, D_{x+i,\\,y+j}
```

```math
\\mathrm{den}(x,y) = \\sum K_{i,j}^2 \\, w_{x+i,\\,y+j}
```

```math
\\mathrm{SNR} = \\frac{\\mathrm{num}}{\\sqrt{\\mathrm{den}}}
```

where ``w = \\mathrm{inv\\_var}``.  When `inv_var = nothing`, uniform
weights ``w = 1`` are assumed and the significance map is in units of the
unknown pixel-level noise σ.  Divide the significance map by your noise
estimate, or provide `inv_var`, to get true SNR.

!!! note "Zero-sum kernel with spatially-varying weights"
    When `normalize_zerosum = true` and `inv_var` varies significantly
    across the kernel footprint, ``\\sum K_i = 0`` alone does **not**
    guarantee ``\\sum K_i w_{x+i,y+j} = 0``, so a constant background
    can leak into the weighted response.  In this case you should
    background-subtract the image before calling [`matched_filter`](@ref).
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
    end

    # ----- 1. Normalise the kernel -----
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
        weighted_img = inv_var .* image
        num = correlate(weighted_img, K, border)

        K_sq = K .^ 2
        # weighted_img is dead after num; reuse its buffer for den.
        den = weighted_img
        correlate!(den, inv_var, K_sq, border)

        # Significance: num ./ sqrt(den).
        # num's eltype is already wide enough (promoted with Float64 kernel),
        # so reuse its buffer for the significance map.
        significance = num
        @inbounds for i in eachindex(den)
            significance[i] = den[i] > 0 ? significance[i] / sqrt(den[i]) : zero(eltype(significance))
        end
        smoothed_inv_var = den
    else
        # Uniform weights: significance = smoothed * kernel_norm.
        # (kernel_norm converts the raw correlation to SNR per unit σ.)
        S = promote_type(T, typeof(kernel_norm))
        significance = similar(image, S)
        significance .= S.(smoothed) .* kernel_norm
        smoothed_inv_var = nothing
    end

    # ----- 3. Find local maxima -----
    peaks = findlocalmaxima(significance; edges=false)
    threshold_peaks(peaks, significance, sigma) = [p for p in peaks if significance[p] >= sigma]
    sig_peaks = threshold_peaks(peaks, significance, sigma)

    # ----- 4. Extract peak information -----
    peak_sigs = T[significance[p] for p in sig_peaks]
    # Flux estimate: for both kernel normalisations, smoothed[p] gives the
    # matched-filter flux estimate directly (kernel scaled so E[corr] = F).
    peak_fluxes = T[smoothed[p] for p in sig_peaks]

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
    matched_filter(image::AbstractMatrix, fwhm::Int; kws...)

Convenience method for matched-filter detection using a
[`CircularGaussianPRF`](@ref) with the specified `fwhm`.
"""
matched_filter(image::AbstractMatrix, fwhm::Int; kws...) = 
    matched_filter(image, CircularGaussianPRF(; fwhm, x=0, y=0, bkg=0, flux=1); kws...)

"""
    matched_filter(image::AbstractMatrix, fwhm::Tuple{Int, Int}; kws...)

Convenience method for matched-filter detection using a
[`GaussianPRF`](@ref) with x- and y-FWHM specified by the `fwhm` tuple.
"""
matched_filter(image::AbstractMatrix, fwhm::Tuple{Int, Int}; kws...) = 
    matched_filter(image, GaussianPRF(; x=0, y=0, bkg=0, flux=1, x_fwhm=fwhm[1],
        y_fwhm=fwhm[2], theta=0); kws...)

# ---------------------------------------------------------------------------
# Local maximum finding
# ---------------------------------------------------------------------------

"""
    findlocalmaxima(img::AbstractMatrix; edges::Bool=true) -> Vector{CartesianIndex}

Return the coordinates of all local maxima in `img`.  A pixel is a local
maximum if its value is strictly greater than all 8 of its immediate
neighbours (3×3 window).  If `edges=false`, pixels on the image boundary
are excluded from consideration.

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
