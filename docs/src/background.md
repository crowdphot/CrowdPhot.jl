# [Background Estimation](@id bkg)

NOTICE: Our background estimation module is adapted from
[BackgroundMeshes.jl](https://juliaastro.org/BackgroundMeshes/stable/).

Accurate sky-background subtraction is a prerequisite for reliable photometry.
In a typical optical or near-infrared image, the sky background arises from a
combination of airglow, zodiacal light, scattered moonlight, and detector bias;
it is spatially smooth on scales of tens to hundreds of pixels but can vary
across an image at the few-percent level.  Point sources, extended objects, and
cosmic-ray hits are local enhancements that must be distinguished from — and
excluded from — the background estimate.

`CrowdPhot.Background` provides two complementary interfaces:

| Interface | When to use |
|-----------|-------------|
| [`estimate_background`](@ref) | Single scalar estimate for the whole image (or a cutout) |
| [`Background2D`](@ref) | Spatially varying map over a mesh grid |

Both interfaces accept a mask that marks pixels to skip (saturated
stars, bad pixels, diffraction spikes, etc.) and support iterative sigma
clipping to suppress residual source contamination. *The mask is `true`
for pixels that should be skipped.*

## Location and RMS estimators

All estimation algorithms are encapsulated in small callable structs.  Choosing
an estimator amounts to trading off noise robustness, sensitivity to source
contamination, and computational cost.

### Location estimators

| Estimator | Description |
|-----------|-------------|
| [`MeanBackground`](@ref) | Arithmetic mean — efficient but sensitive to bright sources |
| [`MedianBackground`](@ref) | Sample median — resistant to outliers up to ~50 % contamination |
| [`SExtractorBackground`](@ref) | Mode approximation `2.5 × median − 1.5 × mean`; falls back to the median when the distribution is strongly skewed |
| [`MMMBackground`](@ref) | DAOPHOT-style `3 × median − 2 × mean`; assumes positive skew from sources |
| [`BiweightLocationBackground`](@ref) | Tukey biweight location; highly robust but requires a good starting median |

**Recommendation:** `SExtractorBackground` (the default) is a good general
choice.  Switch to `BiweightLocationBackground` when your images contain a high
source density and you use a mask to cover the brightest stars, or use
`MedianBackground` for maximum simplicity.

### RMS (scatter) estimators

| Estimator | Description |
|-----------|-------------|
| [`StdRMS`](@ref) | Population standard deviation (default) |
| [`MADStdRMS`](@ref) | Normalised median absolute deviation `1.4826 × MAD`; robust to ~50 % outliers |
| [`BiweightScaleRMS`](@ref) | Biweight midvariance; most robust, slightly higher variance than MAD |

**Recommendation:** `StdRMS` works well when sigma clipping has already removed
source contamination.  Use `MADStdRMS` or `BiweightScaleRMS` for a more
independent robustness guarantee.

The RMS is computed **around the estimated background location**, not around
an independently recomputed mean.  For example, when paired with
`SExtractorBackground`, the scatter is measured around the mode-like estimate
`2.5 × median − 1.5 × mean` rather than the sample mean, so source
contamination does not inflate the RMS through the centering value.

### Sigma clipping

Both [`estimate_background`](@ref) and [`Background2D`](@ref) support iterative
sigma clipping to suppress residual source contamination before the background
estimator runs.  The clipping algorithm is:

1. Compute the median $m$ and standard deviation $s$ of the pixel values in the
   region (or box, for [`Background2D`](@ref)).
2. Reject pixels with values outside $[m - \sigma s, \; m + \sigma s]$.
3. Recompute $m$ and $s$ from the surviving pixels.
4. Repeat steps 2–3 until no more pixels are rejected, or until `maxiters`
   iterations (default 10) have been performed.

The clipping uses **median and standard deviation** as its internal location
and scale measures — it does **not** invoke the configured background estimator
(e.g., `SExtractorBackground`) during the clipping loop.  Only after clipping
converges is the background estimator applied once to the surviving pixels.

Non-finite (`NaN`, `Inf`) and masked pixels are excluded before clipping
begins.  Pass a scalar `sigma` for symmetric clipping (the default); for
asymmetric clipping, pass a length-2 tuple or vector like `sigma = (5.0, 2.0)`;
in this case, a pixel with value `x` will be clipped if `x ≤ median - 5 * std`
or `x ≥ median + 2 * std`, so the high-value tail of the pixel distribution
is being clipped more strongly than the low-value side. Set `sigma = nothing`
to disable clipping entirely.

## Scalar background estimation

For small images or cutouts where a single background level is sufficient, use
[`estimate_background`](@ref):

```julia
using CrowdPhot.Background

r = estimate_background(image)
println(r.bkg)      # background level
println(r.bkg_rms)  # background scatter
```

The default pipeline is: exclude non-finite pixels → apply 3σ iterative sigma
clipping → estimate with `SExtractorBackground` and `StdRMS`.

### Example: scalar estimation with a mask

```@example scalar
using CrowdPhot, CrowdPhot.Background, StableRNGs, Statistics, CairoMakie

rng = StableRNG(1)
# 128×128 image: flat sky at 200 ADU with 30 Gaussian stars
img = make_gaussians_image(200, (128, 128); rng, background = 200.0,
    border = 0, read_noise = 5.0, gain = 1.5)

# Plot image
fig = Figure(size=(500,500),)
ax = Axis(fig[1,1], aspect = DataAspect(), yreversed = true, xlabel = "x", ylabel = "y")

hm = heatmap!(ax, img)
Colorbar(fig[1,2], hm; height = Relative(0.75), valign = :center)
fig
```

```@example scalar
# Estimate without any mask
r_unmasked = estimate_background(img)
println("Unmasked:  bkg = ", round(r_unmasked.bkg; digits=2),
        "  rms = ", round(r_unmasked.bkg_rms; digits=2))

# Build a source mask by thresholding at bkg + 3×rms
src_mask = img .> r_unmasked.bkg + 3 * r_unmasked.bkg_rms

# Re-estimate with the mask
r_masked = estimate_background(img; mask = src_mask)
println("Masked:    bkg = ", round(r_masked.bkg; digits=2),
        "  rms = ", round(r_masked.bkg_rms; digits=2))
println("True bkg:  200.00")
```

## Spatially varying background with Background2D

When the background varies across the field — as is typical for wide-field
cameras or when a large extended source is present — use [`Background2D`](@ref).
The algorithm proceeds in four stages:

1. **Tile** the image into rectangular boxes of size `box_size`.
2. **Estimate** the background (and RMS) in each box after masking and sigma
   clipping.  Boxes with fewer than `exclude_percentile`% valid pixels are
   excluded.
3. **Filter** the resulting low-resolution mesh with a 2-D median filter of
   width `filter_size` to reduce sensitivity to individual contaminated boxes.
4. **Upsample** the smoothed mesh back to the full image resolution using a
   bicubic Catmull-Rom interpolator.

```julia
using CrowdPhot.Background

b = Background2D(image, 64)
# Access the full-resolution maps:
b.background      # background model
b.background_rms  # RMS model
# or the low-resolution mesh:
b.mesh_background
```

### Choosing `box_size`

The box size determines the spatial resolution of the background model.
It should be:

- **Large enough** that each box contains enough sky pixels to estimate
  the background reliably (typically ≥ 50–100 pixels per box after masking).
- **Small enough** to track real spatial variation in the background.

Somewhere in the range of `image_width / 10 ≤ box_size ≤ image_width / 5` is often reasonable; you
should inspect `b.mesh_background` to verify that the mesh looks smooth and
sensible.

### Example: spatially varying background

The example below simulates an image with a gradient background (brighter at
the bottom) and embedded point sources, then estimates and subtracts it.

```@example bkg2d
using CrowdPhot, CrowdPhot.Background, StableRNGs, CairoMakie

rng = StableRNG(42)
H, W = 256, 256

# --- Build a gradient background image ---
# True background: linear ramp from 150 (top) to 300 (bottom) ADU.
true_bkg = [150.0 + 150.0 * (i - 1) / (H - 1) for i in 1:H, _ in 1:W]

# Render stars on the gradient background with realistic noise.
img = make_gaussians_image(100, (H, W); rng,
    background = true_bkg, read_noise = 5.0, gain = 1.5)

# --- Estimate the spatially varying background ---
b = Background2D(img, 16; sigma = 3.0, filter_size = (5, 5))
residual = img .- b.background

# --- Make plot ---
fig = Figure(size = (900, 400))

ax1 = Axis(fig[1, 1]; title = "Image", aspect = DataAspect())
ax2 = Axis(fig[1, 2]; title = "Background model", aspect = DataAspect())
ax3 = Axis(fig[1, 3]; title = "Residual", aspect = DataAspect())

vmin, vmax = 140.0, 320.0
heatmap!(ax1, img'; colorrange = (vmin, vmax))
heatmap!(ax2, b.background'; colorrange = (vmin, vmax))
hm = heatmap!(ax3, residual'; colorrange = (-30.0, 30.0), colormap = :RdBu)
Colorbar(fig[2, 3], hm; vertical = false, label = "ADU")

for ax in (ax1, ax2, ax3)
    hidedecorations!(ax)
end

fig
```

### Example: comparing estimators on the same image

First, with a small 16 pixel box size

```@example bkg2d
rng2 = StableRNG(7)
img2 = make_gaussians_image(50, (128, 128); rng = rng2, background = 200.0,
    border = 0, read_noise = 5.0, gain = 1.5)
colorrange = (170.0, 230.0)

fig2 = Figure(size = (750, 520))

ax_img = Axis(fig2[1, 1]; title = "Image", aspect = DataAspect())
heatmap!(ax_img, img2'; colorrange, colormap = :grays)
hidedecorations!(ax_img)

estimators = (MeanBackground(), MedianBackground(), SExtractorBackground(), 
    MMMBackground(), BiweightLocationBackground())
labels = ("Mean", "Median", "SExtractor", "MMM", "Biweight")

for i in eachindex(estimators)
    b = Background2D(img2, 16; estimator = estimators[i], sigma = 3.0, filter_size = 1)
    row = i ÷ 3 + 1
    col = i % 3 + 1
    ax = Axis(fig2[row, col]; title = labels[i], aspect = DataAspect())
    heatmap!(ax, b.background'; colormap = :grays, colorrange)
    hidedecorations!(ax)
end

fig2
```

Repeating the measurement with a larger 21 pixel box size, which is less affected by contamination from chance arrangments of stars, but will fail to capture any real variations in sky below this scale.

```@example bkg2d
fig3 = Figure(size = (750, 520))

ax_img = Axis(fig3[1, 1]; title = "Image", aspect = DataAspect())
heatmap!(ax_img, img2'; colorrange, colormap = :grays)
hidedecorations!(ax_img)

estimators = (MeanBackground(), MedianBackground(), SExtractorBackground(), 
    MMMBackground(), BiweightLocationBackground())
labels = ("Mean", "Median", "SExtractor", "MMM", "Biweight")

for i in eachindex(estimators)
    b = Background2D(img2, 21; estimator = estimators[i], sigma = 3.0, filter_size = 1)
    row = i ÷ 3 + 1
    col = i % 3 + 1
    ax = Axis(fig3[row, col]; title = labels[i], aspect = DataAspect())
    heatmap!(ax, b.background'; colormap = :grays, colorrange)
    hidedecorations!(ax)
end

fig3
```

### Example: rotated and padded image

CCD images that have been rotated (e.g., to align with a world-coordinate
grid) and placed inside a larger frame padded with `NaN` are common in
multi-extension FITS files and mosaic pipelines.  `Background2D` handles
such images automatically through the `mask_coverage` keyword, which
should be a matrix the same size as the input image where `true` values
indicate a pixel is **out of coverage** and `false` indicate it is
**in coverage**.

```@example rotated
using CrowdPhot, CrowdPhot.Background, StableRNGs, CairoMakie

rng = StableRNG(13)
# 128×128 image with a gradient background (brighter toward the bottom).
true_bkg = [180.0 + 120.0 * (i - 1) / 127 for i in 1:128, _ in 1:128]
img = make_gaussians_image(40, (128, 128); rng,
    background = true_bkg, read_noise = 5.0, gain = 1.5)

# -- Rotate by 25° and embed in a larger NaN-padded frame --
θ = deg2rad(25)
cosθ, sinθ = cos(θ), sin(θ)
H, W = size(img)
cx, cy = (W + 1) / 2, (H + 1) / 2

# Compute the bounding box of the rotated image.
corners = [(1, 1), (1, W), (H, 1), (H, W)]
xs = Float64[]
ys = Float64[]
for (i, j) in corners
    push!(xs, cosθ * (j - cx) - sinθ * (i - cy) + cx)
    push!(ys, sinθ * (j - cx) + cosθ * (i - cy) + cy)
end
xmin, xmax = floor(Int, minimum(xs)), ceil(Int, maximum(xs))
ymin, ymax = floor(Int, minimum(ys)), ceil(Int, maximum(ys))

H_new = ymax - ymin + 1
W_new = xmax - xmin + 1
rotated = fill(0.0, H_new, W_new)

# Fill valid pixels with bilinear interpolation from the original image.
for j_out in 1:W_new, i_out in 1:H_new
    x = j_out + xmin - 1
    y = i_out + ymin - 1
    # Inverse rotation: where in the original image does (x, y) come from?
    j_orig = cosθ * (x - cx) + sinθ * (y - cy) + cx
    i_orig = -sinθ * (x - cx) + cosθ * (y - cy) + cy
    i0, j0 = floor(Int, i_orig), floor(Int, j_orig)
    if i0 >= 1 && i0 < H && j0 >= 1 && j0 < W
        di, dj = i_orig - i0, j_orig - j0
        rotated[i_out, j_out] =
            (1 - di) * (1 - dj) * img[i0, j0] +
            (1 - di) * dj       * img[i0, j0 + 1] +
            di       * (1 - dj) * img[i0 + 1, j0] +
            di       * dj       * img[i0 + 1, j0 + 1]
    end
end

# -- Mark pixels with values 0 as outside data coverage --
coverage_mask = iszero.(rotated)

# -- Estimate the spatially varying background --
b = Background2D(rotated, 16; coverage_mask, sigma = 3.0, filter_size = (5, 5))

# -- Two-panel plot --
fig = Figure(size = (800, 400))

ax1 = Axis(fig[1, 1]; title = "Rotated image", aspect = DataAspect())
ax2 = Axis(fig[1, 2]; title = "Background model", aspect = DataAspect())

vmin, vmax = 160.0, 320.0
heatmap!(ax1, rotated'; colorrange = (vmin, vmax))
heatmap!(ax2, b.background'; colorrange = (vmin, vmax))

for ax in (ax1, ax2)
    hidedecorations!(ax)
end

fig
```

And now we can look at the background-subtracted image

```@example rotated
fig = Figure(size = (800, 400))
ax1 = Axis(fig[1, 1]; title = "Residual", aspect = DataAspect())
heatmap!(ax1, (rotated .- b.background)'; colorrange=(-40.0, 40.0))
hidedecorations!(ax1)
fig
```

### Tips

- **Source masks** significantly improve accuracy at high source densities.
  Build an initial mask by thresholding at a few sigma above a first-pass
  background estimate, then re-run `Background2D`.

- **`filter_size`** should be odd (default `(3, 3)`).  Larger values (e.g.,
  `(5, 5)`) smooth over mesh cells that are corrupted by bright isolated stars;
  but sizes larger than the spatial scale of genuine background variation
  will bias the estimate.

- **`exclude_percentile`** (default 10%) controls how many pixels per box must
  survive masking and sigma clipping for the box to be used.  Lower values
  allow poorly constrained boxes into the mesh; higher values force more
  interpolation.

- **Edge handling**: when the image dimensions are not exact multiples of
  `box_size`, the mesh extends to the next multiple via ceil division.
  Out-of-bounds pixels in partial edge boxes are treated as `NaN` and
  excluded automatically.  The output always has exactly the same size as
  the input, with no internal allocation of a padded image.

## API reference

```@docs
CrowdPhot.Background.estimate_background
CrowdPhot.Background.Background2D
CrowdPhot.Background.AbstractBackgroundEstimator
CrowdPhot.Background.AbstractBackgroundRMSEstimator
CrowdPhot.Background.MeanBackground
CrowdPhot.Background.MedianBackground
CrowdPhot.Background.SExtractorBackground
CrowdPhot.Background.MMMBackground
CrowdPhot.Background.BiweightLocationBackground
CrowdPhot.Background.StdRMS
CrowdPhot.Background.MADStdRMS
CrowdPhot.Background.BiweightScaleRMS
```
