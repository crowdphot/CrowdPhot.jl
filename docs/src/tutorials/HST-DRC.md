# HST DRC

Here we will run photometry on an HST/ACS DRC file which 
is a high-level data product that has been calibrated,
geometrically-corrected, and dither-combined by AstroDrizzle.

```@example hst-drc
using CrowdPhot
using FITSIO
using LazyArtifacts
using CairoMakie
using Makie: LuptonAsinhScale
using PlotUtils: zscale
using Pkg

# Load image from artifact system
artifacts_toml = joinpath(pkgdir(CrowdPhot), "docs", "Artifacts.toml")
artifact_dir = Pkg.Artifacts.ensure_artifact_installed("jbjl03010_drc", artifacts_toml)
fits_path = joinpath(artifact_dir, "jbjl03010_drc.fits")

# Load relevant parts from FITS file
hdr, sci_hdr, img, weights = FITS(fits_path) do fits
    read_header(fits[1]), read_header(fits[2]), read(fits[2]), read(fits[3])
end

# Plot image
fig = Figure(size=(700,700),)
ax = Axis(fig[1,1]; aspect = DataAspect(),
    xlabel = "x", ylabel = "y", 
    title="Leo A, HST F814W, Proposal ID 12273, PI: R. van der Marel",
    titlesize = 20, yreversed = true)

colorrange = zscale(img) # Use PlotUtils.zscale for reasonable color limits
hm = image!(ax, img'; colorrange, colorscale=LuptonAsinhScale())
Colorbar(fig[1,2], hm; height = Relative(0.75), valign = :center)
fig
```

# Background Estimation
DRZ files have already been background-subtracted, but we will will still
run background estimation to get the noise (RMS) map.

```@example hst-drc
# `img` is NaN in areas not covered, allowing us to make a coverage mask
coverage_mask = isnan.(img)
# Estimate 2-D background with 256 pixel mesh size
bkg = Background2D(img, 256; coverage_mask, fill_value=NaN)
img_sub = img .- bkg.background

fig = Figure(size = (700, 300))
ax1 = Axis(fig[1, 1]; title = "Image", aspect = DataAspect(), yreversed = true)
ax2 = Axis(fig[1, 2]; title = "Background model", aspect = DataAspect(), yreversed = true)
ax3 = Axis(fig[1, 3]; title = "Residual", aspect = DataAspect(), yreversed = true)

im1 = image!(ax1, img'; colorrange)
im2 = image!(ax2, bkg.background'; colorrange)
im3 = image!(ax3, img_sub'; colorrange)
# Colorbar(fig[2, 3], hm; vertical = false)
for (i, im) in enumerate((im1, im2, im3))
    Colorbar(fig[2, i], im; vertical = false)
end
for ax in (ax1, ax2, ax3)
    hidedecorations!(ax)
end

fig
```

## Source Detection

We detect point sources with [`matched_filter`](@ref) using a circular
Gaussian kernel with FWHM = 2.0 pixels, the approximate width of the
ACS/WFC PSF at F814W.  For detection we use inverse-variance weights
derived from the background RMS only; including source Poisson noise
would make bright sources harder to detect by inflating their own noise
estimate, which is incorrect for the detection null hypothesis.

```@example hst-drc
# Convert to Float64 for consistency with the PSF kernel
img_sub_f64 = Float64.(img_sub)

# Inverse variance from background-only error; clip NaN regions to zero weight.
inv_var_bkg = fill(0.0, size(img_sub_f64))
valid = @. !isnan(bkg.background_rms) & (bkg.background_rms > 0)
@. inv_var_bkg[valid] = 1 / bkg.background_rms[valid]^2

# PSF FWHM for HST/ACS F814W (~2 pix)
psf_fwhm = 2.0
mf = matched_filter(img_sub_f64, psf_fwhm; inv_var = inv_var_bkg, sigma = 5.0)

println("Detected $(length(mf.peaks)) sources at ≥ 5σ")
```

And now we can plot our detections on a small part of the image:

```@example hst-drc
region_width = 500
ny, nx = size(img_sub)
y_start = 1500
x_start = 1500
y_range = y_start:min(ny, y_start + region_width - 1)
x_range = x_start:min(nx, x_start + region_width - 1)

region_img = img_sub[y_range, x_range]
region_colorrange = zscale(region_img[isfinite.(region_img)])

# CrowdPhot coordinates are image[y, x]; Makie overlays take (x, y).
region_peaks = filter(mf.peaks) do peak
    y, x = Tuple(peak)
    y in y_range && x in x_range
end

# Draw 4-pixel-diameter source circles in image-coordinate units.
source_circles = map(region_peaks) do peak
    y, x = Tuple(peak)
    Circle(Point2f(x, y), 2)
end

fig = Figure(size = (700, 700))
ax = Axis(fig[1, 1];
    aspect = DataAspect(), yreversed = true,
    xlabel = "x", ylabel = "y",
    title = "Detected sources in a 500×500 pixel region")

hm = heatmap!(ax, x_range, y_range, region_img';
    colorrange = region_colorrange, colorscale = LuptonAsinhScale(),
    colormap = :grays, interpolate = false)
poly!(ax, source_circles; color = (:limegreen, 0), strokecolor = :limegreen,
    strokewidth = 1.2)
Colorbar(fig[1, 2], hm; height = Relative(0.75), valign = :center)

fig
```

## Morphological Measurements

Shape statistics want **background-only** inverse-variance weights. `mf`
already carries them (`inv_var_bkg`, above), so pass it straight to
[`measure_star_shapes`](@ref) and let the default apply. Do **not** substitute a
map that includes source Poisson noise, that down-weights the bright
core and biases statistics like the FWHM.

[`measure_star_shapes`](@ref) extracts a cutout around each peak, computes
sub-pixel centroids via [`centroid_poly`](@ref), and measures aperture-based
FWHM, ellipticity components, moment normalization, and rectangular aperture-sum
diagnostics via [`measure_star_shape`](@ref).

It tapers the aperture moments with a Gaussian window matched to the detection
kernel, which bounds the sky-noise contribution, so `half_width` only needs to
be wide enough to contain the source.

```@example hst-drc
# Measure morphology for all detected sources
results = measure_star_shapes(mf; half_width = 2)
keys(results[1])
```

### Flux Diagnostics

At this point in processing we have two quick flux diagnostics. The `flux`
field holds the best estimate available at this stage, which at detection time
is the matched-filter amplitude of the PSF template at the detected peak; it is
intended to estimate the source flux when the source matches the detection
kernel and the local background is handled by the zero-sum filter.
It is not a full PSF fit, so blends, PSF mismatch, and structured backgrounds
can bias it.

The `aperture.aperture_sum` is the unweighted sum of `image - background`
over the small rectangular cutout used for shape measurement. It is useful
because it is simple and tied to exactly the pixels used for the morphology,
but it is not a robust circular aperture measurement. We have not performed
any aperture correction here; it is calculated below after we construct
a PSF model.

```@example hst-drc
# Compute instrumental magnitudes from the rectangular count-rate aperture sum,
# excluding stars with negative aperture sums.
inst_mag = [
    r.aperture.aperture_sum > 0 ?
        -2.5 * log10(r.aperture.aperture_sum) :
        NaN
    for r in results
]

# Compute instrumental magnitudes from the matched-filter flux estimate.
mf_inst_mag = [
    r.flux > 0 ?
        -2.5 * log10(r.flux) :
        NaN
    for r in results
]

# Filter to sources with valid measurements
good = findall(eachindex(results)) do i
    r = results[i]
    r.aperture.fwhm.y > 0 &&
    r.aperture.fwhm.x > 0 &&
    isfinite(r.core.compactness_core) &&
    isfinite(r.core.normalized_curvature) &&
    r.aperture.moment_norm > 0 &&
    isfinite(inst_mag[i]) &&
    isfinite(mf_inst_mag[i])
end

println("$(length(good)) / $(length(results)) sources have valid morphological measurements")

# Compare ST magnitudes from the two quick flux estimates.
stmag_zeropoint = -2.5 * log10(sci_hdr["PHOTFLAM"]) + sci_hdr["PHOTZPT"]
aperture_mags = inst_mag[good] .+ stmag_zeropoint
matched_filter_mags = mf_inst_mag[good] .+ stmag_zeropoint

fig = Figure(size = (500, 430))
ax = Axis(fig[1, 1];
    xlabel = "Aperture-sum ST magnitude",
    ylabel = "Matched-filter ST magnitude",
    title = "Flux Diagnostics")
h = hexbin!(ax, aperture_mags, matched_filter_mags; bins = 80, colorscale = log10)
# lo = minimum((minimum(aperture_mags), minimum(matched_filter_mags)))
# hi = maximum((maximum(aperture_mags), maximum(matched_filter_mags)))
lo, hi = 20, 30
lines!(ax, [lo, hi], [lo, hi]; color = :black, linestyle = :dash)
xlims!(ax, hi, lo)
ylims!(ax, hi, lo)
Colorbar(fig[1, 2], h; label = "Counts")
fig
```

## Morphology Diagnostics

The panels below show how morphological measurements trend with *small-aperture* ST magnitude.
We use the aperture magnitudes here as we used a simple Gaussian for our matched-filter
detection pass, so the flux estimates based on that metric are likely to be biased.
A quick-look estimate from the [encircled energy curves](https://www.stsci.edu/files/live/sites/www/files/home/hst/instrumentation/acs/data-analysis/aperture-corrections/_documents/bohlin2016_wfc_ee-1.txt)
suggests these cutout aperture magnitudes need a correction of roughly
``-0.4`` mag, making them brighter, which we apply below. **This is not a robust
aperture correction, we will discuss how to measure true calibrated magnitudes later.**

```@example hst-drc
using Statistics

# Reuse the aperture-sum ST magnitudes from the flux diagnostics above,
# applying *APPROXIMATE* 0.4 mag aperture correction
mags = aperture_mags .- 0.4

# Unpack data from `results`, which has array-of-structs layout
fwhm_y  = [results[i].aperture.fwhm.y for i in good]
fwhm_x  = [results[i].aperture.fwhm.x for i in good]
fwhm_theta = [results[i].aperture.fwhm.theta for i in good]
e1 = [results[i].aperture.ellipticity1_aperture for i in good]
e2 = [results[i].aperture.ellipticity2_aperture for i in good]
e1_core = [results[i].core.ellipticity1_core for i in good]
e2_core = [results[i].core.ellipticity2_core for i in good]
compactness = [results[i].core.compactness_core for i in good]
normalized_curvature = [results[i].core.normalized_curvature for i in good]
sharpness = [results[i].sharpness for i in good]
sig    = [results[i].significance for i in good]
mf_flux = [results[i].flux for i in good]

fig = Figure(size = (900, 1250))

# Panel 1: FWHM y vs magnitude
ax1 = Axis(fig[1, 1]; xlabel = "Small-aperture ST magnitude",
           ylabel = "FWHM y (pix)", title = "FWHM (y-axis)")
mask = 0.0 .<= fwhm_y .<= 5
scatter!(ax1, mags[mask], fwhm_y[mask]; markersize = 2, color = :black, rasterize = true)
h1 = hexbin!(ax1, mags[mask], fwhm_y[mask]; bins = 80, threshold = 100, colorscale = log10)
Colorbar(fig[1, 2], h1; label = "Counts")

# Panel 2: FWHM x vs magnitude
ax2 = Axis(fig[1, 3]; xlabel = "Small-aperture ST magnitude",
           ylabel = "FWHM x (pix)", title = "FWHM (x-axis)")
mask = 0.0 .<= fwhm_x .<= 5
scatter!(ax2, mags[mask], fwhm_x[mask]; markersize = 2, color = :black, rasterize = true)
h2 = hexbin!(ax2, mags[mask], fwhm_x[mask]; bins = 80, threshold = 100, colorscale = log10)
Colorbar(fig[1, 4], h2; label = "Counts")

# Panel 3: e1 (axis-aligned ellipticity component) vs magnitude
ax3 = Axis(fig[2, 1]; xlabel = "Small-aperture ST magnitude",
           ylabel = "e1", title = "ellipticity1_aperture")
mask = -1 .<= e1 .<= 1 # Restrict range for plotting
scatter!(ax3, mags[mask], e1[mask]; markersize = 2, color = :black, rasterize = true)
h3 = hexbin!(ax3, mags[mask], e1[mask]; bins = 80, threshold = 100, colorscale = log10)
Colorbar(fig[2, 2], h3; label = "Counts")

# Panel 4: e2 (45-degree ellipticity component) vs magnitude
ax4 = Axis(fig[2, 3]; xlabel = "Small-aperture ST magnitude",
           ylabel = "e2", title = "ellipticity2_aperture")
mask = -1 .<= e2 .<= 1 # Restrict range for plotting
scatter!(ax4, mags[mask], e2[mask]; markersize = 2, color = :black, rasterize = true)
h4 = hexbin!(ax4, mags[mask], e2[mask]; bins = 80, threshold = 100, colorscale = log10)
Colorbar(fig[2, 4], h4; label = "Counts")

# Panel 5: normalized curvature vs magnitude
ax5 = Axis(fig[3, 1]; xlabel = "Small-aperture ST magnitude",
           ylabel = "normalized curvature", title = "Normalized Core Curvature (2/FWHM²) / I_0")
ylims!(ax5, -2, 10)
mask = -2 .<= normalized_curvature .<= 10
scatter!(ax5, mags[mask], normalized_curvature[mask]; markersize = 2, color = :black, rasterize = true)
h5 = hexbin!(ax5, mags[mask], normalized_curvature[mask]; bins = 80, colorscale=log10, threshold = 100)
Colorbar(fig[3, 2], h5; label = "Counts")

# Panel 6: compactness vs magnitude
ax6 = Axis(fig[3, 3]; xlabel = "Small-aperture ST magnitude",
           ylabel = "compactness", title = "Compactness 1 / (σ_x² + σ_y²)")
ylims!(ax6, 0, 5)
mask = 0 .<= compactness .<= 5
scatter!(ax6, mags[mask], compactness[mask]; markersize = 2, color = :black, rasterize = true)
h6 = hexbin!(ax6, mags[mask], compactness[mask]; bins = 80, colorscale=log10, threshold = 100)
Colorbar(fig[3, 4], h6; label = "Counts")

# Panel 7: Core vs Aperture e1
ax7 = Axis(fig[4, 1]; xlabel = "e1 aperture",
           ylabel = "e1 core", title = "Core vs Aperture e1",
           limits = ((-1, 1), (-1, 1)))
mask = findall( (-1 .< e1) .& (e1 .< 1) .& (-1 .< e1_core) .& (e1_core .< 1) )
scatter!(ax7, e1[mask], e1_core[mask]; markersize = 2, color = :black, rasterize = true)
h7 = hexbin!(ax7, e1[mask], e1_core[mask]; bins = 80, colorscale=log10, threshold = 100)
Colorbar(fig[4, 2], h7; label = "Counts")

# Panel 8: Core vs Aperture e2
ax8 = Axis(fig[4, 3]; xlabel = "e2 aperture",
           ylabel = "e2 core", title = "Core vs Aperture e2",
           limits = ((-1, 1), (-1, 1)))
mask = findall( (-1 .< e2) .& (e2 .< 1) .& (-1 .< e2_core) .& (e2_core .< 1) )
scatter!(ax8, e2[mask], e2_core[mask]; markersize = 2, color = :black, rasterize = true)
h8 = hexbin!(ax8, e2[mask], e2_core[mask]; bins = 80, colorscale=log10, threshold = 100)
Colorbar(fig[4, 4], h8; label = "Counts")

# Panel 9: median FWHM per magnitude bin
mag_bins = range(minimum(mags), maximum(mags); length = 30)
med_fwhm_y = Float64[]
med_fwhm_x = Float64[]
mag_centers = Float64[]
for (lo, hi) in zip(mag_bins[1:end-1], mag_bins[2:end])
    bin_idx = findall(m -> lo <= m < hi, mags)
    isempty(bin_idx) && continue
    push!(mag_centers, (lo + hi) / 2)
    push!(med_fwhm_y, median(fwhm_y[bin_idx]))
    push!(med_fwhm_x, median(fwhm_x[bin_idx]))
end
ax9 = Axis(fig[5, 1]; xlabel = "Small-aperture ST magnitude",
           ylabel = "Median FWHM (pix)", title = "Median FWHM by magnitude")
scatterlines!(ax9, mag_centers, med_fwhm_y; label = "y", color = :blue)
scatterlines!(ax9, mag_centers, med_fwhm_x; label = "x", color = :red)
axislegend(ax9; position = :rt)

# Panel 10: DAOPHOT SHARP vs magnitude.  Raw peak minus the mean of its
# kernel-footprint neighbors, divided by the source's fitted central height.
# Large for cosmic rays and hot pixels, small for blends and resolved
# sources, tightly clustered for stars.
ax10 = Axis(fig[5, 3]; xlabel = "Small-aperture ST magnitude",
            ylabel = "sharpness", title = "SHARP (DAOPHOT)")
ylims!(ax10, -1, 3)
mask = findall(x -> -1 < x < 3, sharpness)
scatter!(ax10, mags[mask], sharpness[mask]; markersize = 2, color = :black, rasterize = true)
h10 = hexbin!(ax10, mags[mask], sharpness[mask]; bins = 80, colorscale = log10, threshold = 100)
Colorbar(fig[5, 4], h10; label = "Counts")

fig
```

## Pick Stars for PSF Fitting

We select stars suitable for PSF fitting with
`pick_psf_stars`.  The function clips
the instrumental-magnitude distribution to exclude very bright (potentially
saturated) and very faint stars, applies a hard constraint on normalized
core curvature, and then sigma-clips morphological parameters
(`fwhm.y`, `fwhm.x`, `ellipticity1_aperture`, `ellipticity2_aperture`,
`normalized_curvature`) within five instrumental-magnitude bins.

```@example hst-drc
# Select the brightest 50 stars suitable for PSF fitting
show_idx = CrowdPhot.PSF.pick_psf_stars(results, 50)
println("$(length(show_idx)) stars selected for PSF fitting (from $(length(results)) total)")
```

We show image cutouts of the selected stars to visually confirm that the
selection is reasonable. Note that the ePSF fitting routine includes
mechanisms for rejecting provided PSF stars that do not help to improve
the fit, so it is not as important to pre-filter the list of PSF stars
as carefully as for other software.

```@example hst-drc
# Extract cutouts with a 5-pixel half-width (11×11 pixels) for context
half = 5
ny, nx = size(img_sub)
cutouts = map(show_idx) do i
    y, x = Tuple(results[i].pixel)
    yr = max(1, y - half):min(ny, y + half)
    xr = max(1, x - half):min(nx, x + half)
    img_sub[yr, xr]
end

# Compact grid — no axis labels, colorbars, or other decorations
ncols = 10
nrows = cld(length(show_idx), ncols)
fig = Figure(size = (ncols * 70, nrows * 70))

for (k, cutout) in enumerate(cutouts)
    # Per-cutout zscale for robust contrast
    fin = cutout[isfinite.(cutout)]
    zmin, zmax = isempty(fin) ? (0.0, 1.0) : zscale(fin)

    row = (k - 1) ÷ ncols + 1
    col = (k - 1) % ncols + 1
    ax = Axis(fig[row, col]; aspect = DataAspect())
    heatmap!(ax, cutout';
        colorrange = (zmin, zmax), colormap = :grays, interpolate = false)
    hidedecorations!(ax)
end

colgap!(fig.layout, 1)
rowgap!(fig.layout, 1)
fig
```

## Empirical PSF Construction

We now build an empirical PSF from the selected stars using
[`fit_psf`](@ref CrowdPhot.PSF.fit_psf) with the Anderson & King (2000)
iterative residual-stacking method ([Anderson2000](@citet)).
Stars outside the cutout boundary are dropped (`drop_edge=true`), and
the ePSF is supersampled at 4× the detector pixel scale.

```@example hst-drc
# Select bright, morphologically-clean stars
n_psf = 2000
psf_idx = CrowdPhot.PSF.pick_psf_stars(results, n_psf; mag_quantiles=(0.00, 0.95))

# Use sub-pixel centroids from the morphology measurements
psf_y = [results[i].centroid.y for i in psf_idx]
psf_x = [results[i].centroid.x for i in psf_idx]

# Build the empirical PSF
psf_rad = 6
psf, fit_result = CrowdPhot.PSF.fit_psf(
    CrowdPhot.PSF.ImagePSF, img_sub_f64, psf_y, psf_x;
    psf_rad = psf_rad,
    fit_rad = 4,
    oversampling = 4,
    smooth = true,
    recenter = true,
)

n_used = sum(fit_result.used)
println("$(n_used) / $(n_psf) stars used in $(fit_result.iterations) iterations")
```

The returned `ImagePSF` stores the PSF on an oversampled grid.  We convert
the grid axes to detector-pixel offsets from the PSF center for display.
Note that we use a non-linear colorscale to accentuate the PSF features
outside the core. Thick contours show ePSF levels of 0.001, 0.01, and 0.1.

```@example hst-drc
os_y, os_x = psf.oversampling
ny_os, nx_os = size(psf.data)

# Oversampled grid indices relative to the PSF origin, scaled to detector pixels
dy_os = (1:ny_os) .- psf.origin.y
dx_os = (1:nx_os) .- psf.origin.x
y_pix = dy_os ./ os_y
x_pix = dx_os ./ os_x

fig = Figure(size = (700, 600))
ax = Axis(fig[1, 1];
    aspect = DataAspect(),
    xlabel = "Δx (pix)", ylabel = "Δy (pix)",
    title = "Empirical PSF — $(n_used) stars, $(os_y)× oversampled",
    yreversed = true)

zmin, zmax = zscale(psf.data[psf.data .> 0]; contrast=0.05)
psf_hm = heatmap!(ax, x_pix, y_pix, psf.data';
    colorrange = (zmin, zmax), colormap = :grays, interpolate = false, colorscale=LuptonAsinhScale(0.0025, 5.0))
contour!(ax, x_pix, y_pix, psf.data'; 
    levels = logrange(1e-3,1e-1;length=3), color=:red, linewidth = 3)
contour!(ax, x_pix, y_pix, psf.data'; 
    levels = logrange(1e-3,1e-1;length=9), color=:red, linewidth = 1)
Colorbar(fig[1, 2], psf_hm; label = "relative flux", height = Relative(0.8), valign = :center)

fig
```

## Curve of Growth and Aperture Correction

We compute the encircled-energy curve from the empirical PSF using
[`curve_of_growth`](@ref).  The radii are chosen to stay within the
PSF support (``\pm 6`` detector pixels from center).  We also load the
[`reference_cog`](@ref) for ACS/WFC F814W from [Bohlin2016](@citet) to compare
our measured PSF against the standard reference.

```@example hst-drc
# Radii in detector pixels, staying within the PSF cutout extent (psf_rad = 6).
radii = 0.5:0.25:5.5

# Compute and normalize the curve of growth from our empirical PSF.
cog = curve_of_growth(psf, radii)
cog_norm = normalize(cog; method = :sum)

# Load the reference encircled-energy curve for ACS/WFC F814W.
ref = reference_cog(:WFC, :F814W)

# Key encircled-energy radii from our measured PSF.
rhalf = radius_at_energy(cog_norm, 0.5)
r80 = radius_at_energy(cog_norm, 0.80)
rhalf_ref = radius_at_energy(ref, 0.5)
r80_ref = radius_at_energy(ref, 0.80)
println("Half-light radius: measured $(round(rhalf; digits=2)) pix, reference $(round(rhalf_ref; digits=2)) pix")
println("80% EE radius: measured $(round(r80; digits=2)) pix, reference $(round(r80_ref; digits=2)) pix")
```

The reference curve is a time-averaged PSF model.  Comparing our measured
EE to the reference tells us whether the PSF in our drizzle-interpolated image
is broader or sharper than the reference. We multiply our
measured EE by the value of the reference EE in the last radial bin
so their shapes can be more easily compared visually.

```@example hst-drc
# Reference EE at last radius value
max_rad_EE = encircled_energy(reference_cog(:WFC, :F814W), last(radii))

fig = Figure(size = (600, 450))
ax = Axis(fig[1, 1];
    xlabel = "Radius (pixels)", ylabel = "Encircled energy fraction",
    title = "Encircled Energy — ACS/WFC F814W",
    limits = ((0, 6), (0, 1.05)))

lines!(ax, cog_norm.radii, cog_norm.flux .* max_rad_EE;
    linewidth = 2, color = :black, label = "Measured ePSF")
lines!(ax, ref.radii, ref.flux;
    linewidth = 2, color = :red, linestyle = :dash, label = "Bohlin (2016)")

axislegend(ax; position = :rb)

fig
```

After renormalizing to match the reference EE curve in the last radial bin,
our EE curve for our ePSF fit matches the reference well. We can now calculate
the aperture correction to put our measured PSF magnitudes onto the correct
infinite-aperture convention for which the HST zeropoints are defined. We evaluate
the reference EE curve at `psf_rad` here because this is radius to which the flux
of our ePSF model has been normalized.

```@example hst-drc
aper_corr = 2.5 * log10(encircled_energy(reference_cog(:WFC, :F814W), psf_rad))
```

## PSF Fitting Photometry

We prepare to fit our detected sources by preparing the total error
map using [`calc_total_error`](@ref) to add the source Poisson term to the background RMS.
The data are in ``\mathrm{e}^- / \mathrm{s}``
(``\mathtt{BUNIT} = \mathrm{ELECTRONS}/\mathrm{S}``), so the effective gain is
the exposure time, ``g_{\mathrm{eff}} = \mathtt{EXPTIME}``, which converts to
countable units (electrons).

```@example hst-drc
# Extract exposure time from the primary header for the effective gain.
exptime = hdr["EXPTIME"]

# Total 1-sigma error including source Poisson noise, for the PSF fit below.
total_err = calc_total_error.(img_sub_f64, bkg.background_rms, exptime)

# Build inverse-variance map with NaN regions clipped to zero weight.
inv_var = fill(0.0, size(img_sub_f64))
valid_total = @. isfinite(total_err) & (total_err > 0)
@. inv_var[valid_total] = 1 / total_err[valid_total]^2;
```

We now run [`fit_all_stars_multipass`](@ref) on a 250×250 region, using the
empirical PSF we just constructed.  It owns the background and detection as
well as the fit: each pass re-estimates the background from the current
residual, detects anything the model is still missing, and re-fits every source
one at a time against the residual of all the others.  We pass it the *raw*
cutout and the sources we already detected in this region as a warm start.
After the final pass the residual image should be nearly empty -- only noise
should remain where the sources used to be.  We fix each source's local
background pedestal to 0, since the pipeline's own background model already
handles this uncrowded field ($\sim2$ stars per square arcsecond).

```@example hst-drc
# Select a 250×250 sub-region (lower-right corner of the earlier 500×500 region)
y1, y2 = 1750, min(ny, 1999)
x1, x2 = 1750, min(nx, 1999)
phot_y_range = y1:y2
phot_x_range = x1:x2

# Select all sources whose pixel positions fall within this region
region_idx = findall(eachindex(results)) do i
    y, x = Tuple(results[i].pixel)
    y in phot_y_range && x in phot_x_range
end
region_sources = results[region_idx]
println("$(length(region_sources)) sources in the display region")

# Cutout coordinates start at 1, so shift the warm-start catalog to match.
img_cut = Float64.(img[phot_y_range, phot_x_range])
warm = (; y = [s.centroid.y - (y1 - 1) for s in region_sources],
          x = [s.centroid.x - (x1 - 1) for s in region_sources],
          flux = [s.flux for s in region_sources])

# Multi-pass background -> detect -> fit
mp = fit_all_stars_multipass(img_cut, psf, warm, 3;
    inv_var = inv_var[phot_y_range, phot_x_range], fixed = (; bkg = 0.0),
    max_iter = 3, min_iter = 3,
    # blend gating, to prevent bright-star fragmentation
    blend_threshold_initial = 10.0, blend_threshold = 5.0, blend_passes = 1)
phot_result = mp.phot

n_good = length(phot_result.flux)
println("$n_good sources in the final catalog after $(mp.n_detection_passes) detection passes")
```

The [`MultiPassPhotResult`](@ref) stores the final residual image,
`image - background - model`, after all source models have been subtracted.  Below we compare the original
image to the residual for this 250×250 region.  Where sources have been
successfully subtracted, the residual shows only noise.

```@example hst-drc
orig_region = img_sub[phot_y_range, phot_x_range]
resid_region = phot_result.residual
region_colorrange = zscale(orig_region[isfinite.(orig_region)])

# Circle every source in the final catalog, including those found by the
# detection passes.  Catalog positions are cutout-local, so shift them back.
phot_circles = map(zip(phot_result.y, phot_result.x)) do (y, x)
    Circle(Point2f(x + (x1 - 1), y + (y1 - 1)), 2)
end

fig = Figure(size = (500, 900))
ax1 = Axis(fig[1, 1];
    aspect = DataAspect(), yreversed = true,
    title = "Original")
ax2 = Axis(fig[2, 1];
    aspect = DataAspect(), yreversed = true,
    title = "Residual ($n_good stars subtracted)")

hm1 = heatmap!(ax1, phot_x_range, phot_y_range, orig_region';
    colorrange = region_colorrange, colorscale = LuptonAsinhScale(),
    colormap = :grays, interpolate = false)
hm2 = heatmap!(ax2, phot_x_range, phot_y_range, resid_region';
    colorrange = region_colorrange, colorscale = LuptonAsinhScale(),
    colormap = :grays, interpolate = false)

# Overlay detection circles on both panels
poly!(ax1, phot_circles; color = (:limegreen, 0), strokecolor = :limegreen,
    strokewidth = 1.2)
poly!(ax2, phot_circles; color = (:limegreen, 0), strokecolor = :limegreen,
    strokewidth = 1.2)

Colorbar(fig[1, 2], hm1; height = Relative(0.8), valign = :center)
Colorbar(fig[2, 2], hm2; height = Relative(0.8), valign = :center)
rowgap!(fig.layout, 0)

fig
```

We now examine the photometric quality of the PSF-fit results.  Instrumental
magnitudes are computed from the fitted fluxes and placed on the STMAG system
using the PHOTFLAM and PHOTZPT header keywords and the aperture correction
derived above is applied. For descriptions
of the goodness-of-fit statistics, see [`CrowdPhot.MultiPassPhotResult`](@ref).

```@example hst-drc
# Every returned source was fit; pair each with the nearest morphology
# measurement from the region, when there is one within a pixel.
nearest = map(eachindex(phot_result.y)) do j
    d = [hypot(warm.y[i] - phot_result.y[j], warm.x[i] - phot_result.x[j]) for i in eachindex(warm.y)]
    i = argmin(d)
    d[i] < 1 ? i : 0
end
good = nearest .> 0

# Compute ST magnitudes from PSF-fit fluxes
psf_mags = -2.5 .* log10.(phot_result.flux[good]) .+ stmag_zeropoint .+ aper_corr
psf_mag_errs = (2.5 / log(10)) .* phot_result.flux_err[good] ./ phot_result.flux[good]

# Centroids from measure_star_shapes for the same sources
morph_y = warm.y[nearest[good]]
morph_x = warm.x[nearest[good]]
fit_y = phot_result.y[good]
fit_x = phot_result.x[good]
fit_y_err = phot_result.y_err[good]
fit_x_err = phot_result.x_err[good]

# Centroid offset between the two measurement techniques
centroid_offset = @. hypot(morph_y - fit_y, morph_x - fit_x)

# Fitting statistics returned from `fit_all_stars_multipass`
chisq = phot_result.chisq[good]
qfit = phot_result.qfit[good]
qfit_z = phot_result.qfit_z[good]
crowding = phot_result.crowding[good]

fig = Figure(size = (900, 1500))

# Panel 1: magnitude error vs magnitude
ax1 = Axis(fig[1, 1];
    xlabel = "PSF-fit ST magnitude", ylabel = "σ_mag",
    title = "Photometric errors")
scatter!(ax1, psf_mags, psf_mag_errs; markersize = 4, color = :black)

# Panel 2: centroid errors vs magnitude
ax2 = Axis(fig[1, 2];
    xlabel = "PSF-fit ST magnitude", ylabel = "σ (pix)",
    title = "Centroid errors")
scatter!(ax2, psf_mags, fit_y_err; markersize = 4, color = :royalblue, label = "y")
scatter!(ax2, psf_mags, fit_x_err; markersize = 4, color = :crimson, label = "x")
axislegend(ax2; position = :lt)

# Panel 3: centroid offset (morphology vs PSF fit) vs magnitude
ax3 = Axis(fig[2, 1];
    xlabel = "PSF-fit ST magnitude",
    ylabel = "Centroid offset (pix)",
    title = "Centroid offset: quadratic (`measure_star_shapes`) vs PSF fit")
scatter!(ax3, psf_mags, centroid_offset; markersize = 4, color = :black)
ylims!(ax3, 0.0, 0.5)

# Panel 4: Reduced χ² vs magnitude
ax4 = Axis(fig[2,2];
    xlabel = "PSF-fit ST magnitude",
    ylabel = "Reduced χ²",
    title = "Reduced χ²")
scatter!(ax4, psf_mags, chisq; markersize=4, color = :black)
ylims!(ax4, 0.0, 5.0)

# Panel 5: qfit
ax5 = Axis(fig[3,1];
    xlabel = "PSF-fit ST magnitude",
    ylabel = "qfit",
    title = "qfit")
scatter!(ax5, psf_mags, qfit; markersize=4, color = :black)
hlines!(ax5, 0.2; linestyle = :dash, color = :red)

# Panel 6: qfit_z
ax6 = Axis(fig[3,2];
    xlabel = "PSF-fit ST magnitude",
    ylabel = "qfit_z",
    title = "qfit_z")
scatter!(ax6, psf_mags, qfit_z; markersize=4, color = :black)

# Panel 7: Crowding
ax7 = Axis(fig[4,1];
    xlabel = "PSF-fit ST magnitude",
    ylabel = "crowding",
    title = "DOLPHOT crowding")
scatter!(ax7, psf_mags, crowding; markersize=4, color = :black)

fig
```