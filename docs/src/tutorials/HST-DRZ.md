# HST DRX

Here we will run photometry on an HST/ACS DRC file which 
is a high-level data product that has been calibrated,
geometrically-corrected, and dither-combined by AstroDrizzle.

```@example hst-drz
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
hdr, img, weights = FITS(fits_path) do fits
    read_header(fits[1]), read(fits[2]), read(fits[3])
end

# Plot image
fig = Figure(size=(700,700),)
ax = Axis(fig[1,1]; aspect = DataAspect(),
    xlabel = "x", ylabel = "y", 
    title="Leo A, HST F814W, Proposal ID 12273, PI: R. van der Marel",
    titlesize = 20)

colorrange = zscale(img) # Use PlotUtils.zscale for reasonable color limits
hm = image!(ax, img'; colorrange, colorscale=LuptonAsinhScale())
Colorbar(fig[1,2], hm; height = Relative(0.75), valign = :center)
fig
```

# Background Estimation
DRZ files have already been background-subtracted, but we will will still
run background estimation to get the noise (RMS) map.

```@example hst-drz
# `img` is NaN in areas not covered, allowing us to make a coverage mask
coverage_mask = isnan.(img)
# Estimate 2-D background with 256 pixel mesh size
bkg = Background2D(img, 256; coverage_mask, fill_value=NaN)
img_sub = img .- bkg.background

fig = Figure(size = (700, 300))
ax1 = Axis(fig[1, 1]; title = "Image", aspect = DataAspect())
ax2 = Axis(fig[1, 2]; title = "Background model", aspect = DataAspect())
ax3 = Axis(fig[1, 3]; title = "Residual", aspect = DataAspect())

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