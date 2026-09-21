```@meta
CurrentModule = CrowdPhot
```

# Utilities

## Total Error

```@docs
calc_total_error
```

## Sigma Clipping

```@docs
sigma_clip
sigma_clip!
```

## Photometric Calibration

Converting a fitted flux to a calibrated magnitude requires scaling the fitted flux
to a physical unit and applying a zero point. Zeropoint application is generic and so
is included here. Scaling the fitted flux to a physical unit requires instrument-specific
information and is implemented in the observatory modules,
for example [`CrowdPhot.Roman.jansky_per_flux_unit`](@ref) for Roman WFI.

```julia
using CrowdPhot: Roman

f = Roman.jansky_per_flux_unit(l2.meta)
ABmag = abmag.(res.phot.flux .* f)
ABmag_err = abmag_err.(res.phot.flux, res.phot.flux_err)
```

Note that `abmag_err` takes the *uncalibrated* flux and its error. The magnitude
error depends only on their ratio, so it is unaffected by the zero point.

```@docs
abmag
abmag_err
```
