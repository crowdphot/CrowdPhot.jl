```@meta
CurrentModule = CrowdPhot
```

# PSF Fitting Photometry

[`fit_all_stars`](@ref) performs PSF-fitting photometry on all sources in an
image using a DOLPHOT-style multi-pass algorithm. Stars are sorted by
brightness and fitted sequentially against a progressive residual image:
each fitted model is subtracted before the next star is processed, so
fainter neighbors are measured after brighter stars have been removed. On
subsequent passes each star is added back, re-fitted, and re-subtracted,
refining all measurements iteratively.

This avoids the large matrix solves used by fully simultaneous
group-fitting methods such as DAOPHOT/ALLSTAR. For a group containing `N`
stars, the cost of forming normal matrices can scale like `N^2`,
while a dense least-squares solve can scale like `N^3` in the group size.
In contrast, when the fitting radius, the residual-image approach scales
approximately linearly with the number of stars and passes:
`N_passes * N_stars`. By operating on the progressive residual image,
this method can give good results even in crowded fields.

## Result type

```@docs
MultiPassPhotResult
```

### Goodness-of-fit and morphology diagnostics

Alongside the fitted parameters, [`MultiPassPhotResult`](@ref) carries several
per-star diagnostics computed on the final pass over exactly the fitting box:

- `chisq` (reduced ``\chi^2``, L2) and `qfit` / `qfit_expected` / `qfit_z`
  (absolute-residual, L1) measure how well the PSF model fits the data.
- `crowding` flags blend contamination from neighbors.
- `spread_model` / `spread_model_err` is a star/galaxy separator
  for developed for 
  [SExtractor](https://sextractor.readthedocs.io/en/latest/Model.html).
  It compares the data against the local PSF
  and against the PSF convolved with a small circular exponential disk
  (scalelength ``\mathrm{FWHM}/16``); the value is near zero for point sources,
  positive for extended sources, and negative for detections sharper than the
  PSF. The reference disk FWHM defaults to a Gaussian-equivalent estimate from
  the PSF's effective area and can be overridden with the `spread_model_fwhm`
  keyword. `spread_model` is only meaningful when `psf` is an accurate model of
  the true PSF with its shape parameters held fixed via `fixed`.

## Fitting functions

```@docs
fit_all_stars
CrowdPhot.fit_all_stars_simultaneous
CrowdPhot.fit_all_stars_simultaneous_multipass
```
