```@meta
CurrentModule = CrowdPhot
```

# PSF Fitting Photometry

CrowdPhot's crowded-field photometry is a **multi-pass** pipeline.  Each pass
re-estimates the background from the current residual image, detects sources
the model is still missing, estimates initial centroids and fluxes, re-fits the whole
catalog, and prunes unreliable sources; a final pass re-fits without detecting
or pruning.  Two functions run this pipeline and differ only in the fitting
implementation:

- [`fit_all_stars_multipass`](@ref) fits DOLPHOT-style: within a pass the
  catalog is swept brightest to faintest, and each source is fit alone with a
  few Levenberg-Marquardt iterations against the residual of every other
  source.  Each LM step is cheap (a `3x3` or `4x4` solve) and the cost scales
  linearly with the number of sources and sweeps.  It can also fit a local
  background pedestal per source.
- [`fit_all_stars_simultaneous_multipass`](@ref) fits crowdsource-style: every
  source moves together in one damped Levenberg-Marquardt step over the whole
  catalog.  One step accounts for all the couplings between blended neighbors at once.

Background estimation, detection, seeding, pruning, model radius tuning,
diagnostics and morphology are shared, so the two return the same result
structure and are directly comparable.  Errors are computed per fitter, from
each source's own normal-matrix block in both cases.

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
fit_all_stars_multipass
fit_all_stars_simultaneous_multipass
```
