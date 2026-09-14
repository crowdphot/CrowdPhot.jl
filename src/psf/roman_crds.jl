"""
    roman_crds_gridded_epsf(path; defocus=0, spectral_type="G2V", psf_subtype="psf",
                            origin=nothing, normalize=false,
                            pixel_integration=:exact) -> GriddedPSFModel

Read a Roman CRDS ePSF reference file (ASDF) into a `GriddedPSFModel` of
`ImagePSF` nodes. Requires `ASDF.jl` to be loaded (`using ASDF`); this
function is implemented in a package extension (`CrowdPhotASDFExt`).

The oversampling factor is always read from the file's own
`meta.oversample`; it is a fact about how the file's PSF stamps were
tabulated, not a user choice, so it is not a keyword argument here.

`defocus` selects the defocus-waves slice (`0` = in-focus, matching
`romanisim`'s default). `spectral_type` selects the spectral-type slice by
name (default `"G2V"`, matching `romanisim`'s default index). `psf_subtype`
selects between `"psf"` (default, includes interpixel-capacitance) and
`"psf_noipc"`; `"extended_psf"`/`"extended_psf_noipc"` (single non-gridded
stamps) are not supported here.

If the reference file's PSF stamps are the optical-PSF-only convention
(not yet convolved with the detector pixel response, i.e. summing to
approximately 1 rather than approximately `oversample^2`), this function
convolves each node with the detector pixel response before use, following
the same `is_old_format` heuristic as `romancal`'s `get_gridded_psf_model`.
See the package documentation for the full normalization discussion.

`pixel_integration` selects how that convolution is discretized, and is
forwarded to [`pixel_response_kernel`](@ref):

- `:exact` (default) integrates over the pixel exactly, matching the
  continuous-pixel convolution GalSim applies when `romanisim` renders a
  source from the same reference file.
- `:box` reproduces `romancal`, which uses
  `astropy.convolution.Box2DKernel`. That kernel convolves in an extra box
  of one oversampled sample, broadening the model; for `oversample = 4` it
  depresses the modeled peak of a bright star by roughly 4%, which shows up
  as a bright core and a dark ring in the residuals of a `romanisim` image.
  Use it only to reproduce `romancal` results.

!!! note
    This keyword has no effect on "new format" files, whose stamps are
    already pixel-integrated and are used unchanged.
"""
function roman_crds_gridded_epsf end
