module CrowdPhotASDFExt
using CrowdPhot, CrowdPhot.PSF, ASDF
using Statistics: median

# See the base-package docstring (`src/psf/roman_crds.jl`) for the full
# description of arguments and behavior.
function CrowdPhot.PSF.roman_crds_gridded_epsf(path::AbstractString;
        defocus = 0, spectral_type::AbstractString = "G2V", psf_subtype::AbstractString = "psf",
        origin = nothing, normalize::Bool = false, pixel_integration::Symbol = :exact)
    psf_subtype in ("psf", "psf_noipc") || throw(ArgumentError(
        "psf_subtype must be \"psf\" or \"psf_noipc\" (got $(repr(psf_subtype))); " *
        "\"extended_psf\"/\"extended_psf_noipc\" are single non-gridded stamps, " *
        "not supported by GriddedPSFModel — build an ImagePSF directly instead."))
    # `validate_checksum = false`: some ASDF files produced by the Python
    # implementation of ASDF save a checksum computed from the original
    # decompressed file, which does not match what ASDF.jl computes (from
    # the compressed/"used" data, per the current ASDF spec); real Roman
    # CRDS reference files hit this, so checksum validation must be
    # disabled to load them.
    # `extensions = true`: Roman reference files use custom (non-standard)
    # YAML tags, e.g.
    # "asdf://stsci.edu/datamodels/roman/tags/reference_files/epsf-1.0.0".
    # Without this, loading throws "could not determine a constructor for
    # the tag ...". With it, ASDF.jl falls back to a generic (untyped)
    # representation for unrecognized tags and only emits a `@warn` (once
    # per distinct tag), which is expected here.
    af = ASDF.load(path; extensions = true, validate_checksum = false)
    haskey(af.metadata, "roman") || throw(ArgumentError("$path does not contain a top-level \"roman\" key; is this a Roman CRDS reference file?"))
    afr = af["roman"]
    haskey(afr, "meta") || throw(ArgumentError("$path does not contain a \"roman.meta\" key"))
    meta = afr["meta"]
    haskey(afr, psf_subtype) || throw(ArgumentError("$path does not contain \"$psf_subtype\"; available: $(collect(keys(afr)))"))
    psf_data = afr[psf_subtype][]  # Julia-order (x, y, grid_index, spectral_type, defocus)
    ndims(psf_data) == 5 || throw(ArgumentError("$path\'s \"$psf_subtype\" array has $(ndims(psf_data)) dimensions, expected 5 (x, y, grid_index, spectral_type, defocus); is this a gridded (not extended) PSF array?"))
    pixel_x, pixel_y = meta["pixel_x"], meta["pixel_y"]
    length(pixel_x) == length(pixel_y) || throw(ArgumentError("pixel_x and pixel_y have different lengths in reference file $path"))
    spectral_types, defocus_values = meta["spectral_type"], meta["defocus"]

    # `oversample` is always the file's own value -- see Section 3.1 of
    # gridded_psf_crds_plan.md for why this is never a user-settable
    # keyword on this function.
    os = Int(meta["oversample"])
    os > 0 || throw(ArgumentError("$path has non-positive meta.oversample=$os"))

    defocus_idx = findfirst(==(defocus), defocus_values)
    isnothing(defocus_idx) && throw(ArgumentError("defocus=$defocus not found in $path; available: $(collect(defocus_values))"))
    spectral_idx = findfirst(==(spectral_type), spectral_types)
    isnothing(spectral_idx) && throw(ArgumentError("spectral_type=$(repr(spectral_type)) not found in $path; available: $(collect(spectral_types))"))

    n_grid = length(pixel_x)
    size(psf_data, 3) == n_grid || throw(ArgumentError("$path\'s \"$psf_subtype\" array has $(size(psf_data, 3)) grid nodes but meta.pixel_x/pixel_y list $n_grid"))
    T = eltype(psf_data)
    stamps = Matrix{T}[permutedims(view(psf_data, :, :, i, spectral_idx, defocus_idx)) for i in 1:n_grid]

    # romancal's own is_old_format heuristic (romancal/source_catalog/psf.py,
    # get_gridded_psf_model): "old format" stamps are the optical PSF only
    # (STPSF output, sum ~ 1), not yet convolved with the detector's pixel
    # response; "new format" stamps already have that convolution baked in
    # (sum ~ oversample^2). Computed once per (defocus, spectral_type)
    # slice, from the median sum across grid nodes -- not per-node. See
    # "Pixel-response convolution" in gridded_psf_crds_plan.md.
    is_old_format = median(sum.(stamps)) < os^2 / 2
    if is_old_format
        kernel = T.(CrowdPhot.PSF.pixel_response_kernel(os; type = pixel_integration))
        stamps = Matrix{T}[T(os^2) .* CrowdPhot.correlate(stamp, kernel, :zero) for stamp in stamps]
    end
    # NOTE: `oversampling^2` scaling only happens inside the `is_old_format`
    # branch above -- a "new format" stamp already sums to ~oversampling^2
    # and must be used unchanged; scaling it again would be silently wrong.

    psfs = [ImagePSF(stamp; oversampling = os, origin, normalize) for stamp in stamps]
    return GriddedPSFModel(psfs, Vector{T}(pixel_y), Vector{T}(pixel_x))
end

end # module
