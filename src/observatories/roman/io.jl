# Every ASDF call goes through `_load_asdf`.  Keeping the
# dependency's surface to one function makes it easy to move back
# behind a package extension if we need to.
#
# `validate_checksum = false`: some ASDF files produced by the Python
# implementation of ASDF save a checksum computed from the original
# decompressed file, which does not match what ASDF.jl computes (from the
# compressed/"used" data, per the current ASDF spec); real Roman files hit
# this, so checksum validation must be disabled to load them.
#
# `extensions = true`: Roman files use custom (non-standard) YAML tags, e.g.
# "asdf://stsci.edu/datamodels/roman/tags/reference_files/epsf-1.0.0".
# Without this, loading throws "could not determine a constructor for the
# tag ...".  With it, ASDF.jl falls back to a generic (untyped)
# representation and only emits a `@warn` (once per distinct tag).  That
# warning is expected for every Roman file, so it is suppressed rather than
# printed on every load.
function _load_asdf(path::AbstractString)
    return Logging.with_logger(Logging.ConsoleLogger(stderr, Logging.Error)) do
        ASDF.load(path; extensions = true, validate_checksum = false)
    end
end

# Fetch the top-level "roman" tree, with a message that says what was expected.
function _roman_tree(af, path::AbstractString)
    haskey(af.metadata, "roman") || throw(ArgumentError(
        "$path does not contain a top-level \"roman\" key; is this a Roman file?"))
    return af["roman"]
end

"""
    load_l2(fname; dq_flags = (:DO_NOT_USE, :NO_LIN_CORR)) -> NamedTuple

Read a Roman WFI Level 2 (calibrated rate) ASDF file.

# Arguments
- `fname::AbstractString`: path to the `.asdf` file.

# Keyword arguments
- `dq_flags`: the [`DQ_FLAGS`](@ref) names that mark a pixel unusable.  See
  [`parse_dq_mask`](@ref).

# Returns
- `data::Matrix{Float32}`: the science array.
- `dq::BitMatrix`: `true` where any of `dq_flags` is set, i.e. `true` means
  "bad", matching the `bkg_mask` convention of
  [`CrowdPhot.fit_all_stars_multipass`](@ref).
- `err::Matrix{Float32}`: the total error array.
- `var_poisson::Matrix{Float32}`: the Poisson variance array.
- `meta`: the file's `roman.meta` metadata tree, passed through as ASDF returns it.

!!! note
    ASDF.jl reads array data in reverse dimension order relative to the
    Python/C layout the file was written in, so every array here is
    `permutedims`ed back on the way out.  Arrays are returned in CrowdPhot's
    `(y, x)` convention.
"""
function load_l2(fname::AbstractString; dq_flags = (:DO_NOT_USE, :NO_LIN_CORR))
    afr = _roman_tree(_load_asdf(fname), fname)
    meta = afr["meta"]
    image_data = Matrix{Float32}(permutedims(afr["data"][]))
    dq = BitMatrix(permutedims(parse_dq_mask(afr["dq"][]; flags = dq_flags)))
    err = Matrix{Float32}(permutedims(afr["err"][]))
    var_poisson = Matrix{Float32}(permutedims(afr["var_poisson"][]))
    return (; data = image_data, dq, err, var_poisson, meta)
end

"""
    load_area(fname) -> NamedTuple

Read a Roman WFI pixel area map (PAM) reference file.

# Returns
- `data::Matrix{Float32}`: the relative pixel area array, in `(y, x)` order.
- `meta`: the file's `roman.meta` tree, passed through as ASDF returns it.
"""
function load_area(fname::AbstractString)
    afr = _roman_tree(_load_asdf(fname), fname)
    return (; data = Matrix{Float32}(permutedims(afr["data"][])), meta = afr["meta"])
end
