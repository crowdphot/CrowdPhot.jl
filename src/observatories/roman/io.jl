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
    load_l2(fname; dq_flags = (:DO_NOT_USE, :NO_LIN_CORR, :HOT), bad_pixels = false, ramp_weights = :optimal, uniform_jumps = :clip) -> NamedTuple

Read a Roman WFI Level 2 (calibrated rate) ASDF file.

# Arguments
- `fname::AbstractString`: path to the `.asdf` file.

# Keyword arguments
- `dq_flags`: the [`DQ_FLAGS`](@ref) names that mark a pixel unusable.  See
  [`parse_dq_mask`](@ref).
- `bad_pixels`: use heuristics to try to mark unusable pixels that the DQ array does not flag
  (invalid errors, cold and hot pixels; see [`bad_pixel_mask`](@ref)). Mostly used on `romanisim`
  simulations where the DQ array may not capture all problematic pixels.
- `ramp_weights`: `:optimal` returns the file's `data` (the optimally weighted ramp fit);
  `:uniform` returns `data + dumo`, the uniformly weighted ramp fit (`dumo` = "data uniform minus
  optimal", flat-fielded like `data`), with the cosmic rays put back by that fit handled by
  `uniform_jumps` (see [`uniform_delta`](@ref)).  `err` and `var_poisson` are the file's arrays in both
  cases, i.e. they describe the optimal fit, not the noisier uniform one.
- `uniform_jumps`: with `ramp_weights = :uniform`, what happens on JUMP_DET pixels: `:clip` (default)
  keeps the uniform value unless it differs from the optimal one by more than a cosmic ray's worth,
  `:keep` keeps every one (cosmic rays included), `:data` takes the optimal value on all of them.

# Returns
- `data::Matrix{Float32}`: the science array.
- `dq::BitMatrix`: `true` where any of `dq_flags` is set or, with `bad_pixels`,
  where [`bad_pixel_mask`](@ref) fires, i.e. `true` means "bad", matching the
  `bkg_mask` convention of [`CrowdPhot.fit_all_stars_multipass`](@ref).
- `err::Matrix{Float32}`: the total error array.
- `var_poisson::Matrix{Float32}`: the Poisson variance array.
- `meta`: the file's `roman.meta` metadata tree, passed through as ASDF returns it.
- `n_bad`: with `bad_pixels`, the counts `(; n_err, n_cold, n_hot)` of the pixels
  [`bad_pixel_mask`](@ref) added; otherwise `nothing`.
- `n_uniform`: with `ramp_weights = :uniform`, `(; n_jump, n_replaced)`: the JUMP_DET pixels and those
  that kept their optimal value; otherwise `nothing`.

!!! note
    ASDF.jl reads array data in reverse dimension order relative to the
    Python/C layout the file was written in, so every array here is
    `permutedims`ed back on the way out.  Arrays are returned in CrowdPhot's
    `(y, x)` convention.
"""
function load_l2(fname::AbstractString; dq_flags = (:DO_NOT_USE, :NO_LIN_CORR, :HOT), bad_pixels::Bool = false, ramp_weights::Symbol = :optimal,
                 uniform_jumps::Symbol = :clip)
    ramp_weights in (:optimal, :uniform) || throw(ArgumentError("ramp_weights must be :optimal or :uniform, got :$ramp_weights"))
    afr = _roman_tree(_load_asdf(fname), fname)
    meta = afr["meta"]
    image_data = Matrix{Float32}(permutedims(afr["data"][]))
    dq_raw = Matrix{UInt32}(permutedims(afr["dq"][]))
    err = Matrix{Float32}(permutedims(afr["err"][]))
    n_uniform = nothing
    if ramp_weights === :uniform
        haskey(afr, "dumo") || throw(ArgumentError("$fname has no \"dumo\" array, so ramp_weights = :uniform is unavailable"))
        u = uniform_delta(image_data, permutedims(afr["dumo"][]), dq_raw, err; jumps = uniform_jumps)
        image_data .+= u.delta
        n_uniform = (; u.n_jump, u.n_replaced)
    end
    dq = BitMatrix(parse_dq_mask(dq_raw; flags = dq_flags))
    var_poisson = Matrix{Float32}(permutedims(afr["var_poisson"][]))
    n_bad = nothing
    if bad_pixels
        bad = bad_pixel_mask(image_data, err, dq_raw; flags = dq_flags)
        dq .|= bad.mask
        n_bad = (; bad.n_err, bad.n_cold, bad.n_hot)
    end
    return (; data = image_data, dq, err, var_poisson, meta, n_bad, n_uniform)
end

"""
    uniform_delta(data, dumo, dq_raw, err; jumps = :clip, nsigma = 5, frac = 0.2) -> (; delta, n_jump, n_replaced)

What to add to an L2 file's `data` for the uniformly weighted ramp fit, from its `dumo` (all arrays the same shape, `dq_raw`
the raw DQ bits).  romancal computes `dumo` ignoring every resultant flag, so on a JUMP_DET pixel the uniform fit still holds
the cosmic ray the optimal fit took out (a few in 10^4 pixels of a 66 s image, `dumo` many times the sky).  JUMP_DET also
flags most bright-star cores, where the jump is often not real and the uniform value is the one wanted, so those pixels are
not simply masked.  `jumps = :clip` keeps `dumo` on a JUMP_DET pixel unless `|dumo| > nsigma * sqrt(err² + (frac * signal)²)`,
`signal` = the pixel's data minus the image median (0 if negative or not finite); a pixel without a positive finite `err`
is replaced.  `:keep` keeps every one, `:data` replaces every one.  A replaced pixel keeps its optimal value (`delta` 0);
non-finite `dumo` counts as 0.  The rule and its defaults are roman-mast-tools' `roman_dumo.uniform_delta` (`frac = 0.2`
gave the best repeatability of the values tried on program 1043's repeats).
"""
function uniform_delta(data, dumo, dq_raw, err; jumps::Symbol = :clip, nsigma::Real = 5, frac::Real = 0.2)
    jumps in (:clip, :keep, :data) || throw(ArgumentError("uniform_jumps must be :clip, :keep or :data, got :$jumps"))
    size(dumo) == size(data) == size(dq_raw) == size(err) || throw(DimensionMismatch("data, dumo, dq_raw and err must have the same size"))
    delta = Float32[isfinite(x) ? x : 0.0f0 for x in dumo]
    sky = median(Iterators.filter(isfinite, data))
    n_jump = n_replaced = 0
    for i in eachindex(delta)
        dq_raw[i] & DQ_FLAGS[:JUMP_DET] == 0 && continue
        n_jump += 1
        s = data[i] - sky
        s = isfinite(s) ? max(s, zero(s)) : zero(s)
        if jumps === :data || (jumps === :clip && !(err[i] > 0 && abs(delta[i]) <= nsigma * hypot(err[i], frac * s)))
            delta[i] = 0
            n_replaced += 1
        end
    end
    return (; delta, n_jump, n_replaced)
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
