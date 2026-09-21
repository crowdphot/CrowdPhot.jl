# photometry.jl
#
# Minimal end-to-end PSF photometry on one Roman WFI Level 2 exposure.
# Loads the image, resolves and loads its reference files from CRDS, fits every
# source, writes a flat table of results.  `fit_type` selects the simultaneous
# or the sequential fitter.
#
# Run it directly:
#
#     julia --project=. photometry.jl /path/to/r..._cal.asdf
#
# or, better while you are experimenting, from a REPL started with
# `julia --project=.`:
#
#     include("photometry.jl")
#     main("/path/to/r..._cal.asdf"; fit_rad = 4)
#
# The first call spends a while compiling before it prints anything.  Staying in
# one REPL session pays that once instead of once per file.

# Load Julia package code, timing how long it takes.
const J_PACKAGE_LOADS = @elapsed begin
    using CrowdPhot
    using CrowdPhot: Roman
    using LinearAlgebra: BLAS
    import Parquet2
end

# Loads CRDS code, timing how long it takes.
const T_PYTHON = @elapsed include(joinpath(@__DIR__, "crds_query_pythoncall.jl"))

# Print one line per phase to show startup costs.
# On a cold process most of these numbers are Julia compiling, not
# work: a second call to `main` in the same session is far faster.  See the
# REPL paragraph under "Running it" in README.md.
macro step(label, ex)
    quote
        local t = @elapsed local v = $(esc(ex))
        println("  ", rpad($(esc(label)), 34), lpad(round(t; digits = 1), 6), " s")
        v
    end
end

# One BLAS thread per process. The BLAS work is generally pretty small
# so multi-threading does not really speed it up, and
# `photometry_distributed.jl` runs several worker
# processes at once, and each one spawning a full BLAS pool would oversubscribe
# the machine badly, hurting performance.  
# This file is `@everywhere include`d there, so setting it
# at top level here sets it on every worker.
BLAS.set_num_threads(1)

"""
    main(l2_path; outdir = "results", kwargs...) -> String

Photometry on one Roman WFI Level 2 exposure, writing the result to a Parquet
file under `outdir`, named after the input.  Returns the output path.

`fit_type` selects the fitter: `:simultaneous` solves for every source at once,
`:sequential` visits them one at a time.  The remaining keyword arguments are
the ones worth tuning; they are forwarded to whichever fitter is chosen, and
both docstrings document each one in full.  Most are shown at their default
values so the whole control surface is visible in one place.
"""
function main(l2_path::AbstractString;
        outdir::AbstractString = "results",
        fit_type::Symbol = :simultaneous,
        # --- detection ---
        kernel_rad::Integer = 10,       # half-width of the matched filter kernel
        detect_sigma::Real = 5.0,       # detection threshold, in sigma
        bkg_box_size::Integer = 25,     # background mesh size, in pixels
        # --- blend gate ---
        # This parameters control the blend gate, which damps detection in the wings
        # of bright sources.
        blend_threshold::Real = 0.25,
        blend_threshold_initial::Real = 0.5,
        blend_passes::Integer = 2,
        # --- fitting ---
        fit_rad::Real = 5,              # fitting box half-width, in pixels
        max_iter::Integer = 3,          # maximum detection passes
        min_iter::Integer = 1,          # minimum detection passes
        few_sources::Integer = 100,     # pass finding fewer than this ends the run
        λ_init::Real = 1.0e-3,          # initial Levenberg-Marquardt damping
        max_step::Real = 1.0,           # maximum centroid step per iteration, in pixels
        show_trace::Bool = true,        # show informative output
    )
    fit_type in (:simultaneous, :sequential) ||
        throw(ArgumentError("fit_type must be :simultaneous or :sequential, got $(repr(fit_type))"))
    println("Processing ", basename(l2_path), " (", fit_type, ")")
    t_start = time()
    println("  ", rpad("julia package load", 34), lpad(round(J_PACKAGE_LOADS; digits = 1), 6), " s")
    println("  ", rpad("python startup", 34), lpad(round(T_PYTHON; digits = 1), 6), " s")
    img = @step "load L2" Roman.load_l2(l2_path)

    # CRDS picks the ePSF and pixel area map that match this exposure's
    # detector, filter and observation date.  Both are downloaded into
    # $CRDS_PATH on first use and cached afterwards.
    refs = @step "CRDS lookup" get_references(l2_path, ["epsf", "area"])
    psf = @step "load ePSF" Roman.crds_gridded_epsf(refs["epsf"]; psf_subtype = "psf")
    pam = @step "load pixel area map" Roman.load_area(refs["area"])

    # Roman L2 data are in surface brightness units, so a pixel's value does
    # not scale with its area.  Photometry sums pixels, so the area has to be
    # multiplied back in first; skipping this gives wrong fluxes, not just
    # imprecise ones.
    img.data .*= pam.data

    # NaNs mark pixels the detector never covered, which is different from a
    # pixel that was observed and is merely bad. This is typically not needed
    # but included for safety.
    coverage_mask = isnan.(img.data)

    # Inverse variance fit weights from the calibrated error array.  There is no
    # separate bad-pixel keyword on the multipass entry point -- a zero weight
    # means "exclude this pixel from detection and from the fit", so
    # the DQ mask is folded in here.
    inv_var = @step "build weights" begin
        iv = convert(Matrix{eltype(img.data)}, inv.(img.err .^ 2))
        iv[img.dq .| .!isfinite.(img.data) .| .!isfinite.(iv)] .= 0
        iv
    end

    # Background estimation, detection, deblending, fitting and pruning,
    # iterated to convergence.  The two fitters return the same result type and
    # share every keyword exposed through `main`; they differ in how a pass updates sources.
    # Note that there are some keyword arguments that adjust the behavior of the
    # fitters that are specific to each method (e.g., `linear_tol` for simultaneous and
    # `sweeps_per_pass` for the sequential one).  Add them here when tuning.
    mp = @step "fit" if fit_type === :simultaneous
        fit_all_stars_simultaneous_multipass(img.data, psf, fit_rad;
            inv_var, coverage_mask, kernel_rad, detect_sigma, bkg_box_size,
            blend_threshold, blend_threshold_initial, blend_passes,
            max_iter, min_iter, few_sources, λ_init, max_step, show_trace)
    else
        fit_all_stars_multipass(img.data, psf, fit_rad;
            inv_var, coverage_mask, kernel_rad, detect_sigma, bkg_box_size,
            blend_threshold, blend_threshold_initial, blend_passes,
            max_iter, min_iter, few_sources, λ_init, max_step, show_trace)
    end

    println("Fit ", length(mp.phot.y), " sources")

    # The full result is a deep nested schema carrying every intermediate
    # quantity.  `to_table` flattens it to the columns most analyses want, with
    # the PSF referencing already applied to the morphology statistics.
    #
    # Fitted fluxes are in the data units of the pixel-area-corrected image above.
    # `jansky_per_flux_unit` reads the conversion factors from the image metadata
    # to put them on the AB system.  `abmag_err` is scale invariant, so it takes
    # the raw fitted flux and flux error directly and needs no calibration.
    jy = Roman.jansky_per_flux_unit(img.meta) |> eltype(mp.phot.flux)
    tbl = @step "to_table" to_table(mp;
        ABmag = abmag.(mp.phot.flux .* jy),
        ABmag_err = abmag_err.(mp.phot.flux, mp.phot.flux_err))

    mkpath(outdir)
    outpath = joinpath(outdir, replace(basename(l2_path), r"\.asdf$" => "") * ".parquet")
    @step "write parquet" Parquet2.writefile(outpath, tbl)
    println("  ", rpad("TOTAL (excl. python startup)", 34),
            lpad(round(time() - t_start; digits = 1), 6), " s")
    println("Wrote ", outpath)
    return outpath
end

# Only runs when this file is executed as a script, so that
# `photometry_distributed.jl` can `include` it just to get `main`.
if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS[1])
end
