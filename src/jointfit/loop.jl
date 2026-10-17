# The joint detect -> fit -> subtract multipass loop, step for step the single-image
# `_fit_all_stars_multipass`:
#
#   per pass: render the model at θ -> per image, the package's `estimate_background_multipass`
#   on raw - model (host) -> joint detection on all residuals (maps, peaks accepted on
#   `accept >= detect_sigma`, wing gate) -> dedup against the catalog and the discard log (prunes after pass 1) at
#   `min_separation` (frame px) -> seeds: refined peak position and per-band fluxes from
#   `joint_point` -> `fit_pass` -> prune on the joint signed significance (Stouffer over bands of
#   f_b sqrt(H_bb)) and close pairs.
#   Stop on relative catalog growth (min_growth, from pass 3) or max_iter, then a terminal refit
#   without detection or pruning and the finalize step.

"""
    JointLoopOptions(; kwargs...)

- `detector = JointDetector()`, `detect_sigma = 5.0`, `accept = :nu`: peaks are thresholded on
  `accept` (`:nu` or `:snm`) at `detect_sigma`
- `dedup = 1.5`: peak suppression radius (frame px)
- `gate = 2.0`: wing-gate factor applied in every pass (`0` disables it)
- `band_weights = Dict()`: detection weights per band (missing bands weigh 1)
- `min_separation = 1.0`: new peaks within this of a catalog or discarded source are dropped
  (discarded: pruned in a pass after the first, whose background still holds the stars' light)
- `prune_snr_min = 3 detect_sigma / 5`, `prune_separation = 0.5`

  Both are tighter than the single-image defaults (1.5 and 1.0): a blend's second star can only
  enter as a residual peak near the first, which `min_separation = 1.5` rejects.  On simulated
  Roman fields 1.0 / 0.5 resolved 3x more input pairs closer than 1.5 px and gained ~2% true
  sources at a fixed false count; 0.5 / 0.5 gained no more pairs and, with one exposure per
  band, doubled the split-star fragments.
- `min_growth = 0.01`, `max_iter = 10`: from pass 3 on, stop once a pass's new sources that
  survive pruning are at most `min_growth` times the catalog; `max_iter` caps the passes
  (`min_growth = 0` runs exactly `max_iter`).  A relative rule, unlike an absolute count of
  new sources, so the pass count does not grow with the field's area: on simulated
  Roman fields of 6k to 100k sources, growth after passes 2, 3 and 4 was ~17%, ~2.5% and
  0.3-0.8%, so the default stops after 4 passes at every size.
- `fit = JointFitOptions()`, `error_neighbors = 8`
- `background`: the background keywords `bkg_box_size = 20` (one mesh for the level and RMS
  maps, in every pass), `bkg_estimator = SExtractorBackground()`,
  `bkg_rms_estimator = MADStdRMS()`, `bkg_kws = (;)` (passed to the background estimator,
  e.g. `sigma`, `filter_size`); any other key throws.  Pixels with zero fit weight are masked
- `device_background = true`: estimate the background on the backend ([`background2d!`]);
  boxes over $MAX_BOX_PIXELS pixels and the biweight estimators use the host `Background2D`
"""
Base.@kwdef struct JointLoopOptions
    detector::JointDetector = JointDetector()
    detect_sigma::Float64 = 5.0
    accept::Symbol = :nu
    dedup::Float64 = 1.5
    gate::Float64 = 2.0
    band_weights::Dict{Symbol, Float64} = Dict{Symbol, Float64}()
    min_separation::Float64 = 1.0
    prune_snr_min::Float64 = 3 * detect_sigma / 5
    prune_separation::Float64 = 0.5
    min_growth::Float64 = 0.01
    max_iter::Int = 10
    fit::JointFitOptions = JointFitOptions()
    error_neighbors::Int = 8
    background::NamedTuple = (;)
    device_background::Bool = true
end

# The resolved background keywords, with the single-image loop's defaults.
const _BACKGROUND_KEYS = (:bkg_box_size, :bkg_estimator, :bkg_rms_estimator, :bkg_kws)
function _background_settings(kw::NamedTuple)
    bad = setdiff(keys(kw), _BACKGROUND_KEYS)
    isempty(bad) || throw(ArgumentError("unsupported joint background keyword(s) $(join(bad, ", ")); accepted: $(join(_BACKGROUND_KEYS, ", "))"))
    return (; box = get(kw, :bkg_box_size, 20), estimator = get(kw, :bkg_estimator, CrowdPhot.SExtractorBackground()),
              rms_estimator = get(kw, :bkg_rms_estimator, CrowdPhot.MADStdRMS()), kws = get(kw, :bkg_kws, (;)))
end

"""
Announce once per run why the background falls back to the host
`Background2D`: a `@warn` on a GPU, where each affected pass copies every image to the host and
back and the host estimator is ~25x slower than `background2d!`; an `@info` on the CPU, where it
costs ~1.3x.  Nothing is reported when `device_background = false` was asked for.
"""
function _report_background_fallback(b, o::JointLoopOptions, backend)
    o.device_background || return nothing
    why = String[]
    _device_estimator(b.estimator) || push!(why, "bkg_estimator = $(nameof(typeof(b.estimator)))")
    _device_rms(b.rms_estimator) || push!(why, "bkg_rms_estimator = $(nameof(typeof(b.rms_estimator)))")
    _device_box(b.box) || push!(why, "bkg_box_size = $(b.box)")
    isempty(why) && return nothing
    msg = "Joint background falls back to the host Background2D: " * join(why, ", ") *
          ".  The device estimator supports Mean/Median/SExtractor/MMM locations, Std/MADStd RMS and boxes up to $MAX_BOX_PIXELS pixels."
    is_cpu(backend) ? (@info msg) : (@warn msg)
    return nothing
end

# Whole-array steps of `_backgrounds!` as kernels: Base broadcasts over CPU arrays run on one
# thread, and the three outputs after the background share one pass on every backend.
@kernel function sub_kernel!(out, @Const(a), @Const(b))
    i = @index(Global)
    @inbounds out[i] = a[i] - b[i]
end

# detection residual raw - bkg - model and detection inverse variance 1 / rms² (zero where the
# rms or the fit weight is unusable)
@kernel function post_background_kernel!(det, div, @Const(raw), @Const(bkg), @Const(model), @Const(rms), @Const(w))
    i = @index(Global)
    T = eltype(det)
    @inbounds begin
        det[i] = (raw[i] - bkg[i]) - model[i]
        r = rms[i]; wi = w[i]
        div[i] = ifelse(isfinite(r) & (r > 0) & isfinite(wi) & (wi > 0), inv(r * r), zero(T))
    end
end

"""
Per image: the model at θ, then the background of raw - model, on the backend
(`background2d!`) or, for options it does not support, with the host `Background2D` (the
single-image `estimate_background_multipass`, threaded over images).  Writes the background into
`st.bkg` (the fit data is `sc.data - st.bkg`, formed in the cost kernel; `sc.data` stays the raw
images), and the detection residual (raw - background - model) and detection inverse variance
(1 / rms², zero where the fit weight is) into `st`.
"""
function _backgrounds!(st, θ, pass, o::JointLoopOptions)
    sc = st.sc; T = eltype(sc); b = st.bset; E = nimages(sc)
    model = st.work.model
    if size(θ, 2) > 0
        render!(model, sc, θ, footprint(sc, θ, st.rt))
    else
        fill!(model, zero(T))
    end
    if o.device_background && device_background_ok(b.box, b.estimator, b.rms_estimator)
        launch!(sub_kernel!, sc.backend, length(model), st.tmp, sc.data, model)
        for e in 1:E
            v(x) = image_view(sc, x, e)
            kw = (; estimator = b.estimator, rms_estimator = b.rms_estimator, mask = v(st.bmask), b.kws...)
            background2d!(v(st.bkg), v(st.rms), v(st.tmp), b.box; kw...)
        end
    else
        mh = Array(model)
        bk = Vector{Matrix{T}}(undef, E); rm = similar(bk)
        Threads.@threads :dynamic for e in 1:E
            im = st.images[e]; w = im.weight
            m = reshape(view(mh, sc.offsets[e] + 1:sc.offsets[e + 1]), sc.dims[e])
            opts = (; bkg_box_size = b.box, bkg_coarse_passes = 0, bkg_rms_box_size = b.box,
                      bkg_estimator = b.estimator, bkg_rms_estimator = b.rms_estimator, bkg_kws = b.kws, bkg_mask = map(x -> !(isfinite(x) && x > 0), w),
                      coverage_mask = nothing, fixed_inv_var = w)
            r = CrowdPhot.estimate_background_multipass(im.data, m, pass, opts)
            bk[e] = r.background; rm[e] = r.rms
        end
        copyto!(st.bkg, reduce(vcat, vec.(bk))); copyto!(st.rms, reduce(vcat, vec.(rm)))
    end
    launch!(post_background_kernel!, sc.backend, length(model), st.det_data, st.det_iv, sc.data, st.bkg, model, st.rms, sc.weight)
    return nothing
end

# Uniform-bucket index answering "is there a point within `sep`?"
_sep_index(sep) = (; sep, cells = Dict{NTuple{2, Int}, Vector{NTuple{2, Float64}}}())
_sep_add!(ix, u, v) = push!(get!(() -> NTuple{2, Float64}[], ix.cells, (floor(Int, u / ix.sep), floor(Int, v / ix.sep))), (u, v))
function _sep_hit(ix, u, v)
    cu = floor(Int, u / ix.sep); cv = floor(Int, v / ix.sep)
    for a in -1:1, b in -1:1, p in get(ix.cells, (cu + a, cv + b), NTuple{2, Float64}[])
        (p[1] - u)^2 + (p[2] - v)^2 < ix.sep^2 && return true
    end
    return false
end

"Joint detection on the pass's residuals: new (u, v) and per-band seed fluxes after the gate and dedup."
function _detect_new(st, θh, disc, o::JointLoopOptions)
    sc = st.sc; jd = st.jd; K = jd.det.K
    maps = joint_maps(jd, sc, st.det_data, st.det_iv)
    pk = find_peaks(maps; floor = o.detect_sigma, dedup = o.dedup * K, accept = o.accept)
    n_peaks = length(pk.y)
    n_gated = 0
    if st.tmpl !== nothing
        keep = wing_gate(pk, st.tmpl; gate = o.gate)
        n_gated = count(!, keep)
        pk = map(x -> x[keep], pk)
    end
    occ = _sep_index(o.min_separation); dsc = _sep_index(o.min_separation)
    for s in axes(θh, 2); _sep_add!(occ, θh[1, s], θh[2, s]); end
    for (u, v) in disc; _sep_add!(dsc, u, v); end
    us = Float64[]; vs = Float64[]; nd_cat = 0; nd_disc = 0
    for i in eachindex(pk.y)                       # find_peaks orders by snm, descending
        u = (pk.y[i] - 0.5) / K + 0.5; v = (pk.x[i] - 0.5) / K + 0.5
        _sep_hit(occ, u, v) && (nd_cat += 1; continue)
        _sep_hit(dsc, u, v) && (nd_disc += 1; continue)
        push!(us, u); push!(vs, v)
        _sep_add!(occ, u, v)                       # mutual dedup within the batch
    end
    B = nbands(sc)
    flux = zeros(B, length(us))
    if !isempty(us)
        jp = Array(joint_point(jd, sc, st.det_data, st.det_iv, to_device(sc.backend, eltype(sc), permutedims(hcat(us, vs)))))
        flux .= map(x -> isfinite(x) ? Float64(x) : 0.0, jp[4:3 + B, :])
    end
    return (; u = us, v = vs, flux, n_peaks, n_gated, nd_cat, nd_disc)
end

"Joint signed significance Stouffer_b(f_b sqrt(H_bb)) over bands with data (NaN without any)."
function _flux_significance(θh::AbstractMatrix, hdiag::AbstractMatrix)
    P, N = size(θh)
    js = fill(NaN, N)
    for s in 1:N
        acc = 0.0; nb = 0
        for b in 3:P
            h = hdiag[b, s]
            h > 0 || continue
            acc += θh[b, s] * sqrt(h); nb += 1
        end
        nb > 0 && (js[s] = acc / sqrt(nb))
    end
    return js
end

"Keep mask: significance >= snr_min, then close pairs (greedy by significance)."
function _prune_keep(θh, js, snr_min, separation)
    keep = [isfinite(j) && j >= snr_min for j in js]
    n_snr = count(!, keep)
    ix = _sep_index(separation); n_close = 0
    for s in sort(findall(keep); by = s -> -js[s])
        if _sep_hit(ix, θh[1, s], θh[2, s])
            keep[s] = false; n_close += 1
        else
            _sep_add!(ix, θh[1, s], θh[2, s])
        end
    end
    return keep, n_snr, n_close
end

# Rough peak memory (bytes) of a pass with N sources and np pairs: the per-pixel arrays at a constant per
# image pixel (the measured Roman-field peak at 2000 px images less its source rows, so live arrays plus
# uncollected garbage: 87 B on the CPU in Float64, 56 B on Metal; Float32 on the CPU is overestimated),
# plus H (own and pair blocks) and its finalize host copy exactly.
_memory_estimate(sc::JointScene, N, np) = (is_cpu(sc.backend) ? 87 : 56) * length(sc.data) + 2 * nparams(sc)^2 * sizeof(eltype(sc)) * (N + np)

"""
    fit_joint_multipass(images, frame; backend = KernelAbstractions.CPU(), T = Float64,
                        options = JointLoopOptions(), show_trace = false, products = ()) -> NamedTuple

Joint multipass photometry of `images` (`JointImage`s) with source positions on `frame`.
Returns `catalog`, a `StructArray` (a Tables.jl table) with columns `u, v` (positions on
`frame`), `ra, dec`, `u_err, v_err`, then per band `b` (in `bands` order) `flux_b`, `flux_err_b`,
the joint final `significance` and per band `significance_b`, `nimg` (images with data) and
`pass_number` (the pass that first detected the source); the `bands` order; the per-pass
`history`, whose `mem_est` is a rough peak memory in bytes (a warning is logged once on the CPU
or an Apple GPU if it exceeds half of `Sys.total_memory()`); whether the stopping rule fired
(`converged`); and `products`.

`products` lists the arrays to return, on the host in `T`, at the final catalog (the background
and RMS are the terminal pass's, the ones the final fit used); `products.grid` is always the
detection grid (`TanGrid`):
- per image, one matrix per image in `images` order on its pixel grid: `:background`, `:rms`
  (the measured background RMS), `:model`, `:residual` (data - background - model);
- detection maps, one matrix on `products.grid`: `:nu` (the calibrated joint significance),
  `:snm` (`ν / √(χ²_red + 0.25)`, the statistic peaks are found on), `:chi2` (`χ²_red`), each
  computed on data - background, the same three with suffix `_residual` computed on the final
  residual (what a further detection pass would search), and `:nimg` (images with data at each
  node).  `:chi2` needs `method = :dolphot`;
- per-band coadds on `products.grid` ([`coadd`](@ref) of the images with the final background,
  RMS and model, computed from the scene's own arrays), each a `NamedTuple` keyed by band:
  `:coadd_data`, `:coadd_background`, `:coadd_model`, `:coadd_residual`, `:coadd_rms`,
  `:coadd_weight`, `:coadd_nimg`, which `:coadd` requests together (one pass per band), and `:coadd_rms_aperture`,
  the per-pixel noise for sums over large areas such as aperture photometry on the coadd, which
  must be requested by name (it costs two more passes per band and an array of the band's input
  pixels; see `coadd`).  `coadd_options` passes `pixfrac`, `units`, `weight`, `clip`,
  `clip_grow`, `clip_mu` and `clip_min_images` to `coadd`.
"""
function fit_joint_multipass(images::AbstractVector{<:JointImage}, frame::TanGrid; backend = KA.CPU(), T::Type{<:AbstractFloat} = Float64,
        options::JointLoopOptions = JointLoopOptions(), show_trace::Bool = false, products = (), coadd_options::NamedTuple = (;))
    o = options
    products = _check_run_args(o, products, coadd_options)
    t_setup = @elapsed begin
        sc = JointScene(images, frame; backend, T)
        _check_bands(sc.bands)
        psfs = [im.psf for im in images]; offs = [im.psf_offset for im in images]
        f = o.fit
        rt = RadiusTable(sc, psfs, [im.weight for im in images]; model_rad = f.model_rad, nsigma = f.model_rad_nsigma, f.R_fit, f.R_cap)
        jd = JointDetection(sc, psfs, offs, o.detector; weights = o.band_weights)
        bset = _background_settings(o.background)
        _report_background_fallback(bset, o, backend)
        det_data = similar(sc.data); det_iv = similar(sc.data)
        # the template depends only on the exposures' PSFs, orientations and weights: once per run,
        # with the detection buffers (not yet in use) as scratch
        ivs = [median(filter(w -> isfinite(w) && w > 0, vec(im.weight))) for im in images]
        tmpl = o.gate > 0 ? wing_template(jd, sc, ivs; data = det_data, iv = det_iv) : nothing
        # the scene's data stay the raw images; the fit subtracts each pass's background (work.bkg)
        bkg = KA.zeros(sc.backend, T, length(sc.data))
        st = (; sc, images, rt, jd, tmpl, work = LinearizeWork(sc; bkg), bset,
                bkg, rms = similar(sc.data), tmp = similar(sc.data), bmask = map!(w -> !(isfinite(w) && w > 0), KA.allocate(sc.backend, Bool, length(sc.weight)), sc.weight),
                det_data, det_iv)
    end
    P = nparams(sc)
    θh = zeros(P, 0); born = Int[]; disc = NTuple{2, Float64}[]
    λ = f.λ_init; history = NamedTuple[]; converged = false; pass_run = 0; mem_warned = false
    for pass in 1:o.max_iter
        pass_run = pass
        t_bkg = @elapsed _backgrounds!(st, to_device(backend, T, θh), pass, o)
        t_det = @elapsed dn = _detect_new(st, θh, disc, o)
        θh = hcat(θh, vcat(dn.u', dn.v', dn.flux)); append!(born, fill(pass, length(dn.u)))
        n_new = length(dn.u)
        isempty(born) && (push!(history, (; pass, dn.n_peaks, n_new = 0, n_catalog = 0)); continue)
        t_fit = @elapsed r = fit_pass(sc, to_device(backend, T, θh), rt, f, λ; n_new, work = st.work)
        mem_est = _memory_estimate(sc, size(r.θ, 2), size(r.lin.Ho, 3))
        if !mem_warned && (is_cpu(backend) || Sys.isapple()) && mem_est > Sys.total_memory() / 2
            mem_warned = true
            @warn "Estimated peak memory $(round(mem_est / 1e9, digits = 1)) GB at pass $pass ($(size(r.θ, 2)) sources) exceeds half of the system memory ($(round(Sys.total_memory() / 1e9, digits = 1)) GB); fit a smaller area per run."
        end
        θh = Float64.(Array(r.θ)); λ = r.λ
        js = _flux_significance(θh, Array(r.lin.hdiag))
        js[Array(r.lin.nimg) .== 0] .= NaN
        keep, n_snr, n_close = _prune_keep(θh, js, o.prune_snr_min, o.prune_separation)
        # pass 1 fits against a background that still holds every star's light, so its prunes include
        # real faint stars; logging them would block their re-detection for the rest of the run (on
        # a crowded Roman tile, leaving them out kept 123 more true sources for 10 more spurious ones)
        pass > 1 && for s in findall(!, keep); push!(disc, (θh[1, s], θh[2, s])); end
        θh = θh[:, keep]; born = born[keep]
        n_surv = count(==(pass), born)
        h = (; pass, dn.n_peaks, dn.n_gated, n_dup_cat = dn.nd_cat, n_dup_disc = dn.nd_disc, n_new, n_pruned_snr = n_snr,
               n_pruned_close = n_close, n_catalog = length(born), n_new_surviving = n_surv, λ, r.stats..., mem_est, t_bkg, t_det, t_fit)
        push!(history, h)
        show_trace && println(h)
        # growth after passes 1 and 2 is always large, so the rule starts at pass 3
        if pass >= 3 && o.min_growth > 0 && n_surv <= o.min_growth * length(born)
            converged = true
            break
        end
    end
    t_final = @elapsed begin
        cat, prods = _terminal_and_finalize(st, θh, born, λ, pass_run + 1, o, products, coadd_options)
    end
    # every result is on the host: return the device memory now, since a GPU array's is otherwise held
    # until the GC finalizes it, which host memory pressure barely drives (a no-op on the CPU)
    bk = st.jd.banks
    foreach(KA.unsafe_free!, (sc.data, sc.weight, st.bkg, st.rms, st.tmp, st.bmask, st.det_data, st.det_iv, st.work.model, st.work.resid,
                              st.work.mask, bk.P, bk.c, bk.cχ, bk.rng))
    return (; catalog = cat, bands = sc.bands, history, converged, products = prods, t_setup, t_final)
end

"Checks of `fit_joint_multipass`'s arguments that need no data; `products` with `:coadd` expanded."
function _check_run_args(o::JointLoopOptions, products, coadd_options)
    o.accept in (:nu, :snm) || throw(ArgumentError("accept must be :nu or :snm"))
    products = Tuple(unique(Iterators.flatten(p === :coadd ? _COADD_DEFAULT : (p,) for p in products)))
    avail = (_IMAGE_PRODUCTS..., _MAP_PRODUCTS..., _COADD_PRODUCTS...)
    bad = setdiff(products, avail)
    isempty(bad) || throw(ArgumentError("unknown products $(join(bad, ", ")); available: $(join(avail, ", ")), coadd"))
    okc = (:pixfrac, :units, :weight, :clip, :clip_grow, :clip_mu, :clip_min_images)
    badc = setdiff(keys(coadd_options), okc)
    isempty(badc) || throw(ArgumentError("unsupported coadd_options $(join(badc, ", ")); accepted: $(join(okc, ", "))"))
    !uses_chi(o.detector) && any(in(products), (:chi2, :chi2_residual)) &&
        throw(ArgumentError("products :chi2 and :chi2_residual need a detector with a χ² (method = :dolphot), not method = $(repr(o.detector.method))"))
    return products
end

# the per-band catalog columns flux_b, flux_err_b collide only for bands b and err_b
function _check_bands(bands)
    for b in bands
        Symbol(:err_, b) in bands && throw(ArgumentError("bands $b and err_$b give the same catalog column flux_err_$b"))
    end
end

# The `products` of `fit_joint_multipass`: per-image arrays on each image's pixel grid, and
# detection maps on the detection grid (`_residual`: on the final residual, else on data - background).
const _IMAGE_PRODUCTS = (:background, :rms, :model, :residual)
const _MAP_PRODUCTS = (:nu, :snm, :chi2, :nu_residual, :snm_residual, :chi2_residual, :nimg)

"The requested `products` at the final catalog θh, plus the detection `grid`."
function _products(st, θh, products, coadd_options)
    sc = st.sc; be = sc.backend; T = eltype(sc); model = st.work.model
    isempty(products) && return (; grid = st.jd.grid)
    θd = to_device(be, T, θh)
    size(θh, 2) > 0 ? render!(model, sc, θd, footprint(sc, θd, st.rt)) : fill!(model, zero(T))
    # the final residual raw - bkg - model and detection inverse variance, as each pass forms them
    launch!(post_background_kernel!, be, length(model), st.det_data, st.det_iv, sc.data, st.bkg, model, st.rms, sc.weight)
    per_image(x) = (h = Array(x); [reshape(h[(sc.offsets[e] + 1):sc.offsets[e + 1]], sc.dims[e]) for e in 1:nimages(sc)])
    vals = Dict{Symbol, Any}()
    for (p, x) in ((:background, st.bkg), (:rms, st.rms), (:model, model), (:residual, st.det_data))
        p in products && (vals[p] = per_image(x))
    end
    want(names) = any(in(products), names)
    function maps!(data, suffix)
        m = joint_maps(st.jd, sc, data, st.det_iv)
        vals[Symbol(:nu, suffix)] = Array(m.ν); vals[Symbol(:snm, suffix)] = Array(m.snm); vals[Symbol(:chi2, suffix)] = Array(m.χ2)
        vals[:nimg] = Array(m.nimg)
    end
    want((:nu_residual, :snm_residual, :chi2_residual)) && maps!(st.det_data, :_residual)
    if want((:nu, :snm, :chi2)) || (:nimg in products && !haskey(vals, :nimg))
        launch!(sub_kernel!, be, length(model), st.tmp, sc.data, st.bkg)
        maps!(st.tmp, Symbol(""))
    end
    cq = Tuple(q for (q, p) in zip(_COADD_QUANTITIES, _COADD_PRODUCTS) if p in products)
    if !isempty(cq)
        # the scene's arrays and PolyMaps (frame -> image; the detection grid is K times finer)
        inp = (; data = sc.data, weight = sc.weight, bkg = st.bkg, rms = st.rms, model)
        C = _coadd(sc.dev, sc.footprints, sc.bands, st.jd.grid, st.jd.det.K, inp, cq; backend = be, T, coadd_options...)
        for q in cq; vals[Symbol(:coadd_, q)] = C[q]; end
    end
    names = Tuple(p for p in (_IMAGE_PRODUCTS..., _MAP_PRODUCTS..., _COADD_PRODUCTS...) if p in products)
    return NamedTuple{(names..., :grid)}((map(p -> vals[p], names)..., st.jd.grid))
end

"The terminal pass (background, refit, no detection or pruning), finalize and the requested products."
function _terminal_and_finalize(st, θh, born, λ, pass, o::JointLoopOptions, products, coadd_options)
    sc = st.sc; be = sc.backend; T = eltype(sc); f = o.fit
    _backgrounds!(st, to_device(be, T, θh), pass, o)
    if size(θh, 2) > 0
        r = fit_pass(sc, to_device(be, T, θh), st.rt, f, λ; work = st.work)
        θh = Float64.(Array(r.θ))
        js = _flux_significance(θh, Array(r.lin.hdiag))
        ok = [all(isfinite, view(θh, :, s)) && isfinite(js[s]) && js[s] > 0 for s in axes(θh, 2)]
        θh = θh[:, ok]; born = born[ok]
    end
    N = size(θh, 2); B = nbands(sc)
    σ = fill(NaN, 2 + B, N); nimg = zeros(Int, N)
    if N > 0
        lin = linearize(sc, to_device(be, T, θh), st.rt, st.work; f.R_fit, f.stamp_budget)
        σ = Float64.(marginal_errors(HostH(lin); k = o.error_neighbors))
        nimg = Int.(Array(lin.nimg))
    end
    sb, joint = final_significance(θh, σ)
    radec = [grid_to_world(sc.frame, θh[1, s], θh[2, s]) for s in 1:N]
    # one column per band and quantity, named by band symbol
    perband(q, M) = NamedTuple{Tuple(Symbol(q, :_, b) for b in sc.bands)}(Tuple(M[:, k] for k in 1:B))
    cols = merge((; u = θh[1, :], v = θh[2, :], ra = first.(radec), dec = last.(radec), u_err = σ[1, :], v_err = σ[2, :]),
                 perband(:flux, permutedims(θh[3:end, :])), perband(:flux_err, permutedims(σ[3:end, :])), (; significance = joint),
                 perband(:significance, sb), (; nimg, pass_number = born))
    return StructArray(cols), _products(st, θh, products, coadd_options)
end
