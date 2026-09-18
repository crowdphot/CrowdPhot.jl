# Sequential (DOLPHOT-style) fitter for the multi-pass pipeline.
#
# Within a pass the catalog is swept brightest to faintest, as a Gauss-Seidel
# update over the residual image: each source's current model is added back
# to the residual, the source alone is fit with a few damped LM iterations, and
# the new model is subtracted again.  Everything around the fit -- background,
# detection, seeding, geometry, pruning, diagnostics, morphology -- is the shared
# pipeline in `multipass.jl`, so this file differs from the simultaneous fitter
# only in `fit_pass` and `source_errors`.
#
# Two invariants hold between source fits and are what make the sweep correct:
#
#   * `resid == data - model` pixel for pixel (both are updated in the same
#     scatter), and `model` is the pedestal-free render of every source at its
#     current parameters over its own `model_R` box.
#   * A source's local pedestal (`bkg`, when fit) lives only inside its own fit.
#     It never enters `model`, so overlapping footprints cannot accumulate it.

"""
    SequentialFitter{L}

The DOLPHOT-style [`AbstractMultipassFitter`](@ref): every pass sweeps the
catalog `sweeps` times, brightest first, fitting one source at a time against the
residual of all the others.  Built by [`fit_all_stars_multipass`](@ref) from its
fitter keywords.

# Fields

- `sweeps`: sweeps through the catalog per pass.
- `lm_iterations`: `max_iter` of each per-source [`fit_star`](@ref CrowdPhot.PSF.fit_star) call.
- `fit_bkg`: whether each source fits a local pedestal over its fitting box.
- `lambda_init`: the floor of every fit's starting damping factor.
- `skip_tol`: convergence check threshold, in sigma; `0` disables it.
- `lm`: the remaining keyword arguments forwarded to every LM solve.
"""
struct SequentialFitter{L <: NamedTuple} <: AbstractMultipassFitter
    sweeps::Int
    lm_iterations::Int
    fit_bkg::Bool
    lambda_init::Float64
    skip_tol::Float64
    lm::L
end

initial_fit_state(::SequentialFitter, ::Type) = nothing
n_free_per_source(fitter::SequentialFitter, plan::FitPlan) = plan.p + fitter.fit_bkg
# `fit_star` needs more usable pixels than free parameters.
min_stamp_pixels(fitter::SequentialFitter, plan::FitPlan) = n_free_per_source(fitter, plan) + 1
empty_pass_stats(::SequentialFitter, _, ::Type{FT}) where {FT} =
    (; n_lin = 0, n_trials = 0, n_accepted = 0, cost_start = FT(NaN), cost_end = FT(NaN),
       n_star_fits = 0, n_skipped = 0, n_capped = 0, sweep_costs = FT[],
       fit_timing = (; setup = 0.0, fits = 0.0))

# The `fixed` the per-source fits see: the render plan's, minus the pinned `bkg`
# when a local pedestal is fit.
_seq_fit_fixed(fitter::SequentialFitter, plan::FitPlan) =
    fitter.fit_bkg ? Base.structdiff(plan.fixed, NamedTuple{(:bkg,)}) : plan.fixed

# Render one source's unit-flux, pedestal-free model over its `model_R` box into
# the top-left `(2R + 1)^2` of `buf`.  Fixed parameters win over the catalog's,
# exactly as in `PSF.model_from_vector`, so this is the same model
# `_render_model!` draws, divided by its flux.
function _render_unit!(buf, psf, plan::FitPlan, yj, xj, ay::Int, ax::Int, R::Int, scratch)
    FT = eltype(buf)
    m = ConstructionBase.setproperties(psf, merge((; y = yj, x = xj, flux = one(FT)), plan.fixed))
    PSF.render!(buf, m, (ay - R):(ay + R), (ax - R):(ax + R), scratch)
    return buf
end

# Add `coef` times a unit render into `model` and take it out of `resid`, in
# lockstep, and return the flux curvature `H_ff = sum(w * u^2)` over the fitting
# box `anchor +- R_fit` -- the same `colnorm` the simultaneous fitter's flux
# column carries, so both fitters prune on the same `flux * sqrt(H_ff)`.
function _scatter_source!(resid, model, buf, coef, w, ay::Int, ax::Int, R::Int, R_fit::Int,
                          ny::Int, nx::Int)
    FT = eltype(resid)
    hff = zero(FT)
    S = 2R + 1
    @inbounds for jj in 1:S
        gx = ax - R + jj - 1
        (1 <= gx <= nx) || continue
        in_x = abs(gx - ax) <= R_fit
        for ii in 1:S
            gy = ay - R + ii - 1
            (1 <= gy <= ny) || continue
            q = gy + (gx - 1) * ny
            u = buf[ii, jj]
            v = coef * u
            model[q] += v
            resid[q] -= v
            (in_x && abs(gy - ay) <= R_fit) && (hff += w[q] * u * u)
        end
    end
    return hff
end

# Largest predicted Gauss-Newton step, in units of each parameter's own 1-sigma
# error, from the normal equations at the current parameters: `δ = -A⁻¹b`,
# `σ_k = sqrt((A⁻¹)_kk)`.  `Inf` when `A` is not positive definite, so such a
# source is always fit.
function _gn_step_sigma(A::AbstractMatrix, b::AbstractVector)
    F = cholesky(Symmetric(A); check = false)
    issuccess(F) || return oftype(float(zero(eltype(b))), Inf)
    Ai = inv(F)
    δ = Ai * b
    s = zero(eltype(δ))
    for k in eachindex(δ)
        s = max(s, abs(δ[k]) / sqrt(Ai[k, k]))
    end
    return s
end

# One source's fit behind the convergence check.  The normal equations at the
# current parameters are accumulated once with the model's own `fit_star`
# kernel (`PSF._star_problem`); a source whose predicted step is below
# `fitter.skip_tol` sigma in every free parameter is not fit, and otherwise LM
# starts from those same normal equations instead of accumulating them again.
# `image` must hold this source's own light (its model added back).
function _checked_fit(m0, image, inds, fixed, w_mat, λ0, fitter::SequentialFitter, rbuf)
    prob = PSF._star_problem(m0, image, inds, fixed, w_mat)
    FT = eltype(prob.x0)
    p = length(prob.x0)
    A0 = Matrix{FT}(undef, p, p)
    b0 = Vector{FT}(undef, p)
    cost0 = prob.accum!(A0, b0, view(rbuf, 1:prob.nobs), prob.x0, prob.base_weights)
    if fitter.skip_tol > 0 && _gn_step_sigma(A0, b0) < fitter.skip_tol
        return (; skipped = true, best = m0, res = nothing)
    end
    res = lm_irls(prob; max_iter = fitter.lm_iterations, λ_init = λ0,
        initial_normal = (A0, b0, cost0), fitter.lm...)
    best = PSF.model_from_vector(m0, PSF._free_names_val(m0, fixed), res.minimizer, fixed)
    return (; skipped = false, best, res)
end

"""
    fit_pass(fitter::SequentialFitter, data, w, geom, catalog, psf, plan, o, state; kws...) -> NamedTuple

Sweep the catalog `fitter.sweeps` times, brightest first.  For each source:

1. add its current model back into the residual over its `model_R` box;
2. accumulate its normal equations at the current parameters over the
   `anchor +- R_fit` box with the model's `fit_star` kernel, and skip it
   (subtract the same model again) when the predicted Gauss-Newton step is
   below `fitter.skip_tol` sigma in every free parameter;
3. otherwise fit it alone with up to `fitter.lm_iterations` LM iterations,
   starting from those normal equations, `(y, x, flux)` plus the local pedestal
   when `fitter.fit_bkg`;
4. cap its position change at `o.max_step`, refitting only flux (and pedestal)
   at the capped position when the cap binds;
5. subtract the new model, pedestal-free, over the same `model_R` box.

A fit that returns non-finite parameters leaves the source where it was.  The
fitting box and `model_R` are fixed for the whole pass from the pass-start
catalog, which keeps the diagnostics box equal to the box every sweep fit over.

# Keyword arguments

- `move`: `false` runs one sweep with no fits, rebuilding the model and
  `flux_snr` at the incoming parameters, which is what the post-validation
  rebuild wants.
- `freeze_positions`: fix `(y, x)` at their incoming values in every fit.
- `n_new`: unused; part of the fitter interface.

# Returns

The fields [`AbstractMultipassFitter`](@ref) requires, with `state = nothing`.
`stats` holds `n_lin` (sweeps), `n_trials` (LM iterations summed over every
source fit), `n_accepted` (source fits that lowered their own cost),
`n_star_fits`, `n_skipped` (sources the convergence check did not fit),
`n_capped` (fits whose position step hit `max_step`),
`cost_start`, `cost_end` and `sweep_costs` (the global cost after each sweep);
`cost_end` is `sweep_costs`' last entry.

`stats.fit_timing` breaks the pass's wall time into `setup` (allocation and the
opening render) and `fits` (the whole sweep loop).  These are named substeps
timing only significant work; minor calculations are neglected, so the substeps sum to
slightly less than the driver's `t_fit`.
"""
function fit_pass(fitter::SequentialFitter, data::Vector{FT}, w::Vector{FT}, geom,
                  catalog::Catalog{FT}, psf, plan::FitPlan, o, state; ny::Int, nx::Int,
                  pass::Integer = 1, n_new::Integer = 0, freeze_positions::Bool = false,
                  move::Bool = true, show_trace::Bool = false) where {FT}
    t0 = time()
    n = length(catalog)
    npix = length(data)
    R_fit = o.R_fit
    theta = theta_from_catalog(catalog, plan)
    model_R = _model_radii(psf, o.model_rad, o.model_rad_nsigma, R_fit, o.R_cap, w, catalog.flux)
    Rc_side = 2 * maximum(model_R) + 1
    render_buf = Matrix{FT}(undef, Rc_side, Rc_side)
    render_scratch = PSF._render_scratch(psf, Rc_side, FT)

    model = zeros(FT, npix)
    _render_model!(model, psf, plan.free_names_val, plan.fixed, theta, plan.p, model_R,
        geom.anchor_y, geom.anchor_x, ny, nx, trues(n), render_buf, render_scratch)
    union_pix = _touched_pixels(geom.pixels, npix)
    cost_start = _cost!(model, data, w, union_pix)
    resid = data .- model
    resid_mat = reshape(resid, ny, nx)
    w_mat = reshape(w, ny, nx)

    y, x, flux, bkg = copy(catalog.y), copy(catalog.x), copy(catalog.flux), copy(catalog.bkg)
    lambda = copy(catalog.lambda)
    snr = fill(FT(NaN), n)
    fit_fixed = _seq_fit_fixed(fitter, plan)
    max_step = o.max_step
    n_sweeps = move ? fitter.sweeps : 1
    sweep_costs = FT[]
    n_trials = 0
    n_accepted = 0
    n_star_fits = 0
    n_skipped = 0
    n_capped = 0
    rbuf = Vector{FT}(undef, (2R_fit + 1)^2)
    t_setup = time() - t0

    # One timer around the whole sweep loop, not per source: the per-source render
    # and scatter are negligible beside `_checked_fit`, so splitting them would buy
    # a noise floor and put four `time()` calls in the innermost loop.
    t0 = time()
    for sweep in 1:n_sweeps
        cost_before = isempty(sweep_costs) ? cost_start : last(sweep_costs)
        fits_before, trials_before = n_star_fits, n_trials
        for j in sortperm(flux; rev = true)
            ay, ax, R = geom.anchor_y[j], geom.anchor_x[j], model_R[j]
            _render_unit!(render_buf, psf, plan, y[j], x[j], ay, ax, R, render_scratch)
            _scatter_source!(resid, model, render_buf, -flux[j], w, ay, ax, R, R_fit, ny, nx)

            if move
                inds = CartesianIndices(_clamp_inds((ay - R_fit):(ay + R_fit), (ax - R_fit):(ax + R_fit), resid_mat))
                y0, x0 = y[j], x[j]
                m0 = ConstructionBase.setproperties(psf,
                    merge((; y = y0, x = x0, flux = flux[j], bkg = bkg[j]), fit_fixed))
                # Damping memory: with only a few iterations per fit, a source
                # whose Gauss-Newton step overshoots would spend every fit's whole
                # budget climbing back up from `lambda_init` and never accept a
                # step.  Start from where its last fit ended instead, floored at
                # `lambda_init` so a well-behaved source is not over-damped.
                λ0 = isnan(lambda[j]) ? FT(fitter.lambda_init) : max(lambda[j], FT(fitter.lambda_init))
                out = freeze_positions ?
                    _checked_fit(m0, resid_mat, inds, merge(fit_fixed, (; y = y0, x = x0)), w_mat, λ0, fitter, rbuf) :
                    _checked_fit(m0, resid_mat, inds, fit_fixed, w_mat, λ0, fitter, rbuf)
                if out.skipped
                    # `render_buf` still holds this source's unit render, so the
                    # scatter below subtracts exactly what was added back.
                    n_skipped += 1
                else
                best, res = out.best, out.res
                n_star_fits += 1
                n_trials += res.iterations
                n_accepted += res.minimum < res.cost_init
                dy, dx = FT(best.y) - y0, FT(best.x) - x0
                step = hypot(dy, dx)
                if step > max_step
                    # A step this large is a source wandering onto a neighbor's
                    # light, not converging; keep the direction, cap the length,
                    # and re-solve the linear parameters where it now sits.
                    s = max_step / step
                    yc, xc = y0 + s * dy, x0 + s * dx
                    best, res = PSF.fit_star(ConstructionBase.setproperties(best, (; y = yc, x = xc)),
                        resid_mat, inds; fixed = merge(fit_fixed, (; y = yc, x = xc)),
                        inv_var = w_mat, max_iter = fitter.lm_iterations,
                        λ_init = FT(fitter.lambda_init), fitter.lm...)
                    n_capped += 1
                    n_trials += res.iterations
                end
                if isfinite(best.y) && isfinite(best.x) && isfinite(best.flux) && isfinite(best.bkg)
                    y[j], x[j], flux[j] = best.y, best.x, best.flux
                    fitter.fit_bkg && (bkg[j] = best.bkg)
                end
                lambda[j] = res.λ_final
                _render_unit!(render_buf, psf, plan, y[j], x[j], ay, ax, R, render_scratch)
                end
            end

            hff = _scatter_source!(resid, model, render_buf, flux[j], w, ay, ax, R, R_fit, ny, nx)
            snr[j] = flux[j] * sqrt(hff)
        end
        push!(sweep_costs, _cost!(model, data, w, union_pix))
        show_trace && move && _trace_sweep(pass, sweep, cost_before, last(sweep_costs),
            n_star_fits - fits_before, n_trials - trials_before)
    end

    t_fits = time() - t0

    new_catalog = Catalog{FT}(y, x, flux, snr, copy(catalog.pass), bkg, lambda)
    theta = theta_from_catalog(new_catalog, plan)
    # The last sweep already evaluated the cost at this model, over the same
    # `union_pix` and with the same function as `cost_start`.
    stats = (; n_lin = move ? n_sweeps : 0, n_trials, n_accepted, cost_start,
               cost_end = last(sweep_costs), n_star_fits, n_skipped, n_capped, sweep_costs,
               fit_timing = (; setup = t_setup, fits = t_fits))
    return (; catalog = new_catalog, theta, model = reshape(model, ny, nx), model_R, geom, data, w,
              render_buf, render_scratch, state = nothing, stats)
end

function _trace_sweep(pass, sweep, cost, cost_after, n_fits, n_iter)
    pct = cost > 0 ? 100 * (cost_after - cost) / cost : zero(cost)
    println("  sweep ", lpad(sweep, 2), " | cost ", @sprintf("%.4e", cost), " -> ",
        @sprintf("%.4e", cost_after), " (", @sprintf("%+.2f%%", pct), ")",
        " | ", n_fits, " fits, ", n_iter, " LM iterations")
    return nothing
end

"""
    source_errors(fitter::SequentialFitter, fit, psf, plan, cov_est) -> NamedTuple

Per-source errors from each source's own LM normal matrix, evaluated at the
final parameters against the final residual of every other source: one
zero-iteration [`fit_star`](@ref CrowdPhot.PSF.fit_star) per source, which
builds the normal matrix at the incoming parameters and inverts it without
moving anything.  `cov_est` rescales by that source's own `cost / dof` where it
rescales at all.  IRLS reweighting from the sweeps is not reapplied.
"""
function source_errors(fitter::SequentialFitter, fit, psf, plan::FitPlan, cov_est)
    catalog = fit.catalog
    FT = eltype(catalog.y)
    n = length(catalog)
    ny, nx = size(fit.model)
    R_fit = fit.geom.dy_off[end]
    model = copy(vec(fit.model))
    resid = fit.data .- model
    resid_mat = reshape(resid, ny, nx)
    w_mat = reshape(fit.w, ny, nx)
    fit_fixed = _seq_fit_fixed(fitter, plan)
    free_names = PSF.free_params(psf, fit_fixed)[1]
    k_y, k_x = findfirst(==(:y), free_names), findfirst(==(:x), free_names)
    k_f, k_b = findfirst(==(:flux), free_names), findfirst(==(:bkg), free_names)
    y_err, x_err, flux_err, bkg_err = (zeros(FT, n) for _ in 1:4)
    buf, scratch = fit.render_buf, fit.render_scratch
    for j in 1:n
        ay, ax, R = fit.geom.anchor_y[j], fit.geom.anchor_x[j], fit.model_R[j]
        _render_unit!(buf, psf, plan, catalog.y[j], catalog.x[j], ay, ax, R, scratch)
        _scatter_source!(resid, model, buf, -catalog.flux[j], fit.w, ay, ax, R, R_fit, ny, nx)
        inds = CartesianIndices(_clamp_inds((ay - R_fit):(ay + R_fit), (ax - R_fit):(ax + R_fit), resid_mat))
        m0 = ConstructionBase.setproperties(psf, merge((; y = catalog.y[j], x = catalog.x[j],
            flux = catalog.flux[j], bkg = catalog.bkg[j]), fit_fixed))
        _, res = PSF.fit_star(m0, resid_mat, inds; fixed = fit_fixed, inv_var = w_mat,
            max_iter = 0, covariance_estimator = cov_est)
        cov = res.cov
        k_y === nothing || (y_err[j] = sqrt(max(zero(FT), cov[k_y, k_y])))
        k_x === nothing || (x_err[j] = sqrt(max(zero(FT), cov[k_x, k_x])))
        flux_err[j] = sqrt(max(zero(FT), cov[k_f, k_f]))
        k_b === nothing || (bkg_err[j] = sqrt(max(zero(FT), cov[k_b, k_b])))
        _scatter_source!(resid, model, buf, catalog.flux[j], fit.w, ay, ax, R, R_fit, ny, nx)
    end
    return (; y_err, x_err, flux_err, bkg_err)
end


# ==============================================================================
# Public entry point
# ==============================================================================

"""
    fit_all_stars_multipass(image, psf, [sources], fit_rad; kws...)

Iterated background estimation, source detection, and sequential PSF-fitting
photometry.

$(_MULTIPASS_DOC_LOOP)
The fit is DOLPHOT-style: within a pass the catalog is swept brightest to
faintest, and each source is fit alone with a few damped Levenberg-Marquardt
iterations against the residual image, from which every other source's current
model has been subtracted.  Its own model is added back before its fit and
subtracted again after, so fainter neighbors are always fit against the latest
models of the brighter ones.  [`fit_all_stars_simultaneous_multipass`](@ref)
runs the same pipeline with a simultaneous whole-catalog fitter instead; the two
share every stage except the fit and the error computation.

$(_MULTIPASS_DOC_ARGUMENTS)
# Keyword arguments

$(_MULTIPASS_DOC_PIPELINE)
## Fitting

$(_MULTIPASS_DOC_FIT_COMMON)
## Sequential fitter

- `fixed::NamedTuple = (;)`: parameters frozen for all sources.  Shape
  parameters must be fixed; `(y, x, flux)` are free unless listed.  Omitting
  `bkg` fits a **local pedestal** per source over its fitting box, on top of the
  global background model, as the per-star `fit_star` fit always has; passing
  `bkg = 0` disables it.  The pedestal absorbs local background error the mesh
  cannot follow, at the cost of one more free parameter per source.  It is
  never rendered into the model image, so it does not leak into neighbors' fits
  or into detection, and the returned `residual` excludes it.
- `sweeps_per_pass::Integer = 1`: sweeps through the catalog per detection
  pass.  Each sweep fits every source once.
- `lm_iterations::Integer = 5`: `max_iter` of each per-source LM fit, counting
  rejected steps.  It is a cap, not a cost: once seeded, most fits stop on the
  `fit_star` tolerances after one or two iterations, so the cap only binds for
  sources still far from their solution -- typically a seed on a neighbor's
  wing -- and a larger cap settles those in fewer sweeps.  On a crowded test
  field, caps of 1 to 10 took the same time end to end.
- `skip_tol::Real = 0.1`: convergence check before each source fit.  The
  source's normal equations are accumulated at its current parameters against
  the current residual, and the source is not refit on this sweep when the
  predicted Gauss-Newton step is below `skip_tol` times its own 1-sigma error in
  every free parameter.  A neighbor that moves, a source detected nearby, or a
  background update all change that residual, so a skipped source is
  reconsidered automatically on the next sweep or pass.  A source that is fit
  starts its LM solve from these normal equations, so the check costs one extra
  accumulation only for the sources it skips.  On a 250k-source crowded field
  most fits after the first few passes change their source by less than 0.1
  sigma, and skipping them cut wall time by a quarter at one sweep per pass.
  `0` fits every source on every sweep.
- `max_step::Real = 1.0`: cap on a source's position change within one fit, in
  pixels.  When it binds the step is shortened along its own direction and the
  linear parameters are refit at the capped position.
- `λ_init::Real = 1.0e-4`: floor of each fit's starting damping factor.  Each
  source remembers the damping its last fit ended at and its next fit starts
  from the larger of that and `λ_init`.  Without the memory a source whose
  Gauss-Newton step overshoots -- typically a seed far from its true flux next
  to a brighter neighbor -- spends every fit's `lm_iterations` budget on
  rejected steps while the damping climbs back up from `λ_init`, and never
  moves.
- `λ_up::Real = 10.0`, `λ_down::Real = 10.0`, `λ_min::Real = 1.0e-12`,
  `λ_max::Real = 1.0e12`, `damping = MarquardtDamping()`, `x_tol::Real = 1.0e-8`,
  `f_tol::Real = 1.0e-4`, `g_tol::Real = 1.0e-4`, `reweight = nothing`,
  `scale_estimator = nothing`, `weight_reset_tol::Real = 0.1`: forwarded to every
  per-source [`fit_star`](@ref CrowdPhot.PSF.fit_star) call; see there.

In `pass_history`, `n_lin` counts sweeps, `n_trials` LM iterations summed over
all source fits, and `n_accepted` source fits that lowered their own cost;
`n_star_fits`, `n_skipped` (sources the convergence check did not refit),
`n_capped` and `sweep_costs` (the global cost after each sweep) are specific to
this fitter.

Errors come from each source's own LM normal matrix at its final parameters,
evaluated against the final residual of all other sources; like the
simultaneous fitter's they ignore covariance with blended neighbors.

$(_MULTIPASS_DOC_RETURNS)"""
function fit_all_stars_multipass(
        image::AbstractMatrix,
        psf::AbstractPSFModel,
        sources,
        fit_rad::Real;
        fixed::NamedTuple = (;),
        sweeps_per_pass::Integer = 1,
        lm_iterations::Integer = 5,
        λ_init::Real = 1.0e-4,
        λ_up::Real = 10.0,
        λ_down::Real = 10.0,
        λ_min::Real = 1.0e-12,
        λ_max::Real = 1.0e12,
        damping::AbstractLMDamping = MarquardtDamping(),
        x_tol::Real = 1.0e-8,
        f_tol::Real = 1.0e-4,
        g_tol::Real = 1.0e-4,
        reweight::Union{Nothing, LossFunctions.SupervisedLoss} = nothing,
        scale_estimator::Union{Nothing, AbstractScaleEstimator} = nothing,
        weight_reset_tol::Real = 0.1,
        skip_tol::Real = 0.1,
        kws...,
    )
    sweeps_per_pass > 0 || throw(ArgumentError("sweeps_per_pass must be positive"))
    lm_iterations > 0 || throw(ArgumentError("lm_iterations must be positive"))
    fit_bkg = !haskey(fixed, :bkg) && haskey(ConstructionBase.getproperties(psf), :bkg)
    λ_init > 0 || throw(ArgumentError("λ_init must be positive"))
    skip_tol >= 0 || throw(ArgumentError("skip_tol must be non-negative"))
    lm = (; λ_up, λ_down, λ_min, λ_max, damping, x_tol, f_tol, g_tol, reweight,
            scale_estimator, weight_reset_tol)
    fitter = SequentialFitter(Int(sweeps_per_pass), Int(lm_iterations), fit_bkg, Float64(λ_init), Float64(skip_tol), lm)
    return _fit_all_stars_multipass(fitter, image, psf, sources, fit_rad; fixed, kws...)
end

"""
    fit_all_stars_multipass(image, psf, fit_rad; kws...)

Convenience method starting from an empty catalog: the first detection pass
builds it from scratch.  Equivalent to passing `sources = nothing`.
"""
fit_all_stars_multipass(image::AbstractMatrix, psf::AbstractPSFModel, fit_rad::Real; kws...) =
    fit_all_stars_multipass(image, psf, nothing, fit_rad; kws...)
