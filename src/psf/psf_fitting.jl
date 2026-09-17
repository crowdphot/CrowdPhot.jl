"""
    free_params(model::AbstractPSFModel, 
        fixed::NamedTuple=NamedTuple()) -> (free_names, free_idx, x0)

Return the names of the free (non-fixed) parameters, their indices, and their initial values as
a `Vector`. `fixed` is a `NamedTuple` whose keys name the fields to freeze.
"""
function free_params(model::AbstractPSFModel{T}, fixed::NamedTuple = NamedTuple()) where {T}
    all_props = ConstructionBase.getproperties(model)
    all_names = keys(all_props)
    # `structdiff` maintains inferability as it computes the free names from the
    # two NamedTuple *types*, so `free_names` and `free_idx` are compile-time constants
    # and `_free_names_val` below costs nothing.  A generator filtering on `haskey`
    # reads more simply but infers as `Tuple{Vararg{Symbol}}`, which makes every
    # downstream `Val(free_names)` a runtime value (521 ns and 23 allocations per call).
    free_names = keys(Base.structdiff(all_props, fixed))
    free_idx = map(k -> findfirst(isequal(k), all_names)::Int, free_names)
    x0 = T[all_props[k] for k in free_names]
    return free_names, free_idx, x0
end

"""
    _free_names_val(model, fixed) -> Val{free_names}

The free-parameter names of `model` under `fixed`, `Val`-wrapped for
[`model_from_vector`](@ref).

Call this whenever you drive [`lm_irls`](@ref) on a `_star_problem`
yourself and need the fitted model back, as in [`fit_star`](@ref): 
`lm_irls` returns a parameter vector, and turning that vector back
into a model requires the names it was built from.

This exists so that `_star_problem` can return a plain `LMProblem` instead of
pairing it with the names.  That is only viable because the result is a
function of the two argument *types*, so it constant-folds to zero allocations
and any caller can recompute it at no cost.  The folding depends on
`free_params` using `Base.structdiff`, which is why both live here
and why `psf_fitting_tests.jl` asserts `@inferred` on this call: if that
inferability is ever lost, this function silently becomes a per-fit allocation
in the sequential fitter's inner loop.
"""
_free_names_val(model, fixed::NamedTuple) =
    Val(keys(Base.structdiff(ConstructionBase.getproperties(model), fixed)))

"""
    model_from_vector(model, ::Val{names}, x, fixed::NamedTuple) -> updated model

Reconstruct the model from an optimizer vector `x`, merging in the fixed
parameters. `names` is a `Val`-wrapped tuple of the free parameter names
for type-stability, and `fixed` is a `NamedTuple` of the fixed parameters.
"""
function model_from_vector(model, ::Val{names}, x::AbstractVector, fixed::NamedTuple) where {names}
    updates = NamedTuple{names}(ntuple(i -> x[i], Val(length(names))))
    return ConstructionBase.setproperties(model, merge(updates, fixed))
end

function _has_hessian(model::AbstractPSFModel)
    return hasmethod(evaluate_fgh, Tuple{typeof(model), Real, Real})
end

function _has_deriv(model::AbstractPSFModel)
    return hasmethod(evaluate_fg, Tuple{typeof(model), Real, Real})
end

function _prepare_fit_star_inputs(model::AbstractPSFModel{T}, image::AbstractMatrix, inds, fixed::NamedTuple, inv_var) where {T}
    # Validate model capabilities and weight-map shape before allocating LM buffers.
    if !isnothing(inv_var) && size(inv_var) != size(image)
        throw(ArgumentError("`inv_var` must be the same size as `image`"))
    end
    if !(_has_deriv(model))
        throw(
            ArgumentError(
                "model does not implement `evaluate_fg`; " *
                    "Levenberg-Marquardt requires gradient"
            )
        )
    end

    # Convert fit indices and determine which model parameters remain free.
    # CartesianIndices(::CartesianIndices) normalizes offset ranges to 1-based,
    # which would break any fit using pixel indices that don't start at 1.
    # Preserve CartesianIndices as-is; convert tuples / other index types.
    fit_inds = inds isa CartesianIndices ? inds : CartesianIndices(inds)
    free_names, free_idx, x0 = free_params(model, fixed)
    n = length(x0)
    n > 0 || throw(ArgumentError("all model parameters are fixed; nothing to fit"))

    FT = float(T)

    # When inv_var is provided, zero or non-finite values act as a pixel mask.
    # Weights for masked pixels are set to zero so they contribute nothing to
    # the normal equations or the cost, preserving CartesianIndices iteration.
    if isnothing(inv_var)
        base_weights = nothing
        dof = length(fit_inds) - n
    else
        n_valid = 0
        weights = Vector{FT}(undef, length(fit_inds))
        base_k = 0
        @inbounds for idx in fit_inds
            base_k += 1
            w = inv_var[idx]
            wgt = if isfinite(w) && w > 0
                n_valid += 1
                FT(w)
            else
                zero(FT)
            end
            weights[base_k] = wgt
        end
        n_valid > n || throw(
            ArgumentError(
                "too few valid inv_var pixels ($n_valid) " *
                "for $n free parameters"
            )
        )
        base_weights = weights
        dof = n_valid - n
    end

    return (
        inds = fit_inds,
        free_names = free_names,
        free_idx = free_idx,
        x0 = x0,
        free_names_val = Val(free_names),
        FT = FT,
        base_weights = base_weights,
    )
end

"""
    fit_star(model::AbstractPSFModel, image, inds=axes(image);
        fixed=(;), inv_var=nothing,
        damping::AbstractLMDamping=MarquardtDamping(),
        max_iter=200, x_tol=1e-8, f_tol=1e-4, g_tol=1e-4,
        show_trace=false, 
        reweight=nothing,
        scale_estimator=nothing,
        weight_reset_tol=0.1,
        covariance_estimator=nothing)

Fit the free parameters of `model` to `image[inds]` under weighted L2 loss
using the Levenberg-Marquardt algorithm.  The model must implement
`evaluate_fg`.

`fixed` is a `NamedTuple` of field-name → value pairs whose parameters are
frozen during the fit.  All other fields of `model` are free.

Inverse variance weights can be passed via `inv_var`; it must be the same size
as `image`.

Returns `(best_model, result::LMResult)`.

# Algorithm

Minimises ``C(\\mathbf{x}) = \\sum_i w_i [f_i(\\mathbf{x}) - d_i]^2`` using
the Gauss-Newton approximation to the Hessian augmented by a diagonal damping
term:

```math
(J^\\top W J + \\lambda D)\\,\\delta = -J^\\top W r
```

where ``J`` is the Jacobian assembled from `evaluate_fg` gradients (restricted
to the free parameters), ``W = \\mathrm{diag}(w_i)``, and
``r_i = f_i(\\mathbf{x}) - d_i``.  If the candidate step reduces the cost it
is accepted and ``\\lambda`` is decreased; otherwise it is rejected and
``\\lambda`` is increased.

# Iteratively Reweighted Least Squares

Pass `reweight` as a `LossFunctions.SupervisedLoss` (e.g. `LossFunctions.HuberLoss()`,
[`CrowdPhot.TukeyLoss()`](@ref TukeyLoss)) to enable iteratively reweighted least squares (IRLS). After
each accepted LM step, the residuals are used to recompute pixel weights via
``w_i^{\\text{final}} = w_i^{\\text{base}} \\cdot w(r_i/\\sigma)`` where
``w(r) = \\psi(r)/r`` is derived from the loss function's influence function
and ``\\sigma`` is a robust scale estimate. This provides automatic outlier
rejection — useful for cosmic rays, bad pixels, and satellite trails in
astronomical images.

The scale ``\\sigma`` is estimated by `scale_estimator`. If not provided,
it defaults to [`FixedScale`](@ref) inferred from `inv_var` if supplied,
otherwise [`MADScale()`](@ref MADScale) is used. Available scale estimators:
- [`MADScale()`](@ref MADScale) — median absolute deviation (robust, default without `inv_var`)
- [`FixedScale(σ)`](@ref FixedScale) — fixed user-provided scale
- [`MScale(; δ=0.5)`](@ref MScale) — iterative M-scale (most robust, highest cost)

Recommended loss functions for IRLS:
- `LossFunctions.HuberLoss(c)` — soft downweighting of outliers, suitable for mildly
  contaminated data or crowded-field photometry
- [`CrowdPhot.TukeyLoss(; c=4.685)`](@ref TukeyLoss) — complete rejection of extreme
  outliers, ideal for cosmic rays and bad pixels in space-based imaging

# Covariance Estimation
The covariance of the fitted parameters is estimated from the Gauss-Newton Hessian
approximation to the Hessian at the solution. For known good input weights `inv_var`,
the covariance is simply the inverse of this Hessian approximation. Use 
[`covariance_estimator = KnownWeightsCovarianceEstimator()`](@ref KnownWeightsCovarianceEstimator) for this behavior.
However, when IRLS reweighting is used and the weights are estimated from the data,
the covariance is inflated by the reduced cost per degree of freedom to account for
uncertainty in the weights. In this case, use [`ReweightedCovarianceEstimator`](@ref ReweightedCovarianceEstimator).

# Damping Strategies
- [`MarquardtDamping`](@ref CrowdPhot.MarquardtDamping): diagonal entries are scaled by `max(A[i, i], min_diagonal)` to
  prevent small curvature directions from being under-damped
- [`LevenbergDamping`](@ref CrowdPhot.LevenbergDamping): uniform damping with no scaling
- [`NoDamping`](@ref CrowdPhot.NoDamping): no damping; equivalent to Gauss-Newton (not recommended)

# Keyword arguments

- `fixed`: `NamedTuple` of frozen parameter name → value pairs
- `inv_var`: inverse-variance weights, same shape as `image`
- `reweight`: a `LossFunctions.SupervisedLoss` for IRLS reweighting,
  or `nothing` (default) for standard L2 fitting
- `scale_estimator::AbstractScaleEstimator`: scale estimator for IRLS;
  defaults to [`FixedScale`](@ref) (from `inv_var`) or [`MADScale`](@ref) (see above)
- `weight_reset_tol`: reset `λ` to `λ_init` after an IRLS weight update only
  when the symmetric relative L2 norm of the weight change is at least this
  threshold (default `0.1`)
- `covariance_estimator::AbstractCovarianceEstimator`: method for estimating
  the covariance matrix of the fitted parameters; defaults to
  [`ReweightedCovarianceEstimator`](@ref) when IRLS is used and
  [`KnownWeightsCovarianceEstimator`](@ref) otherwise
- `λ_init`: initial damping parameter (default `1e-4`)
- `λ_up`: factor by which `λ` is multiplied on rejection (default `10`)
- `λ_down`: factor by which `λ` is divided on acceptance (default `10`)
- `λ_min`, `λ_max`: lower/upper bounds on the damping parameter
- `damping::AbstractLMDamping`: specifies the damping strategy
  ([`MarquardtDamping`](@ref CrowdPhot.MarquardtDamping),
  [`LevenbergDamping`](@ref CrowdPhot.LevenbergDamping), or
  [`NoDamping`](@ref CrowdPhot.NoDamping))
- `max_iter`: maximum number of LM iterations (default `200`)
- `x_tol`: rescaled parameter-step convergence criterion (default `1e-8`):
  `norm(D .* δ) ≤ x_tol * (norm(D .* x) + x_tol)`,
  where `D[i] = max(sqrt(A[i,i]), D_prev[i])`
  is a monotonically-growing per-parameter scale (analogous to MINPACK's
  `diag` under `mode=1`) that prevents a large-magnitude parameter (e.g.
  flux, in counts) from dominating the norm over a small-magnitude one
  (e.g. position, in pixels). On a rejected step this additionally requires
  `f_tol`'s criterion to also hold in order to consider the fit converged,
  to avoid falsely reporting convergence from a step that is tiny only due
  to heavy damping.
- `f_tol`: cost-decrease convergence criterion (default `1e-4`), mirroring MINPACK's
  three-part `ftol` test. The actual cost decrease `ΔC` must be small,
  the LM quadratic model's *predicted* decrease `prered` must independently
  be small, and their ratio must not indicate that the model is a poor
  local approximation:
  `abs(ΔC) <= f_tol * (abs(C) + f_tol) &&
   prered <= f_tol * (abs(C) + f_tol) &&
   0.5 * ΔC / prered <= 1`. 
   Checked every iteration, not only on accepted steps.
- `g_tol`: dimensionless per-parameter gradient-cosine convergence
  criterion (default `1e-4`), bounded by Cauchy-Schwarz in `[-1, 1]`:
  `max(abs.(b) ./ sqrt.(diag(A) .* C)) <= g_tol`.
- `show_trace`: print per-iteration statistics to stdout (default `false`)
"""
function fit_star(
        model::AbstractPSFModel{T},
        image::AbstractMatrix,
        inds = axes(image);
        fixed::NamedTuple = (;),
        inv_var = nothing,
        λ_init::Real = 1.0e-4,
        λ_up::Real = 10.0,
        λ_down::Real = 10.0,
        λ_min::Real = 1.0e-12,
        λ_max::Real = 1.0e12,
        damping::AbstractLMDamping = MarquardtDamping(),
        max_iter::Integer = 200,
        x_tol::Real = 1.0e-8,
        f_tol::Real = 1.0e-4,
        g_tol::Real = 1.0e-4,
        show_trace::Bool = false,
        reweight::Union{Nothing, LossFunctions.SupervisedLoss} = nothing,
        scale_estimator::Union{Nothing, AbstractScaleEstimator} = nothing,
        weight_reset_tol::Real = 0.1,
        covariance_estimator::Union{Nothing, AbstractCovarianceEstimator} = nothing
    ) where {T}
    problem = _star_problem(model, image, inds, fixed, inv_var)
    result = lm_irls(
        problem;
        λ_init,
        λ_up,
        λ_down,
        λ_min,
        λ_max,
        damping,
        max_iter,
        x_tol,
        f_tol,
        g_tol,
        show_trace,
        reweight,
        scale_estimator,
        weight_reset_tol,
        covariance_estimator
    )
    best_model = model_from_vector(model, _free_names_val(model, fixed), result.minimizer, fixed)
    return best_model, result
end

"""
    _star_problem(model, image, inds, fixed, inv_var) -> LMProblem

The normal-equation problem [`fit_star`](@ref) solves for `model`: validated fit
indices and weights, the free-parameter bookkeeping, and an `accum!` closure using
the fastest accumulator available for `model`'s type (one method per specialized
model below; this one streams `evaluate_fg`).

Split out of `fit_star` so a caller can evaluate the normal equations at the
starting parameters itself -- to test for convergence -- and pass them to
[`lm_irls`](@ref) as `initial_normal` instead of accumulating them twice.  Such a
caller rebuilds the fitted model from `result.minimizer` with
[`model_from_vector`](@ref) and [`_free_names_val`](@ref).
"""
function _star_problem(model::AbstractPSFModel{T}, image::AbstractMatrix, inds, fixed::NamedTuple, inv_var) where {T}
    # Run shared validation and parameter bookkeeping for this fit.
    prepared = _prepare_fit_star_inputs(model, image, inds, fixed, inv_var)
    inds, _, free_idx, x0, free_names_val, FT, base_weights = prepared

    # Custom accumulator for LM that streams over pixels, evaluates the model and Jacobian via `evaluate_fg`,
    # and fills the normal equations without materializing the full Jacobian. This is the core of the algorithm; all model-specific work lives here.
    function accum!(A::AbstractMatrix{FT}, b::AbstractVector{FT}, residuals::AbstractVector{FT}, x::AbstractVector{FT}, weights) where {FT}
        nparams = length(free_idx)
        @assert size(A, 1) == size(A, 2) == length(b) == nparams
        @assert length(residuals) == length(inds)
        fill!(A, zero(FT))
        fill!(b, zero(FT))
        cost = zero(FT)
        m = model_from_vector(model, free_names_val, x, fixed)
        use_weights = !isnothing(weights)
        obs_k = 0
        @inbounds for idx in inds
            obs_k += 1
            w = use_weights ? FT(weights[obs_k]) : one(FT)
            f_val, g_full = evaluate_fg(m, idx)
            r = FT(f_val) - FT(image[idx])
            residuals[obs_k] = r
            wr = w * r
            cost = muladd(wr, r, cost)
            @inbounds for j in 1:nparams
                gj = FT(g_full[free_idx[j]])
                b[j] = muladd(wr, gj, b[j])
                for i in 1:nparams
                    A[i, j] = muladd(w * FT(g_full[free_idx[i]]), gj, A[i, j])
                end
            end
        end
        return cost
    end

    return LMProblem(Vector{FT}(x0), length(inds), accum!, base_weights)
end

################################################################################
# Custom `fit_star` methods for specific PSF models with optimized accumulators.
################################################################################

#-------------------------------------------------------------------------------
# Separable circular Gaussians: `CircularGaussianPSF` and `CircularGaussianPRF`
# share one set of per-axis buffers, one accumulator, one `_star_problem` and
# one `render!`.  Everything model-specific lives in `_separable_params` and
# `_separable_axis!`.
#-------------------------------------------------------------------------------

"""
    _separable_axis_buffers(ny, nx, FT) -> NamedTuple

Per-axis scratch for the separable circular Gaussians (`CircularGaussianPSF`,
`CircularGaussianPRF`): `Ey`, `Gy`, `Hy` of length `ny` and `Ex`, `Gx`, `Hx` of
length `nx` (see [`_separable_axis!`](@ref)), plus `Ee`, `Ge` of length
`max(ny, nx) + 1`, the pixel-edge scratch the `CircularGaussianPRF` fill reuses
for both axes.  Also used by their `_render_scratch` and `_fill_scratch`.
"""
_separable_axis_buffers(ny::Int, nx::Int, ::Type{FT}) where {FT} =
    (; Ey = Vector{FT}(undef, ny), Gy = Vector{FT}(undef, ny), Hy = Vector{FT}(undef, ny),
       Ex = Vector{FT}(undef, nx), Gx = Vector{FT}(undef, nx), Hx = Vector{FT}(undef, nx),
       Ee = Vector{FT}(undef, max(ny, nx) + 1), Ge = Vector{FT}(undef, max(ny, nx) + 1))

"""
    _separable_params(model) -> (; αy, αx, amp, cH, cE, cF)

Scalars that turn the per-axis factors of [`_separable_axis!`](@ref) into a
model's value and gradient.  With `E = Ex[j] Ey[i]` (and `G`, `H` likewise),

- value `= amp E + bkg`,
- `∂/∂y = amp αy Ex Gy`, `∂/∂x = amp αx Gx Ey`,
- `∂/∂fwhm = cH (Hx Ey + Ex Hy) + cE E`,
- `∂/∂flux = cF E`, `∂/∂bkg = 1`.

`αy` and `αx` scale pixel offsets into the axis buffers and are the same
`sqrt(-GAUSS_PRE)/fwhm = 2 sqrt(ln 2)/fwhm` on both axes for circular models.

!!! note
    These scalars assume the model's five parameters are `(y, x, fwhm, flux,
    bkg)` in that order.  A separable model with per-axis widths would need
    per-axis width constants and a different gradient assembly.
"""
function _separable_params(m::CircularGaussianPRF)
    fwhm = float(m.fwhm)
    α = sqrt(-oftype(fwhm, GAUSS_PRE)) / fwhm
    amp = float(m.flux) / 4
    return (; αy = α, αx = α, amp, cH = amp / fwhm, cE = zero(amp), cF = one(amp) / 4)
end

function _separable_params(m::CircularGaussianPSF)
    fwhm = float(m.fwhm)
    α = sqrt(-oftype(fwhm, GAUSS_PRE)) / fwhm
    # Same normalization as the full-parameter path: norm = -(π fwhm² / GAUSS_PRE).
    norm = -(oftype(fwhm, π) * fwhm^2 / oftype(fwhm, GAUSS_PRE))
    amp = float(m.flux) / norm
    return (; αy = α, αx = α, amp, cH = 2 * amp / fwhm, cE = -2 * amp / fwhm, cF = one(amp) / norm)
end

"""
    _separable_axis!(E, G, H, Ee, Ge, model, α, μ, r)

One axis of a separable circular Gaussian, for pixel centers `c` in `r` at
scaled offsets `u = α (c - μ)`.  `Ee` and `Ge` are edge scratch of length
`length(r) + 1`, used by the `CircularGaussianPRF` method and ignored by the
`CircularGaussianPSF` one.

`CircularGaussianPSF` samples the profile at pixel centers:

- `E[k] = exp(-u²)`, `G[k] = 2 u E[k]`, `H[k] = u² E[k]`.

`CircularGaussianPRF` integrates it over the pixel, with edges
`u± = α (c - μ ± 1/2)` and `g(u) = 2 exp(-u²) / √π`:

- `E[k] = erf(u+) - erf(u-)`, `G[k] = g(u-) - g(u+)`, `H[k] = g(u-) u- - g(u+) u+`.

Adjacent pixels share an edge, so the PRF evaluates each edge once: `length(r) +
1` erf/exp pairs per axis instead of four of each per pixel.  It does so in two
`@turbo` passes (edges, then differences) rather than one loop carrying the
previous edge, because the carry would serialize the `erf` calls; the extra pass
over `Ee`/`Ge` is cheap next to vectorizing them.  In both cases the value and
the full five-parameter gradient are products of one x factor and one y factor;
see [`_separable_params`](@ref).
"""
function _separable_axis!(E::AbstractVector{FT}, G::AbstractVector{FT}, H::AbstractVector{FT},
        _Ee, _Ge, ::CircularGaussianPSF, α, μ, r::AbstractUnitRange) where {FT}
    α, μ = FT(α), FT(μ)
    # `@turbo` vectorizes the exponentials (4x over the scalar loop at 11 pixels,
    # and the whole point of factorizing: the per-pixel kernel it replaces got
    # vectorized `exp` too).  The offset is built affinely from `k` because
    # `@turbo` needs affine indices.
    d0 = FT(first(r)) - μ
    LV.@turbo for k in 1:length(r)
        u = α * (d0 + (k - 1))
        e = exp(-u^2)
        E[k] = e
        G[k] = 2 * u * e
        H[k] = u^2 * e
    end
    return nothing
end

function _separable_axis!(E::AbstractVector{FT}, G::AbstractVector{FT}, H::AbstractVector{FT},
        Ee::AbstractVector{FT}, Ge::AbstractVector{FT},
        ::CircularGaussianPRF, α, μ, r::AbstractUnitRange) where {FT}
    tsp = 2 / sqrt(FT(π))
    α, μ = FT(α), FT(μ)
    n = length(r)
    # Offset of the first pixel's lower edge; edge k is `α (d0 + (k - 1))`.
    d0 = FT(first(r)) - μ - FT(0.5)
    LV.@turbo for k in 1:(n + 1)
        u = α * (d0 + (k - 1))
        Ee[k] = erf(u)
        Ge[k] = tsp * exp(-u^2)
    end
    LV.@turbo for k in 1:n
        um = α * (d0 + (k - 1))
        up = um + α
        gm, gp = Ge[k], Ge[k + 1]
        E[k] = Ee[k + 1] - Ee[k]
        G[k] = gm - gp
        H[k] = gm * um - gp * up
    end
    return nothing
end

"""
    _separable_axes!(axis, model, yr, xr) -> params

Fill both axes of `axis` (from [`_separable_axis_buffers`](@ref)) for `model`
over rows `yr` and columns `xr` and return its [`_separable_params`](@ref).
The parameters carry the `α` the fill itself needs, so returning them keeps
each call site to one line and evaluates the constants once.
"""
function _separable_axes!(axis::NamedTuple, model::Union{CircularGaussianPSF, CircularGaussianPRF},
        yr::AbstractUnitRange, xr::AbstractUnitRange)
    p = _separable_params(model)
    _separable_axis!(axis.Ey, axis.Gy, axis.Hy, axis.Ee, axis.Ge, model, p.αy, model.y, yr)
    _separable_axis!(axis.Ex, axis.Gx, axis.Hx, axis.Ee, axis.Ge, model, p.αx, model.x, xr)
    return p
end

"""
    _accum_separable_gaussian!(A, b, residuals, image, inds, model, free_idx, weights, [axis])

Normal-equation accumulator shared by `CircularGaussianPSF` and
`CircularGaussianPRF`.  The per-axis factors of [`_separable_axis!`](@ref) are
computed once per call (`axis` from [`_separable_axis_buffers`](@ref),
allocated when omitted), then every pixel's value and five-parameter gradient
is a product of one x and one y factor.  Pixels are visited in `inds` order, as
the residuals and weights expect.
"""
function _accum_separable_gaussian!(
        A::AbstractMatrix{FT},
        b::AbstractVector{FT},
        residuals::AbstractVector{FT},
        image::AbstractMatrix,
        inds::CartesianIndices,
        model::Union{CircularGaussianPSF, CircularGaussianPRF},
        free_idx,
        weights,
        axis::NamedTuple
    ) where {FT}

    # `@turbo` requires affine array indices and compile-time-constant tuple
    # indices, so the runtime-valued `free_idx` subset can't be used to index
    # a per-pixel gradient tuple inside the loop.  Instead, always accumulate
    # the full 5-parameter cost, gradient, and (upper-triangle, by symmetry)
    # Hessian into scalar reduction variables, then project onto the requested
    # free-parameter block once after the loop.  `inds` is iterated via its
    # `(yr, xr)` ranges directly (rather than `CartesianIndices`) so
    # offset/non-1-based `inds` still work, and the linear index into
    # `residuals` (k) is recovered affinely from `(i, j)`.
    yr, xr = inds.indices
    ny = length(yr)
    p = _separable_axes!(axis, model, yr, xr)
    αy, αx, amp = FT(p.αy), FT(p.αx), FT(p.amp)
    cH, cE, cF = FT(p.cH), FT(p.cE), FT(p.cF)
    bkg = FT(model.bkg)
    Ey, Gy, Hy, Ex, Gx, Hx = axis.Ey, axis.Gy, axis.Hy, axis.Ex, axis.Gx, axis.Hx

    cost = zero(FT)
    b1 = b2 = b3 = b4 = b5 = zero(FT)
    A11 = A12 = A13 = A14 = A15 = zero(FT)
    A22 = A23 = A24 = A25 = zero(FT)
    A33 = A34 = A35 = zero(FT)
    A44 = A45 = zero(FT)
    A55 = zero(FT)

    LV.@turbo for j in eachindex(xr), i in eachindex(yr)
        ex, ey = Ex[j], Ey[i]
        E = ex * ey
        k = (j - 1) * ny + i
        r = muladd(amp, E, bkg) - FT(image[yr[i], xr[j]])
        residuals[k] = r
        w = FT(weights[k])
        wr = w * r
        cost = muladd(wr, r, cost)
        # evaluate_fg's parameter order: y, x, fwhm, flux, bkg.
        g1 = amp * αy * ex * Gy[i]
        g2 = amp * αx * Gx[j] * ey
        g3 = muladd(cH, muladd(Hx[j], ey, ex * Hy[i]), cE * E)
        g4 = cF * E
        g5 = one(FT)
        b1 = muladd(wr, g1, b1)
        b2 = muladd(wr, g2, b2)
        b3 = muladd(wr, g3, b3)
        b4 = muladd(wr, g4, b4)
        b5 = muladd(wr, g5, b5)
        A11 = muladd(w * g1, g1, A11)
        A12 = muladd(w * g1, g2, A12)
        A13 = muladd(w * g1, g3, A13)
        A14 = muladd(w * g1, g4, A14)
        A15 = muladd(w * g1, g5, A15)
        A22 = muladd(w * g2, g2, A22)
        A23 = muladd(w * g2, g3, A23)
        A24 = muladd(w * g2, g4, A24)
        A25 = muladd(w * g2, g5, A25)
        A33 = muladd(w * g3, g3, A33)
        A34 = muladd(w * g3, g4, A34)
        A35 = muladd(w * g3, g5, A35)
        A44 = muladd(w * g4, g4, A44)
        A45 = muladd(w * g4, g5, A45)
        A55 = muladd(w * g5, g5, A55)
    end

    # Project the full 5x5 normal-equation block onto the requested free
    # parameters (identity projection when all 5 parameters are free).
    bfull = (b1, b2, b3, b4, b5)
    Afull = (
        A11, A12, A13, A14, A15,
        A12, A22, A23, A24, A25,
        A13, A23, A33, A34, A35,
        A14, A24, A34, A44, A45,
        A15, A25, A35, A45, A55,
    )
    nparams = length(free_idx)
    @inbounds for j in 1:nparams
        b[j] = bfull[free_idx[j]]
        for i in 1:nparams
            A[i, j] = Afull[(free_idx[j] - 1) * 5 + free_idx[i]]
        end
    end
    return cost
end

function _accum_separable_gaussian!(A, b, residuals, image, inds::CartesianIndices,
        model::Union{CircularGaussianPSF, CircularGaussianPRF}, free_idx, weights)
    FT = eltype(A)
    yr, xr = inds.indices
    # LV.@turbo needs a real weight vector, so the unweighted case gets ones.
    w = isnothing(weights) ? ones(FT, length(inds)) : weights
    return _accum_separable_gaussian!(A, b, residuals, image, inds, model, free_idx, w,
        _separable_axis_buffers(length(yr), length(xr), FT))
end

# Specialized method: separable per-axis factors; see `_accum_separable_gaussian!`.
function _star_problem(
        model::Union{CircularGaussianPSF{T}, CircularGaussianPRF{T}},
        image::AbstractMatrix,
        inds,
        fixed::NamedTuple,
        inv_var
    ) where {T}

    prepared = _prepare_fit_star_inputs(model, image, inds, fixed, inv_var)
    inds, _, free_idx, x0, free_names_val, FT, base_weights = prepared

    # Prepare an all-ones weight vector for the unweighted case
    # so the LV.@turbo IRLS accumulator can always use a weight vector.
    ones_weights = isnothing(base_weights) ? ones(FT, length(inds)) : base_weights
    yr, xr = inds.indices
    axis = _separable_axis_buffers(length(yr), length(xr), FT)
    # Stream pixels through the separable normal-equation kernel.
    function accum!(A::AbstractMatrix{FT}, b::AbstractVector{FT}, residuals::AbstractVector{FT}, x::AbstractVector{FT}, weights) where {FT}
        @assert size(A, 1) == size(A, 2) == length(b) == length(free_idx)
        @assert length(residuals) == length(inds)
        m = model_from_vector(model, free_names_val, x, fixed)
        w = isnothing(weights) ? ones_weights : weights
        return _accum_separable_gaussian!(A, b, residuals, image, inds, m, free_idx, w, axis)
    end

    return LMProblem(Vector{FT}(x0), length(inds), accum!, base_weights)
end

# Documented with the generic method in PSF.jl.  The separable circular
# Gaussians render from the per-axis factors; see the `render!` below.
_render_scratch(::Union{CircularGaussianPSF, CircularGaussianPRF}, S::Int, ::Type{FT}) where {FT} =
    _separable_axis_buffers(S, S, FT)

"""
    render!(buf, model::Union{CircularGaussianPSF, CircularGaussianPRF}, yr, xr, axis)

Separable [`render!`](@ref), reached by passing the buffers
[`_render_scratch`](@ref) builds: `amp · Ex[j] · Ey[i] + bkg` from the per-axis
factors of [`_separable_axis!`](@ref), so an `n x n` box costs `2 (n + 1)`
erf/exp evaluations instead of `4 n²`.
"""
function render!(
        buf::AbstractMatrix{FT}, model::Union{CircularGaussianPSF, CircularGaussianPRF},
        yr::AbstractUnitRange{<:Integer}, xr::AbstractUnitRange{<:Integer}, axis::NamedTuple
    ) where {FT}
    ny, nx = length(yr), length(xr)
    p = _separable_axes!(axis, model, yr, xr)
    amp, bkg = FT(p.amp), FT(model.bkg)
    Ey, Ex = axis.Ey, axis.Ex
    # Not `LV.@turbo`: it is faster below ~40x40 but 1.7-1.8x slower once `buf`
    # outgrows L1 which happens for bright stars with large model radii.
    @inbounds for j in 1:nx
        exj = amp * Ex[j]
        @simd for i in 1:ny
            buf[i, j] = muladd(exj, Ey[i], bkg)
        end
    end
    return view(buf, 1:ny, 1:nx)
end

# Specialized method: 9.8x faster at (5,5), 5.6x at (11,11), and 3.3x at (21,21) over generic method.
function _star_problem(
        model::GaussianPSF{T},
        image::AbstractMatrix,
        inds,
        fixed::NamedTuple,
        inv_var
    ) where {T}

    # Reuse shared validation and fixed-parameter bookkeeping.
    prepared = _prepare_fit_star_inputs(model, image, inds, fixed, inv_var)
    inds, _, free_idx, x0, free_names_val, FT, base_weights = prepared

    # Stream pixels through a scalar Gaussian-specific normal-equation kernel.
    function accum!(A::AbstractMatrix{FT}, b::AbstractVector{FT}, residuals::AbstractVector{FT}, x::AbstractVector{FT}, weights) where {FT}
        @assert size(A, 1) == size(A, 2) == length(b) == length(free_idx)
        @assert length(residuals) == length(inds)
        m = model_from_vector(model, free_names_val, x, fixed)
        return _accum_gaussian!(A, b, residuals, image, inds, m, free_idx, weights)
    end

    # Delegate iteration details to the shared LM implementation.
    return LMProblem(Vector{FT}(x0), length(inds), accum!, base_weights)
end

function _accum_gaussian!(
        A::AbstractMatrix{FT},
        b::AbstractVector{FT},
        residuals::AbstractVector{FT},
        image::AbstractMatrix,
        inds::CartesianIndices,
        model::GaussianPSF,
        free_idx,
        weights
    ) where {FT}

    # Precompute model constants and rotation terms shared by all pixels.
    fill!(A, zero(FT))
    fill!(b, zero(FT))
    γ = FT(GAUSS_PRE)
    x0 = FT(model.x)
    y0 = FT(model.y)
    ax = FT(model.x_fwhm)
    ay = FT(model.y_fwhm)
    flux = FT(model.flux)
    bkg = FT(model.bkg)
    ax² = ax^2
    ay² = ay^2
    θ = deg2rad(FT(model.theta))
    sn, cs = sincos(θ)
    norm = -(FT(π) * ax * ay / γ)
    amp = flux / norm
    degree = deg2rad(one(FT))

    # Evaluate analytic derivatives and accumulate only the active parameter block.
    cost = zero(FT)
    obs_k = 0
    nparams = length(free_idx)
    use_weights = !isnothing(weights)
    @inbounds for idx in inds
        obs_k += 1
        w = use_weights ? FT(weights[obs_k]) : one(FT)
        dx = FT(idx[2]) - x0
        dy = FT(idx[1]) - y0
        u = cs * dx + sn * dy
        v = -sn * dx + cs * dy
        sqmahab = u^2 / ax² + v^2 / ay²
        profile = exp(γ * sqmahab)
        Ag = amp * profile
        f_val = muladd(amp, profile, bkg)
        r = f_val - FT(image[idx])
        residuals[obs_k] = r
        wr = w * r
        cost = muladd(wr, r, cost)

        # Match evaluate_fg's parameter order: y, x, y_fwhm, x_fwhm, theta, flux, bkg.
        D = one(FT) / ax² - one(FT) / ay²
        Qx = -2 * (cs * u / ax² - sn * v / ay²)
        Qy = -2 * (sn * u / ax² + cs * v / ay²)
        Qax = -2 * u^2 / ax^3
        Qay = -2 * v^2 / ay^3
        Qtheta = degree * 2 * u * v * D
        g_full = (
            Ag * γ * Qy,
            Ag * γ * Qx,
            Ag * (-one(FT) / ay + γ * Qay),
            Ag * (-one(FT) / ax + γ * Qax),
            Ag * γ * Qtheta,
            profile / norm,
            one(FT),
        )

        # Accumulate the projected normal-equation block expected by lm_irls.
        for j in 1:nparams
            gj = g_full[free_idx[j]]
            b[j] = muladd(wr, gj, b[j])
            for i in 1:nparams
                A[i, j] = muladd(w * g_full[free_idx[i]], gj, A[i, j])
            end
        end
    end
    return cost
end

# Specialized method: scalar integrated-Gaussian accumulator.
function _star_problem(
        model::GaussianPRF{T},
        image::AbstractMatrix,
        inds,
        fixed::NamedTuple,
        inv_var
    ) where {T}

    # Reuse shared validation and fixed-parameter bookkeeping.
    prepared = _prepare_fit_star_inputs(model, image, inds, fixed, inv_var)
    inds, _, free_idx, x0, free_names_val, FT, base_weights = prepared
    # Stream pixels through a scalar Gaussian-PRF normal-equation kernel.
    function accum!(A::AbstractMatrix{FT}, b::AbstractVector{FT}, residuals::AbstractVector{FT}, x::AbstractVector{FT}, weights) where {FT}
        @assert size(A, 1) == size(A, 2) == length(b) == length(free_idx)
        @assert length(residuals) == length(inds)
        m = model_from_vector(model, free_names_val, x, fixed)
        return _accum_gaussian_prf!(A, b, residuals, image, inds, m, free_idx, weights)
    end

    # Delegate iteration details to the shared LM implementation.
    return LMProblem(Vector{FT}(x0), length(inds), accum!, base_weights)
end

function _accum_gaussian_prf!(
        A::AbstractMatrix{FT},
        b::AbstractVector{FT},
        residuals::AbstractVector{FT},
        image::AbstractMatrix,
        inds::CartesianIndices,
        model::GaussianPRF,
        free_idx,
        weights
    ) where {FT}

    # Precompute integrated-Gaussian constants and rotation terms.
    fill!(A, zero(FT))
    fill!(b, zero(FT))
    c = sqrt(-FT(GAUSS_PRE))
    x0 = FT(model.x)
    y0 = FT(model.y)
    ax = FT(model.x_fwhm)
    ay = FT(model.y_fwhm)
    flux = FT(model.flux)
    bkg = FT(model.bkg)
    αx = c / ax
    αy = c / ay
    θ = deg2rad(FT(model.theta))
    sn, cs = sincos(θ)
    two_sqrtpi = 2 / sqrt(FT(π))
    fl4 = flux / 4
    degree = deg2rad(one(FT))

    # Evaluate difference-of-erf derivatives and accumulate the active block.
    cost = zero(FT)
    obs_k = 0
    nparams = length(free_idx)
    use_weights = !isnothing(weights)
    @inbounds for idx in inds
        obs_k += 1
        w = use_weights ? FT(weights[obs_k]) : one(FT)
        dx = FT(idx[2]) - x0
        dy = FT(idx[1]) - y0
        u = cs * dx + sn * dy
        v = -sn * dx + cs * dy
        u_p = αx * (u + FT(0.5))
        u_m = αx * (u - FT(0.5))
        v_p = αy * (v + FT(0.5))
        v_m = αy * (v - FT(0.5))
        Ex = erf(u_p) - erf(u_m)
        Ey = erf(v_p) - erf(v_m)
        f_val = muladd(fl4, Ex * Ey, bkg)
        r = f_val - FT(image[idx])
        residuals[obs_k] = r
        wr = w * r
        cost = muladd(wr, r, cost)

        # Match evaluate_fg's parameter order: y, x, y_fwhm, x_fwhm, theta, flux, bkg.
        Gxp = two_sqrtpi * exp(-u_p^2)
        Gxm = two_sqrtpi * exp(-u_m^2)
        Gyp = two_sqrtpi * exp(-v_p^2)
        Gym = two_sqrtpi * exp(-v_m^2)
        dEx_du = αx * (Gxm - Gxp)
        dEy_dv = αy * (Gym - Gyp)
        g_full = (
            fl4 * (sn * dEx_du * Ey + cs * dEy_dv * Ex),
            fl4 * (cs * dEx_du * Ey - sn * dEy_dv * Ex),
            fl4 / ay * Ex * (Gym * v_m - Gyp * v_p),
            fl4 / ax * (Gxm * u_m - Gxp * u_p) * Ey,
            fl4 * degree * (dEy_dv * u * Ex - dEx_du * v * Ey),
            Ex * Ey / 4,
            one(FT),
        )

        # Accumulate the projected normal-equation block expected by lm_irls.
        for j in 1:nparams
            gj = g_full[free_idx[j]]
            b[j] = muladd(wr, gj, b[j])
            for i in 1:nparams
                A[i, j] = muladd(w * g_full[free_idx[i]], gj, A[i, j])
            end
        end
    end
    return cost
end

# Specialized method: 5x faster at (5,5), 3.5x at (11,11), and 2.2x at (21,21) over generic method.
function _star_problem(
        model::CircularMoffatPSF{T},
        image::AbstractMatrix,
        inds,
        fixed::NamedTuple,
        inv_var
    ) where {T}

    # Reuse shared validation and fixed-parameter bookkeeping.
    prepared = _prepare_fit_star_inputs(model, image, inds, fixed, inv_var)
    inds, _, free_idx, x0, free_names_val, FT, base_weights = prepared

    # Stream pixels through a CircularMoffat-specific normal-equation kernel.
    function accum!(A::AbstractMatrix{FT}, b::AbstractVector{FT}, residuals::AbstractVector{FT}, x::AbstractVector{FT}, weights) where {FT}
        @assert size(A, 1) == size(A, 2) == length(b) == length(free_idx)
        @assert length(residuals) == length(inds)
        m = model_from_vector(model, free_names_val, x, fixed)
        return _accum_circular_moffat!(A, b, residuals, image, inds, m, free_idx, weights)
    end

    # Delegate iteration details to the shared LM implementation.
    return LMProblem(Vector{FT}(x0), length(inds), accum!, base_weights)
end

function _accum_circular_moffat!(
        A::AbstractMatrix{FT},
        b::AbstractVector{FT},
        residuals::AbstractVector{FT},
        image::AbstractMatrix,
        inds::CartesianIndices,
        model::CircularMoffatPSF,
        free_idx,
        weights
    ) where {FT}

    # Precompute model constants that are shared across all fit pixels.
    fill!(A, zero(FT))
    fill!(b, zero(FT))
    x0 = FT(model.x)
    y0 = FT(model.y)
    α = FT(model.α)
    β = FT(model.β)
    flux = FT(model.flux)
    bkg = FT(model.bkg)
    α² = α^2
    norm = FT(π) * α² / (β - one(FT))
    amp = flux / norm

    # Evaluate analytic derivatives and accumulate only the active parameter block.
    cost = zero(FT)
    obs_k = 0
    nparams = length(free_idx)
    use_weights = !isnothing(weights)
    @inbounds for idx in inds
        obs_k += 1
        w = use_weights ? FT(weights[obs_k]) : one(FT)
        dx = FT(idx[2]) - x0
        dy = FT(idx[1]) - y0
        r2 = dx^2 + dy^2
        u = one(FT) + r2 / α²
        profile = u^(-β)
        Ag = amp * profile
        f_val = muladd(amp, profile, bkg)
        r = f_val - FT(image[idx])
        residuals[obs_k] = r
        wr = w * r
        cost = muladd(wr, r, cost)
        g_full = (
            2 * Ag * β * dy / (α² * u),
            2 * Ag * β * dx / (α² * u),
            Ag * (-2 / α + 2 * β * r2 / (α^3 * u)),
            Ag * (1 / (β - one(FT)) - log(u)),
            profile / norm,
            one(FT),
        )

        # Accumulate the projected normal-equation block expected by lm_irls.
        for j in 1:nparams
            gj = g_full[free_idx[j]]
            b[j] = muladd(wr, gj, b[j])
            for i in 1:nparams
                A[i, j] = muladd(w * g_full[free_idx[i]], gj, A[i, j])
            end
        end
    end
    return cost
end

# Specialized method: 4.9x faster at (5,5), 2.6x at (11,11), and 1.8x at (21,21) over generic method.
function _star_problem(
        model::MoffatPSF{T},
        image::AbstractMatrix,
        inds,
        fixed::NamedTuple,
        inv_var
    ) where {T}

    # Reuse shared validation and fixed-parameter bookkeeping.
    prepared = _prepare_fit_star_inputs(model, image, inds, fixed, inv_var)
    inds, _, free_idx, x0, free_names_val, FT, base_weights = prepared

    # Stream pixels through a scalar Moffat-specific normal-equation kernel.
    function accum!(A::AbstractMatrix{FT}, b::AbstractVector{FT}, residuals::AbstractVector{FT}, x::AbstractVector{FT}, weights) where {FT}
        @assert size(A, 1) == size(A, 2) == length(b) == length(free_idx)
        @assert length(residuals) == length(inds)
        m = model_from_vector(model, free_names_val, x, fixed)
        return _accum_moffat!(A, b, residuals, image, inds, m, free_idx, weights)
    end

    # Delegate iteration details to the shared LM implementation.
    return LMProblem(Vector{FT}(x0), length(inds), accum!, base_weights)
end

function _accum_moffat!(
        A::AbstractMatrix{FT},
        b::AbstractVector{FT},
        residuals::AbstractVector{FT},
        image::AbstractMatrix,
        inds::CartesianIndices,
        model::MoffatPSF,
        free_idx,
        weights
    ) where {FT}

    # Precompute model constants and rotation terms shared by all pixels.
    fill!(A, zero(FT))
    fill!(b, zero(FT))
    x0 = FT(model.x)
    y0 = FT(model.y)
    ax = FT(model.x_α)
    ay = FT(model.y_α)
    β = FT(model.β)
    flux = FT(model.flux)
    bkg = FT(model.bkg)
    ax² = ax^2
    ay² = ay^2
    θ = deg2rad(FT(model.theta))
    sn, cs = sincos(θ)
    norm = FT(π) * ax * ay / (β - one(FT))
    amp = flux / norm
    degree = deg2rad(one(FT))

    # Evaluate analytic derivatives and accumulate only the active parameter block.
    cost = zero(FT)
    obs_k = 0
    nparams = length(free_idx)
    use_weights = !isnothing(weights)
    @inbounds for idx in inds
        obs_k += 1
        w = use_weights ? FT(weights[obs_k]) : one(FT)
        dx = FT(idx[2]) - x0
        dy = FT(idx[1]) - y0
        u = cs * dx + sn * dy
        v = -sn * dx + cs * dy
        q = u^2 / ax² + v^2 / ay²
        h = one(FT) + q
        profile = h^(-β)
        Ag = amp * profile
        f_val = muladd(amp, profile, bkg)
        r = f_val - FT(image[idx])
        residuals[obs_k] = r
        wr = w * r
        cost = muladd(wr, r, cost)

        # Match evaluate_fg's parameter order: y, x, y_α, x_α, theta, β, flux, bkg.
        D = one(FT) / ax² - one(FT) / ay²
        Qx = -2 * (cs * u / ax² - sn * v / ay²)
        Qy = -2 * (sn * u / ax² + cs * v / ay²)
        Qax = -2 * u^2 / ax^3
        Qay = -2 * v^2 / ay^3
        Qtheta = degree * 2 * u * v * D
        scale = -β / h
        g_full = (
            Ag * scale * Qy,
            Ag * scale * Qx,
            Ag * (-one(FT) / ay + scale * Qay),
            Ag * (-one(FT) / ax + scale * Qax),
            Ag * scale * Qtheta,
            Ag * (one(FT) / (β - one(FT)) - log(h)),
            profile / norm,
            one(FT),
        )

        # Accumulate the projected normal-equation block expected by lm_irls.
        for j in 1:nparams
            gj = g_full[free_idx[j]]
            b[j] = muladd(wr, gj, b[j])
            for i in 1:nparams
                A[i, j] = muladd(w * g_full[free_idx[i]], gj, A[i, j])
            end
        end
    end
    return cost
end

# Specialized method: 4.3x faster at (5,5), 1.7x at (11,11), and 1.3x at (21,21) over generic method.
function _star_problem(
        model::AiryPSF{T},
        image::AbstractMatrix,
        inds,
        fixed::NamedTuple,
        inv_var
    ) where {T}

    # Reuse shared validation and fixed-parameter bookkeeping.
    prepared = _prepare_fit_star_inputs(model, image, inds, fixed, inv_var)
    inds, _, free_idx, x0, free_names_val, FT, base_weights = prepared

    # Stream pixels through an Airy-specific normal-equation kernel.
    function accum!(A::AbstractMatrix{FT}, b::AbstractVector{FT}, residuals::AbstractVector{FT}, x::AbstractVector{FT}, weights) where {FT}
        @assert size(A, 1) == size(A, 2) == length(b) == length(free_idx)
        @assert length(residuals) == length(inds)
        m = model_from_vector(model, free_names_val, x, fixed)
        return _accum_airy!(A, b, residuals, image, inds, m, free_idx, weights)
    end

    # Delegate iteration details to the shared LM implementation.
    return LMProblem(Vector{FT}(x0), length(inds), accum!, base_weights)
end

function _accum_airy!(
        A::AbstractMatrix{FT},
        b::AbstractVector{FT},
        residuals::AbstractVector{FT},
        image::AbstractMatrix,
        inds::CartesianIndices,
        model::AiryPSF,
        free_idx,
        weights
    ) where {FT}

    # Precompute model constants and Airy radial scale.
    fill!(A, zero(FT))
    fill!(b, zero(FT))
    x0 = FT(model.x)
    y0 = FT(model.y)
    radius = FT(model.radius)
    flux = FT(model.flux)
    bkg = FT(model.bkg)
    airy_rz = FT(AIRY_RZ)
    a = radius / airy_rz
    norm = a^2 / FT(π) * 4
    amp = flux / norm

    # Evaluate Airy values, analytic derivatives, and projected normal equations.
    cost = zero(FT)
    obs_k = 0
    nparams = length(free_idx)
    use_weights = !isnothing(weights)
    @inbounds for idx in inds
        obs_k += 1
        w = use_weights ? FT(weights[obs_k]) : one(FT)
        dx = FT(idx[2]) - x0
        dy = FT(idx[1]) - y0
        r = sqrt(dx^2 + dy^2)
        u = FT(π) * r / a
        if abs(u) < eps(FT)
            A2 = one(FT)
            dA2_du = zero(FT)
        else
            J0 = besselj0(u)
            J1 = besselj1(u)
            airy = 2 * J1 / u
            # J2 eliminated via the Bessel recurrence J2 = (2/u)J1 - J0:
            # airy_p = (u * (J0 - J2) - 2J1) / u^2 = 2J0/u - 4J1/u^2
            airy_p = 2 * J0 / u - 4 * J1 / u^2
            A2 = airy^2
            dA2_du = 2 * airy * airy_p
        end

        # Accumulate residual and derivative columns, matching evaluate_fg's center branch.
        f_val = muladd(amp, A2, bkg)
        residual = f_val - FT(image[idx])
        residuals[obs_k] = residual
        wr = w * residual
        cost = muladd(wr, residual, cost)
        df_dflux = A2 / norm
        if r == 0
            g_full = (zero(FT), zero(FT), zero(FT), df_dflux, one(FT))
        else
            du_dr = FT(π) / a
            df_dr = amp * dA2_du * du_dr
            df_dx = -df_dr * dx / r
            df_dy = -df_dr * dy / r
            df_da = amp / a * (-u * dA2_du - 2 * A2)
            df_dradius = df_da / airy_rz
            g_full = (df_dy, df_dx, df_dradius, df_dflux, one(FT))
        end

        # Accumulate only the active parameter block expected by lm_irls.
        for j in 1:nparams
            gj = g_full[free_idx[j]]
            b[j] = muladd(wr, gj, b[j])
            for i in 1:nparams
                A[i, j] = muladd(w * g_full[free_idx[i]], gj, A[i, j])
            end
        end
    end
    return cost
end

# Specialized method: faster than generic by 4x at (5,5), 2.2x at (11,11), and 1.4x at (21,21).
function _star_problem(
        model::ImagePSF{T},
        image::AbstractMatrix,
        inds,
        fixed::NamedTuple,
        inv_var
    ) where {T}

    # Share validation and fixed-parameter bookkeeping with the generic fitter.
    prepared = _prepare_fit_star_inputs(model, image, inds, fixed, inv_var)
    inds, _, free_idx, x0, free_names_val, FT, base_weights = prepared

    # Route ImagePSF fits through a scalarized accumulator while preserving
    # arbitrary fixed combinations of x, y, flux, and bkg.
    function accum!(A::AbstractMatrix{FT}, b::AbstractVector{FT}, residuals::AbstractVector{FT}, x::AbstractVector{FT}, weights) where {FT}
        @assert size(A, 1) == size(A, 2) == length(b) == length(free_idx)
        @assert length(residuals) == length(inds)
        m = model_from_vector(model, free_names_val, x, fixed)
        return _accum_image_psf!(A, b, residuals, image, inds, m, free_idx, weights)
    end

    # Keep the LM iteration, damping, IRLS, and covariance semantics shared.
    return LMProblem(Vector{FT}(x0), length(inds), accum!, base_weights)
end

function _accum_image_psf!(
        A::AbstractMatrix{FT},
        b::AbstractVector{FT},
        residuals::AbstractVector{FT},
        image::AbstractMatrix,
        inds::CartesianIndices,
        model::ImagePSF,
        free_idx,
        weights
    ) where {FT}

    # Precompute interpolation metadata before projecting onto free parameters.
    fill!(A, zero(FT))
    fill!(b, zero(FT))
    data = model.data
    sx = FT(model.oversampling[1])
    sy = FT(model.oversampling[2])
    x0 = FT(model.x)
    y0 = FT(model.y)
    flux = FT(model.flux)
    bkg = FT(model.bkg)
    ox = FT(model.origin.x)
    oy = FT(model.origin.y)
    fill_value = FT(model.fill_value)

    # Compute all four derivatives, then accumulate only the requested block.
    cost = zero(FT)
    obs_k = 0
    nparams = length(free_idx)
    use_weights = !isnothing(weights)
    @inbounds for idx in inds
        obs_k += 1
        w = use_weights ? FT(weights[obs_k]) : one(FT)
        u = sx * (FT(idx[2]) - x0) + ox
        v = sy * (FT(idx[1]) - y0) + oy
        p, dpdv, dpdu = bicubic_interpolate(data, v, u; fill_value)
        profile = FT(p)
        r = muladd(flux, profile, bkg) - FT(image[idx])
        residuals[obs_k] = r
        wr = w * r
        cost = muladd(wr, r, cost)
        g_full = (
            -flux * sy * FT(dpdv),
            -flux * sx * FT(dpdu),
            profile,
            one(FT),
        )

        # Project the four ImagePSF parameters onto the active free subset.
        for j in 1:nparams
            gj = g_full[free_idx[j]]
            b[j] = muladd(wr, gj, b[j])
            for i in 1:nparams
                A[i, j] = muladd(w * g_full[free_idx[i]], gj, A[i, j])
            end
        end
    end
    return cost
end

# Specialized method: uses `LV.@turbo` for GriddedPSFModel{ImagePSF}; see
# `_accum_gridded_imagepsf!` for design and benchmark details.
function _star_problem(
        model::GriddedPSFModel{T, M},
        image::AbstractMatrix,
        inds,
        fixed::NamedTuple,
        inv_var
    ) where {T, M <: ImagePSF{T}}

    prepared = _prepare_fit_star_inputs(model, image, inds, fixed, inv_var)
    inds, _, free_idx, x0, free_names_val, FT, base_weights = prepared
    ones_weights = isnothing(base_weights) ? ones(FT, length(inds)) : base_weights

    # Scratch buffers reused across every LM iteration for this fit: three
    # cutout-sized caches (value, dpdv, dpdu) per corner, threading the four
    # elementwise `@turbo` passes into the final reduction pass.
    ny_c, nx_c = length(inds.indices[1]), length(inds.indices[2])
    p_val = ntuple(_ -> Matrix{FT}(undef, ny_c, nx_c), 4)
    p_dpdv = ntuple(_ -> Matrix{FT}(undef, ny_c, nx_c), 4)
    p_dpdu = ntuple(_ -> Matrix{FT}(undef, ny_c, nx_c), 4)

    function accum!(A::AbstractMatrix{FT}, b::AbstractVector{FT}, residuals::AbstractVector{FT}, x::AbstractVector{FT}, weights) where {FT}
        @assert size(A, 1) == size(A, 2) == length(b) == length(free_idx)
        @assert length(residuals) == length(inds)
        m = model_from_vector(model, free_names_val, x, fixed)
        w = isnothing(weights) ? ones_weights : weights
        return _accum_gridded_imagepsf!(A, b, residuals, image, inds, m, free_idx, w, p_val, p_dpdv, p_dpdu)
    end

    return LMProblem(Vector{FT}(x0), length(inds), accum!, base_weights)
end

"""
    _gridded_corner_bicubic_pass!(pv, pdv, pdu, data, ox, oy, sx, sy, fv, ny_d, nx_d, yr, xr, Y, X, y1, x1)

Bicubic value + both partial derivatives (`dpdv`, `dpdu`) for one grid
corner's own tabulated stamp `data`, at every pixel in the cutout `(yr,
xr)`. Writes `ifelse`-clamped results into `pv`/`pdv`/`pdu`, which are
mutated in place and sized to the cutout, not the stamp.

Written once and called once per corner (up to four times) by
[`_accum_gridded_imagepsf!`](@ref) rather than hand-duplicated, since
fusing more than one corner's worth of 4x4-gather bicubic math into a
single `@turbo` loop exceeds an internal LoopVectorization scheduling
limit (a `LoopSet` `ArgumentError` at compile time, not a semantic
restriction -- confirmed to fail compiling even at two matrices' worth of
gathers together, well short of four). Being an ordinary function containing
its own self-contained `@turbo` loop, rather than a macro expanded at each
call site, is sufficient for that: `@turbo` only ever inspects the syntax
of its own loop, not what encloses it, so this compiles identically to
four inlined copies while being compiled once and called four times --
less code generated, not more, for the same result.
"""
function _gridded_corner_bicubic_pass!(
        pv::AbstractMatrix{FT}, pdv::AbstractMatrix{FT}, pdu::AbstractMatrix{FT},
        data::AbstractMatrix, ox, oy, sx, sy, fv, ny_d, nx_d,
        yr, xr, Y, X, y1, x1
    ) where {FT}
    LV.@turbo for j in xr, i in yr
        u = sx * (FT(j) - X) + ox
        v = sy * (FT(i) - Y) + oy
        _inb = isfinite(u) & isfinite(v) & (u >= one(FT)) & (u <= FT(nx_d)) & (v >= one(FT)) & (v <= FT(ny_d))
        uc = ifelse(_inb, u, one(FT))
        vc = ifelse(_inb, v, one(FT))
        lx = clamp(floor(Int32, uc), Int32(1), Int32(nx_d - 1))
        ly = clamp(floor(Int32, vc), Int32(1), Int32(ny_d - 1))
        du = uc - FT(lx)
        dv = vc - FT(ly)
        ix1_ = clamp(lx - Int32(1), Int32(1), Int32(nx_d))
        ix2_ = lx
        ix3_ = lx + Int32(1)
        ix4_ = clamp(lx + Int32(2), Int32(1), Int32(nx_d))
        iy1_ = clamp(ly - Int32(1), Int32(1), Int32(ny_d))
        iy2_ = ly
        iy3_ = ly + Int32(1)
        iy4_ = clamp(ly + Int32(2), Int32(1), Int32(ny_d))

        r1a = data[iy1_, ix1_]
        r1b = data[iy1_, ix2_]
        r1c = data[iy1_, ix3_]
        r1d = data[iy1_, ix4_]
        c1a = (r1c - r1a) / 2
        c4a = r1c - r1b - c1a
        c2a = 3 * c4a - (r1d - r1b) / 2 + c1a
        c3a = c4a - c2a
        row1 = du * (du * (du * c3a + c2a) + c1a) + r1b
        drow1 = du * (3 * du * c3a + 2 * c2a) + c1a

        r2a = data[iy2_, ix1_]
        r2b = data[iy2_, ix2_]
        r2c = data[iy2_, ix3_]
        r2d = data[iy2_, ix4_]
        c1b = (r2c - r2a) / 2
        c4b = r2c - r2b - c1b
        c2b = 3 * c4b - (r2d - r2b) / 2 + c1b
        c3b = c4b - c2b
        row2 = du * (du * (du * c3b + c2b) + c1b) + r2b
        drow2 = du * (3 * du * c3b + 2 * c2b) + c1b

        r3a = data[iy3_, ix1_]
        r3b = data[iy3_, ix2_]
        r3c = data[iy3_, ix3_]
        r3d = data[iy3_, ix4_]
        c1c = (r3c - r3a) / 2
        c4c = r3c - r3b - c1c
        c2c = 3 * c4c - (r3d - r3b) / 2 + c1c
        c3c = c4c - c2c
        row3 = du * (du * (du * c3c + c2c) + c1c) + r3b
        drow3 = du * (3 * du * c3c + 2 * c2c) + c1c

        r4a = data[iy4_, ix1_]
        r4b = data[iy4_, ix2_]
        r4c = data[iy4_, ix3_]
        r4d = data[iy4_, ix4_]
        c1d = (r4c - r4a) / 2
        c4d = r4c - r4b - c1d
        c2d = 3 * c4d - (r4d - r4b) / 2 + c1d
        c3d = c4d - c2d
        row4 = du * (du * (du * c3d + c2d) + c1d) + r4b
        drow4 = du * (3 * du * c3d + 2 * c2d) + c1d

        cv1 = (row3 - row1) / 2
        cv4 = row3 - row2 - cv1
        cv2 = 3 * cv4 - (row4 - row2) / 2 + cv1
        cv3 = cv4 - cv2
        val = dv * (dv * (dv * cv3 + cv2) + cv1) + row2
        dpdv_ = dv * (3 * dv * cv3 + 2 * cv2) + cv1

        cu1 = (drow3 - drow1) / 2
        cu4 = drow3 - drow2 - cu1
        cu2 = 3 * cu4 - (drow4 - drow2) / 2 + cu1
        cu3 = cu4 - cu2
        dpdu_ = dv * (dv * (dv * cu3 + cu2) + cu1) + drow2

        ii = i - y1 + 1
        jj = j - x1 + 1
        pv[ii, jj] = ifelse(_inb, val, fv)
        pdv[ii, jj] = ifelse(_inb, dpdv_, zero(FT))
        pdu[ii, jj] = ifelse(_inb, dpdu_, zero(FT))
    end
    return nothing
end

# Documented with the generic method in PSF.jl; defined here because it pairs
# with the specialized `render!` below.
function _render_scratch(::GriddedPSFModel{T2, M}, S::Int, ::Type{FT}) where {T2, M <: ImagePSF{T2}, FT}
    return ntuple(_ -> ntuple(_ -> Matrix{FT}(undef, S, S), 4), 3)
end

"""
    render!(buf, model::GriddedPSFModel{T,<:ImagePSF{T}}, yr, xr, scratch)

Specialized [`render!`](@ref) reached by passing the buffers
[`_render_scratch`](@ref) builds for `model`.  Corner selection/weights and
each active node's recentered origin depend only on the star's `(Y, X)`, not
on which pixel is being evaluated, so they are computed once here instead of
once per `evaluate` call as the generic method does.  The bicubic gather and
blend reuses [`_gridded_corner_bicubic_pass!`](@ref) (the same kernel
[`_accum_gridded_imagepsf!`](@ref) uses), one branchless pass per corner into
`scratch`'s per-corner buffers, followed by a weighted-sum reduction into
`buf`.

This also sidesteps `_turbo_safe(::Type{<:ImagePSF}) == false`: the reduction
vectorizes because it reads dense cached matrices, never `evaluate`.
"""
function render!(
        buf::AbstractMatrix{FT}, model::GriddedPSFModel{T2, M},
        yr::AbstractUnitRange{<:Integer}, xr::AbstractUnitRange{<:Integer},
        scratch::NTuple{3, NTuple{4, Matrix{FT}}}
    ) where {FT, T2, M <: ImagePSF{T2}}
    ny, nx = length(yr), length(xr)
    Y, X, flux, bkg = FT(model.y), FT(model.x), FT(model.flux), FT(model.bkg)
    pv1, pv2, pv3, pv4 = scratch[1]
    pdv1, pdv2, pdv3, pdv4 = scratch[2]
    pdu1, pdu2, pdu3, pdu4 = scratch[3]

    corners = _grid_corners_dw(model, Y, X)
    idx1, w1 = corners[1][1], FT(corners[1][2])
    idx2, w2 = corners[2][1], FT(corners[2][2])
    idx3, w3 = corners[3][1], FT(corners[3][2])
    idx4, w4 = corners[4][1], FT(corners[4][2])
    # `_grid_corners_dw` returns idx=0 for corners 2-4 only when there is a
    # single node; their weight is exactly 0 there, so remapping to a valid
    # (arbitrary) index is safe (matches `_accum_gridded_imagepsf!`).
    idx2 = idx2 == 0 ? idx1 : idx2
    idx3 = idx3 == 0 ? idx1 : idx3
    idx4 = idx4 == 0 ? idx1 : idx4
    node1, node2, node3, node4 = model.psfs[idx1], model.psfs[idx2], model.psfs[idx3], model.psfs[idx4]

    y1, x1 = first(yr), first(xr)
    _gridded_corner_bicubic_pass!(pv1, pdv1, pdu1, node1.data, FT(node1.origin.x), FT(node1.origin.y),
        FT(node1.oversampling[1]), FT(node1.oversampling[2]), FT(node1.fill_value),
        size(node1.data, 1), size(node1.data, 2), yr, xr, Y, X, y1, x1)
    _gridded_corner_bicubic_pass!(pv2, pdv2, pdu2, node2.data, FT(node2.origin.x), FT(node2.origin.y),
        FT(node2.oversampling[1]), FT(node2.oversampling[2]), FT(node2.fill_value),
        size(node2.data, 1), size(node2.data, 2), yr, xr, Y, X, y1, x1)
    _gridded_corner_bicubic_pass!(pv3, pdv3, pdu3, node3.data, FT(node3.origin.x), FT(node3.origin.y),
        FT(node3.oversampling[1]), FT(node3.oversampling[2]), FT(node3.fill_value),
        size(node3.data, 1), size(node3.data, 2), yr, xr, Y, X, y1, x1)
    _gridded_corner_bicubic_pass!(pv4, pdv4, pdu4, node4.data, FT(node4.origin.x), FT(node4.origin.y),
        FT(node4.oversampling[1]), FT(node4.oversampling[2]), FT(node4.fill_value),
        size(node4.data, 1), size(node4.data, 2), yr, xr, Y, X, y1, x1)

    LV.@turbo for j in 1:nx, i in 1:ny
        buf[i, j] = muladd(flux, w1 * pv1[i, j] + w2 * pv2[i, j] + w3 * pv3[i, j] + w4 * pv4[i, j], bkg)
    end
    return view(buf, 1:ny, 1:nx)
end

"""
    _accum_gridded_imagepsf!(A, b, residuals, image, inds, model, free_idx, weights,
        p_val, p_dpdv, p_dpdu)

`@turbo`-vectorized normal-equation accumulator for
`GriddedPSFModel{T, <:ImagePSF{T}}` fits.

# Design
`GriddedPSFModel` evaluates as a bilinear blend of up to four corner node
PSFs (see `evaluate_fg(model::GriddedPSFModel, ...)`), each independently
bicubic-interpolated. Fusing all four corners' bicubic evaluations (value
plus both partial derivatives, 16 gathers each) into a single `@turbo` loop
exceeds an internal LoopVectorization scheduling limit (confirmed to fail
compiling even at two matrices' worth of gathers together, well short of
four). An earlier design worked around this by pre-blending the four corner
stamps (nodes) into one, exploiting linearity of bicubic interpolation in the
tabulated samples; that made per-`accum!`-call cost scale with stamp size
rather than cutout size, which measured 20-50x slower than the generic path
for a 361x361, oversampling-4 stamp fit with a 5x5 cutout (Roman CRDS ePSF
geometry).

This design avoids both problems: each corner's own bicubic is
computed in its own `@turbo` pass via
[`_gridded_corner_bicubic_pass!`](@ref), writing `ifelse`-clamped results
into per-corner scratch matrices sized to the *cutout*, not the stamp. A
final reduction pass combines the four cached corners via the
(compile-time-fixed-per-call) corner weights and their position
derivatives, matching the chain rule in `evaluate_fg(::GriddedPSFModel,
...)`, and accumulates the normal equations. Five `@turbo` loops total
(four elementwise, one reduction); each elementwise pass recomputes the
shared stencil indices (`ix1_..4_`, `iy1_..4_`, `du`, `dv`) redundantly
rather than caching them separately, since that arithmetic is cheap
relative to the 16 gathers each pass performs.

Because there is no blending step, corner nodes no longer need to share
`size(data)`, `origin`, or `oversampling` -- this handles heterogeneous
grids directly, unlike the earlier blended design.

# Performance
Benchmarked against the generic `fit_star` path (which evaluates
`evaluate_fg` per node per pixel with no vectorization).

At ePSF node size 31x31, oversampling 1:

| type    | n=5   | n=11  | n=21  | n=31  |
|:--------|:------|:------|:------|:------|
| Float32 | 2.4x  | 3.7x  | 5.2x  | 5.7x  |
| Float64 | 1.4x  | 2.2x  | 2.4x  | 2.6x  |

At ePSF node size 361x361, oversampling 4 (Roman CRDS ePSF geometry):

| type    | n=5   | n=8   | n=15  | n=31  |
|:--------|:------|:------|:------|:------|
| Float32 | 2.4x  | 4.0x  | 4.8x  | 5.8x  |
| Float64 | 1.4x  | 2.3x  | 2.4x  | 2.6x  |

Gains at every tested size and stamp geometry, unlike the earlier
blended design, which regressed sharply once the stamp was large relative
to the cutout.
"""
function _accum_gridded_imagepsf!(
        A::AbstractMatrix{FT},
        b::AbstractVector{FT},
        residuals::AbstractVector{FT},
        image::AbstractMatrix,
        inds::CartesianIndices,
        model::GriddedPSFModel,
        free_idx,
        weights,
        p_val::NTuple{4, AbstractMatrix{FT}},
        p_dpdv::NTuple{4, AbstractMatrix{FT}},
        p_dpdu::NTuple{4, AbstractMatrix{FT}}
    ) where {FT}

    Y = FT(model.y)
    X = FT(model.x)
    flux = FT(model.flux)
    bkg = FT(model.bkg)

    # Corner weights and their position derivatives are constant over the
    # whole cutout (they depend only on (Y, X)), so this is computed once
    # per `accum!` call, not per pixel.
    corners = _grid_corners_dw(model, Y, X)
    idx1, w1, dwdy1, dwdx1 = corners[1]
    idx2, w2, dwdy2, dwdx2 = corners[2]
    idx3, w3, dwdy3, dwdx3 = corners[3]
    idx4, w4, dwdy4, dwdx4 = corners[4]
    # `_grid_corners_dw` returns idx=0 for corners 2-4 only when
    # `length(model.psfs) == 1`; their weight is exactly 0 there, so
    # remapping to a valid (arbitrary) index is safe.
    idx2 = idx2 == 0 ? idx1 : idx2
    idx3 = idx3 == 0 ? idx1 : idx3
    idx4 = idx4 == 0 ? idx1 : idx4
    w1, w2, w3, w4 = FT(w1), FT(w2), FT(w3), FT(w4)
    dwdy1, dwdy2, dwdy3, dwdy4 = FT(dwdy1), FT(dwdy2), FT(dwdy3), FT(dwdy4)
    dwdx1, dwdx2, dwdx3, dwdx4 = FT(dwdx1), FT(dwdx2), FT(dwdx3), FT(dwdx4)

    node1, node2, node3, node4 = model.psfs[idx1], model.psfs[idx2], model.psfs[idx3], model.psfs[idx4]
    d1, d2, d3, d4 = node1.data, node2.data, node3.data, node4.data
    ox1, oy1 = FT(node1.origin.x), FT(node1.origin.y)
    sx1, sy1 = FT(node1.oversampling[1]), FT(node1.oversampling[2])
    fv1 = FT(node1.fill_value)
    ox2, oy2 = FT(node2.origin.x), FT(node2.origin.y)
    sx2, sy2 = FT(node2.oversampling[1]), FT(node2.oversampling[2])
    fv2 = FT(node2.fill_value)
    ox3, oy3 = FT(node3.origin.x), FT(node3.origin.y)
    sx3, sy3 = FT(node3.oversampling[1]), FT(node3.oversampling[2])
    fv3 = FT(node3.fill_value)
    ox4, oy4 = FT(node4.origin.x), FT(node4.origin.y)
    sx4, sy4 = FT(node4.oversampling[1]), FT(node4.oversampling[2])
    fv4 = FT(node4.fill_value)
    ny_d1, nx_d1 = size(d1)
    ny_d2, nx_d2 = size(d2)
    ny_d3, nx_d3 = size(d3)
    ny_d4, nx_d4 = size(d4)

    yr, xr = inds.indices
    y1 = first(yr)
    x1 = first(xr)
    ny = length(yr)
    pv1, pv2, pv3, pv4 = p_val
    pdv1, pdv2, pdv3, pdv4 = p_dpdv
    pdu1, pdu2, pdu3, pdu4 = p_dpdu

    _gridded_corner_bicubic_pass!(pv1, pdv1, pdu1, d1, ox1, oy1, sx1, sy1, fv1, ny_d1, nx_d1, yr, xr, Y, X, y1, x1)
    _gridded_corner_bicubic_pass!(pv2, pdv2, pdu2, d2, ox2, oy2, sx2, sy2, fv2, ny_d2, nx_d2, yr, xr, Y, X, y1, x1)
    _gridded_corner_bicubic_pass!(pv3, pdv3, pdu3, d3, ox3, oy3, sx3, sy3, fv3, ny_d3, nx_d3, yr, xr, Y, X, y1, x1)
    _gridded_corner_bicubic_pass!(pv4, pdv4, pdu4, d4, ox4, oy4, sx4, sy4, fv4, ny_d4, nx_d4, yr, xr, Y, X, y1, x1)

    # Reduction pass: combine the four cached corners via the corner weights
    # (matching `evaluate_fg(::GriddedPSFModel, ...)`'s chain rule), no
    # gathers -- just dense per-pixel reads of the cache matrices above.
    cost = zero(FT)
    b1 = b2 = b3 = b4 = zero(FT)
    A11 = A12 = A13 = A14 = zero(FT)
    A22 = A23 = A24 = zero(FT)
    A33 = A34 = zero(FT)
    A44 = zero(FT)
    LV.@turbo for j in xr, i in yr
        ii = i - y1 + 1
        jj = j - x1 + 1
        s = w1 * pv1[ii, jj] + w2 * pv2[ii, jj] + w3 * pv3[ii, jj] + w4 * pv4[ii, jj]
        dsdY = dwdy1 * pv1[ii, jj] - w1 * sy1 * pdv1[ii, jj] +
            dwdy2 * pv2[ii, jj] - w2 * sy2 * pdv2[ii, jj] +
            dwdy3 * pv3[ii, jj] - w3 * sy3 * pdv3[ii, jj] +
            dwdy4 * pv4[ii, jj] - w4 * sy4 * pdv4[ii, jj]
        dsdX = dwdx1 * pv1[ii, jj] - w1 * sx1 * pdu1[ii, jj] +
            dwdx2 * pv2[ii, jj] - w2 * sx2 * pdu2[ii, jj] +
            dwdx3 * pv3[ii, jj] - w3 * sx3 * pdu3[ii, jj] +
            dwdx4 * pv4[ii, jj] - w4 * sx4 * pdu4[ii, jj]
        f_val = muladd(flux, s, bkg)
        r = f_val - FT(image[i, j])
        k = (j - x1) * ny + (i - y1) + 1
        residuals[k] = r
        wgt = FT(weights[k])
        wr = wgt * r
        cost = muladd(wr, r, cost)
        g1v = flux * dsdY
        g2v = flux * dsdX
        g3v = s
        g4v = one(FT)
        b1 = muladd(wr, g1v, b1)
        b2 = muladd(wr, g2v, b2)
        b3 = muladd(wr, g3v, b3)
        b4 = muladd(wr, g4v, b4)
        A11 = muladd(wgt * g1v, g1v, A11)
        A12 = muladd(wgt * g1v, g2v, A12)
        A13 = muladd(wgt * g1v, g3v, A13)
        A14 = muladd(wgt * g1v, g4v, A14)
        A22 = muladd(wgt * g2v, g2v, A22)
        A23 = muladd(wgt * g2v, g3v, A23)
        A24 = muladd(wgt * g2v, g4v, A24)
        A33 = muladd(wgt * g3v, g3v, A33)
        A34 = muladd(wgt * g3v, g4v, A34)
        A44 = muladd(wgt * g4v, g4v, A44)
    end

    # Project the full 4x4 normal-equation block onto the requested free
    # parameters (identity projection when all 4 parameters are free).
    bfull = (b1, b2, b3, b4)
    Afull = (
        A11, A12, A13, A14,
        A12, A22, A23, A24,
        A13, A23, A33, A34,
        A14, A24, A34, A44,
    )
    nparams = length(free_idx)
    @inbounds for j in 1:nparams
        b[j] = bfull[free_idx[j]]
        for i in 1:nparams
            A[i, j] = Afull[(free_idx[j] - 1) * 4 + free_idx[i]]
        end
    end
    return cost
end
