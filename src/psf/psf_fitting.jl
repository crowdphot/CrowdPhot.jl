"""
    free_params(model::AbstractPSFModel, 
        fixed::NamedTuple=NamedTuple()) -> (free_names, free_idx, x0)

Return the names of the free (non-fixed) parameters, their indices, and their initial values as
a `Vector`. `fixed` is a `NamedTuple` whose keys name the fields to freeze.
"""
function free_params(model::AbstractPSFModel{T}, fixed::NamedTuple = NamedTuple()) where {T}
    all_props = ConstructionBase.getproperties(model)
    free_names = Tuple(k for k in keys(all_props) if !haskey(fixed, k))
    free_idx = Tuple(i for (i, k) in enumerate(keys(all_props)) if !haskey(fixed, k))
    x0 = T[all_props[k] for k in free_names]
    return free_names, free_idx, x0
end

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
    # Validate weights and model capabilities before allocating LM work buffers.
    if !isnothing(inv_var)
        if size(inv_var) != size(image)
            throw(ArgumentError("`inv_var` must be the same size as `image`"))
        end
        if !all(x -> isfinite(x) && x > 0, inv_var)
            throw(ArgumentError("`inv_var` must be finite and > 0 everywhere"))
        end
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
    fit_inds = CartesianIndices(inds)
    free_names, free_idx, x0 = free_params(model, fixed)
    n = length(x0)
    n > 0 || throw(ArgumentError("all model parameters are fixed; nothing to fit"))
    dof = length(fit_inds) - n
    if dof < 0
        throw(
            ArgumentError(
                "degrees of freedom must be positive; " *
                    "too many free parameters ($n) for the number of pixels ($(length(fit_inds)))"
            )
        )
    end

    # Restrict inverse-variance weights to the fit pixels once for compact LM accumulation.
    FT = float(T)
    base_weights = if isnothing(inv_var)
        nothing
    else
        weights = Vector{FT}(undef, length(fit_inds))
        base_k = 0
        @inbounds for idx in fit_inds
            base_k += 1
            weights[base_k] = FT(inv_var[idx])
        end
        weights
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
        max_iter=200, x_tol=1e-8, f_tol=1e-8, g_tol=1e-8,
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
- `x_tol`: step-norm convergence criterion:
  ``\\|\\delta\\| \\le x\\_tol\\cdot(\\|x\\|+x\\_tol)`` (default `1e-8`)
- `f_tol`: cost-decrease convergence criterion:
  ``\\Delta C \\le f\\_tol\\cdot(|C|+f\\_tol)`` (default `1e-8`)
- `g_tol`: gradient-norm convergence criterion ``\\|J^\\top W r\\|``
  (default `1e-8`)
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
        f_tol::Real = 1.0e-8,
        g_tol::Real = 1.0e-8,
        show_trace::Bool = false,
        reweight::Union{Nothing, LossFunctions.SupervisedLoss} = nothing,
        scale_estimator::Union{Nothing, AbstractScaleEstimator} = nothing,
        weight_reset_tol::Real = 0.1,
        covariance_estimator::Union{Nothing, AbstractCovarianceEstimator} = nothing
    ) where {T}

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

    problem = LMProblem(Vector{FT}(x0), length(inds), accum!, base_weights)
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
    best_model = model_from_vector(model, free_names_val, result.minimizer, fixed)
    return best_model, result
end

################################################################################
# Custom `fit_star` methods for specific PSF models with optimized accumulators.
################################################################################

# Specialized method: 2--4x faster than generic.
function fit_star(
        model::CircularGaussianPSF{T},
        image::AbstractMatrix,
        inds = axes(image);
        fixed::NamedTuple = (;),
        inv_var = nothing,
        kws...
    ) where {T}

    # Reuse the generic preparation so this dispatch path preserves validation semantics.
    prepared = _prepare_fit_star_inputs(model, image, inds, fixed, inv_var)
    inds, _, free_idx, x0, free_names_val, FT, base_weights = prepared


    # Stream pixels through a CircularGaussian-specific normal-equation kernel.
    # The all-parameters case is fully scalarized; fixed-parameter fits use the
    # same scalar derivatives and project them onto the free parameter subset.
    function accum!(A::AbstractMatrix{FT}, b::AbstractVector{FT}, residuals::AbstractVector{FT}, x::AbstractVector{FT}, weights) where {FT}
        @assert size(A, 1) == size(A, 2) == length(b) == length(free_idx)
        @assert length(residuals) == length(inds)
        m = model_from_vector(model, free_names_val, x, fixed)
        return _accum_circular_gaussian!(A, b, residuals, image, inds, m, free_idx, weights)
    end

    # Delegate iteration, IRLS reweighting, damping, and covariance estimation
    # to the shared LM engine to keep convergence behavior aligned.
    problem = LMProblem(Vector{FT}(x0), length(inds), accum!, base_weights)
    result = lm_irls(problem; kws...)
    best_model = model_from_vector(model, free_names_val, result.minimizer, fixed)
    return best_model, result
end

function _accum_circular_gaussian!(
        A::AbstractMatrix{FT},
        b::AbstractVector{FT},
        residuals::AbstractVector{FT},
        image::AbstractMatrix,
        inds::CartesianIndices,
        model::CircularGaussianPSF,
        free_idx,
        weights
    ) where {FT}

    # Precompute model constants exactly as in the full-parameter path.
    fill!(A, zero(FT))
    fill!(b, zero(FT))
    γ = FT(GAUSS_PRE)
    x0 = FT(model.x)
    y0 = FT(model.y)
    fwhm = FT(model.fwhm)
    fwhm² = fwhm^2
    norm = -(FT(π) * fwhm² / γ)
    amp = FT(model.flux) / norm
    bkg = FT(model.bkg)
    γ_f2 = γ / fwhm²

    # Compute scalar derivatives, then project them onto the requested free
    # parameter subset for fixed-parameter fits.
    cost = zero(FT)
    obs_k = 0
    nparams = length(free_idx)
    use_weights = !isnothing(weights)
    @inbounds for idx in inds
        obs_k += 1
        w = use_weights ? FT(weights[obs_k]) : one(FT)
        dx = FT(idx[1]) - x0
        dy = FT(idx[2]) - y0
        sqmahab = (dx^2 + dy^2) / fwhm²
        g = exp(γ * sqmahab)
        Ag = amp * g
        f_val = muladd(amp, g, bkg)
        r = f_val - FT(image[idx])
        residuals[obs_k] = r
        wr = w * r
        cost = muladd(wr, r, cost)
        g_full = (
            -2 * Ag * γ_f2 * dx,
            -2 * Ag * γ_f2 * dy,
            -2 * Ag * (1 + γ * sqmahab) / fwhm,
            g / norm,
            one(FT),
        )

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

# Specialized method: 9.8x faster at (5,5), 5.6x at (11,11), and 3.3x at (21,21) over generic method.
function fit_star(
        model::GaussianPSF{T},
        image::AbstractMatrix,
        inds = axes(image);
        fixed::NamedTuple = (;),
        inv_var = nothing,
        kws...
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
    problem = LMProblem(Vector{FT}(x0), length(inds), accum!, base_weights)
    result = lm_irls(problem; kws...)
    best_model = model_from_vector(model, free_names_val, result.minimizer, fixed)
    return best_model, result
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
        dx = FT(idx[1]) - x0
        dy = FT(idx[2]) - y0
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

        # Match evaluate_fg's parameter order: x, y, x_fwhm, y_fwhm, theta, flux, bkg.
        D = one(FT) / ax² - one(FT) / ay²
        Qx = -2 * (cs * u / ax² - sn * v / ay²)
        Qy = -2 * (sn * u / ax² + cs * v / ay²)
        Qax = -2 * u^2 / ax^3
        Qay = -2 * v^2 / ay^3
        Qtheta = degree * 2 * u * v * D
        g_full = (
            Ag * γ * Qx,
            Ag * γ * Qy,
            Ag * (-one(FT) / ax + γ * Qax),
            Ag * (-one(FT) / ay + γ * Qay),
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

# Specialized method: 9.7x faster at (5,5), 3.7x at (11,11), and 1.9x at (21,21) over generic method.
function fit_star(
        model::CircularGaussianPRF{T},
        image::AbstractMatrix,
        inds = axes(image);
        fixed::NamedTuple = (;),
        inv_var = nothing,
        kws...
    ) where {T}

    # Reuse shared validation and fixed-parameter bookkeeping.
    prepared = _prepare_fit_star_inputs(model, image, inds, fixed, inv_var)
    inds, _, free_idx, x0, free_names_val, FT, base_weights = prepared

    # Stream pixels through a scalar circular-PRF normal-equation kernel.
    function accum!(A::AbstractMatrix{FT}, b::AbstractVector{FT}, residuals::AbstractVector{FT}, x::AbstractVector{FT}, weights) where {FT}
        @assert size(A, 1) == size(A, 2) == length(b) == length(free_idx)
        @assert length(residuals) == length(inds)
        m = model_from_vector(model, free_names_val, x, fixed)
        return _accum_circular_gaussian_prf!(A, b, residuals, image, inds, m, free_idx, weights)
    end

    # Delegate iteration details to the shared LM implementation.
    problem = LMProblem(Vector{FT}(x0), length(inds), accum!, base_weights)
    result = lm_irls(problem; kws...)
    best_model = model_from_vector(model, free_names_val, result.minimizer, fixed)
    return best_model, result
end

function _accum_circular_gaussian_prf!(
        A::AbstractMatrix{FT},
        b::AbstractVector{FT},
        residuals::AbstractVector{FT},
        image::AbstractMatrix,
        inds::CartesianIndices,
        model::CircularGaussianPRF,
        free_idx,
        weights
    ) where {FT}

    # Precompute model constants used by the integrated Gaussian factors.
    fill!(A, zero(FT))
    fill!(b, zero(FT))
    x0 = FT(model.x)
    y0 = FT(model.y)
    fwhm = FT(model.fwhm)
    flux = FT(model.flux)
    bkg = FT(model.bkg)
    α = 2 * sqrt(FT(log(2))) / fwhm
    two_sqrtpi = 2 / sqrt(FT(π))
    fl4 = flux / 4

    # Evaluate difference-of-erf derivatives and accumulate the active block.
    cost = zero(FT)
    obs_k = 0
    nparams = length(free_idx)
    use_weights = !isnothing(weights)
    @inbounds for idx in inds
        obs_k += 1
        w = use_weights ? FT(weights[obs_k]) : one(FT)
        dx = FT(idx[1]) - x0
        dy = FT(idx[2]) - y0
        u_p = α * (dx + FT(0.5))
        u_m = α * (dx - FT(0.5))
        v_p = α * (dy + FT(0.5))
        v_m = α * (dy - FT(0.5))
        Ex = erf(u_p) - erf(u_m)
        Ey = erf(v_p) - erf(v_m)
        f_val = muladd(fl4, Ex * Ey, bkg)
        r = f_val - FT(image[idx])
        residuals[obs_k] = r
        wr = w * r
        cost = muladd(wr, r, cost)

        # Match evaluate_fg's parameter order: x, y, fwhm, flux, bkg.
        Gxp = two_sqrtpi * exp(-u_p^2)
        Gxm = two_sqrtpi * exp(-u_m^2)
        Gyp = two_sqrtpi * exp(-v_p^2)
        Gym = two_sqrtpi * exp(-v_m^2)
        g_full = (
            fl4 * Ey * α * (Gxm - Gxp),
            fl4 * Ex * α * (Gym - Gyp),
            fl4 / fwhm * ((Gxm * u_m - Gxp * u_p) * Ey + Ex * (Gym * v_m - Gyp * v_p)),
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

# Specialized method: 7.5x faster at (5,5), 3.4x at (11,11), and 1.9x at (21,21) over generic method.
function fit_star(
        model::GaussianPRF{T},
        image::AbstractMatrix,
        inds = axes(image);
        fixed::NamedTuple = (;),
        inv_var = nothing,
        kws...
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
    problem = LMProblem(Vector{FT}(x0), length(inds), accum!, base_weights)
    result = lm_irls(problem; kws...)
    best_model = model_from_vector(model, free_names_val, result.minimizer, fixed)
    return best_model, result
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
        dx = FT(idx[1]) - x0
        dy = FT(idx[2]) - y0
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

        # Match evaluate_fg's parameter order: x, y, x_fwhm, y_fwhm, theta, flux, bkg.
        Gxp = two_sqrtpi * exp(-u_p^2)
        Gxm = two_sqrtpi * exp(-u_m^2)
        Gyp = two_sqrtpi * exp(-v_p^2)
        Gym = two_sqrtpi * exp(-v_m^2)
        dEx_du = αx * (Gxm - Gxp)
        dEy_dv = αy * (Gym - Gyp)
        g_full = (
            fl4 * (cs * dEx_du * Ey - sn * dEy_dv * Ex),
            fl4 * (sn * dEx_du * Ey + cs * dEy_dv * Ex),
            fl4 / ax * (Gxm * u_m - Gxp * u_p) * Ey,
            fl4 / ay * Ex * (Gym * v_m - Gyp * v_p),
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
function fit_star(
        model::CircularMoffatPSF{T},
        image::AbstractMatrix,
        inds = axes(image);
        fixed::NamedTuple = (;),
        inv_var = nothing,
        kws...
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
    problem = LMProblem(Vector{FT}(x0), length(inds), accum!, base_weights)
    result = lm_irls(problem; kws...)
    best_model = model_from_vector(model, free_names_val, result.minimizer, fixed)
    return best_model, result
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
        dx = FT(idx[1]) - x0
        dy = FT(idx[2]) - y0
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
            2 * Ag * β * dx / (α² * u),
            2 * Ag * β * dy / (α² * u),
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
function fit_star(
        model::MoffatPSF{T},
        image::AbstractMatrix,
        inds = axes(image);
        fixed::NamedTuple = (;),
        inv_var = nothing,
        kws...
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
    problem = LMProblem(Vector{FT}(x0), length(inds), accum!, base_weights)
    result = lm_irls(problem; kws...)
    best_model = model_from_vector(model, free_names_val, result.minimizer, fixed)
    return best_model, result
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
        dx = FT(idx[1]) - x0
        dy = FT(idx[2]) - y0
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

        # Match evaluate_fg's parameter order: x, y, x_α, y_α, theta, β, flux, bkg.
        D = one(FT) / ax² - one(FT) / ay²
        Qx = -2 * (cs * u / ax² - sn * v / ay²)
        Qy = -2 * (sn * u / ax² + cs * v / ay²)
        Qax = -2 * u^2 / ax^3
        Qay = -2 * v^2 / ay^3
        Qtheta = degree * 2 * u * v * D
        scale = -β / h
        g_full = (
            Ag * scale * Qx,
            Ag * scale * Qy,
            Ag * (-one(FT) / ax + scale * Qax),
            Ag * (-one(FT) / ay + scale * Qay),
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
function fit_star(
        model::AiryPSF{T},
        image::AbstractMatrix,
        inds = axes(image);
        fixed::NamedTuple = (;),
        inv_var = nothing,
        kws...
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
    problem = LMProblem(Vector{FT}(x0), length(inds), accum!, base_weights)
    result = lm_irls(problem; kws...)
    best_model = model_from_vector(model, free_names_val, result.minimizer, fixed)
    return best_model, result
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
        dx = FT(idx[1]) - x0
        dy = FT(idx[2]) - y0
        r = sqrt(dx^2 + dy^2)
        u = FT(π) * r / a
        if abs(u) < eps(FT)
            A2 = one(FT)
            dA2_du = zero(FT)
        else
            J0 = besselj0(u)
            J1 = besselj1(u)
            J2 = besselj(2, u)
            airy = 2 * J1 / u
            airy_p = (u * (J0 - J2) - 2 * J1) / (u^2)
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
            g_full = (df_dx, df_dy, df_dradius, df_dflux, one(FT))
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
function fit_star(
        model::ImagePSF{T},
        image::AbstractMatrix,
        inds = axes(image);
        fixed::NamedTuple = (;),
        inv_var = nothing,
        kws...
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
    problem = LMProblem(Vector{FT}(x0), length(inds), accum!, base_weights)
    result = lm_irls(problem; kws...)
    best_model = model_from_vector(model, free_names_val, result.minimizer, fixed)
    return best_model, result
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
    ox = FT(model.origin[1])
    oy = FT(model.origin[2])
    fill_value = FT(model.fill_value)

    # Compute all four derivatives, then accumulate only the requested block.
    cost = zero(FT)
    obs_k = 0
    nparams = length(free_idx)
    use_weights = !isnothing(weights)
    @inbounds for idx in inds
        obs_k += 1
        w = use_weights ? FT(weights[obs_k]) : one(FT)
        u = sx * (FT(idx[1]) - x0) + ox
        v = sy * (FT(idx[2]) - y0) + oy
        p, dpdu, dpdv = bicubic_interpolate(data, u, v; fill_value)
        profile = FT(p)
        r = muladd(flux, profile, bkg) - FT(image[idx])
        residuals[obs_k] = r
        wr = w * r
        cost = muladd(wr, r, cost)
        g_full = (
            -flux * sx * FT(dpdu),
            -flux * sy * FT(dpdv),
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
