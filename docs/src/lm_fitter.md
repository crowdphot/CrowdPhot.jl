```@meta
CurrentModule = CrowdPhot
```

# [Levenberg-Marquardt Fitting](@id lm_fitter)

CrowdPhot uses a [Levenberg-Marquardt (LM)](https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm)
optimizer with optional iteratively reweighted least squares (IRLS) for robust PSF fitting.
The core optimizer is generic and model-agnostic; all model-specific work lives in an `accum!`
callback.

## Fit a PSF model to a single star

```@docs
fit_star(::AbstractPSFModel{T}, ::AbstractMatrix, ::Any) where {T}
```

## LM Result

```@docs
LMResult
```

## Damping Strategies

The LM algorithm augments the Gauss-Newton Hessian approximation with a
diagonal damping term. The damping strategy controls how this term is
constructed.

```@docs
MarquardtDamping
LevenbergDamping
NoDamping
```

## Scale Estimators (for IRLS)

When IRLS reweighting is enabled, a robust scale estimator provides the
residual scale $\sigma$ used to compute per-pixel weights.

```@docs
MADScale
FixedScale
MScale
```

## Loss Functions for IRLS

Any loss function supporting the [LossFunctions.jl](https://github.com/JuliaML/LossFunctions.jl)
API can be used. We additionally define [`TukeyLoss`](@ref) which is useful
for outlier rejection.

```@docs
TukeyLoss
weight
```

## Covariance Estimators

When fitting with IRLS, final covariances can either be determined from the
final weights with `ReweightedCovarianceEstimator` or from the initial weights
with `KnownWeightsCovarianceEstimator`. You would typically use
`KnownWeightsCovarianceEstimator` if you had a known-good input inverse variance
weight map.

```@docs
KnownWeightsCovarianceEstimator
ReweightedCovarianceEstimator
```

## API Internals

The main interfaces for defining new optimizations to feed into the LM
optimizer are the `LMProblem` and the `lm_irls` function.

```@docs
LMProblem
lm_irls
```
