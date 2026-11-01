# Damped Gauss-Newton steps on the equilibrated normal equations: CG with a gather SpMV over
# the block-sparse H, all vectors on the device.  On a GPU the CG scalars stay on the device
# too (a host round trip per dot product costs ~0.35 ms on Metal, more than the iteration).

"""
    JointFitOptions(; kwargs...)

Fitter settings, mirroring the single-image simultaneous fitter's keywords.

- `R_fit = 3`: fit-stamp half-width (pixels)
- `model_rad = :auto`, `model_rad_nsigma = 0.5`, `R_cap = nothing`: model half-width rule, as in the
  single-image fitter (`R_cap` is its `model_rad_max`).  `R_cap = nothing` caps the radius at the
  PSFs' extent: a tabulated PSF's stamp half-width, ±10 FWHM for an
  analytic one.  `model_rad_nsigma = 0.25` renders about 30% more pixels than 0.5 for better faint-star
  errors: on simulated fields improves faint-star flux scatter by a few percent.
  Wall time on a full 8-band WFI detector with PSF refinement (Metal): 0.25 is 10% slower than 0.5.
- `linearizations = 1`: Gauss-Newton linearizations per pass
- `cg_iterations = 10`: CG iterations per solve (fixed; no early stop)
- `max_trials = 8`, `λ_init = 1e-3`, `λ_up = 10`, `λ_down = 10`, `λ_min = 1e-12`, `λ_max = 1e12`: damping schedule
- `max_step = 1.0`: per-source position step cap (frame pixels)
- `stamp_budget = 100_000_000`: bytes of derivative stamps held at once (images are batched, never
  fewer than one image's stamps).  It trades memory only: on simulated Roman stacks the
  linearization time was flat (within ~10%) from 17 MB to 1.07 GB.
"""
Base.@kwdef struct JointFitOptions
    R_fit::Int = 3
    model_rad::Union{Symbol, Float64} = :auto
    model_rad_nsigma::Float64 = 0.5
    R_cap::Union{Nothing, Int} = nothing
    linearizations::Int = 1
    cg_iterations::Int = 10
    max_trials::Int = 8
    λ_init::Float64 = 1.0e-3
    λ_up::Float64 = 10.0
    λ_down::Float64 = 10.0
    λ_min::Float64 = 1.0e-12
    λ_max::Float64 = 1.0e12
    max_step::Float64 = 1.0
    stamp_budget::Int = 100_000_000
end

@inline loadblock(H, q, ::Val{P}) where {P} = SMatrix{P, P}(ntuple(k -> H[(q - 1) * P * P + k], Val(P * P)))
@inline loadvec(x, s, ::Val{P}) where {P} = SVector(ntuple(c -> x[(s - 1) * P + c], Val(P)))

# Every CG kernel spans the whole vector.  The two dot products of an iteration come out of the
# kernels that produce their operands, as per-workgroup partial sums that a small one-workgroup
# kernel adds into the device scalar.  (A single workgroup striding over all n was the
# bottleneck: 0.41 ms per dot on Metal at n = 63k, 13 ms at 1e7.)  Kernels that @synchronize
# run on whole groups, so their ndrange is padded and items past the end contribute zero.
const CG_GROUP = 256

# y = (H + λI) x, gathering row i's upper slots and, transposed, the slots where i is the column;
# part[g] = Σ over the group's sources of x_i ⋅ y_i.
@kernel function hmul_kernel!(y, part, @Const(x), @Const(Hd), @Const(Ho), @Const(ptr), @Const(nbr), @Const(cptr), @Const(cslot),
                              @Const(row), λ, N, vp::Val{P}) where {P}
    i = @index(Global); li = @index(Local); g = @index(Group)
    sh = @localmem eltype(y) (CG_GROUP,)
    d = zero(eltype(y))
    if i <= N
        xi = loadvec(x, i, vp)
        acc = λ * xi + loadblock(Hd, i, vp) * xi
        for q in ptr[i]:(ptr[i + 1] - 1)
            acc += loadblock(Ho, q, vp) * loadvec(x, nbr[q], vp)
        end
        for k in cptr[i]:(cptr[i + 1] - 1)
            q = cslot[k]
            acc += loadblock(Ho, q, vp)' * loadvec(x, row[q], vp)
        end
        for r in 1:P
            y[(i - 1) * P + r] = acc[r]
        end
        d = xi' * acc
    end
    sh[li] = d
    @synchronize
    li <= 128 && (sh[li] += sh[li + 128])
    @synchronize
    li <= 64 && (sh[li] += sh[li + 64])
    @synchronize
    li <= 32 && (sh[li] += sh[li + 32])
    @synchronize
    li <= 16 && (sh[li] += sh[li + 16])
    @synchronize
    li <= 8 && (sh[li] += sh[li + 8])
    @synchronize
    li <= 4 && (sh[li] += sh[li + 4])
    @synchronize
    li <= 2 && (sh[li] += sh[li + 2])
    @synchronize
    li == 1 && (part[g] = sh[1] + sh[2])
end

# x += α p, r -= α q (α = s[a] / s[2]) and part[g] = the group's Σ r²; with `init`, r = p = -g ./ D
# instead (x untouched) and the same Σ r².  Once CG has converged (p = 0, so pᵀ(H̃ + λI)p = 0) α is 0, not
# 0/0: with heavy damping CG converges in a few of its fixed iterations, the residual underflows, and a
# GPU that flushes subnormals to zero (Metal) turned the whole step into NaN.
@kernel function cg_xr_kernel!(x, r, p, part, @Const(q), @Const(g0), @Const(D), @Const(s), a, n, init::Bool)
    i = @index(Global); li = @index(Local); g = @index(Group)
    sh = @localmem eltype(x) (CG_GROUP,)
    d = zero(eltype(x))
    if i <= n
        if init
            v = -g0[i] / D[i]
            r[i] = v; p[i] = v; d = v * v
        else
            α = s[2] > 0 ? s[a] / s[2] : zero(eltype(x))
            x[i] += α * p[i]
            v = r[i] - α * q[i]
            r[i] = v; d = v * v
        end
    end
    sh[li] = d
    @synchronize
    li <= 128 && (sh[li] += sh[li + 128])
    @synchronize
    li <= 64 && (sh[li] += sh[li + 64])
    @synchronize
    li <= 32 && (sh[li] += sh[li + 32])
    @synchronize
    li <= 16 && (sh[li] += sh[li + 16])
    @synchronize
    li <= 8 && (sh[li] += sh[li + 8])
    @synchronize
    li <= 4 && (sh[li] += sh[li + 4])
    @synchronize
    li <= 2 && (sh[li] += sh[li + 2])
    @synchronize
    li == 1 && (part[g] = sh[1] + sh[2])
end

# s[k] = Σ part, in one workgroup (part has one entry per group of a wide kernel: few).
@kernel function cg_sum_kernel!(s, k, @Const(part), n)
    li = @index(Local)
    sh = @localmem eltype(s) (CG_GROUP,)
    acc = zero(eltype(s)); t = li
    while t <= n
        acc += part[t]; t += CG_GROUP
    end
    sh[li] = acc
    @synchronize
    li <= 128 && (sh[li] += sh[li + 128])
    @synchronize
    li <= 64 && (sh[li] += sh[li + 64])
    @synchronize
    li <= 32 && (sh[li] += sh[li + 32])
    @synchronize
    li <= 16 && (sh[li] += sh[li + 16])
    @synchronize
    li <= 8 && (sh[li] += sh[li + 8])
    @synchronize
    li <= 4 && (sh[li] += sh[li + 4])
    @synchronize
    li <= 2 && (sh[li] += sh[li + 2])
    @synchronize
    li == 1 && (s[k] = sh[1] + sh[2])
end

# p = r + β p, β = |r_new|² / |r_old|² (0 once the residual is 0; see `cg_xr_kernel!`)
@kernel function cg_p_kernel!(p, @Const(r), @Const(s), a, b)
    i = @index(Global)
    β = s[a] > 0 ? s[b] / s[a] : zero(eltype(p))
    p[i] = r[i] + β * p[i]
end

"CG vectors, device scalars and the partial sums of the wide kernels, for `n = P N` unknowns."
struct CGWork{V}
    x::V
    r::V
    p::V
    q::V
    s::V
    ph::V
    pr::V
end
CGWork(be, ::Type{T}, n, N) where {T} =
    CGWork((KA.zeros(be, T, n) for _ in 1:4)..., KA.zeros(be, T, 3), KA.zeros(be, T, cld(N, CG_GROUP)), KA.zeros(be, T, cld(n, CG_GROUP)))

_cg_sum!(s, k, part, be) = cg_sum_kernel!(be, CG_GROUP)(s, k, part, length(part); ndrange = CG_GROUP)
_padded(n) = cld(n, CG_GROUP) * CG_GROUP

# y = D (θc - θ) (the step in the equilibrated system) and part[g] = the group's Σ g (θc - θ).
@kernel function pred_y_kernel!(y, part, @Const(θc), @Const(θ), @Const(g0), @Const(D), n)
    i = @index(Global); li = @index(Local); g = @index(Group)
    sh = @localmem eltype(y) (CG_GROUP,)
    d = zero(eltype(y))
    if i <= n
        δ = θc[i] - θ[i]
        y[i] = D[i] * δ; d = g0[i] * δ
    end
    sh[li] = d
    @synchronize
    li <= 128 && (sh[li] += sh[li + 128])
    @synchronize
    li <= 64 && (sh[li] += sh[li + 64])
    @synchronize
    li <= 32 && (sh[li] += sh[li + 32])
    @synchronize
    li <= 16 && (sh[li] += sh[li + 16])
    @synchronize
    li <= 8 && (sh[li] += sh[li + 8])
    @synchronize
    li <= 4 && (sh[li] += sh[li + 4])
    @synchronize
    li <= 2 && (sh[li] += sh[li + 2])
    @synchronize
    li == 1 && (part[g] = sh[1] + sh[2])
end

"""
The cost change the linearization predicts for the step `θc - θ` (as taken, after the position cap): 2 gᵀδ + δᵀHδ with the
undamped H, the second term as yᵀH̃y for y = D δ.  Uses the CG vectors r and q, free once the step is applied; one host read.
"""
function predicted_change(w::CGWork, lin::Linearization, θc, θ, be)
    n = length(w.x); P, N = size(lin.D); T = eltype(w.x)
    pred_y_kernel!(be, CG_GROUP)(w.r, w.pr, vec(θc), vec(θ), vec(lin.g), vec(lin.D), n; ndrange = _padded(n))
    _cg_sum!(w.s, 1, w.pr, be)
    hmul_kernel!(be, CG_GROUP)(w.q, w.ph, w.r, lin.Hd, lin.Ho, lin.pat.ptr, lin.pat.nbr, lin.pat.cptr, lin.pat.cslot, lin.pat.row,
                               zero(T), N, Val(P); ndrange = _padded(N))
    _cg_sum!(w.s, 2, w.ph, be)
    s = Array(w.s)
    return 2 * Float64(s[1]) + Float64(s[2])
end

"`itmax` CG iterations on (H̃ + λI) x = -g ./ D from x = 0 (same recurrences as Krylov.cg)."
function cg!(w::CGWork, lin::Linearization, λ, itmax, be)
    n = length(w.x); P, N = size(lin.D); T = eltype(w.x)
    fill!(w.x, zero(T))
    a, b = 1, 3
    cg_xr_kernel!(be, CG_GROUP)(w.x, w.r, w.p, w.pr, w.q, vec(lin.g), vec(lin.D), w.s, a, n, true; ndrange = _padded(n))
    _cg_sum!(w.s, a, w.pr, be)
    for _ in 1:itmax
        hmul_kernel!(be, CG_GROUP)(w.q, w.ph, w.p, lin.Hd, lin.Ho, lin.pat.ptr, lin.pat.nbr, lin.pat.cptr, lin.pat.cslot, lin.pat.row,
                                   T(λ), N, Val(P); ndrange = _padded(N))
        _cg_sum!(w.s, 2, w.ph, be)
        cg_xr_kernel!(be, CG_GROUP)(w.x, w.r, w.p, w.pr, w.q, vec(lin.g), vec(lin.D), w.s, a, n, false; ndrange = _padded(n))
        _cg_sum!(w.s, b, w.pr, be)
        launch!(cg_p_kernel!, be, n, w.p, w.r, w.s, a, b)
        a, b = b, a
    end
    return w.x
end

# θc = θ + x ./ D, with each source's (u, v) step capped at `max_step`.
@kernel function step_kernel!(θc, @Const(θ), @Const(x), @Const(D), max_step)
    s = @index(Global)
    P = size(θ, 1)
    b0 = (s - 1) * P
    d1 = x[b0 + 1] / D[1, s]; d2 = x[b0 + 2] / D[2, s]
    d = sqrt(d1 * d1 + d2 * d2)
    sc = d > max_step ? max_step / d : one(d)
    θc[1, s] = θ[1, s] + d1 * sc
    θc[2, s] = θ[2, s] + d2 * sc
    for k in 3:P
        θc[k, s] = θ[k, s] + x[b0 + k] / D[k, s]
    end
end

"""
    fit_pass(sc, θ, rt, o, λ; n_new = 0, work = LinearizeWork(sc)) -> (; θ, λ, lin, stats)

`o.linearizations` damped Gauss-Newton steps from `θ` with the single-image fitter's
schedule: `λ` floored at `λ_init` when `n_new` sources were just added, divided by `λ_down` on
an accepted step, multiplied by `λ_up` on a rejected one, up to `max_trials` per
linearization.  Trial costs use the linearization's anchors, model radii and cost support.
`lin` is the last linearization (H at the start of the last accepted step).  `stats` holds the
first and last trial's gain ratio (actual / predicted cost change, [`predicted_change`](@ref); ~1
where the linearization describes the cost, ≤ 0 for a rejected step whose predicted decrease did not
happen) as `gain_first` and `gain_last` (NaN without trials).
"""
function fit_pass(sc::JointScene{T}, θ::AbstractMatrix{T}, rt::RadiusTable, o::JointFitOptions, λ::Real;
        n_new::Integer = 0, work::LinearizeWork = LinearizeWork(sc)) where {T}
    be = sc.backend; P, N = size(θ)
    n_new > 0 && (λ = max(λ, o.λ_init))
    cgw = CGWork(be, T, P * N, N)
    θ = copy(θ); θc = similar(θ)
    lin = nothing; n_trials = 0; n_accepted = 0; cost_start = NaN; cost = NaN
    t_lin = t_solve = t_trial = 0.0
    gain_first = gain_last = NaN
    for l in 1:o.linearizations
        # synchronize so t_lin holds linearize's kernels, not t_solve (cg!'s first reduction waits for them anyway)
        t_lin += @elapsed (lin = linearize(sc, θ, rt, work; o.R_fit, o.stamp_budget); KA.synchronize(be))
        cost = lin.cost
        l == 1 && (cost_start = cost)
        accepted = false
        for _ in 1:o.max_trials
            t_solve += @elapsed begin
                x = cg!(cgw, lin, λ, o.cg_iterations, be)
                launch!(step_kernel!, be, N, θc, θ, x, lin.D, T(o.max_step))
                pred = predicted_change(cgw, lin, θc, θ, be)
            end
            t_trial += @elapsed begin
                render!(work.model, sc, θc, lin.fp)
                cc = cost!(work.resid, sc, work.model, work.mask, work.part, work.bkg)
            end
            n_trials += 1
            gain_last = (cc - cost) / pred; isnan(gain_first) && (gain_first = gain_last)
            if cc < cost
                θ, θc = θc, θ
                cost = cc
                λ = max(λ / o.λ_down, o.λ_min)
                accepted = true; n_accepted += 1
                break
            end
            λ = min(λ * o.λ_up, o.λ_max)
        end
        accepted || break
    end
    stats = (; n_trials, n_accepted, cost_start, cost_end = cost, gain_first, gain_last, t_lin, t_solve, t_trial)
    return (; θ, λ, lin, stats)
end
