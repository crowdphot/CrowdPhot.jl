# Background covariance: the spatial covariance of the background left in a
# residual image, as a parametric function of pixel offset fitted per mesh cell.
# Used by the multipass finalize step (`background_covariance = true`).

"""
    BackgroundCovariance

Per-mesh-cell parametric model of the background covariance at pixel offset
`Δ = (dy, dx)`,

    Σ_B(Δ) = a A(Δ) + b exp(-|Δ|² / 2ℓ²),

fitted by [`_background_covariance`](@ref).  `A` is the PSF autocorrelation
(normalized to `A(0) = 1`) at the cell center: a confusion-like term with the
shape unresolved sources give.  The second term is smooth structure.  `a` and `b`
are variances in the residual's units squared.

# Fields

- `box`: mesh cell size in pixels; cell `(i, j)` covers rows
  `(i - 1) * box + 1 : i * box` and the same for columns.
- `ny`, `nx`: the image size.
- `L`: the lag range, per axis, the parameters were fitted over.  It does not limit
  the tables [`_sigma_b_lags`](@ref) builds.
- `a`, `b`, `ell`: the per-cell parameters.
- `acf`: `acf(y, x)` returns the PSF autocorrelation table at lags `-La:La` per axis,
  indexed from its center, for some `La >= L`; tables of `Σ_B` can reach lag `La`.
"""
struct BackgroundCovariance{T, F}
    box::Int
    ny::Int
    nx::Int
    L::Int
    a::Matrix{T}
    b::Matrix{T}
    ell::Matrix{T}
    acf::F
end

_cell_center(i, box, n) = min(n, (i - 1) * box + (box + 1) ÷ 2)

"""
    _background_covariance(resid, valid, inv_var, box, L, acf; half_width, ells) -> BackgroundCovariance

Fit [`BackgroundCovariance`](@ref) to the residual image `resid`.

For each `box`-pixel mesh cell, the autocovariance of `resid` is measured at
lags up to `L` pixels along the axes and diagonals (the model is isotropic, so
its radial profile is all the fit needs) over a window of the cell and its neighbors reaching
`half_width` pixels from the cell center (rounded to whole cells, at least one
ring), using only pixels where `valid` is `true`: the caller excludes sources'
light, zero-weight pixels and non-finite values.  The window mean is subtracted,
and the mean pixel noise variance `1 / inv_var` over the valid pixels is taken off
the zero lag, so what remains is the background's excess covariance.  `(a, b, ℓ)`
are then a nonnegative least-squares fit of the parametric form to it over all
lags, `ℓ` chosen from the grid `ells` (pixels; by default `2L × {1/3, …, 16/3}`) by
residual.  A cell whose window has no valid pixel pair at some lag gets `a = b = 0`
(no correction).

`L` is only the range of lags fitted.  `acf(y, x)` must return a table at least
`2L + 1` wide, and may return a larger one, which [`_sigma_b_lags`](@ref) can then
use for longer lags than were fitted.

Each pixel pair is counted once per window that covers it: the per-lag sums are
accumulated per cell once and then summed over each window, so the cost is one
pass over the image per lag regardless of the window size.
"""
function _background_covariance(resid::AbstractMatrix, valid::AbstractMatrix{Bool},
        inv_var::AbstractMatrix, box::Integer, L::Integer, acf;
        half_width::Real = 1.5 * box, ells = 2L .* (1 / 3, 2 / 3, 4 / 3, 8 / 3, 16 / 3))
    T = float(eltype(resid))
    ny, nx = size(resid)
    box, L = Int(box), Int(L)
    ncy, ncx = cld(ny, box), cld(nx, box)
    rv = Matrix{Float64}(undef, ny, nx)
    vv = Matrix{Float64}(undef, ny, nx)
    @inbounds for i in eachindex(rv)
        ok = valid[i] && isfinite(resid[i]) && isfinite(inv_var[i]) && inv_var[i] > 0
        rv[i] = ok ? Float64(resid[i]) : 0.0
        vv[i] = ok ? 1.0 : 0.0
    end
    # The model is isotropic, so its radial profile is all the fit needs: lags along
    # the two axes and two diagonals, one half-plane of each since C(Δ) = C(-Δ).
    # That is `4L + 1` lags instead of the `2L² + 2L + 1` of the full half-plane.
    lags = [(0, 0); [(d, 0) for d in 1:L]; [(0, d) for d in 1:L]; [(d, d) for d in 1:L]; [(d, -d) for d in 1:L]]
    nl = length(lags)
    l0 = findfirst(==((0, 0)), lags)
    S = zeros(nl, ncy, ncx); A1 = zeros(nl, ncy, ncx); A2 = zeros(nl, ncy, ncx); NP = zeros(nl, ncy, ncx)
    SR = zeros(ncy, ncx); NV = zeros(ncy, ncx); NN = zeros(ncy, ncx)
    for cj in 1:ncx, ci in 1:ncy
        ys = ((ci - 1) * box + 1):min(ci * box, ny)
        xs = ((cj - 1) * box + 1):min(cj * box, nx)
        sr = 0.0; nv = 0.0; nn = 0.0
        @inbounds for x in xs, y in ys
            if vv[y, x] > 0
                sr += rv[y, x]; nv += 1; nn += 1 / Float64(inv_var[y, x])
            end
        end
        SR[ci, cj], NV[ci, cj], NN[ci, cj] = sr, nv, nn
        nv == 0 && continue
        for (l, (dy, dx)) in enumerate(lags)
            s = 0.0; a1 = 0.0; a2 = 0.0; np = 0.0
            @inbounds for x in max(first(xs), 1 - dx):min(last(xs), nx - dx)
                @simd for y in first(ys):min(last(ys), ny - dy)
                    rp, vp = rv[y, x], vv[y, x]
                    rq, vq = rv[y + dy, x + dx], vv[y + dy, x + dx]
                    s += rp * rq; a1 += rp * vq; a2 += vp * rq; np += vp * vq
                end
            end
            S[l, ci, cj], A1[l, ci, cj], A2[l, ci, cj], NP[l, ci, cj] = s, a1, a2, np
        end
    end
    k = max(1, ceil(Int, (half_width - box / 2) / box))
    a = zeros(T, ncy, ncx); b = zeros(T, ncy, ncx); ell = fill(T(first(ells)), ncy, ncx)
    C = zeros(nl)
    wts = ones(nl)
    Kl = [[exp(-(dy^2 + dx^2) / (2 * l^2)) for (dy, dx) in lags] for l in ells]
    Ap = zeros(nl)
    for cj in 1:ncx, ci in 1:ncy
        wy = max(1, ci - k):min(ncy, ci + k)
        wx = max(1, cj - k):min(ncx, cj + k)
        nv = sum(view(NV, wy, wx))
        nv > 0 || continue
        μ = sum(view(SR, wy, wx)) / nv
        nbar = sum(view(NN, wy, wx)) / nv
        ok = true
        for l in 1:nl
            np = sum(view(NP, l, wy, wx))
            np > 0 || (ok = false; break)
            C[l] = (sum(view(S, l, wy, wx)) - μ * (sum(view(A1, l, wy, wx)) + sum(view(A2, l, wy, wx))) + np * μ^2) / np
        end
        ok || continue
        C[l0] -= nbar
        At = acf(_cell_center(ci, box, ny), _cell_center(cj, box, nx))
        c = _acf_center(At, L)
        for (l, (dy, dx)) in enumerate(lags)
            Ap[l] = At[c + dy, c + dx]
        end
        best = (Inf, 0.0, 0.0, Float64(first(ells)))
        for (il, K) in enumerate(Kl)
            f = _nnls2(Ap, K, C, wts)
            f[3] < best[1] && (best = (f[3], f[1], f[2], Float64(ells[il])))
        end
        a[ci, cj], b[ci, cj], ell[ci, cj] = T(best[2]), T(best[3]), T(best[4])
    end
    return BackgroundCovariance(box, ny, nx, L, a, b, ell, acf)
end

# Weighted nonnegative least squares of `y ≈ α u + β v` by enumerating the active
# sets.  Returns `(α, β, residual)`.
function _nnls2(u, v, y, w)
    suu = suv = svv = suy = svy = syy = 0.0
    @inbounds for i in eachindex(y)
        suu += w[i] * u[i]^2; suv += w[i] * u[i] * v[i]; svv += w[i] * v[i]^2
        suy += w[i] * u[i] * y[i]; svy += w[i] * v[i] * y[i]; syy += w[i] * y[i]^2
    end
    return _nnls2(suu, suv, svv, suy, svy, syy)
end

# The closed-form part, apart so that `res` captures only values assigned once
# (a closure over the accumulators above would box them).
function _nnls2(suu, suv, svv, suy, svy, syy)
    res(α, β) = syy - 2α * suy - 2β * svy + α^2 * suu + 2α * β * suv + β^2 * svv
    best = (0.0, 0.0, syy)
    det = suu * svv - suv^2
    if det > 0
        α = (svv * suy - suv * svy) / det
        β = (suu * svy - suv * suy) / det
        α >= 0 && β >= 0 && return (α, β, res(α, β))
    end
    suu > 0 && (α = max(0.0, suy / suu); r = res(α, 0.0); r < best[3] && (best = (α, 0.0, r)))
    svv > 0 && (β = max(0.0, svy / svv); r = res(0.0, β); r < best[3] && (best = (0.0, β, r)))
    return best
end

# Center index of a square autocorrelation table, which must reach lag `L`.
function _acf_center(At::AbstractMatrix, L::Integer)
    n = size(At, 1)
    size(At, 2) == n && isodd(n) && n >= 2L + 1 ||
        throw(DimensionMismatch("PSF autocorrelation table is $(size(At)); need a square table of odd size at least $(2L + 1)"))
    return n ÷ 2 + 1
end

# Linear index of the mesh cell holding pixel `(y, x)`, clamped to the mesh.
function _cell_id(bc::BackgroundCovariance, y::Integer, x::Integer)
    ncy, ncx = size(bc.a)
    ci = clamp(fld(y - 1, bc.box) + 1, 1, ncy)
    cj = clamp(fld(x - 1, bc.box) + 1, 1, ncx)
    return LinearIndices(bc.a)[ci, cj]
end

"""
    _sigma_b_lags(bc::BackgroundCovariance, y, x, L = bc.L) -> Matrix

The `(2L+1, 2L+1)` table of `Σ_B(Δ)` for the mesh cell containing pixel `(y, x)`,
indexed `[dy + L + 1, dx + L + 1]`.  Every source in a cell uses that cell's fit
and the PSF autocorrelation at the cell center.  `L` may exceed the fitted lag range
`bc.L` up to the reach of `bc.acf`'s tables.  Returns a new matrix.
"""
function _sigma_b_lags(bc::BackgroundCovariance{T}, y::Integer, x::Integer, L::Integer = bc.L) where {T}
    ci, cj = Tuple(CartesianIndices(bc.a)[_cell_id(bc, y, x)])
    a, b, ℓ = bc.a[ci, cj], bc.b[ci, cj], bc.ell[ci, cj]
    out = zeros(T, 2L + 1, 2L + 1)
    (iszero(a) && iszero(b)) && return out
    At = iszero(a) ? nothing : bc.acf(_cell_center(ci, bc.box, bc.ny), _cell_center(cj, bc.box, bc.nx))
    c = At === nothing ? 0 : _acf_center(At, L)
    @inbounds for dx in -L:L, dy in -L:L
        v = iszero(b) ? zero(T) : b * exp(-(dy^2 + dx^2) / (2ℓ^2))
        At === nothing || (v += a * At[c + dy, c + dx])
        out[dy + L + 1, dx + L + 1] = v
    end
    return out
end
