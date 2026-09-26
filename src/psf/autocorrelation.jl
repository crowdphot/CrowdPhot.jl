# PSF autocorrelation at integer pixel lags: the shape of the confusion-like term of
# the background covariance model used by the multipass finalize step.

# Autocorrelation table for a PSF whose shape does not depend on position.
struct _StaticACF{T}
    A::Matrix{T}
end

# Cross-correlation tables of every pair of nodes that share a grid square, keyed
# `(i, j)` with `i <= j`; a position's table is their bilinear-weighted combination.
struct _GriddedACF{T, M <: GriddedPSFModel}
    psf::M
    C::Dict{Tuple{Int, Int}, Matrix{T}}
    L::Int
end

"""
    _psf_acf(psf, L, R_crop; oversampling = 4) -> acf

Prepare the autocorrelation of `psf`'s unit-flux profile at integer pixel lags
`-L:L` in each axis,

    A(Δ) = ∫ p(u) p(u + Δ) du,

with `p(u)` the model's value for a pixel whose center lies `u` from the source,
integrated over sub-pixel `u` on a grid `oversampling` times finer than the
pixels.  Averaging over sub-pixel phase is the right shape for the covariance of
randomly placed faint sources.  The profile is cropped to half-width `R_crop`
pixels about its center before correlating: the squared profile is far more
concentrated than the light, so a crop at `fit_rad` changes the shape negligibly
(< 0.3% in the resulting flux variance for Roman WFI ePSFs at `R_crop >= 3`) at a
small fraction of the cost.

A [`GriddedPSFModel`](@ref) is a bilinear blend `p = Σ wᵢ pᵢ` of its nodes, so its
autocorrelation at any position is exactly `Σᵢⱼ wᵢ wⱼ Cᵢⱼ`, with `Cᵢⱼ` the
cross-correlation of nodes `i` and `j`.  Those tables are computed here, once, for
every pair of nodes sharing a grid square.  When every node is an
[`ImagePSF`](@ref) with the same size, origin and oversampling and a zero
`fill_value`, the tables come straight from the nodes' tabulated data at their
own oversampling; otherwise each node is sampled like any other model.

Returns an object for [`_psf_acf_at`](@ref).
"""
function _psf_acf(psf::AbstractPSFModel, L::Integer, R_crop::Integer; oversampling::Integer = 4)
    T = float(eltype(psf))
    m = ConstructionBase.setproperties(psf, (; y = zero(T), x = zero(T), flux = one(T), bkg = zero(T)))
    f = _acf_samples(m, R_crop, oversampling)
    return _StaticACF(_cross_corr(f, f, Int(L), oversampling, oversampling))
end

function _psf_acf(psf::GriddedPSFModel{T}, L::Integer, R_crop::Integer; oversampling::Integer = 4) where {T}
    nodes = psf.psfs
    ig = psf.index_grid
    samples, sy, sx = if _acf_tabulated(nodes)
        n1 = first(nodes)
        s_x, s_y = n1.oversampling   # `ImagePSF` stores `(x, y)` oversampling
        oy, ox = round(Int, n1.origin.y), round(Int, n1.origin.x)
        hy, hx = s_y * Int(R_crop), s_x * Int(R_crop)
        ny, nx = size(n1.data)
        yr = max(1, oy - hy):min(ny, oy + hy)
        xr = max(1, ox - hx):min(nx, ox + hx)
        [n.data[yr, xr] for n in nodes], s_y, s_x
    else
        [_acf_samples(_recenter_unit(n, zero(T), zero(T)), R_crop, oversampling) for n in nodes],
            oversampling, oversampling
    end
    C = Dict{Tuple{Int, Int}, Matrix{T}}()
    for iy in 1:max(1, size(ig, 1) - 1), ix in 1:max(1, size(ig, 2) - 1)
        sq = unique(vec(ig[iy:min(iy + 1, end), ix:min(ix + 1, end)]))
        for a in sq, b in sq
            a <= b || continue
            haskey(C, (a, b)) || (C[(a, b)] = _cross_corr(samples[a], samples[b], Int(L), sy, sx))
        end
    end
    return _GriddedACF(psf, C, Int(L))
end

_acf_tabulated(nodes) = all(n -> n isa ImagePSF && iszero(n.fill_value), nodes) &&
    all(n -> size(n.data) == size(first(nodes).data) && n.origin == first(nodes).origin &&
        n.oversampling == first(nodes).oversampling, nodes)

# `p(u)` for sub-pixel offsets `u = k / os`, `|k| <= os * R`, of a model centered at 0.
function _acf_samples(m::AbstractPSFModel{T}, R::Integer, os::Integer) where {T}
    k = (-os * R):(os * R)
    return [T(evaluate(m, i / os, j / os)) for i in k, j in k]
end

# `C[di, dj] = Σ a[u] b[u + (sy di, sx dj)] / (sy sx)`, lags `-L:L`, zero outside the arrays.
function _cross_corr(a::AbstractMatrix{T}, b::AbstractMatrix{T}, L::Int, sy::Int, sx::Int) where {T}
    n1, n2 = size(a)
    C = zeros(T, 2L + 1, 2L + 1)
    for dj in -L:L, di in -L:L
        oi, oj = sy * di, sx * dj
        s = zero(T)
        @inbounds for j in max(1, 1 - oj):min(n2, n2 - oj)
            @simd for i in max(1, 1 - oi):min(n1, n1 - oi)
                s += a[i, j] * b[i + oi, j + oj]
            end
        end
        C[di + L + 1, dj + L + 1] = s / (sy * sx)
    end
    return C
end

"""
    _psf_acf_at(acf, y, x) -> Matrix

The `(2L+1, 2L+1)` autocorrelation table prepared by [`_psf_acf`](@ref) at
detector position `(y, x)`, indexed `A[dy + L + 1, dx + L + 1]` and normalized to
`A(0) = 1`.  Only a [`GriddedPSFModel`](@ref)'s table depends on the position.
Returns a new matrix.
"""
function _psf_acf_at(acf::_StaticACF, y, x)
    A = copy(acf.A)
    return A ./= A[size(A, 1) ÷ 2 + 1, size(A, 2) ÷ 2 + 1]
end

function _psf_acf_at(acf::_GriddedACF{T}, y, x) where {T}
    L = acf.L
    A = zeros(T, 2L + 1, 2L + 1)
    corners = _grid_corners(acf.psf, T(y), T(x))
    for (ia, wa) in corners, (ib, wb) in corners
        (iszero(wa) || iszero(wb)) && continue
        w = wa * wb
        if ia <= ib
            A .+= w .* acf.C[(ia, ib)]
        else
            # C_ji(Δ) = C_ij(-Δ)
            Cij = acf.C[(ib, ia)]
            @inbounds for dj in 1:(2L + 1), di in 1:(2L + 1)
                A[di, dj] += w * Cij[2L + 2 - di, 2L + 2 - dj]
            end
        end
    end
    return A ./= A[L + 1, L + 1]
end
