# One Gauss-Newton linearization of the joint problem, streamed into the normal matrix H.
#
# Per (source, image) with a fit stamp, the weighted, column-equilibrated derivatives of the
# model with respect to the *local* parameters (pixel y, x, image flux g f_b) are filled into
# `V[3, S2, stamp]` with column norms `cn[3, stamp]`, exactly as the single-image fitter's
# stamps.  The chain matrix C = d(local)/d(global) = [J 0; 0 g] (J from the PolyMap) maps them
# to the global parameters (u, v, f_band).  Then
#   g     += C' (cn .* V' r)                         per stamp
#   H_ss  += C' (cn cn' .* V_s' V_s) C               own blocks, one work item per source
#   H_ij  += C_i' (cn_i cn_j' .* V_i' V_j) C_j       pair blocks, one work item per pair slot
# so every H and g write is owned by one work item (no atomics).  Images are processed in
# batches bounded by the stamp memory budget.  Pair slots are the source pairs (i < j) whose
# fit stamps can overlap in some image, built on the device from a cell grid (`PairPattern`).

"""
Global index of every (source, image) fit stamp: `pidx[s, e]` (0: none), the inverse
`src`/`img`, and the host range `first[e] + 1 : first[e + 1]` of image `e`'s stamps.
"""
struct StampIndex{MI, VI}
    pidx::MI
    src::VI
    img::VI
    first::Vector{Int}
end

@kernel function stamp_eligible_kernel!(el, @Const(ay), @Const(ax), @Const(mR), d, R)
    t = @index(Global)
    N = size(ay, 1)
    s = (t - 1) % N + 1; e = (t - 1) ÷ N + 1
    a = ay[s, e]; b = ax[s, e]
    ok = (mR[s, e] > 0) & (a + R >= 1) & (a - R <= d.dims[1, e]) & (b + R >= 1) & (b - R <= d.dims[2, e])
    el[s, e] = ifelse(ok, Int32(1), Int32(0))
end

@kernel function stamp_scatter_kernel!(src, img, pidx, @Const(el))
    t = @index(Global)
    N = size(pidx, 1)
    if el[t] > 0
        k = pidx[t]
        src[k] = Int32((t - 1) % N + 1); img[k] = Int32((t - 1) ÷ N + 1)
    else
        pidx[t] = Int32(0)
    end
end

# Per-image stamp counts from the inclusive scan of the column-major (N, E) eligibility.
@kernel function stamp_count_kernel!(counts, @Const(pidx), N)
    e = @index(Global)
    counts[e] = pidx[N, e] - (e > 1 ? pidx[N, e - 1] : zero(eltype(pidx)))
end

# Images in which each source has a stamp.
@kernel function nimg_kernel!(nimg, @Const(pidx))
    s = @index(Global)
    c = Int32(0)
    for e in axes(pidx, 2)
        c += pidx[s, e] > 0
    end
    nimg[s] = c
end

# The stamp index in two halves around the host read of its per-image counts (`counts`, device
# Int32), so `linearize` can read them together with its other sizes.  Counts come from the scan
# rather than `sum!(...; dims)`: Metal compiles its dims reductions anew for every array shape
# (~0.2 s each), and N changes every pass.
function stamp_counts(sc::JointScene, fp::Footprint, R_fit)
    be = sc.backend; N, E = size(fp.ay)
    N * E < typemax(Int32) || throw(ArgumentError("$N sources x $E images exceed the Int32 stamp index"))
    el = KA.allocate(be, Int32, N, E)
    launch!(stamp_eligible_kernel!, be, N * E, el, fp.ay, fp.ax, fp.mR, sc.dev, Int32(R_fit))
    pidx = reshape(_scan(vec(el)), N, E)
    counts = KA.zeros(be, Int32, E)
    N > 0 && launch!(stamp_count_kernel!, be, E, counts, pidx, N)
    return (; el, pidx, counts)
end

function stamp_index(sc::JointScene, c, counts::AbstractVector{<:Integer})
    N, E = size(c.el)
    first = cumsum([0; counts])
    src = KA.allocate(sc.backend, Int32, first[end]); img = KA.allocate(sc.backend, Int32, first[end])
    launch!(stamp_scatter_kernel!, sc.backend, N * E, src, img, c.pidx, c.el)
    return StampIndex(c.pidx, src, img, first)
end

# ------------------------------------------------------------------------------------------
# Stamps
# ------------------------------------------------------------------------------------------

# Weighted (dmodel/dy, dmodel/dx, dmodel/dflux) of one stamp pixel (0 outside the image or masked).
@inline function stamp_pixel(d, w, e, m, a, b, R, y::T, x::T, f::T, pc) where {T}
    S = 2R + 1
    gy = a - R + Int32((m - 1) % S); gx = b - R + Int32((m - 1) ÷ S)
    ny = d.dims[1, e]; nx = d.dims[2, e]
    ((1 <= gy <= ny) & (1 <= gx <= nx)) || return zero(T), zero(T), zero(T)
    wq = w[d.dims[3, e] + gy + (gx - 1) * ny]
    wq > 0 || return zero(T), zero(T), zero(T)
    sw = sqrt(wq)
    val, dY, dX = psf_value_grad(d.psf, pc, gy, gx, y, x)
    return sw * f * dY, sw * f * dX, sw * val
end

# One work item per stamp of the batch: fill raw values while accumulating the column norms,
# then rescale in place (the single-image `_equilibrate_star!`, norms floored at eps).
@kernel function fill_kernel!(V, cn, @Const(θ), d, @Const(w), @Const(ay), @Const(ax), @Const(src), @Const(img), k0, R)
    t = @index(Global)
    T = eltype(V)
    k = k0 + t
    s = src[k]; e = img[k]
    _, y, x = source_pixel(d, e, θ[1, s], θ[2, s])
    f = d.ip[IP_GAIN, e] * θ[2 + d.band[e], s]
    pc = psf_context(d.psf, e, y, x)
    a = ay[s, e]; b = ax[s, e]
    c1 = c2 = c3 = zero(T)
    S2 = (2R + 1)^2
    for m in 1:S2
        r1, r2, r3 = stamp_pixel(d, w, e, m, a, b, R, y, x, f, pc)
        V[1, m, t] = r1; V[2, m, t] = r2; V[3, m, t] = r3
        c1 += r1 * r1; c2 += r2 * r2; c3 += r3 * r3
    end
    n1 = max(sqrt(c1), eps(T)); n2 = max(sqrt(c2), eps(T)); n3 = max(sqrt(c3), eps(T))
    cn[1, t] = n1; cn[2, t] = n2; cn[3, t] = n3
    i1 = inv(n1); i2 = inv(n2); i3 = inv(n3)
    for m in 1:S2
        V[1, m, t] *= i1; V[2, m, t] *= i2; V[3, m, t] *= i3
    end
end

"The (i, j) block V_i' V_j of equilibrated stamps over their shared pixels; `(dy, dx)` = anchor_j - anchor_i."
@inline function cross3(V, i, j, dy, dx, R)
    T = eltype(V)
    S = 2R + 1
    c1 = max(-R, dx - R); c2 = min(R, dx + R)
    r1 = max(-R, dy - R); r2 = min(R, dy + R)
    a11 = a21 = a31 = a12 = a22 = a32 = a13 = a23 = a33 = zero(T)
    for ox in c1:c2
        mi0 = (ox + R) * S + R + 1
        mj0 = (ox - dx + R) * S - dy + R + 1
        for oy in r1:r2
            mi = mi0 + oy; mj = mj0 + oy
            x1 = V[1, mi, i]; x2 = V[2, mi, i]; x3 = V[3, mi, i]
            y1 = V[1, mj, j]; y2 = V[2, mj, j]; y3 = V[3, mj, j]
            a11 = muladd(x1, y1, a11); a21 = muladd(x2, y1, a21); a31 = muladd(x3, y1, a31)
            a12 = muladd(x1, y2, a12); a22 = muladd(x2, y2, a22); a32 = muladd(x3, y2, a32)
            a13 = muladd(x1, y3, a13); a23 = muladd(x2, y3, a23); a33 = muladd(x3, y3, a33)
        end
    end
    return SMatrix{3, 3, T}(a11, a21, a31, a12, a22, a32, a13, a23, a33)
end

# d(local y, x, flux)/d(global u, v, f_band) of source s in image e.
@inline function chain(d, θ, s, e)
    T = eltype(θ)
    _, _, _, yu, yv, xu, xv = source_pixel(d, e, θ[1, s], θ[2, s])
    return SMatrix{3, 3, T}(yu, xu, zero(T), yv, xv, zero(T), zero(T), zero(T), d.ip[IP_GAIN, e])
end

@inline cnvec(cn, t) = SVector(cn[1, t], cn[2, t], cn[3, t])

# Gradient and own block of source s from every image of the batch, in image order.
@kernel function own_kernel!(g, Hd, @Const(V), @Const(cn), @Const(r), @Const(θ), d, @Const(ay), @Const(ax), @Const(pidx),
                             e0, nb, k0, R)
    s = @index(Global)
    T = eltype(V)
    S = 2R + 1
    for e in (e0 + 1):(e0 + nb)
        k = pidx[s, e]
        k > 0 || continue
        t = k - k0
        a = ay[s, e]; b = ax[s, e]; ny = d.dims[1, e]; nx = d.dims[2, e]; off = d.dims[3, e]
        z1 = z2 = z3 = zero(T)
        m = 0
        for kx in 1:S, ky in 1:S
            m += 1
            gy = a - R + ky - 1; gx = b - R + kx - 1
            ((1 <= gy <= ny) & (1 <= gx <= nx)) || continue
            rq = r[off + gy + (gx - 1) * ny]
            z1 += V[1, m, t] * rq; z2 += V[2, m, t] * rq; z3 += V[3, m, t] * rq
        end
        c = cnvec(cn, t)
        C = chain(d, θ, s, e)
        gl = C' * (c .* SVector(z1, z2, z3))
        G = C' * ((c * c') .* cross3(V, t, t, 0, 0, R)) * C
        ix = (1, 2, 2 + Int(d.band[e]))
        for q in 1:3
            g[ix[q], s] += gl[q]
            for p in 1:3
                Hd[ix[p], ix[q], s] += G[p, q]
            end
        end
    end
end

# Pair blocks: one work item per stored slot q = (row[q], nbr[q]), over the batch's images.
@kernel function pair_kernel!(Ho, @Const(V), @Const(cn), @Const(θ), d, @Const(ay), @Const(ax), @Const(pidx), @Const(row), @Const(nbr),
                              e0, nb, k0, R)
    q = @index(Global)
    i = row[q]; j = nbr[q]
    for e in (e0 + 1):(e0 + nb)
        ki = pidx[i, e]; kj = pidx[j, e]
        ((ki > 0) & (kj > 0)) || continue
        dy = ay[j, e] - ay[i, e]; dx = ax[j, e] - ax[i, e]
        ((abs(dy) <= 2R) & (abs(dx) <= 2R)) || continue
        ti = ki - k0; tj = kj - k0
        G = chain(d, θ, i, e)' * ((cnvec(cn, ti) * cnvec(cn, tj)') .* cross3(V, ti, tj, dy, dx, R)) * chain(d, θ, j, e)
        ix = (1, 2, 2 + Int(d.band[e]))
        for c in 1:3, r in 1:3
            Ho[ix[r], ix[c], q] += G[r, c]
        end
    end
end

# ------------------------------------------------------------------------------------------
# Pair pattern
# ------------------------------------------------------------------------------------------

"""
Source pairs `i < j` whose fit stamps can overlap in some image, as CSR (`ptr`, `nbr`) by row,
with each slot's row (`row`) and the transposed index (`cptr`, `cslot`) the gather SpMV reads.
"""
struct PairPattern{VI}
    ptr::VI
    nbr::VI
    row::VI
    cptr::VI
    cslot::VI
    np::Int
end

"""
    pair_reach(sc, R) -> Float64

Chebyshev frame distance beyond which two sources' `(2R+1)^2` stamps overlap in no image:
anchors differ by at most `2R` per axis and rounding adds up to 1 px, so `(2R+1)` pixels
times the largest frame step per pixel (`‖J⁻¹‖∞`, sampled over each PolyMap's footprint).
"""
function pair_reach(sc::JointScene, R)
    jmax = 0.0
    for m in sc.maps
        for u in range(m.bbox[1], m.bbox[2]; length = 9), v in range(m.bbox[3], m.bbox[4]; length = 9)
            _, _, J = m(u, v)
            jmax = max(jmax, opnorm(inv(J), Inf))
        end
    end
    return (2R + 1) * jmax * 1.01
end

# The pattern is built on the device (adapted from the solver study's `pattern_dev.jl`): a cell
# grid of side `h` over the sources, a counting sort of sources into cells, a count and then a write
# of each source's candidates in its 3x3 cells, and the transposed index.  Atomics make the slot
# order run-dependent, so each row and each column segment is then insertion-sorted (they are a few
# entries long): the result is deterministic and identical to a serial build, with sorted rows
# (`hblock` relies on that).

@inline function pair_cell(u, v, cg)
    a = clamp(unsafe_trunc(Int32, floor((u - cg[1]) / cg[3])), Int32(0), Int32(cg[4]) - Int32(1))
    b = clamp(unsafe_trunc(Int32, floor((v - cg[2]) / cg[3])), Int32(0), Int32(cg[5]) - Int32(1))
    return a, b
end
@inline pair_cell_index(a, b, nu, nv) = ((0 <= a < nu) & (0 <= b < nv)) ? a + nu * b + Int32(1) : Int32(0)
@inline is_pair(j, i, ui, vi, θ, h) = (j > i) && ((abs(θ[1, j] - ui) <= h) & (abs(θ[2, j] - vi) <= h))

@kernel function cell_count_kernel!(cell, ccount, @Const(θ), cg)
    i = @index(Global)
    a, b = pair_cell(θ[1, i], θ[2, i], cg)
    c = a + Int32(cg[4]) * b + Int32(1)
    cell[i] = c
    Atomix.@atomic ccount[c + 1] += Int32(1)
end

@kernel function cell_scatter_kernel!(order, cfill, @Const(cell), @Const(cstart))
    i = @index(Global)
    c = cell[i]
    k = Atomix.@atomic cfill[c] += Int32(1)    # the post-increment value
    order[cstart[c] + k - Int32(1)] = i
end

# Count (`write = false`) or write (`true`) source i's candidates j > i in its 3x3 cells.
@kernel function pair_scan_kernel!(cnt, nbr, row, ccol, @Const(ptr), @Const(θ), @Const(cell), @Const(cstart), @Const(order), cg, write::Bool)
    i = @index(Global)
    ui = θ[1, i]; vi = θ[2, i]; h = cg[3]
    nu = Int32(cg[4]); nv = Int32(cg[5])
    c = cell[i] - Int32(1); a = c % nu; b = c ÷ nu
    n = Int32(0); k = write ? ptr[i] : Int32(0)
    for db in Int32(-1):Int32(1), da in Int32(-1):Int32(1)
        cc = pair_cell_index(a + da, b + db, nu, nv)
        cc == 0 && continue
        for q in cstart[cc]:(cstart[cc + 1] - Int32(1))
            j = order[q]
            is_pair(j, i, ui, vi, θ, h) || continue
            if write
                nbr[k] = j; row[k] = i
                Atomix.@atomic ccol[j + 1] += Int32(1)
                k += Int32(1)
            else
                n += Int32(1)
            end
        end
    end
    write || (cnt[i + 1] = n)
end

@kernel function col_scatter_kernel!(cslot, cfill, @Const(nbr), @Const(cptr))
    q = @index(Global)
    j = nbr[q]
    k = Atomix.@atomic cfill[j] += Int32(1)
    cslot[cptr[j] + k - Int32(1)] = q
end

# Insertion sort of x[ptr[i] : ptr[i+1]-1] for each segment i.
@kernel function segment_sort_kernel!(x, @Const(ptr))
    i = @index(Global)
    lo = ptr[i]; hi = ptr[i + 1] - Int32(1)
    for p in (lo + Int32(1)):hi
        v = x[p]; q = p - Int32(1)
        while q >= lo && x[q] > v
            x[q + 1] = x[q]; q -= Int32(1)
        end
        x[q + 1] = v
    end
end

# Inclusive scan in x's element type: `cumsum` widens Int32 to Int64 on every backend.
_scan(x) = cumsum!(similar(x), x)

# Counts written into x[2:end]; returns starts from 1: [1, 1 + x[2], 1 + x[2] + x[3], ...].
_scan_from_one(x) = (fill!(view(x, 1:1), one(eltype(x))); _scan(x))

"""
    PairPattern(be, θ, h; ext = nothing) -> PairPattern

Source pairs `i < j` within Chebyshev frame distance `h` of each other (positions are rows 1-2 of
the device `P x N` matrix `θ`; non-finite positions get no pairs).  The cell grid spans
`[0.5, ext + 0.5]` per axis when the frame size `ext = (ny, nx)` is given (positions outside
clamp into edge cells, which still finds every pair since the true distance is tested), else the
sources' bounding box (four extra host reads).  One host synchronization, to size the pair
arrays; `linearize` runs the two halves (`pair_count`, `pair_fill`) around its single read.
"""
function PairPattern(be, θ::AbstractMatrix, h::Real; ext = nothing)
    c = pair_count(be, θ, h, ext)
    N = size(θ, 2)
    np = N > 0 ? Int(Array(view(c.ptr, (N + 1):(N + 1)))[1]) - 1 : 0
    return pair_fill(be, θ, c, np)
end

# Cells, the counting sort into cells and each row's pair count and start (`ptr`); no host read
# beyond the bounding box when `ext` is not given.
function pair_count(be, θ::AbstractMatrix, h::Real, ext)
    T = eltype(θ); N = size(θ, 2)
    if ext === nothing
        big = typemax(T); small = typemin(T)        # captured as values: a captured type is not a valid GPU kernel argument
        lo(r) = mapreduce(x -> isfinite(x) ? x : big, min, view(θ, r, :); init = big)
        hi(r) = mapreduce(x -> isfinite(x) ? x : small, max, view(θ, r, :); init = small)
        u0, u1, v0, v1 = N > 0 ? (lo(1), hi(1), lo(2), hi(2)) : (zero(T), zero(T), zero(T), zero(T))
        isfinite(u0) || (u0 = u1 = v0 = v1 = zero(T))
    else
        u0, v0 = T(0.5), T(0.5); u1, v1 = T(ext[1] + 0.5), T(ext[2] + 0.5)
    end
    nu = floor(Int, (u1 - u0) / h) + 1; nv = floor(Int, (v1 - v0) / h) + 1
    cg = (T(u0), T(v0), T(h), T(nu), T(nv))
    z(n) = KA.zeros(be, Int32, n)
    cell = z(N); ccount = z(nu * nv + 1); cfill = z(nu * nv); order = z(N); cnt = z(N + 1)
    launch!(cell_count_kernel!, be, N, cell, ccount, θ, cg)
    cstart = _scan_from_one(ccount)
    launch!(cell_scatter_kernel!, be, N, order, cfill, cell, cstart)
    launch!(pair_scan_kernel!, be, N, cnt, cnt, cnt, cnt, cnt, θ, cell, cstart, order, cg, false)
    return (; cell, cstart, order, cnt, cg, ptr = _scan_from_one(cnt))
end

# The pair arrays for `np` pairs (read from `c.ptr[N + 1] - 1`), written, row-sorted and transposed.
function pair_fill(be, θ::AbstractMatrix, c, np::Integer)
    N = size(θ, 2)
    ccol = KA.zeros(be, Int32, N + 1); cfillc = KA.zeros(be, Int32, N)
    nbr = KA.allocate(be, Int32, np); row = KA.allocate(be, Int32, np); cslot = KA.allocate(be, Int32, np)
    launch!(pair_scan_kernel!, be, N, c.cnt, nbr, row, ccol, c.ptr, θ, c.cell, c.cstart, c.order, c.cg, true)
    launch!(segment_sort_kernel!, be, N, nbr, c.ptr)                         # rows: row[] is unaffected
    cptr = _scan_from_one(ccol)
    launch!(col_scatter_kernel!, be, np, cslot, cfillc, nbr, cptr)
    launch!(segment_sort_kernel!, be, N, cslot, cptr)
    return PairPattern(c.ptr, nbr, row, cptr, cslot, np)
end

# ------------------------------------------------------------------------------------------
# Linearization
# ------------------------------------------------------------------------------------------

"""
The normal equations at one θ: raw gradient `g` (`P x N`), the equilibration `D` (`P x N`,
`sqrt(diag H)` or 1 without data), the equilibrated own blocks `Hd` (`P x P x N`) and pair
blocks `Ho` (`P x P x np`), the raw diagonal `hdiag`, and the cost.  `nimg[s]` counts the
images in which source `s` has a fit stamp.
"""
struct Linearization{M, A3, MI, VI, PP}
    g::M
    D::M
    hdiag::M
    Hd::A3
    Ho::A3
    pat::PP
    cost::Float64
    fp::Footprint{MI}
    nimg::VI
end

@kernel function equilibrate_own_kernel!(D, hdiag, Hd)
    s = @index(Global)
    T = eltype(D)
    P = size(D, 1)
    for k in 1:P
        h = Hd[k, k, s]
        hdiag[k, s] = h
        D[k, s] = h > 0 ? sqrt(h) : one(T)
    end
    for c in 1:P, r in 1:P
        Hd[r, c, s] /= D[r, s] * D[c, s]
    end
end

@kernel function equilibrate_pair_kernel!(Ho, @Const(D), @Const(row), @Const(nbr))
    q = @index(Global)
    P = size(D, 1)
    i = row[q]; j = nbr[q]
    for c in 1:P, r in 1:P
        Ho[r, c, q] /= D[r, i] * D[c, j]
    end
end

"""
Device buffers reused across linearizations and trials: the per-pixel model, weighted residual
and cost mask, and the cost's per-workgroup partial sums (`cost_parts`).  `bkg`, when given, is a
per-pixel background the fit data is `sc.data - bkg` against (the loop keeps `sc.data` raw and
its background here, rather than a background-subtracted copy).
`stamp_budget` bytes bound the stamp array; batches split at image boundaries.
"""
struct LinearizeWork{V, VB, VP, BK}
    model::V
    resid::V
    mask::VB
    part::VP
    bkg::BK
end

LinearizeWork(sc::JointScene{T}; bkg = nothing) where {T} =
    LinearizeWork(KA.zeros(sc.backend, T, length(sc.data)), KA.zeros(sc.backend, T, length(sc.data)), KA.zeros(sc.backend, Bool, length(sc.data)),
                  cost_parts(sc.backend, T), bkg)

"""
    linearize(sc, θ, rt, work; R_fit, stamp_budget = 100_000_000) -> Linearization

Footprint, cost support and residual, stamps, gradient and equilibrated H at `θ`.  `work.model`
holds the model at `θ` and `work.resid` the weighted residual on return.  One host
synchronization: the per-image stamp counts, the pair count and the cost partials are read
together.
"""
function linearize(sc::JointScene{T}, θ::AbstractMatrix{T}, rt::RadiusTable, work::LinearizeWork; R_fit::Integer,
        stamp_budget::Integer = 100_000_000) where {T}
    be = sc.backend; P, N = size(θ); E = nimages(sc); R = Int32(R_fit)
    fp = footprint(sc, θ, rt)
    fit_mask!(work.mask, sc, fp, R)
    render!(work.model, sc, θ, fp)
    cost_partials!(work.resid, sc, work.model, work.mask, work.part, work.bkg)
    stc = stamp_counts(sc, fp, R)
    pc = pair_count(be, θ, pair_reach(sc, R), (sc.frame.ny, sc.frame.nx))
    # the one read: [stamp counts; ptr[N + 1]; the cost partials' bits]
    pbits = reinterpret(Int32, work.part)
    stage = KA.allocate(be, Int32, E + 1 + length(pbits))
    copyto!(view(stage, 1:E), stc.counts); copyto!(view(stage, (E + 1):(E + 1)), view(pc.ptr, (N + 1):(N + 1)))
    copyto!(view(stage, (E + 2):length(stage)), pbits)
    hs = Array(stage)
    cost = sum(Float64, reinterpret(T, hs[(E + 2):end]))
    si = stamp_index(sc, stc, view(hs, 1:E))
    pat = pair_fill(be, θ, pc, Int(hs[E + 1]) - 1)
    g = KA.zeros(be, T, P, N); Hd = KA.zeros(be, T, P, P, N); Ho = KA.zeros(be, T, P, P, pat.np)
    S2 = (2R + 1)^2
    nmax = max(stamp_budget ÷ (3 * S2 * sizeof(T)), maximum(diff(si.first)))
    V = KA.allocate(be, T, 3, S2, min(nmax, si.first[end])); cn = KA.allocate(be, T, 3, size(V, 3))
    e0 = 0
    while e0 < E
        nb = 1                                     # grow the batch while its stamps fit
        while e0 + nb < E && si.first[e0 + nb + 2] - si.first[e0 + 1] <= nmax
            nb += 1
        end
        k0 = si.first[e0 + 1]; ns = si.first[e0 + nb + 1] - k0
        launch!(fill_kernel!, be, ns, V, cn, θ, sc.dev, sc.weight, fp.ay, fp.ax, si.src, si.img, k0, R)
        launch!(own_kernel!, be, N, g, Hd, V, cn, work.resid, θ, sc.dev, fp.ay, fp.ax, si.pidx, e0, nb, k0, R)
        launch!(pair_kernel!, be, pat.np, Ho, V, cn, θ, sc.dev, fp.ay, fp.ax, si.pidx, pat.row, pat.nbr, e0, nb, k0, R)
        e0 += nb
    end
    D = KA.allocate(be, T, P, N); hdiag = KA.allocate(be, T, P, N)
    launch!(equilibrate_own_kernel!, be, N, D, hdiag, Hd)
    launch!(equilibrate_pair_kernel!, be, pat.np, Ho, D, pat.row, pat.nbr)
    nimg = KA.allocate(be, Int32, N)
    launch!(nimg_kernel!, be, N, nimg, si.pidx)
    return Linearization(g, D, hdiag, Hd, Ho, pat, cost, fp, nimg)
end
