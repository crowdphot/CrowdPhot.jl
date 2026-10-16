# Device-resident `Background2D` (KernelAbstractions): the per-pass background of the joint
# loop runs where the residuals live, with no host round trip.
#
# Algorithm per mesh box (one workgroup per box): load the box into local memory with
# masked / non-finite / out-of-image pixels set to +Inf, bitonic-sort it (across the
# workgroup on a GPU; one work item per box runs the same network on the CPU), and then every
# statistic is cheap because sigma clipping keeps a contiguous range of the sorted box:
# the median is an index, mean/std come from one shifted-sum reduction, the new clip range
# from two binary searches, and the median absolute deviation is the k-th smallest
# |x - loc|, i.e. the minimum over size-k windows of max(|x[i] - loc|, |x[i+k-1] - loc|).
#
# Matches `Background2D` to ~1e-14 (Float64) and ~1e-5 (Float32) relative where no mesh
# cell is excluded.  Differences from the CPU `Background2D`:
# - excluded mesh cells are filled by Jacobi (order independent) neighbor averaging; the CPU
#   fills in place in loop order (Gauss-Seidel), so filled cells differ (maps by up to ~1%);
# - no warning for unmasked non-finite pixels, and an all-excluded mesh returns NaN maps
#   instead of throwing (both would need a device -> host sync);
# - only Mean/Median/SExtractor/MMM location and Std/MADStd RMS estimators;
# - boxes are limited to `MAX_BOX_PIXELS` pixels (local memory).

const MAX_BOX_PIXELS = 4096

_est_code(::MeanBackground) = (0, 0.0, 0.0)
_est_code(::MedianBackground) = (1, 0.0, 0.0)
_est_code(::SExtractorBackground) = (2, 0.0, 0.0)
_est_code(e::MMMBackground) = (3, Float64(e.median_factor), Float64(e.mean_factor))
_est_code(e) = throw(ArgumentError("background2d! does not support estimator $(typeof(e))"))
_rms_code(::StdRMS) = 0
_rms_code(::MADStdRMS) = 1
_rms_code(e) = throw(ArgumentError("background2d! does not support rms_estimator $(typeof(e))"))

_kernel_mask(be, m) = m
_kernel_mask(be, m::BitArray) = to_device(be, Array{Bool}(m))

@inline _getmask(::Nothing, i, j) = false
@inline _getmask(m, i, j) = @inbounds m[i, j]

# Median of the sorted range x[a+1:a+n], as Statistics.median! (middle(a, b) = a/2 + b/2).
@inline function _sorted_median(x, a, n)
    h = a + (n + 1) ÷ 2
    return isodd(n) ? x[h] : x[h] / 2 + x[h + 1] / 2
end

# First index in a+1:a+n with x[p] >= v (a+n+1 if none); last index with x[p] <= v (a if none).
@inline function _first_ge(x, v, a, n)
    lo, hi = a + 1, a + n + 1
    while lo < hi
        m = (lo + hi) >>> 1
        x[m] < v ? (lo = m + 1) : (hi = m)
    end
    return lo
end
@inline function _last_le(x, v, a, n)
    lo, hi = a, a + n
    while lo < hi
        m = (lo + hi + 1) >>> 1
        x[m] <= v ? (lo = m) : (hi = m - 1)
    end
    return lo
end

@inline function _location(code, f1, f2, m, s, med)
    code == 0 && return m
    code == 1 && return med
    code == 3 && return f1 * med - f2 * m
    s == 0 && return m
    abs(m - med) / s > oftype(s, 0.3) && return med
    return oftype(m, 2.5) * med - oftype(m, 1.5) * m
end

# Sequential branchless bitonic sort of x[1:N] (N a power of 2); inner loops are contiguous.
@inline function _bitonic_sort!(x, ::Val{N}) where {N}
    @inbounds for kk in 1:trailing_zeros(N), jj in kk:-1:1
        jb = 1 << (jj - 1)
        for s in 0:(2jb):(N - 1)
            up = (s & (1 << kk)) == 0
            @simd for o in 1:jb
                a = x[s + o]
                b = x[s + o + jb]
                lo, hi = min(a, b), max(a, b)
                x[s + o] = ifelse(up, lo, hi)
                x[s + o + jb] = ifelse(up, hi, lo)
            end
        end
    end
    return x
end

# One workgroup per mesh box.  `N` is the padded (power of 2) box length, `WG` the static
# workgroup size.  State shared across @synchronize lives in local memory: ist = (range
# start a, range length n, done flag, covered count); st = (loc, rms).
@kernel function _mesh_kernel!(mbkg, mrms, @Const(img), mask, cov, bh, bw, min_valid, slo, shi, maxiters,
                               est, f1, f2, rmscode, ::Val{N}, ::Val{WG}) where {N, WG}
    g = @index(Group, Linear)
    t = @index(Local, Linear)
    @uniform T = eltype(mbkg)
    @uniform NSTAGE = WG == 1 ? 0 : trailing_zeros(N)
    @uniform NHALF = N ÷ 2
    x = @localmem T (N,)
    red = @localmem T (WG, 2)
    cnt = @localmem Int (WG, 2)
    ist = @localmem Int (4,)
    st = @localmem T (2,)

    # --- load (masked / non-finite / outside image -> +Inf, sorted to the end) ---
    nmr = size(mbkg, 1)
    r0 = ((g - 1) % nmr) * bh
    c0 = ((g - 1) ÷ nmr) * bw
    nv = 0
    nc = 0
    bi = (t - 1) % bh
    bj = (t - 1) ÷ bh
    for p in t:WG:N
        v = typemax(T)
        valid = false
        if bj < bw
            i = r0 + bi + 1
            j = c0 + bj + 1
            if i <= size(img, 1) && j <= size(img, 2)
                u = @inbounds img[i, j]
                ok = isfinite(u) && !_getmask(cov, i, j)
                nc += ok
                if ok && !_getmask(mask, i, j)
                    v = T(u)
                    valid = true
                end
            end
        end
        @inbounds x[p] = v
        nv += valid
        bi += WG
        while bi >= bh
            bi -= bh
            bj += 1
        end
    end
    # One work item sorts the whole box on the CPU backend (a comparison sort mispredicts
    # ~8 us per 400-px box; this branchless network takes ~3.6 us).
    WG == 1 && _bitonic_sort!(x, Val(N))
    @inbounds cnt[t, 1] = nv
    @inbounds cnt[t, 2] = nc
    @synchronize

    # --- the same bitonic network spread across the workgroup (GPU) ---
    for kk in 1:NSTAGE
        for jj in kk:-1:1
            for q in (t - 1):WG:(NHALF - 1)
                jb = 1 << (jj - 1)
                i0 = 2q - (q & (jb - 1))
                l0 = i0 + jb
                a = @inbounds x[i0 + 1]
                b = @inbounds x[l0 + 1]
                if (a > b) == ((i0 & (1 << kk)) == 0)
                    @inbounds x[i0 + 1] = b
                    @inbounds x[l0 + 1] = a
                end
            end
            @synchronize
        end
    end

    if t == 1
        nv1 = 0
        nc1 = 0
        for w in 1:WG
            nv1 += @inbounds cnt[w, 1]
            nc1 += @inbounds cnt[w, 2]
        end
        @inbounds ist[1] = 0
        @inbounds ist[2] = nv1
        @inbounds ist[3] = (nv1 == 0 || maxiters == 0) ? 1 : 0
        @inbounds ist[4] = nc1
    end
    @synchronize

    # --- sigma clipping: the retained set stays a contiguous sorted range ---
    for it in 1:maxiters
        a1 = @inbounds ist[1]
        n1 = @inbounds ist[2]
        s1 = zero(T)
        s2 = zero(T)
        if @inbounds(ist[3]) == 0
            K = _sorted_median(x, a1, n1)
            for p in (a1 + t):WG:(a1 + n1)
                d = @inbounds(x[p]) - K
                s1 += d
                s2 += d * d
            end
        end
        @inbounds red[t, 1] = s1
        @inbounds red[t, 2] = s2
        @synchronize
        if t == 1 && @inbounds(ist[3]) == 0
            a2 = @inbounds ist[1]
            n2 = @inbounds ist[2]
            S1 = zero(T)
            S2 = zero(T)
            for w in 1:WG
                S1 += @inbounds red[w, 1]
                S2 += @inbounds red[w, 2]
            end
            med = _sorted_median(x, a2, n2)
            sd = sqrt(max(zero(T), S2 / n2 - (S1 / n2)^2))
            if sd == 0
                @inbounds ist[3] = 1
            else
                pf = _first_ge(x, med - slo * sd, a2, n2)
                pl = _last_le(x, med + shi * sd, a2, n2)
                nn = max(0, pl - pf + 1)
                if nn == n2
                    @inbounds ist[3] = 1
                else
                    @inbounds ist[1] = pf - 1
                    @inbounds ist[2] = nn
                    nn == 0 && (@inbounds ist[3] = 1)
                end
            end
        end
        @synchronize
    end

    # --- final moments about the median of the retained range ---
    a3 = @inbounds ist[1]
    n3 = @inbounds ist[2]
    s3 = zero(T)
    s4 = zero(T)
    if n3 > 0
        K3 = _sorted_median(x, a3, n3)
        for p in (a3 + t):WG:(a3 + n3)
            d = @inbounds(x[p]) - K3
            s3 += d
            s4 += d * d
        end
    end
    @inbounds red[t, 1] = s3
    @inbounds red[t, 2] = s4
    @synchronize
    if t == 1
        a4 = @inbounds ist[1]
        n4 = @inbounds ist[2]
        loc = zero(T)
        rms = zero(T)
        if n4 > 0
            S3 = zero(T)
            S4 = zero(T)
            for w in 1:WG
                S3 += @inbounds red[w, 1]
                S4 += @inbounds red[w, 2]
            end
            K4 = _sorted_median(x, a4, n4)
            m1 = S3 / n4
            sd = sqrt(max(zero(T), S4 / n4 - m1^2))
            loc = _location(est, f1, f2, K4 + m1, sd, K4)
            δ = loc - K4
            rms = sqrt(max(zero(T), S4 / n4 - 2δ * m1 + δ^2))
        end
        @inbounds st[1] = loc
        @inbounds st[2] = rms
    end
    @synchronize

    # --- MAD about loc: k-th smallest |x - loc| = min over size-k windows of the end deviations ---
    a5 = @inbounds ist[1]
    n5 = @inbounds ist[2]
    l5 = @inbounds st[1]
    k1 = (n5 + 1) ÷ 2
    k2 = n5 ÷ 2 + 1
    w1 = typemax(T)
    w2 = typemax(T)
    if rmscode == 1 && n5 > 0
        for p in (a5 + t):WG:(a5 + n5)
            dp = abs(@inbounds(x[p]) - l5)
            p + k1 - 1 <= a5 + n5 && (w1 = min(w1, max(dp, abs(@inbounds(x[p + k1 - 1]) - l5))))
            p + k2 - 1 <= a5 + n5 && (w2 = min(w2, max(dp, abs(@inbounds(x[p + k2 - 1]) - l5))))
        end
    end
    @inbounds red[t, 1] = w1
    @inbounds red[t, 2] = w2
    @synchronize
    if t == 1
        n6 = @inbounds ist[2]
        rms6 = @inbounds st[2]
        if rmscode == 1 && n6 > 0
            d1 = typemax(T)
            d2 = typemax(T)
            for w in 1:WG
                d1 = min(d1, @inbounds red[w, 1])
                d2 = min(d2, @inbounds red[w, 2])
            end
            rms6 = (isodd(n6) ? d1 : d1 / 2 + d2 / 2) * T(CrowdPhot.mad_constant)
        end
        excluded = @inbounds(ist[4]) == 0 || n6 == 0 || n6 < min_valid
        mi = (g - 1) % size(mbkg, 1) + 1
        mj = (g - 1) ÷ size(mbkg, 1) + 1
        @inbounds mbkg[mi, mj] = excluded ? T(NaN) : st[1]
        @inbounds mrms[mi, mj] = excluded ? T(NaN) : rms6
    end
end

# Fill NaN mesh cells of `a` and `b` by repeated averaging of finite 8-neighbors (Jacobi
# sweeps).  Single workgroup; stops once a sweep leaves no NaN, makes no progress (nothing
# finite to propagate from), or after `maxsweeps`.  go = (continue flag, sweeps done).
@kernel function _fill_nans_kernel!(a, b, ta, tb, maxsweeps, ::Val{WG}) where {WG}
    t = @index(Local, Linear)
    @uniform T = eltype(a)
    before = @localmem Int (WG,)
    after = @localmem Int (WG,)
    go = @localmem Int (2,)
    if t == 1
        @inbounds go[1] = 1
        @inbounds go[2] = 0
    end
    @synchronize
    while @inbounds(go[1]) == 1
        nb0 = 0
        na0 = 0
        M, Nc = size(a)
        for c in t:WG:(M * Nc)
            i = (c - 1) % M + 1
            j = (c - 1) ÷ M + 1
            va = @inbounds a[i, j]
            vb = @inbounds b[i, j]
            if isnan(va) || isnan(vb)
                nb0 += 1
                sa = zero(T); ka = 0
                sb = zero(T); kb = 0
                for di in -1:1, dj in -1:1
                    (di == 0 && dj == 0) && continue
                    ii = clamp(i + di, 1, M)
                    jj = clamp(j + dj, 1, Nc)
                    ua = @inbounds a[ii, jj]
                    ub = @inbounds b[ii, jj]
                    isnan(ua) || (sa += ua; ka += 1)
                    isnan(ub) || (sb += ub; kb += 1)
                end
                isnan(va) && ka > 0 && (va = sa / ka)
                isnan(vb) && kb > 0 && (vb = sb / kb)
                na0 += isnan(va) | isnan(vb)
            end
            @inbounds ta[c] = va
            @inbounds tb[c] = vb
        end
        @inbounds before[t] = nb0
        @inbounds after[t] = na0
        @synchronize
        s0 = 0
        for w in 1:WG
            s0 += @inbounds before[w]
        end
        # Only copy back when this sweep changed something (every thread sees the same s0).
        if s0 > 0
            for c in t:WG:length(a)
                @inbounds a[c] = ta[c]
                @inbounds b[c] = tb[c]
            end
        end
        @synchronize
        if t == 1
            s1 = 0
            s2 = 0
            for w in 1:WG
                s1 += @inbounds after[w]
                s2 += @inbounds before[w]
            end
            @inbounds go[2] += 1
            @inbounds go[1] = (s1 > 0 && s1 < s2 && go[2] < maxsweeps) ? 1 : 0
        end
        @synchronize
    end
end

# Median filter of both meshes with replicated borders; FN = fh * fw (odd).
@kernel function _median_filter_kernel!(oa, ob, @Const(a), @Const(b), fh, fw, ::Val{FN}) where {FN}
    i, j = @index(Global, NTuple)
    @uniform T = eltype(a)
    buf = @private T (FN,)
    buf2 = @private T (FN,)
    M, Nc = size(a)
    if i <= M && j <= Nc
        hh = fh ÷ 2
        hw = fw ÷ 2
        k = 0
        for di in -hh:hh, dj in -hw:hw
            k += 1
            ii = clamp(i + di, 1, M)
            jj = clamp(j + dj, 1, Nc)
            @inbounds buf[k] = a[ii, jj]
            @inbounds buf2[k] = b[ii, jj]
        end
        # insertion sort; FN is small (9 for the default 3x3)
        for p in 2:FN
            va = @inbounds buf[p]
            vb = @inbounds buf2[p]
            q = p - 1
            while q >= 1 && @inbounds(buf[q]) > va
                @inbounds buf[q + 1] = buf[q]
                q -= 1
            end
            @inbounds buf[q + 1] = va
            q = p - 1
            while q >= 1 && @inbounds(buf2[q]) > vb
                @inbounds buf2[q + 1] = buf2[q]
                q -= 1
            end
            @inbounds buf2[q + 1] = vb
        end
        @inbounds oa[i, j] = buf[(FN + 1) ÷ 2]
        @inbounds ob[i, j] = buf2[(FN + 1) ÷ 2]
    end
end

@inline function _cr_weights(t::F) where {F}
    t2 = t * t
    t3 = t2 * t
    return ((-t + 2 * t2 - t3) / 2, (2 - 5 * t2 + 3 * t3) / 2, (t + 4 * t2 - 3 * t3) / 2, (-t2 + t3) / 2)
end

# Mesh coordinate of output pixel k: index of the floor node and fractional offset.  Box-centered
# and clamped to [1, Km], as the host `_zoom_coord`.
@inline function _zoom_idx(::Type{T}, k, Kout, Km) where {T}
    cf = clamp((T(k) - T(0.5)) * T(Km) / T(Kout) + T(0.5), one(T), T(Km))
    m = unsafe_trunc(Int, cf)
    return (m, cf - m)
end

@inline function _cr_row(mesh, r, j1, j2, j3, j4, w1, w2, w3, w4)
    return @inbounds muladd(w4, mesh[r, j4], muladd(w3, mesh[r, j3], muladd(w2, mesh[r, j2], w1 * mesh[r, j1])))
end

# Catmull-Rom bicubic upsampling of both meshes (box-centered on the padded extent padH x padW,
# as `_bicubic_zoom`) plus the coverage fill.
@kernel function _zoom_kernel!(bkg, rms, @Const(ma), @Const(mb), cov, padH, padW, fill_value)
    i, j = @index(Global, NTuple)
    @uniform T = eltype(bkg)
    if i <= size(bkg, 1) && j <= size(bkg, 2)
        if _getmask(cov, i, j)
            @inbounds bkg[i, j] = fill_value
            @inbounds rms[i, j] = fill_value
        else
            M, Nc = size(ma)
            im, tr = _zoom_idx(T, i, padH, M)
            jm, tc = _zoom_idx(T, j, padW, Nc)
            r1 = clamp(im - 1, 1, M); r2 = clamp(im, 1, M); r3 = clamp(im + 1, 1, M); r4 = clamp(im + 2, 1, M)
            j1 = clamp(jm - 1, 1, Nc); j2 = clamp(jm, 1, Nc); j3 = clamp(jm + 1, 1, Nc); j4 = clamp(jm + 2, 1, Nc)
            wc1, wc2, wc3, wc4 = _cr_weights(tc)
            wr1, wr2, wr3, wr4 = _cr_weights(tr)
            q1 = _cr_row(ma, r1, j1, j2, j3, j4, wc1, wc2, wc3, wc4)
            q2 = _cr_row(ma, r2, j1, j2, j3, j4, wc1, wc2, wc3, wc4)
            q3 = _cr_row(ma, r3, j1, j2, j3, j4, wc1, wc2, wc3, wc4)
            q4 = _cr_row(ma, r4, j1, j2, j3, j4, wc1, wc2, wc3, wc4)
            @inbounds bkg[i, j] = muladd(wr4, q4, muladd(wr3, q3, muladd(wr2, q2, wr1 * q1)))
            q1 = _cr_row(mb, r1, j1, j2, j3, j4, wc1, wc2, wc3, wc4)
            q2 = _cr_row(mb, r2, j1, j2, j3, j4, wc1, wc2, wc3, wc4)
            q3 = _cr_row(mb, r3, j1, j2, j3, j4, wc1, wc2, wc3, wc4)
            q4 = _cr_row(mb, r4, j1, j2, j3, j4, wc1, wc2, wc3, wc4)
            @inbounds rms[i, j] = muladd(wr4, q4, muladd(wr3, q3, muladd(wr2, q2, wr1 * q1)))
        end
    end
end

# Launch geometry per backend, the one place that knows about it.  Kernel objects are
# instantiated with a static workgroup size (dynamic sizes make CPU @private type-unstable).
_mesh_wg(::KA.CPU, N) = 1
_mesh_wg(::KA.Backend, N) = min(N ÷ 2, 32)  # fastest on Apple M-series (tried 8-256)
_fill_wg(::KA.Backend) = 256
_pix_wg(::KA.CPU) = (64, 4)
_pix_wg(::KA.Backend) = (32, 8)

"""
    background2d!(bkg, rms, resid, box_size; estimator = SExtractorBackground(),
        rms_estimator = MADStdRMS(), mask = nothing, coverage_mask = nothing, fill_value = 0,
        filter_size = (3, 3), exclude_percentile = 10, sigma = 3.0, maxiters = 10,
        mesh = nothing) -> (bkg, rms, mesh_bkg, mesh_rms)

Device-resident `Background2D`: writes the full-resolution background and RMS maps of
`resid` into `bkg` and `rms` (same size and backend).  Keywords mean what they mean for
`Background2D`; `sigma = nothing` disables clipping.  `mesh` may pass preallocated
`(mesh_bkg, mesh_rms, tmp_a, tmp_b)` device matrices of size `cld.(size(resid), box)`.
Asynchronous: nothing is synchronized with the host.
"""
function background2d!(bkg::AbstractMatrix{T}, rms::AbstractMatrix{T}, resid::AbstractMatrix, box_size;
        estimator = SExtractorBackground(), rms_estimator = MADStdRMS(), mask = nothing,
        coverage_mask = nothing, fill_value::Real = 0, filter_size = (3, 3),
        exclude_percentile::Real = 10.0, sigma = 3.0, maxiters::Integer = 10, mesh = nothing) where {T}
    H, W = size(resid)
    size(bkg) == size(rms) == (H, W) || throw(DimensionMismatch("bkg and rms must match resid"))
    for (m, name) in ((mask, "mask"), (coverage_mask, "coverage_mask"))
        m === nothing || size(m) == (H, W) || throw(DimensionMismatch("$name must be the same size as resid"))
    end
    bh, bw = box_size isa Integer ? (box_size, box_size) : Tuple(box_size)
    fh, fw = filter_size isa Integer ? (filter_size, filter_size) : Tuple(filter_size)
    isodd(fh) && isodd(fw) || throw(ArgumentError("filter_size must be odd; got ($fh, $fw)"))
    0 <= exclude_percentile <= 100 || throw(ArgumentError("exclude_percentile must be in [0, 100]"))
    bh * bw <= MAX_BOX_PIXELS ||
        throw(ArgumentError("box of $(bh * bw) pixels exceeds $MAX_BOX_PIXELS; use the CPU Background2D"))
    est, f1, f2 = _est_code(estimator)
    rc = _rms_code(rms_estimator)
    slo, shi = sigma === nothing ? (0.0, 0.0) : sigma isa Real ? (sigma, sigma) : (sigma[1], sigma[2])
    niter = sigma === nothing ? 0 : Int(maxiters)

    backend = KA.get_backend(resid)
    # a BitArray is not a valid kernel argument on compiled backends
    mask = _kernel_mask(backend, mask); coverage_mask = _kernel_mask(backend, coverage_mask)
    M, Nc = cld(H, bh), cld(W, bw)
    ma, mb, ta, tb = something(mesh, ntuple(_ -> similar(bkg, M, Nc), 4))
    size(ma) == (M, Nc) || throw(DimensionMismatch("mesh buffers must be $M x $Nc"))

    N = nextpow(2, max(2, bh * bw))
    WG = _mesh_wg(backend, N)
    _mesh_kernel!(backend, WG)(ma, mb, resid, mask, coverage_mask, bh, bw, T(exclude_percentile / 100 * bh * bw),
        T(slo), T(shi), niter, est, T(f1), T(f2), rc, Val(N), Val(WG); ndrange = M * Nc * WG)
    FW = _fill_wg(backend)
    _fill_nans_kernel!(backend, FW)(ma, mb, ta, tb, M + Nc, Val(FW); ndrange = FW)
    src_a, src_b = ma, mb
    if fh * fw > 1
        _median_filter_kernel!(backend, _pix_wg(backend))(ta, tb, ma, mb, fh, fw, Val(fh * fw); ndrange = (M, Nc))
        src_a, src_b = ta, tb
    end
    _zoom_kernel!(backend, _pix_wg(backend))(bkg, rms, src_a, src_b, coverage_mask, M * bh, Nc * bw, T(fill_value);
        ndrange = (H, W))
    # The unfiltered mesh is `ma, mb` (Background2D reports the unfiltered NaN-filled mesh).
    return (bkg, rms, ma, mb)
end

_device_box(box) = prod(box isa Integer ? (box, box) : box) <= MAX_BOX_PIXELS
_device_estimator(e) = e isa Union{MeanBackground, MedianBackground, SExtractorBackground, MMMBackground}
_device_rms(e) = e isa Union{StdRMS, MADStdRMS}

"Whether `background2d!` supports these options (otherwise the loop uses the host `Background2D`)."
device_background_ok(box, estimator, rms_estimator) = _device_box(box) && _device_estimator(estimator) && _device_rms(rms_estimator)
