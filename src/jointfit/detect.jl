# Joint multi-image, multi-band detection statistic on an oversampled detection grid.
#
# At a sky position every image i of band f is fitted with the same flux s_f:
# model_i = g_i s_f P_i(x - x_i(u)) [+ b_i], weights c iv inside the fit aperture c (DOLPHOT
# style).  All sums are per image, so the joint fit accumulates
#
#   fixed level:  N = g Σ cPe iv, D = g² Σ cP² iv, V = g² Σ c²P² iv       (s = ΣN/ΣD, ν_f = ΣN/√ΣV)
#   free level b_i per image, profiled out (Schur complement), r = E/F:
#                 Ñ = N - rG, D̃ = D - rE, Ṽ = V - 2r M12 + r² M22
#                 E = g Σ cP iv, F = Σ c iv, G = Σ c e iv, M12 = g Σ c²P iv, M22 = Σ c² iv
#   χ² inside c_χ with ivχ = 1/(1/iv + μ max(e,0)²), as a quadratic in s per image after
#   substituting b_i(s) = (G - sE)/F:  χ²_i(s) = α + βs + γs²  (summed over a band's images).
#
# Per band: s_f = ΣÑ/ΣD̃, ν_f = ΣÑ/√ΣṼ, χ²_f = Σα + Σβ s_f + Σγ s_f².
# Across bands: ν = Σ w_f ν_f / √Σ w_f² (N(0,1) under the null for any weights);
# χ²_red = Σ_f χ²_f / (Σ c_χ - #bands - #levels);  snm = ν / √(χ²_red + 0.25).
# `method = :matched_filter` uses a unit box with the level and no χ².
#
# The per-image sums (`image_terms`) are one inline function shared by the map kernel, the
# per-peak kernel and the host-side wing template.

"""
    JointDetector(; method = :dolphot, rap = 1.4, rchi = 1.0, level = true, μ = 0.003,
                    mf_half = 5, Q = 32, K = 2, psf_regions = 1)

Joint detection settings.

- `method`: `:dolphot` (PSF fit in an aperture of `rap` FWHM, χ² in `rchi` FWHM) or
  `:matched_filter` (a `(2 mf_half + 1)²` unit box with the level, no χ²)
- `level`: fit a local level per image (required with real ePSFs)
- `μ`: model-error term of the χ² variance
- `Q`: subpixel phase steps per native pixel of the stamp banks
- `K`: detection grid oversampling relative to the fit frame
- `psf_regions`: `n` builds an `n x n` grid of banks per image for a spatially varying PSF
"""
Base.@kwdef struct JointDetector
    method::Symbol = :dolphot
    rap::Float64 = 1.4
    rchi::Float64 = 1.0
    level::Bool = true
    μ::Float64 = 0.003
    mf_half::Int = 5
    Q::Int = 32
    K::Int = 2
    psf_regions::Int = 1
    function JointDetector(method, rap, rchi, level, μ, mf_half, Q, K, psf_regions)
        method in (:dolphot, :matched_filter) || throw(ArgumentError("method must be :dolphot or :matched_filter, got $(repr(method))"))
        (Q > 0 && K > 0 && psf_regions > 0 && mf_half > 0) || throw(ArgumentError("Q, K, psf_regions and mf_half must be positive"))
        return new(method, rap, rchi, level, μ, mf_half, Q, K, psf_regions)
    end
end

uses_chi(det::JointDetector) = det.method === :dolphot

# Stamps of the unit PRF and the apertures for a source at offset δ = k/Q - 1/2 (k = 0..Q per
# axis) from the stamp's center pixel, index k + 1, built at PSF-frame position (Y, X).  The
# aperture weights are stored as UInt8 (round(255 w)): interior weights of 1 stay exact, edge
# weights move by at most 1/510, and the maps read a quarter (Float32) or eighth of the bytes.
function _bank(psf, det::JointDetector, Y, X)
    Q = det.Q
    fwhm = PSF.effective_fwhm(unit_psf(psf, Y, X))
    mf = det.method === :matched_filter
    h = mf ? det.mf_half : ceil(Int, det.rap * fwhm + 0.5)
    n = 2h + 1
    P = zeros(n, n, Q + 1, Q + 1); c = zeros(UInt8, size(P)); cχ = zero(c)   # cχ stays zero for :matched_filter (no χ²)
    ap(r, δy, δx) = (a = CrowdPhot.CircularAperture(h + 1 + δy, h + 1 + δx, float(r));
                     [round(UInt8, 255 * CrowdPhot.aperture_weight(a, i, j, CrowdPhot.ExactOverlap())) for i in 1:n, j in 1:n])
    for kx in 0:Q, ky in 0:Q
        δy = ky / Q - 0.5; δx = kx / Q - 0.5
        m = unit_psf(psf, Y + δy, X + δx)
        P[:, :, ky + 1, kx + 1] .= [PSF.evaluate(m, Y + a - h - 1, X + b - h - 1) for a in 1:n, b in 1:n]
        if mf
            c[:, :, ky + 1, kx + 1] .= 0xff
        else
            c[:, :, ky + 1, kx + 1] .= ap(det.rap * fwhm, δy, δx)
            cχ[:, :, ky + 1, kx + 1] .= ap(min(det.rchi, det.rap) * fwhm, δy, δx)
        end
    end
    return (; h, P, c, cχ)
end

"""
Stamp banks of every image on the device, each at its own size.  Bank `k` (half-width `h[k]`,
`n = 2h[k] + 1`) holds `(n, n, Q+1, Q+1)` stamps stored column-major in the flat vectors `P`,
`c`, `cχ` (UInt8 aperture weights, `w = c / 255`) starting after element `off[k]`
(`bank_entry`).  Image `e` uses bank `base[e] + region` of its `nr x nr` regions of `(sy, sx)`
pixels (`reg` is `(4, E)` Int32: base, nr, sy, sx).  Images with identical bank inputs (dims,
`psf_offset` and unit PSF, e.g. dithers of one detector and filter) share a base, so their
banks are built and stored once.  `hmax` is the largest half-width (the reach of an image's
apertures).  `rng` holds, per bank, phase and stamp column, the rows `(lo, hi)` (UInt8) where
the fit aperture is nonzero, starting after element `roff[k]` (`_aperture_runs`); `image_terms`
loops over those rows only.
"""
struct DeviceBanks{V, VA, VI, VO, MI}
    P::V
    c::VA
    cχ::VA
    h::VI
    off::VO
    reg::MI
    rng::VA
    roff::VO
    hmax::Int32
    Q::Int32
end
Adapt.@adapt_structure DeviceBanks

# Rows of each stamp column where the fit aperture weight is nonzero, as (lo, hi) per (column,
# phase); the aperture is a disk, so they are one run (lo > hi marks an empty column).  The χ²
# aperture is a smaller concentric disk, so its nonzero pixels are inside these runs.
function _aperture_runs(c::AbstractArray{UInt8, 4})
    n = size(c, 1)
    n <= typemax(UInt8) || throw(ArgumentError("stamp side $n exceeds the UInt8 run table"))
    r = zeros(UInt8, 2, n, size(c, 3), size(c, 4))
    for kx in axes(c, 4), ky in axes(c, 3), b in 1:n
        rows = findall(>(0), view(c, :, b, ky, kx))
        if isempty(rows)
            r[1, b, ky, kx] = 0x01
        else
            rows == first(rows):last(rows) || error("nonzero aperture rows of a stamp column are not contiguous")
            r[1, b, ky, kx] = first(rows); r[2, b, ky, kx] = last(rows)
        end
    end
    return r
end

function DeviceBanks(sc::JointScene{T}, psfs, offsets, det::JointDetector) where {T}
    E = nimages(sc); nr = det.psf_regions
    # group images by bank inputs; unit_psf drops position, flux and background
    units = [unit_psf(p, 0, 0) for p in psfs]
    reps, slot = _groups((r, e) -> sc.dims[r] == sc.dims[e] && offsets[r] == offsets[e] && _same_content(units[r], units[e]), E)
    centers = [(e, i, j) for e in reps for j in 1:nr for i in 1:nr]
    banks = Vector{Any}(undef, length(centers))
    Threads.@threads for k in eachindex(centers)
        e, i, j = centers[k]
        ny, nx = sc.dims[e]; sy, sx = cld(ny, nr), cld(nx, nr)
        Y = min((i - 0.5) * sy + 0.5, ny) + offsets[e][1]; X = min((j - 0.5) * sx + 0.5, nx) + offsets[e][2]
        banks[k] = _bank(psfs[e], det, Y, X)
    end
    # Int64 offsets: at Roman scale the banks together can exceed typemax(Int32) entries
    off = cumsum([0; [length(b.P) for b in banks]])
    flat(f) = reduce(vcat, [vec(f(b)) for b in banks])
    runs = [_aperture_runs(b.c) for b in banks]
    roff = cumsum([0; length.(runs)])
    reg = Int32[k == 1 ? (slot[e] - 1) * nr * nr : k == 2 ? nr : k == 3 ? cld(sc.dims[e][1], nr) : cld(sc.dims[e][2], nr) for k in 1:4, e in 1:E]
    be = sc.backend
    return DeviceBanks(to_device(be, T, flat(b -> b.P)), to_device(be, flat(b -> b.c)), to_device(be, flat(b -> b.cχ)), to_device(be, Int32[b.h for b in banks]), to_device(be, off[1:end-1]),
                       to_device(be, reg), to_device(be, reduce(vcat, vec.(runs))), to_device(be, roff[1:end-1]), Int32(maximum(b.h for b in banks)), Int32(det.Q))
end

Base.@propagate_inbounds function bank_index(bk::DeviceBanks, e, py, px)
    nr = bk.reg[2, e]
    return bk.reg[1, e] + min(cld(py, bk.reg[3, e]), nr) + (min(cld(px, bk.reg[4, e]), nr) - 1) * nr
end

"Flat index of stamp pixel (a, b) at phase (ky, kx) of bank kb."
Base.@propagate_inbounds function bank_entry(bk::DeviceBanks, kb, a, b, ky, kx)
    n = 2 * Int(bk.h[kb]) + 1
    return bk.off[kb] + a + (b - 1) * n + ((ky - 1) + (kx - 1) * (Int(bk.Q) + 1)) * n * n
end

@inline phase_index(δ::T, Q) where {T} = clamp(unsafe_trunc(Int32, round((δ + T(0.5)) * Q)), Int32(0), Q) + Int32(1)

"""
Per-image sums at continuous pixel position `(y, x)` of the flat image `[off + 1 : off + ny nx]`
of `data`/`iv`, with bank `kb`: `(Ñ, D̃, Ṽ, α, β, γ, dof, ok)` (`dof`: Σ c_χ minus this
image's level).  `g` is the image gain.

With the level, the sums are about the aperture means p̄ = Σcwp/Σcw and ē = Σcwe/Σcw:
Ñ = g Σ cw (p-p̄)(e-ē) is N - rG exactly, and the χ² quadratic is α = Σ wχ (e-ē)²,
β = -2g Σ wχ (e-ē)(p-p̄), γ = g² Σ wχ (p-p̄)².  Without the level, p̄ = ē = 0 and the same
sums are the fixed-level ones.

One pass forms raw moments of e - e0, with e0 the data at the center pixel, and the centered
sums follow from the usual identities (Ñ = Σcw p e - p̄ Σcw e, ...).  Centering is invariant to
the shift, and the shift keeps a pedestal or bright wings under the aperture from cancelling
(Float32 with a 1e4 pedestal stays within 2e-4 σ of Float64, as the earlier two-pass form did).
Only the rows of each stamp column inside the fit aperture are visited (`bk.rng`), the UInt8
weights are summed as integers and scaled by 1/255 once, and the inner loop is branch-free
(masked pixels get zero weight and data) so it vectorizes on the CPU and does not diverge on a
GPU.  Against the two-pass form: ν within 1e-13 in Float64 and 6e-5 in Float32.
"""
Base.@propagate_inbounds function image_terms(data, iv, off, ny, nx, y::T, x::T, g::T, bk::DeviceBanks, kb, level::Bool, chi::Bool, μ::T) where {T}
    z = zero(T)
    py = unsafe_trunc(Int32, round(y)); px = unsafe_trunc(Int32, round(x))
    ((1 <= py <= ny) & (1 <= px <= nx)) || return z, z, z, z, z, z, z, false
    Q = bk.Q
    ky = phase_index(y - T(py), Q); kx = phase_index(x - T(px), Q)
    h = bk.h[kb]; n = 2 * Int(h) + 1
    # stamp pixel (i, j) of this phase is entry s0 + (i - py) + (j - px) n
    s0 = bank_entry(bk, kb, Int(h) + 1, Int(h) + 1, ky, kx)
    ir0 = max(Int32(1), py - h); ir1 = min(Int32(ny), py + h)
    rbase = bk.roff[kb] + 2 * n * ((ky - 1) + (kx - 1) * (Int(Q) + 1))
    qc = off + py + (px - 1) * ny
    e0 = (level & (iv[qc] > 0)) ? data[qc] : z
    S0 = Sp = Se = Spp = Spe = V0 = Vp = Vpp = z
    W0 = We = Wee = Wp = Wpp = Wep = dd = z
    @inbounds for j in max(Int32(1), px - h):min(Int32(nx), px + h)
        b = Int(j - px + h)
        i0 = max(ir0, py - h - Int32(1) + Int32(bk.rng[rbase + 2b + 1]))
        i1 = min(ir1, py - h - Int32(1) + Int32(bk.rng[rbase + 2b + 2]))
        qb = off + (j - 1) * ny; sb = s0 + (j - px) * n - py
        @simd for i in i0:i1
            q = qb + i; s = sb + i
            w0 = iv[q]; m = w0 > 0
            w = ifelse(m, w0, z); ev = ifelse(m, data[q], z)
            c = T(bk.c[s]); p = bk.P[s]; e = ifelse(m, ev - e0, z)
            cw = c * w
            S0 += cw; Sp += cw * p; Se += cw * e; Spp += cw * p * p; Spe += cw * p * e
            V0 += c * cw; Vp += c * cw * p; Vpp += c * cw * p * p
            if chi
                cx = T(bk.cχ[s]); ep = max(ev, z)
                # cx / (1/w0 + μ ep²) with one division (exact for w0 > 0); precise division is slow on Metal
                wχ = ifelse(m, Base.FastMath.div_fast(cx * w0, 1 + μ * ep * ep * w0), z)
                W0 += wχ; We += wχ * e; Wee += wχ * e * e; Wp += wχ * p; Wpp += wχ * p * p; Wep += wχ * e * p
                dd += ifelse(m, cx, z)
            end
        end
    end
    pb = eb = z
    if level
        S0 > 0 || return z, z, z, z, z, z, z, false
        pb = Sp / S0; eb = Se / S0
    end
    N = Spe - pb * Se; Dd = Spp - pb * Sp; V = Vpp - 2pb * Vp + pb * pb * V0
    A = Wee - 2eb * We + eb * eb * W0; Bx = Wep - pb * We - eb * Wp + pb * eb * W0; Cc = Wpp - 2pb * Wp + pb * pb * W0
    Dd > 0 || return z, z, z, z, z, z, z, false
    k = inv(T(255))                                # the weights were summed as 0..255 integers
    N *= k; Dd *= k; V *= k * k; A *= k; Bx *= k; Cc *= k; dd *= k
    return g * N, g * g * Dd, g * g * V, A, -2g * Bx, g * g * Cc, dd - ifelse(level, one(T), z), true
end

# Fold one band's summed terms into the cross-band sums (num, den, χ, nf): s_f = Ñ/D̃,
# ν_f = Ñ/√Ṽ, num += w_f ν_f, den += w_f², χ += α + β s_f + γ s_f².
@inline function fold_band(t::SVector{6, T}, w, num, den, χ, nf) where {T}
    ((t[2] > 0) & (t[3] > 0)) || return num, den, χ, nf
    s = t[1] / t[2]
    return num + w * t[1] / sqrt(t[3]), den + w * w, χ + t[4] + t[5] * s + t[6] * s * s, nf + 1
end

# (ν, snm, χ²_red) from the cross-band sums; ν = NaN where no band has data.
@inline function finish_bands(num::T, den, χ, nf, dof, chi::Bool) where {T}
    den > 0 || return T(NaN), T(NaN), T(NaN)
    ν = num / sqrt(den)
    chi || return ν, ν, T(NaN)
    χr = χ / max(dof - nf, one(T))
    return ν, ν / sqrt(max(χr + T(0.25), T(1e-12))), χr
end

# Per-band outputs of `joint_point` (s_f and ν_f into column k of `out`); nothing for the maps.
@inline band_out!(::Nothing, f, t, nb) = nothing
@inline function band_out!(sink::Tuple, f, t::SVector{6, T}, nb) where {T}
    out, k = sink
    out[3 + f, k] = t[2] > 0 ? t[1] / t[2] : T(NaN)
    out[3 + nb + f, k] = t[3] > 0 ? t[1] / sqrt(t[3]) : T(NaN)
    return nothing
end

"""
The joint statistic at frame position (u, v) from the images `imgs[i0:i1]`, which are sorted
by band: each band's six sums are folded into the cross-band sums when the loop leaves it, so
the state is independent of the number of bands.  Returns `(ν, snm, χ²_red, nimg)`.
The kernels pass `level` and `chi` as compile-time constants.
"""
@inline function node_statistic(data, iv, d, bk, imgs, i0, i1, u::T, v::T, wf, level, chi, μ, sink, nb) where {T}
    num = den = χ = dof = zero(T); nf = 0; cnt = Int32(0)
    cur = Int32(0); acc = zero(SVector{6, T})
    @inbounds for k in i0:(i1 + 1)                # the extra step flushes the last band
        e = k <= i1 ? imgs[k] : zero(eltype(imgs))   # unused on the flush step (the list may be empty)
        f = k <= i1 ? d.band[e] : Int32(0)
        if f != cur
            if cur > 0
                num, den, χ, nf = fold_band(acc, wf[cur], num, den, χ, nf)
                band_out!(sink, cur, acc, nb)
            end
            cur = f; acc = zero(SVector{6, T})
        end
        k <= i1 || break
        ok, y, x = source_pixel(d, e, u, v)
        ok || continue
        ny = d.dims[1, e]; nx = d.dims[2, e]
        py = unsafe_trunc(Int32, round(y)); px = unsafe_trunc(Int32, round(x))
        ((1 <= py <= ny) & (1 <= px <= nx)) || continue
        t1, t2, t3, t4, t5, t6, dd, good = image_terms(data, iv, d.dims[3, e], ny, nx, y, x, d.ip[IP_GAIN, e], bk, bank_index(bk, e, py, px), level, chi, μ)
        good || continue
        acc += SVector(t1, t2, t3, t4, t5, t6)
        dof += dd; cnt += Int32(1)
    end
    a, b, c = finish_bands(num, den, χ, nf, dof, chi)
    return a, b, c, cnt
end

# Detection grid node (r, c) -> fit frame (u, v), for a grid K times finer with the same edges.
@inline node_frame(r, c, K, ::Type{T}) where {T} = ((T(r) - T(0.5)) / K + T(0.5), (T(c) - T(0.5)) / K + T(0.5))

@kernel function joint_maps_kernel!(ν, snm, χ2, nimg, @Const(data), @Const(iv), d, bk, @Const(tptr), @Const(timg), @Const(wf),
                                    K, tile, ntr, ::Val{level}, ::Val{chi}, μ, ::Val{NB}) where {level, chi, NB}
    r, c = @index(Global, NTuple)
    T = eltype(ν)
    @inbounds if (r <= size(ν, 1)) & (c <= size(ν, 2))
        t = (r - 1) ÷ tile + ((c - 1) ÷ tile) * ntr + 1
        u, v = node_frame(r, c, K, T)
        a, b, x, cnt = node_statistic(data, iv, d, bk, timg, tptr[t], tptr[t + 1] - 1, u, v, wf, level, chi, μ, nothing, NB)
        if cnt > 0
            ν[r, c] = a; snm[r, c] = b; χ2[r, c] = x
        end
        nimg[r, c] = UInt8(min(cnt, 255))
    end
end

@kernel function joint_point_kernel!(out, @Const(uv), @Const(data), @Const(iv), d, bk, @Const(tptr), @Const(timg), @Const(wf),
                                     K, tile, ntr, ngy, ngx, ::Val{level}, ::Val{chi}, μ, ::Val{NB}) where {level, chi, NB}
    k = @index(Global)
    T = eltype(out)
    u = uv[1, k]; v = uv[2, k]
    # tile of the nearest detection-grid node
    r = clamp(unsafe_trunc(Int32, round(K * (u - T(0.5)) + T(0.5))), Int32(1), Int32(ngy))
    c = clamp(unsafe_trunc(Int32, round(K * (v - T(0.5)) + T(0.5))), Int32(1), Int32(ngx))
    t = (r - 1) ÷ tile + ((c - 1) ÷ tile) * ntr + 1
    for f in 1:(2NB)
        out[3 + f, k] = T(NaN)                    # bands without data stay NaN
    end
    a, b, x, cnt = node_statistic(data, iv, d, bk, timg, tptr[t], tptr[t + 1] - 1, u, v, wf, level, chi, μ, (out, k), NB)
    out[1, k] = a; out[2, k] = b; out[3, k] = x
end

"""
    JointDetection(sc, psfs, offsets, det; weights = Dict()) -> JointDetection

Everything the detector reuses across passes: device banks, the detection grid
(`oversample(sc.frame, det.K)`), per-tile image lists and band weights (missing bands weigh 1).
`psfs`/`offsets` are the images' host PSFs and PSF offsets.
"""
struct JointDetection{B, VI, V}
    det::JointDetector
    grid::TanGrid
    banks::B
    tptr::VI
    timg::VI
    wf::V
    tile::Int
    ntr::Int
end

function JointDetection(sc::JointScene{T}, psfs, offsets, det::JointDetector; weights = Dict{Symbol, Float64}(), tile::Int = 64) where {T}
    be = sc.backend
    grid = oversample(sc.frame, det.K)
    banks = DeviceBanks(sc, psfs, offsets, det)
    hpad = maximum(Array(banks.h)) + 1           # an image reaches nodes within its aperture's half-width
    ntr, ntc = cld(grid.ny, tile), cld(grid.nx, tile)
    lists = [Int32[] for _ in 1:(ntr * ntc)]
    for (e, fp) in enumerate(sc.footprints)
        # grid rows/cols of the footprint padded by the aperture (native px -> grid px)
        r0 = floor(Int, det.K * (fp[1] - hpad - 0.5) + 0.5); r1 = ceil(Int, det.K * (fp[2] + hpad - 0.5) + 0.5)
        c0 = floor(Int, det.K * (fp[3] - hpad - 0.5) + 0.5); c1 = ceil(Int, det.K * (fp[4] + hpad - 0.5) + 0.5)
        for tc in max(1, cld(c0, tile)):min(ntc, cld(c1, tile)), tr in max(1, cld(r0, tile)):min(ntr, cld(r1, tile))
            push!(lists[tr + (tc - 1) * ntr], e)
        end
    end
    bands = Array(sc.dev.band)
    foreach(l -> sort!(l; by = e -> bands[e]), lists)   # `node_statistic` folds each band as it leaves it
    tptr = Int32.(cumsum([1; length.(lists)])); timg = reduce(vcat, lists; init = Int32[])
    wf = T[get(weights, b, 1.0) for b in sc.bands]
    return JointDetection(det, grid, banks, to_device(be, tptr), to_device(be, timg), to_device(be, wf), tile, ntr)
end

"""
    joint_maps(jd, sc, data, iv) -> (; ν, snm, χ2, nimg)

The joint statistic at every detection-grid node from flat per-image `data` (background-
subtracted residuals) and detection inverse variance `iv`, both in the scene's layout, on its
backend.  Maps stay on the device; `ν` is NaN where no image covers a node.
"""
function joint_maps(jd::JointDetection, sc::JointScene{T}, data::AbstractVector, iv::AbstractVector) where {T}
    be = sc.backend; g = jd.grid
    ν = KA.allocate(be, T, g.ny, g.nx); snm = KA.allocate(be, T, g.ny, g.nx); χ2 = KA.allocate(be, T, g.ny, g.nx)
    fill!(ν, T(NaN)); fill!(snm, T(NaN)); fill!(χ2, T(NaN))
    nimg = KA.zeros(be, UInt8, g.ny, g.nx)
    det = jd.det
    joint_maps_kernel!(be, is_cpu(be) ? (64, 1) : (16, 16))(ν, snm, χ2, nimg, data, iv, sc.dev, jd.banks, jd.tptr, jd.timg, jd.wf,
        T(det.K), jd.tile, jd.ntr, Val(det.level), Val(uses_chi(det)), T(det.μ), Val(nbands(sc)); ndrange = (g.ny, g.nx))
    return (; ν, snm, χ2, nimg)
end

"""
    joint_point(jd, sc, data, iv, uv) -> Matrix (3 + 2B, n)

The joint statistic at `n` continuous fit-frame positions `uv` (`2 x n`, device), exact per
image: rows `ν, snm, χ²_red, s_1..s_B` (per-band flux, NaN without data) and `ν_1..ν_B`.
"""
function joint_point(jd::JointDetection, sc::JointScene{T}, data, iv, uv::AbstractMatrix) where {T}
    be = sc.backend; B = nbands(sc); det = jd.det
    out = KA.allocate(be, T, 3 + 2B, size(uv, 2))
    launch!(joint_point_kernel!, be, size(uv, 2), out, uv, data, iv, sc.dev, jd.banks, jd.tptr, jd.timg, jd.wf, T(det.K), jd.tile,
            jd.ntr, jd.grid.ny, jd.grid.nx, Val(det.level), Val(uses_chi(det)), T(det.μ), Val(B))
    return out
end

# ------------------------------------------------------------------------------------------
# Peaks (flagged and refined on the device) and the wing gate (host)
# ------------------------------------------------------------------------------------------

@inline _finite_or_min(v::T) where {T} = ifelse(isfinite(v), v, typemin(T))

# 3x3 maxima of snm (strictly above all 8 neighbors; non-finite counts as -Inf) whose acceptance
# statistic passes `floor`, away from the border.
@kernel function peak_flag_kernel!(flag, @Const(S), @Const(A), floor, border)
    i, j = @index(Global, NTuple)
    ny, nx = size(S)
    f = Int32(0)
    if (border < i <= ny - border) & (border < j <= nx - border) & (1 < i < ny) & (1 < j < nx)
        a = A[i, j]
        if isfinite(S[i, j]) & isfinite(a) & (a >= floor)
            c = S[i, j]; m = true
            for dj in -1:1, di in -1:1
                ((di == 0) & (dj == 0)) && continue
                m &= c > _finite_or_min(S[i + di, j + dj])
            end
            m && (f = Int32(1))
        end
    end
    flag[i, j] = f
end

# Separable-parabola refinement of each flagged peak into column idx[t] of `out`: (y, x, ν, snm).
@kernel function peak_collect_kernel!(out, @Const(idx), @Const(flag), @Const(S), @Const(ν))
    t = @index(Global)
    if flag[t] > 0
        T = eltype(out); ny = size(S, 1); k = idx[t]
        i = (t - 1) % ny + 1; j = (t - 1) ÷ ny + 1
        c = T(S[i, j])
        s1 = T(_finite_or_min(S[i + 1, j])); s0 = T(_finite_or_min(S[i - 1, j]))
        r1 = T(_finite_or_min(S[i, j + 1])); r0 = T(_finite_or_min(S[i, j - 1]))
        dy = (s1 - s0) / (2 * (2c - s1 - s0)); dx = (r1 - r0) / (2 * (2c - r1 - r0))
        out[1, k] = T(i) + clamp(ifelse(isfinite(dy), dy, zero(T)), T(-0.5), T(0.5))
        out[2, k] = T(j) + clamp(ifelse(isfinite(dx), dx, zero(T)), T(-0.5), T(0.5))
        out[3, k] = T(ν[i, j]); out[4, k] = c
    end
end

"""
    find_peaks(maps; floor, dedup = 0.0, border = 2, accept = :nu) -> (; y, x, ν, snm, stat)

3x3 maxima of `snm`, separable-parabola refinement, and greedy suppression of weaker peaks (by
snm) within `dedup` grid pixels.  `accept` picks the statistic peaks are thresholded on
(`floor`): `:nu` (calibrated) or `:snm` (χ²-penalized).  Positions are in grid pixels.  The maps
may live on any backend: maxima are flagged, compacted and refined there, and only the
candidates come back for the sort and the suppression.
"""
function find_peaks(m; floor::Real, dedup::Real = 0.0, border::Integer = 2, accept::Symbol = :nu)
    accept in (:nu, :snm) || throw(ArgumentError("accept must be :nu or :snm, got $(repr(accept))"))
    S = m.snm; A = accept === :nu ? m.ν : m.snm; be = KA.get_backend(S); T = eltype(S)
    length(S) < typemax(Int32) || throw(ArgumentError("$(length(S)) detection grid nodes exceed the Int32 peak scan; tile the frame or lower K"))
    flag = KA.allocate(be, Int32, size(S))
    peak_flag_kernel!(be, is_cpu(be) ? (64, 1) : (16, 16))(flag, S, A, T(floor), border; ndrange = size(S))
    idx = _scan(vec(flag))                      # Int32: one per grid node
    n = isempty(idx) ? 0 : Int(Array(view(idx, length(idx):length(idx)))[1])
    out = KA.allocate(be, T, 4, max(n, 1))
    n > 0 && launch!(peak_collect_kernel!, be, length(flag), out, idx, flag, S, m.ν)
    cand = Float64.(Array(out))
    y = Float64[]; x = Float64[]; νs = Float64[]; ss = Float64[]
    cell = Dict{NTuple{2, Int}, Vector{Int}}(); cs = max(dedup, 1.0)
    # candidates are in column-major order, as a serial scan; the sort keeps ties in that order
    for p in sortperm(view(cand, 4, 1:n); rev = true)
        yy = cand[1, p]; xx = cand[2, p]
        if dedup > 0
            ky = Base.floor(Int, yy / cs); kx = Base.floor(Int, xx / cs); hit = false
            for a in -1:1, b in -1:1, k in get(cell, (ky + a, kx + b), Int[])
                (y[k] - yy)^2 + (x[k] - xx)^2 < dedup^2 && (hit = true; break)
            end
            hit && continue
            push!(get!(() -> Int[], cell, (ky, kx)), length(y) + 1)
        end
        push!(y, yy); push!(x, xx); push!(νs, cand[3, p]); push!(ss, cand[4, p])
    end
    return (; y, x, ν = νs, snm = ss, stat = accept === :nu ? νs : ss)
end

"""
    wing_template(jd, sc, ivs; R = 20.0, flux = 1e6, data = similar(sc.data), iv = similar(sc.data)) -> (; W, h, K)

The joint ν response to one isolated noiseless star, relative to its peak, on detection-grid
offsets within ±`R` native px (`W[h+1+Δr, h+1+Δc]`).  Every image covering the star
contributes its own PSF at the star's pixel position, so the template carries the diffraction
structure and relative orientations of the actual exposures.  The star (flux `flux` in every
band) sits at the frame center (a tile's core center), or, if fewer images cover the center
than cover some other point of a coarse grid of the frame (a detector gap, a mosaic edge), at
the nearest best-covered grid point; image `e` has the constant inverse variance
`ivs[e]` (its median fit weight in the loop).  The star is rendered with `render!` and the
statistic evaluated with one `joint_point` launch; `data` and `iv` are scene-sized scratch.
"""
function wing_template(jd::JointDetection, sc::JointScene{T}, ivs::AbstractVector; R::Real = 20.0, flux::Real = 1e6,
        data::AbstractVector = similar(sc.data), iv::AbstractVector = similar(sc.data)) where {T}
    be = sc.backend; K = jd.det.K; E = nimages(sc)
    # images covering (u, v), by the per-image test `joint_point` applies
    covered(u, v) = count(1:E) do e
        m = sc.maps[e]; ny, nx = sc.dims[e]
        ((m.bbox[1] <= u <= m.bbox[2]) & (m.bbox[3] <= v <= m.bbox[4])) || return false
        y, x = m(u, v)
        return (1 <= round(y) <= ny) & (1 <= round(x) <= nx)
    end
    c = ((sc.frame.ny + 1) / 2, (sc.frame.nx + 1) / 2)
    span(n, m) = range(min(1 + R, m), max(n - R, m); length = 17)       # keep the ±R template inside the frame
    cand = [c; vec([(u, v) for u in span(sc.frame.ny, c[1]), v in span(sc.frame.nx, c[2])])]
    ncov = [covered(p...) for p in cand]
    maximum(ncov) > 0 || throw(ArgumentError("no image covers the frame's template grid"))
    u0, v0 = cand[argmin(k -> (ncov[k] < maximum(ncov), hypot((cand[k] .- c)...)), eachindex(cand))]
    # render far enough that every aperture within ±R of the star sees all of its light
    reach = ceil(Int, R) + Int(jd.banks.hmax) + 2
    θ = to_device(be, T, reshape([u0; v0; fill(flux, nbands(sc))], :, 1))
    rt = RadiusTable(KA.zeros(be, T, reach, E), KA.zeros(be, T, E), Int32(reach), Int32(reach))
    render!(data, sc, θ, footprint(sc, θ, rt))
    for e in 1:E
        fill!(view(iv, (sc.offsets[e] + 1):sc.offsets[e + 1]), T(ivs[e]))
    end
    h = ceil(Int, R * K)
    uv = Matrix{Float64}(undef, 2, (2h + 1)^2 + 1); uv[:, 1] .= (u0, v0)
    k = 1
    for b in -h:h, a in -h:h
        uv[:, k += 1] .= (u0 + a / K, v0 + b / K)
    end
    ν = Float64.(Array(joint_point(jd, sc, data, iv, to_device(be, T, uv)))[1, :])
    return (; W = reshape(ν[2:end] ./ ν[1], 2h + 1, 2h + 1), h, K)
end

"""
    wing_gate(pk, tmpl; gate = 2.0, rcore = 3.0) -> Vector{Bool}

Single-pass analog of the blend gate: keep a peak only if its ν exceeds `gate` times the ν
every brighter peak within the template's reach predicts at its position through the
detector's own response (`wing_template`).  Offsets inside `rcore` native px are left to peak
finding, so close companions are never gated.  Peak positions in detection-grid pixels.
"""
function wing_gate(pk, tmpl; gate::Real = 2.0, rcore::Real = 3.0)
    h = tmpl.h; n = length(pk.y); keep = trues(n); rc2 = (rcore * tmpl.K)^2
    ord = sortperm(pk.ν; rev = true); cells = Dict{NTuple{2, Int}, Vector{Int}}()
    for i in ord
        cy = floor(Int, pk.y[i] / h); cx = floor(Int, pk.x[i] / h); pred = 0.0
        for a in -1:1, b in -1:1, j in get(cells, (cy + a, cx + b), Int[])
            dr = round(Int, pk.y[i] - pk.y[j]); dc = round(Int, pk.x[i] - pk.x[j])
            (abs(dr) <= h && abs(dc) <= h && dr^2 + dc^2 > rc2) || continue
            w = tmpl.W[dr + h + 1, dc + h + 1]
            isfinite(w) || continue            # offsets no image covered when the template was made predict nothing
            pred = max(pred, pk.ν[j] * w)
        end
        keep[i] = pk.ν[i] > gate * pred
        push!(get!(() -> Int[], cells, (cy, cx)), i)      # rejected peaks still predict: they are wings of the same star
    end
    return keep
end
