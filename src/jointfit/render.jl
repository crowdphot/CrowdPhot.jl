# Per-source box operations over every image (model render, fit-pixel mask) and the cost.
#
# Launch geometries share the per-pixel math (`box_prepare`, `psf_value`):
#   CPU: one work item per (row chunk, image).  Every source's box is clipped to the chunk's
#        rows, so no two work items write the same pixel and no atomics are needed (float
#        atomics are CAS loops on the CPU backend).  Gridded-PSF renders use the constant-weight
#        filter (`render_rows_cw!`), which differs from the GPU's per-pixel evaluation in the
#        last bits; the mask and Gaussian PSFs use `box_rows_kernel!`.
#   GPU render: each (source, image)'s box evaluated by one small workgroup into a fixed stamp
#        buffer, band of tile columns by band, then one workgroup per 16 x 16 pixel tile that has
#        sources, one item per pixel, adds the tile's sources' stamp values in ascending source
#        order (`render_tiles!`).  Each pixel adds its sources in the CPU kernel's order, so the
#        model is deterministic and bitwise equal to the CPU's (float atomics would add in a
#        run-dependent order).
#   GPU mask: one workgroup per (source, image); its items stride over the box pixels (every
#        write stores `true`).

"""
Per-(source, image) placement held fixed within a linearization, as the single-image fitter
holds its anchors and model radii: `ay`, `ax` are the rounded pixel position, `mR` the model
half-width (0: the source does not reach the image).  All `(N, E)` Int32.
"""
struct Footprint{MI}
    ay::MI
    ax::MI
    mR::MI
end

"""
`:auto` model-radius rule of the single-image fitter per image (`_model_radius_table`):
`sb` `(R_cap, E)` and `thresh` `(E,)` on the device.  A scalar `model_rad` is the table that
gives that radius to every source.
"""
struct RadiusTable{M, V}
    sb::M
    thresh::V
    R_fit::Int32
    R_cap::Int32
end

function RadiusTable(sc::JointScene{T}, psfs, weights; model_rad = :auto, nsigma = 1, R_fit::Integer, R_cap::Integer) where {T}
    E = nimages(sc)
    if model_rad isa Real
        r = clamp(round(Int, round(model_rad, RoundUp)), R_fit, R_cap)
        return RadiusTable(KA.zeros(sc.backend, T, max(r, 1), E), KA.zeros(sc.backend, T, E), Int32(r), Int32(r))
    end
    model_rad === :auto || throw(ArgumentError("`model_rad` must be a number or :auto"))
    sb = Matrix{T}(undef, R_cap, E); th = Vector{T}(undef, E)
    Threads.@threads for e in 1:E
        s, t = CrowdPhot._model_radius_table(psfs[e], nsigma, R_cap, vec(weights[e]), T)
        sb[:, e] .= s; th[e] = t
    end
    return RadiusTable(to_device(sc.backend, sb), to_device(sc.backend, th), Int32(R_fit), Int32(R_cap))
end

@kernel function footprint_kernel!(ay, ax, mR, @Const(θ), d, @Const(sb), @Const(thresh), R_fit, R_cap)
    t = @index(Global)
    N = size(θ, 2)
    s = (t - 1) % N + 1; e = (t - 1) ÷ N + 1
    T = eltype(θ)
    ok, y, x = source_pixel(d, e, θ[1, s], θ[2, s])
    r = Int32(0)
    if ok & isfinite(y) & isfinite(x)
        a = unsafe_trunc(Int32, round(y)); b = unsafe_trunc(Int32, round(x))
        F = max(d.ip[IP_SCALE, e] * θ[2 + d.band[e], s], zero(T))
        r = R_fit
        while r < R_cap && F * sb[r, e] > thresh[e]
            r += Int32(1)
        end
        ny = d.dims[1, e]; nx = d.dims[2, e]
        ((a + r >= 1) & (a - r <= ny) & (b + r >= 1) & (b - r <= nx)) || (r = Int32(0))
        ay[s, e] = a; ax[s, e] = b
    end
    mR[s, e] = r
end

"Anchors and model radii of every (source, image) at `θ` (`P x N`)."
function footprint(sc::JointScene, θ::AbstractMatrix, rt::RadiusTable)
    N, E = size(θ, 2), nimages(sc)
    fp = Footprint(KA.zeros(sc.backend, Int32, N, E), KA.zeros(sc.backend, Int32, N, E), KA.zeros(sc.backend, Int32, N, E))
    launch!(footprint_kernel!, sc.backend, N * E, fp.ay, fp.ax, fp.mR, θ, sc.dev, rt.sb, rt.thresh, rt.R_fit, rt.R_cap)
    return fp
end

# ------------------------------------------------------------------------------------------
# Box operations
# ------------------------------------------------------------------------------------------

"Add each source's model at `θ` (flux = flux_scale x band flux)."
struct RenderOp{TH, D}
    θ::TH
    d::D
end
Adapt.@adapt_structure RenderOp

@inline box_radius(::RenderOp, mR, s, e) = mR[s, e]

@inline function box_prepare(op::RenderOp, s, e)
    d = op.d
    _, y, x = source_pixel(d, e, op.θ[1, s], op.θ[2, s])
    f = d.ip[IP_SCALE, e] * op.θ[2 + d.band[e], s]
    return y, x, f, psf_context(d.psf, e, y, x)
end

# The CPU row kernel's per-pixel add (the GPU render is `render_tiles!`).
@inline function box_pixel!(op::RenderOp, out, c, q, gy, gx)
    y, x, f, pc = c
    @inbounds out[q] += f * psf_value(op.d.psf, pc, gy, gx, y, x)
    return nothing
end

"Mark the usable pixels (`w > 0`) of every fit stamp (half-width `R_fit`) of sources that reach the image."
struct MaskOp{V}
    w::V
    R_fit::Int32
end
Adapt.@adapt_structure MaskOp

@inline box_radius(op::MaskOp, mR, s, e) = ifelse(mR[s, e] > 0, op.R_fit, Int32(0))
@inline box_prepare(::MaskOp, s, e) = nothing
@inline function box_pixel!(op::MaskOp, out, c, q, gy, gx)
    # concurrent GPU writes all store `true`
    @inbounds op.w[q] > 0 && (out[q] = true)
    return nothing
end

@kernel function box_rows_kernel!(op, out, @Const(ay), @Const(ax), @Const(mR), d, nchunk)
    t = @index(Global)
    c = (t - 1) % nchunk + 1; e = (t - 1) ÷ nchunk + 1
    ny = d.dims[1, e]; nx = d.dims[2, e]; off = d.dims[3, e]
    h = cld(ny, nchunk)
    r0 = (c - 1) * h + 1; r1 = min(c * h, ny)
    for s in 1:size(ay, 1)
        R = box_radius(op, mR, s, e)
        R > 0 || continue
        a = ay[s, e]; b = ax[s, e]
        lo = max(r0, a - R); hi = min(r1, a + R)
        lo <= hi || continue
        ctx = box_prepare(op, s, e)
        for gx in max(Int32(1), b - R):min(nx, b + R), gy in lo:hi
            box_pixel!(op, out, ctx, off + gy + (gx - 1) * ny, gy, gx)
        end
    end
end

# CPU render of gridded PSFs as a constant-weight filter.  With integer oversampling s a node's
# coordinate moves by exactly s per pixel, so a source's fractional offset in the node grid, and
# with it its 4-tap Catmull-Rom weights, is the same at every pixel of its box; pixel (i, j) reads
# the node block starting s rows and s columns further per pixel.  Where no stencil is clamped
# (the interior: one contiguous range of rows and of columns per source and node), the bicubic is
# a separable filter, 4 taps along each needed node row (contiguous in memory) and then 4 taps down
# every s-th row: no per-pixel index arithmetic and no gathers, 3-4x the per-pixel evaluation of
# `box_rows_kernel!` (full Roman detector, Apple M5 Max, 1 or 18 threads, Float32 or Float64).
# Pixels outside the interior, and nodes with non-integer oversampling, use `node_value_grad`.
# TODO: the loops use @inbounds @simd: LV.@turbo did not speed them up on Apple ARM (128-bit NEON,
# 7-16% slower here); test @turbo on x86 (AVX2 / AVX-512), where its wider vectors may pay.

# Pixels p in p0:p1 whose stencil start l0 + s (p - pa) lies in 2:(n - 2) (no clamping).
cw_interior(p0, p1, pa, l0, s, n) = (max(p0, pa + cld(2 - l0, s)), min(p1, pa + fld(n - 2 - l0, s)))

# Node k's value at rows lo:hi, columns c0:c1 for a source at (Y, X), into P (h x w); H holds the
# row pass.  The weights are taken at the unclipped box origin (pa, qa), so a pixel's value does not
# depend on the row chunk (or the thread count).
function cw_node!(P, H, nodes, par, k, lo, hi, c0, c1, pa, qa, Y::T, X::T) where {T}
    ox = par[1, k]; oy = par[2, k]; sx = par[3, k]; sy = par[4, k]
    nyd = size(nodes, 1); nxd = size(nodes, 2)
    ia, ib, ja, jb = 1, 0, 1, 0
    s_x = round(Int, sx); s_y = round(Int, sy)
    ua = sx * (T(qa) - X) + ox; va = sy * (T(pa) - Y) + oy
    if (sx == s_x) & (sy == s_y) & (s_x > 0) & (s_y > 0) & isfinite(ua) & isfinite(va)
        lxa = floor(Int, ua); lya = floor(Int, va)
        ja, jb = cw_interior(c0, c1, qa, lxa, s_x, nxd); ia, ib = cw_interior(lo, hi, pa, lya, s_y, nyd)
        if (ja <= jb) & (ia <= ib)
            wu1, wu2, wu3, wu4 = cr_weights(ua - T(lxa)); wv1, wv2, wv3, wv4 = cr_weights(va - T(lya))
            # first stencil row and column of the interior; the rows its pixel rows read
            R0 = lya + s_y * (ia - pa) - 1; X0 = lxa + s_x * (ja - qa) - 1
            h = ib - ia + 1; w = jb - ja + 1; nR = s_y * (h - 1) + 4
            @inbounds for jj in 1:w
                xb = X0 + s_x * (jj - 1)
                @simd for r in 1:nR
                    q = R0 + r - 1
                    H[r, jj] = wu1 * nodes[q, xb, k] + wu2 * nodes[q, xb + 1, k] + wu3 * nodes[q, xb + 2, k] + wu4 * nodes[q, xb + 3, k]
                end
            end
            io = ia - lo; jo = ja - c0
            @inbounds for jj in 1:w
                @simd for ii in 1:h
                    rb = s_y * (ii - 1)
                    P[io + ii, jo + jj] = wv1 * H[rb + 1, jj] + wv2 * H[rb + 2, jj] + wv3 * H[rb + 3, jj] + wv4 * H[rb + 4, jj]
                end
            end
        end
    end
    @inbounds for (jj, j) in enumerate(c0:c1), (ii, i) in enumerate(lo:hi)
        ((ia <= i <= ib) & (ja <= j <= jb)) && continue
        P[ii, jj] = node_value_grad(nodes, par, k, i, j, Y, X)[1]
    end
    return P
end

# Item (chunk c, image e) of `render_rows_cw!`: its rows of every source's box, sources ascending.
function cw_rows!(out, op::RenderOp, fp::Footprint, e, c, nchunk, pv, acc, H)
    d = op.d; ny = Int(d.dims[1, e]); nx = Int(d.dims[2, e]); off = Int(d.dims[3, e])
    nodes = d.psf.nodes; par = d.psf.par; T = eltype(out)
    hc = cld(ny, nchunk); r0 = (c - 1) * hc + 1; r1 = min(c * hc, ny)
    @inbounds for s in axes(fp.ay, 1)
        R = Int(fp.mR[s, e]); R > 0 || continue
        a = Int(fp.ay[s, e]); b = Int(fp.ax[s, e])
        lo = max(r0, a - R); hi = min(r1, a + R); c0 = max(1, b - R); c1 = min(nx, b + R)
        ((lo <= hi) & (c0 <= c1)) || continue
        y, x, f, (ks, ws, _, _) = box_prepare(op, s, e)
        h = hi - lo + 1; w = c1 - c0 + 1
        A = view(acc, 1:h, 1:w); P = view(pv, 1:h, 1:w)
        fill!(A, zero(T))
        for q in 1:4
            iszero(ws[q]) && continue                 # as psf_value: zero-weight corners are skipped
            cw_node!(P, H, nodes, par, ks[q], lo, hi, c0, c1, a - R, b - R, y, x)
            wq = ws[q]
            for jj in 1:w
                @simd for ii in 1:h
                    A[ii, jj] += wq * P[ii, jj]
                end
            end
        end
        base = off + lo - 1 + (c0 - 1) * ny
        for jj in 1:w
            @simd for ii in 1:h
                out[base + ii + (jj - 1) * ny] += f * A[ii, jj]
            end
        end
    end
end

"Add the model of footprint `fp`'s sources into `out` (CPU, gridded PSFs) by the constant-weight filter, in row chunks."
function render_rows_cw!(out, op::RenderOp, sc::JointScene, fp::Footprint, nchunk = max(1, cld(8 * Threads.nthreads(), nimages(sc))))
    E = nimages(sc); T = eltype(out); nt = Threads.nthreads()
    S = 2 * Int(maximum(fp.mR)) + 1
    nH = ceil(Int, maximum(view(op.d.psf.par, 4, :))) * (S - 1) + 4
    # scratch per task, not per item (a box's value and sum, the row pass)
    pool = Channel{NTuple{3, Matrix{T}}}(nt)
    for _ in 1:nt
        put!(pool, (Matrix{T}(undef, S, S), Matrix{T}(undef, S, S), Matrix{T}(undef, nH, S)))
    end
    Threads.@threads :dynamic for t in 1:(nchunk * E)
        buf = take!(pool)
        cw_rows!(out, op, fp, (t - 1) ÷ nchunk + 1, (t - 1) % nchunk + 1, nchunk, buf...)
        put!(pool, buf)
    end
    return out
end

const BOX_GROUP = 64

@kernel function box_groups_kernel!(op, out, @Const(ay), @Const(ax), @Const(mR), d, N)
    li = @index(Local)
    g = @index(Group)
    s = (g - 1) % N + 1; e = (g - 1) ÷ N + 1
    R = box_radius(op, mR, s, e)
    if R > 0
        ny = d.dims[1, e]; nx = d.dims[2, e]; off = d.dims[3, e]
        a = ay[s, e]; b = ax[s, e]
        ctx = box_prepare(op, s, e)
        S = 2R + 1
        m = li
        while m <= S * S
            gy = a - R + Int32((m - 1) % S); gx = b - R + Int32((m - 1) ÷ S)
            if (1 <= gy <= ny) & (1 <= gx <= nx)
                box_pixel!(op, out, ctx, off + gy + (gx - 1) * ny, gy, gx)
            end
            m += BOX_GROUP
        end
    end
end

const TILE = 16
const TILE_G = TILE * TILE

# The images' tiles: `nty[e]` tiles per column and `toff[e]` (0-based) the first in the flat list.
function tile_layout(sc::JointScene)
    nty = Int32[cld(d[1], TILE) for d in sc.dims]; ntx = Int32[cld(d[2], TILE) for d in sc.dims]
    return nty, Int32.(cumsum([0; nty .* ntx]))
end

# The image of tile g: the last e with toff[e] < g (binary search; toff is sorted).
@inline function tile_image(g, toff, E)
    lo = Int32(1); hi = Int32(E)
    while lo < hi
        m = (lo + hi + Int32(1)) ÷ Int32(2)
        toff[m] < g ? (lo = m) : (hi = m - Int32(1))
    end
    return lo
end

# Image and 0-based first row and column of tile g, in Int32: Metal's group index is UInt32, so
# mixed arithmetic would promote a negative value to a huge unsigned one (an InexactError).
@inline function tile_origin(g, toff, nty, E)
    e = tile_image(g, toff, E); t = (g - 1 - toff[e]) % Int32
    return e, (t % nty[e]) * Int32(TILE), (t ÷ nty[e]) * Int32(TILE)
end

# 0-based (row, column) of item li in its tile: each 32-item warp covers an 8 x 4 block, which a
# typical box crosses with fewer partly idle warps than a 16 x 2 column pair (~10% faster).
@inline function tile_pixel(li)
    w = (li - 1) ÷ 32; l = (li - 1) % 32
    return (w % 2) * 8 + l % 8, (w ÷ 2) * 4 + l ÷ 8
end

# Count (`fill = false`) or place (`true`) every (source, image) box into the tiles it overlaps.
@kernel function tile_bin_kernel!(tcnt, tlist, tfill, @Const(tptr), @Const(ay), @Const(ax), @Const(mR), d, @Const(nty), @Const(toff), fill::Bool)
    t = @index(Global)
    N = size(ay, 1); s = (t - 1) % N + 1; e = (t - 1) ÷ N + 1
    R = mR[s, e]
    if R > 0
        ny = d.dims[1, e]; nx = d.dims[2, e]; a = ay[s, e]; b = ax[s, e]
        T0 = Int32(TILE)
        r0 = max(Int32(1), a - R); r1 = min(ny, a + R); c0 = max(Int32(1), b - R); c1 = min(nx, b + R)
        for tx in ((c0 - Int32(1)) ÷ T0):((c1 - Int32(1)) ÷ T0), ty in ((r0 - Int32(1)) ÷ T0):((r1 - Int32(1)) ÷ T0)
            tile = toff[e] + tx * nty[e] + ty + Int32(1)
            if fill
                k = Atomix.@atomic tfill[tile] += Int32(1)    # the post-increment value
                tlist[tptr[tile] + k - Int32(1)] = s % Int32
            else
                Atomix.@atomic tcnt[tile + 1] += Int32(1)
            end
        end
    end
end

# Per tile, the sources whose boxes overlap it (ascending: the atomic slots are then sorted), the
# batches of TILE_G sources the render needs for the fullest tile, and the non-empty tiles (`tid`,
# and `tidh` on the host), the only ones the render launches: empty tiles add nothing, and launching
# them cost a sparse or one-source render most of its time.
function tile_bins(sc::JointScene, fp::Footprint)
    be = sc.backend; N, E = size(fp.ay)
    nty, toff = tile_layout(sc); NT = Int(toff[end])
    ntyd = to_device(be, nty); toffd = to_device(be, toff)
    tptr = KA.zeros(be, Int32, NT + 1)
    launch!(tile_bin_kernel!, be, N * E, tptr, tptr, tptr, tptr, fp.ay, fp.ax, fp.mR, sc.dev, ntyd, toffd, false)
    fill!(view(tptr, 1:1), one(Int32)); scan!(tptr)
    ph = Array(tptr); nl = Int(ph[end]) - 1
    cnt = diff(ph); nb = nl == 0 ? 0 : cld(maximum(cnt), TILE_G)
    tidh = Int32.(findall(>(0), cnt)); tid = to_device(be, tidh)
    tlist = KA.allocate(be, Int32, max(nl, 1)); tfill = KA.zeros(be, Int32, NT)
    launch!(tile_bin_kernel!, be, N * E, tptr, tlist, tfill, tptr, fp.ay, fp.ax, fp.mR, sc.dev, ntyd, toffd, true)
    launch!(segment_sort_kernel!, be, NT, tlist, tptr)
    return (; tptr, tlist, tid, tidh, nty = ntyd, toff = toffd, nb, tfill)
end

# The tile gather evaluated the PSF per pixel, and on small boxes most items of a SIMD group sat
# idle (a 7 x 7 box covers few pixels of the 8 x 4 blocks it touches; ~12% of the lane work was
# useful on a Roman detector).  `render_tiles!` evaluates each (source, image)'s box first, one
# RENDER_G-item group per box, into a stamp buffer, and the gather then reads the stamps: the same
# products added in the same order, so the model is unchanged bit for bit (and equal to the CPU
# row kernel's).  The render runs during the fit while the linearization is live, i.e. at the run's
# peak memory, so the buffer has a fixed size (RENDER_WORDS) and is filled band by band: a band is a
# run of tile columns of one image whose boxes' stamps fit.  Every tile lies in one band, so every
# pixel is finished within a band; a box crossing a band edge is evaluated in each band it touches.
# The pairs are bucketed once by the first tile column their box touches, so a band's candidates
# are one contiguous slice of the buckets.  One small readback per render (besides tile_bins')
# plans the bands; the band loop has no host reads.
const RENDER_WORDS = 2^24      # stamp buffer words (64 MiB in Float32): ~1% of a Roman detector's peak
const RENDER_G = 32            # items per stamp group (one SIMD group on Apple and NVIDIA GPUs)
const RENDER_SPAN = 256        # bins of the box column-span histogram (tile columns)

# The box of (s, e) clipped to the image, as tile_bin_kernel! clips it (empty: r0 > r1 or c0 > c1).
@inline function render_box(ay, ax, mR, d, s, e)
    R = mR[s, e]; ny = d.dims[1, e]; nx = d.dims[2, e]; a = ay[s, e]; b = ax[s, e]
    return R, max(Int32(1), a - R), min(ny, a + R), max(Int32(1), b - R), min(nx, b + R)
end
@inline render_active(R, r0, r1, c0, c1) = (R > 0) & (r0 <= r1) & (c0 <= c1)

# Per pair, into the packed H (1 : NC words per tile column, NC+1 : 2NC pairs per first column,
# then the histogram of column spans): its stamp size added to every column its box touches, a
# count in its first column's bucket and its span.  Integer atomics: the sums are order-free (Int32:
# 64-bit atomics are not portable; a column's total stays far below 2^31).
@kernel function render_cols_kernel!(H, @Const(ay), @Const(ax), @Const(mR), d, @Const(coff), NC)
    t = @index(Global)
    N = size(ay, 1); s = (t - 1) % N + 1; e = (t - 1) ÷ N + 1
    R, r0, r1, c0, c1 = render_box(ay, ax, mR, d, s, e)
    if render_active(R, r0, r1, c0, c1)
        S = (Int32(2) * R + Int32(1)) * (Int32(2) * R + Int32(1))
        t0 = (c0 - Int32(1)) ÷ Int32(TILE); t1 = (c1 - Int32(1)) ÷ Int32(TILE)
        for tx in t0:t1
            Atomix.@atomic H[coff[e] + tx + 1] += S
        end
        Atomix.@atomic H[NC + coff[e] + t0 + 1] += Int32(1)
        Atomix.@atomic H[2NC + min(t1 - t0, Int32(RENDER_SPAN - 1)) + 1] += Int32(1)
    end
end

# Every active pair's source in its first column's bucket (bstart: 1-based bucket starts).  The
# order within a bucket is run-dependent; it only places stamps in the buffer, and the gather finds
# each source's stamp through `off`, so the model does not depend on it.
@kernel function render_bucket_kernel!(plist, bfill, @Const(bstart), @Const(ay), @Const(ax), @Const(mR), d, @Const(coff))
    t = @index(Global)
    N = size(ay, 1); s = (t - 1) % N + 1; e = (t - 1) ÷ N + 1
    R, r0, r1, c0, c1 = render_box(ay, ax, mR, d, s, e)
    if render_active(R, r0, r1, c0, c1)
        b = coff[e] + (c0 - Int32(1)) ÷ Int32(TILE) + 1
        k = Atomix.@atomic bfill[b] += Int32(1)              # the post-increment value
        plist[bstart[b] + k - Int32(1)] = s % Int32
    end
end

# Band candidate i (source plist[lo + i - 1] of image e): its stamp size into bsz[i + 1] if its box
# touches tile columns ca : cb, else 0.
@kernel function render_size_kernel!(bsz, @Const(plist), lo, @Const(ay), @Const(ax), @Const(mR), d, e, ca, cb)
    i = @index(Global)
    R, r0, r1, c0, c1 = render_box(ay, ax, mR, d, plist[lo + i - 1], e)
    v = Int32(0)
    if ((c0 - Int32(1)) ÷ Int32(TILE) <= cb) & ((c1 - Int32(1)) ÷ Int32(TILE) >= ca)
        v = (Int32(2) * R + Int32(1)) * (Int32(2) * R + Int32(1))
    end
    bsz[i + 1] = v
end

# Each band source's stamp start (bsz scanned from 1), by source, for the gather.
@kernel function render_off_kernel!(off, @Const(bsz), @Const(plist), lo)
    i = @index(Global)
    bsz[i + 1] > bsz[i] && (off[plist[lo + i - 1]] = bsz[i])
end

# Each band source's setup (y, x, f and the PSF context) once, so that the stamp groups start at once.
@kernel function render_ctx_kernel!(C, op, @Const(bsz), @Const(plist), lo, e)
    i = @index(Global)
    if bsz[i + 1] > bsz[i]
        y, x, f, c = box_prepare(op, plist[lo + i - 1], e)
        C[1, i] = y; C[2, i] = x; C[3, i] = f
        ctx_store!(view(C, 4:(3 + CTX_N), :), i, op.d.psf, c)
    end
end

# One RENDER_G-item group per band candidate k: f P over the whole box, column-major, at bsz[k]
# (0 off the image; those pixels are never read).
@kernel function render_stamp_kernel!(stamp, op, @Const(C), @Const(bsz), @Const(plist), lo, @Const(ay), @Const(ax), @Const(mR), e)
    li = @index(Local); k = Int32(@index(Group))
    if bsz[k + 1] > bsz[k]
        s = plist[lo + k - Int32(1)]
        d = op.d; ny = d.dims[1, e]; nx = d.dims[2, e]
        R = mR[s, e]; a = ay[s, e]; b = ax[s, e]; S = Int32(2) * R + Int32(1)
        y = C[1, k]; x = C[2, k]; f = C[3, k]; c = ctx_load(d.psf, view(C, 4:(3 + CTX_N), :), k)
        o = bsz[k] - Int32(1); m = Int32(li)
        while m <= S * S
            gy = a - R + (m - Int32(1)) % S; gx = b - R + (m - Int32(1)) ÷ S
            v = zero(eltype(stamp))
            ((Int32(1) <= gy <= ny) & (Int32(1) <= gx <= nx)) && (v = f * psf_value(d.psf, c, gy, gx, y, x))
            stamp[o + m] = v
            m += Int32(RENDER_G)
        end
    end
end

# Group k gathers the band's tile tid[k0 + k] of image e: per batch of TILE_G sources, item j stages
# source j's box and stamp start; then every item adds, for its pixel, the batch's sources whose
# boxes cover it, in list order.  The CPU backend keeps no variables across @synchronize, so each
# segment recomputes its indices and the loop runs `nb` (the fullest tile's) batches in every group.
@kernel function render_gather_kernel!(out, @Const(stamp), @Const(off), e, d, @Const(tid), k0, @Const(tptr), @Const(tlist), @Const(ay), @Const(ax),
                                       @Const(mR), @Const(nty), @Const(toff), E, nb)
    sA = @localmem Int32 (TILE_G,); sB = @localmem Int32 (TILE_G,); sR = @localmem Int32 (TILE_G,); sO = @localmem Int32 (TILE_G,)
    acc = @private eltype(out) (1,)
    acc[1] = zero(eltype(out))
    for bt in 1:nb
        li = @index(Local, Linear); g = tid[k0 + @index(Group, Linear)]
        l = tptr[g] + (bt - 1) * TILE_G + li - 1
        if l < tptr[g + 1]
            s = tlist[l]
            sA[li] = ay[s, e]; sB[li] = ax[s, e]; sR[li] = mR[s, e]; sO[li] = off[s]
        end
        @synchronize
        li = @index(Local, Linear); g = tid[k0 + @index(Group, Linear)]
        _, ty0, tx0 = tile_origin(g, toff, nty, E)
        py, px = tile_pixel(li)
        gy = ty0 + py + Int32(1); gx = tx0 + px + Int32(1)
        for j in 1:min(TILE_G, tptr[g + 1] - tptr[g] - (bt - 1) * TILE_G)
            R = sR[j]; dy = gy - sA[j]; dx = gx - sB[j]
            if (abs(dy) <= R) & (abs(dx) <= R)
                acc[1] += stamp[sO[j] + (dx + R) * (Int32(2) * R + Int32(1)) + dy + R]
            end
        end
        @synchronize
    end
    li = @index(Local, Linear); g = tid[k0 + @index(Group, Linear)]
    _, ty0, tx0 = tile_origin(g, toff, nty, E)
    py, px = tile_pixel(li)
    gy = ty0 + py + Int32(1); gx = tx0 + px + Int32(1)
    ny = d.dims[1, e]
    ((gy <= ny) & (gx <= d.dims[2, e])) && (out[d.dims[3, e] + gy + (gx - 1) * ny] += acc[1])
end

"""
Add the model of footprint `fp`'s sources into `out`: per-box stamps evaluated band by band into
a buffer of `words` (at least the fullest tile column's stamps), then the tile gather (any
backend; the GPU render).
"""
function render_tiles!(out, op::RenderOp, sc::JointScene, fp::Footprint; words::Integer = RENDER_WORDS)
    be = sc.backend; N, E = size(fp.ay); T = eltype(out)
    N == 0 && return out
    t = tile_bins(sc, fp)
    if t.nb > 0
        nty, toff = tile_layout(sc); ntx = Int32[cld(dm[2], TILE) for dm in sc.dims]
        coff = Int32.(cumsum([0; ntx])); NC = Int(coff[end]); coffd = to_device(be, coff[1:(end - 1)])
        H = KA.zeros(be, Int32, 2NC + RENDER_SPAN)
        launch!(render_cols_kernel!, be, N * E, H, fp.ay, fp.ax, fp.mR, sc.dev, coffd, Int32(NC))
        Hh = Int.(Array(H))
        W = view(Hh, 1:NC); hs = view(Hh, (2NC + 1):(2NC + RENDER_SPAN))
        span = hs[end] > 0 ? Int(maximum(ntx)) : findlast(>(0), hs) - 1      # columns a box spans beyond its first
        bstart = Int32.(cumsum([1; view(Hh, (NC + 1):2NC)]))                # bucket b: plist[bstart[b] : bstart[b + 1] - 1]
        plist = KA.allocate(be, Int32, Int(bstart[end]) - 1); bfill = KA.zeros(be, Int32, NC); bstartd = to_device(be, bstart)
        launch!(render_bucket_kernel!, be, N * E, plist, bfill, bstartd, fp.ay, fp.ax, fp.mR, sc.dev, coffd)
        nw = Int(min(max(words, maximum(W)), sum(W)))
        bands = NTuple{3, Int}[]                                             # (image, first and last tile column, 0-based)
        for e in 1:E
            ca = 0; acc = 0
            for c in 0:(ntx[e] - 1)
                w = W[coff[e] + c + 1]
                if acc + w > nw && c > ca
                    push!(bands, (e, ca, c - 1)); ca = c; acc = 0
                end
                acc += w
            end
            push!(bands, (e, ca, ntx[e] - 1))
        end
        clo(e, ca) = Int(bstart[coff[e] + max(0, ca - span) + 1]); chi(e, cb) = Int(bstart[coff[e] + cb + 2]) - 1
        ncmax = maximum(chi(e, cb) - clo(e, ca) + 1 for (e, ca, cb) in bands)
        stamp = KA.allocate(be, T, nw); bsz = KA.zeros(be, Int32, ncmax + 1); off = KA.zeros(be, Int32, N); C = KA.allocate(be, T, 3 + CTX_N, max(ncmax, 1))
        for (e, ca, cb) in bands
            k0 = searchsortedfirst(t.tidh, toff[e] + ca * nty[e] + 1) - 1; k1 = searchsortedlast(t.tidh, toff[e] + (cb + 1) * nty[e])
            lo = clo(e, ca); nc = chi(e, cb) - lo + 1
            (k1 > k0 && nc > 0) || continue
            launch!(render_size_kernel!, be, nc, bsz, plist, Int32(lo), fp.ay, fp.ax, fp.mR, sc.dev, Int32(e), Int32(ca), Int32(cb))
            fill!(view(bsz, 1:1), one(Int32)); scan!(view(bsz, 1:(nc + 1)))
            launch!(render_off_kernel!, be, nc, off, bsz, plist, Int32(lo))
            launch!(render_ctx_kernel!, be, nc, C, op, bsz, plist, Int32(lo), Int32(e))
            render_stamp_kernel!(be, RENDER_G)(stamp, op, C, bsz, plist, Int32(lo), fp.ay, fp.ax, fp.mR, Int32(e); ndrange = RENDER_G * nc)
            render_gather_kernel!(be, TILE_G)(out, stamp, off, Int32(e), sc.dev, t.tid, Int32(k0), t.tptr, t.tlist, fp.ay, fp.ax, fp.mR, t.nty,
                                              t.toff, E, t.nb; ndrange = TILE_G * (k1 - k0))
        end
        KA.synchronize(be)
        foreach(KA.unsafe_free!, (H, coffd, plist, bfill, bstartd, stamp, bsz, off, C))
    end
    KA.synchronize(be)
    foreach(KA.unsafe_free!, (t.tptr, t.tlist, t.tid, t.nty, t.toff, t.tfill))
    return out
end

"Apply `op` to every (source, image) box of footprint `fp`, accumulating into the flat `out`."
function box_apply!(op, out, sc::JointScene, fp::Footprint)
    be = sc.backend; N, E = size(fp.ay)
    N == 0 && return out
    if is_cpu(be) && op isa RenderOp && op.d.psf isa DevGridded
        render_rows_cw!(out, op, sc, fp)
    elseif is_cpu(be)
        nchunk = max(1, cld(8 * Threads.nthreads(), E))
        launch!(box_rows_kernel!, be, nchunk * E, op, out, fp.ay, fp.ax, fp.mR, sc.dev, nchunk)
    elseif op isa RenderOp
        render_tiles!(out, op, sc, fp)
    else
        box_groups_kernel!(be, BOX_GROUP)(op, out, fp.ay, fp.ax, fp.mR, sc.dev, N; ndrange = BOX_GROUP * N * E)
    end
    return out
end

"Model of all sources at `θ` into the flat `model` (zeroed first), with anchors and radii from `fp`."
function render!(model::AbstractVector, sc::JointScene, θ::AbstractMatrix, fp::Footprint)
    fill!(model, zero(eltype(model)))
    return box_apply!(RenderOp(θ, sc.dev), model, sc, fp)
end

"Union of the fit stamps' usable pixels, as a flat Bool mask (the cost's support)."
function fit_mask!(mask::AbstractVector{Bool}, sc::JointScene, fp::Footprint, R_fit::Integer)
    fill!(mask, false)
    return box_apply!(MaskOp(sc.weight, Int32(R_fit)), mask, sc, fp)
end

const COST_GROUP = 256

# The fit data at pixel i: the scene's data, less the background when one is given.
@inline _fit_data(data, ::Nothing, i) = data[i]
@inline _fit_data(data, bkg, i) = data[i] - bkg[i]

# Weighted residual into buf and its Σ² per workgroup into part[g], in one pass.  On the CPU each
# item covers `step` consecutive pixels (a serial, cache-friendly run per item); on a GPU items
# stride by the grid (`step` is the grid size).
@kernel function cost_kernel!(buf, part, @Const(w), @Const(model), @Const(data), bkg, @Const(mask), n, step, cpu::Bool)
    li = @index(Local); g = @index(Group); t = @index(Global)
    T = eltype(buf)
    sh = @localmem eltype(buf) (COST_GROUP,)
    acc = zero(T)
    i = cpu ? (t - 1) * step + 1 : t
    i1 = cpu ? min(t * step, n) : n
    while i <= i1
        v = ifelse(mask[i], sqrt(w[i]) * (model[i] - _fit_data(data, bkg, i)), zero(T))
        buf[i] = v; acc += v * v
        i += cpu ? 1 : step
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
    li == 1 && (part[g] = sh[1] + sh[2])
end

"""
Per-workgroup partial sums for `cost!`: 256 on every backend, so the summation order, and with it
the fit, does not depend on the thread count.
"""
cost_parts(be, ::Type{T}) where {T} = KA.zeros(be, T, 256)

"Launch the cost kernel: `buf` and the partials `part` are written, nothing is read back."
function cost_partials!(buf, sc::JointScene, model, mask, part, bkg = nothing)
    be = sc.backend; n = length(buf); G = length(part)
    step = is_cpu(be) ? cld(n, G * COST_GROUP) : G * COST_GROUP
    cost_kernel!(be, COST_GROUP)(buf, part, sc.weight, model, sc.data, bkg, mask, n, step, is_cpu(be); ndrange = G * COST_GROUP)
    return part
end

"""
    cost!(buf, sc, model, mask, part, bkg = nothing) -> Float64

Weighted squared residual `Σ w (model - d)²` over the mask, with the fit data `d = sc.data - bkg`
(`sc.data` without a background); `buf` receives the weighted residual `sqrt(w) (model - d)`
(zero off the mask).  One fused kernel writes per-workgroup partial sums into `part`
(`cost_parts`), which the host adds in Float64.
"""
cost!(buf::AbstractVector, sc::JointScene, model::AbstractVector, mask::AbstractVector{Bool}, part::AbstractVector, bkg = nothing) =
    sum(Float64, Array(cost_partials!(buf, sc, model, mask, part, bkg)))
