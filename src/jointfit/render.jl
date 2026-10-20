# Per-source box operations over every image (model render, fit-pixel mask) and the cost.
#
# Launch geometries share the per-pixel math (`box_prepare`, `psf_value`):
#   CPU: one work item per (row chunk, image).  Every source's box is clipped to the chunk's
#        rows, so no two work items write the same pixel and no atomics are needed (float
#        atomics are CAS loops on the CPU backend).
#   GPU render: one workgroup per 16 x 16 pixel tile that has sources, one item per pixel, over
#        the tile's sources in ascending order (`render_tiles!`).  Each pixel adds its sources in
#        the CPU kernel's order, so the model is deterministic and bitwise equal to the CPU's
#        (float atomics would add in a run-dependent order).
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
# batches of TILE_G sources the render needs for the fullest tile, and the non-empty tiles (`tid`),
# the only ones the render launches: empty tiles add nothing, and launching them cost a sparse or
# one-source render most of its time.
function tile_bins(sc::JointScene, fp::Footprint)
    be = sc.backend; N, E = size(fp.ay)
    nty, toff = tile_layout(sc); NT = Int(toff[end])
    ntyd = to_device(be, nty); toffd = to_device(be, toff)
    tptr = KA.zeros(be, Int32, NT + 1)
    launch!(tile_bin_kernel!, be, N * E, tptr, tptr, tptr, tptr, fp.ay, fp.ax, fp.mR, sc.dev, ntyd, toffd, false)
    fill!(view(tptr, 1:1), one(Int32)); scan!(tptr)
    ph = Array(tptr); nl = Int(ph[end]) - 1
    cnt = diff(ph); nb = nl == 0 ? 0 : cld(maximum(cnt), TILE_G)
    tid = to_device(be, Int32.(findall(>(0), cnt)))
    tlist = KA.allocate(be, Int32, max(nl, 1)); tfill = KA.zeros(be, Int32, NT)
    launch!(tile_bin_kernel!, be, N * E, tptr, tlist, tfill, tptr, fp.ay, fp.ax, fp.mR, sc.dev, ntyd, toffd, true)
    launch!(segment_sort_kernel!, be, NT, tlist, tptr)
    return (; tptr, tlist, tid, nty = ntyd, toff = toffd, nb, tfill)
end

# Group k renders the non-empty tile tid[k].  Per batch, item j sets up the batch's source j (`box_prepare`,
# into local memory); then every item adds, for its pixel, the batch's sources whose boxes cover
# it, in list order.  The CPU backend keeps no variables across @synchronize, so each segment
# recomputes its indices and the loop runs `nb` (the fullest tile's) batches in every group.
@kernel function render_tiles_kernel!(out, op, @Const(tid), @Const(tptr), @Const(tlist), @Const(ay), @Const(ax), @Const(mR), @Const(nty), @Const(toff), E, nb)
    sY = @localmem eltype(out) (TILE_G,); sX = @localmem eltype(out) (TILE_G,); sF = @localmem eltype(out) (TILE_G,)
    sC = @localmem eltype(out) (CTX_N, TILE_G)
    sA = @localmem Int32 (TILE_G,); sB = @localmem Int32 (TILE_G,); sR = @localmem Int32 (TILE_G,)
    acc = @private eltype(out) (1,)
    acc[1] = zero(eltype(out))
    for b in 1:nb
        li = @index(Local, Linear); g = tid[@index(Group, Linear)]
        l = tptr[g] + (b - 1) * TILE_G + li - 1
        if l < tptr[g + 1]
            e = tile_image(g, toff, E); s = tlist[l]
            y, x, f, c = box_prepare(op, s, e)
            sY[li] = y; sX[li] = x; sF[li] = f
            ctx_store!(sC, li, op.d.psf, c)
            sA[li] = ay[s, e]; sB[li] = ax[s, e]; sR[li] = mR[s, e]
        end
        @synchronize
        li = @index(Local, Linear); g = tid[@index(Group, Linear)]
        e, ty0, tx0 = tile_origin(g, toff, nty, E)
        py, px = tile_pixel(li)
        gy = ty0 + py + Int32(1); gx = tx0 + px + Int32(1)
        for j in 1:min(TILE_G, tptr[g + 1] - tptr[g] - (b - 1) * TILE_G)
            R = sR[j]
            if (abs(gy - sA[j]) <= R) & (abs(gx - sB[j]) <= R)
                acc[1] += sF[j] * psf_value(op.d.psf, ctx_load(op.d.psf, sC, j), gy, gx, sY[j], sX[j])
            end
        end
        @synchronize
    end
    li = @index(Local, Linear); g = tid[@index(Group, Linear)]
    e, ty0, tx0 = tile_origin(g, toff, nty, E)
    py, px = tile_pixel(li)
    gy = ty0 + py + Int32(1); gx = tx0 + px + Int32(1)
    d = op.d; ny = d.dims[1, e]
    ((gy <= ny) & (gx <= d.dims[2, e])) && (out[d.dims[3, e] + gy + (gx - 1) * ny] += acc[1])
end

"Add the model of footprint `fp`'s sources into `out` by tile gather (any backend; the GPU render)."
function render_tiles!(out, op::RenderOp, sc::JointScene, fp::Footprint)
    be = sc.backend; N, E = size(fp.ay)
    N == 0 && return out
    t = tile_bins(sc, fp)
    t.nb > 0 && render_tiles_kernel!(be, TILE_G)(out, op, t.tid, t.tptr, t.tlist, fp.ay, fp.ax, fp.mR, t.nty, t.toff, E, t.nb; ndrange = TILE_G * length(t.tid))
    KA.synchronize(be)
    foreach(KA.unsafe_free!, (t.tptr, t.tlist, t.tid, t.nty, t.toff, t.tfill))
    return out
end

"Apply `op` to every (source, image) box of footprint `fp`, accumulating into the flat `out`."
function box_apply!(op, out, sc::JointScene, fp::Footprint)
    be = sc.backend; N, E = size(fp.ay)
    N == 0 && return out
    if is_cpu(be)
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
