## Multi-pass photometry, functional formulation
#
# Iterated background -> detect -> fit -> prune, in the spirit of crowdsource's
# `fit_im`, written as a pipeline of functions that take values and return
# values instead of a set of long-lived mutable structs that verbs are applied
# to in a required order.
#
# The organizing principle here is not lifetime, it is *dataflow*.  Every stage
# is a function whose output is the next stage's input:
#
#     bkg     = estimate_background_multipass(image, model, pass, o)
#     det     = detect_sources(catalog, disc, bkg, model, pass, o)
#     geom    = stamp_geometry(catalog, w, R_fit, ny, nx)
#     fit     = fit_pass(data, w, geom, catalog, psf, plan, o, lambda)
#     catalog = catalog_from_theta(catalog, fit, plan)
#     catalog = drop!(disc, catalog, keep, reasons, pass)
#
# so the pass body is a straight line and the only mutable things that outlive
# one stage are the two that genuinely accumulate: `disc` (an append-only
# discard log) and `history` (the per-pass reports).
#
# What that buys, concretely -- these are the bug classes the struct-based
# version has to hold off with invariants and comments:
#
#   * No buffer is shared between passes.  Every array `fit_pass` works in is
#     allocated inside `fit_pass`, so there is no such thing as a stale entry
#     left by an earlier pass over a different source set, and no "this buffer
#     must be zeroed because a later consumer reads past what the earlier one
#     wrote".
#   * No compaction in lockstep.  `Catalog` is subset by `catalog[keep]`, which
#     copies.  Nothing else is source-indexed and long-lived, so there is no
#     second array that must take the same mask, and no `compact!` whose
#     exemptions ("`stamp` and `union_pix` are deliberately exempt...") have to
#     be argued from properties of the loop.
#   * No "which buffer is current" flag.  An accepted damping trial swaps the
#     `model` and `cand` bindings, so the name `model` always denotes the render
#     at the current `theta`.  That replaces a `model_valid::Bool` field and a
#     `current_model(state)` accessor.
#   * No shared free-parameter plan duplicated into the fit state.  `FitPlan`
#     is built once from `(psf, fixed)` and passed by value.
#
# The cost is allocation: the per-source Jacobian stamps and the image-sized
# work vectors are reallocated every pass rather than resized in place.  That
# is deliberate.  The run is dominated by PSF rendering and gradient
# evaluation inside `_fill_stamps!` / `_render_model!` / the Krylov matvecs,
# all of which are unchanged here -- they are the same kernels the struct-based
# version calls.

# ==============================================================================
# Stamp derivative operator
# ==============================================================================

"""
    StampDerivatives{T, I <: Integer}

Per-star Jacobian values, stored as stamps; `J` is never materialized.

# Fields

- `values`: `(p, S², n_active)` array of weighted, column-equilibrated
  derivatives (`raw ./ colnorm`), where `p` is the number of free parameters
  per star and `S²` the number of stamp pixels.
- `pixels`: `(S², n_active)` flat pixel indices into the image, with `0`
  marking masked or off-image pixels.
- `colnorm`: `(p, n_active)` true per-column norm, kept from fill time.
- `npix`: number of pixels in the (flattened) image.
- `p`: number of free parameters per star.
- `S2`: number of stamp pixels (`S²`).
"""
struct StampDerivatives{T, I <: Integer}
    values::Array{T, 3}
    pixels::Matrix{I}
    colnorm::Matrix{T}
    npix::Int
    p::Int
    S2::Int
end

"""
    apply_JT!(z, Jm, u, live, sbuf)
    apply_JT!(z, Jm, u, live)

Compute `z = Jm' * u` where `u` is the *weighted* residual
`sqrt.(w) .* (model .- data)` and `z` is the equilibrated gradient
`D⁻¹ J' r` (i.e. `b_scaled`).  `z` is filled in place.  `sbuf` is a
length-`S²` scratch vector.  Non-`live` (frozen) stars are skipped, leaving
their `z` slice at `0`.  The convenience form allocates `sbuf`.
"""
function apply_JT!(z::AbstractVector, Jm::StampDerivatives, u::AbstractVector, live, sbuf::AbstractVector)
    p = Jm.p
    S2 = Jm.S2
    n_active = size(Jm.values, 3)
    V = Jm.values
    fill!(z, zero(eltype(z)))
    @inbounds for a in 0:(n_active - 1)
        live[a + 1] || continue
        base = a * p
        # Gather this star's residual pixels once into `sbuf`, then a dense,
        # non-aliased `p × S²` reduction that vectorizes.  A masked entry
        # (`fi == 0`) gathers the clamped `u[1]`, but pairs with
        # `Jm.values == 0`, so it contributes exactly `0`.
        for m in 1:S2
            fi = Jm.pixels[m, a + 1]
            sbuf[m] = u[ifelse(fi == zero(fi), one(fi), fi)]
        end
        LV.@turbo for k in 1:p
            acc = zero(eltype(z))
            for m in 1:S2
                acc += V[k, m, a + 1] * sbuf[m]
            end
            z[base + k] = acc
        end
    end
    return z
end

function apply_JT!(z::AbstractVector, Jm::StampDerivatives, u::AbstractVector, live)
    return apply_JT!(z, Jm, u, live, Vector{eltype(Jm.values)}(undef, Jm.S2))
end

"""
    apply_J!(y, Jm, v, live, sbuf)
    apply_J!(y, Jm, v)

Compute `y = Jm * v` (`y` filled in place).  `sbuf` is a length-`S²` scratch
vector.  Non-`live` (frozen) stars are skipped.  The convenience form
allocates `live`/`sbuf`.
"""
function apply_J!(y::AbstractVector, Jm::StampDerivatives, v::AbstractVector, live, sbuf::AbstractVector)
    p = Jm.p
    S2 = Jm.S2
    n_active = size(Jm.values, 3)
    V = Jm.values
    fill!(y, zero(eltype(y)))
    @inbounds for a in 0:(n_active - 1)
        live[a + 1] || continue
        base = a * p
        # Dense, non-aliased per-star product into `sbuf` (no pixel access, so
        # it vectorizes); a masked stamp entry has `Jm.values == 0`, so `sbuf`
        # is `0` there and the plain masked scatter below can skip it.  The
        # `p == 3` branch (the common y/x/flux-all-free case) hoists the three
        # RHS components to scalars, which the generic `v[base + k]` load inside
        # the reduction is not.
        if p == 3
            v1 = v[base + 1]
            v2 = v[base + 2]
            v3 = v[base + 3]
            LV.@turbo for m in 1:S2
                sbuf[m] = v1 * V[1, m, a + 1] + v2 * V[2, m, a + 1] + v3 * V[3, m, a + 1]
            end
        else
            LV.@turbo for m in 1:S2
                acc = zero(eltype(sbuf))
                for k in 1:p
                    acc += V[k, m, a + 1] * v[base + k]
                end
                sbuf[m] = acc
            end
        end
        for m in 1:S2
            fi = Jm.pixels[m, a + 1]
            fi != 0 || continue
            y[fi] += sbuf[m]
        end
    end
    return y
end

function apply_J!(y::AbstractVector, Jm::StampDerivatives, v::AbstractVector)
    return apply_J!(y, Jm, v, trues(size(Jm.values, 3)), Vector{eltype(Jm.values)}(undef, Jm.S2))
end

"""
    _jacobian_operator(stamp, live, sbuf, npix, n) -> LinearOperator

Wrap the weighted, column-equilibrated Jacobian as a matrix-free
`npix × n` `LinearOperator`: `op * v` calls [`apply_J!`](@ref), `op' * u`
calls [`apply_JT!`](@ref).  The closures capture `stamp` (whose `values` are
refreshed in place each outer iteration), `live` (mutated as stars freeze),
and `sbuf` (a length-`S²` scratch shared by the forward and adjoint products,
which the linear solver never runs concurrently), so a single operator built
once is valid for the whole fit.
"""
function _jacobian_operator(stamp::StampDerivatives{FT}, live, sbuf, npix::Int, n::Int) where {FT}
    fwd = (res, v) -> apply_J!(res, stamp, v, live, sbuf)
    adj = (res, u) -> apply_JT!(res, stamp, u, live, sbuf)
    return LinearOperators.LinearOperator(FT, npix, n, false, false, fwd, adj, adj)
end

# ==============================================================================
# Fixed stamp footprint
# ==============================================================================

# Morton (Z-order) key of a 2D integer coordinate: interleave the bits of `y`
# and `x` so that sorting by the key visits points along a locality-preserving
# space-filling curve.  Coordinates are image indices (>= 1, well under 2^32).
function _morton2d(y::Integer, x::Integer)
    spread(v::UInt64) = begin
        v &= 0x00000000ffffffff
        v = (v | (v << 16)) & 0x0000ffff0000ffff
        v = (v | (v << 8))  & 0x00ff00ff00ff00ff
        v = (v | (v << 4))  & 0x0f0f0f0f0f0f0f0f
        v = (v | (v << 2))  & 0x3333333333333333
        v = (v | (v << 1))  & 0x5555555555555555
        v
    end
    return spread(UInt64(y)) | (spread(UInt64(x)) << 1)
end

"""
    _model_radii(psf, model_rad, nsigma, R_fit, R_cap, w, flux_init) -> Vector{Int}

Per-source model-stamp half-width for [`fit_all_stars_simultaneous_multipass`](@ref).  The
model stamp is the box each star's PSF is rendered/subtracted over (distinct from
the `fit_rad` Jacobian/cost box); making it per-source keeps the faint bulk cheap
while still subtracting bright stars' wings far enough out that neighbors' cores
are clean (DAOPHOT's FITRAD vs PSFRAD split).

- A scalar `model_rad` returns one value (rounded up, clamped to `[R_fit, R_cap]`)
  for every source.
- `model_rad === :auto`: the half-width is the smallest integer `r` at which the
  source's annular-mean wing surface brightness `flux_init * <SB_PSF(r)>` drops
  below `nsigma * sigma_bg`, where `<SB_PSF>` comes from the PSF's own curve of
  growth and `sigma_bg` is the background noise estimated from the inverse weight
  vector `w`.  Clamped to `[R_fit, R_cap]`.
"""
function _model_radii(psf, model_rad, nsigma, R_fit::Int, R_cap::Int,
                      w::AbstractVector, flux_init::AbstractVector{FT}) where {FT}
    n = length(flux_init)
    n == 0 && return Int[]  # every source was dropped; `_w` below would be empty
    model_rad isa Real &&
        return fill(clamp(round(Int, round(model_rad, RoundUp)), R_fit, R_cap), n)
    # Quick, conservative background noise estimate from the inverse weight vector.
    # High w (inverse variance) = low noise so sigma_bg estimate will be low,
    # leading to a conservatively large model_rad estimate.
    step = max(1, length(w) ÷ 65536) # downsample to ~65k for speed, still robust
    _w = w[1:step:end]
    _w = _w[(_w .> 0) .& isfinite.(_w)]
    # A fixed stride can miss every valid pixel on a periodically masked frame,
    # and `quantile` of an empty vector throws.  Rescanning the whole map is the
    # expensive path this stride exists to avoid, so take it only when the cheap
    # sample came up empty.
    isempty(_w) && (_w = w[(w .> 0) .& isfinite.(w)])
    w_val = isempty(_w) ? zero(FT) : FT(quantile(_w, 0.84))
    sigma_bg = w_val > 0 ? FT(sqrt(1 / w_val)) : one(FT)
    unit = ConstructionBase.setproperties(psf,
        (; y = zero(FT), x = zero(FT), flux = one(FT), bkg = zero(FT)))
    cog = curve_of_growth(unit, FT.(1:R_cap))
    ee = cog.flux ./ cog.flux[end]
    sb = Vector{FT}(undef, R_cap)
    prev = zero(FT)
    for r in 1:R_cap
        sb[r] = max((ee[r] - prev) / (FT(π) * (r^2 - (r - 1)^2)), zero(FT))
        prev = ee[r]
    end
    thr = FT(nsigma) * sigma_bg
    out = Vector{Int}(undef, n)
    for j in 1:n
        F = max(flux_init[j], zero(FT))
        r = R_fit
        while r < R_cap && F * sb[r] > thr
            r += 1
        end
        out[j] = r
    end
    return out
end
# Sorted, unique flat indices of every image pixel covered by at least one
# stamp -- the support over which the cost and residual are evaluated.  A
# length-`npix` mask (not a growing vector + sort) keeps this linear in the
# stamp count and cheap in memory even at whole-frame sizes.
function _touched_pixels(pixels, npix::Int)
    mask = falses(npix)
    @inbounds for fi in pixels
        fi != 0 && (mask[fi] = true)
    end
    return findall(mask)
end


# ==============================================================================
# Fill (render value + gradient), model render, cost
# ==============================================================================

function _fill_stamps!(
        stamp::StampDerivatives{FT}, model_template, free_names_val, fixed,
        θ, w, grad_col, dy_off, dx_off, anchor_y, anchor_x,
        row_y, row_x, row_flux, live, _fill_scratch_buf
    ) where {FT}
    p = stamp.p
    S2 = stamp.S2
    n_active = size(stamp.values, 3)
    # This fills only the `fit_rad` Jacobian columns; `model_img` (which may
    # extend past the fit box, per-source `model_rad`) is built separately by
    # `_render_model!`.
    # No blanket `fill!(stamp.colnorm, ...)`: a frozen star's column must stay
    # bitwise untouched between freeze and the end of the fit (reset only the
    # live columns being recomputed below), or the eps(FT) floor two lines
    # down turns a stale zero into a ~4.5e15 rescale applied again every
    # iteration, overflowing stamp.values to Inf within a couple of outer
    # iterations and leaking NaN into a still-live star's operator column.
    @inbounds for a in 0:(n_active - 1)
        live[a + 1] || continue
        base = a * p
        for k in 1:p
            stamp.colnorm[k, a + 1] = zero(FT)
        end
        m = PSF.model_from_vector(model_template, free_names_val, view(θ, base + 1:base + p), fixed)
        ay = anchor_y[a + 1]
        ax = anchor_x[a + 1]
        for mi in 1:S2
            fi = stamp.pixels[mi, a + 1]
            if fi == 0
                # `apply_J!`/`apply_JT!` still touch masked entries (the adjoint
                # gathers the clamped `u[1]`) and rely on the derivative being
                # exactly zero to contribute nothing.  The store's buffer is
                # reused across passes, so a skipped slot would otherwise keep a
                # derivative from a different source/anchor/mask.
                for k in 1:p
                    stamp.values[k, mi, a + 1] = zero(FT)
                end
                continue
            end
            gy = ay + dy_off[mi]
            gx = ax + dx_off[mi]
            _, g = evaluate_fg(m, gy, gx)
            g1, g2, g3 = g[row_y], g[row_x], g[row_flux]
            sw = sqrt(w[fi])
            for k in 1:p
                gc = grad_col[k]
                gv = gc == 1 ? g1 : (gc == 2 ? g2 : g3)
                raw = sw * gv
                stamp.values[k, mi, a + 1] = raw
                stamp.colnorm[k, a + 1] += raw * raw
            end
        end
    end
    # Column equilibration: values ./= colnorm, floor colnorm at eps(T).
    @inbounds for a in 1:n_active
        live[a] || continue
        for k in 1:p
            cn = sqrt(stamp.colnorm[k, a])
            cn = ifelse(cn < eps(FT), eps(FT), cn)
            stamp.colnorm[k, a] = cn
            invcn = inv(cn)
            for mi in 1:S2
                stamp.values[k, mi, a] *= invcn
            end
        end
    end
    return stamp
end

"""
    _fill_scratch(psf, S, ::Type{FT}) where {FT}

Per-solve scratch for [`_fill_stamps!`](@ref)'s specialized
`GriddedPSFModel{T,<:ImagePSF{T}}` method: the same per-corner
`(value, d/dv, d/du)` `S x S` buffers [`PSF._render_scratch`](@ref) builds, plus three
length-`S^2` reduction outputs (`dS/dY`, `dS/dX`, `S`, in
`evaluate_fg(::GriddedPSFModel,...)`'s notation). `nothing` for every other
model type, which the generic `_fill_stamps!` method ignores. Kept
independent of `PSF._render_scratch`'s buffers (rather than shared) even though
`_fill_stamps!` and `_render_model!` never run concurrently within one
solve -- the tiny extra allocation is not worth the coupling.
"""
_fill_scratch(::AbstractPSFModel, S::Int, ::Type{FT}) where {FT} = nothing
function _fill_scratch(::GriddedPSFModel{T2, M}, S::Int, ::Type{FT}) where {T2, M <: ImagePSF{T2}, FT}
    corner = ntuple(_ -> ntuple(_ -> Matrix{FT}(undef, S, S), 4), 3)
    reductions = ntuple(_ -> Vector{FT}(undef, S * S), 3)
    return corner, reductions
end

"""
    _fill_stamps!(stamp, model_template::GriddedPSFModel{T,<:ImagePSF{T}}, ...)

Specialized `LV.@turbo` value+gradient fill for a
`GriddedPSFModel{T,<:ImagePSF{T}}` PSF, mirroring the specialized
[`_render_model!`](@ref): corner selection/weights and each active node's
`(oversampling, origin, fill_value)` depend only on the star's `(Y, X)`, so
they are computed once per star rather than once per SIMD batch. The actual
bicubic gather reuses [`PSF._gridded_corner_bicubic_pass!`](@ref) (one
branchless `@turbo` pass per corner, already used by the sequential
`fit_star` path), and a `@turbo` reduction combines the four corners' value
and both partial derivatives into `S`, `dS/dY`, `dS/dX` -- the same chain
rule `evaluate_fg(::GriddedPSFModel,...)` and `_accum_gridded_imagepsf!` use
-- before the per-pixel masked accumulation into `stamp.values`/`colnorm`
proceeds exactly as the generic method's.
"""
function _fill_stamps!(
        stamp::StampDerivatives{FT}, model_template::GriddedPSFModel{T2, M}, free_names_val, fixed,
        θ, w, grad_col, dy_off, dx_off, anchor_y, anchor_x,
        row_y, row_x, row_flux, live,
        fill_scratch_buf::Tuple{NTuple{3, NTuple{4, Matrix{FT}}}, NTuple{3, Vector{FT}}}
    ) where {FT, T2, M <: ImagePSF{T2}}
    p = stamp.p
    S2 = stamp.S2
    S = round(Int, sqrt(S2))
    R = maximum(dy_off)
    n_active = size(stamp.values, 3)
    # Jacobian columns only; `model_img` is built by `_render_model!`.
    (p_val, p_dpdv, p_dpdu), (dsdY_buf, dsdX_buf, s_buf) = fill_scratch_buf
    pv1, pv2, pv3, pv4 = p_val
    pdv1, pdv2, pdv3, pdv4 = p_dpdv
    pdu1, pdu2, pdu3, pdu4 = p_dpdu
    dsdY_mat = reshape(dsdY_buf, S, S)
    dsdX_mat = reshape(dsdX_buf, S, S)
    s_mat = reshape(s_buf, S, S)
    # No blanket `fill!(stamp.colnorm, ...)` -- see the generic method's comment.
    @inbounds for a in 0:(n_active - 1)
        live[a + 1] || continue
        base = a * p
        for k in 1:p
            stamp.colnorm[k, a + 1] = zero(FT)
        end
        m = PSF.model_from_vector(model_template, free_names_val, view(θ, base + 1:base + p), fixed)
        Y, X, flux, bkg = FT(m.y), FT(m.x), FT(m.flux), FT(m.bkg)
        ay = anchor_y[a + 1]
        ax = anchor_x[a + 1]
        yr = (ay - R):(ay + R)
        xr = (ax - R):(ax + R)

        corners = PSF._grid_corners_dw(model_template, Y, X)
        idx1, w1, dwdy1, dwdx1 = corners[1]
        idx2, w2, dwdy2, dwdx2 = corners[2]
        idx3, w3, dwdy3, dwdx3 = corners[3]
        idx4, w4, dwdy4, dwdx4 = corners[4]
        idx2 = idx2 == 0 ? idx1 : idx2
        idx3 = idx3 == 0 ? idx1 : idx3
        idx4 = idx4 == 0 ? idx1 : idx4
        w1, w2, w3, w4 = FT(w1), FT(w2), FT(w3), FT(w4)
        dwdy1, dwdy2, dwdy3, dwdy4 = FT(dwdy1), FT(dwdy2), FT(dwdy3), FT(dwdy4)
        dwdx1, dwdx2, dwdx3, dwdx4 = FT(dwdx1), FT(dwdx2), FT(dwdx3), FT(dwdx4)
        node1, node2, node3, node4 = model_template.psfs[idx1], model_template.psfs[idx2],
            model_template.psfs[idx3], model_template.psfs[idx4]
        sx1, sy1 = FT(node1.oversampling[1]), FT(node1.oversampling[2])
        sx2, sy2 = FT(node2.oversampling[1]), FT(node2.oversampling[2])
        sx3, sy3 = FT(node3.oversampling[1]), FT(node3.oversampling[2])
        sx4, sy4 = FT(node4.oversampling[1]), FT(node4.oversampling[2])

        y1, x1 = ay - R, ax - R
        PSF._gridded_corner_bicubic_pass!(pv1, pdv1, pdu1, node1.data, FT(node1.origin.x), FT(node1.origin.y),
            sx1, sy1, FT(node1.fill_value), size(node1.data, 1), size(node1.data, 2), yr, xr, Y, X, y1, x1)
        PSF._gridded_corner_bicubic_pass!(pv2, pdv2, pdu2, node2.data, FT(node2.origin.x), FT(node2.origin.y),
            sx2, sy2, FT(node2.fill_value), size(node2.data, 1), size(node2.data, 2), yr, xr, Y, X, y1, x1)
        PSF._gridded_corner_bicubic_pass!(pv3, pdv3, pdu3, node3.data, FT(node3.origin.x), FT(node3.origin.y),
            sx3, sy3, FT(node3.fill_value), size(node3.data, 1), size(node3.data, 2), yr, xr, Y, X, y1, x1)
        PSF._gridded_corner_bicubic_pass!(pv4, pdv4, pdu4, node4.data, FT(node4.origin.x), FT(node4.origin.y),
            sx4, sy4, FT(node4.fill_value), size(node4.data, 1), size(node4.data, 2), yr, xr, Y, X, y1, x1)

        # Chain rule reduction combining the four corners -- matches
        # `evaluate_fg(::GriddedPSFModel,...)` / `_accum_gridded_imagepsf!`.
        LV.@turbo for jj in 1:S, ii in 1:S
            s_mat[ii, jj] = w1 * pv1[ii, jj] + w2 * pv2[ii, jj] + w3 * pv3[ii, jj] + w4 * pv4[ii, jj]
            dsdY_mat[ii, jj] = dwdy1 * pv1[ii, jj] - w1 * sy1 * pdv1[ii, jj] +
                dwdy2 * pv2[ii, jj] - w2 * sy2 * pdv2[ii, jj] +
                dwdy3 * pv3[ii, jj] - w3 * sy3 * pdv3[ii, jj] +
                dwdy4 * pv4[ii, jj] - w4 * sy4 * pdv4[ii, jj]
            dsdX_mat[ii, jj] = dwdx1 * pv1[ii, jj] - w1 * sx1 * pdu1[ii, jj] +
                dwdx2 * pv2[ii, jj] - w2 * sx2 * pdu2[ii, jj] +
                dwdx3 * pv3[ii, jj] - w3 * sx3 * pdu3[ii, jj] +
                dwdx4 * pv4[ii, jj] - w4 * sx4 * pdu4[ii, jj]
        end

        for mi in 1:S2
            fi = stamp.pixels[mi, a + 1]
            if fi == 0
                for k in 1:p          # see the generic method's comment
                    stamp.values[k, mi, a + 1] = zero(FT)
                end
                continue
            end
            s_val = s_buf[mi]
            gy = flux * dsdY_buf[mi]
            gx = flux * dsdX_buf[mi]
            gflux = s_val
            sw = sqrt(w[fi])
            for k in 1:p
                gc = grad_col[k]
                gv = gc == 1 ? gy : (gc == 2 ? gx : gflux)
                raw = sw * gv
                stamp.values[k, mi, a + 1] = raw
                stamp.colnorm[k, a + 1] += raw * raw
            end
        end
    end
    # Column equilibration: values ./= colnorm, floor colnorm at eps(T).
    @inbounds for a in 1:n_active
        live[a] || continue
        for k in 1:p
            cn = sqrt(stamp.colnorm[k, a])
            cn = ifelse(cn < eps(FT), eps(FT), cn)
            stamp.colnorm[k, a] = cn
            invcn = inv(cn)
            for mi in 1:S2
                stamp.values[k, mi, a] *= invcn
            end
        end
    end
    return stamp
end

"""
    _source_errors!(errs, stamp, cov_est, cost, dof) -> errs

Per-source 1-sigma parameter errors from the `p x p` diagonal block of the
normal equations, for a catalog whose Jacobian is stored as `stamp`.

`stamp` must already hold the derivatives at the final `θ`, filled with every
source live (see [`_fill_stamps!`](@ref)): a source frozen during the fit has
zeroed columns, and its reported errors must not inherit that.

Each block is `D J' J D` restricted to source `j`'s stamp pixels, where `D` is
the column equilibration `stamp.colnorm` applied at fill time and unwound here.
The diagonal is then ridged by `1e-12 * tr` before inversion: a source whose
stamp retains too few unmasked pixels gives a singular block, and the ridge
keeps [`covariance!`](@ref) on its Cholesky path instead of the `pinv`
fallback.

`errs` is filled as `(p, n)` in the stamp's own column order, so `errs[k, j]`
is the error on free parameter `k` (the `k`-th entry of `free_names`) of
source `j`.  Callers scatter it into whatever layout they report.

Used by [`fit_all_stars_simultaneous_multipass`](@ref); `fit_all_stars` takes its
errors from the per-star Levenberg-Marquardt covariance instead.
"""
function _source_errors!(errs::AbstractMatrix{FT}, stamp::StampDerivatives{FT},
                         cov_est, cost, dof) where {FT}
    p, S2 = stamp.p, stamp.S2
    n = size(stamp.values, 3)
    size(errs) == (p, n) ||
        throw(DimensionMismatch("`errs` must be ($p, $n); got $(size(errs))"))
    # `covariance!` factors in place, so `blk` is rebuilt per source anyway.
    blk = zeros(FT, p, p)
    for j in 1:n
        fill!(blk, zero(FT))
        @inbounds for m in 1:S2
            stamp.pixels[m, j] != 0 || continue
            for k in 1:p, l in 1:p
                blk[k, l] += stamp.values[k, m, j] * stamp.values[l, m, j]
            end
        end
        for k in 1:p, l in 1:p
            blk[k, l] *= stamp.colnorm[k, j] * stamp.colnorm[l, j]
        end
        tr = zero(FT)
        for k in 1:p
            tr += blk[k, k]
        end
        for k in 1:p
            blk[k, k] += FT(1.0e-12) * tr
        end
        cov = covariance!(cov_est, blk, cost, dof)
        for k in 1:p
            errs[k, j] = sqrt(max(zero(FT), cov[k, k]))
        end
    end
    return errs
end

"""
    _render_model!(model_img, model_template, free_names_val, fixed, θ, p,
                   model_R, anchor_y, anchor_x, ny, nx, live, render_buf, render_scratch)

Render every live star at its current `θ` and scatter-add the result into the
flat image `model_img`.  Each star is rendered over its own `model_R[j]` box
into `render_buf` (sized to the largest such box) by [`PSF.render!`](@ref);
`render_scratch` comes from [`PSF._render_scratch`](@ref) and selects the
render path for `model_template`'s type.

The scatter is a separate plain loop because stamps of different stars alias
the same image pixel.  Off-image pixels are rendered (harmless, the buffer is
scratch) but not scattered.
"""
function _render_model!(
        model_img::AbstractVector{FT}, model_template, free_names_val, fixed,
        θ, p, model_R, anchor_y, anchor_x, ny::Int, nx::Int,
        live, render_buf::AbstractMatrix{FT}, render_scratch
    ) where {FT}
    fill!(model_img, zero(FT))
    return _accum_model!(model_img, model_template, free_names_val, fixed, θ, p, model_R,
        anchor_y, anchor_x, ny, nx, live, render_buf, render_scratch, one(FT))
end

"""
    _accum_model!(model_img, model_template, free_names_val, fixed, θ, p, model_R,
                  anchor_y, anchor_x, ny, nx, live, render_buf, render_scratch, coef)

The accumulating half of [`_render_model!`](@ref): add `coef` times each `live`
source's render into `model_img` *without* clearing it first.  `coef = -1`
removes a subset of sources from a model that is already rendered, which costs
`O(n_subset * model_R^2)` instead of the `O(n_active * model_R^2)` of a full
re-render.
"""
function _accum_model!(
        model_img::AbstractVector{FT}, model_template, free_names_val, fixed,
        θ, p, model_R, anchor_y, anchor_x, ny::Int, nx::Int,
        live, render_buf::AbstractMatrix{FT}, render_scratch, coef::FT
    ) where {FT}
    n_active = length(anchor_y)
    @inbounds for a in 0:(n_active - 1)
        live[a + 1] || continue
        base = a * p
        m = PSF.model_from_vector(model_template, free_names_val, view(θ, base + 1:base + p), fixed)
        ay = anchor_y[a + 1]
        ax = anchor_x[a + 1]
        R = model_R[a + 1]
        S = 2R + 1
        PSF.render!(render_buf, m, (ay - R):(ay + R), (ax - R):(ax + R), render_scratch)
        for jj in 1:S, ii in 1:S
            gy = ay - R + ii - 1
            gx = ax - R + jj - 1
            (1 <= gy <= ny) & (1 <= gx <= nx) || continue
            model_img[gy + (gx - 1) * ny] += coef * render_buf[ii, jj]
        end
    end
    return model_img
end

function _residual_cost!(wt_resid, model_img, data, w, union_pix)
    cost = zero(eltype(data))
    @inbounds for q in union_pix
        r = model_img[q] - data[q]
        wr = sqrt(w[q]) * r
        wt_resid[q] = wr
        cost += wr * wr
    end
    return cost
end

function _cost!(model_img, data, w, union_pix)
    cost = zero(eltype(data))
    @inbounds for q in union_pix
        r = model_img[q] - data[q]
        cost += w[q] * r * r
    end
    return cost
end


# ==============================================================================
# Position step cap
# ==============================================================================

function _cap_position_step!(δ, max_step, p, k_y, k_x)
    FT = eltype(δ)
    max_step = FT(max_step)
    (k_y === nothing && k_x === nothing) && return δ
    n_active = div(length(δ), p)
    @inbounds for a in 0:(n_active - 1)
        base = a * p
        dy = k_y === nothing ? zero(FT) : δ[base + k_y]
        dx = k_x === nothing ? zero(FT) : δ[base + k_x]
        len = sqrt(dy * dy + dx * dx)
        len > max_step || continue
        scale = max_step / len
        k_y === nothing || (δ[base + k_y] *= scale)
        k_x === nothing || (δ[base + k_x] *= scale)
    end
    return δ
end

# ==============================================================================
# Separation grid
# ==============================================================================

"""
    _SepGrid{T}(sep, ys, xs)
    _sep_grid(T, sep, ys, xs)

Uniform-bucket spatial index answering "is there a point within `sep` of this
position?" -- every such query in the multi-pass loop: detection dedup against
live and discarded positions, mutual dedup within a batch of new peaks, and the
close-pair prune.

The constructor returns an *empty* grid sized for the coordinates it is given;
points are added with [`_insert!`](@ref), which is what the greedy "accept this
one, then reject anything too close to it" passes require.  `_sep_grid` builds
one and inserts them all.  Sizing from the coordinates rather than from the
image keeps the grid independent of any frame, so every call site can build one
from the vectors it already has.

A KD-tree would be the general tool, but every query here is a fixed-radius
neighbor test in two dimensions at roughly uniform density -- the uniform grid's
best case -- and a KD-tree cannot be built incrementally, which the greedy passes
need.

# Cell size

`cell` is the larger of two floors:

- `sep`, below which the 3x3 scan would stop covering the query radius; and
- the mean nearest-neighbor spacing `sqrt(area / 4n)`, below which the grid
  would be almost entirely empty cells.

The second floor is what bounds the memory.  Sized at `sep` alone, a
`sep = 0.05` grid over a 4000^2 frame would be 6.4e9 cells (25 GB); the spacing
floor holds `head` at `4 * _SEPGRID_CELLS_PER_POINT` bytes per point for *any*
`sep`.  It is also faster when the catalog is sparser than `sep`, since the
candidate count per query is already near zero and only the grid overhead
changes.

# Fields

- `sep2`: `sep^2`, the exact test applied to every candidate.
- `inv_cell`: `1 / cell`.
- `ncy`, `ncx`: grid dimensions.
- `head`: `ncy x ncx` (flattened) index of the first point in each cell, `0` if
  empty.
- `next`: per point, the next point in the same cell, `0` at the end.  Together
  with `head` this is an intrusive singly-linked list per cell -- no hashing and
  no per-cell allocation, measured ~5x faster and ~4x smaller than the
  `Dict`-of-vectors it replaced, whose cell occupancy is near 1 at whole-frame
  source counts (one vector allocated per source, ~10 tuple hashes per query).
- `y`, `x`: inserted coordinates, in insertion order.

!!! note
    Cell indices are clamped into range, so points and queries outside the
    coordinate extent need no bounds check.  That is exact, not an
    approximation: `cell >= sep` means an in-range query can only be within
    `sep` of an out-of-range point while it sits in the first cell or two, and
    clamping moves such a point in by exactly one cell -- which the 3x3 window
    absorbs.  The `sep2` test still runs on every candidate, so a distant
    out-of-range point folded into an edge cell is rejected as it would be
    anywhere else.

Non-finite coordinates are never inserted and never match.
"""
struct _SepGrid{T}
    sep2::T
    inv_cell::T
    ncy::Int
    ncx::Int
    head::Vector{Int32}
    next::Vector{Int32}
    y::Vector{T}
    x::Vector{T}
end

# Cells per point once the catalog is sparser than `sep`.  Sets both the
# expected candidates per 3x3 query (9 / this) and `head`'s footprint
# (4 bytes * this, per point).
const _SEPGRID_CELLS_PER_POINT = 4

function _SepGrid{T}(sep::Real, ys, xs) where {T}
    n = 0
    ymax = one(T)
    xmax = one(T)
    for i in eachindex(ys, xs)
        (isfinite(ys[i]) && isfinite(xs[i])) || continue
        n += 1
        ymax = max(ymax, T(ys[i]))
        xmax = max(xmax, T(xs[i]))
    end
    cell = max(T(sep), sqrt(ymax * xmax / T(_SEPGRID_CELLS_PER_POINT * max(n, 1))))
    ncy = ceil(Int, ymax / cell) + 1
    ncx = ceil(Int, xmax / cell) + 1
    g = _SepGrid{T}(T(sep)^2, inv(cell), ncy, ncx, zeros(Int32, ncy * ncx), Int32[], T[], T[])
    sizehint!(g.next, n)
    sizehint!(g.y, n)
    sizehint!(g.x, n)
    return g
end

@inline function _cell(g::_SepGrid, y, x)
    cy = clamp(floor(Int, y * g.inv_cell) + 1, 1, g.ncy)
    cx = clamp(floor(Int, x * g.inv_cell) + 1, 1, g.ncx)
    return (cy, cx)
end

function _insert!(g::_SepGrid{T}, y, x) where {T}
    (isfinite(y) && isfinite(x)) || return g
    push!(g.y, T(y))
    push!(g.x, T(x))
    cy, cx = _cell(g, y, x)
    c = cy + (cx - 1) * g.ncy
    @inbounds push!(g.next, g.head[c])
    @inbounds g.head[c] = Int32(length(g.y))
    return g
end

function _has_neighbor(g::_SepGrid, y, x)
    (isfinite(y) && isfinite(x)) || return false
    cy, cx = _cell(g, y, x)
    @inbounds for dx in -1:1, dy in -1:1
        a = cy + dy
        b = cx + dx
        (1 <= a <= g.ncy && 1 <= b <= g.ncx) || continue
        i = g.head[a + (b - 1) * g.ncy]
        while i != 0
            ((g.y[i] - y)^2 + (g.x[i] - x)^2) <= g.sep2 && return true
            i = g.next[i]
        end
    end
    return false
end

function _sep_grid(::Type{T}, sep::Real, ys, xs) where {T}
    g = _SepGrid{T}(sep, ys, xs)
    for i in eachindex(ys, xs)
        _insert!(g, ys[i], xs[i])
    end
    return g
end


# ==============================================================================
# Catalog
# ==============================================================================

"""
    Catalog{T}

The evolving source list.  Five parallel vectors and nothing else: no discard
memory (that is [`Discards`](@ref)) and no methods that mutate it in place.

Subsetting is `catalog[keep]`, which copies.  That is what makes the fit safe to
write functionally: a prune produces a *new* catalog rather than compacting one
that other arrays are silently indexed against.

# Fields

- `y`, `x`, `flux`: current parameter estimates.
- `flux_snr`: signed curvature significance `flux * sqrt(H_ff)` from the last
  linearization, which is what [`prune_mask`](@ref) cuts on.  `NaN` until a fit
  has run.  Deliberately not `flux / flux_err`: it uses the diagonal flux
  curvature rather than the inverted per-source block, so it ignores the
  flux-position covariance and runs optimistic.
- `pass`: pass on which each source was detected (`0` for warm-start sources).
"""
struct Catalog{T}
    y::Vector{T}
    x::Vector{T}
    flux::Vector{T}
    flux_snr::Vector{T}
    pass::Vector{Int}
end

Catalog{T}() where {T} = Catalog{T}(T[], T[], T[], T[], Int[])

Base.length(catalog::Catalog) = length(catalog.y)
Base.isempty(catalog::Catalog) = isempty(catalog.y)

Base.getindex(catalog::Catalog{T}, I) where {T} = Catalog{T}(catalog.y[I], catalog.x[I],
    catalog.flux[I], catalog.flux_snr[I], catalog.pass[I])

"""
    Catalog{T}(sources, psf) -> Catalog{T}

Seed a catalog from any format [`_extract_source_catalog`](@ref) accepts, or
return an empty one for `sources === nothing`.  The detection pass is recorded
as `0`.  A row with a non-finite position is rejected with a warning, since the
caller would otherwise see a shorter catalog with no explanation.
"""
function Catalog{T}(sources, psf) where {T}
    sources === nothing && return Catalog{T}()
    params, _ = _extract_source_catalog(sources, psf, T)
    prop_names = collect(keys(ConstructionBase.getproperties(psf)))
    ry, rx, rf = findfirst(==(:y), prop_names), findfirst(==(:x), prop_names), findfirst(==(:flux), prop_names)
    (ry === nothing || rx === nothing || rf === nothing) &&
        throw(ArgumentError("PSF model must have `y`, `x` and `flux` fields"))
    a = append_sources(Catalog{T}(), view(params, ry, :), view(params, rx, :),
                       view(params, rf, :), 0)
    n_input = size(params, 2)
    a.n_added == n_input ||
        @warn "$(n_input - a.n_added) of $n_input supplied sources have non-finite " *
              "positions and were rejected"
    return a.catalog
end

"""
    Discards{T}

Append-only log of sources removed from the catalog: where they were, on which
pass, and why (`:snr`, `:close`, `:no_pixels`, `:nonfinite`).

Detection reads it so a discarded position is not re-detected forever, and the
final result reads it for `n_failed` / `failure_msgs`.  It is the one piece of
per-run state that genuinely accumulates, so it is the one thing here that is
mutated in place.

"Rejected" and "discarded" are not interchangeable: a *peak* is rejected when a
detection pass declines to add it; a *source* is discarded when it leaves the
catalog.  Landing on a discarded position is one of the reasons a peak is
rejected, so overloading one word made the trace unreadable.
"""
struct Discards{T}
    y::Vector{T}
    x::Vector{T}
    pass::Vector{Int}
    reason::Vector{Symbol}
end

Discards{T}() where {T} = Discards{T}(T[], T[], Int[], Symbol[])
Base.length(d::Discards) = length(d.y)

"""
    drop!(disc, catalog, keep, reason, pass) -> Catalog

Remove the sources where `keep` is `false` from `catalog`, logging each one on
`disc` with `pass` and its reason, and return the surviving catalog.

`reason` is either a single `Symbol` for the whole drop or a vector indexed like
`keep` (read only where `keep` is `false`).

This is the *only* way a source leaves the catalog, which is what keeps the
discard log complete: forgetting to record a removal would mean not calling the
function that removes it.
"""
function drop!(disc::Discards, catalog::Catalog, keep::AbstractVector{Bool}, reason, pass::Integer)
    length(keep) == length(catalog) ||
        throw(ArgumentError("keep mask length $(length(keep)) != catalog length $(length(catalog))"))
    all(keep) && return catalog
    @inbounds for j in eachindex(keep)
        keep[j] && continue
        push!(disc.y, catalog.y[j])
        push!(disc.x, catalog.x[j])
        push!(disc.pass, Int(pass))
        push!(disc.reason, reason isa Symbol ? reason : reason[j])
    end
    return catalog[keep]
end

# Morton key of the rounded anchor.  Clamped at 1 only for the key: a source
# that has drifted off the top/left edge would otherwise fail `UInt64(::Int)`.
# Such sources are dropped as `:no_pixels` by `stamp_geometry`, but the sort
# runs first.
_anchor_key2(y, x) = _morton2d(max(1, isfinite(y) ? round(Int, y) : 1),
                               max(1, isfinite(x) ? round(Int, x) : 1))

"""
    sort_morton(catalog) -> Catalog

Reorder into Morton (Z-order) order of the rounded `(y, x)` anchor.

This is a cache-locality heuristic for the scatter/gather in
[`apply_J!`](@ref) / [`apply_JT!`](@ref), which the Krylov solve issues 20-40
times per pass, so the sort costs about 1% of the work it accelerates.
"""
function sort_morton(catalog::Catalog)
    length(catalog) > 1 || return catalog
    keys_ = [_anchor_key2(catalog.y[j], catalog.x[j]) for j in eachindex(catalog.y)]
    return catalog[sortperm(keys_)]
end

"""
    append_sources(catalog, y, x, flux, pass; min_separation = 0)
        -> (; catalog, n_added)

Append sources, then re-sort into Morton order.

`min_separation` deduplicates *within the incoming batch* only (greedy, in the
order given).  Deduplication against the existing catalog and against discarded
positions is [`detect_sources`](@ref)'s job, because it reports the two
rejection reasons separately.  Non-finite positions are always rejected.
"""
function append_sources(catalog::Catalog{T}, y, x, flux, pass::Integer;
                        min_separation::Real = 0) where {T}
    grid = min_separation > 0 ? _SepGrid{T}(min_separation, y, x) : nothing
    ys, xs, fs = T[], T[], T[]
    for i in eachindex(y, x, flux)
        yi, xi = T(y[i]), T(x[i])
        (isfinite(yi) && isfinite(xi)) || continue
        if grid !== nothing
            _has_neighbor(grid, yi, xi) && continue
            _insert!(grid, yi, xi)
        end
        push!(ys, yi)
        push!(xs, xi)
        push!(fs, T(flux[i]))
    end
    n_added = length(ys)
    n_added == 0 && return (; catalog, n_added)
    nan = fill(T(NaN), n_added)
    out = Catalog{T}(vcat(catalog.y, ys), vcat(catalog.x, xs), vcat(catalog.flux, fs),
                     vcat(catalog.flux_snr, nan), vcat(catalog.pass, fill(Int(pass), n_added)))
    return (; catalog = sort_morton(out), n_added)
end


# ==============================================================================
# Free-parameter plan
# ==============================================================================

"""
    FitPlan(psf, fixed) -> FitPlan

The free-parameter plan, resolved once from the PSF's property order and the
`fixed` NamedTuple and then passed by value to everything that needs it.

Holding it in one place is what keeps the property-order convention -- gradients
follow the PSF struct's own field order, `y` before `x` -- from being restated
at each site that indexes a gradient or a `theta` slice.

# Fields

- `free_names`: the free parameters, in PSF property order.  Always a subset of
  `(:y, :x, :flux)` here; `flux` is required.
- `free_names_val`: `Val(free_names)`, for [`PSF.model_from_vector`](@ref).
- `p`: parameters per source, `length(free_names)`; the `theta` stride.
- `grad_col[k]`: which gradient component free parameter `k` reads, `1` = `dy`,
  `2` = `dx`, `3` = `dflux`.
- `k_y`, `k_x`, `k_flux`: index of each parameter within a source's `theta`
  slice (`nothing` when fixed).
- `row_y`, `row_x`, `row_flux`, `row_bkg`: index of each parameter within the
  PSF's *full* property tuple, which is what `evaluate_fg`'s gradient is
  indexed by.
- `fixed`: the fixed-parameter NamedTuple, with `bkg` pinned to zero.
"""
struct FitPlan{FN, F <: NamedTuple}
    free_names::Vector{Symbol}
    free_names_val::Val{FN}
    p::Int
    grad_col::Vector{Int}
    k_y::Union{Nothing, Int}
    k_x::Union{Nothing, Int}
    k_flux::Int
    row_y::Int
    row_x::Int
    row_flux::Int
    row_bkg::Union{Nothing, Int}
    fixed::F
end

function FitPlan(psf, fixed::NamedTuple)
    free_names, free_idx, _ = PSF.free_params(psf, fixed)
    isempty(free_idx) && throw(ArgumentError("all model parameters are fixed; nothing to fit"))
    offenders = setdiff(free_names, (:y, :x, :flux))
    isempty(offenders) || throw(ArgumentError(
        "fit_all_stars_simultaneous_multipass fits only (y, x, flux) per star; got free " *
        "parameters $(offenders). `bkg` is fixed automatically; fix all shape parameters."))

    prop_names = collect(keys(ConstructionBase.getproperties(psf)))
    row_y = findfirst(==(:y), prop_names)
    row_x = findfirst(==(:x), prop_names)
    row_flux = findfirst(==(:flux), prop_names)
    row_y === nothing && throw(ArgumentError("PSF model has no `y` field"))
    row_x === nothing && throw(ArgumentError("PSF model has no `x` field"))
    row_flux === nothing && throw(ArgumentError("PSF model has no `flux` field"))

    p = length(free_idx)
    grad_col = [nm === :y ? 1 : (nm === :x ? 2 : 3) for nm in free_names]
    k_flux = findfirst(==(3), grad_col)
    k_flux === nothing && throw(ArgumentError(
        "`flux` must be free: the multi-pass loop prunes on flux significance"))
    return FitPlan(collect(Symbol, free_names), Val(Tuple(free_names)), p, grad_col,
        findfirst(==(1), grad_col), findfirst(==(2), grad_col), k_flux,
        row_y, row_x, row_flux, findfirst(==(:bkg), prop_names), fixed)
end

"""
    theta_from_catalog(catalog, plan) -> Vector

Flatten the catalog into the fit's parameter vector, `p` entries per source in
`plan.free_names` order.
"""
function theta_from_catalog(catalog::Catalog{FT}, plan::FitPlan) where {FT}
    p = plan.p
    theta = Vector{FT}(undef, p * length(catalog))
    @inbounds for j in eachindex(catalog.y), k in 1:p
        nm = plan.free_names[k]
        theta[(j - 1) * p + k] = nm === :y ? catalog.y[j] : (nm === :x ? catalog.x[j] : catalog.flux[j])
    end
    return theta
end

"""
    catalog_from_theta(catalog, fit, plan) -> Catalog

The catalog with the fitted parameters and the fresh `flux_snr` written back.
Fit indices *are* catalog indices, so this is element-wise; it returns a new
catalog rather than mutating, so the pre-fit values remain valid where they are
still referenced.
"""
function catalog_from_theta(catalog::Catalog{FT}, fit, plan::FitPlan) where {FT}
    p = plan.p
    y, x, flux = copy(catalog.y), copy(catalog.x), copy(catalog.flux)
    snr = Vector{FT}(undef, length(catalog))
    theta, colnorm = fit.theta, fit.stamp.colnorm
    @inbounds for j in eachindex(y)
        base = (j - 1) * p
        for k in 1:p
            nm = plan.free_names[k]
            v = theta[base + k]
            nm === :y ? (y[j] = v) : nm === :x ? (x[j] = v) : (flux[j] = v)
        end
        # Signed curvature significance `flux * sqrt(H_ff)`: column equilibration
        # already computed `colnorm[k_flux, j] == sqrt(H_ff)`, so pruning needs
        # no extra linear algebra.  One accepted step stale, since `colnorm` was
        # filled at the theta of the last linearization.
        snr[j] = theta[base + plan.k_flux] * colnorm[plan.k_flux, j]
    end
    return Catalog{FT}(y, x, flux, snr, copy(catalog.pass))
end


# ==============================================================================
# Background
# ==============================================================================

"""
    estimate_background_multipass(image, model, pass, o) -> NamedTuple

Estimate the background level, RMS, weight maps and detection residual for one
pass, from `image - model`.

The level and the RMS want different mesh scales, so they come from two
`Background2D` calls; the level call's struct is returned with the RMS call's
map written into it, so its `box_size` and `mesh_background_rms` describe the
*level* mesh only.  The coarse -> fine level schedule (`pass <= coarse_passes`
uses the coarse mesh) lives here, so it is one branch in one function rather
than a condition threaded through the pass body.

# Returns

- `result`: the `Background2D`, for the caller's result.
- `background`, `rms`: full-resolution level and RMS maps.
- `detect_inv_var`: `1 / rms^2`, zeroed wherever `o.fixed_inv_var` is zero,
  negative or non-finite.  Background-only and smooth; never source-inclusive.
- `fit_inv_var`: the LM weights, `o.fixed_inv_var` when supplied and
  `detect_inv_var` otherwise (the same array, not a copy -- nothing writes to
  it).
- `resid`: `image - background - model`, the detection image.
- `coarse`, `box`: which level mesh this pass used.
"""
function estimate_background_multipass(image::AbstractMatrix, model::AbstractMatrix{FT},
        pass::Integer, o) where {FT}
    work = @. FT(image) - model
    coarse = pass <= o.bkg_coarse_passes
    box = coarse ? o.bkg_box_size_coarse : o.bkg_box_size
    common = merge((; mask = o.mask, coverage_mask = o.coverage_mask), o.bkg_kws)
    level = Background2D(work, box; estimator = o.bkg_estimator, common...)
    rmsb = Background2D(work, o.bkg_rms_box_size; rms_estimator = o.bkg_rms_estimator, common...)
    level.background_rms .= rmsb.background_rms
    background, rms = level.background, level.background_rms

    fx = o.fixed_inv_var
    detect_inv_var = Matrix{FT}(undef, size(background))
    @inbounds for i in eachindex(detect_inv_var)
        r = rms[i]
        iv = (isfinite(r) && r > 0) ? inv(r * r) : zero(FT)
        # The `inv_var` contract: a zero weight excludes the pixel from detection
        # as well as from the fit.  Without this a DQ-flagged or saturated pixel
        # could raise a detection that is then pruned, permanently blinding that
        # position through the discard log.
        if fx !== nothing
            wv = fx[i]
            (isfinite(wv) && wv > 0) || (iv = zero(FT))
        end
        detect_inv_var[i] = iv
    end
    # `work` has served its purpose and `Background2D` returns freshly upsampled
    # maps rather than views of its input, so it can become the residual.  The
    # subtraction is re-done from `image` rather than as `work - background`:
    # the two associate differently in floating point, and matching the order
    # keeps the detection stream identical to a full re-derivation.
    resid = work
    @. resid = FT(image) - background - model
    return (; result = level, background, rms, detect_inv_var,
              fit_inv_var = fx === nothing ? detect_inv_var : fx, resid, coarse, box)
end

"""
    pass_weights(image, bkg, FT) -> (data, w)

The background-subtracted data vector and the weight vector the LM fit sees,
both flat and length `npix`.  A non-finite datum is zeroed and given zero
weight, which is the same "ignore this pixel" convention `inv_var` already uses.
"""
function pass_weights(image::AbstractMatrix, bkg, ::Type{FT}) where {FT}
    npix = length(image)
    data = Vector{FT}(undef, npix)
    w = Vector{FT}(undef, npix)
    @inbounds for i in 1:npix
        d = FT(image[i]) - bkg.background[i]
        wv = bkg.fit_inv_var[i]
        wv = (isfinite(wv) && wv > 0) ? FT(wv) : zero(FT)
        if !isfinite(d)
            d = zero(FT)
            wv = zero(FT)
        end
        data[i] = d
        w[i] = wv
    end
    return data, w
end


# ==============================================================================
# Detection
# ==============================================================================


# Significance of the *current model* on the same footing as the data
# significance: one extra correlation, reusing the detection result's kernel and
# its `smoothed_inv_var` denominator rather than re-entering `matched_filter`.
function _model_significance(mfr::MatchedFilterResult{T}, inv_var, model) where {T}
    den = mfr.smoothed_inv_var
    den === nothing && return fill(zero(T), size(model))
    # Mirrors `matched_filter`'s own guard so a NaN at a zero-weight pixel cannot
    # leak through as `0 * NaN`.
    wm = map((w, v) -> iszero(w) ? zero(w * v) : w * v, inv_var, model)
    num = correlate(wm, mfr.kernel, :replicate)
    @inbounds for i in eachindex(num)
        num[i] = den[i] > 0 ? num[i] / sqrt(den[i]) : zero(eltype(num))
    end
    return num
end

"""
    detect_sources(catalog, disc, bkg, model, pass, o) -> NamedTuple

Correlate the residual, optionally gate on model significance, reject peaks that
land on a catalog source or a discarded position, seed the survivors with
[`measure_star_shapes`](@ref), and append them.

Only the survivors are seeded: existing sources already have better parameters
from the fit, and the full morphology pass runs once at the end.  The cutouts
come from the residual, where a new source's flux is still present.

Returns the new catalog plus every counter the pass report wants
(`n_peaks`, `n_blend_rejected`, `n_dup_catalog`, `n_dup_discarded`, `n_new`) and
the `MatchedFilterResult`.
"""
function detect_sources(catalog::Catalog{FT}, disc::Discards{FT}, bkg, model,
                        pass::Integer, o) where {FT}
    # Run detection on *current residual image*
    mfr = matched_filter(bkg.resid, o.kernel; inv_var = bkg.detect_inv_var,
                         normalize_zerosum = o.normalize_zerosum, sigma = o.detect_sigma)
    surv = collect(eachindex(mfr.peaks))
    n_peaks = length(surv)
    n_blend_rejected = 0
    n_dup_catalog = 0
    n_dup_discarded = 0

    # Gating is a per-pass property: whichever of the two thresholds is in force
    # decides it, and a `nothing` threshold means no gate on this pass.  Neither
    # knob is a master switch, so "strict early, none afterwards" is expressible.
    thr = pass <= o.blend_passes ? o.blend_threshold_initial : o.blend_threshold
    if thr !== nothing && !isempty(surv)
        msig = _model_significance(mfr, bkg.detect_inv_var, model)
        keep = [mfr.peak_significances[i] > thr * msig[mfr.peaks[i]] for i in surv]
        n_blend_rejected = count(!, keep)
        surv = surv[keep]
    end

    sep = o.min_separation
    if sep > 0 && !isempty(surv)
        py = [FT(Tuple(mfr.peaks[i])[1]) for i in surv]
        px = [FT(Tuple(mfr.peaks[i])[2]) for i in surv]
        gl = _sep_grid(FT, sep, catalog.y, catalog.x)
        keep = [!_has_neighbor(gl, py[a], px[a]) for a in eachindex(surv)]
        n_dup_catalog = count(!, keep)
        surv, py, px = surv[keep], py[keep], px[keep]
        if !isempty(surv)
            gd = _sep_grid(FT, sep, disc.y, disc.x)
            keep = [!_has_neighbor(gd, py[a], px[a]) for a in eachindex(surv)]
            n_dup_discarded = count(!, keep)
            surv = surv[keep]
        end
    end

    isempty(surv) && return (; catalog, mfr, n_peaks, n_blend_rejected, n_dup_catalog,
                               n_dup_discarded, n_new = 0)

    # mfr contains only new sources detected this iteration; get estimates of their
    # centroids and fluxes from measure_star_shapes to seed fitting, then append them to the catalog.
    hw = o.morph_half_width === nothing ? _default_half_width(mfr) : o.morph_half_width
    shapes = measure_star_shapes(mfr; peaks = surv, half_width = hw)
    a = append_sources(catalog, [s.centroid.y for s in shapes], [s.centroid.x for s in shapes],
                       [s.flux for s in shapes], pass; min_separation = sep)
    return (; catalog = a.catalog, mfr, n_peaks, n_blend_rejected, n_dup_catalog, n_dup_discarded,
              n_new = a.n_added)
end


# ==============================================================================
# Stamp geometry
# ==============================================================================

"""
    stamp_geometry(catalog, w, R, ny, nx) -> NamedTuple

The fixed `(2R + 1)^2` stamp footprint, each source's rounded anchor, and the
flat image index of every stamp pixel (`0` for off-image or zero-weight).

`ok[j]` is `false` for a source with non-finite parameters or no usable pixel at
all.  The caller drops those and calls this again rather than compacting the
columns in place: dropping is rare (edge and fully masked sources), the rebuild
is one pass over the catalog, and it removes any possibility of the pixel
columns disagreeing with the catalog they were built from.

# Returns

`anchor_y`, `anchor_x`, `dy_off`, `dx_off`, `pixels` (`S2 x n`), `S2`, `ok`.
"""
function stamp_geometry(catalog::Catalog, w::AbstractVector, R::Int, ny::Int, nx::Int)
    S = 2R + 1
    S2 = S * S
    dy_off = Vector{Int}(undef, S2)
    dx_off = Vector{Int}(undef, S2)
    m = 0
    for dx in -R:R, dy in -R:R
        m += 1
        dy_off[m] = dy
        dx_off[m] = dx
    end

    n = length(catalog)
    pixels = zeros(Int32, S2, n)
    anchor_y = Vector{Int}(undef, n)
    anchor_x = Vector{Int}(undef, n)
    ok = trues(n)
    @inbounds for j in 1:n
        yj, xj, fj = catalog.y[j], catalog.x[j], catalog.flux[j]
        if !(isfinite(yj) && isfinite(xj) && isfinite(fj))
            ok[j] = false
            anchor_y[j] = 1
            anchor_x[j] = 1
            continue
        end
        ay = round(Int, yj)
        ax = round(Int, xj)
        anchor_y[j] = ay
        anchor_x[j] = ax
        any_pix = false
        for mi in 1:S2
            gy = ay + dy_off[mi]
            gx = ax + dx_off[mi]
            if 1 <= gy <= ny && 1 <= gx <= nx
                fi = gy + (gx - 1) * ny
                if w[fi] > 0
                    pixels[mi, j] = fi
                    any_pix = true
                end
            end
        end
        ok[j] = any_pix
    end
    return (; anchor_y, anchor_x, dy_off, dx_off, pixels, S2, ok)
end


# ==============================================================================
# One pass's fit
# ==============================================================================

"""
    fit_pass(data, w, geom, catalog, psf, plan, o, lambda; kws...) -> NamedTuple

Run `linearizations` damped Gauss-Newton linearizations of the whole catalog
against `data`, starting from the catalog's own parameters and the damping
factor `lambda`.

Every buffer this needs is allocated here and dies here, so nothing it writes
can be read at a stale value by a later pass.  The Jacobian is never
materialized: `_fill_stamps!` fills per-source, column-equilibrated derivative
stamps and [`_jacobian_operator`](@ref) wraps them for Krylov.

Each linearization runs a damping-trial loop: solve, unscale, cap the position
step, render the candidate, compare costs.  An accepted trial *swaps* the
`model` and `cand` bindings, so `model` always names the render at the current
`theta` and the caller needs no flag to ask which buffer is live.  A
linearization with no accepted trial ends the pass.

# Keyword arguments

- `linearizations`: how many linearizations to attempt.
- `max_trials`: damping trials per linearization.  `0` means "linearize only" --
  fill the stamps and render the model at the incoming `theta` without moving
  it, which is what the post-validation rebuild wants.
- `freeze_positions`: zero the position columns of `J` *and* cap the position
  step at zero.  Zeroing alone would still spend the solve on positions and
  leave a flux component that is not the flux-only Gauss-Newton step; capping
  alone would let the solve waste its budget on directions that are then
  discarded.

# Returns

`theta`, `model` (`ny x nx`, at the final `theta`), `cost`, `cost_start`,
`gnorm`, `dof`, `lambda`, the counters `n_lin`/`n_trials`/`n_accepted`, and the
pieces the finalization re-reads: `stamp`, `model_R`, `render_buf`,
`render_scratch`, `fill_scratch`, `live`.
"""
function fit_pass(data::Vector{FT}, w::Vector{FT}, geom, catalog::Catalog{FT}, psf,
                  plan::FitPlan, o, lambda::FT; ny::Int, nx::Int,
                  linearizations::Int, max_trials::Int = o.max_trials,
                  freeze_positions::Bool = false, pass::Integer = 1,
                  show_trace::Bool = false) where {FT}
    n = length(catalog)
    p = plan.p
    n_par = p * n
    npix = length(data)
    S2 = geom.S2

    theta = theta_from_catalog(catalog, plan)
    model_R = _model_radii(psf, o.model_rad, o.model_rad_nsigma, o.R_fit, o.R_cap, w, catalog.flux)
    Rc_side = 2 * (isempty(model_R) ? o.R_fit : maximum(model_R)) + 1

    stamp = StampDerivatives{FT, Int32}(Array{FT, 3}(undef, p, S2, n), geom.pixels,
                                        Matrix{FT}(undef, p, n), npix, p, S2)
    union_pix = _touched_pixels(geom.pixels, npix)
    dof = max(length(union_pix) - n_par, 1)
    live = trues(n)
    sbuf = Vector{FT}(undef, S2)

    model = zeros(FT, npix)
    cand = zeros(FT, npix)
    # Zeroed, not `undef`: `_residual_cost!` writes only `union_pix`, but `mrhs`
    # is the length-`npix` Krylov right-hand side and copies all of it.
    wt_resid = zeros(FT, npix)
    mrhs = Vector{FT}(undef, npix)
    b_scaled = Vector{FT}(undef, n_par)
    delta = Vector{FT}(undef, n_par)
    theta_cand = Vector{FT}(undef, n_par)

    render_buf = Matrix{FT}(undef, Rc_side, Rc_side)
    render_scratch = PSF._render_scratch(psf, Rc_side, FT)
    fill_scratch = _fill_scratch(psf, 2 * o.R_fit + 1, FT)
    J_op = _jacobian_operator(stamp, live, sbuf, npix, n_par)
    ws = Krylov.krylov_workspace(Val(o.solver), npix, n_par, Vector{FT})

    max_step = freeze_positions ? zero(FT) : FT(o.max_step)
    cost = FT(NaN)
    cost_start = FT(NaN)
    gnorm = FT(NaN)
    n_lin = 0
    n_trials = 0
    n_accepted = 0

    for lin in 1:linearizations
        _fill_stamps!(stamp, psf, plan.free_names_val, plan.fixed, theta, w, plan.grad_col,
            geom.dy_off, geom.dx_off, geom.anchor_y, geom.anchor_x,
            plan.row_y, plan.row_x, plan.row_flux, live, fill_scratch)
        if freeze_positions
            V = stamp.values
            @inbounds for k in (plan.k_y, plan.k_x)
                k === nothing && continue
                for a in 1:n, mi in 1:S2
                    V[k, mi, a] = zero(FT)
                end
            end
        end
        _render_model!(model, psf, plan.free_names_val, plan.fixed, theta, p, model_R,
            geom.anchor_y, geom.anchor_x, ny, nx, live, render_buf, render_scratch)
        cost = _residual_cost!(wt_resid, model, data, w, union_pix)
        n_lin += 1
        lin == 1 && (cost_start = cost)
        @. mrhs = -wt_resid
        apply_JT!(b_scaled, stamp, wt_resid, live, sbuf)

        colnorm_flat = reshape(stamp.colnorm, n_par)
        gnorm = _scaled_gradient_norm(b_scaled, colnorm_flat, cost / dof)

        accepted = false
        for trial in 1:max_trials
            Krylov.krylov_solve!(ws, J_op, mrhs; λ = sqrt(lambda),
                itmax = o.linear_iterations, atol = o.linear_tol, btol = o.linear_tol)
            sol = Krylov.solution(ws)
            @. delta = sol / colnorm_flat
            _cap_position_step!(delta, max_step, p, plan.k_y, plan.k_x)
            @. theta_cand = theta + delta
            _render_model!(cand, psf, plan.free_names_val, plan.fixed, theta_cand, p, model_R,
                geom.anchor_y, geom.anchor_x, ny, nx, live, render_buf, render_scratch)
            cost_cand = _cost!(cand, data, w, union_pix)

            n_trials += 1
            ok = cost_cand < cost
            show_trace && _trace_fit_step(pass, lin, trial, cost, cost_cand, lambda, gnorm,
                ok ? "accepted" : "rejected")
            if ok
                # The candidate arrays become the current ones.  `model` is now
                # the render at `theta`, by construction rather than by flag.
                theta, theta_cand = theta_cand, theta
                model, cand = cand, model
                cost = cost_cand
                lambda = max(lambda / o.lambda_down, o.lambda_min)
                accepted = true
                n_accepted += 1
                break
            end
            lambda = min(lambda * o.lambda_up, o.lambda_max)
            trial == max_trials &&
                @warn "max_damping_trials reached without an accepted step" pass lin λ = lambda
        end
        accepted || break
    end

    return (; theta, model = reshape(model, ny, nx), cost, cost_start, gnorm, dof, lambda,
              n_lin, n_trials, n_accepted, stamp, model_R, live, data, w, geom,
              render_buf, render_scratch, fill_scratch, union_pix)
end

# Cosine-scaled gradient norm: `|b_true| / sqrt(A_ii * cost/dof)` per column,
# maximized.  Dimensionless and comparable across passes and source counts.
function _scaled_gradient_norm(b_scaled::AbstractVector{FT}, colnorm_flat, C_r) where {FT}
    g_tiny = eps(FT)
    gnorm = zero(FT)
    @inbounds for i in eachindex(b_scaled)
        cn = colnorm_flat[i]
        gnorm = max(gnorm, abs(cn * b_scaled[i]) / (sqrt(cn * cn * C_r) + g_tiny))
    end
    return gnorm
end

"""
    subtract_sources!(fit, psf, plan, drop_mask, ny, nx) -> model

Remove the sources marked `true` in `drop_mask` from `fit.model`, in place.

Called with the pruned mask while `fit` still lines up with the pre-prune
catalog: the dropped sources are rendered at exactly the `theta`, anchors and
`model_R` they were rendered with, so subtracting them reproduces the model of
the surviving catalog for `O(n_dropped * model_R^2)` instead of the
`O(n_catalog * model_R^2)` of a full re-render.

!!! note
    Exact in exact arithmetic but not bitwise identical to re-rendering the
    survivors: `sum(all) - sum(dropped)` and `sum(survivors)` round differently.
    The gap is at the last ulp of the model image and reaches the next pass
    through the background.
"""
function subtract_sources!(fit, psf, plan::FitPlan, drop_mask::AbstractVector{Bool},
                           ny::Int, nx::Int)
    FT = eltype(fit.model)
    any(drop_mask) || return fit.model
    _accum_model!(vec(fit.model), psf, plan.free_names_val, plan.fixed, fit.theta, plan.p,
        fit.model_R, fit.geom.anchor_y, fit.geom.anchor_x, ny, nx, drop_mask,
        fit.render_buf, fit.render_scratch, -one(FT))
    return fit.model
end


# ==============================================================================
# Pruning
# ==============================================================================

"""
    prune_mask(catalog, snr_min, separation) -> (; keep, reasons, n_snr, n_close)

Which sources survive the post-fit cuts, and why the others do not.

Two cuts, in order: `catalog.flux_snr` below `snr_min`, then, of any surviving pair
closer than `separation`, the lower-SNR one (greedy by descending SNR, so the
brighter member is inserted first and survives).

Returns a mask rather than a modified catalog: the caller needs the same mask
for [`subtract_sources!`](@ref) as for [`drop!`](@ref), and separating "decide"
from "apply" is what lets both see the same indexing.
"""
function prune_mask(catalog::Catalog{T}, snr_min::Real, separation::Real) where {T}
    n = length(catalog)
    keep = trues(n)
    reasons = fill(:none, n)
    n_snr = 0
    for j in 1:n
        s = catalog.flux_snr[j]
        if !(isfinite(s) && s >= T(snr_min))
            keep[j] = false
            reasons[j] = :snr
            n_snr += 1
        end
    end
    n_close = 0
    if separation > 0
        order = sort(findall(keep); by = j -> -catalog.flux_snr[j])
        grid = _SepGrid{T}(separation, catalog.y, catalog.x)
        for j in order
            if _has_neighbor(grid, catalog.y[j], catalog.x[j])
                keep[j] = false
                reasons[j] = :close
                n_close += 1
            else
                _insert!(grid, catalog.y[j], catalog.x[j])
            end
        end
    end
    return (; keep, reasons, n_snr, n_close)
end


# ==============================================================================
# Trace statistics
# ==============================================================================

# Order statistics of a map, from a strided ~65k-element sample of its finite
# entries.  Trace-only, so an estimate is enough and an exact sort of a
# whole-frame map is not worth its allocation.
function _sample_quantiles(m::AbstractArray, qs, ::Type{FT}) where {FT}
    step = max(1, length(m) ÷ 65536)
    s = FT[]
    sizehint!(s, cld(length(m), step))
    @inbounds for i in 1:step:length(m)
        v = FT(m[i])
        isfinite(v) && push!(s, v)
    end
    isempty(s) && return map(_ -> FT(NaN), qs)
    return FT.(quantile(sort!(s), qs; sorted = true))
end


function _trace_pass_header(pass, max_iter, last_pass)
    label = last_pass ? "terminal pass" : "pass $pass / $max_iter"
    println("---- ", label, " ", "-"^max(0, 62 - length(label)))
    return nothing
end

function _trace_background(r::NamedTuple)
    println("  background   level box: ", lpad(r.bkg_box, 3), " pix (", r.bkg_coarse ? "coarse" : "fine",
        ")   rms box: ", lpad(r.bkg_rms_box, 3), " pix")
    println("               median ", @sprintf("%.4e", r.bkg_median),
        "   rms median ", @sprintf("%.4e", r.rms_median),
        "  [p10 ", @sprintf("%.2e", r.rms_p10), ", p90 ", @sprintf("%.2e", r.rms_p90), "]")
    return nothing
end

function _trace_detection(r::NamedTuple)
    println("  detection    ", r.n_peaks, " peaks above threshold")
    if r.n_blend_rejected > 0
        println("               rejected: ", r.n_blend_rejected, " blend-gated")
    end
    println("               rejected: ", r.n_dup_catalog, " already in catalog, ",
        r.n_dup_discarded, " previously discarded")
    println("               -> ", r.n_new, " new")
    println("               catalog ", r.n_catalog, " sources")
    return nothing
end

function _trace_fit_step(pass, lin, trial, cost, cost_cand, lambda, gnorm, status)
    pct = cost > 0 ? 100 * (cost_cand - cost) / cost : zero(cost)
    println("  lin ", lpad(lin, 2), " trial ", lpad(trial, 2),
        " | cost ", @sprintf("%.4e", cost), " -> ", @sprintf("%.4e", cost_cand),
        " (", @sprintf("%+.2f%%", pct), ")",
        " | lam ", @sprintf("%.1e", lambda),
        " | |g| ", @sprintf("%.2f", gnorm), " | ", status)
    return nothing
end

function _trace_prune(r::NamedTuple)
    println("  prune        dropped: ", r.n_pruned_snr, " low SNR, ", r.n_pruned_close,
        " close pairs, ", r.n_pruned_nopix, " no pixels")
    frac = r.n_new > 0 ? 100 * r.n_new_surviving / r.n_new : 0.0
    println("               of ", r.n_new, " new this pass, ", r.n_new_surviving,
        " survived (", @sprintf("%.1f%%", frac), ")")
    println("               catalog ", r.n_catalog, " sources")
    return nothing
end

function _trace_timing(r::NamedTuple)
    println("  timing       bkg ", @sprintf("%.2fs", r.t_background),
        "  detect ", @sprintf("%.2fs", r.t_detect),
        "  fit ", @sprintf("%.2fs", r.t_fit),
        "  prune ", @sprintf("%.2fs", r.t_prune),
        "  render ", @sprintf("%.2fs", r.t_render))
    return nothing
end

function _trace_setup(t_setup)
    println("---- setup ", "-"^58)
    println("  timing       ", @sprintf("%.2fs", t_setup))
    return nothing
end

function _trace_finalize(t_finalize)
    println("---- finalize ", "-"^55)
    println("  timing       ", @sprintf("%.2fs", t_finalize),
        "   (errors, morphology, diagnostics)")
    return nothing
end

function _trace_summary(history, converged, criterion, t_setup, t_finalize)
    println("==== multipass summary ", "-"^45)
    println("  pass      new    pruned   catalog     time")
    println("  ", lpad("setup", 4), lpad("-", 9), lpad("-", 10), lpad("-", 10),
        @sprintf("%9.2fs", t_setup))
    total_new = 0
    total_pruned = 0
    total_time = t_setup + t_finalize
    for r in history
        pruned = r.n_pruned_snr + r.n_pruned_close + r.n_pruned_nopix
        t = r.t_background + r.t_detect + r.t_fit + r.t_prune + r.t_render
        total_new += r.n_new
        total_pruned += pruned
        total_time += t
        label = r.last_pass ? "term" : string(r.pass)
        println("  ", lpad(label, 4), lpad(r.n_new, 9), lpad(pruned, 10),
            lpad(r.n_catalog, 10), @sprintf("%9.2fs", t))
    end
    println("  ", lpad("final", 4), lpad("-", 9), lpad("-", 10), lpad("-", 10),
        @sprintf("%9.2fs", t_finalize))
    println("  passes run ", length(history), ", detected ", total_new, ", pruned ", total_pruned,
        ", final catalog ", isempty(history) ? 0 : last(history).n_catalog)
    println("  ", converged ? "converged" : "stopped", " (", criterion, ")   total ",
        @sprintf("%.2fs", total_time), "  [setup ", @sprintf("%.2fs", t_setup),
        ", finalize ", @sprintf("%.2fs", t_finalize), "]")
    return nothing
end


# ==============================================================================
# Driver
# ==============================================================================

"""
    fit_all_stars_simultaneous_multipass(image, psf, [sources], fit_rad; kws...)

Iterated background estimation, source detection, and simultaneous PSF-fitting
photometry.

Each *pass* re-estimates the background from the current residual, detects new
sources on that residual, merges them into the working catalog, and then re-fits
the whole catalog simultaneously with a damped Gauss-Newton / Levenberg-Marquardt
step.  Each step's linear subproblem is solved with a method from Krylov.jl
(LSQR by default) on the weighted, column-equilibrated Jacobian `J`, applied
matrix-free straight from the per-source stamp derivatives -- `J` is never
materialized and no explicit normal matrix is built.  Passes repeat
until a detection pass yields no net new sources or `max_iter` is reached,
followed by one terminal pass that re-fits without detecting or pruning.

# Arguments

- `image::AbstractMatrix`: the **raw** (not background-subtracted) image.  The
  background is re-estimated every pass and is part of the returned result.
- `psf::AbstractPSFModel`: PSF model shared by all sources.
- `sources`: optional initial catalog, in any format accepted by
  [`fit_all_stars`](@ref), or `nothing` (the default) to start
  empty and let the first detection pass build the catalog from scratch.  A
  supplied catalog is a warm start only: pass 1 still runs its own detection and
  appends anything the supplied catalog missed (existing entries suppress
  duplicates via `min_separation`).
- `fit_rad::Real`: fitting radius in detector pixels.

# Keyword arguments

## Background

The background level and the background RMS want different mesh scales, so they
are estimated by two separate [`Background2D`](@ref) calls.

- `bkg_box_size::Integer = 20`: mesh size for the background *level* once the
  model is good enough to trust a fine mesh.
- `bkg_box_size_coarse::Integer = 8 * bkg_box_size`: mesh size for the level
  during the first `bkg_coarse_passes` passes, when unsubtracted starlight would
  otherwise be absorbed into a fine mesh and over-subtracted.  The multiplier is
  calibrated against `bkg_box_size = 20`: it reproduces crowdsource's coarse
  scale (`50 x` the rough PSF FWHM, so ~150 px at FWHM 3) for that fine mesh
  only.  If you retune `bkg_box_size`, set this explicitly -- the physical scale
  that matters is the PSF's, not the fine mesh's.
- `bkg_coarse_passes::Integer = 2`: number of leading passes using the coarse mesh.
- `bkg_rms_box_size::Integer = bkg_box_size`: mesh size for the RMS map.  Not
  scheduled: the RMS map wants a fine mesh throughout, since its job is to track
  small-scale confusion noise, and over-subtraction is not a failure mode for it.
- `bkg_estimator = SExtractorBackground()`: background level estimator.
- `bkg_rms_estimator = MADStdRMS()`: background RMS estimator.
- `bkg_kws::NamedTuple = (;)`: extra keyword arguments merged into (and
  overriding) the defaults for both `Background2D` calls, e.g.
  `(; filter_size = (5, 5), sigma = (2.0, 5.0))`.
- `mask`, `coverage_mask`: forwarded to `Background2D` (see its docstring).

## Detection

- `detection_kernel::Union{Nothing, AbstractMatrix} = nothing`: matched-filter
  kernel.  `nothing` renders `psf` over a `±kernel_rad` box at the image center.
- `kernel_rad::Integer = 5`: half-width of the rendered detection kernel.
- `detect_sigma::Real = 5.0`: detection threshold, in units of the significance map.
- `normalize_zerosum::Bool = true`: forwarded to [`matched_filter`](@ref).  Kept
  `true` even though this function owns the background and detects on
  `image - background - model` -- this would typically make a zero-sum kernel
  unnecessary, but err on the side of eliminating bias at the cost of some sensitivity
  by default.

  Setting `normalize_zerosum=false` increases sensitivity as `kernel_norm` rises from
  `sqrt(sum(P^2) - sum(P)^2 / N)` to `sqrt(sum(P^2))`, worth 2-9% in SNR over
  plausible FWHM/`kernel_rad` combinations (worst when the kernel box is tight
  around the PSF).  That cost is uniform and predictable -- a couple of hundredths
  of a magnitude of depth, everywhere, which an artificial-star completeness test
  measures correctly.

  It costs pedestal sensitivity: a uniform residual background error `delta`
  shifts the significance map by `delta * sum(P) / sqrt(sum(P^2))`, which is 3-6
  sigma per 1 sigma of pedestal for typical PSFs (independent of `kernel_rad`,
  since `sum(P) = 1` for a normalized PSF).  That cost is **spatially correlated
  with crowding**: the mesh background is biased high by unsubtracted starlight
  exactly where the field is dense, so `image - background` is biased low there,
  and the effective detection threshold tightens in the crowded core while
  staying put in the outskirts.  For a crowded-field luminosity function that is
  the worst available failure mode -- it recovers fewer faint sources where it is
  dense, which mimics the astrophysical signal, and a completeness correction
  only partly absorbs it because the bias tracks the local background *error*
  rather than local density.

  A stable threshold beats a nominally deeper but spatially variable one, so we
  choose to take the few-percent loss in sensitivity to gain the spatial stability
  by default.
- `min_separation::Real = 1.5`: a new peak is rejected when it falls within this
  distance (pixels) of a source already in the catalog, or of a position
  discarded on an earlier pass.

## Blend gating (off by default)

crowdsource has a feature that compares each candidate peak's significance against the
significance of the *current model* at the same pixel, to avoid splitting a
poorly-modeled bright star into fragments.  

In our implementation, the gate builds a second significance map from the
*current model* on the same footing as the data map, and keeps a candidate peak at
pixel `p` only when

```
S_resid(p) > t * S_model(p)

S_resid = correlate(w .* R, K) / sqrt(correlate(w, K.^2))
S_model = correlate(w .* M, K) / sqrt(correlate(w, K.^2))
```

where `R = image - background - model` is the residual being searched, `M` is the
current model image, `w` is `detect_inv_var`, `K` is the normalized detection
kernel, and `t` is the threshold in force on this pass.

Because both maps share the denominator, the test reduces to
`correlate(w .* R, K) > t * correlate(w .* M, K)`: **`t` is a
residual-to-model flux ratio in the
matched-filter sense**, not a number of sigma. `t = 0.25` means "the light left
over here must be at least a quarter of the model's own light here". The gate
is therefore only effective near modeled sources -- where nothing is modeled,
`S_model ≈ 0` and every peak passes -- and it is insensitive to the noise
normalization, though not to `normalize_zerosum` (see the note there: a zero-sum
kernel removes the local mean, which shrinks `correlate(w .* M, K)` for a smooth
model).

Gating is a **per-pass** property. On each pass the effective threshold is

```
t = pass <= blend_passes ? blend_threshold_initial : blend_threshold
```

and `t === nothing` means no gate on that pass (the extra correlation is skipped).
The following combinations are expressible:

| `blend_threshold` | `blend_threshold_initial` | passes `1:blend_passes` | later passes |
|---|---|---|---|
| `nothing` | `nothing` | off | off |
| `0.2` | `2.0` | `2.0` | `0.2` |
| `nothing` | `2.0` | `2.0` | **off** |
| `0.2` | `nothing` | off | `0.2` |

The third row is worth knowing about: the gate's *benefit* is largest early,
when a bright star's own fit is worst and its residual is therefore biggest, and
decays as the fit converges, while its *cost* -- rejecting genuine faint
companions, which sit exactly where `S_model` is large -- does not decay. That
argues for annealing the gate off rather than leaving a floor in place forever.

- `blend_threshold::Union{Nothing, Real} = nothing`: the `t` active from pass
  `blend_passes + 1` onward; `nothing` means those passes are not gated.
- `blend_threshold_initial::Union{Nothing, Real}`: the `t` on passes
  `1:blend_passes`; `nothing` means those passes are not gated. Defaults to
  `nothing` when `blend_threshold` is `nothing` (so the gate is off by default,
  entirely) and to `2.0` otherwise -- larger than a typical `blend_threshold`,
  i.e. **stricter**, demanding more leftover light before admitting a neighbor,
  so it splits less.
- `blend_passes::Integer = 2`: how many leading passes use
  `blend_threshold_initial`.

  !!! note
      Pass 1 can never gate anything, whatever the thresholds. `model` is all
      zeros when pass 1 detects -- the loop starts it at zero and the first fit
      has not run yet, warm-start catalog or not -- so `S_model = 0` everywhere
      and every peak clears any finite `t`. `blend_passes = 2` therefore starts the
      strict threshold on pass 2.

Tuning, if bright stars fragment:

1. Check `n_blend_rejected` in `pass_history` first. It is `0` under the
   defaults (both thresholds `nothing`), so the gate is not active and
   the cause is elsewhere -- most likely source Poisson noise
   having leaked into the detection weights (see `inv_var`), which walks the
   matched-filter peak onto the wings. Fragments landing within
   `min_separation` of the star are already rejected as duplicates; the gate is
   only for fragments further out than that.
2. Enable it by setting a threshold, and start small: as a flux ratio,
   `0.1`-`0.3` already rejects a lot. Raise it until the fragments go and watch
   `n_blend_rejected` climb.
3. If fragments appear only on the first passes -- the usual case -- set
   `blend_threshold_initial` and `blend_passes` and leave `blend_threshold` at
   `nothing`. That puts the strict bar in place only while the model is bad and
   stops gating entirely once it is good, rather than leaving a floor that keeps
   rejecting real companions for the rest of the run.

The cost of raising `t` is real companions: a true source one FWHM from a bright
star sits exactly where `S_model` is large. That cost is partly recoverable --
a blend-gated peak is rejected **for that pass only** and is *not* added to the
discard list, so it can be detected again on a later pass once the bright star
subtracts more cleanly. This is what makes the pass-scheduled threshold safe,
and it is why the gate is preferable to raising `min_separation`, whose
rejections are permanent.

## Pruning

- `prune::Bool = true`: drop unreliable sources from the catalog after each fit.
- `prune_snr_min::Real = 3 * detect_sigma / 5`: minimum **signed** curvature
  significance, `flux * sqrt(H_ff)`.  This is proportional to but not equal to
  `flux / flux_err`: it uses the diagonal flux curvature, while the reported
  `flux_err` inverts the full per-source block, so the two differ by the
  flux-position covariance and this threshold runs optimistic.  Note that this is
  signed, not absolute, so it will also prune sources that resolve to have negative flux).
- `prune_separation::Real = 1.0`: if two sources are closer than this, drop the
  one with lower SNR.

## Morphology

- `morph_half_width::Union{Nothing, Integer} = nothing`: cutout half-width for
  the final per-source morphology pass, and for the seeding measurement on new
  stars detected each pass.  `nothing` uses `max(3, ceil(fit_rad))` for the former
  and the kernel-derived default for the latter.

  The morphology cutout is an aperture, so every pure-sky pixel it contains
  costs signal-to-noise on the shape, with an `r^2` lever arm on the second
  moments.  The aperture moments are tapered by a Gaussian window matched
  to the PSF FWHM, which bounds the sky-noise contribution.
  The window's compression is divided back out, so `fwhm`, the
  ellipticity components and `compactness_aperture` stay absolute; see
  [`GaussianWindow`](@ref). Generally `fit_rad` is a reasonable scale
  because it was already chosen to encompass enough of the profile to carry
  the signal, not so much that sky dominates.

## Fitting

- `inv_var::Union{Nothing, AbstractMatrix} = nothing`: inverse variance for the
  LM fit.  A matrix (e.g., built from the calibrated `err` array) is used
  unchanged for every pass and is interpreted as the total inverse variance;
  `nothing` derives background-only weights from the pass's RMS map, and so
  will overstate the precision of bright stars and is only supported for testing
  purposes; it is not recommended for actual science measurements.
  This also serves as the bad pixel mask; set `inv_var` to zero at
  every pixel that should be excluded from detection and fitting.
- `fixed::NamedTuple = (;)`: parameters frozen for all sources.  `bkg` is always
  fixed to zero (this function subtracts its own background model; a free
  per-source pedestal on top of that is double-counting) and does not need to be
  listed.  Only `(y, x, flux)` may be free.
- `linearizations_per_pass::Integer = 1`: LM linearizations per detection pass.
  crowdsource does exactly one, which is the default here.  Values > 1 trade detection
  passes for fit accuracy within a pass.
- `linear_iterations::Integer = 10`: iteration cap for the inner Krylov solve
  per damping trial.  Kept small on purpose; see the note under `model_rad_nsigma`.
- `linear_tol::Real = 1.0e-4`: `atol`/`btol` for the linear solve.
  Each linearization is an approximation of a nonlinear objective, so
  solving it precisely is a waste of time.
- `λ_init::Real = 1.0e-3`: initial damping factor. Because λ is floored
  rather than reset here, it is pinned near this value for most of a run instead
  of decaying, so it should be set for the ill-conditioned case rather than as
  a starting guess that adapts away.
- `max_damping_trials::Integer = 8`: max damping retries per linearization.
- `max_iter::Integer = 10`: maximum number of *detection* passes.  A run performs
  at most `max_iter` of them and then one additional terminal pass, so at most
  `max_iter + 1` passes in total.  `max_iter = 1` therefore means one detection
  pass with pruning, followed by the terminal pass.  Must be at least 1.  This
  matches crowdsource, whose zero-indexed `titer` schedules `lastiter = titer + 1`
  at `titer == maxiter - 1`, giving `maxiter` detection iterations plus a terminal
  one.
- `min_iter::Integer = 4`: minimum number of detection passes before the
  early-convergence test (`few_sources`) can fire.  It does not gate the
  `max_iter` budget, so setting it above `max_iter` is legal and simply means the
  run always uses its full budget rather than stopping early.
- `few_sources::Integer = 100`: schedule the terminal pass when a pass
  contributes this many or fewer *surviving* new sources (counted after dedup
  and after that pass's pruning).  Matches crowdsource's
  `fewstars`.  Detection has a long tail -- late passes keep turning up a handful
  of marginal sources -- and every one of them costs a full re-fit of the entire
  catalog, so stopping early is beneficial.
- `freeze_positions_final::Bool = false`: hold `(y, x)` fixed on the terminal
  pass, fitting only fluxes, so the reported diagnostics correspond to a single
  fixed model.  crowdsource does this (`tpsfderiv = False` when
  `titer == lastiter`); with a real LM optimizer it should not be needed.  It is
  only meaningful because the terminal pass runs no detection -- applied to a
  pass that had just detected, it would freeze new sources at their
  matched-filter centroids before they ever moved.
- `solver::Symbol = :lsqr`: the Krylov.jl method used for the linearized
  subproblem, `:lsqr` or `:lsmr`.
- `model_rad::Union{Symbol, Real} = :auto`: half-width of the box each source's
  model is rendered/subtracted over -- distinct from the `fit_rad` Jacobian box
  (DAOPHOT's PSF radius vs fitting radius).  `fit_rad` can safely be kept small
  (~1-1.5 FWHM: it only sets which pixels constrain a source, and fitting a
  fixed normalized PSF over a truncated core is unbiased, just slightly noisier)
  as long as `model_rad` reaches out to where the PSF is below the noise, so a
  bright star's wings are subtracted from its neighbors' cores.  With a single
  radius (`model_rad == fit_rad`) a truncated stamp leaves bright wings
  unmodeled and neighboring free fluxes absorb them, biasing bright stars low in
  crowded fields.

  `:auto` sets `model_rad` per source: the smallest radius at which that
  source's wing surface brightness (from the PSF curve of growth) drops below
  `model_rad_nsigma` times the background noise (from the fit weights), so faint
  sources cost little and bright ones reach far.  A scalar applies one value to
  every source.  Both forms are clamped to `[ceil(fit_rad), ceil(model_rad_max)]`.
- `model_rad_max::Real = 10 * fit_rad`: hard cap on the auto `model_rad` (and
  the size of the internal render buffers).
- `model_rad_nsigma::Real = 1.0`: the auto threshold, in units of the
  background per-pixel noise.  Larger values give smaller `model_rad`.

  !!! note
      Raising `linear_iterations` or tightening `linear_tol` to make the solve
      more exact is **not** a safe way to improve bright-star precision in a
      crowded field.  A near-degenerate, tightly blended group's weakly
      determined directions (e.g. two overlapping stars whose fluxes are nearly
      exchangeable) get resolved more and more precisely the more the linear
      solver is allowed to work, amplifying noise into large (sometimes
      non-positive) flux swings rather than converging to a better answer.
      `λ` and step acceptance are driven by the *global* cost, so a step that
      blows up one small group can still be accepted.
- `max_step::Real = 1.0`: per-source position step cap in pixels.  Too large
  risks overshooting the minimum in a single step; too small slows convergence.
- `λ_up::Real = 10.0`: damping increase factor on a failed trial.
- `λ_down::Real = 10.0`: damping decrease factor on a successful trial.
- `λ_min::Real = 1.0e-12`, `λ_max::Real = 1.0e12`: damping bounds.
- `covariance_estimator = nothing`: an [`AbstractCovarianceEstimator`](@ref)
  used for the final per-source errors.  `nothing` selects
  [`KnownWeightsCovarianceEstimator`](@ref) when `inv_var` is given and
  [`ReweightedCovarianceEstimator`](@ref) otherwise.  Errors invert each
  source's diagonal block of `JᵀJ`, ignoring covariance with blended neighbors,
  so they are not correct marginal errors in a crowded field.
- `spread_model_fwhm::Union{Nothing, Real} = nothing`: FWHM of the reference
  exponential disk for `spread_model`; `nothing` derives it from the PSF's
  effective area (see [`MultiPassPhotResult`](@ref)).

!!! note "No `damping` keyword"
    This function does not accept a `damping` argument.
    Damping is applied through the Krylov
    solver's own Tikhonov term, and because the operator works in
    column-equilibrated coordinates (`δ_scaled = D·δ`), penalizing
    `λ‖δ_scaled‖²` is exactly Marquardt's `λ‖D·δ‖²`.  The damping form is
    therefore fixed by the solve, not selectable.
    `LevenbergDamping`'s uniform `λI` is not expressible here at all
    without dropping the column equilibration or appending explicit penalty
    rows, since Krylov's `λ` only supports an isotropic penalty in the solved
    variable.
- `show_trace::Bool = false`: print progress to stdout.  A `setup` section, then
  per pass a header with background and detection statistics, a line per damping
  trial, a pruning summary and a wall-clock split, then a `finalize` section, then
  a summary table.  The same numbers are returned whether or not tracing is on
  (`pass_history`, `t_setup`, `t_finalize`).

  Every phase of the call is timed and every timed phase is counted into the
  summary's total, so that total is comparable with the caller's own wall clock
  rather than silently omitting work: `setup` (validation, the long-lived state,
  the `inv_var` copy, seeding the catalog), the five per-pass buckets
  (`bkg`/`detect`/`fit`/`prune`/`render`), and `finalize` (the validity gate,
  covariance, morphology and diagnostics).  `setup` scales with the image,
  `finalize` with the catalog; on a 2000x2000 frame with ~55k sources the split
  is roughly 1% setup, 75% passes, 24% finalize, and the traced total accounts
  for ~99.8% of wall time.  Sample:

  ```
  ---- setup ----------------------------------------------------------
    timing       4.31s
  ---- pass 3 / 10 ---------------------------------------------------
    background   level box:  20 pix (fine)   rms box:  20 pix
                 median 1.0243e+00   rms median 9.0312e-02  [p10 8.11e-02, p90 1.18e-01]
    detection    2114 peaks above threshold
                 rejected: 308 already in catalog, 892 previously discarded
                 -> 914 new
                 catalog 41207 sources
    lin  1 trial  1 | cost 4.1382e+05 -> 4.1104e+05 (-0.67%) | lam 3.2e-05 | |g| 18.42 | accepted
    prune        dropped: 412 low SNR, 38 close pairs, 6 no pixels
                 of 914 new this pass, 512 survived (56.0%)
                 catalog 40751 sources
    timing       bkg 1.82s  detect 3.41s  fit 46.20s  prune 0.09s  render 2.14s
  ---- finalize -------------------------------------------------------
    timing       37.60s   (errors, morphology, diagnostics)
  ```

# Returns

A `NamedTuple`:

- `phot::MultiPassPhotResult`: the fit for the final catalog.  Its `n_passes`
  field counts LM linearizations, not detection passes; use
  `n_detection_passes` below.  Its `residual` is `image - background - model`.
  Its `valid` field is all-`true` by construction: this function owns its
  catalog and drops unfittable sources rather than masking them, so every
  returned row is a source that was actually fit.  `n_failed`/`failure_msgs` report what was dropped and why.

  !!! note
      The terminal pass does not prune, so the returned catalog can contain
      sources that fall below `prune_snr_min` on their final fit -- they passed
      the cut on the previous pass and were then re-fit.  Sources with non-finite
      or non-positive flux are still removed by the final gate.  Apply a
      significance cut on the returned `flux`/`flux_err` if you need one.

  Its `morphology` field holds one [`measure_star_shape_ref`](@ref) result per
  source, aligned index-for-index with the photometry and with `pixel`,
  `significance` and `flux` merged in.  Measured on
  **neighbor-subtracted** cutouts: each source's own model is added back into
  the residual, so its aperture moments are not contaminated by the light of
  its neighbors.

  Each entry also carries a populated `psf_ref` block: every statistic
  recomputed on the noiseless render at the same sub-pixel phase, weights and
  anchor.  Prefer the comparison against `psf_ref` (ratio for the concentration
  measures, difference for the ellipticity components) over the raw value; it
  cancels the pixel-phase and PSF-width dependence that otherwise dominates
  these statistics' scatter.  The raw values are kept because they are what you
  inspect to validate the PSF model itself: structure remaining in the
  *normalized* quantity is model error, structure only in the raw one is not.
  Which comparison each statistic takes is tabulated under "PSF-normalized
  statistics" in the Centroid Refinement and Morphology manual page.
- `background::Background2D`: the final background model.  Its `background` comes
  from a `bkg_box_size` mesh and its `background_rms` from a separate
  `bkg_rms_box_size` mesh, overwritten into the same struct.  The two
  full-resolution maps are the usable products; the struct's `box_size` and
  `mesh_background_rms` describe only the level mesh, so do not read them as
  metadata for the RMS map.
- `detection::MatchedFilterResult`: the result of the last pass that actually
  ran detection.  The terminal pass skips detection, so this is normally from the
  pass before it.
- `pass_number::Vector{Int}`: the pass on which each surviving source was first
  detected (crowdsource's `passno`).
- `n_detection_passes::Int`: number of detection passes actually run.
- `converged::Bool`: whether the loop exited on the no-new-sources test rather
  than on `max_iter`.
- `n_pruned::Int`: total sources discarded across all passes, for every reason
  (`:snr`, `:close`, `:no_pixels`, `:nonfinite`), not only the two pruning cuts.
  `pass_history`'s `n_pruned_snr` / `n_pruned_close` / `n_pruned_nopix` break it
  down per pass.
- `pass_history::Vector{<:NamedTuple}`: one report per pass, with every
  detection, pruning and timing counter
  the run produced.  Populated regardless of `show_trace`.  Its five timing
  fields -- `t_background`, `t_detect`, `t_fit`, `t_prune`, `t_render` -- sum to
  that pass's wall time.
- `t_setup::Float64`: seconds spent before the first pass (validation, the
  long-lived state, the `inv_var` copy, seeding the catalog).  Scales with the
  image.
- `t_finalize::Float64`: seconds spent after the last pass (the final validity
  gate, covariance, morphology, diagnostics).  Scales with the catalog, and is
  typically the larger of the two.

  `t_setup + sum(pass timings) + t_finalize` is the whole call, so it can be
  compared against an `@elapsed` around it without an unexplained remainder.
"""
function fit_all_stars_simultaneous_multipass(
        image::AbstractMatrix{T},
        psf::AbstractPSFModel,
        sources,
        fit_rad::Real;
        # --- background ---
        bkg_box_size::Integer = 20,
        bkg_box_size_coarse::Integer = 8 * bkg_box_size,
        bkg_coarse_passes::Integer = 2,
        bkg_rms_box_size::Integer = bkg_box_size,
        bkg_estimator = SExtractorBackground(),
        bkg_rms_estimator = MADStdRMS(),
        bkg_kws::NamedTuple = (;),
        mask::Union{Nothing, AbstractMatrix{Bool}} = nothing,
        coverage_mask::Union{Nothing, AbstractMatrix{Bool}} = nothing,
        # --- detection ---
        detection_kernel::Union{Nothing, AbstractMatrix} = nothing,
        kernel_rad::Integer = 5,
        detect_sigma::Real = 5.0,
        normalize_zerosum::Bool = true,
        min_separation::Real = 1.5,
        blend_threshold::Union{Nothing, Real} = nothing,
        blend_threshold_initial::Union{Nothing, Real} = isnothing(blend_threshold) ? nothing : 2.0,
        blend_passes::Integer = 2,
        # --- pruning ---
        prune::Bool = true,
        prune_snr_min::Real = 3 * detect_sigma / 5,
        prune_separation::Real = 1.0,
        # --- morphology ---
        morph_half_width::Union{Nothing, Integer} = nothing,
        # --- fitting ---
        fixed::NamedTuple = (;),
        inv_var::Union{Nothing, AbstractMatrix} = nothing,
        spread_model_fwhm::Union{Nothing, Real} = nothing,
        solver::Symbol = :lsqr,
        linear_iterations::Integer = 10,
        linear_tol::Real = 1.0e-4,
        model_rad::Union{Symbol, Real} = :auto,
        model_rad_max::Real = 10 * fit_rad,
        model_rad_nsigma::Real = 1.0,
        max_step::Real = 1.0,
        max_damping_trials::Integer = 8,
        linearizations_per_pass::Integer = 1,
        max_iter::Integer = 10,
        min_iter::Integer = 4,
        few_sources::Integer = 100,
        freeze_positions_final::Bool = false,
        λ_init::Real = 1.0e-3,
        λ_up::Real = 10.0,
        λ_down::Real = 10.0,
        λ_min::Real = 1.0e-12,
        λ_max::Real = 1.0e12,
        show_trace::Bool = false,
        covariance_estimator = nothing,
    ) where {T}

    FT = float(T)
    t_setup_start = time()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    max_iter > 0 || throw(ArgumentError("max_iter must be positive"))
    min_iter > 0 || throw(ArgumentError("min_iter must be positive"))
    # No `min_iter <= max_iter` check: the two gate different clauses of the
    # termination test, so `min_iter > max_iter` is well defined -- the
    # early-convergence clause simply never fires.
    linearizations_per_pass > 0 || throw(ArgumentError("linearizations_per_pass must be positive"))
    linear_iterations > 0 || throw(ArgumentError("linear_iterations must be positive"))
    linear_tol > 0 || throw(ArgumentError("linear_tol must be positive"))
    max_damping_trials > 0 || throw(ArgumentError("max_damping_trials must be positive"))
    (model_rad === :auto || (model_rad isa Real && model_rad > 0)) ||
        throw(ArgumentError("model_rad must be :auto or a positive real, got $(repr(model_rad))"))
    model_rad_max >= fit_rad || throw(ArgumentError("model_rad_max must be >= fit_rad"))
    model_rad_nsigma > 0 || throw(ArgumentError("model_rad_nsigma must be positive"))
    solver in (:lsqr, :lsmr) || throw(ArgumentError("solver must be :lsqr or :lsmr, got $(repr(solver))"))
    fit_rad > 0 || throw(ArgumentError("fit_rad must be positive"))
    few_sources >= 0 || throw(ArgumentError("few_sources must be non-negative"))
    bkg_coarse_passes >= 0 || throw(ArgumentError("bkg_coarse_passes must be non-negative"))
    bkg_box_size > 0 || throw(ArgumentError("bkg_box_size must be positive"))
    bkg_box_size_coarse > 0 || throw(ArgumentError("bkg_box_size_coarse must be positive"))
    bkg_rms_box_size > 0 || throw(ArgumentError("bkg_rms_box_size must be positive"))
    kernel_rad > 0 || throw(ArgumentError("kernel_rad must be positive"))
    # This function owns the background model, so a free per-source pedestal on
    # top of it is double-counting: pin `bkg` rather than making the caller know to.
    if haskey(fixed, :bkg) && !iszero(fixed.bkg)
        throw(ArgumentError("fixed.bkg must be zero (or omitted): this function owns the " *
            "background model, and a nonzero per-source pedestal is added once per source " *
            "footprint, accumulating wherever footprints overlap.  Got $(repr(fixed.bkg))."))
    end
    ny, nx = size(image)
    if inv_var !== nothing
        size(inv_var) == size(image) || throw(ArgumentError("`inv_var` must be the same size as `image`"))
    end

    # ------------------------------------------------------------------
    # The immutable plans: free parameters, options, detection kernel
    # ------------------------------------------------------------------
    plan = FitPlan(psf, merge(fixed, (; bkg = zero(FT))))
    fixed_iv = inv_var === nothing ? nothing : Matrix{FT}(inv_var)

    kernel = if detection_kernel === nothing
        # Field-constant by construction: one kernel rendered from `psf` at the
        # image center.  A spatially varying PSF would need one per region.
        yc, xc = (ny + 1) ÷ 2, (nx + 1) ÷ 2
        m = ConstructionBase.setproperties(psf, (; y = FT(yc), x = FT(xc), flux = one(FT), bkg = zero(FT)))
        R = Int(kernel_rad)
        Matrix{FT}(PSF.render!(Matrix{FT}(undef, 2R + 1, 2R + 1), m, (yc - R):(yc + R), (xc - R):(xc + R)))
    else
        Matrix{FT}(detection_kernel)
    end

    R_fit = round(Int, round(fit_rad, RoundUp))
    o = (;
        # background
        bkg_box_size = Int(bkg_box_size), bkg_box_size_coarse = Int(bkg_box_size_coarse),
        bkg_coarse_passes = Int(bkg_coarse_passes), bkg_rms_box_size = Int(bkg_rms_box_size),
        bkg_estimator, bkg_rms_estimator, bkg_kws, mask, coverage_mask, fixed_inv_var = fixed_iv,
        # detection
        kernel, detect_sigma = FT(detect_sigma), normalize_zerosum,
        min_separation = FT(min_separation),
        blend_threshold = blend_threshold === nothing ? nothing : FT(blend_threshold),
        blend_threshold_initial = blend_threshold_initial === nothing ? nothing : FT(blend_threshold_initial),
        blend_passes = Int(blend_passes),
        morph_half_width = morph_half_width === nothing ? nothing : Int(morph_half_width),
        # fit
        R_fit, R_cap = ceil(Int, model_rad === :auto ? model_rad_max : max(float(model_rad), float(fit_rad))),
        model_rad, model_rad_nsigma = FT(model_rad_nsigma), solver,
        linear_iterations = Int(linear_iterations), linear_tol = FT(linear_tol),
        max_step = FT(max_step), max_trials = Int(max_damping_trials),
        lambda_up = FT(λ_up), lambda_down = FT(λ_down),
        lambda_min = FT(λ_min), lambda_max = FT(λ_max),
    )

    # Same default-selection rule as `lm_irls`; with no IRLS reweighting here the
    # choice reduces to whether `inv_var` was given.
    cov_est = covariance_estimator !== nothing ? covariance_estimator :
        (inv_var !== nothing ? KnownWeightsCovarianceEstimator() : ReweightedCovarianceEstimator())

    # ------------------------------------------------------------------
    # Run state: five plain locals plus two accumulators
    # ------------------------------------------------------------------
    catalog = Catalog{FT}(sources, psf)
    disc = Discards{FT}()
    history = NamedTuple[]
    # Zeros to start, so pass 1 estimates the background on the raw image
    # (matching crowdsource's first iteration) even with a warm-start catalog.
    model = zeros(FT, ny, nx)
    lambda = FT(λ_init)
    fit = nothing
    bkg = nothing
    mfr = nothing

    t_setup = time() - t_setup_start
    show_trace && _trace_setup(t_setup)

    last_pass = false
    converged = false
    criterion = "max_iter"
    n_detection_passes = 0
    n_lin_total = 0
    pass_run = 0

    # Main loop over detection and fitting passes
    for pass in 1:(max_iter + 1)
        pass_run = pass
        show_trace && _trace_pass_header(pass, max_iter, last_pass)

        # --- background, RMS, weights, residual ---
        t0 = time()
        bkg = estimate_background_multipass(image, model, pass, o)
        bkg_median = _sample_quantiles(bkg.background, (0.5,), FT)[1]
        rms_p10, rms_median, rms_p90 = _sample_quantiles(bkg.rms, (0.1, 0.5, 0.9), FT)
        t_background = time() - t0
        # The `_trace_*` printers only read fields, so each gets exactly the ones
        # that exist at the point it is called -- which is what keeps the trace in
        # the order the work happened rather than all of it after the pass.
        show_trace && _trace_background((; bkg_box = bkg.box, bkg_coarse = bkg.coarse,
            bkg_rms_box = o.bkg_rms_box_size, bkg_median, rms_median, rms_p10, rms_p90))

        # --- detect, gate, dedup, seed, append ---
        t0 = time()
        det = if last_pass
            (; catalog, mfr, n_peaks = 0, n_blend_rejected = 0, n_dup_catalog = 0,
               n_dup_discarded = 0, n_new = 0)
        else
            n_detection_passes += 1
            detect_sources(catalog, disc, bkg, model, pass, o)
        end
        catalog = det.catalog
        det.mfr === nothing || (mfr = det.mfr)
        t_detect = time() - t0
        (show_trace && !last_pass) && _trace_detection((; det.n_peaks, det.n_blend_rejected,
            det.n_dup_catalog, det.n_dup_discarded, det.n_new, n_catalog = length(catalog)))

        # Inserting a faint source beside a bright one manufactures a
        # near-degenerate flux-exchange direction, and acceptance is global, so a
        # step that resolves it into a large wrong exchange can still lower the
        # total cost.  The floor is the ridge term; the adaptive ratchet above it
        # keeps doing trust-region work.
        det.n_new > 0 && (lambda = max(lambda, FT(λ_init)))
        lambda_start = lambda

        n_pruned_nopix = 0
        n_pruned_snr = 0
        n_pruned_close = 0
        t_fit = 0.0
        t_prune = 0.0
        t_render = 0.0
        cost_start = FT(NaN)
        cost_end = FT(NaN)
        gnorm = FT(NaN)
        n_lin = 0
        n_trials = 0
        n_accepted = 0

        if isempty(catalog)
            fill!(model, zero(FT))
        else
            # --- fit ---
            t0 = time()
            data, w = pass_weights(image, bkg, FT)
            # A source with no usable pixel -- off image, fully masked, or
            # non-finite parameters -- leaves the catalog here rather than being
            # carried through the fit as a validity mask, so the fit sees a dense
            # `1:length(catalog)`.  Rebuilding the geometry after the drop is one
            # cheap pass over the survivors and cannot get the pairing wrong,
            # which compacting its columns in place could.
            geom = stamp_geometry(catalog, w, R_fit, ny, nx)
            if !all(geom.ok)
                n_pruned_nopix = count(!, geom.ok)
                catalog = drop!(disc, catalog, geom.ok, :no_pixels, pass)
                geom = stamp_geometry(catalog, w, R_fit, ny, nx)
            end

            # ...and that drop can empty the catalog, which is why this is asked
            # twice rather than once at the top of the pass.
            if isempty(catalog)
                fill!(model, zero(FT))
                t_fit = time() - t0
            else
                fit = fit_pass(data, w, geom, catalog, psf, plan, o, lambda;
                    ny, nx, linearizations = Int(linearizations_per_pass),
                    freeze_positions = last_pass && freeze_positions_final,
                    pass, show_trace)
                lambda = fit.lambda
                cost_start, cost_end, gnorm = fit.cost_start, fit.cost, fit.gnorm
                n_lin, n_trials, n_accepted = fit.n_lin, fit.n_trials, fit.n_accepted
                n_lin_total += fit.n_lin
                catalog = catalog_from_theta(catalog, fit, plan)
                t_fit = time() - t0

                # --- prune ---
                t0 = time()
                if !last_pass && prune
                    pr = prune_mask(catalog, prune_snr_min, prune_separation)
                    n_pruned_snr, n_pruned_close = pr.n_snr, pr.n_close
                    # Before the catalog shrinks, while `pr.keep` still indexes
                    # `fit`: take the pruned sources back out of the model the fit
                    # already rendered, instead of re-rendering the survivors.
                    subtract_sources!(fit, psf, plan, .!pr.keep, ny, nx)
                    catalog = drop!(disc, catalog, pr.keep, pr.reasons, pass)
                end
                t_prune = time() - t0
                (show_trace && !last_pass) && _trace_prune((; n_pruned_snr, n_pruned_close,
                    n_pruned_nopix, n_new = det.n_new, n_catalog = length(catalog),
                    n_new_surviving = count(==(pass), catalog.pass)))

                # --- carry the model forward ---
                # `fit.model` is already the render at the final `theta`, minus
                # anything just pruned.  Timed in its own bucket so the trace
                # stays comparable with the re-rendering formulation.
                t0 = time()
                copyto!(model, fit.model)
                t_render = time() - t0
            end
        end

        n_new_surviving = count(==(pass), catalog.pass)
        report = (; pass, last_pass, bkg_box = bkg.box, bkg_rms_box = o.bkg_rms_box_size,
                    bkg_coarse = bkg.coarse, bkg_median, rms_median, rms_p10, rms_p90,
                    n_peaks = det.n_peaks, n_blend_rejected = det.n_blend_rejected,
                    n_dup_catalog = det.n_dup_catalog, n_dup_discarded = det.n_dup_discarded,
                    n_new = det.n_new, n_catalog = length(catalog),
                    n_lin, n_trials, n_accepted, cost_start, cost_end,
                    lambda_start, lambda_end = lambda, gnorm,
                    n_pruned_snr, n_pruned_close, n_pruned_nopix, n_new_surviving,
                    t_background, t_detect, t_fit, t_prune, t_render)
        push!(history, report)
        show_trace && _trace_timing(report)

        # Schedule a terminal pass rather than exiting here, so the last fit sees
        # the final catalog.
        last_pass && break
        if pass >= min_iter && n_new_surviving <= few_sources
            last_pass = true
            converged = true
            criterion = "n_new_surviving = $n_new_surviving <= few_sources = $few_sources"
        elseif pass == max_iter
            last_pass = true
            converged = false
            criterion = "max_iter budget = $max_iter"
        end
    end

    # ------------------------------------------------------------------
    # Finalize
    # ------------------------------------------------------------------
    t_finalize_start = time()

    # Final gate on non-finite or non-positive flux.  Dropping changes the source
    # set the stamps were built for, so the fit is rebuilt (one linearization, no
    # damping trials, so nothing moves) rather than compacted.
    if !isempty(catalog)
        # Local binding prevents boxing of catalog from list comprehension
        cy, cx, cf = catalog.y, catalog.x, catalog.flux
        ok = [isfinite(cy[j]) && isfinite(cx[j]) && isfinite(cf[j]) && cf[j] > 0
              for j in eachindex(cy)]
        if !all(ok)
            catalog = drop!(disc, catalog, ok, :nonfinite, pass_run)
            if !isempty(catalog)
                data, w = pass_weights(image, bkg, FT)
                geom = stamp_geometry(catalog, w, R_fit, ny, nx)
                if !all(geom.ok)
                    catalog = drop!(disc, catalog, geom.ok, :no_pixels, pass_run)
                    geom = stamp_geometry(catalog, w, R_fit, ny, nx)
                end
                if !isempty(catalog)
                    fit = fit_pass(data, w, geom, catalog, psf, plan, o, lambda;
                        ny, nx, linearizations = 1, max_trials = 0, pass = pass_run)
                    copyto!(model, fit.model)
                end
            end
        end
    end

    n_failed = count(r -> r === :no_pixels || r === :nonfinite, disc.reason)
    failure_msgs = String[]
    for k in eachindex(disc.reason)
        (disc.reason[k] === :no_pixels || disc.reason[k] === :nonfinite) || continue
        length(failure_msgs) < 10 || break
        push!(failure_msgs, "source at (y = $(disc.y[k]), x = $(disc.x[k])) " *
              "excluded on pass $(disc.pass[k]) ($(disc.reason[k]))")
    end

    if isempty(catalog) || fit === nothing
        # `model` may still hold sources the validation above discarded; with no
        # sources the correct model is identically zero.
        isempty(catalog) && fill!(model, zero(FT))
        residual = @. FT(image) - bkg.background - model
        phot = MultiPassPhotResult(FT[], FT[], FT[], FT[], FT[], FT[], FT[], FT[],
            falses(0), falses(0), FT[], FT[], FT[], FT[], FT[], FT[], FT[], Int[], 0,
            n_failed, failure_msgs, residual, NamedTuple[])
        return (; phot, background = bkg.result, detection = mfr,
                  pass_number = Int[], n_detection_passes, converged,
                  n_pruned = length(disc), pass_history = history,
                  t_setup, t_finalize = time() - t_finalize_start)
    end

    phot = finalize_multipass(image, psf, catalog, fit, bkg, mfr, plan, o, model, cov_est,
        fit_rad, spread_model_fwhm, n_lin_total, n_failed, failure_msgs)

    t_finalize = time() - t_finalize_start
    if show_trace
        _trace_finalize(t_finalize)
        _trace_summary(history, converged, criterion, t_setup, t_finalize)
    end
    return (; phot, background = bkg.result, detection = mfr,
              pass_number = copy(catalog.pass), n_detection_passes,
              converged, n_pruned = length(disc), pass_history = history, t_setup, t_finalize)
end

"""
    fit_all_stars_simultaneous_multipass(image, psf, fit_rad; kws...)

Convenience method starting from an empty catalog: the first detection pass
builds it from scratch.  Equivalent to passing `sources = nothing`.
"""
fit_all_stars_simultaneous_multipass(image::AbstractMatrix, psf::AbstractPSFModel, fit_rad::Real; kws...) =
    fit_all_stars_simultaneous_multipass(image, psf, nothing, fit_rad; kws...)

# Sub-view of a stamp rendered over `(yr_u, xr_u)` covering the inner box
# `(yr, xr)`, which must be contained in it.  Both boxes come from
# `_clamp_inds` on concentric ranges, so containment is structural.
function _inner_view(rend, yr_u::AbstractUnitRange, xr_u::AbstractUnitRange,
                     yr::AbstractUnitRange, xr::AbstractUnitRange)
    oy = first(yr) - first(yr_u) + 1
    ox = first(xr) - first(xr_u) + 1
    return view(rend, oy:(oy + length(yr) - 1), ox:(ox + length(xr) - 1))
end


# Per-source half of `finalize_multipass`: one render over the union of the
# diagnostics and morphology boxes, the goodness-of-fit diagnostics on the inner
# fit box, then the morphology on the `morph_hw` box.  `ctx` carries everything
# that does not vary with the source.
#
# A named function rather than the body of a `do` block inside
# `finalize_multipass`: the trailing `(; ...)` merge is constant-folded only when
# the optimizer sees a small enough unit, and inlined into the driver it fell
# back to a generic runtime `merge`, costing 42 allocations per source (measured
# at 54 MB over 11k sources, a 48% finalize slowdown).
function _finalize_source(ctx, m, j::Int, ay::Int, ax::Int, R_sub::Int, significance)
    FT = eltype(ctx.render_buf)
    image_bs, residual = ctx.image_bs, ctx.residual
    yr_u, xr_u = _clamp_inds((ay - ctx.R_u):(ay + ctx.R_u),
                             (ax - ctx.R_u):(ax + ctx.R_u), image_bs)
    rend = PSF.render!(ctx.render_buf, m, yr_u, xr_u, ctx.render_scratch)

    # We evaluate fit quality diagnostics on the same box the fit actually used,
    # `stamp_geometry`'s `anchor +- R_fit`. This runs before the
    # `bkg` pedestal comes off `rend`, since `_star_diagnostics!` expects the
    # rendered model with its pedestal and subtracts it itself.
    yr, xr = _clamp_inds((ay - ctx.R_fit):(ay + ctx.R_fit),
                         (ax - ctx.R_fit):(ax + ctx.R_fit), image_bs)
    if !isempty(yr) && !isempty(xr)
        ms = _inner_view(rend, yr_u, xr_u, yr, xr)
        gs = ctx.spread_kernel === nothing ? nothing :
            correlate!(view(ctx.g_stamp, axes(ms)...), ms, ctx.spread_kernel, :zero)
        _star_diagnostics!(ctx.diag, j, m, view(image_bs, yr, xr), view(residual, yr, xr),
            ms, gs, view(ctx.fit_iv, yr, xr), ctx.p)
    end

    rend .-= FT(m.bkg)   # `rend` is now the noiseless reference source
    yr_m, xr_m = _clamp_inds((ay - ctx.morph_hw):(ay + ctx.morph_hw),
                             (ax - ctx.morph_hw):(ax + ctx.morph_hw), image_bs)
    rend_m = _inner_view(rend, yr_u, xr_u, yr_m, xr_m)
    nym, nxm = length(yr_m), length(xr_m)

    # Isolated cutout: this star added back into the all-sources-removed
    # residual.  Neighbor light is gone; neighbor *noise* is not.
    clean = view(ctx.clean_buf, 1:nym, 1:nxm)
    clean .= view(residual, yr_m, xr_m)
    # Add back only where this source's model was actually subtracted.  Past
    # `model_R` the star's own wings were never removed, so they are already in
    # `residual` and rendering further would double-count them.
    iy = intersect(yr_m, (ay - R_sub):(ay + R_sub))
    ix = intersect(xr_m, (ax - R_sub):(ax + R_sub))
    if !isempty(iy) && !isempty(ix)
        ry = (first(iy) - first(yr_m) + 1):(last(iy) - first(yr_m) + 1)
        rx = (first(ix) - first(xr_m) + 1):(last(ix) - first(xr_m) + 1)
        view(clean, ry, rx) .+= view(rend_m, ry, rx)
    end

    # Anchor on the brightest pixel within +-1 of the fit anchor: `centroid_poly`
    # fits a quadratic and needs a local maximum.  The search range is built from
    # the anchor, not from the running best, so the neighborhood cannot drift
    # mid-scan.
    ci = ay - first(yr_m) + 1
    cj = ax - first(xr_m) + 1
    i0, j0, best = ci, cj, FT(-Inf)
    for i in max(1, ci - 1):min(nym, ci + 1), k in max(1, cj - 1):min(nxm, cj + 1)
        v = clean[i, k]
        if isfinite(v) && v > best
            best, i0, j0 = v, i, k
        end
    end

    # `morph_half_width < R_fit` can clamp the box to nothing for a source whose
    # anchor drifted off the frame.  Everything below degenerates to `NaN` on an
    # empty cutout; only this reduction would throw.
    height = isempty(rend_m) ? FT(NaN) : maximum(rend_m)
    shapes = measure_star_shape_ref(clean, rend_m, i0, j0, height;
        inv_var = ctx.morph_w === nothing ? nothing : view(ctx.morph_w, yr_m, xr_m),
        sharp_half_width = ctx.sharp_hw, window = ctx.morph_window,
        y_offset = first(yr_m) - 1, x_offset = first(xr_m) - 1)
    return (;
        pixel = CartesianIndex(i0 + first(yr_m) - 1, j0 + first(xr_m) - 1),
        significance,
        flux = FT(m.flux),
        shapes...,
    )
end

"""
    finalize_multipass(...) -> MultiPassPhotResult

Turn the converged fit into the returned photometry: per-source errors from the
`p x p` normal blocks, then one pass over the sources computing both the
goodness-of-fit diagnostics and the morphology.

Split out of the driver because it is one verb applied once, and because it is
the only part of the run that scales with the catalog rather than with the
image; keeping it here leaves the pass loop readable in one screen.
"""
function finalize_multipass(image, psf, catalog::Catalog{FT}, fit, bkg, mfr, plan::FitPlan, o,
                            model, cov_est, fit_rad, spread_model_fwhm, n_lin_total,
                            n_failed, failure_msgs) where {FT}
    ny, nx = size(image)
    n_src = length(catalog)
    p = plan.p
    stamp = fit.stamp

    # --- errors and covariance ---
    # Refill the stamps unmasked: `freeze_positions` may have zeroed the position
    # columns during the fit, and the reported position errors must not inherit
    # that.
    _fill_stamps!(stamp, psf, plan.free_names_val, plan.fixed, fit.theta, fit.w,
        plan.grad_col, fit.geom.dy_off, fit.geom.dx_off, fit.geom.anchor_y, fit.geom.anchor_x,
        plan.row_y, plan.row_x, plan.row_flux, trues(n_src), fit.fill_scratch)

    y_err = zeros(FT, n_src)
    x_err = zeros(FT, n_src)
    flux_err = zeros(FT, n_src)
    errs = Matrix{FT}(undef, p, n_src)
    _source_errors!(errs, stamp, cov_est, fit.cost, fit.dof)
    for j in 1:n_src, k in 1:p
        plan.grad_col[k] == 1 ? (y_err[j] = errs[k, j]) :
            plan.grad_col[k] == 2 ? (x_err[j] = errs[k, j]) : (flux_err[j] = errs[k, j])
    end

    # --- per-source diagnostics and morphology ---
    image_bs = reshape(fit.data, ny, nx)
    diag = _diagnostic_sinks(FT, n_src)

    residual = image_bs .- model
    fit_iv = reshape(fit.w, ny, nx)
    # Background-only weights, not source-inclusive.
    # Inverse-variance weighting is right for estimating a mean and wrong for
    # weighting a moment.  Where the variance tracks the signal, `1/var` is the
    # *anti*-matched filter -- it downweights the core by the source's own
    # Poisson term, and for a bright star it inverts the weight profile
    # entirely, so the moment sum measures the square cutout's geometry instead
    # of the source's. The wings are where the star/galaxy signal lives,
    # but the way to weight them up is a profile (Gaussian) window, not `1/var`.
    #
    # These stay the variance of the *data* even though the morphology runs on
    # neighbor-subtracted cutouts: removing a neighbor's model removes its
    # signal, not the Poisson noise of the photons it contributed.  `inv_var`'s
    # mask reaches here for free -- `detect_inv_var` is already zeroed wherever
    # the caller's map is.
    morph_iv = bkg.detect_inv_var
    morph_hw = o.morph_half_width !== nothing ? Int(o.morph_half_width) : max(3, o.R_fit)

    # The PSF's Gaussian-effective width, used for the `spread_model` reference disk
    # (unless overridden by `spread_model_fwhm`), the SHARP footprint,
    # and the Gaussian-windowed morphological moments.
    psf_fwhm = FT(PSF.effective_fwhm(ConstructionBase.setproperties(psf,
        (; y = catalog.y[1], x = catalog.x[1]))))
    spread_fwhm = spread_model_fwhm === nothing ? psf_fwhm : FT(spread_model_fwhm)
    # Resolved to tuple form once, which stops `correlate!` re-running its
    # separability test -- an SVD for a matrix kernel -- on every source
    spread_kernel = isfinite(spread_fwhm) && spread_fwhm > 0 ?
        _canonicalize(_exp_disk_kernel_bandlimited(spread_fwhm, FT; half = o.R_fit)) : nothing
    # DAOPHOT's SHARP footprint scales with the PSF, not with the detection
    # kernel, and not with `spread_fwhm`, which the user may override.
    sharp_hw = _sharp_half_width(psf_fwhm)
    # Aperture-moment window, matched to the PSF to maximize sensitivity 
    # to small departures from the PSF (see `GaussianWindow`).
    # Built once: the tables are indexed by integer offset from the anchor,
    # so every source reuses them.
    morph_window = isfinite(psf_fwhm) && psf_fwhm > 0 ?
        GaussianWindow(psf_fwhm) : GaussianWindow(one(FT))

    # One render per source over the union of the diagnostics and morphology
    # boxes.  Both are centered on the fit anchor: the diagnostics box has to be
    # the one the fit minimized over, and centering the morphology box anywhere
    # else would slide it off the footprint this source's model was subtracted
    # over.
    S_fit = 2 * o.R_fit + 1
    R_u = max(o.R_fit, morph_hw)
    S_u = 2 * R_u + 1
    render_buf = Matrix{FT}(undef, S_u, S_u)
    # Sized for the union box rather than reusing `fit.render_scratch`, which is
    # built for the largest *model* box and need not cover `morph_half_width`.
    render_scratch = PSF._render_scratch(psf, S_u, FT)
    g_stamp = Matrix{FT}(undef, S_fit, S_fit)
    clean_buf = Matrix{FT}(undef, S_u, S_u)

    # Loop-invariant context, bundled so the per-source worker takes a readable
    # argument list.
    ctx = (; image_bs, residual, fit_iv, morph_w = morph_iv, render_buf, render_scratch,
             g_stamp, clean_buf, spread_kernel, R_fit = o.R_fit, R_u, morph_hw,
             sharp_hw, morph_window, p, diag)
    morphology = map(1:n_src) do j
        m = PSF.model_from_vector(psf, plan.free_names_val,
                                  view(fit.theta, (j - 1) * p + 1:j * p), plan.fixed)
        _finalize_source(ctx, m, j, fit.geom.anchor_y[j], fit.geom.anchor_x[j],
                         fit.model_R[j], FT(catalog.flux_snr[j]))
    end

    bkg_vals = plan.row_bkg === nothing ? zeros(FT, n_src) : fill(FT(getfield(plan.fixed, :bkg)), n_src)
    return MultiPassPhotResult(
        copy(catalog.y), copy(catalog.x), y_err, x_err, copy(catalog.flux), flux_err,
        bkg_vals, zeros(FT, n_src), trues(n_src), trues(n_src), diag.chisq, diag.qfit,
        diag.qfit_expected, diag.qfit_z, diag.crowding, diag.spread_model,
        diag.spread_model_err, fill(n_lin_total, n_src),
        n_lin_total, n_failed, failure_msgs, residual, morphology)
end
