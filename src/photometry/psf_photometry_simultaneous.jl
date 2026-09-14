# Simultaneous PSF fitting photometry for a single image.
#
# A whole-image fitting algorithm that optimizes the parameters of
# every source at once, swappable for the DOLPHOT-style sequential
# `fit_all_stars` with every other pipeline stage (background, detection,
# morphology, PSF model) held fixed.
#
# The linear subproblem of each damped Gauss-Newton step is solved with Krylov.jl
# on the weighted, column-equilibrated Jacobian `J`, applied
# matrix-free straight from the per-star stamp derivatives -- `J` is never
# materialized and no explicit normal matrix or star-pair structure is built.
# Pixels shared between overlapping stars are handled implicitly by the
# scatter in the forward product `apply_J!`.

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

Per-source model-stamp half-width for [`fit_all_stars_simultaneous`](@ref).  The
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
    w_val = quantile(_w, 0.84)
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

function _build_stamps!(image, params, row_y, row_x, row_flux, fit_rad, w, FT)
    ny, nx = size(image)
    n_stars = size(params, 2)

    R = round(Int, round(fit_rad, RoundUp))
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

    anchor_y = round.(Int, view(params, row_y, :))
    anchor_x = round.(Int, view(params, row_x, :))

    pixels = zeros(Int32, S2, n_stars)
    for i in 1:n_stars
        ay = anchor_y[i]
        ax = anchor_x[i]
        for m in 1:S2
            gy = ay + dy_off[m]
            gx = ax + dx_off[m]
            if 1 <= gy <= ny && 1 <= gx <= nx
                fi = gy + (gx - 1) * ny
                if w[fi] > 0
                    pixels[m, i] = fi
                end
            end
        end
    end

    # Active stars: at least one usable pixel and finite initial y/x/flux.
    active = Int[]
    valid = trues(n_stars)
    failure_msgs = String[]
    for i in 1:n_stars
        ok = any(!iszero, view(pixels, :, i)) &&
             isfinite(params[row_y, i]) && isfinite(params[row_x, i]) &&
             isfinite(params[row_flux, i])
        if ok
            push!(active, i)
        else
            valid[i] = false
            push!(failure_msgs, "star $i: excluded before the solve (no usable pixels or non-finite initial parameters)")
        end
    end

    # Spatially cluster the active stars (Morton / Z-order on the integer
    # anchor) so catalog-order-scattered neighbors become adjacent in every
    # per-star loop.  The scatter/gather into the image-shaped
    # `model_img`/`wt_resid`/step vectors is then cache-local -- measured
    # ~1.4x on `apply_J!`/`apply_JT!` at whole-frame star counts.  Everything
    # downstream keys on the 1:n_active position via `active[j]` -> catalog
    # index, so the order is invisible to the result (up to the
    # already-order-dependent floating-point rounding of the overlapping-pixel
    # scatter-add).
    active = active[sortperm([_morton2d(anchor_y[i], anchor_x[i]) for i in active])]

    n_active = length(active)
    pixels_act = zeros(Int32, S2, n_active)
    anchor_y_act = Vector{Int}(undef, n_active)
    anchor_x_act = Vector{Int}(undef, n_active)
    for (j, i) in enumerate(active)
        pixels_act[:, j] .= view(pixels, :, i)
        anchor_y_act[j] = anchor_y[i]
        anchor_x_act[j] = anchor_x[i]
    end

    return (
        pixels = pixels_act, anchor_y = anchor_y_act, anchor_x = anchor_x_act,
        dy_off = dy_off, dx_off = dx_off, active = active, valid = valid,
        n_failed = n_stars - n_active, failure_msgs = failure_msgs, S2 = S2,
    )
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

Shared by [`fit_all_stars_simultaneous`](@ref) and
[`fit_all_stars_simultaneous_multipass`](@ref); `fit_all_stars` takes its
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
# Position step cap (§6.3)
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
# Per-star convergence and freezing
# ==============================================================================

"""
    _freeze_star!(θ, data_work, model_template, free_names_val, fixed, p,
                  model_R, anchor_y, anchor_x, ny, nx, j, render_buf, render_scratch)

Permanently remove star `j`'s current model contribution from `data_work` by
rendering it once over its own `model_rad` box (`O(model_R[j]^2)`, not a masked
call into [`_render_model!`](@ref), which would redo `O(n_active * model_R^2)`
work to extract one star) and subtracting the result. Called exactly once, at
the outer iteration a star freezes: from then on the star is skipped by every
per-star loop (`_fill_stamps!`, `apply_JT!`, `_render_model!`), and its light
is already accounted for in `data_work`, so it never needs to be rendered
again. The subtraction spans the full `model_rad` box (not the `fit_rad`
stamp), so a frozen bright star's wings stop contaminating still-live
neighbors' fits. `θ` for a frozen star simply stops changing thereafter, and
this renders through the same [`PSF.render!`](@ref) path `_render_model!` uses,
so the final diagnostics render (against the untouched original `data`, over
every star unconditionally) reproduces exactly what was subtracted here.

`render_buf` is the shared render scratch; it holds no live result at the point
in the outer loop where stars freeze.
"""
function _freeze_star!(
        θ, data_work, model_template, free_names_val, fixed, p,
        model_R, anchor_y, anchor_x, ny::Int, nx::Int, j,
        render_buf::AbstractMatrix, render_scratch
    )
    base = (j - 1) * p
    m = PSF.model_from_vector(model_template, free_names_val, view(θ, base + 1:base + p), fixed)
    ay = anchor_y[j]
    ax = anchor_x[j]
    R = model_R[j]
    S = 2R + 1
    PSF.render!(render_buf, m, (ay - R):(ay + R), (ax - R):(ax + R), render_scratch)
    @inbounds for jj in 1:S, ii in 1:S
        gy = ay - R + ii - 1
        gx = ax - R + jj - 1
        (1 <= gy <= ny) & (1 <= gx <= nx) || continue
        data_work[gy + (gx - 1) * ny] -= render_buf[ii, jj]
    end
    return data_work
end

"""
    _freeze_remaining!(live, frozen_at, star_converged, iter)

Mark every currently-live star converged and frozen at outer iteration
`iter`, without rendering or subtracting anything. Used when a *global*
convergence test (the existing whole-system `g_tol`/`x_tol`/`f_tol` checks)
fires: the outer loop is about to exit, and the final diagnostics render
(§8 of `fit_all_stars_simultaneous`) unconditionally re-renders every star
at its final `θ` against the untouched original `data` regardless of `live`,
so no per-star subtraction is needed here.
"""
function _freeze_remaining!(live, frozen_at, star_converged, iter)
    @inbounds for j in eachindex(live)
        if live[j]
            live[j] = false
            frozen_at[j] = iter
            star_converged[j] = true
        end
    end
    return live
end


"""
    _g_tol_local(j, p, colnorm_flat, b_scaled, C_r, g_tol)

Per-star analogue of the global gradient-cosine test in the outer loop
(`gnorm`): the same `max_k |b_true_k| / (sqrt(A_kk * C_r) + eps)` expression,
restricted to star `j`'s own `p`-parameter slice and reusing the same
shared, global `C_r = cost / dof` (`fitting_sparse_plan.md` §4.4 "Deviation
2": `C_r`'s denominator is deliberately global, not something to recompute
per star).
"""
function _g_tol_local(j, p, colnorm_flat, b_scaled, C_r, g_tol)
    FT = eltype(b_scaled)
    g_tiny = eps(FT)
    base = (j - 1) * p
    gnorm = zero(FT)
    @inbounds for k in 1:p
        i = base + k
        b_true = colnorm_flat[i] * b_scaled[i]
        A_ii = colnorm_flat[i]^2
        gnorm = max(gnorm, abs(b_true) / (sqrt(A_ii * C_r) + g_tiny))
    end
    return gnorm <= FT(g_tol)
end

"""
    _x_tol_local(j, p, D, colnorm_flat, δ_scaled, θ, x_tol)

Per-star analogue of the global rescaled-step-norm test in the outer loop
(`x_converged`), restricted to star `j`'s own `p`-parameter slice of the same
shared `D`/`colnorm_flat` arrays.
"""
function _x_tol_local(j, p, D, colnorm_flat, δ_scaled, θ, x_tol)
    FT = eltype(θ)
    base = (j - 1) * p
    num = zero(FT)
    xnorm = zero(FT)
    @inbounds for k in 1:p
        i = base + k
        v = (D[i] / colnorm_flat[i]) * δ_scaled[i]
        num += v * v
        xv = D[i] * θ[i]
        xnorm += xv * xv
    end
    return sqrt(num) <= FT(x_tol) * (sqrt(xnorm) + FT(x_tol))
end

# ==============================================================================
# Public entry point
# ==============================================================================

"""
    fit_all_stars_simultaneous(image, psf, sources, fit_rad; kws...) -> MultiPassPhotResult

Simultaneous PSF-fitting photometry: optimize the parameters of every source
at once with a damped Gauss-Newton / Levenberg-Marquardt loop.  Each step's
linear subproblem is solved with a method from Krylov.jl (LSQR by default)
on the weighted, column-equilibrated Jacobian `J`, applied matrix-free straight
from the per-star stamp derivatives -- `J` is never materialized and no explicit
normal matrix or star-pair structure is built.

Every non-fitting concern (background, detection, morphology, PSF model,
diagnostics) is shared with [`fit_all_stars`](@ref); only the optimizer
differs.  The two functions return the same [`MultiPassPhotResult`](@ref),
with one field whose *unit* differs (see `# Returns` below).

Stars are frozen out of the fit independently, as soon as each one satisfies
a per-star convergence test: its model is subtracted out of the working
image once and it is never re-rendered or re-linearized again, so later
outer iterations only pay for stars still moving.  This has no effect on the
converged answer for an isolated star.  For a star in a genuinely blended
group, the freeze test only looks at that star's own step size/gradient
given its neighbors' *current* (not yet converged) positions -- there is no
gate on neighbor convergence, matching how DAOPHOT's ALLSTAR (the closest
precedent for a whole-frame simultaneous fitter) drops converged stars.  In
a strongly degenerate flux-sharing blend this can let a star freeze slightly
before the pair is jointly optimal, since it can no longer respond to
further motion from a still-active neighbor afterward -- a bias in the point
estimate itself, not just in the (already-approximate, see below) reported
error.

# Comparability contract

- Only `(y, x, flux)` may be free per star.  Fix `bkg` (subtract the
  background first and pass `fixed = (; bkg = zero(eltype(image)))`) and fix
  all shape parameters.  Any other free parameter raises an `ArgumentError`.
- `fit_rad` (rounded up to the nearest integer) is the half-width of the
  Jacobian/cost stamp and the diagnostics box, so `qfit`, `chisq`, and
  `crowding` are computed over the same pixels in both arms.  Each star's
  model is *rendered and subtracted* over a separate, generally larger
  `model_rad` box (see below) -- this only affects how far a star's light is
  removed from its neighbors, not which pixels constrain its parameters.
- No robust reweighting: `reweight`, `scale_estimator`, and `weight_reset_tol`
  are not accepted.
- Errors invert the per-star diagonal block of `H = JᵀJ`; like `fit_all_stars`
  this ignores covariance with blended neighbors, so `flux_err`/`y_err`/`x_err`
  compare directly but are not correct marginal errors in a crowded field.

# Keyword arguments

- `inv_var::Union{Nothing, AbstractMatrix} = nothing`: per-pixel inverse
  variance for the weighted fit, interpreted as the **total** inverse variance
  including Poisson noise of source flux, the same shape as `image`.
  This also serves as the bad pixel mask; set `inv_var` to zero at
  every pixel that should be excluded from detection and fitting.
  The map is used exactly as given;
  supplying one also selects [`KnownWeightsCovarianceEstimator`](@ref) unless
  overwritten by the `covariance_estimator` keyword. This estimator assumes the
  supplied `inv_var` is correct, so a background-only inverse variance map yields position
  and flux errors that are too small for bright sources.
- `solver::Symbol = :lsqr`: symbol specifying the algorithm used to solve the
  linearized subproblem; any solver available through
  the generic interface of [Krylov.jl](https://jso.dev/Krylov.jl/stable/generic_interface/#Krylov.krylov_solve) can be used; `:lsqr` and `:lsmr` are common choices.
- `inner_iterations::Int = 10`: iteration cap for the inner linear solve per
  damping trial. Kept small on purpose -- solving the linear subproblem
  accurately against a stale linearization is wasted work.
- `linear_tol::Real = 1.0e-6`: `atol`/`btol` for the inner linear solve; lets an
  isolated or lightly-blended star's subsystem exit early instead of always
  burning the full cap.
- `model_rad::Union{Symbol, Real} = :auto`: half-width of the box each star's
  model is rendered/subtracted over -- distinct from the `fit_rad` Jacobian
  box (DAOPHOT's PSF radius vs fitting radius).  `fit_rad` can safely be kept
  small (~1-1.5 FWHM: it only sets which pixels constrain a star, and fitting
  a fixed normalized PSF over a truncated core is unbiased, just slightly
  noisier) as long as `model_rad` reaches out to where the PSF is below the
  noise, so a bright star's wings are subtracted from its neighbors' cores.
  With a single radius (`model_rad == fit_rad`) a truncated stamp leaves
  bright wings unmodeled and neighboring free fluxes absorb them, biasing
  bright stars low in crowded fields.

  `:auto` (default) sets `model_rad` per source: the smallest radius at which
  that source's wing surface brightness (from the PSF curve of growth) drops
  below `model_rad_nsigma` times the background noise (from `inv_var`), so
  faint sources cost little and bright ones reach far.  Requires `inv_var`.
  A scalar applies one value to every source (DAOPHOT-style; for testing and
  validation).  Both forms are clamped to `[ceil(fit_rad), ceil(model_rad_max)]`.
- `model_rad_max::Real = 10 * fit_rad`: hard cap on the auto `model_rad` (and
  the size of the internal render buffers).
- `model_rad_nsigma::Real = 1.0`: the auto threshold, in units of the
  background per-pixel noise.  Larger values give smaller `model_rad`.

  !!! note
      Raising `inner_iterations` or tightening `linear_tol` to make the solve
      more exact is **not** a safe way to improve bright-star precision in a
      crowded field. A genuinely near-degenerate, tightly-blended group's
      weakly determined directions (e.g. two overlapping stars whose fluxes
      are nearly exchangeable) get resolved more and more precisely the more
      the linear solver is allowed to work, amplifying noise into large
      (sometimes non-positive) flux swings rather than converging to a
      better answer. Nothing in the outer loop guards against this: `λ` and
      step acceptance are driven by the *global* cost, so a step that blows
      up one small group can still be accepted. The fix belongs in the
      damping/acceptance logic, not the inner-solve stopping rule. (LSQR
      works with `κ(J)` rather than `κ(J)²` for the normal equations, so it
      is better conditioned than CG-on-`H` would be, but the failure mode is
      the same.)
- `max_step::Real = 0.25`: per-star position step cap in pixels. Too large
  risks shooting past the actual minimum in the direction of the gradient
  in a single step; too small risks slowing convergence.  Recommend setting
  this to a small factor of the initial centroid uncertainty so that a star
  can reach its true position in a few steps, but not catastrophically
  overshoot it.
- `max_trials::Int = 8`: damping retries per outer iteration.
- `max_iter::Integer = 40` maximum number of outer iterations
- `x_tol::Real = 1.0e-5`: per-star rescaled step-norm convergence test
- `f_tol::Real = 1.0e-4`: whole-system relative cost improvement test
- `g_tol::Real = 1.0e-4`: per-star gradient-cosine convergence test
- `λ_init::Real = 1.0e-4`: initial damping factor
- `λ_up::Real = 10.0`: damping increase factor on a failed trial
- `λ_down::Real = 10.0`: damping decrease factor on a successful trial
- `λ_min::Real = 1.0e-12`: minimum damping factor
- `λ_max::Real = 1.0e12`: maximum damping factor

  !!! note
      The damping *form* is not selectable here (there is no `damping` keyword,
      and `damp!` is never called).  `λ` is applied as the Krylov solver's own
      Tikhonov term, and since the operator works in column-equilibrated
      coordinates (`δ_scaled = D·δ`), penalizing `λ‖δ_scaled‖²` is exactly
      Marquardt's `λ‖D·δ‖²`.  `LevenbergDamping`'s uniform `λI` would require
      dropping the equilibration or appending explicit penalty rows, since
      Krylov's `λ` only supports an isotropic penalty in the solved variable.
- `show_trace::Bool = false`: print a trace of the fitting process
- `covariance_estimator`: a [`AbstractCovarianceEstimator`](@ref) instance to
  compute the covariance of the final fit.  If `nothing`, uses
  [`KnownWeightsCovarianceEstimator`](@ref) if `inv_var` is not `nothing`, otherwise
  [`ReweightedCovarianceEstimator`](@ref).

# Returns

A [`MultiPassPhotResult`](@ref).  `converged[i]` is `true` if star `i` froze
via its own `x_tol`/`g_tol` test before `max_iter` iterations or the global
`f_tol` tolerance was reached; `false` otherwise.  `n_iter[i]` is the outer
iteration at which star `i` froze (or the final iteration count, for a star
never frozen). `n_passes` is the number of outer iterations actually run for
the fit as a whole. `n_failed`/`failure_msgs` report stars excluded before the solve.
"""
function fit_all_stars_simultaneous(
        image::AbstractMatrix{T},
        psf::AbstractPSFModel,
        sources,
        fit_rad::Real;
        fixed::NamedTuple = (;),
        inv_var = nothing,
        spread_model_fwhm::Union{Nothing, Real} = nothing,
        solver::Symbol = :lsqr,
        inner_iterations::Int = 10,
        linear_tol::Real = 1.0e-6,
        model_rad::Union{Symbol, Real} = :auto,
        model_rad_max::Real = 10 * fit_rad,
        model_rad_nsigma::Real = 1.0,
        max_step::Real = 0.25,
        max_trials::Int = 8,
        max_iter::Integer = 40,
        x_tol::Real = 1.0e-5,
        f_tol::Real = 1.0e-4,
        g_tol::Real = 1.0e-4,
        λ_init::Real = 1.0e-4,
        λ_up::Real = 10.0,
        λ_down::Real = 10.0,
        λ_min::Real = 1.0e-12,
        λ_max::Real = 1.0e12,
        show_trace::Bool = false,
        covariance_estimator = nothing,
    ) where {T}

    FT = float(T)
    max_iter > 0 || throw(ArgumentError("max_iter must be positive"))
    inner_iterations > 0 || throw(ArgumentError("inner_iterations must be positive"))
    linear_tol > 0 || throw(ArgumentError("linear_tol must be positive"))
    max_trials > 0 || throw(ArgumentError("max_trials must be positive"))
    (model_rad === :auto || (model_rad isa Real && model_rad > 0)) ||
        throw(ArgumentError("model_rad must be :auto or a positive real, got $(repr(model_rad))"))
    model_rad_max >= fit_rad || throw(ArgumentError("model_rad_max must be >= fit_rad"))
    model_rad_nsigma > 0 || throw(ArgumentError("model_rad_nsigma must be positive"))
    solver in (:lsqr, :lsmr) || throw(ArgumentError("solver must be :lsqr or :lsmr, got $(repr(solver))"))
    fit_rad > 0 || throw(ArgumentError("fit_rad must be positive"))

    # Same default-selection rule as `lm_irls` (levenberg_marquardt.jl); there is
    # no IRLS reweighting here, so the choice reduces to whether `inv_var` was given.
    covariance_estimator = if !isnothing(covariance_estimator)
        covariance_estimator
    elseif !isnothing(inv_var)
        KnownWeightsCovarianceEstimator()
    else
        ReweightedCovarianceEstimator()
    end

    # -------------------------------------------------------------------
    # 1. Catalog, free-parameter metadata
    # -------------------------------------------------------------------
    params, errors = _extract_source_catalog(sources, psf, FT)
    n_params, n_stars = size(params)
    n_stars == 0 && return MultiPassPhotResult(
        FT[], FT[], FT[], FT[], FT[], FT[], FT[], FT[],
        falses(0), falses(0), FT[], FT[], FT[], FT[], FT[], FT[], FT[], Int[], Int(0), Int(0), String[], Matrix{FT}(undef, 0, 0),
        NamedTuple[],
    )

    prop_names = collect(keys(ConstructionBase.getproperties(psf)))
    @assert length(prop_names) == n_params

    free_names, free_idx, _ = PSF.free_params(psf, fixed)
    isempty(free_idx) && throw(ArgumentError("all model parameters are fixed; nothing to fit"))
    offenders = setdiff(free_names, (:y, :x, :flux))
    isempty(offenders) || throw(ArgumentError(
        "fit_all_stars_simultaneous fits only (y, x, flux) per star; got free " *
        "parameters $(offenders). Fix `bkg` (subtract the background first and " *
        "pass `fixed = (; bkg = zero(eltype(image)))`), and fix all shape " *
        "parameters. Pass the same `fixed` to `fit_all_stars` for a " *
        "like-for-like comparison."
    ))
    p = length(free_idx)

    row_y = findfirst(==(:y), prop_names)
    row_x = findfirst(==(:x), prop_names)
    row_flux = findfirst(==(:flux), prop_names)
    row_bkg = findfirst(==(:bkg), prop_names)
    row_y === nothing && throw(ArgumentError("PSF model has no `y` field"))
    row_x === nothing && throw(ArgumentError("PSF model has no `x` field"))
    row_flux === nothing && throw(ArgumentError("PSF model has no `flux` field"))

    # grad_col[k] = 1 (dy), 2 (dx), or 3 (dflux) for free param k.
    grad_col = [free_names[k] === :y ? 1 : (free_names[k] === :x ? 2 : 3) for k in 1:p]
    k_y = findfirst(==(1), grad_col)
    k_x = findfirst(==(2), grad_col)

    free_names_val = Val(free_names)
    is_fixed = Tuple(setdiff(1:n_params, free_idx))
    for k in is_fixed
        errors[k, :] .= zero(FT)
    end

    # -------------------------------------------------------------------
    # 2. Data and weights
    # -------------------------------------------------------------------
    ny, nx = size(image)
    npix = ny * nx
    data = vec(Matrix{FT}(image))
    if inv_var === nothing
        all(isfinite, data) || throw(ArgumentError(
            "the image contains non-finite values but `inv_var` was not provided; " *
            "pass `inv_var` to mask them or clean the image first"
        ))
        w = ones(FT, npix)
    else
        size(inv_var) == size(image) || throw(ArgumentError("`inv_var` must be the same size as `image`"))
        w = Vector{FT}(undef, npix)
        iv = inv_var
        @inbounds for idx in eachindex(image)
            wv = iv[idx]
            w[idx] = if isfinite(wv) && wv > 0
                FT(wv)
            else
                zero(FT)
            end
        end
    end
    # Non-finite data pixels are zeroed AND zero-weighted
    @inbounds for q in 1:npix
        if !isfinite(data[q])
            data[q] = zero(FT)
            w[q] = zero(FT)
        end
    end

    # -------------------------------------------------------------------
    # 3. Stamp footprint (built once)
    # -------------------------------------------------------------------
    built = _build_stamps!(image, params, row_y, row_x, row_flux, fit_rad, w, FT)
    pixels = built.pixels
    anchor_y = built.anchor_y
    anchor_x = built.anchor_x
    dy_off = built.dy_off
    dx_off = built.dx_off
    active = built.active
    valid = built.valid
    n_failed = built.n_failed
    failure_msgs = built.failure_msgs
    S2 = built.S2
    n_active = length(active)

    n_active == 0 && return _empty_simultaneous_result(
        n_stars, params, errors, row_y, row_x, row_flux, row_bkg, valid, failure_msgs, FT, ny, nx)

    union_pix = _touched_pixels(pixels, npix)

    # Per-source model-render/subtraction box (distinct from the `fit_rad`
    # Jacobian/cost stamp): auto-scaled from the PSF curve of growth and the
    # image noise so a bright star's wings are subtracted far enough out that
    # neighbors' `fit_rad` cores stay clean, without paying that box for the
    # faint bulk.  One shared ring-ordered offset table (sized to the largest
    # per-source half-width actually chosen, not the cap) serves every source.
    (model_rad === :auto && inv_var === nothing) && throw(ArgumentError(
        "model_rad=:auto requires inv_var (it needs a noise scale); pass inv_var " *
        "or an explicit scalar model_rad"
    ))
    R_fit = round(Int, round(fit_rad, RoundUp))
    R_cap = ceil(Int, model_rad === :auto ? model_rad_max : max(float(model_rad), float(fit_rad)))
    model_R = _model_radii(psf, model_rad, model_rad_nsigma, R_fit, R_cap, w,
        FT[params[row_flux, i] for i in active])
    R_max = maximum(model_R)
    Rc_side = 2 * R_max + 1

    # -------------------------------------------------------------------
    # 4. Working buffers
    # -------------------------------------------------------------------
    n = p * n_active
    θ = Vector{FT}(undef, n)
    for (j, i) in enumerate(active)
        for k in 1:p
            θ[(j - 1) * p + k] = FT(params[free_idx[k], i])
        end
    end

    stamp = StampDerivatives{FT, Int32}(
        zeros(FT, p, S2, n_active), pixels, zeros(FT, p, n_active), npix, p, S2,
    )
    model_img = zeros(FT, npix)
    model_cand = zeros(FT, npix)
    wt_resid = zeros(FT, npix)
    mrhs = Vector{FT}(undef, npix)    # -wt_resid, the linear solve right-hand side
    render_buf = Matrix{FT}(undef, Rc_side, Rc_side)  # sized to the largest model box
    render_scratch = PSF._render_scratch(psf, Rc_side, FT)
    fill_scratch = _fill_scratch(psf, round(Int, sqrt(S2)), FT)

    b_scaled = Vector{FT}(undef, n)   # Jᵀ·wt_resid, for the g_tol cosine tests
    δ_scaled = Vector{FT}(undef, n)
    δ = Vector{FT}(undef, n)
    D = Vector{FT}(undef, n)
    sbuf = Vector{FT}(undef, S2)      # matrix-free operator scratch: one star's stamp

    # Per-star freezing state (§ "Per-star convergence and freezing" above).
    live = trues(n_active)
    frozen_at = zeros(Int, n_active)
    star_converged = falses(n_active)
    data_work = copy(data)

    # Matrix-free `J` operator and reusable workspace.  The operator
    # closes over `stamp`/`live`, both mutated in place across outer iterations,
    # so it is built once.
    J_op = _jacobian_operator(stamp, live, sbuf, npix, n)
    ws = Krylov.krylov_workspace(Val(solver), npix, n, Vector{FT})

    colnorm_flat = reshape(stamp.colnorm, n)

    # -------------------------------------------------------------------
    # 5. Initial fill and cost
    # -------------------------------------------------------------------
    _fill_stamps!(stamp, psf, free_names_val, fixed, θ, w,
        grad_col, dy_off, dx_off, anchor_y, anchor_x, row_y, row_x, row_flux, live, fill_scratch)
    _render_model!(model_img, psf, free_names_val, fixed, θ, p,
        model_R, anchor_y, anchor_x, ny, nx, live, render_buf, render_scratch)
    cost = _residual_cost!(wt_resid, model_img, data_work, w, union_pix)
    D .= colnorm_flat

    dof = max(length(union_pix) - n, 1)
    λ = FT(λ_init)
    converged = false
    n_run = 0

    # -------------------------------------------------------------------
    # 6. Outer damped Gauss-Newton / LM loop
    # -------------------------------------------------------------------
    for iter in 1:max_iter
        _fill_stamps!(stamp, psf, free_names_val, fixed, θ, w,
            grad_col, dy_off, dx_off, anchor_y, anchor_x, row_y, row_x, row_flux, live, fill_scratch)
        _render_model!(model_img, psf, free_names_val, fixed, θ, p,
            model_R, anchor_y, anchor_x, ny, nx, live, render_buf, render_scratch)
        cost = _residual_cost!(wt_resid, model_img, data_work, w, union_pix)
        @. mrhs = -wt_resid
        apply_JT!(b_scaled, stamp, wt_resid, live, sbuf)

        # g_converged (levenberg_marquardt.jl:535-540), reduced-cost denominator.
        # Global fallback: frozen stars' b_scaled is exactly 0 (apply_JT! skips
        # them), so they never dominate this max; a pass here means the whole
        # remaining live set has stopped moving.
        C_r = cost / dof
        g_tiny = eps(FT)
        gnorm = zero(FT)
        @inbounds for i in 1:n
            b_true = colnorm_flat[i] * b_scaled[i]
            A_ii = colnorm_flat[i]^2
            gnorm = max(gnorm, abs(b_true) / (sqrt(A_ii * C_r) + g_tiny))
        end
        if gnorm <= g_tol
            show_trace && println("g_tol convergence triggered")
            _freeze_remaining!(live, frozen_at, star_converged, iter - 1)
            converged = true
            n_run = iter - 1
            break
        end

        accepted = false
        f_converged = false
        for trial in 1:max_trials
            # Damping goes through the solver's own `λ` (Tikhonov) term: since
            # `δ_scaled = D·δ`, penalizing `λ²‖δ_scaled‖²` is exactly Marquardt's
            # `‖D·δ‖²`, and the operator is reused across trials with no rebuild.
            λ_solve = sqrt(FT(λ))
            Krylov.krylov_solve!(ws, J_op, mrhs; λ = λ_solve, itmax = inner_iterations,
                atol = FT(linear_tol), btol = FT(linear_tol))
            δ_scaled .= Krylov.solution(ws)

            @. δ = δ_scaled / colnorm_flat
            _cap_position_step!(δ, max_step, p, k_y, k_x)
            @. δ_scaled = colnorm_flat * δ

            _render_model!(model_cand, psf, free_names_val, fixed, θ .+ δ, p,
                model_R, anchor_y, anchor_x, ny, nx, live, render_buf, render_scratch)
            cost_cand = _cost!(model_cand, data_work, w, union_pix)
            Δcost = cost - cost_cand

            # f_converged: realized relative cost improvement
            f_converged = (cost_cand < cost) && (Δcost / cost <= FT(f_tol))

            if show_trace
                status = cost_cand < cost ? "accepted" : "rejected"
                println("Iter $(lpad(iter, 4)) trial $(lpad(trial, 2)) | converged = $(@sprintf("%5.2f%%", round(100 * count(!, live) / length(live), RoundDown; digits=2))) | cost = $(@sprintf("%.4e", cost)) -> $(@sprintf("%.4e", cost_cand)) | λ = $(@sprintf("%.2e", λ)) | ||g|| = $(@sprintf("%8.4f", gnorm)) | $status")
                if trial == max_trials
                    println("*** Max damping trials (`max_trials`) reached during outer iteration $(iter) ***")
                end
            end

            if cost_cand < cost
                accepted = true
                θ .+= δ
                cost = cost_cand
                λ = max(λ / FT(λ_down), FT(λ_min))
                @. D = max(D, colnorm_flat)
                break
            else
                λ = min(λ * FT(λ_up), FT(λ_max))
            end
        end

        n_run = iter
        if f_converged
            show_trace && println("f_tol convergence triggered")
            _freeze_remaining!(live, frozen_at, star_converged, iter)
            converged = true
            break
        end
        if accepted
            # Per-star freeze test, only meaningful for a step that was
            # actually applied: evaluated only here, not on a rejected trial.
            for j in 1:n_active
                live[j] || continue
                if _x_tol_local(j, p, D, colnorm_flat, δ_scaled, θ, x_tol) ||
                        _g_tol_local(j, p, colnorm_flat, b_scaled, C_r, g_tol)
                    _freeze_star!(θ, data_work, psf, free_names_val, fixed, p,
                        model_R, anchor_y, anchor_x, ny, nx, j, render_buf, render_scratch)
                    live[j] = false
                    frozen_at[j] = iter
                    star_converged[j] = true
                end
            end
            if !any(live)
                converged = true
                break
            end
        end
    end

    # -------------------------------------------------------------------
    # 7. Final parameter bookkeeping and validity gate
    # -------------------------------------------------------------------
    for (j, i) in enumerate(active)
        for k in 1:p
            params[free_idx[k], i] = θ[(j - 1) * p + k]
        end
    end
    # Fixed parameters take the `fixed` value (like fit_all_stars).
    for fn in keys(fixed)
        row = findfirst(==(fn), prop_names)
        row === nothing && continue
        params[row, :] .= FT(getfield(fixed, fn))
        errors[row, :] .= zero(FT)
    end

    # Errors: rebuild stamp values at final θ (every star, live or frozen --
    # this is a one-time end-of-fit pass, not the per-iteration loop, so
    # there is no per-star skip to apply here), invert per-star diagonal block.
    _fill_stamps!(stamp, psf, free_names_val, fixed, θ, w,
        grad_col, dy_off, dx_off, anchor_y, anchor_x, row_y, row_x, row_flux, trues(n_active), fill_scratch)

    errs = Matrix{FT}(undef, p, n_active)
    _source_errors!(errs, stamp, covariance_estimator, cost, dof)
    for (j, i) in enumerate(active)
        for k in 1:p
            errors[free_idx[k], i] = errs[k, j]
        end
    end

    # Final validity gate: non-finite or non-positive flux after convergence.
    for i in 1:n_stars
        if valid[i] && !(isfinite(params[row_y, i]) && isfinite(params[row_x, i]) &&
                isfinite(params[row_flux, i]) && params[row_flux, i] > 0)
            valid[i] = false
        end
    end

    # -------------------------------------------------------------------
    # 8. Diagnostics and residual image
    # -------------------------------------------------------------------
    # Every star, live or frozen, against the untouched original `data` (not
    # data_work): a frozen star's θ has simply stopped changing since it froze.
    _render_model!(model_img, psf, free_names_val, fixed, θ, p,
        model_R, anchor_y, anchor_x, ny, nx, trues(n_active), render_buf, render_scratch)

    diag = _diagnostic_sinks(FT, n_stars)

    global_resid = data .- model_img
    resid_mat = reshape(global_resid, ny, nx)
    # Small per-star model-render buffer for the diagnostics, reused across
    # stars.  The diagnostics box is the fit's own `anchor +- R_fit`, so exactly
    # this wide.  `dy_off` is the stamp footprint `_build_stamps!` laid out, so
    # taking `R_fit` from it keeps the two from drifting apart.
    R_fit = maximum(dy_off)
    S_max = 2 * R_fit + 1
    model_stamp = Matrix{FT}(undef, S_max, S_max)
    # spread_model reference: one field-constant exponential-disk kernel + a
    # reused buffer for the per-star PSF-convolved-with-disk stamp.
    spread_fwhm = isnothing(spread_model_fwhm) ?
        FT(PSF.effective_fwhm(ConstructionBase.setproperties(psf,
            (; y = FT(params[row_y, 1]), x = FT(params[row_x, 1]))))) : FT(spread_model_fwhm)
    # Resolved to tuple form once, which stops `correlate!` re-running its
    # separability test -- an SVD for a matrix kernel -- on every source
    spread_kernel = isfinite(spread_fwhm) && spread_fwhm > 0 ?
        _canonicalize(_exp_disk_kernel_bandlimited(spread_fwhm, FT; half = R_fit)) : nothing
    g_stamp = Matrix{FT}(undef, S_max, S_max)

    for (j, i) in enumerate(active)
        valid[i] || continue
        m = PSF.model_from_vector(psf, free_names_val, view(θ, (j - 1) * p + 1:j * p), fixed)
        # Use the box the fit actually used.
        ay, ax = anchor_y[j], anchor_x[j]
        yr, xr = _clamp_inds((ay - R_fit):(ay + R_fit), (ax - R_fit):(ax + R_fit), image)
        (isempty(yr) || isempty(xr)) && continue
        ms = PSF.render!(model_stamp, m, yr, xr)
        resid_stamp = view(resid_mat, yr, xr)
        ivv = inv_var === nothing ? nothing : view(inv_var, yr, xr)
        gs = spread_kernel === nothing ? nothing :
            correlate!(view(g_stamp, axes(ms)...), ms, spread_kernel, :zero)
        _star_diagnostics!(diag, i, m, view(image, yr, xr), resid_stamp, ms, gs, ivv, p)
    end

    residual = reshape(global_resid, ny, nx)

    # -------------------------------------------------------------------
    # 9. Assemble result
    # -------------------------------------------------------------------
    y = params[row_y, :]
    x = params[row_x, :]
    flux = params[row_flux, :]
    bkg = row_bkg === nothing ? zeros(FT, n_stars) : params[row_bkg, :]

    y_err = errors[row_y, :]
    x_err = errors[row_x, :]
    flux_err = errors[row_flux, :]
    bkg_err = row_bkg === nothing ? zeros(FT, n_stars) : errors[row_bkg, :]

    # Per-star converged/n_iter: star_converged/frozen_at are indexed by the
    # stable 1:n_active index; map back to catalog index i = active[j], the
    # same pattern already used above for θ/errors. A star still live when
    # the loop exited (max_iter reached) is not converged; its n_iter is the
    # final outer-iteration count n_run.
    converged = falses(n_stars)
    n_iter = zeros(Int, n_stars)
    for (j, i) in enumerate(active)
        converged[i] = star_converged[j]
        n_iter[i] = live[j] ? n_run : frozen_at[j]
    end

    return MultiPassPhotResult(
        y, x, y_err, x_err, flux, flux_err, bkg, bkg_err,
        converged, valid, diag.chisq, diag.qfit, diag.qfit_expected, diag.qfit_z,
        diag.crowding, diag.spread_model, diag.spread_model_err,
        n_iter, n_run, n_failed, failure_msgs, residual, NamedTuple[],
    )
end

function _empty_simultaneous_result(n_stars, params, errors, row_y, row_x, row_flux, row_bkg, valid, failure_msgs, FT, ny, nx)
    bkg = row_bkg === nothing ? zeros(FT, n_stars) : params[row_bkg, :]
    bkg_err = row_bkg === nothing ? zeros(FT, n_stars) : errors[row_bkg, :]
    diag = _diagnostic_sinks(FT, n_stars)
    return MultiPassPhotResult(
        params[row_y, :], params[row_x, :], errors[row_y, :], errors[row_x, :],
        params[row_flux, :], errors[row_flux, :], bkg, bkg_err,
        falses(n_stars), valid, diag.chisq, diag.qfit, diag.qfit_expected, diag.qfit_z,
        diag.crowding, diag.spread_model, diag.spread_model_err,
        zeros(Int, n_stars), Int(0), n_stars, failure_msgs, zeros(FT, ny, nx),
        NamedTuple[],
    )
end
