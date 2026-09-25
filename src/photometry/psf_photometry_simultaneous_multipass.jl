# Simultaneous (crowdsource-style) fitter for the multi-pass pipeline.
#
# Each pass re-optimizes every source at once with damped Gauss-Newton steps.
# The linear subproblem is solved with Krylov.jl on the weighted,
# column-equilibrated Jacobian `J`, applied matrix-free straight from per-source
# derivative stamps -- `J` is never materialized and no explicit normal matrix
# or star-pair structure is built.  Pixels shared between overlapping sources
# are handled implicitly by the scatter in the forward product `apply_J!`.
#
# An accepted damping trial *swaps* the `model` and `cand` bindings, so `model`
# always names the render at the current `theta` without a validity flag.

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
`D⁻¹ J' r`.  `z` is filled in place.  `sbuf` is a
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
refreshed in place each outer iteration), `live`, and `sbuf` (a length-`S²`
scratch shared by the forward and adjoint products, which the linear solver never
runs concurrently), so a single operator built once is valid for the whole fit.
"""
function _jacobian_operator(stamp::StampDerivatives{FT}, live, sbuf, npix::Int, n::Int) where {FT}
    fwd = (res, v) -> apply_J!(res, stamp, v, live, sbuf)
    adj = (res, u) -> apply_JT!(res, stamp, u, live, sbuf)
    return LinearOperators.LinearOperator(FT, npix, n, false, false, fwd, adj, adj)
end

function _fill_stamps_generic!(
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

Per-solve scratch for `_fill_stamps!`'s specialized
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

# Every model without a specialized fill streams `evaluate_fg` per stamp pixel.
_fill_stamps!(stamp::StampDerivatives, model_template, free_names_val, fixed, θ, w, grad_col,
        dy_off, dx_off, anchor_y, anchor_x, row_y, row_x, row_flux, live, fill_scratch_buf) =
    _fill_stamps_generic!(stamp, model_template, free_names_val, fixed, θ, w, grad_col,
        dy_off, dx_off, anchor_y, anchor_x, row_y, row_x, row_flux, live, fill_scratch_buf)

_fill_scratch(::Union{CircularGaussianPSF, CircularGaussianPRF}, S::Int, ::Type{FT}) where {FT} =
    PSF._separable_axis_buffers(S, S, FT)

"""
    _fill_stamps!(stamp, model_template::Union{CircularGaussianPSF, CircularGaussianPRF}, ...)

Separable value+gradient fill for the separable circular Gaussians, the stamp
counterpart of `PSF._accum_separable_gaussian!`: the per-axis factors of
`PSF._separable_axis!` are computed once per star over its `anchor +- R` rows
and columns, and each stamp pixel's `(dS/dY, dS/dX, S)` is a product of one x
factor and one y factor. Masking and column equilibration are as in the generic
method.
"""
function _fill_stamps!(
        stamp::StampDerivatives{FT}, model_template::Union{CircularGaussianPSF, CircularGaussianPRF}, free_names_val, fixed,
        θ, w, grad_col, dy_off, dx_off, anchor_y, anchor_x,
        row_y, row_x, row_flux, live, axis::NamedTuple
    ) where {FT}
    p = stamp.p
    S2 = stamp.S2
    R = maximum(dy_off)
    S = 2R + 1
    n_active = size(stamp.values, 3)
    n_active == 0 && return stamp
    Ey, Gy, Hy, Ex, Gx, Hx = axis.Ey, axis.Gy, axis.Hy, axis.Ex, axis.Gx, axis.Hx
    @inbounds for a in 0:(n_active - 1)
        live[a + 1] || continue
        base = a * p
        for k in 1:p
            stamp.colnorm[k, a + 1] = zero(FT)
        end
        m = PSF.model_from_vector(model_template, free_names_val, view(θ, base + 1:base + p), fixed)
        ay = anchor_y[a + 1]
        ax = anchor_x[a + 1]
        pm = PSF._separable_axes!(axis, m, (ay - R):(ay + R), (ax - R):(ax + R))
        αy, αx, amp, cF = FT(pm.αy), FT(pm.αx), FT(pm.amp), FT(pm.cF)
        mi = 0
        # Stamp layout is `for dx in -R:R, dy in -R:R`: x outer, y inner.
        for kx in 1:S, ky in 1:S
            mi += 1
            fi = stamp.pixels[mi, a + 1]
            if fi == 0
                for k in 1:p          # see the generic method's comment
                    stamp.values[k, mi, a + 1] = zero(FT)
                end
                continue
            end
            ex, ey = Ex[kx], Ey[ky]
            g1 = amp * αy * ex * Gy[ky]
            g2 = amp * αx * Gx[kx] * ey
            g3 = cF * ex * ey
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
    _cross_block!(B, t, V, i, j, dy, dx, R) -> B

Add `V_i' V_j`, summed over the pixels the stamps of sources `i` and `j` share,
into `B[:, :, t]`.  `V` is `StampDerivatives.values` and `(dy, dx)` is `j`'s
anchor minus `i`'s.  Stamps are the `(2R + 1)^2` footprints of
[`stamp_geometry`](@ref), stored `x`-major (`m = (dx + R) * S + dy + R + 1`), so
the shared pixels are one contiguous run of rows per shared column.  With
`i == j` and zero offset this is source `i`'s own block of `J' J`.

A masked stamp entry has `V == 0`, so it drops out of the sum without a test.
"""
function _cross_block!(B::AbstractArray{FT, 3}, t, V::AbstractArray{FT, 3}, i, j, dy, dx, R) where {FT}
    p = size(V, 1)
    S = 2R + 1
    @inbounds for ox in max(-R, dx - R):min(R, dx + R)
        mi0 = (ox + R) * S + R + 1
        mj0 = (ox - dx + R) * S + R - dy + 1
        rows = max(-R, dy - R):min(R, dy + R)
        if p == 3
            # Nine scalar accumulators: `B[k, l, t] +=` in the loop would store
            # through memory on every pixel.
            b11 = b21 = b31 = b12 = b22 = b32 = b13 = b23 = b33 = zero(FT)
            @simd for oy in rows
                u1, u2, u3 = V[1, mi0 + oy, i], V[2, mi0 + oy, i], V[3, mi0 + oy, i]
                v1, v2, v3 = V[1, mj0 + oy, j], V[2, mj0 + oy, j], V[3, mj0 + oy, j]
                b11 += u1 * v1; b21 += u2 * v1; b31 += u3 * v1
                b12 += u1 * v2; b22 += u2 * v2; b32 += u3 * v2
                b13 += u1 * v3; b23 += u2 * v3; b33 += u3 * v3
            end
            B[1, 1, t] += b11; B[2, 1, t] += b21; B[3, 1, t] += b31
            B[1, 2, t] += b12; B[2, 2, t] += b22; B[3, 2, t] += b32
            B[1, 3, t] += b13; B[2, 3, t] += b23; B[3, 3, t] += b33
        else
            for oy in rows, l in 1:p, k in 1:p
                B[k, l, t] += V[k, mi0 + oy, i] * V[l, mj0 + oy, j]
            end
        end
    end
    return B
end

"""
The most overlapping neighbors each source keeps as candidates for its error
group, and so the largest allowed `error_neighbors`.  Bounds the cost of the
coupling blocks, which would otherwise be expensive for large `fit_rad`
in a crowded field.  A source's most
strongly coupled neighbors are typically among its closest as well, so the cap has
minimal effect on the errors -- in Gaussian-PSF fields with ~200 overlapping neighbors
per source, capping at 32 changed no error by more than 1e-8 relative to no cap, and
halved the cost of the error step.
"""
const MAX_NEIGHBORS_ERR = 32

"""
    _overlap_pairs(stamp, anchor_y, anchor_x, cap = MAX_NEIGHBORS_ERR) -> NamedTuple

Pairs of sources whose stamps overlap, with the `p x p` off-diagonal block of the
equilibrated normal matrix `J' J` each pair contributes.

Two `(2R + 1)`-wide stamps overlap when their anchors differ by at most `2R` on
both axes, so bucketing anchors into cells `2R + 1` wide means every overlapping
partner lies in the same cell or one of its eight neighbors.

Each source keeps at most its `cap` nearest overlapping neighbors.

# Returns

- `B`: `(p, p, n_pairs)`, `B[:, :, t] = V_i' V_j` for the pair's first source `i`.
- `ptr`, `nbr`, `pair`: the adjacency in compressed form.  Source `a`'s
  neighbors are `nbr[ptr[a]:(ptr[a + 1] - 1)]`; the matching `pair` entry is `t`
  when `a` is the pair's first source (block `B[:, :, t]`) and `-t` when it is
  the second (block `B[:, :, t]'`).  Each block is stored once, not per direction,
  since at high densities the blocks outnumber the sources by an order of magnitude.
"""
function _overlap_pairs(stamp::StampDerivatives{FT}, anchor_y, anchor_x, cap::Integer = MAX_NEIGHBORS_ERR) where {FT}
    p, n = stamp.p, size(stamp.values, 3)
    S = isqrt(stamp.S2)
    R = S ÷ 2
    n == 0 && return (; B = zeros(FT, p, p, 0), ptr = ones(Int, 1), nbr = Int32[], pair = Int32[])
    # Cell-sorted source order: `cstart[c]:(cstart[c + 1] - 1)` indexes `order`.
    # One empty cell of padding on every side keeps the 3x3 scan in range.
    ymin, ymax = extrema(anchor_y)
    xmin, xmax = extrema(anchor_x)
    cy0 = fld(ymin, S) - 2
    cx0 = fld(xmin, S) - 2
    ncy = fld(ymax, S) - cy0 + 2
    ncx = fld(xmax, S) - cx0 + 2
    cell = [(fld(anchor_x[j], S) - cx0) * ncy + fld(anchor_y[j], S) - cy0 + 1 for j in 1:n]
    cstart = zeros(Int, ncy * ncx + 1)
    for c in cell
        cstart[c + 1] += 1
    end
    cstart[1] = 1
    cumsum!(cstart, cstart)
    order = sortperm(cell)

    # Call `f(i, d2)` for every source `i` whose stamp overlaps `j`'s, where `d2` is
    # their squared anchor distance.
    function each_overlap(f, j)
        @inbounds for ddx in -1:1, ddy in -1:1
            c = cell[j] + ddx * ncy + ddy
            for q in cstart[c]:(cstart[c + 1] - 1)
                i = order[q]
                dy, dx = anchor_y[i] - anchor_y[j], anchor_x[i] - anchor_x[j]
                (i != j && abs(dy) <= 2R && abs(dx) <= 2R) && f(i, dy * dy + dx * dx)
            end
        end
    end
    # `cut[j]` is the `(d2, index)` key of the last neighbor `j` keeps; `j` keeps
    # `i` when `(d2, i) <= cut[j]`.  Sources with at most `cap` neighbors keep all.
    cut = fill((typemax(Int), typemax(Int)), n)
    cand = Tuple{Int, Int}[]
    for j in 1:n
        # Every candidate lies in the 3x3 cells around `j`, whose counts are already
        # known, so only sources with more than `cap` of those need ranking.
        n_near = -1
        for ddx in -1:1, ddy in -1:1
            c = cell[j] + ddx * ncy + ddy
            n_near += cstart[c + 1] - cstart[c]
        end
        n_near > cap || continue
        empty!(cand)
        each_overlap((i, d2) -> push!(cand, (d2, Int(i))), j)
        length(cand) > cap && (cut[j] = partialsort!(cand, cap))
    end
    ia, ib = Int32[], Int32[]
    for j in 1:n
        each_overlap(j) do i, d2
            if i < j && ((d2, Int(i)) <= cut[j] || (d2, j) <= cut[i])
                push!(ia, i)
                push!(ib, j)
            end
        end
    end
    n_pairs = length(ia)
    B = zeros(FT, p, p, n_pairs)
    for t in 1:n_pairs
        i, j = ia[t], ib[t]
        _cross_block!(B, t, stamp.values, i, j, anchor_y[j] - anchor_y[i], anchor_x[j] - anchor_x[i], R)
    end

    ptr = zeros(Int, n + 1)
    for t in 1:n_pairs
        ptr[ia[t] + 1] += 1
        ptr[ib[t] + 1] += 1
    end
    ptr[1] = 1
    cumsum!(ptr, ptr)
    nbr = Vector{Int32}(undef, 2n_pairs)
    pair = Vector{Int32}(undef, 2n_pairs)
    next = ptr[1:n]
    for t in 1:n_pairs
        i, j = ia[t], ib[t]
        nbr[next[i]], pair[next[i]] = j, t
        nbr[next[j]], pair[next[j]] = i, -t
        next[i] += 1
        next[j] += 1
    end
    return (; B, ptr, nbr, pair)
end

"""
    _source_errors!(errs, stamp, geom, cov_est, cost, dof, max_neighbors, cap = MAX_NEIGHBORS_ERR) -> errs

Per-source 1-sigma parameter errors that account for covariance with blended
neighbors, for a catalog whose Jacobian is stored as `stamp`.

`stamp` must already hold the derivatives at the final `θ`, filled with every
source live (see `_fill_stamps!`): a source frozen during the fit has
zeroed columns, and its reported errors must not inherit that.  `geom` supplies
the stamp anchors (`anchor_y`, `anchor_x`) that `stamp` was filled on.

# Method

Under a linearized Gaussian likelihood the parameter covariance is `inv(J' J)`,
the inverse Fisher information; under the Gauss-Newton approximation the
second-order term `Σ r ∇²m` of the χ² Hessian is dropped, leaving only `J' J` .
Each source's marginal errors are its diagonal block of that inverse.
Computing those blocks needs a
factorization of the whole `pn x pn` normal matrix, for number of free parameters
per source `p` and number of sources `n`.  Its fill-in grows
steeply with the number of overlapping neighbors and the inversion becomes impractical.
Instead each source gets its own small problem.

Source `a` and up to `max_neighbors` of the sources whose stamps overlap its
own, ranked by the Frobenius norm of their coupling block, form a principal
submatrix of `J' J`, with `a` placed last.  Its Cholesky factor's last diagonal
block `L_aa` gives the Schur complement `L_aa L_aa'`, the Gauss-Newton curvature
of χ² in `a`'s parameters with those neighbors marginalized out, which  `covariance!`
subsequently inverts to give the marginal covariance of `a`'s parameters.
This treats every source outside the group as
fixed, so the variance is still a lower bound on the full-matrix solution `inv(J' J)` but
is much better than the own-block variance alone.  It increases
monotonically toward the full-matrix variance as the group grows, and the principal submatrix
of a positive-definite `J' J` is itself positive definite, so no neighbor choice can
break the factorization.  Measured against a full selected inverse on fields up to 145k sources
with ~16 overlapping neighbors each, `max_neighbors = 8` came within 1% of the full-matrix variance for
99% of sources and within 4% for 99.9%, while the own-block variance alone was low by a factor of
up to ~20 at the first percentile.

Ranking is by coupling strength, and deliberately not thresholded.  A blended
pair's Schur complement is small, so even a weak coupling to a third source
moves its variance by far more than the coupling itself suggests.

Neighbors are chosen from each source's `cap` nearest candidates (see
[`_overlap_pairs`](@ref)), so `max_neighbors` may not exceed `cap`.

`max_neighbors = 0` skips the neighbor search and inverts each source's own
block, ignoring its neighbors.  A group whose factorization fails (only
possible through rounding in a nearly degenerate blend) falls back to the same
own-block inversion for that source.

`errs` is filled as `(p, n)` in the stamp's own column order, so `errs[k, j]`
is the error on free parameter `k` (the `k`-th entry of `free_names`) of
source `j`.  Callers scatter it into whatever layout they report.

Used by [`fit_all_stars_simultaneous_multipass`](@ref); [`fit_all_stars_multipass`](@ref)
takes its errors from each source's own Levenberg-Marquardt normal matrix instead.
"""
function _source_errors!(errs::AbstractMatrix{FT}, stamp::StampDerivatives{FT}, geom,
                         cov_est, cost, dof, max_neighbors::Integer, cap::Integer = MAX_NEIGHBORS_ERR) where {FT}
    p, n = stamp.p, size(stamp.values, 3)
    size(errs) == (p, n) ||
        throw(DimensionMismatch("`errs` must be ($p, $n); got $(size(errs))"))
    max_neighbors >= 0 || throw(ArgumentError("max_neighbors must be non-negative, got $max_neighbors"))
    max_neighbors <= cap || throw(ArgumentError("max_neighbors must be at most $cap, got $max_neighbors"))
    R = isqrt(stamp.S2) ÷ 2
    # Everything is assembled in the column-equilibrated coordinates the stamp
    # stores (unit diagonal wherever a column has data), and only unscaled by
    # `stamp.colnorm` at the end.  The `1e-12 * tr` ridge must go on these
    # equilibrated blocks: on the raw block a bright source's position curvatures
    # (`~ flux^2`) exceed its flux curvature by many orders of magnitude, so the
    # ridge rivaled the flux diagonal and shrank a 7e5-count star's flux error by 20%.
    D = zeros(FT, p, p, n)
    for j in 1:n
        _cross_block!(D, j, stamp.values, j, j, 0, 0, R)
        tr = zero(FT)
        for k in 1:p
            tr += D[k, k, j]
        end
        for k in 1:p
            D[k, k, j] += max(FT(1.0e-12), eps(FT)) * tr
        end
    end
    # With no neighbors wanted, an empty adjacency of the same type skips the pair search.
    g = max_neighbors > 0 ? _overlap_pairs(stamp, geom.anchor_y, geom.anchor_x, cap) :
        (; B = zeros(FT, p, p, 0), ptr = ones(Int, n + 1), nbr = Int32[], pair = Int32[])
    # No source can use more neighbors than it has, so size the scratch by that.
    K = min(Int(max_neighbors), maximum(a -> g.ptr[a + 1] - g.ptr[a], 1:n; init = 0))
    H = Matrix{FT}(undef, p * (K + 1), p * (K + 1))
    blk = Matrix{FT}(undef, p, p)
    Bs = zeros(FT, p, p, 1)   # a coupling block built on the spot, for pairs the cap dropped
    group = Vector{Int32}(undef, K + 1)
    slot = Vector{Int}(undef, K)   # adjacency slot of each chosen neighbor
    score, rank = FT[], Int[]
    # Copy a pair's block, as seen from the source whose adjacency slot `q` holds
    # it, into `H` at offset `(ro, co)`.
    function put_block!(ro, co, q)
        t = g.pair[q]
        @inbounds for c in 1:p, r in 1:p
            H[ro + r, co + c] = t > 0 ? g.B[r, c, t] : g.B[c, r, -t]
        end
    end
    for a in 1:n
        lo, hi = g.ptr[a], g.ptr[a + 1] - 1
        k = min(K, hi - lo + 1)
        if k < hi - lo + 1
            resize!(score, hi - lo + 1)
            resize!(rank, hi - lo + 1)
            for q in lo:hi
                score[q - lo + 1] = sum(abs2, view(g.B, :, :, abs(g.pair[q])))
            end
            partialsortperm!(rank, score, 1:k; rev = true)
            for u in 1:k
                slot[u] = lo - 1 + rank[u]
            end
        else
            for u in 1:k
                slot[u] = lo - 1 + u
            end
        end
        ok = false
        if k > 0
            m = k + 1
            M = p * m
            for u in 1:k
                group[u] = g.nbr[slot[u]]
            end
            group[m] = a
            # Lower triangle only: own blocks, then `a`'s row, then neighbor pairs.
            for u in 1:m
                H[(u - 1) * p .+ (1:p), (u - 1) * p .+ (1:p)] .= view(D, :, :, group[u])
            end
            for u in 1:k
                put_block!((m - 1) * p, (u - 1) * p, slot[u])
            end
            for u in 1:k, v in (u + 1):k
                gu, gv = group[u], group[v]
                q = findfirst(==(gu), view(g.nbr, g.ptr[gv]:(g.ptr[gv + 1] - 1)))
                if q === nothing
                    # Not in the pair list: either the stamps do not overlap, and
                    # `_cross_block!` returns zero, or the cap dropped the pair.
                    fill!(Bs, zero(FT))
                    _cross_block!(Bs, 1, stamp.values, gv, gu, geom.anchor_y[gu] - geom.anchor_y[gv],
                        geom.anchor_x[gu] - geom.anchor_x[gv], R)
                    H[(v - 1) * p .+ (1:p), (u - 1) * p .+ (1:p)] .= view(Bs, :, :, 1)
                else
                    put_block!((v - 1) * p, (u - 1) * p, g.ptr[gv] - 1 + q)
                end
            end
            F = cholesky!(Symmetric(view(H, 1:M, 1:M), :L); check = false)
            if issuccess(F)
                # `a`'s Schur complement `L_aa L_aa'`, from the lower-triangular last block.
                L = view(H, (M - p + 1):M, (M - p + 1):M)
                for c in 1:p, r in 1:p
                    acc = zero(FT)
                    for s in 1:min(r, c)
                        acc += L[r, s] * L[c, s]
                    end
                    blk[r, c] = acc
                end
                ok = true
            end
        end
        ok || copyto!(blk, view(D, :, :, a))
        # `inv(D B D) = D⁻¹ inv(B) D⁻¹`, and both estimators are `inv` times a scalar.
        cov = covariance!(cov_est, blk, cost, dof)
        for c in 1:p
            errs[c, a] = sqrt(max(zero(FT), cov[c, c])) / stamp.colnorm[c, a]
        end
    end
    return errs
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
# Fitter
# ==============================================================================

"""
    SimultaneousFitter{FT}

The crowdsource-style `AbstractMultipassFitter`: every pass runs
`linearizations` damped Gauss-Newton steps over all sources at once, each solved
matrix-free with Krylov.jl.  Built by [`fit_all_stars_simultaneous_multipass`](@ref)
from its fitter keywords, which the fields mirror.  The damping factor `λ` is the
state carried from pass to pass.
"""
struct SimultaneousFitter{FT} <: AbstractMultipassFitter
    solver::Symbol
    linearizations::Int
    linear_iterations::Int
    linear_tol::FT
    max_trials::Int
    lambda_init::FT
    lambda_up::FT
    lambda_down::FT
    lambda_min::FT
    lambda_max::FT
    error_neighbors::Int
end

initial_fit_state(fitter::SimultaneousFitter, ::Type{FT}) where {FT} = FT(fitter.lambda_init)
n_free_per_source(::SimultaneousFitter, plan::FitPlan) = plan.p
min_stamp_pixels(::SimultaneousFitter, ::FitPlan) = 1
empty_pass_stats(::SimultaneousFitter, lambda, ::Type{FT}) where {FT} =
    (; n_lin = 0, n_trials = 0, n_accepted = 0, cost_start = FT(NaN), cost_end = FT(NaN),
       lambda_start = lambda, lambda_end = lambda,
       fit_timing = (; setup = 0.0, stamps = 0.0, render = 0.0, solve = 0.0))

"""
    catalog_from_theta(catalog, fit, plan) -> Catalog

The catalog with the fitted parameters and the `flux_snr` written back.
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
        # no extra linear algebra.
        snr[j] = theta[base + plan.k_flux] * colnorm[plan.k_flux, j]
    end
    return Catalog{FT}(y, x, flux, snr, copy(catalog.pass), copy(catalog.bkg), copy(catalog.lambda))
end

"""
    fit_pass(fitter::SimultaneousFitter, data, w, geom, catalog, psf, plan, o, lambda; kws...) -> NamedTuple

Run `fitter.linearizations` damped Gauss-Newton linearizations of the whole
catalog against `data`, starting from the catalog's own parameters and the
damping factor `lambda`.

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

- `n_new`: sources detected this pass.  Any new source floors `lambda` at
  `fitter.lambda_init` first as a safety measure, because inserting a faint source
  beside a bright one manufactures a near-degenerate flux-exchange direction,
  and step acceptance is global, so a step that resolves it into a large wrong exchange
  can still lower the total cost.
- `move`: `false` runs one linearization with no damping trials -- fill the
  stamps and render the model at the incoming `theta` without moving it, which
  is what the post-validation rebuild wants.
- `freeze_positions`: zero the position columns of `J` *and* cap the position
  step at zero.  Zeroing alone would still spend the solve on positions and
  leave a flux component that is not the flux-only Gauss-Newton step; capping
  alone would let the solve waste its budget on directions that are then
  discarded.

# Returns

The fields [`AbstractMultipassFitter`](@ref) requires, plus `stamp`, `cost`,
`dof` and `fill_scratch` for [`source_errors`](@ref).  `stats` holds `n_lin`,
`n_trials` (damping trials), `n_accepted` (accepted trials), `cost_start`,
`cost_end`, `lambda_start` and `lambda_end`.

`stats.fit_timing` breaks the pass's wall time into `setup` (allocation and the
Krylov workspace), `stamps`, `render` and `solve` (the Krylov solve and the step it
produces). Some work is not timed (e.g., calculating the `cost`) so
the sum of the segments will generally be slightly less than the total `t_fit`.
"""
function fit_pass(fitter::SimultaneousFitter, data::Vector{FT}, w::Vector{FT}, geom,
                  catalog::Catalog{FT}, psf, plan::FitPlan, o, lambda::FT; ny::Int, nx::Int,
                  pass::Integer = 1, n_new::Integer = 0, freeze_positions::Bool = false,
                  move::Bool = true, show_trace::Bool = false) where {FT}
    t0 = time()
    n = length(catalog)
    p = plan.p
    n_par = p * n
    npix = length(data)
    S2 = geom.S2
    n_new > 0 && (lambda = max(lambda, fitter.lambda_init))
    lambda_start = lambda
    linearizations = move ? fitter.linearizations : 1
    max_trials = move ? fitter.max_trials : 0

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
    delta = Vector{FT}(undef, n_par)
    theta_cand = Vector{FT}(undef, n_par)

    render_buf = Matrix{FT}(undef, Rc_side, Rc_side)
    render_scratch = PSF._render_scratch(psf, Rc_side, FT)
    fill_scratch = _fill_scratch(psf, 2 * o.R_fit + 1, FT)
    J_op = _jacobian_operator(stamp, live, sbuf, npix, n_par)
    ws = Krylov.krylov_workspace(Val(fitter.solver), npix, n_par, Vector{FT})

    max_step = freeze_positions ? zero(FT) : o.max_step
    cost = FT(NaN)
    cost_start = FT(NaN)
    n_lin = 0
    n_trials = 0
    n_accepted = 0
    t_stamps = 0.0
    t_render = 0.0
    t_solve = 0.0
    t_setup = time() - t0

    for lin in 1:linearizations
        t0 = time()
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
        t_stamps += time() - t0
        t0 = time()
        _render_model!(model, psf, plan.free_names_val, plan.fixed, theta, p, model_R,
            geom.anchor_y, geom.anchor_x, ny, nx, live, render_buf, render_scratch)
        t_render += time() - t0
        cost = _residual_cost!(wt_resid, model, data, w, union_pix)
        n_lin += 1
        lin == 1 && (cost_start = cost)
        @. mrhs = -wt_resid
        colnorm_flat = reshape(stamp.colnorm, n_par)

        accepted = false
        for trial in 1:max_trials
            t0 = time()
            Krylov.krylov_solve!(ws, J_op, mrhs; λ = sqrt(lambda),
                itmax = fitter.linear_iterations, atol = fitter.linear_tol, btol = fitter.linear_tol)
            sol = Krylov.solution(ws)
            @. delta = sol / colnorm_flat
            _cap_position_step!(delta, max_step, p, plan.k_y, plan.k_x)
            @. theta_cand = theta + delta
            t_solve += time() - t0
            t0 = time()
            _render_model!(cand, psf, plan.free_names_val, plan.fixed, theta_cand, p, model_R,
                geom.anchor_y, geom.anchor_x, ny, nx, live, render_buf, render_scratch)
            t_render += time() - t0
            cost_cand = _cost!(cand, data, w, union_pix)

            n_trials += 1
            ok = cost_cand < cost
            show_trace && _trace_fit_step(pass, lin, trial, cost, cost_cand, lambda,
                ok ? "accepted" : "rejected")
            if ok
                # The candidate arrays become the current ones.  `model` is now
                # the render at `theta`, by construction rather than by flag.
                theta, theta_cand = theta_cand, theta
                model, cand = cand, model
                cost = cost_cand
                lambda = max(lambda / fitter.lambda_down, fitter.lambda_min)
                accepted = true
                n_accepted += 1
                break
            end
            lambda = min(lambda * fitter.lambda_up, fitter.lambda_max)
            trial == max_trials &&
                @warn "max_damping_trials reached without an accepted step" pass lin λ = lambda
        end
        accepted || break
    end

    # Refill at the returned `theta`, not the one the last linearization started
    # from to refresh `colnorm`, which is what `catalog_from_theta` reads for `flux_snr`
    # which the driver then prunes on, so without this the pruning cut would act on an SNR one
    # accepted step stale. 26-09-18: Stamp filling is expensive and colnorm being one step out
    # of date does not significantly alter the pruning behavior.  Measured on 1000x1000 with
    # 5850 sources: the refill costs ~4.8 ms against ~73 ms of `t_fit` per pass (~7%), and
    # leaves `flux_snr` at most 6e-5 relative from its refilled value -- `flux` is read fresh
    # from `theta`, and the stale factor `sqrt(H_ff)` is a unit-flux render norm, nearly
    # translation-invariant over the <=1 px step.  Over a full 5-pass run that still tips the
    # occasional source sitting on the `prune_snr_min` cut (4 of 5850 differ in membership,
    # fitted positions by <0.04 px), which is the arbitrary end of the cut.  Not measured for
    # hard-masked weights or a `GriddedPSFModel`, where `H_ff` varies less smoothly with
    # position; restore this if morphology or pruning there looks unstable.
    # if n_accepted > 0
    #     t0 = time()
    #     _fill_stamps!(stamp, psf, plan.free_names_val, plan.fixed, theta, w, plan.grad_col,
    #         geom.dy_off, geom.dx_off, geom.anchor_y, geom.anchor_x,
    #         plan.row_y, plan.row_x, plan.row_flux, live, fill_scratch)
    #     t_stamps += time() - t0
    # end

    fit = (; theta, model = reshape(model, ny, nx), cost, dof, stamp, model_R, geom, data, w,
             render_buf, render_scratch, fill_scratch)
    stats = (; n_lin, n_trials, n_accepted, cost_start, cost_end = cost, lambda_start,
               lambda_end = lambda,
               fit_timing = (; setup = t_setup, stamps = t_stamps, render = t_render,
                               solve = t_solve))
    return (; fit..., catalog = catalog_from_theta(catalog, fit, plan), state = lambda, stats)
end

"""
    source_errors(fitter::SimultaneousFitter, fit, psf, plan, cov_est) -> NamedTuple

Per-source errors from the normal matrix at the final `theta`, each marginalized
over up to `fitter.error_neighbors` blended neighbors (see `_source_errors!`),
with the global `cost / dof` for estimators that rescale by it.

The stamps are refilled at `fit.theta` first, for two reasons.
`freeze_positions` may have zeroed the position columns during the fit, and the
reported position errors must not inherit that.  The stamp refill also allows the
errors to reflect the parameters the final pass returned, since `fit_pass` leaves `stamp`
at the last linearization's, one accepted step behind.
"""
function source_errors(fitter::SimultaneousFitter, fit, psf, plan::FitPlan, cov_est)
    FT = eltype(fit.theta)
    n_src = length(fit.catalog)
    p = plan.p
    stamp = fit.stamp
    _fill_stamps!(stamp, psf, plan.free_names_val, plan.fixed, fit.theta, fit.w,
        plan.grad_col, fit.geom.dy_off, fit.geom.dx_off, fit.geom.anchor_y, fit.geom.anchor_x,
        plan.row_y, plan.row_x, plan.row_flux, trues(n_src), fit.fill_scratch)
    y_err = zeros(FT, n_src)
    x_err = zeros(FT, n_src)
    flux_err = zeros(FT, n_src)
    errs = Matrix{FT}(undef, p, n_src)
    _source_errors!(errs, stamp, fit.geom, cov_est, fit.cost, fit.dof, fitter.error_neighbors)
    for j in 1:n_src, k in 1:p
        plan.grad_col[k] == 1 ? (y_err[j] = errs[k, j]) :
            plan.grad_col[k] == 2 ? (x_err[j] = errs[k, j]) : (flux_err[j] = errs[k, j])
    end
    return (; y_err, x_err, flux_err, bkg_err = zeros(FT, n_src))
end


# ==============================================================================
# Public entry point
# ==============================================================================

"""
    fit_all_stars_simultaneous_multipass(image, psf, [sources], fit_rad; kws...)

Iterated background estimation, source detection, and simultaneous PSF-fitting
photometry.

$(_MULTIPASS_DOC_LOOP)
The fit re-optimizes the whole catalog simultaneously with damped
Gauss-Newton / Levenberg-Marquardt steps.  Each step's linear subproblem is
solved with a method from Krylov.jl (LSQR by default) on the weighted,
column-equilibrated Jacobian `J`, applied matrix-free straight from the
per-source stamp derivatives -- `J` is never materialized and no explicit normal
matrix is built.  [`fit_all_stars_multipass`](@ref) runs the same pipeline with
a sequential, one-source-at-a-time fitter instead.

$(_MULTIPASS_DOC_ARGUMENTS)
# Keyword arguments

$(_MULTIPASS_DOC_PIPELINE)
## Fitting

$(_MULTIPASS_DOC_FIT_COMMON)
## Simultaneous fitter

- `fixed::NamedTuple = (;)`: parameters frozen for all sources.  `bkg` is always
  fixed to zero (this function subtracts its own background model; a free
  per-source pedestal on top of that is double-counting) and does not need to be
  listed.  Only `(y, x, flux)` may be free.
- `linearizations_per_pass::Integer = 1`: LM linearizations per detection pass.
  crowdsource does exactly one, which is the default here.  Values > 1 trade detection
  passes for fit accuracy within a pass.
- `linear_iterations::Integer = 10`: iteration cap for the inner Krylov solve
  per damping trial.  Kept small on purpose; see the note below.
- `linear_tol::Real = 1.0e-4`: `atol`/`btol` for the linear solve.
  Each linearization is an approximation of a nonlinear objective, so
  solving it precisely is a waste of time.
- `solver::Symbol = :lsqr`: the Krylov.jl method used for the linearized
  subproblem, `:lsqr` or `:lsmr`.
- `max_step::Real = 1.0`: per-source position step cap in pixels, applied to
  every damping trial.  Too large risks overshooting the minimum in a single
  step; too small slows convergence.
- `λ_init::Real = 1.0e-3`: initial damping factor. Because λ is floored
  rather than reset here, it is pinned near this value for most of a run instead
  of decaying, so it should be set for the ill-conditioned case rather than as
  a starting guess that adapts away.
- `max_damping_trials::Integer = 8`: max damping retries per linearization.
- `λ_up::Real = 10.0`: damping increase factor on a failed trial.
- `λ_down::Real = 10.0`: damping decrease factor on a successful trial.
- `λ_min::Real = 1.0e-12`, `λ_max::Real = 1.0e12`: damping bounds.
- `error_neighbors::Integer = 8`: the most overlapping neighbors each source's
  reported errors are marginalized over.  Each source's errors come from a small
  problem made of itself and its `error_neighbors` most strongly coupled
  neighbors (defined as having overlapping fitting regions based on `fit_rad`),
  so they account for the flux and position covariances with blended
  neighbors.  Only each source's `CrowdPhot.MAX_NEIGHBORS_ERR` (32) nearest
  overlapping neighbors are considered, which bounds the cost when using large
  `fit_rad` in a crowded field, and `error_neighbors` may not exceed it.
  The errors approach those of the full linearized covariance
  `inv(J' J)` as `error_neighbors` grows, but large values are typically not
  necessary; `8` is within 1% for 99% of sources even
  with ~16 overlapping neighbors per source.  `0` ignores neighbors entirely,
  which underestimates errors of blended sources, relative to that full
  covariance, by up to an order of magnitude.

In `pass_history`, `n_lin` counts linearizations, `n_trials` damping trials and
`n_accepted` accepted trials; `lambda_start` / `lambda_end` bracket the
damping factor.

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

$(_MULTIPASS_DOC_RETURNS)"""
function fit_all_stars_simultaneous_multipass(
        image::AbstractMatrix{T},
        psf::AbstractPSFModel,
        sources,
        fit_rad::Real;
        solver::Symbol = :lsqr,
        linear_iterations::Integer = 10,
        linear_tol::Real = 1.0e-4,
        max_damping_trials::Integer = 8,
        linearizations_per_pass::Integer = 1,
        λ_init::Real = 1.0e-3,
        λ_up::Real = 10.0,
        λ_down::Real = 10.0,
        λ_min::Real = 1.0e-12,
        λ_max::Real = 1.0e12,
        error_neighbors::Integer = 8,
        kws...,
    ) where {T}
    FT = float(T)
    linearizations_per_pass > 0 || throw(ArgumentError("linearizations_per_pass must be positive"))
    linear_iterations > 0 || throw(ArgumentError("linear_iterations must be positive"))
    linear_tol > 0 || throw(ArgumentError("linear_tol must be positive"))
    max_damping_trials > 0 || throw(ArgumentError("max_damping_trials must be positive"))
    0 <= error_neighbors <= MAX_NEIGHBORS_ERR ||
        throw(ArgumentError("error_neighbors must be between 0 and $MAX_NEIGHBORS_ERR, got $error_neighbors"))
    solver in (:lsqr, :lsmr) || throw(ArgumentError("solver must be :lsqr or :lsmr, got $(repr(solver))"))
    fitter = SimultaneousFitter{FT}(solver, Int(linearizations_per_pass), Int(linear_iterations),
        linear_tol, Int(max_damping_trials), λ_init, λ_up, λ_down, λ_min, λ_max, Int(error_neighbors))
    return _fit_all_stars_multipass(fitter, image, psf, sources, fit_rad; kws...)
end

"""
    fit_all_stars_simultaneous_multipass(image, psf, fit_rad; kws...)

Convenience method starting from an empty catalog: the first detection pass
builds it from scratch.  Equivalent to passing `sources = nothing`.
"""
fit_all_stars_simultaneous_multipass(image::AbstractMatrix, psf::AbstractPSFModel, fit_rad::Real; kws...) =
    fit_all_stars_simultaneous_multipass(image, psf, nothing, fit_rad; kws...)
