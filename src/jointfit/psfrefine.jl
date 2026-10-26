# PSF refinement in the joint loop.
#
# After a pass's fit and prune, a joint MAP step updates each PSF group's ePSF node samples
# (images whose PSFs are identical form a group, as `device_psf` shares their nodes) inside a
# window about the origin, together with the selected stars' (u, v, band flux).  The model is
# linear in the samples; the prior is the reference ePSF with a hierarchical fractional Gaussian
# (a correction common to the group's nodes plus each node's own); the gauge fixes the sum and
# first moments of every phase sub-lattice of the window; the likelihood is Huber with
# gross-outlier rejection, weights taken at the warm start.  (`gauge = :derivative` fixes, in place of
# the first moments, the projections onto the reference's shift modes.)  The step is solved by projected
# preconditioned CG with all star couplings, the gauge applied in the preconditioner's metric.
# Design and validation:
# experiments/multi-image/psf-refinement/report/psf-refinement-report.tex.
#
# Stamp space.  For integer oversampling s, a star anchored at pixel (ay, ax) reads, for its stamp
# pixel (ay + di, ax + dj), the node rows s di + L0y - 2 + a (a = 1:4, Catmull-Rom) and columns
# s dj + L0x - 2 + b, where L0 = floor(s (anchor - position) + origin) and the stencil weights
# depend only on the star's sub-pixel phase.  Stars sharing (L0y, L0x), the stamp half-width and
# the bilinear node cell (a batch) share that index map; per star only the 64 coefficients
# A = f w_m cv_a cu_b (flux, corner weight, stencil weights) differ.  The PSF part of J is, per
# batch, A against the batch's gathered samples (forward) and an outer-product accumulation of A
# against the stars' stamp residuals (adjoint), then fixed gathers.  No kernel needs atomics: the
# adjoint writes one partial per (batch, chunk of stars), and each window sample sums the partials
# that land on it through a list built with the problem.

"""
    PSFRefineOptions(; kwargs...)

PSF refinement in [`fit_joint_multipass`](@ref) (`JointLoopOptions(psf_refine = PSFRefineOptions())`):
after the fit and prune of every pass in `passes`, a joint MAP step on the ePSF node samples of
each PSF group (the images whose PSFs are identical) within `window` native pixels of the
origin, and on the selected stars' positions and band fluxes, whose updates are kept.  The
images' PSFs must be `ImagePSF`s or `GriddedPSFModel`s of them; the run's result carries the
refined PSFs (`psfs`).  Not supported by `fit_joint_tiled`.

- `passes = 1:typemax(Int)`
- `window = nothing`: half-width (native px) of the refinement box, by default the fit's `R_fit`.
  Each star contributes its pixels within `min(window, model radius)`.  A window beyond `R_fit` is logged
  as a warning: its samples outside the fit stamp are constrained only by this step and its prior,
  which can lead to the fit and the PSF refinement diverging when running many passes
- `snr_min = 20`, `max_stars = 100_000`: stars with band S/N >= `snr_min` constrain their band's
  PSFs; beyond `max_stars` per band (one list for all of the band's PSF groups), a fixed random
  subset
- `eps = 0.1`, `eps_node = 0.05`, `tau = 1e-3`: prior on the correction of each sample of the reference
  PSF `P0` (its node samples): a part common to the group's nodes with scale
  `eps mean|P0| + tau max P0` and one per node with scale `eps_node |P0| + tau max P0`
- `c = 2`, `mu = 0.02`, `reject_z = 20`, `irls = 1`: Huber weights (`c`) of
  `z = residual / sqrt(1/W + (mu model)²)`, zero beyond `|z| = reject_z`; computed at the
  warm start, then `irls >= 1` reweightings against the refined model
- `cg_iterations = 10`, `cg_tol = 1e-6`
- `max_step = 1`: cap on a star's position update (frame px)
- `reserve_frac = 0`, `reserve_seed = 1`: the fraction of sources, drawn at detection (seeded,
  in detection order), held out as reserved stars: they stay in the step with their own
  positions and fluxes free but never constrain the PSF, for validation (`psf_stars`).
  Recommended 0.1-0.2 when validating on data without truth: the reserved stars' χ²/dof in the
  run history is a held-out check of the PSF.  At 0.15 the catalog changes were within the
  scatter between settings on test data
- `gauge = :moment`: what the correction leaves unchanged on every phase sub-lattice of the window,
  (the sum is always preserved). `:moment`: the first moments (the window's center of light). `:derivative`:
  the projections onto the reference's shift modes ∂P0/∂x, ∂P0/∂y (of the group's mean
  reference), so a least-squares fit with uniform weights places a star where it placed it on the
  reference PSF (to first order) -- this ensures positions keep the reference's centroid convention, as an
  external WCS fitted with that PSF expects.  On simulations the moment gauge recovered the true
  PSF shape better (window mismatch ‖P - P_true‖ / ‖P0 - P_true‖ for the refined PSF `P`, 0 exact
  and 1 no better than the reference: 0.13-0.17 vs 0.44-0.52 with the exact catalog,
  psf-refinement report): a real PSF
  difference generally has a component along the shift modes, which this gauge forbids.
  `:derivative` is a diagnostic for alignment / WCS offsets (whether positions move because the
  refinement shifts the PSF's centroid convention or because of the data), not for production fits
- `max_masked = 0.25`: a star whose stamps' in-image pixels are more than this fraction masked
  (zero weight; e.g. on a saturated star's flagged core) does not constrain the PSF (as a rejected
  star): its band flux rests on the few pixels left and can be far off
- `star_nsigma = nothing`, `star_max_remove = 0.01`: with a number, the star-level outlier test
  after the first IRLS round: a fitting star whose χ² over its usable pixels (robust scale, pixels
  already rejected excluded) exceeds the `star_nsigma` point of the χ² distribution for its
  pixels, times max(1, the median χ²/dof of the fitting stars), stops constraining the PSF in the
  later rounds (as a reserved star); at most a fraction `star_max_remove` of them, the worst
  first.  Re-tested from scratch in every refinement.  `star_nsigma` recommended off: no threshold
  improved the held-out χ² or the flux bias on test data; if used, 5.5-8.  `star_max_remove`
  recommended 0.01: larger
  caps let the first refinements, while the reference PSF is still wrong, remove the stars the
  PSF needs; on test data that gave 1-8% more fragments at 0.05 and 12-20% more uncapped, even
  though the stars re-enter later
"""
Base.@kwdef struct PSFRefineOptions
    passes::UnitRange{Int} = 1:typemax(Int)
    window::Union{Nothing, Int} = nothing
    snr_min::Float64 = 20.0
    max_stars::Int = 100_000
    eps::Float64 = 0.1
    eps_node::Float64 = 0.05
    tau::Float64 = 1e-3
    c::Float64 = 2.0
    mu::Float64 = 0.02
    reject_z::Float64 = 20.0
    irls::Int = 1
    cg_iterations::Int = 10
    cg_tol::Float64 = 1e-6
    max_step::Float64 = 1.0
    reserve_frac::Float64 = 0.0
    reserve_seed::Int = 1
    star_nsigma::Union{Nothing, Float64} = nothing
    star_max_remove::Float64 = 0.01
    max_masked::Float64 = 0.25
    gauge::Symbol = :moment
end

function _check_refine_options(o::PSFRefineOptions)
    o.irls >= 1 || throw(ArgumentError("psf_refine needs irls >= 1: the later rounds reweight against the refined PSF, and the star-level test runs between rounds"))
    0 <= o.reserve_frac < 1 || throw(ArgumentError("psf_refine reserve_frac must be in [0, 1)"))
    0 <= o.star_max_remove <= 1 || throw(ArgumentError("psf_refine star_max_remove must be in [0, 1]"))
    0 <= o.max_masked <= 1 || throw(ArgumentError("psf_refine max_masked must be in [0, 1]"))
    o.gauge in (:moment, :derivative) || throw(ArgumentError("psf_refine gauge must be :moment or :derivative, got :$(o.gauge)"))
    (o.star_nsigma === nothing || o.star_nsigma > 0) || throw(ArgumentError("psf_refine star_nsigma must be positive or nothing"))
    return o
end

# ------------------------------------------------------------------------------------------
# Run-level setup: groups, windows, prior blocks, gauge basis
# ------------------------------------------------------------------------------------------

"""
Per PSF group, fixed for the run: images, band, device node indices `nodes` (local node `n` is
`nodes[n]`), the refined half-width `window` (native px), window geometry (node rows `r0 : r0 + nw - 1`, columns `c0 : c0 + nw - 1`), the
reference window values `ref` (`nw x nw x N`, Float64), the prior precision blocks `Pk`
(`N x N x nw²`), the gauge basis `gq` (`3 x nw²`: each sample's orthonormal vectors of its
phase sub-lattice) with the sub-lattices' first rows `ra` and columns `ca` and each window row's and
column's phase index `rph` / `cph`, and the host model `hpsf` the refinement updates.
"""
function _refine_group_setup(sc, g, imgs, band, nodes, window, o::PSFRefineOptions)
    T = eltype(sc); be = sc.backend
    n1 = g.psfs[1]
    sx, sy = n1.oversampling
    all(n -> n.oversampling == n1.oversampling && n.origin == n1.origin && size(n.data) == size(n1.data), g.psfs) ||
        throw(ArgumentError("psf_refine needs every node of a PSF to share its origin, oversampling and size"))
    hw = max(sx, sy) * window + 2
    oyf, oxf = Float64(n1.origin.y), Float64(n1.origin.x)
    r0 = round(Int, oyf) - hw; c0 = round(Int, oxf) - hw; nw = 2hw + 1
    nd1, nd2 = size(n1.data)
    (r0 >= 1 && c0 >= 1 && r0 + nw - 1 <= nd1 && c0 + nw - 1 <= nd2) ||
        throw(ArgumentError("psf_refine window ($window px) exceeds the PSF's $(nd1)x$(nd2) node data"))
    N = length(g.psfs); ns = nw * nw
    ref = [Float64(nd.data[r0 + r - 1, c0 + c - 1]) for r in 1:nw, c in 1:nw, nd in g.psfs]
    τ = o.tau * maximum(maximum(Float64, nd.data) for nd in g.psfs)
    Pk = Array{Float64}(undef, N, N, ns)
    for c in 1:nw, r in 1:nw
        p0 = [abs(ref[r, c, n]) for n in 1:N]
        C = fill((o.eps * sum(p0) / N + τ)^2, N, N)
        N > 1 && (C .+= Diagonal((o.eps_node .* p0 .+ τ) .^ 2))
        Pk[:, :, (c - 1) * nw + r] .= inv(C)
    end
    # sum and first moments (or shift-mode projections) of each phase sub-lattice (samples whose offsets
    # from the origin agree mod s)
    gq = zeros(3, ns); ra = zeros(Int32, sy); ca = zeros(Int32, sx)
    phy(r) = mod(round(Int, r0 + r - 1 - oyf), sy); phx(c) = mod(round(Int, c0 + c - 1 - oxf), sx)
    # shift modes: central differences of the mean reference, the Catmull-Rom interpolant's slopes at the nodes
    P0 = o.gauge === :derivative ? sum(nd -> Float64.(nd.data), g.psfs) / N : nothing
    P0at(r, c) = P0[clamp(r0 + r - 1, 1, nd1), clamp(c0 + c - 1, 1, nd2)]
    mode = o.gauge === :derivative ?
        (s -> (r = (s - 1) % nw + 1; c = (s - 1) ÷ nw + 1; [P0at(r, c + 1) - P0at(r, c - 1), P0at(r + 1, c) - P0at(r - 1, c)])) :
        (s -> [(c0 + (s - 1) ÷ nw) - oxf, (r0 + (s - 1) % nw) - oyf])
    for a in 0:(sy - 1), b in 0:(sx - 1)
        idx = [(c - 1) * nw + r for c in 1:nw for r in 1:nw if phy(r) == a && phx(c) == b]
        Cm = reduce(hcat, [[1.0; mode(s)] for s in idx])'
        F = svd(Cm); rk = count(>(1e-10 * F.S[1]), F.S)
        gq[1:rk, idx] .= permutedims(F.U[:, 1:rk])
    end
    for r in 1:nw; ra[phy(r) + 1] == 0 && (ra[phy(r) + 1] = r); end
    for c in 1:nw; ca[phx(c) + 1] == 0 && (ca[phx(c) + 1] = c); end
    hpsf = GriddedPSFModel([ImagePSF(copy(nd.data); origin = nd.origin, oversampling = nd.oversampling, fill_value = nd.fill_value) for nd in g.psfs],
                           g.grid_y, g.grid_x)
    return (; images = imgs, band, nodes, window, N, sy, sx, r0, c0, nw, ref, Pk, gq = to_device(be, T, gq), ra = to_device(be, ra),
              ca = to_device(be, ca), rph = to_device(be, Int32[phy(r) + 1 for r in 1:nw]), cph = to_device(be, Int32[phx(c) + 1 for c in 1:nw]),
              hpsf, xcur = zeros(N * ns))
end

"Groups of the run's PSFs (the device's node sharing), each set up by `_refine_group_setup` with the window (`R_fit` by default)."
function _refine_setup(sc::JointScene, psfs, o::PSFRefineOptions, R_fit)
    _check_refine_options(o)
    sc.dev.psf isa DevGridded || throw(ArgumentError("psf_refine needs ImagePSF or GriddedPSFModel PSFs (not CircularGaussianPRF)"))
    window = something(o.window, R_fit)
    window > R_fit && @warn "psf_refine window ($window px) exceeds R_fit ($R_fit px): the PSF samples outside the fit stamp are constrained only by the refinement and its prior, and can diverge over the passes"
    gs = map(_as_gridded, psfs)
    units = [unit_psf(g, 0, 0) for g in gs]
    reps, slot = _groups((r, e) -> _same_content(units[r], units[e]), length(gs))   # as `device_psf`
    base = cumsum([0; [length(gs[r].psfs) for r in reps]])
    bands = Array(sc.dev.band)
    return map(enumerate(reps)) do (i, r)
        imgs = findall(==(i), slot)
        all(e -> bands[e] == bands[r], imgs) || throw(ArgumentError("psf_refine: images sharing a PSF must share a band"))
        _refine_group_setup(sc, gs[r], imgs, Int(bands[r]), Int32.(base[i] .+ (1:length(gs[r].psfs))), window, o)
    end
end

# ------------------------------------------------------------------------------------------
# Kernels.  Each is one work item per output element; nothing is written twice.
# ------------------------------------------------------------------------------------------

# Per entry (star, image): pixel position, d(y, x)/d(u, v), node corners (the batch key), the stencil
# base L0 and the sub-pixel fractions, as the fitter's model places the star.
@kernel function rf_entry_kernel!(pos, jac, ks, L, ph, @Const(eimg), @Const(estar), @Const(eay), @Const(eax), @Const(θ), d)
    k = @index(Global)
    T = eltype(pos)
    e = eimg[k]; s = estar[k]
    _, y, x, ya, yb, xa, xb = source_pixel(d, e, θ[1, s], θ[2, s])
    kk, _, _, _ = psf_context(d.psf, e, y, x)
    par = d.psf.par
    v0 = par[4, kk[1]] * (T(eay[k]) - y) + par[2, kk[1]]
    u0 = par[3, kk[1]] * (T(eax[k]) - x) + par[1, kk[1]]
    fy = floor(v0); fx = floor(u0)
    pos[1, k] = y; pos[2, k] = x
    jac[1, k] = ya; jac[2, k] = xa; jac[3, k] = yb; jac[4, k] = xb
    for m in 1:4
        ks[m, k] = kk[m]
    end
    L[1, k] = unsafe_trunc(Int32, fy); L[2, k] = unsafe_trunc(Int32, fx)
    ph[1, k] = v0 - fy; ph[2, k] = u0 - fx
end

# Per entry: its 64 coefficients A[(stencil cell, corner), k] and, per stamp pixel with a row, the
# star-parameter derivatives S = (d/du, d/dv, d/dflux) of its model at the current nodes.
@kernel function rf_stamp_kernel!(A, S, @Const(pos), @Const(jac), @Const(ph), @Const(eay), @Const(eax), @Const(eimg), @Const(estar),
                                  @Const(ebatch), @Const(soff), @Const(bP), @Const(srow), @Const(θ), d)
    k = @index(Global)
    T = eltype(A)
    e = eimg[k]; s = estar[k]
    fs = d.ip[IP_SCALE, e]; f = fs * θ[2 + d.band[e], s]
    cv = cr_weights(ph[1, k]); cu = cr_weights(ph[2, k])
    y = pos[1, k]; x = pos[2, k]; yu = jac[1, k]; xu = jac[2, k]; yv = jac[3, k]; xv = jac[4, k]
    # node corners and weights, as rf_entry_kernel! placed them, with the weights' derivatives (which
    # d/du and d/dv include, as psf_value_grad does): recomputed here, cheaper than carrying them
    kk, w, dwy, dwx = psf_context(d.psf, e, y, x)
    for bb in 1:4, a in 1:4, m in 1:4
        A[((bb - 1) * 4 + a - 1) * 4 + m, k] = f * w[m] * cv[a] * cu[bb]
    end
    P = bP[ebatch[k]]; R = (P - Int32(1)) ÷ Int32(2); o = soff[k]
    for p in 1:(P * P)
        f0 = o + p
        if srow[f0] == 0
            S[1, f0] = zero(T); S[2, f0] = zero(T); S[3, f0] = zero(T)
            continue
        end
        i = eay[k] + Int32((p - 1) % P) - R; j = eax[k] + Int32((p - 1) ÷ P) - R
        val = zero(T); dv = zero(T); du = zero(T); wy = zero(T); wx = zero(T); sx = one(T); sy = one(T)
        for m in 1:4
            iszero(w[m]) && continue
            vm, dvm, dum, sxm, sym = node_value_grad(d.psf.nodes, d.psf.par, kk[m], i, j, y, x)
            val += w[m] * vm; dv += w[m] * dvm; du += w[m] * dum; wy += dwy[m] * vm; wx += dwx[m] * vm; sx = sxm; sy = sym
        end
        Py = f * (wy - sy * dv); Px = f * (wx - sx * du)
        S[1, f0] = Py * yu + Px * xu; S[2, f0] = Py * yv + Px * xv; S[3, f0] = fs * val
    end
end

# Per row (pixel): fit residual data - background - model, weight and model.
@kernel function rf_rows_kernel!(r, W, m, @Const(rowpix), @Const(data), @Const(bkg), @Const(model), @Const(w))
    q = @index(Global)
    i = rowpix[q]
    r[q] = (data[i] - bkg[i]) - model[i]; W[q] = w[i]; m[q] = model[i]
end

# Window sample (r, c) of node n read by batch b's (stencil cell, corner) i at stamp pixel p.
@inline function rf_sample(i, p, P, L0y, L0x, r0, c0, sy, sx)
    m = (i - 1) % 4 + 1; ab = (i - 1) ÷ 4 + 1; a = (ab - 1) % 4 + 1; bb = (ab - 1) ÷ 4 + 1
    R = (P - 1) ÷ 2; di = (p - 1) % P - R; dj = (p - 1) ÷ P - R
    return m, sy * di + L0y - 2 + a - r0 + 1, sx * dj + L0x - 2 + bb - c0 + 1
end

# Forward, step 1: each batch's samples Xb[(cell, corner), p] (zero outside the window).
@kernel function rf_xgather_kernel!(Xb, @Const(x), @Const(bL), @Const(bP), @Const(bks), @Const(xoff), M, r0, c0, nw, sy, sx)
    t = @index(Global)
    b = (t - 1) ÷ M + 1; o = (t - 1) % M + 1
    P = bP[b]
    if o <= 64 * P * P
        i = (o - 1) % 64 + 1; p = (o - 1) ÷ 64 + 1
        m, r, c = rf_sample(i, p, P, bL[1, b], bL[2, b], r0, c0, sy, sx)
        n = bks[m, b]
        v = zero(eltype(Xb))
        if (1 <= r <= nw) & (1 <= c <= nw)
            v = x[(n - 1) * nw * nw + (c - 1) * nw + r]
        end
        Xb[xoff[b] + o] = v
    end
end

# Forward, step 2: each stamp pixel's value A[:, k]ᵀ Xb[:, p] (PSF part) plus S ⋅ (du, dv, df) (star
# part).  One work item per quad of an entry's stamp pixels (fq: each quad's first stamp pixel), so
# each load of A[i, k] serves four pixels (1.5x one pixel per item on the CPU, equal on Metal); the
# entry's last 1-3 pixels one at a time.
@kernel function rf_fwd_kernel!(V, @Const(A), @Const(Xb), @Const(S), @Const(ystar), @Const(fq), @Const(fent), @Const(soff),
                                @Const(ebatch), @Const(eloc), @Const(xoff), @Const(bP), withx::Bool, withy::Bool)
    q = @index(Global)
    T = eltype(V)
    f = fq[q]; k = fent[f]; p = f - soff[k]
    b = ebatch[k]; PP = bP[b] * bP[b]; s = eloc[k]; b1 = xoff[b] + (p - 1) * 64
    y1 = withy ? ystar[1, s] : zero(T); y2 = withy ? ystar[2, s] : zero(T); y3 = withy ? ystar[3, s] : zero(T)
    # reassociating the sums (SIMD on the CPU, ILP on GPUs) is 2.5x faster than one serial chain;
    # bounds checks would block the CPU vectorizer
    @inbounds if p + 3 <= PP
        c1 = c2 = c3 = c4 = zero(T)
        if withx
            @simd for i in 1:64
                a = A[i, k]
                c1 += Xb[b1 + i] * a; c2 += Xb[b1 + 64 + i] * a; c3 += Xb[b1 + 128 + i] * a; c4 += Xb[b1 + 192 + i] * a
            end
        end
        if withy
            c1 += S[1, f] * y1 + S[2, f] * y2 + S[3, f] * y3
            c2 += S[1, f + 1] * y1 + S[2, f + 1] * y2 + S[3, f + 1] * y3
            c3 += S[1, f + 2] * y1 + S[2, f + 2] * y2 + S[3, f + 2] * y3
            c4 += S[1, f + 3] * y1 + S[2, f + 3] * y2 + S[3, f + 3] * y3
        end
        V[f] = c1; V[f + 1] = c2; V[f + 2] = c3; V[f + 3] = c4
    else
        for j in 0:(PP - p)
            c = zero(T)
            if withx
                @simd for i in 1:64
                    c += Xb[b1 + 64j + i] * A[i, k]
                end
            end
            withy && (c += S[1, f + j] * y1 + S[2, f + j] * y2 + S[3, f + j] * y3)
            V[f + j] = c
        end
    end
end

# Forward, step 3: each row sums the stamp pixels on it (its stars).
@kernel function rf_rowsum_kernel!(out, @Const(V), @Const(rptr), @Const(rlist))
    q = @index(Global)
    acc = zero(eltype(out))
    for l in rptr[q]:(rptr[q + 1] - 1)
        acc += V[rlist[l]]
    end
    out[q] = acc
end

# Adjoint, step 1: each stamp pixel takes its row's value (zero without a row).
@kernel function rf_sgather_kernel!(Y, @Const(w), @Const(srow))
    f = @index(Global)
    q = srow[f]
    Y[f] = q > 0 ? w[q] : zero(eltype(Y))
end

# Adjoint, step 2: one workgroup per (batch, chunk of its stars).  Each item owns a register tile
# of IB consecutive cell-corners by PB consecutive stamp pixels and sums A[i, k] Y[p, k] (A² with
# `sq`) over the chunk's stars in order: per star IB loads of A (contiguous: SIMD on the CPU, vector
# loads on GPUs) and PB of Y for IB PB multiply-adds.  The chunk's last star reads up to PB - 1
# entries past its stamp (Y is padded by RF_YPAD; those products are dropped).
@inline rf_load_a(A, i0, k, sq, ::Val{IB}) where {IB} = ntuple(u -> (a = A[i0 + u, k]; sq ? a * a : a), Val(IB))
# (a function, not a closure over the loop's accumulators: those would be boxed)
@inline function rf_fma_tile(acc, a, Y, so, ::Val{IB}, ::Val{PB}) where {IB, PB}
    ntuple(Val(IB * PB)) do t
        u = (t - 1) % IB + 1; j = (t - 1) ÷ IB + 1
        acc[t] + a[u] * Y[so + j]
    end
end

@kernel function rf_adj_kernel!(part, @Const(A), @Const(Y), @Const(cb), @Const(ck0), @Const(ck1), @Const(cpoff), @Const(bP),
                                @Const(soff), sq::Bool, vi::Val{IB}, vp::Val{PB}) where {IB, PB}
    li = @index(Local); g = @index(Group)
    G = @groupsize()[1]
    b = cb[g]; P = bP[b]; PP = P * P; ni = 64 ÷ IB; no = ni * cld(PP, PB)
    k0 = ck0[g]; k1 = ck1[g]; base = cpoff[g]
    o = Int(li)
    @inbounds while o <= no
        i0 = ((o - 1) % ni) * IB; p0 = ((o - 1) ÷ ni) * PB
        acc = ntuple(_ -> zero(eltype(part)), Val(IB * PB))
        for k in k0:k1
            acc = rf_fma_tile(acc, rf_load_a(A, i0, k, sq, vi), Y, soff[k] + p0, vi, vp)
        end
        for t in 1:(IB * PB)
            u = (t - 1) % IB + 1; j = (t - 1) ÷ IB + 1
            p0 + j <= PP && (part[base + (p0 + j - 1) * 64 + i0 + u] = acc[t])
        end
        o += G
    end
end

# rf_adj_kernel!'s tile (cell-corners x stamp pixels): best of those tried on 18 CPU threads and on an
# M5 Max (2026-10-20, experiments/multi-image/psf-refinement/experiments/touchstone/cg_unify)
rf_adj_tile(be) = is_cpu(be) ? (Val(16), Val(4)) : (Val(4), Val(8))
const RF_YPAD = 8             # >= the largest tile's PB - 1

# Adjoint, step 3, at build time: each window sample (r, c) of node n lists the adjoint partials
# (chunk, output) that land on it, in a fixed order (the node's (batch, corner) list, stencil row,
# column, chunk): a count pass (`fill = false`, into cnt[t + 1]) and, after a scan, a fill pass.
@kernel function rf_xlist_kernel!(cnt, idx, @Const(ptr), @Const(nbptr), @Const(nbb), @Const(nbm), @Const(bL), @Const(bP), @Const(bc0),
                                  @Const(bc1), @Const(cpoff), r0, c0, nw, sy, sx, fill::Bool)
    t = @index(Global)
    n = (t - 1) ÷ (nw * nw) + 1; rc = (t - 1) % (nw * nw); c = rc ÷ nw + 1; r = rc % nw + 1
    ty0 = r0 + r - 1; tx0 = c0 + c - 1
    k = 0
    for l in nbptr[n]:(nbptr[n + 1] - 1)
        b = nbb[l]; m = nbm[l]; P = bP[b]; R = (P - 1) ÷ 2
        ty = ty0 - bL[1, b] + 2; tx = tx0 - bL[2, b] + 2
        # only stencil rows a ≡ ty (mod sy) land on this sample: one per pixel row when sy >= 4
        for a in (mod(ty - 1, sy) + 1):sy:4
            di = (ty - a) ÷ sy
            abs(di) <= R || continue
            for bb in (mod(tx - 1, sx) + 1):sx:4
                dj = (tx - bb) ÷ sx
                abs(dj) <= R || continue
                o = ((dj + R) * P + di + R) * 64 + ((bb - 1) * 4 + a - 1) * 4 + m
                for ch in bc0[b]:bc1[b]
                    fill && (idx[ptr[t] + k] = cpoff[ch] + o)
                    k += 1
                end
            end
        end
    end
    fill || (cnt[t + 1] = k)
end

# Adjoint, step 4: each window sample sums its listed partials (the chunks' sum and the scatter to
# the samples in one gather: 11x (CPU) and 3x (Metal) the two kernels it replaced).
@kernel function rf_xgather_adj_kernel!(gx, @Const(part), @Const(xptr), @Const(xidx))
    t = @index(Global)
    acc = zero(eltype(gx))
    for l in xptr[t]:(xptr[t + 1] - 1)
        acc += part[xidx[l]]
    end
    gx[t] = acc
end

# Adjoint, star part: each star sums S ⋅ Y over its entries' stamp pixels.
@kernel function rf_star_adj_kernel!(gy, @Const(S), @Const(Y), @Const(sptr), @Const(slist), @Const(soff), @Const(ebatch), @Const(bP))
    s = @index(Global)
    a1 = a2 = a3 = zero(eltype(gy))
    for l in sptr[s]:(sptr[s + 1] - 1)
        k = slist[l]; P = bP[ebatch[k]]; o = soff[k]
        for p in 1:(P * P)
            y = Y[o + p]
            a1 += S[1, o + p] * y; a2 += S[2, o + p] * y; a3 += S[3, o + p] * y
        end
    end
    gy[1, s] = a1; gy[2, s] = a2; gy[3, s] = a3
end

# Each star's own 3x3 block Σ W S Sᵀ and, for the preconditioner, a regularized inverse of it.
@kernel function rf_star_block_kernel!(H, Hi, @Const(S), @Const(Wst), @Const(sptr), @Const(slist), @Const(soff), @Const(ebatch), @Const(bP))
    s = @index(Global)
    T = eltype(H)
    h11 = h12 = h13 = h22 = h23 = h33 = zero(T)
    for l in sptr[s]:(sptr[s + 1] - 1)
        k = slist[l]; P = bP[ebatch[k]]; o = soff[k]
        for p in 1:(P * P)
            w = Wst[o + p]; a = S[1, o + p]; b = S[2, o + p]; c = S[3, o + p]
            h11 += w * a * a; h12 += w * a * b; h13 += w * a * c; h22 += w * b * b; h23 += w * b * c; h33 += w * c * c
        end
    end
    Hm = SMatrix{3, 3, T}(h11, h12, h13, h12, h22, h23, h13, h23, h33)
    # inverted Jacobi-scaled (unit diagonal) with a relative ridge λ: the plain cofactor inverse of
    # H + εI fails in Float32 for a star with 1-2 usable pixels (rank-deficient block: det rounds to 0,
    # NaN; or to garbage, an indefinite preconditioner) and for an extreme flux (cofactors overflow).
    # An all-rejected star (H = 0) still gets 1/ε.
    ε = T(1e-9); λ = T(1e-2)
    D = SVector{3, T}(inv(sqrt(h11 + ε)), inv(sqrt(h22 + ε)), inv(sqrt(h33 + ε)))
    DD = D * D'
    Hinv = DD .* inv(SMatrix{3, 3, T}(h11 + ε, h12, h13, h12, h22 + ε, h23, h13, h23, h33 + ε) .* DD + λ * one(SMatrix{3, 3, T})) .* (1 + λ)
    for q in 1:9
        H[q, s] = Hm[q]; Hi[q, s] = Hinv[q]
    end
end

# y[:, s] (+)= B[:, :, s] x[:, s] per window sample s across the group's N nodes (prior, preconditioner).
@kernel function rf_block_kernel!(y, @Const(x), @Const(B), N, ns, add::Bool)
    s = @index(Global)
    for n in 1:N
        acc = zero(eltype(y))
        for m in 1:N
            acc += B[n, m, s] * x[(m - 1) * ns + s]
        end
        y[(n - 1) * ns + s] = add ? y[(n - 1) * ns + s] + acc : acc
    end
end

# z[:, s] = Hi[:, :, s] r[:, s] per star.
@kernel function rf_star_prec_kernel!(z, @Const(r), @Const(Hi))
    s = @index(Global)
    a = r[1, s]; b = r[2, s]; c = r[3, s]
    z[1, s] = Hi[1, s] * a + Hi[4, s] * b + Hi[7, s] * c
    z[2, s] = Hi[2, s] * a + Hi[5, s] * b + Hi[8, s] * c
    z[3, s] = Hi[3, s] * a + Hi[6, s] * b + Hi[9, s] * c
end

# Remove from v the span of each (node, phase sub-lattice)'s constraint vectors.
@kernel function rf_proj_kernel!(v, @Const(gq), @Const(ra), @Const(ca), nw, sy, sx)
    t = @index(Global)
    nsub = sy * sx; n = (t - 1) ÷ nsub + 1; l = (t - 1) % nsub
    a = l % sy + 1; b = l ÷ sy + 1
    base = (n - 1) * nw * nw
    T = eltype(v)
    d1 = d2 = d3 = zero(T)
    for c in ca[b]:sx:nw, r in ra[a]:sy:nw
        s = (c - 1) * nw + r; val = v[base + s]
        d1 += gq[1, s] * val; d2 += gq[2, s] * val; d3 += gq[3, s] * val
    end
    for c in ca[b]:sx:nw, r in ra[a]:sy:nw
        s = (c - 1) * nw + r
        v[base + s] -= gq[1, s] * d1 + gq[2, s] * d2 + gq[3, s] * d3
    end
end

# The gauge in the preconditioner's metric.  With B the node-block preconditioner and C the gauge
# vectors (row q of `gq` on one node's phase sub-lattice), the preconditioned residual is
# z = B r - B Cᵀ (C B Cᵀ)⁻¹ C B r, which satisfies the gauge exactly; projecting B r afterwards
# (Euclidean) instead distorts the window-edge samples, where the moment vectors are largest, and
# left the preconditioned PSF block about 3x slower to converge on the test fields.  B mixes nodes
# only within a sample, so C B Cᵀ is block diagonal over the sub-lattices, one (3N x 3N) block each
# (rows (n, q)).

# G[(n, q), (n', q'), l] = Σ over sub-lattice l's samples s of gq[q, s] Bi[n, n', s] gq[q', s].
@kernel function rf_gauge_G_kernel!(G, @Const(Bi), @Const(gq), @Const(ra), @Const(ca), nw, sy, sx, N)
    t = @index(Global)
    K = 3N; i = (t - 1) % K + 1; j = ((t - 1) ÷ K) % K + 1; l = (t - 1) ÷ (K * K) + 1
    a = (l - 1) % sy + 1; b = (l - 1) ÷ sy + 1
    n = (i - 1) ÷ 3 + 1; q = (i - 1) % 3 + 1; n2 = (j - 1) ÷ 3 + 1; q2 = (j - 1) % 3 + 1
    acc = zero(eltype(G))
    for c in ca[b]:sx:nw, r in ra[a]:sy:nw
        s = (c - 1) * nw + r
        acc += gq[q, s] * Bi[n, n2, s] * gq[q2, s]
    end
    G[i, j, l] = acc
end

# cv[(n, q), l] = Σ over sub-lattice l of node n of gq[q, s] z[n, s].
@kernel function rf_gauge_c_kernel!(cv, @Const(z), @Const(gq), @Const(ra), @Const(ca), nw, sy, sx)
    t = @index(Global)
    nsub = sy * sx; n = (t - 1) ÷ nsub + 1; l = (t - 1) % nsub + 1
    a = (l - 1) % sy + 1; b = (l - 1) ÷ sy + 1
    base = (n - 1) * nw * nw
    T = eltype(cv)
    d1 = d2 = d3 = zero(T)
    for c in ca[b]:sx:nw, r in ra[a]:sy:nw
        s = (c - 1) * nw + r; val = z[base + s]
        d1 += gq[1, s] * val; d2 += gq[2, s] * val; d3 += gq[3, s] * val
    end
    cv[3n - 2, l] = d1; cv[3n - 1, l] = d2; cv[3n, l] = d3
end

# yv[:, l] = Gi[:, :, l] cv[:, l].
@kernel function rf_gauge_y_kernel!(yv, @Const(Gi), @Const(cv), K)
    t = @index(Global)
    i = (t - 1) % K + 1; l = (t - 1) ÷ K + 1
    acc = zero(eltype(yv))
    for j in 1:K
        acc += Gi[i, j, l] * cv[j, l]
    end
    yv[i, l] = acc
end

# z[n', s] -= Σ_n Bi[n', n, s] Σ_q gq[q, s] yv[(n, q), l(s)] per window sample s (`rph` / `cph`: the
# phase index of each window row / column, as `_refine_group_setup`).
@kernel function rf_gauge_corr_kernel!(z, @Const(Bi), @Const(gq), @Const(yv), @Const(rph), @Const(cph), nw, sy, N)
    s = @index(Global)
    r = (s - 1) % nw + 1; c = (s - 1) ÷ nw + 1
    l = (cph[c] - 1) * sy + rph[r]
    g1 = gq[1, s]; g2 = gq[2, s]; g3 = gq[3, s]
    for m in 1:N
        acc = zero(eltype(z))
        for n in 1:N
            acc += Bi[m, n, s] * (g1 * yv[3n - 2, l] + g2 * yv[3n - 1, l] + g3 * yv[3n, l])
        end
        z[(m - 1) * nw * nw + s] -= acc
    end
end

# Huber weight of each row's whitened residual, zero beyond `kz`.
@kernel function rf_robust_kernel!(ρ, @Const(e), @Const(W), @Const(m), c, μ, kz)
    q = @index(Global)
    T = eltype(ρ)
    z = abs(e[q] * sqrt(W[q] / (1 + W[q] * (μ * m[q])^2)))
    ρ[q] = ifelse(z > kz, zero(T), min(one(T), c / z))
end

# Zero the PSF coefficients of the entries of stars with smask = 0 (reserved or rejected): their
# pixels then constrain only their own positions and fluxes, not the PSF.
@kernel function rf_mask_kernel!(A, @Const(eloc), @Const(smask))
    k = @index(Global)
    if smask[eloc[k]] == 0
        for i in 1:64
            A[i, k] = zero(eltype(A))
        end
    end
end

# The selected stars' band fluxes.
@kernel function rf_fstar_kernel!(fstar, @Const(θ), @Const(stars), row)
    i = @index(Global)
    fstar[i] = θ[row, stars[i]]
end

# Per star, over its stamps' usable pixels not rejected (ρ > 0): the χ² of the whitened residual
# (the robust scale), the pixel count, and by integer radius from the anchor (0 : nbin - 1) the
# sums of residual / band flux and their pixel counts.
@kernel function rf_star_chi_kernel!(chi, cnt, prof, pcnt, @Const(e), @Const(W), @Const(m), @Const(ρ), @Const(srow), @Const(sptr),
                                     @Const(slist), @Const(soff), @Const(ebatch), @Const(bP), @Const(fstar), μ)
    s = @index(Global)
    T = eltype(chi); nbin = size(prof, 1)
    for b in 1:nbin
        prof[b, s] = zero(T); pcnt[b, s] = zero(eltype(pcnt))
    end
    c2 = zero(T); n = zero(eltype(cnt)); fi = one(T) / fstar[s]
    for l in sptr[s]:(sptr[s + 1] - 1)
        k = slist[l]; P = bP[ebatch[k]]; R = (P - 1) ÷ 2; o = soff[k]
        for p in 1:(P * P)
            q = srow[o + p]
            if (q > 0) && (ρ[q] > 0)
                z2 = e[q]^2 * W[q] / (1 + W[q] * (μ * m[q])^2)
                c2 += z2; n += one(eltype(cnt))
                di = (p - 1) % P - R; dj = (p - 1) ÷ P - R
                b = unsafe_trunc(Int, round(sqrt(T(di * di + dj * dj)))) + 1
                if b <= nbin
                    prof[b, s] += e[q] * fi; pcnt[b, s] += one(eltype(pcnt))
                end
            end
        end
    end
    chi[s] = c2; cnt[s] = n
end

# Whole-vector steps as kernels (Base broadcasts are single-threaded on the CPU and, on GPUs,
# compiled per array length): out = a x + b y, out = x .* y.
@kernel function rf_lincomb_kernel!(out, @Const(x), @Const(y), a, b)
    i = @index(Global)
    out[i] = a * x[i] + b * y[i]
end
@kernel function rf_mul_kernel!(out, @Const(x), @Const(y))
    i = @index(Global)
    out[i] = x[i] * y[i]
end

const RF_GROUP = 256

# Per-workgroup partial sums of x ⋅ y (with `cnt`, of x > 1/2), as `cost_kernel!`: on the CPU each
# item covers `step` consecutive elements, on a GPU items stride by the grid.
@kernel function rf_dot_kernel!(part, @Const(x), @Const(y), n, step, cpu::Bool, cnt::Bool)
    li = @index(Local); g = @index(Group); t = @index(Global)
    T = eltype(part)
    sh = @localmem eltype(part) (RF_GROUP,)
    acc = zero(T)
    i = cpu ? (t - 1) * step + 1 : t
    i1 = cpu ? min(t * step, n) : n
    while i <= i1
        acc += cnt ? ifelse(x[i] > T(0.5), one(T), zero(T)) : x[i] * y[i]
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

# The CG's scalars stay on the device: a host read per dot product left the GPU idle ~0.3 ms each
# (~40% of a Metal solve).  Slots of `sv`: rᵀz and its successor alternate in 1 and 2; then pᵀq,
# rᵀr, r0, the "still iterating" flag and the count of iterations taken.
const RF_PQ = 3; const RF_RR = 4; const RF_R0 = 5; const RF_ACT = 6; const RF_CNT = 7

# sv[k] = the sum of the n partials in one workgroup; `mode` then updates the CG state from it:
# 1 (k = rᵀr) the convergence test for the next iteration (once stopped, it stays stopped), 2 (k =
# rᵀr at the start) r0 and the first test, 3 (k = pᵀq) counts the iteration if still iterating.
@kernel function rf_ssum_kernel!(sv, k, @Const(part), n, mode, tol)
    li = @index(Local)
    # (no local T: the CPU backend drops variables across @synchronize)
    sh = @localmem eltype(sv) (RF_GROUP,)
    acc = zero(eltype(sv)); t = li
    while t <= n
        acc += part[t]; t += RF_GROUP
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
    if li == 1
        v = sh[1] + sh[2]
        sv[k] = v
        if mode == 1
            sv[RF_ACT] = ifelse((sv[RF_ACT] > zero(eltype(sv))) & (sqrt(v) > tol * sv[RF_R0]), one(eltype(sv)), zero(eltype(sv)))
        elseif mode == 2
            sv[RF_R0] = sqrt(v)
            sv[RF_ACT] = ifelse(sqrt(v) > tol * sqrt(v), one(eltype(sv)), zero(eltype(sv)))
        elseif mode == 3
            sv[RF_CNT] += sv[RF_ACT]
        end
    end
end

# z += α p, r -= α q (α = sv[a] / pᵀq while iterating, else 0) and the per-workgroup partials of
# the new rᵀr, with rf_dot_kernel!'s geometry.
@kernel function rf_cg_xr_kernel!(z, r, part, @Const(pk), @Const(q), @Const(sv), a, n, step, cpu::Bool)
    li = @index(Local); g = @index(Group); t = @index(Global)
    T = eltype(part)
    sh = @localmem eltype(part) (RF_GROUP,)
    α = ifelse(sv[RF_ACT] > zero(T), sv[a] / sv[RF_PQ], zero(T))
    acc = zero(T)
    i = cpu ? (t - 1) * step + 1 : t
    i1 = cpu ? min(t * step, n) : n
    while i <= i1
        z[i] += α * pk[i]
        v = r[i] - α * q[i]
        r[i] = v; acc += v * v
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

# p = z + (sv[b] / sv[a]) p while iterating (once stopped p no longer matters: α = 0).
@kernel function rf_cg_p_kernel!(pk, @Const(zz), @Const(sv), a, b)
    i = @index(Global)
    sv[RF_ACT] > zero(eltype(pk)) && (pk[i] = zz[i] + (sv[b] / sv[a]) * pk[i])
end

const RF_NMAX = 16            # node count up to which the preconditioner blocks are inverted on the device

# Per window sample: Bi = inv(diag(dJ) + prior block), Gauss-Jordan without pivoting (the block is
# symmetric positive definite).
@kernel function rf_precond_kernel!(Bi, @Const(dJ), @Const(Pk), N, ns)
    s = @index(Global)
    T = eltype(Bi)
    M = @private eltype(Bi) (RF_NMAX, RF_NMAX)
    V = @private eltype(Bi) (RF_NMAX, RF_NMAX)
    for j in 1:N, i in 1:N
        M[i, j] = Pk[i, j, s] + ifelse(i == j, dJ[(i - 1) * ns + s], zero(T))
        V[i, j] = ifelse(i == j, one(T), zero(T))
    end
    for k in 1:N
        piv = inv(M[k, k])
        for j in 1:N
            M[k, j] *= piv; V[k, j] *= piv
        end
        for i in 1:N
            i == k && continue
            f = M[i, k]
            for j in 1:N
                M[i, j] -= f * M[k, j]; V[i, j] -= f * V[k, j]
            end
        end
    end
    for j in 1:N, i in 1:N
        Bi[i, j, s] = V[i, j]
    end
end

# ------------------------------------------------------------------------------------------
# One group's step
# ------------------------------------------------------------------------------------------

const RF_CHUNK = 256          # stars per adjoint workgroup

"Band `b`'s selected stars (catalog indices), shared by its PSF groups: S/N >= snr_min, a fixed random subset beyond max_stars."
function _refine_select(snr, b, o::PSFRefineOptions)
    cand = [s for s in axes(snr, 2) if snr[b, s] >= o.snr_min]
    length(cand) > o.max_stars && (cand = Random.shuffle(Random.MersenneTwister(b), cand)[1:o.max_stars])
    sort!(cand; by = s -> -snr[b, s])
    return cand
end

# The rows (the stamps' usable pixels, in band-local pixel order) and the CSR lists between rows
# and stamp pixels, built on the device.  Usable pixels are marked and numbered by a scan; the
# atomics that count and place each row's stamp pixels leave the slot order run-dependent, so each
# row's segment is then insertion-sorted (a pixel lies in about one stamp): deterministic.

# The entry of each stamp pixel.
@kernel function rf_fent_kernel!(fent, @Const(soff), @Const(ebatch), @Const(bP))
    k = @index(Global)
    P = bP[ebatch[k]]; o = soff[k]
    for p in 1:(P * P)
        fent[o + p] = k % Int32
    end
end

# Each stamp pixel's band-local pixel (0 off the image or at zero weight) and a mark on that pixel
# (concurrent writes all store 1).
@kernel function rf_mark_kernel!(qf, mark, @Const(fent), @Const(soff), @Const(ebatch), @Const(bP), @Const(eimg), @Const(eay),
                                 @Const(eax), @Const(dims), @Const(loff), @Const(w))
    f = @index(Global)
    k = fent[f]; p = f - soff[k]; P = bP[ebatch[k]]; R = (P - 1) ÷ 2
    i = eay[k] + (p - 1) % P - R; j = eax[k] + (p - 1) ÷ P - R
    e = eimg[k]; ny = dims[1, e]
    q = zero(eltype(qf))
    if (1 <= i <= ny) & (1 <= j <= dims[2, e])
        l = (j - 1) * ny + i
        if w[dims[3, e] + l] > 0
            q = (loff[e] + l) % eltype(qf); mark[q] = one(eltype(mark))
        end
    end
    qf[f] = q
end

# Each stamp pixel's row from the scanned marks, each row's scene pixel (every stamp pixel of a row
# stores the same value) and the count of stamp pixels per row into rcnt[row + 1].
@kernel function rf_srow_kernel!(srow, rowpix, rcnt, @Const(qf), @Const(cs), @Const(fent), @Const(eimg), @Const(dims), @Const(loff))
    f = @index(Global)
    q = qf[f]; r = zero(eltype(srow))
    if q > 0
        r = cs[q]; e = eimg[fent[f]]
        rowpix[r] = dims[3, e] + q - loff[e]
        Atomix.@atomic rcnt[r + 1] += one(eltype(rcnt))
    end
    srow[f] = r
end

@kernel function rf_rlist_kernel!(rlist, rfill, @Const(srow), @Const(rptr))
    f = @index(Global)
    r = srow[f]
    if r > 0
        k = Atomix.@atomic rfill[r] += one(eltype(rfill))    # the post-increment value
        rlist[rptr[r] + k - 1] = f % Int32
    end
end

# The forward's quads: each entry's stamp pixels 1, 5, 9, ... from its quad offset qoff.
@kernel function rf_fquad_kernel!(fq, @Const(qoff), @Const(soff), @Const(ebatch), @Const(bP))
    k = @index(Global)
    P = bP[ebatch[k]]; o = soff[k]; q0 = qoff[k]
    for j in 1:cld(P * P, 4)
        fq[q0 + j] = (o + 4j - 3) % Int32
    end
end

"""
The group's rows and stamp-pixel tables on the device, for `n` entries with `nf` stamp pixels in
batch order (`dd`, the uploaded entry tables): `fent`, `srow`, `rowpix`, `rptr`, `rlist` and `nr`.
"""
function _refine_rows(sc, g, dd, n, nf)
    be = sc.backend
    loff = zeros(Int32, nimages(sc)); nl = 0
    for e in g.images
        loff[e] = nl; nl += prod(sc.dims[e])
    end
    loff = to_device(be, loff)
    fent = KA.allocate(be, Int32, nf); qf = KA.allocate(be, Int32, nf); mark = KA.zeros(be, Int32, nl)
    launch!(rf_fent_kernel!, be, n, fent, dd.soff, dd.ebatch, dd.bP)
    launch!(rf_mark_kernel!, be, nf, qf, mark, fent, dd.soff, dd.ebatch, dd.bP, dd.eimg, dd.eay, dd.eax, sc.dev.dims, loff, sc.weight)
    scan!(mark)
    nrh = zeros(Int32, 1); copyto!(nrh, 1, mark, nl, 1); nr = Int(nrh[1])
    srow = KA.allocate(be, Int32, nf); rowpix = KA.allocate(be, Int32, nr); rptr = KA.zeros(be, Int32, nr + 1)
    launch!(rf_srow_kernel!, be, nf, srow, rowpix, rptr, qf, mark, fent, dd.eimg, sc.dev.dims, loff)
    fill!(view(rptr, 1:1), one(Int32)); scan!(rptr)
    rlist = KA.zeros(be, Int32, nf); rfill = KA.zeros(be, Int32, nr)
    launch!(rf_rlist_kernel!, be, nf, rlist, rfill, srow, rptr)
    launch!(segment_sort_kernel!, be, nr, rlist, rptr)
    KA.synchronize(be)
    foreach(KA.unsafe_free!, (loff, qf, mark, rfill))
    return (; fent, srow, rowpix, rptr, rlist, nr)
end

# The window samples' lists of adjoint partials (rf_xlist_kernel!) for the uploaded tables dd.
function _rf_xlists(dd, g, be)
    nt = g.N * g.nw * g.nw
    xptr = KA.zeros(be, Int32, nt + 1)
    args = (dd.nbptr, dd.nbb, dd.nbm, dd.bL, dd.bP, dd.bc0, dd.bc1, dd.cpoff, g.r0, g.c0, g.nw, g.sy, g.sx)
    launch!(rf_xlist_kernel!, be, nt, xptr, xptr, xptr, args..., false)
    fill!(view(xptr, 1:1), one(Int32)); scan!(xptr)
    nh = zeros(Int32, 1); copyto!(nh, 1, xptr, nt + 1, 1)
    xidx = KA.allocate(be, Int32, Int(nh[1]) - 1)
    launch!(rf_xlist_kernel!, be, nt, xptr, xidx, xptr, args..., true)
    return (; xptr, xidx)
end

"""
The group's stamp-space problem at the current catalog (device arrays and the host tables the
kernels index): entries (star, image) sorted into batches, their stamps, the rows (unique pixels)
with residual, weight and model, the CSR lists between rows and stamp pixels and between stars
and entries, the adjoint's chunks and each window sample's list of adjoint partials.  `nothing` without selected stars.  `reserve` (per catalog
source) marks reserved stars: in the step, their PSF coefficients are zero (`smask`).
"""
function _refine_problem(st, g, θd, snr, ay, ax, mR, o::PSFRefineOptions; reserve = nothing)
    sc = st.sc; be = sc.backend; T = eltype(sc)
    cand = _refine_select(snr, g.band, o)
    eimg = Int32[]; estar = Int32[]; erb = Int32[]
    for e in g.images, s in cand
        mR[s, e] > 0 || continue
        push!(eimg, e); push!(estar, s); push!(erb, min(g.window, mR[s, e]))
    end
    n = length(eimg)
    n == 0 && return nothing
    stars = unique(estar); sloc = Dict(s => Int32(i) for (i, s) in enumerate(stars))
    eay = Int32[ay[estar[k], eimg[k]] for k in 1:n]; eax = Int32[ax[estar[k], eimg[k]] for k in 1:n]
    # placement on the device, as the fitter's model places the star
    dv = map(x -> to_device(be, x), (; eimg, estar, eay, eax))
    pos = KA.allocate(be, T, 2, n); jac = KA.allocate(be, T, 4, n); ks = KA.allocate(be, Int32, 4, n)
    L = KA.allocate(be, Int32, 2, n); ph = KA.allocate(be, T, 2, n)
    launch!(rf_entry_kernel!, be, n, pos, jac, ks, L, ph, dv.eimg, dv.estar, dv.eay, dv.eax, θd, sc.dev)
    ksh = Array(ks); Lh = Array(L)
    base = g.nodes[1] - Int32(1)
    key(k) = (Lh[1, k], Lh[2, k], erb[k], ksh[1, k], ksh[2, k], ksh[3, k], ksh[4, k])
    perm = sortperm(1:n; by = key)
    # batches: runs of equal keys
    bstart = [1; [i for i in 2:n if key(perm[i]) != key(perm[i - 1])]]; nb = length(bstart)
    bend = [bstart[2:end] .- 1; n]
    bP = Int32[2erb[perm[bstart[b]]] + 1 for b in 1:nb]
    bL = Int32[Lh[r, perm[bstart[b]]] for r in 1:2, b in 1:nb]
    bks = Int32[ksh[m, perm[bstart[b]]] - base for m in 1:4, b in 1:nb]
    ebatch = zeros(Int32, n); soff = zeros(Int32, n); qoff = zeros(Int32, n); nf = 0; nq = 0
    for b in 1:nb, i in bstart[b]:bend[b]
        ebatch[i] = b; soff[i] = nf; nf += Int(bP[b])^2
        qoff[i] = nq; nq += cld(Int(bP[b])^2, 4)
    end
    # entries in batch order from here on
    eimg = eimg[perm]; estar = estar[perm]; eay = eay[perm]; eax = eax[perm]
    eloc = Int32[sloc[s] for s in estar]
    # stars -> entries
    ns_ = length(stars); scount = zeros(Int32, ns_)
    for k in 1:n; scount[eloc[k]] += 1; end
    sptr = Int32.(cumsum([1; scount])); sfill = copy(sptr[1:ns_]); slist = zeros(Int32, n)
    for k in 1:n; slist[sfill[eloc[k]]] = k; sfill[eloc[k]] += 1; end
    # adjoint chunks and batch output offsets
    cb = Int32[]; ck0 = Int32[]; ck1 = Int32[]; cpoff = Int32[]; bc0 = zeros(Int32, nb); bc1 = zeros(Int32, nb); po = 0
    goff = zeros(Int32, nb); go = 0
    for b in 1:nb
        bc0[b] = length(cb) + 1
        for k0 in bstart[b]:RF_CHUNK:bend[b]
            push!(cb, b); push!(ck0, k0); push!(ck1, min(k0 + RF_CHUNK - 1, bend[b])); push!(cpoff, po); po += 64 * Int(bP[b])^2
        end
        bc1[b] = length(cb); goff[b] = go; go += 64 * Int(bP[b])^2
    end
    # node -> (batch, corner)
    nbb = [Int32[] for _ in 1:g.N]; nbm = [Int32[] for _ in 1:g.N]
    for b in 1:nb, m in 1:4
        push!(nbb[bks[m, b]], b); push!(nbm[bks[m, b]], m)
    end
    nbptr = Int32.(cumsum([1; length.(nbb)]))
    d = (; ebatch, soff, qoff, eloc, eay, eax, eimg, estar, bP, bL, bks, goff, cb, ck0, ck1, cpoff, bc0, bc1, nbptr,
           nbb = reduce(vcat, nbb), nbm = reduce(vcat, nbm), sptr, slist)
    dd = map(x -> to_device(be, x), d)
    rows = _refine_rows(sc, g, dd, n, nf)
    fq = KA.allocate(be, Int32, nq)
    launch!(rf_fquad_kernel!, be, n, fq, dd.qoff, dd.soff, dd.ebatch, dd.bP)
    nr = rows.nr; dd = merge(dd, Base.structdiff(rows, (; nr = 0)), (; fq), _rf_xlists(dd, g, be))
    # placement arrays, permuted to batch order
    pos = to_device(be, Array(pos)[:, perm]); jac = to_device(be, Array(jac)[:, perm]); ph = to_device(be, Array(ph)[:, perm])
    # coefficients, derivative stamps and the rows' data
    A = KA.allocate(be, T, 64, n); S = KA.allocate(be, T, 3, nf)
    launch!(rf_stamp_kernel!, be, n, A, S, pos, jac, ph, dd.eay, dd.eax, dd.eimg, dd.estar, dd.ebatch, dd.soff, dd.bP, dd.srow, θd, sc.dev)
    # the in-image stamp pixels of each star and those usable (srow > 0: nonzero weight)
    srowh = Array(dd.srow); nin = zeros(Int, ns_); nok = zeros(Int, ns_)
    for k in 1:n
        P = Int(bP[ebatch[k]]); R = (P - 1) ÷ 2; ny, nx = sc.dims[eimg[k]]
        nin[eloc[k]] += (min(eay[k] + R, ny) - max(eay[k] - R, 1) + 1) * (min(eax[k] + R, nx) - max(eax[k] - R, 1) + 1)
        nok[eloc[k]] += count(>(0), view(srowh, soff[k] .+ (1:P * P)))
    end
    masked = nok .< (1 - o.max_masked) .* nin
    res = Bool[reserve !== nothing && reserve[s] for s in stars]
    smask = to_device(be, Int32.(.!(res .| masked)))
    any(res .| masked) && launch!(rf_mask_kernel!, be, n, A, dd.eloc, smask)
    fstar = KA.allocate(be, T, ns_)
    launch!(rf_fstar_kernel!, be, ns_, fstar, θd, to_device(be, Int32.(stars)), 2 + g.band)
    r = KA.allocate(be, T, nr); W = KA.allocate(be, T, nr); m = KA.allocate(be, T, nr)
    launch!(rf_rows_kernel!, be, nr, r, W, m, dd.rowpix, sc.data, st.bkg, st.work.model, sc.weight)
    KA.synchronize(be)
    foreach(KA.unsafe_free!, (values(dv)..., pos, jac, ks, L, ph))
    Pmax = Int(maximum(bP))
    return (; n, nf, nq, nr, nb, ns = ns_, stars, reserve = res, masked, smask, fstar, M = 64 * Pmax^2, nx = go, npart = po, nchunk = length(cb), d = dd, A, S, r, W, m)
end

# The operators' work buffers for problem p (Y padded for rf_adj_kernel!'s tiles).
_rf_buffers(p, be, ::Type{T}) where {T} = (; Xb = KA.zeros(be, T, p.nx), V = KA.zeros(be, T, p.nf), Y = KA.zeros(be, T, p.nf + RF_YPAD),
                                            part = KA.zeros(be, T, p.npart), ys = KA.zeros(be, T, 3, 1))

# Rough peak device memory (bytes) of group g's step on problem p: per stamp pixel fent, srow, rlist,
# a quarter of fq (Int32) and S, V, Y; per row rowpix, rptr (Int32) and r, W, m plus the solve's five
# row vectors; per adjoint partial part and xidx; per entry A and nine Int32 tables; Xb; per window
# sample the prior and preconditioner blocks, xptr and the CG's six vectors (with their star parts).
_refine_memory(p, g, ::Type{T}) where {T} = (t = sizeof(T); np = g.N * g.nw^2;
    (13 + 5t) * p.nf + (8 + 8t) * p.nr + (4 + t) * p.npart + (36 + 64t) * p.n + t * p.nx + (2g.N * t + 6t + 4) * np + 18t * p.ns)

# J v: rows of the PSF part (x) plus the star part (y); either may be `nothing`.
function _rf_J!(out, x, y, p, g, buf, be)
    d = p.d
    x === nothing || launch!(rf_xgather_kernel!, be, p.M * p.nb, buf.Xb, x, d.bL, d.bP, d.bks, d.goff, p.M, g.r0, g.c0, g.nw, g.sy, g.sx)
    launch!(rf_fwd_kernel!, be, p.nq, buf.V, p.A, buf.Xb, p.S, y === nothing ? buf.ys : y, d.fq, d.fent, d.soff, d.ebatch, d.eloc, d.goff,
            d.bP, x !== nothing, y !== nothing)
    launch!(rf_rowsum_kernel!, be, p.nr, out, buf.V, d.rptr, d.rlist)
    return out
end

# Jᵀ w: the PSF part into gx (window samples), the star part into gy; with `sq`, the PSF part of
# the diagonal Σ J² w (cross terms dropped) instead.
function _rf_Jt!(gx, gy, w, p, g, buf, be; sq = false)
    d = p.d
    launch!(rf_sgather_kernel!, be, p.nf, buf.Y, w, d.srow)
    rf_adj_kernel!(be, 256)(buf.part, p.A, buf.Y, d.cb, d.ck0, d.ck1, d.cpoff, d.bP, d.soff, sq, rf_adj_tile(be)...; ndrange = 256 * p.nchunk)
    launch!(rf_xgather_adj_kernel!, be, length(gx), gx, buf.part, d.xptr, d.xidx)
    gy === nothing || launch!(rf_star_adj_kernel!, be, p.ns, gy, p.S, buf.Y, d.sptr, d.slist, d.soff, d.ebatch, d.bP)
    return gx
end

const RF_KMAX = 3RF_NMAX      # gauge block size up to which the blocks are inverted on the device

# Gi[:, :, l] = inv(G[:, :, l]) per sub-lattice block: one workgroup per block, one item per row,
# Gauss-Jordan without pivoting in local memory (the blocks are symmetric positive definite).  Rows
# of a rank-deficient sub-lattice's basis are zero in `gq`; they get a unit diagonal (their
# components are zero too).  The blocks are well conditioned (condition numbers 35-103 on WFI01), so
# the working precision is enough for a preconditioner (the step changed by ~1e-6 against Float64).
# Each item writes only its own row and reads only its own and the pivot row, so one barrier per step
# (the pivot row scaled before the others read it) is all the ordering needed.
@kernel function rf_gauge_inv_kernel!(Gi, @Const(G), K)
    i = @index(Local, Linear); l = @index(Group, Linear)
    @uniform T = eltype(Gi)
    M = @localmem T (RF_KMAX, RF_KMAX)
    V = @localmem T (RF_KMAX, RF_KMAX)
    if i <= K
        for j in 1:K
            M[i, j] = G[i, j, l]; V[i, j] = ifelse(i == j, one(T), zero(T))
        end
        iszero(M[i, i]) && (M[i, i] = one(T))
    end
    for k in 1:K
        if i == k
            piv = inv(M[k, k])
            for j in 1:K
                M[k, j] *= piv; V[k, j] *= piv
            end
        end
        @synchronize
        if i <= K && i != k
            f = M[i, k]
            for j in 1:K
                M[i, j] -= f * M[k, j]; V[i, j] -= f * V[k, j]
            end
        end
    end
    if i <= K
        for j in 1:K
            Gi[i, j, l] = V[i, j]
        end
    end
end

"""
Write into `Gi` the inverse of every sub-lattice's block of C B Cᵀ for the node blocks `Bi` of group
`g`: on the device up to `RF_KMAX` rows per block, on the host beyond (as the node blocks beyond
`RF_NMAX`).
"""
function _rf_gauge_G!(Gi, Bi, g, be)
    nsub = g.sy * g.sx; K = 3g.N
    G = KA.allocate(be, eltype(Gi), K, K, nsub)
    launch!(rf_gauge_G_kernel!, be, K * K * nsub, G, Bi, g.gq, g.ra, g.ca, g.nw, g.sy, g.sx, g.N)
    if K <= RF_KMAX
        rf_gauge_inv_kernel!(be, RF_KMAX)(Gi, G, K; ndrange = RF_KMAX * nsub)
    else
        Gh = Array(G)
        for l in 1:nsub
            M = Gh[:, :, l]
            for i in 1:K; iszero(M[i, i]) && (M[i, i] = one(eltype(M))); end
            Gh[:, :, l] .= inv(M)
        end
        copyto!(Gi, Gh)
    end
    KA.unsafe_free!(G)
    return Gi
end


"""
The group's MAP step: projected preconditioned CG on the PSF correction (relative to the
reference, warm-started at the current one) and the stars' (u, v, flux) updates, with IRLS and,
with `star_nsigma`, the star-level test after the first round (which zeroes the rejected stars'
PSF coefficients in `p.A`).  Returns the correction, the star updates (`3 x ns`), their
information blocks (`9 x ns`), the fraction of rows with robust weight above 1/2, and per star at the solution
its χ² and usable pixel count, whether it was rejected (by the test or, from the start, `p.masked`), and its ring sums of residual / flux and
their pixel counts (`(window + 1) x ns`).
"""
function _refine_solve(p, g, o::PSFRefineOptions, be, ::Type{T}) where {T}
    N = g.N; nws = g.nw * g.nw; np = N * nws; ns = p.ns; nz = np + 3ns
    buf = _rf_buffers(p, be, T)
    Pk = to_device(be, T, g.Pk)
    vx(v) = view(v, 1:np); vy(v) = reshape(view(v, (np + 1):nz), 3, ns)
    proj!(v) = (launch!(rf_proj_kernel!, be, N * g.sy * g.sx, vx(v), g.gq, g.ra, g.ca, g.nw, g.sy, g.sx); v)
    tmp = KA.zeros(be, T, p.nr); Wρ = similar(tmp); ρ = similar(tmp); e = similar(tmp)
    lc!(out, x, y, a, b) = launch!(rf_lincomb_kernel!, be, length(out), out, x, y, T(a), T(b))
    mul!_(out, x, y) = launch!(rf_mul_kernel!, be, length(out), out, x, y)
    part = KA.zeros(be, T, 256)                   # fixed: the sums do not depend on the thread count
    G = length(part)
    dstep(n) = is_cpu(be) ? cld(n, G * RF_GROUP) : G * RF_GROUP
    function dot_(x, y; cnt = false)
        rf_dot_kernel!(be, RF_GROUP)(part, x, y, length(x), dstep(length(x)), is_cpu(be), cnt; ndrange = G * RF_GROUP)
        return sum(Float64, Array(part))
    end
    sv = KA.zeros(be, T, 8)                       # the CG's device scalars (RF_PQ ...)
    ssum!(k, mode) = rf_ssum_kernel!(be, RF_GROUP)(sv, k, part, G, mode, T(o.cg_tol); ndrange = RF_GROUP)
    function ddot!(k, x, y, mode)
        rf_dot_kernel!(be, RF_GROUP)(part, x, y, length(x), dstep(length(x)), is_cpu(be), false; ndrange = G * RF_GROUP)
        ssum!(k, mode)
    end
    # t = r + J x_cur: the residual of the model at the reference plus the correction
    z = KA.zeros(be, T, nz); copyto!(z, 1, T.(g.xcur), 1, np); proj!(z)
    t = _rf_J!(KA.zeros(be, T, p.nr), to_device(be, T, g.xcur), nothing, p, g, buf, be); lc!(t, t, p.r, 1, 1)
    lc!(e, t, _rf_J!(tmp, vx(z), nothing, p, g, buf, be), 1, -1)
    rob!(ρ, e) = launch!(rf_robust_kernel!, be, p.nr, ρ, e, p.W, p.m, T(o.c), T(o.mu), T(o.reject_z))
    rob!(ρ, e)
    H = KA.zeros(be, T, 9, ns); Hi = KA.zeros(be, T, 9, ns); Bi = KA.zeros(be, T, N, N, nws); dJb = KA.zeros(be, T, np)
    # the gauge in the preconditioner's metric (rf_gauge_G_kernel!)
    nsub = g.sy * g.sx; K = 3N
    Gi = KA.zeros(be, T, K, K, nsub); cv = KA.zeros(be, T, K, nsub); yv = KA.zeros(be, T, K, nsub)
    function gauge!(x)
        launch!(rf_gauge_c_kernel!, be, N * nsub, cv, x, g.gq, g.ra, g.ca, g.nw, g.sy, g.sx)
        launch!(rf_gauge_y_kernel!, be, K * nsub, yv, Gi, cv, K)
        launch!(rf_gauge_corr_kernel!, be, nws, x, Bi, g.gq, yv, g.rph, g.cph, g.nw, g.sy, N)
        return x
    end
    function Aop!(out, v)
        _rf_J!(tmp, vx(v), vy(v), p, g, buf, be)
        mul!_(tmp, tmp, Wρ)
        _rf_Jt!(vx(out), vy(out), tmp, p, g, buf, be)
        launch!(rf_block_kernel!, be, nws, vx(out), vx(v), Pk, N, nws, true)
        lc!(view(out, (np + 1):nz), view(out, (np + 1):nz), view(v, (np + 1):nz), 1, 1e-9)
        return out
    end
    # the trailing projection only removes rounding from the gauge-metric step
    function Mi!(out, r)
        launch!(rf_block_kernel!, be, nws, vx(out), vx(r), Bi, N, nws, false)
        gauge!(vx(out))
        launch!(rf_star_prec_kernel!, be, ns, vy(out), vy(r), Hi)
        return proj!(out)
    end
    r = similar(z); q = similar(z); zz = similar(z); pk = similar(z); b = similar(z)
    nbin = g.window + 1
    chi = KA.zeros(be, T, ns); cnt = KA.zeros(be, Int32, ns); prof = KA.zeros(be, T, nbin, ns); pcnt = KA.zeros(be, Int32, nbin, ns)
    starchi!() = launch!(rf_star_chi_kernel!, be, ns, chi, cnt, prof, pcnt, e, p.W, p.m, ρ, p.d.srow, p.d.sptr, p.d.slist, p.d.soff,
                         p.d.ebatch, p.d.bP, p.fstar, T(o.mu))
    rejected = copy(p.masked)
    cnts = KA.zeros(be, T, o.irls + 1)            # CG iterations per round, read with the diagnostics
    for round in 0:o.irls
        mul!_(Wρ, p.W, ρ)
        # preconditioner: star blocks and, per sample, inv(diag(Jᵀ W J) + prior block)
        launch!(rf_sgather_kernel!, be, p.nf, buf.Y, Wρ, p.d.srow)
        launch!(rf_star_block_kernel!, be, ns, H, Hi, p.S, buf.Y, p.d.sptr, p.d.slist, p.d.soff, p.d.ebatch, p.d.bP)
        dJ = _rf_Jt!(dJb, nothing, Wρ, p, g, buf, be; sq = true)
        if N <= RF_NMAX
            launch!(rf_precond_kernel!, be, nws, Bi, dJ, Pk, N, nws)
        else
            dh = Array(dJ); Bh = Array{Float64}(undef, N, N, nws)
            for s in 1:nws
                Bh[:, :, s] .= inv(Diagonal([Float64(dh[(n - 1) * nws + s]) for n in 1:N]) + g.Pk[:, :, s])
            end
            copyto!(Bi, T.(Bh))
        end
        _rf_gauge_G!(Gi, Bi, g, be)
        mul!_(tmp, Wρ, t)
        _rf_Jt!(vx(b), vy(b), tmp, p, g, buf, be)
        lc!(r, b, Aop!(q, z), 1, -1); proj!(r)
        Mi!(zz, r); copyto!(pk, zz)
        # cg_iterations sweeps with the scalars on the device, no host read: once ‖r‖ <= cg_tol r0 a
        # flag stops the updates (α = 0), as a host loop's early exit would
        fill!(sv, zero(T))
        a, bz = 1, 2
        ddot!(a, r, zz, 0); ddot!(RF_RR, r, r, 2)
        for _ in 1:o.cg_iterations
            Aop!(q, pk); proj!(q)
            ddot!(RF_PQ, pk, q, 3)
            rf_cg_xr_kernel!(be, RF_GROUP)(z, r, part, pk, q, sv, a, nz, dstep(nz), is_cpu(be); ndrange = G * RF_GROUP)
            ssum!(RF_RR, 1)
            Mi!(zz, r); ddot!(bz, r, zz, 0)
            launch!(rf_cg_p_kernel!, be, nz, pk, zz, sv, a, bz)
            a, bz = bz, a
        end
        copyto!(cnts, round + 1, sv, RF_CNT, 1)
        round == o.irls && break
        lc!(e, t, _rf_J!(tmp, vx(z), vy(z), p, g, buf, be), 1, -1)
        rob!(ρ, e)
        # the star-level test, against the model after the first round's solve
        if round == 0 && o.star_nsigma !== nothing
            starchi!()
            rej = _star_outliers(Float64.(Array(chi)), Int.(Array(cnt)), p.reserve .| rejected, o)
            if any(rej)
                rejected .|= rej
                copyto!(p.smask, Int32.(.!(p.reserve .| rejected)))
                launch!(rf_mask_kernel!, be, p.n, p.A, p.d.eloc, p.smask)
            end
        end
    end
    wfrac = dot_(ρ, ρ; cnt = true) / p.nr
    # per-star diagnostics at the solution
    lc!(e, t, _rf_J!(tmp, vx(z), vy(z), p, g, buf, be), 1, -1)
    rob!(ρ, e); starchi!()
    zh = Array(z); iters = Int.(Array(cnts))
    out = (; x = Float64.(zh[1:np]), d = Float64.(reshape(zh[(np + 1):nz], 3, ns)), H = Float64.(Array(H)), iters, wfrac,
             chi2 = Float64.(Array(chi)), npix = Int.(Array(cnt)), rejected, prof = Float64.(Array(prof)), pcnt = Int.(Array(pcnt)))
    foreach(KA.unsafe_free!, (buf..., Pk, tmp, Wρ, ρ, e, H, Hi, Bi, Gi, cv, yv, r, q, zz, pk, b, z, t, part, sv, cnts, dJb, chi, cnt, prof, pcnt))
    return out
end

"""
The star-level test: fitting stars (not `reserve`d, more than 3 usable pixels) whose χ² exceeds
the `star_nsigma` point of the χ² distribution with (pixels - 3) degrees of freedom, times
max(1, the fitting stars' median χ²/dof); at most `star_max_remove` of the fitting stars, those
with the largest excess.
"""
function _star_outliers(chi, npix, reserve, o::PSFRefineOptions)
    ns = length(chi); rej = falses(ns)
    fit = [!reserve[i] && npix[i] > 3 for i in 1:ns]
    any(fit) || return rej
    factor = max(1.0, median(chi[fit] ./ (npix[fit] .- 3)))
    prob = erfc(o.star_nsigma / sqrt(2.0))
    thr = Dict(k => 2 * gamma_inc_inv(k / 2, 1 - prob, prob) for k in unique(npix[fit] .- 3))
    excess = [fit[i] ? chi[i] - thr[npix[i] - 3] * factor : -Inf for i in 1:ns]
    cand = findall(>(0), excess)
    nmax = ceil(Int, o.star_max_remove * count(fit))         # as PIFF: small groups can still remove one
    length(cand) > nmax && (cand = cand[partialsortperm(excess[cand], 1:nmax; rev = true)])
    rej[cand] .= true
    return rej
end

# Write the correction x (window samples, relative to the reference) into the group's host model
# (in its element type) and the device nodes; the group's next warm start is what was written.
function _refine_apply!(g, x, sc)
    T = eltype(sc); nodes = sc.dev.psf.nodes; nws = g.nw * g.nw
    for (n, nd) in enumerate(g.hpsf.psfs)
        P = nd.data
        for c in 1:g.nw, r in 1:g.nw
            v = convert(eltype(P), g.ref[r, c, n] + x[(n - 1) * nws + (c - 1) * g.nw + r])
            P[g.r0 + r - 1, g.c0 + c - 1] = v
            g.xcur[(n - 1) * nws + (c - 1) * g.nw + r] = Float64(v) - g.ref[r, c, n]
        end
        len = length(P)
        copyto!(nodes, (g.nodes[n] - 1) * len + 1, convert(Array{T}, P), 1, len)
    end
end

# S/N bins of the residual profiles in the refinement statistics
const RF_SNR_EDGES = (20.0, 50.0, 200.0, Inf)

"""
One refinement after a pass's fit and prune at catalog `θh` (`hd`: the fit's precision diagonal
for `θh`; `reserve`: the reserved sources): every group's step, the node update, and the stars'
updates combined over groups (positions by their information blocks, capped at `max_step`; band
fluxes from their band's group).  Returns the updated state (PSF-dependent pieces rebuilt), `θh`
with the star updates, per-group statistics, and one row per (star, group) of the step's stars
(`col`: the star's column of `θh`).
"""
function _refine!(st, groups, θh, hd, lo; reserve = nothing)
    o = lo.psf_refine; sc = st.sc; be = sc.backend; T = eltype(sc)
    N = size(θh, 2); B = nbands(sc)
    θd = to_device(be, T, θh)
    fp = footprint(sc, θd, st.rt)
    render!(st.work.model, sc, θd, fp)
    ay = Array(fp.ay); ax = Array(fp.ax); mR = Array(fp.mR)
    snr = [hd[2 + b, s] > 0 ? θh[2 + b, s] * sqrt(hd[2 + b, s]) : -Inf for b in 1:B, s in 1:N]
    Hp = zeros(3, N); gp = zeros(2, N); Hf = zeros(size(θh)); gf = zeros(size(θh))
    stats = []
    rows = (; col = Int[], group = Int[], band = Symbol[], reserve = Bool[], rejected = Bool[], fit = Bool[], snr = Float64[],
              chi2 = Float64[], dof = Int[])
    nsb = length(RF_SNR_EDGES) - 1
    for (gi, g) in enumerate(groups)
        nbin = g.window + 1
        t_build = @elapsed p = _refine_problem(st, g, θd, snr, ay, ax, mR, o; reserve)
        if p === nothing
            push!(stats, (; band = sc.bands[g.band], nsel = 0, n_reserve = 0, n_masked = 0, n_rejected = 0, rows = 0, iters = Int[], wfrac = NaN,
                          chi2_fit = NaN, chi2_reserve = NaN, prof_fit = fill(NaN, nbin, nsb), prof_reserve = fill(NaN, nbin, nsb), t_build, t_solve = 0.0,
                          mem_est = 0))
            continue
        end
        t_solve = @elapsed sol = _refine_solve(p, g, o, be, T)
        _refine_apply!(g, sol.x, sc)
        row = 2 + g.band
        for (i, s) in enumerate(p.stars)
            d = view(sol.d, :, i); h = view(sol.H, :, i)
            (all(isfinite, d) && all(isfinite, h)) || continue
            Hp[1, s] += h[1]; Hp[2, s] += h[4]; Hp[3, s] += h[5]
            gp[1, s] += h[1] * d[1] + h[4] * d[2]; gp[2, s] += h[2] * d[1] + h[5] * d[2]
            Hf[row, s] += h[9]; gf[row, s] += h[9] * d[3]
        end
        # the step's stars: fitting (constrained the PSF to the end), reserved, rejected (star test or max_masked)
        fit = .!(p.reserve .| sol.rejected); dof = sol.npix .- 3
        ssnr = [snr[g.band, s] for s in p.stars]
        red(m) = (k = m .& (dof .> 0); any(k) ? median(sol.chi2[k] ./ dof[k]) : NaN)
        # mean residual / flux per pixel, by ring (rows) and S/N bin (columns)
        function prof(m)
            P = fill(NaN, nbin, nsb)
            for j in 1:nsb
                k = findall(i -> m[i] && RF_SNR_EDGES[j] <= ssnr[i] < RF_SNR_EDGES[j + 1], eachindex(m))
                isempty(k) && continue
                c = vec(sum(view(sol.pcnt, :, k); dims = 2))
                P[:, j] .= ifelse.(c .> 0, vec(sum(view(sol.prof, :, k); dims = 2)) ./ max.(c, 1), NaN)
            end
            return P
        end
        push!(stats, (; band = sc.bands[g.band], nsel = length(p.stars), n_reserve = count(p.reserve), n_masked = count(p.masked), n_rejected = count(sol.rejected),
                      rows = p.nr, sol.iters, sol.wfrac, chi2_fit = red(fit), chi2_reserve = red(p.reserve), prof_fit = prof(fit),
                      prof_reserve = prof(p.reserve), t_build, t_solve, mem_est = _refine_memory(p, g, T)))
        append!(rows.col, p.stars); append!(rows.group, fill(gi, length(p.stars))); append!(rows.band, fill(sc.bands[g.band], length(p.stars)))
        append!(rows.reserve, p.reserve); append!(rows.rejected, sol.rejected); append!(rows.fit, fit); append!(rows.snr, ssnr)
        append!(rows.chi2, sol.chi2); append!(rows.dof, dof)
        foreach(x -> x isa AbstractArray && KA.unsafe_free!(x), (values(p.d)..., p.A, p.S, p.r, p.W, p.m, p.smask, p.fstar))
    end
    Δθ = zeros(size(θh))
    for s in 1:N
        a, b, c = Hp[1, s], Hp[2, s], Hp[3, s]
        dt = a * c - b * b
        dt > 0 || continue                      # no position information (every pixel of the star rejected)
        du = (c * gp[1, s] - b * gp[2, s]) / dt; dv = (a * gp[2, s] - b * gp[1, s]) / dt
        (isfinite(du) && isfinite(dv)) || continue
        nrm = hypot(du, dv)
        nrm > o.max_step && (du *= o.max_step / nrm; dv *= o.max_step / nrm)
        Δθ[1, s] = du; Δθ[2, s] = dv
    end
    for i in eachindex(Hf); Hf[i] > 0 && (Δθ[i] = gf[i] / Hf[i]); end
    # the PSF-dependent pieces: radius table, detection banks, wing template (the device nodes are already updated)
    psfs = copy(st.psfs)
    for g in groups, e in g.images; psfs[e] = g.hpsf; end
    f = lo.fit
    rt = RadiusTable(sc, psfs, [im.weight for im in st.images]; model_rad = f.model_rad, nsigma = f.model_rad_nsigma, f.R_fit, f.R_cap)
    bk = st.jd.banks
    foreach(KA.unsafe_free!, (bk.P, bk.c, bk.cχ, bk.rng))
    jd = JointDetection(sc, psfs, st.offs, lo.detector; weights = lo.band_weights)
    tmpl = lo.gate > 0 ? wing_template(jd, sc, st.ivs; data = st.det_data, iv = st.det_iv) : nothing
    return merge(st, (; rt, jd, tmpl, psfs)), θh .+ Δθ, stats, rows
end

# The (star, group) rows of a refinement after a catalog keep mask: rows of dropped stars removed,
# `col` renumbered to the kept columns.
function _remap_psf_stars(rows, keep)
    rows === nothing && return nothing
    newcol = cumsum(keep)
    idx = findall(c -> keep[c], rows.col)
    r = map(v -> v[idx], rows)
    return merge(r, (; col = newcol[r.col]))
end
