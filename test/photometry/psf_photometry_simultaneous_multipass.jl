using CrowdPhot
using CrowdPhot: Catalog, Discards, FitPlan, append_sources, sort_morton, drop!,
    prune_mask, estimate_background_multipass, detect_sources, stamp_geometry, theta_from_catalog,
    _anchor_key2, _SepGrid, _has_neighbor, _insert!, matched_filter, measure_star_shapes,
    StampDerivatives, apply_JT!, apply_J!, _jacobian_operator, _fill_stamps!, _model_radii,
    _render_model!, _accum_model!
using CrowdPhot.PSF
using CrowdPhot.PSF: CircularGaussianPSF, GriddedPSFModel, GaussianPRF
using CrowdPhot.Background: SExtractorBackground, MADStdRMS
using ConstructionBase
using Krylov: lsqr!, lsmr!, LsqrWorkspace, LsmrWorkspace, solution
using LinearAlgebra: dot, I
using StableRNGs
using Statistics: median, mean
using Test

const PSF_FWHM = 3.0
const TEST_PSF = CircularGaussianPSF(0.0, 0.0, PSF_FWHM, 1.0, 0.0)
const TEST_FIXED = (; fwhm = PSF_FWHM)

# Smooth, spatially varying background: a gradient plus a broad bump.  A constant
# background exercises none of the coarse -> fine schedule or the RMS adaptivity
# that motivated the design.
function varying_background(ny, nx; level = 50.0)
    yy = (1:ny) .* ones(nx)'
    xx = ones(ny) .* (1:nx)'
    return @. level + 0.05 * yy + 0.03 * xx +
        0.6 * level * exp(-((yy - ny / 2)^2 + (xx - nx / 2)^2) / (2 * (ny / 4)^2))
end

function test_field(; ny = 160, nx = 160, n = 90, seed = 20240905, flux = (800.0, 20000.0))
    rng = StableRNG(seed)
    bg = varying_background(ny, nx)
    img, src = simulate_image((ny, nx), TEST_PSF, n; background = bg, flux, rng, border = 8)
    return img, src, bg
end

# Build the full (npix x n) Jacobian from a StampDerivatives so the operator
# and LSQR step can be checked against dense linear algebra.
function dense_J(stamp::StampDerivatives)
    p, S2, n_active = size(stamp.values)
    J = zeros(stamp.npix, n_active * p)
    for a in 0:(n_active - 1)
        for m in 1:S2
            fi = stamp.pixels[m, a + 1]
            fi != 0 || continue
            for k in 1:p
                J[fi, a * p + k] = stamp.values[k, m, a + 1]
            end
        end
    end
    return J
end

# Column-equilibrate a stamp (as the real fill does) so H has unit diagonal.
function equilibrate!(stamp::StampDerivatives)
    p, S2, n_active = size(stamp.values)
    for a in 1:n_active
        for k in 1:p
            s = zero(eltype(stamp.values))
            for m in 1:S2
                s += stamp.values[k, m, a]^2
            end
            s = max(sqrt(s), eps())
            for m in 1:S2
                stamp.values[k, m, a] /= s
            end
        end
    end
    return stamp
end

@testset "stamp operator kernels" begin
    rng = StableRNG(1234)

    @testset "_fill_stamps! preserves frozen colnorm/values" begin
        psf = CircularGaussianPSF(y=0.0, x=0.0, fwhm=2.0, flux=1.0, bkg=0.0)
        fixed = (; fwhm=2.0, bkg=0.0)
        free_names, free_idx, _ = PSF.free_params(psf, fixed)
        p = length(free_idx)
        prop_names = collect(keys(ConstructionBase.getproperties(psf)))
        row_y = findfirst(==(:y), prop_names)
        row_x = findfirst(==(:x), prop_names)
        row_flux = findfirst(==(:flux), prop_names)
        grad_col = [free_names[k] === :y ? 1 : (free_names[k] === :x ? 2 : 3) for k in 1:p]
        free_names_val = Val(free_names)

        R = 1
        dy_off = Int[]
        dx_off = Int[]
        for dx in -R:R, dy in -R:R
            push!(dy_off, dy)
            push!(dx_off, dx)
        end
        S2 = length(dy_off)
        n_active = 2
        ny = 30
        anchor_y = [10, 10]
        anchor_x = [10, 20]  # far apart: no shared pixels between the two stars
        pixels = zeros(Int32, S2, n_active)
        for a in 1:n_active, m in 1:S2
            gy = anchor_y[a] + dy_off[m]
            gx = anchor_x[a] + dx_off[m]
            pixels[m, a] = gy + (gx - 1) * ny
        end
        npix = ny * 30
        θ = zeros(p * n_active)
        for a in 1:n_active
            θ[(a - 1) * p .+ (1:p)] .= (Float64(anchor_y[a]), Float64(anchor_x[a]), 100.0)
        end
        w = ones(npix)
        stamp = StampDerivatives{Float64, Int32}(
            zeros(p, S2, n_active), pixels, zeros(p, n_active), npix, p, S2)

        live = trues(n_active)
        live[2] = false  # star 2 frozen from the start
        _fill_stamps!(stamp, psf, free_names_val, fixed, θ, w,
            grad_col, dy_off, dx_off, anchor_y, anchor_x, row_y, row_x, row_flux, live, nothing)
        values_before = copy(stamp.values)
        colnorm_before = copy(stamp.colnorm)
        @test all(isfinite, values_before)
        @test all(isfinite, colnorm_before)

        # Repeated fills with a perturbed θ for the still-live star only: the
        # frozen star's column must stay bitwise untouched, not drift or blow
        # up via a stale colnorm floor (the bug this test regresses).
        θ2 = copy(θ)
        for _ in 1:5
            θ2[1] += 0.1
            _fill_stamps!(stamp, psf, free_names_val, fixed, θ2, w,
                grad_col, dy_off, dx_off, anchor_y, anchor_x, row_y, row_x, row_flux, live, nothing)
        end
        @test stamp.values[:, :, 2] == values_before[:, :, 2]
        @test stamp.colnorm[:, 2] == colnorm_before[:, 2]
        @test all(isfinite, stamp.values)
        @test all(isfinite, stamp.colnorm)
    end

    @testset "_accum_model! removes a subset" begin
        # The multipass pass loop carries its model forward by subtracting pruned
        # sources instead of re-rendering the survivors (`unrender!`).  The two
        # agree in exact arithmetic; this pins that they agree numerically, and
        # that the subtraction covers the full per-source `model_R` box.
        psf = CircularGaussianPSF(y = 0.0, x = 0.0, fwhm = 2.5, flux = 1.0, bkg = 0.0)
        fixed = (; fwhm = 2.5, bkg = 0.0)
        free_names, _, _ = PSF.free_params(psf, fixed)
        fnv = Val(free_names)
        p = length(free_names)
        ny = nx = 48
        npix = ny * nx
        rng = StableRNG(99)
        n = 12
        anchor_y = rand(rng, 8:41, n)
        anchor_x = rand(rng, 8:41, n)
        model_R = rand(rng, 3:6, n)
        θ = Float64[]
        for j in 1:n
            append!(θ, (anchor_y[j] + 0.3, anchor_x[j] - 0.25, 500.0 * j))
        end
        Rmax = maximum(model_R)
        rbuf = Matrix{Float64}(undef, 2Rmax + 1, 2Rmax + 1)
        rs = PSF._render_scratch(psf, 2Rmax + 1, Float64)

        keep = trues(n); keep[[2, 5, 9]] .= false
        all_img = zeros(npix); sur_img = zeros(npix)
        _render_model!(all_img, psf, fnv, fixed, θ, p, model_R, anchor_y, anchor_x,
            ny, nx, trues(n), rbuf, rs)
        _render_model!(sur_img, psf, fnv, fixed, θ, p, model_R, anchor_y, anchor_x,
            ny, nx, keep, rbuf, rs)
        # subtract the dropped sources from the full render
        _accum_model!(all_img, psf, fnv, fixed, θ, p, model_R, anchor_y, anchor_x,
            ny, nx, .!keep, rbuf, rs, -1.0)
        @test all_img ≈ sur_img rtol = 1.0e-12
        @test maximum(abs, all_img .- sur_img) < 1.0e-9 * maximum(abs, sur_img)
        # A no-op mask leaves the image untouched, bitwise.
        before = copy(sur_img)
        _accum_model!(sur_img, psf, fnv, fixed, θ, p, model_R, anchor_y, anchor_x,
            ny, nx, falses(n), rbuf, rs, -1.0)
        @test sur_img == before
    end

    @testset "_fill_stamps! zeros masked entries" begin
        # `apply_J!`/`apply_JT!` touch masked entries (the adjoint gathers the
        # clamped `u[1]`) and rely on the derivative being exactly zero.  The
        # multipass `StampStore` reuses one buffer across passes, so a skipped
        # slot would otherwise keep a derivative from a different source/mask.
        psf = CircularGaussianPSF(y = 0.0, x = 0.0, fwhm = 2.0, flux = 1.0, bkg = 0.0)
        fixed = (; fwhm = 2.0, bkg = 0.0)
        free_names, free_idx, _ = PSF.free_params(psf, fixed)
        p = length(free_idx)
        prop_names = collect(keys(ConstructionBase.getproperties(psf)))
        row_y, row_x, row_flux = findfirst(==(:y), prop_names), findfirst(==(:x), prop_names), findfirst(==(:flux), prop_names)
        grad_col = [free_names[k] === :y ? 1 : (free_names[k] === :x ? 2 : 3) for k in 1:p]
        free_names_val = Val(free_names)

        R, ny = 1, 30
        dy_off = Int[]; dx_off = Int[]
        for dx in -R:R, dy in -R:R
            push!(dy_off, dy); push!(dx_off, dx)
        end
        S2 = length(dy_off)
        npix = ny * 30
        anchor_y = [10]; anchor_x = [10]
        pixels = zeros(Int32, S2, 1)
        for m in 1:S2                      # every other stamp entry masked out
            isodd(m) || continue
            pixels[m, 1] = (anchor_y[1] + dy_off[m]) + (anchor_x[1] + dx_off[m] - 1) * ny
        end
        θ = Float64[anchor_y[1], anchor_x[1], 100.0]
        w = ones(npix)
        stamp = StampDerivatives{Float64, Int32}(
            fill(NaN, p, S2, 1), pixels, zeros(p, 1), npix, p, S2)   # poisoned buffer

        _fill_stamps!(stamp, psf, free_names_val, fixed, θ, w, grad_col, dy_off, dx_off,
            anchor_y, anchor_x, row_y, row_x, row_flux, trues(1), nothing)
        for m in 1:S2
            pixels[m, 1] == 0 || continue
            @test all(iszero, stamp.values[:, m, 1])
        end
        @test all(isfinite, stamp.values)
        # The adjoint must stay finite: a masked slot left at NaN poisons all of it.
        z = zeros(p)
        apply_JT!(z, stamp, ones(npix), trues(1), Vector{Float64}(undef, S2))
        @test all(isfinite, z)
    end

    @testset "adjoint identity" begin
        p = 3
        S2 = 9
        n_active = 4
        npix = 20
        pixels = zeros(Int32, S2, n_active)
        # Give stars overlapping footprints.
        for a in 1:n_active
            for m in 1:S2
                fi = mod(m + (a - 1) * 2 - 1, npix) + 1
                pixels[m, a] = fi
            end
        end
        stamp = StampDerivatives{Float64, Int32}(
            randn(rng, p, S2, n_active), pixels, zeros(p, n_active), npix, p, S2)
        u = randn(rng, npix)
        v = randn(rng, p * n_active)
        y = zeros(npix)
        z = zeros(p * n_active)
        apply_J!(y, stamp, v)
        apply_JT!(z, stamp, u, trues(n_active))
        @test dot(u, y) ≈ dot(z, v) rtol = 1e-12

        # Both products agree with the explicit dense Jacobian.
        J = dense_J(stamp)
        @test y ≈ J * v rtol = 1e-12
        @test z ≈ J' * u rtol = 1e-12

        # A frozen star drops its columns from J: `apply_J!` ignores that
        # star's slice of `v`, and `apply_JT!` leaves its slice of `z` at 0.
        live = trues(n_active); live[2] = false
        Jm = copy(J); Jm[:, (2 - 1) * p + 1:2 * p] .= 0
        apply_J!(y, stamp, v, live, zeros(S2))
        apply_JT!(z, stamp, u, live)
        @test y ≈ Jm * v rtol = 1e-12
        @test z ≈ Jm' * u rtol = 1e-12
        @test all(iszero, view(z, (2 - 1) * p + 1:2 * p))
    end

    @testset "matrix-free operator matches dense J" begin
        p = 3
        S2 = 9
        n_active = 6
        npix = 30
        pixels = zeros(Int32, S2, n_active)
        for a in 1:n_active, m in 1:S2
            pixels[m, a] = mod(m + (a - 1) * 2 - 1, npix) + 1
        end
        stamp = StampDerivatives{Float64, Int32}(
            randn(rng, p, S2, n_active), pixels, zeros(p, n_active), npix, p, S2)
        equilibrate!(stamp)
        J = dense_J(stamp)
        n = p * n_active

        live = trues(n_active)
        op = _jacobian_operator(stamp, live, zeros(S2), npix, n)
        v = randn(rng, n)
        u = randn(rng, npix)
        @test op * v ≈ J * v rtol = 1e-12
        @test op' * u ≈ J' * u rtol = 1e-12

        # `op` closes over `live`: freezing a star zeroes its columns in place.
        live[3] = false
        Jm = copy(J); Jm[:, (3 - 1) * p + 1:3 * p] .= 0
        @test op * v ≈ Jm * v rtol = 1e-12
        @test op' * u ≈ Jm' * u rtol = 1e-12
    end

    @testset "LSQR/LSMR step matches dense damped normal-equation solve" begin
        p = 3
        S2 = 9
        n_active = 6
        npix = 8 * n_active + 8
        pixels = zeros(Int32, S2, n_active)
        # Light overlap (1 shared pixel between adjacent stars) so JᵀJ is
        # well-conditioned and the Krylov solves converge tightly.
        for a in 1:n_active, m in 1:S2
            pixels[m, a] = mod(m + (a - 1) * 8 - 1, npix) + 1
        end
        stamp = StampDerivatives{Float64, Int32}(
            randn(rng, p, S2, n_active), pixels, zeros(p, n_active), npix, p, S2)
        equilibrate!(stamp)
        J = dense_J(stamp)
        n = p * n_active

        μ = 1.0e-3                       # Marquardt damping (the outer-loop `λ`)
        b = randn(rng, npix)             # weighted residual RHS
        δ_dense = (J' * J + μ * Matrix{Float64}(I, n, n)) \ (J' * b)

        for (wsT, solve!) in ((LsqrWorkspace, lsqr!), (LsmrWorkspace, lsmr!))
            live = trues(n_active)
            op = _jacobian_operator(stamp, live, zeros(S2), npix, n)
            ws = wsT(npix, n, Vector{Float64})
            solve!(ws, op, b; λ = sqrt(μ), itmax = 200, atol = 1e-12, btol = 1e-12)
            @test solution(ws) ≈ δ_dense rtol = 1e-7 atol = 1e-9

            # A frozen star: its slice of the step stays exactly 0 (its columns
            # of `op` are structurally zero and the solver starts from 0).
            live[2] = false
            solve!(ws, op, randn(rng, npix); λ = sqrt(μ), itmax = 200, atol = 1e-12, btol = 1e-12)
            @test all(iszero, view(solution(ws), (2 - 1) * p + 1:2 * p))
        end
    end
    @testset "_model_radii" begin
        psf = CircularGaussianPSF(y = 0.0, x = 0.0, fwhm = 2.5, flux = 1.0, bkg = 0.0)
        R_fit, R_cap = 2, 20
        w = fill(1 / 125.0, 4000)               # sigma_bg ~ 11.2 ADU
        # Scalar path: one clamped value for all.
        @test _model_radii(psf, 7.0, 1.0, R_fit, R_cap, w, Float64[100, 5e4]) == fill(7, 2)
        @test _model_radii(psf, 1.0, 1.0, R_fit, R_cap, w, Float64[100]) == [R_fit]   # clamped up
        @test _model_radii(psf, 999.0, 1.0, R_fit, R_cap, w, Float64[100]) == [R_cap] # clamped down
        # :auto path: monotone non-decreasing in flux, faint -> R_fit, bright grows.
        fl = Float64[50, 500, 5_000, 50_000, 500_000]
        rr = _model_radii(psf, :auto, 1.0, R_fit, R_cap, w, fl)
        @test issorted(rr)
        @test all(R_fit .<= rr .<= R_cap)
        @test rr[1] == R_fit          # a faint source's wings are below the noise
        @test rr[end] > rr[1]         # a bright source needs a larger box
        # A larger nsigma (looser threshold) never needs a larger box.
        @test all(_model_radii(psf, :auto, 3.0, R_fit, R_cap, w, fl) .<= rr)

        # `sigma_bg` comes from a strided sample of `w`, taken for speed on a
        # full frame.  A periodically masked map can put every valid pixel off
        # that stride, which used to reach `quantile` with an empty vector and
        # throw `ArgumentError: empty data vector`.  The full-map fallback must
        # recover the *same* radii the dense map gives, not just avoid the
        # crash -- landing on the `sigma_bg = 1` default would silently mis-size
        # every box.
        fl2 = Float64[3.0e4, 1.0e3]
        npix = 160_000                      # > 131072, so the stride is 2
        striped = zeros(npix); striped[2:2:end] .= 1 / 100.0
        @test all(iszero, striped[1:2:end])   # the sample really is empty
        dense = fill(1 / 100.0, npix)
        @test _model_radii(psf, :auto, 1.0, R_fit, R_cap, striped, fl2) ==
              _model_radii(psf, :auto, 1.0, R_fit, R_cap, dense, fl2)
        @test _model_radii(psf, :auto, 1.0, R_fit, R_cap, striped, fl2) !=
              _model_radii(psf, :auto, 1.0, R_fit, R_cap, zeros(npix), fl2)
        # No positive finite weight anywhere is genuinely degenerate: `sigma_bg`
        # falls back to 1 and the radius is set by the curve of growth alone.
        @test all(R_fit .<= _model_radii(psf, :auto, 1.0, R_fit, R_cap,
                                         zeros(npix), fl2) .<= R_cap)
    end
    @testset "_model_radii for GriddedPSFModel" begin
        # The :auto path renders a unit PSF and takes its curve of growth; for
        # an image/gridded PSF that uses the generic (pixel-integrated)
        # `curve_of_growth` and `ConstructionBase.setproperties`.
        circ = CircularGaussianPSF(y = 0.0, x = 0.0, fwhm = 2.0, flux = 1.0, bkg = 0.0)
        node = ImagePSF(render(circ); y = 0.0, x = 0.0, flux = 1.0, bkg = 0.0, oversampling = 1, normalize = true)
        gpsf = GriddedPSFModel([node], [0.0], [0.0]; y = 0.0, x = 0.0, flux = 1.0, bkg = 0.0)
        w = fill(1 / 125.0, 4000)
        rr = _model_radii(gpsf, :auto, 1.0, 2, 15, w, Float64[50, 500, 5_000, 50_000])

        @test issorted(rr) && all(2 .<= rr .<= 15) && rr[end] > rr[1]

    end



    @testset "_source_errors! is scale-invariant for bright sources" begin
        # The ridge that keeps a starved block invertible must be applied in
        # equilibrated coordinates.  On the raw block a bright source's position
        # curvature (~flux^2) dwarfs its flux curvature, so a trace-relative ridge
        # there shrank the flux error of a 7e5-count star by 20%.
        psf = CircularGaussianPSF(y = 0.0, x = 0.0, fwhm = 2.9, flux = 1.0, bkg = 0.0)
        fixed = (; fwhm = 2.9, bkg = 0.0)
        free_names, free_idx, _ = PSF.free_params(psf, fixed)
        p = length(free_idx)
        prop_names = collect(keys(ConstructionBase.getproperties(psf)))
        row_y, row_x, row_flux = findfirst(==(:y), prop_names), findfirst(==(:x), prop_names), findfirst(==(:flux), prop_names)
        grad_col = [free_names[k] === :y ? 1 : (free_names[k] === :x ? 2 : 3) for k in 1:p]
        R, ny = 5, 30
        dy_off = Int[]; dx_off = Int[]
        for dx in -R:R, dy in -R:R
            push!(dy_off, dy); push!(dx_off, dx)
        end
        S2 = length(dy_off)
        npix = ny * ny
        anchor_y, anchor_x = [15], [15]
        pixels = zeros(Int32, S2, 1)
        for m in 1:S2
            pixels[m, 1] = (15 + dy_off[m]) + (15 + dx_off[m] - 1) * ny
        end
        θ = [15.2, 14.9, 7.0e5]
        w = fill(1 / 120.0, npix)
        stamp = StampDerivatives{Float64, Int32}(zeros(p, S2, 1), pixels, zeros(p, 1), npix, p, S2)
        _fill_stamps!(stamp, psf, Val(free_names), fixed, θ, w, grad_col, dy_off, dx_off,
            anchor_y, anchor_x, row_y, row_x, row_flux, trues(1), nothing)
        errs = zeros(p, 1)
        CrowdPhot._source_errors!(errs, stamp, CrowdPhot.KnownWeightsCovarianceEstimator(), 1.0, 1)
        J = dense_J(stamp) .* reshape(stamp.colnorm, 1, p)   # undo the equilibration
        exact = sqrt.([inv(J' * J)[k, k] for k in 1:p])
        @test vec(errs) ≈ exact rtol = 1e-8
    end
end

@testset "Separation grid" begin
    g = _SepGrid{Float64}(2.0, [10.0, 30.0], [10.0, 30.0])
    _insert!(g, 10.0, 10.0)
    @test _has_neighbor(g, 10.5, 10.5)
    @test !_has_neighbor(g, 20.0, 20.0)
    # Exactly at the radius counts as a neighbor; just beyond does not.
    @test _has_neighbor(g, 12.0, 10.0)
    @test !_has_neighbor(g, 12.001, 10.0)
    # Non-finite coordinates never match and are never inserted.
    @test !_has_neighbor(g, NaN, 10.0)
    _insert!(g, NaN, 5.0)
    @test length(g.y) == 1

    # The cell size never drops below `sep`, so `head` stays bounded no matter
    # how small `sep` is: sized at `sep` alone this would be 1.6e9 cells.
    tiny = _SepGrid{Float64}(0.01, [1000.0, 2000.0], [1000.0, 2000.0])
    @test length(tiny.head) < 100
    _insert!(tiny, 1500.0, 1500.0)
    @test _has_neighbor(tiny, 1500.005, 1500.0)
    @test !_has_neighbor(tiny, 1500.02, 1500.0)

    # Points and queries outside the coordinate extent are clamped into edge
    # cells, which is exact rather than approximate: a neighbor across the
    # boundary is still found, and a distant one is still rejected.
    edge = _SepGrid{Float64}(1.5, [1.0, 50.0], [1.0, 50.0])
    _insert!(edge, -0.4, -0.2)
    _insert!(edge, 51.0, 51.0)
    @test _has_neighbor(edge, 0.3, 0.5)
    @test _has_neighbor(edge, 50.5, 50.5)
    @test !_has_neighbor(edge, 25.0, 25.0)
    far = _SepGrid{Float64}(1.5, [1.0, 50.0], [1.0, 50.0])
    _insert!(far, -900.0, -900.0)
    @test !_has_neighbor(far, 0.5, 0.5)

    # An empty grid is well formed and matches nothing.
    empty_g = _SepGrid{Float64}(1.0, Float64[], Float64[])
    @test !_has_neighbor(empty_g, 5.0, 5.0)

    # Agreement with a brute-force fixed-radius test over a spilling field.
    rng = StableRNG(4242)
    n, sep = 3000, 1.5
    ys = rand(rng, n) .* 210 .- 5
    xs = rand(rng, n) .* 210 .- 5
    ref = falses(n)
    gg = _SepGrid{Float64}(sep, ys, xs)
    acc_y = Float64[]
    acc_x = Float64[]
    for j in 1:n
        hit = any(k -> (acc_y[k] - ys[j])^2 + (acc_x[k] - xs[j])^2 <= sep^2, eachindex(acc_y))
        ref[j] = hit
        if hit
            @test _has_neighbor(gg, ys[j], xs[j])
        else
            @test !_has_neighbor(gg, ys[j], xs[j])
            _insert!(gg, ys[j], xs[j])
            push!(acc_y, ys[j])
            push!(acc_x, xs[j])
        end
    end
    @test count(ref) > 0
end

@testset "Catalog" begin
    @testset "seeding formats" begin
        empty_cat = Catalog{Float64}(nothing, TEST_PSF)
        @test isempty(empty_cat)
        @test length(empty_cat) == 0

        nt = (; y = [10.0, 30.0, 20.0], x = [10.0, 30.0, 20.0], flux = [1.0, 3.0, 2.0])
        c_nt = Catalog{Float64}(nt, TEST_PSF)
        @test length(c_nt) == 3
        @test sort(c_nt.flux) == [1.0, 2.0, 3.0]
        @test all(==(0), c_nt.pass)             # warm-start sources carry pass 0
        @test all(isnan, c_nt.flux_snr)         # no SNR until a fit has run

        img, _, _ = test_field(; n = 20, seed = 11)
        mfr = matched_filter(img .- median(img), TEST_PSF; sigma = 8.0)
        c_mf = Catalog{Float64}(mfr, TEST_PSF)
        @test length(c_mf) == length(mfr.peaks)
        shapes = measure_star_shapes(mfr)
        c_sh = Catalog{Float64}(shapes, TEST_PSF)
        @test length(c_sh) == length(shapes)
    end

    @testset "subsetting copies every column together" begin
        c = Catalog{Float64}([1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0],
                             [10.0, 11.0, 12.0], [1, 1, 2])
        s = c[[true, false, true]]
        @test s.y == [1.0, 3.0] && s.x == [4.0, 6.0] && s.flux == [7.0, 9.0]
        @test s.flux_snr == [10.0, 12.0] && s.pass == [1, 2]
        # A subset is a copy, not a view: writing to it cannot corrupt the source.
        s.y[1] = -1.0
        @test c.y[1] == 1.0
    end

    @testset "Morton ordering and tie determinism" begin
        ys = [40.0, 10.0, 10.0, 25.0]
        xs = [40.0, 10.0, 10.0, 25.0]
        a = append_sources(Catalog{Float64}(), ys, xs, ones(4), 1)
        @test a.n_added == 4
        catalog = a.catalog
        @test issorted(_anchor_key2.(catalog.y, catalog.x))

        # Appending re-sorts and keeps every parallel vector together.
        cat3 = append_sources(catalog, [5.0], [5.0], [7.0], 2).catalog
        @test length(cat3) == 5
        j = findfirst(==(7.0), cat3.flux)
        @test cat3.pass[j] == 2
        @test cat3.y[j] == 5.0 && cat3.x[j] == 5.0
        @test issorted(_anchor_key2.(cat3.y, cat3.x))

        one_src = Catalog{Float64}([1.0], [1.0], [1.0], [1.0], [1])
        @test sort_morton(one_src) === one_src

        # Non-finite positions are rejected outright.
        @test append_sources(cat3, [NaN], [1.0], [1.0], 3).n_added == 0

        # Tie handling, on rows that are actually *distinguishable*.  Sources
        # sharing an anchor pixel share a Morton key, and `apply_J!`'s
        # overlapping-pixel scatter-add is order-dependent in floating point, so
        # the order among them has to be a function of the input rather than of
        # the sorting algorithm.  Two sources tied at flux 1.0 -- as the fixture
        # above has them -- cannot show this: permuting identical rows is
        # invisible, so `issorted` and a rerun comparison both pass under a sort
        # that reverses ties.  Give the tied pair distinct fluxes instead.
        tied = Catalog{Float64}([10.0, 10.0, 40.0], [10.0, 10.0, 40.0],
                                [11.0, 22.0, 33.0], fill(5.0, 3), fill(1, 3))
        st = sort_morton(tied)
        @test st.flux[1:2] == [11.0, 22.0]        # ties keep their input order
        # Swapping the input swaps the output, which is what pins the order to
        # the input rather than to some fixed rule of the sorter's own.
        swapped = sort_morton(tied[[2, 1, 3]])
        @test swapped.flux[1:2] == [22.0, 11.0]
        # Idempotent, so repeated passes cannot walk a tied pair around.
        @test sort_morton(st).flux == st.flux
    end

    @testset "batch dedup at min_separation" begin
        a = append_sources(Catalog{Float64}(), [10.0, 10.4, 30.0], [10.0, 10.4, 30.0],
                           ones(3), 1; min_separation = 1.5)
        @test a.n_added == 2
        @test length(a.catalog) == 2
    end
end

@testset "Discards and drop!" begin
    make() = Catalog{Float64}([10.0, 20.0, 30.0, 40.0], [10.0, 20.0, 30.0, 40.0],
                              [1.0, 2.0, 3.0, 4.0], fill(5.0, 4), fill(1, 4))
    disc = Discards{Float64}()
    @test length(disc) == 0

    catalog = make()
    kept = drop!(disc, catalog, [true, false, true, false], :snr, 2)
    @test kept.flux == [1.0, 3.0]
    @test catalog.flux == [1.0, 2.0, 3.0, 4.0]  # the input is untouched
    @test length(disc) == 2
    @test disc.reason == [:snr, :snr]
    @test disc.pass == [2, 2]
    @test disc.y == [20.0, 40.0] && disc.x == [20.0, 40.0]

    # A per-source reason vector is read only where `keep` is false.
    disc2 = Discards{Float64}()
    reasons = [:none, :close, :none, :no_pixels]
    drop!(disc2, make(), [true, false, true, false], reasons, 3)
    @test disc2.reason == [:close, :no_pixels]
    @test all(==(3), disc2.pass)

    # Dropping nothing logs nothing and returns the catalog itself.
    disc3 = Discards{Float64}()
    c = make()
    @test drop!(disc3, c, trues(4), :snr, 1) === c
    @test length(disc3) == 0

    # Dropping everything empties the catalog and logs all of it.
    disc4 = Discards{Float64}()
    @test isempty(drop!(disc4, make(), falses(4), :snr, 1))
    @test length(disc4) == 4

    @test_throws "keep mask length" drop!(Discards{Float64}(), make(), trues(3), :snr, 1)
end

@testset "prune_mask" begin
    # SNR cut, then the lower-SNR member of a close pair.
    catalog = Catalog{Float64}([10.0, 10.5, 40.0, 60.0], [10.0, 10.5, 40.0, 60.0],
                           ones(4), [20.0, 10.0, 1.0, 30.0], fill(1, 4))
    pr = prune_mask(catalog, 3.0, 1.0)
    @test pr.n_snr == 1                       # the SNR-1 source at y = 40
    @test pr.n_close == 1                     # the fainter of the y ~ 10 pair
    @test pr.keep == [true, false, false, true]
    @test pr.reasons == [:none, :close, :snr, :none]
    @test count(!, pr.keep) == pr.n_snr + pr.n_close

    # Greedy by descending SNR, so the pair's survivor does not depend on order.
    rev = catalog[[4, 3, 2, 1]]
    pr_rev = prune_mask(rev, 3.0, 1.0)
    @test Set(zip(rev.y[pr_rev.keep], rev.x[pr_rev.keep])) ==
          Set(zip(catalog.y[pr.keep], catalog.x[pr.keep]))

    # A non-finite SNR is cut like a low one.
    nanc = Catalog{Float64}([10.0], [10.0], [1.0], [NaN], [1])
    @test prune_mask(nanc, 3.0, 1.0).reasons == [:snr]

    # `separation = 0` disables the close-pair cut entirely.
    pr0 = prune_mask(catalog, 3.0, 0.0)
    @test pr0.n_close == 0
    @test pr0.keep == [true, true, false, true]
end

@testset "estimate_background_multipass" begin
    ny, nx = 80, 80
    level = 123.0
    img = fill(level, ny, nx)
    iv = fill(1.0, ny, nx)
    iv[1, 1] = 0.0
    iv[2, 2] = -1.0
    iv[3, 3] = NaN
    o = (; bkg_box_size = 10, bkg_box_size_coarse = 40, bkg_coarse_passes = 2,
           bkg_rms_box_size = 10, bkg_estimator = SExtractorBackground(),
           bkg_rms_estimator = MADStdRMS(), bkg_kws = (;), bkg_mask = nothing,
           coverage_mask = nothing, fixed_inv_var = iv)
    model = zeros(ny, nx)

    b1 = estimate_background_multipass(img, model, 1, o)
    @test b1.coarse
    @test b1.box == 40
    @test all(≈(level), b1.background)
    @test all(v -> abs(v) < 1e-9, b1.resid)
    # detect_inv_var inherits the caller's zeros / negatives / non-finites.
    @test b1.detect_inv_var[1, 1] == 0
    @test b1.detect_inv_var[2, 2] == 0
    @test b1.detect_inv_var[3, 3] == 0
    # A caller-supplied inv_var is used unchanged as the fit weights.
    @test b1.fit_inv_var === iv

    @test estimate_background_multipass(img, model, 2, o).coarse
    b3 = estimate_background_multipass(img, model, 3, o)
    @test !b3.coarse                            # coarse -> fine at pass > coarse_passes
    @test b3.box == 10

    # Without a caller inv_var the fit weights are the background-only weights.
    o2 = merge(o, (; bkg_coarse_passes = 0, fixed_inv_var = nothing))
    rng = StableRNG(5)
    noisy = level .+ 2.0 .* randn(rng, ny, nx)
    b = estimate_background_multipass(noisy, model, 1, o2)
    @test !b.coarse
    @test median(b.background) ≈ level atol = 0.5
    @test median(b.rms) ≈ 2.0 atol = 0.5
    @test b.fit_inv_var === b.detect_inv_var
    @test median(b.resid) ≈ 0.0 atol = 0.5

    # Each call returns fresh arrays; nothing is shared with the previous pass.
    @test b1.background !== b3.background
    @test b1.resid !== b3.resid
end

@testset "detect_sources" begin
    img, src, _ = test_field(; n = 40, seed = 31)
    ny, nx = size(img)
    R = 5
    kern = Matrix{Float64}(CrowdPhot.PSF.render!(Matrix{Float64}(undef, 2R + 1, 2R + 1),
        CircularGaussianPSF(R + 1.0, R + 1.0, PSF_FWHM, 1.0, 0.0), 1:(2R + 1), 1:(2R + 1)))
    o = (; bkg_box_size = 20, bkg_box_size_coarse = 80, bkg_coarse_passes = 1,
           bkg_rms_box_size = 20, bkg_estimator = SExtractorBackground(),
           bkg_rms_estimator = MADStdRMS(), bkg_kws = (;), bkg_mask = nothing,
           coverage_mask = nothing, fixed_inv_var = nothing,
           kernel = kern, detect_sigma = 5.0, normalize_zerosum = true,
           min_separation = 1.5, blend_threshold = nothing,
           blend_threshold_initial = nothing, blend_passes = 0,
           # As the driver resolves them: already an `Int`, never `nothing`.
           morph_half_width = max(3, ceil(Int, 3 * PSF_FWHM / (2 * sqrt(2 * log(2))))),
           psf_fwhm = PSF_FWHM)

    model = zeros(ny, nx)
    bkg = estimate_background_multipass(img, model, 1, o)
    disc = Discards{Float64}()
    d1 = detect_sources(Catalog{Float64}(), disc, bkg, model, 1, o)
    @test d1.mfr isa CrowdPhot.MatchedFilterResult
    @test d1.n_new == length(d1.catalog)
    @test d1.n_peaks >= d1.n_new
    @test d1.n_dup_catalog == 0 && d1.n_dup_discarded == 0
    @test 0.5 <= d1.n_new / length(src.y) <= 1.2
    @test all(==(1), d1.catalog.pass)

    # A second detection pass on the same (unchanged) residual must reject
    # essentially everything as already in the catalog.
    d2 = detect_sources(d1.catalog, disc, bkg, model, 2, o)
    @test d2.n_dup_catalog > 0
    @test d2.n_new < d1.n_new
    @test length(d2.catalog) == d1.n_new + d2.n_new
    # Discarded positions suppress re-detection just as catalog positions do.
    disc3 = Discards{Float64}()
    drop!(disc3, d1.catalog, falses(length(d1.catalog)), :snr, 1)
    d3 = detect_sources(Catalog{Float64}(), disc3, bkg, model, 3, o)
    @test d3.n_dup_discarded > 0

    # The blend gate is per-pass: `blend_passes` decides which threshold applies.
    o_gate = merge(o, (; blend_threshold_initial = 1e6, blend_passes = 1))
    model_bright = fill(1.0, ny, nx)
    g1 = detect_sources(Catalog{Float64}(), Discards{Float64}(), bkg, model_bright, 1, o_gate)
    @test g1.n_blend_rejected > 0
    g2 = detect_sources(Catalog{Float64}(), Discards{Float64}(), bkg, model_bright, 2, o_gate)
    @test g2.n_blend_rejected == 0            # past `blend_passes`, no gate
end

@testset "stamp_geometry" begin
    ny, nx = 40, 40
    w = ones(ny * nx)
    R = 2
    S2 = (2R + 1)^2

    catalog = Catalog{Float64}([20.0, 20.4], [20.0, 30.0], [1.0, 2.0], fill(5.0, 2), [1, 1])
    g = stamp_geometry(catalog, w, R, ny, nx)
    @test g.S2 == S2
    @test size(g.pixels) == (S2, 2)
    @test all(g.ok)
    @test g.anchor_y == [20, 20] && g.anchor_x == [20, 30]
    @test length(g.dy_off) == S2 && length(g.dx_off) == S2
    # Column-major stamp order, and every pixel resolves to its flat image index.
    mid = (S2 + 1) ÷ 2
    @test g.dy_off[mid] == 0 && g.dx_off[mid] == 0
    @test g.pixels[mid, 1] == 20 + (20 - 1) * ny
    @test all(>(0), g.pixels)

    # A source off the image edge keeps only its on-image pixels.
    edge = Catalog{Float64}([1.0], [1.0], [1.0], [5.0], [1])
    ge = stamp_geometry(edge, w, R, ny, nx)
    @test ge.ok == [true]
    @test count(!=(0), ge.pixels) == 9        # the 3x3 corner of a 5x5 stamp

    # Entirely off the image, or non-finite: no usable pixel at all.
    off = Catalog{Float64}([-50.0, NaN, 20.0], [20.0, 20.0, 20.0], [1.0, 1.0, NaN],
                           fill(5.0, 3), fill(1, 3))
    go = stamp_geometry(off, w, R, ny, nx)
    @test go.ok == [false, false, false]

    # Zero-weight pixels are excluded, and a fully masked source is not `ok`.
    wz = copy(w)
    for dy in -R:R, dx in -R:R
        wz[(20 + dy) + ((20 + dx) - 1) * ny] = 0.0
    end
    gz = stamp_geometry(catalog, wz, R, ny, nx)
    @test gz.ok == [false, true]
end

@testset "FitPlan" begin
    plan = FitPlan(TEST_PSF, (; fwhm = PSF_FWHM, bkg = 0.0))
    @test plan.free_names == [:y, :x, :flux]  # PSF property order, y before x
    @test plan.p == 3
    @test plan.grad_col == [1, 2, 3]
    @test (plan.k_y, plan.k_x, plan.k_flux) == (1, 2, 3)
    @test plan.fixed.bkg == 0.0

    # Fixing a position parameter drops it from `theta` but not from the gradient
    # row mapping, which indexes the PSF's full property tuple.
    pf = FitPlan(TEST_PSF, (; fwhm = PSF_FWHM, bkg = 0.0, y = 1.0))
    @test pf.free_names == [:x, :flux]
    @test pf.p == 2
    @test pf.k_y === nothing
    @test (pf.k_x, pf.k_flux) == (1, 2)
    @test pf.row_y == plan.row_y              # unchanged by what is free

    @test_throws "fits only (y, x, flux)" FitPlan(TEST_PSF, (; bkg = 0.0))
    @test_throws "`flux` must be free" FitPlan(TEST_PSF, (; fwhm = PSF_FWHM, bkg = 0.0, flux = 1.0))
    @test_throws "nothing to fit" FitPlan(TEST_PSF, (; y = 1.0, x = 1.0, fwhm = PSF_FWHM, flux = 1.0, bkg = 0.0))
end

@testset "theta round trip" begin
    plan = FitPlan(TEST_PSF, (; fwhm = PSF_FWHM, bkg = 0.0))
    catalog = Catalog{Float64}([10.0, 20.0], [30.0, 40.0], [100.0, 200.0], fill(NaN, 2), [1, 1])
    theta = theta_from_catalog(catalog, plan)
    @test theta == [10.0, 30.0, 100.0, 20.0, 40.0, 200.0]

    pf = FitPlan(TEST_PSF, (; fwhm = PSF_FWHM, bkg = 0.0, y = 1.0))
    @test theta_from_catalog(catalog, pf) == [30.0, 100.0, 40.0, 200.0]
end

@testset "argument validation" begin
    img = fill(1.0, 40, 40)
    @test_throws "max_iter must be positive" fit_all_stars_simultaneous_multipass(
        img, TEST_PSF, 3.0; fixed = TEST_FIXED, max_iter = 0)
    @test_throws "min_iter must be positive" fit_all_stars_simultaneous_multipass(
        img, TEST_PSF, 3.0; fixed = TEST_FIXED, min_iter = 0)
    @test_throws "fit_rad must be positive" fit_all_stars_simultaneous_multipass(
        img, TEST_PSF, 0.0; fixed = TEST_FIXED)
    @test_throws "few_sources must be non-negative" fit_all_stars_simultaneous_multipass(
        img, TEST_PSF, 3.0; fixed = TEST_FIXED, few_sources = -1)
    @test_throws "solver must be :lsqr or :lsmr" fit_all_stars_simultaneous_multipass(
        img, TEST_PSF, 3.0; fixed = TEST_FIXED, solver = :cg)
    # Only (y, x, flux) may be free; `bkg` is pinned automatically.
    @test_throws "fits only (y, x, flux)" fit_all_stars_simultaneous_multipass(
        img, TEST_PSF, 3.0)
    @test_throws ArgumentError fit_all_stars_simultaneous_multipass(
        img, TEST_PSF, 3.0; fixed = TEST_FIXED, inv_var = ones(3, 3))
    # A nonzero per-source pedestal would be added once per source footprint and
    # accumulate wherever footprints overlap.
    @test_throws "fixed.bkg must be zero" fit_all_stars_simultaneous_multipass(
        img, TEST_PSF, 3.0; fixed = (; TEST_FIXED..., bkg = 5.0))
end

@testset "empty result" begin
    img, src, bg = test_field(; ny = 80, nx = 80, n = 10)
    # Detection suppressed and no warm start: the residual is still the real
    # `image - background - model`, not a fabricated zero image.
    r = fit_all_stars_simultaneous_multipass(img, TEST_PSF, 3.0;
        fixed = (; TEST_FIXED..., bkg = 0.0), detect_sigma = 1.0e6)  # bkg = 0 is accepted
    @test isempty(r.phot.flux)
    @test size(r.phot.residual) == size(img)
    @test r.phot.residual ≈ img .- r.background.background

    # Every pixel masked: all warm-start sources are dropped for having no usable
    # pixels, and the run returns an empty result instead of throwing out of
    # `_model_radii` (whose positive-weight quantile would see an empty vector).
    catalog = (; y = src.y, x = src.x, flux = src.flux)
    r0 = fit_all_stars_simultaneous_multipass(img, TEST_PSF, catalog, 3.0;
        fixed = TEST_FIXED, inv_var = zeros(size(img)))
    @test isempty(r0.phot.flux)
    @test r0.phot.n_failed == length(src.y)
    @test r0.phot.residual ≈ img .- r0.background.background
end

@testset "pass loop scheduling" begin
    img, src, _ = test_field(; n = 60, seed = 77)

    # `max_iter = 1`: one detection pass with pruning, then the terminal pass.
    r1 = fit_all_stars_simultaneous_multipass(img, TEST_PSF, 4.0; fixed = TEST_FIXED,
        max_iter = 1)
    @test r1.n_detection_passes == 1
    @test length(r1.pass_history) == 2
    @test r1.pass_history[end].last_pass
    @test r1.pass_history[end].n_peaks == 0        # the terminal pass skips detection
    @test !r1.converged                            # stopped on the max_iter budget
    for h in r1.pass_history
        # Named buckets, not a partition of `t_fit`: cheap steps are left out, so they
        # may sum to less but never to more.  No tolerance on the difference, which is
        # wall-clock and would be flaky.
        @test keys(h.fit_timing) == (:setup, :stamps, :render, :solve)
        @test all(t -> isfinite(t) && t >= 0, values(h.fit_timing))
        @test sum(values(h.fit_timing)) <= h.t_fit
    end

    # A blank field converges on pass 2 rather than burning the budget: pass 1
    # detects nothing, 3k schedules the terminal pass, pass 2 exits converged.
    blank = fill(100.0, 100, 100)
    rb = fit_all_stars_simultaneous_multipass(blank, TEST_PSF, 4.0; fixed = TEST_FIXED,
        max_iter = 10, min_iter = 1)
    @test rb.converged
    @test length(rb.pass_history) == 2
    @test rb.n_detection_passes == 1
    @test isempty(rb.phot.y)
    @test isempty(rb.phot.morphology)
    @test size(rb.phot.residual) == size(blank)
    @test rb.background isa CrowdPhot.Background2D
    # An empty catalog goes through `empty_pass_stats`, which must offer the same
    # `fit_timing` keys or `pass_history` is heterogeneous within a run.
    @test all(h -> keys(h.fit_timing) == (:setup, :stamps, :render, :solve),
              rb.pass_history)

    # `min_iter > max_iter` is legal: the early-convergence clause never fires.
    r2 = fit_all_stars_simultaneous_multipass(img, TEST_PSF, 4.0; fixed = TEST_FIXED,
        max_iter = 2, min_iter = 10)
    @test !r2.converged
    @test r2.n_detection_passes == 2
    @test length(r2.pass_history) == 3

    # Early convergence on `few_sources`.
    r3 = fit_all_stars_simultaneous_multipass(img, TEST_PSF, 4.0; fixed = TEST_FIXED,
        max_iter = 10, min_iter = 1, few_sources = 1000)
    @test r3.converged
    @test r3.n_detection_passes == 1
end

@testset "end-to-end recovery" begin
    img, src, _ = test_field(; n = 90, seed = 20240905)
    res = fit_all_stars_simultaneous_multipass(img, TEST_PSF, 4.0; fixed = TEST_FIXED,
        max_iter = 4, min_iter = 2)
    ph = res.phot

    @test length(ph.y) == length(ph.morphology) == length(res.pass_number)
    @test all(ph.flux .> 0)
    @test all(isfinite, ph.flux_err)
    @test all(res.pass_number .>= 1)
    @test res.n_pruned >= 0
    @test res.detection isa CrowdPhot.MatchedFilterResult
    @test size(ph.residual) == size(img)

    # Completeness and photometric accuracy on truth.
    n_matched = 0
    dmag = Float64[]
    dpos = Float64[]
    for i in eachindex(src.y)
        d, j = findmin([hypot(ph.y[k] - src.y[i], ph.x[k] - src.x[i]) for k in eachindex(ph.y)])
        d <= 1.0 || continue
        n_matched += 1
        push!(dmag, -2.5 * log10(ph.flux[j] / src.flux[i]))
        push!(dpos, d)
    end
    completeness = n_matched / length(src.y)
    @test completeness > 0.9
    @test abs(median(dmag)) < 0.05
    @test median(abs.(dmag .- median(dmag))) < 0.1
    @test median(dpos) < 0.2
    # Spurious rate: output sources with no input match within 1 px.
    spurious = count(k -> minimum([hypot(ph.y[k] - src.y[i], ph.x[k] - src.x[i])
                                   for i in eachindex(src.y)]) > 1.0, eachindex(ph.y))
    @test spurious / length(ph.y) < 0.25

    # Determinism: two runs on identical input are bitwise identical.  This is
    # what the Morton tie-break exists for.
    res2 = fit_all_stars_simultaneous_multipass(img, TEST_PSF, 4.0; fixed = TEST_FIXED,
        max_iter = 4, min_iter = 2)
    @test res2.phot.y == ph.y
    @test res2.phot.x == ph.x
    @test res2.phot.flux == ph.flux
    @test res2.phot.flux_err == ph.flux_err
    @test res2.phot.residual == ph.residual
end

@testset "pass history and tracing" begin
    img, _, _ = test_field(; n = 50, seed = 909)
    res = fit_all_stars_simultaneous_multipass(img, TEST_PSF, 4.0; fixed = TEST_FIXED,
        max_iter = 3, min_iter = 1)
    hist = res.pass_history
    @test !isempty(hist)
    @test hist isa Vector{<:NamedTuple}
    for (k, r) in enumerate(hist)
        @test r.pass == k
        @test r.n_new_surviving <= r.n_new
        @test r.n_accepted <= r.n_trials
        # Every peak is classified into exactly one outcome: gated by the blend
        # test, a duplicate of a catalog source, a duplicate of a discarded
        # position, or inserted.  A double count or a dropped peak breaks this.
        @test r.n_blend_rejected + r.n_dup_catalog + r.n_dup_discarded + r.n_new ==
              r.n_peaks
        @test r.bkg_box > 0 && r.bkg_rms_box > 0
        @test r.t_background >= 0 && r.t_detect >= 0 && r.t_fit >= 0 && r.t_prune >= 0
        @test r.t_render >= 0
    end
    # Every phase of the call is timed: setup + the five per-pass buckets +
    # finalize.  Measured on the *same* call as the wall clock -- comparing one
    # call's timers against another call's wall time is only a measurement of
    # run-to-run noise.
    @test res.t_setup >= 0
    @test res.t_finalize >= 0
    total_traced(r) = r.t_setup + r.t_finalize +
        sum(p.t_background + p.t_detect + p.t_fit + p.t_prune + p.t_render for p in r.pass_history)
    wall = @elapsed timed = fit_all_stars_simultaneous_multipass(img, TEST_PSF, 4.0;
        fixed = TEST_FIXED, max_iter = 3, min_iter = 1)
    # The timers partition the call, so their sum can never exceed it, and must
    # account for the bulk of it rather than a sliver.
    @test total_traced(timed) <= wall
    @test total_traced(timed) >= 0.5 * wall
    @test hist[1].bkg_coarse                       # bkg_coarse_passes defaults to 2
    # The final non-finite/non-positive-flux gate runs after the last pass, so the
    # returned catalog can be smaller than the last report's count but never larger.
    @test hist[end].n_catalog >= length(res.phot.y)

    # The trace prints from the same report the caller gets back, so tracing must
    # not change the numbers.
    tracefile = tempname()
    res_t = open(tracefile, "w") do io
        redirect_stdout(io) do
            fit_all_stars_simultaneous_multipass(img, TEST_PSF, 4.0; fixed = TEST_FIXED,
                max_iter = 3, min_iter = 1, show_trace = true)
        end
    end
    s = read(tracefile, String)
    rm(tracefile; force = true)
    @test occursin("background", s)
    @test occursin("detection", s)
    @test occursin("terminal pass", s)
    @test occursin("multipass summary", s)
    @test occursin(string(hist[1].n_peaks), s)
    @test res_t.phot.flux == res.phot.flux
    @test [r.n_new for r in res_t.pass_history] == [r.n_new for r in hist]
end

@testset "options" begin
    img, _, _ = test_field(; n = 50, seed = 3131)
    iv = fill(1 / 81.0, size(img))

    # A supplied `inv_var` reaches both the fit and the covariance.  Scaling it
    # uniformly leaves the weighted least-squares solution untouched but scales
    # every reported error by `sqrt(100) = 10`, which only holds if the map got
    # all the way through to `_source_errors!`.  Detection is unaffected because
    # it reads the RMS-mesh weights, not these.  `model_rad` is pinned because
    # `:auto` sizes the model box from the noise level it reads out of `inv_var`,
    # which would otherwise change the geometry along with the scale.
    scale_kw = (; fixed = TEST_FIXED, max_iter = 2, min_iter = 1, prune = false,
                  model_rad = 6.0)
    r_hi = fit_all_stars_simultaneous_multipass(img, TEST_PSF, 4.0; inv_var = iv, scale_kw...)
    r_lo = fit_all_stars_simultaneous_multipass(img, TEST_PSF, 4.0; inv_var = iv ./ 100,
        scale_kw...)
    @test length(r_lo.phot.y) == length(r_hi.phot.y) > 0
    @test r_lo.phot.flux ≈ r_hi.phot.flux rtol = 1e-12
    @test all(≈(10.0; rtol = 1e-6), r_lo.phot.flux_err ./ r_hi.phot.flux_err)
    @test all(≈(10.0; rtol = 1e-6), r_lo.phot.y_err ./ r_hi.phot.y_err)

    # A supplied `inv_var` is the bad-pixel mask too: zero-weight pixels are
    # excluded from detection as well as from the fit.
    iv_masked = copy(iv)
    iv_masked[:, 1:20] .= 0
    r_mask = fit_all_stars_simultaneous_multipass(img, TEST_PSF, 4.0; fixed = TEST_FIXED,
        max_iter = 2, min_iter = 1, inv_var = iv_masked)
    @test all(r_mask.phot.x .> 15)

    # Reference run for the two comparisons below.
    r_def = fit_all_stars_simultaneous_multipass(img, TEST_PSF, 4.0; fixed = TEST_FIXED,
        max_iter = 1, bkg_kws = (; filter_size = (3, 3), sigma = 3.0))

    # `bkg_kws` reach the estimator: a different median `filter_size` has to
    # produce a different background map, not merely be accepted.
    r_kws = fit_all_stars_simultaneous_multipass(img, TEST_PSF, 4.0; fixed = TEST_FIXED,
        max_iter = 1, bkg_kws = (; filter_size = (5, 5), sigma = (2.0, 5.0)))
    @test length(r_kws.phot.y) > 0
    @test maximum(abs.(r_kws.background.background .- r_def.background.background)) > 1e-3

    # An explicit detection kernel is the one actually correlated against.
    # `matched_filter` stores the zero-sum-normalized kernel rather than the
    # input, so compare against that transform: `K = (P - mean(P)) / (sum(P^2) -
    # sum(P)^2/N)`.  The default kernel is rendered over `+-kernel_rad`, so its
    # shape differs too.
    kern = CrowdPhot.PSF.render(CrowdPhot.PSF.CircularGaussianPSF(0.0, 0.0, PSF_FWHM, 1.0, 0.0))
    r_kern = fit_all_stars_simultaneous_multipass(img, TEST_PSF, 4.0; fixed = TEST_FIXED,
        max_iter = 1, detection_kernel = kern)
    @test length(r_kern.phot.y) > 0
    @test size(r_kern.detection.kernel) == size(kern) != size(r_def.detection.kernel)
    n_k = length(kern)
    @test r_kern.detection.kernel ≈
          (kern .- sum(kern) / n_k) ./ (sum(abs2, kern) - sum(kern)^2 / n_k) rtol = 1e-12

    # `prune = false` keeps everything the fit produced.
    r_np = fit_all_stars_simultaneous_multipass(img, TEST_PSF, 4.0; fixed = TEST_FIXED,
        max_iter = 2, min_iter = 1, prune = false)
    @test all(r.n_pruned_snr == 0 && r.n_pruned_close == 0 for r in r_np.pass_history)
end

@testset "blend gate schedule" begin
    # Gating is a per-pass property: whichever threshold is in force on a pass
    # decides it, and `nothing` means that pass is not gated.  Neither knob is a
    # master switch, so all four combinations must be expressible.
    img, _, _ = test_field(; ny = 200, nx = 200, n = 120, seed = 4242)
    base = (; fixed = TEST_FIXED, max_iter = 5, min_iter = 5, few_sources = 0)
    rejected(r) = [p.n_blend_rejected for p in r.pass_history]
    go(; kws...) = fit_all_stars_simultaneous_multipass(img, TEST_PSF, 4.0; base..., kws...)

    # Both `nothing` (the default): never gated.
    off = go()
    @test all(iszero, rejected(off))

    # Steady-state only: gated on the later passes, not the early ones.
    late = go(blend_threshold = 0.2, blend_threshold_initial = nothing, blend_passes = 3)
    @test all(iszero, rejected(late)[1:3])
    @test any(!iszero, rejected(late)[4:end])

    # Initial only: gated early, then off -- the schedule that anneals the gate
    # away, which the old master-switch API could not express.
    early = go(blend_threshold = nothing, blend_threshold_initial = 2.0, blend_passes = 3)
    @test any(!iszero, rejected(early)[2:3])
    @test all(iszero, rejected(early)[4:end])

    # Both set: gated throughout.
    both = go(blend_threshold = 0.2, blend_passes = 3)
    @test any(!iszero, rejected(both)[2:3])
    @test any(!iszero, rejected(both)[4:end])
    # `blend_threshold_initial` defaults to 2.0 once `blend_threshold` is set, so
    # `both` and `early` must gate the early passes identically.
    @test rejected(both)[1:3] == rejected(early)[1:3]

    # Pass 1 can never gate: the model is all zeros when it detects, so
    # `S_model = 0` and every peak clears any finite threshold.
    @test rejected(go(blend_threshold = 1.0e6, blend_passes = 0))[1] == 0

    # A zero threshold is equivalent to no gate, since every candidate peak has
    # significance >= detect_sigma > 0.  (The old idiom for "off"; still valid.)
    zero_thr = go(blend_threshold = 0.0, blend_passes = 0)
    @test zero_thr.phot.flux == off.phot.flux
    @test zero_thr.phot.y == off.phot.y
end

@testset "GriddedPSFModel" begin
    # The auto-rendered detection kernel and the specialized `_fill_stamps!` /
    # `_render_model!` paths all take a different branch for a gridded PSF.  All
    # four grid nodes here are the *same* `GaussianPRF`, so the blended model is
    # that PRF everywhere and the gridded run has to reproduce the single-PSF run
    # exactly: any error in the node interpolation, in the gridded render, or in
    # the specialized stamp fill shows up as a disagreement below.  Agreement is
    # at the 1e-16 level in practice, so the tolerances carry ~1e6 of headroom
    # while a real interpolation bug would be orders of magnitude larger.
    node() = GaussianPRF(y = 0.0, x = 0.0, y_fwhm = PSF_FWHM, x_fwhm = PSF_FWHM,
                         theta = 0.0, flux = 1.0, bkg = 0.0)
    gm = GriddedPSFModel([node() for _ in 1:4], [0.0, 0.0, 200.0, 200.0],
        [0.0, 200.0, 0.0, 200.0]; y = 0.0, x = 0.0, flux = 1.0, bkg = 0.0)
    img, src, _ = test_field(; ny = 150, nx = 150, n = 40, seed = 555)
    kw = (; fixed = (; y_fwhm = PSF_FWHM, x_fwhm = PSF_FWHM, theta = 0.0),
            max_iter = 2, min_iter = 1)
    rg = fit_all_stars_simultaneous_multipass(img, gm, 4.0; kw...)
    rs = fit_all_stars_simultaneous_multipass(img, node(), 4.0; kw...)

    @test size(rg.detection.kernel) == (11, 11)     # rendered from the gridded PSF
    @test rg.detection.kernel ≈ rs.detection.kernel rtol = 1e-12
    @test length(rg.phot.y) == length(rs.phot.y) > 0.5 * length(src.y)
    @test rg.phot.y ≈ rs.phot.y rtol = 1e-10
    @test rg.phot.x ≈ rs.phot.x rtol = 1e-10
    @test rg.phot.flux ≈ rs.phot.flux rtol = 1e-10
    @test rg.phot.flux_err ≈ rs.phot.flux_err rtol = 1e-9
    @test all(isfinite, rg.phot.flux_err)
end

@testset "freeze_positions_final" begin
    # Warm-start from truth positions offset by 0.6 px with detection suppressed:
    # with the terminal pass frozen, only pass 1 can move the centroids, so the
    # residual offsets must be larger than when it is free.
    rng = StableRNG(616)
    img, src = simulate_image((120, 120), TEST_PSF, 30; background = 100.0,
        flux = (5000.0, 20000.0), rng, border = 12)
    imgbs = img .- 100.0
    iv = fill(1 / 100.0, size(imgbs))
    warm = (; y = src.y .+ 0.6, x = src.x .+ 0.6, flux = src.flux)
    kws = (; fixed = TEST_FIXED, inv_var = iv, detect_sigma = 1e9, max_iter = 1,
             min_iter = 1, prune = false, max_step = 0.25)
    free = fit_all_stars_simultaneous_multipass(imgbs, TEST_PSF, warm, 4.0;
        freeze_positions_final = false, kws...)
    frozen = fit_all_stars_simultaneous_multipass(imgbs, TEST_PSF, warm, 4.0;
        freeze_positions_final = true, kws...)
    @test length(free.phot.y) == length(frozen.phot.y) == length(src.y)
    err(r) = median([minimum([hypot(r.phot.y[j] - src.y[i], r.phot.x[j] - src.x[i])
                              for j in eachindex(r.phot.y)]) for i in eachindex(src.y)])
    @test err(frozen) > err(free)
end

@testset "morphology box geometry" begin
    # Noiseless frame whose sources *are* the PSF, warm started from truth with
    # detection suppressed, so the fit lands essentially on the input positions.
    # The morphology box is centered on the fit anchor and is shared with the
    # `psf_ref` render, so the normalized ratios must sit at 1 and the reported
    # centroid must be in global coordinates matching the fitted position.  An
    # off-by-one in the box origin, in `_inner_view`, or in the `y_offset` /
    # `x_offset` handed to `measure_star_shape_ref` shifts the centroid by a
    # whole pixel and pulls the ratios well off 1.
    #
    # The last source sits close enough to the corner that `morph_half_width`
    # clamps its morphology box on both axes while its fitting box stays inside
    # the frame: `psf_ref` is clamped identically, so the ratios still hold.
    src = (; y = [30.3, 50.7, 70.5, 6.4], x = [30.6, 55.2, 71.8, 6.6],
             flux = fill(3.0e4, 4))
    # A small amount of seeded Gaussian noise, and the background handed over
    # un-subtracted so the fitter estimates it itself.  The morphology weights
    # are background-only, so a perfectly noiseless frame would give zero RMS,
    # those weights would be guarded to zero, and every weighted moment would
    # collapse.  `read_noise = 0.1` against ~2800-count peaks is enough for the
    # RMS mesh to measure while contaminating the ratios below only at the
    # 1e-3 level.
    rng = StableRNG(90210)
    img = simulate_image((100, 100), TEST_PSF, src; background = 100.0,
        noise = :gaussian, read_noise = 0.1, rng)
    iv = fill(1 / 100.0, size(img))
    res = fit_all_stars_simultaneous_multipass(img, TEST_PSF, src, 4.0;
        fixed = TEST_FIXED, inv_var = iv, detect_sigma = 1e9, prune = false,
        max_iter = 3, min_iter = 1, morph_half_width = 8)
    ph = res.phot
    @test length(ph.morphology) == length(ph.y) == 4

    for (j, m) in enumerate(ph.morphology)
        # A box centered anywhere else would put the peak outside the +-1 search.
        @test abs(m.pixel[1] - round(Int, ph.y[j])) <= 1
        @test abs(m.pixel[2] - round(Int, ph.x[j])) <= 1
        # Lifted back to global coordinates by `y_offset`/`x_offset`; an
        # off-by-one there lands a whole pixel out.  The 0.02 px actually seen is
        # the quadratic-fit centroid's own bias, not a geometry error.
        @test m.centroid.y ≈ ph.y[j] atol = 0.1
        @test m.centroid.x ≈ ph.x[j] atol = 0.1
        # These ratios do not test *where* the box is -- measurement and
        # reference share it -- they test that `clean` and the `rend` handed
        # alongside it describe the same pixels, which is `_inner_view`'s
        # offset arithmetic.  They agree to ~1e-3 here, set by the background
        # ripple above; a one-pixel mismatch would move them by a few percent.
        @test m.sharpness / m.psf_ref.sharpness ≈ 1 rtol = 1.0e-2
        @test m.aperture.fwhm.y / m.psf_ref.aperture.fwhm.y ≈ 1 rtol = 1.0e-2
        @test m.aperture.fwhm.x / m.psf_ref.aperture.fwhm.x ≈ 1 rtol = 1.0e-2
        @test m.aperture.aperture_sum / m.psf_ref.aperture.aperture_sum ≈ 1 rtol = 1.0e-2
        @test isfinite(m.aperture.ellipticity1_aperture)
        # The uncertainties reach the caller with the background-only weights
        # `finalize_multipass` hard-codes, so they must be real numbers rather
        # than the NaN an unweighted path would give.
        @test isfinite(m.sharpness_err) && m.sharpness_err > 0
        @test isfinite(m.core.normalized_curvature_err) && m.core.normalized_curvature_err > 0
    end
end

@testset "morphology box clamped to nothing" begin
    # `morph_half_width` below `fit_rad` lets the morphology box clamp to zero
    # pixels for a source whose anchor has drifted off the frame while its wider
    # fit box still holds a weighted pixel, so `stamp_geometry` keeps it.  Every
    # statistic must come back `NaN` rather than throwing on the empty cutout.
    rng = StableRNG(19937)
    src = (; y = [-2.0, 50.7], x = [30.6, 55.2], flux = [3.0e4, 3.0e4])
    img = simulate_image((100, 100), TEST_PSF, src; background = 100.0,
        noise = :gaussian, read_noise = 0.1, rng)
    iv = fill(1 / 100.0, size(img))
    res = fit_all_stars_simultaneous_multipass(img, TEST_PSF, src, 4.0;
        fixed = TEST_FIXED, inv_var = iv, detect_sigma = 1e9, prune = false,
        max_iter = 1, min_iter = 1, morph_half_width = 1)
    ph = res.phot
    @test length(ph.morphology) == 2
    # The off-frame source is the one whose box is empty; the in-frame one is
    # measured normally, which is what makes this a guard and not a bypass.
    off = findfirst(<(0), ph.y)
    @test off !== nothing
    m = ph.morphology[off]
    @test isnan(m.sharpness) && isnan(m.sharpness_err)
    @test isnan(m.ellipticity_sq_resid) && isnan(m.ellipticity_sq_resid_err)
    @test isnan(m.aperture.ellipticity1_aperture)
    @test isnan(m.core.compactness_core)
    @test m.aperture.aperture_area == 0
    @test m.aperture.moment_norm == 0
    on = off == 1 ? 2 : 1
    @test isfinite(ph.morphology[on].sharpness)
    @test isfinite(ph.morphology[on].aperture.ellipticity1_aperture)
end
