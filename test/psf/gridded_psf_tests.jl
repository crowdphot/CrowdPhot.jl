import CrowdPhot
using CrowdPhot.PSF: GriddedPSFModel, GaussianPRF, CircularGaussianPRF, CircularGaussianPSF, ImagePSF, AbstractPSFModel, evaluate, evaluate_fg, extent, centroid, integral, background, peak, render, render!, add_star!, subtract_star!, fit_star
import ConstructionBase
using StableRNGs: StableRNG
using Test

# Stage 1: construction and validation of GriddedPSFModel only.
# Stage 2: `evaluate` (bilinear node blending).
# Stage 3: `evaluate_fg` (analytic gradients wrt y, x, flux, bkg).
# Stage 4: `extent` (union of active corners' recentered extents), and the
# generic centroid/integral/background/peak/render/add_star!/subtract_star!
# API surface (all work for free via the generic AbstractPSFModel defaults).
# Stage 5: `fit_star` integration (no source changes; the generic
# `AbstractPSFModel` fitting machinery works once `evaluate_fg` and the
# `ConstructionBase` interface exist).

@testset "GriddedPSFModel construction and validation" begin
    psfs4 = [GaussianPRF(y = 0.0, x = 0.0, y_fwhm = 3.0, x_fwhm = 3.0, theta = 0.0, flux = 1.0, bkg = 0.0) for _ in 1:4]
    gy4 = [0.0, 0.0, 10.0, 10.0]
    gx4 = [0.0, 10.0, 0.0, 10.0]

    @testset "basic 2x2 grid" begin
        m = GriddedPSFModel(psfs4, gy4, gx4; y = 5.0, x = 5.0, flux = 100.0, bkg = 1.0)
        @test m isa GriddedPSFModel{Float64, GaussianPRF{Float64}}
        @test m.ygrid == [0.0, 10.0]
        @test m.xgrid == [0.0, 10.0]
        @test m.index_grid == [1 2; 3 4]
        @test m.y == 5.0 && m.x == 5.0 && m.flux == 100.0 && m.bkg == 1.0
        @test m.psfs == psfs4
        @test m.psfs !== psfs4 # owned copy, not aliased
    end

    @testset "ConstructionBase getproperties/setproperties" begin
        m = GriddedPSFModel(psfs4, gy4, gx4; y = 5.0, x = 5.0, flux = 100.0, bkg = 1.0)
        @test ConstructionBase.getproperties(m) == (y = 5.0, x = 5.0, flux = 100.0, bkg = 1.0)
        m2 = ConstructionBase.setproperties(m, (y = 1.0, x = 2.0))
        @test m2.y == 1.0 && m2.x == 2.0
        @test m2.flux == m.flux && m2.bkg == m.bkg
        # Updating fit parameters should reuse the immutable model's node PSFs
        # and grid metadata instead of making a copy (mirrors ImagePSF).
        @test m2.psfs === m.psfs
        @test m2.grid_y === m.grid_y
        @test m2.grid_x === m.grid_x
        @test m2.ygrid === m.ygrid
        @test m2.xgrid === m.xgrid
        @test m2.index_grid === m.index_grid
    end

    @testset "defaults" begin
        m = GriddedPSFModel(psfs4, gy4, gx4)
        @test m.y == 0.0 && m.x == 0.0 && m.flux == 1.0 && m.bkg == 0.0
    end

    @testset "N == 1 skips rectangular-grid requirement" begin
        m1 = GriddedPSFModel([psfs4[1]], [5.0], [5.0])
        @test m1.index_grid == reshape([1], 1, 1)
        @test m1.ygrid == [5.0] && m1.xgrid == [5.0]
    end

    @testset "mismatched vector lengths error" begin
        @test_throws "same length" GriddedPSFModel(psfs4, gy4[1:3], gx4)
        @test_throws "same length" GriddedPSFModel(psfs4, gy4, gx4[1:3])
    end

    @testset "non-rectangular grid errors" begin
        # Only 3 of the 4 corners of a 2x2 grid supplied.
        @test_throws "rectangular" GriddedPSFModel(psfs4[1:3], gy4[1:3], gx4[1:3])
    end

    @testset "duplicate node position errors" begin
        @test_throws "duplicate" GriddedPSFModel(psfs4, [0.0, 0.0, 0.0, 10.0], gx4)
    end

    @testset "1D grid (only one axis varies) errors unless N == 1" begin
        @test_throws "at least 2 unique" GriddedPSFModel(psfs4, [0.0, 0.0, 10.0, 10.0], zeros(4))
    end

    @testset "empty psfs errors" begin
        @test_throws "non-empty" GriddedPSFModel(GaussianPRF{Float64}[], Float64[], Float64[])
    end

    @testset "mixed eltype nodes error" begin
        mixed = [
            GaussianPRF(y = 0.0, x = 0.0, y_fwhm = 3.0, x_fwhm = 3.0, theta = 0.0, flux = 1.0, bkg = 0.0),
            GaussianPRF(y = 0.0f0, x = 0.0f0, y_fwhm = 3.0f0, x_fwhm = 3.0f0, theta = 0.0f0, flux = 1.0f0, bkg = 0.0f0),
        ]
        @test_throws "element type" GriddedPSFModel(mixed, [0.0, 10.0], [0.0, 10.0])
    end

    @testset "unparameterized abstract vector errors with actionable message" begin
        het = AbstractPSFModel[
            GaussianPRF(y = 0.0, x = 0.0, y_fwhm = 3.0, x_fwhm = 3.0, theta = 0.0, flux = 1.0, bkg = 0.0),
            CircularGaussianPRF(y = 0.0, x = 10.0, fwhm = 3.0, flux = 1.0, bkg = 0.0),
        ]
        @test_throws "must be a subtype of" GriddedPSFModel(het, [0.0, 10.0], [0.0, 10.0])
    end

    @testset "heterogeneous node types with a shared T work" begin
        het2 = AbstractPSFModel{Float64}[
            GaussianPRF(y = 0.0, x = 0.0, y_fwhm = 3.0, x_fwhm = 3.0, theta = 0.0, flux = 1.0, bkg = 0.0),
            GaussianPRF(y = 0.0, x = 10.0, y_fwhm = 3.0, x_fwhm = 3.0, theta = 0.0, flux = 1.0, bkg = 0.0),
            CircularGaussianPRF(y = 10.0, x = 0.0, fwhm = 3.0, flux = 1.0, bkg = 0.0),
            CircularGaussianPRF(y = 10.0, x = 10.0, fwhm = 3.0, flux = 1.0, bkg = 0.0),
        ]
        m = GriddedPSFModel(het2, [0.0, 0.0, 10.0, 10.0], [0.0, 10.0, 0.0, 10.0])
        @test m isa GriddedPSFModel{Float64, AbstractPSFModel{Float64}}
    end

    @testset "non-finite grid positions error" begin
        @test_throws "finite" GriddedPSFModel(psfs4, [0.0, 0.0, 10.0, NaN], gx4)
        @test_throws "finite" GriddedPSFModel(psfs4, gy4, [0.0, Inf, 0.0, 10.0])
    end

    @testset "non-uniform grid spacing is allowed" begin
        gy_nonuniform = [0.0, 0.0, 3.0, 3.0]
        gx_nonuniform = [0.0, 25.0, 0.0, 25.0]
        m = GriddedPSFModel(psfs4, gy_nonuniform, gx_nonuniform)
        @test m.ygrid == [0.0, 3.0]
        @test m.xgrid == [0.0, 25.0]
    end

    @testset "3x2 rectangular grid" begin
        psfs6 = [GaussianPRF(y = 0.0, x = 0.0, y_fwhm = 3.0, x_fwhm = 3.0, theta = 0.0, flux = 1.0, bkg = 0.0) for _ in 1:6]
        gy6 = [0.0, 0.0, 5.0, 5.0, 10.0, 10.0]
        gx6 = [0.0, 10.0, 0.0, 10.0, 0.0, 10.0]
        m = GriddedPSFModel(psfs6, gy6, gx6)
        @test m.ygrid == [0.0, 5.0, 10.0]
        @test m.xgrid == [0.0, 10.0]
        @test size(m.index_grid) == (3, 2)
        @test sort(vec(m.index_grid)) == 1:6
    end

    @testset "node models must follow (y, x, ..., flux, bkg) property order" begin
        struct _BadOrderPSF{T} <: AbstractPSFModel{T}
            x::T
            y::T
            flux::T
            bkg::T
        end
        ConstructionBase.getproperties(m::_BadOrderPSF) = (x = m.x, y = m.y, flux = m.flux, bkg = m.bkg)
        bad = [_BadOrderPSF(0.0, 0.0, 1.0, 0.0), _BadOrderPSF(10.0, 0.0, 1.0, 0.0), _BadOrderPSF(0.0, 10.0, 1.0, 0.0), _BadOrderPSF(10.0, 10.0, 1.0, 0.0)]
        @test_throws "properties ordered" GriddedPSFModel(bad, [0.0, 0.0, 10.0, 10.0], [0.0, 10.0, 0.0, 10.0])
    end
end

@testset "GriddedPSFModel evaluate (bilinear node blending)" begin
    # Four distinct-shape unit-flux Gaussian nodes on a 2x2 grid; distinct
    # FWHM per node makes it easy to tell which node(s) contributed.
    mkpsf(fwhm) = GaussianPRF(y = 0.0, x = 0.0, y_fwhm = fwhm, x_fwhm = fwhm, theta = 0.0, flux = 1.0, bkg = 0.0)
    psfs = [mkpsf(3.0), mkpsf(6.0), mkpsf(1.0), mkpsf(9.0)] # (0,0) (0,10) (10,0) (10,10)
    gy = [0.0, 0.0, 10.0, 10.0]
    gx = [0.0, 10.0, 0.0, 10.0]

    @testset "exact recovery at each node position" begin
        for (i, (Y, X)) in enumerate(zip(gy, gx))
            m = GriddedPSFModel(psfs, gy, gx; y = Y, x = X, flux = 100.0, bkg = 1.0)
            recentered = ConstructionBase.setproperties(psfs[i], (y = Y, x = X, flux = 1.0, bkg = 0.0))
            for (py, px) in ((Y, X), (Y + 1.3, X - 0.7))
                @test evaluate(m, py, px) ≈ 100.0 * evaluate(recentered, py, px) + 1.0
            end
        end
    end

    @testset "manual bilinear blend at interior point" begin
        Y, X = 2.5, 7.5
        m = GriddedPSFModel(psfs, gy, gx; y = Y, x = X, flux = 100.0, bkg = 1.0)
        w_ll = (10 - X) * (10 - Y) / 100
        w_lr = (X - 0) * (10 - Y) / 100
        w_ul = (10 - X) * (Y - 0) / 100
        w_ur = (X - 0) * (Y - 0) / 100
        @test w_ll + w_lr + w_ul + w_ur ≈ 1.0
        py, px = 3.0, 8.0
        manual = 100 * (
            w_ll * evaluate(ConstructionBase.setproperties(psfs[1], (y = Y, x = X, flux = 1.0, bkg = 0.0)), py, px) +
                w_lr * evaluate(ConstructionBase.setproperties(psfs[2], (y = Y, x = X, flux = 1.0, bkg = 0.0)), py, px) +
                w_ul * evaluate(ConstructionBase.setproperties(psfs[3], (y = Y, x = X, flux = 1.0, bkg = 0.0)), py, px) +
                w_ur * evaluate(ConstructionBase.setproperties(psfs[4], (y = Y, x = X, flux = 1.0, bkg = 0.0)), py, px)
        ) + 1.0
        @test evaluate(m, py, px) ≈ manual
    end

    @testset "weights sum to 1 and are non-negative across many interior points" begin
        for Y in range(0.0, 10.0; length = 11), X in range(0.0, 10.0; length = 11)
            corners = CrowdPhot.PSF._grid_corners(GriddedPSFModel(psfs, gy, gx), Y, X)
            ws = last.(corners)
            @test all(≥(0), ws)
            @test sum(ws) ≈ 1.0
        end
    end

    @testset "outside-grid positions freeze the shape blend at the nearest edge cell" begin
        for (Y, X, corner_idx) in ((-50.0, -50.0, 1), (-50.0, 50.0, 2), (50.0, -50.0, 3), (50.0, 50.0, 4))
            m = GriddedPSFModel(psfs, gy, gx; y = Y, x = X, flux = 100.0, bkg = 1.0)
            recentered = ConstructionBase.setproperties(psfs[corner_idx], (y = Y, x = X, flux = 1.0, bkg = 0.0))
            @test evaluate(m, Y, X) ≈ 100.0 * evaluate(recentered, Y, X) + 1.0
        end
        # A point far outside along only one axis should blend the two nodes
        # on the nearest edge (not freeze to a single corner).
        m = GriddedPSFModel(psfs, gy, gx; y = -50.0, x = 5.0, flux = 100.0, bkg = 1.0)
        corners = CrowdPhot.PSF._grid_corners(m, -50.0, 5.0)
        active = [w for (_, w) in corners if w != 0]
        @test length(active) == 2
    end

    @testset "N == 1 constant PSF everywhere" begin
        m1 = GriddedPSFModel([psfs[1]], [5.0], [5.0]; y = -30.0, x = 200.0, flux = 7.0, bkg = 2.0)
        recentered = ConstructionBase.setproperties(psfs[1], (y = -30.0, x = 200.0, flux = 1.0, bkg = 0.0))
        @test evaluate(m1, -30.0, 200.0) ≈ 7.0 * evaluate(recentered, -30.0, 200.0) + 2.0
    end

    @testset "ImagePSF nodes work identically to analytic nodes" begin
        data1 = [exp(-((i - 4)^2 + (j - 4)^2) / 5.0) for i in 1:7, j in 1:7]
        data2 = [exp(-((i - 4)^2 + (j - 4)^2) / 20.0) for i in 1:7, j in 1:7]
        img_psfs = [
            ImagePSF(data1; y = 0.0, x = 0.0, flux = 1.0, bkg = 0.0, oversampling = 2, normalize = true),
            ImagePSF(data2; y = 0.0, x = 0.0, flux = 1.0, bkg = 0.0, oversampling = 2, normalize = true),
            ImagePSF(data2; y = 0.0, x = 0.0, flux = 1.0, bkg = 0.0, oversampling = 2, normalize = true),
            ImagePSF(data1; y = 0.0, x = 0.0, flux = 1.0, bkg = 0.0, oversampling = 2, normalize = true),
        ]
        img_gy = [0.0, 0.0, 10.0, 10.0]
        img_gx = [0.0, 10.0, 0.0, 10.0]
        m = GriddedPSFModel(img_psfs, img_gy, img_gx; y = 0.0, x = 3.0, flux = 50.0, bkg = 0.5)
        w_lr = 3.0 / 10.0
        w_ll = 1 - w_lr
        r1 = ConstructionBase.setproperties(img_psfs[1], (y = 0.0, x = 3.0, flux = 1.0, bkg = 0.0))
        r2 = ConstructionBase.setproperties(img_psfs[2], (y = 0.0, x = 3.0, flux = 1.0, bkg = 0.0))
        expected = 50.0 * (w_ll * evaluate(r1, 1.0, 2.0) + w_lr * evaluate(r2, 1.0, 2.0)) + 0.5
        @test evaluate(m, 1.0, 2.0) ≈ expected
    end
end

@testset "GriddedPSFModel evaluate_fg (analytic gradients)" begin
    # Reuse a 2x2 grid of Gaussians with distinct shapes so the interpolated
    # gradient genuinely depends on position, not just a single node's shape.
    psfs = [
        GaussianPRF(y = 0.0, x = 0.0, y_fwhm = 3.0, x_fwhm = 3.0, theta = 0.0, flux = 1.0, bkg = 0.0),
        GaussianPRF(y = 0.0, x = 0.0, y_fwhm = 5.0, x_fwhm = 4.0, theta = 0.2, flux = 1.0, bkg = 0.0),
        GaussianPRF(y = 0.0, x = 0.0, y_fwhm = 4.0, x_fwhm = 6.0, theta = -0.3, flux = 1.0, bkg = 0.0),
        GaussianPRF(y = 0.0, x = 0.0, y_fwhm = 6.0, x_fwhm = 5.0, theta = 0.5, flux = 1.0, bkg = 0.0),
    ]
    gy = [0.0, 0.0, 10.0, 10.0]
    gx = [0.0, 10.0, 0.0, 10.0]

    function fd_gradient(model, py, px; h = 1.0e-6)
        p0 = (model.y, model.x, model.flux, model.bkg)
        fd = zeros(4)
        for k in 1:4
            pplus = collect(Float64, p0)
            pminus = collect(Float64, p0)
            pplus[k] += h
            pminus[k] -= h
            mplus = ConstructionBase.setproperties(model, (y = pplus[1], x = pplus[2], flux = pplus[3], bkg = pplus[4]))
            mminus = ConstructionBase.setproperties(model, (y = pminus[1], x = pminus[2], flux = pminus[3], bkg = pminus[4]))
            fd[k] = (evaluate(mplus, py, px) - evaluate(mminus, py, px)) / (2h)
        end
        return fd
    end

    @testset "f matches evaluate" begin
        m = GriddedPSFModel(psfs, gy, gx; y = 4.3, x = 6.7, flux = 37.0, bkg = 1.5)
        f, G = evaluate_fg(m, 5.1, 5.9)
        @test f ≈ evaluate(m, 5.1, 5.9)
        @test length(G) == 4
        @test G[4] == 1.0 # d(bkg)/d(bkg) is always exactly 1
    end

    @testset "gradient matches finite differences: interior points" begin
        m = GriddedPSFModel(psfs, gy, gx; y = 4.3, x = 6.7, flux = 37.0, bkg = 1.5)
        for (py, px) in ((4.3, 6.7), (1.0, 1.0), (8.0, 8.0), (0.5, 0.5), (9.9, 0.1))
            _, G = evaluate_fg(m, py, px)
            fd = fd_gradient(m, py, px)
            @test collect(G) ≈ fd rtol = 1.0e-5 atol = 1.0e-5
        end
    end

    @testset "gradient matches finite differences: outside grid (clamped)" begin
        for (Y, X) in ((-3.0, 15.0), (-50.0, -50.0), (50.0, 50.0), (-50.0, 5.0))
            m = GriddedPSFModel(psfs, gy, gx; y = Y, x = X, flux = 37.0, bkg = 1.5)
            _, G = evaluate_fg(m, Y, X)
            fd = fd_gradient(m, Y, X)
            @test collect(G) ≈ fd rtol = 1.0e-5 atol = 1.0e-5
        end
    end

    @testset "N == 1 constant PSF gradient" begin
        m1 = GriddedPSFModel([psfs[1]], [5.0], [5.0]; y = -30.0, x = 200.0, flux = 7.0, bkg = 2.0)
        _, G = evaluate_fg(m1, -30.0, 200.0)
        fd = fd_gradient(m1, -30.0, 200.0)
        @test collect(G) ≈ fd rtol = 1.0e-5 atol = 1.0e-5
    end

    @testset "ImagePSF node gradients" begin
        data1 = [exp(-((i - 4)^2 + (j - 4)^2) / 5.0) for i in 1:7, j in 1:7]
        data2 = [exp(-((i - 4)^2 + (j - 4)^2) / 20.0) for i in 1:7, j in 1:7]
        img_psfs = [
            ImagePSF(data1; y = 0.0, x = 0.0, flux = 1.0, bkg = 0.0, oversampling = 2, normalize = true),
            ImagePSF(data2; y = 0.0, x = 0.0, flux = 1.0, bkg = 0.0, oversampling = 2, normalize = true),
            ImagePSF(data2; y = 0.0, x = 0.0, flux = 1.0, bkg = 0.0, oversampling = 2, normalize = true),
            ImagePSF(data1; y = 0.0, x = 0.0, flux = 1.0, bkg = 0.0, oversampling = 2, normalize = true),
        ]
        img_gy = [0.0, 0.0, 10.0, 10.0]
        img_gx = [0.0, 10.0, 0.0, 10.0]
        m = GriddedPSFModel(img_psfs, img_gy, img_gx; y = 3.4, x = 6.1, flux = 50.0, bkg = 0.5)
        _, G = evaluate_fg(m, 4.0, 5.0)
        fd = fd_gradient(m, 4.0, 5.0)
        @test collect(G) ≈ fd rtol = 1.0e-5 atol = 1.0e-4
    end
end

@testset "GriddedPSFModel extent (union of active corner extents)" begin
    psfs = [
        GaussianPRF(y = 0.0, x = 0.0, y_fwhm = 3.0, x_fwhm = 3.0, theta = 0.0, flux = 1.0, bkg = 0.0),
        GaussianPRF(y = 0.0, x = 0.0, y_fwhm = 5.0, x_fwhm = 4.0, theta = 0.2, flux = 1.0, bkg = 0.0),
        GaussianPRF(y = 0.0, x = 0.0, y_fwhm = 4.0, x_fwhm = 6.0, theta = -0.3, flux = 1.0, bkg = 0.0),
        GaussianPRF(y = 0.0, x = 0.0, y_fwhm = 6.0, x_fwhm = 5.0, theta = 0.5, flux = 1.0, bkg = 0.0),
    ]
    gy = [0.0, 0.0, 10.0, 10.0]
    gx = [0.0, 10.0, 0.0, 10.0]

    @testset "matches manual union of active recentered corner extents" begin
        for (Y, X) in ((2.0, 2.0), (7.5, 3.2), (-50.0, -50.0), (-50.0, 5.0))
            m = GriddedPSFModel(psfs, gy, gx; y = Y, x = X, flux = 1.0, bkg = 0.0)
            corners = CrowdPhot.PSF._grid_corners(m, Y, X)
            active = [(idx, w) for (idx, w) in corners if w != 0]
            exts = [extent(ConstructionBase.setproperties(psfs[idx], (y = Y, x = X))) for (idx, _) in active]
            ymin = minimum(e -> e[1][1], exts)
            ymax = maximum(e -> e[1][2], exts)
            xmin = minimum(e -> e[2][1], exts)
            xmax = maximum(e -> e[2][2], exts)
            @test extent(m) == ((ymin, ymax), (xmin, xmax))
            @test extent(m, 5) == extent(m)
        end
    end

    @testset "N == 1 extent matches the single node's own extent" begin
        m1 = GriddedPSFModel([psfs[1]], [5.0], [5.0]; y = -30.0, x = 200.0)
        r1 = ConstructionBase.setproperties(psfs[1], (y = -30.0, x = 200.0))
        @test extent(m1) == extent(r1)
    end

    @testset "extent(Int, ...) and CartesianIndices work via generic machinery" begin
        m = GriddedPSFModel(psfs, gy, gx; y = 2.0, x = 2.0)
        exi = CrowdPhot.PSF.extent(Int, m)
        (ylo, yhi), (xlo, xhi) = exi
        @test exi isa Tuple{Tuple{Int, Int}, Tuple{Int, Int}}
        ci = CartesianIndices(m)
        @test ci == CartesianIndices((ylo:yhi, xlo:xhi))
    end

    @testset "identical ImagePSF nodes: extent equals the single node's extent exactly" begin
        data1 = [exp(-((i - 4)^2 + (j - 4)^2) / 5.0) for i in 1:7, j in 1:7]
        img_psfs = [
            ImagePSF(data1; y = 0.0, x = 0.0, flux = 1.0, bkg = 0.0, oversampling = 2, normalize = true)
            for _ in 1:4
        ]
        m = GriddedPSFModel(img_psfs, gy, gx; y = 2.0, x = 3.0)
        r = ConstructionBase.setproperties(img_psfs[1], (y = 2.0, x = 3.0))
        @test extent(m) == extent(r)
    end

    @testset "centroid/integral/background/peak/render/add_star!/subtract_star! (generic defaults)" begin
        m = GriddedPSFModel(psfs, gy, gx; y = 4.3, x = 6.7, flux = 37.0, bkg = 1.5)
        @test centroid(m) == (4.3, 6.7)
        @test integral(m) == 37.0
        @test background(m) == 1.5
        @test peak(m) ≈ evaluate(m, 4.3, 6.7)

        img = render(m)
        @test all(isodd, size(img)) # render always produces an odd-sized matrix
        @test all(isfinite, img)

        out = zeros(50, 50)
        add_star!(out, m)
        @test any(!=(0), out)
        base = copy(out)
        subtract_star!(out, m)
        @test all(abs.(out) .< 1.0e-10) # subtracting what was just added returns (near) zero
        @test base != out
    end
end

@testset "render! scratch dispatch" begin
    # `_render_scratch` returns `nothing` for models with no specialized path,
    # so forwarding its result is always correct; passing the buffers it builds
    # for a gridded ImagePSF model must reproduce the per-pixel `evaluate` path.
    data1 = [exp(-((i - 6)^2 + (j - 6)^2) / 5.0) for i in 1:11, j in 1:11]
    data2 = [exp(-((i - 6)^2 + (j - 6)^2) / 12.0) for i in 1:11, j in 1:11]
    nodes = [ImagePSF(d; y = 0.0, x = 0.0, flux = 1.0, bkg = 0.0, oversampling = 2, normalize = true)
             for d in (data1, data2, data2, data1)]
    m = GriddedPSFModel(nodes, [0.0, 0.0, 10.0, 10.0], [0.0, 10.0, 0.0, 10.0];
                        y = 12.3, x = 7.6, flux = 50.0, bkg = 0.5)

    S = 9
    scratch = CrowdPhot.PSF._render_scratch(m, S, Float64)
    @test scratch isa NTuple{3, NTuple{4, Matrix{Float64}}}
    @test all(sz -> sz == (S, S), size.(scratch[1]))

    yr, xr = 9:16, 4:12                       # deliberately non-square, <= S
    ref = fill(NaN, S, S)
    got = fill(NaN, S, S)
    render!(ref, m, yr, xr)                   # scratch defaults to `nothing`
    render!(got, m, yr, xr, scratch)
    @test view(got, 1:length(yr), 1:length(xr)) ≈ view(ref, 1:length(yr), 1:length(xr)) rtol = 1.0e-12
    # Oversized-buffer contract: only the top-left block is written.
    @test all(isnan, got[(length(yr) + 1):end, :])
    @test all(isnan, got[:, (length(xr) + 1):end])

    # A model with no specialized path pairs with `nothing`, and a scratch it
    # cannot use is a MethodError rather than a silent fallback.
    plain = nodes[1]   # a bare ImagePSF has no specialized render path
    @test CrowdPhot.PSF._render_scratch(plain, S, Float64) === nothing
    @test_throws MethodError render!(got, plain, yr, xr, scratch)
end

@testset "GriddedPSFModel fit_star integration" begin
    # A grid whose node shapes vary substantially so that the true (y, x)
    # position materially affects the local PSF shape being fit.
    psfs = [
        CircularGaussianPRF(y = 0.0, x = 0.0, fwhm = 3.0, flux = 1.0, bkg = 0.0),
        CircularGaussianPRF(y = 0.0, x = 0.0, fwhm = 6.0, flux = 1.0, bkg = 0.0),
        CircularGaussianPRF(y = 0.0, x = 0.0, fwhm = 4.0, flux = 1.0, bkg = 0.0),
        CircularGaussianPRF(y = 0.0, x = 0.0, fwhm = 8.0, flux = 1.0, bkg = 0.0),
    ]
    gy = [0.0, 0.0, 20.0, 20.0]
    gx = [0.0, 20.0, 0.0, 20.0]
    truth = GriddedPSFModel(psfs, gy, gx; y = 25.4, x = 15.7, flux = 5000.0, bkg = 12.0)
    init = GriddedPSFModel(psfs, gy, gx; y = 24.0, x = 16.0, flux = 4000.0, bkg = 0.0)

    @testset "noiseless recovery" begin
        image = evaluate.(truth, 1:50, (1:50)')
        best, result = fit_star(init, image)
        @test result.converged
        @test best.y ≈ truth.y atol = 1.0e-4
        @test best.x ≈ truth.x atol = 1.0e-4
        @test best.flux ≈ truth.flux rtol = 1.0e-4
        @test best.bkg ≈ truth.bkg atol = 1.0e-3
    end

    @testset "noisy recovery" begin
        rng = StableRNG(42)
        image = evaluate.(truth, 1:50, (1:50)') .+ randn(rng, 50, 50) .* 2.0
        best, result = fit_star(init, image)
        @test result.converged
        @test best.y ≈ truth.y atol = 0.1
        @test best.x ≈ truth.x atol = 0.1
        @test best.flux ≈ truth.flux rtol = 0.05
    end

    @testset "fixed background" begin
        image = evaluate.(truth, 1:50, (1:50)')
        best, result = fit_star(init, image; fixed = (bkg = truth.bkg,))
        @test result.converged
        @test best.bkg == truth.bkg
        @test best.y ≈ truth.y atol = 1.0e-4
        @test best.x ≈ truth.x atol = 1.0e-4
        @test best.flux ≈ truth.flux rtol = 1.0e-4
    end
end
