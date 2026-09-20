import CrowdPhot
using CrowdPhot.Roman: crds_gridded_epsf, load_l2, load_area, DQ_FLAGS, dq_mask_value, parse_dq_mask
using CrowdPhot.PSF: GriddedPSFModel, ImagePSF, pixel_response_kernel
using ASDF
using OrderedCollections: OrderedDict
using Test

# No synthetic fixture here touches the real (170 MB) Roman CRDS reference
# file; everything is built as tiny in-memory ASDF trees written to a
# temporary directory.

# Build a minimal synthetic Roman CRDS ePSF-like ASDF tree.
#
# `data` must already be in the materialized Julia axis order
# `(x, y, grid_index, spectral_type, defocus)` (i.e. what
# `crds_gridded_epsf` expects to read back after round-tripping
# through ASDF) -- this helper reverses `size(data)` to produce the
# ASDF/Python-order `shape` field `NDArray` expects, exactly mirroring how
# the real files are laid out (see gridded_psf_crds_plan.md, Section 3).
function _write_synthetic_epsf(dir, data::Array{Float32, 5}; pixel_x, pixel_y,
                               spectral_type = ["A0V", "G2V"], defocus = [0], oversample = 2, name = "synthetic_epsf.asdf")
    lbh = ASDF.LazyBlockHeaders()
    nd = ASDF.NDArray(lbh, nothing, data, reverse(collect(size(data))), "float32", nothing)
    tree = OrderedDict("roman" => OrderedDict(
        "meta" => OrderedDict(
            "pixel_x" => pixel_x, "pixel_y" => pixel_y,
            "oversample" => oversample, "spectral_type" => spectral_type,
            "defocus" => defocus,
        ),
        "psf" => nd,
    ))
    path = joinpath(dir, name)
    ASDF.save(path, tree)
    return path
end

@testset "crds_gridded_epsf" begin
    @testset "transpose correctness, grid pass-through, defocus/spectral_type selection" begin
        # 2x2 grid, 2 spectral types, 1 defocus value. Each node's stamp has
        # a single asymmetric "marker" pixel at a distinct, non-central
        # (x, y) location -- a symmetric test stamp could pass even with a
        # transposed axis order, so this specifically catches that bug.
        nx, ny, ngrid, nspec, ndef = 7, 5, 4, 2, 1
        data = fill(0.001f0, nx, ny, ngrid, nspec, ndef) # near-zero background
        marker_xy = [(2, 4), (6, 1), (1, 1), (5, 3)] # distinct (x, y) per node
        for (i, (mx, my)) in enumerate(marker_xy)
            data[mx, my, i, 1, 1] = 100f0 # spectral_type = "A0V"
            data[mx, my, i, 2, 1] = 200f0 # spectral_type = "G2V"
        end
        pixel_x = [0.0, 10.0, 0.0, 10.0]
        pixel_y = [0.0, 0.0, 20.0, 20.0]

        mktempdir() do dir
            path = _write_synthetic_epsf(dir, data; pixel_x, pixel_y)

            model = crds_gridded_epsf(path; defocus = 0, spectral_type = "G2V")
            @test model isa GriddedPSFModel
            @test model.ygrid == [0.0, 20.0]
            @test model.xgrid == [0.0, 10.0]
            @test length(model.psfs) == 4

            for (i, (mx, my)) in enumerate(marker_xy)
                stamp = model.psfs[i].data # already transposed to (y, x) convention
                @test size(stamp) == (ny, nx)
                # Marker values (100/200) are far above oversample^2/2 = 2
                # (oversample defaults to 2 in `_write_synthetic_epsf`), so
                # this fixture is classified "new format" and used
                # unchanged (no pixel-response convolution) -- the argmax
                # location is exact, and lands at (my, mx), not (mx, my).
                @test Tuple(argmax(stamp)) == (my, mx)
            end

            # Selecting the other spectral_type picks a different (scaled) marker value.
            model_a0v = crds_gridded_epsf(path; defocus = 0, spectral_type = "A0V")
            @test maximum(model_a0v.psfs[1].data) < maximum(model.psfs[1].data)
        end
    end

    @testset "error handling" begin
        nx, ny, ngrid, nspec, ndef = 5, 5, 1, 1, 1
        data = fill(0.001f0, nx, ny, ngrid, nspec, ndef)
        data[3, 3, 1, 1, 1] = 1.0f0

        mktempdir() do dir
            path = _write_synthetic_epsf(dir, data; pixel_x = [0.0], pixel_y = [0.0],
                spectral_type = ["G2V"], defocus = [0])

            @test_throws "psf_subtype must be" crds_gridded_epsf(path; psf_subtype = "extended_psf")
            @test_throws "defocus=99 not found" crds_gridded_epsf(path; defocus = 99)
            @test_throws "spectral_type=\"Z9Z\" not found" crds_gridded_epsf(path; spectral_type = "Z9Z")

            # Missing top-level "roman" key.
            bad_tree = OrderedDict("not_roman" => OrderedDict())
            bad_path = joinpath(dir, "bad.asdf")
            ASDF.save(bad_path, bad_tree)
            @test_throws "does not contain a top-level \"roman\" key" crds_gridded_epsf(bad_path)
        end
    end

    @testset "normalize/origin/oversample plumbing" begin
        nx, ny, ngrid, nspec, ndef = 5, 5, 1, 1, 1
        data = fill(0.001f0, nx, ny, ngrid, nspec, ndef)
        data[3, 3, 1, 1, 1] = 1.0f0

        mktempdir() do dir
            path = _write_synthetic_epsf(dir, data; pixel_x = [7.0], pixel_y = [3.0],
                spectral_type = ["G2V"], defocus = [0], oversample = 3)

            model = crds_gridded_epsf(path)
            node = model.psfs[1]
            @test node.oversampling == (3, 3) # read from meta.oversample, not a keyword
            @test node.origin == (y = 3.0, x = 3.0) # default origin: geometric center of a 5x5 stamp
            @test model.ygrid == [3.0]
            @test model.xgrid == [7.0]

            model_origin = crds_gridded_epsf(path; origin = (y = 1.0, x = 1.0))
            @test model_origin.psfs[1].origin == (y = 1.0, x = 1.0)

            # `normalize = false` (default) preserves the (post pixel-response-
            # convolution) native flux scale; `normalize = true` forces
            # sum(data) == oversampling^2.
            model_norm = crds_gridded_epsf(path; normalize = true)
            @test sum(model_norm.psfs[1].data) ≈ 9.0 # oversample^2 = 3^2
            @test !(sum(model.psfs[1].data) ≈ 9.0)
        end
    end

    @testset "pixel-response convolution: old vs. new format" begin
        # "Old format": raw per-node stamps sum to ~1 (median across nodes
        # < oversample^2 / 2). Must be convolved with the pixel-response
        # kernel and scaled by oversample^2 before use.
        oversample = 4
        # The stamp must be wide enough for the full kernel support to fit
        # around the central sample, or the `:zero` border drops flux and the
        # flux-conservation check below fails for reasons that have nothing to
        # do with the loader. The :exact kernel has half-width 4 * oversample.
        nx, ny = 41, 41
        cen = (nx + 1) ÷ 2
        old_stamp = zeros(Float32, nx, ny)
        old_stamp[cen, cen] = 1.0f0 # sums to ~1

        # "New format": stamps already pixel-integrated, summing to
        # ~oversample^2. Must be used unchanged (no convolution, no extra
        # scaling).
        new_stamp = zeros(Float32, nx, ny)
        new_stamp[cen, cen] = Float32(oversample^2) # sums to oversample^2

        data_old = reshape(old_stamp, nx, ny, 1, 1, 1)
        data_new = reshape(new_stamp, nx, ny, 1, 1, 1)

        mktempdir() do dir
            path_old = _write_synthetic_epsf(dir, data_old; pixel_x = [0.0], pixel_y = [0.0],
                spectral_type = ["G2V"], defocus = [0], oversample, name = "old_format.asdf")
            model_old = crds_gridded_epsf(path_old)
            kernel = Float32.(pixel_response_kernel(oversample))
            expected_old = Float32(oversample^2) .* CrowdPhot.correlate(
                permutedims(old_stamp), kernel, :zero,
            )
            @test model_old.psfs[1].data ≈ expected_old
            @test sum(model_old.psfs[1].data) ≈ oversample^2 # convolution preserves total flux, then scaled

            # `pixel_integration` selects the quadrature; :box reproduces
            # romancal's Box2DKernel convolution.
            model_box = crds_gridded_epsf(path_old; pixel_integration = :box)
            expected_box = Float32(oversample^2) .* CrowdPhot.correlate(
                permutedims(old_stamp), Float32.(pixel_response_kernel(oversample; type = :box)), :zero,
            )
            @test model_box.psfs[1].data ≈ expected_box
            @test !(model_box.psfs[1].data ≈ model_old.psfs[1].data)

            path_new = _write_synthetic_epsf(dir, data_new; pixel_x = [0.0], pixel_y = [0.0],
                spectral_type = ["G2V"], defocus = [0], oversample, name = "new_format.asdf")
            model_new = crds_gridded_epsf(path_new; defocus = 0)
            @test model_new.psfs[1].data == permutedims(new_stamp) # used unchanged, no extra scaling
        end
    end

    @testset "pixel_response_kernel type=:box matches astropy Box2DKernel" begin
        # Hardcoded from astropy.convolution.Box2DKernel(width=n).array,
        # verified earlier (see gridded_psf_crds_plan.md, "Pixel-response
        # convolution"). n=3 (odd) is a naive uniform 3x3 box; n=4 (even)
        # is a tapered 5x5 kernel, not a naive 4x4 box.
        k3 = pixel_response_kernel(3; type = :box)
        @test k3 ≈ fill(1 / 9, 3, 3)

        k4 = pixel_response_kernel(4; type = :box)
        @test size(k4) == (5, 5)
        marginal4 = [0.125, 0.25, 0.25, 0.25, 0.125]
        @test k4 ≈ marginal4 * marginal4'
        @test sum(k4) ≈ 1.0
    end

    @testset "pixel_response_kernel type=:exact implements sinc(n f)" begin
        for n in (2, 3, 4, 8)
            k = pixel_response_kernel(n) # :exact is the default
            @test size(k) == (8n + 1, 8n + 1)
            @test sum(k) ≈ 1.0
            @test k ≈ k' # separable and symmetric
            @test k ≈ reverse(k; dims = 1)

            # The 1D marginal's DTFT must be the transfer function of a box of
            # width n, sinc(n f). The Hann window costs accuracy only in the
            # last few percent of the band, so test the interior tightly and
            # the band edge loosely.
            w = k[:, (8n + 1) ÷ 2 + 1]
            w = w ./ sum(w)
            j = -(4n):(4n)
            dtft(f) = sum(w .* cospi.(2 * f .* j))
            @test dtft(0) ≈ 1 atol = 1e-12 # unit gain at DC: preserves total flux
            for f in range(-0.45, 0.45; length = 101)
                @test dtft(f) ≈ sinc(n * f) atol = 0.015
            end
            for f in range(-0.5, 0.5; length = 11)
                @test dtft(f) ≈ sinc(n * f) atol = 0.08
            end
        end
    end

    @testset "pixel_response_kernel edge cases" begin
        # n == 1: nothing to integrate over, identity for either type.
        @test pixel_response_kernel(1) == fill(1.0, 1, 1)
        @test pixel_response_kernel(1; type = :box) == fill(1.0, 1, 1)

        @test_throws "type must be :exact or :box" pixel_response_kernel(4; type = :sinc)
        @test_throws "must be positive" pixel_response_kernel(0)

        # :box is broader than :exact by an extra box of one oversampled
        # sample. Correlating a delta function with either kernel returns the
        # kernel itself, so comparing their peaks compares the quadratures.
        n = 4
        kb = pixel_response_kernel(n; type = :box)
        ke = pixel_response_kernel(n; type = :exact)
        @test maximum(kb) < maximum(ke)
    end
end

@testset "data quality flags" begin
    # Bit positions, checked against `roman_datamodels`' `pixel` enum.  A
    # shifted bit here silently masks the wrong pixels, so the values are
    # pinned rather than recomputed from the table they came from.
    @test DQ_FLAGS[:DO_NOT_USE] == UInt32(1)
    @test DQ_FLAGS[:SATURATED] == UInt32(2) << 0
    @test DQ_FLAGS[:NO_LIN_CORR] == UInt32(1) << 20
    @test DQ_FLAGS[:REFERENCE_PIXEL] == UInt32(1) << 31
    # Every value is a single distinct bit; a duplicate or a non-power-of-two
    # would make two names alias or a mask span flags it did not request.
    @test all(count_ones(v) == 1 for v in values(DQ_FLAGS))
    @test length(unique(values(DQ_FLAGS))) == length(DQ_FLAGS)

    @test dq_mask_value() == DQ_FLAGS[:DO_NOT_USE]
    @test dq_mask_value(()) == UInt32(0)
    @test dq_mask_value((:DO_NOT_USE, :NO_LIN_CORR)) == UInt32(0x00100001)
    @test dq_mask_value((:DO_NOT_USE, :DO_NOT_USE)) == DQ_FLAGS[:DO_NOT_USE]  # OR is idempotent
    # A typo must name the alternatives, not throw a bare KeyError.
    @test_throws "unknown Roman DQ flag" dq_mask_value((:DONOTUSE,))
    @test_throws "valid flags are" dq_mask_value((:DO_NOT_USE, :NOPE))

    # `true` means bad, matching the `bkg_mask` convention.
    dq = UInt32[0 1; 2 (1 | 1 << 20)]
    @test parse_dq_mask(dq) == Bool[0 1; 0 1]
    @test parse_dq_mask(dq; flags = (:SATURATED,)) == Bool[0 0; 1 0]
    @test parse_dq_mask(dq; flags = (:DO_NOT_USE, :SATURATED)) == Bool[0 1; 1 1]
    @test parse_dq_mask(dq; flags = (:HOT,)) == Bool[0 0; 0 0]   # flag absent everywhere
    @test parse_dq_mask(dq) isa BitMatrix
end

@testset "load_l2 and load_area" begin
    # Round-trip through a synthetic L2 tree.  The load path is one
    # `permutedims` per array, so the thing worth testing is that the
    # transpose happens and that `dq` comes back as a mask, not raw bits.
    ny, nx = 3, 4
    data = Float32[10i + j for i in 1:nx, j in 1:ny]          # written x-fast
    err = Float32[0.1(10i + j) for i in 1:nx, j in 1:ny]
    var_poisson = Float32[0.01(10i + j) for i in 1:nx, j in 1:ny]
    dq_raw = zeros(UInt32, nx, ny)
    dq_raw[2, 1] = DQ_FLAGS[:DO_NOT_USE]
    dq_raw[3, 2] = DQ_FLAGS[:NO_LIN_CORR]
    dq_raw[4, 3] = DQ_FLAGS[:HOT]        # not in the default flag set

    nd(a, T) = ASDF.NDArray(ASDF.LazyBlockHeaders(), nothing, a,
                            reverse(collect(size(a))), T, nothing)
    mktempdir() do dir
        tree = OrderedDict("roman" => OrderedDict(
            "meta" => OrderedDict("exposure" => OrderedDict("effective_exposure_time" => 100.0)),
            "data" => nd(data, "float32"), "err" => nd(err, "float32"),
            "var_poisson" => nd(var_poisson, "float32"), "dq" => nd(dq_raw, "uint32"),
        ))
        path = joinpath(dir, "l2.asdf")
        ASDF.save(path, tree)

        l2 = load_l2(path)
        @test size(l2.data) == (ny, nx)                 # transposed to (y, x)
        @test l2.data == permutedims(data)
        @test l2.err == permutedims(err)
        @test l2.var_poisson == permutedims(var_poisson)
        @test l2.data isa Matrix{Float32}
        @test l2.dq isa BitMatrix
        @test size(l2.dq) == (ny, nx)
        # Default flags catch DO_NOT_USE and NO_LIN_CORR but not HOT.
        @test l2.dq[1, 2] && l2.dq[2, 3] && !l2.dq[3, 4]
        @test count(l2.dq) == 2
        @test l2.meta["exposure"]["effective_exposure_time"] == 100.0
        # The flag set is a keyword, not hardcoded.
        @test count(load_l2(path; dq_flags = (:HOT,)).dq) == 1

        area_tree = OrderedDict("roman" => OrderedDict(
            "meta" => OrderedDict("useafter" => "2020-01-01"), "data" => nd(data, "float32")))
        area_path = joinpath(dir, "area.asdf")
        ASDF.save(area_path, area_tree)
        pam = load_area(area_path)
        @test pam.data == permutedims(data)
        @test size(pam.data) == (ny, nx)
        @test pam.meta["useafter"] == "2020-01-01"

        bad = joinpath(dir, "bad.asdf")
        ASDF.save(bad, OrderedDict("not_roman" => OrderedDict()))
        @test_throws "does not contain a top-level \"roman\" key" load_l2(bad)
        @test_throws "does not contain a top-level \"roman\" key" load_area(bad)
    end
end
