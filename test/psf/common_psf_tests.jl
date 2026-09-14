using CrowdPhot.PSF: CircularGaussianPSF, GaussianPSF, CircularGaussianPRF, GaussianPRF, CircularMoffatPSF, MoffatPSF, evaluate, centroid, integral, evaluate_fg, evaluate_fgh, AbstractPSFModel, extent, render, theta, amplitude, background, fwhm, peak, effective_area, effective_fwhm, AiryPSF, ImagePSF, GriddedPSFModel, add_star!, subtract_star!, render!
import ConstructionBase
using Test

# Tests generic API, type return, etc
function test_common(model::AbstractPSFModel{T}) where {T}
    # Evaluation
    @test @inferred(evaluate(model, centroid(model)...)) isa T
    cy, cx = round.(Int, centroid(model))
    @test @inferred(evaluate(model, CartesianIndex(cy, cx))) isa T
    @test model(centroid(model)...) ≈ evaluate(model, centroid(model)...)
    @test model(CartesianIndex(cy, cx)) ≈ evaluate(model, CartesianIndex(cy, cx))
    ex = @inferred extent(model)
    @test ex isa Tuple{Tuple{T, T}, Tuple{T, T}}
    y, x = range(ex[1][1], ex[1][2]; step = one(T)), range(ex[2][1], ex[2][2]; step = one(T))
    ex_round = @inferred extent(Int, model)
    @test ex_round == ((floor(Int, ex[1][1]), ceil(Int, ex[1][2])), (floor(Int, ex[2][1]), ceil(Int, ex[2][2])))
    m = evaluate.(model, y, x')
    @test m isa Matrix{T}
    @test size(m) == (length(y), length(x))
    y, x = ex_round[1][1]:ex_round[1][2], ex_round[2][1]:ex_round[2][2]
    m = evaluate.(model, CartesianIndices((y, x)))
    @test m isa Matrix{T}
    @test size(m) == (length(y), length(x))
    @test @inferred(render(model)) isa Matrix{T}
    # Check rendering into a larger image.
    # add_star!/subtract_star! evaluate the model via @turbo, which may use
    # FMA/reassociation and so can differ from a plain scalar `evaluate` call
    # by a few ULP; compare with a tolerance rather than exact equality.
    image = zeros(T, 20, 20)
    inds = CartesianIndices(model)
    add_star!(image, model, inds)
    for i in inds
        checkbounds(Bool, image, i) || continue
        @test image[i] ≈ evaluate(model, i) rtol = sqrt(eps(T))
    end
    # Check that doing it again accumulates flux
    add_star!(image, model, inds)
    for i in inds
        checkbounds(Bool, image, i) || continue
        @test image[i] ≈ 2 * evaluate(model, i) rtol = sqrt(eps(T))
    end
    # Check automatic inds
    fill!(image, 0)
    add_star!(image, model)
    for i in inds
        checkbounds(Bool, image, i) || continue
        @test image[i] ≈ evaluate(model, i) rtol = sqrt(eps(T))
    end
    subtract_star!(image, model)
    @test all(iszero, image)

    # render!: stamp-local write over an explicit image-coordinate box.
    ex_i = extent(Int, model)
    yr = ex_i[1][1]:ex_i[1][2]
    xr = ex_i[2][1]:ex_i[2][2]
    buf = fill(T(NaN), length(yr) + 3, length(xr) + 3)  # oversized on purpose
    ms = render!(buf, model, yr, xr)
    @test size(ms) == (length(yr), length(xr))
    for (i, yy) in enumerate(yr), (jj, xx) in enumerate(xr)
        @test ms[i, jj] ≈ evaluate(model, yy, xx) rtol = sqrt(eps(T))
    end
    @test all(isnan, buf[(length(yr) + 1):end, :])  # untouched block preserved

    # Conversion rules
    other_elt = T === Float64 ? Float32 : Float64
    convert_type = if model isa GriddedPSFModel
        ConstructionBase.constructorof(typeof(model)){
            other_elt,
            ConstructionBase.constructorof(typeof(model.psfs[1])){other_elt}
        }
    else
        ConstructionBase.constructorof(typeof(model)){other_elt}
    end
    @test convert(convert_type, model) isa convert_type
    @test convert(typeof(model), model) === model
    # add_star!/subtract_star! must fall back to a plain scalar loop (rather
    # than @turbo) when eltype(out) != T, since LoopVectorization/
    # VectorizationBase cannot unify SIMD vector widths across mismatched
    # floating-point types and throws inside `vconvert` if forced to.
    mixed_rtol = sqrt(max(eps(T), eps(other_elt))) # storage into `other_elt` may be coarser than T
    image_mixed = zeros(other_elt, 20, 20)
    add_star!(image_mixed, model)
    for i in inds
        checkbounds(Bool, image_mixed, i) || continue
        @test image_mixed[i] ≈ evaluate(model, i) rtol = mixed_rtol
    end
    subtract_star!(image_mixed, model)
    @test all(x -> abs(x) < sqrt(eps(other_elt)), image_mixed)
    # render! eltype-mismatch fallback (scalar loop, no @turbo)
    buf_mixed = fill(other_elt(NaN), length(inds.indices[1]) + 1, length(inds.indices[2]) + 1)
    ms_mixed = render!(buf_mixed, model, inds.indices...)
    for (i, yy) in enumerate(inds.indices[1]), (jj, xx) in enumerate(inds.indices[2])
        @test ms_mixed[i, jj] ≈ evaluate(model, yy, xx) rtol = mixed_rtol
    end

    # API functions
    @test @inferred(centroid(model)) isa Tuple{T, T}
    @test @inferred(integral(model)) isa T
    @test @inferred(peak(model)) isa T
    @test @inferred(amplitude(model)) isa T
    @test @inferred(background(model)) isa T
    ea = @inferred effective_area(model)
    @test ea isa T
    @test ea > 0
    # The model method and the scalar method must agree, and both must preserve
    # the model's float type (this loop runs Float32 models too).
    @test @inferred(effective_fwhm(model)) isa T
    @test @inferred(effective_fwhm(ea)) isa T
    @test effective_fwhm(model) ≈ effective_fwhm(ea)
    @test effective_fwhm(ea) > 0
    # @test fwhm(model) isa T # no generic method yet
    return @test @inferred(theta(model)) isa T
end

@testset "conversion preserves model family" begin
    model = CircularGaussianPSF(x = 1.0, y = 2.0, fwhm = 3.0, flux = 4.0, bkg = 5.0)
    @test_throws MethodError convert(AiryPSF{Float64}, model)
end

for model in (
        AiryPSF(x = 1.3, y = 2.4, radius = 3.0, flux = 120.0, bkg = 10.0),
        AiryPSF(x = 1.3f0, y = 2.4f0, radius = 3.0f0, flux = 120.0f0, bkg = 10.0f0),
        CircularMoffatPSF(x = 1.3, y = 2.4, α = 3.0, β = 3.5, flux = 120.0, bkg = 10.0),
        CircularMoffatPSF(x = 1.3f0, y = 2.4f0, α = 3.0f0, β = 3.5f0, flux = 120.0f0, bkg = 10.0f0),
        MoffatPSF(x = 2.5, y = 5.0, x_α = 3.0, y_α = 4.0, theta = 35.0, β = 3.5, flux = 120.0, bkg = 10.0),
        MoffatPSF(x = 2.5f0, y = 5.0f0, x_α = 3.0f0, y_α = 4.0f0, theta = 35.0f0, β = 3.5f0, flux = 120.0f0, bkg = 10.0f0),
        CircularGaussianPSF(x = 1.3, y = 2.4, fwhm = 3.0, flux = 120.0, bkg = 10.0),
        CircularGaussianPSF(x = 1.3f0, y = 2.4f0, fwhm = 3.0f0, flux = 120.0f0, bkg = 10.0f0),
        GaussianPSF(x = 2.5, y = 5.0, x_fwhm = 3.0, y_fwhm = 4.0, theta = 35, flux = 120.0, bkg = 10),
        GaussianPSF(x = 2.5f0, y = 5.0f0, x_fwhm = 3.0f0, y_fwhm = 4.0f0, theta = 35.0f0, flux = 120.0f0, bkg = 10.0f0),
        CircularGaussianPRF(x = 1.3, y = 2.4, fwhm = 3.0, flux = 120.0, bkg = 10.0),
        CircularGaussianPRF(x = 1.3f0, y = 2.4f0, fwhm = 3.0f0, flux = 120.0f0, bkg = 10.0f0),
        GaussianPRF(x = 2.5, y = 5.0, x_fwhm = 3.0, y_fwhm = 4.0, theta = 35.0, flux = 120.0, bkg = 10),
        GaussianPRF(x = 2.5f0, y = 5.0f0, x_fwhm = 3.0f0, y_fwhm = 4.0f0, theta = 35.0f0, flux = 120.0f0, bkg = 10.0f0),
        ImagePSF(rand(7, 7); x = 3.0, y = 4.0, flux = 120.0, bkg = 7.0, oversampling = 2, normalize = false),
        ImagePSF(rand(Float32, 7, 7); x = 3.0f0, y = 4.0f0, flux = 120.0f0, bkg = 7.0f0, oversampling = 2, normalize = false),
        GriddedPSFModel(
            [CircularGaussianPRF(y = gy, x = gx, fwhm = 3.0, flux = 1.0, bkg = 0.0) for (gy, gx) in ((0.0, 0.0), (0.0, 10.0), (10.0, 0.0), (10.0, 10.0))],
            [0.0, 0.0, 10.0, 10.0], [0.0, 10.0, 0.0, 10.0]; y = 2.4, x = 1.3, flux = 120.0, bkg = 10.0,
        ),
        GriddedPSFModel(
            [CircularGaussianPRF(y = gy, x = gx, fwhm = 3.0f0, flux = 1.0f0, bkg = 0.0f0) for (gy, gx) in ((0.0f0, 0.0f0), (0.0f0, 10.0f0), (10.0f0, 0.0f0), (10.0f0, 10.0f0))],
            [0.0f0, 0.0f0, 10.0f0, 10.0f0], [0.0f0, 10.0f0, 0.0f0, 10.0f0]; y = 2.4f0, x = 1.3f0, flux = 120.0f0, bkg = 10.0f0,
        ),
        GriddedPSFModel(
            [ImagePSF(rand(7, 7); y = gy, x = gx, flux = 1.0, bkg = 0.0, oversampling = 2, normalize = true) for (gy, gx) in ((0.0, 0.0), (0.0, 10.0), (10.0, 0.0), (10.0, 10.0))],
            [0.0, 0.0, 10.0, 10.0], [0.0, 10.0, 0.0, 10.0]; y = 2.4, x = 1.3, flux = 120.0, bkg = 10.0,
        ),
    )
    @testset "API: $(typeof(model))" begin
        test_common(model)
    end
end

@testset "positional constructors use y, x order" begin
    # Positional constructors must agree with image-index order so callers can
    # pass centroids as `(y, x)` without swapping coordinates.
    @test centroid(CircularGaussianPSF(2.0, 1.0, 3.0, 4.0, 5.0)) == (2.0, 1.0)
    @test centroid(GaussianPSF(2.0, 1.0, 3.0, 4.0, 35.0, 6.0, 7.0)) == (2.0, 1.0)
    @test centroid(CircularGaussianPRF(2.0, 1.0, 3.0, 4.0, 5.0)) == (2.0, 1.0)
    @test centroid(GaussianPRF(2.0, 1.0, 3.0, 4.0, 35.0, 6.0, 7.0)) == (2.0, 1.0)
    @test centroid(CircularMoffatPSF(2.0, 1.0, 3.0, 2.5, 4.0, 5.0)) == (2.0, 1.0)
    @test centroid(MoffatPSF(2.0, 1.0, 3.0, 4.0, 35.0, 2.5, 6.0, 7.0)) == (2.0, 1.0)
    @test centroid(AiryPSF(2.0, 1.0, 3.0, 4.0, 5.0)) == (2.0, 1.0)
    @test centroid(ImagePSF(ones(4, 4), 2.0, 1.0, 3.0, 4.0)) == (2.0, 1.0)
end

@testset "CircularMoffatPSF" begin
    @testset "constructor promotion" begin
        @test CircularMoffatPSF(x = 1.3, y = 2.4, α = 3.0, β = 2.5, flux = 120.0, bkg = 10) isa CircularMoffatPSF{Float64}
        @test CircularMoffatPSF(x = 1.3f0, y = 2.4f0, α = 3.0f0, β = 2.5f0, flux = 120.0f0, bkg = 10.0f0) isa CircularMoffatPSF{Float32}
        @test CircularMoffatPSF(x = 1, y = 2, α = 3, β = 2, flux = 120, bkg = 10) isa CircularMoffatPSF{Float64}
        @test CircularMoffatPSF(x = BigFloat(1.3), y = BigFloat(2.4), α = BigFloat(3.0), β = BigFloat(2.5), flux = BigFloat(120.0), bkg = BigFloat(10.0)) isa CircularMoffatPSF{BigFloat}
    end

    m = CircularMoffatPSF(x = 0, y = 0, α = 5, β = 3, flux = 50, bkg = 10)
    @test centroid(m) == (0.0, 0.0)
    @test integral(m) == 50.0
    @test all(x -> isapprox(x, 5.098245285339587), fwhm(m))
    @test effective_area(m) ≈ 98.17477042468103 rtol = 1.0e-6
    @test background(m) == 10.0
    @test peak(m) ≈ amplitude(m) + background(m)
    @test theta(m) == 0.0
    r1 = evaluate(m, 2, 1)
    @test r1 isa Float64
    @test r1 ≈ 10.736828440240256 ≈ m(2, 1)
    let (f, g) = evaluate_fg(m, 2, 1)
        @test f ≈ evaluate(m, 2, 1)
        @test g ≈ [0.29473137609610256, 0.14736568804805128, -0.14736568804805128, 0.23407451180546335, 0.014736568804805127, 1.0]
    end
end

@testset "MoffatPSF" begin
    @testset "constructor promotion" begin
        @test MoffatPSF(x = 1.3, y = 2.4, x_α = 3.0, y_α = 4.0, theta = 35, β = 2.5, flux = 120.0, bkg = 10.0) isa MoffatPSF{Float64}
        @test MoffatPSF(x = 1.3f0, y = 2.4f0, x_α = 3.0f0, y_α = 4.0f0, theta = 35.0f0, β = 2.5f0, flux = 120.0f0, bkg = 10.0f0) isa MoffatPSF{Float32}
        @test MoffatPSF(x = 1, y = 2, x_α = 3, y_α = 4, theta = 35, β = 2, flux = 120, bkg = 10) isa MoffatPSF{Float64}
        @test MoffatPSF(x = BigFloat(1.3), y = BigFloat(2.4), x_α = BigFloat(3.0), y_α = BigFloat(4.0), theta = BigFloat(35), β = BigFloat(2.5), flux = BigFloat(120.0), bkg = BigFloat(10.0)) isa MoffatPSF{BigFloat}
    end

    m = MoffatPSF(x = 0, y = 0, x_α = 5, y_α = 3, theta = 30, β = 3, flux = 50, bkg = 10)
    @test centroid(m) == (0.0, 0.0)
    @test integral(m) == 50.0
    @test all(isapprox.(fwhm(m), (3.058947171203752, 5.098245285339587)))
    @test effective_area(m) ≈ 58.90486225480862 rtol = 1.0e-6
    @test background(m) == 10.0
    @test peak(m) ≈ amplitude(m) + background(m)
    @test theta(m) == 30.0
    r1 = evaluate(m, 2, 1)
    @test r1 isa Float64
    @test r1 ≈ 10.948401781459316
    let (f, g) = evaluate_fg(m, 2, 1)
        @test f ≈ evaluate(m, 2, 1)
        @test g ≈  [0.6781570383450282, -0.016559689619557116, -0.07153854864611547, -0.0684867059819571, 0.012414115378633376, 0.2195970122399787, 0.01896803562918631, 1.0]
    end

    # equal α values and theta=0 reduce to CircularMoffatPSF
    mc = CircularMoffatPSF(x = 1.5, y = 2.5, α = 8, β = 2.5, flux = 3, bkg = 0)
    mm = MoffatPSF(x = 1.5, y = 2.5, x_α = 8, y_α = 8, theta = 0, β = 2.5, flux = 3, bkg = 0)
    @test evaluate(mc, 3, 4) ≈ evaluate(mm, 3, 4)
end

# Test specific models; verify return values
@testset "CircularGaussianPSF" begin
    @testset "constructor promotion" begin
        @test CircularGaussianPSF(x = 1.3, y = 2.4, fwhm = 3.0, flux = 120.0, bkg = 10) isa CircularGaussianPSF{Float64}
        @test CircularGaussianPSF(x = 1.3f0, y = 2.4f0, fwhm = 3.0f0, flux = 120.0f0, bkg = 10.0f0) isa CircularGaussianPSF{Float32}
        @test CircularGaussianPSF(x = 1, y = 2, fwhm = 3, flux = 120, bkg = 10) isa CircularGaussianPSF{Float64}
        @test CircularGaussianPSF(x = BigFloat(1.3), y = BigFloat(2.4), fwhm = BigFloat(3.0), flux = BigFloat(120.0), bkg = BigFloat(10.0)) isa CircularGaussianPSF{BigFloat}
    end

    m = CircularGaussianPSF(x = 0, y = 0, fwhm = 10, flux = 1, bkg = 10)
    @test centroid(m) == (0.0, 0.0)
    @test integral(m) == 1.0
    @test fwhm(m) == (10.0, 10.0)
    @test effective_area(m) ≈ 226.6180070913597 rtol = 1.0e-6
    @test background(m) == 10.0
    @test peak(m) ≈ amplitude(m) + background(m)
    @test theta(m) == 0.0
    r1 = evaluate(m, 2, 1)
    @test r1 isa Float64
    @test r1 ≈ 10.0076829778398427705 ≈ m(2, 1)
    let (f, g) = evaluate_fg(m, 2, 1)
        @test f ≈ evaluate(m, 2, 1)
        @test g ≈ [0.0008520695084786488, 0.0004260347542393244, -0.001323578190848892, 0.0076829778398427705, 1.0]
    end
    let (f, g, h) = evaluate_fgh(m, 2, 1)
        @test f ≈ evaluate(m, 2, 1)
        @test g ≈ [0.0008520695084786488, 0.0004260347542393244, -0.001323578190848892, 0.0076829778398427705, 1.0]
        @test h ≈ [-0.00033153722184843266 4.724876619544591e-5 -0.00031720342029373656 0.0008520695084786488 0.0; 4.724876619544591e-5 -0.0004024103711416015 -0.00015860171014686828 0.0004260347542393244 0.0; -0.00031720342029373656 -0.00015860171014686828 0.00031777260218123345 -0.0013235781908488918 0.0; 0.0008520695084786488 0.0004260347542393244 -0.0013235781908488918 0.0 0.0; 0.0 0.0 0.0 0.0 0.0]
    end
end

@testset "GaussianPSF" begin
    @testset "constructor promotion" begin
        @test GaussianPSF(x = 1.3, y = 2.4, x_fwhm = 3.0, y_fwhm = 4.0, theta = 35, flux = 120.0, bkg = 10.0) isa GaussianPSF{Float64}
        @test GaussianPSF(x = 1.3f0, y = 2.4f0, x_fwhm = 3.0f0, y_fwhm = 4.0f0, theta = 35.0f0, flux = 120.0f0, bkg = 10.0f0) isa GaussianPSF{Float32}
        @test GaussianPSF(x = 1, y = 2, x_fwhm = 3, y_fwhm = 4, theta = 35, flux = 120, bkg = 10) isa GaussianPSF{Float64}
        @test GaussianPSF(x = BigFloat(1.3), y = BigFloat(2.4), x_fwhm = BigFloat(3.0), y_fwhm = BigFloat(4.0), theta = BigFloat(35), flux = BigFloat(120.0), bkg = BigFloat(10.0)) isa GaussianPSF{BigFloat}
    end

    m = GaussianPSF(x = 0, y = 0, x_fwhm = 10, y_fwhm = 6, theta = 30, flux = 1, bkg = 10)
    @test centroid(m) == (0.0, 0.0)
    @test integral(m) == 1.0
    @test fwhm(m) == (6.0, 10.0)
    @test effective_area(m) ≈ 135.9708042548158 rtol = 1.0e-6
    @test background(m) == 10.0
    @test peak(m) ≈ amplitude(m) + background(m)
    @test theta(m) == 30.0
    r1 = evaluate(m, 2, 1)
    @test r1 isa Float64
    @test r1 ≈ 10.011881854589938992
    let (f, g) = evaluate_fg(m, 2, 1)
        @test f ≈ evaluate(m, 2, 1)
        @test g ≈ [0.0025675279951917654, -6.269560630626032e-5, -0.0015172854575571531, -0.0009587636050457796, 4.700030666638217e-5, 0.01188185458993899, 1.0]
    end
    let (f, g, h) = evaluate_fgh(m, 2, 1)
        @test f ≈ evaluate(m, 2, 1)
        @test g ≈ [0.0025675279951917654, -6.269560630626032e-5, -0.0015172854575571531, -0.0009587636050457796, 4.700030666638217e-5, 0.01188185458993899, 1.0]
        @test h ≈ [-0.0009825507698117963 0.0004936505237653131 -0.0009787987399725283 -0.0003301242561598406 -3.547464474559758e-5 0.0025675279951917654 0.0; 0.0004936505237653131 -0.0009513701779307769 0.0003838214631391412 -0.00020789110903162168 -2.986906138973509e-6 -6.26956063062603e-5 0.0; -0.0009787987399725283 0.0003838214631391412 0.0002922935533913823 0.00012243190355172497 -3.048115727009258e-5 -0.001517285457557153 0.0; -0.0003301242561598406 -0.00020789110903162168 0.00012243190355172497 0.0001273559772475686 1.4950134657622455e-6 -0.0009587636050457795 0.0; -3.547464474559758e-5 -2.986906138973509e-6 -3.048115727009258e-5 1.4950134657622455e-6 -5.148866586397458e-7 4.700030666638217e-5 0.0; 0.0025675279951917654 -6.26956063062603e-5 -0.001517285457557153 -0.0009587636050457795 4.700030666638217e-5 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0]
    end
    # equal fwhm + theta=0 reduces to CircularGaussianPSF
    mc = CircularGaussianPSF(x = 1.5, y = 2.5, fwhm = 8, flux = 3, bkg = 0)
    mg = GaussianPSF(x = 1.5, y = 2.5, x_fwhm = 8, y_fwhm = 8, theta = 0, flux = 3, bkg = 0)
    @test evaluate(mc, 3, 4) ≈ evaluate(mg, 3, 4)
end

@testset "CircularGaussianPRF" begin
    @testset "constructor promotion" begin
        @test CircularGaussianPRF(x = 1.3, y = 2.4, fwhm = 3.0, flux = 120.0, bkg = 10) isa CircularGaussianPRF{Float64}
        @test CircularGaussianPRF(x = 1.3f0, y = 2.4f0, fwhm = 3.0f0, flux = 120.0f0, bkg = 10.0f0) isa CircularGaussianPRF{Float32}
        @test CircularGaussianPRF(x = 1, y = 2, fwhm = 3, flux = 120, bkg = 10) isa CircularGaussianPRF{Float64}
        @test CircularGaussianPRF(x = BigFloat(1.3), y = BigFloat(2.4), fwhm = BigFloat(3.0), flux = BigFloat(120.0), bkg = BigFloat(10.0)) isa CircularGaussianPRF{BigFloat}
    end

    m = CircularGaussianPRF(x = 0, y = 0, fwhm = 10, flux = 1, bkg = 10)
    @test centroid(m) == (0.0, 0.0)
    @test integral(m) == 1.0
    @test fwhm(m) == (10.0, 10.0)
    # Larger than the CircularGaussianPSF value (226.618...) because the PRF is
    # pixel-integrated, which broadens the profile.
    @test effective_area(m) ≈ 227.66592874914255 rtol = 1.0e-6
    @test background(m) == 10.0
    @test peak(m) ≈ amplitude(m) + background(m)
    @test theta(m) == 0.0
    r1 = evaluate(m, 2, 1)
    @test r1 isa Float64
    @test r1 ≈ 10.007652480658708 ≈ m(2, 1)
    let (f, g) = evaluate_fg(m, 2, 1)
        @test f ≈ evaluate(m, 2, 1)
        @test g ≈ [0.0008447735386125179, 0.00042238646952845753, -0.0013132200987741396, 0.007652480658708134, 1.0]
    end
end

@testset "GaussianPRF" begin
    @testset "constructor promotion" begin
        @test GaussianPRF(x = 1.3, y = 2.4, x_fwhm = 3.0, y_fwhm = 4.0, theta = 35.0, flux = 120.0, bkg = 10.0) isa GaussianPRF{Float64}
        @test GaussianPRF(x = 1.3f0, y = 2.4f0, x_fwhm = 3.0f0, y_fwhm = 4.0f0, theta = 35.0f0, flux = 120.0f0, bkg = 10.0f0) isa GaussianPRF{Float32}
        @test GaussianPRF(x = 1, y = 2, x_fwhm = 3, y_fwhm = 4, theta = 35, flux = 120, bkg = 10) isa GaussianPRF{Float64}
        @test GaussianPRF(x = BigFloat(1.3), y = BigFloat(2.4), x_fwhm = BigFloat(3.0), y_fwhm = BigFloat(4.0), theta = BigFloat(35.0), flux = BigFloat(120.0), bkg = BigFloat(10.0)) isa GaussianPRF{BigFloat}
    end

    m = GaussianPRF(x = 0, y = 0, x_fwhm = 10, y_fwhm = 6, theta = 55.0, flux = 1, bkg = 10)
    @test centroid(m) == (0.0, 0.0)
    @test integral(m) == 1.0
    @test fwhm(m) == (6.0, 10.0)
    # Larger than the GaussianPSF value (135.970...): pixel integration.
    @test effective_area(m) ≈ 137.15649102556438 rtol = 1.0e-6
    @test background(m) == 10.0
    @test peak(m) ≈ amplitude(m) + background(m)
    @test theta(m) == 55.0
    r1 = evaluate(m, 2, 1)
    @test r1 isa Float64
    @test r1 ≈ 10.012636019260277
    let (f, g) = evaluate_fg(m, 2, 1)
        @test f ≈ evaluate(m, 2, 1)
        @test g ≈ [0.0016252015373576577, 0.00036857849874118, -0.002045098287018763, -0.0009181257637637817, 1.549930112518202e-5, 0.012636019260278195, 1.0]
    end

    # equal x/y fwhm with theta=0 collapses to CircularGaussianPRF
    mc = CircularGaussianPRF(x = 1.5, y = 2.5, fwhm = 8, flux = 3, bkg = 0)
    mg = GaussianPRF(x = 1.5, y = 2.5, x_fwhm = 8, y_fwhm = 8, theta = 0, flux = 3, bkg = 0)
    @test evaluate(mc, 3, 4) ≈ evaluate(mg, 3, 4)
end

@testset "AiryPSF" begin
    @testset "constructor promotion" begin
        @test AiryPSF(x = 1.3, y = 2.4, radius = 3.0, flux = 120.0, bkg = 10) isa AiryPSF{Float64}
        @test AiryPSF(x = 1.3f0, y = 2.4f0, radius = 3.0f0, flux = 120.0f0, bkg = 10.0f0) isa AiryPSF{Float32}
        @test AiryPSF(x = 1, y = 2, radius = 3, flux = 120, bkg = 10) isa AiryPSF{Float64}
        @test AiryPSF(x = BigFloat(1.3), y = BigFloat(2.4), radius = BigFloat(3.0), flux = BigFloat(120.0), bkg = BigFloat(10.0)) isa AiryPSF{BigFloat}
    end

    m = AiryPSF(x = 0, y = 0, radius = 10, flux = 50, bkg = 10)
    @test centroid(m) == (0.0, 0.0)
    @test integral(m) == 50.0
    @test evaluate(m, centroid(m)...) ≈ peak(m) # Ensure r=0 is correct, as this is a special case in the code
    @test all(x -> isapprox(x, 8.436659602162363; rtol = 1.0e-6), fwhm(m))
    @test effective_area(m) ≈ 186.21997265876772 rtol = 1.0e-6
    @test background(m) == 10.0
    @test peak(m) ≈ amplitude(m) + background(m)
    @test theta(m) == 0.0
    r1 = evaluate(m, 2, 1)
    @test r1 isa Float64
    @test r1 ≈ 10.484822848946342 ≈ m(2, 1)
    let (f, g) = evaluate_fg(m, 2, 1)
        @test f ≈ evaluate(m, 2, 1)
        @test g ≈ [0.07346384950430244, 0.03673192475215122, -0.0785986074131928, 0.00969645697892684, 1.0]
    end
end

# ---------------------------------------------------------------------------
# render — odd-size guarantee and centering
# ---------------------------------------------------------------------------
@testset "render" begin

    @testset "odd-size guarantee across sub-pixel centroids and FWHMs" begin
        for x0 in (10.0, 10.1, 10.3, 10.5, 10.7, 10.9)
            for fwhm in (2.0, 3.0, 5.0, 7.2)
                m = CircularGaussianPSF(; y=20.4, x=x0, fwhm, flux=100.0, bkg=0.0)
                kern = render(m)
                sz = size(kern)
                @test isodd(sz[1]) && isodd(sz[2])
                cr, cc = sz[1] ÷ 2 + 1, sz[2] ÷ 2 + 1
                maxval, maxidx = findmax(kern)
                @test maxidx == CartesianIndex(cr, cc)
            end
        end
    end

    @testset "symmetry about center for integer-centroid model" begin
        # A circular Gaussian centered exactly on a pixel should produce a
        # rendered kernel that is symmetric about its center pixel.
        for fwhm in (3.0, 5.0, 8.0)
            m = CircularGaussianPSF(; y=25.0, x=15.0, fwhm, flux=50.0, bkg=0.0)
            kern = render(m)
            cr, cc = size(kern, 1) ÷ 2 + 1, size(kern, 2) ÷ 2 + 1
            for r in 1:size(kern, 1), c in 1:size(kern, 2)
                dr, dc = r - cr, c - cc
                @test kern[cr + dr, cc + dc] ≈ kern[cr - dr, cc - dc]
                @test kern[cr + dr, cc + dc] ≈ kern[cr - dr, cc + dc]
                @test kern[cr + dr, cc + dc] ≈ kern[cr + dr, cc - dc]
            end
        end
    end

    @testset "integral conservation (well-sampled PSF)" begin
        # For a well-sampled Gaussian (FWHM ≫ 1 px), the sum of the rendered
        # kernel should approximate the true flux (pixel area = 1.0).
        for fwhm in (4.0, 6.0, 10.0)
            # Place centroid at half-pixel offset — worst case for alignment.
            m = CircularGaussianPSF(; y=25.5, x=15.5, fwhm, flux=100.0, bkg=0.0)
            kern = render(m)
            @test sum(kern) ≈ 100.0 rtol = 0.01
        end
    end

    @testset "extent is fully covered" begin
        # The rendered kernel must cover the full floating-point extent
        # returned by `extent(model)`.
        for (y0, x0) in ((20.0, 10.0), (20.7, 10.3), (20.4, 10.6))
            for fwhm in (2.5, 5.0)
                m = CircularGaussianPSF(; y=y0, x=x0, fwhm, flux=100.0, bkg=0.0)
                (y_lo, y_hi), (x_lo, x_hi) = extent(m)
                kern = render(m)
                cr, cc = size(kern, 1) ÷ 2 + 1, size(kern, 2) ÷ 2 + 1
                hy = cr - 1
                hx = cc - 1
                xc = round(Int, x0)
                yc = round(Int, y0)
                @test xc - hx ≤ x_lo
                @test xc + hx ≥ x_hi
                @test yc - hy ≤ y_lo
                @test yc + hy ≥ y_hi
            end
        end
    end
end

@testset "effective_area / effective_fwhm" begin
    # Exact for a Gaussian, via the analytic `effective_area` method.
    for fw in (2.0, 3.0, 5.5)
        @test effective_fwhm(CircularGaussianPSF(; y = 0.0, x = 0.0, fwhm = fw,
                                                 flux = 1.0, bkg = 0.0)) ≈ fw rtol = 1e-12
    end

    # The generic fallback (render -> matrix) must reproduce the analytic value.
    m = CircularGaussianPSF(; y = 2.4, x = 1.3, fwhm = 3.0, flux = 120.0, bkg = 10.0)
    @test effective_area(render(ConstructionBase.setproperties(m, (; bkg = 0.0)))) ≈
          effective_area(m) rtol = 1e-3

    # It must ignore `flux` (cancels from the ratio) and `bkg` (does not, so the
    # model method has to zero it before rendering).  Checked on a
    # `GriddedPSFModel`, which has no analytic method and so exercises the
    # fallback itself.
    nodes(fw) = [CircularGaussianPRF(; y = gy, x = gx, fwhm = fw, flux = 1.0, bkg = 0.0)
                 for (gy, gx) in ((0.0, 0.0), (0.0, 10.0), (10.0, 0.0), (10.0, 10.0))]
    g = GriddedPSFModel(nodes(3.0), [0.0, 0.0, 10.0, 10.0], [0.0, 10.0, 0.0, 10.0];
                        y = 5.0, x = 5.0, flux = 1.0, bkg = 0.0)
    @test effective_area(g) ≈ effective_area(ConstructionBase.setproperties(g, (; flux = 1.0e6))) rtol = 1e-12
    @test effective_area(g) ≈ effective_area(ConstructionBase.setproperties(g, (; bkg = 50.0))) rtol = 1e-12
    # Above the nominal 3.0 because the nodes are PRFs: the profile is
    # pixel-integrated and so broader than the underlying Gaussian.  The
    # render-based fallback and the analytic PRF method now agree on this.
    @test effective_fwhm(g) ≈ 3.0 rtol = 0.10
    @test effective_fwhm(g) > 3.0
    @test effective_area(g) ≈
          effective_area(CircularGaussianPRF(; y = 0.0, x = 0.0, fwhm = 3.0,
                                             flux = 1.0, bkg = 0.0)) rtol = 1e-6

    # Spatially varying, non-zero origin: the value has to follow the local PSF,
    # which is set by `y`/`x` alone.
    gv = GriddedPSFModel(
        [CircularGaussianPRF(; y = gy, x = gx, fwhm = fw, flux = 1.0, bkg = 0.0)
         for ((gy, gx), fw) in zip(((100.0, 100.0), (100.0, 110.0), (110.0, 100.0), (110.0, 110.0)),
                                   (2.0, 2.0, 6.0, 6.0))],
        [100.0, 100.0, 110.0, 110.0], [100.0, 110.0, 100.0, 110.0];
        y = 100.0, x = 105.0, flux = 1.0, bkg = 0.0)
    lo = effective_fwhm(gv)
    hi = effective_fwhm(ConstructionBase.setproperties(gv, (; y = 110.0)))
    @test lo ≈ 2.0 rtol = 0.10       # pixel integration broadens the narrow node most
    @test hi ≈ 6.0 rtol = 0.10
    @test hi > lo

    # PRF effective area is the pixel-integrated one, so it exceeds the PSF's,
    # by more the worse the sampling.  Checked against a direct render, which is
    # the definition; agreement is exact once the profile is well sampled.
    for (fw, tol) in ((3.0, 1e-6), (5.0, 1e-7), (10.0, 1e-7))
        pr = CircularGaussianPRF(; y = 25.0, x = 25.0, fwhm = fw, flux = 1.0, bkg = 0.0)
        ps = CircularGaussianPSF(; y = 25.0, x = 25.0, fwhm = fw, flux = 1.0, bkg = 0.0)
        @test effective_area(pr) > effective_area(ps)
        @test effective_area(pr) ≈ effective_area(render(pr)) rtol = tol
    end
    # Broadening grows as sampling degrades: 1.02 at FWHM 5, 1.21 at FWHM 1.5.
    ratio(fw) = effective_area(CircularGaussianPRF(; y = 0.0, x = 0.0, fwhm = fw, flux = 1.0, bkg = 0.0)) /
                effective_area(CircularGaussianPSF(; y = 0.0, x = 0.0, fwhm = fw, flux = 1.0, bkg = 0.0))
    @test ratio(5.0) ≈ 1.0185 rtol = 1e-3
    @test ratio(1.5) ≈ 1.2111 rtol = 1e-3
    @test ratio(1.5) > ratio(3.0) > ratio(5.0)

    # GaussianPRF: `det(Σ + I/12)` is rotation invariant, so `theta` drops out.
    ea_th(th) = effective_area(GaussianPRF(; y = 0.0, x = 0.0, y_fwhm = 3.0, x_fwhm = 5.0,
                                           theta = th, flux = 1.0, bkg = 0.0))
    for th in (30.0, 55.0, 90.0)
        @test ea_th(th) ≈ ea_th(0.0) rtol = 1e-12
    end
    # It reduces to the circular case to within the Gaussian-pixel approximation
    # (the circular method uses the exact form, hence the 1e-3 rather than 1e-12).
    @test effective_area(GaussianPRF(; y = 0.0, x = 0.0, y_fwhm = 3.0, x_fwhm = 3.0,
                                     theta = 0.0, flux = 1.0, bkg = 0.0)) ≈
          effective_area(CircularGaussianPRF(; y = 0.0, x = 0.0, fwhm = 3.0,
                                             flux = 1.0, bkg = 0.0)) rtol = 1e-3

    # Scalar method: type preserving, and the inverse of the Gaussian relation.
    @test effective_fwhm(1.0f0) isa Float32
    @test effective_fwhm(1) isa Float64
    @test effective_fwhm(pi * 3.0^2 / (2 * log(2))) ≈ 3.0 rtol = 1e-12
end
