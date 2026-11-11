using CrowdPhot
using CrowdPhot.Joint
using CrowdPhot.Joint: oversample
using StableRNGs: StableRNG
using Statistics: mean, var
using Test
import FITSIO

include("coadd_cases.jl")

# drizzle-library outputs for the cases (see ~/Work/STScI/experiments/coadd/README.md)
const COADD_REF = joinpath(@__DIR__, "data", "coadd")
read_ref(case, name, S, dims) = (a = Array{S}(undef, dims); read!(joinpath(COADD_REF, "$(case)_$(name).bin"), a); ltoh.(a))

@testset "coadd matches drizzle" begin
    # drizzle (push form) leaves out drops whose mapped center falls off the output grid, so pixels
    # near the border get only part of the drops overlapping them; coadd (gather form) uses every
    # overlapping drop.  Compare away from a 3-pixel border.
    for c in coadd_cases()
        dims = (c.grid.ny, c.grid.nx)
        R = coadd(c.images, c.grid; quantities = (:data, :rms, :nimg), weight = :input, pixfrac = c.pixfrac)
        d = R.data.A; s = R.rms.A
        rd = read_ref(c.name, "data", Float32, dims); rv = read_ref(c.name, "var", Float32, dims); rn = read_ref(c.name, "nimg", Int32, dims)
        inner = falses(dims); inner[4:(end - 3), 4:(end - 3)] .= true
        @test (isfinite.(d) .& inner) == (isfinite.(rd) .& inner)
        @test R.nimg.A[inner] == rn[inner]
        k = isfinite.(d) .& inner
        # drizzle stores float32; with strong distortion the area ratio taken per output pixel here and
        # per input pixel by drizzle differ at second order (its gradient across half a pixel)
        tol_d, tol_v = c.name == "projection_distortion" ? (0.1, 5e-3) : (1e-3, 1e-5)
        @test maximum(abs.(d[k] .- rd[k]) ./ s[k]) < tol_d
        @test maximum(abs.(s[k] .^ 2 .- rv[k]) ./ s[k] .^ 2) < tol_v
    end
end

@testset "coadd geometry, flux and units" begin
    g = TanGrid(COADD_RA0, COADD_DEC0, 0.1, 10, 10)
    img = reshape(collect(1.0:100.0), 10, 10)
    on_grid(dy, dx) = (y, x) -> grid_to_world(g, y + dy, x + dx)
    R = coadd([JointImage(img, ones(10, 10), nothing, on_grid(0, 0); band = :A)], g; quantities = (:data, :rms, :weight, :nimg), weight = :input)
    @test R.data.A ≈ img rtol = 1e-9
    @test all(≈(1; rtol = 1e-9), R.weight.A)          # Σ a w' with a = 1, w = 1, f = 1
    @test all(≈(1; rtol = 1e-9), R.rms.A) && all(==(1), R.nimg.A)
    # half an input pixel along x: each output pixel averages two inputs (the first only half covered)
    R = coadd([JointImage(img, ones(10, 10), nothing, on_grid(0, 0.5); band = :A)], g; quantities = (:data, :rms), weight = :input)
    @test R.data.A[5, 1:4] ≈ [5, 10, 20, 30] rtol = 1e-9
    @test R.rms.A[5, 2] ≈ sqrt(0.5) rtol = 1e-9

    # a source's sum over the coadd is its flux, for rotated, distorted and coarser inputs; surface
    # brightness is flux over the output pixel's solid angle
    g = TanGrid(COADD_RA0, COADD_DEC0, 0.05, 60, 60)
    for (θ, q, s) in ((0.0, (0.0, 0.0, 0.0), 0.1), (-20.0, (2e-4, -1e-4, 1e-4), 0.1), (10.0, (0.0, 0.0, 0.0), 0.17))
        sky = case_sky(COADD_RA0, COADD_DEC0, s, θ, 20.5, 20.5; q)
        d = [(rc = world_to_grid(g, sky(i, j)...); exp(-((rc[1] - 31.3)^2 + (rc[2] - 29.6)^2) / 18)) for i in 1:40, j in 1:40]
        im = JointImage(d, ones(40, 40), nothing, sky; band = :A)
        F = coadd([im], g; weight = :input).data.A
        @test sum(filter(isfinite, F)) ≈ sum(d) rtol = 1e-6
        S = coadd([im], g; weight = :input, units = :surface_brightness).data.A
        @test S[31, 30] / F[31, 30] ≈ (1 + ((31 - g.cy)^2 + (30 - g.cx)^2) * (0.05 / 206264.80624709636)^2)^1.5 / 0.05^2 rtol = 1e-10
    end

    # bands are coadded separately and keyed by name; Float32 agrees with Float64
    c = coadd_cases()[2]
    ims = [JointImage(im.data, im.weight, nothing, im.to_sky; band = k == 2 ? :B : :A) for (k, im) in enumerate(c.images)]
    R = coadd(ims, c.grid; quantities = (:data, :nimg), weight = :input, pixfrac = c.pixfrac)
    @test keys(R.data) == (:A, :B) && maximum(R.nimg.A) == 2 && maximum(R.nimg.B) == 1
    R32 = coadd(ims, c.grid; quantities = (:data,), weight = :input, pixfrac = c.pixfrac, T = Float32)
    k = isfinite.(R.data.A)
    @test eltype(R32.data.A) == Float32 && maximum(abs.(R32.data.A[k] .- R.data.A[k])) < 1e-3 * maximum(abs, R.data.A[k])

    # drop overlaps at full-detector pixel coordinates: in Float32 the area once vanished in the
    # shoelace sum's cancellation there (a full-frame Roman coadd lost 8% of its pixels)
    for (y, x) in (Float64.(Float32.((3.3, 7.7))), Float64.(Float32.((4000.3, 4087.7))))   # inputs both precisions hold exactly
        e = Float64.(Float32.((0.5, 0.05, -0.04, 0.5)))                                  # (ya, xa, yb, xb)
        ov(T, i, j) = CrowdPhot.Joint._overlap(T(y), T(x), T.(e)..., T(i - 0.5), T(i + 0.5), T(j - 0.5), T(j + 0.5))
        pix = [(i, j) for i in (floor(Int, y) - 1):(floor(Int, y) + 2) for j in (floor(Int, x) - 1):(floor(Int, x) + 2)]
        @test all(p -> isapprox(ov(Float32, p...), ov(Float64, p...); atol = 1e-5), pix)
        @test sum(p -> ov(Float32, p...), pix) ≈ abs(e[1] * e[4] - e[3] * e[2]) rtol = 1e-5
    end
end

@testset "coadd noise maps" begin
    # :rms_aperture / :rms is Fruchter & Hook's R for a uniformly filled dither (offsets spread over a
    # whole input pixel), onto a grid twice as fine
    fh(r) = r >= 1 ? r / (1 - 1 / (3r)) : 1 / (1 - r / 3)
    g = TanGrid(COADD_RA0, COADD_DEC0, 0.05, 40, 40)
    N = 20
    ims = vec([JointImage(ones(24, 24), ones(24, 24), nothing, case_sky(COADD_RA0, COADD_DEC0, 0.1, 0.0, 12.5 + (i - 0.5) / N, 12.5 + (j - 0.5) / N); band = :A)
               for i in 1:N, j in 1:N])
    for p in (1.0, 0.5)
        R = coadd(ims, g; quantities = (:rms, :rms_aperture), weight = :input, pixfrac = p)
        @test all(≈(fh(p / 0.5); rtol = 0.01), (R.rms_aperture.A ./ R.rms.A)[11:30, 11:30])
    end
    # :rms is the per-pixel noise: Monte Carlo over noise realizations of three rolled images with
    # different weights
    skies = [case_sky(COADD_RA0, COADD_DEC0, 0.1, θ, 12.5 + 0.3k, 12.5 - 0.2k) for (k, θ) in enumerate((0.0, 25.0, 60.0))]
    w = [fill(1 / s^2, 24, 24) for s in (1.0, 1.5, 0.8)]
    base = coadd([JointImage(zeros(24, 24), w[k], nothing, skies[k]; band = :A) for k in 1:3], g; quantities = (:rms,), weight = :input, pixfrac = 0.7)
    rng = StableRNG(11)
    S = cat([coadd([JointImage(randn(rng, 24, 24) ./ sqrt.(w[k]), w[k], nothing, skies[k]; band = :A) for k in 1:3], g; weight = :input, pixfrac = 0.7).data.A
             for _ in 1:300]...; dims = 3)
    @test mean(vec(var(S[12:29, 12:29, :]; dims = 3)) ./ vec(base.rms.A[12:29, 12:29] .^ 2)) ≈ 1 atol = 0.02
end

@testset "coadd inputs, quantities and masks" begin
    c = coadd_cases()[5]   # partial overlap, a masked block with NaN data, varying weights
    R = coadd(c.images, c.grid; quantities = (:data, :rms, :nimg), weight = :input)
    @test all(isnan, R.data.A[R.nimg.A .== 0]) && all(isfinite, R.data.A[R.nimg.A .> 0]) && any(==(0), R.nimg.A)
    # the residual coadd is data - background - model, with the same weights
    bkg = [fill(10.0, size(im.data)) for im in c.images]; mdl = [0.5 .* max.(im.data .- 10, 0) for im in c.images]
    rms = [fill(5.0, size(im.data)) for im in c.images]
    Q = coadd(c.images, c.grid; quantities = (:data, :background, :model, :residual), background = bkg, model = mdl, rms)
    k = isfinite.(Q.data.A)
    @test Q.residual.A[k] ≈ Q.data.A[k] .- Q.background.A[k] .- Q.model.A[k] rtol = 1e-10
    # weight = :rms is weight = :input with 1 / rms²
    ims = [JointImage(im.data, ifelse.(im.weight .> 0, 1 / 25, 0.0), nothing, im.to_sky; band = im.band) for im in c.images]
    @test isequal(coadd(c.images, c.grid; rms).data.A, coadd(ims, c.grid; weight = :input).data.A)
    # without a background, Background2D estimates one
    @test all(isfinite, coadd(c.images, c.grid; quantities = (:background,)).background.A[k])
    @test_throws "unknown coadd quantities flux" coadd(c.images, c.grid; quantities = (:flux,))
    @test_throws "quantity :model needs `model`" coadd(c.images, c.grid; quantities = (:model,))
    @test_throws "pixfrac must be in (0, 1]" coadd(c.images, c.grid; pixfrac = 0)
    @test_throws "units must be" coadd(c.images, c.grid; units = :jansky)
    @test_throws DimensionMismatch coadd(c.images, c.grid; background = bkg[1:1])
end

# `n` dithered 0.1" images (side `m`, small rolls) of a bright star with its exact model, sky 10,
# onto a 0.05" grid of side `gn` (star near its center).  Noise as data: background variance 25
# plus the star's Poisson variance (the model, in counts); the images' weights are that full
# inverse variance, while `rms` is the background's 5 (the coadd's weights).  The images share
# `data`, so outliers can be added after.
function clip_stack(n, m, gn; seed)
    g = TanGrid(COADD_RA0, COADD_DEC0, 0.05, gn, gn)
    rng = StableRNG(seed)
    skies = [case_sky(COADD_RA0, COADD_DEC0, 0.1, 10 * (rand(rng) - 0.5), (m + 1) / 2 + rand(rng) - 0.5, (m + 1) / 2 + rand(rng) - 0.5) for _ in 1:n]
    s0 = (gn + 1) / 2
    mdl = [[(rc = world_to_grid(g, s(i, j)...); 2000 * exp(-((rc[1] - s0)^2 + (rc[2] - s0 - 1)^2) / 8)) for i in 1:m, j in 1:m] for s in skies]
    data = [10 .+ x .+ sqrt.(25 .+ x) .* randn(rng, m, m) for x in mdl]
    ims = [JointImage(data[k], 1 ./ (25 .+ mdl[k]), nothing, skies[k]; band = :A) for k in 1:n]
    return (; ims, g, data, mdl, bkg = [fill(10.0, m, m) for _ in 1:n], rms = [fill(5.0, m, m) for _ in 1:n])
end

@testset "coadd clipping" begin
    J = CrowdPhot.Joint
    # independent of the kernels' two median paths: the exact weighted median and test at every
    # node from the same per-image statistics, rejections grown, then the images left per node
    function ref_nimg(S; k = 4.0, mu = 0.003, nmin = 5, grow = 1, model = S.mdl)
        geo = J._image_geometry(S.ims, S.g, J.KA.CPU(), Float64; degree = 5, max_degree = 9, margin = 40, tol = 1e-4)
        E = length(S.ims); ny, nx = S.g.ny, S.g.nx
        flat(xs) = reduce(vcat, vec.(xs))
        # the base arrays, as the kernels read them (clipping variances come from the input weights)
        inp = (; data = flat(S.data), weight = flat([im.weight for im in S.ims]), bkg = flat(S.bkg), rms = flat(S.rms), model = flat(model))
        has = falses(E, ny, nx); rej = falses(E, ny, nx)
        for c in 1:nx, r in 1:ny
            st = [J._clip_stat(geo.dev, e, Float64(r), Float64(c), 1.0, 1.0, inp, Val(:rms), mu) for e in 1:E]
            idx = findall(s -> s[1] > 0, st); has[idx, r, c] .= true
            length(idx) >= nmin || continue
            x = [st[e][2] for e in idx]; v = [st[e][3] for e in idx]
            o = sortperm(x); W = sum(1 ./ v)
            med = x[o[findfirst(>=(W / 2), cumsum(1 ./ v[o]))]]
            rej[idx[(x .- med) .^ 2 .> k^2 .* (v .+ π / 2 / W)], r, c] .= true
        end
        return [count(e -> has[e, r, c] && !any(rej[e, max(1, r - grow):min(ny, r + grow), max(1, c - grow):min(nx, c + grow)]), 1:E) for r in 1:ny, c in 1:nx]
    end
    kw(S) = (; quantities = (:data, :residual, :nimg), background = S.bkg, S.rms, model = S.mdl)

    # 8 images (the private-array path): one image with an outlier, and two with outliers at one spot
    S = clip_stack(8, 24, 40; seed = 21)
    S.data[2][5:6, 7:8] .+= 500; S.data[4][18:19, 4:5] .+= 400; S.data[6][18:19, 4:5] .+= 450
    plain = coadd(S.ims, S.g; kw(S)...)
    C = coadd(S.ims, S.g; kw(S)..., clip = 4)
    @test maximum(plain.nimg.A) == 8 && C.nimg.A == ref_nimg(S)
    hit = C.nimg.A .< plain.nimg.A
    # the single outlier (rows < 15; ~12 in the coadd of 8, noise ~0.4) loses its image; where the
    # pair overlaps (rows >= 15) both are rejected, and the clipped residual is noise
    one = argmax(ifelse.(isfinite.(plain.residual.A[1:14, :]), plain.residual.A[1:14, :], -Inf))
    @test plain.residual.A[one] > 10 && plain.nimg.A[one] - C.nimg.A[one] == 1 && abs(C.residual.A[one]) < 2
    @test maximum((plain.nimg.A .- C.nimg.A)[15:end, :]) == 2
    @test maximum(abs, filter(isfinite, C.residual.A[15:end, 1:12])) < 3 < maximum(filter(isfinite, plain.residual.A[15:end, 1:12]))
    star = (19:23, 20:24)                                               # around the star at (20.5, 21.5)
    @test C.nimg.A[star...] == plain.nimg.A[star...]                    # the star's residual is noise
    @test isequal(C.data.A[.!hit], plain.data.A[.!hit])
    @test count(hit) > count(coadd(S.ims, S.g; kw(S)..., clip = 4, clip_grow = 0).nimg.A .< plain.nimg.A)
    # a 15% model error in one image (300 counts at the peak, against Poisson noise of 45 there):
    # covered by clip_mu (√0.003 ≈ 5.5% of the model, 110), clipped without it.  (A 5% error is
    # already within the star's Poisson noise, which the input weights carry.)
    bad = copy(S.mdl); bad[3] = 0.85 .* S.mdl[3]
    nm(mu) = coadd(S.ims, S.g; kw(S)..., model = bad, clip = 4, clip_mu = mu).nimg.A
    @test nm(0.003)[star...] == plain.nimg.A[star...] && nm(0.003) == ref_nimg(S; model = bad)
    @test any(nm(0.0)[star...] .< plain.nimg.A[star...])

    # the input weights set the test's variances: with the star's Poisson noise in them the star is
    # not clipped even without clip_mu; background-only input weights (1/25) reject its core, though
    # the coadd weights (1/rms²) are the same in both runs
    S = clip_stack(8, 24, 40; seed = 26)
    bkg_only = [JointImage(S.data[k], fill(1 / 25, 24, 24), nothing, S.ims[k].to_sky; band = :A) for k in 1:8]
    plain = coadd(S.ims, S.g; kw(S)...)
    @test !any(plain.nimg.A[star...] .> coadd(S.ims, S.g; kw(S)..., clip = 4, clip_mu = 0).nimg.A[star...])
    @test count(plain.nimg.A[star...] .> coadd(bkg_only, S.g; kw(S)..., clip = 4, clip_mu = 0).nimg.A[star...]) >= 5

    # 300 images (pixels beyond the CPU kernel's arrays: the median by bisection; bit masks of 5
    # words), 90 outliers
    S = clip_stack(300, 16, 28; seed = 22)
    rng = StableRNG(23)
    for _ in 1:90
        S.data[rand(rng, 1:300)][rand(rng, 4:13), rand(rng, 4:13)] += 300
    end
    plain = coadd(S.ims, S.g; kw(S)...)
    C = coadd(S.ims, S.g; kw(S)..., clip = 4)
    @test maximum(plain.nimg.A) > J._clip_fast(J.KA.CPU())
    @test count(C.nimg.A .!= ref_nimg(S)) <= 2          # bisection finds the median to 0.05 of its σ
    # no outliers, noise including the star's Poisson noise (which the input weights carry): with
    # clip_mu = 0 the rejected fraction of node-image samples, star included, is the Gaussian
    # two-sided tail at k (k = 3 for ~180 events; erfc(3 / √2) = 0.0027), times 0.92 at ~100 images:
    # the test adds the median's variance (π/2) / W, while each image's own pull on the median
    # reduces the variance of its difference by 2 / W.  The default clip_mu widens it at the star.
    S = clip_stack(100, 16, 28; seed = 24)
    plain = coadd(S.ims, S.g; kw(S)...)
    deep = plain.nimg.A .>= 5
    nrej(mu) = (C = coadd(S.ims, S.g; kw(S)..., clip = 3, clip_grow = 0, clip_mu = mu); plain.nimg.A .- C.nimg.A)
    r0 = nrej(0.0)
    @test 0.7 < sum(r0[deep]) / (0.92 * 0.0026997960632601866 * sum(plain.nimg.A[deep])) < 1.3
    near = [hypot(r - 14.5, c - 15.5) <= 6 for r in 1:28, c in 1:28]
    @test sum(nrej(0.003)[near]) < sum(r0[near]) / 2

    # a deep Float32 stack near 1e4: the bisection's tolerance (σ / 20 of the median, 4e-5) is below one
    # ulp (1e-3) there, so it must stop on adjacent floats (it once looped forever); values a few ulps apart
    g1 = TanGrid(COADD_RA0, COADD_DEC0, 0.1, 4, 4)
    ims = [JointImage(fill(1e4 + 1e-3 * (k % 5), 4, 4), fill(1e4, 4, 4), nothing, (y, x) -> grid_to_world(g1, y, x); band = :A) for k in 1:257]
    C = coadd(ims, g1; T = Float32, weight = :input, clip = 4, clip_mu = 0, background = [zeros(4, 4) for _ in 1:257], quantities = (:nimg,))
    @test all(==(257), C.nimg.A)

    # 4 images, below clip_min_images: nothing is clipped, however large the outlier
    S = clip_stack(4, 24, 40; seed = 25)
    S.data[2][5:6, 7:8] .+= 5000
    plain = coadd(S.ims, S.g; kw(S)...)
    @test isequal(coadd(S.ims, S.g; kw(S)..., clip = 4).data.A, plain.data.A)
    @test any(coadd(S.ims, S.g; kw(S)..., clip = 4, clip_min_images = 3).nimg.A .< plain.nimg.A)
    @test_throws "clip_min_images must be at least 3" coadd(S.ims, S.g; clip = 4, clip_min_images = 2)
end

@testset "TanGrid(images), fits_wcs and FITSIO" begin
    c = coadd_cases()[3]   # 0.1" and 0.17" images
    G = TanGrid(c.images)
    @test G.scale ≈ (0.1 + 0.17) / 2 rtol = 1e-4   # median of two
    for im in c.images, (y, x) in ((0.5, 0.5), (0.5, size(im.data, 2) + 0.5), (size(im.data, 1) + 0.5, 0.5), size(im.data) .+ 0.5)
        r, cc = world_to_grid(G, im.to_sky(y, x)...)
        @test 0.5 - 1e-9 <= r <= G.ny + 0.5 + 1e-9 && 0.5 - 1e-9 <= cc <= G.nx + 0.5 + 1e-9
    end
    @test TanGrid(c.images; scale = 0.05).scale == 0.05
    # fits_wcs through the FITS TAN equations reproduces grid_to_world (axis 1 = column)
    h = Dict(k => v for (k, v, _) in fits_wcs(G))
    for (p1, p2) in ((1.0, 1.0), (3.0, G.ny), (G.nx, 7.5))
        ξ = deg2rad(h["CD1_1"] * (p1 - h["CRPIX1"]) + h["CD1_2"] * (p2 - h["CRPIX2"]))
        η = deg2rad(h["CD2_1"] * (p1 - h["CRPIX1"]) + h["CD2_2"] * (p2 - h["CRPIX2"]))
        δ0 = deg2rad(h["CRVAL2"])
        ra = h["CRVAL1"] + rad2deg(atan(ξ, cos(δ0) - η * sin(δ0)))
        dec = rad2deg(atan((sin(δ0) + η * cos(δ0)) * cos(deg2rad(ra - h["CRVAL1"])), cos(δ0) - η * sin(δ0)))
        @test all(isapprox.((mod(ra, 360), dec), grid_to_world(G, p2, p1); atol = 1e-10))
    end
    # write(f, A, g) puts the column on axis 1 with the header; permutedims(read(...)) reads A back
    hdr = FITSIO.FITSHeader(G)
    @test hdr["CTYPE1"] == "RA---TAN" && hdr["CD1_1"] ≈ -G.scale / 3600 && hdr["CRPIX1"] == G.cx
    g = TanGrid(G.ra0, G.dec0, G.scale, 7, 5, 3.2, 2.6)     # non-square: a transpose cannot pass
    A = reshape(Float32.(1:35), 7, 5)
    mktempdir() do dir
        f = joinpath(dir, "coadd.fits")
        FITSIO.FITS(f, "w") do io
            write(io, A, g; name = "SCI")
            @test_throws "a 5 x 7 matrix is not on a 7 x 5 TanGrid" write(io, permutedims(A), g)
        end
        FITSIO.FITS(f) do io
            h2 = FITSIO.read_header(io["SCI"])
            @test (h2["NAXIS1"], h2["NAXIS2"]) == (5, 7) && h2["CRPIX1"] == 2.6 && h2["CRPIX2"] == 3.2
            @test permutedims(read(io["SCI"])) == A
        end
    end
end
