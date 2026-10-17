# Coadd validation cases, shared by coadd_tests.jl and ~/Work/STScI/experiments/coadd/gen_inputs.jl
# (which writes them for the drizzle library; the drizzle outputs are committed in data/coadd/).
# Changing anything here invalidates those references: regenerate them (see that directory's README).
using CrowdPhot.Joint: TanGrid, grid_to_world, world_to_grid, JointImage
using StableRNGs: StableRNG

const COADD_RA0, COADD_DEC0 = 80.0, -69.0

"""
`to_sky` of an image whose pixel offsets from `(cy, cx)` are rotated by `θ` (degrees,
counterclockwise from north-up/east-left), scaled by `scale` arcsec/px, optionally distorted by
`q` (quadratic terms added to the offsets, in px), and projected about `(ra_t, dec_t)` by
`proj` (`:tan` or `:sin`).
"""
function case_sky(ra_t, dec_t, scale, θ, cy, cx; q = (0.0, 0.0, 0.0), proj = :tan)
    s, c = sincosd(θ)
    g1 = TanGrid(ra_t, dec_t, 1.0, 1, 1, 0.0, 0.0)          # rows/cols in arcsec about the tangent point
    return function (y, x)
        dy = y - cy; dx = x - cx
        dy, dx = dy + q[1] * (dx^2 - dy^2), dx + q[2] * dx * dy + q[3] * dy^2
        r = scale * (c * dy - s * dx); k = scale * (s * dy + c * dx)  # north and west offsets, arcsec
        proj === :tan && return grid_to_world(g1, r, k)
        # orthographic (SIN): plane offsets are direction cosines
        l = -k / 206264.80624709636; m = r / 206264.80624709636; n = sqrt(1 - l^2 - m^2)
        sa, ca = sincosd(ra_t); sd, cd = sincosd(dec_t)
        p = n .* (cd * ca, cd * sa, sd) .+ l .* (-sa, ca, 0.0) .+ m .* (-sd * ca, -sd * sa, cd)
        return (mod(atand(p[2], p[1]), 360), asind(p[3]))
    end
end

# Sky + Gaussian sources (positions on `grid`, so all images see the same scene) + noise.
function case_image(rng, to_sky, dims, grid, srcs; sigma = 5.0, sky = 10.0, band = :A)
    data = [begin
                r, c = world_to_grid(grid, to_sky(i, j)...)
                sky + sum(f * exp(-((r - sr)^2 + (c - sc)^2) / (2 * w^2)) for (sr, sc, f, w) in srcs; init = 0.0)
            end for i in 1:dims[1], j in 1:dims[2]]
    data .+= sigma .* randn(rng, dims...)
    return JointImage(data, fill(1 / sigma^2, dims), nothing, to_sky; band)
end

"""
    coadd_cases() -> Vector of (; name, images, grid, pixfrac)

Edge cases for the drizzle comparison: subpixel shifts, roll angles, mixed input scales, a
non-TAN projection with distortion, partial overlap with masked and NaN pixels and varying
weights, and an output coarser than the inputs.
"""
function coadd_cases()
    rng = StableRNG(2026)
    out(scale, n) = TanGrid(COADD_RA0, COADD_DEC0, scale, n, n)
    srcs(grid, k) = [(grid.ny * (0.2 + 0.6rand(rng)), grid.nx * (0.2 + 0.6rand(rng)), 200 + 800rand(rng), 2 + rand(rng)) for _ in 1:k]
    sky(scale, θ, n; dy = 0.0, dx = 0.0, kw...) = case_sky(COADD_RA0, COADD_DEC0, scale, θ, (n + 1) / 2 + dy, (n + 1) / 2 + dx; kw...)
    cases = []
    g = out(0.05, 40); S = srcs(g, 5)
    push!(cases, (; name = "shift", grid = g, pixfrac = 1.0,
                  images = [case_image(rng, sky(0.1, 0.0, 24; dy = 0.3, dx = 0.7), (24, 24), g, S), case_image(rng, sky(0.1, 0.0, 24; dy = -0.4, dx = 0.1), (24, 24), g, S)]))
    g = out(0.05, 48); S = srcs(g, 6)
    push!(cases, (; name = "roll", grid = g, pixfrac = 0.7,
                  images = [case_image(rng, sky(0.1, θ, 26; dy = 0.25k, dx = -0.3k), (26, 26), g, S) for (k, θ) in enumerate((0.0, 30.0, 77.0))]))
    g = out(0.05, 48); S = srcs(g, 6)
    push!(cases, (; name = "mixed_scale", grid = g, pixfrac = 1.0,
                  images = [case_image(rng, sky(0.1, 10.0, 24), (24, 24), g, S), case_image(rng, sky(0.17, 0.0, 14; dy = 0.2), (14, 14), g, S)]))
    # SIN projected about a point 4000 px (400") away, placed so the image center lands near the
    # grid center (13 px off: north rotates between the two tangent points), and a TAN image
    # with quadratic distortion
    g = out(0.05, 48); S = srcs(g, 6)
    s20, c20 = sincosd(20.0)
    far_t = grid_to_world(TanGrid(COADD_RA0, COADD_DEC0, 1.0, 1, 1, 0.0, 0.0), 400c20, 400s20)
    far = case_sky(far_t..., 0.1, 20.0, 12.5 + 4000, 12.5; proj = :sin)
    push!(cases, (; name = "projection_distortion", grid = g, pixfrac = 0.8,
                  images = [case_image(rng, far, (24, 24), g, S), case_image(rng, sky(0.1, -15.0, 24; q = (5e-4, -4e-4, 3e-4)), (24, 24), g, S)]))
    # images reaching past the grid, a masked block with NaN data, spatially varying weights
    g = out(0.05, 40); S = srcs(g, 4)
    a = case_image(rng, sky(0.1, 5.0, 24; dy = 8.0, dx = 6.0), (24, 24), g, S)
    a.data[3:8, 10:15] .= NaN; a.weight[3:8, 10:15] .= 0
    b = case_image(rng, sky(0.1, -20.0, 24; dy = -9.0), (24, 24), g, S)
    b.weight .*= [1 + 0.5 * sin(i / 3) * cos(j / 4) for i in 1:24, j in 1:24]
    push!(cases, (; name = "partial_masked", grid = g, pixfrac = 1.0, images = [a, b]))
    g = out(0.2, 16); S = srcs(g, 4)
    push!(cases, (; name = "coarse_output", grid = g, pixfrac = 1.0,
                  images = [case_image(rng, sky(0.1, θ, 30; dy = 0.3k, dx = 0.2k), (30, 30), g, S) for (k, θ) in enumerate((0.0, 45.0, 90.0))]))
    return cases
end

"""
    case_drizzle_inputs(img, grid, pixfrac) -> (; data, weight, var, pixmap)

One image as drizzle inputs that make drizzle compute what `coadd` does: values in flux per
output pixel (`d f`, `f` the output pixel's area in input pixels), weights `p² w / f³` (drizzle
weights by the fraction of the drop in the output pixel; `coadd` by the overlap area, `p²/f`
times that, with the inverse variance `w / f²` of `d f`), variance `f² / w`, and the 0-based
output `(x, y)` of each input pixel center.  `f` is evaluated at each input pixel from the
exact map; masked pixels get weight 0 and value 0.
"""
function case_drizzle_inputs(img, grid, pixfrac)
    ny, nx = size(img.data)
    rc(y, x) = world_to_grid(grid, img.to_sky(y, x)...)
    h = 1e-3
    f = [begin   # |d(r, c)/d(y, x)| by central differences, then its inverse
             a = rc(i + h, j) .- rc(i - h, j); b = rc(i, j + h) .- rc(i, j - h)
             (2h)^2 / abs(a[1] * b[2] - a[2] * b[1])
         end for i in 1:ny, j in 1:nx]
    ok = isfinite.(img.data) .& (img.weight .> 0)
    pm = Array{Float64}(undef, ny, nx, 2)
    for j in 1:nx, i in 1:ny
        r, c = rc(i, j); pm[i, j, 1] = c - 1; pm[i, j, 2] = r - 1
    end
    return (; data = ifelse.(ok, img.data .* f, 0.0), weight = ifelse.(ok, pixfrac^2 .* img.weight ./ f .^ 3, 0.0),
              var = ifelse.(ok, f .^ 2 ./ img.weight, 0.0), pixmap = pm)
end
