# When adding a benchmark suite, update SUITE_NAMES and SUITE_TITLES, then
# define its BenchmarkGroup under SUITE using the same suite name.
using CrowdPhot: make_gaussians_image, simulate_image, centroid_poly, _centroid_poly3, correlate, findlocalmaxima
using CrowdPhot.PSF: GaussianPSF, CircularGaussianPSF, CircularGaussianPRF, evaluate, evaluate_fg, fit_star, fit_psf, TukeyLoss, bicubic_interpolate, fill_grid_holes!, ImagePSF
using CrowdPhot.Background
import BackgroundMeshes as BM
using BenchmarkTools
using ImageFiltering: imfilter, findlocalmaxima as _if_findlocalmaxima
import LossFunctions
using OffsetArrays: centered
using PrettyTables: pretty_table
using StableRNGs: StableRNG

function show_benchmarks(results)
    # Collect results — results may be a flat Dict or a nested BenchmarkGroup;
    # flatten first so that every value is a Trial.
    flat = flatten_results(results)
    # Sort so that paired CrowdPhot / external-package benchmarks appear
    # adjacent, with the CrowdPhot entry first in each pair.
    sorted  = sort(collect(flat), by=pair -> begin
        key = pair.first
        base = replace(key, r"\s*,?\s*(?:BackgroundMeshes|ImageFiltering)\.jl" => "")
        ext  = occursin(r"(?:BackgroundMeshes|ImageFiltering)\.jl", key)
        return (base, ext)
    end)
    names   = [k for (k,_) in sorted]
    trials  = [v for (_,v) in sorted]

    # Pack into matrix
    data = hcat(
        names,
        [BenchmarkTools.prettytime(median(t).time) for t in trials],
        [BenchmarkTools.prettymemory(median(t).memory) for t in trials],
        [median(t).allocs for t in trials]
    )

    # Make pretty table
    pretty_table(data;
        column_labels = ["Benchmark", "Median Time", "Memory", "Allocs"],
        alignment     = [:l, :r, :r, :r]
    )
end

function flatten_results(group)
    # Recursively flatten a (possibly nested) BenchmarkGroup of Trial results
    # into a flat Dict{String, Trial}.  Dict inputs are returned as-is so that
    # the function is idempotent.
    flat = Dict{String, Any}()
    _flatten_results!(flat, group, "")
    return flat
end

function _flatten_results!(flat, group, prefix)
    for (k, v) in group
        fullname = isempty(prefix) ? string(k) : "$prefix/$k"
        if v isa BenchmarkGroup
            _flatten_results!(flat, v, fullname)
        else
            flat[fullname] = v
        end
    end
end

const SUITE_NAMES = ["parametric", "fitting", "empirical", "background", "centroids", "correlation", "peakfinding"]
const SUITE_TITLES = Dict(
    "parametric" => "Parametric Suite",
    "fitting" => "Fitting Suite",
    "empirical" => "Empirical Suite",
    "background" => "Background Estimation Suite",
    "centroids" => "Centroids Suite",
    "correlation" => "Correlation Suite",
    "peakfinding" => "Peak Finding Suite",
)

function selected_suite_names(args)
    # Default to every suite when no command-line filters are provided.
    isempty(args) && return SUITE_NAMES

    # Fail early on misspelled suite names so benchmark output is unambiguous.
    unknown = setdiff(args, SUITE_NAMES)
    if !isempty(unknown)
        println(stderr, "Unknown benchmark suite(s): ", join(unknown, ", "))
        println(stderr, "Usage: julia benchmarks.jl [", join(SUITE_NAMES, "|"), "]...")
        exit(1)
    end

    # Preserve the user's requested order while avoiding duplicate runs.
    return unique(args)
end

function print_suite_header(name)
    # Format each selected suite consistently with the existing benchmark output.
    println("⎯⎯⎯ ", SUITE_TITLES[name], " ", repeat("⎯", 42))
end

function run_selected_suites(args)
    # Parse command-line filters once before executing any benchmark work.
    names = selected_suite_names(args)

    # Run and render each selected suite independently.
    for name in names
        results = run(SUITE[name], verbose=true)
        print_suite_header(name)
        show_benchmarks(results)
    end
end

const SUITE = BenchmarkGroup()
SUITE["parametric"] = BenchmarkGroup()

let model = CircularGaussianPSF(x=15.0, y=15.0, fwhm=4.0, flux=10.0, bkg=1.0)
    idx = CartesianIndex(15, 15)
    SUITE["parametric"]["circular_gaussian_evaluate_fg"] = @benchmarkable evaluate_fg($model, $idx)
end

# ---------------------------------------------------------------------------
# LM fitting benchmarks
# ---------------------------------------------------------------------------
SUITE["fitting"] = BenchmarkGroup()

let model = CircularGaussianPSF(x=15.0, y=15.0, fwhm=4.0, flux=10.0, bkg=1.0)
    inds  = (1:30, 1:30)
    image = evaluate.(model, inds[1], inds[2]')
    init = CircularGaussianPSF(x=15.5, y=14.5, fwhm=3.5, flux=9.0, bkg=1.2)
    SUITE["fitting"]["fit_star CircularGaussianPSF (L2)"] = @benchmarkable fit_star($init, $image, $inds)
    SUITE["fitting"]["fit_star CircularGaussianPSF (Huber IRLS)"] = @benchmarkable fit_star($init, $image, $inds;
        reweight=$(LossFunctions.HuberLoss(1.0)))
    SUITE["fitting"]["fit_star CircularGaussianPSF (Tukey IRLS)"] = @benchmarkable fit_star($init, $image, $inds;
        reweight=$(TukeyLoss()))
end

# ---------------------------------------------------------------------------
# Empirical model benchmarks
# ---------------------------------------------------------------------------
SUITE["empirical"] = BenchmarkGroup()
SUITE["empirical"]["bicubic_interpolate"] = @benchmarkable bicubic_interpolate(x, $3.5, $3.5) setup=(x=rand(7,7))
SUITE["empirical"]["fill_grid_holes!"] = @benchmarkable fill_grid_holes!(x) setup=(x=rand(21,21); inds=([9, 4, 15, 13, 1, 1], [6, 15, 17, 1, 17, 1]); x[inds...] .= NaN) evals=1

for n in (5, 11, 21)
    SUITE["empirical"]["ImagePSF fit_star, size=($n, $n)"] = @benchmarkable fit_star(init, image, inds; max_iter = 100) setup=(begin
        origin = (($n + 1) / 2, ($n + 1) / 2)
        grid_model = CircularGaussianPRF(x = origin[1], y = origin[2], fwhm = 2.4, flux = 1, bkg = 0)
        psf_data = evaluate.(grid_model, 1:$n, (1:$n)')
        truth = ImagePSF(psf_data; x = origin[1] + 0.35, y = origin[2] - 0.25,
            flux = 300.0, bkg = 4.0, origin, normalize = true)
        image = evaluate.(truth, 1:$n, (1:$n)')
        init = ImagePSF(psf_data; x = origin[1], y = origin[2] + 0.1,
            flux = 260.0, bkg = 3.5, origin, normalize = true)
        inds = (1:$n, 1:$n)
    end)
end

for n in (50, 100)
    SUITE["empirical"]["ImagePSF fit_psf, n=$n"] = @benchmarkable psf, result =
        fit_psf(ImagePSF, image, sources.x, sources.y;
            psf_rad = 5.0, oversampling = 2, smooth = true, recenter = false,
            reweight = nothing) setup=(begin
                truth_model = CircularGaussianPRF(x = 0, y = 0, fwhm = 1.8, flux = 1, bkg = 0)
                image, sources = simulate_image((128, 128), truth_model, $n;
                    background = 20.0, noise = :none, flux = (600.0, 900.0),
                    min_separation = 7, border = 8, model_radius = 6)
            end) evals=1 samples=100
end

SUITE["background"] = BenchmarkGroup()
# Direct test of estimators
SUITE["background"]["estimators"] = BenchmarkGroup()
for t in (
    (MMMBackground(), BM.MMMBackground()), 
    (BiweightLocationBackground(), BM.BiweightLocationBackground()),
    (StdRMS(), BM.StdRMS()),
    (BiweightScaleRMS(), BM.BiweightScaleRMS()),
)
    img = make_gaussians_image(100, (100, 100); rng = StableRNG(7), background = 200.0, read_noise = 5.0, gain = 1.5)
    SUITE["background"]["estimators"]["$(typeof(t[1])) (100, 100)"] = @benchmarkable $t[1]($img)
    SUITE["background"]["estimators"]["$(typeof(t[1])) BackgroundMeshes.jl (100, 100)"] = @benchmarkable $t[2]($img)
end
# Test estimate_background on a range of image sizes, comparing against BackgroundMeshes.jl
for s in (30, 500, 2000)
    img = make_gaussians_image(s, (s, s); rng = StableRNG(7), background = 200.0, read_noise = 5.0, gain = 1.5)
    SUITE["background"]["MMMBackground, size=($s, $s)"] = @benchmarkable estimate_background($img; estimator = $MMMBackground(), rms_estimator = $StdRMS(), maxiters=$0)
    SUITE["background"]["MMMBackground, size=($s, $s), nclip=5"] = @benchmarkable estimate_background($img; estimator = $MMMBackground(), rms_estimator = $StdRMS(), maxiters=$5) # Test sigma clipping overhead
    SUITE["background"]["MMMBackground, BackgroundMeshes.jl, size=($s, $s)"] = @benchmarkable BM.estimate_background($img; location = $BM.MMMBackground(), rms = $BM.StdRMS())
end

let img = make_gaussians_image(1000, (1000, 1000); rng = StableRNG(7), background = 200.0, read_noise = 5.0, gain = 1.5)
    SUITE["background"]["Background2D (1000, 1000), 64 box, no filter"] = @benchmarkable Background2D($img, $64; estimator = MMMBackground(), rms_estimator = StdRMS(), maxiters=0)
    SUITE["background"]["Background2D BackgroundMeshes.jl (1000, 1000), 64 box, no filter"] = @benchmarkable BM.estimate_background($img, $64; location = $BM.MMMBackground(), rms = $BM.StdRMS())
end
# SUITE["background"]["Background2D (1000, 1000), BackgroundMeshes.jl, 64 box"] = @benchmarkable BM.Background2D($img, 64) setup=(img = make_gaussians_image(1000, (1000, 1000); rng = StableRNG(7), background = 200.0, read_noise = 5.0, gain = 1.5))
# SUITE["background"]["Background2D (1000, 1000), 64 box"] = @benchmarkable Background2D($img, 64) setup=(img = make_gaussians_image(1000, (1000, 1000); rng = StableRNG(7), background = 200.0, read_noise = 5.0, gain = 1.5))

# ---------------------------------------------------------------------------
# Centroids benchmarks
# ---------------------------------------------------------------------------
SUITE["centroids"] = BenchmarkGroup()

# Benchmark _centroid_poly3 directly on a 3×3 patch
let model = CircularGaussianPSF(x=2.0, y=2.0, fwhm=4.0, flux=10.0, bkg=0.0)
    inds = (1:3, 1:3)
    patch = evaluate.(model, inds[1], inds[2]')
    inv_var = ones(3, 3)
    SUITE["centroids"]["_centroid_poly3 (3x3)"] = @benchmarkable _centroid_poly3($patch, $inv_var)
end

# Benchmark centroid_poly on a larger cutout
for n in (7, 15)
    let model = CircularGaussianPSF(x=4.0, y=4.0, fwhm=4.0, flux=10.0, bkg=0.0)
        inds = (1:n, 1:n)
        img = evaluate.(model, inds[1], inds[2]')
        SUITE["centroids"]["centroid_poly ($n×$n)"] = @benchmarkable centroid_poly($img)
    end
end

# Benchmark centroid_poly on a larger cutout with explicit inverse variance
let model = CircularGaussianPSF(x=8.0, y=8.0, fwhm=4.0, flux=10.0, bkg=0.0)
    inds = (1:15, 1:15)
    img = evaluate.(model, inds[1], inds[2]')
    ivar = ones(15, 15)
    SUITE["centroids"]["centroid_poly (15×15), explicit ivar"] = @benchmarkable centroid_poly($img, $ivar)
end

# ---------------------------------------------------------------------------
# Correlation benchmarks — CrowdPhot.jl vs ImageFiltering.jl
# Note that ImageFiltering.jl has auto-algorithm selection; for small, separable
# kernels it uses direct, multi-threaded convolution `FIRTiled` with `CPUThreads`,
# while for large kernels it uses `FFT()`. For kernel sizes 5, 11 the direct convolution
# is used, while for 21 the FFT is used.
# ---------------------------------------------------------------------------
SUITE["correlation"] = BenchmarkGroup()

# Separable kernel: 2D Gaussian (always rank-1, SVD detection should fire).
function _bench_gaussian(σ, k)
    x = LinRange(-(k ÷ 2), k ÷ 2, k)
    g = exp.(-0.5 .* (x ./ σ) .^ 2)
    g ./= sum(g)
    return g * g'
end

for ksize in ((5, 5), (11, 11), (21, 21))
    for imsize in ((500, 500), (1000, 1000), (2000, 2000))
        # ---- separable (Gaussian) ----
        let k = ksize, sz = imsize
            σ = k[1] / 5
            label = "$sz, k=$k, separable"
            kern  = _bench_gaussian(σ, k[1])
            ckern = centered(kern)
            SUITE["correlation"]["$label, CrowdPhot.jl"] = @benchmarkable correlate(img, $(kern), :replicate) setup=(
                img = rand($sz...)
            ) samples=3
            SUITE["correlation"]["$label, ImageFiltering.jl"] = @benchmarkable imfilter(img, $(ckern), "replicate") setup=(
                img = rand($sz...)
            ) samples=3
        end
        # ---- non-separable (random full-rank) ----
        let k = ksize, sz = imsize
            label = "$sz, k=$k, non-separable"
            SUITE["correlation"]["$label, CrowdPhot.jl"] = @benchmarkable correlate(img, kern, :replicate) setup=(
                img = rand($sz...); kern = rand($k...)
            ) samples=3
            SUITE["correlation"]["$label, ImageFiltering.jl"] = @benchmarkable imfilter(img, ckern, "replicate") setup=(
                img = rand($sz...); kern = rand($k...); ckern = centered(kern)
            ) samples=3
        end
    end
end

# ---------------------------------------------------------------------------
# Peak-finding benchmarks — CrowdPhot.jl vs ImageFiltering.jl
# ---------------------------------------------------------------------------
SUITE["peakfinding"] = BenchmarkGroup()

for (sz, label) in [((100, 100), "100x100"), ((500, 500), "500x500"), ((2000, 2000), "2000x2000")]
    let s = sz, lbl = label
        SUITE["peakfinding"]["$lbl, CrowdPhot.jl"] = @benchmarkable findlocalmaxima(img) setup=(
            img = rand($s...)
        ) samples=10
        SUITE["peakfinding"]["$lbl, ImageFiltering.jl"] = @benchmarkable _if_findlocalmaxima(img; edges=true) setup=(
            img = rand($s...)
        ) samples=10
    end
end

# If not on CI, we'll show a nice table
if get(ENV, "CI", "false") == "false"
    # Run the requested benchmarks and print a table for each suite.
    run_selected_suites(ARGS)
end
