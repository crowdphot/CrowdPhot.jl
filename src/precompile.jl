using PrecompileTools: @setup_workload, @compile_workload

# @compile_workload begin
#     for T in (Float32, Float64)
#         # vv = [CircularGaussianPRF(y = T(gy), x = T(gx), fwhm = T(3.0), flux = T(1.0), bkg = T(0.0)) for (gy, gx) in ((0.0, 0.0), (0.0, 10.0), (10.0, 0.0), (10.0, 10.0))]
#         # Note that different keyword arguments shapes (e.g., ommitting some) 
#         # require different compilations, so this precompiles
#         # only for the specific set of keyword arguments used here. If you want
#         # to precompile for other sets of keyword arguments, you need to call
#         # GriddedPSFModel with those arguments as well.
#         # psf = GriddedPSFModel(
#         #            vv,
#         #            T[0.0, 0.0, 10.0, 10.0], T[0.0, 10.0, 0.0, 10.0];
#         #             y = T(2.4), x = T(1.3), flux = T(120.0), bkg = T(10.0),
#         #         )

#         Background2D(randn(T, 50, 50), 5; coverage_mask=zeros(Bool, 50, 50), fill_value=T(NaN))
#         mf = matched_filter(zeros(T, 10, 10), zeros(T, 3, 3); inv_var=ones(T, 10, 10), sigma=T(5.0))
#         measure_star_shapes(mf; half_width=2)
#     end
# end

# Test for if Julia is precompiling:
# https://discourse.julialang.org/t/is-it-possible-to-detect-if-julia-is-ahead-of-time-precompiling/78631/3

# if ccall(:jl_generating_output, Cint, ()) == 1 

# Common calls *on* PSFs.  The forms of `render!` / `add_star!` / `subtract_star!`
# taking extra UnitRange{Int} args are the methods fitters call once per source; the other
# `add_star!` / `subtract_star!` methods are the whole-frame convenience entry points.
for psf in (PSF.CircularGaussianPSF, CircularGaussianPRF)
            # (PSF.AiryPSF, PSF.GaussianPSF, PSF.GaussianPRF, PSF.CircularMoffatPSF, PSF.MoffatPSF)
    for T in (Float32, Float64)
        psfT = psf{T}
        precompile(evaluate, (psfT, T, T))
        precompile(render, (psfT,))
        precompile(PSF.render!, (Matrix{T}, psfT, UnitRange{Int}, UnitRange{Int}, Nothing))
        precompile(PSF.add_star!, (Matrix{T}, psfT))
        precompile(PSF.subtract_star!, (Matrix{T}, psfT))
        precompile(PSF.add_star!, (Matrix{T}, psfT, UnitRange{Int}, UnitRange{Int}))
        precompile(PSF.subtract_star!, (Matrix{T}, psfT, UnitRange{Int}, UnitRange{Int}))
    end
end

# The empirical models carry a second type parameter -- the backing array, and
# for the grid the node model -- so they cannot go through the loop above.
# `GriddedPSFModel{T, <:ImagePSF}` also has the scratch-accelerated `render!`
# specialization, which is the path the simultaneous fitters take.
for T in (Float32, Float64)
    ip = PSF.ImagePSF{T, Matrix{T}}
    gp = PSF.GriddedPSFModel{T, ip}
    for m in (ip, gp)
        precompile(evaluate, (m, T, T))
        precompile(render, (m,))
        precompile(PSF.render!, (Matrix{T}, m, UnitRange{Int}, UnitRange{Int}, Nothing))
        precompile(PSF.add_star!, (Matrix{T}, m, UnitRange{Int}, UnitRange{Int}))
        precompile(PSF.subtract_star!, (Matrix{T}, m, UnitRange{Int}, UnitRange{Int}))
    end
    precompile(PSF._render_scratch, (gp, Int, Type{T}))
    precompile(PSF.render!, (Matrix{T}, gp, UnitRange{Int}, UnitRange{Int},
                             NTuple{3, NTuple{4, Matrix{T}}}))
end

for T in (Float32, Float64)
    precompile(AiryPSF, (T, T, T, T, T))
    precompile(CircularGaussianPSF, (T, T, T, T, T))
    precompile(CircularGaussianPRF, (T, T, T, T, T))
    precompile(GaussianPSF, (T, T, T, T, T, T, T))
    precompile(GaussianPRF, (T, T, T, T, T, T, T))
    precompile(CircularMoffatPSF, (T, T, T, T, T, T))
    precompile(MoffatPSF, (T, T, T, T, T, T, T, T))
    precompile(ImagePSF, (Matrix{T},))
    # precompile(GriddedPSFModel, (Vector{<:AbstractPSFModel}, Vector{T}, Vector{T}))
    # precompile(GriddedPSFModel, (Vector{CircularGaussianPRF{T}}, Vector{T}, Vector{T}))
    precompile(Core.kwcall, (NamedTuple{(:coverage_mask, :fill_value)}, typeof(Background2D), Matrix{T}, Int))
    precompile(Core.kwcall, (NamedTuple{(:inv_var, :sigma), Tuple{Matrix{T}, T}}, typeof(matched_filter), Matrix{T}, Matrix{T}))
    precompile(Core.kwcall, (NamedTuple{(:half_width,), Tuple{Int}}, typeof(measure_star_shapes), MatchedFilterResult{T}))
end

# ==============================================================================
# Workload
# ==============================================================================

# `precompile` directives above only cache method signatures the caller names.
# A workload additionally caches every specialization the call actually reaches,
# including ones inside other packages -- which is the point here: more than half
# of `Roman.load_l2`'s first-call cost is inference and codegen inside ASDF.jl,
# and no `precompile` directive we could write reaches it.
#
# Measured on a 4088x4088 Roman L2: running `load_l2` once on a tiny synthetic
# file first drops a subsequent real load from 7.9 s of compilation to 0.6 s.
# This moves that 7.9 s to package build time.
#
# Everything runs at Float32, the eltype Roman data actually uses.  Adding
# Float64 would roughly double both the precompile cost and the cache size for a
# path no Roman user takes.
@setup_workload begin
    # A 4-node grid of 25x25 Gaussian renders.  The node model type is what the
    # simultaneous fitter specializes its render and stamp-fill paths on, and it
    # is the same type `crds_gridded_epsf` returns, so the two share this work.
    nodes = [ImagePSF(Matrix{Float32}(PSF.render!(Matrix{Float32}(undef, 25, 25),
                                                  CircularGaussianPSF(13.0f0, 13.0f0, 3.0f0, 1.0f0, 0.0f0),
                                                  1:25, 1:25)))
             for _ in 1:4]
    # One coordinate per node, not grid axes: a 2x2 grid spanning the frame below.
    gpsf = GriddedPSFModel(nodes, Float32[1, 1, 60, 60], Float32[1, 60, 1, 60])

    # A 5-D CRDS-style ePSF array, and a tiny L2 with the element types the real
    # files use: data Float32, err and var_poisson Float16, dq UInt32.  The
    # element types matter; the array sizes do not.
    epsf5 = fill(1.0f0, 9, 9, 4, 1, 1)
    # `ASDF` is imported inside the `Roman` submodule, not here.
    nd(a, t) = Roman.ASDF.NDArray(Roman.ASDF.LazyBlockHeaders(), nothing, a,
                                  reverse(collect(size(a))), t, nothing)

    @compile_workload begin
        mktempdir() do dir
            l2 = joinpath(dir, "l2.asdf")
            Roman.ASDF.save(l2, Dict("roman" => Dict(
                "meta" => Dict("exposure" => Dict("effective_exposure_time" => 1.0)),
                "data" => nd(rand(Float32, 8, 8), "float32"),
                "err" => nd(ones(Float16, 8, 8), "float16"),
                "var_poisson" => nd(ones(Float16, 8, 8), "float16"),
                "dq" => nd(zeros(UInt32, 8, 8), "uint32"))))
            Roman.load_l2(l2)
            Roman.load_area(l2)

            ep = joinpath(dir, "epsf.asdf")
            Roman.ASDF.save(ep, Dict("roman" => Dict(
                "meta" => Dict("pixel_x" => [0.0, 40.0, 0.0, 40.0],
                               "pixel_y" => [0.0, 0.0, 40.0, 40.0],
                               "oversample" => 1, "spectral_type" => ["G2V"], "defocus" => [0]),
                "psf" => nd(epsf5, "float32"))))
            Roman.crds_gridded_epsf(ep)
        end

        img, _ = simulate_image((60, 60), gpsf, 6; background = 50.0f0,
                                rng = Random.Xoshiro(1))
        res = fit_all_stars_simultaneous_multipass(Float32.(img), gpsf, 3.0f0;
            max_iter = 2, min_iter = 1, show_trace = false)
        to_table(res)
    end
end