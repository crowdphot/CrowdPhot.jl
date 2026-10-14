using SafeTestsets
using Test

# Run the PSF API, parametric fitting, and empirical PSF coverage together.
@safetestset "PSF API tests" include("psf/common_psf_tests.jl")
@safetestset "PSF Fitting" include("psf/psf_fitting_tests.jl")
@safetestset "PSF Fit Parity" include("psf/psf_fit_parity_tests.jl")
@safetestset "Empirical PSF models" include("psf/empirical_model_tests.jl")
@safetestset "Simulation tests" include("simulation_test.jl")
@safetestset "Background estimation" include("background_tests.jl")
@safetestset "Centroids" include("centroids_tests.jl")
@safetestset "Correlation" include("correlation_tests.jl")
@safetestset "Detection" include("detection_tests.jl")
