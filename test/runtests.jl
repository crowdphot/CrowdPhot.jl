using SafeTestsets
using Test

# Documenter evaluates @meta blocks in documentation source files
# (CurrentModule, etc.) via Core.eval(Main, ...).
# When run under @safetestset, each test file gets its own anonymous module, so
# imports in doctests.jl don't create bindings in Main.  We import here so that
# CrowdPhot is visible where Documenter expects it.
import CrowdPhot

# Run the PSF API, parametric fitting, and empirical PSF coverage together.
@safetestset "PSF API tests" include("psf/common_psf_tests.jl")
@safetestset "PSF Fitting" include("psf/psf_fitting_tests.jl")
@safetestset "PSF Fit Parity" include("psf/psf_fit_parity_tests.jl")
@safetestset "Empirical PSF models" include("psf/empirical_model_tests.jl")
@safetestset "Gridded PSF model" include("psf/gridded_psf_tests.jl")
@safetestset "Roman CRDS ePSF" include("psf/roman_crds_epsf_tests.jl")
@safetestset "Simulation tests" include("simulation_test.jl")
@safetestset "Background estimation" include("background_tests.jl")
@safetestset "Centroids" include("centroids_tests.jl")
@safetestset "Morphology" include("morphology_tests.jl")
@safetestset "Correlation" include("correlation_tests.jl")
@safetestset "Apertures" include("apertures_tests.jl")
@safetestset "Curve of Growth" include("curve_of_growth_tests.jl")
@safetestset "Detection" include("detection_tests.jl")
@safetestset "Utilities" include("utilities/calc_total_error_tests.jl")
@safetestset "Bessel functions" include("bessels.jl")
@safetestset "Photometry" include("photometry/psf_photometry_single.jl")
@safetestset "Multi-pass simultaneous photometry" include("photometry/psf_photometry_simultaneous_multipass.jl")
@safetestset "Doctests" include("doctests.jl")
