module CrowdPhot

import ConstructionBase
using FillArrays: Fill
import Krylov
using LinearAlgebra: cholesky, cholesky!, ldiv!, dot, norm, I, Symmetric, pinv, PosDefException, svd
import LinearOperators
import LoopVectorization as LV
import LossFunctions
using Printf: @sprintf
import Random
import SparseArrays
using StaticArrays: SMatrix, SVector, @SMatrix, @SVector
using Statistics: median, median!, mean, std, quantile

export Background2D
export sigma_clip, sigma_clip!, calc_total_error
export simulate_sources, simulate_image, make_gaussians_image, centroid_poly, choose_centroid
export matched_filter, MatchedFilterResult
export measure_star_shape, measure_star_shape_ref, measure_star_shapes, FlatWindow, GaussianWindow
export MultiPassPhotResult, fit_all_stars, fit_all_stars_simultaneous_multipass
export CurveOfGrowth, curve_of_growth, encircled_energy, radius_at_energy, normalize, reference_cog

include("correlation.jl")
include("utilities.jl")
include("bessels.jl")
using .Bessels
include("levenberg_marquardt.jl")
include("psf/PSF.jl")
using .PSF
include("detection.jl")
include("simulation.jl")
include("background/background.jl")
using .Background
include("centroids.jl")
include("apertures.jl")
include("morphology.jl")
include("curve_of_growth.jl")
include("photometry/psf_photometry_diagnostics.jl")
include("photometry/psf_photometry_single.jl")
include("photometry/psf_photometry_simultaneous_multipass.jl")
include("precompile.jl")

end # module CrowdPhot
