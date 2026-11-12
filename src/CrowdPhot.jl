module CrowdPhot

import ConstructionBase
using FillArrays: Fill
using LinearAlgebra: cholesky, cholesky!, ldiv!, I, Symmetric, pinv, PosDefException, svd
import LossFunctions
import Random
using StaticArrays: SMatrix, @SMatrix, @SVector
using Statistics: median, median!, mean, std

export Background2D
export simulate_sources, simulate_image, make_gaussians_image, centroid_poly, choose_centroid
export matched_filter, MatchedFilterResult

include("correlation.jl")
include("utilities.jl")
include("levenberg_marquardt.jl")
include("psf/PSF.jl")
using .PSF
include("detection.jl")
include("simulation.jl")
include("background/background.jl")
using .Background
include("centroids.jl")

end # module CrowdPhot
