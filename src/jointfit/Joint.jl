"""
    CrowdPhot.Joint

Joint photometry of many images (dithers, epochs, bands) with one parameter set per source:
sky position on a common [`TanGrid`](@ref) frame and one flux per band.  Built on
KernelAbstractions, so the same kernels run on the CPU (threaded) and on GPUs; pass
`backend = MetalBackend()` (or another KA backend) where a function takes one.
"""
module Joint

import Adapt
import Atomix
using KernelAbstractions
const KA = KernelAbstractions
using SpecialFunctions: erf, erfc, gamma_inc_inv
using StaticArrays: SMatrix, SVector, MVector
using StructArrays: StructArrays, StructArray
using SHA: sha256
import TOML
using LinearAlgebra: opnorm, dot, svd, Diagonal
import Random
using Statistics: median

import ..CrowdPhot
import ..PSF
using ..PSF: CircularGaussianPRF, ImagePSF, GriddedPSFModel
using ..Background: MeanBackground, MedianBackground, SExtractorBackground, MMMBackground, StdRMS, MADStdRMS

export TanGrid, world_to_grid, grid_to_world, oversample, fits_wcs, PolyMap, JointImage, JointScene
export JointDetector, JointFitOptions, JointLoopOptions, PSFRefineOptions, fit_joint_multipass, fit_joint_tiled, plan_tiles, coadd
export ImageFootprint, fit_tile, merge_tiles, write_plan, read_plan, read_job
export NeighborList, pairs_within, cross_pairs, nearest_within

include("frame.jl")
include("polymap.jl")
include("launch.jl")
include("psfs.jl")
include("scene.jl")
include("render.jl")
include("neighbors.jl")
include("linearize.jl")
include("solve.jl")
include("detect.jl")
include("background.jl")
include("finalize.jl")
include("psfrefine.jl")
include("loop.jl")
include("coadd.jl")
include("tiling.jl")
include("planfile.jl")

end # module Joint
