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
using SpecialFunctions: erf
using StaticArrays: SMatrix, SVector
using StructArrays: StructArray
using LinearAlgebra: opnorm
using Statistics: median

import ..CrowdPhot
import ..PSF
using ..PSF: CircularGaussianPRF, ImagePSF, GriddedPSFModel
using ..Background: MeanBackground, MedianBackground, SExtractorBackground, MMMBackground, StdRMS, MADStdRMS

export TanGrid, world_to_grid, grid_to_world, PolyMap, JointImage, JointScene
export JointDetector, JointFitOptions, JointLoopOptions, fit_joint_multipass

include("frame.jl")
include("polymap.jl")
include("launch.jl")
include("psfs.jl")
include("scene.jl")
include("render.jl")
include("linearize.jl")
include("solve.jl")
include("detect.jl")
include("background.jl")
include("finalize.jl")
include("loop.jl")

end # module Joint
