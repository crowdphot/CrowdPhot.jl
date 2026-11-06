"""
    CrowdPhot.Roman

Everything specific to the Nancy Grace Roman Space Telescope.
Includes reading WFI data products, CRDS reference files, and interpreting
data quality flags.
"""
module Roman

using ASDF
import Logging
using Statistics: median
using StructArrays: StructArray

using ..CrowdPhot: correlate
import ..CrowdPhot.PSF
using ..CrowdPhot.PSF: ImagePSF, GriddedPSFModel
using ..CrowdPhot.Joint: TanGrid, grid_to_world

export DQ_FLAGS, dq_mask_value, parse_dq_mask, load_l2, load_area, crds_gridded_epsf,
    jansky_per_flux_unit, ProjectionRegion, projection_regions, projection_region, region_to_world, skycells

include("dq.jl")
include("io.jl")
include("photom.jl")
include("epsf.jl")
include("skycells.jl")

end # module Roman
