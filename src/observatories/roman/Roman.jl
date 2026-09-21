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

using ..CrowdPhot: correlate
import ..CrowdPhot.PSF
using ..CrowdPhot.PSF: ImagePSF, GriddedPSFModel

export DQ_FLAGS, dq_mask_value, parse_dq_mask, load_l2, load_area, crds_gridded_epsf,
    jansky_per_flux_unit

include("dq.jl")
include("io.jl")
include("photom.jl")
include("epsf.jl")

end # module Roman
