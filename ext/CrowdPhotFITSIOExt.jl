module CrowdPhotFITSIOExt

using CrowdPhot.Joint: TanGrid, fits_wcs
import FITSIO

"""
    FITSIO.FITSHeader(g::TanGrid)

A header holding [`fits_wcs`](@ref)`(g)`, for writing a matrix on `g` (as is) with FITSIO.
"""
function FITSIO.FITSHeader(g::TanGrid)
    cards = fits_wcs(g)
    return FITSIO.FITSHeader(String[c[1] for c in cards], Any[c[2] for c in cards], String[c[3] for c in cards])
end

end # module
