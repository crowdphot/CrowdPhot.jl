module CrowdPhotFITSIOExt

using CrowdPhot.Joint: TanGrid, fits_wcs
import FITSIO

"""
    FITSIO.FITSHeader(g::TanGrid)

A header holding [`fits_wcs`](@ref)`(g)`, for a matrix `A` on `g` written as `permutedims(A)`;
`write(f, A, g)` writes both.
"""
function FITSIO.FITSHeader(g::TanGrid)
    cards = fits_wcs(g)
    return FITSIO.FITSHeader(String[c[1] for c in cards], Any[c[2] for c in cards], String[c[3] for c in cards])
end

"""
    write(f::FITSIO.FITS, A::AbstractMatrix{<:Real}, g::TanGrid; name, ver)

Write `A`, a matrix on `g` in CrowdPhot's `(row, col)` layout, as an image HDU in the usual FITS
layout (axis 1 the column, north up and east left in a viewer) with `g`'s WCS:
`permutedims(A)` with header `FITSHeader(g)`.  `permutedims(read(f[name]))` reads it back.  The
size check cannot catch an already transposed `A` on a square grid.
"""
function Base.write(f::FITSIO.FITS, A::AbstractMatrix{<:Real}, g::TanGrid; kwargs...)
    size(A) == (g.ny, g.nx) || throw(DimensionMismatch("a $(size(A, 1)) x $(size(A, 2)) matrix is not on a $(g.ny) x $(g.nx) TanGrid"))
    return write(f, permutedims(A); header = FITSIO.FITSHeader(g), kwargs...)
end

end # module
