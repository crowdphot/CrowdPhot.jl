"""
    TanGrid(ra0, dec0, scale, ny, nx[, cy, cx])

A gnomonic (TAN) projection about the tangent point `(ra0, dec0)` (degrees) with a uniform
pixel grid of `scale` arcsec per pixel, north up and east left; `(row, col)` are 1-based and
`(cy, cx)` is the grid position of the tangent point (default: the grid center).

Joint-fit source positions live on one of these (the *fit frame*); the joint detector
samples a finer one with the same tangent point.  A Roman skycell is a `TanGrid` with fixed
parameters.
"""
struct TanGrid
    ra0::Float64
    dec0::Float64
    scale::Float64
    ny::Int
    nx::Int
    cy::Float64
    cx::Float64
end
TanGrid(ra0, dec0, scale, ny, nx) = TanGrid(ra0, dec0, scale, ny, nx, (ny + 1) / 2, (nx + 1) / 2)

const ARCSEC_PER_RAD = 180 * 3600 / π

"`(row, col)` of `(ra, dec)` (degrees) on grid `g`."
function world_to_grid(g::TanGrid, ra, dec)
    α = deg2rad(ra - g.ra0); δ = deg2rad(dec); δ0 = deg2rad(g.dec0)
    sδ, cδ = sincos(δ); sδ0, cδ0 = sincos(δ0); sα, cα = sincos(α)
    cosc = sδ0 * sδ + cδ0 * cδ * cα
    # TAN diverges 90 deg from the tangent point and mirrors beyond it
    cosc > 0 || throw(ArgumentError("($ra, $dec) is at least 90 deg from the TanGrid tangent point ($(g.ra0), $(g.dec0)); the projection is undefined there"))
    ξ = cδ * sα / cosc
    η = (cδ0 * sδ - sδ0 * cδ * cα) / cosc
    return (g.cy + η * ARCSEC_PER_RAD / g.scale, g.cx - ξ * ARCSEC_PER_RAD / g.scale)
end

"`(ra, dec)` (degrees) of grid position `(row, col)`."
function grid_to_world(g::TanGrid, row, col)
    η = (row - g.cy) * g.scale / ARCSEC_PER_RAD; ξ = -(col - g.cx) * g.scale / ARCSEC_PER_RAD
    ρ = hypot(ξ, η); δ0 = deg2rad(g.dec0); sδ0, cδ0 = sincos(δ0)
    ρ == 0 && return (g.ra0, g.dec0)
    c = atan(ρ); sc, cc = sincos(c)
    δ = asin(cc * sδ0 + η * sc * cδ0 / ρ)
    α = atan(ξ * sc, ρ * cδ0 * cc - η * sδ0 * sc)
    return (mod(g.ra0 + rad2deg(α), 360.0), rad2deg(δ))
end

"""
    oversample(g::TanGrid, K) -> TanGrid

The grid `K` times finer than `g` with the same tangent point and sky coverage: position
`(u, v)` on `g` is `(K (u - g.cy) + cy', K (v - g.cx) + cx')` on the result.
"""
oversample(g::TanGrid, K::Integer) =
    TanGrid(g.ra0, g.dec0, g.scale / K, K * g.ny, K * g.nx, K * (g.cy - 0.5) + 0.5, K * (g.cx - 0.5) + 0.5)

"""
    fits_wcs(g::TanGrid) -> Vector{Tuple{String, Any, String}}

FITS WCS keywords `(keyword, value, comment)` for a matrix on `g` written as is: Julia's first
index is FITS axis 1, so axis 1 runs north and axis 2 west, and the `CD` matrix has zero
diagonal.  Pass them to a FITS writer as the header (with FITSIO loaded, `FITSHeader(g)` builds
one).  The projection is exactly `g`'s gnomonic one; the celestial frame is whatever `to_sky`
returned (FITS assumes ICRS when `RADESYS` is absent).
"""
function fits_wcs(g::TanGrid)
    s = g.scale / 3600
    return Tuple{String, Any, String}[
        ("WCSAXES", 2, "number of WCS axes"),
        ("CTYPE1", "RA---TAN", "gnomonic projection"),
        ("CTYPE2", "DEC--TAN", "gnomonic projection"),
        ("CRVAL1", g.ra0, "[deg] RA of the tangent point"),
        ("CRVAL2", g.dec0, "[deg] Dec of the tangent point"),
        ("CRPIX1", g.cy, "axis-1 (row) pixel of the tangent point"),
        ("CRPIX2", g.cx, "axis-2 (column) pixel of the tangent point"),
        ("CD1_1", 0.0, "[deg/px] RA-like coordinate along axis 1"),
        ("CD1_2", -s, "[deg/px] RA-like coordinate along axis 2 (east is -col)"),
        ("CD2_1", s, "[deg/px] Dec-like coordinate along axis 1 (north is +row)"),
        ("CD2_2", 0.0, "[deg/px] Dec-like coordinate along axis 2"),
        ("CUNIT1", "deg", "units of CRVAL1 and CD1_j"),
        ("CUNIT2", "deg", "units of CRVAL2 and CD2_j")]
end
