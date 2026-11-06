# The Roman skymap: the sky tessellation the archive's L3 mosaics are stored on (Roman data handbook,
# https://roman-docs.stsci.edu/data-handbook/wfi-data-levels-and-products/skymap-tessellation; romancal
# `romancal/skycell/skymap.py`).  It is the CRDS SKYCELLS reference `roman_wfi_skycells_0001.asdf` (1.3 GB): a table
# of 4058 projection regions and one of 8,045,034 skycells.  The region table is bundled
# (`data/roman_wfi_skycells_0001_projection_regions.bin`, the bytes of the file's block 0) and every skycell is
# rebuilt from it by the rules below; test/observatories/skycells_tests.jl checks every cell against the file.

"Arcsec per skycell pixel, skycell side and border (px): the SKYCELLS reference's `meta.pixel_scale`, `nxy_skycell`, `skycell_border_pixels`."
const SKYCELL_SCALE = 0.055
const SKYCELL_NXY = 5000
const SKYCELL_BORDER = 100
"Skycell spacing (px): neighbors overlap by twice the border, so each one's core is `SKYCELL_STEP` px square."
const SKYCELL_STEP = SKYCELL_NXY - 2 * SKYCELL_BORDER

"""
    ProjectionRegion

One row of the Roman skymap's projection-region table: a gnomonic projection about
`(ra_tangent, dec_tangent)` (deg) onto an `nx x ny` grid of `SKYCELL_SCALE` arcsec pixels, with the tangent
point at 0-based pixel `(x_tangent, y_tangent)`, `+x` east and `+y` north (`orientat` is 0 for every region).
A point belongs to the region whose bounds hold it: RA in `[ra_min, ra_max)` (wrapping through 0) and Dec in
`[dec_min, dec_max)`, the north cap including the pole.  `index` and the skycell range
`skycell_start:skycell_end - 1` are 0-based, as in the reference file.
"""
struct ProjectionRegion
    index::Int32
    ra_tangent::Float64
    dec_tangent::Float64
    ra_min::Float64
    ra_max::Float64
    dec_min::Float64
    dec_max::Float64
    orientat::Float32
    x_tangent::Float64
    y_tangent::Float64
    nx::Int32
    ny::Int32
    skycell_start::Int32
    skycell_end::Int32
end

const _REGION_TABLE = joinpath(dirname(dirname(dirname(@__DIR__))), "data", "roman_wfi_skycells_0001_projection_regions.bin")
const _REGION_RECORD = 88   # bytes per row: the reference file's packed little-endian layout

# The table's rows, from the bytes of the reference file's block 0.
function _read_regions(bytes::AbstractVector{UInt8})
    length(bytes) % _REGION_RECORD == 0 || throw(ArgumentError("projection-region table of $(length(bytes)) bytes is not whole $(_REGION_RECORD)-byte rows"))
    field(T, o) = ltoh(reinterpret(T, bytes[(o + 1):(o + sizeof(T))])[1])
    return map(0:(length(bytes) ÷ _REGION_RECORD - 1)) do k
        o = k * _REGION_RECORD
        ProjectionRegion(field(Int32, o), field(Float64, o + 4), field(Float64, o + 12), field(Float64, o + 20), field(Float64, o + 28),
                         field(Float64, o + 36), field(Float64, o + 44), field(Float32, o + 52), field(Float64, o + 56), field(Float64, o + 64),
                         field(Int32, o + 72), field(Int32, o + 76), field(Int32, o + 80), field(Int32, o + 84))
    end
end

include_dependency(_REGION_TABLE)
const PROJECTION_REGIONS = _read_regions(read(_REGION_TABLE))

"The Roman skymap's 4058 projection regions ([`ProjectionRegion`](@ref)), in the reference file's order (`index` 0 first)."
projection_regions() = PROJECTION_REGIONS

_is_polar(r::ProjectionRegion) = r.dec_max == 90 || r.dec_min == -90

# romancal's `_ra_in_range` and `ProjectionRegion.contains_radec`.
function _in_region(r::ProjectionRegion, ra, dec)
    a = mod(ra, 360); lo = mod(r.ra_min, 360); hi = mod(r.ra_max, 360)
    inra = lo == hi || (lo < hi ? lo <= a < hi : (a >= lo) | (a < hi))
    return inra & (dec >= r.dec_min) & ((dec < r.dec_max) | (r.dec_max == 90))
end

"""
    projection_region(ra, dec) -> ProjectionRegion

The projection region whose bounds hold `(ra, dec)` (deg); every point is in exactly one.
"""
function projection_region(ra, dec)
    i = findfirst(r -> _in_region(r, ra, dec), PROJECTION_REGIONS)
    i === nothing && throw(ArgumentError("no projection region holds ($ra, $dec)"))
    return PROJECTION_REGIONS[i]
end

"""
    region_to_world(r::ProjectionRegion, x, y) -> (ra, dec)

`(ra, dec)` (deg, RA in `[0, 360)`) of 0-based pixel `(x, y)` of region `r`'s grid: romancal's skycell WCS (TAN about
the tangent point, `CDELT = SKYCELL_SCALE` on both axes, identity PC), so `+x` is east and `+y` north.  This is
CrowdPhot's `TanGrid` mirrored east-west: row `y` and column `-x` of a `TanGrid` with the tangent point at
`(y_tangent, -x_tangent)`.
"""
region_to_world(r::ProjectionRegion, x, y) = grid_to_world(TanGrid(r.ra_tangent, r.dec_tangent, SKYCELL_SCALE, r.ny, r.nx, r.y_tangent, -r.x_tangent), y, -x)

# Skycell name: the tangent point's rounded RA and Dec, then the cell's lattice position from the center cell
# (`x` along the region's y, `y` along its x, both offset by 50).
function _skycell_name(r::ProjectionRegion, i, j)
    ra = mod(Int(round(r.ra_tangent)), 360); dec = Int(round(r.dec_tangent))
    return string(lpad(ra, 3, '0'), dec >= 0 ? 'p' : 'm', lpad(abs(dec), 2, '0'), 'x', lpad(50 + j, 2, '0'), 'y', lpad(50 + i, 2, '0'))
end

# Region pixel of skycell (i, j)'s pixel 0: cell (0, 0) is centered on the tangent point.
_cell_origin(r::ProjectionRegion, i, j) = (r.x_tangent - (SKYCELL_NXY - 1) / 2 + SKYCELL_STEP * i, r.y_tangent - (SKYCELL_NXY - 1) / 2 + SKYCELL_STEP * j)

# Whether skycell (i, j) exists: a corner of its core lies inside the region's bounds.
function _has_cell(r::ProjectionRegion, i, j)
    x0, y0 = _cell_origin(r, i, j)
    lo = SKYCELL_BORDER - 0.5; hi = SKYCELL_NXY - SKYCELL_BORDER - 0.5
    return any(((a, b),) -> _in_region(r, region_to_world(r, x0 + a, y0 + b)...), ((lo, lo), (hi, lo), (hi, hi), (lo, hi)))
end

"""
    skycells(r::ProjectionRegion) -> StructArray

Every skycell of region `r`, in the reference file's order and with its columns, rebuilt from the region alone:
- the cells lie on a lattice of `SKYCELL_STEP` px with cell `(i, j) = (0, 0)` centered on the tangent point (`i`
  along the region's x, `j` along its y), each `SKYCELL_NXY` px square, so neighbors overlap by `2 SKYCELL_BORDER`;
- a cell exists if a corner of its core (cell pixels `SKYCELL_BORDER` to `SKYCELL_NXY - SKYCELL_BORDER - 1`) lies in
  the region's bounds;
- cells are ordered by `j`, then `i`.

Columns: `name` (e.g. `270m27x50y29`), `i`, `j`, `ra_center`, `dec_center` (cell pixel `(2499.5, 2499.5)`), `orientat`,
`x_tangent`, `y_tangent` (the tangent point in the cell's 0-based pixels), and the corners `ra_corn1, dec_corn1`, ...,
`ra_corn4, dec_corn4` at cell pixels `(-0.5, -0.5)`, `(4999.5, -0.5)`, `(4999.5, 4999.5)`, `(-0.5, 4999.5)`.
`orientat` is the reference's: `ra_tangent - ra_center` (deg, `Float32`), wrapped to `(-180, 180]` except in the
polar caps.  It is the cell center's RA offset from the tangent meridian, not the angle of local north.
"""
function skycells(r::ProjectionRegion)
    n = cld(max(r.nx, r.ny), 2 * SKYCELL_STEP) + 2
    ij = [(i, j) for j in -n:n for i in -n:n if _has_cell(r, i, j)]
    any(((i, j),) -> max(abs(i), abs(j)) == n, ij) && error("region $(r.index): a skycell on the edge of the $(2n + 1)-cell search box")
    c = (SKYCELL_NXY - 1) / 2
    rows = map(ij) do (i, j)
        x0, y0 = _cell_origin(r, i, j)
        ra, dec = region_to_world(r, x0 + c, y0 + c)
        orientat = Float32(_is_polar(r) ? r.ra_tangent - ra : rem(r.ra_tangent - ra, 360, RoundNearest))
        k = map(((a, b),) -> region_to_world(r, x0 + a, y0 + b), ((-0.5, -0.5), (SKYCELL_NXY - 0.5, -0.5), (SKYCELL_NXY - 0.5, SKYCELL_NXY - 0.5), (-0.5, SKYCELL_NXY - 0.5)))
        (; name = _skycell_name(r, i, j), i, j, ra_center = ra, dec_center = dec, orientat, x_tangent = c - SKYCELL_STEP * i, y_tangent = c - SKYCELL_STEP * j,
         ra_corn1 = k[1][1], dec_corn1 = k[1][2], ra_corn2 = k[2][1], dec_corn2 = k[2][2], ra_corn3 = k[3][1], dec_corn3 = k[3][2],
         ra_corn4 = k[4][1], dec_corn4 = k[4][2])
    end
    return StructArray(rows)
end
