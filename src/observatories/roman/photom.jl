"""
    jansky_per_flux_unit(meta) -> Float64

The factor converting a flux fitted on a Roman WFI Level 2 exposure into janskys,
read from that exposure's own `meta.photometry` block:

    conversion_megajanskys * 1e6 * pixel_area

`conversion_megajanskys` is in MJy/sr and `pixel_area` is in steradians, so the
product converts a flux in L2 data units to Jy.  Pair it with
[`CrowdPhot.abmag`](@ref) to get AB magnitudes.

!!! warning
    This assumes the pixel area map has already been multiplied into the image,
    as [`load_area`](@ref) supplies and the `examples/roman` scripts do.  Roman L2
    data are in surface brightness units, so a flux fitted to an uncorrected frame
    is not on the scale this factor assumes, and the magnitudes will be wrong by a
    position-dependent amount.

# Examples
```jldoctest
julia> using CrowdPhot: Roman

julia> meta = Dict("photometry" => Dict("conversion_megajanskys" => 2.0,
                                        "pixel_area" => 5.0e-13));

julia> Roman.jansky_per_flux_unit(meta)
1.0e-6
```
"""
function jansky_per_flux_unit(meta)
    haskey(meta, "photometry") || throw(ArgumentError(
        "`meta` has no \"photometry\" key; is this the `meta` from a Roman L2 file?"))
    p = meta["photometry"]
    for k in ("conversion_megajanskys", "pixel_area")
        haskey(p, k) || throw(ArgumentError(
            "`meta[\"photometry\"]` has no \"$k\" key; cannot calibrate fluxes"))
    end
    return Float64(p["conversion_megajanskys"]) * 1.0e6 * Float64(p["pixel_area"])
end
