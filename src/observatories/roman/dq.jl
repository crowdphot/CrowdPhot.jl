# --- Data Quality bit flag definitions -------------------------------------
# Bit position => (value, name).
#
# Owned upstream by `roman_datamodels`
# (https://github.com/spacetelescope/roman_datamodels), whose `pixel` enum
# this mirrors.  Bits 14, 26, 27 and 29 are reserved there and have no name.
"""
    DQ_FLAGS::Dict{Symbol, UInt32}

Roman pixel-level data quality flags, mirroring the `pixel` enum in
`roman_datamodels`.  Each entry maps a flag name to its bit value.  Used by
[`dq_mask_value`](@ref) and [`parse_dq_mask`](@ref).
"""
const DQ_FLAGS = Dict(
    :DO_NOT_USE        => UInt32(1)          , # bit 0
    :SATURATED         => UInt32(2)          , # bit 1
    :JUMP_DET          => UInt32(4)          , # bit 2
    :DROPOUT           => UInt32(8)          , # bit 3
    :GW_AFFECTED_DATA  => UInt32(16)         , # bit 4
    :PERSISTENCE       => UInt32(32)         , # bit 5
    :AD_FLOOR          => UInt32(64)         , # bit 6
    :OUTLIER           => UInt32(128)        , # bit 7 (PIXELDQ meaning)
    :UNRELIABLE_ERROR  => UInt32(256)        , # bit 8
    :NON_SCIENCE       => UInt32(512)        , # bit 9
    :DEAD              => UInt32(1024)       , # bit 10
    :HOT               => UInt32(2048)       , # bit 11
    :WARM              => UInt32(4096)       , # bit 12
    :LOW_QE            => UInt32(8192)       , # bit 13
    :TELEGRAPH         => UInt32(32768)      , # bit 15
    :NONLINEAR         => UInt32(65536)      , # bit 16
    :BAD_REF_PIXEL     => UInt32(131072)     , # bit 17
    :NO_FLAT_FIELD     => UInt32(262144)     , # bit 18
    :NO_GAIN_VALUE     => UInt32(524288)     , # bit 19
    :NO_LIN_CORR       => UInt32(1048576)    , # bit 20
    :NO_SAT_CHECK      => UInt32(2097152)    , # bit 21
    :UNRELIABLE_BIAS   => UInt32(4194304)    , # bit 22
    :UNRELIABLE_DARK   => UInt32(8388608)    , # bit 23
    :UNRELIABLE_SLOPE  => UInt32(16777216)   , # bit 24
    :UNRELIABLE_FLAT   => UInt32(33554432)   , # bit 25
    :UNRELIABLE_RESET  => UInt32(268435456)  , # bit 28
    :OTHER_BAD_PIXEL   => UInt32(1073741824) , # bit 30
    :REFERENCE_PIXEL   => UInt32(0x80000000) , # bit 31
)

"""
    dq_mask_value(flags = (:DO_NOT_USE,)) → UInt32

Combine the named flags (see [`DQ_FLAGS`](@ref)) into a single bitmask via
bitwise OR.  Throws an `ArgumentError` naming the valid flags if `flags`
contains a name that is not in the table.

# Examples
```jldoctest
julia> using CrowdPhot.Roman: dq_mask_value

julia> dq_mask_value((:DO_NOT_USE, :SATURATED))
0x00000003
```
"""
function dq_mask_value(flags = (:DO_NOT_USE,))
    for f in flags
        # A typo here silently masks nothing, so it is worth naming the
        # alternatives rather than letting `Dict` throw a bare `KeyError`.
        haskey(DQ_FLAGS, f) || throw(ArgumentError(
            "unknown Roman DQ flag $(repr(f)); valid flags are " *
            join(sort!(collect(keys(DQ_FLAGS))), ", ")))
    end
    reduce(|, (DQ_FLAGS[f] for f in flags); init = UInt32(0))
end

"""
    parse_dq_mask(mask::AbstractMatrix{UInt32}; flags = (:DO_NOT_USE,)) -> BitMatrix

Read a DQ array and return a boolean array indicating which pixels have *any*
of the specified flags set.  Pixels are `true` where these flags are present
and `false` where they are not.  The default flag is `:DO_NOT_USE`.

`true` means "bad", matching the `bkg_mask` convention of
[`CrowdPhot.fit_all_stars_multipass`](@ref), so the result can be passed
straight through.

# Examples
```jldoctest
julia> using CrowdPhot.Roman: parse_dq_mask

julia> parse_dq_mask(UInt32[0 1; 2 3])
2×2 BitMatrix:
 0  1
 0  1
```
"""
function parse_dq_mask(mask::AbstractMatrix{UInt32};
                       flags = (:DO_NOT_USE,))
    bitmask = dq_mask_value(flags)
    return (mask .& bitmask) .!= 0
end
