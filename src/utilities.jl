# The mad, mad!, mean_and_std functions are reproduced from StatsBase.jl and are licensed under the MIT License.

Base.@irrational mad_constant 1.4826022185056018 big"""
1.482602218505601860547076529360423431326703202590312896536266275245674447622701
"""

"""
    mad(x; center=median(x), normalize=true)

Compute the median absolute deviation (MAD) of collection `x` around `center`
(by default, around the median).

If `normalize` is set to `true`, the MAD is multiplied by
`1 / quantile(Normal(), 3/4) ≈ 1.4826`, in order to obtain a consistent estimator
of the standard deviation under the assumption that the data is normally distributed.
"""
function mad(x; center=nothing, normalize::Union{Bool, Nothing}=nothing, constant=nothing)
    mad!(Base.copymutable(x); center=center, normalize=normalize, constant=constant)
end
mad(data, ::Colon; kwargs...) = mad(data; kwargs...)
mad(data, dims; kwargs...) = mapslices(x -> mad(x; kwargs...), data; dims)

"""
    StatsBase.mad!(x; center=median!(x), normalize=true)

Compute the median absolute deviation (MAD) of array `x` around `center`
(by default, around the median), overwriting `x` in the process.

If `normalize` is set to `true`, the MAD is multiplied by
`1 / quantile(Normal(), 3/4) ≈ 1.4826`, in order to obtain a consistent estimator
of the standard deviation under the assumption that the data is normally distributed.
"""
function mad!(x::AbstractArray;
              center=median!(x),
              normalize::Union{Bool,Nothing}=true,
              constant=nothing)
    isempty(x) && throw(ArgumentError("mad is not defined for empty arrays"))
    c = center === nothing ? median!(x) : center
    T = promote_type(typeof(c), eltype(x))
    U = eltype(x)
    x2 = U == T ? x : isconcretetype(U) && isconcretetype(T) && sizeof(U) == sizeof(T) ? reinterpret(T, x) : similar(x, T)
    x2 .= abs.(x .- c)
    m = median!(x2)
    if normalize isa Nothing
        Base.depwarn("the `normalize` keyword argument will be false by default in future releases: set it explicitly to silence this deprecation", :mad)
        normalize = true
    end
    if !isa(constant, Nothing)
        Base.depwarn("keyword argument `constant` is deprecated, use `normalize` instead or apply the multiplication directly", :mad)
        m * constant
    elseif normalize
        m * mad_constant
    else
        m
    end
end

# """
#     mean_and_std(x, [dim]; corrected=true) -> (mean, std)

# Return the mean and standard deviation of collection `x`. If `x` is an `AbstractArray`,
# `dim` can be specified as a tuple to compute statistics over these dimensions.
# A weighting vector `w` can be specified to weight the estimates.
# Finally, bias correction is applied to the
# standard deviation calculation if `corrected=true`.
# See [`std`](@ref) documentation for more details.
# """
# function mean_and_std(x; corrected::Bool=true)
#     m = mean(x)
#     s = std(x, mean=m, corrected=corrected)
#     m, s
# end
# function mean_and_std(x::AbstractArray{<:Real}, dims; corrected::Bool=true)
#     m = mean(x, dims=dims)
#     s = std(x, dims=dims, mean=m, corrected=corrected)
#     m, s
# end
# # mean_and_std(x::AbstractArray{<:Real}, ::Colon; corrected::Bool=true) =
# #     mean_and_std(x; corrected=corrected)