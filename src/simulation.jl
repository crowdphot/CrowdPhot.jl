function _sample_flux(rng::Random.AbstractRNG, flux, distribution, power)
    # A scalar flux means every simulated source has the same brightness.
    if flux isa Number
        return float(flux)
    elseif flux isa Tuple || flux isa AbstractVector
        # A two-value flux range is sampled from the requested distribution.
        length(flux) == 2 || throw(ArgumentError("flux range must have two values"))
        lo, hi = float(flux[1]), float(flux[2])
        lo > 0 && hi ≥ lo || throw(ArgumentError("flux range must be positive and ordered"))
        u = rand(rng)
        if distribution === :uniform
            return lo + u * (hi - lo)
        elseif distribution === :loguniform
            return exp(log(lo) + u * (log(hi) - log(lo)))
        elseif distribution === :powerlaw
            α = float(power)
            if abs(α - 1) < eps(typeof(α)) * 10
                return exp(log(lo) + u * (log(hi) - log(lo)))
            end
            q = 1 - α
            return (lo^q + u * (hi^q - lo^q))^(1 / q)
        else
            throw(ArgumentError("unsupported flux distribution `$distribution`"))
        end
    else
        throw(ArgumentError("`flux` must be a scalar or two-value range"))
    end
end

@doc raw"""
    flux_for_snr(model, snr; background, read_noise=0, gain=1)

Approximate source flux, in the same data units as `background` and
`read_noise` (for example, ADU), required for matched-filter `snr` for a PSF
model on a flat background. `gain` is in electrons per data unit. The estimate
uses [`effective_area(model)`](@ref CrowdPhot.PSF.effective_area), includes source
shot noise, and a background/read-noise variance per pixel of
`background / gain + read_noise^2`. The returned flux is in data units; the
corresponding source flux in electrons is `F * gain`:

```math
\mathrm{snr} \approx \frac{F}{\sqrt{F / g + A_\mathrm{eff}\,\sigma_b^2}},
\quad
\sigma_b^2 = \max\left(0, \frac{\mathrm{background}}{g} + \mathrm{read\_noise}^2\right)
```

Solving the quadratic for the positive flux gives

```math
F \approx \frac{1}{2}\left[
\frac{\mathrm{snr}^2}{g} +
\sqrt{\left(\frac{\mathrm{snr}^2}{g}\right)^2
+ 4\,\mathrm{snr}^2 A_\mathrm{eff}\sigma_b^2}
\right],
\quad
A_\mathrm{eff} = \mathrm{effective\_area}(\mathrm{model})
```
"""
function flux_for_snr(
        model::AbstractPSFModel,
        snr::Real;
        background::Real,
        read_noise::Real = 0,
        gain::Real = 1
    )
    snr ≥ 0 || throw(ArgumentError("`snr` must be non-negative"))
    bv, g = _snr_noise(model, background, read_noise, gain)
    return _flux_for_snr(float(snr), bv, g)
end

# The SNR noise model shared by `flux_for_snr`, `snr_for_flux` and `simulate_sources`.
# `bv` is the background variance over the PSF, `effective_area(model) * σb2`; everything that
# does not depend on the source is in it, so callers with a fixed PSF compute it once.
function _snr_noise(model::AbstractPSFModel, background::Real, read_noise::Real, gain::Real)
    gain > 0 || throw(ArgumentError("`gain` must be positive"))
    background ≥ 0 || throw(ArgumentError("`background` must be non-negative"))
    read_noise ≥ 0 || throw(ArgumentError("`read_noise` must be non-negative"))
    background, read_noise, gain = float.((background, read_noise, gain))
    σb2 = background / gain + read_noise^2
    return effective_area(model) * σb2, gain
end

# Positive root of snr² (F/g + bv) = F², i.e. the flux including source shot noise.
@inline function _flux_for_snr(snr, bv, gain)
    source_noise_term = snr^2 / gain
    return (source_noise_term + sqrt(source_noise_term^2 + 4 * snr^2 * bv)) / 2
end

@doc raw"""
    snr_for_flux(model, flux; background, read_noise=0, gain=1)

Approximate matched-filter signal-to-noise ratio for a source of `flux`, in
the same data units as `background` and `read_noise` (for example, ADU), for
a PSF model on a flat background. `gain` is in electrons per data unit. This
is the inverse of [`flux_for_snr`](@ref) and uses the same noise model:

```math
\mathrm{snr} \approx \frac{F}{\sqrt{F / g + A_\mathrm{eff}\,\sigma_b^2}},
\quad
\sigma_b^2 = \max\left(0, \frac{\mathrm{background}}{g} + \mathrm{read\_noise}^2\right),
\quad
A_\mathrm{eff} = \mathrm{effective\_area}(\mathrm{model})
```

Note that `snr_for_flux(model, 0; background=0, read_noise=0)` returns `NaN`,
since both source shot noise and background/read noise vanish and the ratio
is indeterminate.
"""
function snr_for_flux(
        model::AbstractPSFModel,
        flux::Real;
        background::Real,
        read_noise::Real = 0,
        gain::Real = 1
    )
    flux ≥ 0 || throw(ArgumentError("`flux` must be non-negative"))
    bv, g = _snr_noise(model, background, read_noise, gain)
    flux = float(flux)
    # Include source shot noise in the SNR calculation.
    return flux / sqrt(flux / g + bv)
end

### Source placement utilities with grid-based acceleration for large `min_separation` ###

function _source_grid_cell(y, x, ymin, xmin, inv_cell_size)
    # Anchor grid coordinates to the valid placement region.
    return (floor(Int, (y - ymin) * inv_cell_size), floor(Int, (x - xmin) * inv_cell_size))
end

function _candidate_is_separated(y, x, ys, xs, grid, next_source, minsep2, ymin, xmin, inv_cell_size)
    # Limit exact separation checks to sources in nearby grid cells.
    cy, cx = _source_grid_cell(y, x, ymin, xmin, inv_cell_size)
    for iy in (cy - 1):(cy + 1), ix in (cx - 1):(cx + 1)
        k = get(grid, (iy, ix), 0)
        while k != 0
            if (y - ys[k])^2 + (x - xs[k])^2 < minsep2
                return false
            end
            k = next_source[k]
        end
    end
    return true
end

function _insert_source!(grid, next_source, source_index, y, x, ymin, xmin, inv_cell_size)
    # Link the source into its grid cell without allocating a per-cell vector.
    cell = _source_grid_cell(y, x, ymin, xmin, inv_cell_size)
    next_source[source_index] = get(grid, cell, 0)
    grid[cell] = source_index
    return grid
end

"""
    simulate_sources(shape, n_sources; kwargs...) -> NamedTuple

Generate random source positions and fluxes for an image
with dimensions `shape == (ny, nx)`. Result is a `NamedTuple` with keys
`(id, y, x, flux)` where `id` is a 1-based source index, `y` and `x`
are source positions in pixel coordinates (y = row, x = column), and
`flux` is the source brightness in data units (e.g., ADU). See keyword
arguments below for controlling the source properties and placement.

Source fluxes can be specified in one of two mutually exclusive ways:

**Branch 1 — Flux-based (default):** `flux`, `flux_distribution`, `flux_power`
are used to draw fluxes directly; `psf`, `background`, `read_noise`, and
`gain` are ignored.

- `flux`: scalar or `(lo, hi)` range. Defaults to `(100, 1000)`.
- `flux_distribution`: `:uniform`, `:loguniform`, or `:powerlaw`.
- `flux_power`: power-law exponent for `:powerlaw` (`dN/dF ∝ F^-flux_power`).

**Branch 2 — SNR-based (when `snr` is set):** fluxes are derived from a
requested signal-to-noise ratio via [`flux_for_snr`](@ref), which requires
the PSF model and noise properties. `flux` and `flux_distribution` are
ignored in this branch.

- `snr`: scalar SNR or `(lo, hi)` range. If a scalar, every source is drawn
  at the flux required for that SNR. If a two-value range, each source gets a
  uniform random SNR in `[lo, hi]` before converting to flux.
- `psf`: an [`AbstractPSFModel`](@ref). Required when `snr` is provided in
  order to compute the effective area for flux conversion.
- `background`: scalar background level or a 2-D `AbstractMatrix`
  (image-sized). If a matrix, its median is used as a representative scalar.
  Defaults to `0`.
- `read_noise`: Gaussian read-noise standard deviation in data units (e.g.,
  ADU). Defaults to `0`.
- `gain`: CCD gain in electrons per data unit. Must be positive. Defaults to `1`.

**Common keywords (used in both branches):**

- `min_separation`: minimum source-center separation in pixels.
- `border`: scalar or `(yborder, xborder)` excluded image border.
"""
function simulate_sources(
        shape::Tuple{<:Integer, <:Integer},
        n_sources::Integer;
        flux = (100.0, 1000.0),
        flux_distribution = :uniform,
        flux_power::Real = 2,
        min_separation::Real = 0,
        border = 0,
        rng::Random.AbstractRNG = Random.default_rng(),
        snr = nothing,
        psf::Union{Nothing, AbstractPSFModel} = nothing,
        background::Union{Real, AbstractMatrix} = 0,
        read_noise::Real = 0,
        gain::Real = 1,
        max_attempts::Integer = max(10_000, 200 * n_sources)
    )
    # Define the valid placement region after excluding the image border.
    n_sources ≥ 0 || throw(ArgumentError("`n_sources` must be non-negative"))
    # Treat scalar inputs as identical values on both axes.
    by, bx = if length(border) == 1
        (border, border)
    elseif length(border) == 2
        (border[1], border[2])
    else
        throw(ArgumentError("`border` must be a scalar or length-2 tuple/vector"))
    end
    xmin, xmax = 1 + float(bx), float(shape[2]) - float(bx)
    ymin, ymax = 1 + float(by), float(shape[1]) - float(by)
    xmin ≤ xmax && ymin ≤ ymax || throw(ArgumentError("border leaves no valid source-placement area"))
    minsep = float(min_separation)
    minsep ≥ 0 || throw(ArgumentError("`min_separation` must be non-negative"))
    minsep2 = minsep^2
    use_grid = !iszero(minsep)
    inv_cell_size = inv(minsep)
    # SNR branch: only the SNR draw varies per source, so the PSF's effective area, the
    # representative background (a matrix's median) and the noise model are evaluated once.
    bv = g = snr_lo = snr_span = 0.0
    if !isnothing(snr)
        isnothing(psf) && throw(ArgumentError("`psf` is required when generating fluxes from `snr`"))
        if snr isa Number
            snr_lo = float(snr)
        elseif snr isa Tuple || snr isa AbstractVector
            # Allow source-to-source SNR variation over a uniform range.
            length(snr) == 2 || throw(ArgumentError("`snr` range must have two values"))
            snr_lo, snr_span = float(snr[1]), float(snr[2]) - float(snr[1])
        else
            throw(ArgumentError("`snr` must be a scalar or two-value range"))
        end
        snr_lo ≥ 0 && snr_lo + snr_span ≥ 0 || throw(ArgumentError("`snr` must be non-negative"))
        bv, g = _snr_noise(psf, background isa AbstractMatrix ? median(background) : background, read_noise, gain)
    end
    xs = Vector{Float64}(undef, n_sources)
    ys = Vector{Float64}(undef, n_sources)
    fs = Vector{Float64}(undef, n_sources)
    next_source = use_grid ? Vector{Int}(undef, n_sources) : Int[]
    # Track a grid of source indices at minsep resolution to limit expensive separation checks to nearby sources.
    grid = Dict{Tuple{Int, Int}, Int}()
    use_grid && sizehint!(grid, n_sources)
    attempts = 0
    n_generated = 0
    while n_generated < n_sources && attempts < max_attempts
        # Draw a candidate position and reject it if the local grid neighborhood is occupied too closely.
        attempts += 1
        x = xmin + rand(rng) * (xmax - xmin)
        y = ymin + rand(rng) * (ymax - ymin)
        !use_grid || _candidate_is_separated(y, x, ys, xs, grid, next_source, minsep2, ymin, xmin, inv_cell_size) || continue
        # Draw either flux directly or infer it from a requested SNR.
        f = if isnothing(snr)
            _sample_flux(rng, flux, flux_distribution, flux_power)
        else
            _flux_for_snr(snr isa Number ? snr_lo : snr_lo + rand(rng) * snr_span, bv, g)
        end
        n_generated += 1
        xs[n_generated] = x
        ys[n_generated] = y
        fs[n_generated] = f
        # Log the source in the grid for efficient separation checks of future candidates.
        !use_grid || _insert_source!(grid, next_source, n_generated, y, x, ymin, xmin, inv_cell_size)
    end
    if attempts >= max_attempts
        @warn "Reached maximum attempts ($max_attempts) with only $(n_generated) sources generated; consider reducing `min_separation` or increasing `max_attempts`. Returning the $(n_generated) sources that were generated."
    end
    resize!(xs, n_generated)
    resize!(ys, n_generated)
    resize!(fs, n_generated)
    return (id = 1:n_generated, y = ys, x = xs, flux = fs)
end

function _source_vectors(sources)
    # NamedTuple/table-like inputs expose x, y, and flux directly.
    if hasproperty(sources, :x) && hasproperty(sources, :y) && hasproperty(sources, :flux)
        return sources.y, sources.x, sources.flux
    end
    # Fall back to iterating row-like objects with x/y/flux properties.
    ys = Float64[]
    xs = Float64[]
    fs = Float64[]
    for src in sources
        push!(ys, getproperty(src, :y))
        push!(xs, getproperty(src, :x))
        push!(fs, getproperty(src, :flux))
    end
    return ys, xs, fs
end

function _source_ranges(model, y, x, model_radius, image)
    # Use the model extent unless the caller supplies a fixed render radius.
    if isnothing(model_radius)
        (ylo, yhi), (xlo, xhi) = PSF.extent(model)
    else
        ylo, yhi = y - model_radius, y + model_radius
        xlo, xhi = x - model_radius, x + model_radius
    end
    # Clip the render footprint to the output image in matrix order.
    yr = max(first(axes(image, 1)), floor(Int, ylo)):min(last(axes(image, 1)), ceil(Int, yhi))
    xr = max(first(axes(image, 2)), floor(Int, xlo)):min(last(axes(image, 2)), ceil(Int, xhi))
    return yr, xr
end

"""
    render_sources!(image, model, sources; model_radius=nothing)

Add sources to `image` in place using `model`. `sources` must provide `y`, `x`,
and `flux` fields or columns. The model background is set to zero while each
source is rendered.
"""
function render_sources!(
        image::AbstractMatrix,
        model::AbstractPSFModel,
        sources;
        model_radius = nothing
    )
    # Normalize source containers to coordinate and flux vectors.
    ys, xs, fs = _source_vectors(sources)
    length(ys) == length(xs) == length(fs) || throw(ArgumentError("source x/y/flux lengths must match"))
    for k in eachindex(ys, xs, fs)
        # Render each source with zero model background; image background is added separately.
        m = ConstructionBase.setproperties(model, (x = xs[k], y = ys[k], flux = fs[k], bkg = zero(float(eltype(image)))))
        yr, xr = _source_ranges(m, ys[k], xs[k], model_radius, image)
        if PSF._turbo_safe(m)
            LV.@turbo for j in xr, i in yr
                image[i, j] += evaluate(m, i, j)
            end
        else
            @inbounds for j in xr
                @simd for i in yr
                    image[i, j] += evaluate(m, i, j)
                end
            end
        end
    end
    return image
end

function rand_poisson(rng::Random.AbstractRNG, λ::Real)
    # Use exact inversion for small means and Gaussian approximation for large means.
    function _rand_poisson_knuth(rng::Random.AbstractRNG, λ::T) where T
        # Draw before the first comparison: for λ below ~1e-16, `exp(-λ)` rounds to
        # 1 and the draw-after form returned -1.
        L = exp(-λ)
        k = 0
        p = rand(rng, T)
        while p > L
            k += 1
            p *= rand(rng, T)
        end
        return k
    end

    function _rand_poisson_normal(rng::Random.AbstractRNG, λ::T) where T
        σ = sqrt(λ)

        while true
            x = round(Int, λ + σ * randn(rng, T))
            x >= 0 && return x
        end
    end

    λ < 0 && throw(DomainError(λ, "Poisson rate λ must be nonnegative"))
    λ == 0 && return 0

    if λ < 30
        return _rand_poisson_knuth(rng, float(λ))
    else
        return _rand_poisson_normal(rng, float(λ))
    end
end
rand_poisson(λ::Real) = rand_poisson(Random.default_rng(), λ)


"""
    add_noise!(image; noise=:poisson_gaussian, read_noise=0, gain=1, rng=Random.default_rng())

Apply in-place noise to an expected-count image. `:poisson` samples shot noise,
`:gaussian` adds read noise, and `:poisson_gaussian` applies both.
"""
function add_noise!(
        image::AbstractMatrix;
        noise = :poisson_gaussian,
        read_noise::Real = 0,
        gain::Real = 1,
        rng::Random.AbstractRNG = Random.default_rng()
    )
    # Dispatch the requested noise model explicitly.
    gain > 0 || throw(ArgumentError("`gain` must be positive"))
    if isnothing(noise) || noise === :none
        return image
    elseif noise === :poisson
        for idx in eachindex(image)
            image[idx] = rand_poisson(rng, max(zero(float(image[idx])), float(image[idx]) * gain)) / gain
        end
    elseif noise === :gaussian
        for idx in eachindex(image)
            image[idx] += read_noise * randn(rng)
        end
    elseif noise === :poisson_gaussian
        add_noise!(image; noise = :poisson, gain, rng)
        add_noise!(image; noise = :gaussian, read_noise, rng)
    else
        throw(ArgumentError("unsupported noise model `$noise`"))
    end
    return image
end

"""
    simulate_image(shape, psf, sources; background=0, noise=:poisson_gaussian, kwargs...)

Create an artificial image by rendering `sources` through `psf`, adding scalar or
matrix `background`, and optionally injecting Poisson/read noise.
"""
function simulate_image(
        shape::Tuple{<:Integer, <:Integer},
        psf::AbstractPSFModel,
        sources;
        background::Union{Real, AbstractMatrix} = 0,
        noise = :poisson_gaussian,
        read_noise::Real = 0,
        gain::Real = 1,
        model_radius = nothing,
        rng::Random.AbstractRNG = Random.default_rng()
    )
    # Allocate the image from either scalar or matrix background.
    T = promote_type(float(eltype(ConstructionBase.getproperties(psf))), background isa AbstractMatrix ? eltype(background) : typeof(background))
    T = T <: Integer ? Float64 : float(T)
    image = if background isa AbstractMatrix
        size(background) == shape || throw(ArgumentError("background matrix must match `shape`"))
        Matrix{T}(background)
    else
        fill(T(background), shape)
    end
    # Add source model values and then apply the requested noise model.
    render_sources!(image, psf, sources; model_radius)
    add_noise!(image; noise, read_noise, gain, rng)
    return image
end

"""
    simulate_image(shape, psf, n_sources; kwargs...) -> (image, sources)

Generate random sources with `simulate_sources`, render them through `psf`, and
return both the image and source table.
"""
function simulate_image(
        shape::Tuple{<:Integer, <:Integer},
        psf::AbstractPSFModel,
        n_sources::Integer;
        background::Union{Real, AbstractMatrix} = 0,
        noise = :poisson_gaussian,
        read_noise::Real = 0,
        gain::Real = 1,
        model_radius = nothing,
        rng::Random.AbstractRNG = Random.default_rng(),
        source_kwargs...
    )
    # Generate a source catalog with the same noise/background assumptions.
    sources = simulate_sources(
        shape, n_sources;
        rng, psf, background, read_noise, gain, source_kwargs...
    )
    # Render the generated catalog into an image.
    image = simulate_image(
        shape, psf, sources;
        background, noise, read_noise, gain, model_radius, rng
    )
    return image, sources
end

"""
    make_gaussians_image(n_stars, image_size=(256, 256);
        rng            = Random.default_rng(),
        background     = 200.0,
        read_noise     = 5.0,
        gain           = 1.5,
        sigma_psf      = 2.0,
        flux_range     = (500.0, 50_000.0),
        border         = 10,
        noise          = :poisson_gaussian) -> Matrix{Float64}

Generate a realistic mock CCD image containing `n_stars` point sources
with Gaussian profiles on top of a background.

`background` may be a scalar (for a flat sky) or a matrix the same size
as the requested output (for a spatially varying background).  In both
cases the values are interpreted as counts (ADU).

Stars are placed uniformly at random (with a `border`-pixel exclusion zone),
with fluxes drawn log-uniformly from `flux_range`.  The pixel model is:

```
expected[i,j] = background[i,j] + Σ_k  flux_k / (2π σ²) · exp(−r²_k / 2σ²)
```

where `r²_k = (i − y_k)² + (j − x_k)²` and `σ = sigma_psf`.

Noise is applied as: shot noise on `expected * gain` electrons, then a
Gaussian read-noise draw with standard deviation `read_noise` (in ADU).
The returned image is in ADU (counts).  Pass `noise = :none` to return
the noiseless expected image.

The same `rng` will always produce the same image, making this function
suitable for deterministic doctests and benchmarks.

# Examples

```jldoctest
julia> using CrowdPhot, StableRNGs

julia> rng = StableRNG(42);

julia> img = make_gaussians_image(30, (128, 128); rng);

julia> size(img)
(128, 128)

julia> img[1, 1] > 0
true
```
"""
function make_gaussians_image(
        n_stars::Integer,
        image_size::Tuple{<:Integer, <:Integer} = (256, 256);
        rng::Random.AbstractRNG = Random.default_rng(),
        background::Union{Real, AbstractMatrix} = 200.0,
        read_noise::Real = 5.0,
        gain::Real = 1.5,
        sigma_psf::Real = 2.0,
        flux_range = (500.0, 50_000.0),
        border::Integer = 10,
        noise = :poisson_gaussian,
    )
    n_stars ≥ 0 || throw(ArgumentError("`n_stars` must be non-negative"))
    gain > 0     || throw(ArgumentError("`gain` must be positive"))
    sigma_psf > 0 || throw(ArgumentError("`sigma_psf` must be positive"))

    H, W = image_size
    img  = if background isa AbstractMatrix
        size(background) == (H, W) || throw(ArgumentError(
            "`background` matrix must match `image_size`; got $(size(background))"))
        float.(background)
    else
        fill(float(background), H, W)
    end

    # Draw source positions and fluxes.
    xmin, xmax = 1 + border, W - border
    ymin, ymax = 1 + border, H - border
    lo, hi = float(flux_range[1]), float(flux_range[2])
    lo > 0 && hi ≥ lo || throw(ArgumentError("`flux_range` must be positive and ordered"))

    # Render each Gaussian source by scanning a ±4σ footprint.
    half = ceil(Int, 4 * sigma_psf)
    norm = 1.0 / (2π * sigma_psf^2)
    inv2σ2 = 1.0 / (2 * sigma_psf^2)

    for _ in 1:n_stars
        x0  = xmin + rand(rng) * (xmax - xmin)
        y0  = ymin + rand(rng) * (ymax - ymin)
        # Log-uniform flux.
        f   = exp(log(lo) + rand(rng) * (log(hi) - log(lo)))
        fval = f * norm
        ilo, ihi = max(1, floor(Int, y0) - half), min(H, ceil(Int, y0) + half)
        jlo, jhi = max(1, floor(Int, x0) - half), min(W, ceil(Int, x0) + half)
        LV.@turbo for j in jlo:jhi, i in ilo:ihi
            img[i, j] += fval * exp(-((j - x0)^2 + (i - y0)^2) * inv2σ2)
        end
    end

    # Apply Poisson shot noise then Gaussian read noise.
    add_noise!(img; noise, read_noise, gain, rng)
    return img
end
