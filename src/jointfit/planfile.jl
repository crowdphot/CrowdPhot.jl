# Plan folders: a `plan_tiles` plan as TOML files that separate processes read, so they agree on it.
# `plan.toml` is the index (no cutouts), `tiles/shard_<k>.toml` hold the tiles with their cutouts, and
# `jobs.toml` groups tiles into jobs.  Design: run-planning-design.tex, "The plan as a file".

"The plan folder's format version: readers refuse any other."
const PLAN_VERSION = 1

# ---- options as TOML values ----

# A value as TOML: `nothing` as "nothing" (TOML has no null), Symbols as strings, tuples as arrays, option
# structs and NamedTuples as tables, background estimators by name (a table with `type` when they have fields).
_toml(::Nothing) = "nothing"
_toml(x::Symbol) = string(x)
_toml(x::Union{Bool, Integer, AbstractFloat, AbstractString}) = x
_toml(x::Union{Tuple, AbstractVector}) = Any[_toml(y) for y in x]
_toml(x::AbstractDict) = Dict{String, Any}(string(k) => _toml(v) for (k, v) in x)
_toml(x::NamedTuple) = Dict{String, Any}(string(k) => _toml(v) for (k, v) in pairs(x))
_toml(x::Union{JointLoopOptions, JointDetector, JointFitOptions}) = Dict{String, Any}(string(f) => _toml(getfield(x, f)) for f in fieldnames(typeof(x)))
function _toml(x::Union{CrowdPhot.Background.AbstractBackgroundEstimator, CrowdPhot.Background.AbstractBackgroundRMSEstimator})
    E = typeof(x)
    parentmodule(E) === CrowdPhot.Background || throw(ArgumentError("a plan can name only CrowdPhot.Background's estimators, not $E"))
    isempty(fieldnames(E)) && return string(nameof(E))
    return merge(Dict{String, Any}("type" => string(nameof(E))), Dict{String, Any}(string(f) => _toml(getfield(x, f)) for f in fieldnames(E)))
end
_toml(x) = throw(ArgumentError("a plan file can't hold a $(typeof(x))"))

# The one reader.  A schema is a NamedTuple of declared types (`_COADD_OPTIONS`, `_BACKGROUND_OPTIONS`) or a
# struct's fields; every value is read by its declared type, from its TOML value or from an in-memory value
# (as `plan_tiles` normalizes options), so a plan in memory and the same plan read from its folder can't differ.
_schema(::Type{T}) where {T} = NamedTuple{fieldnames(T)}(fieldtypes(T))
# `background` is declared only as a NamedTuple
_schema(::Type{JointLoopOptions}) = merge(NamedTuple{fieldnames(JointLoopOptions)}(fieldtypes(JointLoopOptions)), (; background = _BACKGROUND_OPTIONS))

"""
The options struct `T` from its TOML table `d` (or a NamedTuple), every field converted by its declared
type (`"nothing"` is `nothing` where the type allows it).  Unknown keys are an error; a missing key is
an error when `strict` (a plan, which records every field), else `T`'s default (a user's configuration).
`path` prefixes the keys in messages.
"""
_options_from(::Type{T}, d, path::AbstractString; strict::Bool) where {T} = T(; _options_from(_schema(T), d, path; strict, required = strict)...)

# The NamedTuple of the keys of `d` (a table or a NamedTuple) that `schema` names, in its order, each read by
# its type; unknown keys are an error, and missing ones too when `required` (a struct's, in a plan).
function _options_from(schema::NamedTuple, d, path::AbstractString; strict::Bool, required::Bool = false)
    d isa NamedTuple && (d = Dict{String, Any}(string(k) => v for (k, v) in pairs(d)))
    d isa AbstractDict || throw(ArgumentError("option $(rstrip(path, '.')): $(repr(d)) is not a table"))
    bad = sort!(collect(setdiff(keys(d), string.(keys(schema)))))
    isempty(bad) || throw(ArgumentError("unknown option$(length(bad) > 1 ? "s" : "") $(join(path .* bad, ", ")); valid: $(join(keys(schema), ", "))"))
    for k in keys(schema)
        required && !haskey(d, string(k)) && throw(ArgumentError("option $path$k is missing: the plan was written by another CrowdPhot; re-plan"))
    end
    return NamedTuple(k => _option_value(schema[k], d[string(k)], "$path$k"; strict) for k in keys(schema) if haskey(d, string(k)))
end

const _ESTIMATOR = Union{CrowdPhot.Background.AbstractBackgroundEstimator, CrowdPhot.Background.AbstractBackgroundRMSEstimator}

# An option of declared type FT (or a nested schema) from its TOML or in-memory value v.
function _option_value(FT, v, path; strict)
    FT isa NamedTuple && return _options_from(FT, v, path * "."; strict)
    v == "nothing" && Nothing <: FT && return nothing
    FT <: _ESTIMATOR && return _estimator(FT, v, path; strict)
    if v isa AbstractDict
        for S in (JointDetector, JointFitOptions, PSFRefineOptions)
            S <: FT && return _options_from(S, v, path * "."; strict)
        end
        FT <: AbstractDict && return FT(Symbol(k) => _option_value(valtype(FT), x, "$path.$k"; strict) for (k, x) in v)   # band_weights
    end
    v isa AbstractString && Symbol <: FT && return Symbol(v)
    v isa FT && return v
    # to the member of FT of v's shape (TOML arrays are tuples): `convert` to a Union of more than `nothing`
    # and one type has no method, e.g. `sigma = 3` to Union{Nothing, Float64, NTuple{2, Float64}}
    w = v isa AbstractVector ? Tuple(v) : v
    S = w isa Tuple ? typeintersect(FT, Tuple) : w isa Real ? typeintersect(FT, Real) : FT
    try
        S === Union{} || return convert(S, w)
    catch err
        err isa Union{MethodError, InexactError} || rethrow()
    end
    throw(ArgumentError("option $path: $(repr(v)) is not a $FT"))
end

# A background estimator of type A: one of CrowdPhot.Background's, from its name, or a table with `type` and its
# fields, each made with its keyword constructor, its fields Float64 (they are numeric type parameters, so declared
# `Any`; the device background converts them to its own precision anyway).  An instance is normalized the same way,
# but another module's, or a function (which Background2D also takes), is kept as is: only a plan file is limited
# to Background's (`_toml` refuses the rest).
function _estimator(A, v, path; strict)
    v isa Function && return v
    B = CrowdPhot.Background
    if v isa A
        E = typeof(v)
        (parentmodule(E) === B && !isempty(fieldnames(E))) || return v
        v = Dict{String, Any}("type" => string(nameof(E)), (string(f) => getfield(v, f) for f in fieldnames(E))...)
    end
    name = v isa AbstractString ? v : v isa AbstractDict ? get(v, "type", nothing) : nothing
    name isa AbstractString || throw(ArgumentError("$path: an estimator is a name, or a table with `type` and its parameters"))
    s = Symbol(name)
    E = isdefined(B, s) ? getfield(B, s) : nothing
    E isa Type && E <: A || throw(ArgumentError("$path: $(repr(name)) is not one of CrowdPhot.Background's $(nameof(A))s"))
    v isa AbstractString && return E()
    schema = map(FT -> FT === Any ? Float64 : FT, _schema(E))
    return E(; _options_from(schema, Dict(k => x for (k, x) in v if k != "type"), path * "."; strict, required = strict)...)
end

# ---- the plan's tables ----

_range(r) = [first(r), last(r)]
_range_from(v) = Int(v[1]):Int(v[2])

# What every tile's result depends on, the same in the index and in each shard (and part of every hash).
_plan_table(plan) = Dict{String, Any}("version" => PLAN_VERSION, "overlap" => plan.overlap, "bands" => string.(plan.bands),
    "options" => _toml(plan.options), "products" => string.(collect(plan.products)), "coadd_options" => _toml(plan.coadd_options),
    "product_eltype" => string(plan.product_eltype),
    "frame" => Dict{String, Any}(string(f) => getfield(plan.frame, f) for f in fieldnames(TanGrid)))

function _plan_from(p)
    f = p["frame"]
    pe = Symbol(p["product_eltype"])
    E = isdefined(Base, pe) ? getfield(Base, pe) : nothing
    E isa Type && E <: AbstractFloat || throw(ArgumentError("plan: product_eltype $(repr(p["product_eltype"])) is not a floating-point type"))
    return (; overlap = Int(p["overlap"]), bands = Symbol.(p["bands"]), options = _options_from(JointLoopOptions, p["options"], "options."; strict = true),
            products = Tuple(Symbol.(p["products"])), coadd_options = _options_from(_COADD_OPTIONS, p["coadd_options"], "coadd_options."; strict = true), product_eltype = E,
            frame = TanGrid(f["ra0"], f["dec0"], f["scale"], f["ny"], f["nx"], f["cy"], f["cx"]))
end

_image_table(im) = Dict{String, Any}("name" => im.name, "dims" => collect(im.dims), "band" => string(im.band))
_image_from(d) = (; name = String(d["name"]), dims = (Int(d["dims"][1]), Int(d["dims"][2])), band = Symbol(d["band"]))

_hash(d) = bytes2hex(sha256(sprint(io -> TOML.print(io, d; sorted = true))))

# Two kinds of hash.  A file's (integrity): its whole content in canonical form but the hash itself and the
# CrowdPhot version, so any edit to any field is refused.  A tile's (identity, what lets a re-plan reuse a
# finished tile): what its fit depends on, the plan table, its geometry and its cutouts, each with its image's
# name, size and band looked up through the image table (a cutout holds only the index), and not the image's
# index or the tile's number, cell, shard or mem_est, which a re-plan may change without changing the fit.
# An image is identified by name only: its pixels, weights, WCS, PSF and flux_scale, the CrowdPhot version
# and fit_tile's T are not hashed (see write_plan's docstring).
_content_hash(d) = _hash(Dict{String, Any}(k => v for (k, v) in d if k ∉ ("hash", "crowdphot")))

_tile_hash(ptable, t, images) =
    _hash(Dict{String, Any}("plan" => ptable, "core" => t["core"], "padded" => t["padded"], "offset" => t["offset"],
                            "cutouts" => [(im = images[c["image"]]; Dict{String, Any}("name" => im["name"], "dims" => im["dims"], "band" => im["band"],
                                                                                    "rows" => c["rows"], "cols" => c["cols"])) for c in t["cutouts"]]))

function _check_version(d, file)
    v = get(d, "version", nothing)
    v == PLAN_VERSION || throw(ArgumentError("$file has plan version $(repr(v)); this CrowdPhot reads version $PLAN_VERSION"))
end

# ---- the folder ----

"""
    write_plan(dir, plan; shard_size = 256) -> String

Write `plan` (a [`plan_tiles`](@ref) result) as the plan folder `dir`, which must not exist, and
return the plan's hash.  The folder holds
- `plan.toml`, the index: the plan without its cutouts, with each tile's shard and hash;
- `tiles/shard_<k>.toml`: up to `shard_size` tiles each, with their cutouts, the images those use and
  everything else a tile's fit needs, so a unit reads only its shard;
- `jobs.toml`: one job per cell (an uncovered cell's job has no tiles), with each tile's shard, and
  the plan table every shard carries (what the fits need besides the tiles).
Every value is written, `nothing` as `"nothing"`.  Each file carries a hash of its whole content (the
index's is the plan's hash), and each tile an identity hash of what its fit depends on, which a re-plan
keeps for tiles it doesn't change.  The folder is written next to `dir` and renamed into place, so a
complete folder is the signal that planning finished.  [`read_plan`](@ref) and [`read_job`](@ref) read
it back, checking its version and hashes.

A tile's identity hash knows each image only by its name, size and band, not by its contents: the
pixels, weights, WCS, PSF and `flux_scale` are not part of it, nor are the CrowdPhot version and
`fit_tile`'s `T`.  So a re-plan reuses a finished tile's result even when an image it used has
changed under the same name (re-reduced, re-aligned, recalibrated, or replaced in shared storage
such as an S3 bucket that others write to).  Treat a name as a fixed version of an image, and re-fit
from scratch, rather than reuse results, when inputs may have changed.
"""
function write_plan(dir::AbstractString, plan; shard_size::Integer = 256)
    ispath(dir) && throw(ArgumentError("$dir exists: a plan folder is written once (re-plan into a new folder)"))
    shard_size > 0 || throw(ArgumentError("shard_size must be positive"))
    # (a Dict's values in a read_job result, and an all-uncovered plan has no tiles, which is fine)
    all(t -> hasproperty(t, :cutouts), plan.tiles) ||
        throw(ArgumentError("write_plan needs plan_tiles' result: read_plan and read_job don't keep every tile's cutouts (copy the folder, or re-plan)"))
    ptable = _plan_table(plan)
    images = [_image_table(im) for im in plan.images]
    shard(k) = cld(k, shard_size)
    entries = map(enumerate(plan.tiles)) do (k, t)
        e = Dict{String, Any}("tile" => k, "core" => _range.(collect(t.core)), "padded" => _range.(collect(t.padded)), "offset" => collect(t.offset),
                              "cell" => t.cell, "mem_est" => t.mem_est, "shard" => shard(k),
                              "cutouts" => [Dict{String, Any}("image" => c.image, "rows" => _range(c.rows), "cols" => _range(c.cols)) for c in t.cutouts])
        e["hash"] = _tile_hash(ptable, e, images)
        e
    end
    index = Dict{String, Any}("version" => PLAN_VERSION, "crowdphot" => string(pkgversion(CrowdPhot)), "plan" => ptable, "images" => images,
                              "N" => _toml(plan.N), "M" => _toml(plan.M), "tile_size" => _toml(plan.tile_size),
                              "excluded" => [Dict{String, Any}("name" => x.name, "reason" => x.reason) for x in plan.excluded],
                              "cells" => [_range.(collect(c)) for c in plan.cells], "uncovered" => collect(plan.uncovered),
                              "tiles" => [Dict{String, Any}(k => e[k] for k in ("tile", "core", "padded", "offset", "cell", "mem_est", "shard", "hash")) for e in entries])
    index["hash"] = _content_hash(index)
    jobs = Dict{String, Any}("version" => PLAN_VERSION, "plan_hash" => index["hash"], "plan" => ptable,
                             "jobs" => [(ks = [k for (k, t) in enumerate(plan.tiles) if t.cell == c];
                                         Dict{String, Any}("cells" => [c], "tiles" => ks, "shards" => shard.(ks))) for c in eachindex(plan.cells)])
    jobs["hash"] = _content_hash(jobs)
    # without trailing separators, dirname would give the target itself ("a/plan/" -> "a/plan") and basename ""
    target = rstrip(abspath(dir), ['/', '\\'])
    # beside the target (so the rename is atomic) and named after it (so one a killed write leaves is recognizable);
    # mkdir, not mktempdir, which makes the folder owner-only (0700) whatever the umask, and the rename keeps that
    tmp = joinpath(dirname(target), "." * basename(target) * ".tmp-" * Random.randstring(8))
    mkdir(tmp)
    try
        mkdir(joinpath(tmp, "tiles"))
        for s in 1:shard(length(entries))
            es = entries[((s - 1) * shard_size + 1):min(s * shard_size, length(entries))]
            used = sort!(unique(c["image"] for e in es for c in e["cutouts"]))
            sd = Dict{String, Any}("version" => PLAN_VERSION, "plan_hash" => index["hash"], "plan" => ptable,
                                   "images" => [merge(Dict{String, Any}("index" => i), images[i]) for i in used],
                                   "tiles" => [Dict{String, Any}(k => v for (k, v) in e if k != "shard") for e in es])
            sd["hash"] = _content_hash(sd)
            open(io -> TOML.print(io, sd; sorted = true), joinpath(tmp, "tiles", "shard_$s.toml"), "w")
        end
        open(io -> TOML.print(io, index; sorted = true), joinpath(tmp, "plan.toml"), "w")
        open(io -> TOML.print(io, jobs; sorted = true), joinpath(tmp, "jobs.toml"), "w")
        mv(tmp, target)
    catch
        rm(tmp; recursive = true, force = true)
        rethrow()
    end
    return index["hash"]
end

"""
    read_plan(dir) -> NamedTuple

The plan in folder `dir` (see [`write_plan`](@ref)) from its index, without cutouts: the fields of
[`plan_tiles`](@ref)'s result, its tiles with their `shard` and `hash` in place of `cutouts`, plus the
plan's `hash`.  This is what [`merge_tiles`](@ref), reports and inspection need.  Throws if the index's
version is not this CrowdPhot's or its content doesn't match its hash (an edited or damaged file:
re-plan).
"""
function read_plan(dir::AbstractString)
    file = joinpath(dir, "plan.toml")
    d = TOML.parsefile(file)
    _check_version(d, file)
    _content_hash(d) == d["hash"] || throw(ArgumentError("$file doesn't match its hash: edited or damaged; re-plan"))
    p = _plan_from(d["plan"])
    sentinel(v) = v == "nothing" ? nothing : Int(v)
    tiles = map(enumerate(d["tiles"])) do (k, t)
        t["tile"] == k || throw(ArgumentError("$file: tile $(t["tile"]) is in place $k"))
        padded = Tuple(_range_from.(t["padded"])); offset = (Int(t["offset"][1]), Int(t["offset"][2]))
        (; core = Tuple(_range_from.(t["core"])), padded, offset, frame = _tile_frame(p.frame, padded, offset), cell = Int(t["cell"]),
           mem_est = Float64(t["mem_est"]), shard = Int(t["shard"]), hash = String(t["hash"]))
    end
    return (; N = sentinel(d["N"]), M = sentinel(d["M"]), tile_size = sentinel(d["tile_size"]), p.overlap, p.bands, p.options, p.products,
            p.coadd_options, p.product_eltype, p.frame, images = [_image_from(im) for im in d["images"]],
            excluded = [(; name = String(x["name"]), reason = String(x["reason"])) for x in d["excluded"]],
            cells = [Tuple(_range_from.(c)) for c in d["cells"]], uncovered = Int.(d["uncovered"]), tiles, hash = String(d["hash"]))
end

"""
    read_job(dir, j) -> NamedTuple

Job `j` of the plan folder `dir` (see [`write_plan`](@ref)) from the job file and the shards it names,
without reading the index: `(; job, cells, plan_hash, overlap, bands, options, products, coadd_options,
product_eltype, frame, images, tiles)`.  `tiles` maps each of the job's tile numbers to the tile as
[`plan_tiles`](@ref) made it, cutouts included, plus its identity `hash` (which [`fit_tile`](@ref)
records in its result, for [`merge_tiles`](@ref) to match it by), and `images` maps the plan index of each image its
cutouts use to `(; name, dims, band)`, so a unit loads those files by name.  The result stands in for
the plan in [`fit_tile`](@ref).  Throws if a file's version is not this CrowdPhot's, the job file or a
shard doesn't match its hash, a shard belongs to another plan, or a tile doesn't match its identity hash.
"""
function read_job(dir::AbstractString, j::Integer)
    jfile = joinpath(dir, "jobs.toml")
    jd = TOML.parsefile(jfile)
    _check_version(jd, jfile)
    _content_hash(jd) == jd["hash"] || throw(ArgumentError("$jfile doesn't match its hash: edited or damaged; regroup or re-plan"))
    1 <= j <= length(jd["jobs"]) || throw(ArgumentError("$jfile has jobs 1 to $(length(jd["jobs"])), not $j"))
    job = jd["jobs"][j]
    want = Int.(job["tiles"])
    p = _plan_from(jd["plan"]); phash = _hash(jd["plan"])
    tiles = Dict{Int, Any}(); images = Dict{Int, Any}()
    for s in unique(Int.(job["shards"]))
        file = joinpath(dir, "tiles", "shard_$s.toml")
        sd = TOML.parsefile(file)
        _check_version(sd, file)
        _content_hash(sd) == sd["hash"] || throw(ArgumentError("$file doesn't match its hash: edited or damaged; re-plan"))
        sd["plan_hash"] == jd["plan_hash"] || throw(ArgumentError("$file belongs to plan $(sd["plan_hash"]), not the job file's $(jd["plan_hash"])"))
        _hash(sd["plan"]) == phash || throw(ArgumentError("$file's plan table differs from $jfile's: edited or damaged; re-plan"))
        imtab = Dict(Int(im["index"]) => im for im in sd["images"])
        for t in sd["tiles"]
            k = Int(t["tile"])
            k in want || continue
            _tile_hash(sd["plan"], t, imtab) == t["hash"] || throw(ArgumentError("$file: tile $k doesn't match its hash: edited or damaged; re-plan"))
            padded = Tuple(_range_from.(t["padded"])); offset = (Int(t["offset"][1]), Int(t["offset"][2]))
            # each cutout's name from the image table, the one its identity hash took it from
            cutouts = [(; image = Int(c["image"]), name = String(imtab[Int(c["image"])]["name"]), rows = _range_from(c["rows"]), cols = _range_from(c["cols"]))
                       for c in t["cutouts"]]
            tiles[k] = (; core = Tuple(_range_from.(t["core"])), padded, offset, frame = _tile_frame(p.frame, padded, offset), cell = Int(t["cell"]),
                          mem_est = Float64(t["mem_est"]), cutouts, hash = String(t["hash"]))
            for c in cutouts; images[c.image] = _image_from(imtab[c.image]); end
        end
    end
    missed = setdiff(want, keys(tiles))
    isempty(missed) || throw(ArgumentError("job $j's tiles $missed are not in the shards it names"))
    return (; job = Int(j), cells = Int.(job["cells"]), plan_hash = String(jd["plan_hash"]), p.overlap, p.bands, p.options, p.products,
            p.coadd_options, p.product_eltype, p.frame, images, tiles)
end
