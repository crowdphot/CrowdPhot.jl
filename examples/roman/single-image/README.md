# Roman WFI photometry: single image

A minimal end-to-end photometry run. Point it at one Roman WFI Level 2 (`_cal.asdf`)
exposure and it resolves the matching reference files from CRDS, runs photometry,
and writes a flat table of results.

| file | what it does |
|---|---|
| `photometry.jl` | `main(l2_path)` — one exposure, start to finish |
| `photometry_distributed.jl` | runs `main` over a directory, several workers at a time |
| `crds_query_pythoncall.jl` | resolves reference files by calling CRDS through Python |

## Setup

Copy the contents of this directory to wherever you want to run from. `Project.toml`
pulls CrowdPhot straight from GitHub, so the project configuration is portable.

From that directory, start Julia on this project and instantiate it once:

```
julia --project=.
```
```julia
julia> ]                       # enter the package manager
(single-image) pkg> instantiate
```

This clones CrowdPhot and resolves the rest of the dependencies. It takes a few
minutes the first time. Backspace leaves the package manager.

Two paths in the scripts to check before you run:

| where | default | what to change it to |
|---|---|---|
| `outdir` keyword of `main` in `photometry.jl` | `"results"`, relative to your working directory | directory to write the Parquet catalogs |
| `l2dir` at the top of `photometry_distributed.jl` | `l2_asdf/` next to the script | the directory holding your `_cal.asdf` exposures |

`photometry.jl` takes the exposure path as an argument, so it needs no editing.
`photometry_distributed.jl` finds its inputs with `readdir(l2dir)`, so it does.

## Running it

```
julia --project=. photometry.jl /path/to/r0100601001001001001_0001_wfi01_f087_cal.asdf
```

Output is a Parquet file under `results/`, named after the input. Any Parquet
reader should be able to open it; e.g., from Python, `pandas.read_parquet`.

**Prefer a REPL while you are experimenting.** Due to Julia's just-in-time compilation
process, the first `main` call each time you restart the `julia` process recompiles
a lot of code. That cost is paid once per session, not once per file, making multiple
`main` calls to test different keyword arguments from the same session is much faster
than re-running the script from the shell like `julia --project=. photometry.jl 
/path/to/r0100601001001001001_0001_wfi01_f087_cal.asdf`. Prefer this workflow:

```julia
julia --project=.

julia> include("photometry.jl")
julia> main("/path/to/exposure.asdf")
# Try another set of parameters, from the same REPL session
julia> main("/path/to/exposure.asdf"; fit_rad = 4, detect_sigma = 3.0)
```

Most of the relevant keyword arguments to tune the photometry routine
are keyword arguments on `main`, listed with their defaults at the top of `photometry.jl`.
If you edit the `main` function in `photometry.jl` on disk, the change does not
immediately get picked up from an active REPL session, you need to re-include it
`include("photometry.jl")` to get the updated `main` definition.

For many exposures, edit `l2dir` at the top of `photometry_distributed.jl` and
run it. It uses 6 worker processes by default; budget about 6 GB of memory each.

## CRDS setup

Reference files (the ePSF and the pixel area map) are resolved by CRDS, the
same service the Roman pipeline uses. There is no Julia implementation, so this
example calls the `crds` Python package through
[PythonCall.jl](https://github.com/JuliaPy/PythonCall.jl).

### Two environment variables you must set

These are needed no matter which Python is used:

```sh
export CRDS_PATH=/somewhere/with/space/crds-cache
export CRDS_SERVER_URL=https://roman-crds.stsci.edu
```

`CRDS_PATH` is a local cache directory. Reference files are large — a single
ePSF is roughly 180 MB, and there is a distinct one per (detector, filter)
pair — so point it somewhere with room. The example checks both variables at
load time and fails with a clear message rather than a Python traceback.

### Where Python comes from

**By default, you do not have to do anything.** `CondaPkg.toml` declares `crds`
as a dependency, and on first run CondaPkg downloads a private Python
environment and installs CRDS into it. This is self-contained and does not
touch any Python you already have. A full installation is ~1.4 GB.

**To use a Python you already have**, one with `crds` installed, set both of:

```sh
export JULIA_CONDAPKG_BACKEND=Null
export JULIA_PYTHONCALL_EXE=/path/to/your/python
```

These should be set in the environment *before* Julia starts. PythonCall picks
its interpreter when the module initializes, so they cannot be set after
you execute `using PythonCall`. If you
want them set from Julia, they have to be assigned at the very top of a script,
above the `include` of `crds_query_pythoncall.jl`.

## What the example does

1. `Roman.load_l2` reads the science array, error array and DQ flags,
   transposing into CrowdPhot's `(y, x)` convention.
2. CRDS resolves the ePSF and pixel area map for this exposure.
3. The pixel area map is multiplied into the image. Roman L2 data are in
   surface brightness units, so this is required for correct fluxes, not an
   optional refinement.
4. Inverse-variance weights are built from the error array, with DQ-flagged and
   non-finite pixels zeroed. Zero weight means "ignore this pixel", which is
   how bad pixels are excluded from both detection and fitting.
5. `fit_all_stars_simultaneous_multipass` iterates background estimation,
   detection, deblending, a simultaneous fit of every source at once, and
   pruning, until it converges.
6. `to_table` flattens the result to the columns most analyses want, with the
   PSF-referencing already applied to the morphology statistics.

## What it leaves out

Everything that is analysis rather than pipeline: no plots, no crossmatching
against an input catalog, no completeness or bias modeling. It also uses the
calibrated error array directly rather than adding an empirical confusion term,
which matters in the most crowded fields.
