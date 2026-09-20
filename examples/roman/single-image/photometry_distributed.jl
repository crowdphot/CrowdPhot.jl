# photometry_distributed.jl
#
# Run `photometry.jl`'s `main` over a directory of Roman L2 exposures, several
# at a time, using persistent Julia worker processes.  See
# https://docs.julialang.org/en/v1/manual/distributed-computing/
#
# Run as
#     julia --project=. photometry_distributed.jl
# If running on a cluster, scale the number of worker processes you request for
# the job to match the number of Julia worker processes created below by `addprocs`,
# adding 1 additional worker for the main orchestrator process.

using Distributed

# Practical limit to worker count is memory usage. You should budget roughly 6 GB
# per worker process for the simultaneous fitter.
addprocs(6)

# The path is interpolated with `$` so it is resolved here and shipped as a
# literal; workers do not share this process's working directory.
const SCRIPT = joinpath(@__DIR__, "photometry.jl")
@everywhere include($SCRIPT) # @everywhere loads code across all worker processes

# Edit this to point at your own data.
const l2dir = joinpath(@__DIR__, "l2_asdf")

jobs = sort(filter(endswith(".asdf"), readdir(l2dir; join = true)))
println("Found $(length(jobs)) exposures in $l2dir")

# `pmap` hands tasks to workers one at a time as they finish.  The alternative,
# `@sync @distributed for ...`, splits the loop into equal chunks up front,
# which stalls on the slowest chunk when exposures take unequal time -- and
# they do, since run time scales with the number of sources detected.
#
# `on_error = identity` returns the exception as that job's result instead of
# aborting the whole `pmap`.
results = pmap(jobs; on_error = identity) do l2_path
    println("STARTING $(basename(l2_path)) ON WORKER $(myid())")
    outpath = main(l2_path)
    println("FINISHED $(basename(l2_path)) ON WORKER $(myid())")
    return outpath
end

# Check if any jobs failed.
failed = [(job, r) for (job, r) in zip(jobs, results) if r isa Exception]
println("\n$(length(jobs) - length(failed)) of $(length(jobs)) exposures succeeded")
for (job, err) in failed
    println("FAILED $(basename(job)): $err")
end
