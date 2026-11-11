# [Reproducibility](@id reproducibility)

The joint fit (`CrowdPhot.Joint`) aims to give results that are reproducible across its
supported backends (the CPU and GPUs) and parallelization options (such as the number of CPU
threads). This page states which changes to a run leave its results exactly the same, which
change them at the level of floating-point rounding, and how large such changes are in a full
catalog.

## What gives identical results

A run repeated with the same inputs, options, backend, element type `T`, and package versions
gives the same catalog, bit for bit. The computations are written so that no result depends on
the number of threads used or the order in which threads finish -- sums that combine many work-items
are taken over a fixed number of groups, and lists filled with atomic operations are sorted before use.

## What changes results at the rounding level

These change how floating-point sums are rounded, and therefore the results:

- the element type `T` (`Float64` or `Float32`);
- the backend (CPU or a GPU), whose compilers order and fuse arithmetic differently;
- the version of KernelAbstractions on the CPU: version 0.9 runs kernels as Julia code, while
  0.10 compiles them through SPIR-V for PoCL, which sums in a different order and uses fused
  multiply-adds, and some functions inside kernels (`erf`, for example) are replaced by
  PoCL's own versions of those functions.

A single fit of a fixed set of sources changes only at that level: between KernelAbstractions
0.9 and 0.10 on the CPU, positions agreed to ``2 \times 10^{-12}`` px and fluxes to
``10^{-11}`` relative. However, the multipass loop structure of CrowdPhot amplifies these differences.
Detection is a threshold, so even minute numerical differences can result in a peak near the threshold
being found in one run and not in the other; from that pass on the two runs fit different catalogs, and the
differences between the catalogs grow in subsequent passes.

In simulated Roman test data of the LMC touchstone field, KernelAbstractions 0.9 and 0.10 gave the same first
pass result exactly, and the same cost to 15 digits in the second pass. One detection differed in the sixth pass.
The final catalogs:

| Matching radius | Sources without a match |
|---|---|
| 0.05 px | 0.58 % |
| 0.5 px | 0.11 % |
| 1 px | 0.08 % |

The sources without a match are typically not the faintest stars in the catalogs. Most are blends divided differently
between the two runs. Matched fluxes agree to a median of ``2 \times 10^{-4}`` relative, and
0.2 % of the matched sources differ by more than 5 %. The same field on the Metal backend gave
identical catalogs with KernelAbstractions 0.9 (Metal.jl 1.11) and 0.10 (Metal.jl 1.12).

These numbers come from one field and one pair of runs and are only meant to be illustrative of the scale of the effect.

## Comparing two runs

Some recommendations on how to compare runs that are not bitwise-identical:

- Match the catalogs by position (about 1 px), not by row: a different catalog history changes
  the row order and the set of sources.
- Compare matched fluxes in units of their errors. Differences much smaller than the errors are the
  expected rounding effects; systematic offsets are not.
- Record the backend, the element type, the number of threads, and the package versions (the
  `Manifest.toml`) with every run, so that a run can be repeated exactly.
