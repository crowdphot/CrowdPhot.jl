# Backend plumbing shared by every kernel family.  The math lives in `@inline` functions that
# both CPU and GPU kernels call; only launch geometry differs by backend, and it is chosen here
# and in each family's host driver.

"Copy host array `x` to backend `be`."
to_device(be, x::AbstractArray) = copyto!(KA.allocate(be, eltype(x), size(x)), x)
to_device(be, ::Type{T}, x::AbstractArray) where {T} = to_device(be, convert(Array{T}, x))

# The workgroup size must be a compile-time constant: a dynamic size makes the CPU backend's
# `@private` arrays type-unstable (100-250x slower kernels).  The CPU backend splits the
# workgroups into `nthreads()` contiguous, equal-count chunks, one task per thread and no work
# stealing; each task runs its groups' work items serially.  The sizes aim for at least 8 groups
# per thread so the split's granularity stays small (it cannot balance uneven work per group).
cpu_groupsize(n) = (t = 8 * Threads.nthreads(); n >= 256t ? 256 : n >= 16t ? 16 : 1)

"Launch kernel `k` over `n` work items with the backend's default 1-D geometry."
launch!(k, be::KA.CPU, n, args...) = (n > 0 && k(be, cpu_groupsize(n))(args...; ndrange = n); nothing)
launch!(k, be, n, args...) = (n > 0 && k(be, 128)(args...; ndrange = n); nothing)

is_cpu(be) = be isa KA.CPU
