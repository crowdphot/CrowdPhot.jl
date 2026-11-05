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

# In-place inclusive prefix sum.  Each group of SCAN_G items scans a tile of SCAN_G·K consecutive
# elements (K per item, serially, then a Hillis-Steele scan of the items' sums in local memory)
# and adds the inclusive sum of the tiles before it; the tiles' sums are scanned recursively.
# Longer tiles on the CPU amortize its serial execution of a group's items and synchronizations.
const SCAN_G = 256

@kernel function scan_sum_kernel!(tsum, @Const(x), n, K)
    li = @index(Local); g = @index(Group)
    sh = @localmem eltype(tsum) (SCAN_G,)
    base = ((g - 1) * SCAN_G + li - 1) * K
    acc = zero(eltype(tsum))
    for j in 1:K
        base + j <= n && (acc += x[base + j])
    end
    sh[li] = acc
    @synchronize
    li <= 128 && (sh[li] += sh[li + 128])
    @synchronize
    li <= 64 && (sh[li] += sh[li + 64])
    @synchronize
    li <= 32 && (sh[li] += sh[li + 32])
    @synchronize
    li <= 16 && (sh[li] += sh[li + 16])
    @synchronize
    li <= 8 && (sh[li] += sh[li + 8])
    @synchronize
    li <= 4 && (sh[li] += sh[li + 4])
    @synchronize
    li <= 2 && (sh[li] += sh[li + 2])
    @synchronize
    li == 1 && (tsum[g] = sh[1] + sh[2])
end

@kernel function scan_tile_kernel!(x, @Const(toff), n, K, off::Bool)
    li = @index(Local); g = @index(Group)
    sh = @localmem eltype(x) (SCAN_G,)
    tmp = @localmem eltype(x) (SCAN_G,)
    base = ((g - 1) * SCAN_G + li - 1) * K
    acc = zero(eltype(x))
    for j in 1:K
        base + j <= n && (acc += x[base + j])
    end
    sh[li] = acc
    @synchronize
    for d in (1, 2, 4, 8, 16, 32, 64, 128)       # a literal: the CPU backend drops variables across @synchronize
        tmp[li] = li > d ? sh[li - d] : zero(eltype(x))
        @synchronize
        sh[li] += tmp[li]
        @synchronize
    end
    b = ((g - 1) * SCAN_G + li - 1) * K
    run = (li > 1 ? sh[li - 1] : zero(eltype(x))) + ((off && g > 1) ? toff[g - 1] : zero(eltype(x)))
    for j in 1:K
        if b + j <= n
            run += x[b + j]; x[b + j] = run
        end
    end
end

"In-place inclusive prefix sum of the device vector `x`, in its element type."
function scan!(x)
    n = length(x); n == 0 && return x
    be = KA.get_backend(x); K = is_cpu(be) ? 64 : 16
    nt = cld(n, SCAN_G * K)
    if nt == 1
        scan_tile_kernel!(be, SCAN_G)(x, x, n, K, false; ndrange = SCAN_G)
    else
        ts = KA.allocate(be, eltype(x), nt)       # small; left to the GC (freeing would need a synchronize)
        scan_sum_kernel!(be, SCAN_G)(ts, x, n, K; ndrange = SCAN_G * nt)
        scan!(ts)
        scan_tile_kernel!(be, SCAN_G)(x, ts, n, K, true; ndrange = SCAN_G * nt)
    end
    return x
end

# Insertion sort of x[ptr[i] : ptr[i+1]-1] for each segment i.
@kernel function segment_sort_kernel!(x, @Const(ptr))
    i = @index(Global)
    lo = ptr[i]; hi = ptr[i + 1] - Int32(1)
    for p in (lo + Int32(1)):hi
        v = x[p]; q = p - Int32(1)
        while q >= lo && x[q] > v
            x[q + 1] = x[q]; q -= Int32(1)
        end
        x[q + 1] = v
    end
end

# Inclusive scan in x's element type, out of place (`cumsum` widens Int32 to Int64; Metal's
# `cumsum!` compiles anew for every array length).
_scan(x) = scan!(copy(x))

# Counts written into x[2:end]; returns starts from 1: [1, 1 + x[2], 1 + x[2] + x[3], ...].
_scan_from_one(x) = (fill!(view(x, 1:1), one(eltype(x))); _scan(x))
