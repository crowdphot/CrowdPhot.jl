# ------------------------------------------------------------------------------------------
# Cell grid and neighbor lists within a radius
# ------------------------------------------------------------------------------------------
#
# Fixed-radius neighbor searches in two dimensions on a uniform cell grid, on any backend: the
# points are counting-sorted into cells, and each query point scans its 3x3 cells.  A KD-tree is
# the general tool, but these searches are one radius at roughly uniform density, the grid's best
# case.  `cell_grid` is also the grid of `PairPattern` (linearize.jl: Chebyshev, one point set,
# with a transposed index); the searches here are Euclidean, with an optional second point set.

"""
    NeighborList{VI}

Neighbors as compressed rows: row `i`'s neighbors are `idx[ptr[i]:ptr[i+1]-1]`, in ascending order.
From [`pairs_within`](@ref) (one point set; row `i` lists its neighbors `j > i`, so every pair
appears once) or [`cross_pairs`](@ref) (rows are the query points, `idx` indexes the reference
points).  `ptr` and `idx` are `Int32` vectors on the backend the search ran on.
"""
struct NeighborList{VI}
    ptr::VI
    idx::VI
end

# Cell of (u, v), zero-based and clamped into the grid; cg = (u0, v0, cell side, nu, nv).  A point
# outside the grid clamps to an edge cell, which still finds every point within the cell side: those
# lie in the cells next to the grid's edge.
@inline function grid_cell(u, v, cg)
    a = unsafe_trunc(Int32, clamp(floor((u - cg[1]) / cg[3]), zero(u), cg[4] - one(u)))
    b = unsafe_trunc(Int32, clamp(floor((v - cg[2]) / cg[3]), zero(v), cg[5] - one(v)))
    return a, b
end
@inline grid_cell_index(a, b, nu, nv) = ((0 <= a < nu) & (0 <= b < nv)) ? a + nu * b + Int32(1) : Int32(0)

@kernel function grid_cell_kernel!(cell, ccount, @Const(R), cg)
    j = @index(Global)
    u = R[1, j]; v = R[2, j]
    if isfinite(u) && isfinite(v)
        a, b = grid_cell(u, v, cg)
        c = a + unsafe_trunc(Int32, cg[4]) * b + Int32(1)
        cell[j] = c
        Atomix.@atomic ccount[c + 1] += Int32(1)
    else
        cell[j] = Int32(0)                         # in no cell: never a neighbor
    end
end

@kernel function grid_scatter_kernel!(order, cfill, @Const(cell), @Const(cstart))
    j = @index(Global)
    c = cell[j]
    if c > 0
        k = Atomix.@atomic cfill[c] += Int32(1)    # the post-increment value
        order[cstart[c] + k - Int32(1)] = j
    end
end

# The cell grid of points R (rows 1-2) for searches within h: each point's cell (`cell`, 0 for a
# non-finite position) and the points counting-sorted by cell (`order`, cell c's from `cstart[c]`).
# Cells are at least h wide (so the 3x3 scan covers the search) and, like `_SepGrid`, no more than
# about 4 per point (so a small h over a large field does not allocate mostly empty cells).  The grid
# spans [0.5, ext + 0.5] per axis when `ext = (ny, nx)` is given, else R's finite bounding box (four
# host reads).
function cell_grid(be, R, h, ext = nothing)
    T = eltype(R); N = size(R, 2)
    if ext === nothing
        big = typemax(T); small = typemin(T)                 # values, not the type, in the closures (GPU kernels)
        lo(k) = mapreduce(x -> isfinite(x) ? x : big, min, view(R, k, :); init = big)
        hi(k) = mapreduce(x -> isfinite(x) ? x : small, max, view(R, k, :); init = small)
        u0, u1, v0, v1 = N > 0 ? (lo(1), hi(1), lo(2), hi(2)) : (zero(T), zero(T), zero(T), zero(T))
        isfinite(u0) && isfinite(v0) || (u0 = u1 = v0 = v1 = zero(T))
    else
        u0, v0 = T(0.5), T(0.5); u1, v1 = T(ext[1] + 0.5), T(ext[2] + 0.5)
    end
    du = u1 - u0; dv = v1 - v0
    w = max(T(h), sqrt(du * dv / T(4 * max(N, 1))), eps(T))
    while (floor(Int, du / w) + 1) * (floor(Int, dv / w) + 1) > 16 * N + 16; w *= 2; end   # a degenerate (line-like) box
    nu = floor(Int, du / w) + 1; nv = floor(Int, dv / w) + 1
    cg = (T(u0), T(v0), w, T(nu), T(nv))
    z(n) = KA.zeros(be, Int32, n)
    cell = z(N); ccount = z(nu * nv + 1); cfill = z(nu * nv); order = z(N)
    launch!(grid_cell_kernel!, be, N, cell, ccount, R, cg)
    cstart = _scan_from_one(ccount)
    launch!(grid_scatter_kernel!, be, N, order, cfill, cell, cstart)
    return (; cg, cell, cstart, order)
end

# Count (`write = false`) or write (`true`) query i's reference points within the radius (squared:
# r2); with `self` (queries and references the same set) only j > i.
@kernel function nb_scan_kernel!(cnt, idx, @Const(ptr), @Const(Q), @Const(R), @Const(cstart), @Const(order), cg, r2, self::Bool, write::Bool)
    i = @index(Global)
    qu = Q[1, i]; qv = Q[2, i]
    n = Int32(0); k = write ? ptr[i] : Int32(0)
    if isfinite(qu) && isfinite(qv)
        a, b = grid_cell(qu, qv, cg)
        nu = unsafe_trunc(Int32, cg[4]); nv = unsafe_trunc(Int32, cg[5])
        for db in Int32(-1):Int32(1), da in Int32(-1):Int32(1)
            cc = grid_cell_index(a + da, b + db, nu, nv)
            cc == 0 && continue
            for q in cstart[cc]:(cstart[cc + Int32(1)] - Int32(1))
                j = order[q]
                (self & (j <= i)) && continue
                du = R[1, j] - qu; dv = R[2, j] - qv
                du * du + dv * dv <= r2 || continue
                if write
                    idx[k] = j; k += Int32(1)
                else
                    n += Int32(1)
                end
            end
        end
    end
    write || (cnt[i + 1] = n)
end

# Query i's nearest neighbor in the list (0 without one); ties go to the lower index (rows ascend).
@kernel function nb_nearest_kernel!(out, @Const(ptr), @Const(idx), @Const(Q), @Const(R))
    i = @index(Global)
    best = Int32(0); bd = typemax(eltype(Q))
    for q in ptr[i]:(ptr[i + 1] - Int32(1))
        j = idx[q]
        du = R[1, j] - Q[1, i]; dv = R[2, j] - Q[2, i]; dd = du * du + dv * dv
        dd < bd && (bd = dd; best = j)
    end
    out[i] = best
end

function nb_search(be, Q, R, r, self)
    T = eltype(R); NQ = size(Q, 2)
    g = cell_grid(be, R, r)
    cnt = KA.zeros(be, Int32, NQ + 1)
    launch!(nb_scan_kernel!, be, NQ, cnt, cnt, cnt, Q, R, g.cstart, g.order, g.cg, T(r)^2, self, false)
    ptr = _scan_from_one(cnt)
    np = Int(Array(view(ptr, (NQ + 1):(NQ + 1)))[1]) - 1     # one host read, to size the list
    idx = KA.zeros(be, Int32, np)
    launch!(nb_scan_kernel!, be, NQ, cnt, idx, ptr, Q, R, g.cstart, g.order, g.cg, T(r)^2, self, true)
    launch!(segment_sort_kernel!, be, NQ, idx, ptr)           # atomics make the order run-dependent: sort each row
    return NeighborList(ptr, idx)
end

"""
    pairs_within(be, P, r) -> NeighborList

Every pair of points `i < j` of `P` (positions in rows 1-2 of a `2 x N` or taller matrix on backend
`be`) within Euclidean distance `r` (inclusive): row `i` of the result lists its `j > i`, ascending.
Non-finite positions have no neighbors.  Deterministic, and the same on every backend.
"""
pairs_within(be, P::AbstractMatrix, r::Real) = nb_search(be, P, P, r, true)

"""
    cross_pairs(be, A, B, r) -> NeighborList

For each point of `A`, the points of `B` within Euclidean distance `r` (inclusive), ascending
(positions in rows 1-2 of each matrix, on backend `be`; the same element type).  Non-finite
positions have no neighbors.
"""
cross_pairs(be, A::AbstractMatrix, B::AbstractMatrix, r::Real) = nb_search(be, A, B, r, false)

"""
    nearest_within(be, A, B, r) -> Vector{Int32}

For each point of `A`, the index of its nearest point of `B` within Euclidean distance `r`, or 0
if there is none; equal distances go to the lower index.  An `Int32` vector on backend `be`.
"""
function nearest_within(be, A::AbstractMatrix, B::AbstractMatrix, r::Real)
    nl = cross_pairs(be, A, B, r)
    out = KA.zeros(be, Int32, size(A, 2))
    launch!(nb_nearest_kernel!, be, size(A, 2), out, nl.ptr, nl.idx, A, B)
    return out
end
