using CrowdPhot
using CrowdPhot.Joint: NeighborList, pairs_within, cross_pairs, nearest_within
using StableRNGs, Test

const KA = CrowdPhot.Joint.KA
const CPU = KA.CPU()

rows(nl::NeighborList) = [Int.(nl.idx[nl.ptr[i]:(nl.ptr[i + 1] - 1)]) for i in 1:(length(nl.ptr) - 1)]
inside(A, i, B, j, r) = all(isfinite, (A[1, i], A[2, i], B[1, j], B[2, j])) && (A[1, i] - B[1, j])^2 + (A[2, i] - B[2, j])^2 <= r^2
brute_self(P, r) = [[j for j in (i + 1):size(P, 2) if inside(P, i, P, j, r)] for i in 1:size(P, 2)]
brute_cross(A, B, r) = [[j for j in 1:size(B, 2) if inside(A, i, B, j, r)] for i in 1:size(A, 2)]
function brute_nearest(A, B, r)
    map(1:size(A, 2)) do i
        best = 0; bd = Inf
        for j in 1:size(B, 2)
            inside(A, i, B, j, r) || continue
            dd = (A[1, i] - B[1, j])^2 + (A[2, i] - B[2, j])^2
            dd < bd && (bd = dd; best = j)
        end
        best
    end
end

# uniform points, a cluster of near-duplicates, exact duplicates, a pair exactly r apart, non-finite positions
function points(rng, T, n; r = T(1))
    P = T.(200 .* rand(rng, 2, n))
    P[:, 1:40] .= T(50) .+ T(0.3) .* rand(rng, T, 2, 40)
    P[:, 41:43] .= P[:, 44:44]
    P[:, 45] .= (T(10), T(10)); P[:, 46] .= (T(10) + r, T(10))
    P[1, 47] = T(NaN); P[2, 48] = T(Inf)
    return P
end

@testset "pairs_within" begin
    rng = StableRNG(1)
    for T in (Float64, Float32), r in (0.5, 1.0, 3.0, 50.0)
        P = points(rng, T, 1500; r = T(r))
        nl = pairs_within(CPU, P, r)
        @test rows(nl) == brute_self(P, T(r))
        @test 46 in rows(nl)[45]                                    # exactly r apart: inclusive
        @test isempty(rows(nl)[47]) && isempty(rows(nl)[48]) && all(i -> !(47 in i) && !(48 in i), rows(nl))
        @test all(issorted, rows(nl))
        nl2 = pairs_within(CPU, P, r); @test nl2.ptr == nl.ptr && nl2.idx == nl.idx   # deterministic
    end
    # a radius far below the spacing over a large field: the grid stays small and nothing is missed
    P = 1e4 .* rand(rng, 2, 2000); P[:, 2] .= P[:, 1] .+ (1e-5, 0.0)
    nl = pairs_within(CPU, P, 1e-4)
    @test rows(nl) == brute_self(P, 1e-4) && rows(nl)[1] == [2]
    # rows 1-2 of a taller matrix; empty and single-point sets
    Q = vcat(P, rand(rng, 3, 2000)); @test rows(pairs_within(CPU, Q, 50.0)) == brute_self(P, 50.0)
    @test rows(pairs_within(CPU, zeros(2, 0), 1.0)) == Vector{Int}[]
    @test rows(pairs_within(CPU, [1.0; 2.0;;], 1.0)) == [Int[]]
end

@testset "cross_pairs and nearest_within" begin
    rng = StableRNG(2)
    for T in (Float64, Float32), r in (0.5, 2.0, 30.0)
        B = points(rng, T, 2000; r = T(r))
        A = T.(-20 .+ 240 .* rand(rng, 2, 800))                      # some queries outside B's bounding box
        A[:, 1] .= (T(-1e6), T(5)); A[1, 2] = T(NaN)                 # far outside; non-finite
        A[:, 3] .= B[:, 45]                                          # on a B point
        nl = cross_pairs(CPU, A, B, r)
        @test rows(nl) == brute_cross(A, B, T(r))
        @test isempty(rows(nl)[1]) && isempty(rows(nl)[2]) && 45 in rows(nl)[3]
        @test Int.(nearest_within(CPU, A, B, r)) == brute_nearest(A, B, T(r))
    end
    # equal distances go to the lower index
    B = [0.0 2.0 -2.0; 0.0 0.0 0.0]; A = [0.0; 0.0;;]
    @test nearest_within(CPU, A .+ [1.0; 0.0;;], B, 1.0) == Int32[1]   # B 1 and 2 both at 1: the lower
    @test nearest_within(CPU, A, B, 3.0) == Int32[1]
    @test nearest_within(CPU, A .+ [10.0; 0.0;;], B, 1.0) == Int32[0]
    # empty sets on either side
    @test rows(cross_pairs(CPU, zeros(2, 0), B, 1.0)) == Vector{Int}[]
    @test rows(cross_pairs(CPU, B, zeros(2, 0), 1.0)) == [Int[], Int[], Int[]]
    @test nearest_within(CPU, B, zeros(2, 0), 1.0) == Int32[0, 0, 0]
end
