# Finalize: neighbor-marginalized errors of the joint fit and the final significance.
#
# The multi-image generalization of the single-image `_source_errors!`: for source a, its
# active parameters (u, v and every band with data) and those of its `k` most strongly coupled
# neighbors (Frobenius norm of the equilibrated coupling block over a's active columns) form a
# principal submatrix of the Gauss-Newton H with a last.  The Cholesky factor's last block
# gives a's Schur complement, whose inverse is a's covariance with those neighbors
# marginalized.  `k = 0` is the own block only.  H sums every image's information, so a band's
# flux error is the joint (all-exposure) error of that band.  Host side, threaded over sources,
# with a hand-written Cholesky: at these sizes (L <= (k+1)P, median ~40) it is as fast as LAPACK
# per matrix, and LAPACK called from many threads stopped scaling past ~4 tasks even with one
# BLAS thread (51 ms with 17 BLAS threads, 11 ms with 1, 2 ms hand-written; 6326 sources, 18 threads).

"Host copy of a linearization's equilibrated H blocks and pattern, with a block lookup."
struct HostH{T}
    Hd::Array{T, 3}
    Ho::Array{T, 3}
    D::Matrix{T}
    hdiag::Matrix{T}
    ptr::Vector{Int32}
    nbr::Vector{Int32}
    cptr::Vector{Int32}
    cslot::Vector{Int32}
    row::Vector{Int32}
end

HostH(lin::Linearization) = HostH(Array(lin.Hd), Array(lin.Ho), Array(lin.D), Array(lin.hdiag), Array(lin.pat.ptr), Array(lin.pat.nbr),
                                  Array(lin.pat.cptr), Array(lin.pat.cslot), Array(lin.pat.row))

"Equilibrated block H[a, b] (rows a's parameters), or `nothing` if the pair has no slot."
function hblock(h::HostH, a, b)
    a == b && return view(h.Hd, :, :, a)
    i, j = minmax(a, b)
    r = h.ptr[i]:(h.ptr[i + 1] - 1)
    k = searchsortedfirst(view(h.nbr, r), j)
    (k <= length(r) && h.nbr[r[k]] == j) || return nothing
    B = view(h.Ho, :, :, r[k])
    return a < b ? B : transpose(B)
end

"""
    marginal_errors(h::HostH; k = 8) -> Matrix (P x N)

1σ errors of every parameter with the `k` most strongly coupled neighbors marginalized (NaN
for parameters without data).
"""
function marginal_errors(h::HostH{T}; k::Integer = 8) where {T}
    P, N = size(h.D)
    σ = fill(T(NaN), P, N)
    Lmax = (k + 1) * P
    # per-task buffers; column k + 1 of `act` holds a's active parameters
    chunks = collect(Iterators.partition(1:N, max(1, cld(N, 8 * Threads.nthreads()))))
    Threads.@sync for ch in chunks
        Threads.@spawn begin
            M = Matrix{T}(undef, Lmax, Lmax); V = Matrix{T}(undef, P, P)
            nb = zeros(Int, k); sc = zeros(T, k)
            srcs = zeros(Int, k + 1); act = zeros(Int, P, k + 1); na = zeros(Int, k + 1); off = zeros(Int, k + 2)
            for a in ch
                m = 0
                for p in 1:P; h.hdiag[p, a] > 0 && (act[m += 1, k + 1] = p); end
                m == 0 && continue
                # the k neighbors most strongly coupled over a's active parameters, descending
                cnt = 0
                for q in h.ptr[a]:(h.ptr[a + 1] - 1)
                    s = zero(T); for c in 1:P, t in 1:m; s += abs2(h.Ho[act[t, k + 1], c, q]); end
                    cnt = _topk_insert!(nb, sc, cnt, k, h.nbr[q], s)
                end
                for t2 in h.cptr[a]:(h.cptr[a + 1] - 1)
                    q = h.cslot[t2]
                    s = zero(T); for t in 1:m, r in 1:P; s += abs2(h.Ho[r, act[t, k + 1], q]); end
                    cnt = _topk_insert!(nb, sc, cnt, k, h.row[q], s)
                end
                # blocks: the neighbors, then a last
                nsrc = cnt + 1
                for ib in 1:cnt
                    b = nb[ib]; c = 0
                    for p in 1:P; h.hdiag[p, b] > 0 && (act[c += 1, ib] = p); end
                    srcs[ib] = b; na[ib] = c
                end
                srcs[nsrc] = a; na[nsrc] = m
                for t in 1:m; act[t, nsrc] = act[t, k + 1]; end
                off[1] = 0; for ib in 1:nsrc; off[ib + 1] = off[ib] + na[ib]; end
                L = off[nsrc + 1]
                fill!(view(M, 1:L, 1:L), zero(T))
                for jb in 1:nsrc, ib in 1:jb
                    B = hblock(h, srcs[ib], srcs[jb])
                    B === nothing && continue
                    for c in 1:na[jb], r in 1:na[ib]
                        M[off[ib] + r, off[jb] + c] = B[act[r, ib], act[c, jb]]
                    end
                end
                tr_ = zero(T); for i in 1:L; tr_ += M[i, i]; end
                for i in 1:L; M[i, i] += T(1e-12) * tr_; end
                # a's covariance is (UᵀU)⁻¹ for U the last m x m block of the factor, so its
                # diagonal is the row norms of U⁻¹
                o = L - m
                if !_chol_upper!(M, L)
                    o = 0                              # own block, as the single-image fitter falls back
                    for c in 1:m, r in 1:m; M[r, c] = h.Hd[act[r, nsrc], act[c, nsrc], a]; end
                    _chol_upper!(M, m) || continue
                end
                for c in 1:m, r in m:-1:1              # V = U⁻¹ by back substitution
                    t = r == c ? one(T) : zero(T)
                    for i in (r + 1):m; t -= M[o + r, o + i] * V[i, c]; end
                    V[r, c] = t / M[o + r, o + r]
                end
                for t in 1:m
                    s = zero(T); for j in 1:m; s += abs2(V[t, j]); end
                    p = act[t, nsrc]; σ[p, a] = sqrt(max(s, zero(T))) / h.D[p, a]
                end
            end
        end
    end
    return σ
end

# Insert neighbor j with score s into the descending top-k lists (nb, sc) holding cnt entries.
function _topk_insert!(nb, sc, cnt, k, j, s)
    (s > 0 && (cnt < k || (k > 0 && s > sc[k]))) || return cnt
    pos = cnt < k ? (cnt += 1) : k
    while pos > 1 && sc[pos - 1] < s
        sc[pos] = sc[pos - 1]; nb[pos] = nb[pos - 1]; pos -= 1
    end
    sc[pos] = s; nb[pos] = j
    return cnt
end

# Upper Cholesky of M[1:L, 1:L] in place (M = UᵀU); false if not positive definite.
function _chol_upper!(M, L)
    @inbounds for j in 1:L
        s = M[j, j]
        @simd for i in 1:(j - 1); s -= M[i, j] * M[i, j]; end
        s > 0 || return false
        d = sqrt(s); M[j, j] = d
        for c in (j + 1):L
            t = M[j, c]
            @simd for i in 1:(j - 1); t -= M[i, j] * M[i, c]; end
            M[j, c] = t / d
        end
    end
    return true
end

"Per-band f/σ (N x B) and their Stouffer combination over bands with data."
function final_significance(θ::AbstractMatrix, σ::AbstractMatrix)
    P, N = size(θ); B = P - 2
    sb = fill(NaN, N, B); joint = fill(NaN, N)
    for s in 1:N
        acc = 0.0; nb = 0
        for b in 1:B
            e = σ[2 + b, s]
            (isfinite(e) && e > 0) || continue
            sb[s, b] = θ[2 + b, s] / e; acc += sb[s, b]; nb += 1
        end
        nb > 0 && (joint[s] = acc / sqrt(nb))
    end
    return sb, joint
end
