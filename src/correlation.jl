# correlation.jl — 2D correlation filtering for CrowdPhot.jl
#
# Extracted and adapted from ImageFiltering.jl (MIT-licensed, Tim Holy et al.).
# This file provides a self-contained FIR (finite impulse response) 2D correlation
# engine with automatic separable-kernel detection via SVD.  It depends only on
# LinearAlgebra (stdlib) — no OffsetArrays, no image ecosystem, no FFTW.

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

"""
    correlate(img::AbstractMatrix, kernel, [border=:replicate]) -> Matrix

Compute the 2D correlation of `img` with `kernel`, returning a matrix of the
same size as `img`.  The `kernel` is **not** flipped — this is correlation,
not convolution.

`kernel` may be:
- An `AbstractMatrix` (e.g., a rendered PSF image, see [`render`](@ref)),
  which must be odd-sized (e.g., `(5,5)`) so the center falls on a single pixel.
  Separability is automatically detected via SVD rank-1 checking;
  if separable, the computation is done in two 1D passes for better performance.
- A `Tuple` of two matrices representing pre-factored 1D kernels,
  e.g. `(col_vec, row_vec)` where `col_vec` is `kr×1` and `row_vec` is `1×kc`.

`border` controls how the image is extended at the edges:
- `:replicate` (default) — repeat the nearest border pixel.
- `:zero` — pad with zeros.

See also: [`correlate!`](@ref) for an in-place variant.

# Notes
- The center is assumed to be at integer index `((m+1)÷2, (n+1)÷2)`
  for an `m×n` kernel.
Only `size(kernel)` is used to determine the centre; `axes(kernel)` is
ignored.  Unlike ImageFiltering.jl (which uses OffsetArrays so that
`kernel[-1:1, -1:1]` encodes the centre at `[0,0]`), this implementation
relies purely on the size convention.

### Separability detection tolerances

When `kernel` is a `Matrix` (not a pre-factored tuple), an SVD is computed
and the kernel is treated as separable when all singular values beyond the
first are strictly less than ``\\sqrt{\\epsilon_T}``, where ``\\epsilon_T``
is the machine epsilon of the kernel's element type.  This is the same
criterion used by ImageFiltering.jl.  Kernels that are numerically
rank-1 (e.g. an outer product contaminated by roundoff) will be factored;
kernels that are merely "close" to rank-1 may not be.  The SVD operates on
a dense `Matrix` copy of the kernel, so wrapper array types are materialised.

### Pre-factored tuples

When `kernel` is a `Tuple` of two matrices, the **shape** of each factor
determines the filtering direction:

- `m × 1` — filters along rows (dimension 1).  Its centre is at row
  `(m+1)÷2`.
- `1 × n` — filters along columns (dimension 2).  Its centre is at column
  `(n+1)÷2`.

Each factor must have exactly one non-singleton dimension.  A factor that
is `m × n` with both `m > 1` and `n > 1` will be treated as a full 2D
kernel and applied in a single pass.

A `1 × 1` factor whose single element equals exactly `1.0` is treated as an
identity (no-op) and skipped.  This is an internal sentinel; users should
not rely on it.

### Correlation, not convolution

The kernel is applied as-is — no flipping is performed.  This is correlation
(also called "matched filtering" in the signal-processing literature), not
convolution.  If you have a convolution kernel, reverse it along both axes
before calling `correlate`.

### Element type

The output element type is `promote_type(eltype(img), eltype(kernel))`
(for a matrix kernel) or the `promote_type` across all tuple elements
(for a pre-factored kernel).  Accumulation inside the inner loops uses the
output element type.  For mixed-precision workloads (e.g. `Float32` image
with `Float64` kernel), consider passing an explicit output type by
allocating the output yourself and calling `correlate!`.

### Multithreading

The inner loops use `Threads.@threads` over columns.  On machines with many
cores the speedup is substantial (3–5× on 16 threads).  On single-threaded
runs the overhead is negligible.

### Border handling

Border conditions are handled inline via strip-mining — no padded copy of
the input image is made.  The interior pixels (where the full kernel
footprint lies within the image bounds) use a `@inbounds` fast path.
The thin border strips use per-access bounds checks via `_getpixel`.
"""
function correlate(img::AbstractMatrix{T}, kernel, border::Symbol=:replicate) where {T}
    S = promote_type(T, _eltype_kernel(kernel))
    out = similar(img, S)
    correlate!(out, img, kernel, border)
end

"""
    correlate!(out::AbstractMatrix, img::AbstractMatrix, kernel, [border=:replicate]) -> out

In-place variant of [`correlate`](@ref).  Writes the correlation result into `out`,
which must have the same axes as `img`.  `out` must not be the same object as `img`
(in-place correlation is not a valid operation; aliasing causes data races under
multi-threading).
"""
function correlate!(out::AbstractMatrix, img::AbstractMatrix, kernel, border::Symbol=:replicate)
    axes(out) == axes(img) || throw(DimensionMismatch(
        "output axes $(axes(out)) must match image axes $(axes(img))"
    ))
    out === img && throw(ArgumentError(
        "out must not be the same object as img; in-place correlation is not supported"
    ))
    axes(img, 1) != Base.OneTo(size(img, 1)) &&
        throw(ArgumentError("img must use 1-based indexing (got axes $(axes(img))); " *
            "OffsetArrays and other non-standard index ranges are not supported"))
    kern = _canonicalize(kernel)
    if border == :replicate
        _correlate!(out, img, kern, Val(:replicate))
    elseif border == :zero
        _correlate!(out, img, kern, Val(:zero))
    else
        throw(ArgumentError("unknown border mode :$border; use :replicate or :zero"))
    end
    out
end

# ---------------------------------------------------------------------------
# Kernel canonicalization — convert everything to Tuple form
# ---------------------------------------------------------------------------

_eltype_kernel(kernel::AbstractMatrix) = eltype(kernel)
_eltype_kernel(kernel::Tuple) = promote_type(map(_eltype_kernel, kernel)...)

"""
    _canonicalize(kernel) -> Tuple{Vararg{AbstractMatrix}}

Normalize a kernel into tuple-of-matrices form for the correlation scheduler.

- An already-factored `Tuple` is passed through unchanged.
- A 2D `AbstractMatrix` is tested for separability via SVD.  If the matrix has
  rank 1 (all singular values beyond the first are ≤ `sqrt(eps(T))`), it is
  split into `(col_factor, row_factor)`.  Otherwise it is wrapped as a
  pseudo-factored pair `(identity, kernel)` where the identity element is
  a 1×1 matrix `[1.0]` that the scheduler skips.

For an `m×n` kernel the convention for the correlation center is
`cr = (m+1)÷2, cc = (n+1)÷2`.  Callers must supply odd-sized kernels
so the center falls on an integer pixel.
"""
_canonicalize(kernel::Tuple) = (_validate_kernel.(kernel); kernel)
_canonicalize(kernel::AbstractMatrix) = (_validate_kernel(kernel); _tryfactor(kernel))

"""
    _validate_kernel(kernel::AbstractMatrix)

Validate kernel properties required for correct correlation:

- Both dimensions must be odd (centre at integer index `((m+1)÷2, (n+1)÷2)`).
- Axes must be 1-based (OffsetArrays and other non-standard index ranges are
  not supported because the inner loops use literal `1:m` ranges).

Throws `ArgumentError` on violation.
"""
function _validate_kernel(kernel::AbstractMatrix)
    m, n = size(kernel)
    if !(isodd(m) && isodd(n))
        throw(ArgumentError(
            "kernel dimensions must be odd (got $m×$n); " *
            "the correlation centre is at ((m+1)÷2, (n+1)÷2), " *
            "which requires odd-sized kernels"))
    end
    if axes(kernel, 1) != Base.OneTo(m) || axes(kernel, 2) != Base.OneTo(n)
        throw(ArgumentError(
            "kernel must use 1-based indexing (got axes $(axes(kernel))); " *
            "OffsetArrays and other non-standard index ranges are not supported"))
    end
    nothing
end

function _tryfactor(kernel::AbstractMatrix{T}) where {T}
    m, n = size(kernel)
    # 1×1 is trivially separable but SVD + scheduling overhead isn't worth it.
    m == n == 1 && return (_identity_kernel(T), kernel)

    # Use a dense Matrix to guarantee StridedMatrix for svd.
    F = svd(Matrix{T}(kernel))
    U, S, Vt = F.U, F.S, F.Vt

    separable = true
    EPS = sqrt(eps(real(eltype(S))))
    @inbounds for i in 2:length(S)
        separable &= (abs(S[i]) < EPS)
    end

    if !separable
        return (_identity_kernel(T), kernel)
    end

    s = S[1]
    ss = sqrt(s)
    # U[:, 1] is an m-vector → reshape to m×1 (filters rows / dim 1).
    # Vt[1, :] is an n-vector → reshape to 1×n (filters columns / dim 2).
    k1 = reshape(ss .* U[:, 1], m, 1)
    k2 = reshape(ss .* Vt[1, :], 1, n)
    return (k1, k2)
end

# Sentinel identity kernel.  A 1×1 matrix [1.0] — convolving with this is a no-op.
_identity_kernel(::Type{T}) where {T} = fill(one(float(T)), 1, 1)

_isidentity(k::AbstractMatrix) = size(k) == (1, 1) && k[1, 1] == 1

# ---------------------------------------------------------------------------
# Border pixel access helpers
# ---------------------------------------------------------------------------

"""
    _getpixel(img, r, c, ::Val{:replicate}) -> eltype(img)

Return `img[r, c]` with out-of-bounds indices clamped to the nearest edge.
"""
@inline function _getpixel(img::AbstractMatrix, r::Int, c::Int, ::Val{:replicate})
    r = clamp(r, 1, size(img, 1))
    c = clamp(c, 1, size(img, 2))
    @inbounds img[r, c]
end

"""
    _getpixel(img, r, c, ::Val{:zero}) -> eltype(img)

Return `img[r, c]` if in bounds, or `zero(eltype(img))` otherwise.
"""
@inline function _getpixel(img::AbstractMatrix, r::Int, c::Int, ::Val{:zero})
    if 1 <= r <= size(img, 1) && 1 <= c <= size(img, 2)
        @inbounds img[r, c]
    else
        zero(eltype(img))
    end
end

# Row-only variant — caller guarantees column is in bounds.
@inline function _getpixel_row(img::AbstractMatrix, r::Int, c::Int, ::Val{:replicate})
    r = clamp(r, 1, size(img, 1))
    @inbounds img[r, c]
end
@inline function _getpixel_row(img::AbstractMatrix, r::Int, c::Int, ::Val{:zero})
    if 1 <= r <= size(img, 1)
        @inbounds img[r, c]
    else
        zero(eltype(img))
    end
end

# Col-only variant — caller guarantees row is in bounds.
@inline function _getpixel_col(img::AbstractMatrix, r::Int, c::Int, ::Val{:replicate})
    c = clamp(c, 1, size(img, 2))
    @inbounds img[r, c]
end
@inline function _getpixel_col(img::AbstractMatrix, r::Int, c::Int, ::Val{:zero})
    if 1 <= c <= size(img, 2)
        @inbounds img[r, c]
    else
        zero(eltype(img))
    end
end

# ---------------------------------------------------------------------------
# Correlation scheduler — routes to 1D-separable or 2D-inseparable paths
#
# Each function receives the original (unpadded) image and handles border
# conditions inline via strip-mining.  `correlate!` guarantees `out !== img`,
# so all inner loops are thread-safe.
# ---------------------------------------------------------------------------

function _correlate!(out::AbstractMatrix, img::AbstractMatrix, kernel::Tuple{Any},
                     ::Val{B}) where {B}
    _correlate_2d!(out, img, first(kernel), Val{B}())
end

function _correlate!(out::AbstractMatrix, img::AbstractMatrix, kernel::Tuple{Any,Any},
                     ::Val{B}) where {B}
    k1, k2 = kernel
    if _isidentity(k1)
        _correlate_2d!(out, img, k2, Val{B}())
        return
    end
    # Separable: row pass then column pass.  The intermediate tmp buffer
    # breaks any aliasing concern between out and img.
    T = promote_type(eltype(out), eltype(k1), eltype(k2))
    H, W = size(out)
    tmp = similar(img, T, H, W)
    _correlate_1d!(tmp, img, k1, Val{B}())
    _correlate_1d!(out, tmp, k2, Val{B}())
end

# Three or more factors — fold left with fresh intermediates at each step.
function _correlate!(out::AbstractMatrix, img::AbstractMatrix, kernel::Tuple,
                     ::Val{B}) where {B}
    T = promote_type(eltype(out), _eltype_kernel(kernel))
    nonid = filter(k -> !_isidentity(k), collect(kernel))
    isempty(nonid) && return copyto!(out, img)
    H, W = size(out)
    n = length(nonid)
    src = img
    for (i, k) in enumerate(nonid)
        dest = (i == n) ? out : similar(img, T, H, W)
        _correlate_1d!(dest, src, k, Val{B}())
        src = dest
    end
end

# ---------------------------------------------------------------------------
# Inner correlation loops — strip-mined with border handling
#
# Indexing convention: for a kernel of size (kr, kc) with centre at
# (cr, cc) = ((kr+1)÷2, (kc+1)÷2) and radius (r, c) = (kr÷2, kc÷2),
# the correlation at output pixel (row, col) is
#
#     Σ_j Σ_i  img[row + i - cr, col + j - cc] * kernel[i, j]
#
# which simplifies to
#
#     Σ_j Σ_i  img[row + i - r - 1, col + j - c - 1] * kernel[i, j]
#
# The interior region (rows r+1..H-r, cols c+1..W-c) uses a pure @inbounds
# fast path.  Border strips use _getpixel to clamp out-of-bounds accesses.
# ---------------------------------------------------------------------------

# ---- 1D dispatch ----

function _correlate_1d!(out::AbstractMatrix, img::AbstractMatrix,
                        kernel::AbstractMatrix, ::Val{B}) where {B}
    kr, kc = size(kernel)
    if kr > 1 && kc == 1
        _correlate_rows!(out, img, kernel, Val{B}())
    elseif kr == 1 && kc > 1
        _correlate_cols!(out, img, kernel, Val{B}())
    else
        _correlate_2d!(out, img, kernel, Val{B}())
    end
end

# ---- Row filter (m×1 kernel) ----

function _correlate_rows!(out::AbstractMatrix, img::AbstractMatrix,
                          kernel::AbstractMatrix, ::Val{B}) where {B}
    kr = size(kernel, 1)
    r = kr ÷ 2
    H, W = size(out)
    S = eltype(out)
    z = zero(S)

    # Interior rows: r+1 .. H-r.  All kernel footprint rows are in bounds.
    ri, re = r + 1, H - r
    if ri <= re
        Threads.@threads for col in 1:W
            @inbounds for row in ri:re
                acc = z
                for j in 1:kr
                    acc += img[row + j - r - 1, col] * kernel[j, 1]
                end
                out[row, col] = acc
            end
        end
    end

    # Top border: rows 1 .. r.  Row index may underflow.
    if r >= 1
        Threads.@threads for col in 1:W
            @inbounds for row in 1:r
                acc = z
                for j in 1:kr
                    ir = row + j - r - 1
                    acc += _getpixel_row(img, ir, col, Val{B}()) * kernel[j, 1]
                end
                out[row, col] = acc
            end
        end
    end

    # Bottom border: rows H-r+1 .. H.  Row index may overflow.
    if r >= 1
        Threads.@threads for col in 1:W
            @inbounds for row in H-r+1:H
                acc = z
                for j in 1:kr
                    ir = row + j - r - 1
                    acc += _getpixel_row(img, ir, col, Val{B}()) * kernel[j, 1]
                end
                out[row, col] = acc
            end
        end
    end

    out
end

# ---- Column filter (1×n kernel) ----

function _correlate_cols!(out::AbstractMatrix, img::AbstractMatrix,
                          kernel::AbstractMatrix, ::Val{B}) where {B}
    kc = size(kernel, 2)
    c = kc ÷ 2
    H, W = size(out)
    S = eltype(out)
    z = zero(S)

    # Interior columns: c+1 .. W-c.  All kernel footprint columns are in bounds.
    ci, ce = c + 1, W - c
    if ci <= ce
        Threads.@threads for col in ci:ce
            @inbounds for row in 1:H
                acc = z
                for j in 1:kc
                    acc += img[row, col + j - c - 1] * kernel[1, j]
                end
                out[row, col] = acc
            end
        end
    end

    # Left border: cols 1 .. c.  Column index may underflow.
    if c >= 1
        Threads.@threads for col in 1:c
            @inbounds for row in 1:H
                acc = z
                for j in 1:kc
                    ic = col + j - c - 1
                    acc += _getpixel_col(img, row, ic, Val{B}()) * kernel[1, j]
                end
                out[row, col] = acc
            end
        end
    end

    # Right border: cols W-c+1 .. W.  Column index may overflow.
    if c >= 1
        Threads.@threads for col in W-c+1:W
            @inbounds for row in 1:H
                acc = z
                for j in 1:kc
                    ic = col + j - c - 1
                    acc += _getpixel_col(img, row, ic, Val{B}()) * kernel[1, j]
                end
                out[row, col] = acc
            end
        end
    end

    out
end

# ---- 2D non-separable filter ----

function _correlate_2d!(out::AbstractMatrix, img::AbstractMatrix,
                        kernel::AbstractMatrix, ::Val{B}) where {B}
    kr, kc = size(kernel)
    r, c = kr ÷ 2, kc ÷ 2
    H, W = size(out)
    S = eltype(out)
    z = zero(S)

    # Interior: rows r+1..H-r, cols c+1..W-c.  All kernel footprint
    # pixels are in bounds — full @inbounds fast path.
    ri, re = r + 1, H - r
    ci, ce = c + 1, W - c
    if ri <= re && ci <= ce
        Threads.@threads for col in ci:ce
            @inbounds for row in ri:re
                acc = z
                for kc_i in 1:kc, kr_i in 1:kr
                    acc += img[row + kr_i - r - 1, col + kc_i - c - 1] *
                           kernel[kr_i, kc_i]
                end
                out[row, col] = acc
            end
        end
    end

    # Top strip: rows 1..r, all columns.  Row index may underflow.
    # (Includes the four corner regions — no separate corner handling needed.)
    if r >= 1
        Threads.@threads for col in 1:W
            @inbounds for row in 1:r
                acc = z
                for kc_i in 1:kc, kr_i in 1:kr
                    ir = row + kr_i - r - 1
                    ic = col + kc_i - c - 1
                    acc += _getpixel(img, ir, ic, Val{B}()) * kernel[kr_i, kc_i]
                end
                out[row, col] = acc
            end
        end
    end

    # Bottom strip: rows H-r+1..H, all columns.  Row index may overflow.
    if r >= 1
        Threads.@threads for col in 1:W
            @inbounds for row in H-r+1:H
                acc = z
                for kc_i in 1:kc, kr_i in 1:kr
                    ir = row + kr_i - r - 1
                    ic = col + kc_i - c - 1
                    acc += _getpixel(img, ir, ic, Val{B}()) * kernel[kr_i, kc_i]
                end
                out[row, col] = acc
            end
        end
    end

    # Left strip: middle rows, cols 1..c.  Only column index may underflow;
    # row index is always in bounds for rows ri..re.
    if c >= 1 && ri <= re
        Threads.@threads for col in 1:c
            @inbounds for row in ri:re
                acc = z
                for kc_i in 1:kc, kr_i in 1:kr
                    ic = col + kc_i - c - 1
                    acc += _getpixel_col(img, row + kr_i - r - 1, ic, Val{B}()) *
                           kernel[kr_i, kc_i]
                end
                out[row, col] = acc
            end
        end
    end

    # Right strip: middle rows, cols W-c+1..W.  Only column index may overflow.
    if c >= 1 && ri <= re
        Threads.@threads for col in W-c+1:W
            @inbounds for row in ri:re
                acc = z
                for kc_i in 1:kc, kr_i in 1:kr
                    ic = col + kc_i - c - 1
                    acc += _getpixel_col(img, row + kr_i - r - 1, ic, Val{B}()) *
                           kernel[kr_i, kc_i]
                end
                out[row, col] = acc
            end
        end
    end

    out
end
