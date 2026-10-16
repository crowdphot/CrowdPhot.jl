"""
    PolyMap(to_sky, dims, frame::TanGrid; degree = 5, max_degree = 9, n = 41, margin = 40, tol = 1e-4)

Bivariate polynomial map from fit-frame position `(u, v)` to image pixel `(y, x)`, fitted once
per image so that kernels can place a source and get its Jacobian `d(y, x)/d(u, v)` without
evaluating the WCS.  `to_sky(y, x) -> (ra, dec)` (degrees, 1-based pixel centers) is the only
thing needed from the image's WCS; `dims = (ny, nx)`.

The fit is least squares on an `n x n` lattice spanning the image plus `margin` pixels (the
model wings of sources just off the image), in monomials of the normalized `((u - u0) / s,
(v - v0) / s)`, ordered `(i, j)` for `i in 0:degree`, `j in 0:degree-i`.  It is validated on
the staggered lattice.  If the maximum error exceeds `tol` pixels the degree is raised by 2
(with a warning) up to `max_degree`; construction throws if that is not enough either.
"""
struct PolyMap
    degree::Int
    u0::Float64
    v0::Float64
    s::Float64
    cy::Vector{Float64}
    cx::Vector{Float64}
    bbox::NTuple{4, Float64}   # (umin, umax, vmin, vmax) of the fitted lattice
    maxerr::Float64
end

nmonomials(d) = (d + 1) * (d + 2) ÷ 2

# Degree-`d` coefficients `c` in the degree-`D` monomial order (zero above `d`), so one device
# degree serves images whose maps escalated to different degrees.
function _lift_coefs(c, d, D)
    out = zeros(nmonomials(D)); k = kd = 0
    for i in 0:D, j in 0:(D - i)
        k += 1
        i + j <= d && (out[k] = c[kd += 1])
    end
    return out
end

function _poly_design(d, u0, v0, s, us, vs)
    A = Matrix{Float64}(undef, length(us), nmonomials(d))
    for k in eachindex(us)
        a = (us[k] - u0) / s; b = (vs[k] - v0) / s
        m = 0
        for i in 0:d, j in 0:(d - i)
            A[k, m += 1] = a^i * b^j
        end
    end
    return A
end

function PolyMap(to_sky, dims::NTuple{2, Integer}, frame::TanGrid; degree::Integer = 5, max_degree::Integer = 9, n::Integer = 41,
        margin::Real = 40, tol::Real = 1e-4)
    ny, nx = dims
    # `stagger` shifts the lattice by half a step on both axes: the validation points between the fitting nodes
    function lattice(m, stagger)
        oy = stagger ? (ny + 2margin) / 2m : 0.0; ox = stagger ? (nx + 2margin) / 2m : 0.0
        return [(y, x) for y in range(1 - margin + oy, ny + margin - oy; length = m) for x in range(1 - margin + ox, nx + margin - ox; length = m)]
    end
    to_uv(p) = world_to_grid(frame, to_sky(p...)...)
    P = lattice(n, false)
    uv = map(to_uv, P)
    us = first.(uv); vs = last.(uv)
    all(isfinite, us) && all(isfinite, vs) || throw(ArgumentError("`to_sky` returned non-finite coordinates on the image lattice"))
    u0 = sum(us) / length(us); v0 = sum(vs) / length(vs)
    s = max(maximum(abs, us .- u0), maximum(abs, vs .- v0))
    Q = lattice(n - 1, true)
    uq = map(to_uv, Q)
    nmonomials(max_degree) <= n^2 || throw(ArgumentError("max_degree = $max_degree needs more than the $(n^2) lattice points; raise `n`"))
    err = Inf
    for d in degree:2:max_degree
        A = _poly_design(d, u0, v0, s, us, vs)
        cy = A \ first.(P); cx = A \ last.(P)
        Aq = _poly_design(d, u0, v0, s, first.(uq), last.(uq))
        err = max(maximum(abs, Aq * cy .- first.(Q)), maximum(abs, Aq * cx .- last.(Q)))
        err <= tol && return PolyMap(d, u0, v0, s, cy, cx, (minimum(us), maximum(us), minimum(vs), maximum(vs)), err)
        d + 2 <= max_degree && @warn "PolyMap: validation error $(round(err; sigdigits = 3)) px exceeds tol = $tol px at degree $d; retrying at degree $(d + 2)"
    end
    throw(ArgumentError("PolyMap: validation error $err px exceeds tol = $tol px up to max_degree = $max_degree; check `to_sky`, " *
                        "or split the image into smaller cutouts"))
end

"""
    poly_eval(cy, cx, e, d, a, b) -> (y, x, dy/da, dy/db, dx/da, dx/db)

Degree-`d` polynomial of column `e` of the coefficient matrices at normalized `(a, b)`, with
its derivatives.  Powers are built incrementally (no `^`) so the same code runs in kernels.
"""
@inline function poly_eval(cy, cx, e, d, a, b)
    T = typeof(a)
    y = x = ya = yb = xa = xb = zero(T)
    k = 0
    ai = one(T); dai = zero(T)            # a^i and i a^(i-1)
    @inbounds for i in 0:d
        bj = one(T); dbj = zero(T)        # b^j and j b^(j-1)
        for j in 0:(d - i)
            k += 1
            cyk = cy[k, e]; cxk = cx[k, e]
            t = ai * bj; ta = dai * bj; tb = ai * dbj
            y += cyk * t; x += cxk * t
            ya += cyk * ta; yb += cyk * tb
            xa += cxk * ta; xb += cxk * tb
            dbj = T(j + 1) * bj; bj *= b
        end
        dai = T(i + 1) * ai; ai *= a
    end
    return y, x, ya, yb, xa, xb
end

"Pixel position and Jacobian `[dy/du dy/dv; dx/du dx/dv]` of frame position `(u, v)` (host)."
function (pm::PolyMap)(u, v)
    a = (u - pm.u0) / pm.s; b = (v - pm.v0) / pm.s
    y, x, ya, yb, xa, xb = poly_eval(reshape(pm.cy, :, 1), reshape(pm.cx, :, 1), 1, pm.degree, a, b)
    is = 1 / pm.s
    return y, x, SMatrix{2, 2, Float64, 4}(ya * is, xa * is, yb * is, xb * is)
end
