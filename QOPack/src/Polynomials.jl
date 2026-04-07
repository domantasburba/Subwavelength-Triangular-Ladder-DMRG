#=
Polynomials of degree n are represented as arrays of length (n+1) where the
(i+1)-th element is the coefficient of x^i.

E.g.:
f(x) = 1 + 3x^2 - πx^3 -> f_coeffs = [1, 0, 3, -π]
=#

function eval_poly(x, polynomial_coeffs)
    # y = big(zero(x))
    y = zero(x)
    for (idx, c) in enumerate(polynomial_coeffs)
        n = idx-1

        y += c * x^n
    end # for

    return y
end # function

function eval_poly_xs(xs, polynomial_coeffs)
    # ys = zeros(BigFloat, length(xs))
    ys = zeros(eltype(xs), length(xs))
    for (i, x) in enumerate(xs)
        ys[i] = eval_poly(x, polynomial_coeffs)
    end # for

    return ys
end # function

function eval_poly_Horner(x, polynomial_coeffs)
    if length(polynomial_coeffs) == 0
        return zero(x)
    end # if
    if length(polynomial_coeffs) == 1
        return polynomial_coeffs[1]
    end # if

    y = polynomial_coeffs[end-1] + polynomial_coeffs[end]*big(x)
    for i in (length(polynomial_coeffs)-2):(-1):1
        y = polynomial_coeffs[i] + x*y
    end # for

    return y
end # function

function add_polynomial_coeffs(p1_coeffs, p2_coeffs)
    p_length = max(length(p1_coeffs), length(p2_coeffs))
    p_coeffs = zeros(eltype(p1_coeffs), p_length)
    p_coeffs[1:length(p1_coeffs)] += p1_coeffs[:]
    p_coeffs[1:length(p2_coeffs)] += p2_coeffs[:]

    return p_coeffs
end # function

"""Multiply polynomial by x^p."""
function mul_xp_polynomial_coeffs(p, polynomial_coeffs)
    mul_polynomial_coeffs = zeros(eltype(polynomial_coeffs), length(polynomial_coeffs)+p)
    mul_polynomial_coeffs[(1+p):end] = polynomial_coeffs[:]

    return mul_polynomial_coeffs
end # function

"""Derivative of polynomial."""
function Dx_polynomial_coeffs(polynomial_coeffs)
    if length(polynomial_coeffs) ≤ 1
        return []
    end # if

    der_polynomial_coeffs = zeros(eltype(polynomial_coeffs), length(polynomial_coeffs)-1)
    for i in 2:length(polynomial_coeffs)
        der_polynomial_coeffs[i-1] = (i-1)*polynomial_coeffs[i]
    end # for

    return der_polynomial_coeffs
end # function

"""Returns vector (length max_order) where the (n+1)-th element is the n-th order
Hermite polynomial, represented in the usual way."""
function calc_Hermite_polynomial_coeffs_vec(max_order)
    Hermite_vec = [[1]]
    # Hermite_vec = [[big(1)]]
    for n in 1:(max_order-1)
        Hnm1_coeffs = Hermite_vec[end]

        p1_coeffs = 2 .* mul_xp_polynomial_coeffs(1, Hnm1_coeffs)
        p2_coeffs = -1 .* Dx_polynomial_coeffs(Hnm1_coeffs)
        Hn_coeffs = add_polynomial_coeffs(p1_coeffs, p2_coeffs)

        push!(Hermite_vec, Hn_coeffs)
    end # for

    return Hermite_vec
end # function
