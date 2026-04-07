# module Math

using EllipsisNotation
using LinearAlgebra
using Printf
using FFTW

# export Pauli_x, Pauli_y, Pauli_z, x2index

function ⊗(A, B)
    return Base.kron(A, B)
end # function

function sgn(x)
    if x > 0
        return one(x)
    elseif x == 0
        return zero(x)
    else
        return -one(x)
    end # if
end # function

function Kronecker_δ(i, j)
    return Int(i == j)
end # function

function Levi_Civita_ϵ(i, j)
    if (i == 1) && (j == 2)
        return 1
    elseif (i == 2) && (j == 1)
        return -1
    else
        return 0
    end # if
end # function

function Levi_Civita_ϵ(i, j, k)
    if (i, j, k) == (1, 2, 3) || (i, j, k) == (2, 3, 1) || (i, j, k) == (3, 1, 2)
        return 1
    elseif (i, j, k) == (3, 2, 1) || (i, j, k) == (1, 3, 2) || (i, j, k) == (2, 1, 3)
        return -1
    else
        return 0
    end # if
end # function

function inBounds(x, x_span)
    return (x_span[1] <= x <= x_span[2])
end # function

function inBounds(r, x_span, y_span)
    return (inBounds(r[1], x_span) && inBounds(r[2], y_span))
end # function

function avg(arr)
    return sum(arr)[1] / length(arr)
end # function

function std_dev(arr)
    arr_avg = avg(arr)

    return sqrt(sum((arr .- arr_avg).^2)[1] / (length(arr) - 1))
end # function

function hsum(H)
    return sum(abs.(H - adjoint(H)))[1]
end # function

function meshgrid(xs, ys)
    # Taken from Chris Rackauskas's answer in https://discourse.julialang.org/t/meshgrid-function-in-julia/48679/4
    Xs = xs' .* ones(length(ys))
    Ys = ones(length(xs))' .* ys

    return Xs, Ys
end # function

function sqrt_Gaussian(x, μ, σ)
    return exp(-0.25 * ((x - μ) / σ)^2) / (sqrt(σ) * (2π)^(1/4))
end # function

function Gaussian(x, μ, σ)
    return exp(-0.5 * ((x - μ) / σ)^2) / (σ * sqrt(2π))
end # function

function comm(A, B)
    return A*B - B*A
end # function

function anticomm(A, B)
    return A*B + B*A
end # function

function id(N::Integer)
    id = zeros(ComplexF64, N, N)
    for i in 1:N
        id[i, i] = 1.0
    end # for

    return id
end # function

function add_ijv!(i, j, v, rows, cols, data)
    append!(rows, i)
    append!(cols, j)
    append!(data, v)
end # function

# function index_dict(arr)
#     d = Dict{eltype(arr), Vector{Int}}()
#     for (i, x) in pairs(arr)
#         if haskey(d, x)
#             push!(d[x], i)
#         else
#             d[x] = [i]
#         end # if
#     end # for

#     return d
# end # function

function Gram_Schmidt(v)
    @assert length(size(v)) == 2

    u = zero(v)
    for i in 1:size(v)[2]
        u_i = v[:, i]
        for j in 1:(i-1)
            u_i = u_i - ((u[:, j])' * v[:, i]) * u[:, j]
        end # for

        u_i = u_i / sqrt(u_i' * u_i)

        u[:, i] = u_i
    end # for

    return u
end # function

function invert_dict(dict)
    return Dict(el[2]=>el[1] for el in dict)
end # function

function sum_non_diag(A)
    return sum(A) - tr(A)
end # function

"""Assumes A is block diagonal."""
function split_block_diag(A)
    block_vec = Vector{typeof(A)}([])

    N = size(A, 1)
    @assert size(A, 1) == size(A, 2)

    i = 1
    while i <= N
        row = A[i, i:end]
        col = A[i:end, i]
        max_row_idx = findlast(x -> x!=0, row)
        max_col_idx = findlast(x -> x!=0, col)
        if max_row_idx === nothing
            max_row_idx = i
        end # if
        if max_col_idx === nothing
            max_col_idx = i
        end # if
        max_row_idx += i-1
        max_col_idx += i-1
        max_idx = max(max_row_idx, max_col_idx)

        push!(block_vec, A[i:max_idx, i:max_idx])

        i = max_idx + 1
    end # while

    return block_vec
end # function

function find_split_idxs(v)
    current_val = v[1]
    split_idxs = [1]
    for (i, v_i) in enumerate(v)
        if !(v_i ≈ current_val)
            current_val = v_i
            push!(split_idxs, i-1)
        end # if
    end # for
    push!(split_idxs, length(v))

    return split_idxs
end # function

# TODO: Add sparse option
"""Options: "Dirichlet", "Neumann", "Periodic"."""
function calc_Dx(dim::Integer, Δx::Real; BC="Dirichlet")
    half_inv_Δx = 0.5/Δx
    Dx = diagm(1 => fill(half_inv_Δx, dim-1), -1 => fill(-half_inv_Δx, dim-1))

    if BC == "Dirichlet"
        return Dx
    elseif BC == "Neumann"
        Dx[1, :] .= zero(half_inv_Δx)
        Dx[end, :] .= zero(half_inv_Δx)
        return Dx
    elseif BC == "Periodic"
        Dx[1, end] = -half_inv_Δx
        Dx[end, 1] = half_inv_Δx
        return Dx
    else
        throw(ArgumentError("Invalid boundary condition (BC). Please check available options in documentation."))
    end # if
end # function

"""Options: "Dirichlet", "Neumann", "Periodic"."""
function calc_D2x(dim::Integer, Δx::Real; BC="Dirichlet")
    inv_sq_Δx = 1.0/(Δx)^2
    off_diag = fill(inv_sq_Δx, dim-1)
    main_diag = fill(-2.0*inv_sq_Δx, dim)
    D2x = diagm(1 => off_diag, 0 => main_diag, -1 => off_diag)

    if BC == "Dirichlet"
        return D2x
    elseif BC == "Neumann"
        D2x[1, 1] = -inv_sq_Δx
        D2x[end, end] = -inv_sq_Δx
        return D2x
    elseif BC == "Periodic"
        D2x[1, end] = inv_sq_Δx
        D2x[end, 1] = inv_sq_Δx
        return D2x
    else
        throw(ArgumentError("Invalid boundary condition (BC). Please check available options in documentation."))
    end # if
end # function

# function σx()
#     return Pauli_x()
# end # function

# function σy()
#     return Pauli_y()
# end # function

# function σz()
#     return Pauli_z()
# end # function

function Pauli_x()
    return [0. 1.;
            1. 0.]
end # function

function Pauli_y()
    return [0. -im;
            im 0.]
end # function

function Pauli_z()
    return [1. 0.;
            0. -1.]
end # function

"""Convention: 0 - identity; 1 - Pauli_x; 2 - Pauli_y; 3 - Pauli_z;"""
function Pauli_i(i::Integer)
    if i == 0
        return id(2)
    elseif i == 1
        return Pauli_x()
    elseif i == 2
        return Pauli_y()
    elseif i == 3
        return Pauli_z()
    else
        throw(ArgumentError("Invalid i for Pauli_i matrix. Please use either 0,1,2,3 or x,y,z."))
    end # if
end # function

function Pauli_i(i::Char)
    if i == 'x'
        return Pauli_x()
    elseif i == 'y'
        return Pauli_y()
    elseif i == 'z'
        return Pauli_z()
    else
        throw(ArgumentError("Invalid i for Pauli_i matrix. Please use either 0,1,2,3 or x,y,z."))
    end # if
end # function

function Pauli_i(i::String)
    if i == "x"
        return Pauli_x()
    elseif i == "y"
        return Pauli_y()
    elseif i == "z"
        return Pauli_z()
    else
        throw(ArgumentError("Invalid i for Pauli_i matrix. Please use either 0,1,2,3 or x,y,z."))
    end # if
end # function

function create_S_plus(S)
    @assert isinteger(2S)  # True for both half-spin and integer spin.
    N_m = round(Int, 2S + 1)
    off_diag = Array{ComplexF64}(undef, N_m-1)
    for idx_m in 1:(N_m-1)
        m = S - idx_m + 1
        off_diag[idx_m] = sqrt(S*(S+1) - (m-1)*m)
    end # for

    return diagm(1 => off_diag)
end # function

function create_S_minus(S)
    @assert isinteger(2S)  # True for both half-spin and integer spin.
    return transpose(create_S_plus(S))
end # function

function create_S_x(S)
    @assert isinteger(2S)  # True for both half-spin and integer spin.
    return 1/2 .* (create_S_plus(S) .+ create_S_minus(S))
end # function

function create_S_y(S)
    @assert isinteger(2S)  # True for both half-spin and integer spin.
    return 1/(2im) .* (create_S_plus(S) .- create_S_minus(S))
end # function

function create_S_z(S)
    @assert isinteger(2S)  # True for both half-spin and integer spin.
    N_m = round(Int, 2S + 1)
    S_z = zeros(ComplexF64, N_m, N_m)
    for idx_m in 1:N_m
        m = S - idx_m + 1
        S_z[idx_m, idx_m] = m
    end # for

    return S_z
end # function

function create_S_squared(S)
    @assert isinteger(2S)  # True for both half-spin and integer spin.
    N_m = round(Int, 2S + 1)
    return S*(S+1) .* id(N_m)
end # function

function calc_orthogonal_vectors_3D(v)
    v_norm = v ./ sqrt(dot(v, v))

    x = cross(v_norm, [1, 0, 0])
    if dot(x, x) < 1e-6
        x = cross(v_norm, [0, 1, 0])
    end # if
    x = x ./ sqrt(dot(x, x))

    y = cross(v_norm, x)
    y = y ./ sqrt(dot(y, y))

    return x, y
end # function

function normalize_state(state)
    return (1.0/sqrt(state'*state)) .* state
end # function

# function normalize_wf!(ψ, Δx)
#     d
# end # function

"""
Finite difference derivative.
Assumes 1D y_arr and equidistant spacing.
Special case of np.gradient"""
function dx_y_FD(y_arr, Δx; order=6)
    N_y = length(y_arr)
    double_Δx = 2*Δx

    if order == 2
        ##### SECOND ORDER ACCURACY #####
        dx_y_arr = Vector{typeof(y_arr[1])}(undef, N_y)
        for idx_y in 2:(N_y-1)
            # Central difference
            dx_y_arr[idx_y] = (y_arr[idx_y+1] - y_arr[idx_y-1]) / double_Δx
        end # for

        ### EDGE CASES ###
        # Forward difference
        # dx_y_arr[1] = (y_arr[2] - y_arr[1]) / Δx
        dx_y_arr[1] = (-1/2*y_arr[3] + 2*y_arr[2] - 3/2*y_arr[1]) / Δx

        # Backward difference
        # dx_y_arr[N_y] = (y_arr[N_y] - y_arr[N_y-1]) / Δx
        dx_y_arr[N_y] = -(-1/2*y_arr[N_y-2] + 2*y_arr[N_y-1] - 3/2*y_arr[N_y]) / Δx
        ### END OF EDGE CASES ###
        ##### END OF SECOND ORDER ACCURACY #####
    elseif order == 4
        ##### FOURTH ORDER ACCURACY #####
        dx_y_arr = Vector{typeof(y_arr[1])}(undef, N_y)
        for idx_y in 3:(N_y-2)
            # Central difference
            dx_y_arr[idx_y] = (-1/12*y_arr[idx_y+2] + 2/3*y_arr[idx_y+1] - 2/3*y_arr[idx_y-1] + 1/12*y_arr[idx_y-2]) / Δx
        end # for

        ### EDGE CASES ###
        # Forward difference
        dx_y_arr[1] = (-1/4*y_arr[5] + 4/3*y_arr[4] - 3*y_arr[3] + 4*y_arr[2] - 25/12*y_arr[1]) / Δx
        dx_y_arr[2] = (-1/4*y_arr[6] + 4/3*y_arr[5] - 3*y_arr[4] + 4*y_arr[3] - 25/12*y_arr[2]) / Δx

        # Backward difference
        dx_y_arr[N_y] = -(-1/4*y_arr[N_y-4] + 4/3*y_arr[N_y-3] - 3*y_arr[N_y-2] + 4*y_arr[N_y-1] - 25/12*y_arr[N_y]) / Δx
        dx_y_arr[N_y-1] = -(-1/4*y_arr[N_y-5] + 4/3*y_arr[N_y-4] - 3*y_arr[N_y-3] + 4*y_arr[N_y-2] - 25/12*y_arr[N_y-1]) / Δx
        ### END OF EDGE CASES ###
        ##### END OF FOURTH ORDER ACCURACY #####
    elseif order == 6
        ##### SIXTH ORDER ACCURACY #####
        dx_y_arr = Vector{typeof(y_arr[1])}(undef, N_y)
        for idx_y in 4:(N_y-3)
            # Central difference
            dx_y_arr[idx_y] = (1/60*y_arr[idx_y+3] - 3/20*y_arr[idx_y+2] + 3/4*y_arr[idx_y+1] - 3/4*y_arr[idx_y-1] + 3/20*y_arr[idx_y-2] - 1/60*y_arr[idx_y-3]) / Δx
        end # for

        ### EDGE CASES ###
        # Forward difference
        dx_y_arr[1] = (-1/6*y_arr[7] + 6/5*y_arr[6] - 15/4*y_arr[5] + 20/3*y_arr[4] - 15/2*y_arr[3] + 6*y_arr[2] - 49/20*y_arr[1]) / Δx
        dx_y_arr[2] = (-1/6*y_arr[8] + 6/5*y_arr[7] - 15/4*y_arr[6] + 20/3*y_arr[5] - 15/2*y_arr[4] + 6*y_arr[3] - 49/20*y_arr[2]) / Δx
        dx_y_arr[3] = (-1/6*y_arr[9] + 6/5*y_arr[8] - 15/4*y_arr[7] + 20/3*y_arr[6] - 15/2*y_arr[5] + 6*y_arr[4] - 49/20*y_arr[3]) / Δx

        # Backward difference
        dx_y_arr[N_y] = -(-1/6*y_arr[N_y-6] + 6/5*y_arr[N_y-5] - 15/4*y_arr[N_y-4] + 20/3*y_arr[N_y-3] - 15/2*y_arr[N_y-2] + 6*y_arr[N_y-1] - 49/20*y_arr[N_y]) / Δx
        dx_y_arr[N_y-1] = -(-1/6*y_arr[N_y-7] + 6/5*y_arr[N_y-6] - 15/4*y_arr[N_y-5] + 20/3*y_arr[N_y-4] - 15/2*y_arr[N_y-3] + 6*y_arr[N_y-2] - 49/20*y_arr[N_y-1]) / Δx
        dx_y_arr[N_y-2] = -(-1/6*y_arr[N_y-8] + 6/5*y_arr[N_y-7] - 15/4*y_arr[N_y-6] + 20/3*y_arr[N_y-5] - 15/2*y_arr[N_y-4] + 6*y_arr[N_y-3] - 49/20*y_arr[N_y-2]) / Δx
        ### END OF EDGE CASES ###
        ##### END OF SIXTH ORDER ACCURACY #####
    end # if

    return dx_y_arr
end # function

function d2x_y_FD(y_arr, Δx; order=6)
    N_y = length(y_arr)
    square_Δx = Δx^2

    if order == 2
        ##### SECOND ORDER ACCURACY #####
        d2x_y_arr = Vector{typeof(y_arr[1])}(undef, N_y)
        for idx_y in 2:(N_y-1)
            # Central difference
            d2x_y_arr[idx_y] = (y_arr[idx_y+1] - 2*y_arr[idx_y] + y_arr[idx_y-1]) / square_Δx
        end # for

        ### EDGE CASES ###
        # Forward difference
        # d2x_y_arr[1] = (y_arr[3] - 2*y_arr[2] + y_arr[1]) / square_Δx
        d2x_y_arr[1] = (-1*y_arr[4] + 4*y_arr[3] - 5*y_arr[2] + 2*y_arr[1]) / square_Δx

        # Backward difference
        # d2x_y_arr[N_y] = (y_arr[N_y] - 2*y_arr[N_y-1] + y_arr[N_y-2]) / square_Δx
        d2x_y_arr[N_y] = (-1*y_arr[N_y-3] + 4*y_arr[N_y-2] - 5*y_arr[N_y-1] + 2*y_arr[N_y]) / square_Δx
        ### END OF EDGE CASES ###
        ##### END OF SECOND ORDER ACCURACY #####
    elseif order == 4
        ##### FOURTH ORDER ACCURACY #####
        d2x_y_arr = Vector{typeof(y_arr[1])}(undef, N_y)
        for idx_y in 3:(N_y-2)
            # Central difference
            d2x_y_arr[idx_y] = (-1/12*y_arr[idx_y+2] + 4/3*y_arr[idx_y+1] - 5/2*y_arr[idx_y] + 4/3*y_arr[idx_y-1] - 1/12*y_arr[idx_y-2]) / square_Δx
        end # for

        ### EDGE CASES ###
        # Forward difference
        d2x_y_arr[1] = (-5/6*y_arr[6] + 61/12*y_arr[5] - 13*y_arr[4] + 107/6*y_arr[3] - 77/6*y_arr[2] + 15/4*y_arr[1]) / square_Δx
        d2x_y_arr[2] = (-5/6*y_arr[7] + 61/12*y_arr[6] - 13*y_arr[5] + 107/6*y_arr[4] - 77/6*y_arr[3] + 15/4*y_arr[2]) / square_Δx

        # Backward difference
        d2x_y_arr[N_y] = (-5/6*y_arr[N_y-5] + 61/12*y_arr[N_y-4] - 13*y_arr[N_y-3] + 107/6*y_arr[N_y-2] - 77/6*y_arr[N_y-1] + 15/4*y_arr[N_y]) / square_Δx
        d2x_y_arr[N_y-1] = (-5/6*y_arr[N_y-6] + 61/12*y_arr[N_y-5] - 13*y_arr[N_y-4] + 107/6*y_arr[N_y-3] - 77/6*y_arr[N_y-2] + 15/4*y_arr[N_y-1]) / square_Δx
        ### END OF EDGE CASES ###
        ##### END OF FOURTH ORDER ACCURACY #####
    elseif order == 6
        ##### SIXTH ORDER ACCURACY #####
        d2x_y_arr = Vector{typeof(y_arr[1])}(undef, N_y)
        for idx_y in 4:(N_y-3)
            # Central difference
            d2x_y_arr[idx_y] = (1/90*y_arr[idx_y+3] - 3/20*y_arr[idx_y+2] + 3/2*y_arr[idx_y+1] - 49/18*y_arr[idx_y] + 3/2*y_arr[idx_y-1] - 3/20*y_arr[idx_y-2] + 1/90*y_arr[idx_y-3]) / square_Δx
        end # for

        ### EDGE CASES ###
        # Forward difference
        for idx_y in 1:3
            d2x_y_arr[idx_y] = (-7/10*y_arr[idx_y+7] + 1019/180*y_arr[idx_y+6] - 201/10*y_arr[idx_y+5] + 41*y_arr[idx_y+4] - 949/18*y_arr[idx_y+3] + 879/20*y_arr[idx_y+2] - 223/10*y_arr[idx_y+1] + 469/90*y_arr[idx_y]) / square_Δx
        end # for

        # Backward difference
        for idx_y in (N_y-2):N_y
            d2x_y_arr[idx_y] = (-7/10*y_arr[idx_y-7] + 1019/180*y_arr[idx_y-6] - 201/10*y_arr[idx_y-5] + 41*y_arr[idx_y-4] - 949/18*y_arr[idx_y-3] + 879/20*y_arr[idx_y-2] - 223/10*y_arr[idx_y-1] + 469/90*y_arr[idx_y]) / square_Δx
        end # for
        ### END OF EDGE CASES ###
        ##### END OF SIXTH ORDER ACCURACY #####
    end # if

    # ##### SPECTRAL ACCURACY #####
    # k_arr = 2*π .* fftfreq(N_y, 1/Δx)

    # y_bar = fft(y_arr)
    # d2x_y_bar = -k_arr.^2 .* y_bar
    # d2x_y_arr = real.(ifft(d2x_y_bar))
    # ##### END OF SPECTRAL ACCURACY #####

    return d2x_y_arr
end # function

function dx_y_FFT(y_arr, Δx)
    k = 2π .* fftfreq(length(y_arr), 1/Δx)
    y_bar = fft(y_arr)
    dx_y_bar = im.*k.*y_bar
    dx_y_arr = ifft(dx_y_bar)

    return dx_y_arr
end # function

function d2x_y_FFT(y_arr, Δx)
    k = 2π .* fftfreq(length(y_arr), 1/Δx)
    y_bar = fft(y_arr)
    d2x_y_bar = -(k.^2).*y_bar
    d2x_y_arr = ifft(d2x_y_bar)

    return d2x_y_arr
end # function

"""Recreation of MATLAB toeplitz function. Quoting MATLAB documentation: "T =
toeplitz(c,r) returns a nonsymmetric Toeplitz matrix with c as its first column
and r as its first row. If the first elements of c and r differ, toeplitz issues
a warning and uses the column element for the diagonal." """
function toeplitz(c, r)
    @assert length(size(c)) == length(size(r)) == 1
    @assert eltype(c) == eltype(r)
    @assert c[1] ≈ r[1]
    @assert length(c) == length(r)  # Function could be more general and also allow non-square matrices.

    N = length(c)
    mat = zeros(eltype(c), N, N)
    for (i, c_i) in enumerate(c)
        mat += diagm(-(i-1) => repeat([c_i], N-i+1))
    end # for
    for (i, r_i) in enumerate(r[2:end])
        mat += diagm(i => repeat([r_i], N-i))
    end # for

    return mat
end # function

function toeplitz(r)
    return toeplitz(r, r)
end # function

"""Differentiation matrix for regular, periodic grid. This grid does not include
same point twice, it excludes first point, e.g., takes 2π, but does not take 0.
See Trefethen's "Spectral Methods in MATLAB", pg. 21, Eq. (3.10)."""
function create_periodic_diff_matrix(x_vec)
    N = length(x_vec)
    Δx = x_vec[2] - x_vec[1]
    # Δx = (x_vec[end] - x_vec[1]) / (N-1)

    cot_col = [0.0; 0.5*(-1).^(1:(N-1)).*cot.((1:N-1)*Δx/2)]
    cot_row = [0.0; 0.5*(-1).^((N-1):-1:1).*cot.(((N-1):-1:1)*Δx/2)]
    D = toeplitz(cot_col, cot_row)

    return D
end # function

"""Differentiation matrix for regular, periodic grid. This grid does not include
same point twice, it excludes first point, e.g., takes 2π, but does not take 0.
See Trefethen's "Spectral Methods in MATLAB", pg. 23, Eq. (3.12)."""
function create_periodic_diff2_matrix(x_vec)
    N = length(x_vec)
    Δx = x_vec[2] - x_vec[1]

    csc_col = [-π^2/(3*Δx^2) - 1.0/6; -0.5*(-1).^(1:(N-1))./(sin.((1:N-1)*Δx/2).^2)]
    D2 = toeplitz(csc_col)

    return D2
end # function

function create_x_Cheb(L_x, N_x)
    x_cheb = cos.(π/N_x .* collect(0:N_x)) .* 0.5*L_x

    return x_cheb
end # function

"""Adapted from Trefethen's "Spectral Methods in MATLAB", page 53, 54.  Note
that the grid points are the Chebyshev points - they are NOT equidistantly
spaced."""
function create_Cheb_diff_matrix(x_cheb)
    @assert x_cheb[1] ≈ -x_cheb[end]

    x_max = 0.5 * abs(x_cheb[end] - x_cheb[1])
    y_cheb = @. x_cheb / x_max

    N_x = length(x_cheb)
    M = N_x-1

    D = zeros(ComplexF64, N_x, N_x)

    ### Edge elements
    D[1, 1] = (2.0*(M^2) + 1.0) / 6.0
    D[end, end] = -D[1, 1]
    D[1, end] = 0.5 * (-1)^M
    D[end, 1] = -D[1, end]

    ### Diagonal elements
    for i in 2:M
    # Threads.@threads for i in 2:M
        y = y_cheb[i]

        D[i, i] = -0.5 * y / (1.0 - y^2)
    end # for

    ### Outer elements
    for i in 2:M
    # Threads.@threads for i in 2:M
        y = y_cheb[i]

        D[1, i] = 2.0 * (-1)^(i-1) / (1.0 - y)
        D[i, 1] = -0.5 * (-1)^(i-1) / (1.0 - y)
        D[end, i] = -2.0 * (-1)^(M+i-1) / (1.0 + y)
        D[i, end] = 0.5 * (-1)^(M+i-1) / (1.0 + y)
    end # for

    ### Inner elements
    for i in 2:M
    # Threads.@threads for i in 2:M
        for j in 2:M
        # Threads.@threads for j in 2:M
            if i != j
                y_i = y_cheb[i]
                y_j = y_cheb[j]

                D[i, j] = (-1)^(i+j-2) / (y_i - y_j)
            end # for
        end # for
    end # for

    D /= x_max

    return D
end # function

@views function trapz(y_arr, Δx)
    integral = Δx/2 * (y_arr[1] + 2*sum(y_arr[2:end-1]) + y_arr[end])

    return integral
end # function

# Taken from https://discourse.julialang.org/t/simpsons-rule/84114/2
@views function simps(y_arr, Δx)
    if length(y_arr) < 5  # For very small arrays
        integral = Δx * sum(y_arr)
    elseif length(y_arr) % 2 == 1
        integral = Δx/3 * (y_arr[1] + 2*sum(y_arr[3:2:end-2]) + 4*sum(y_arr[2:2:end-1]) + y_arr[end])
    else
        integral = Δx/3 * (y_arr[1] + 2*sum(y_arr[3:2:end-5]) + 4*sum(y_arr[2:2:end-4]) + y_arr[end-3]) +
                   (3*Δx/8) * (y_arr[end-3] + 3*y_arr[end-2] + 3*y_arr[end-1] + y_arr[end])
    end # if

    return integral
end # function

"""Assumes that size(w_arr) = (N_x, N_y)."""
function simps_2D(w_arr, Δx, Δy)
    return simps([simps(w_arr[:, j], Δx) for j in 1:size(w_arr, 2)], Δy)
end # function

function x2index(x_span, N_x, select_x)
    return Int64(round((select_x - x_span[1]) / (x_span[2] - x_span[1]) * (N_x - 1) + 1))
end # function

"""Assumes 1D array and equidistant spacing."""
function x2index(select_x, x_arr)
    return Int64(round((select_x - x_arr[1]) / (x_arr[end] - x_arr[1]) * (length(x_arr) - 1) + 1))
end # function

function x2index_floor(x_span, N_x, select_x)
    return Int64(floor((select_x - x_span[1]) / (x_span[2] - x_span[1]) * (N_x - 1) + 1))
end # function

"""Assumes 1D array and equidistant spacing."""
function x2index_floor(select_x, x_arr)
    return Int64(floor((select_x - x_arr[1]) / (x_arr[end] - x_arr[1]) * (length(x_arr) - 1) + 1))
end # function

function x2index_ceil(x_span, N_x, select_x)
    return Int64(ceil((select_x - x_span[1]) / (x_span[2] - x_span[1]) * (N_x - 1) + 1))
end # function

"""Assumes 1D array and equidistant spacing."""
function x2index_ceil(select_x, x_arr)
    return Int64(ceil((select_x - x_arr[1]) / (x_arr[end] - x_arr[1]) * (length(x_arr) - 1) + 1))
end # function

function translate_y(untranslated_y_arr, x_span, x_shift)
    N_y = size(untranslated_y_arr)[1]
    shift_index = Int64(round(x_shift / (x_span[2] - x_span[1]) * (N_y - 1)))

    translated_y_arr = zero(untranslated_y_arr)
    if shift_index > 0
        translated_y_arr[shift_index+1:end, ..] = untranslated_y_arr[1:end-shift_index, ..]
    elseif shift_index < 0
        translated_y_arr[1:end+shift_index, ..] = untranslated_y_arr[1-shift_index:end, ..]
    else
        translated_y_arr = copy(untranslated_y_arr)
    end # if

    return translated_y_arr
end # function

function is_approx_equal(A, B; abstol=1e-6)
    return sum(abs.(A .- B)) < abstol
end # function

function check_if_real(arr; name="arr")
    if !all(imag.(arr) .≈ zero(arr))
        @printf("WARNING: %s isn't real.\n", name)
    end # if
end # function

function check_if_imag(arr; name="arr")
    if !all(real.(arr) .≈ zero(arr))
        @printf("WARNING: %s isn't imaginary.\n", name)
    end # if
end # function

function check_if_int(arr, name="arr")
    if !all(arr .≈ round.(Int, arr))
        @printf("WARNING: %s isn't an integer.\n", name)
    end # if
end # function

function check_if_Hermitian(arr; name="arr", epsilon=1E-5)
    arr_hermiticity = sum(abs2.(arr - adjoint(arr)))
    # println(arr_hermiticity)

    if arr_hermiticity > epsilon
        @printf("WARNING: %s isn't Hermitian.\n", name)
    end # if
end # function

function check_if_close(arr1, arr2; name1="arr1", name2="arr2")
    # if !all(arr1 .≈ arr2)
    if !isapprox(arr1, arr2)
        @printf("WARNING: %s and %s aren't close to equal.\n", arr1, arr2)
    end # if
end # function

# end # module
