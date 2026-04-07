# module PlaneWaves
import Base.conj
import Base.adjoint
import Base.+
import Base.-
import Base.*
import Base./
import Base.vcat
import Base.kron
# using LinearAlgebra
using SparseArrays
# using Cubature
using HCubature

struct PWRep
    N::Int64
    D::Int64
    m_vec2::Vector{Vector{Int64}}
    coeff_vec::Vector{ComplexF64}

    function PWRep(m_vec2, coeff_vec)
        @assert length(m_vec2) == length(coeff_vec)
        N = length(m_vec2)
        D = length(m_vec2[1])

        new(N, D, m_vec2, coeff_vec)
    end # function
end # struct

function index_dict(arr)
    d = Dict{eltype(arr), Vector{Int}}()
    for (i, x) in enumerate(arr)  # pairs(arr)
        if haskey(d, x)
            push!(d[x], i)
        else
            d[x] = [i]
        end # if
    end # for

    return d
end # function

function vcat(v1::PWRep, v2::PWRep)
    return PWRep(vcat(v1.m_vec2, v2.m_vec2), vcat(v1.coeff_vec, v2.coeff_vec))
end # function

function conj(v::PWRep)
    new_m_vec2 = Vector{Vector{Int64}}([])
    new_coeff_vec = Vector{ComplexF64}([])
    for i in 1:v.N
        push!(new_m_vec2, (-1) .* v.m_vec2[i])
        push!(new_coeff_vec, conj(v.coeff_vec[i]))
    end # for
    new_v = PWRep(new_m_vec2, new_coeff_vec)
    new_v = add_repeating(new_v)

    return new_v
end # function

function adjoint(v::PWRep)
    return conj(v)
end # function

function +(α::T, v::PWRep) where {T<:Number}
    return +(α * PW_id_D(v.D), v)
end # function

function +(v::PWRep, α::T) where {T<:Number}
    return +(v, α * PW_id_D(v.D))
end # function

function +(v1::PWRep, v2::PWRep)
    return add_repeating(vcat(v1, v2))
end # function

function -(α::T, v::PWRep) where {T<:Number}
    return -(α * PW_id_D(v.D), v)
end # function

function -(v::PWRep, α::T) where {T<:Number}
    return -(v, α * PW_id_D(v.D))
end # function

function -(v1::PWRep, v2::PWRep)
    neg_v2 = PWRep(v2.m_vec2, (-1.0) .* v2.coeff_vec)

    return add_repeating(vcat(v1, neg_v2))
end # function

function *(α::T, v::PWRep) where {T<:Number}
    return PWRep(v.m_vec2, α .* v.coeff_vec)
end # function

function *(v::PWRep, α::T) where {T<:Number}
    return *(α, v)
end # function

function *(v1::PWRep, v2::PWRep)
    new_m_vec2 = Vector{Vector{Int64}}([])
    new_coeff_vec = Vector{ComplexF64}([])
    for i in 1:v1.N
        for j in 1:v2.N
            push!(new_m_vec2, v1.m_vec2[i] .+ v2.m_vec2[j])
            push!(new_coeff_vec, v1.coeff_vec[i] * v2.coeff_vec[j])
        end # for
    end # for
    new_v = PWRep(new_m_vec2, new_coeff_vec)
    new_v = add_repeating(new_v)

    return new_v
end # function

function /(v::PWRep, α::T) where {T<:Number}
    return *(1.0/α, v)
end # function

function kron(v1::PWRep, v2::PWRep)
    new_m_vec2 = Vector{Vector{Int64}}([])
    new_coeff_vec = Vector{ComplexF64}([])
    for i in 1:v1.N
        for j in 1:v2.N
            push!(new_m_vec2, vcat(v1.m_vec2[i], v2.m_vec2[j]))
            push!(new_coeff_vec, v1.coeff_vec[i] * v2.coeff_vec[j])
        end # for
    end # for
    new_v = PWRep(new_m_vec2, new_coeff_vec)
    new_v = add_repeating(new_v)

    return new_v
end # function

# TODO Could also sort by harmonic index
# TODO Could also remove modes with zero coefficients
function add_repeating(v::PWRep)
    occ_dict = index_dict(v.m_vec2)

    new_m_vec2 = Vector{Vector{Int64}}([])
    new_coeff_vec = Vector{ComplexF64}([])
    for (m_vec, idxs) in pairs(occ_dict)
        push!(new_m_vec2, m_vec)
        push!(new_coeff_vec, sum(v.coeff_vec[idxs]))
    end # for
    new_v = PWRep(new_m_vec2, new_coeff_vec)

    return new_v
end # function

function PW_id()
    return PWRep([[0]], [1.0])
end # function

function PW_id_D(D::Int)
    if D == 1
        id_D = PW_id()
    else
        id_vec = [PW_id() for i in 1:D]
        id_D = kron(id_vec...)
    end # if

    return id_D
end # function

function PW_exp_inqx(n::Int)
    return PWRep([[n]], [1.0])
end # function

function PW_cos_nqx(n::Int)
    return PWRep([[n], [-n]], [0.5, 0.5])
end # function

function PW_sin_nqx(n::Int)
    return PWRep([[n], [-n]], [-0.5im, 0.5im])
end # function

function PW_exp_iqx()
    return PW_exp_inqx(1)
end # function

function PW_exp_imqx()
    return PW_exp_inqx(-1)
end # function

function PW_cos_qx()
    return PW_cos_nqx(1)
end # function

function PW_sin_qx()
    return PW_sin_nqx(1)
end # function

function convert_PW_to_nested_vec(v::PWRep)
    PW_vec = Vector{Vector{Any}}([])
    for i in 1:v.N
        m_vec = v.m_vec2[i]
        coeff = v.coeff_vec[i]
        push!(PW_vec, [m_vec, coeff])
    end # for

    return PW_vec
end # function

function convert_nested_vec_to_PW(PW_vec::Vector{Vector{T}}) where {T<:Any}
    m_vec2 = Vector{Vector{Int64}}([])
    coeff_vec = Vector{ComplexF64}([])
    for i in 1:length(PW_vec)
        push!(m_vec2, PW_vec[i][1])
        push!(coeff_vec, PW_vec[i][2])
    end # for

    return PWRep(m_vec2, coeff_vec)
end # function

function convert_PW_to_matrix(v::PWRep, N_Fourier::Int)
    N2_Fourier = 2*N_Fourier + 1
    N_Hilbert = N2_Fourier^v.D

    matrix_op = spzeros(ComplexF64, N_Hilbert, N_Hilbert)
    for i in 1:v.N
        coeff = v.coeff_vec[i]

        matrix_vec = Vector{SparseMatrixCSC{ComplexF64, Int64}}([])
        for d in 1:v.D
            m = v.m_vec2[i][d]

            push!(matrix_vec, spdiagm(m => repeat([1.0], N2_Fourier-abs(m))))
        end # for

        if length(matrix_vec) == 1
            matrix_op += coeff * matrix_vec[1]
        else
            matrix_op += coeff * kron(matrix_vec...)
        end # if
    end # for

    return matrix_op
end # function

function convert_PW_to_vector(v::PWRep, N_Fourier::Int)
    N2_Fourier = 2*N_Fourier + 1
    N_Hilbert = N2_Fourier^v.D

    vector_ket = spzeros(ComplexF64, N_Hilbert)
    for i in 1:v.N
        coeff = v.coeff_vec[i]

        vector_vec = Vector{SparseVector{ComplexF64, Int64}}([])
        for d in 1:v.D
            m = v.m_vec2[i][d]

            # NOTE: First component should be N_Fourier mode, last component
            # should be -N_Fourier mode.
            # push!(vector_vec, sparsevec([N_Fourier+m+1], [1.0], N2_Fourier))
            push!(vector_vec, sparsevec([N_Fourier-m+1], [1.0], N2_Fourier))
        end # for

        if length(vector_vec) == 1
            vector_ket += coeff * vector_vec[1]
        else
            vector_ket += coeff * kron(vector_vec...)
        end # if
    end # for

    return vector_ket
end # function

# TODO Only works for 2D
function convert_vector_to_PW(vector_ket::Vector, D::Int, N_Fourier::Int; ϵ=1e-8)
    N2_Fourier = 2*N_Fourier + 1
    N_Hilbert = length(vector_ket)

    m_vec2 = Vector{Vector{Int64}}([])
    coeff_vec = Vector{ComplexF64}([])
    for i in 1:N_Hilbert
        coeff = vector_ket[i]

        if abs(coeff) > ϵ
            m_vec = (-1) .* [(i-1)÷N2_Fourier, (i-1)%N2_Fourier] .+ N_Fourier

            push!(m_vec2, m_vec)
            push!(coeff_vec, coeff)
        end # if
    end # for

    return PWRep(m_vec2, coeff_vec)
end # function

# NOTE: Assumes periodicity of 2π in all arguments.
function convert_PW_to_function_r(v::PWRep)
    function PW_function_r(r)
        res = complex(0.0, 0.0)
        for i in 1:v.N
            coeff = v.coeff_vec[i]

            α = 0.0
            for d in 1:v.D
                m = v.m_vec2[i][d]

                α += m * r[d]
            end # for
            # α *= 2π

            res += coeff * exp(im * α)
        end # for

        return res
    end # function

    return PW_function_r
end # function

# NOTE: Assumes periodicity of 2π in all arguments.
function convert_vector_to_function_r(vector_ket::Vector, D::Int, N_Fourier::Int)
    return convert_PW_to_function_r(convert_vector_to_PW(vector_ket, D, N_Fourier))
end # function

# NOTE: Assumes periodicity of 2π in all arguments.
function convert_full_ket_to_function_r(ψ, N_state, D, N_Fourier)
    N_Hilbert = length(ψ)
    div_N_Hilbert = N_Hilbert ÷ N_state

    calc_ψ_j_vec = []
    for j in 1:N_state
        idx1 = (j-1) * div_N_Hilbert + 1
        idx2 = j * div_N_Hilbert
        push!(calc_ψ_j_vec, convert_vector_to_function_r(ψ[idx1:idx2], D, N_Fourier))
    end # for

    function function_r(r)
        res_vec = Vector{ComplexF64}([])
        for calc_ψ_j in calc_ψ_j_vec
            push!(res_vec, calc_ψ_j(r))
        end # for

        return res_vec
    end # function

    return function_r
end # function

# NOTE: Assumes periodicity of 2π in all arguments.
# NOTE: Only Fourier modes in range (-N_Fourier):N_Fourier are considered. Thus,
# in general, returned plane wave (PW) vector will be approximation of given
# function.
function approx_function_r_with_PW(f::Function, D::Int, N_Fourier::Int; ϵ=1e-8)
    m_vec2 = Vector{Vector{Int64}}([])
    coeff_vec = Vector{ComplexF64}([])
    for I in CartesianIndices(Tuple([(-N_Fourier):N_Fourier for d in 1:D]))
        m_vec = collect(Tuple(I))

        function integrand(r)
            α = 0.0
            for d in 1:D
                α += m_vec[d] * r[d]
            end # for
            # α *= 2π

            return f(r) * exp(-im * α)
        end # function

        coeff = hcubature(integrand, [-π for d in 1:D], [π for d in 1:D]; rtol=1e-6, atol=1e-6)[1] / (2π)^D
        # re_coeff = hcubature(r -> real(integrand(r)), [-π for d in 1:D], [π for d in 1:D])[1] / (2π)^D
        # im_coeff = hcubature(r -> imag(integrand(r)), [-π for d in 1:D], [π for d in 1:D])[1] / (2π)^D
        # coeff = complex(re_coeff, im_coeff)

        if abs(coeff) > ϵ
            push!(m_vec2, m_vec)
            push!(coeff_vec, coeff)
        end # if
    end # for

    return PWRep(m_vec2, coeff_vec)
end # function

# NOTE: Assumes periodicity of 2π in all arguments.
function approx_function_r_with_PW_vector(f::Function, D::Int, N_Fourier::Int)
    return convert_PW_to_vector(approx_function_r_with_PW(f, D, N_Fourier), N_Fourier)
end # function

# NOTE: Assumes periodicity of 2π in all arguments.
function calc_function_r_PW_approx(f::Function, D::Int, N_Fourier::Int)
    return convert_PW_to_function_r(approx_function_r_with_PW(f, D, N_Fourier))
end # function

# ########## OLD MATRIX IMPLEMENTATION ##########
# function id(N2_Fourier)
#     return diagm(0 => repeat([1], N2_Fourier))
# end # function

# function exp_inqx(N2_Fourier; n=1)
#     return diagm(n => repeat([1], N2_Fourier-n))
# end # function

# function exp_minqx(N2_Fourier; n=1)
#     return diagm(-n => repeat([1], N2_Fourier-n))
# end # function

# function cos_nqx(N2_Fourier; n=1)
#     return 0.5 * (exp_inqx(N2_Fourier; n=n) + exp_minqx(N2_Fourier; n=n))
# end # function

# function sin_nqx(N2_Fourier; n=1)
#     return -0.5im * (exp_inqx(N2_Fourier; n=n) - exp_minqx(N2_Fourier; n=n))
# end # function

# function exp_iqx(N2_Fourier)
#     return exp_inqx(N2_Fourier; n=1)
# end # function

# function exp_miqx(N2_Fourier)
#     return exp_minqx(N2_Fourier; n=1)
# end # function

# function cos_qx(N2_Fourier)
#     return cos_nqx(N2_Fourier; n=1)
# end # function

# function sin_qx(N2_Fourier)
#     return sin_nqx(N2_Fourier; n=1)
# end # function
# ########## END OF OLD MATRIX IMPLEMENTATION ##########
# end # module
