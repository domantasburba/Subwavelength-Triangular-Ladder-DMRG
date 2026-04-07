# This version uses my conventions. Matrices will differ from Tana's, but
# physical results should agree perfectly.

# Lexicographic order: https://en.wikipedia.org/wiki/Lexicographic_order

module EDBosons

using SparseArrays
using LinearAlgebra

include("../Math.jl")

########## GENERAL EXACT DIAGONALIZATION ##########
"""
    create_boson_basis(N_lat, N_part)

Return Fock basis and basis dictionary for `N_part` bosons in a lattice of `N_lat` sites.

Assumes spin 0 bosons. Basis ordered lexicographically. Order of states is determined by comparing them in the following way:
1) Look at which state has more particles on the left. If one has more, it is higher order.
2) If the states have the same amount of particles, move to the right and repeat the comparison.
"""
function create_boson_basis(N_lat, N_part; max_N_part=nothing)
    basis = Vector{Vector{Int}}([])

    # Create highest lexicographic order state and append it to basis array
    first_state = zeros(Int, N_lat)
    first_state[1] = N_part
    push!(basis, first_state)

    # Create lowest order state
    last_state = zeros(Int, N_lat)
    last_state[end] = N_part

    state = copy(first_state)
    #=
    From the current state (starting with the highest order state), the loop
    generates the next, lower order state until it reaches the lowest order.

    This is done by keeping track of the right-most set of particles in the
    given state.  If they are not near the edge, simply take one right-most
    particle and move it to the right. If they are at the edge, remove them,
    also remove one particle from the right-most set and add all the removed
    particles to the site that is one to the right from the right-most set.
    Checking of edge occupancy is given by state[N_lat] != 0.

    Finally, there is a nuance: if the two last sites are occupied, then one
    should follow the normal move to the right procedure, NOT the edge
    procedure. This is handled by state[N_lat-1] == 0.
    =#
    while state != last_state
        if state[N_lat] != 0 && state[N_lat-1] == 0
            for i in (N_lat-2):-1:1
                if state[i] != 0
                    state[i] -= 1
                    state[i+1] += state[N_lat]+1
                    state[N_lat] = 0
                    break
                end # if
            end # for
        else
            for i in (N_lat-1):-1:1
                if state[i] != 0
                    state[i] -= 1
                    state[i+1] += 1
                    break
                end # if
            end # for
        end # if

        push!(basis, copy(state))
    end # while

    if max_N_part != nothing
        @assert isa(max_N_part, Int)

        cutoff_basis = Vector{Vector{Int}}([])
        for state in basis
            satisfiesCutoff = true
            for i in 1:N_lat
                if state[i] > max_N_part
                    satisfiesCutoff = false
                    break
                end # if
            end # for

            if satisfiesCutoff
                push!(cutoff_basis, copy(state))
            end # if
        end # for

        basis = cutoff_basis
    end # if

    basis_dict = Dict(key=>value for (value, key) in enumerate(basis))

    return basis, basis_dict
end # function

"""
Creates Fock basis containing all states from N_part=1 to N_part=max_N_part
"""
function create_boson_ext_basis(N_lat, max_N_part)
    # NOTE: Including vacuum state (N_part=0) causes issues and gives no benefit in our case.
    basis, _ = create_boson_basis(N_lat, 1)
    for N_part in 2:max_N_part
        basis_n, _ = create_boson_basis(N_lat, N_part)
        basis = vcat(basis, basis_n)
    end # for
    basis_dict = Dict(key=>value for (value, key) in enumerate(basis))

    return basis, basis_dict
end # function

function choose_boson_basis(N_lat, N_part, max_N_part, conserveParticleNumber)
    if conserveParticleNumber
        basis, basis_dict = create_boson_basis(N_lat, N_part; max_N_part=max_N_part)
    else
        basis, basis_dict = create_boson_ext_basis(N_lat, max_N_part)
    end # if

    return basis, basis_dict
end # function

function create_boson_Ham_chem_pot(basis, μ)
    diag = Vector{Float64}(undef, length(basis))
    for (i, state) in enumerate(basis)
        summ = 0
        for n_j in state
            summ += n_j
        end # for

        diag[i] = -μ * summ
    end # for

    return spdiagm(0 => diag)
end # function

function create_boson_Ham_int(basis, U)
    diag = Vector{Float64}(undef, length(basis))
    for (i, state) in enumerate(basis)
        summ = 0
        for n_j in state
            summ += n_j*(n_j-1)
        end # for

        diag[i] = U/2 * summ
    end # for

    return spdiagm(0 => diag)
end # function

"""Options: PBC, OBC."""
function _calc_boson_Ham_int_Vj_mat_el(state, V_arr, N_lat; BC="PBC")
    Vj_mat_el = 0.0
    for (j, Vj) in enumerate(V_arr[2:end])
        summ = 0.0
        for i in 1:N_lat
            if BC == "PBC"
                BC_cond = true
            elseif BC == "OBC"
                BC_cond = (i+j <= N_lat)
            else
                throw(ArgumentError("Invalid boundary condition (BC). Please check options in function documentation."))
            end # if

            if BC_cond
                ipj = mod1(i+j, N_lat)
                n_i = state[i]
                n_ipj = state[ipj]
                summ += n_i*n_ipj
            end # if
        end # for
        Vj_mat_el += Vj * summ
    end # for

    return Vj_mat_el
end # function

"""Options: PBC, OBC."""
function create_boson_Ham_int(basis, V_arr, N_lat; BC="PBC")
    diag = Vector{Float64}(undef, length(basis))

    for (ii, state) in enumerate(basis)
        V0_mat_el = 0.0

        V0 = V_arr[1]
        summ = 0.0
        for n_i in state
            summ += n_i*(n_i-1)
        end # for
        V0_mat_el += V0/2 * summ

        Vj_mat_el = _calc_boson_Ham_int_Vj_mat_el(state, V_arr, N_lat; BC=BC)

        diag[ii] = V0_mat_el + Vj_mat_el
    end # for

    return spdiagm(0 => diag)
end # function

"""Options: PBC, OBC."""
function create_boson_Ham_tunneling(basis, basis_dict, N_lat, J; BC="PBC")
    rows = Vector{Int}([])
    cols = Vector{Int}([])
    data = Vector{Float64}([])

    for (i, state) in enumerate(basis)
        for j in 1:N_lat
            if state[j] != 0
                for pm in [-1, 1]
                    if BC == "PBC"
                        BC_cond = true
                    elseif BC == "OBC"
                        BC_cond = (1 <= j+pm <= N_lat)
                    else
                        throw(ArgumentError("Invalid boundary condition (BC). Please check options in function documentation."))
                    end # if

                    if BC_cond
                        jj = mod1(j+pm, N_lat)  # j+pm

                        neighbour_state = copy(state)
                        neighbour_state[j] -= 1
                        neighbour_state[jj] += 1

                        if haskey(basis_dict, neighbour_state)
                            append!(rows, basis_dict[neighbour_state])
                            append!(cols, i)
                            append!(data, -J*sqrt(state[j])*sqrt(neighbour_state[jj]))
                        end # if
                    end # if
                end # for
            end # if
        end # for
    end # for

    return sparse(rows, cols, data, length(basis), length(basis))
end # function

"""Options: PBC, OBC."""
function create_boson_Ham_tunneling(basis, basis_dict, J_arr, N_lat, N_part; BC="PBC")
    rows = Vector{Int}([])
    cols = Vector{Int}([])
    data = Vector{Float64}([])

    for (ii, state) in enumerate(basis)
        for i in 1:N_lat
            if state[i] != 0
                for (j, Jj) in enumerate(J_arr)
                    if BC == "PBC"
                        BC_cond = true
                    elseif BC == "OBC"
                        BC_cond = (i+j <= N_lat)
                    else
                        throw(ArgumentError("Invalid boundary condition (BC). Please check options in function documentation."))
                    end # if

                    ipj = mod1(i+j, N_lat)
                    if BC_cond && state[ipj] != N_part
                        first_state = copy(state)
                        first_state[i] -= 1
                        first_state[ipj] += 1

                        if haskey(basis_dict, first_state)
                            append!(rows, basis_dict[first_state])
                            append!(cols, ii)
                            append!(data, -Jj*sqrt(state[i])*sqrt(first_state[ipj]))
                        end # if
                    end # if
                end # for
            end # if
        end # for

        for i in 1:N_lat
            if state[i] != N_part
                for (j, Jj) in enumerate(J_arr)
                    if BC == "PBC"
                        BC_cond = true
                    elseif BC == "OBC"
                        BC_cond = (i+j <= N_lat)
                    else
                        throw(ArgumentError("Invalid boundary condition (BC). Please check options in function documentation."))
                    end # if

                    ipj = mod1(i+j, N_lat)
                    if BC_cond && state[ipj] != 0
                        second_state = copy(state)
                        second_state[i] += 1
                        second_state[ipj] -= 1

                        if haskey(basis_dict, second_state)
                            append!(rows, basis_dict[second_state])
                            append!(cols, ii)
                            append!(data, -Jj*sqrt(second_state[i])*sqrt(state[ipj]))
                        end # if
                    end # if
                end # for
            end # if
        end # for
    end # for

    return sparse(rows, cols, data, length(basis), length(basis))
end # function

"""Options: PBC, OBC."""
function create_boson_Ham_pair_tunneling(basis, basis_dict, N_lat, P; BC="PBC")
    rows = Vector{Int}([])
    cols = Vector{Int}([])
    data = Vector{Float64}([])

    for (i, state) in enumerate(basis)
        for j in 1:N_lat
            if state[j] > 1
                for pm in [-1, 1]
                    if BC == "PBC"
                        BC_cond = true
                    elseif BC == "OBC"
                        BC_cond = (1 <= j+pm <= N_lat)
                    else
                        throw(ArgumentError("Invalid boundary condition (BC). Please check options in function documentation."))
                    end # if

                    if BC_cond
                        jj = mod1(j+pm, N_lat)  # j+pm

                        neighbour_state = copy(state)
                        neighbour_state[j] -= 2
                        neighbour_state[jj] += 2

                        if haskey(basis_dict, neighbour_state)
                            append!(rows, basis_dict[neighbour_state])
                            append!(cols, i)
                            append!(data, -P * sqrt(state[jj]+2)*sqrt(state[jj]+1) * sqrt(state[j]-1)*sqrt(state[j]))
                        end # if
                    end # if
                end # for
            end # if
        end # for
    end # for

    return sparse(rows, cols, data, length(basis), length(basis))
end # function

### WARNING: Would only work in grand canonical ensemble (constant μ, changing
### N_part) because these operators do not preserve particle number. If one
### works in canonical ensemble (typical case), these operators won't work since
### fixed particle number basis is taken.
# """Boson annihilation operator"""
# function create_boson_a_j(j, basis)
#     rows = Vector{Int}([])
#     cols = Vector{Int}([])
#     data = Vector{Float64}([])

#     for (ii, state) in enumerate(basis)
#         if state[j] != 0
#             new_state = copy(state)
#             new_state[j] -= 1
#             append!(rows, basis_dict[new_state])
#             append!(cols, ii)
#             append!(data, sqrt(state[j]))
#         end # if
#     end # for

#     return sparse(rows, cols, data, length(basis), length(basis))
# end # function

# """Boson creation operator"""
# function create_boson_a_dag_j(j, basis, N_part)
#     rows = Vector{Int}([])
#     cols = Vector{Int}([])
#     data = Vector{Float64}([])

#     for (ii, state) in enumerate(basis)
#         if state[j] != N_part
#             new_state = copy(state)
#             new_state[j] += 1
#             append!(rows, basis_dict[new_state])
#             append!(cols, ii)
#             append!(data, sqrt(new_state[j]))
#         end # if
#     end # for

#     return sparse(rows, cols, data, length(basis), length(basis))
# end # function

function create_boson_n_j(j, basis)
    rows = Vector{Int}([])
    cols = Vector{Int}([])
    data = Vector{Float64}([])

    for (ii, state) in enumerate(basis)
        append!(rows, ii)
        append!(cols, ii)
        append!(data, state[j])
    end # for

    return sparse(rows, cols, data, length(basis), length(basis))
end # function

function create_boson_g1_ij(i, j, basis, basis_dict, N_part)
    rows = Vector{Int}([])
    cols = Vector{Int}([])
    data = Vector{Float64}([])

    for (ii, state) in enumerate(basis)
        if state[i] != N_part && state[j] != 0
            new_state = copy(state)
            new_state[i] += 1
            new_state[j] -= 1

            if haskey(basis_dict, new_state)
                append!(rows, basis_dict[new_state])
                append!(cols, ii)
                append!(data, sqrt(state[j])*sqrt(new_state[i]))
            end # if
        end # if
    end # for

    return sparse(rows, cols, data, length(basis), length(basis))
end # function

function create_boson_g2_ij(i, j, basis, basis_dict, N_part)
    rows = Vector{Int}([])
    cols = Vector{Int}([])
    data = Vector{Float64}([])

    for (ii, state) in enumerate(basis)
        if state[i] < (N_part-1) && state[j] > 1
            new_state = copy(state)
            new_state[i] += 2
            new_state[j] -= 2

            if haskey(basis_dict, new_state)
                append!(rows, basis_dict[new_state])
                append!(cols, ii)
                append!(data, sqrt(state[j])*sqrt(state[j]-1) * sqrt(new_state[i])*sqrt(new_state[i]-1))
            end # if
        end # if
    end # for

    return sparse(rows, cols, data, length(basis), length(basis))
end # function

function create_boson_κ_j(j, basis, basis_dict, N_lat, N_part)
    rows = Vector{Int}([])
    cols = Vector{Int}([])
    data = Vector{ComplexF64}([])

    if j != N_lat
        for (ii, state) in enumerate(basis)
            if state[j] != 0 && state[j+1] != N_part
                first_state = copy(state)
                first_state[j] -= 1
                first_state[j+1] += 1
                append!(rows, basis_dict[first_state])
                append!(cols, ii)
                append!(data, im*sqrt(state[j])*sqrt(first_state[j+1]))
            end # if

            if state[j] != N_part && state[j+1] != 0
                second_state = copy(state)
                second_state[j] += 1
                second_state[j+1] -= 1
                append!(rows, basis_dict[second_state])
                append!(cols, ii)
                append!(data, -im*sqrt(second_state[j])*sqrt(state[j+1]))
            end # if
        end # for
    else
        throw(ArgumentError("j=N_lat is not supported for κ_j."))
    end # if

    return sparse(rows, cols, data, length(basis), length(basis))
end # function

function create_boson_κ2_ij(i, j, basis, basis_dict, N_lat, N_part)
    κ_i = create_boson_κ_j(i, basis, basis_dict, N_lat, N_part)
    κ_j = create_boson_κ_j(j, basis, basis_dict, N_lat, N_part)

    return κ_i*κ_j
end # function
########## END OF GENERAL EXACT DIAGONALIZATION ##########

########## STATES ##########
"""Assumes canonical ensemble."""
function create_ideal_SF(basis)
    N_lat = length(basis[1])
    N_part = sum(basis[1])

    ψ_SF = zeros(ComplexF64, length(basis))
    for (ii, state) in enumerate(basis)
        coeff = sqrt(factorial(N_part) / N_lat^N_part)
        for n_j in state
            coeff /= sqrt(factorial(n_j))
        end # for
        ψ_SF[ii] = coeff
    end # for

    return ψ_SF
end # function

# TODO
function create_ideal_PSF(basis)
    N_lat = length(basis[1])
    N_part = sum(basis[1])

    ψ_PSF = zeros(ComplexF64, length(basis))
    for (ii, state) in enumerate(basis)
        isEvenOcc = true
        for n_j in state
            if isodd(n_j)
                isEvenOcc = false
                break
            end # if
        end # for

        if isEvenOcc
            coeff = 1.0
            for n_j in state
                coeff *= sqrt(factorial(n_j)) / factorial(div(n_j, 2))
            end # for
            ψ_PSF[ii] = coeff
        end # for
    end # for

    # TODO Why does this normalization not work?
    # ψ_PSF *= sqrt(factorial(div(N_part, 2))) * (2.0 * (2.0*N_part + N_lat))^(-N_part/4)
    ψ_PSF /= sqrt(ψ_PSF' * ψ_PSF)

    return ψ_PSF
end # function
########## END OF STATES ##########

########## HAMILTONIANS ##########
function create_BH_Ham(basis, basis_dict, N_lat, conserveParticleNumber, μ; J, U, BC="PBC")
    Ham_tunneling = create_boson_Ham_tunneling(basis, basis_dict, N_lat, J; BC=BC)
    Ham_int = create_boson_Ham_int(basis, U)
    if conserveParticleNumber
        Ham = Ham_tunneling + Ham_int
    else
        Ham_chem = create_boson_Ham_chem_pot(basis, μ)
        Ham = Ham_tunneling + Ham_int + Ham_chem
    end # if

    return Ham
end # function

function create_Zhou_PSF_deep_lat_Ham(basis, basis_dict, N_lat, conserveParticleNumber, μ; J, U, BC="PBC")
    Ham_pair_tun = create_boson_Ham_pair_tunneling(basis, basis_dict, N_lat, J; BC=BC)
    Ham_int = create_boson_Ham_int(basis, U)
    if conserveParticleNumber
        Ham = Ham_pair_tun + Ham_int
    else
        Ham_chem = create_boson_Ham_chem_pot(basis, μ)
        Ham = Ham_pair_tun + Ham_int + Ham_chem
    end # if

    return Ham
end # function

function create_EBH_Ham(basis, basis_dict, J_arr, V_arr, N_lat, N_part; μ=0.0, BC="PBC")
    Ham_tunneling = create_boson_Ham_tunneling(basis, basis_dict, J_arr, N_lat, N_part; BC=BC)
    Ham_int = create_boson_Ham_int(basis, V_arr, N_lat; BC=BC)
    Ham_chem = create_boson_Ham_chem_pot(basis, μ)
    # Ham = Ham_tunneling + Ham_int
    Ham = Ham_tunneling + Ham_int + Ham_chem

    return Ham
end # function

# TODO CHECK THIS CODE
# TODO DOES NOT SUPPORT g_z
function create_brick_wall_Ham(basis, basis_dict, N_lat, N_part, conserveParticleNumber, μ; J_1, J_2, g_0, g_x, g_z, G_000, G_001, G_011, BC="PBC")
    Ham_tunneling = create_boson_Ham_tunneling(basis, basis_dict, [J_1, J_2], N_lat, N_part; BC=BC)
    U = 2.0 * (g_0 + 0.5*g_x)*G_000
    V = 2.0*g_0*G_011
    Ham_int = create_boson_Ham_int(basis, [U, V], N_lat; BC=BC)
    P = 0.5*g_x*G_011
    Ham_pair_tun = create_boson_Ham_pair_tunneling(basis, basis_dict, N_lat, P; BC=BC)
    if conserveParticleNumber
        Ham = Ham_tunneling + Ham_int + Ham_pair_tun
    else
        Ham_chem = create_boson_Ham_chem_pot(basis, μ)
        Ham = Ham_tunneling + Ham_int + Ham_pair_tun + Ham_chem
    end # if

    return Ham
end # function

function create_boson_Ham(par)
    N_lat, N_part, conserveParticleNumber, max_N_part, μ, BC, J_arr, V_arr = par.N, par.N_part, par.conserveParticleNumber, par.max_N_part, par.μ, par.BC, par.J_arr, par.V_arr

    if conserveParticleNumber
        basis, basis_dict = create_boson_basis(N_lat, N_part)
        Ham = create_EBH_Ham(basis, basis_dict, J_arr, V_arr, N_lat, N_part; μ=0.0, BC=BC)
    else
        basis, basis_dict = create_boson_ext_basis(N_lat, max_N_part)
        Ham = create_EBH_Ham(basis, basis_dict, J_arr, V_arr, N_lat, N_part; μ=μ, BC=BC)
    end # if

    return Ham
end # function
########## END OF HAMILTONIANS ##########

########## EXPECTATION VALUES ##########
function calc_expect_n_arr(ψ, basis, N_lat)
    expect_n_arr = zeros(Float64, N_lat)
    for j in 1:N_lat
        n_j = create_boson_n_j(j, basis)
        expect_n_arr[j] = ψ' * n_j * ψ
    end # for

    return expect_n_arr
end # function

function calc_expect_n2_matrix(ψ, basis, N_lat)
    expect_n2_matrix = zeros(Float64, N_lat, N_lat)
    for i in 1:N_lat
        n_i = create_boson_n_j(i, basis)
        for j in 1:N_lat
            n_j = create_boson_n_j(j, basis)
            expect_n2_matrix[i, j] = ψ' * n_i*n_j * ψ
        end # for
    end # for

    return expect_n2_matrix
end # function

function calc_expect_Δn_arr(ψ, basis, N_lat)
    expect_Δn_arr = zeros(Float64, N_lat)
    for j in 1:N_lat
        n_j = create_boson_n_j(j, basis)
        expect_Δn_arr[j] = ψ' * n_j*n_j * ψ - (ψ' * n_j * ψ)^2
    end # for

    return expect_Δn_arr
end # function

function calc_expect_δN(expect_n_arr, N_lat)
    avg_n = avg(expect_n_arr)
    expect_δN = 0.0
    for j in 1:N_lat
        expect_δN += 1.0/N_lat * (-1)^j * (expect_n_arr[j] - avg_n)
    end # for

    return expect_δN
end # function

function calc_expect_S_k_arr(expect_n2_matrix, expect_n_arr, k_arr, basis, N_lat)
    expect_S_k_arr = zeros(ComplexF64, length(k_arr))
    for (idx, k) in enumerate(k_arr)
        for i in 1:N_lat
            for j in 1:N_lat
                # expect_S_k_arr[idx] += exp(-im*k*(i-j)) * (expect_n2_matrix[i, j] - expect_n_arr[i]*expect_n_arr[j])
                expect_S_k_arr[idx] += exp(-im*k*(i-j)) * expect_n2_matrix[i, j]
            end # for
        end # for
    end # for
    expect_S_k_arr /= N_lat

    return expect_S_k_arr
end # function
########## END OF EXPECTATION VALUES ##########
end # module
