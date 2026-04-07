#=
Exact diagonalization program for spin-s fermionic systems with N_lat lattice
sites, composed of N_part particles. Constructed Fock basis includes all
possible configurations, including double occupancy and higher occupancy states.
=#
# state[j] corresponds to highest spin projection at site j, state[N_lat+j] - second highest at site j and so on.

# This version uses my conventions. Matrices will differ from Tana's, but
# physical results should agree perfectly.

# Lexicographic order: https://en.wikipedia.org/wiki/Lexicographic_order

module EDFermions

using Optim
using StatsBase
using LinearAlgebra
using SparseArrays
using KrylovKit
using Expokit  # Expokit's expmv uses less memory than FastExpm functions

include("../Math.jl")

########## GENERAL EXACT DIAGONALIZATION ##########
function calc_no_of_fermion_states(N_s, N_lat, N_part)
    prod = 1
    for i in 0:(N_part-1)
        prod *= N_s*N_lat-i
    end # for
    prod = prod÷factorial(N_part)

    return prod
end # function

"""For fermion basis. Assumes N_lat = N_part."""
function calc_no_of_single_occ_states(N_s, N_lat)
    return N_s^N_lat
end # function

"""
Basis ordered lexicographically. Representated in terms of 0s and 1s since
either 0 or 1 particles may exist in a given state (due to Pauli exclusion
principle).
"""
function create_fermion_basis(N_s, N_lat, N_part)
    basis = Vector{Vector{Int}}([])

    # Create highest lexicographic order state and append it to basis array
    first_state = vcat([1 for i in 1:N_part], [0 for i in (N_part+1):(N_s*N_lat)])
    push!(basis, first_state)

    # Create lowest order state
    last_state = vcat([0 for i in 1:(N_s*N_lat-N_part)], [1 for i in (N_s*N_lat-N_part+1):(N_s*N_lat)])

    # length(state) = N_s*N_lat
    state = copy(first_state)
    while state != last_state
        # E.g., 0{1}00[11] -> 00{1}[11]0
        if state[N_s*N_lat] != 0
            # j = Number of 1s in [...] (for A in A -> B)
            j = 0
            while state[N_s*N_lat-j] != 0
                state[N_s*N_lat-j] -= 1
                j += 1
            end # while

            # Move {1} to right by one and append [...] right after it
            for i in (N_s*N_lat-j-1):-1:1
                if state[i] != 0
                    state[i] -= 1
                    state[(i+1):(i+1+j)] .+= 1
                    break
                end # if
            end # for
        # E.g., 101{1}00 -> 1010{1}0
        else
            # Move {1} to right by one
            for i in (N_s*N_lat-1):-1:1
                if state[i] != 0
                    state[i] -= 1
                    state[i+1] += 1
                    break
                end # if
            end # for
        end # if

        push!(basis, copy(state))
    end # while

    basis_dict = Dict(key=>value for (value, key) in enumerate(basis))

    # display(basis)

    return basis, basis_dict
end # function

function create_single_occ_fermion_basis(N_s, N_lat, N_part)
    basis, basis_dict = create_fermion_basis(N_s, N_lat, N_part)
    I_single_occ, trunc_basis = calc_single_occ_projection_matrix(basis, N_s, N_lat)
    trunc_basis_dict = Dict(key=>value for (value, key) in enumerate(trunc_basis))

    return trunc_basis, trunc_basis_dict
end # function

"""For fermion basis."""
function calc_single_occ_projection_matrix(basis, N_s, N_lat)
    # Calculate occupancy at each lattice site j
    basis_occ = Vector{Vector{Int}}([])
    for state in basis
        occ_vec = [sum(state[j:N_lat:end]) for j in 1:N_lat]
        push!(basis_occ, occ_vec)
    end # for

    # Determine if state is single occupied for every state
    is_single_occ_vec = Vector{Bool}([])
    for occ_vec in basis_occ
        more_than_single_j = [occ_vec[j] > 1 for j in 1:N_lat]
        is_single_occ = !any(more_than_single_j)
        push!(is_single_occ_vec, is_single_occ)
    end # for

    single_occ_idx_vec = findall(is_single_occ_vec)
    trunc_basis = basis[single_occ_idx_vec]
    @assert calc_no_of_single_occ_states(N_s, N_lat) == length(single_occ_idx_vec)

    I_single_occ = zeros(Int, length(basis), length(single_occ_idx_vec))
    for (i, idx_single_occ) in enumerate(single_occ_idx_vec)
        I_single_occ[idx_single_occ, i] = 1
    end # for

    return I_single_occ, trunc_basis
end # function

function calc_time_evo(Ham, wf_initial, t_arr)
    Δt = t_arr[2] - t_arr[1]
    wf_arr = Array{ComplexF64}(undef, length(wf_initial), length(t_arr))
    wf_arr[:, 1] = copy(wf_initial)
    # U_propagator = fastExpm(-im*Δt*Ham)
    for i in 2:length(t_arr)
        # @show i
        wf_arr[:, i] = expmv(-im*Δt, Ham, wf_arr[:, i-1])
        # wf_arr[:, i] = U_propagator * wf_arr[:, i-1]
    end # for

    return wf_arr
end # function

function calc_expectation_value(A, wf)
    return real(adjoint(wf) * A * wf)
end # function

function calc_1D_expectation_values(A, wf_arr)
    expect_arr = Array{Float64}(undef, size(wf_arr, 2))
    for i in 1:size(wf_arr, 2)
        expect_arr[i] = calc_expectation_value(A, wf_arr[:, i])
    end # for

    return expect_arr
end # function

function calc_2D_expectation_values(A, wf_arr)
    expect_arr = Array{Float64}(undef, length(A), size(wf_arr, 2))
    for i in 1:size(wf_arr, 2)
        for j in 1:length(A)
            expect_arr[j, i] = calc_expectation_value(A[j], wf_arr[:, i])
        end # for
    end # for

    return expect_arr
end # function
########## END OF GENERAL EXACT DIAGONALIZATION ##########

########## OPERATORS ##########
function create_fermion_S_plus(basis, basis_dict, m_s_arr, s, N_s, N_lat)
    S_plus = Vector{SparseMatrixCSC}([])

    for j in 1:N_lat
        rows = Vector{Int}([])
        cols = Vector{Int}([])
        data = Vector{Float64}([])

        for (i, state) in enumerate(basis)
            # state[j] corresponds to highest spin projection at site j, state[N_lat+j] - second highest at site j and so on.
            for n_s in 2:N_s  # Exclude n_s=1 because highest spin projection can't be raised higher.
                idx_higher = (n_s-2)*N_lat+j  # Higher spin projection state index
                idx_lower = (n_s-1)*N_lat+j  # Lower spin projection state index

                if state[idx_higher] == 0 && state[idx_lower] != 0
                    raised_state = copy(state)

                    m = m_s_arr[n_s]
                    α = sqrt(s*(s+1) - m*(m+1))

                    A = (-1)^sum(state[1:(idx_lower-1)])
                    raised_state[idx_lower] -= 1
                    A *= (-1)^sum(raised_state[1:(idx_higher-1)])
                    raised_state[idx_higher] += 1

                    append!(rows, basis_dict[raised_state])
                    append!(cols, i)
                    # sqrt(state[idx_lower])*sqrt(raised_state[idx_higher]) should be 1 (fermions).
                    append!(data, α*A*sqrt(state[idx_lower])*sqrt(raised_state[idx_higher]))
                end # if
            end # for
        end # for

        S_plus_j = sparse(rows, cols, data, length(basis), length(basis))
        push!(S_plus, S_plus_j)
    end # for

    return S_plus
end # function

function create_fermion_S_minus(basis, basis_dict, m_s_arr, s, N_s, N_lat)
    S_minus = Vector{SparseMatrixCSC}([])

    for j in 1:N_lat
        rows = Vector{Int}([])
        cols = Vector{Int}([])
        data = Vector{Float64}([])

        for (i, state) in enumerate(basis)
            # state[j] corresponds to highest spin projection at site j, state[N_lat+j] - second highest at site j and so on.
            for n_s in 1:(N_s-1)  # Exclude n_s=N_s because lowest spin projection can't be lowered further.
                idx_higher = (n_s-1)*N_lat+j  # Higher spin projection state index
                idx_lower = n_s*N_lat+j  # Lower spin projection state index

                if state[idx_lower] == 0 && state[idx_higher] != 0
                    lowered_state = copy(state)

                    m = m_s_arr[n_s+1]
                    α = sqrt(s*(s+1) - m*(m+1))

                    A = (-1)^sum(state[1:(idx_higher-1)])
                    lowered_state[idx_higher] -= 1
                    A *= (-1)^sum(lowered_state[1:(idx_lower-1)])
                    lowered_state[idx_lower] += 1

                    append!(rows, basis_dict[lowered_state])
                    append!(cols, i)
                    # sqrt(state[idx_lower])*sqrt(lowered_state[idx_higher]) should be 1 (fermions).
                    append!(data, α*A*sqrt(state[idx_higher])*sqrt(lowered_state[idx_lower]))
                end # if
            end # for
        end # for

        S_minus_j = sparse(rows, cols, data, length(basis), length(basis))
        push!(S_minus, S_minus_j)
    end # for

    return S_minus
end # function

function create_fermion_S_j_arrow_m_to_μ(j, m, μ, basis, basis_dict, m_s_arr, s, N_lat)
    @assert j <= N_lat

    @assert m ∈ m_s_arr
    @assert μ ∈ m_s_arr

    idx_m = findfirst(isequal(m), m_s_arr)
    idx_μ = findfirst(isequal(μ), m_s_arr)
    idx_initial = (idx_m-1)*N_lat+j
    idx_final = (idx_μ-1)*N_lat+j

    rows = Vector{Int}([])
    cols = Vector{Int}([])
    data = Vector{Float64}([])
    for (i, state) in enumerate(basis)
        # state[j] corresponds to highest spin projection at site j, state[N_lat+j] - second highest at site j and so on.
        if (state[idx_final] == 0 && state[idx_initial] != 0) ||
           (m == μ && state[idx_initial] != 0)
            exchanged_state = copy(state)

            A = (-1)^sum(state[1:(idx_initial-1)])
            exchanged_state[idx_initial] -= 1
            A *= (-1)^sum(exchanged_state[1:(idx_final-1)])
            exchanged_state[idx_final] += 1

            append!(rows, basis_dict[exchanged_state])
            append!(cols, i)
            # sqrt(state[idx_lower])*sqrt(exchanged_state[idx_higher]) should be 1 (fermions).
            append!(data, A*sqrt(state[idx_initial])*sqrt(exchanged_state[idx_final]))
        end # if
    end # for

    S_j_arrow_m_to_μ = sparse(rows, cols, data, length(basis), length(basis))

    return S_j_arrow_m_to_μ
end # function

function create_fermion_S_z_j(j, basis, basis_dict, m_s_arr, s, N_lat)
    S_z_j = spzeros(length(basis), length(basis))
    for m in m_s_arr
        S_z_j += m * create_fermion_S_j_arrow_m_to_μ(j, m, m, basis, basis_dict, m_s_arr, s, N_lat)
    end # for

    return S_z_j
end # function

function create_fermion_N_m(m, basis, basis_dict, m_s_arr, s, N_lat)
    N_m = spzeros(length(basis), length(basis))
    for j in 1:N_lat
        N_m += create_fermion_S_j_arrow_m_to_μ(j, m, m, basis, basis_dict, m_s_arr, s, N_lat)
    end # for

    return N_m
end # function

function create_fermion_N_ms(basis, basis_dict, m_s_arr, s, N_lat)
    return [create_fermion_N_m(m, basis, basis_dict, m_s_arr, s, N_lat) for m in m_s_arr]
end # function

function create_fermion_S_x(S_plus, S_minus)
    return 1/2 .* (S_plus .+ S_minus)
end # function

function create_fermion_S_y(S_plus, S_minus)
    return 1/(2*im) .* (S_plus .- S_minus)
end # function

function create_fermion_S_z(basis, basis_dict, m_s_arr, s, N_lat)
    return [create_fermion_S_z_j(j, basis, basis_dict, m_s_arr, s, N_lat) for j ∈ 1:N_lat]
end # function

function create_fermion_S_plus_j(j, basis, basis_dict, m_s_arr, s, N_lat)
    S_plus_j = spzeros(length(basis), length(basis))
    for m in m_s_arr[2:end]
        α = sqrt(s*(s+1) - m*(m+1))
        S_plus_j += α * create_fermion_S_j_arrow_m_to_μ(j, m, m+1, basis, basis_dict, m_s_arr, s, N_lat)
    end # for

    return S_plus_j
end # function

function create_fermion_S_minus_j(j, basis, basis_dict, m_s_arr, s, N_lat)
    S_minus_j = spzeros(length(basis), length(basis))
    for m in m_s_arr[2:end]
        α = sqrt(s*(s+1) - m*(m+1))
        S_minus_j += α * create_fermion_S_j_arrow_m_to_μ(j, m+1, m, basis, basis_dict, m_s_arr, s, N_lat)
    end # for

    return S_minus_j
end # function
########## END OF OPERATORS ##########

########## STATES ##########
function create_max_state(N_s, N_lat, basis, basis_dict)
    occ_arr = zeros(Int8, N_s*N_lat)
    occ_arr[1:N_lat] = fill(1, (N_lat,))

    max_state = zeros(ComplexF64, length(basis))
    # max_state[1] = 1.0  # This is simpler, but it relies on particular state ordering
    max_state[basis_dict[occ_arr]] = 1.0

    return max_state
end # function

function create_CSS_initial_wf(N_s, N_lat, θ_CSS, ϕ_CSS, basis, basis_dict, collective_S_y, collective_S_z)
    wf_initial = create_max_state(N_s, N_lat, basis, basis_dict)

    expmv!(-im*θ_CSS, collective_S_y, wf_initial)
    expmv!(-im*ϕ_CSS, collective_S_z, wf_initial)

    return wf_initial
end # function
########## END OF STATES ##########

########## HAMILTONIANS ##########
function calc_fermion_Ham_int(basis, N_s, N_lat, U)
    diag = Vector{Float64}(undef, length(basis))
    for (i, state) in enumerate(basis)
        total_int = 0.0

        for j in 1:N_lat
            n_j = 0
            for idx_m in 1:N_s
                n_j += state[(idx_m-1)*N_lat+j]
            end # for

            total_int += U/2 * n_j*(n_j-1)
        end # for

        diag[i] = total_int
    end # for

    return spdiagm(0 => diag)
end # function

"""Assumes PBC."""
function calc_fermion_Ham_tunneling(basis, basis_dict, N_s, N_lat, J)
    rows = Vector{Int}([])
    cols = Vector{Int}([])
    data = Vector{Float64}([])

    for (i, state) in enumerate(basis)
        for s in 0:(N_s-1)  # σ ∈ [s, s-1, s-2, ..., -s]
            for j in 1:N_lat
                α = s*N_lat+j
                if state[α] != 0
                    for pm in [-1, 1]
                        β = s*N_lat+mod(j+pm-1, N_lat)+1
                        if state[β] == 0
                            neighbour_state = copy(state)

                            A = (-1)^sum(state[1:(α-1)])
                            neighbour_state[α] -= 1
                            A *= (-1)^sum(neighbour_state[1:(β-1)])
                            neighbour_state[β] += 1

                            append!(rows, basis_dict[neighbour_state])
                            append!(cols, i)
                            append!(data, -A*J*sqrt(state[α])*sqrt(neighbour_state[β]))
                        end # if
                    end # for
                end # if
            end # for
        end # for
    end # for

    return sparse(rows, cols, data, length(basis), length(basis))
end # function

function calc_Ham_SOC(S_plus, S_minus, N_lat, whichSOC, J_SOC, ϕ_SOC, ϕ_0_SOC)
    Ham_SOC = spzeros(ComplexF64, size(S_plus[1]))
    for j in 1:N_lat
        if whichSOC == "Exp"
            Ham_SOC += J_SOC *
                (exp(im*(ϕ_SOC*j-ϕ_0_SOC)) * S_plus[j] +
                exp(-im*(ϕ_SOC*j-ϕ_0_SOC)) * S_minus[j])

            # S_x_j = (S_plus[j] + S_minus[j]) / 2
            # S_y_j = (S_plus[j] - S_minus[j]) / (2im)
            # Ham_SOC += 2*J_SOC *
            #     (cos(ϕ_SOC*j-ϕ_0_SOC)*S_x_j - sin(ϕ_SOC*j-ϕ_0_SOC)*S_y_j)
            # Ham_SOC += 2*J_SOC *
            #     (-cos(ϕ_SOC*j-ϕ_0_SOC)*S_x_j + sin(ϕ_SOC*j-ϕ_0_SOC)*S_y_j)
        elseif whichSOC == "Sin1"
            S_z_j = comm(S_plus[j], S_minus[j])/2
            Ham_SOC += J_SOC * sin(j*ϕ_SOC) * S_z_j
        elseif whichSOC == "Sin1"
            S_z_j = comm(S_plus[j], S_minus[j])/2
            Ham_SOC += J_SOC * sin(j*ϕ_SOC) * S_z_j^2
        elseif whichSOC == "ExpTensor"
            S_z_j = comm(S_plus[j], S_minus[j])/2
            b = 0.5
            Ham_SOC += J_SOC * (
                exp(im*(ϕ_SOC*j-ϕ_0_SOC)) * (S_plus[j] + b*anticomm(S_plus[j], S_z_j)) +
                exp(-im*(ϕ_SOC*j-ϕ_0_SOC)) * (S_minus[j] + b*anticomm(S_minus[j], S_z_j))
            )
        else
            throw(ArgumentError("Invalid whichSOC. Change variable whichSOC to one of the possible options. Possible options are given in main."))
        end # if
    end # for

    return Ham_SOC
end # function

function calc_PBC_FHM_Ham(basis, basis_dict, S_plus, S_minus, N_s, N_lat, whichSOC, J, U, Ω, ϕ_SOC, ϕ_0_SOC)
    return calc_fermion_Ham_tunneling(basis, basis_dict, N_s, N_lat, J) +
           calc_fermion_Ham_int(basis, N_s, N_lat, U) +
           calc_Ham_SOC(S_plus, S_minus, N_lat, whichSOC, Ω/2, ϕ_SOC, ϕ_0_SOC)
end # function

function calc_PBC_SE_Ham(basis, basis_dict, S_plus, S_minus, m_s_arr, s, N_lat, whichSOC, J_SE, J_SOC, ϕ_SOC, ϕ_0_SOC)
    Ham = spzeros(length(basis), length(basis))
    for j in 1:N_lat
        # b = -0.8
        # r = rand()
        for m in m_s_arr
            for μ in m_s_arr
                if m != μ
                    S_j_arrow_m_to_m = create_fermion_S_j_arrow_m_to_μ(j, m, m, basis, basis_dict, m_s_arr, s, N_lat)
                    S_jp1_arrow_μ_to_μ = create_fermion_S_j_arrow_m_to_μ(j%N_lat+1, μ, μ, basis, basis_dict, m_s_arr, s, N_lat)
                    S_j_arrow_μ_to_m = create_fermion_S_j_arrow_m_to_μ(j, μ, m, basis, basis_dict, m_s_arr, s, N_lat)
                    S_jp1_arrow_m_to_μ = create_fermion_S_j_arrow_m_to_μ(j%N_lat+1, m, μ, basis, basis_dict, m_s_arr, s, N_lat)

                    Ham += J_SE * (-S_j_arrow_m_to_m*S_jp1_arrow_μ_to_μ + S_j_arrow_μ_to_m*S_jp1_arrow_m_to_μ)
                    # Ham += J_SE * (1.0 + b*r) * (-S_j_arrow_m_to_m*S_jp1_arrow_μ_to_μ + S_j_arrow_μ_to_m*S_jp1_arrow_m_to_μ)
                end # if
            end # for
        end # for
    end # for
    Ham += calc_Ham_SOC(S_plus, S_minus, N_lat, whichSOC, J_SOC, ϕ_SOC, ϕ_0_SOC)

    # for j in 1:N_lat
    #     Ham += -J_SE *
    #         (create_fermion_S_j_arrow_m_to_μ(j, 1/2, 1/2, basis, basis_dict, m_s_arr, s, N_lat) *
    #          create_fermion_S_j_arrow_m_to_μ(j%N_lat+1, -1/2, -1/2, basis, basis_dict, m_s_arr, s, N_lat))
    #     Ham += J_SE *
    #         (create_fermion_S_j_arrow_m_to_μ(j, -1/2, 1/2, basis, basis_dict, m_s_arr, s, N_lat) *
    #          create_fermion_S_j_arrow_m_to_μ(j%N_lat+1, 1/2, -1/2, basis, basis_dict, m_s_arr, s, N_lat))
    #     Ham += -J_SE *
    #         (create_fermion_S_j_arrow_m_to_μ(j, -1/2, -1/2, basis, basis_dict, m_s_arr, s, N_lat) *
    #          create_fermion_S_j_arrow_m_to_μ(j%N_lat+1, 1/2, 1/2, basis, basis_dict, m_s_arr, s, N_lat))
    #     Ham += J_SE *
    #         (create_fermion_S_j_arrow_m_to_μ(j, 1/2, -1/2, basis, basis_dict, m_s_arr, s, N_lat) *
    #          create_fermion_S_j_arrow_m_to_μ(j%N_lat+1, -1/2, 1/2, basis, basis_dict, m_s_arr, s, N_lat))
    # end # for

    # ### Yields Heisenberg XXX for spin 1/2
    # id = sparse(diagm(0 => [1.0 for j ∈ 1:length(basis)]))
    # for j in 1:N_lat
    #     Ham += J_SE *
    #         (create_fermion_S_j_arrow_m_to_μ(j, -1/2, 1/2, basis, basis_dict, m_s_arr, s, N_lat) *
    #          create_fermion_S_j_arrow_m_to_μ(j%N_lat+1, 1/2, -1/2, basis, basis_dict, m_s_arr, s, N_lat))
    #     Ham += J_SE *
    #         (create_fermion_S_j_arrow_m_to_μ(j, 1/2, -1/2, basis, basis_dict, m_s_arr, s, N_lat) *
    #          create_fermion_S_j_arrow_m_to_μ(j%N_lat+1, -1/2, 1/2, basis, basis_dict, m_s_arr, s, N_lat))
    #     Ham += 1/2*J_SE *
    #         (create_fermion_S_j_arrow_m_to_μ(j, 1/2, 1/2, basis, basis_dict, m_s_arr, s, N_lat) *
    #          create_fermion_S_j_arrow_m_to_μ(j%N_lat+1, 1/2, 1/2, basis, basis_dict, m_s_arr, s, N_lat))
    #     Ham += -1/2*J_SE *
    #         (create_fermion_S_j_arrow_m_to_μ(j, -1/2, -1/2, basis, basis_dict, m_s_arr, s, N_lat) *
    #          create_fermion_S_j_arrow_m_to_μ(j%N_lat+1, 1/2, 1/2, basis, basis_dict, m_s_arr, s, N_lat))
    #     Ham += -1/2*J_SE *
    #         (create_fermion_S_j_arrow_m_to_μ(j, 1/2, 1/2, basis, basis_dict, m_s_arr, s, N_lat) *
    #          create_fermion_S_j_arrow_m_to_μ(j%N_lat+1, -1/2, -1/2, basis, basis_dict, m_s_arr, s, N_lat))
    #     Ham += 1/2*J_SE *
    #         (create_fermion_S_j_arrow_m_to_μ(j, -1/2, -1/2, basis, basis_dict, m_s_arr, s, N_lat) *
    #          create_fermion_S_j_arrow_m_to_μ(j%N_lat+1, -1/2, -1/2, basis, basis_dict, m_s_arr, s, N_lat))
    #     Ham += -1/2*J_SE*id
    # end # for

    return Ham
end # function

function calc_PBC_Heisenberg_XXX_Ham(basis, basis_dict, m_s_arr, S_plus, S_minus, s, N_s, N_lat, whichSOC, J_SE, J_SOC, ϕ_SOC, ϕ_0_SOC)
    ### OPERATORS ###
    # id = sparse([j for j ∈ 1:length(basis)], [j for j ∈ 1:length(basis)], [1.0 for j ∈ 1:length(basis)])
    id = sparse(diagm(0 => [1.0 for j ∈ 1:length(basis)]))
    S_plus = create_fermion_S_plus(basis, basis_dict, m_s_arr, s, N_s, N_lat)
    S_minus = create_fermion_S_minus(basis, basis_dict, m_s_arr, s, N_s, N_lat)
    S_x = create_fermion_S_x(S_plus, S_minus)
    S_y = create_fermion_S_y(S_plus, S_minus)
    S_z = create_fermion_S_z(basis, basis_dict, m_s_arr, s, N_lat)
    ### END OF OPERATORS ###

    Ham = 2J_SE .* sum([S_x[j]*S_x[j%N_lat+1] + S_y[j]*S_y[j%N_lat+1] + S_z[j]*S_z[j%N_lat+1] - 1/4*id for j ∈ 1:N_lat])
    Ham += calc_Ham_SOC(S_plus, S_minus, N_lat, whichSOC, J_SOC, ϕ_SOC, ϕ_0_SOC)

    return Ham
end # function

function calc_OAT_Ham(Ω, ϕ_SOC, J_SE, N_lat, s, collective_S_z)
    # WARNING: Taken from Hubert's derivation, may be incorrect.
    # χ = Ω^2 / (4*(1-cos(ϕ_SOC))) * 1/J_SE * 1/(2*N_lat*s - 1)

    χ = Ω^2 / (4*(1-cos(ϕ_SOC))) * 1/J_SE * 1/(2*N_lat*s - 1)
    # χ = Ω^2 / (16*(1-cos(ϕ_SOC))) * 1/J_SE * 1/(2*N_lat*s - 1) * (2*s*N_lat)
    # χ = Ω^2 / (4*(1-cos(ϕ_SOC))) * 1/J_SE * 1/(2*N_lat*s - 1) * 1.6
    # χ = Ω^2 / (4*(1-cos(ϕ_SOC))) * 1/J_SE * 1/(2*N_lat*s - 1)*(2*s)

    Ham = χ * collective_S_z^2

    return Ham
end # function
########## END OF HAMILTONIANS ##########

########## SQUEEZING ##########
function calc_expect_n_uvw(expect_S_x_arr, expect_S_y_arr, expect_S_z_arr, N_t)
    expect_n_u_arr = Array{Float64}(undef, 3, N_t)
    expect_n_v_arr = Array{Float64}(undef, 3, N_t)
    expect_n_w_arr = Array{Float64}(undef, 3, N_t)
    for idx_t in 1:N_t
        expect_n_u_arr[:, idx_t] = [expect_S_x_arr[idx_t], expect_S_y_arr[idx_t], expect_S_z_arr[idx_t]]
        expect_n_v_arr[:, idx_t], expect_n_w_arr[:, idx_t] = calc_orthogonal_vectors_3D(expect_n_u_arr[:, idx_t])
    end # for

    return expect_n_u_arr, expect_n_v_arr, expect_n_w_arr
end # function

function calc_squeezing_ξ_squared(squeezing_φ, ket_ψ, expect_n_v, expect_n_w, expect_S_squared, collective_S_x, collective_S_y, collective_S_z, s, N_lat)
    n_φ = sin(squeezing_φ)*expect_n_v + cos(squeezing_φ)*expect_n_w
    # S_n = n_φ[1]*ops.collective_S_x + n_φ[2]*ops.collective_S_y + n_φ[3]*ops.collective_S_z
    S_n = n_φ[1]*collective_S_x + n_φ[2]*collective_S_y + n_φ[3]*collective_S_z

    var_S_n = real(adjoint(ket_ψ) * S_n*S_n * ket_ψ - (adjoint(ket_ψ) * S_n * ket_ψ)^2)

    squeezing_ξ_squared = 2*s*N_lat * var_S_n / expect_S_squared

    return squeezing_ξ_squared
end # function

function calc_optimal_squeezing_ξ_squared(s, N_lat, N_t, collective_S_x, collective_S_y, collective_S_z, wf_arr, expect_n_v_arr, expect_n_w_arr, expect_S_squared_arr)
    squeezing_ξ_squared_arr = Array{Float64}(undef, N_t)
    for i in 1:N_t
        # f = squeezing_φ -> calc_squeezing_ξ_squared(squeezing_φ[1], wf_arr[:, i], expect_n_v_arr[:, i], expect_n_w_arr[:, i], expect_S_squared_arr[i], ops, par.N_lat)
        f = squeezing_φ -> calc_squeezing_ξ_squared(squeezing_φ[1], wf_arr[:, i], expect_n_v_arr[:, i], expect_n_w_arr[:, i], expect_S_squared_arr[i], collective_S_x, collective_S_y, collective_S_z, s, N_lat)
        # function g!(storage, squeezing_ϕ)
        #     Δϕ = 1E-5
        #     storage[1] = (f(squeezing_ϕ[1]+Δϕ) - f(squeezing_ϕ[1]-Δϕ)) / (2.0*Δϕ)
        # end # function
        # res = Optim.optimize(f, [0.0])
        res = Optim.optimize(f, 0, 2π)
        # res = Optim.optimize(f, g!, [0.0])
        squeezing_ξ_squared_arr[i] = Optim.minimum(res)
    end # for

    return squeezing_ξ_squared_arr
end # function

function full_calc_squeezing_ξ_squared(s, N_s, N_lat, θ_CSS, ϕ_CSS, N_t, t_arr, basis, basis_dict, Ham, collective_S_x, collective_S_y, collective_S_z; U_transformation=nothing)
    wf_initial = create_CSS_initial_wf(N_s, N_lat, θ_CSS, ϕ_CSS, basis, basis_dict, collective_S_y, collective_S_z)
    if U_transformation == nothing
        # size(wf_arr) = (length(basis), N_t)
        wf_arr = calc_time_evo(Ham, wf_initial, t_arr)
    else
        transformed_wf_initial = U_transformation' * wf_initial
        transformed_Ham = U_transformation' * Ham * U_transformation
        transformed_wf_arr = calc_time_evo(transformed_Ham, transformed_wf_initial, t_arr)
        wf_arr = Array{ComplexF64}(undef, length(wf_initial), length(t_arr))
        for i in 1:N_t
            wf_arr[:, i] = U_transformation * transformed_wf_arr[:, i]
        end # for
    end # if

    ##### EXPECTATION VALUES #####
    expect_S_x_arr = calc_1D_expectation_values(collective_S_x, wf_arr)
    expect_S_y_arr = calc_1D_expectation_values(collective_S_y, wf_arr)
    expect_S_z_arr = calc_1D_expectation_values(collective_S_z, wf_arr)
    expect_S_squared_arr = expect_S_x_arr.^2 + expect_S_y_arr.^2 + expect_S_z_arr.^2

    expect_n_u_arr, expect_n_v_arr, expect_n_w_arr = calc_expect_n_uvw(expect_S_x_arr, expect_S_y_arr, expect_S_z_arr, N_t)
    ##### END OF EXPECTATION VALUES #####

    squeezing_ξ_squared_arr = calc_optimal_squeezing_ξ_squared(s, N_lat, N_t, collective_S_x, collective_S_y, collective_S_z, wf_arr, expect_n_v_arr, expect_n_w_arr, expect_S_squared_arr)

    return squeezing_ξ_squared_arr
end # function
########## END OF SQUEEZING ##########

########## POPULATIONS ##########
function full_calc_manifold_populations(N_t, t_arr, basis, basis_dict, Ham_SE_no_SOC, Ham_SE, collective_S_squared, collective_S_z, collective_S_y; U_transformation=nothing)
    wf_initial = create_CSS_initial_wf(N_s, N_lat, θ_CSS, ϕ_CSS, basis, basis_dict, collective_S_y, collective_S_z)
    if U_transformation == nothing
        # size(wf_arr) = (length(basis), N_t)
        wf_arr = calc_time_evo(Ham_SE, wf_initial, t_arr)
    else
        transformed_wf_initial = U_transformation' * wf_initial
        transformed_Ham_SE = U_transformation' * Ham_SE * U_transformation
        transformed_wf_arr = calc_time_evo(transformed_Ham_SE, transformed_wf_initial, t_arr)
        wf_arr = Array{ComplexF64}(undef, length(wf_initial), length(t_arr))
        for i in 1:N_t
            wf_arr[:, i] = U_transformation * transformed_wf_arr[:, i]
        end # for
    end # if

    # WARNING: Only perform these calculations with single occupation basis. Full Fock basis can be too large.
    F = eigen(Hermitian(Matrix(Ham_SE_no_SOC + collective_S_squared + collective_S_z)))
    v = F.vectors

    # sparse_display(v'*Ham_SE*v)
    # sparse_display(v'*collective_S_squared*v)
    # sparse_display(v'*collective_S_z*v)

    # Operators must commute for there to be a common eigenbasis.
    @assert sum(abs.(comm(Ham_SE_no_SOC, collective_S_squared))) < 1e-10
    @assert sum(abs.(comm(Ham_SE_no_SOC, collective_S_z))) < 1e-10
    @assert sum(abs.(comm(collective_S_squared, collective_S_z))) < 1e-10

    # Make sure commuting operators are simultaneously diagonalized.
    @assert sum_non_diag(abs.(v'*Ham_SE_no_SOC*v)) / length(basis)^2 < 1e-10
    @assert sum_non_diag(abs.(v'*collective_S_squared*v)) / length(basis)^2 < 1e-10
    @assert sum_non_diag(abs.(v'*collective_S_z*v)) / length(basis)^2 < 1e-10

    unsorted_λ_S = diag(real.(v'*collective_S_squared*v))
    perm = reverse(sortperm(unsorted_λ_S))
    λ_S = round.(unsorted_λ_S[perm], digits=1)
    sorted_v = v[:, perm]
    countmap_λ_S = reverse(sort(collect(countmap(λ_S)), by=x->x[1]))
    N_S = length(countmap_λ_S)
    idx_vec = pushfirst!(cumsum([pair[2] for pair in countmap_λ_S]), 0)

    # println("***** SPINS ******")
    # # display(λ_S[1:16])
    # display(λ_S[1:20])
    # display(λ_S[21:40])
    # println("***** END OF SPINS ******")
    # display(idx_vec)

    # println("***** ENERGIES ******")
    # # unsorted_E = diag(real.(v'*Ham_SE_no_SOC*v))
    # unsorted_E = diag(real.(v'*Ham_SE*v))
    # perm = reverse(sortperm(unsorted_E))
    # E = round.(unsorted_E[perm], digits=10)
    # E_sorted_v = v[:, perm]
    # # display(E[1:16])
    # display(E[1:20])
    # display(E[21:40])
    # println("***** END OF ENERGIES ******")

    # @show countmap_λ_S
    # @show length(countmap_λ_S)
    # @show pushfirst!(cumsum([pair[2] for pair in countmap_λ_S]), 0)

    man_pop_arr = zeros(Float64, N_t, N_S)
    for i in 1:N_S
        for j in 1:N_t
            for k in (idx_vec[i]+1):idx_vec[i+1]
                man_pop_arr[j, i] += abs2((sorted_v[:, k])' * wf_arr[:, j])
            end # for
        end # for
    end # for

    return man_pop_arr
end # function

function full_calc_m_z_populations(N_s, m_s_arr, N_t, t_arr, basis, basis_dict, Ham, N_ms, collective_S_y, collective_S_z; U_transformation=nothing)
    wf_initial = create_CSS_initial_wf(N_s, N_lat, θ_CSS, ϕ_CSS, basis, basis_dict, collective_S_y, collective_S_z)
    if U_transformation == nothing
        # size(wf_arr) = (length(basis), N_t)
        wf_arr = calc_time_evo(Ham, wf_initial, t_arr)
    else
        transformed_wf_initial = U_transformation' * wf_initial
        transformed_Ham = U_transformation' * Ham * U_transformation
        transformed_wf_arr = calc_time_evo(transformed_Ham, transformed_wf_initial, t_arr)
        wf_arr = Array{ComplexF64}(undef, length(wf_initial), length(t_arr))
        for i in 1:N_t
            wf_arr[:, i] = U_transformation * transformed_wf_arr[:, i]
        end # for
    end # if

    m_z_pop_arr = zeros(Float64, N_t, N_s)
    for (i, m_z) in enumerate(m_s_arr)
        for j in 1:N_t
            m_z_pop_arr[j, i] = calc_expectation_value(N_ms[i], wf_arr[:, j])
        end # for
    end # for

    return m_z_pop_arr
end # function
########## END OF POPULATIONS ##########

############### RESTRICTED s=1 (SU(3)) ###############
# NOTE: Not fully general, many assumptions such as: *) s=1; *) PBC;
# *) ϕ_0_SOC = 0; *) ϕ_SOC commensurate; *) ϕ_SOC \neq π; *) Probably more that I forgot;

########## GENERAL ##########
function create_dicts_i_fromto_np(N_lat)
    # Not strictly necessary, it is just easier than finding inverse functions.
    # np states ordered as follows: |0,0⟩ → 1, |0,1⟩ → 2, ..., |0,N_lat⟩ → N_lat+1, |1,0⟩ → N_lat+2, ..., |M,0⟩ → (N_lat+1)(N_lat+2)/2;

    i = 1
    dict_i_to_np = Dict{Int, Tuple{Int, Int}}()
    for n in 0:N_lat
        for p in 0:(N_lat-n)
            push!(dict_i_to_np, i => (n, p))
            i = i + 1
        end # for
    end # for

    dict_np_to_i = invert_dict(dict_i_to_np)

    return dict_i_to_np, dict_np_to_i
end # function

function calc_Fock_to_np_transformation_matrix(np_states)
    # NOTE: That np_states and U are stored differently, but have exactly the
    # same matrix elements, so this function is simply for convenience.

    # Fock → np: ψ' = U'*ψ, H' = U'*H*U;
    # Function calculates U;

    # 0 ≤ n+p ≤ N_lat ⟹ No. of np states = (N_lat+1)(N_lat+2)/2
    # np states ordered as follows: |0,0⟩ → 1, |0,1⟩ → 2, ..., |0,N_lat⟩ → N_lat+1, |1,0⟩ → N_lat+2, ..., |M,0⟩ → (N_lat+1)(N_lat+2)/2;

    # U_ij = ⟨Fock|np⟩; U'_ij = ⟨np|Fock⟩;
    U = zeros(ComplexF64, length(np_states[1]), length(np_states))
    for (j, np_state) in enumerate(np_states)
        U[:, j] = np_state
    end # for

    return U
end # function

function calc_effective_dynamics_matrices(N_s, N_lat, J, U, Ω, ϕ_SOC, basis, basis_dict, collective_S1_minus, collective_S3_minus)
    ##### EFFECTIVE DYNAMICS #####
    np_states = create_np_states(N_s, N_lat, basis, basis_dict, collective_S1_minus, collective_S3_minus)
    dict_i_to_np, dict_np_to_i = create_dicts_i_fromto_np(N_lat)
    U_F_to_np = calc_Fock_to_np_transformation_matrix(np_states)

    N_np = length(np_states)

    ### EFFECTIVE HAMILTONIAN ###
    energy_gap = 4*J^2/U * (1 - cos(ϕ_SOC))
    amp_Ham = Ω^2/(2*energy_gap)

    Ham_eff = zeros(ComplexF64, N_np, N_np)
    for i in 1:N_np
        n, p = dict_i_to_np[i]

        # DIAGONAL ELEMENTS
        g1_np = calc_g1_np(n, p, N_lat)
        g2_np = calc_g2_np(n, p, N_lat)

        Ham_eff[i, i] += -amp_Ham * (g1_np^(-2) + g2_np^(-2) - (N_lat - p - 2*n))
        # END OF DIAGONAL ELEMENTS

        # OFF-DIAGONAL ELEMENTS
        if n + p + 2 <= N_lat
            i_from_np1_p = dict_np_to_i[(n+1, p)]
            i_from_n_pp2 = dict_np_to_i[(n, p+2)]

            off_diag_mul = -2.0/(N_lat-1.0) * sqrt((n+1)*(p+1)*(p+2)*(N_lat-n-p-1))

            Ham_eff[i_from_np1_p, i_from_n_pp2] += -amp_Ham * off_diag_mul
            Ham_eff[i_from_n_pp2, i_from_np1_p] += -amp_Ham * off_diag_mul
        end # if
        # END OF OFF-DIAGONAL ELEMENTS
    end # for
    ### END OF EFFECTIVE HAMILTONIAN ###
    ##### END OF EFFECTIVE DYNAMICS #####

    return Ham_eff, U_F_to_np
end # function
########## END OF GENERAL ##########

########## OPERATORS ##########
function create_fermion_restricted_n_s_S_plus(basis, basis_dict, n_s, m_s_arr, s, N_s, N_lat)
    restricted_n_s_S_plus = Vector{SparseMatrixCSC}([])

    for j in 1:N_lat
        rows = Vector{Int}([])
        cols = Vector{Int}([])
        data = Vector{Float64}([])

        for (i, state) in enumerate(basis)
            # state[j] corresponds to highest spin projection at site j, state[N_lat+j] - second highest at site j and so on.
            idx_higher = (n_s-1)*N_lat+j  # Higher spin projection state index
            idx_lower = n_s*N_lat+j  # Lower spin projection state index

            if state[idx_higher] == 0 && state[idx_lower] != 0
                raised_state = copy(state)

                m = m_s_arr[n_s+1]
                α = sqrt(s*(s+1) - m*(m+1))

                A = (-1)^sum(state[1:(idx_lower-1)])
                raised_state[idx_lower] -= 1
                A *= (-1)^sum(raised_state[1:(idx_higher-1)])
                raised_state[idx_higher] += 1

                append!(rows, basis_dict[raised_state])
                append!(cols, i)
                # sqrt(state[idx_lower])*sqrt(raised_state[idx_higher]) should be 1 (fermions).
                append!(data, α*A*sqrt(state[idx_lower])*sqrt(raised_state[idx_higher]))
            end # if
        end # for

        restricted_n_s_S_plus_j = sparse(rows, cols, data, length(basis), length(basis))
        push!(restricted_n_s_S_plus, restricted_n_s_S_plus_j)
    end # for

    return restricted_n_s_S_plus
end # function

function create_fermion_restricted_n_s_S_minus(basis, basis_dict, n_s, m_s_arr, s, N_s, N_lat)
    restricted_n_s_S_minus = Vector{SparseMatrixCSC}([])

    for j in 1:N_lat
        rows = Vector{Int}([])
        cols = Vector{Int}([])
        data = Vector{Float64}([])

        for (i, state) in enumerate(basis)
            # state[j] corresponds to highest spin projection at site j, state[N_lat+j] - second highest at site j and so on.
            idx_higher = (n_s-1)*N_lat+j  # Higher spin projection state index
            idx_lower = n_s*N_lat+j  # Lower spin projection state index

            if state[idx_lower] == 0 && state[idx_higher] != 0
                lowered_state = copy(state)

                m = m_s_arr[n_s+1]
                α = sqrt(s*(s+1) - m*(m+1))

                A = (-1)^sum(state[1:(idx_higher-1)])
                lowered_state[idx_higher] -= 1
                A *= (-1)^sum(lowered_state[1:(idx_lower-1)])
                lowered_state[idx_lower] += 1

                append!(rows, basis_dict[lowered_state])
                append!(cols, i)
                # sqrt(state[idx_lower])*sqrt(lowered_state[idx_higher]) should be 1 (fermions).
                append!(data, α*A*sqrt(state[idx_higher])*sqrt(lowered_state[idx_lower]))
            end # if
        end # for

        restricted_n_s_S_minus_j = sparse(rows, cols, data, length(basis), length(basis))
        push!(restricted_n_s_S_minus, restricted_n_s_S_minus_j)
    end # for

    return restricted_n_s_S_minus
end # function

function create_fermion_restricted_n_s_S_z(restricted_n_s_S_plus, restricted_n_s_S_minus)
    N_lat = length(restricted_n_s_S_plus)
    restricted_n_s_S_z = Vector{SparseMatrixCSC}([])
    for j in 1:N_lat
        push!(restricted_n_s_S_z, 0.5*comm(restricted_n_s_S_plus[j], restricted_n_s_S_minus[j]))
    end # for

    return restricted_n_s_S_z
end # function

function create_O_ϕ_from_O_j(O_j_vec, ϕ; ϕ_0=0.0)
    O_ϕ = zero(O_j_vec[1])
    for (j, O_j) in enumerate(O_j_vec)
        O_ϕ += exp(-im*(ϕ*j-ϕ_0)) * O_j
    end # for

    return O_ϕ
end # function
########## END OF OPERATORS ##########

########## STATES ##########
function calc_d_np(n, p, M)
    return sqrt(2.0^(-p-n) * factorial(M-p-n)/(factorial(M)*factorial(n)*factorial(p)))
end # function

function calc_g1_np(n, p, M)
    return sqrt((M - 1) / (2 * (M - p - 1) * (M - p - n)))
end # function

function calc_g2_np(n, p, M)
    return sqrt((M - 1) / (2 * p * (M - n - 1)))
end # function

function calc_L_npα(n, p, M)
    return -sqrt(n*p / ((M-n) * (M-p)))
end # function

function create_generalized_Dicke_np_state(n, p, N_s, N_lat, basis, basis_dict, collective_S1_minus, collective_S3_minus)
    np_state = create_max_state(N_s, N_lat, basis, basis_dict)

    for i in 1:p
        np_state = collective_S1_minus * np_state
    end # for
    for i in 1:n
        np_state = collective_S3_minus * np_state
    end # for

    d_np = calc_d_np(n, p, N_lat)
    np_state *= d_np

    return np_state
end # function

function create_np_states(N_s, N_lat, basis, basis_dict, collective_S1_minus, collective_S3_minus)
    # np states ordered as follows: |0,0⟩ → 1, |0,1⟩ → 2, ..., |0,N_lat⟩ → N_lat+1, |1,0⟩ → N_lat+2, ..., |M,0⟩ → (N_lat+1)(N_lat+2)/2;
    np_states = Vector{Vector{ComplexF64}}([])
    for n in 0:N_lat
        for p in 0:(N_lat-n)
            np_state = create_generalized_Dicke_np_state(n, p, N_s, N_lat, basis, basis_dict, collective_S1_minus, collective_S3_minus)
            push!(np_states, np_state)
        end # for
    end # for

    return np_states
end # function
########## END OF STATES ##########
############### END OF RESTRICTED s=1 (SU(3)) ###############

########## FISHER INFORMATION ##########
########## END OF FISHER INFORMATION ##########
end # module
