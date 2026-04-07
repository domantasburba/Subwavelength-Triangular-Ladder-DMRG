using Dates
using ITensors
using ITensorMPS
using LinearAlgebra
using NPZ
using HDF5
using Random

import PyPlot as plt
ENV["MPLBACKEND"] = "Qt5Agg"
rcParams = plt.PyDict(plt.matplotlib."rcParams")
rcParams["figure.figsize"] = [12, 8]
rcParams["font.size"] = 16
set_interactive_plt = bool -> (bool ? plt.ion() : plt.ioff())
set_interactive_plt(true)  # true - shows plots; false - doesn't show plots
# plt.ioff()  # Doesn't show plots
# plt.ion()  # Shows plots

# ### https://itensor.github.io/ITensors.jl/stable/Multithreading.html
# # @show BLAS.get_config()
# # BLAS.set_num_threads(22)
# BLAS.set_num_threads(1)
# ITensors.Strided.set_num_threads(1)
# ITensors.enable_threaded_blocksparse(false)

using QOPack
using QOPack.DMRG
using QOPack.QMBPhaseDiagram

mutable struct XXZHamiltonianParameters
    N_lat::Int64

    conserve_Sz::Bool
    h::Float64

    J::Float64
    Δ::Float64

    N_sweeps::Int64
    max_dim::Int64
    cutoff::Float64
    noise
end # function

##### OPERATORS #####
function ITensors.op(::OpName"id", ::SiteType"Boson", d::Int)
    o = zeros(Float64, d, d)
    for idx in 1:d
        o[idx, idx] = 1.0
    end # for

    return o
end # function

function ITensors.op(::OpName"adag2", ::SiteType"Boson", d::Int)
    o = zeros(Float64, d, d)
    for i in 0:(d-3)
        idx = i+1
        o[idx+2, idx] = sqrt(i+1) * sqrt(i+2)
    end # for

    return o
end # function

function ITensors.op(::OpName"a2", ::SiteType"Boson", d::Int)
    o = zeros(Float64, d, d)
    for i in 2:(d-1)
        idx = i+1
        o[idx-2, idx] = sqrt(i) * sqrt(i-1)
    end # for

    return o
end # function

"""Projection operator, which acts as identity for occupation numbers 0 and 2."""
function ITensors.op(::OpName"proj_Q", ::SiteType"Boson", d::Int)
    o = zeros(Float64, d, d)
    o[1, 1] = 1.0
    o[3, 3] = 1.0
    # for idx in 3:d
    #     o[idx, idx] = 1.0
    # end # for

    return o
end # function

# WARNING: Should be changed with respect to lattice size.
"""Projection operator, which acts as identity for occupation numbers 0 and 2
and gives some weight less than unity to occupation number 1. This metric should
be more stable w.r.t. defects in DMRG numerics and also should give lattice size
independent results."""
function ITensors.op(::OpName"proj_Q_mod", ::SiteType"Boson", d::Int)
    o = zeros(Float64, d, d)
    o[1, 1] = 1.0
    o[2, 2] = exp(-0.1/4)  # exp(-α_Q/N_lat)
    o[3, 3] = 1.0

    return o
end # function

"""Projection operator, which acts as identity for occupation numbers 0 and 1."""
function ITensors.op(::OpName"proj_P", ::SiteType"Boson", d::Int)
    o = zeros(Float64, d, d)
    o[1, 1] = 1.0
    o[2, 2] = 1.0
    # for idx in 3:d
    #     o[idx, idx] = 1.0
    # end # for

    return o
end # function

function ITensors.op(::OpName"exp_iπn", ::SiteType"Boson", d::Int)
    o = zeros(Float64, d, d)
    for idx in 1:d
        o[idx, idx] = (-1.0)^(idx-1)
    end # for

    return o
end # function

function ITensors.op(::OpName"exp_iπnm1", ::SiteType"Boson", d::Int)
    o = zeros(Float64, d, d)
    for idx in 1:d
        o[idx, idx] = (-1.0)^(idx-2)
    end # for

    return o
end # function

function calc_prod_expectation_value(ψ, sites, N_lat, op_name, op_range)
    if op_range[1] == 1
        os = OpSum() + (op_name, 1)
    else
        os = OpSum() + ("id", 1)
    end # if

    for i in 2:(op_range[1]-1)
        os_i = OpSum() + ("id", i)
        os *= os_i
    end # for
    for i in max(op_range[1], 2):(op_range[end])
        os_i = OpSum() + (op_name, i)
        os *= os_i
    end # for
    for i in (op_range[end]+1):N_lat
        os_i = OpSum() + ("id", i)
        os *= os_i
    end # for
    prod_mpo = MPO(Ops.expand(os), sites)
    prod_expect = inner(ψ', prod_mpo, ψ)

    return prod_expect
end # function

# TODO UNFINISHED
function calc_string_order(ψ, sites, N_lat, op_range)
    if op_range[1] == 1
        os = OpSum() + ("exp_iπnm1", 1)
    else
        os = OpSum() + ("id", 1)
    end # if

    # TODO UNFINISHED
    for i in 2:(op_range[1]-1)
        os_i = OpSum() + ("id", i)
        os *= os_i
    end # for
    for i in max(op_range[1], 2):(op_range[end])
        os_i = OpSum() + ("exp_iπnm1", i)
        os *= os_i
    end # for
    # TODO UNFINISHED
    for i in (op_range[end]+1):N_lat
        os_i = OpSum() + ("id", i)
        os *= os_i
    end # for
    prod_mpo = MPO(Ops.expand(os), sites)
    prod_expect = inner(ψ', prod_mpo, ψ)

    return prod_expect
end # function
##### END OF OPERATORS #####

function select_par_by_sweep_no(i_sweep, par_vec)
    return par_vec[min(i_sweep, length(par_vec))]
end # function

##### STATE CREATION #####
function try_add_particle(part_arr, boson_dim)
    j = rand(1:length(part_arr))
    max_bosons = boson_dim-1
    if part_arr[j] < max_bosons
        part_arr[j] += 1
        return part_arr
    else
        try_add_particle(part_arr, boson_dim)
    end # if
end # function

function create_ψ0(sites, N_lat, N_part, boson_dim; linkdims=250)
    part_arr = zeros(Int64, N_lat)
    ii = max(N_lat÷N_part, 1)
    for i in 1:N_part
        # part_arr[mod1(ii*i, N_lat)] += 1
        # part_arr[rand(1:N_lat)] += 1
        part_arr = try_add_particle(part_arr, boson_dim)
    end # for
    state = string.(part_arr)
    # ψ0 = randomMPS(sites, state, 250)
    ψ0 = randomMPS(sites, state, linkdims)
    # ψ0 = random_mps(sites, state; linkdims=250)

    # state = [isodd(n) ? "1" : "0" for n in 1:N_lat]
    # ψ0 = MPS(sites, state)

    return ψ0
end # function

# TODO Choose magnetization sector, for now always S_z=0
function create_ψ0_spin(par, sites)
    N_lat, conserve_Sz, max_dim = par.N_lat, par.conserve_Sz, par.max_dim

    if conserve_Sz
        # state = [isodd(j) ? "Up" : "Dn" for j in 1:N_lat]
        # state = vcat(["Up" for j in 1:(N_lat÷2)], ["Dn" for j in (N_lat÷2+1):N_lat])
        state = fill("Up", N_lat)
        for j in (shuffle(1:N_lat)[1:(N_lat÷2)])
            state[j] = "Dn"
        end # for
        ψ0 = randomMPS(sites, state; linkdims=max_dim)
    else
        ψ0 = randomMPS(sites; linkdims=max_dim)
    end # if

    return ψ0
end # function

function create_homogeneous_ψ0(sites, N_lat, N_part, boson_dim)
    part_arr = zeros(Int64, N_lat)
    ii = max(N_lat÷N_part, 1)
    for i in 1:N_part
        # part_arr[mod1(ii*i, N_lat)] += 1
        part_arr[rand(1:N_lat)] += 1
        # part_arr = try_add_particle(part_arr, boson_dim)
    end # for
    state = string.(part_arr)
    ψ0 = randomMPS(sites, state, 250)
    # ψ0 = random_mps(sites, state; linkdims=250)

    # state = [isodd(n) ? "1" : "0" for n in 1:N_lat]
    # ψ0 = MPS(sites, state)

    return ψ0
end # function

function create_pair_ψ0(sites, N_lat, N_part, boson_dim; linkdims=100)
    part_arr = zeros(Int64, N_lat)
    ii = max(N_lat÷N_part, 1)
    for i in 1:(N_part÷2)
        part_arr[2*i] += 2
    end # for
    if isodd(N_part)
        part_arr[2] += 1
    end # for
    state = string.(part_arr)
    ψ0 = randomMPS(sites, state, linkdims)
    # ψ0 = random_mps(sites, state; linkdims=250)

    return ψ0
end # function

function create_Mott_ψ0(sites, N_lat, N_part; createRandom=false)
    part_arr = zeros(Int64, N_lat)
    for i in 1:N_part
        part_arr[mod1(i, N_lat)] += 1
    end # for
    state = string.(part_arr)
    if createRandom
        ψ_Mott = randomMPS(sites, state, 250)
    else
        ψ_Mott = MPS(sites, state)
    end # if

    return ψ_Mott
end # function
##### END OF STATE CREATION #####

##### HAMILTONIANS #####
function calc_BH_Ham(sites, N_lat, conserveParticleNumber, μ; J, U)
    os = OpSum()
    for i in 1:(N_lat-1)
    # for i in 1:N_lat
        os += -J, "adag", i, "a", mod1(i+1, N_lat)
        os += -J, "adag", mod1(i+1, N_lat), "a", i
    end # for
    for i in 1:N_lat
        os += U/2, "n", i, "n", i
        os += -U/2, "n", i
    end # for
    if !conserveParticleNumber
        for i in 1:N_lat
            os += -μ, "n", i
        end # for
    end # if
    Ham = MPO(os, sites)

    return Ham
end # function

function calc_brick_wall_Ham(sites, N_lat::Int, conserveParticleNumber, μ; J_1, J_2, g_0, g_x, g_z, G_000, G_001, G_011)
    os = OpSum()
    for i in 1:(N_lat-1)
        os += -J_1, "adag", i, "a", i+1
        os += -J_1, "adag", i+1, "a", i
    end # for
    for i in 1:(N_lat-2)
        os += -J_2, "adag", i, "a", i+2
        os += -J_2, "adag", i+2, "a", i
    end # for
    for i in 1:N_lat
        os += (g_0 + 0.5*g_x)*G_000, "n", i, "n", i
        os += -(g_0 + 0.5*g_x)*G_000, "n", i

        # A = 1e3
        # os += 2*A, "n", i
        # os += -A, "n", i, "n", i
    end # for
    for i in 1:(N_lat-1)
        os += 2*g_0*G_011, "n", i, "n", i+1

        os += g_z*G_001, "adag", i+1, "adag", i+1, "a", i+1, "a", i
        os += g_z*G_001, "adag", i, "adag", i, "a", i, "a", i+1
        os += g_z*G_001, "adag", i, "adag", i+1, "a", i+1, "a", i+1
        os += g_z*G_001, "adag", i+1, "adag", i, "a", i, "a", i

        os += -0.5*g_x*G_011, "adag", i+1, "adag", i+1, "a", i, "a", i
        os += -0.5*g_x*G_011, "adag", i, "adag", i, "a", i+1, "a", i+1
    end # for
    if !conserveParticleNumber
        for i in 1:N_lat
            os += -μ, "n", i
        end # for
    end # if
    Ham = MPO(os, sites)

    return Ham
end # function

function calc_brick_wall_Ham(sites, lat_range::UnitRange{Int}, conserveParticleNumber, μ; J_1, J_2, g_0, g_x, g_z, G_000, G_001, G_011)
    os = OpSum()
    for i in (lat_range[1]):(lat_range[end]-1)
        os += -J_1, "adag", i, "a", i+1
        os += -J_1, "adag", i+1, "a", i
    end # for
    for i in (lat_range[1]):(lat_range[end]-2)
        os += -J_2, "adag", i, "a", i+2
        os += -J_2, "adag", i+2, "a", i
    end # for
    for i in lat_range
        os += (g_0 + 0.5*g_x)*G_000, "n", i, "n", i
        os += -(g_0 + 0.5*g_x)*G_000, "n", i
    end # for
    for i in (lat_range[1]):(lat_range[end]-1)
        os += 2*g_0*G_011, "n", i, "n", i+1

        os += g_z*G_001, "adag", i+1, "adag", i+1, "a", i+1, "a", i
        os += g_z*G_001, "adag", i, "adag", i, "a", i, "a", i+1
        os += g_z*G_001, "adag", i, "adag", i+1, "a", i+1, "a", i+1
        os += g_z*G_001, "adag", i+1, "adag", i, "a", i, "a", i

        os += -0.5*g_x*G_011, "adag", i+1, "adag", i+1, "a", i, "a", i
        os += -0.5*g_x*G_011, "adag", i, "adag", i, "a", i+1, "a", i+1
    end # for
    if !conserveParticleNumber
        for i in lat_range
            os += -μ, "n", i
        end # for
    end # if
    Ham = MPO(os, sites)

    return Ham
end # function

function choose_brick_wall_Ham(whichBand, sites, N_lat::Int, conserveParticleNumber, μ; abs_J1, g_0, g_x, g_z)
    if whichBand == "s"
        Ham = calc_brick_wall_Ham(sites, N_lat, conserveParticleNumber, μ;
            J_1=abs_J1, J_2=-abs_J1*(10^(0.5521)), g_0=g_0, g_x=g_x, g_z=g_z, G_000=1.0000, G_001=-0.2122, G_011=0.1666)  # s-band
    elseif whichBand == "p"
        Ham = calc_brick_wall_Ham(sites, N_lat, conserveParticleNumber, μ;
            J_1=abs_J1, J_2=abs_J1*(10^(0.375)), g_0=g_0, g_x=g_x, g_z=g_z, G_000=1.0000, G_001=-0.5000, G_011=0.5000)  # p-band
    else
        throw(ArgumentError("Invalid whichBand."))
    end # if
end # function

function choose_brick_wall_Ham(whichBand, sites, lat_range::UnitRange{Int}, conserveParticleNumber, μ; abs_J1, g_0, g_x, g_z)
    if whichBand == "s"
        Ham = calc_brick_wall_Ham(sites, lat_range, conserveParticleNumber, μ;
            # J_1=-abs_J1, J_2=abs_J1*(10^(0.5521)), g_0=g_0, g_x=g_x, g_z=g_z, G_000=1.0000, G_001=-0.2122, G_011=0.1666)  # s-band
            J_1=abs_J1, J_2=-abs_J1*(10^(0.5521)), g_0=g_0, g_x=g_x, g_z=g_z, G_000=1.0000, G_001=-0.2122, G_011=0.1666)  # s-band
    elseif whichBand == "p"
        Ham = calc_brick_wall_Ham(sites, lat_range, conserveParticleNumber, μ;
            # J_1=-abs_J1, J_2=-abs_J1*(10^(0.375)), g_0=g_0, g_x=g_x, g_z=g_z, G_000=1.0000, G_001=-0.5000, G_011=0.5000)  # p-band
            J_1=abs_J1, J_2=abs_J1*(10^(0.375)), g_0=g_0, g_x=g_x, g_z=g_z, G_000=1.0000, G_001=-0.5000, G_011=0.5000)  # p-band
    else
        throw(ArgumentError("Invalid whichBand."))
    end # if
end # function

# NOTE: Zhou considered MFT.
function calc_Zhou_PSF_deep_lat_Ham(sites, N_lat, conserveParticleNumber, μ; J, U)
    # ZhouPRA2009
    os = OpSum()
    for i in 1:(N_lat-1)
    # for i in 1:N_lat
        # os += -1e-5, "adag", i, "a", i+1
        # os += -1e-5, "adag", i+1, "a", i
        os += -J, "adag", i, "adag", i, "a", mod1(i+1,N_lat), "a", mod1(i+1,N_lat)
        os += -J, "adag", mod1(i+1,N_lat), "adag", mod1(i+1,N_lat), "a", i, "a", i
    end # for
    for i in 1:N_lat
        os += U/2, "n", i, "n", i
        os += -U/2, "n", i

        # W = 1.0
        # os += W/6, "n", i, "n", i, "n", i
        # os += -3*W/6, "n", i, "n", i
        # os += 2*W/6, "n", i
    end # for
    if !conserveParticleNumber
        for i in 1:N_lat
            os += -μ, "n", i
        end # for
    end # if
    Ham = MPO(os, sites)

    return Ham
end # function

# WARNING: CHANGED
function calc_Huber_interacting_Creutz_ladder(sites, N_lat, conserveParticleNumber, μ; t, U, m, ϵ, δ)
    # HuberPRB2013
    os = OpSum()
    for i in 1:N_lat
        os += U/4, "n", i, "n", i
        os += -U/4, "n", i
    end # for
    for i in 1:(N_lat-1)
        os += U/2, "n", i, "n", i+1
    end # for
    for i in 1:(N_lat-1)
    # for i in 1:N_lat
        os += -U/8, "adag", i, "adag", i, "a", mod1(i+1,N_lat), "a", mod1(i+1,N_lat)
        os += -U/8, "adag", mod1(i+1,N_lat), "adag", mod1(i+1,N_lat), "a", i, "a", i
    end # for
    for i in 1:(N_lat-1)
        os += -(m+2δ)*t/2, "adag", i, "a", i+1
        os += -(m+2δ)*t/2, "adag", i+1, "a", i
    end # for
    for i in 1:(N_lat-2)
        os += -(ϵ+0.5*δ^2)*t/2, "adag", i, "a", i+2
        os += -(ϵ+0.5*δ^2)*t/2, "adag", i+2, "a", i
    end # for
    if !conserveParticleNumber
        for i in 1:N_lat
            os += -μ, "n", i
        end # for
    end # if
    Ham = MPO(os, sites)

    return Ham
end # function

function calc_extended_int_BH_Ham(sites, N_lat, conserveParticleNumber, μ; J_1, J_2, U, V, P, D)
    os = OpSum()
    for i in 1:(N_lat-1)
        os += -J_1, "adag", i, "a", i+1
        os += -J_1, "adag", i+1, "a", i
    end # for
    for i in 1:(N_lat-2)
        os += -J_2, "adag", i, "a", i+2
        os += -J_2, "adag", i+2, "a", i
    end # for
    for i in 1:N_lat
        os += U/2, "n", i, "n", i
        os += -U/2, "n", i
    end # for
    for i in 1:(N_lat-1)
        os += V, "n", i, "n", i+1

        os += -D, "adag", i+1, "adag", i+1, "a", i+1, "a", i
        os += -D, "adag", i, "adag", i, "a", i, "a", i+1

        os += -P, "adag", i+1, "adag", i+1, "a", i, "a", i
        os += -P, "adag", i, "adag", i, "a", i+1, "a", i+1
    end # for
    if !conserveParticleNumber
        for i in 1:N_lat
            os += -μ, "n", i
        end # for
    end # if
    Ham = MPO(os, sites)

    return Ham
end # function

function calc_XXZ_Hamiltonian(par, sites)
    N_lat, conserve_Sz, h, J, Δ = par.N_lat, par.conserve_Sz, par.h, par.J, par.Δ

    os = OpSum()
    for j in 1:(N_lat-1)
        # os += -J, "Sx", j, "Sx", j+1
        # os += -J, "Sy", j, "Sy", j+1
        os += -J/2.0, "S+", j, "S-", j+1
        os += -J/2.0, "S+", j+1, "S-", j
        os += -J*Δ, "Sz", j, "Sz", j+1
    end # for
    if !conserve_Sz
        for j in 1:N_lat
            os += -2*h, "Sz", j
        end # for
    end # if
    # os += 0.05, "Sz", 1
    Ham = MPO(os, sites)

    return Ham
end # function
##### END OF HAMILTONIANS #####

##### EXPECTATION VALUES #####
function calc_expectation_values(ψ, Ham, sites, k_arr, N_lat)
    expectation_values = Dict{String, Any}()

    # k_arr INDICES
    idx_k0 = x2index(0, k_arr)
    idx_kπ = x2index(π, k_arr)
    idx_k2π3 = x2index(2*π/3, k_arr)
    # END OF k_arr INDICES

    # BULK
    bulk_range = (N_lat÷10):(N_lat-N_lat÷10)
    bulk_size = bulk_range[end] - bulk_range[1] + 1
    merge!(expectation_values, Dict("bulk_range" => bulk_range))
    merge!(expectation_values, Dict("bulk_size" => bulk_size))

    i_start = bulk_range[1]
    i_end = bulk_range[end]
    # i_dist = 100
    # i_start = (N_lat - i_dist) ÷ 2
    # i_end = i_start + i_dist
    merge!(expectation_values, Dict("i_start" => i_start))
    merge!(expectation_values, Dict("i_end" => i_end))
    # END OF BULK

    # DENSITY
    n_arr = expect(ψ, "n")
    ni_nj_matrix = correlation_matrix(ψ, "n", "n")
    C_corr_matrix = ni_nj_matrix - n_arr * n_arr'
    n2_arr = diag(ni_nj_matrix)
    var_n_arr = n2_arr .- n_arr.^2
    merge!(expectation_values, Dict("n_arr" => n_arr))
    merge!(expectation_values, Dict("ni_nj_matrix" => ni_nj_matrix))
    merge!(expectation_values, Dict("C_corr_matrix" => C_corr_matrix))
    merge!(expectation_values, Dict("n2_arr" => n2_arr))
    merge!(expectation_values, Dict("var_n_arr" => var_n_arr))
    # END OF DENSITY

    # DW, PERIOD 2 ORDER #
    avg_n = sum(n_arr)[1] / length(n_arr)
    # NOTE: Odd lattice site number is better for breaking DW degeneracy.
    δN = 0.0
    for i in 1:N_lat
        δN += 1.0/N_lat * (-1)^i * (n_arr[i] - avg_n)
    end # for
    merge!(expectation_values, Dict("δN" => δN))
    # END OF DW, PERIOD 2 ORDER #

    # SF ORDER #
    os = OpSum()
    os += "adag", i_start, "a", i_end
    g1_mpo = MPO(os, sites)
    g1 = inner(ψ', g1_mpo, ψ)
    merge!(expectation_values, Dict("g1" => g1))

    g1_vec = Vector{Float64}([])
    for i in i_start:i_end
        os = OpSum()
        os += "adag", i_start, "a", i
        g1_mpo = MPO(os, sites)
        g1_i = inner(ψ', g1_mpo, ψ)
        push!(g1_vec, g1_i)
    end # for
    merge!(expectation_values, Dict("g1_vec" => g1_vec))

    g1_matrix = correlation_matrix(ψ, "adag", "a")
    merge!(expectation_values, Dict("g1_matrix" => g1_matrix))
    # END OF SF ORDER #

    # PAIR SF ORDER #
    os = OpSum()
    os += "adag", i_start, "adag", i_start, "a", i_end, "a", i_end
    g2_mpo = MPO(os, sites)
    g2 = inner(ψ', g2_mpo, ψ)
    merge!(expectation_values, Dict("g2" => g2))

    g2_vec = Vector{Float64}([])
    for i in i_start:i_end
        os = OpSum()
        os += "adag", i_start, "adag", i_start, "a", i, "a", i
        g2_mpo = MPO(os, sites)
        g2_i = inner(ψ', g2_mpo, ψ)
        push!(g2_vec, g2_i)
    end # for
    merge!(expectation_values, Dict("g2_vec" => g2_vec))

    g2_matrix = correlation_matrix(ψ, "adag2", "a2")
    merge!(expectation_values, Dict("g2_matrix" => g2_matrix))
    # END OF PAIR SF ORDER #

    # CHIRAL SF ORDER #
    os = OpSum()
    os += -0.25, "a", i_start, "adag", i_start+1, "a", i_end, "adag", i_end+1
    os += 0.25, "a", i_start, "adag", i_start+1, "adag", i_end, "a", i_end+1
    os += 0.25, "adag", i_start, "a", i_start+1, "a", i_end, "adag", i_end+1
    os += -0.25, "adag", i_start, "a", i_start+1, "adag", i_end, "a", i_end+1
    κ2_mpo = MPO(os, sites)
    κ2 = inner(ψ', κ2_mpo, ψ)
    merge!(expectation_values, Dict("κ2" => κ2))

    κ2_vec = Vector{Float64}([])
    for i in i_start:i_end
        os = OpSum()
        os += -0.25, "a", i_start, "adag", i_start+1, "a", i, "adag", i+1
        os += 0.25, "a", i_start, "adag", i_start+1, "adag", i, "a", i+1
        os += 0.25, "adag", i_start, "a", i_start+1, "a", i, "adag", i+1
        os += -0.25, "adag", i_start, "a", i_start+1, "adag", i, "a", i+1
        κ2_mpo = MPO(os, sites)
        κ2_i = inner(ψ', κ2_mpo, ψ)
        push!(κ2_vec, κ2_i)
    end # for
    merge!(expectation_values, Dict("κ2_vec" => κ2_vec))
    # END OF CHIRAL SF ORDER #

    # BOW ORDER #
    ΔB = 0.0
    for i in i_start:i_end
        os = OpSum()
        os += 1.0, "adag", i, "a", i+1
        os += 1.0, "adag", i+1, "a", i
        B_i_mpo = MPO(os, sites)
        B_i = inner(ψ', B_i_mpo, ψ)

        os = OpSum()
        os += 1.0, "adag", i+1, "a", i+2
        os += 1.0, "adag", i+2, "a", i+1
        B_ip1_mpo = MPO(os, sites)
        B_ip1 = inner(ψ', B_ip1_mpo, ψ)

        ΔB += B_i + B_ip1
    end # for

    ΔB /= bulk_size
    merge!(expectation_values, Dict("ΔB" => ΔB))
    # END OF BOW ORDER #

    # DROPLET ORDER #
    avg_NN_g1 = sum(diag(g1_matrix, 1))
    avg_NN_g2 = sum(diag(g2_matrix, 1))
    avg_NN_n = sum(diag(ni_nj_matrix, 1))
    avg_sep2_n = sum(diag(ni_nj_matrix, 2))

    merge!(expectation_values, Dict("avg_NN_g1" => avg_NN_g1))
    merge!(expectation_values, Dict("avg_NN_g2" => avg_NN_g2))
    merge!(expectation_values, Dict("avg_NN_n" => avg_NN_n))
    merge!(expectation_values, Dict("avg_sep2_n" => avg_sep2_n))
    # END OF DROPLET ORDER #

    # MOMENTUM DISTRIBUTION #
    # ChandaBarbiero et al., Recent progress on quantum simulations of
    # non-standard Bose-Hubbard models
    M_k_vec = Vector{ComplexF64}([])
    for k in k_arr
        M_k = complex(0.0, 0.0)

        for i in i_start:i_end
        # for i in 1:N_lat
            for j in i_start:i_end
            # for j in 1:N_lat
                 M_k += exp(im*k*(i-j)) * g1_matrix[i, j] / N_lat^2
            end # for
        end # for

        push!(M_k_vec, M_k)
    end # for
    merge!(expectation_values, Dict("M_k_vec" => M_k_vec))
    @show sum(abs.(imag.(M_k_vec)))

    M_k0 = real(M_k_vec[idx_k0])
    M_kπ = real(M_k_vec[idx_kπ])
    M_k2π3 = real(M_k_vec[idx_k2π3])
    merge!(expectation_values, Dict("M_k0" => M_k0))
    merge!(expectation_values, Dict("M_kπ" => M_kπ))
    merge!(expectation_values, Dict("M_k2π3" => M_k2π3))
    # END OF MOMENTUM DISTRIBUTION #

    # Several different versions of structure factor are used.
    # STRUCTURE FACTOR #
    S_k_vec = Vector{ComplexF64}([])
    for k in k_arr
        S_k = complex(0.0, 0.0)

        for i in i_start:i_end
        # for i in 1:N_lat
            for j in i_start:i_end
            # for j in 1:N_lat
                # if i != j
                # os = OpSum()
                # os += "n", i, "n", j
                # ni_nj_mpo = MPO(os, sites)
                # ni_nj = inner(ψ', ni_nj_mpo, ψ)

                ni_nj = ni_nj_matrix[i, j]
                S_k += exp(im*k*(i-j)) * ni_nj / N_lat^2
                # end # if
            end # for
        end # for
        # for i in 1:N_lat
        #     # ni_nj = ni_nj_matrix[i, 1]
        #     ni_nj = ni_nj_matrix[i, 1]
        #     S_k += exp(im*k*(i-1)) * ni_nj
        # end # for

        push!(S_k_vec, S_k)
    end # for
    merge!(expectation_values, Dict("S_k_vec" => S_k_vec))
    # @assert sum(abs.(imag.(S_k_vec))) < 1e-6
    @show sum(abs.(imag.(S_k_vec)))

    S_k0 = real(S_k_vec[idx_k0])
    S_kπ = real(S_k_vec[idx_kπ])
    S_k2π3 = real(S_k_vec[idx_k2π3])
    merge!(expectation_values, Dict("S_k0" => S_k0))
    merge!(expectation_values, Dict("S_kπ" => S_kπ))
    merge!(expectation_values, Dict("S_k2π3" => S_k2π3))
    # END OF STRUCTURE FACTOR #

    # PARITY #
    # O_odd = calc_prod_expectation_value(ψ, sites, "exp_iπn", bulk_range)
    # O_even = calc_prod_expectation_value(ψ, sites, "exp_iπnm1", bulk_range)

    O_odd = 0.5 * (calc_prod_expectation_value(ψ, sites, N_lat, "exp_iπn", bulk_range) + calc_prod_expectation_value(ψ, sites, N_lat, "exp_iπn", (bulk_range[1]):(bulk_range[end]+1)))
    O_even = 0.5 * (calc_prod_expectation_value(ψ, sites, N_lat, "exp_iπnm1", bulk_range) + calc_prod_expectation_value(ψ, sites, N_lat, "exp_iπnm1", (bulk_range[1]):(bulk_range[end]+1)))

    # merge!(expectation_values, Dict("O_odd_matrix" => O_odd_matrix))
    # merge!(expectation_values, Dict("O_odd_vec" => O_odd_vec))
    merge!(expectation_values, Dict("O_odd" => O_odd))
    # merge!(expectation_values, Dict("O_even_matrix" => O_even_matrix))
    # merge!(expectation_values, Dict("O_even_vec" => O_even_vec))
    merge!(expectation_values, Dict("O_even" => O_even))
    # END OF PARITY #

    # STRING ORDER #
    O_string = calc_string_order(ψ, sites, N_lat, bulk_range)
    merge!(expectation_values, Dict("O_string" => O_string))
    # END OF STRING ORDER #

    # CENTRAL CHARGE #
    entropy_S_vec = [DMRG.calc_entanglement_entropy(ψ, j) for j in 1:N_lat]
    merge!(expectation_values, Dict("entropy_S_vec" => entropy_S_vec))

    SvN_half = DMRG.calc_entanglement_entropy(ψ, N_lat÷2)
    merge!(expectation_values, Dict("SvN_half" => SvN_half))

    # NOTE: Central charge doesn't always converge, disabling for now.
    # c, c_resid = 99.0, 99.0
    # c, c_resid = DMRG.full_calc_central_charge(ψ, N_lat)
    c, c_resid = DMRG.calc_central_charge(entropy_S_vec, N_lat)
    merge!(expectation_values, Dict("c" => c))
    merge!(expectation_values, Dict("c_resid" => c_resid))
    # END OF CENTRAL CHARGE #

    # OCCUPATION PROBABILITIES #
    # https://itensor.discourse.group/t/operator-definition-as-a-product-of-operators/673/7
    os = OpSum() + ("proj_Q", 1)
    for i in 2:N_lat
        os_i = OpSum() + ("proj_Q", i)
        os *= os_i
    end # for
    proj_Q_mpo = MPO(Ops.expand(os), sites)
    prob_Q = inner(ψ', proj_Q_mpo, ψ)
    prob_Q_bulk = calc_prod_expectation_value(ψ, sites, N_lat, "proj_Q", bulk_range)

    os = OpSum() + ("proj_Q_mod", 1)
    for i in 2:N_lat
        os_i = OpSum() + ("proj_Q_mod", i)
        os *= os_i
    end # for
    proj_Q_mod_mpo = MPO(Ops.expand(os), sites)
    prob_Q_mod = inner(ψ', proj_Q_mod_mpo, ψ)
    prob_Q_mod_bulk = calc_prod_expectation_value(ψ, sites, N_lat, "proj_Q_mod", bulk_range)

    os = OpSum() + ("proj_P", 1)
    for i in 2:N_lat
        os_i = OpSum() + ("proj_P", i)
        os *= os_i
    end # for
    proj_P_mpo = MPO(Ops.expand(os), sites)
    prob_P = inner(ψ', proj_P_mpo, ψ)
    prob_P_bulk = calc_prod_expectation_value(ψ, sites, N_lat, "proj_P", bulk_range)

    merge!(expectation_values, Dict("prob_Q" => prob_Q))
    merge!(expectation_values, Dict("prob_Q_bulk" => prob_Q_bulk))
    merge!(expectation_values, Dict("prob_Q_mod" => prob_Q_mod))
    merge!(expectation_values, Dict("prob_Q_mod_bulk" => prob_Q_mod_bulk))
    merge!(expectation_values, Dict("prob_P" => prob_P))
    merge!(expectation_values, Dict("prob_P_bulk" => prob_P_bulk))
    # END OF OCCUPATION PROBABILITIES #

    # LOCALIZATION #
    IPR = tr(ni_nj_matrix) / sum(n_arr)^2

    merge!(expectation_values, Dict("IPR" => IPR))
    # END OF LOCALIZATION #

    # ENERGIES #
    # TODO Calculate bulk values.

    expect_E2 = inner(Ham, ψ, Ham, ψ)
    expect_E = inner(ψ', Ham, ψ)
    @show expect_E2 - expect_E^2
    merge!(expectation_values, Dict("expect_E" => expect_E))
    merge!(expectation_values, Dict("expect_E2" => expect_E2))
    merge!(expectation_values, Dict("var_E" => (expect_E2 - expect_E^2)))

    @show expect_E/N_lat
    merge!(expectation_values, Dict("E/N_lat" => (expect_E/N_lat)))
    # END OF ENERGIES #

    ##### SAMPLING #####
    N_samples = 100
    orthogonalize!(ψ, 1)
    ψ_samples = zeros(Int, N_samples, N_lat)
    for i in 1:N_samples
        ψ_samples[i, :] = (sample(ψ) .- 1)
    end # for
    merge!(expectation_values, Dict("ψ_samples" => ψ_samples))
    ##### END OF SAMPLING #####

    # plt.plot(expect_n_arr)

    return expectation_values
end # function

function calc_spin_expectation_values(ψ, sites, N_lat)
    expectation_values = Dict{String, Any}()

    bulk_range = (N_lat÷10):(N_lat-N_lat÷10)
    bulk_size = bulk_range[end] - bulk_range[1] + 1
    merge!(expectation_values, Dict("bulk_range" => bulk_range))
    merge!(expectation_values, Dict("bulk_size" => bulk_size))

    Sz_vec = expect(ψ, "Sz")
    mz = sum(Sz_vec[bulk_range]) / bulk_size
    SpSm_matrix = correlation_matrix(ψ, "S+", "S-")
    SpSm_vec = SpSm_matrix[bulk_range[1], bulk_range]
    SpSm = SpSm_matrix[bulk_range[1], bulk_range[end]]
    mz_stag = sum([(-1)^j * Sz for (j, Sz) in enumerate(Sz_vec[bulk_range])]) / bulk_size
    SzSz_matrix = correlation_matrix(ψ, "Sz", "Sz")
    SzSz_vec = SzSz_matrix[bulk_range[1], bulk_range]
    SzSz = SzSz_matrix[bulk_range[1], bulk_range[end]]
    merge!(expectation_values, Dict("Sz_vec" => Sz_vec))
    merge!(expectation_values, Dict("mz" => mz))
    merge!(expectation_values, Dict("SpSm_matrix" => SpSm_matrix))
    merge!(expectation_values, Dict("SpSm_vec" => SpSm_vec))
    merge!(expectation_values, Dict("SpSm" => SpSm))
    merge!(expectation_values, Dict("mz_stag" => mz_stag))
    merge!(expectation_values, Dict("SzSz_matrix" => SzSz_matrix))
    merge!(expectation_values, Dict("SzSz_vec" => SzSz_vec))
    merge!(expectation_values, Dict("SzSz" => SzSz))

    return expectation_values
end # function
##### END OF EXPECTATION VALUES #####

##### PROPER PHASE DIAGRAM #####
function map_2D_data_arr_to_2D_int_arr(data_arr)
    unique_vals = unique(data_arr)
    mapping_dict = Dict(v => i for (i, v) in enumerate(unique_vals))

    # TODO Generalize to multidimensional.
    int_arr = zeros(Int, size(data_arr)...)
    for i in 1:size(data_arr, 1)
        for j in 1:size(data_arr, 2)
            int_arr[i, j] = mapping_dict[data_arr[i, j]]
        end # for
    end # for

    return mapping_dict, int_arr
end # function

function classify_qmb_phase_v1(expectation_values; printWarning=true)
    crit_val_dict = Dict(
        "O_even" => 0.1,
        "g1" => 1e-2,
        "κ2" => 5e-2,
        "δN" => 0.10,
        "S_kπ" => 0.025,
        # "g2" => 5e-4,
        "g2" => 2.1e-3,
    )

    O_even, g1, κ2, δN, S_kπ, g2 = expectation_values["O_even"], expectation_values["g1"], expectation_values["κ2"], expectation_values["δN"], expectation_values["S_kπ"], expectation_values["g2"]
    crit_O_even, crit_g1, crit_κ2, crit_δN, crit_S_kπ, crit_g2 = crit_val_dict["O_even"], crit_val_dict["g1"], crit_val_dict["κ2"], crit_val_dict["δN"], crit_val_dict["S_kπ"], crit_val_dict["g2"]

    # g1_matrix = expectation_values["g1_matrix"]
    # g1 = abs(g1_matrix[75, 125]) + abs(g1_matrix[75, 126])

    # g2_matrix = expectation_values["g2_matrix"]
    # g2 = abs(g2_matrix[75, 125]) + abs(g2_matrix[75, 126])

    # if (abs(O_even) > crit_O_even) && (abs(g1) < crit_g1) &&
    #    (abs(κ2) < crit_κ2) && (abs(δN) < crit_δN) &&
    #    (abs(g2) < crit_g2)
    # if (abs(O_even) > crit_O_even) && (abs(g1) < crit_g1)
    if (abs(κ2) > crit_κ2) && (abs(g1) > crit_g1)
    # if (abs(κ2) > crit_κ2)
        return "CSF"
    elseif (abs(O_even) > crit_O_even)
        return "MI"
    # elseif (abs(O_even) < crit_O_even) && (abs(g1) < crit_g1) &&
    #    (abs(κ2) < crit_κ2) && (abs(δN) > crit_δN) &&
    #    (abs(g2) < crit_g2)
    # elseif (abs(g1) < crit_g1) && (abs(δN) > crit_δN)
    elseif (abs(g1) < crit_g1) && (abs(S_kπ) > crit_S_kπ)
        return "DW"
    # elseif (abs(O_even) < crit_O_even) && (abs(g1) < crit_g1) &&
    #    (abs(κ2) < crit_κ2) && (abs(δN) < crit_δN) &&
    #    (abs(g2) > crit_g2)
    elseif (abs(g1) < crit_g1) && (abs(g2) > crit_g2) && (abs(κ2) < crit_κ2)
        return "PSF"
    # elseif (abs(O_even) < crit_O_even) && (abs(g1) > crit_g1) &&
    #    (abs(κ2) > crit_κ2) && (abs(δN) < crit_δN) &&
    #    (abs(g2) > crit_g2)
    else
        if printWarning
            println("WARNING: Could not classify phase.")
        end # if

        return "NONE"
    end # if
end # function

"""Based on top-to-bottom and bottom-to-top vertical cuts in parameter space to
find upper and lower phase curves."""
function calc_phase_curve_v1(phase_name, x_vec, y_vec, phase_name_arr)
    @assert y_vec == sort(y_vec)  # Assuming y_vec is in ascending order.

    N_x = length(x_vec)
    N_y = length(y_vec)
    @assert N_x > 2
    @assert N_y > 2

    x_phase_curve_vec = Vector{Float64}([])
    y_phase_top_curve_vec = Vector{Float64}([])
    y_phase_bottom_curve_vec = Vector{Float64}([])
    for m in 1:N_x
        x = x_vec[m]
        isPhase = in(phase_name, phase_name_arr[m, :])

        if isPhase
            push!(x_phase_curve_vec, x)

            # TOP CURVE
            topPhase = (phase_name_arr[m, end] == phase_name)

            if topPhase
                push!(y_phase_top_curve_vec, y_vec[end])
            else
                for n in (N_y-1):(-1):1
                    prev_phase_name = phase_name_arr[m, n+1]
                    cur_phase_name = phase_name_arr[m, n]

                    if (prev_phase_name != phase_name) && (cur_phase_name == phase_name)
                        trans_y = 0.5 * (y_vec[n+1] + y_vec[n])

                        push!(y_phase_top_curve_vec, trans_y)
                        break
                    end # if
                end # for
            end # if
            # END OF TOP CURVE

            # BOTTOM CURVE
            bottomPhase = (phase_name_arr[m, 1] == phase_name)

            if bottomPhase
                push!(y_phase_bottom_curve_vec, y_vec[1])
            else
                for n in 2:N_y
                    prev_phase_name = phase_name_arr[m, n-1]
                    cur_phase_name = phase_name_arr[m, n]

                    if (prev_phase_name != phase_name) && (cur_phase_name == phase_name)
                        trans_y = 0.5 * (y_vec[n-1] + y_vec[n])

                        push!(y_phase_bottom_curve_vec, trans_y)
                        break
                    end # if
                end # for
            end # if
            # END OF BOTTOM CURVE
        end # if
    end # for

    return x_phase_curve_vec, y_phase_top_curve_vec, y_phase_bottom_curve_vec
end # function
##### END OF PROPER PHASE DIAGRAM #####

##### PLOTTING #####
function plot_y_vs_x(x_arr, y_arr; x_name="x", y_name="y", x_scale="linear", y_scale="linear", color="blue", fig_ax=nothing, file_name=nothing)
    fig, ax = get_fig_ax(fig_ax)

    if file_name == nothing
        file_name = "$(y_name)_vs_$(x_name)"
    end # if

    if fig_ax == nothing
        fig.set_size_inches(11, 6)
    end # if
    ax.set_xscale(x_scale)
    ax.set_yscale(y_scale)
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    ax.plot(x_arr, y_arr, color=color)

    if fig_ax == nothing
        set_plot_defaults(fig, ax)
        save_plot(file_name, @__DIR__)
        # plt.show()
    end # if
end # function

function plot_u_vs_xy(x_arr, y_arr, u_arr; x_name="x", y_name="y", u_name="u", vlim=nothing, fig_ax=nothing, file_name="u_vs_xy")
    fig, ax = get_fig_ax(fig_ax)

    if fig_ax == nothing
        fig.set_size_inches(9, 6)
    end # if
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    ax.set_title(u_name)
    if vlim == nothing
        pmesh = ax.pcolormesh(x_arr, y_arr, u_arr, cmap=plt.get_cmap("viridis"), shading="nearest")
    else
        pmesh = ax.pcolormesh(x_arr, y_arr, u_arr, cmap=plt.get_cmap("viridis"), vmin=vlim[1], vmax=vlim[2], shading="nearest")
    end # if
    cbar = fig.colorbar(pmesh, ax=ax, location="right", aspect=30)
    # cbar.ax.xaxis.label.set_size(12)
    cbar.ax.tick_params(which="both", length=2, direction="in")
    cbar.ax.xaxis.set_ticks_position("both")

    if fig_ax == nothing
        set_plot_defaults(fig, ax, addGrid=false)
        save_plot(file_name, @__DIR__)
        # plt.show()
    end # if
end # function

function plot_phase_diagram_by_name(x_vec, y_vec, phase_name_arr; x_label=raw"$x$", y_label=raw"$y$", fig_ax=nothing, file_name="Phase_Diagram_by_Name_vs_xy")
    mapping_dict, phase_id_arr = map_2D_data_arr_to_2D_int_arr(phase_name_arr)

    fig, ax = get_fig_ax(fig_ax)

    if fig_ax == nothing
        fig.set_size_inches(9, 6)
    end # if
    ax.set_xlim(x_vec[1], x_vec[end])
    ax.set_ylim(y_vec[1], y_vec[end])
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    pmesh = ax.pcolormesh(x_vec, y_vec, transpose(phase_id_arr), cmap=plt.get_cmap("tab10"), shading="nearest")

    categories = minimum(phase_id_arr):maximum(phase_id_arr)
    norm = plt.matplotlib.colors.Normalize(minimum(phase_id_arr), maximum(phase_id_arr))
    colors = [plt.cm.tab10(norm(cat)) for cat in categories]
    patches = [plt.matplotlib.patches.Patch(color=c, label=invert_dict(mapping_dict)[cat]) for (cat, c) in zip(categories, colors)]
    ax.legend(handles=patches, title="Phases", loc="center left", bbox_to_anchor=(1, 0.5))

    if fig_ax == nothing
        set_plot_defaults(fig, ax, addGrid=false)
        plt.subplots_adjust(right=0.8, bottom=0.15)
        save_plot(file_name, @__DIR__)
        # plt.show()
    end # if
end # function

# TODO
# NOTE: Very specific to current brick wall problem. Very difficult/impossible to generalize.
function plot_phase_diagram_vs_J1_gx(abs_J1_vec, gx_vec, CSF_curve_vec2, MI_curve_vec2, DW_curve_vec2, PSF_curve_vec2; file_name="Phase_Diagram_vs_J1_gx")
    fig, ax = get_fig_ax()
    fig.set_size_inches(12, 9)

    phase_name_size = 42
    colors = ["blue", "cyan", "red", "green"]

    ax.set_xlim(abs_J1_vec[1], abs_J1_vec[end])
    ax.set_ylim(gx_vec[1], gx_vec[end])
    ax.set_xlabel(raw"$|J_1|/g_0$")
    ax.set_ylabel(raw"$g_x/g_0$")

    function plot_phase_region(phase_curve_vec2, ax, i)
        x_phase_curve_vec, y_phase_top_curve_vec, y_phase_bottom_curve_vec = phase_curve_vec2
        ax.plot(x_phase_curve_vec, y_phase_top_curve_vec, color="black", marker="o")
        ax.plot(x_phase_curve_vec, y_phase_bottom_curve_vec, color="black", marker="o")
        ax.plot([x_phase_curve_vec[1], x_phase_curve_vec[1]], [y_phase_top_curve_vec[1], y_phase_bottom_curve_vec[1]], color="black", marker="o")
        ax.fill_between(x_phase_curve_vec, y_phase_top_curve_vec, y_phase_bottom_curve_vec, color=colors[i], alpha=0.6)
    end # function

    # CSF
    ax.text(0.6, 0.6, "CSF", fontsize=phase_name_size, transform=ax.transAxes)
    plot_phase_region(CSF_curve_vec2, ax, 1)
    # END OF CSF

    # MI
    ax.text(0.1, 0.85, "MI", fontsize=phase_name_size, transform=ax.transAxes)
    plot_phase_region(MI_curve_vec2, ax, 2)
    # END OF MI

    # DW
    ax.text(0.05, 0.5, "DW", fontsize=phase_name_size, transform=ax.transAxes)
    plot_phase_region(DW_curve_vec2, ax, 3)
    # END OF DW

    # PSF
    ax.text(0.2, 0.15, "PSF", fontsize=phase_name_size, transform=ax.transAxes)
    plot_phase_region(PSF_curve_vec2, ax, 4)
    # END OF PSF

    set_plot_defaults(fig, ax, addGrid=false)
    plt.subplots_adjust(hspace=0.4)
    save_plot(file_name, @__DIR__, file_type="pdf")
end # function

function plotN_expectation_values_v1(N_lat, k_arr, n_arr, S_k_vec, g1_vec, g2_vec, g1_matrix, M_k_vec; file_name="Expectation_Values_v1")
# function plotN_expectation_values_v1(N_lat, k_arr, n_arr, S_k_vec, g1_vec, g2_vec, g1_matrix, g2_matrix; file_name="Expectation_Values_v1")
    fig, ax = plt.subplots(3, 3)
    fig.set_size_inches(24, 15)

    idx_arr = Array(1:N_lat)
    plot_y_vs_x(idx_arr, n_arr; x_name="j", y_name=raw"$\langle n \rangle$", color="black", fig_ax=(fig, ax[1, 1]))
    plot_y_vs_x(k_arr, real.(S_k_vec); x_name=raw"$k$", y_name=raw"$S(k)$", color="green", fig_ax=(fig, ax[2, 1]))
    dist_arr = Array(1:length(g1_vec))
    plot_y_vs_x(dist_arr, g1_vec; x_name="i-j", y_name=raw"$g_1(|i-j|)$", x_scale="log", y_scale="log", color="blue", fig_ax=(fig, ax[1, 2]))
    plot_y_vs_x(dist_arr, g2_vec; x_name="i-j", y_name=raw"$g_2(|i-j|)$", x_scale="log", y_scale="log", color="cyan", fig_ax=(fig, ax[2, 2]))
    plot_u_vs_xy(idx_arr, idx_arr, transpose(g1_matrix), x_name="i", y_name="j", u_name=raw"$g_1(|i-j|)$", fig_ax=(fig, ax[1, 3]))
    # plot_u_vs_xy(idx_arr, idx_arr, transpose(g2_matrix), x_name="i", y_name="j", u_name=raw"$g_2(|i-j|)$", fig_ax=(fig, ax[2, 3]))
    plot_y_vs_x(k_arr, real.(M_k_vec); x_name=raw"$k$", y_name=raw"$M(k)$", color="purple", fig_ax=(fig, ax[2, 3]))
    # plot_y_vs_x(dist_arr, abs.(O_odd_vec); x_name=raw"$i-j$", y_name=raw"$|O_{\mathrm{odd}}(|i-j|)|$", color="red", fig_ax=(fig, ax[3, 1]))
    # plot_y_vs_x(dist_arr, abs.(O_even_vec); x_name=raw"$i-j$", y_name=raw"$|O_{\mathrm{even}}(|i-j|)|$", color="orange", fig_ax=(fig, ax[3, 2]))

    for i in 1:2
        for j in 1:2
            ax[i, j].grid(true)
        end # for
    end # for

    set_plot_defaults(fig, ax, addGrid=false)
    plt.subplots_adjust(hspace=0.4)
    save_plot(file_name, @__DIR__, file_type="pdf")
    # plt.close("all")
    # plt.show()
end # function

function plotN_expectation_values_poster(N_lat, g1_vec, g2_vec, g1_matrix, g2_matrix; fig_title=nothing, file_name="Expectation_Values_v2")
    fig, ax = plt.subplots(2, 2)
    fig.set_size_inches(16, 10)
    if fig_title != nothing
        fig.suptitle(fig_title)
    end # if

    idx_arr = Array(1:N_lat)
    dist_arr = Array(1:length(g1_vec))
    plot_y_vs_x(dist_arr, g1_vec; x_name="i-j", y_name=raw"$g_1(|i-j|)$", x_scale="log", y_scale="log", color="blue", fig_ax=(fig, ax[1, 1]))
    plot_y_vs_x(dist_arr, g2_vec; x_name="i-j", y_name=raw"$g_2(|i-j|)$", x_scale="log", y_scale="log", color="cyan", fig_ax=(fig, ax[2, 1]))
    plot_u_vs_xy(idx_arr, idx_arr, transpose(g1_matrix), x_name="i", y_name="j", u_name=raw"$g_1(|i-j|)$", fig_ax=(fig, ax[1, 2]))
    plot_u_vs_xy(idx_arr, idx_arr, transpose(g2_matrix), x_name="i", y_name="j", u_name=raw"$g_2(|i-j|)$", fig_ax=(fig, ax[2, 2]))

    for i in 1:2
        for j in 1:2
            ax[i, j].grid(true)
        end # for
    end # for

    set_plot_defaults(fig, ax, addGrid=false)
    plt.subplots_adjust(hspace=0.4)
    save_plot(file_name, @__DIR__, file_type="pdf")
    # plt.show()
end # function

function plotN_PSF_phase_curves_v1(g0_vec, δN_arr, g1_arr, g2_arr; file_name="PSF_Phase_Curves")
    fig, ax = plt.subplots(3, 1)
    fig.set_size_inches(8, 14)

    plot_y_vs_x(g0_vec, abs.(δN_arr); x_name=raw"$g_0$", y_name=raw"$|\delta N|$", color="red", fig_ax=(fig, ax[1, 1]))
    plot_y_vs_x(g0_vec, g1_arr; x_name=raw"$g_0$", y_name=raw"$g_1(|i-j|)$", color="blue", fig_ax=(fig, ax[2, 1]))
    plot_y_vs_x(g0_vec, g2_arr; x_name=raw"$g_0$", y_name=raw"$g_2(|i-j|)$", color="cyan", fig_ax=(fig, ax[3, 1]))

    set_plot_defaults(fig, ax, addGrid=true)
    plt.subplots_adjust(hspace=0.4)
    save_plot(file_name, @__DIR__, file_type="pdf")
    # plt.show()
end # function

function plotN_SS_phase_curves_v1(abs_J2_vec, δN_arr, g1_arr, g2_arr; file_name="SS_Phase_Curves")
    fig, ax = plt.subplots(3, 1)
    fig.set_size_inches(8, 14)

    plot_y_vs_x(abs_J2_vec, abs.(δN_arr); x_name=raw"$|J_2|$", y_name=raw"$|\delta N|$", color="red", fig_ax=(fig, ax[1, 1]))
    plot_y_vs_x(abs_J2_vec, g1_arr; x_name=raw"$|J_2|$", y_name=raw"$g_1(|i-j|)$", color="blue", fig_ax=(fig, ax[2, 1]))
    plot_y_vs_x(abs_J2_vec, g2_arr; x_name=raw"$|J_2|$", y_name=raw"$g_2(|i-j|)$", color="cyan", fig_ax=(fig, ax[3, 1]))

    set_plot_defaults(fig, ax, addGrid=true)
    plt.subplots_adjust(hspace=0.4)
    save_plot(file_name, @__DIR__, file_type="pdf")
    # plt.show()
end # function

function plotN_order_parameters_vs_gx(gx_vec, δN_vec, g1_vec, g2_vec, integ_g1_vec, integ_g2_vec, integ_S_k_vec, O_odd_vec, O_even_vec, expect_E_vec, S_k0_vec, M_k0_vec, S_kπ_vec, M_kπ_vec, c_vec, prob_Q_vec; file_name="Order_parameters_vs_gx")
    fig, ax = plt.subplots(4, 5)
    fig.set_size_inches(30, 22)

    plot_y_vs_x(gx_vec, δN_vec; x_name=raw"$g_x/g_0$", y_name=raw"$\delta N$", fig_ax=(fig, ax[1, 1]))
    plot_y_vs_x(gx_vec, g1_vec; x_name=raw"$g_x/g_0$", y_name=raw"$g_1$", fig_ax=(fig, ax[1, 2]))
    plot_y_vs_x(gx_vec, g2_vec; x_name=raw"$g_x/g_0$", y_name=raw"$g_2$", fig_ax=(fig, ax[1, 3]))
    plot_y_vs_x(gx_vec, integ_g1_vec; x_name=raw"$g_x/g_0$", y_name=raw"$\sum |g_1|$", fig_ax=(fig, ax[1, 4]))
    plot_y_vs_x(gx_vec, integ_g2_vec; x_name=raw"$g_x/g_0$", y_name=raw"$\sum |g_2|$", fig_ax=(fig, ax[2, 1]))
    plot_y_vs_x(gx_vec, integ_S_k_vec; x_name=raw"$g_x/g_0$", y_name=raw"$\sum |S_k|$", fig_ax=(fig, ax[2, 2]))
    # plot_y_vs_x(gx_vec, integ_M_k_vec; x_name=raw"$g_x/g_0$", y_name=raw"$\sum |M_k|$", fig_ax=(fig, ax[2, 3]))
    plot_y_vs_x(gx_vec, expect_E_vec; x_name=raw"$g_x/g_0$", y_name=raw"$E$", fig_ax=(fig, ax[2, 3]))
    # plot_y_vs_x(gx_vec, S_k0_vec; x_name=raw"$g_x/g_0$", y_name=raw"$S_{k=0}$", fig_ax=(fig, ax[3, 1]))
    # plot_y_vs_x(gx_vec, M_k0_vec; x_name=raw"$g_x/g_0$", y_name=raw"$M_{k=0}$", fig_ax=(fig, ax[3, 2]))
    plot_y_vs_x(gx_vec, O_odd_vec; x_name=raw"$g_x/g_0$", y_name=raw"$O_{\mathrm{odd}}$", fig_ax=(fig, ax[3, 1]))
    plot_y_vs_x(gx_vec, O_even_vec; x_name=raw"$g_x/g_0$", y_name=raw"$O_{\mathrm{even}}$", fig_ax=(fig, ax[3, 2]))
    plot_y_vs_x(gx_vec, S_kπ_vec; x_name=raw"$g_x/g_0$", y_name=raw"$S_{k=\pi}$", fig_ax=(fig, ax[3, 3]))
    plot_y_vs_x(gx_vec, M_kπ_vec; x_name=raw"$g_x/g_0$", y_name=raw"$M_{k=\pi}$", fig_ax=(fig, ax[3, 4]))
    plot_y_vs_x(gx_vec, prob_Q_vec; x_name=raw"$g_x/g_0$", y_name=raw"$P_Q$", fig_ax=(fig, ax[4, 1]))
    # plot_y_vs_x(gx_vec, prob_P_vec; x_name=raw"$g_x/g_0$", y_name=raw"$P_P$", fig_ax=(fig, ax[4, 2]))

    set_plot_defaults(fig, ax, addGrid=false)
    plt.subplots_adjust(hspace=0.4)
    save_plot(file_name, @__DIR__, file_type="pdf")
    # plt.close("all")
    # plt.show()
end # function

function plotN_order_parameters_gaps_vs_gx(gx_vec, δN_vec, g1_vec, g2_vec, Δ1_vec, Δ2_vec, integ_g1_vec, integ_g2_vec, integ_S_k_vec, O_odd_vec, O_even_vec, expect_E_vec, S_k0_vec, M_k0_vec, S_kπ_vec, M_kπ_vec, c_vec, prob_Q_vec, avg_NN_g1_vec, avg_NN_g2_vec, avg_NN_n_vec; file_name="Order_parameters_gaps_vs_gx")
    fig, ax = plt.subplots(4, 5)
    fig.set_size_inches(30, 22)

    plot_y_vs_x(gx_vec, δN_vec; x_name=raw"$g_x/g_0$", y_name=raw"$\delta N$", fig_ax=(fig, ax[1, 1]))
    plot_y_vs_x(gx_vec, g1_vec; x_name=raw"$g_x/g_0$", y_name=raw"$g_1$", fig_ax=(fig, ax[1, 2]))
    plot_y_vs_x(gx_vec, g2_vec; x_name=raw"$g_x/g_0$", y_name=raw"$g_2$", fig_ax=(fig, ax[1, 3]))
    plot_y_vs_x(gx_vec, integ_g1_vec; x_name=raw"$g_x/g_0$", y_name=raw"$\sum |g_1|$", fig_ax=(fig, ax[1, 4]))
    plot_y_vs_x(gx_vec, integ_g2_vec; x_name=raw"$g_x/g_0$", y_name=raw"$\sum |g_2|$", fig_ax=(fig, ax[2, 1]))
    plot_y_vs_x(gx_vec, integ_S_k_vec; x_name=raw"$g_x/g_0$", y_name=raw"$\sum |S_k|$", fig_ax=(fig, ax[2, 2]))
    # plot_y_vs_x(gx_vec, integ_M_k_vec; x_name=raw"$g_x/g_0$", y_name=raw"$\sum |M_k|$", fig_ax=(fig, ax[2, 3]))
    plot_y_vs_x(gx_vec, expect_E_vec; x_name=raw"$g_x/g_0$", y_name=raw"$E$", fig_ax=(fig, ax[2, 3]))
    plot_y_vs_x(gx_vec, Δ1_vec; x_name=raw"$g_x/g_0$", y_name=raw"$\Delta_1$", fig_ax=(fig, ax[1, 5]))
    plot_y_vs_x(gx_vec, Δ2_vec; x_name=raw"$g_x/g_0$", y_name=raw"$\Delta_2$", fig_ax=(fig, ax[2, 5]))
    plot_y_vs_x(gx_vec, (sgn.(Δ1_vec .- Δ2_vec)); x_name=raw"$g_x/g_0$", y_name=raw"sgn($\Delta_1 - \Delta_2$)", fig_ax=(fig, ax[2, 4]))
    # plot_y_vs_x(gx_vec, abs_O_odd_vec; x_name=raw"$g_x/g_0$", y_name=raw"$|O_{\mathrm{odd}}|$", fig_ax=(fig, ax[2, 4]))
    # plot_y_vs_x(gx_vec, S_k0_vec; x_name=raw"$g_x/g_0$", y_name=raw"$S_{k=0}$", fig_ax=(fig, ax[3, 1]))
    # plot_y_vs_x(gx_vec, M_k0_vec; x_name=raw"$g_x/g_0$", y_name=raw"$M_{k=0}$", fig_ax=(fig, ax[3, 2]))
    plot_y_vs_x(gx_vec, O_odd_vec; x_name=raw"$g_x/g_0$", y_name=raw"$O_{\mathrm{odd}}$", fig_ax=(fig, ax[3, 1]))
    plot_y_vs_x(gx_vec, O_even_vec; x_name=raw"$g_x/g_0$", y_name=raw"$O_{\mathrm{even}}$", fig_ax=(fig, ax[3, 2]))
    plot_y_vs_x(gx_vec, S_kπ_vec; x_name=raw"$g_x/g_0$", y_name=raw"$S_{k=\pi}$", fig_ax=(fig, ax[3, 3]))
    plot_y_vs_x(gx_vec, M_kπ_vec; x_name=raw"$g_x/g_0$", y_name=raw"$M_{k=\pi}$", fig_ax=(fig, ax[3, 4]))
    plot_y_vs_x(gx_vec, c_vec; x_name=raw"$g_x/g_0$", y_name=raw"$c$", fig_ax=(fig, ax[3, 5]))
    plot_y_vs_x(gx_vec, prob_Q_vec; x_name=raw"$g_x/g_0$", y_name=raw"$P_Q$", fig_ax=(fig, ax[4, 1]))
    plot_y_vs_x(gx_vec, avg_NN_g1_vec; x_name=raw"$g_x/g_0$", y_name=raw"NN $g_1$", fig_ax=(fig, ax[4, 2]))
    plot_y_vs_x(gx_vec, avg_NN_g2_vec; x_name=raw"$g_x/g_0$", y_name=raw"NN $g_2$", fig_ax=(fig, ax[4, 3]))
    plot_y_vs_x(gx_vec, avg_NN_n_vec; x_name=raw"$g_x/g_0$", y_name=raw"NN $n$", fig_ax=(fig, ax[4, 4]))
    plot_y_vs_x(gx_vec, avg_NN_n_vec; x_name=raw"$g_x/g_0$", y_name=raw"NN $n$", fig_ax=(fig, ax[4, 4]))
    # plot_y_vs_x(gx_vec, avg_sep2_n_vec; x_name=raw"$g_x/g_0$", y_name=raw"sep2 $n$", fig_ax=(fig, ax[4, 5]))

    set_plot_defaults(fig, ax, addGrid=false)
    plt.subplots_adjust(hspace=0.4)
    save_plot(file_name, @__DIR__, file_type="pdf")
    # plt.close("all")
    # plt.show()
end # function

function plotN_order_parameters_vs_J1_gx(abs_J1_vec, gx_vec, δN_arr, g1_arr, g2_arr, integ_g1_arr, integ_g2_arr, integ_S_k_arr, integ_M_k_arr; file_name="Order_parameters_vs_J1_gx")
    fig, ax = plt.subplots(2, 4)
    fig.set_size_inches(24, 12)

    plot_u_vs_xy(abs_J1_vec, gx_vec, δN_arr'; x_name=raw"$|J_1|/g_0$", y_name=raw"$g_x/g_0$", u_name=raw"$\delta N$", fig_ax=(fig, ax[1, 1]))
    plot_u_vs_xy(abs_J1_vec, gx_vec, g1_arr'; x_name=raw"$|J_1|/g_0$", y_name=raw"$g_x/g_0$", u_name=raw"$g_1$", fig_ax=(fig, ax[1, 2]))
    plot_u_vs_xy(abs_J1_vec, gx_vec, g2_arr'; x_name=raw"$|J_1|/g_0$", y_name=raw"$g_x/g_0$", u_name=raw"$g_2$", fig_ax=(fig, ax[1, 3]))
    plot_u_vs_xy(abs_J1_vec, gx_vec, integ_g1_arr'; x_name=raw"$|J_1|/g_0$", y_name=raw"$g_x/g_0$", u_name=raw"$\sum |g_1|$", fig_ax=(fig, ax[1, 4]))
    plot_u_vs_xy(abs_J1_vec, gx_vec, integ_g2_arr'; x_name=raw"$|J_1|/g_0$", y_name=raw"$g_x/g_0$", u_name=raw"$\sum |g_2|$", fig_ax=(fig, ax[2, 1]))
    plot_u_vs_xy(abs_J1_vec, gx_vec, integ_S_k_arr'; x_name=raw"$|J_1|/g_0$", y_name=raw"$g_x/g_0$", u_name=raw"$\sum |S_k|$", fig_ax=(fig, ax[2, 2]))
    plot_u_vs_xy(abs_J1_vec, gx_vec, integ_M_k_arr'; x_name=raw"$|J_1|/g_0$", y_name=raw"$g_x/g_0$", u_name=raw"$\sum |M_k|$", fig_ax=(fig, ax[2, 3]))
    # plot_u_vs_xy(abs_J1_vec, gx_vec, abs_O_odd_arr'; x_name=raw"$|J_1|/g_0$", y_name=raw"$g_x/g_0$", u_name=raw"$|O_{\mathrm{odd}}|$", fig_ax=(fig, ax[2, 4]))

    set_plot_defaults(fig, ax, addGrid=false)
    plt.subplots_adjust(hspace=0.4)
    save_plot(file_name, @__DIR__, file_type="pdf")
    # plt.show()
end # function

function plotN_order_parameters_vs_J1_gx_v2(abs_J1_vec, gx_vec, g1_arr, g2_arr, δN_arr, S_kπ_arr, O_even_arr, O_odd_arr, κ2_arr, M_k0_arr, M_kπ_arr, c_arr, prob_Q_arr; file_name="Order_parameters_vs_J1_gx")
    fig, ax = plt.subplots(3, 4)
    fig.set_size_inches(24, 18)

    plot_u_vs_xy(abs_J1_vec, gx_vec, g1_arr'; x_name=raw"$|J_1|/g_0$", y_name=raw"$g_x/g_0$", u_name=raw"$g_1$", fig_ax=(fig, ax[1, 1]))
    plot_u_vs_xy(abs_J1_vec, gx_vec, g2_arr'; x_name=raw"$|J_1|/g_0$", y_name=raw"$g_x/g_0$", u_name=raw"$g_2$", fig_ax=(fig, ax[1, 2]))
    plot_u_vs_xy(abs_J1_vec, gx_vec, δN_arr'; x_name=raw"$|J_1|/g_0$", y_name=raw"$g_x/g_0$", u_name=raw"$\delta N$", fig_ax=(fig, ax[1, 3]))
    plot_u_vs_xy(abs_J1_vec, gx_vec, S_kπ_arr'; x_name=raw"$|J_1|/g_0$", y_name=raw"$g_x/g_0$", u_name=raw"$S_k\pi$", fig_ax=(fig, ax[1, 4]))
    plot_u_vs_xy(abs_J1_vec, gx_vec, O_even_arr'; x_name=raw"$|J_1|/g_0$", y_name=raw"$g_x/g_0$", u_name=raw"$O_{\mathrm{even}}$", fig_ax=(fig, ax[2, 1]))
    plot_u_vs_xy(abs_J1_vec, gx_vec, κ2_arr'; x_name=raw"$|J_1|/g_0$", y_name=raw"$g_x/g_0$", u_name=raw"$\kappa_2$", fig_ax=(fig, ax[2, 2]))
    plot_u_vs_xy(abs_J1_vec, gx_vec, M_k0_arr'; x_name=raw"$|J_1|/g_0$", y_name=raw"$g_x/g_0$", u_name=raw"$M_k0$", fig_ax=(fig, ax[2, 3]))
    plot_u_vs_xy(abs_J1_vec, gx_vec, M_kπ_arr'; x_name=raw"$|J_1|/g_0$", y_name=raw"$g_x/g_0$", u_name=raw"$M_k\pi$", fig_ax=(fig, ax[2, 4]))
    plot_u_vs_xy(abs_J1_vec, gx_vec, prob_Q_arr'; x_name=raw"$|J_1|/g_0$", y_name=raw"$g_x/g_0$", u_name=raw"$P_Q$", fig_ax=(fig, ax[3, 1]))
    plot_u_vs_xy(abs_J1_vec, gx_vec, O_odd_arr'; x_name=raw"$|J_1|/g_0$", y_name=raw"$g_x/g_0$", u_name=raw"$O_{\mathrm{odd}}$", fig_ax=(fig, ax[3, 2]))
    plot_u_vs_xy(abs_J1_vec, gx_vec, c_arr'; x_name=raw"$|J_1|/g_0$", y_name=raw"$g_x/g_0$", u_name=raw"$c$", vlim=[-1, 3], fig_ax=(fig, ax[3, 3]))
    # plot_u_vs_xy(abs_J1_vec, gx_vec, kmax_M_arr'; x_name=raw"$|J_1|/g_0$", y_name=raw"$g_x/g_0$", u_name=raw"$kmax M$", fig_ax=(fig, ax[3, 3]))
    # plot_u_vs_xy(abs_J1_vec, gx_vec, M_kmax_arr'; x_name=raw"$|J_1|/g_0$", y_name=raw"$g_x/g_0$", u_name=raw"$M kmax$", fig_ax=(fig, ax[3, 4]))

    set_plot_defaults(fig, ax, addGrid=false)
    plt.subplots_adjust(hspace=0.4)
    save_plot(file_name, @__DIR__, file_type="pdf")
    # plt.show()
end # function

function plotN_fit_res_vs_J1_gx(abs_J1_vec, gx_vec, α_g1_arr, res_g1_arr, α_g2_arr, res_g2_arr, exp_ξ_g2_arr, exp_res_g2_arr; vlim=nothing, file_name="fit_res_vs_J1_gx")
    fig, ax = plt.subplots(2, 4)
    fig.set_size_inches(24, 12)

    plot_u_vs_xy(abs_J1_vec, gx_vec, α_g1_arr'; x_name=raw"$|J_1|/g_0$", y_name=raw"$g_x/g_0$", u_name=raw"$\alpha, g_1$", vlim=vlim, fig_ax=(fig, ax[1, 1]))
    plot_u_vs_xy(abs_J1_vec, gx_vec, log10.(res_g1_arr)'; x_name=raw"$|J_1|/g_0$", y_name=raw"$g_x/g_0$", u_name=raw"log10 res$, g_1$", fig_ax=(fig, ax[1, 2]))
    plot_u_vs_xy(abs_J1_vec, gx_vec, α_g2_arr'; x_name=raw"$|J_1|/g_0$", y_name=raw"$g_x/g_0$", u_name=raw"$\alpha, g_2$", vlim=vlim, fig_ax=(fig, ax[1, 3]))
    plot_u_vs_xy(abs_J1_vec, gx_vec, log10.(res_g2_arr)'; x_name=raw"$|J_1|/g_0$", y_name=raw"$g_x/g_0$", u_name=raw"log10 res$, g_2$", fig_ax=(fig, ax[1, 4]))

    α_crit = 1.1
    plot_u_vs_xy(abs_J1_vec, gx_vec, (α_g1_arr .< α_crit)'; x_name=raw"$|J_1|/g_0$", y_name=raw"$g_x/g_0$", u_name=raw"$\alpha, g_1 < $"*"$(α_crit)", fig_ax=(fig, ax[2, 1]))
    plot_u_vs_xy(abs_J1_vec, gx_vec, (α_g2_arr .< α_crit)'; x_name=raw"$|J_1|/g_0$", y_name=raw"$g_x/g_0$", u_name=raw"$\alpha, g_2 < $"*"$(α_crit)", fig_ax=(fig, ax[2, 2]))

    plot_u_vs_xy(abs_J1_vec, gx_vec, exp_ξ_g2_arr'; x_name=raw"$|J_1|/g_0$", y_name=raw"$g_x/g_0$", u_name=raw"$\xi, g_2$", vlim=vlim, fig_ax=(fig, ax[2, 3]))
    plot_u_vs_xy(abs_J1_vec, gx_vec, (res_g2_arr .< exp_res_g2_arr)'; x_name=raw"$|J_1|/g_0$", y_name=raw"$g_x/g_0$", u_name=raw"$g_2$"*" power law or exp?", fig_ax=(fig, ax[2, 4]))

    set_plot_defaults(fig, ax, addGrid=false)
    plt.subplots_adjust(hspace=0.4)
    save_plot(file_name, @__DIR__, file_type="pdf")
    # plt.show()
end # function

function plotN_order_parameters_gaps_vs_J1_gx(abs_J1_vec, gx_vec, δN_arr, g1_arr, g2_arr, Δ1_arr, Δ2_arr, integ_g1_arr, integ_g2_arr, integ_S_k_arr, integ_M_k_arr, S_k0_arr, M_k0_arr, S_kπ_arr, M_kπ_arr, c_arr, prob_Q_arr, avg_NN_g1_arr, avg_NN_g2_arr, avg_NN_n_arr; file_name="Order_parameters_gaps_vs_J1_gx")
    fig, ax = plt.subplots(4, 5)
    fig.set_size_inches(30, 22)

    plot_u_vs_xy(abs_J1_vec, gx_vec, δN_arr'; x_name=raw"$|J_1|/g_0$", y_name=raw"$g_x/g_0$", u_name=raw"$\delta N$", fig_ax=(fig, ax[1, 1]))
    plot_u_vs_xy(abs_J1_vec, gx_vec, g1_arr'; x_name=raw"$|J_1|/g_0$", y_name=raw"$g_x/g_0$", u_name=raw"$g_1$", fig_ax=(fig, ax[1, 2]))
    plot_u_vs_xy(abs_J1_vec, gx_vec, g2_arr'; x_name=raw"$|J_1|/g_0$", y_name=raw"$g_x/g_0$", u_name=raw"$g_2$", fig_ax=(fig, ax[1, 3]))
    plot_u_vs_xy(abs_J1_vec, gx_vec, integ_g1_arr'; x_name=raw"$|J_1|/g_0$", y_name=raw"$g_x/g_0$", u_name=raw"$\sum |g_1|$", fig_ax=(fig, ax[1, 4]))
    plot_u_vs_xy(abs_J1_vec, gx_vec, integ_g2_arr'; x_name=raw"$|J_1|/g_0$", y_name=raw"$g_x/g_0$", u_name=raw"$\sum |g_2|$", fig_ax=(fig, ax[2, 1]))
    plot_u_vs_xy(abs_J1_vec, gx_vec, integ_S_k_arr'; x_name=raw"$|J_1|/g_0$", y_name=raw"$g_x/g_0$", u_name=raw"$\sum |S_k|$", fig_ax=(fig, ax[2, 2]))
    plot_u_vs_xy(abs_J1_vec, gx_vec, integ_M_k_arr'; x_name=raw"$|J_1|/g_0$", y_name=raw"$g_x/g_0$", u_name=raw"$\sum |M_k|$", fig_ax=(fig, ax[2, 3]))
    plot_u_vs_xy(abs_J1_vec, gx_vec, Δ1_arr'; x_name=raw"$|J_1|/g_0$", y_name=raw"$g_x/g_0$", u_name=raw"$\Delta_1$", fig_ax=(fig, ax[1, 5]))
    plot_u_vs_xy(abs_J1_vec, gx_vec, Δ2_arr'; x_name=raw"$|J_1|/g_0$", y_name=raw"$g_x/g_0$", u_name=raw"$\Delta_2$", fig_ax=(fig, ax[2, 5]))
    plot_u_vs_xy(abs_J1_vec, gx_vec, (sgn.(Δ1_arr .- Δ2_arr))'; x_name=raw"$|J_1|/g_0$", y_name=raw"$g_x/g_0$", u_name=raw"sgn($\Delta_1 - \Delta_2$)", fig_ax=(fig, ax[2, 4]))
    # plot_u_vs_xy(abs_J1_vec, gx_vec, abs_O_odd_arr'; x_name=raw"$|J_1|/g_0$", y_name=raw"$g_x/g_0$", u_name=raw"$|O_{\mathrm{odd}}|$", fig_ax=(fig, ax[2, 4]))
    plot_u_vs_xy(abs_J1_vec, gx_vec, S_k0_arr'; x_name=raw"$|J_1|/g_0$", y_name=raw"$g_x/g_0$", u_name=raw"$S_{k=0}$", fig_ax=(fig, ax[3, 1]))
    plot_u_vs_xy(abs_J1_vec, gx_vec, M_k0_arr'; x_name=raw"$|J_1|/g_0$", y_name=raw"$g_x/g_0$", u_name=raw"$M_{k=0}$", fig_ax=(fig, ax[3, 2]))
    plot_u_vs_xy(abs_J1_vec, gx_vec, S_kπ_arr'; x_name=raw"$|J_1|/g_0$", y_name=raw"$g_x/g_0$", u_name=raw"$S_{k=\pi}$", fig_ax=(fig, ax[3, 3]))
    plot_u_vs_xy(abs_J1_vec, gx_vec, M_kπ_arr'; x_name=raw"$|J_1|/g_0$", y_name=raw"$g_x/g_0$", u_name=raw"$M_{k=\pi}$", fig_ax=(fig, ax[3, 4]))
    plot_u_vs_xy(abs_J1_vec, gx_vec, c_arr'; x_name=raw"$|J_1|/g_0$", y_name=raw"$g_x/g_0$", u_name=raw"$c$", fig_ax=(fig, ax[3, 5]))
    plot_u_vs_xy(abs_J1_vec, gx_vec, prob_Q_arr'; x_name=raw"$|J_1|/g_0$", y_name=raw"$g_x/g_0$", u_name=raw"$P_Q$", fig_ax=(fig, ax[4, 1]))
    plot_u_vs_xy(abs_J1_vec, gx_vec, avg_NN_g1_arr'; x_name=raw"$|J_1|/g_0$", y_name=raw"$g_x/g_0$", u_name=raw"NN $g_1$", fig_ax=(fig, ax[4, 2]))
    plot_u_vs_xy(abs_J1_vec, gx_vec, avg_NN_g2_arr'; x_name=raw"$|J_1|/g_0$", y_name=raw"$g_x/g_0$", u_name=raw"NN $g_2$", fig_ax=(fig, ax[4, 3]))
    plot_u_vs_xy(abs_J1_vec, gx_vec, avg_NN_n_arr'; x_name=raw"$|J_1|/g_0$", y_name=raw"$g_x/g_0$", u_name=raw"NN $n$", fig_ax=(fig, ax[4, 4]))

    set_plot_defaults(fig, ax, addGrid=false)
    plt.subplots_adjust(hspace=0.4)
    save_plot(file_name, @__DIR__, file_type="pdf")
    # plt.show()
end # function

function plotN_order_parameters_gaps_vs_J1_gx_v2(abs_J1_vec, gx_vec, g1_arr, g2_arr, δN_arr, S_kπ_arr, O_even_arr, O_odd_arr, κ2_arr, Δ1_arr, Δ2_arr, Δ1_bulk_arr, Δ2_bulk_arr, prob_Q_arr; file_name="Order_parameters_gaps_vs_J1_gx")
    fig, ax = plt.subplots(3, 4)
    fig.set_size_inches(24, 18)

    Δ1_vlim = [0.0, 3.0]
    Δ2_vlim = [0.0, 0.1]

    plot_u_vs_xy(abs_J1_vec, gx_vec, g1_arr'; x_name=raw"$|J_1|/g_0$", y_name=raw"$g_x/g_0$", u_name=raw"$g_1$", fig_ax=(fig, ax[1, 1]))
    plot_u_vs_xy(abs_J1_vec, gx_vec, g2_arr'; x_name=raw"$|J_1|/g_0$", y_name=raw"$g_x/g_0$", u_name=raw"$g_2$", fig_ax=(fig, ax[1, 2]))
    plot_u_vs_xy(abs_J1_vec, gx_vec, δN_arr'; x_name=raw"$|J_1|/g_0$", y_name=raw"$g_x/g_0$", u_name=raw"$\delta N$", fig_ax=(fig, ax[1, 3]))
    plot_u_vs_xy(abs_J1_vec, gx_vec, S_kπ_arr'; x_name=raw"$|J_1|/g_0$", y_name=raw"$g_x/g_0$", u_name=raw"$S_k\pi$", fig_ax=(fig, ax[1, 4]))
    plot_u_vs_xy(abs_J1_vec, gx_vec, O_even_arr'; x_name=raw"$|J_1|/g_0$", y_name=raw"$g_x/g_0$", u_name=raw"$O_{\mathrm{even}}$", fig_ax=(fig, ax[2, 1]))
    plot_u_vs_xy(abs_J1_vec, gx_vec, κ2_arr'; x_name=raw"$|J_1|/g_0$", y_name=raw"$g_x/g_0$", u_name=raw"$\kappa_2$", fig_ax=(fig, ax[2, 2]))
    plot_u_vs_xy(abs_J1_vec, gx_vec, Δ1_bulk_arr'; x_name=raw"$|J_1|/g_0$", y_name=raw"$g_x/g_0$", u_name=raw"bulk $\Delta_1$", vlim=Δ1_vlim, fig_ax=(fig, ax[2, 3]))
    plot_u_vs_xy(abs_J1_vec, gx_vec, Δ2_bulk_arr'; x_name=raw"$|J_1|/g_0$", y_name=raw"$g_x/g_0$", u_name=raw"bulk $\Delta_2$", vlim=Δ2_vlim, fig_ax=(fig, ax[2, 4]))
    plot_u_vs_xy(abs_J1_vec, gx_vec, prob_Q_arr'; x_name=raw"$|J_1|/g_0$", y_name=raw"$g_x/g_0$", u_name=raw"$P_Q$", fig_ax=(fig, ax[3, 1]))
    # plot_u_vs_xy(abs_J1_vec, gx_vec, O_odd_arr'; x_name=raw"$|J_1|/g_0$", y_name=raw"$g_x/g_0$", u_name=raw"$O_{\mathrm{odd}}$", fig_ax=(fig, ax[3, 2]))
    plot_u_vs_xy(abs_J1_vec, gx_vec, Δ1_arr'; x_name=raw"$|J_1|/g_0$", y_name=raw"$g_x/g_0$", u_name=raw"$\Delta_1$", vlim=Δ1_vlim, fig_ax=(fig, ax[3, 2]))
    plot_u_vs_xy(abs_J1_vec, gx_vec, Δ2_arr'; x_name=raw"$|J_1|/g_0$", y_name=raw"$g_x/g_0$", u_name=raw"$\Delta_2$", vlim=Δ2_vlim, fig_ax=(fig, ax[3, 3]))
    plot_u_vs_xy(abs_J1_vec, gx_vec, (sgn.(Δ1_arr - Δ2_arr))'; x_name=raw"$|J_1|/g_0$", y_name=raw"$g_x/g_0$", u_name=raw"sgn $\Delta_1 - \Delta_2$", fig_ax=(fig, ax[3, 4]))
    # plot_u_vs_xy(abs_J1_vec, gx_vec, (sgn.(Δ1_bulk_arr - Δ2_bulk_arr))'; x_name=raw"$|J_1|/g_0$", y_name=raw"$g_x/g_0$", u_name=raw"sgn $\Delta_1 - \Delta_2$", fig_ax=(fig, ax[3, 4]))

    set_plot_defaults(fig, ax, addGrid=false)
    plt.subplots_adjust(hspace=0.4)
    save_plot(file_name, @__DIR__, file_type="pdf")
    # plt.show()
end # function

function plotN_order_parameters_vs_gx_Npart(gx_vec, N_part_vec, δN_arr, g1_arr, g2_arr, integ_g1_arr, integ_g2_arr, integ_S_k_arr, integ_M_k_arr, abs_O_odd_arr; file_name="Order_parameters_vs_gx_Npart")
    fig, ax = plt.subplots(2, 4)
    fig.set_size_inches(24, 12)

    plot_u_vs_xy(gx_vec, N_part_vec, δN_arr'; x_name=raw"$g_x/g_0$", y_name=raw"Particle number", u_name=raw"$\delta N$", fig_ax=(fig, ax[1, 1]))
    plot_u_vs_xy(gx_vec, N_part_vec, g1_arr'; x_name=raw"$g_x/g_0$", y_name=raw"Particle number", u_name=raw"$g_1$", fig_ax=(fig, ax[1, 2]))
    plot_u_vs_xy(gx_vec, N_part_vec, g2_arr'; x_name=raw"$g_x/g_0$", y_name=raw"Particle number", u_name=raw"$g_2$", fig_ax=(fig, ax[1, 3]))
    plot_u_vs_xy(gx_vec, N_part_vec, integ_g1_arr'; x_name=raw"$g_x/g_0$", y_name=raw"Particle number", u_name=raw"$\sum |g_1|$", fig_ax=(fig, ax[1, 4]))
    plot_u_vs_xy(gx_vec, N_part_vec, integ_g2_arr'; x_name=raw"$g_x/g_0$", y_name=raw"Particle number", u_name=raw"$\sum |g_2|$", fig_ax=(fig, ax[2, 1]))
    plot_u_vs_xy(gx_vec, N_part_vec, integ_S_k_arr'; x_name=raw"$g_x/g_0$", y_name=raw"Particle number", u_name=raw"$\sum |S_k|$", fig_ax=(fig, ax[2, 2]))
    plot_u_vs_xy(gx_vec, N_part_vec, integ_M_k_arr'; x_name=raw"$g_x/g_0$", y_name=raw"Particle number", u_name=raw"$\sum |M_k|$", fig_ax=(fig, ax[2, 3]))
    # plot_u_vs_xy(abs_J1_vec, gx_vec, abs_O_odd_arr'; x_name=raw"$g_x/g_0$", y_name=raw"$g_x/g_0$", u_name=raw"$|O_{\mathrm{odd}}|$", fig_ax=(fig, ax[2, 4]))

    set_plot_defaults(fig, ax, addGrid=false)
    plt.subplots_adjust(hspace=0.4)
    save_plot(file_name, @__DIR__, file_type="pdf")
    # plt.show()
end # function

function plotN_order_parameters_vs_gz_gx(gz_vec, gx_vec, δN_arr, g1_arr, g2_arr, integ_S_k_arr, integ_M_k_arr; file_name="Order_parameters_vs_gz_gx")
    fig, ax = plt.subplots(2, 3)
    fig.set_size_inches(16, 12)

    plot_u_vs_xy(gz_vec, gx_vec, δN_arr'; x_name=raw"$g_z/g_0$", y_name=raw"$g_x/g_0$", u_name=raw"$\delta N$", fig_ax=(fig, ax[1, 1]))
    plot_u_vs_xy(gz_vec, gx_vec, g1_arr'; x_name=raw"$g_z/g_0$", y_name=raw"$g_x/g_0$", u_name=raw"$g_1$", fig_ax=(fig, ax[1, 2]))
    plot_u_vs_xy(gz_vec, gx_vec, g2_arr'; x_name=raw"$g_z/g_0$", y_name=raw"$g_x/g_0$", u_name=raw"$g_2$", fig_ax=(fig, ax[1, 3]))
    plot_u_vs_xy(gz_vec, gx_vec, integ_S_k_arr'; x_name=raw"$g_z/g_0$", y_name=raw"$g_x/g_0$", u_name=raw"$\sum S_k$", fig_ax=(fig, ax[2, 1]))
    plot_u_vs_xy(gz_vec, gx_vec, integ_M_k_arr'; x_name=raw"$g_z/g_0$", y_name=raw"$g_x/g_0$", u_name=raw"$\sum M_k$", fig_ax=(fig, ax[2, 2]))

    set_plot_defaults(fig, ax, addGrid=false)
    plt.subplots_adjust(hspace=0.4)
    save_plot(file_name, @__DIR__, file_type="pdf")
    # plt.show()
end # function
##### END OF PLOTTING #####

##### LOWEST ORDER ROUTINES #####
function routine_single(N_lat, N_part, boson_dim, conserveParticleNumber, μ, N_sweeps, max_dim, cutoff, noise; showPlot=true)
    k_arr = LinRange(0.0, 2π, N_lat+1)[1:N_lat]
    # i_start = max(N_lat÷10, 1)
    # i_end = min(N_lat - N_lat÷10, N_lat)
    # k_arr = LinRange(0.0, 2π, i_end-i_start+2)[1:(i_end-i_start+1)]

    # sites = siteinds("Boson", N_lat; dim=boson_dim, conserve_qns=true)
    sites = siteinds("Boson", N_lat; dim=boson_dim, conserve_qns=conserveParticleNumber)

    ##### HAMILTONIAN #####
    # Ham = calc_BH_Ham(sites, N_lat, conserveParticleNumber, μ; J=1.0, U=-7.0)
    # J = 0.1; U = 1.0;
    # Ham = calc_BH_Ham(sites, N_lat, conserveParticleNumber, μ; J=J, U=U)
    abs_J1 = 0.0
    # abs_J_2 = 0.000  # 0.035
    Ham = calc_brick_wall_Ham(sites, N_lat, conserveParticleNumber, μ;
          # J_1=0.0, J_2=0.0, g_0=1.0, g_x=0.0, g_z=0.0, G_000=1.0000, G_001=-0.5000, G_011=0.5000)  # MI, p-band
          # J_1=0.0, J_2=0.0, g_0=1.0, g_x=0.0, g_z=1.0, G_000=1.0000, G_001=-0.5000, G_011=0.5000)  # SF, p-band
          # J_1=0.0, J_2=0.0, g_0=0.1, g_x=1.0, g_z=0.0, G_000=1.0000, G_001=0.2122, G_011=0.1666)  # PSF, s-band
          # J_1=0.0, J_2=0.0, g_0=0.5, g_x=-0.7, g_z=0.0, G_000=1.0000, G_001=0.2122, G_011=0.1666)  # Giedrius PSF, s-band (Failed for unit filling)
          # J_1=0.3, J_2=0.0, g_0=1.0, g_x=-5.0, g_z=0.0, G_000=1.0000, G_001=0.2122, G_011=0.1666)  # Giedrius PSF, s-band (Main one, 2025-11)
          # J_1=0.0, J_2=0.0, g_0=1.0, g_x=-3.9, g_z=0.0, G_000=1.0000, G_001=0.2122, G_011=0.1666)  # s-band
          J_1=abs_J1, J_2=-abs_J1*(10^(0.5521)), g_0=1.0, g_x=-3.9, g_z=0.0, G_000=1.0000, G_001=0.2122, G_011=0.1666)  # s-band
          # J_1=-abs_J1, J_2=abs_J1*(10^(0.5521)), g_0=1.0, g_x=0.0, g_z=-1.5, G_000=1.0000, G_001=0.2122, G_011=0.1666)  # s-band
          # J_1=0.0, J_2=0.0, g_0=0.0, g_x=1.0, g_z=0.0, G_000=1.0000, G_001=-0.5000, G_011=0.5000)  # PSF, p-band
          # J_1=0.0, J_2=0.0, g_0=0.5, g_x=-0.7, g_z=0.0, G_000=1.0000, G_001=-0.5000, G_011=0.5000)  # Giedrius PSF, p-band
          # J_1=0.0, J_2=0.0, g_0=1.0, g_x=-3.0, g_z=0.0, G_000=1.0000, G_001=-0.5000, G_011=0.5000)  # Giedrius PSF, p-band
          # J_1=-abs_J_2/(10^(0.5521)), J_2=abs_J_2, g_0=0.5, g_x=0.0, g_z=0.0, G_000=1.0000, G_001=0.2122, G_011=0.1666)  # SS, s-band
          # J_1=-abs_J_2/(10^(0.375)), J_2=-abs_J_2, g_0=0.5, g_x=0.0, g_z=0.0, G_000=1.0000, G_001=-0.5000, G_011=0.5000)  # SS, p-band
          # J_1=-abs_J_2/(10^(0.375)), J_2=-abs_J_2, g_0=0.5, g_x=-0.5, g_z=0.05, G_000=1.0000, G_001=-0.5000, G_011=0.5000)  # SS, p-band
          # J_1=0.0, J_2=0.0, g_0=0.5, g_x=-0.5, g_z=0.0, G_000=1.0000, G_001=-0.5000, G_011=0.5000)  # DW, p-band
          # J_1=0.0, J_2=0.0, g_0=0.8, g_x=-0.5, g_z=0.0, G_000=1.0000, G_001=0.2122, G_011=0.1666)  # DWx 7., s-band (Giedrius, 2024-10-14)
          # J_1=0.0, J_2=0.0, g_0=0.8, g_x=-0.5, g_z=0.0, G_000=1.0000, G_001=-0.5000, G_011=0.5000)  # DWx 7., p-band (Giedrius, 2024-10-14)
    # Ham = calc_Zhou_PSF_deep_lat_Ham(sites, N_lat, conserveParticleNumber, μ;
    #                                  J=1.0, U=1.0)
    # Ham = calc_Huber_interacting_Creutz_ladder(sites, N_lat, conserveParticleNumber, μ;
    #                                            # t=1.0, U=1.0, m=0.05, ϵ=0.0, δ=0.0)
                                               # t=1.0, U=1.0, m=0.00, ϵ=0.15, δ=0.0)
    # Ham = calc_extended_int_BH_Ham(sites, N_lat, conserveParticleNumber, μ;
    #         J_1=1.0, J_2=0.0, U=1.0, V=0.0, P=0.0, D=0.0)
            # J_1=0.0, J_2=0.0, U=1.0, V=1.0*0.1666, P=0.5*0.1666, D=0.0)
            # J_1=0.0, J_2=0.0, U=(2*g_0+g_x)*1.0000, V=2*g_0*0.1666, P=0.5*g_x*0.1666, D=-g_z*0.2122)

    # bulk_range = (N_lat÷10):(N_lat-N_lat÷10)
    # Ham_bulk = calc_brick_wall_Ham(sites, bulk_range, conserveParticleNumber, μ;
    #       J_1=1.0, J_2=-1.0, g_0=1.0, g_x=0.0, g_z=0.0, G_000=1.0000, G_001=0.2122, G_011=0.1666)  # s-band
    ##### END OF HAMILTONIAN #####

    # Ham_mat = Array(Ham, sites[1]', sites[2]', sites[1], sites[2])
    # @show sum(abs.(Ham_mat - Ham_mat'))

    ##### INITIAL STATE #####
    ψ0 = create_ψ0(sites, N_lat, N_part, boson_dim; linkdims=200)
    # ψ0 = create_ψ0(sites, N_lat, N_part, boson_dim)
    # ψ0 = create_homogeneous_ψ0(sites, N_lat, N_part, boson_dim)
    # ψ0 = create_Mott_ψ0(sites, N_lat, N_part; createRandom=true)
    # ψ0 = create_Mott_ψ0(sites, N_lat, N_part; createRandom=false)
    # ψ0 = create_pair_ψ0(sites, N_lat, N_part, boson_dim)
    # plt.plot(expect(ψ0, "n"), color="magenta", ls="--")
    ##### END OF INITIAL STATE #####

    E, ψ = dmrg(Ham, ψ0; nsweeps=N_sweeps, maxdim=max_dim, cutoff=cutoff, noise=noise)

    ##### EXPECTATION VALUES #####
    expectation_values = calc_expectation_values(ψ, Ham, sites, k_arr, N_lat)
    # expectation_values = calc_expectation_values(ψ, Ham_bulk, sites, k_arr, N_lat)
    expect_E, δN, n_arr, g1, g1_vec, g1_matrix, g2, g2_vec, g2_matrix, κ2, M_k_vec, S_k_vec, M_k0, M_kπ, M_k2π3, S_kπ, O_odd, O_even, c, prob_Q, prob_Q_bulk, prob_Q_mod, prob_Q_mod_bulk, avg_NN_g1, avg_NN_g2, avg_NN_n = expectation_values["expect_E"], expectation_values["δN"], expectation_values["n_arr"], expectation_values["g1"], expectation_values["g1_vec"], expectation_values["g1_matrix"], expectation_values["g2"], expectation_values["g2_vec"], expectation_values["g2_matrix"], expectation_values["κ2"], expectation_values["M_k_vec"], expectation_values["S_k_vec"], expectation_values["M_k0"], expectation_values["M_kπ"], expectation_values["M_k2π3"], expectation_values["S_kπ"], expectation_values["O_odd"], expectation_values["O_even"], expectation_values["c"], expectation_values["prob_Q"], expectation_values["prob_Q_bulk"], expectation_values["prob_Q_mod"], expectation_values["prob_Q_mod_bulk"], expectation_values["avg_NN_g1"], expectation_values["avg_NN_g2"], expectation_values["avg_NN_n"]
    # S_kπ = real(S_k_vec[N_lat÷2+1])
    integ_S_kπ = sum(abs.(S_k_vec[(N_lat÷2-10):(N_lat÷2+10)]))
    @show expect_E
    @show δN
    @show g1
    @show g2
    @show κ2
    @show S_kπ
    @show M_k0
    @show M_kπ
    @show M_k2π3
    @show integ_S_kπ
    @show c
    @show prob_Q
    @show prob_Q_bulk
    @show prob_Q_mod
    @show prob_Q_mod_bulk
    @show O_odd
    @show O_even
    @show avg_NN_g1
    @show avg_NN_g2
    @show avg_NN_n

    merge!(expectation_values, Dict("E" => E))
    # merge!(expectation_values, Dict("ψ" => ψ))

    # ε_strong = calc_1D_BH_energy_per_site_strong_U_expansion(J, U)
    # ε_weak = calc_1D_BH_energy_per_site_weak_U_expansion(J, U)
    # @show ε_strong
    # @show ε_weak

    ψ_Mott = create_Mott_ψ0(sites, N_lat, N_part; createRandom=false)
    expect_E2_Mott = inner(Ham, ψ_Mott, Ham, ψ_Mott)
    expect_E_Mott = inner(ψ_Mott', Ham, ψ_Mott)
    @show expect_E_Mott
    @show expect_E2_Mott - expect_E_Mott^2
    ##### END OF EXPECTATION VALUES #####

    ##### SAMPLING #####
    N_samples = 100
    orthogonalize!(ψ, 1)
    ψ_samples = zeros(Int, N_samples, N_lat)
    for i in 1:N_samples
        ψ_samples[i, :] = (sample(ψ) .- 1)
    end # for
    println("********** SAMPLES **********")
    display([ψ_samples[i, :] for i in 1:N_samples])
    println("********** END OF SAMPLES **********")
    display([ψ_samples[i, (N_lat÷2-15):(N_lat÷2+15)] for i in 1:N_samples])
    ##### END OF SAMPLING #####

    ##### SAVING RESULTS ######
    save_path = "$(@__DIR__)/NPY/$(Dates.today())/Expectation_Values"
    date_string = """$(Dates.format(Dates.now(), "Y-mm-dd!HH.MM.SS"))"""
    mkpath(save_path)
    npzwrite("$(save_path)/Expectation_Values!$(date_string).npz", expectation_values)
    ##### END OF SAVING RESULTS ######

    ##### PLOTTING ######
    if showPlot
        # plotN_expectation_values_v1(N_lat, k_arr, n_arr, S_k_vec, g1_vec, g2_vec, g1_matrix, g2_matrix)
        plotN_expectation_values_v1(N_lat, k_arr, n_arr, S_k_vec, g1_vec, g2_vec, g1_matrix, M_k_vec)
        plotN_expectation_values_poster(N_lat, g1_vec, g2_vec, g1_matrix, g2_matrix; fig_title=raw"$|\delta N|=$"*"$(round(abs(δN), digits=3))")
    end # if
    ##### END OF PLOTTING ######

    return E, ψ, expectation_values
end # function

function routine_single_repeat(N_lat, N_part, boson_dim, conserveParticleNumber, μ, N_repeat, N_sweeps, max_dim, cutoff, noise; showPlot=true)
    k_arr = LinRange(0.0, 2π, N_lat+1)[1:N_lat]
    # @show k_arr[x2index(π, k_arr)]

    sites = siteinds("Boson", N_lat; dim=boson_dim, conserve_qns=conserveParticleNumber)

    ##### HAMILTONIAN #####
    # Ham = calc_BH_Ham(sites, N_lat, conserveParticleNumber, μ; J=1.0, U=-7.0)
    # Ham = calc_BH_Ham(sites, N_lat, conserveParticleNumber, μ; J=10.0, U=1.0)
    abs_J1 = 0.0
    Ham = calc_brick_wall_Ham(sites, N_lat, conserveParticleNumber, μ;
          # J_1=0.0, J_2=0.0, g_0=0.0, g_x=1.0, g_z=0.0, G_000=1.0000, G_001=0.2122, G_011=0.1666)
          # J_1=-abs_J1_m, J_2=abs_J1_m*(10^(0.5521)), g_0=1.0, g_x=gx_n, g_z=0.0, G_000=1.0000, G_001=-0.2122, G_011=0.1666)  # s-band
          J_1=-abs_J1, J_2=-abs_J1*(10^(0.375)), g_0=1.0, g_x=-1.0, g_z=0.0, G_000=1.0000, G_001=-0.5000, G_011=0.5000)  # p-band
    # Ham = calc_Zhou_PSF_deep_lat_Ham(sites, N_lat, conserveParticleNumber, μ;
    #                                  J=1.0, U=1.0)
    # Ham = calc_Huber_interacting_Creutz_ladder(sites, N_lat, conserveParticleNumber, μ;
    #                                            t=1.0, U=1.0, m=0.05, ϵ=0.0, δ=0.0)
                                               # t=0.0, U=1.0, m=0.00, ϵ=0.0, δ=0.0)
    ##### END OF HAMILTONIAN #####

    E_vec = Vector{Float64}([])
    ψ_vec = Vector{MPS}([])
    for ii in 1:N_repeat
        # ψ = random_mps(sites, ψ; linkdims=250)
        # ψ0 = create_ψ0(sites, N_lat, N_part, boson_dim)
        ψ0 = create_pair_ψ0(sites, N_lat, N_part, boson_dim)

        # E, ψ = dmrg(Ham, ψ0; nsweeps=N_sweeps, maxdim=max_dim, cutoff=cutoff, noise=noise)
        E, ψ = dmrg(Ham, ψ_vec, ψ0; nsweeps=N_sweeps, maxdim=max_dim, cutoff=cutoff, noise=noise)
        push!(E_vec, E)
        push!(ψ_vec, ψ)
        println("********* $ii/$(N_repeat) DONE *********")
    end # for
    E = E_vec[argmin(E_vec)]
    ψ = ψ_vec[argmin(E_vec)]

    ##### EXPECTATION VALUES #####
    # δN, n_arr, g1, g1_vec, g2, g2_vec, S_k_vec, O_odd, O_even = calc_expectation_values(ψ, sites, k_arr, N_lat)
    expectation_values = calc_expectation_values(ψ, Ham, sites, k_arr, N_lat)
    δN, n_arr, g1, g1_vec, g1_matrix, g2, g2_vec, g2_matrix, S_k_vec, M_k_vec, O_odd, O_even = expectation_values["δN"], expectation_values["n_arr"], expectation_values["g1"], expectation_values["g1_vec"], expectation_values["g1_matrix"], expectation_values["g2"], expectation_values["g2_vec"], expectation_values["g2_matrix"], expectation_values["S_k_vec"], expectation_values["M_k_vec"], expectation_values["O_odd"], expectation_values["O_even"]
    integ_g1 = sum(abs.(g1_matrix)) - tr(abs.(g1_matrix))
    integ_g2 = sum(abs.(g2_matrix)) - tr(abs.(g2_matrix))
    integ_S_k = sum(abs.(S_k_vec[(N_lat÷10):end]))
    integ_M_k = sum(abs.(M_k_vec[(N_lat÷10):end]))
    @show δN
    @show g1
    @show g2
    @show integ_g1
    @show integ_g2
    @show integ_S_k
    @show integ_M_k

    # expect_E2 = inner(Ham, ψ, Ham, ψ)
    # expect_E = inner(ψ', Ham, ψ)
    # @show expect_E2 - expect_E^2
    ##### END OF EXPECTATION VALUES #####

    # TODO Generalize somehow
    ##### COMPARISON #####
    println("********** COMPARISON **********")

    E_min = E_vec[argmin(E_vec)]
    E_max = E_vec[argmax(E_vec)]
    @show E_min
    @show E_max
    print("E_vec = "); display(E_vec)

    δN_vec = []
    g1_vec = []
    g2_vec = []
    integ_g1_vec = []
    integ_g2_vec = []
    integ_S_k_vec = []
    integ_M_k_vec = []
    for i in 1:N_repeat
        expectation_values = calc_expectation_values(ψ_vec[i], Ham, sites, k_arr, N_lat)

        δN, n_arr, g1, g1_vec, g1_matrix, g2, g2_vec, g2_matrix, S_k_vec, M_k_vec, O_odd, O_even = expectation_values["δN"], expectation_values["n_arr"], expectation_values["g1"], expectation_values["g1_vec"], expectation_values["g1_matrix"], expectation_values["g2"], expectation_values["g2_vec"], expectation_values["g2_matrix"], expectation_values["S_k_vec"], expectation_values["M_k_vec"], expectation_values["O_odd"], expectation_values["O_even"]
        integ_g1 = sum(abs.(g1_matrix)) - tr(abs.(g1_matrix))
        integ_g2 = sum(abs.(g2_matrix)) - tr(abs.(g2_matrix))
        integ_S_k = sum(abs.(S_k_vec[(N_lat÷10):end]))
        integ_M_k = sum(abs.(M_k_vec[(N_lat÷10):end]))

        push!(δN_vec, δN)
        push!(g1_vec, g1)
        push!(g2_vec, g2)
        push!(integ_g1_vec, integ_g1)
        push!(integ_g2_vec, integ_g2)
        push!(integ_S_k_vec, integ_S_k)
        push!(integ_M_k_vec, integ_M_k)
    end # for
    avg_δN = sum(δN_vec) / length(δN_vec)
    avg_g1 = sum(g1_vec) / length(g1_vec)
    avg_g2 = sum(g2_vec) / length(g2_vec)
    avg_integ_g1 = sum(integ_g1_vec) / length(integ_g1_vec)
    avg_integ_g2 = sum(integ_g2_vec) / length(integ_g2_vec)
    avg_integ_S_k = sum(integ_S_k_vec) / length(integ_S_k_vec)
    avg_integ_M_k = sum(integ_M_k_vec) / length(integ_M_k_vec)
    @show avg_δN
    @show avg_g1
    @show avg_g2
    @show avg_integ_g1
    @show avg_integ_g2
    @show avg_integ_S_k
    @show avg_integ_M_k
    std_δN = sqrt(sum(δN_vec.^2 .- avg_δN^2) / length(δN_vec))
    std_g1 = sqrt(sum(g1_vec.^2 .- avg_g1^2) / length(g1_vec))
    std_g2 = sqrt(sum(g2_vec.^2 .- avg_g2^2) / length(g2_vec))
    std_integ_g1 = sqrt(sum(integ_g1_vec.^2 .- avg_integ_g1^2) / length(integ_g1_vec))
    std_integ_g2 = sqrt(sum(integ_g2_vec.^2 .- avg_integ_g2^2) / length(integ_g2_vec))
    std_integ_S_k = sqrt(sum(integ_S_k_vec.^2 .- avg_integ_S_k^2) / length(integ_S_k_vec))
    std_integ_M_k = sqrt(sum(integ_M_k_vec.^2 .- avg_integ_M_k^2) / length(integ_M_k_vec))
    @show std_δN
    @show std_g1
    @show std_g2
    @show std_integ_g1
    @show std_integ_g2
    @show std_integ_S_k
    @show std_integ_M_k

    println("********** END OF COMPARISON **********")
    ##### END OF COMPARISON #####

    # ##### PLOTTING ######
    # if showPlot
    #     plotN_expectation_values_v1(N_lat, k_arr, n_arr, S_k_vec, g1_vec, g2_vec, g1_matrix, g2_matrix)
    # end # if
    # ##### END OF PLOTTING ######

    return E, ψ
end # function

function routine_multi_gaps(N_lat, N_part, boson_dim, conserveParticleNumber, μ, N_sweeps, max_dim, cutoff, noise)
    N_part_arr = [N_part-2, N_part-1, N_part, N_part+1, N_part+2]

    E_vec = Vector{Float64}([])
    expectation_values_vec = Vector([])
    for (i, N_part) in enumerate(N_part_arr)
        E, ψ, expectation_values = routine_single(N_lat, N_part, boson_dim, conserveParticleNumber, μ, N_sweeps, max_dim, cutoff, noise; showPlot=false)
        push!(E_vec, E)
        merge!(expectation_values, Dict("E" => E))
        push!(expectation_values_vec, expectation_values)
        println("********* $i/$(length(N_part_arr)) DONE *********")
    end # for

    Δ1 = E_vec[2] + E_vec[4] - 2.0*E_vec[3]
    Δ2 = E_vec[1] + E_vec[5] - 2.0*E_vec[3]
    @show Δ1
    @show Δ2

    ########## CENTRAL POINT ##########
    k_arr = LinRange(0.0, 2π, N_lat+1)[1:N_lat]
    expectation_values = expectation_values_vec[3]

    δN, n_arr, g1, g1_vec, g1_matrix, g2, g2_vec, g2_matrix, S_k_vec, O_odd, O_even = expectation_values["δN"], expectation_values["n_arr"], expectation_values["g1"], expectation_values["g1_vec"], expectation_values["g1_matrix"], expectation_values["g2"], expectation_values["g2_vec"], expectation_values["g2_matrix"], expectation_values["S_k_vec"], expectation_values["O_odd"], expectation_values["O_even"]

    ##### SAVING RESULTS ######
    save_path = "$(@__DIR__)/NPY/$(Dates.today())/Expectation_Values"
    date_string = """$(Dates.format(Dates.now(), "Y-mm-dd!HH.MM.SS"))"""
    mkpath(save_path)
    npzwrite("$save_path/Expectation_Values!$date_string.npz", expectation_values)
    ##### END OF SAVING RESULTS ######

    ##### PLOTTING ######
    # if showPlot
    plotN_expectation_values_v1(N_lat, k_arr, n_arr, S_k_vec, g1_vec, g2_vec, g1_matrix, g2_matrix)
    plotN_expectation_values_poster(N_lat, g1_vec, g2_vec, g1_matrix, g2_matrix;
            fig_title=raw"$|\delta N|=$"*"$(round(abs(δN), digits=3)), " * raw"$\Delta_1=$"*"$(round(Δ1, digits=3)), " * raw"$\Delta_2=$"*"$(round(Δ2, digits=3))")
    # end # if
    ##### END OF PLOTTING ######
    ########## END OF CENTRAL POINT ##########
end # function
##### END OF LOWEST ORDER ROUTINES #####

##### PHASE CURVE ROUTINES #####
function routine_phase_curve_PSF_v1(N_lat, N_part, boson_dim, conserveParticleNumber, μ, N_sweeps, max_dim, cutoff, noise)
    sites = siteinds("Boson", N_lat; dim=boson_dim, conserve_qns=conserveParticleNumber)
    k_arr = LinRange(0.0, 2π, N_lat+1)[1:N_lat]

    save_path = "$(@__DIR__)/NPY/$(Dates.today())/Expectation_Values_Vec"
    date_string = """$(Dates.format(Dates.now(), "Y-mm-dd!HH.MM.SS"))"""
    mkpath(save_path)

    ##### PHASE CURVE #####
    N_g0 = 2  # 21
    g0_vec = LinRange(0.0, 0.5, N_g0)
    display(Array(g0_vec))

    δN_arr = zeros(Float64, N_g0)
    g1_arr = zeros(Float64, N_g0)
    g2_arr = zeros(Float64, N_g0)

    # for n in 1:N_g0
    Threads.@threads for n in 1:N_g0
        g0_n = g0_vec[n]

        Ham = calc_brick_wall_Ham(sites, N_lat, conserveParticleNumber, μ;
              J_1=0.0, J_2=0.0, g_0=g0_n, g_x=1.0, g_z=0.0, G_000=1.0000, G_001=-0.5000, G_011=0.5000)

        ψ0 = create_ψ0(sites, N_lat, N_part, boson_dim)
        # E, ψ = dmrg(Ham, ψ0; nsweeps=N_sweeps, maxdim=max_dim, cutoff=cutoff, noise=noise)
        if n == 1
            E, ψ = dmrg(Ham, ψ0; nsweeps=N_sweeps, maxdim=max_dim, cutoff=cutoff, noise=noise,
                        outputlevel=1)
        else
            E, ψ = dmrg(Ham, ψ0; nsweeps=N_sweeps, maxdim=max_dim, cutoff=cutoff, noise=noise,
                        outputlevel=0)
        end # if

        expectation_values = calc_expectation_values(ψ, Ham, sites, k_arr, N_lat)
        δN, g1, g2 = expectation_values["δN"], expectation_values["g1"], expectation_values["g2"]
        δN_arr[n], g1_arr[n], g2_arr[n] = δN, g1, g2
        # push!(expectation_values_vec, expectation_values)

        npzwrite("$save_path/Expectation_Values!g0=$(round(g0_n; digits=4))!$date_string.npz", expectation_values)

        # println("******* $(n)/$(N_g0) DONE *******")
    end # for

    if Threads.threadid() == 1
        ##### PLOTTING #####
        plotN_PSF_phase_curves_v1(g0_vec, δN_arr, g1_arr, g2_arr)
        ##### END OF PLOTTING #####
    end # if
    ##### END OF PHASE CURVE #####
end # function

function routine_phase_curve_SS_v1(N_lat, N_part, boson_dim, conserveParticleNumber, μ, N_sweeps, max_dim, cutoff, noise)
    sites = siteinds("Boson", N_lat; dim=boson_dim, conserve_qns=conserveParticleNumber)
    k_arr = LinRange(0.0, 2π, N_lat+1)[1:N_lat]

    save_path = "$(@__DIR__)/NPY/$(Dates.today())/Expectation_Values_Vec"
    date_string = """$(Dates.format(Dates.now(), "Y-mm-dd!HH.MM.SS"))"""
    mkpath(save_path)

    ##### PHASE CURVE #####
    N_J2 = 2  # 21
    abs_J2_vec = LinRange(0.0, 0.100, N_J2)
    display(Array(abs_J2_vec))

    δN_arr = zeros(Float64, N_J2)
    g1_arr = zeros(Float64, N_J2)
    g2_arr = zeros(Float64, N_J2)

    # for n in 1:N_J2
    Threads.@threads for n in 1:N_J2
        abs_J2_n = abs_J2_vec[n]

        Ham = calc_brick_wall_Ham(sites, N_lat, conserveParticleNumber, μ;
                J_1=-abs_J2_n/(10^(0.375)), J_2=-abs_J2_n, g_0=0.5, g_x=-0.5, g_z=0.0, G_000=1.0000, G_001=-0.5000, G_011=0.5000)  # SS, p-band

        ψ0 = create_ψ0(sites, N_lat, N_part, boson_dim)
        # E, ψ = dmrg(Ham, ψ0; nsweeps=N_sweeps, maxdim=max_dim, cutoff=cutoff, noise=noise)
        if n == 1
            E, ψ = dmrg(Ham, ψ0; nsweeps=N_sweeps, maxdim=max_dim, cutoff=cutoff, noise=noise,
                        outputlevel=1)
        else
            E, ψ = dmrg(Ham, ψ0; nsweeps=N_sweeps, maxdim=max_dim, cutoff=cutoff, noise=noise,
                        outputlevel=0)
        end # if

        expectation_values = calc_expectation_values(ψ, Ham, sites, k_arr, N_lat)
        δN, g1, g2 = expectation_values["δN"], expectation_values["g1"], expectation_values["g2"]
        δN_arr[n], g1_arr[n], g2_arr[n] = δN, g1, g2
        # push!(expectation_values_vec, expectation_values)

        npzwrite("$save_path/Expectation_Values!abs_J2=$(round(abs_J2_n; digits=4))!$date_string.npz", expectation_values)

        # println("******* $(n)/$(N_J2) DONE *******")
    end # for

    if Threads.threadid() == 1
        ##### PLOTTING #####
        plotN_SS_phase_curves_v1(abs_J2_vec, δN_arr, g1_arr, g2_arr)
        ##### END OF PLOTTING #####
    end # if
    ##### END OF PHASE CURVE #####
end # function

function routine_phase_curve_SS_repeat_v1(N_lat, N_part, boson_dim, conserveParticleNumber, μ, N_sweeps, max_dim, cutoff, noise)
    sites = siteinds("Boson", N_lat; dim=boson_dim, conserve_qns=conserveParticleNumber)
    k_arr = LinRange(0.0, 2π, N_lat+1)[1:N_lat]

    save_path = "$(@__DIR__)/NPY/$(Dates.today())/Expectation_Values_Vec"
    date_string = """$(Dates.format(Dates.now(), "Y-mm-dd!HH.MM.SS"))"""
    mkpath(save_path)

    ##### PHASE CURVE #####
    N_J2 = 2  # 21
    abs_J2_vec = LinRange(0.0, 0.100, N_J2)
    display(Array(abs_J2_vec))

    δN_arr = zeros(Float64, N_J2)
    g1_arr = zeros(Float64, N_J2)
    g2_arr = zeros(Float64, N_J2)

    N_repeat = 5
    # for n in 1:N_J2
    Threads.@threads for n in 1:N_J2
        abs_J2_n = abs_J2_vec[n]

        Ham = calc_brick_wall_Ham(sites, N_lat, conserveParticleNumber, μ;
                J_1=-abs_J2_n/(10^(0.375)), J_2=-abs_J2_n, g_0=0.5, g_x=-0.5, g_z=0.0, G_000=1.0000, G_001=-0.5000, G_011=0.5000)  # SS, p-band

        E_vec = Vector{Float64}([])
        ψ_vec = Vector{MPS}([])
        for ii in 1:N_repeat
            ψ0 = create_ψ0(sites, N_lat, N_part, boson_dim)
            # E, ψ = dmrg(Ham, ψ0; nsweeps=N_sweeps, maxdim=max_dim, cutoff=cutoff, noise=noise)
            if n == 1
                E, ψ = dmrg(Ham, ψ0; nsweeps=N_sweeps, maxdim=max_dim, cutoff=cutoff, noise=noise,
                            outputlevel=1)
            else
                E, ψ = dmrg(Ham, ψ0; nsweeps=N_sweeps, maxdim=max_dim, cutoff=cutoff, noise=noise,
                            outputlevel=0)
            end # if
            push!(E_vec, E)
            push!(ψ_vec, ψ)
            if n == 1
                println("********* $ii/$(N_repeat) DONE *********")
            end # if
        end # for
        E = E_vec[argmin(E_vec)]
        ψ = ψ_vec[argmin(E_vec)]

        expectation_values = calc_expectation_values(ψ, Ham, sites, k_arr, N_lat)
        δN, g1, g2 = expectation_values["δN"], expectation_values["g1"], expectation_values["g2"]
        δN_arr[n], g1_arr[n], g2_arr[n] = δN, g1, g2
        # push!(expectation_values_vec, expectation_values)

        npzwrite("$save_path/Expectation_Values!abs_J2=$(round(abs_J2_n; digits=4))!$date_string.npz", expectation_values)

        # println("******* $(n)/$(N_J2) DONE *******")
    end # for

    if Threads.threadid() == 1
        ##### PLOTTING #####
        plotN_SS_phase_curves_v1(abs_J2_vec, δN_arr, g1_arr, g2_arr)
        ##### END OF PLOTTING #####
    end # if
    ##### END OF PHASE CURVE #####
end # function

function routine_phase_curve_gx(N_lat, N_part, boson_dim, conserveParticleNumber, μ, N_repeat, N_sweeps, max_dim, cutoff, noise)
    ##### SPECIFIC PARAMETERS #####
    N_gx = 31  # 12  # 22
    gx_vec = LinRange(-5.0, 0.0, N_gx)

    abs_J1 = 0.0

    band_options = Dict(
        1 => "s",
        2 => "p"
    )
    whichBand = band_options[1]

    N_sweeps_small = N_sweeps÷5
    ##### END OF SPECIFIC PARAMETERS #####

    N_part_arr = [N_part]

    sites = siteinds("Boson", N_lat; dim=boson_dim, conserve_qns=conserveParticleNumber)
    k_arr = LinRange(0.0, 2π, N_lat+1)[1:N_lat]

    save_path = "$(@__DIR__)/NPY/$(Dates.today())/Phase_Curve_gx"
    date_string = """$(Dates.format(Dates.now(), "Y-mm-dd!HH.MM.SS"))"""
    mkpath(save_path)

    ##### PHASE CURVE #####
    δN_vec = zeros(Float64, N_gx)
    g1_vec = zeros(Float64, N_gx)
    g2_vec = zeros(Float64, N_gx)
    Δ1_vec = zeros(Float64, N_gx)
    Δ2_vec = zeros(Float64, N_gx)
    S_k0_vec = zeros(Float64, N_gx)
    M_k0_vec = zeros(Float64, N_gx)
    S_kπ_vec = zeros(Float64, N_gx)  # Assumes N_lat is even
    M_kπ_vec = zeros(Float64, N_gx)  # Assumes N_lat is even
    integ_g1_vec = zeros(Float64, N_gx)
    integ_g2_vec = zeros(Float64, N_gx)
    integ_S_k_vec = zeros(Float64, N_gx)
    O_odd_vec = zeros(Float64, N_gx)
    O_even_vec = zeros(Float64, N_gx)
    expect_E_vec = zeros(Float64, N_gx)
    # integ_M_k_vec = zeros(Float64, N_gx)
    c_vec = zeros(Float64, N_gx)
    prob_Q_vec = zeros(Float64, N_gx)
    avg_NN_g1_vec = zeros(Float64, N_gx)
    avg_NN_g2_vec = zeros(Float64, N_gx)
    avg_NN_n_vec = zeros(Float64, N_gx)

    # Threads.@threads for n in 1:N_gx
    for n in 1:N_gx
        gx_n = gx_vec[n]

        Ham = choose_brick_wall_Ham(whichBand, sites, N_lat, conserveParticleNumber, μ;
            abs_J1=abs_J1, g_0=1.0, g_x=gx_n, g_z=0.0)

        E_vec = Vector{Float64}([])
        expectation_values_vec = Vector([])
        for (i, N_part) in enumerate(N_part_arr)
            E_repeat_vec = Vector{Float64}([])
            ψ_repeat_vec = Vector{MPS}([])
            for ii in 1:N_repeat
                ψ0 = create_ψ0(sites, N_lat, N_part, boson_dim; linkdims=100)
                E, ψ = dmrg(Ham, ψ0; nsweeps=N_sweeps_small, maxdim=max_dim, cutoff=cutoff, noise=noise)

                push!(E_repeat_vec, E)
                push!(ψ_repeat_vec, ψ)
            end # for
            E = E_repeat_vec[argmin(E_repeat_vec)]
            ψ = ψ_repeat_vec[argmin(E_repeat_vec)]

            E, ψ = dmrg(Ham, ψ; nsweeps=N_sweeps, maxdim=max_dim, cutoff=cutoff, noise=noise)
            expectation_values = calc_expectation_values(ψ, Ham, sites, k_arr, N_lat)

            push!(E_vec, E)
            push!(expectation_values_vec, expectation_values)

            println("******* $(i)/$(length(N_part_arr)) DONE *******")
        end # for

        expectation_values = expectation_values_vec[1]
        expect_E, δN, g1, g2, g1_matrix, g2_matrix, S_k_vec, M_k_vec, O_odd, O_even, c, prob_Q, prob_Q_bulk, prob_Q_mod_bulk, avg_NN_g1, avg_NN_g2, avg_NN_n = expectation_values["expect_E"], expectation_values["δN"], expectation_values["g1"], expectation_values["g2"], expectation_values["g1_matrix"], expectation_values["g2_matrix"], expectation_values["S_k_vec"], expectation_values["M_k_vec"], expectation_values["O_odd"], expectation_values["O_even"], expectation_values["c"], expectation_values["prob_Q"], expectation_values["prob_Q_bulk"], expectation_values["prob_Q_mod_bulk"], expectation_values["avg_NN_g1"], expectation_values["avg_NN_g2"], expectation_values["avg_NN_n"]
        δN_vec[n], g1_vec[n], g2_vec[n] = δN, g1, g2
        S_k0_vec[n], M_k0_vec[n] = real(S_k_vec[1]), real(M_k_vec[1])
        S_kπ_vec[n], M_kπ_vec[n] = real(S_k_vec[N_lat÷2+1]), real(M_k_vec[N_lat÷2+1])
        integ_g1_vec[n] = sum(abs.(g1_matrix)) - tr(abs.(g1_matrix))
        integ_g2_vec[n] = sum(abs.(g2_matrix)) - tr(abs.(g2_matrix))
        integ_S_k_vec[n] = sum(abs.(S_k_vec[(N_lat÷2-10):(N_lat÷2+10)]))  # sum(abs.(S_k_vec[(N_lat÷10):end]))
        O_odd_vec[n] = O_odd
        O_even_vec[n] = O_even
        expect_E_vec[n] = expect_E
        # integ_M_k_vec[n] = sum(abs.(M_k_vec[(N_lat÷10):end]))
        c_vec[n] = c
        prob_Q_vec[n] = prob_Q_mod_bulk  # prob_Q
        avg_NN_g1_vec[n] = avg_NN_g1
        avg_NN_g2_vec[n] = avg_NN_g2
        avg_NN_n_vec[n] = avg_NN_n

        npzwrite("$save_path/Expectation_Values!abs_J1=$(round(abs_J1; digits=4))!gx=$(round(gx_n; digits=4))!$date_string.npz", expectation_values)

        # if m == 1
        #     println("******* $(n)/$(N_gx) DONE *******")
        # end # if
        println("************** $(n)/$(N_gx) DONE **************")
    end # for

    if Threads.threadid() == 1
        ##### PLOTTING #####
        plotN_order_parameters_gaps_vs_gx(gx_vec, δN_vec, g1_vec, g2_vec, Δ1_vec, Δ2_vec, integ_g1_vec, integ_g2_vec, integ_S_k_vec, O_odd_vec, O_even_vec, expect_E_vec, S_k0_vec, M_k0_vec, S_kπ_vec, M_kπ_vec, c_vec, prob_Q_vec, avg_NN_g1_vec, avg_NN_g2_vec, avg_NN_n_vec)
        ##### END OF PLOTTING #####
    end # if
    ##### END OF PHASE CURVE #####
end # function

function routine_phase_curve_gaps_gx(N_lat, N_part, boson_dim, conserveParticleNumber, μ, N_repeat, N_sweeps, max_dim, cutoff, noise)
    ##### SPECIFIC PARAMETERS #####
    N_gx = 31  # 31  # 12  # 22
    # gx_vec = LinRange(-3.0, 3.0, N_gx)
    gx_vec = LinRange(-5.0, 0.0, N_gx)

    abs_J1 = 0.0

    band_options = Dict(
        1 => "s",
        2 => "p"
    )
    whichBand = band_options[1]

    N_sweeps_small = N_sweeps÷5
    ##### END OF SPECIFIC PARAMETERS #####

    N_part_arr = [N_part-2, N_part-1, N_part, N_part+1, N_part+2]

    sites = siteinds("Boson", N_lat; dim=boson_dim, conserve_qns=conserveParticleNumber)
    k_arr = LinRange(0.0, 2π, N_lat+1)[1:N_lat]

    save_path = "$(@__DIR__)/NPY/$(Dates.today())/Phase_Curve_Gaps_gx"
    date_string = """$(Dates.format(Dates.now(), "Y-mm-dd!HH.MM.SS"))"""
    mkpath(save_path)

    ##### PHASE CURVE #####
    δN_vec = zeros(Float64, N_gx)
    g1_vec = zeros(Float64, N_gx)
    g2_vec = zeros(Float64, N_gx)
    Δ1_vec = zeros(Float64, N_gx)
    Δ2_vec = zeros(Float64, N_gx)
    S_k0_vec = zeros(Float64, N_gx)
    M_k0_vec = zeros(Float64, N_gx)
    S_kπ_vec = zeros(Float64, N_gx)  # Assumes N_lat is even
    M_kπ_vec = zeros(Float64, N_gx)  # Assumes N_lat is even
    integ_g1_vec = zeros(Float64, N_gx)
    integ_g2_vec = zeros(Float64, N_gx)
    integ_S_k_vec = zeros(Float64, N_gx)
    O_odd_vec = zeros(Float64, N_gx)
    O_even_vec = zeros(Float64, N_gx)
    expect_E_vec = zeros(Float64, N_gx)
    # integ_M_k_vec = zeros(Float64, N_gx)
    c_vec = zeros(Float64, N_gx)
    prob_Q_vec = zeros(Float64, N_gx)
    avg_NN_g1_vec = zeros(Float64, N_gx)
    avg_NN_g2_vec = zeros(Float64, N_gx)
    avg_NN_n_vec = zeros(Float64, N_gx)

    # Threads.@threads for n in 1:N_gx
    for n in 1:N_gx
        gx_n = gx_vec[n]

        Ham = choose_brick_wall_Ham(whichBand, sites, N_lat, conserveParticleNumber, μ;
            abs_J1=abs_J1, g_0=1.0, g_x=gx_n, g_z=0.0)

        bulk_range = (N_lat÷10):(N_lat-N_lat÷10)
        Ham_bulk = choose_brick_wall_Ham(whichBand, sites, bulk_range, conserveParticleNumber, μ;
            abs_J1=abs_J1, g_0=1.0, g_x=gx_n, g_z=0.0)

        E_vec = Vector{Float64}([])
        expectation_values_vec = Vector([])
        for (i, N_part) in enumerate(N_part_arr)
            E_repeat_vec = Vector{Float64}([])
            ψ_repeat_vec = Vector{MPS}([])
            for ii in 1:N_repeat
                ψ0 = create_ψ0(sites, N_lat, N_part, boson_dim; linkdims=100)
                E, ψ = dmrg(Ham, ψ0; nsweeps=N_sweeps_small, maxdim=max_dim, cutoff=cutoff, noise=noise)

                push!(E_repeat_vec, E)
                push!(ψ_repeat_vec, ψ)
            end # for
            E = E_repeat_vec[argmin(E_repeat_vec)]
            ψ = ψ_repeat_vec[argmin(E_repeat_vec)]

            E, ψ = dmrg(Ham, ψ; nsweeps=N_sweeps, maxdim=max_dim, cutoff=cutoff, noise=noise)
            expectation_values = calc_expectation_values(ψ, Ham_bulk, sites, k_arr, N_lat)

            push!(E_vec, E)
            push!(expectation_values_vec, expectation_values)

            println("******* $(i)/$(length(N_part_arr)) DONE *******")
        end # for

        expectation_values = expectation_values_vec[3]
        expect_E, δN, g1, g2, g1_matrix, g2_matrix, S_k_vec, M_k_vec, O_odd, O_even, c, prob_Q_mod_bulk, avg_NN_g1, avg_NN_g2, avg_NN_n = expectation_values["expect_E"], expectation_values["δN"], expectation_values["g1"], expectation_values["g2"], expectation_values["g1_matrix"], expectation_values["g2_matrix"], expectation_values["S_k_vec"], expectation_values["M_k_vec"], expectation_values["O_odd"], expectation_values["O_even"], expectation_values["c"], expectation_values["prob_Q_mod_bulk"], expectation_values["avg_NN_g1"], expectation_values["avg_NN_g2"], expectation_values["avg_NN_n"]
        δN_vec[n], g1_vec[n], g2_vec[n] = δN, g1, g2
        S_k0_vec[n], M_k0_vec[n] = real(S_k_vec[1]), real(M_k_vec[1])
        S_kπ_vec[n], M_kπ_vec[n] = real(S_k_vec[N_lat÷2+1]), real(M_k_vec[N_lat÷2+1])
        integ_g1_vec[n] = sum(abs.(g1_matrix)) - tr(abs.(g1_matrix))
        integ_g2_vec[n] = sum(abs.(g2_matrix)) - tr(abs.(g2_matrix))
        integ_S_k_vec[n] = sum(abs.(S_k_vec[(N_lat÷2-10):(N_lat÷2+10)]))  # sum(abs.(S_k_vec[(N_lat÷10):end]))
        O_odd_vec[n] = O_odd
        O_even_vec[n] = O_even
        expect_E_vec[n] = expect_E
        # integ_M_k_vec[n] = sum(abs.(M_k_vec[(N_lat÷10):end]))
        c_vec[n] = c
        prob_Q_vec[n] = prob_Q_mod_bulk
        avg_NN_g1_vec[n] = avg_NN_g1
        avg_NN_g2_vec[n] = avg_NN_g2
        avg_NN_n_vec[n] = avg_NN_n

        Δ1 = E_vec[2] + E_vec[4] - 2.0*E_vec[3]
        Δ2 = E_vec[1] + E_vec[5] - 2.0*E_vec[3]
        Δ1_vec[n] = Δ1
        Δ2_vec[n] = Δ2
        merge!(expectation_values, Dict("Δ1" => Δ1))
        merge!(expectation_values, Dict("Δ2" => Δ2))
        @show Δ1
        @show Δ2

        npzwrite("$save_path/Expectation_Values!abs_J1=$(round(abs_J1; digits=4))!gx=$(round(gx_n; digits=4))!$date_string.npz", expectation_values)

        # if m == 1
        #     println("******* $(n)/$(N_gx) DONE *******")
        # end # if
        println("************** $(n)/$(N_gx) DONE **************")
    end # for

    if Threads.threadid() == 1
        ##### PLOTTING #####
        plotN_order_parameters_gaps_vs_gx(gx_vec, δN_vec, g1_vec, g2_vec, Δ1_vec, Δ2_vec, integ_g1_vec, integ_g2_vec, integ_S_k_vec, O_odd_vec, O_even_vec, expect_E_vec, S_k0_vec, M_k0_vec, S_kπ_vec, M_kπ_vec, c_vec, prob_Q_vec, avg_NN_g1_vec, avg_NN_g2_vec, avg_NN_n_vec)
        ##### END OF PLOTTING #####
    end # if
    ##### END OF PHASE CURVE #####
end # function

function routine_brick_wall_compare_to_XXZ_gx(N_lat, N_part, boson_dim, conserveParticleNumber, μ, N_repeat, N_sweeps, max_dim, cutoff, noise)
    # ##### SPECIFIC PARAMETERS #####
    # N_gx = 31  # 31  # 12  # 22
    # # gx_vec = LinRange(-3.0, 3.0, N_gx)
    # gx_vec = LinRange(-5.0, 0.0, N_gx)

    # band_options = Dict(
    #     1 => "s",
    #     2 => "p"
    # )
    # whichBand = band_options[1]

    # N_sweeps_spin = 20
    # ##### END OF SPECIFIC PARAMETERS #####

    # gx_n = gx_vec[n]

    ########## SINGLE ##########
    ##### SPECIFIC PARAMETERS #####
    g_x = -5.0

    N_sweeps_spin = N_sweeps
    ##### END OF SPECIFIC PARAMETERS #####

    # ### SPIN ###
    # @assert conserveParticleNumber

    # g_0 = 1.0
    # G_000 = 1.0
    # G_001 = -0.2122
    # G_011 = 0.1666

    # U = (2.0*g_0 + g_x) * G_000
    # V = 2.0*g_0 * G_011
    # @show U
    # @show V

    # J_XXZ = 2.0*g_x*G_011
    # Δ_XXZ = -4.0*g_0/g_x
    # @show J_XXZ
    # @show Δ_XXZ

    # # NOTE: Not sure if h=μ
    # par_XXZ = XXZHamiltonianParameters(N_lat, conserveParticleNumber, μ, J_XXZ, Δ_XXZ, N_sweeps_spin, max_dim, cutoff, noise)

    # sites = siteinds("S=1/2", N_lat; conserve_qns=conserveParticleNumber)

    # Ham = calc_XXZ_Hamiltonian(par_XXZ, sites)

    # ψ0 = create_ψ0_spin(par_XXZ, sites)
    # E, ψ = dmrg(Ham, ψ0; nsweeps=N_sweeps_spin, maxdim=max_dim, cutoff=cutoff, noise=noise)
    # expectation_values = calc_spin_expectation_values(ψ, sites, N_lat)

    # mz, SpSm, mz_stag, SzSz = expectation_values["mz"], expectation_values["SpSm"], expectation_values["mz_stag"], expectation_values["SzSz"]
    # # NOTE: Parity of lattice sites matters.
    # E_shifted = E + (U + V) * N_lat

    # @show E
    # @show E_shifted
    # @show mz
    # @show SpSm
    # @show mz_stag
    # @show SzSz

    # println("********** SPIN DONE **********")
    # ### END OF SPIN ###

    ### BOSON ###
    k_arr = LinRange(0.0, 2π, N_lat+1)[1:N_lat]

    sites = siteinds("Boson", N_lat; dim=boson_dim, conserve_qns=conserveParticleNumber)

    Ham = calc_brick_wall_Ham(sites, N_lat, conserveParticleNumber, μ;
          J_1=0.0, J_2=0.0, g_0=1.0, g_x=g_x, g_z=0.0, G_000=1.0000, G_001=0.2122, G_011=0.1666)  # s-band

    ψ0 = create_ψ0(sites, N_lat, N_part, boson_dim; linkdims=200)

    E, ψ = dmrg(Ham, ψ0; nsweeps=N_sweeps, maxdim=max_dim, cutoff=cutoff, noise=noise)
    expectation_values = calc_expectation_values(ψ, Ham, sites, k_arr, N_lat)

    expect_E, δN, n_arr, g1, g1_vec, g1_matrix, g2, g2_vec, g2_matrix, κ2, M_k_vec, S_k_vec, M_k0, M_kπ, M_k2π3, S_kπ, O_odd, O_even, c, prob_Q, prob_Q_bulk, prob_Q_mod, prob_Q_mod_bulk, avg_NN_g1, avg_NN_g2, avg_NN_n = expectation_values["expect_E"], expectation_values["δN"], expectation_values["n_arr"], expectation_values["g1"], expectation_values["g1_vec"], expectation_values["g1_matrix"], expectation_values["g2"], expectation_values["g2_vec"], expectation_values["g2_matrix"], expectation_values["κ2"], expectation_values["M_k_vec"], expectation_values["S_k_vec"], expectation_values["M_k0"], expectation_values["M_kπ"], expectation_values["M_k2π3"], expectation_values["S_kπ"], expectation_values["O_odd"], expectation_values["O_even"], expectation_values["c"], expectation_values["prob_Q"], expectation_values["prob_Q_bulk"], expectation_values["prob_Q_mod"], expectation_values["prob_Q_mod_bulk"], expectation_values["avg_NN_g1"], expectation_values["avg_NN_g2"], expectation_values["avg_NN_n"]
    # S_kπ = real(S_k_vec[N_lat÷2+1])
    integ_S_kπ = sum(abs.(S_k_vec[(N_lat÷2-10):(N_lat÷2+10)]))
    @show expect_E
    @show δN
    @show g1
    @show g2
    @show κ2
    @show S_kπ
    @show M_k0
    @show M_kπ
    @show M_k2π3
    @show integ_S_kπ
    @show c
    @show prob_Q
    @show prob_Q_bulk
    @show prob_Q_mod
    @show prob_Q_mod_bulk
    @show O_odd
    @show O_even
    @show avg_NN_g1
    @show avg_NN_g2
    @show avg_NN_n

    println("********** BOSON DONE **********")
    ### END OF BOSON ###
    ########## END OF SINGLE ##########
end # function
##### END OF PHASE CURVE ROUTINES #####

##### PHASE DIAGRAM ROUTINES #####
function routine_phase_diagram_J1_gx(N_lat, N_part, boson_dim, conserveParticleNumber, μ, N_sweeps, max_dim, cutoff, noise)
    sites = siteinds("Boson", N_lat; dim=boson_dim, conserve_qns=conserveParticleNumber)
    k_arr = LinRange(0.0, 2π, N_lat+1)[1:N_lat]

    save_path = "$(@__DIR__)/NPY/$(Dates.today())/Phase_Diagram_J1_gx"
    date_string = """$(Dates.format(Dates.now(), "Y-mm-dd!HH.MM.SS"))"""
    mkpath(save_path)

    ##### PHASE DIAGRAM #####
    N_J1 = 11  # 4  # 21
    N_gx = 12  # 12  # 22

    abs_J1_vec = LinRange(0.0, 1.0/sqrt(10.0), N_J1)
    gx_vec = LinRange(-10.0, 10.0, N_gx)

    δN_arr = zeros(Float64, N_J1, N_gx)
    g1_arr = zeros(Float64, N_J1, N_gx)
    g2_arr = zeros(Float64, N_J1, N_gx)
    integ_g1_arr = zeros(Float64, N_J1, N_gx)
    integ_g2_arr = zeros(Float64, N_J1, N_gx)
    integ_S_k_arr = zeros(Float64, N_J1, N_gx)
    integ_M_k_arr = zeros(Float64, N_J1, N_gx)

    for m in 1:N_J1
    # Threads.@threads for m in 1:N_J1
        for n in 1:N_gx
            abs_J1_m = abs_J1_vec[m]
            gx_n = gx_vec[n]

            Ham = calc_brick_wall_Ham(sites, N_lat, conserveParticleNumber, μ;
                                      J_1=-abs_J1_m, J_2=abs_J1_m*(10^(0.5521)), g_0=1.0, g_x=gx_n, g_z=0.0, G_000=1.0000, G_001=-0.2122, G_011=0.1666)  # s-band
                                      # J_1=-abs_J1_m, J_2=-abs_J1_m*(10^(0.375)), g_0=1.0, g_x=gx_n, g_z=0.0, G_000=1.0000, G_001=-0.5000, G_011=0.5000)  # p-band

            ψ0 = create_ψ0(sites, N_lat, N_part, boson_dim; linkdims=100)
            E, ψ = dmrg(Ham, ψ0; nsweeps=N_sweeps, maxdim=max_dim, cutoff=cutoff, noise=noise)
            # if m == 1
            #     E, ψ = dmrg(Ham, ψ0; nsweeps=N_sweeps, maxdim=max_dim, cutoff=cutoff, noise=noise,
            #                 outputlevel=1)
            # else
            #     E, ψ = dmrg(Ham, ψ0; nsweeps=N_sweeps, maxdim=max_dim, cutoff=cutoff, noise=noise,
            #                 outputlevel=0)
            # end # if

            expectation_values = calc_expectation_values(ψ, Ham, sites, k_arr, N_lat)
            δN, g1, g2, g1_matrix, g2_matrix, S_k_vec, M_k_vec = expectation_values["δN"], expectation_values["g1"], expectation_values["g2"], expectation_values["g1_matrix"], expectation_values["g2_matrix"], expectation_values["S_k_vec"], expectation_values["M_k_vec"]
            integ_g1_arr[m, n] = sum(abs.(g1_matrix)) - tr(abs.(g1_matrix))
            integ_g2_arr[m, n] = sum(abs.(g2_matrix)) - tr(abs.(g2_matrix))
            δN_arr[m, n], g1_arr[m, n], g2_arr[m, n] = δN, g1, g2
            integ_S_k_arr[m, n] = sum(abs.(S_k_vec[(N_lat÷10):end]))
            integ_M_k_arr[m, n] = sum(abs.(M_k_vec[(N_lat÷10):end]))
            # push!(expectation_values_vec, expectation_values)

            merge!(expectation_values, Dict("abs_J1_vec" => abs_J1_vec))
            merge!(expectation_values, Dict("gx_vec" => gx_vec))

            npzwrite("$save_path/Expectation_Values!abs_J1=$(round(abs_J1_m; digits=4))!gx=$(round(gx_n; digits=4))!$date_string.npz", expectation_values)

            # if m == 1
            #     println("******* $(n)/$(N_gx) DONE *******")
            # end # if
            println("******* $(m)/$(N_J1) && $(n)/$(N_gx) DONE *******")
        end # for
    end # for

    if Threads.threadid() == 1
        ##### PLOTTING #####
        plotN_order_parameters_vs_J1_gx(abs_J1_vec, gx_vec, δN_arr, g1_arr, g2_arr, integ_g1_arr, integ_g2_arr, integ_S_k_arr, integ_M_k_arr)
        ##### END OF PLOTTING #####
    end # if
    ##### END OF PHASE DIAGRAM #####
end # function

function routine_phase_diagram_gaps_J1_gx(N_lat, N_part, boson_dim, conserveParticleNumber, μ, N_sweeps, max_dim, cutoff, noise)
    N_part_arr = [N_part-2, N_part-1, N_part, N_part+1, N_part+2]

    sites = siteinds("Boson", N_lat; dim=boson_dim, conserve_qns=conserveParticleNumber)
    k_arr = LinRange(0.0, 2π, N_lat+1)[1:N_lat]

    save_path = "$(@__DIR__)/NPY/$(Dates.today())/Phase_Diagram_Gaps_J1_gx"
    date_string = """$(Dates.format(Dates.now(), "Y-mm-dd!HH.MM.SS"))"""
    mkpath(save_path)

    ##### PHASE DIAGRAM #####
    N_J1 = 7  # 4  # 21
    N_gx = 7  # 12  # 22

    abs_J1_vec = LinRange(0.0, 1.0/sqrt(10.0), N_J1)
    gx_vec = LinRange(-5.0, 0.0, N_gx)

    δN_arr = zeros(Float64, N_J1, N_gx)
    g1_arr = zeros(Float64, N_J1, N_gx)
    g2_arr = zeros(Float64, N_J1, N_gx)
    Δ1_arr = zeros(Float64, N_J1, N_gx)
    Δ2_arr = zeros(Float64, N_J1, N_gx)
    S_k0_arr = zeros(Float64, N_J1, N_gx)
    M_k0_arr = zeros(Float64, N_J1, N_gx)
    S_kπ_arr = zeros(Float64, N_J1, N_gx)  # Assumes N_lat is even
    M_kπ_arr = zeros(Float64, N_J1, N_gx)  # Assumes N_lat is even
    integ_g1_arr = zeros(Float64, N_J1, N_gx)
    integ_g2_arr = zeros(Float64, N_J1, N_gx)
    integ_S_k_arr = zeros(Float64, N_J1, N_gx)
    integ_M_k_arr = zeros(Float64, N_J1, N_gx)
    c_arr = zeros(Float64, N_J1, N_gx)
    prob_Q_arr = zeros(Float64, N_J1, N_gx)
    avg_NN_g1_arr = zeros(Float64, N_J1, N_gx)
    avg_NN_g2_arr = zeros(Float64, N_J1, N_gx)
    avg_NN_n_arr = zeros(Float64, N_J1, N_gx)

    # TODO Different components should be abstracted away.
    for m in 1:N_J1
    # Threads.@threads for m in 1:N_J1
        for n in 1:N_gx
            abs_J1_m = abs_J1_vec[m]
            gx_n = gx_vec[n]

            E_vec = Vector{Float64}([])
            expectation_values_vec = Vector([])
            for (i, N_part) in enumerate(N_part_arr)
                Ham = calc_brick_wall_Ham(sites, N_lat, conserveParticleNumber, μ;
                                          J_1=-abs_J1_m, J_2=abs_J1_m*(10^(0.5521)), g_0=1.0, g_x=gx_n, g_z=0.0, G_000=1.0000, G_001=-0.2122, G_011=0.1666)  # s-band
                                          # J_1=-abs_J1_m, J_2=-abs_J1_m*(10^(0.375)), g_0=1.0, g_x=gx_n, g_z=0.0, G_000=1.0000, G_001=-0.5000, G_011=0.5000)  # p-band

                ψ0 = create_ψ0(sites, N_lat, N_part, boson_dim; linkdims=100)
                E, ψ = dmrg(Ham, ψ0; nsweeps=N_sweeps, maxdim=max_dim, cutoff=cutoff, noise=noise)
                # if m == 1
                #     E, ψ = dmrg(Ham, ψ0; nsweeps=N_sweeps, maxdim=max_dim, cutoff=cutoff, noise=noise,
                #                 outputlevel=1)
                # else
                #     E, ψ = dmrg(Ham, ψ0; nsweeps=N_sweeps, maxdim=max_dim, cutoff=cutoff, noise=noise,
                #                 outputlevel=0)
                # end # if

                expectation_values = calc_expectation_values(ψ, Ham, sites, k_arr, N_lat)

                push!(E_vec, E)
                push!(expectation_values_vec, expectation_values)
            end # for

            expectation_values = expectation_values_vec[3]
            δN, g1, g2, g1_matrix, g2_matrix, S_k_vec, M_k_vec, c, prob_Q, avg_NN_g1, avg_NN_g2, avg_NN_n = expectation_values["δN"], expectation_values["g1"], expectation_values["g2"], expectation_values["g1_matrix"], expectation_values["g2_matrix"], expectation_values["S_k_vec"], expectation_values["M_k_vec"], expectation_values["c"], expectation_values["prob_Q"], expectation_values["avg_NN_g1"], expectation_values["avg_NN_g2"], expectation_values["avg_NN_n"]
            δN_arr[m, n], g1_arr[m, n], g2_arr[m, n] = δN, g1, g2
            S_k0_arr[m, n], M_k0_arr[m, n] = real(S_k_vec[1]), real(M_k_vec[1])
            S_kπ_arr[m, n], M_kπ_arr[m, n] = real(S_k_vec[N_lat÷2+1]), real(M_k_vec[N_lat÷2+1])
            integ_g1_arr[m, n] = sum(abs.(g1_matrix)) - tr(abs.(g1_matrix))
            integ_g2_arr[m, n] = sum(abs.(g2_matrix)) - tr(abs.(g2_matrix))
            integ_S_k_arr[m, n] = sum(abs.(S_k_vec[(N_lat÷10):end]))
            integ_M_k_arr[m, n] = sum(abs.(M_k_vec[(N_lat÷10):end]))
            c_arr[m, n] = c
            prob_Q_arr[m, n] = prob_Q
            avg_NN_g1_arr[m, n] = avg_NN_g1
            avg_NN_g2_arr[m, n] = avg_NN_g2
            avg_NN_n_arr[m, n] = avg_NN_n

            Δ1 = E_vec[2] + E_vec[4] - 2.0*E_vec[3]
            Δ2 = E_vec[1] + E_vec[5] - 2.0*E_vec[3]
            Δ1_arr[m, n] = Δ1
            Δ2_arr[m, n] = Δ2
            merge!(expectation_values, Dict("Δ1" => Δ1))
            merge!(expectation_values, Dict("Δ2" => Δ2))
            @show Δ1
            @show Δ2

            npzwrite("$save_path/Expectation_Values!abs_J1=$(round(abs_J1_m; digits=4))!gx=$(round(gx_n; digits=4))!$date_string.npz", expectation_values)

            # if m == 1
            #     println("******* $(n)/$(N_gx) DONE *******")
            # end # if
            println("******* $(m)/$(N_J1) && $(n)/$(N_gx) DONE *******")
        end # for
    end # for

    if Threads.threadid() == 1
        ##### PLOTTING #####
        plotN_order_parameters_gaps_vs_J1_gx(abs_J1_vec, gx_vec, δN_arr, g1_arr, g2_arr, Δ1_arr, Δ2_arr, integ_g1_arr, integ_g2_arr, integ_S_k_arr, integ_M_k_arr, S_k0_arr, M_k0_arr, S_kπ_arr, M_kπ_arr, c_arr, prob_Q_arr, avg_NN_g1_arr, avg_NN_g2_arr, avg_NN_n_arr)
        ##### END OF PLOTTING #####
    end # if
    ##### END OF PHASE DIAGRAM #####
end # function

function routine_phase_diagram_gx_gz(N_lat, N_part, boson_dim, conserveParticleNumber, μ, N_sweeps, max_dim, cutoff, noise)
    sites = siteinds("Boson", N_lat; dim=boson_dim, conserve_qns=conserveParticleNumber)
    k_arr = LinRange(0.0, 2π, N_lat+1)[1:N_lat]

    save_path = "$(@__DIR__)/NPY/$(Dates.today())/Expectation_Values_Vec"
    date_string = """$(Dates.format(Dates.now(), "Y-mm-dd!HH.MM.SS"))"""
    mkpath(save_path)

    ##### PHASE CURVE #####
    N_gz = 11  # 21
    N_gx = 12  # 22
    abs_J1 = 0.1 * sqrt(10.0)

    gz_vec = LinRange(0.0, 1.0, N_gz)
    gx_vec = LinRange(-10.0, 10.0, N_gx)

    δN_arr = zeros(Float64, N_gz, N_gx)
    g1_arr = zeros(Float64, N_gz, N_gx)
    g2_arr = zeros(Float64, N_gz, N_gx)
    integ_S_k_arr = zeros(Float64, N_gz, N_gx)
    integ_M_k_arr = zeros(Float64, N_gz, N_gx)

    # for n in 1:N_gz
    Threads.@threads for m in 1:N_gz
        for n in 1:N_gx
            gz_m = gz_vec[m]
            gx_n = gx_vec[n]

            Ham = calc_brick_wall_Ham(sites, N_lat, conserveParticleNumber, μ;
                                      J_1=-abs_J1_m, J_2=abs_J1_m*(10^(0.5521)), g_0=1.0, g_x=gx_n, g_z=0.0, G_000=1.0000, G_001=-0.2122, G_011=0.1666)  # s-band
                                      # J_1=-abs_J1_m, J_2=-abs_J1_m*(10^(0.375)), g_0=1.0, g_x=gx_n, g_z=0.0, G_000=1.0000, G_001=-0.5000, G_011=0.5000)  # p-band

            ψ0 = create_ψ0(sites, N_lat, N_part, boson_dim; linkdims=100)
            # E, ψ = dmrg(Ham, ψ0; nsweeps=N_sweeps, maxdim=max_dim, cutoff=cutoff, noise=noise)
            if m == 1
                E, ψ = dmrg(Ham, ψ0; nsweeps=N_sweeps, maxdim=max_dim, cutoff=cutoff, noise=noise,
                            outputlevel=1)
            else
                E, ψ = dmrg(Ham, ψ0; nsweeps=N_sweeps, maxdim=max_dim, cutoff=cutoff, noise=noise,
                            outputlevel=0)
            end # if

            expectation_values = calc_expectation_values(ψ, Ham, sites, k_arr, N_lat)
            δN, g1, g2, S_k_vec, M_k_vec = expectation_values["δN"], expectation_values["g1"], expectation_values["g2"], expectation_values["S_k_vec"], expectation_values["M_k_vec"]
            δN_arr[m, n], g1_arr[m, n], g2_arr[m, n] = δN, g1, g2
            integ_S_k_arr[m, n] = sum(abs.(S_k_vec[(N_lat÷10):end]))
            integ_M_k_arr[m, n] = sum(abs.(M_k_vec[(N_lat÷10):end]))
            # push!(expectation_values_vec, expectation_values)

            npzwrite("$save_path/Expectation_Values!gz=$(round(gz_m; digits=4))!gx=$(round(gx_n; digits=4))!$date_string.npz", expectation_values)

            if m == 1
                println("******* $(n)/$(N_gx) DONE *******")
            end # if
        end # for
    end # for

    if Threads.threadid() == 1
        ##### PLOTTING #####
        plotN_order_parameters_vs_gz_gx(gz_vec, gx_vec, δN_arr, g1_arr, g2_arr, integ_S_k_arr, integ_M_k_arr)
        ##### END OF PLOTTING #####
    end # if
    ##### END OF PHASE CURVE #####
end # function

function routine_phase_diagram_J1_gx_HPC(N_lat, N_part, boson_dim, conserveParticleNumber, μ, N_repeat, N_sweeps, max_dim, cutoff, noise)
    N_sweeps_small = 1*N_sweeps÷5

    sites = siteinds("Boson", N_lat; dim=boson_dim, conserve_qns=conserveParticleNumber)
    k_arr = LinRange(0.0, 2π, N_lat+1)[1:N_lat]

    slurm_array_job_id = parse(Int, ARGS[3])
    save_path = "$(@__DIR__)/NPY/$(Dates.today())!$(slurm_array_job_id)/Expectation_Values_Vec"
    date_string = """$(Dates.format(Dates.now(), "Y-mm-dd!HH.MM.SS"))"""
    mkpath(save_path)

    ##### PHASE CURVE #####
    N_J1 = 21  # 4  # 21
    N_gx = 22  # 12  # 22

    abs_J1_vec = LinRange(0.0, 0.3, N_J1)
    gx_vec = LinRange(-5.0, 0.0, N_gx)

    δN_arr = zeros(Float64, N_J1, N_gx)
    g1_arr = zeros(Float64, N_J1, N_gx)
    g2_arr = zeros(Float64, N_J1, N_gx)
    integ_g1_arr = zeros(Float64, N_J1, N_gx)
    integ_g2_arr = zeros(Float64, N_J1, N_gx)
    integ_S_k_arr = zeros(Float64, N_J1, N_gx)
    integ_M_k_arr = zeros(Float64, N_J1, N_gx)

    ##### SLURM #####
    N_total = N_J1 * N_gx

    slurm_task_id = parse(Int, ARGS[1])
    slurm_task_count = parse(Int, ARGS[2])

    calc_count = (slurm_task_id <= (N_total % slurm_task_count)) ? (N_total ÷ slurm_task_count + 1) : (N_total ÷ slurm_task_count)
    idx_vec = [slurm_task_id + i * slurm_task_count for i in 0:(calc_count - 1)]
    m_vec = [((idx - 1) ÷ N_gx) + 1 for idx in idx_vec]
    n_vec = [mod1(idx, N_gx) for idx in idx_vec]
    ##### END OF SLURM #####

    # # for n in 1:N_J1
    # Threads.@threads for m in 1:N_J1
    #     for n in 1:N_gx

    for i in 1:calc_count
        m = m_vec[i]
        n = n_vec[i]

        abs_J1_m = abs_J1_vec[m]
        gx_n = gx_vec[n]
        @show abs_J1_m
        @show gx_n

        Ham = calc_brick_wall_Ham(sites, N_lat, conserveParticleNumber, μ;
                                  J_1=abs_J1_m, J_2=-abs_J1_m*(10^(0.5521)), g_0=1.0, g_x=gx_n, g_z=0.0, G_000=1.0000, G_001=-0.2122, G_011=0.1666)  # s-band
                                  # J_1=-abs_J1_m, J_2=abs_J1_m*(10^(0.5521)), g_0=1.0, g_x=gx_n, g_z=0.0, G_000=1.0000, G_001=-0.2122, G_011=0.1666)  # s-band
                                  # J_1=-abs_J1_m, J_2=-abs_J1_m*(10^(0.375)), g_0=1.0, g_x=gx_n, g_z=0.0, G_000=1.0000, G_001=-0.5000, G_011=0.5000)  # p-band

        E_repeat_vec = Vector{Float64}([])
        ψ_repeat_vec = Vector{MPS}([])
        for ii in 1:N_repeat
            ψ0 = create_ψ0(sites, N_lat, N_part, boson_dim; linkdims=100)

            E, ψ = dmrg(Ham, ψ0; nsweeps=N_sweeps_small, maxdim=max_dim, cutoff=cutoff, noise=noise)

            expectation_values = calc_expectation_values(ψ, Ham, sites, k_arr, N_lat)
            δN, S_kπ, g2, var_E = expectation_values["δN"], expectation_values["S_kπ"], expectation_values["g2"], expectation_values["var_E"]
            @show δN
            @show S_kπ
            @show g2
            @show var_E

            push!(E_repeat_vec, E)
            push!(ψ_repeat_vec, ψ)
        end # for
        E = E_repeat_vec[argmin(E_repeat_vec)]
        ψ = ψ_repeat_vec[argmin(E_repeat_vec)]

        E, ψ = dmrg(Ham, ψ; nsweeps=N_sweeps, maxdim=max_dim, cutoff=cutoff, noise=noise)
        # ψ0 = create_ψ0(sites, N_lat, N_part, boson_dim; linkdims=100)
        # E, ψ = dmrg(Ham, ψ0; nsweeps=N_sweeps, maxdim=max_dim, cutoff=cutoff, noise=noise)
        # # # if m == 1
        # # if m == 1 && n == 1
        # #     E, ψ = dmrg(Ham, ψ0; nsweeps=N_sweeps, maxdim=max_dim, cutoff=cutoff, noise=noise,
        # #                 outputlevel=1)
        # # else
        # #     E, ψ = dmrg(Ham, ψ0; nsweeps=N_sweeps, maxdim=max_dim, cutoff=cutoff, noise=noise,
        # #                 outputlevel=0)
        # # end # if

        expectation_values = calc_expectation_values(ψ, Ham, sites, k_arr, N_lat)
        δN, g1, g2, g1_matrix, g2_matrix, S_k_vec, M_k_vec = expectation_values["δN"], expectation_values["g1"], expectation_values["g2"], expectation_values["g1_matrix"], expectation_values["g2_matrix"], expectation_values["S_k_vec"], expectation_values["M_k_vec"]
        integ_g1_arr[m, n] = sum(abs.(g1_matrix)) - tr(abs.(g1_matrix))
        integ_g2_arr[m, n] = sum(abs.(g2_matrix)) - tr(abs.(g2_matrix))
        δN_arr[m, n], g1_arr[m, n], g2_arr[m, n] = δN, g1, g2
        integ_S_k_arr[m, n] = sum(abs.(S_k_vec[(N_lat÷10):end]))
        integ_M_k_arr[m, n] = sum(abs.(M_k_vec[(N_lat÷10):end]))
        # push!(expectation_values_vec, expectation_values)

        merge!(expectation_values, Dict("N_lat" => N_lat))
        merge!(expectation_values, Dict("abs_J1_vec" => abs_J1_vec))
        merge!(expectation_values, Dict("gx_vec" => gx_vec))

        npzwrite("$(save_path)/Expectation_Values!m=$(m)!n=$(n)!abs_J1=$(round(abs_J1_m; digits=4))!gx=$(round(gx_n; digits=4))!$date_string.npz", expectation_values)

        # if m == 1
        #      println("******* $(n)/$(N_gx) DONE *******")
        # end # if

        println("******* $(m)/$(N_J1) && $(n)/$(N_gx) DONE *******")
    end # for

    #     end # for
    # end # for

    # if Threads.threadid() == 1
    #     ##### PLOTTING #####
    #     plotN_order_parameters_vs_J1_gx(abs_J1_vec, gx_vec, δN_arr, g1_arr, g2_arr, integ_g1_arr, integ_g2_arr, integ_S_k_arr, integ_M_k_arr)
    #     ##### END OF PLOTTING #####
    # end # if
    ##### END OF PHASE CURVE #####
end # function

function routine_phase_diagram_J1_gx_gaps_HPC(N_lat, N_part, N_repeat, boson_dim, conserveParticleNumber, μ, N_sweeps, max_dim, cutoff, noise)
    N_sweeps_small = N_sweeps÷5

    N_part_arr = [N_part-2, N_part-1, N_part, N_part+1, N_part+2]

    sites = siteinds("Boson", N_lat; dim=boson_dim, conserve_qns=conserveParticleNumber)
    k_arr = LinRange(0.0, 2π, N_lat+1)[1:N_lat]

    slurm_array_job_id = parse(Int, ARGS[3])
    save_path = "$(@__DIR__)/NPY/$(Dates.today())!$(slurm_array_job_id)/Expectation_Values_Vec"
    date_string = """$(Dates.format(Dates.now(), "Y-mm-dd!HH.MM.SS"))"""
    mkpath(save_path)

    ##### PHASE CURVE #####
    N_J1 = 21  # 4  # 21
    N_gx = 22  # 12  # 22

    abs_J1_vec = LinRange(0.0, 0.3, N_J1)
    gx_vec = LinRange(-5.0, 0.0, N_gx)

    δN_arr = zeros(Float64, N_J1, N_gx)
    g1_arr = zeros(Float64, N_J1, N_gx)
    g2_arr = zeros(Float64, N_J1, N_gx)
    Δ1_arr = zeros(Float64, N_J1, N_gx)
    Δ2_arr = zeros(Float64, N_J1, N_gx)
    Δ1_bulk_arr = zeros(Float64, N_J1, N_gx)
    Δ2_bulk_arr = zeros(Float64, N_J1, N_gx)
    integ_g1_arr = zeros(Float64, N_J1, N_gx)
    integ_g2_arr = zeros(Float64, N_J1, N_gx)
    integ_S_k_arr = zeros(Float64, N_J1, N_gx)
    integ_M_k_arr = zeros(Float64, N_J1, N_gx)

    ##### SLURM #####
    N_total = N_J1 * N_gx

    slurm_task_id = parse(Int, ARGS[1])
    slurm_task_count = parse(Int, ARGS[2])

    calc_count = (slurm_task_id <= (N_total % slurm_task_count)) ? (N_total ÷ slurm_task_count + 1) : (N_total ÷ slurm_task_count)
    idx_vec = [slurm_task_id + i * slurm_task_count for i in 0:(calc_count - 1)]
    m_vec = [((idx - 1) ÷ N_gx) + 1 for idx in idx_vec]
    n_vec = [mod1(idx, N_gx) for idx in idx_vec]
    ##### END OF SLURM #####

    # # for n in 1:N_J1
    # Threads.@threads for m in 1:N_J1
    #     for n in 1:N_gx

    for i in 1:calc_count
        m = m_vec[i]
        n = n_vec[i]

        abs_J1_m = abs_J1_vec[m]
        gx_n = gx_vec[n]
        @show abs_J1_m
        @show gx_n

        Ham = calc_brick_wall_Ham(sites, N_lat, conserveParticleNumber, μ;
                                  J_1=abs_J1_m, J_2=-abs_J1_m*(10^(0.5521)), g_0=1.0, g_x=gx_n, g_z=0.0, G_000=1.0000, G_001=-0.2122, G_011=0.1666)  # s-band
                                  # J_1=-abs_J1_m, J_2=abs_J1_m*(10^(0.5521)), g_0=1.0, g_x=gx_n, g_z=0.0, G_000=1.0000, G_001=-0.2122, G_011=0.1666)  # s-band
                                  # J_1=-abs_J1_m, J_2=-abs_J1_m*(10^(0.375)), g_0=1.0, g_x=gx_n, g_z=0.0, G_000=1.0000, G_001=-0.5000, G_011=0.5000)  # p-band

        # NOTE: Check that Ham and Ham_bulk are consistent.
        bulk_range = (N_lat÷10):(N_lat-N_lat÷10)
        Ham_bulk = calc_brick_wall_Ham(sites, bulk_range, conserveParticleNumber, μ;
                                  J_1=abs_J1_m, J_2=-abs_J1_m*(10^(0.5521)), g_0=1.0, g_x=gx_n, g_z=0.0, G_000=1.0000, G_001=-0.2122, G_011=0.1666)  # s-band

        # ψ0 = create_ψ0(sites, N_lat, N_part, boson_dim; linkdims=100)
        # E, ψ = dmrg(Ham, ψ0; nsweeps=N_sweeps, maxdim=max_dim, cutoff=cutoff, noise=noise)
        # # # if m == 1
        # # if m == 1 && n == 1
        # #     E, ψ = dmrg(Ham, ψ0; nsweeps=N_sweeps, maxdim=max_dim, cutoff=cutoff, noise=noise,
        # #                 outputlevel=1)
        # # else
        # #     E, ψ = dmrg(Ham, ψ0; nsweeps=N_sweeps, maxdim=max_dim, cutoff=cutoff, noise=noise,
        # #                 outputlevel=0)
        # # end # if
        E_vec = Vector{Float64}([])
        E_bulk_vec = Vector{Float64}([])
        expectation_values_vec = Vector([])
        for (i, N_part) in enumerate(N_part_arr)
            E_repeat_vec = Vector{Float64}([])
            ψ_repeat_vec = Vector{MPS}([])
            for ii in 1:N_repeat
                ψ0 = create_ψ0(sites, N_lat, N_part, boson_dim; linkdims=100)
                E, ψ = dmrg(Ham, ψ0; nsweeps=N_sweeps_small, maxdim=max_dim, cutoff=cutoff, noise=noise)

                push!(E_repeat_vec, E)
                push!(ψ_repeat_vec, ψ)
            end # for
            E = E_repeat_vec[argmin(E_repeat_vec)]
            ψ = ψ_repeat_vec[argmin(E_repeat_vec)]

            E, ψ = dmrg(Ham, ψ; nsweeps=N_sweeps, maxdim=max_dim, cutoff=cutoff, noise=noise)
            expectation_values = calc_expectation_values(ψ, Ham_bulk, sites, k_arr, N_lat)
            E_bulk = expectation_values["expect_E"]

            push!(E_vec, E)
            push!(E_bulk_vec, E_bulk)
            push!(expectation_values_vec, expectation_values)

            println("******* $(i)/$(length(N_part_arr)) DONE *******")
        end # for

        expectation_values = expectation_values_vec[3]
        δN, g1, g2, g1_matrix, g2_matrix, S_k_vec, M_k_vec = expectation_values["δN"], expectation_values["g1"], expectation_values["g2"], expectation_values["g1_matrix"], expectation_values["g2_matrix"], expectation_values["S_k_vec"], expectation_values["M_k_vec"]
        integ_g1_arr[m, n] = sum(abs.(g1_matrix)) - tr(abs.(g1_matrix))
        integ_g2_arr[m, n] = sum(abs.(g2_matrix)) - tr(abs.(g2_matrix))
        δN_arr[m, n], g1_arr[m, n], g2_arr[m, n] = δN, g1, g2
        integ_S_k_arr[m, n] = sum(abs.(S_k_vec[(N_lat÷10):end]))
        integ_M_k_arr[m, n] = sum(abs.(M_k_vec[(N_lat÷10):end]))
        # push!(expectation_values_vec, expectation_values)

        Δ1 = E_vec[2] + E_vec[4] - 2.0*E_vec[3]
        Δ2 = E_vec[1] + E_vec[5] - 2.0*E_vec[3]
        Δ1_arr[m, n] = Δ1
        Δ2_arr[m, n] = Δ2
        merge!(expectation_values, Dict("Δ1" => Δ1))
        merge!(expectation_values, Dict("Δ2" => Δ2))
        @show Δ1
        @show Δ2

        Δ1_bulk = E_bulk_vec[2] + E_bulk_vec[4] - 2.0*E_bulk_vec[3]
        Δ2_bulk = E_bulk_vec[1] + E_bulk_vec[5] - 2.0*E_bulk_vec[3]
        Δ1_bulk_arr[m, n] = Δ1_bulk
        Δ2_bulk_arr[m, n] = Δ2_bulk
        merge!(expectation_values, Dict("Δ1_bulk" => Δ1_bulk))
        merge!(expectation_values, Dict("Δ2_bulk" => Δ2_bulk))
        @show Δ1_bulk
        @show Δ2_bulk

        merge!(expectation_values, Dict("abs_J1_vec" => abs_J1_vec))
        merge!(expectation_values, Dict("gx_vec" => gx_vec))

        npzwrite("$(save_path)/Expectation_Values!m=$(m)!n=$(n)!abs_J1=$(round(abs_J1_m; digits=4))!gx=$(round(gx_n; digits=4))!$date_string.npz", expectation_values)

        # if m == 1
        #      println("******* $(n)/$(N_gx) DONE *******")
        # end # if

        println("******* $(m)/$(N_J1) && $(n)/$(N_gx) DONE *******")
    end # for

    #     end # for
    # end # for

    # if Threads.threadid() == 1
    #     ##### PLOTTING #####
    #     plotN_order_parameters_vs_J1_gx(abs_J1_vec, gx_vec, δN_arr, g1_arr, g2_arr, integ_g1_arr, integ_g2_arr, integ_S_k_arr, integ_M_k_arr)
    #     ##### END OF PLOTTING #####
    # end # if
    ##### END OF PHASE CURVE #####
end # function

function routine_phase_diagram_J1_gx_HPC_checkpoints(N_lat, N_part, boson_dim, conserveParticleNumber, μ, N_repeat, n_checkpoint, loadCheckpoint, checkpoint_job_id, N_sweeps, max_dim, cutoff, noise)
    sites = siteinds("Boson", N_lat; dim=boson_dim, conserve_qns=conserveParticleNumber)
    k_arr = LinRange(0.0, 2π, N_lat+1)[1:N_lat]

    slurm_array_job_id = parse(Int, ARGS[3])
    current_job_id = loadCheckpoint ? checkpoint_job_id : slurm_array_job_id

    checkpoint_path = "$(@__DIR__)/Checkpoints/$(current_job_id)"
    mkpath(checkpoint_path)

    save_path = "$(@__DIR__)/NPY/$(current_job_id)/Expectation_Values_Vec"
    mkpath(save_path)

    ##### PHASE DIAGRAM #####
    N_J1 = 21  # 4  # 21
    N_gx = 22  # 12  # 22

    abs_J1_vec = LinRange(0.0, 0.3, N_J1)
    gx_vec = LinRange(-5.0, 0.0, N_gx)

    ##### SLURM #####
    N_total = N_J1 * N_gx * N_repeat

    slurm_task_id = parse(Int, ARGS[1])
    slurm_task_count = parse(Int, ARGS[2])

    q = N_total ÷ slurm_task_count
    r = N_total % slurm_task_count
    calc_count = (slurm_task_id <= r) ? (q + 1) : q
    idx_vec = [slurm_task_id + i * slurm_task_count for i in 0:(calc_count - 1)]

    I_vec = (vec(CartesianIndices(zeros(N_repeat, N_gx, N_J1))))[idx_vec]
    ir_vec = [I[1] for I in I_vec]
    n_vec = [I[2] for I in I_vec]
    m_vec = [I[3] for I in I_vec]
    ##### END OF SLURM #####

    for i in 1:calc_count
        ir = ir_vec[i]
        n = n_vec[i]
        m = m_vec[i]

        abs_J1_m = abs_J1_vec[m]
        gx_n = gx_vec[n]
        @show abs_J1_m
        @show gx_n

        info_string = "m=$(m)!n=$(n)!ir=$(ir)!abs_J1=$(round(abs_J1_m; digits=4))!gx=$(round(gx_n; digits=4))"

        Ham = calc_brick_wall_Ham(sites, N_lat, conserveParticleNumber, μ;
                                  J_1=abs_J1_m, J_2=-abs_J1_m*(10^(0.5521)), g_0=1.0, g_x=gx_n, g_z=0.0, G_000=1.0000, G_001=-0.2122, G_011=0.1666)  # s-band
                                  # J_1=-abs_J1_m, J_2=abs_J1_m*(10^(0.5521)), g_0=1.0, g_x=gx_n, g_z=0.0, G_000=1.0000, G_001=-0.2122, G_011=0.1666)  # s-band
                                  # J_1=-abs_J1_m, J_2=-abs_J1_m*(10^(0.375)), g_0=1.0, g_x=gx_n, g_z=0.0, G_000=1.0000, G_001=-0.5000, G_011=0.5000)  # p-band

        checkpoint_filename = "$(checkpoint_path)/Checkpoint!$(info_string).h5"
        if loadCheckpoint
            ψ0 = create_ψ0(sites, N_lat, N_part, boson_dim; linkdims=10)
            ψ = h5open(checkpoint_filename, "r") do f
                read(f, "MPS", typeof(ψ0))
            end # h5open
        else
            ψ = create_ψ0(sites, N_lat, N_part, boson_dim; linkdims=100)
        end # if

        for is in 1:N_sweeps
            E, ψ = dmrg(Ham, ψ; nsweeps=1,
                maxdim=select_par_by_sweep_no(is, max_dim),
                cutoff=select_par_by_sweep_no(is, cutoff),
                noise=select_par_by_sweep_no(is, noise))

            if is % n_checkpoint == 0
                f = h5open(checkpoint_filename, "w") do f
                    write(f, "MPS", ψ)
                end # h5open
            end # if
        end # for

        expectation_values = calc_expectation_values(ψ, Ham, sites, k_arr, N_lat)
        npzwrite("$(save_path)/Expectation_Values!$(info_string).npz", expectation_values)

        println("******* $(m)/$(N_J1) && $(n)/$(N_gx) && $(ir)/$(N_repeat) DONE *******")
    end # for
    ##### END OF PHASE DIAGRAM #####
end # function

# TODO Generalize this code to any two parameters
# TODO Adapt this code to normal computer
function routine_phase_diagram_gx_Npart_HPC(N_lat, boson_dim, conserveParticleNumber, μ, N_sweeps, max_dim, cutoff, noise)
    sites = siteinds("Boson", N_lat; dim=boson_dim, conserve_qns=conserveParticleNumber)
    k_arr = LinRange(0.0, 2π, N_lat+1)[1:N_lat]

    save_path = "$(@__DIR__)/NPY/$(Dates.today())/Expectation_Values_Vec"
    date_string = """$(Dates.format(Dates.now(), "Y-mm-dd!HH.MM.SS"))"""
    mkpath(save_path)

    ##### PHASE CURVE #####
    N_gx = 12  # 12  # 22
    N_N_part = 11  # 4  # 21
    # ΔN_part = N_lat ÷ N_N_part

    gx_vec = LinRange(-10.0, 10.0, N_gx)
    # N_part_vec = collect(N_lat:ΔN_part:(2*N_lat))
    N_part_vec = round.(Int, LinRange(N_lat, 2*N_lat, N_N_part))

    # @show N_N_part
    # @show length(N_part_vec)
    @assert N_N_part == length(N_part_vec)

    δN_arr = zeros(Float64, N_gx, N_N_part)
    g1_arr = zeros(Float64, N_gx, N_N_part)
    g2_arr = zeros(Float64, N_gx, N_N_part)
    integ_g1_arr = zeros(Float64, N_gx, N_N_part)
    integ_g2_arr = zeros(Float64, N_gx, N_N_part)
    integ_S_k_arr = zeros(Float64, N_gx, N_N_part)
    integ_M_k_arr = zeros(Float64, N_gx, N_N_part)

    ##### SLURM #####
    N_total = N_gx * N_N_part

    slurm_task_id = parse(Int, ARGS[1])
    slurm_task_count = parse(Int, ARGS[2])

    calc_count = (slurm_task_id <= (N_total % slurm_task_count)) ? (N_total ÷ slurm_task_count + 1) : (N_total ÷ slurm_task_count)
    idx_vec = [slurm_task_id + i * slurm_task_count for i in 0:(calc_count - 1)]
    m_vec = [((idx - 1) ÷ N_N_part) + 1 for idx in idx_vec]
    n_vec = [mod1(idx, N_N_part) for idx in idx_vec]
    ##### END OF SLURM #####

    # @show length(m_vec)
    # @show length(gx_vec)
    # @show length(n_vec)
    # @show length(N_part_vec)
    # @assert length(m_vec) == length(gx_vec)
    # @assert length(n_vec) == length(N_part_vec)

    # # for n in 1:N_gx
    # Threads.@threads for m in 1:N_gx
    #     for n in 1:N_N_part

    for i in 1:calc_count
        m = m_vec[i]
        n = n_vec[i]

        gx_m = gx_vec[m]
        N_part_n = N_part_vec[n]

        abs_J1 = 0.0
        Ham = calc_brick_wall_Ham(sites, N_lat, conserveParticleNumber, μ;
                                  # J_1=-abs_J1, J_2=abs_J1*(10^(0.5521)), g_0=1.0, g_x=gx_m, g_z=0.0, G_000=1.0000, G_001=-0.2122, G_011=0.1666)  # s-band
                                  J_1=-abs_J1, J_2=-abs_J1*(10^(0.375)), g_0=1.0, g_x=gx_m, g_z=0.0, G_000=1.0000, G_001=-0.5000, G_011=0.5000)  # p-band

        ψ0 = create_ψ0(sites, N_lat, N_part_n, boson_dim; linkdims=100)
        E, ψ = dmrg(Ham, ψ0; nsweeps=N_sweeps, maxdim=max_dim, cutoff=cutoff, noise=noise)
        # # if m == 1
        # if m == 1 && n == 1
        #     E, ψ = dmrg(Ham, ψ0; nsweeps=N_sweeps, maxdim=max_dim, cutoff=cutoff, noise=noise,
        #                 outputlevel=1)
        # else
        #     E, ψ = dmrg(Ham, ψ0; nsweeps=N_sweeps, maxdim=max_dim, cutoff=cutoff, noise=noise,
        #                 outputlevel=0)
        # end # if

        expectation_values = calc_expectation_values(ψ, Ham, sites, k_arr, N_lat)
        δN, g1, g2, g1_matrix, g2_matrix, S_k_vec, M_k_vec = expectation_values["δN"], expectation_values["g1"], expectation_values["g2"], expectation_values["g1_matrix"], expectation_values["g2_matrix"], expectation_values["S_k_vec"], expectation_values["M_k_vec"]
        integ_g1_arr[m, n] = sum(abs.(g1_matrix)) - tr(abs.(g1_matrix))
        integ_g2_arr[m, n] = sum(abs.(g2_matrix)) - tr(abs.(g2_matrix))
        δN_arr[m, n], g1_arr[m, n], g2_arr[m, n] = δN, g1, g2
        integ_S_k_arr[m, n] = sum(abs.(S_k_vec[(N_lat÷10):end]))
        integ_M_k_arr[m, n] = sum(abs.(M_k_vec[(N_lat÷10):end]))
        # push!(expectation_values_vec, expectation_values)

        npzwrite("$(save_path)/Expectation_Values!m=$(m)!n=$(n)!gx=$(round(gx_m; digits=4))!N_part=$(N_part_n)!$date_string.npz", expectation_values)

        # if m == 1
        #      println("******* $(n)/$(N_N_part) DONE *******")
        # end # if

        println("******* $(m)/$(N_gx) && $(n)/$(N_N_part) DONE *******")
    end # for

    #     end # for
    # end # for

    # if Threads.threadid() == 1
    #     ##### PLOTTING #####
    #     plotN_order_parameters_vs_J1_gx(abs_J1_vec, gx_vec, δN_arr, g1_arr, g2_arr, integ_g1_arr, integ_g2_arr, integ_S_k_arr, integ_M_k_arr)
    #     ##### END OF PLOTTING #####
    # end # if
    ##### END OF PHASE CURVE #####
end # function

function routine_phase_diagram_J1_Npart_HPC(N_lat, boson_dim, conserveParticleNumber, μ, N_sweeps, max_dim, cutoff, noise)
    sites = siteinds("Boson", N_lat; dim=boson_dim, conserve_qns=conserveParticleNumber)
    k_arr = LinRange(0.0, 2π, N_lat+1)[1:N_lat]

    save_path = "$(@__DIR__)/NPY/$(Dates.today())/Expectation_Values_Vec"
    date_string = """$(Dates.format(Dates.now(), "Y-mm-dd!HH.MM.SS"))"""
    mkpath(save_path)

    ##### PHASE CURVE #####
    N_J1 = 12  # 12  # 22
    N_N_part = 11  # 4  # 21
    # ΔN_part = N_lat ÷ N_N_part

    abs_J1_vec = LinRange(0.0, 0.10, N_J1)
    # N_part_vec = collect(N_lat:ΔN_part:(2*N_lat))
    N_part_vec = round.(Int, LinRange(N_lat, 2*N_lat, N_N_part))

    # @show N_N_part
    # @show length(N_part_vec)
    @assert N_N_part == length(N_part_vec)

    δN_arr = zeros(Float64, N_J1, N_N_part)
    g1_arr = zeros(Float64, N_J1, N_N_part)
    g2_arr = zeros(Float64, N_J1, N_N_part)
    integ_g1_arr = zeros(Float64, N_J1, N_N_part)
    integ_g2_arr = zeros(Float64, N_J1, N_N_part)
    integ_S_k_arr = zeros(Float64, N_J1, N_N_part)
    integ_M_k_arr = zeros(Float64, N_J1, N_N_part)

    ##### SLURM #####
    N_total = N_J1 * N_N_part

    slurm_task_id = parse(Int, ARGS[1])
    slurm_task_count = parse(Int, ARGS[2])

    calc_count = (slurm_task_id <= (N_total % slurm_task_count)) ? (N_total ÷ slurm_task_count + 1) : (N_total ÷ slurm_task_count)
    idx_vec = [slurm_task_id + i * slurm_task_count for i in 0:(calc_count - 1)]
    m_vec = [((idx - 1) ÷ N_N_part) + 1 for idx in idx_vec]
    n_vec = [mod1(idx, N_N_part) for idx in idx_vec]
    ##### END OF SLURM #####

    for i in 1:calc_count
        m = m_vec[i]
        n = n_vec[i]

        abs_J1_m = abs_J1_vec[m]
        N_part_n = N_part_vec[n]

        g_x = 0.0
        Ham = calc_brick_wall_Ham(sites, N_lat, conserveParticleNumber, μ;
                                  J_1=-abs_J1_m, J_2=abs_J1_m*(10^(0.5521)), g_0=1.0, g_x=g_x, g_z=0.0, G_000=1.0000, G_001=-0.2122, G_011=0.1666)  # s-band
                                  # J_1=-abs_J1_m, J_2=-abs_J1_m*(10^(0.375)), g_0=1.0, g_x=g_x, g_z=0.0, G_000=1.0000, G_001=-0.5000, G_011=0.5000)  # p-band

        ψ0 = create_ψ0(sites, N_lat, N_part_n, boson_dim; linkdims=100)
        E, ψ = dmrg(Ham, ψ0; nsweeps=N_sweeps, maxdim=max_dim, cutoff=cutoff, noise=noise)
        # # if m == 1
        # if m == 1 && n == 1
        #     E, ψ = dmrg(Ham, ψ0; nsweeps=N_sweeps, maxdim=max_dim, cutoff=cutoff, noise=noise,
        #                 outputlevel=1)
        # else
        #     E, ψ = dmrg(Ham, ψ0; nsweeps=N_sweeps, maxdim=max_dim, cutoff=cutoff, noise=noise,
        #                 outputlevel=0)
        # end # if

        expectation_values = calc_expectation_values(ψ, Ham, sites, k_arr, N_lat)
        δN, g1, g2, g1_matrix, g2_matrix, S_k_vec, M_k_vec = expectation_values["δN"], expectation_values["g1"], expectation_values["g2"], expectation_values["g1_matrix"], expectation_values["g2_matrix"], expectation_values["S_k_vec"], expectation_values["M_k_vec"]
        integ_g1_arr[m, n] = sum(abs.(g1_matrix)) - tr(abs.(g1_matrix))
        integ_g2_arr[m, n] = sum(abs.(g2_matrix)) - tr(abs.(g2_matrix))
        δN_arr[m, n], g1_arr[m, n], g2_arr[m, n] = δN, g1, g2
        integ_S_k_arr[m, n] = sum(abs.(S_k_vec[(N_lat÷10):end]))
        integ_M_k_arr[m, n] = sum(abs.(M_k_vec[(N_lat÷10):end]))
        # push!(expectation_values_vec, expectation_values)

        npzwrite("$(save_path)/Expectation_Values!m=$(m)!n=$(n)!abs_J1=$(round(abs_J1_m; digits=4))!N_part=$(N_part_n)!$date_string.npz", expectation_values)

        # if m == 1
        #      println("******* $(n)/$(N_N_part) DONE *******")
        # end # if

        println("******* $(m)/$(N_J1) && $(n)/$(N_N_part) DONE *******")
    end # for

    #     end # for
    # end # for

    # if Threads.threadid() == 1
    #     ##### PLOTTING #####
    #     plotN_order_parameters_vs_J1_gx(abs_J1_vec, gx_vec, δN_arr, g1_arr, g2_arr, integ_g1_arr, integ_g2_arr, integ_S_k_arr, integ_M_k_arr)
    #     ##### END OF PLOTTING #####
    # end # if
    ##### END OF PHASE CURVE #####
end # function
##### END OF PHASE DIAGRAM ROUTINES #####

##### READING ROUTINES #####
function routine_read_phase_diagram_J1_gx(N_lat, N_part, boson_dim, conserveParticleNumber, μ, N_sweeps, max_dim, cutoff, noise)
    k_arr = LinRange(0.0, 2π, N_lat+1)[1:N_lat]

    # save_path = "$(@__DIR__)/NPY/2024-12-07/15:25/Expectation_Values_Vec"
    # save_path = "$(@__DIR__)/NPY/2024-12-07/19:21/Expectation_Values_Vec"
    # save_path = "$(@__DIR__)/NPY/2024-12-09/15:47/Expectation_Values_Vec"
    # save_path = "$(@__DIR__)/NPY/2024-12-11/01:17/Expectation_Values_Vec"
    # save_path = "$(@__DIR__)/NPY/2024-12-12/15:24/Expectation_Values_Vec"
    # save_path = "$(@__DIR__)/NPY/2024-12-14/01:27/Expectation_Values_Vec"
    # save_path = "$(@__DIR__)/NPY/2024-12-15/17:21/Expectation_Values_Vec"
    save_path = "$(@__DIR__)/NPY/2024-12-28/17:40/Expectation_Values_Vec"
    file_names = readdir(save_path)

    ##### PHASE DIAGRAM #####
    N_J1 = 11  # 4  # 21
    N_gx = 12  # 12  # 22

    abs_J1_vec = LinRange(0.0, 0.05, N_J1)
    gx_vec = LinRange(-10.0, 10.0, N_gx)
    # gx_vec = LinRange(0.0, 100.0, N_gx)

    δN_arr = zeros(Float64, N_J1, N_gx)
    g1_arr = zeros(Float64, N_J1, N_gx)
    g2_arr = zeros(Float64, N_J1, N_gx)
    integ_g1_arr = zeros(Float64, N_J1, N_gx)
    integ_g2_arr = zeros(Float64, N_J1, N_gx)
    integ_S_k_arr = zeros(Float64, N_J1, N_gx)
    integ_M_k_arr = zeros(Float64, N_J1, N_gx)
    abs_O_odd_arr = zeros(Float64, N_J1, N_gx)

    # for n in 1:N_J1
    for m in 1:N_J1
        for n in 1:N_gx
            abs_J1_m = abs_J1_vec[m]
            gx_n = gx_vec[n]

            expectation_values = npzread("$(save_path)/$(file_names[findfirst(contains("m=$(m)!n=$(n)"), file_names)])")
            δN, g1, g2, g1_matrix, g2_matrix, S_k_vec, M_k_vec, O_odd, O_even = expectation_values["δN"], expectation_values["g1"], expectation_values["g2"], expectation_values["g1_matrix"], expectation_values["g2_matrix"], expectation_values["S_k_vec"], expectation_values["M_k_vec"], expectation_values["O_odd"], expectation_values["O_even"]
            integ_g1_arr[m, n] = sum(abs.(g1_matrix)) - tr(abs.(g1_matrix))
            integ_g2_arr[m, n] = sum(abs.(g2_matrix)) - tr(abs.(g2_matrix))
            δN_arr[m, n], g1_arr[m, n], g2_arr[m, n] = δN, g1, g2
            integ_S_k_arr[m, n] = sum(abs.(S_k_vec[(N_lat÷10):end]))
            integ_M_k_arr[m, n] = sum(abs.(M_k_vec[(N_lat÷10):end]))
            abs_O_odd_arr[m, n] = abs(O_odd)
        end # for
    end # for

    if Threads.threadid() == 1
        ##### PLOTTING #####
        plotN_order_parameters_vs_J1_gx(abs_J1_vec, gx_vec, δN_arr, g1_arr, g2_arr, integ_g1_arr, integ_g2_arr, integ_S_k_arr, integ_M_k_arr, abs_O_odd_arr)
        ##### END OF PLOTTING #####
    end # if
    ##### END OF PHASE DIAGRAM #####
end # function

function routine_read_phase_diagram_J1_gx_v2(N_lat, N_part, boson_dim, conserveParticleNumber, μ, N_sweeps, max_dim, cutoff, noise)
    # save_path = "$(@__DIR__)/Selected NPY/Phase_Diagram_J1_gx, 2025-11-27/Expectation_Values_Vec"  # N_lat = 201
    # save_path = "$(@__DIR__)/Selected NPY/Phase_Diagram_J1_gx, 2025-12-14/Expectation_Values_Vec"  # N_lat = 401
    # save_path = "$(@__DIR__)/Selected NPY/Phase_Diagram_J1_gx, 2025-12-13/Expectation_Values_Vec"  # N_lat = 601
    # save_path = "$(@__DIR__)/Selected NPY/Phase_Diagram_J1_gx, 2026-01-16, 1/Expectation_Values_Vec"  # N_lat = 601
    # save_path = "$(@__DIR__)/Selected NPY/Phase_Diagram_J1_gx, 2026-01-16, 2/Expectation_Values_Vec"  # N_lat = 601
    # save_path = "$(@__DIR__)/Selected NPY/Phase_Diagram_J1_gx, 2026-01-18, 1/Expectation_Values_Vec"  # N_lat = 201
    # save_path = "$(@__DIR__)/Selected NPY/Phase_Diagram_J1_gx, 2026-01-18, 2/Expectation_Values_Vec"  # N_lat = 201
    # save_path = "$(@__DIR__)/Selected NPY/Phase_Diagram_J1_gx, Repeat, 2026-01-19, 1/Expectation_Values_Vec"  # N_lat = 601
    # save_path = "$(@__DIR__)/Selected NPY/Phase_Diagram_J1_gx, chi=300, 2026-01-19, 1/Expectation_Values_Vec"  # N_lat = 601
    # save_path = "$(@__DIR__)/Selected NPY/Phase_Diagram_J1_gx, chi=100, 2026-01-23, 1/Expectation_Values_Vec"  # N_lat = 601
    # save_path = "$(@__DIR__)/Selected NPY/Phase_Diagram_J1_gx, chi=200, 2026-01-24, 1/Expectation_Values_Vec"  # N_lat = 601
    # save_path = "$(@__DIR__)/Selected NPY/Phase_Diagram_J1_gx, chi=300, 2026-01-25, 1/Expectation_Values_Vec"  # N_lat = 601
    save_path = "$(@__DIR__)/Selected NPY/Phase_Diagram_J1_gx, chi=500, 2026-01-29/Expectation_Values_Vec"  # N_lat = 601
    file_names = readdir(save_path)

    ##### PHASE DIAGRAM #####
    expectation_values = npzread("$(save_path)/$(file_names[findfirst(contains("m=1!n=1"), file_names)])")
    abs_J1_vec = expectation_values["abs_J1_vec"]
    gx_vec = expectation_values["gx_vec"]
    # N_lat = expectation_values["N_lat"]
    N_J1 = length(abs_J1_vec)
    N_gx = length(gx_vec)

    k_arr = LinRange(0.0, 2π, N_lat+1)[1:N_lat]

    g1_arr = zeros(Float64, N_J1, N_gx)
    g2_arr = zeros(Float64, N_J1, N_gx)
    δN_arr = zeros(Float64, N_J1, N_gx)
    S_kπ_arr = zeros(Float64, N_J1, N_gx)
    O_even_arr = zeros(Float64, N_J1, N_gx)
    O_odd_arr = zeros(Float64, N_J1, N_gx)
    κ2_arr = zeros(Float64, N_J1, N_gx)
    M_k0_arr = zeros(Float64, N_J1, N_gx)
    M_kπ_arr = zeros(Float64, N_J1, N_gx)
    c_arr = zeros(Float64, N_J1, N_gx)
    # kmax_M_arr = zeros(Float64, N_J1, N_gx)
    # M_kmax_arr = zeros(Float64, N_J1, N_gx)
    prob_Q_arr = zeros(Float64, N_J1, N_gx)

    # for n in 1:N_J1
    for m in 1:N_J1
        for n in 1:N_gx
            abs_J1_m = abs_J1_vec[m]
            gx_n = gx_vec[n]

            expectation_values = npzread("$(save_path)/$(file_names[findfirst(contains("m=$(m)!n=$(n)"), file_names)])")
            g1, g2, bulk_range, g1_matrix, g2_matrix, δN, S_kπ, O_even, O_odd, κ2, M_k0, M_kπ, M_k_vec, entropy_S_vec, prob_Q_mod_bulk = expectation_values["g1"], expectation_values["g2"], expectation_values["bulk_range"], expectation_values["g1_matrix"], expectation_values["g2_matrix"], expectation_values["δN"], expectation_values["S_kπ"], expectation_values["O_even"], expectation_values["O_odd"], expectation_values["κ2"], expectation_values["M_k0"], expectation_values["M_kπ"], expectation_values["M_k_vec"], expectation_values["entropy_S_vec"], expectation_values["prob_Q_mod_bulk"]
            g1_arr[m, n] = g1
            g2_arr[m, n] = g2
            # g2_arr[m, n] = sum(abs.(g2_matrix[bulk_range[1], (bulk_range[1]+100):(bulk_range[1]+200)]))
            δN_arr[m, n] = δN
            S_kπ_arr[m, n] = S_kπ
            O_even_arr[m, n] = O_even
            O_odd_arr[m, n] = O_odd
            κ2_arr[m, n] = κ2
            M_k0_arr[m, n] = M_k0
            M_kπ_arr[m, n] = M_kπ

            c_arr[m, n], _ = DMRG.calc_central_charge(entropy_S_vec, N_lat)
            # kmax_M_arr[m, n] = k_arr[argmax(real.(M_k_vec[1:(N_lat÷2)]))]
            # M_kmax_arr[m, n] = maximum(real.(M_k_vec))
            prob_Q_arr[m, n] = prob_Q_mod_bulk

            println("******* $(m)/$(N_J1) && $(n)/$(N_gx) DONE *******")
        end # for
    end # for

    ### FITTING CORRELATIONS ###
    # FITTING PARAMETERS #
    idx_start = 100  # 200
    fit_range = 100:200
    lower = [-100.0, 0.0]
    upper = [100.0, 10.0]
    exp_lower = [-100.0, 1.0]
    exp_upper = [100.0, 1000.0]
    # END OF FITTING PARAMETERS #

    # α_g1_arr = zeros(Float64, N_J1, N_gx)
    # res_g1_arr = zeros(Float64, N_J1, N_gx)
    # α_g2_arr = zeros(Float64, N_J1, N_gx)
    # res_g2_arr = zeros(Float64, N_J1, N_gx)
    # exp_ξ_g2_arr = zeros(Float64, N_J1, N_gx)
    # exp_res_g2_arr = zeros(Float64, N_J1, N_gx)
    # for m in 1:N_J1
    #     for n in 1:N_gx
    #         abs_J1_m = abs_J1_vec[m]
    #         gx_n = gx_vec[n]

    #         expectation_values = npzread("$(save_path)/$(file_names[findfirst(contains("m=$(m)!n=$(n)"), file_names)])")
    #         g1, g2, bulk_range, g1_matrix, g2_matrix, δN, S_kπ, O_even, O_odd, κ2, M_k0, M_kπ, M_k_vec, entropy_S_vec, prob_Q_mod_bulk = expectation_values["g1"], expectation_values["g2"], expectation_values["bulk_range"], expectation_values["g1_matrix"], expectation_values["g2_matrix"], expectation_values["δN"], expectation_values["S_kπ"], expectation_values["O_even"], expectation_values["O_odd"], expectation_values["κ2"], expectation_values["M_k0"], expectation_values["M_kπ"], expectation_values["M_k_vec"], expectation_values["entropy_S_vec"], expectation_values["prob_Q_mod_bulk"]

    #         # NOTE: Here fit is done for abs correlations, not correlations themselves.
    #         g1_fit_params, g1_fit_resid, g1_fit_vec = calc_fit_1D(ln_power_law_fit, fit_range, log.(abs.(g1_matrix[idx_start, idx_start .+ fit_range])), [1.0, 1.0])
    #         # g1_fit_params, g1_fit_resid, g1_fit_vec = calc_fit_1D_MC(ln_power_law_fit, fit_range, log.(abs.(g1_matrix[idx_start, idx_start .+ fit_range])), lower, upper; N_p0=1)
    #         # g1_fit_params, g1_fit_resid, g1_fit_vec = calc_fit_1D_MC(power_law_fit, fit_range, abs.(g1_matrix[idx_start, idx_start .+ fit_range]), lower, upper; N_p0=1)
    #         α_g1_arr[m, n] = g1_fit_params[2]
    #         res_g1_arr[m, n] = sum(abs2.(g1_fit_resid))
    #         g2_fit_params, g2_fit_resid, g2_fit_vec = calc_fit_1D(ln_power_law_fit, fit_range, log.(abs.(g2_matrix[idx_start, idx_start .+ fit_range])), [1.0, 1.0])
    #         # g2_fit_params, g2_fit_resid, g2_fit_vec = calc_fit_1D_MC(ln_power_law_fit, fit_range, log.(abs.(g2_matrix[idx_start, idx_start .+ fit_range])), lower, upper; N_p0=1)
    #         # g2_fit_params, g2_fit_resid, g2_fit_vec = calc_fit_1D_MC(power_law_fit, fit_range, abs.(g2_matrix[idx_start, idx_start .+ fit_range]), lower, upper; N_p0=1)
    #         α_g2_arr[m, n] = g2_fit_params[2]
    #         res_g2_arr[m, n] = sum(abs2.(g2_fit_resid))

    #         # exp_g2_fit_params, exp_g2_fit_resid, exp_g2_fit_vec = calc_fit_1D_MC(ln_exp_law_fit, fit_range, log.(abs.(g2_matrix[idx_start, idx_start .+ fit_range])), exp_lower, exp_upper; N_p0=1)
    #         exp_g2_fit_params, exp_g2_fit_resid, exp_g2_fit_vec = calc_fit_1D(ln_exp_law_fit, fit_range, log.(abs.(g2_matrix[idx_start, idx_start .+ fit_range])), [1.0, 1.0])
    #         exp_ξ_g2_arr[m, n] = exp_g2_fit_params[2]
    #         exp_res_g2_arr[m, n] = sum(abs2.(exp_g2_fit_resid))

    #         println("******* $(m)/$(N_J1) && $(n)/$(N_gx) DONE *******")
    #     end # for
    # end # for
    # ### END OF FITTING CORRELATIONS ###

    if Threads.threadid() == 1
        ##### PLOTTING #####
        plotN_order_parameters_vs_J1_gx_v2(abs_J1_vec, gx_vec, g1_arr, g2_arr, δN_arr, S_kπ_arr, O_even_arr, O_odd_arr, κ2_arr, M_k0_arr, M_kπ_arr, c_arr, prob_Q_arr)
        # plotN_fit_res_vs_J1_gx(abs_J1_vec, gx_vec, α_g1_arr, res_g1_arr, α_g2_arr, res_g2_arr, exp_ξ_g2_arr, exp_res_g2_arr; vlim=[lower[2], upper[2]])
        ##### END OF PLOTTING #####
    end # if
    ##### END OF PHASE DIAGRAM #####

    # ##### CLASSIFICATION #####
    # phase_name_arr = Array{String}(undef, N_J1, N_gx)
    # for m in 1:N_J1
    #     for n in 1:N_gx
    #         expectation_values = npzread("$(save_path)/$(file_names[findfirst(contains("m=$(m)!n=$(n)"), file_names)])")

    #         phase_name_arr[m, n] = classify_qmb_phase_v1(expectation_values)
    #     end # for
    # end # for

    # plot_phase_diagram_by_name(abs_J1_vec, gx_vec, phase_name_arr;
    #     x_label=raw"$|J_1|/g_0$", y_label=raw"$g_x/g_0$", file_name="Phase_Diagram_by_Name_vs_J1_gx")
    # ##### END OF CLASSIFICATION #####

    ########## PROPER PHASE DIAGRAM ##########
    ##### PHASE SEPARATING CURVES #####
    phase_name_arr = Array{String}(undef, N_J1, N_gx)
    for m in 1:N_J1
        for n in 1:N_gx
            expectation_values = npzread("$(save_path)/$(file_names[findfirst(contains("m=$(m)!n=$(n)"), file_names)])")

            phase_name_arr[m, n] = classify_qmb_phase_v1(expectation_values)
        end # for
    end # for

    CSF_curve_vec2 = calc_phase_curve_v1("CSF", abs_J1_vec, gx_vec, phase_name_arr)
    MI_curve_vec2 = calc_phase_curve_v1("MI", abs_J1_vec, gx_vec, phase_name_arr)
    DW_curve_vec2 = calc_phase_curve_v1("DW", abs_J1_vec, gx_vec, phase_name_arr)
    PSF_curve_vec2 = calc_phase_curve_v1("PSF", abs_J1_vec, gx_vec, phase_name_arr)

    plot_phase_diagram_vs_J1_gx(abs_J1_vec, gx_vec, CSF_curve_vec2, MI_curve_vec2, DW_curve_vec2, PSF_curve_vec2)
    ##### END OF PHASE SEPARATING CURVES #####
    # END OF PROPER PHASE DIAGRAM ##########

    # ##### PHASE CURVE (HORIZONTAL CUT / FIXED g_x) #####
    # S_gx = -0.0
    # S_n = x2index(S_gx, gx_vec)
    # gx_n = gx_vec[S_n]
    # @show gx_n

    # # expect_list = ["O_even", "g1", "κ2"]  # MI-CSF
    # # expect_list = ["g2", "δN", "S_kπ"]  # DW-PSF
    # # expect_list = ["κ2", "g1", "O_odd", "δN", "S_kπ"]  # DW-CSF
    # # expect_list = ["κ2", "g1", "g2", "O_odd"]  # PSF-CSF
    # # expect_list = ["O_odd", "O_even", "δN", "S_kπ"]  # MI-DW
    # expect_list = ["O_even", "O_odd", "g1", "g2", "κ2", "δN", "S_kπ"]  # All
    # N_expect = length(expect_list)

    # expect_vec2 = []
    # for (i, expect_name) in enumerate(expect_list)
    #     expect_vec = []
    #     for m in 1:N_J1
    #         abs_J1_m = abs_J1_vec[m]

    #         expectation_values = npzread("$(save_path)/$(file_names[findfirst(contains("m=$(m)!n=$(S_n)"), file_names)])")
    #         expect_val = expectation_values[expect_name]

    #         push!(expect_vec, expect_val)
    #     end # for

    #     push!(expect_vec2, expect_vec)
    # end # for

    # plotN_ys_vs_x(abs_J1_vec, expect_vec2, @__DIR__;
    #   ax_title=raw"$g_x$ = "*"$(round(gx_n, digits=4))",
    #   x_label_vec=vcat(["" for i in 1:(N_expect-1)], [raw"$|J_1|/g_0$"]),
    #   y_label_vec=expect_list, marker="o",
    #   saveDirectly=false, file_name="Expect_vs_J1")
    # ##### END OF PHASE CURVE (HORIZONTAL CUT / FIXED g_x) #####

    ##### PHASE CURVE (VERTICAL CUT / FIXED J_1) #####
    S_J1 = 0.0  # 0.075
    S_m = x2index(S_J1, abs_J1_vec)
    abs_J1_m = abs_J1_vec[S_m]
    @show abs_J1_m

    # expect_list = ["O_even", "g1", "κ2"]  # MI-CSF
    # expect_list = ["g2", "δN", "S_kπ"]  # DW-PSF
    # expect_list = ["κ2", "g1", "O_odd", "δN", "S_kπ"]  # DW-CSF
    # expect_list = ["κ2", "g1", "g2", "O_odd"]  # PSF-CSF
    # expect_list = ["O_odd", "O_even", "δN", "S_kπ"]  # MI-DW
    expect_list = ["O_even", "O_odd", "g1", "g2", "κ2", "δN", "S_kπ"]  # All
    N_expect = length(expect_list)

    expect_vec2 = []
    for (i, expect_name) in enumerate(expect_list)
        expect_vec = []
        for n in 1:N_gx
            gx_n = gx_vec[n]

            expectation_values = npzread("$(save_path)/$(file_names[findfirst(contains("m=$(S_m)!n=$(n)"), file_names)])")
            # if expect_name == "g2"
            #     g2_matrix = expectation_values["g2_matrix"]
            #     expect_val = sum(abs.(g2_matrix[100, 300:400]))
            #     # expect_val = sum(abs.(g2_matrix[10, 100]))
            # else
            #     expect_val = expectation_values[expect_name]
            # end # if
            expect_val = expectation_values[expect_name]
            # if expect_name == "ni_nj"
            #     i_start, i_end, ni_nj_matrix = expectation_values["i_start"], expectation_values["i_end"], expectation_values["ni_nj_matrix"]
            #     expect_val = ni_nj_matrix[i_start, i_end]
            # else
            #     expect_val = abs(expectation_values[expect_name])
            # end # if

            push!(expect_vec, expect_val)
        end # for

        push!(expect_vec2, expect_vec)
    end # for

    plotN_ys_vs_x(gx_vec, expect_vec2, @__DIR__;
      ax_title=raw"$|J_1|$ = "*"$(round(abs_J1_m, digits=4))",
      x_lims_vec=[[gx_vec[end], gx_vec[1]] for i in 1:N_expect],
      x_label_vec=vcat(["" for i in 1:(N_expect-1)], [raw"$g_x/g_0$"]),
      y_label_vec=expect_list, marker="o",
      saveDirectly=false, file_name="Expect_vs_gx")
    ##### END OF PHASE CURVE (VERTICAL CUT / FIXED J_1) #####

    # ##### SINGLE SELECTED POINT #####
    # S_J1 = 0.0
    # S_gx = -4.2

    # S_m = x2index(S_J1, abs_J1_vec)
    # S_n = x2index(S_gx, gx_vec)

    # abs_J1_m = abs_J1_vec[S_m]
    # gx_n = gx_vec[S_n]

    # expectation_values = npzread("$(save_path)/$(file_names[findfirst(contains("m=$(S_m)!n=$(S_n)"), file_names)])")
    # g1, g2, δN, S_kπ, O_even, O_odd, κ2, M_k0, M_kπ, prob_Q_mod_bulk = expectation_values["g1"], expectation_values["g2"], expectation_values["δN"], expectation_values["S_kπ"], expectation_values["O_even"], expectation_values["O_odd"], expectation_values["κ2"], expectation_values["M_k0"], expectation_values["M_kπ"], expectation_values["prob_Q_mod_bulk"]
    # n_arr, S_k_vec, g1_vec, g2_vec, g1_matrix, g2_matrix, M_k_vec = expectation_values["n_arr"], expectation_values["S_k_vec"], expectation_values["g1_vec"], expectation_values["g2_vec"], expectation_values["g1_matrix"], expectation_values["g2_matrix"], expectation_values["M_k_vec"]

    # @show abs_J1_m
    # @show gx_n
    # @show g1
    # @show δN
    # @show g1
    # @show g1_vec[1:10]
    # @show g1_vec[N_lat÷2]
    # @show g1_vec[N_lat÷2+1]
    # @show g1_vec[N_lat÷2:(N_lat÷2+10)]
    # @show g2
    # @show g2_vec[1:10]
    # @show g2_vec[N_lat÷2]
    # @show g2_vec[N_lat÷2+1]
    # @show g2_vec[N_lat÷2:(N_lat÷2+10)]
    # @show κ2
    # @show S_kπ
    # @show M_k0
    # @show M_kπ
    # @show prob_Q_mod_bulk
    # @show O_odd
    # @show O_even

    # # FITTING CORRELATIONS #
    # # # g1_fit_params, g1_fit_resid, g1_fit_vec = calc_fit_1D_MC(ln_power_law_fit, fit_range, log.(abs.(g1_matrix[idx_start, idx_start .+ fit_range])), lower, upper; N_p0=100)
    # # g1_fit_params, g1_fit_resid, g1_fit_vec = calc_fit_1D_MC(power_law_fit, fit_range, abs.(g1_matrix[idx_start, idx_start .+ fit_range]), lower, upper; N_p0=100)
    # # α_g1 = g1_fit_params[2]
    # # @show α_g1
    # # @show sum(abs2.(g1_fit_resid))

    # # g2_fit_params, g2_fit_resid, g2_fit_vec = calc_fit_1D(ln_power_law_fit, fit_range, log.(abs.(g2_matrix[idx_start, idx_start .+ fit_range])), [1.0, 1.0])
    # # g2_fit_params, g2_fit_resid, g2_fit_vec = calc_fit_1D_MC(ln_power_law_fit, fit_range, log.(abs.(g2_matrix[idx_start, idx_start .+ fit_range])), lower, upper; N_p0=100)
    # # g2_fit_params, g2_fit_resid, g2_fit_vec = calc_fit_1D_MC(power_law_fit, fit_range, abs.(g2_matrix[idx_start, idx_start .+ fit_range]), lower, upper; N_p0=100)
    # g2_fit_params, g2_fit_resid, g2_fit_vec = calc_fit_1D(ln_exp_law_fit, fit_range, log.(abs.(g2_matrix[idx_start, idx_start .+ fit_range])), [1.0, 1.0])
    # # g2_fit_params, g2_fit_resid, g2_fit_vec = calc_fit_1D_MC(ln_exp_law_fit, fit_range, log.(abs.(g2_matrix[idx_start, idx_start .+ fit_range])), exp_lower, exp_upper; N_p0=100)
    # # g2_fit_params, g2_fit_resid, g2_fit_vec = calc_fit_1D_MC(exp_law_fit, fit_range, abs.(g2_matrix[idx_start, idx_start .+ fit_range]), exp_lower, exp_upper; N_p0=100)
    # α_g2 = g2_fit_params[2]
    # @show α_g2
    # @show sum(abs2.(g2_fit_resid))
    # # END OF FITTING CORRELATIONS #

    # # ψ_samples = expectation_values["ψ_samples"]
    # # println("********** SAMPLES **********")
    # # display([ψ_samples[i, :] for i in 1:size(ψ_samples, 1)])
    # # println("********** END OF SAMPLES **********")
    # # display([ψ_samples[i, (N_lat÷2-15):(N_lat÷2+15)] for i in 1:size(ψ_samples, 1)])

    # ### PLOTTING ###
    # # plotN_expectation_values_v1(N_lat, k_arr, n_arr, S_k_vec, g1_vec, g2_vec, g1_matrix, M_k_vec)
    # plotN_expectation_values_v1(N_lat, k_arr, n_arr, S_k_vec, g1_vec, abs.(g2_vec), g1_matrix, M_k_vec)

    # # ax_title = raw"$g_1(x) = $"*"$(round(g1_fit_params[1], digits=4)) "*raw"$x$"*"^(-$(round(g1_fit_params[2], digits=4)))"
    # # # plot_ys_vs_x(fit_range, [abs.(g1_matrix[idx_start, idx_start .+ fit_range]), exp.(g1_fit_vec)], @__DIR__; x_label="i-j", y_label=raw"$g_1(i-j)$", x_scale="log", y_scale="log", ax_title=ax_title, color_vec=["blue", "red"], saveDirectly=false, file_name="g1_vs_x_reg_and_fit")
    # # plot_ys_vs_x(fit_range, [abs.(g1_matrix[idx_start, idx_start .+ fit_range]), g1_fit_vec], @__DIR__; x_label="i-j", y_label=raw"$g_1(i-j)$", x_scale="log", y_scale="log", ax_title=ax_title, color_vec=["blue", "red"], saveDirectly=false, file_name="g1_vs_x_reg_and_fit")

    # # ax_title = raw"$g_2(x) = $"*"$(round(g2_fit_params[1], digits=4)) "*raw"$x$"*"^(-$(round(g2_fit_params[2], digits=4)))"
    # ax_title = raw"$g_2(x) = $"*"$(round(exp(g2_fit_params[1]), digits=4)) "*raw"exp(-$x$"*"/$(round(g2_fit_params[2], digits=4)))"
    # plot_ys_vs_x(fit_range, [abs.(g2_matrix[idx_start, idx_start .+ fit_range]), exp.(g2_fit_vec)], @__DIR__; x_label="i-j", y_label=raw"$g_2(i-j)$", x_scale="log", y_scale="log", ax_title=ax_title, color_vec=["blue", "red"], saveDirectly=false, file_name="g2_vs_x_reg_and_fit")
    # # plot_ys_vs_x(fit_range, [abs.(g2_matrix[idx_start, idx_start .+ fit_range]), g2_fit_vec], @__DIR__; x_label="i-j", y_label=raw"$g_2(i-j)$", x_scale="log", y_scale="log", ax_title=ax_title, color_vec=["blue", "red"], saveDirectly=false, file_name="g2_vs_x_reg_and_fit")
    # ### END OF PLOTTING ###
    # ##### END OF SINGLE SELECTED POINT #####
end # function

# TODO Check if energy selection correct
function routine_multiple_read_phase_diagram_J1_gx_v2(N_lat, N_part, boson_dim, conserveParticleNumber, μ, N_sweeps, max_dim, cutoff, noise)
    save_path_list = [
        "$(@__DIR__)/Selected NPY/Phase_Diagram_J1_gx, 2025-11-27/Expectation_Values_Vec",
        "$(@__DIR__)/Selected NPY/Phase_Diagram_J1_gx, 2026-01-18, 1/Expectation_Values_Vec",
        "$(@__DIR__)/Selected NPY/Phase_Diagram_J1_gx, 2026-01-18, 2/Expectation_Values_Vec"]  # N_lat = 201
    # save_path_list = [
    #     "$(@__DIR__)/Selected NPY/Phase_Diagram_J1_gx, 2025-12-13/Expectation_Values_Vec",
    #     "$(@__DIR__)/Selected NPY/Phase_Diagram_J1_gx, 2026-01-16, 1/Expectation_Values_Vec",
    #     "$(@__DIR__)/Selected NPY/Phase_Diagram_J1_gx, 2026-01-16, 2/Expectation_Values_Vec",
    #     "$(@__DIR__)/Selected NPY/Phase_Diagram_J1_gx, Repeat, 2026-01-19, 1/Expectation_Values_Vec",
    #     "$(@__DIR__)/Selected NPY/Phase_Diagram_J1_gx, chi=300, 2026-01-19, 1/Expectation_Values_Vec",
    #     "$(@__DIR__)/Selected NPY/Phase_Diagram_J1_gx, chi=100, 2026-01-23, 1/Expectation_Values_Vec",
    #     "$(@__DIR__)/Selected NPY/Phase_Diagram_J1_gx, chi=200, 2026-01-24, 1/Expectation_Values_Vec",
    #     "$(@__DIR__)/Selected NPY/Phase_Diagram_J1_gx, chi=300, 2026-01-25, 1/Expectation_Values_Vec",
    #     "$(@__DIR__)/Selected NPY/Phase_Diagram_J1_gx, chi=500, 2026-01-29/Expectation_Values_Vec",
    # ]  # N_lat = 601

    N_files = length(save_path_list)
    save_path = save_path_list[1]
    file_names = readdir(save_path)

    ##### PHASE DIAGRAM #####
    expectation_values = npzread("$(save_path)/$(file_names[findfirst(contains("m=1!n=1"), file_names)])")
    abs_J1_vec = expectation_values["abs_J1_vec"]
    gx_vec = expectation_values["gx_vec"]
    # N_lat = expectation_values["N_lat"]
    N_J1 = length(abs_J1_vec)
    N_gx = length(gx_vec)

    k_arr = LinRange(0.0, 2π, N_lat+1)[1:N_lat]

    # FITTING PARAMETERS #
    idx_start = 50  # 200
    fit_range = 50:100
    lower = [-100.0, 0.0]
    upper = [100.0, 10.0]
    exp_lower = [-100.0, 1.0]
    exp_upper = [100.0, 10000.0]
    # END OF FITTING PARAMETERS #

    g1_arr = zeros(Float64, N_J1, N_gx)
    g2_arr = zeros(Float64, N_J1, N_gx)
    δN_arr = zeros(Float64, N_J1, N_gx)
    S_kπ_arr = zeros(Float64, N_J1, N_gx)
    O_even_arr = zeros(Float64, N_J1, N_gx)
    O_odd_arr = zeros(Float64, N_J1, N_gx)
    κ2_arr = zeros(Float64, N_J1, N_gx)
    M_k0_arr = zeros(Float64, N_J1, N_gx)
    M_kπ_arr = zeros(Float64, N_J1, N_gx)
    c_arr = zeros(Float64, N_J1, N_gx)
    # kmax_M_arr = zeros(Float64, N_J1, N_gx)
    # M_kmax_arr = zeros(Float64, N_J1, N_gx)
    prob_Q_arr = zeros(Float64, N_J1, N_gx)

    # for n in 1:N_J1
    for m in 1:N_J1
        for n in 1:N_gx
            abs_J1_m = abs_J1_vec[m]
            gx_n = gx_vec[n]

            idx_best = 0
            E_best = 9e9
            E_vec = []
            for (idx, save_path) in enumerate(save_path_list)
                file_names = readdir(save_path)
                expectation_values = npzread("$(save_path)/$(file_names[findfirst(contains("m=$(m)!n=$(n)"), file_names)])")

                E_idx = expectation_values["expect_E"]
                if expectation_values["expect_E"] < E_best
                    idx_best = idx
                    E_best = E_idx
                end # if

                push!(E_vec, E_idx)
            end # for
            save_path = save_path_list[idx_best]
            file_names = readdir(save_path)
            expectation_values = npzread("$(save_path)/$(file_names[findfirst(contains("m=$(m)!n=$(n)"), file_names)])")
            @show save_path
            σ_E = std_dev(E_vec)
            @show σ_E

            g1, g2, bulk_range, g1_matrix, g2_matrix, δN, S_kπ, O_even, O_odd, κ2, M_k0, M_kπ, M_k_vec, entropy_S_vec, prob_Q_bulk, prob_Q_mod_bulk = expectation_values["g1"], expectation_values["g2"], expectation_values["bulk_range"], expectation_values["g1_matrix"], expectation_values["g2_matrix"], expectation_values["δN"], expectation_values["S_kπ"], expectation_values["O_even"], expectation_values["O_odd"], expectation_values["κ2"], expectation_values["M_k0"], expectation_values["M_kπ"], expectation_values["M_k_vec"], expectation_values["entropy_S_vec"], expectation_values["prob_Q_bulk"], expectation_values["prob_Q_mod_bulk"]
            g1_arr[m, n] = g1
            g2_arr[m, n] = g2
            # g2_arr[m, n] = sum(abs.(g2_matrix[bulk_range[1], (bulk_range[1]+100):(bulk_range[1]+200)]))
            δN_arr[m, n] = δN
            S_kπ_arr[m, n] = S_kπ
            O_even_arr[m, n] = O_even
            O_odd_arr[m, n] = O_odd
            κ2_arr[m, n] = κ2
            M_k0_arr[m, n] = M_k0
            M_kπ_arr[m, n] = M_kπ

            c_arr[m, n], _ = DMRG.calc_central_charge(entropy_S_vec, N_lat)
            # kmax_M_arr[m, n] = k_arr[argmax(real.(M_k_vec[1:(N_lat÷2)]))]
            # M_kmax_arr[m, n] = maximum(real.(M_k_vec))
            # prob_Q_arr[m, n] = prob_Q_bulk
            prob_Q_arr[m, n] = prob_Q_mod_bulk

            println("******* $(m)/$(N_J1) && $(n)/$(N_gx) DONE *******")
        end # for
    end # for

    ### FITTING CORRELATIONS ###
    # α_g1_arr = zeros(Float64, N_J1, N_gx)
    # res_g1_arr = zeros(Float64, N_J1, N_gx)
    # α_g2_arr = zeros(Float64, N_J1, N_gx)
    # res_g2_arr = zeros(Float64, N_J1, N_gx)
    # exp_ξ_g2_arr = zeros(Float64, N_J1, N_gx)
    # exp_res_g2_arr = zeros(Float64, N_J1, N_gx)
    # for m in 1:N_J1
    #     for n in 1:N_gx
    #         abs_J1_m = abs_J1_vec[m]
    #         gx_n = gx_vec[n]

    #         idx_best = 0
    #         E_best = 9e9
    #         E_vec = []
    #         for (idx, save_path) in enumerate(save_path_list)
    #             file_names = readdir(save_path)
    #             expectation_values = npzread("$(save_path)/$(file_names[findfirst(contains("m=$(m)!n=$(n)"), file_names)])")

    #             E_idx = expectation_values["expect_E"]
    #             if expectation_values["expect_E"] < E_best
    #                 idx_best = idx
    #                 E_best = E_idx
    #             end # if

    #             push!(E_vec, E_idx)
    #         end # for
    #         save_path = save_path_list[idx_best]
    #         file_names = readdir(save_path)
    #         expectation_values = npzread("$(save_path)/$(file_names[findfirst(contains("m=$(m)!n=$(n)"), file_names)])")
    #         @show save_path
    #         σ_E = std_dev(E_vec)
    #         @show σ_E

    #         g1, g2, bulk_range, g1_matrix, g2_matrix, δN, S_kπ, O_even, O_odd, κ2, M_k0, M_kπ, M_k_vec, entropy_S_vec, prob_Q_mod_bulk = expectation_values["g1"], expectation_values["g2"], expectation_values["bulk_range"], expectation_values["g1_matrix"], expectation_values["g2_matrix"], expectation_values["δN"], expectation_values["S_kπ"], expectation_values["O_even"], expectation_values["O_odd"], expectation_values["κ2"], expectation_values["M_k0"], expectation_values["M_kπ"], expectation_values["M_k_vec"], expectation_values["entropy_S_vec"], expectation_values["prob_Q_mod_bulk"]

    #         # NOTE: Here fit is done for abs correlations, not correlations themselves.
    #         g1_fit_params, g1_fit_resid, g1_fit_vec = calc_fit_1D(ln_power_law_fit, fit_range, log.(abs.(g1_matrix[idx_start, idx_start .+ fit_range])), [1.0, 1.0])
    #         # g1_fit_params, g1_fit_resid, g1_fit_vec = calc_fit_1D_MC(ln_power_law_fit, fit_range, log.(abs.(g1_matrix[idx_start, idx_start .+ fit_range])), lower, upper; N_p0=1)
    #         # g1_fit_params, g1_fit_resid, g1_fit_vec = calc_fit_1D_MC(power_law_fit, fit_range, abs.(g1_matrix[idx_start, idx_start .+ fit_range]), lower, upper; N_p0=1)
    #         α_g1_arr[m, n] = g1_fit_params[2]
    #         res_g1_arr[m, n] = sum(abs2.(g1_fit_resid))
    #         g2_fit_params, g2_fit_resid, g2_fit_vec = calc_fit_1D(ln_power_law_fit, fit_range, log.(abs.(g2_matrix[idx_start, idx_start .+ fit_range])), [1.0, 1.0])
    #         # g2_fit_params, g2_fit_resid, g2_fit_vec = calc_fit_1D_MC(ln_power_law_fit, fit_range, log.(abs.(g2_matrix[idx_start, idx_start .+ fit_range])), lower, upper; N_p0=1)
    #         # g2_fit_params, g2_fit_resid, g2_fit_vec = calc_fit_1D_MC(power_law_fit, fit_range, abs.(g2_matrix[idx_start, idx_start .+ fit_range]), lower, upper; N_p0=1)
    #         α_g2_arr[m, n] = g2_fit_params[2]
    #         res_g2_arr[m, n] = sum(abs2.(g2_fit_resid))

    #         # exp_g2_fit_params, exp_g2_fit_resid, exp_g2_fit_vec = calc_fit_1D_MC(ln_exp_law_fit, fit_range, log.(abs.(g2_matrix[idx_start, idx_start .+ fit_range])), exp_lower, exp_upper; N_p0=1)
    #         exp_g2_fit_params, exp_g2_fit_resid, exp_g2_fit_vec = calc_fit_1D(ln_exp_law_fit, fit_range, log.(abs.(g2_matrix[idx_start, idx_start .+ fit_range])), [1.0, 1.0])
    #         exp_ξ_g2_arr[m, n] = exp_g2_fit_params[2]
    #         exp_res_g2_arr[m, n] = sum(abs2.(exp_g2_fit_resid))

    #         println("******* $(m)/$(N_J1) && $(n)/$(N_gx) DONE *******")
    #     end # for
    # end # for
    # ### END OF FITTING CORRELATIONS ###

    if Threads.threadid() == 1
        ##### PLOTTING #####
        plot_u_vs_xy(abs_J1_vec, gx_vec, prob_Q_arr'; x_name=raw"$|J_1|/g_0$", y_name=raw"$g_x/g_0$", u_name=raw"$P_Q$")

        plotN_order_parameters_vs_J1_gx_v2(abs_J1_vec, gx_vec, g1_arr, g2_arr, δN_arr, S_kπ_arr, O_even_arr, O_odd_arr, κ2_arr, M_k0_arr, M_kπ_arr, c_arr, prob_Q_arr)
        # plotN_fit_res_vs_J1_gx(abs_J1_vec, gx_vec, α_g1_arr, res_g1_arr, α_g2_arr, res_g2_arr, exp_ξ_g2_arr, exp_res_g2_arr; vlim=[lower[2], upper[2]])
        ##### END OF PLOTTING #####
    end # if
    ##### END OF PHASE DIAGRAM #####

    ########## PROPER PHASE DIAGRAM ##########
    ##### PHASE SEPARATING CURVES #####
    phase_name_arr = Array{String}(undef, N_J1, N_gx)
    for m in 1:N_J1
        for n in 1:N_gx
            idx_best = 0
            E_best = 9e9
            for (idx, save_path) in enumerate(save_path_list)
                file_names = readdir(save_path)
                expectation_values = npzread("$(save_path)/$(file_names[findfirst(contains("m=$(m)!n=$(n)"), file_names)])")

                E_idx = expectation_values["expect_E"]
                if expectation_values["expect_E"] < E_best
                    idx_best = idx
                    E_best = E_idx
                end # if
            end # for
            save_path = save_path_list[idx_best]
            file_names = readdir(save_path)
            expectation_values = npzread("$(save_path)/$(file_names[findfirst(contains("m=$(m)!n=$(n)"), file_names)])")

            phase_name_arr[m, n] = classify_qmb_phase_v1(expectation_values)
        end # for
    end # for

    CSF_curve_vec2 = calc_phase_curve_v1("CSF", abs_J1_vec, gx_vec, phase_name_arr)
    MI_curve_vec2 = calc_phase_curve_v1("MI", abs_J1_vec, gx_vec, phase_name_arr)
    DW_curve_vec2 = calc_phase_curve_v1("DW", abs_J1_vec, gx_vec, phase_name_arr)
    PSF_curve_vec2 = calc_phase_curve_v1("PSF", abs_J1_vec, gx_vec, phase_name_arr)

    phase_diagram_curves = Dict{String, Any}()
    merge!(phase_diagram_curves, Dict("abs_J1_vec" => abs_J1_vec))
    merge!(phase_diagram_curves, Dict("gx_vec" => gx_vec))
    merge!(phase_diagram_curves, Dict("CSF_J1_vec" => CSF_curve_vec2[1]))
    merge!(phase_diagram_curves, Dict("CSF_gx_top_vec" => CSF_curve_vec2[2]))
    merge!(phase_diagram_curves, Dict("CSF_gx_bottom_vec" => CSF_curve_vec2[3]))
    merge!(phase_diagram_curves, Dict("MI_J1_vec" => MI_curve_vec2[1]))
    merge!(phase_diagram_curves, Dict("MI_gx_top_vec" => MI_curve_vec2[2]))
    merge!(phase_diagram_curves, Dict("MI_gx_bottom_vec" => MI_curve_vec2[3]))
    merge!(phase_diagram_curves, Dict("DW_J1_vec" => DW_curve_vec2[1]))
    merge!(phase_diagram_curves, Dict("DW_gx_top_vec" => DW_curve_vec2[2]))
    merge!(phase_diagram_curves, Dict("DW_gx_bottom_vec" => DW_curve_vec2[3]))
    merge!(phase_diagram_curves, Dict("PSF_J1_vec" => PSF_curve_vec2[1]))
    merge!(phase_diagram_curves, Dict("PSF_gx_top_vec" => PSF_curve_vec2[2]))
    merge!(phase_diagram_curves, Dict("PSF_gx_bottom_vec" => PSF_curve_vec2[3]))

    # ### SAVING RESULTS ###
    # save_path = "$(@__DIR__)/NPY/$(Dates.today())/Phase_Diagram_Curves_J1_gx"
    # date_string = """$(Dates.format(Dates.now(), "Y-mm-dd!HH.MM.SS"))"""
    # mkpath(save_path)
    # npzwrite("$(save_path)/Phase_Diagram_Curves_J1_gx!$(date_string).npz", phase_diagram_curves)
    # ### END OF SAVING RESULTS ###

    plot_phase_diagram_vs_J1_gx(abs_J1_vec, gx_vec, CSF_curve_vec2, MI_curve_vec2, DW_curve_vec2, PSF_curve_vec2)
    # ##### END OF PHASE SEPARATING CURVES #####
    # # END OF PROPER PHASE DIAGRAM ##########

    # ##### PHASE CURVE (HORIZONTAL CUT / FIXED g_x) #####
    # S_gx = -0.0
    # S_n = x2index(S_gx, gx_vec)
    # gx_n = gx_vec[S_n]
    # @show gx_n

    # # expect_list = ["O_even", "g1", "κ2"]  # MI-CSF
    # # expect_list = ["g2", "δN", "S_kπ"]  # DW-PSF
    # # expect_list = ["κ2", "g1", "O_odd", "δN", "S_kπ"]  # DW-CSF
    # # expect_list = ["κ2", "g1", "g2", "O_odd"]  # PSF-CSF
    # # expect_list = ["O_odd", "O_even", "δN", "S_kπ"]  # MI-DW
    # expect_list = ["O_even", "O_odd", "g1", "g2", "κ2", "δN", "S_kπ"]  # All
    # N_expect = length(expect_list)

    # expect_vec2 = []
    # for (i, expect_name) in enumerate(expect_list)
    #     expect_vec = []
    #     for m in 1:N_J1
    #         abs_J1_m = abs_J1_vec[m]

    #         idx_best = 0
    #         E_best = 9e9
    #         for (idx, save_path) in enumerate(save_path_list)
    #             file_names = readdir(save_path)
    #             expectation_values = npzread("$(save_path)/$(file_names[findfirst(contains("m=$(m)!n=$(S_n)"), file_names)])")

    #             E_idx = expectation_values["expect_E"]
    #             if expectation_values["expect_E"] < E_best
    #                 idx_best = idx
    #                 E_best = E_idx
    #             end # if
    #         end # for
    #         save_path = save_path_list[idx_best]
    #         file_names = readdir(save_path)
    #         expectation_values = npzread("$(save_path)/$(file_names[findfirst(contains("m=$(m)!n=$(S_n)"), file_names)])")

    #         expect_val = expectation_values[expect_name]

    #         push!(expect_vec, expect_val)
    #     end # for

    #     push!(expect_vec2, expect_vec)
    # end # for

    # plotN_ys_vs_x(abs_J1_vec, expect_vec2, @__DIR__;
    #   ax_title=raw"$g_x$ = "*"$(round(gx_n, digits=4))",
    #   x_label_vec=vcat(["" for i in 1:(N_expect-1)], [raw"$|J_1|/g_0$"]),
    #   y_label_vec=expect_list, marker="o",
    #   saveDirectly=false, file_name="Expect_vs_J1")
    # ##### END OF PHASE CURVE (HORIZONTAL CUT / FIXED g_x) #####

    ##### PHASE CURVE (VERTICAL CUT / FIXED J_1) #####
    S_J1 = 0.07  # 0.075
    S_m = x2index(S_J1, abs_J1_vec)
    abs_J1_m = abs_J1_vec[S_m]
    @show abs_J1_m

    # expect_list = ["O_even", "g1", "κ2"]  # MI-CSF
    # expect_list = ["g2", "δN", "S_kπ"]  # DW-PSF
    # expect_list = ["κ2", "g1", "O_odd", "δN", "S_kπ"]  # DW-CSF
    # expect_list = ["κ2", "g1", "g2", "O_odd"]  # PSF-CSF
    # expect_list = ["O_odd", "O_even", "δN", "S_kπ"]  # MI-DW
    expect_list = ["O_even", "O_odd", "g1", "g2", "κ2", "δN", "S_kπ"]  # All
    N_expect = length(expect_list)

    O_even_vec = zeros(Float64, N_gx)
    g1_vec = zeros(Float64, N_gx)
    g2_vec = zeros(Float64, N_gx)
    κ2_vec = zeros(Float64, N_gx)
    S_kπ_vec = zeros(Float64, N_gx)

    expect_vec2 = []
    for (i, expect_name) in enumerate(expect_list)
        expect_vec = []
        for n in 1:N_gx
            gx_n = gx_vec[n]

            idx_best = 0
            E_best = 9e9
            for (idx, save_path) in enumerate(save_path_list)
                file_names = readdir(save_path)
                expectation_values = npzread("$(save_path)/$(file_names[findfirst(contains("m=$(S_m)!n=$(n)"), file_names)])")

                E_idx = expectation_values["expect_E"]
                if expectation_values["expect_E"] < E_best
                    idx_best = idx
                    E_best = E_idx
                end # if
            end # for
            save_path = save_path_list[idx_best]
            file_names = readdir(save_path)
            expectation_values = npzread("$(save_path)/$(file_names[findfirst(contains("m=$(S_m)!n=$(n)"), file_names)])")
            if i == 1
                @show save_path
            end # if

            # if expect_name == "g2"
            #     bulk_range = expectation_values["bulk_range"]
            #     g2_matrix = expectation_values["g2_matrix"]
            #     # @show abs.(g2_matrix[bulk_range[1], (bulk_range[end]-10):bulk_range[end]])
            #     expect_val = sum(abs.(g2_matrix[bulk_range[1], (bulk_range[end]-10):bulk_range[end]]))
            #     # expect_val = sum(abs.(g2_matrix[50, 100:150]))
            #     # expect_val = sum(abs.(g2_matrix[50, 250]))
            # else
            #     expect_val = expectation_values[expect_name]
            # end # if
            expect_val = expectation_values[expect_name]
            # if expect_name == "ni_nj"
            #     i_start, i_end, ni_nj_matrix = expectation_values["i_start"], expectation_values["i_end"], expectation_values["ni_nj_matrix"]
            #     expect_val = ni_nj_matrix[i_start, i_end]
            # else
            #     expect_val = abs(expectation_values[expect_name])
            # end # if

            if expect_name == "O_even"
                O_even_vec[n] = expectation_values[expect_name]
            elseif expect_name == "g1"
                g1_vec[n] = expectation_values[expect_name]
            elseif expect_name == "g2"
                g2_vec[n] = -expectation_values[expect_name]
                # g2_vec[n] = sum(abs.(expectation_values["g2_matrix"][100, 300:400]))
            elseif expect_name == "κ2"
                κ2_vec[n] = expectation_values[expect_name]
            elseif expect_name == "S_kπ"
                S_kπ_vec[n] = expectation_values[expect_name]
            end # if

            push!(expect_vec, expect_val)
        end # for

        push!(expect_vec2, expect_vec)
    end # for

    # J1_cut_expectation_values = Dict{String, Any}()
    # merge!(J1_cut_expectation_values, Dict("abs_J1_vec" => abs_J1_vec))
    # merge!(J1_cut_expectation_values, Dict("gx_vec" => gx_vec))
    # merge!(J1_cut_expectation_values, Dict("S_J1" => S_J1))
    # merge!(J1_cut_expectation_values, Dict("S_m" => S_m))
    # merge!(J1_cut_expectation_values, Dict("abs_J1_m" => abs_J1_m))
    # merge!(J1_cut_expectation_values, Dict("O_even_vec" => O_even_vec))
    # merge!(J1_cut_expectation_values, Dict("g1_vec" => g2_vec))
    # merge!(J1_cut_expectation_values, Dict("g2_vec" => g2_vec))
    # merge!(J1_cut_expectation_values, Dict("κ2_vec" => κ2_vec))
    # merge!(J1_cut_expectation_values, Dict("S_kπ_vec" => S_kπ_vec))

    # ### SAVING RESULTS ###
    # save_path = "$(@__DIR__)/NPY/$(Dates.today())/J1_Cut_Expectation_Values"
    # date_string = """$(Dates.format(Dates.now(), "Y-mm-dd!HH.MM.SS"))"""
    # mkpath(save_path)
    # npzwrite("$(save_path)/J1_Cut_Expectation_Values!$(date_string).npz", J1_cut_expectation_values)
    # ### END OF SAVING RESULTS ###

    plotN_ys_vs_x(gx_vec, expect_vec2, @__DIR__;
      ax_title=raw"$|J_1|$ = "*"$(round(abs_J1_m, digits=4))",
      x_lims_vec=[[gx_vec[end], gx_vec[1]] for i in 1:N_expect],
      x_label_vec=vcat(["" for i in 1:(N_expect-1)], [raw"$g_x/g_0$"]),
      y_label_vec=expect_list, marker="o",
      saveDirectly=false, file_name="Expect_vs_gx")
    ##### END OF PHASE CURVE (VERTICAL CUT / FIXED J_1) #####

    ##### SINGLE SELECTED POINT #####
    S_J1 = 0.05
    S_gx = -3.0

    S_m = x2index(S_J1, abs_J1_vec)
    S_n = x2index(S_gx, gx_vec)

    abs_J1_m = abs_J1_vec[S_m]
    gx_n = gx_vec[S_n]

    idx_best = 0
    E_best = 9e9
    for (idx, save_path) in enumerate(save_path_list)
        file_names = readdir(save_path)
        expectation_values = npzread("$(save_path)/$(file_names[findfirst(contains("m=$(S_m)!n=$(S_n)"), file_names)])")

        E_idx = expectation_values["expect_E"]
        if expectation_values["expect_E"] < E_best
            idx_best = idx
            E_best = E_idx
        end # if
    end # for
    save_path = save_path_list[idx_best]
    file_names = readdir(save_path)
    expectation_values = npzread("$(save_path)/$(file_names[findfirst(contains("m=$(S_m)!n=$(S_n)"), file_names)])")

    g1, g2, δN, S_kπ, O_even, O_odd, κ2, M_k0, M_kπ, prob_Q_mod_bulk = expectation_values["g1"], expectation_values["g2"], expectation_values["δN"], expectation_values["S_kπ"], expectation_values["O_even"], expectation_values["O_odd"], expectation_values["κ2"], expectation_values["M_k0"], expectation_values["M_kπ"], expectation_values["prob_Q_mod_bulk"]
    n_arr, S_k_vec, g1_vec, g2_vec, g1_matrix, g2_matrix, M_k_vec = expectation_values["n_arr"], expectation_values["S_k_vec"], expectation_values["g1_vec"], expectation_values["g2_vec"], expectation_values["g1_matrix"], expectation_values["g2_matrix"], expectation_values["M_k_vec"]

    @show abs_J1_m
    @show gx_n
    @show g1
    @show δN
    @show g1
    @show g1_vec[1:10]
    @show g1_vec[N_lat÷2]
    @show g1_vec[N_lat÷2+1]
    @show g1_vec[N_lat÷2:(N_lat÷2+10)]
    @show g2
    @show g2_vec[1:10]
    @show g2_vec[N_lat÷2]
    @show g2_vec[N_lat÷2+1]
    @show g2_vec[N_lat÷2:(N_lat÷2+10)]
    @show κ2
    @show S_kπ
    @show M_k0
    @show M_kπ
    @show prob_Q_mod_bulk
    @show O_odd
    @show O_even

    # FITTING CORRELATIONS #
    # # g1_fit_params, g1_fit_resid, g1_fit_vec = calc_fit_1D_MC(ln_power_law_fit, fit_range, log.(abs.(g1_matrix[idx_start, idx_start .+ fit_range])), lower, upper; N_p0=100)
    # g1_fit_params, g1_fit_resid, g1_fit_vec = calc_fit_1D_MC(power_law_fit, fit_range, abs.(g1_matrix[idx_start, idx_start .+ fit_range]), lower, upper; N_p0=100)
    # α_g1 = g1_fit_params[2]
    # @show α_g1
    # @show sum(abs2.(g1_fit_resid))

    g2_fit_params, g2_fit_resid, g2_fit_vec = calc_fit_1D(ln_power_law_fit, fit_range, log.(abs.(g2_matrix[idx_start, idx_start .+ fit_range])), [1.0, 1.0])
    # g2_fit_params, g2_fit_resid, g2_fit_vec = calc_fit_1D_MC(ln_power_law_fit, fit_range, log.(abs.(g2_matrix[idx_start, idx_start .+ fit_range])), lower, upper; N_p0=100)
    # g2_fit_params, g2_fit_resid, g2_fit_vec = calc_fit_1D_MC(power_law_fit, fit_range, abs.(g2_matrix[idx_start, idx_start .+ fit_range]), lower, upper; N_p0=100)
    # g2_fit_params, g2_fit_resid, g2_fit_vec = calc_fit_1D(ln_exp_law_fit, fit_range, log.(abs.(g2_matrix[idx_start, idx_start .+ fit_range])), [1.0, 1.0])
    # g2_fit_params, g2_fit_resid, g2_fit_vec = calc_fit_1D_MC(ln_exp_law_fit, fit_range, log.(abs.(g2_matrix[idx_start, idx_start .+ fit_range])), exp_lower, exp_upper; N_p0=100)
    # g2_fit_params, g2_fit_resid, g2_fit_vec = calc_fit_1D_MC(exp_law_fit, fit_range, abs.(g2_matrix[idx_start, idx_start .+ fit_range]), exp_lower, exp_upper; N_p0=100)
    α_g2 = g2_fit_params[2]
    @show α_g2
    @show sum(abs2.(g2_fit_resid))
    # END OF FITTING CORRELATIONS #

    ψ_samples = expectation_values["ψ_samples"]
    println("********** SAMPLES **********")
    display([ψ_samples[i, :] for i in 1:size(ψ_samples, 1)])
    println("********** END OF SAMPLES **********")
    display([ψ_samples[i, (N_lat÷2-15):(N_lat÷2+15)] for i in 1:size(ψ_samples, 1)])

    ### PLOTTING ###
    # plotN_expectation_values_v1(N_lat, k_arr, n_arr, S_k_vec, g1_vec, g2_vec, g1_matrix, M_k_vec)
    plotN_expectation_values_v1(N_lat, k_arr, n_arr, S_k_vec, g1_vec, abs.(g2_vec), g1_matrix, M_k_vec)

    # ax_title = raw"$g_1(x) = $"*"$(round(g1_fit_params[1], digits=4)) "*raw"$x$"*"^(-$(round(g1_fit_params[2], digits=4)))"
    # # plot_ys_vs_x(fit_range, [abs.(g1_matrix[idx_start, idx_start .+ fit_range]), exp.(g1_fit_vec)], @__DIR__; x_label="i-j", y_label=raw"$g_1(i-j)$", x_scale="log", y_scale="log", ax_title=ax_title, color_vec=["blue", "red"], saveDirectly=false, file_name="g1_vs_x_reg_and_fit")
    # plot_ys_vs_x(fit_range, [abs.(g1_matrix[idx_start, idx_start .+ fit_range]), g1_fit_vec], @__DIR__; x_label="i-j", y_label=raw"$g_1(i-j)$", x_scale="log", y_scale="log", ax_title=ax_title, color_vec=["blue", "red"], saveDirectly=false, file_name="g1_vs_x_reg_and_fit")

    ax_title = raw"$g_2(x) = $"*"$(round(g2_fit_params[1], digits=4)) "*raw"$x$"*"^(-$(round(g2_fit_params[2], digits=4)))"
    # ax_title = raw"$g_2(x) = $"*"$(round(exp(g2_fit_params[1]), digits=4)) "*raw"exp(-$x$"*"/$(round(g2_fit_params[2], digits=4)))"
    plot_ys_vs_x(fit_range, [abs.(g2_matrix[idx_start, idx_start .+ fit_range]), exp.(g2_fit_vec)], @__DIR__; x_label="i-j", y_label=raw"$g_2(i-j)$", x_scale="log", y_scale="log", ax_title=ax_title, color_vec=["blue", "red"], saveDirectly=false, file_name="g2_vs_x_reg_and_fit")
    # plot_ys_vs_x(fit_range, [abs.(g2_matrix[idx_start, idx_start .+ fit_range]), g2_fit_vec], @__DIR__; x_label="i-j", y_label=raw"$g_2(i-j)$", x_scale="log", y_scale="log", ax_title=ax_title, color_vec=["blue", "red"], saveDirectly=false, file_name="g2_vs_x_reg_and_fit")
    ### END OF PLOTTING ###
    # ##### END OF SINGLE SELECTED POINT #####
end # function

function routine_read_phase_diagram_gaps_J1_gx(N_lat, N_part, boson_dim, conserveParticleNumber, μ, N_sweeps, max_dim, cutoff, noise)
    k_arr = LinRange(0.0, 2π, N_lat+1)[1:N_lat]

    # save_path = "$(@__DIR__)/Selected NPY/Phase_Diagram_Gaps_J1_gx, 2025-08-15/Phase_Diagram_Gaps_J1_gx/"
    # save_path = "$(@__DIR__)/Selected NPY/Phase_Diagram_Gaps_J1_gx, Incomplete, 2025-09-17/Phase_Diagram_Gaps_J1_gx/"  # Only N_J1=9 out of N_J1=11
    save_path = "$(@__DIR__)/Selected NPY/Phase_Diagram_Gaps_J1_gx, 2025-09-17/Phase_Diagram_Gaps_J1_gx/"
    file_names = readdir(save_path)

    ##### PHASE DIAGRAM #####
    N_J1 = 11
    N_gx = 12

    abs_J1_vec = LinRange(0.0, 1.0/sqrt(10.0), N_J1)
    gx_vec = LinRange(-10.0, 10.0, N_gx)

    δN_arr = zeros(Float64, N_J1, N_gx)
    g1_arr = zeros(Float64, N_J1, N_gx)
    g2_arr = zeros(Float64, N_J1, N_gx)
    Δ1_arr = zeros(Float64, N_J1, N_gx)
    Δ2_arr = zeros(Float64, N_J1, N_gx)
    S_k0_arr = zeros(Float64, N_J1, N_gx)
    M_k0_arr = zeros(Float64, N_J1, N_gx)
    S_kπ_arr = zeros(Float64, N_J1, N_gx)  # Assumes N_lat is even
    M_kπ_arr = zeros(Float64, N_J1, N_gx)  # Assumes N_lat is even
    integ_g1_arr = zeros(Float64, N_J1, N_gx)
    integ_g2_arr = zeros(Float64, N_J1, N_gx)
    integ_S_k_arr = zeros(Float64, N_J1, N_gx)
    integ_M_k_arr = zeros(Float64, N_J1, N_gx)
    c_arr = zeros(Float64, N_J1, N_gx)
    prob_Q_arr = zeros(Float64, N_J1, N_gx)
    avg_NN_g1_arr = zeros(Float64, N_J1, N_gx)
    avg_NN_g2_arr = zeros(Float64, N_J1, N_gx)
    avg_NN_n_arr = zeros(Float64, N_J1, N_gx)

    for m in 1:N_J1
    # Threads.@threads for m in 1:N_J1
        for n in 1:N_gx
            abs_J1_m = abs_J1_vec[m]
            gx_n = gx_vec[n]

            expectation_values = npzread("$(save_path)/$(file_names[findfirst(contains("abs_J1=$(round(abs_J1_m, digits=4))!gx=$(round(gx_n, digits=4))"), file_names)])")

            δN, g1, g2, g1_matrix, g2_matrix, S_k_vec, M_k_vec, c, Δ1, Δ2 = expectation_values["δN"], expectation_values["g1"], expectation_values["g2"], expectation_values["g1_matrix"], expectation_values["g2_matrix"], expectation_values["S_k_vec"], expectation_values["M_k_vec"], expectation_values["c"], expectation_values["Δ1"], expectation_values["Δ2"]
            δN_arr[m, n], g1_arr[m, n], g2_arr[m, n] = δN, g1, g2
            Δ1_arr[m, n] = Δ1
            Δ2_arr[m, n] = Δ2
            S_k0_arr[m, n], M_k0_arr[m, n] = real(S_k_vec[1]), real(M_k_vec[1])
            S_kπ_arr[m, n], M_kπ_arr[m, n] = real(S_k_vec[N_lat÷2+1]), real(M_k_vec[N_lat÷2+1])
            integ_g1_arr[m, n] = sum(abs.(g1_matrix)) - tr(abs.(g1_matrix))
            integ_g2_arr[m, n] = sum(abs.(g2_matrix)) - tr(abs.(g2_matrix))
            integ_S_k_arr[m, n] = sum(abs.(S_k_vec[(N_lat÷10):end]))
            integ_M_k_arr[m, n] = sum(abs.(M_k_vec[(N_lat÷10):end]))
            c_arr[m, n] = c
            prob_Q_arr[m, n] = prob_Q
            avg_NN_g1_arr[m, n] = avg_NN_g1
            avg_NN_g2_arr[m, n] = avg_NN_g2
            avg_NN_n_arr[m, n] = avg_NN_n

            # if m == 1
            #     println("******* $(n)/$(N_gx) DONE *******")
            # end # if
            println("******* $(m)/$(N_J1) && $(n)/$(N_gx) DONE *******")
        end # for
    end # for

    if Threads.threadid() == 1
        ##### PLOTTING #####
        plotN_order_parameters_gaps_vs_J1_gx(abs_J1_vec, gx_vec, δN_arr, g1_arr, g2_arr, Δ1_arr, Δ2_arr, integ_g1_arr, integ_g2_arr, integ_S_k_arr, integ_M_k_arr, S_k0_arr, M_k0_arr, S_kπ_arr, M_kπ_arr, c_arr, prob_Q_arr, avg_NN_g1_arr, avg_NN_g2_arr, avg_NN_n_arr)
        ##### END OF PLOTTING #####
    end # if
    ##### END OF PHASE DIAGRAM #####

    ##### SINGLE SELECTED POINT #####
    S_m = 1
    S_n = 6

    abs_J1_m = abs_J1_vec[S_m]
    gx_n = gx_vec[S_n]

    expectation_values = npzread("$(save_path)/$(file_names[findfirst(contains("abs_J1=$(round(abs_J1_m, digits=4))!gx=$(round(gx_n, digits=4))"), file_names)])")
    δN, n_arr, g1, g2, g1_vec, g2_vec, g1_matrix, g2_matrix, S_k_vec, M_k_vec, c, O_odd_vec, O_even_vec, Δ1, Δ2 = expectation_values["δN"], expectation_values["n_arr"], expectation_values["g1"], expectation_values["g2"], expectation_values["g1_vec"], expectation_values["g2_vec"], expectation_values["g1_matrix"], expectation_values["g2_matrix"], expectation_values["S_k_vec"], expectation_values["M_k_vec"], expectation_values["c"], expectation_values["O_odd_vec"], expectation_values["O_even_vec"], expectation_values["Δ1"], expectation_values["Δ2"]

    println("***** SINGLE SELECTED POINT *****")
    @show abs_J1_m
    @show gx_n
    @show δN
    @show g1
    @show g2
    @show c
    @show Δ1
    @show Δ2
    println("***** END OF SINGLE SELECTED POINT *****")

    plotN_expectation_values_v1(N_lat, k_arr, n_arr, S_k_vec, g1_vec, g2_vec, g1_matrix, M_k_vec, O_odd_vec, O_even_vec)
    ##### END OF SINGLE SELECTED POINT #####
end # function

function routine_read_phase_diagram_gaps_J1_gx_v2(N_lat, N_part, boson_dim, conserveParticleNumber, μ, N_sweeps, max_dim, cutoff, noise)
    # save_path = "$(@__DIR__)/Selected NPY/Phase_Diagram_Gaps_J1_gx, 2026-01-07/Expectation_Values_Vec"  # N_lat = 201
    # save_path = "$(@__DIR__)/Selected NPY/Phase_Diagram_Gaps_J1_gx, 2026-01-08/Expectation_Values_Vec"  # N_lat = 201
    # save_path = "$(@__DIR__)/Selected NPY/Phase_Diagram_Gaps_J1_gx, 2026-01-28/Expectation_Values_Vec"  # N_lat = 200
    save_path = "$(@__DIR__)/Selected NPY/Phase_Diagram_Gaps_J1_gx, 2026-02-18/Expectation_Values_Vec"  # N_lat = 200
    file_names = readdir(save_path)

    ##### PHASE DIAGRAM #####
    expectation_values = npzread("$(save_path)/$(file_names[findfirst(contains("m=1!n=1"), file_names)])")
    abs_J1_vec = expectation_values["abs_J1_vec"]
    gx_vec = expectation_values["gx_vec"]
    # N_lat = expectation_values["N_lat"]
    N_J1 = length(abs_J1_vec)
    N_gx = length(gx_vec)

    k_arr = LinRange(0.0, 2π, N_lat+1)[1:N_lat]

    # FITTING PARAMETERS #
    idx_start = 50
    fit_range = 50:100
    lower = [0.0, 0.0]
    upper = [100.0, 10.0]
    # END OF FITTING PARAMETERS #

    g1_arr = zeros(Float64, N_J1, N_gx)
    g2_arr = zeros(Float64, N_J1, N_gx)
    δN_arr = zeros(Float64, N_J1, N_gx)
    S_kπ_arr = zeros(Float64, N_J1, N_gx)
    O_even_arr = zeros(Float64, N_J1, N_gx)
    O_odd_arr = zeros(Float64, N_J1, N_gx)
    κ2_arr = zeros(Float64, N_J1, N_gx)
    Δ1_arr = zeros(Float64, N_J1, N_gx)
    Δ2_arr = zeros(Float64, N_J1, N_gx)
    Δ1_bulk_arr = zeros(Float64, N_J1, N_gx)
    Δ2_bulk_arr = zeros(Float64, N_J1, N_gx)
    # c_arr = zeros(Float64, N_J1, N_gx)
    # kmax_M_arr = zeros(Float64, N_J1, N_gx)
    # M_kmax_arr = zeros(Float64, N_J1, N_gx)
    prob_Q_arr = zeros(Float64, N_J1, N_gx)
    prob_Q_gx_J1_arr = zeros(Float64, N_gx, N_J1)

    # for n in 1:N_J1
    for m in 1:N_J1
        for n in 1:N_gx
            abs_J1_m = abs_J1_vec[m]
            gx_n = gx_vec[n]

            expectation_values = npzread("$(save_path)/$(file_names[findfirst(contains("m=$(m)!n=$(n)"), file_names)])")
            g1, g2, bulk_range, g1_matrix, g2_matrix, δN, S_kπ, O_even, O_odd, κ2, Δ1, Δ2, Δ1_bulk, Δ2_bulk, M_k0, M_kπ, M_k_vec, entropy_S_vec, prob_Q_mod_bulk = expectation_values["g1"], expectation_values["g2"], expectation_values["bulk_range"], expectation_values["g1_matrix"], expectation_values["g2_matrix"], expectation_values["δN"], expectation_values["S_kπ"], expectation_values["O_even"], expectation_values["O_odd"], expectation_values["κ2"], expectation_values["Δ1"], expectation_values["Δ2"], expectation_values["Δ1_bulk"], expectation_values["Δ2_bulk"], expectation_values["M_k0"], expectation_values["M_kπ"], expectation_values["M_k_vec"], expectation_values["entropy_S_vec"], expectation_values["prob_Q_mod_bulk"]
            g1_arr[m, n] = g1
            g2_arr[m, n] = g2
            δN_arr[m, n] = δN
            S_kπ_arr[m, n] = S_kπ
            O_even_arr[m, n] = O_even
            O_odd_arr[m, n] = O_odd
            κ2_arr[m, n] = κ2
            Δ1_arr[m, n] = Δ1
            Δ2_arr[m, n] = Δ2
            Δ1_bulk_arr[m, n] = Δ1_bulk
            Δ2_bulk_arr[m, n] = Δ2_bulk

            # c_arr[m, n], _ = DMRG.calc_central_charge(entropy_S_vec, N_lat)
            # kmax_M_arr[m, n] = k_arr[argmax(real.(M_k_vec[1:(N_lat÷2)]))]
            # M_kmax_arr[m, n] = maximum(real.(M_k_vec))
            prob_Q_arr[m, n] = prob_Q_mod_bulk
            prob_Q_gx_J1_arr[n, m] = prob_Q_mod_bulk

            println("******* $(m)/$(N_J1) && $(n)/$(N_gx) DONE *******")
        end # for
    end # for

    ### FITTING CORRELATIONS ###
    # α_g1_arr = zeros(Float64, N_J1, N_gx)
    # res_g1_arr = zeros(Float64, N_J1, N_gx)
    # α_g2_arr = zeros(Float64, N_J1, N_gx)
    # res_g2_arr = zeros(Float64, N_J1, N_gx)
    # for m in 1:N_J1
    #     for n in 1:N_gx
    #         abs_J1_m = abs_J1_vec[m]
    #         gx_n = gx_vec[n]

    #         expectation_values = npzread("$(save_path)/$(file_names[findfirst(contains("m=$(m)!n=$(n)"), file_names)])")
    #         g1, g2, bulk_range, g1_matrix, g2_matrix, δN, S_kπ, O_even, O_odd, κ2, M_k0, M_kπ, M_k_vec, entropy_S_vec, prob_Q_mod_bulk = expectation_values["g1"], expectation_values["g2"], expectation_values["bulk_range"], expectation_values["g1_matrix"], expectation_values["g2_matrix"], expectation_values["δN"], expectation_values["S_kπ"], expectation_values["O_even"], expectation_values["O_odd"], expectation_values["κ2"], expectation_values["M_k0"], expectation_values["M_kπ"], expectation_values["M_k_vec"], expectation_values["entropy_S_vec"], expectation_values["prob_Q_mod_bulk"]

    #         # NOTE: Here fit is done for abs correlations, not correlations themselves.
    #         g1_fit_params, g1_fit_resid, g1_fit_vec = calc_fit_1D_MC(ln_power_law_fit, fit_range, log.(abs.(g1_matrix[idx_start, idx_start .+ fit_range])), lower, upper; N_p0=100)
    #         # g1_fit_params, g1_fit_resid, g1_fit_vec = calc_fit_1D_MC(power_law_fit, fit_range, abs.(g1_matrix[idx_start, idx_start .+ fit_range]), lower, upper; N_p0=100)
    #         α_g1_arr[m, n] = g1_fit_params[2]
    #         res_g1_arr[m, n] = sum(abs2.(g1_fit_resid))
    #         g2_fit_params, g2_fit_resid, g2_fit_vec = calc_fit_1D_MC(ln_power_law_fit, fit_range, log.(abs.(g2_matrix[idx_start, idx_start .+ fit_range])), lower, upper; N_p0=100)
    #         # g2_fit_params, g2_fit_resid, g2_fit_vec = calc_fit_1D_MC(power_law_fit, fit_range, abs.(g2_matrix[idx_start, idx_start .+ fit_range]), lower, upper; N_p0=100)
    #         α_g2_arr[m, n] = g2_fit_params[2]
    #         res_g2_arr[m, n] = sum(abs2.(g2_fit_resid))

    #         println("******* $(m)/$(N_J1) && $(n)/$(N_gx) DONE *******")
    #     end # for
    # end # for
    # ### END OF FITTING CORRELATIONS ###

    prob_Q_res = Dict{String, Any}()
    merge!(prob_Q_res, Dict("abs_J1_vec" => abs_J1_vec))
    merge!(prob_Q_res, Dict("gx_vec" => gx_vec))
    merge!(prob_Q_res, Dict("prob_Q_gx_J1_arr" => prob_Q_gx_J1_arr))

    ### SAVING RESULTS ###
    save_path_prob_Q = "$(@__DIR__)/NPY/$(Dates.today())/Prob_Q_Res"
    date_string = """$(Dates.format(Dates.now(), "Y-mm-dd!HH.MM.SS"))"""
    mkpath(save_path_prob_Q)
    npzwrite("$(save_path_prob_Q)/Prob_Q_Res!$(date_string).npz", prob_Q_res)
    ### END OF SAVING RESULTS ###

    if Threads.threadid() == 1
        ##### PLOTTING #####
        plot_u_vs_xy(abs_J1_vec, gx_vec, prob_Q_arr'; x_name=raw"$J_1/g_0$", y_name=raw"$g_x/g_0$", u_name=raw"$P_Q$")

        plotN_order_parameters_gaps_vs_J1_gx_v2(abs_J1_vec, gx_vec, g1_arr, g2_arr, δN_arr, S_kπ_arr, O_even_arr, O_odd_arr, κ2_arr, Δ1_arr, Δ2_arr, Δ1_bulk_arr, Δ2_bulk_arr, prob_Q_arr)
        # plotN_fit_res_vs_J1_gx(abs_J1_vec, gx_vec, α_g1_arr, res_g1_arr, α_g2_arr, res_g2_arr; vlim=[lower[2], upper[2]])
        ##### END OF PLOTTING #####
    end # if
    ##### END OF PHASE DIAGRAM #####

    ########## PROPER PHASE DIAGRAM ##########
    # ##### CLASSIFICATION #####
    # phase_name_arr = Array{String}(undef, N_J1, N_gx)
    # for m in 1:N_J1
    #     for n in 1:N_gx
    #         expectation_values = npzread("$(save_path)/$(file_names[findfirst(contains("m=$(m)!n=$(n)"), file_names)])")

    #         phase_name_arr[m, n] = classify_qmb_phase_v1(expectation_values)
    #     end # for
    # end # for

    # plot_phase_diagram_by_name(abs_J1_vec, gx_vec, phase_name_arr;
    #     x_label=raw"$|J_1|/g_0$", y_label=raw"$g_x/g_0$", file_name="Phase_Diagram_by_Name_vs_J1_gx")
    # ##### END OF CLASSIFICATION #####

    # ##### PHASE SEPARATING CURVES #####
    # phase_name_arr = Array{String}(undef, N_J1, N_gx)
    # for m in 1:N_J1
    #     for n in 1:N_gx
    #         expectation_values = npzread("$(save_path)/$(file_names[findfirst(contains("m=$(m)!n=$(n)"), file_names)])")

    #         phase_name_arr[m, n] = classify_qmb_phase_v1(expectation_values)
    #     end # for
    # end # for

    # CSF_curve_vec2 = calc_phase_curve_v1("CSF", abs_J1_vec, gx_vec, phase_name_arr)
    # MI_curve_vec2 = calc_phase_curve_v1("MI", abs_J1_vec, gx_vec, phase_name_arr)
    # DW_curve_vec2 = calc_phase_curve_v1("DW", abs_J1_vec, gx_vec, phase_name_arr)
    # PSF_curve_vec2 = calc_phase_curve_v1("PSF", abs_J1_vec, gx_vec, phase_name_arr)

    # phase_diagram_curves = Dict{String, Any}()
    # merge!(phase_diagram_curves, Dict("abs_J1_vec" => abs_J1_vec))
    # merge!(phase_diagram_curves, Dict("gx_vec" => gx_vec))
    # merge!(phase_diagram_curves, Dict("CSF_J1_vec" => CSF_curve_vec2[1]))
    # merge!(phase_diagram_curves, Dict("CSF_gx_top_vec" => CSF_curve_vec2[2]))
    # merge!(phase_diagram_curves, Dict("CSF_gx_bottom_vec" => CSF_curve_vec2[3]))
    # merge!(phase_diagram_curves, Dict("MI_J1_vec" => MI_curve_vec2[1]))
    # merge!(phase_diagram_curves, Dict("MI_gx_top_vec" => MI_curve_vec2[2]))
    # merge!(phase_diagram_curves, Dict("MI_gx_bottom_vec" => MI_curve_vec2[3]))
    # merge!(phase_diagram_curves, Dict("DW_J1_vec" => DW_curve_vec2[1]))
    # merge!(phase_diagram_curves, Dict("DW_gx_top_vec" => DW_curve_vec2[2]))
    # merge!(phase_diagram_curves, Dict("DW_gx_bottom_vec" => DW_curve_vec2[3]))
    # merge!(phase_diagram_curves, Dict("PSF_J1_vec" => PSF_curve_vec2[1]))
    # merge!(phase_diagram_curves, Dict("PSF_gx_top_vec" => PSF_curve_vec2[2]))
    # merge!(phase_diagram_curves, Dict("PSF_gx_bottom_vec" => PSF_curve_vec2[3]))

    # ### SAVING RESULTS ###
    # save_path_curves = "$(@__DIR__)/NPY/$(Dates.today())/Phase_Diagram_Curves_J1_gx"
    # date_string = """$(Dates.format(Dates.now(), "Y-mm-dd!HH.MM.SS"))"""
    # mkpath(save_path_curves)
    # npzwrite("$(save_path_curves)/Phase_Diagram_Curves_J1_gx!$(date_string).npz", phase_diagram_curves)
    # ### END OF SAVING RESULTS ###

    # plot_phase_diagram_vs_J1_gx(abs_J1_vec, gx_vec, CSF_curve_vec2, MI_curve_vec2, DW_curve_vec2, PSF_curve_vec2)
    # ##### END OF PHASE SEPARATING CURVES #####

    ##### PLOTTING (FILLING BETWEEN CURVES) #####
    ##### c
    ##### END OF PLOTTING (FILLING BETWEEN CURVES) #####
    ########## END OF PROPER PHASE DIAGRAM ##########

    # ##### PHASE CURVE (HORIZONTAL CUT / FIXED g_x) #####
    # S_gx = -1.0
    # S_n = x2index(S_gx, gx_vec)
    # gx_n = gx_vec[S_n]
    # @show gx_n

    # expect_list = ["O_even", "g1", "κ2"]  # MI-CSF
    # # expect_list = ["g2", "δN", "S_kπ"]  # DW-PSF
    # # expect_list = ["κ2", "g1", "O_odd", "δN", "S_kπ"]  # DW-CSF
    # # expect_list = ["κ2", "g1", "g2", "O_odd"]  # PSF-CSF
    # # expect_list = ["O_odd", "O_even", "δN", "S_kπ"]  # MI-DW
    # N_expect = length(expect_list)

    # expect_vec2 = []
    # for (i, expect_name) in enumerate(expect_list)
    #     expect_vec = []
    #     for m in 1:N_J1
    #         abs_J1_m = abs_J1_vec[m]

    #         expectation_values = npzread("$(save_path)/$(file_names[findfirst(contains("m=$(m)!n=$(S_n)"), file_names)])")
    #         expect_val = expectation_values[expect_name]

    #         push!(expect_vec, expect_val)
    #     end # for

    #     push!(expect_vec2, expect_vec)
    # end # for

    # plotN_ys_vs_x(abs_J1_vec, expect_vec2, @__DIR__;
    #   ax_title=raw"$g_x$ = "*"$(round(gx_n, digits=4))",
    #   x_label_vec=vcat(["" for i in 1:(N_expect-1)], [raw"$|J_1|/g_0$"]),
    #   y_label_vec=expect_list, marker="o",
    #   saveDirectly=false, file_name="Expect_vs_J1")
    # ##### END OF PHASE CURVE (HORIZONTAL CUT / FIXED g_x) #####

    # ##### PHASE CURVE (VERTICAL CUT / FIXED J_1) #####
    # S_J1 = 0.07
    # S_m = x2index(S_J1, abs_J1_vec)
    # abs_J1_m = abs_J1_vec[S_m]
    # @show abs_J1_m

    # # expect_list = ["O_even", "g1", "κ2"]  # MI-CSF
    # # expect_list = ["g2", "δN", "S_kπ"]  # DW-PSF
    # # expect_list = ["κ2", "g1", "O_odd", "δN", "S_kπ"]  # DW-CSF
    # # expect_list = ["κ2", "g1", "g2", "O_odd"]  # PSF-CSF
    # # expect_list = ["O_odd", "O_even", "δN", "S_kπ"]  # MI-DW
    # expect_list = ["O_even", "O_odd", "g1", "g2", "κ2", "S_kπ", "Δ1", "Δ2"]  # All
    # # expect_list = ["O_even", "O_odd", "g1", "g2", "κ2", "δN", "Δ1_bulk", "Δ2_bulk"]  # All
    # N_expect = length(expect_list)

    # O_even_vec = zeros(Float64, N_gx)
    # g1_vec = zeros(Float64, N_gx)
    # g2_vec = zeros(Float64, N_gx)
    # κ2_vec = zeros(Float64, N_gx)
    # S_kπ_vec = zeros(Float64, N_gx)

    # expect_vec2 = []
    # for (i, expect_name) in enumerate(expect_list)
    #     expect_vec = []
    #     for n in 1:N_gx
    #         gx_n = gx_vec[n]

    #         expectation_values = npzread("$(save_path)/$(file_names[findfirst(contains("m=$(S_m)!n=$(n)"), file_names)])")
    #         expect_val = expectation_values[expect_name]

    #         push!(expect_vec, expect_val)

    #         if expect_name == "O_even"
    #             O_even_vec[n] = expectation_values[expect_name]
    #         elseif expect_name == "g1"
    #             g1_vec[n] = expectation_values[expect_name]
    #         elseif expect_name == "g2"
    #             g2_vec[n] = expectation_values[expect_name]
    #             # g2_vec[n] = sum(abs.(expectation_values["g2_matrix"][100, 300:400]))
    #         elseif expect_name == "κ2"
    #             κ2_vec[n] = expectation_values[expect_name]
    #         elseif expect_name == "S_kπ"
    #             S_kπ_vec[n] = expectation_values[expect_name]
    #         end # if
    #     end # for

    #     push!(expect_vec2, expect_vec)
    # end # for

    # J1_cut_expectation_values = Dict{String, Any}()
    # merge!(J1_cut_expectation_values, Dict("abs_J1_vec" => abs_J1_vec))
    # merge!(J1_cut_expectation_values, Dict("gx_vec" => gx_vec))
    # merge!(J1_cut_expectation_values, Dict("S_J1" => S_J1))
    # merge!(J1_cut_expectation_values, Dict("S_m" => S_m))
    # merge!(J1_cut_expectation_values, Dict("abs_J1_m" => abs_J1_m))
    # merge!(J1_cut_expectation_values, Dict("O_even_vec" => O_even_vec))
    # merge!(J1_cut_expectation_values, Dict("g1_vec" => g2_vec))
    # merge!(J1_cut_expectation_values, Dict("g2_vec" => g2_vec))
    # merge!(J1_cut_expectation_values, Dict("κ2_vec" => κ2_vec))
    # merge!(J1_cut_expectation_values, Dict("S_kπ_vec" => S_kπ_vec))

    # # ### SAVING RESULTS ###
    # # save_path_J1 = "$(@__DIR__)/NPY/$(Dates.today())/J1_Cut_Expectation_Values"
    # # date_string = """$(Dates.format(Dates.now(), "Y-mm-dd!HH.MM.SS"))"""
    # # mkpath(save_path_J1)
    # # npzwrite("$(save_path_J1)/J1_Cut_Expectation_Values!$(date_string).npz", J1_cut_expectation_values)
    # # ### END OF SAVING RESULTS ###

    # plotN_ys_vs_x(gx_vec, expect_vec2, @__DIR__;
    #   ax_title=raw"$|J_1|$ = "*"$(round(abs_J1_m, digits=4))",
    #   x_lims_vec=[[gx_vec[end], gx_vec[1]] for i in 1:N_expect],
    #   x_label_vec=vcat(["" for i in 1:(N_expect-1)], [raw"$g_x/g_0$"]),
    #   y_label_vec=expect_list, marker="o",
    #   saveDirectly=false, file_name="Expect_vs_gx")
    # ##### END OF PHASE CURVE (VERTICAL CUT / FIXED g_x) #####

    # ##### SINGLE SELECTED POINT #####
    # S_J1 = 0.00
    # S_gx = -2.5

    # S_m = x2index(S_J1, abs_J1_vec)
    # S_n = x2index(S_gx, gx_vec)

    # abs_J1_m = abs_J1_vec[S_m]
    # gx_n = gx_vec[S_n]

    # expectation_values = npzread("$(save_path)/$(file_names[findfirst(contains("m=$(S_m)!n=$(S_n)"), file_names)])")
    # g1, g2, δN, S_kπ, O_even, O_odd, κ2, M_k0, M_kπ, prob_Q_mod_bulk = expectation_values["g1"], expectation_values["g2"], expectation_values["δN"], expectation_values["S_kπ"], expectation_values["O_even"], expectation_values["O_odd"], expectation_values["κ2"], expectation_values["M_k0"], expectation_values["M_kπ"], expectation_values["prob_Q_mod_bulk"]
    # n_arr, S_k_vec, g1_vec, g2_vec, g1_matrix, g2_matrix, M_k_vec = expectation_values["n_arr"], expectation_values["S_k_vec"], expectation_values["g1_vec"], expectation_values["g2_vec"], expectation_values["g1_matrix"], expectation_values["g2_matrix"], expectation_values["M_k_vec"]

    # @show δN
    # @show g1
    # @show g1_vec[N_lat÷2]
    # @show g1_vec[N_lat÷2+1]
    # @show g1_vec[N_lat÷2:(N_lat÷2+10)]
    # @show g2
    # @show g2_vec[N_lat÷2]
    # @show g2_vec[N_lat÷2+1]
    # @show g2_vec[N_lat÷2:(N_lat÷2+10)]
    # @show κ2
    # @show S_kπ
    # @show M_k0
    # @show M_kπ
    # @show prob_Q_mod_bulk
    # @show O_odd
    # @show O_even

    # # FITTING CORRELATIONS #
    # # g1_fit_params, g1_fit_resid, g1_fit_vec = calc_fit_1D_MC(ln_power_law_fit, fit_range, log.(abs.(g1_matrix[idx_start, idx_start .+ fit_range])), lower, upper; N_p0=100)
    # # g1_fit_params, g1_fit_resid, g1_fit_vec = calc_fit_1D_MC(power_law_fit, fit_range, abs.(g1_matrix[idx_start, idx_start .+ fit_range]), lower, upper; N_p0=100)
    # # α_g1 = g1_fit_params[2]
    # # @show α_g1
    # # @show sum(abs2.(g1_fit_resid))

    # g2_fit_params, g2_fit_resid, g2_fit_vec = calc_fit_1D_MC(ln_power_law_fit, fit_range, log.(abs.(g2_matrix[idx_start, idx_start .+ fit_range])), lower, upper; N_p0=100)
    # # g2_fit_params, g2_fit_resid, g2_fit_vec = calc_fit_1D_MC(power_law_fit, fit_range, abs.(g2_matrix[idx_start, idx_start .+ fit_range]), lower, upper; N_p0=100)
    # α_g2 = g2_fit_params[2]
    # @show α_g2
    # @show sum(abs2.(g2_fit_resid))
    # # END OF FITTING CORRELATIONS #

    # ψ_samples = expectation_values["ψ_samples"]
    # println("********** SAMPLES **********")
    # display([ψ_samples[i, :] for i in 1:size(ψ_samples, 1)])
    # println("********** END OF SAMPLES **********")
    # display([ψ_samples[i, (N_lat÷2-15):(N_lat÷2+15)] for i in 1:size(ψ_samples, 1)])

    # ### PLOTTING ###
    # plotN_expectation_values_v1(N_lat, k_arr, n_arr, S_k_vec, g1_vec, g2_vec, g1_matrix, M_k_vec)

    # # ax_title = raw"$g_1(x) = $"*"$(round(g1_fit_params[1], digits=4)) "*raw"$x$"*"^(-$(round(g1_fit_params[2], digits=4)))"
    # # # plot_ys_vs_x(fit_range, [abs.(g1_matrix[idx_start, idx_start .+ fit_range]), exp.(g1_fit_vec)], @__DIR__; x_label="i-j", y_label=raw"$g_1(i-j)$", x_scale="log", y_scale="log", ax_title=ax_title, color_vec=["blue", "red"], saveDirectly=false, file_name="g1_vs_x_reg_and_fit")
    # # plot_ys_vs_x(fit_range, [abs.(g1_matrix[idx_start, idx_start .+ fit_range]), g1_fit_vec], @__DIR__; x_label="i-j", y_label=raw"$g_1(i-j)$", x_scale="log", y_scale="log", ax_title=ax_title, color_vec=["blue", "red"], saveDirectly=false, file_name="g1_vs_x_reg_and_fit")

    # ax_title = raw"$g_2(x) = $"*"$(round(g2_fit_params[1], digits=4)) "*raw"$x$"*"^(-$(round(g2_fit_params[2], digits=4)))"
    # plot_ys_vs_x(fit_range, [abs.(g2_matrix[idx_start, idx_start .+ fit_range]), exp.(g2_fit_vec)], @__DIR__; x_label="i-j", y_label=raw"$g_2(i-j)$", x_scale="log", y_scale="log", ax_title=ax_title, color_vec=["blue", "red"], saveDirectly=false, file_name="g2_vs_x_reg_and_fit")
    # # plot_ys_vs_x(fit_range, [abs.(g2_matrix[idx_start, idx_start .+ fit_range]), g2_fit_vec], @__DIR__; x_label="i-j", y_label=raw"$g_2(i-j)$", x_scale="log", y_scale="log", ax_title=ax_title, color_vec=["blue", "red"], saveDirectly=false, file_name="g2_vs_x_reg_and_fit")
    # ### END OF PLOTTING ###
    # ##### END OF SINGLE SELECTED POINT #####
end # function

function routine_read_phase_curve_gx(N_lat, N_part, boson_dim, conserveParticleNumber, μ, N_repeat, N_sweeps, max_dim, cutoff, noise)
    ##### SPECIFIC PARAMETERS #####
    N_gx = 31  # 12  # 22
    gx_vec = LinRange(-5.0, 0.0, N_gx)
    ##### END OF SPECIFIC PARAMETERS #####

    k_arr = LinRange(0.0, 2π, N_lat+1)[1:N_lat]

    # save_path = "$(@__DIR__)/Selected NPY/Phase_Curve_Gaps_gx, 2025-09-02/Phase_Curve_Gaps_gx"  # boson_dim=6
    # save_path = "$(@__DIR__)/Selected NPY/Phase_Curve_Gaps_gx, 2025-09-11/Phase_Curve_Gaps_gx"  # boson_dim=3
    # save_path = "$(@__DIR__)/Selected NPY/Phase_Curve_Gaps_gx, 2025-09-27/Phase_Curve_Gaps_gx"  # boson_dim=3, N_gx=51, N_lat=60, gx_vec = LinRange(-5.0, 0.0, N_gx)
    # save_path = "$(@__DIR__)/Selected NPY/Phase_Curve_Gaps_gx, 2025-09-28/Phase_Curve_Gaps_gx"  # boson_dim=3, N_gx=31, N_lat=100, gx_vec = LinRange(-5.0, 0.0, N_gx)
    # save_path = "$(@__DIR__)/Selected NPY/Phase_Curve_Gaps_gx, 2025-10-23/Phase_Curve_Gaps_gx"  # boson_dim=3, N_gx=31, N_lat=400, gx_vec = LinRange(-5.0, 0.0, N_gx)
    # save_path = "$(@__DIR__)/Selected NPY/Phase_Curve_gx, 2025-11-16/Phase_Curve_gx"  # boson_dim=3, N_gx=31, N_lat=401, gx_vec = LinRange(-5.0, 0.0, N_gx)
    # save_path = "$(@__DIR__)/Selected NPY/Phase_Curve_gx, Half-Filling, 2025-11-17/Phase_Curve_gx"  # boson_dim=3, N_gx=31, N_lat=401, gx_vec = LinRange(-5.0, 0.0, N_gx)
    save_path = "$(@__DIR__)/Selected NPY/Phase_Curve_gx, 801 sites, 2025-11-19/Phase_Curve_gx"  # boson_dim=3, N_gx=31, N_lat=801, gx_vec = LinRange(-5.0, 0.0, N_gx)

    file_names = readdir(save_path)

    ##### PHASE CURVE #####
    δN_vec = zeros(Float64, N_gx)
    g1_vec = zeros(Float64, N_gx)
    g2_vec = zeros(Float64, N_gx)
    S_k0_vec = zeros(Float64, N_gx)
    M_k0_vec = zeros(Float64, N_gx)
    S_kπ_vec = zeros(Float64, N_gx)  # Assumes N_lat is even
    M_kπ_vec = zeros(Float64, N_gx)  # Assumes N_lat is even
    integ_g1_vec = zeros(Float64, N_gx)
    integ_g2_vec = zeros(Float64, N_gx)
    integ_S_k_vec = zeros(Float64, N_gx)
    O_odd_vec = zeros(Float64, N_gx)
    O_even_vec = zeros(Float64, N_gx)
    expect_E_vec = zeros(Float64, N_gx)
    # integ_M_k_vec = zeros(Float64, N_gx)
    c_vec = zeros(Float64, N_gx)
    prob_Q_vec = zeros(Float64, N_gx)
    prob_P_vec = zeros(Float64, N_gx)

    for n in 1:N_gx
        gx_n = gx_vec[n]

        expectation_values = npzread("$(save_path)/$(file_names[findfirst(contains("gx=$(round(gx_n; digits=4))"), file_names)])")

        expect_E, δN, g1, g2, g1_matrix, g2_matrix, ni_nj_matrix, S_k_vec, M_k_vec, O_odd, O_even, c, prob_Q, avg_NN_g1, avg_NN_g2, avg_NN_n = expectation_values["expect_E"], expectation_values["δN"], expectation_values["g1"], expectation_values["g2"], expectation_values["g1_matrix"], expectation_values["g2_matrix"], expectation_values["ni_nj_matrix"], expectation_values["S_k_vec"], expectation_values["M_k_vec"], expectation_values["O_odd"], expectation_values["O_even"], expectation_values["c"], expectation_values["prob_Q"], expectation_values["avg_NN_g1"], expectation_values["avg_NN_g2"], expectation_values["avg_NN_n"]
        # expect_E, δN, g1, g2, g1_matrix, g2_matrix, ni_nj_matrix, S_k_vec, M_k_vec, O_odd, O_even, c, prob_Q, prob_Q_mod_bulk, avg_NN_g1, avg_NN_g2, avg_NN_n = expectation_values["expect_E"], expectation_values["δN"], expectation_values["g1"], expectation_values["g2"], expectation_values["g1_matrix"], expectation_values["g2_matrix"], expectation_values["ni_nj_matrix"], expectation_values["S_k_vec"], expectation_values["M_k_vec"], expectation_values["O_odd"], expectation_values["O_even"], expectation_values["c"], expectation_values["prob_Q"], expectation_values["prob_Q_mod_bulk"], expectation_values["avg_NN_g1"], expectation_values["avg_NN_g2"], expectation_values["avg_NN_n"]
        δN_vec[n], g1_vec[n], g2_vec[n] = δN, g1, g2  # g2_matrix[30, 370]
        S_k0_vec[n], M_k0_vec[n] = real(S_k_vec[1]), real(M_k_vec[1])
        S_kπ_vec[n], M_kπ_vec[n] = real(S_k_vec[N_lat÷2+1]), real(M_k_vec[N_lat÷2+1])
        integ_g1_vec[n] = sum(abs.(g1_matrix)) - tr(abs.(g1_matrix))
        integ_g2_vec[n] = sum(abs.(g2_matrix)) - tr(abs.(g2_matrix))
        integ_S_k_vec[n] = sum(abs.(S_k_vec[(N_lat÷2-10):(N_lat÷2+10)]))  # sum(abs.(S_k_vec[(N_lat÷10):end]))
        O_odd_vec[n] = O_odd  # O_odd_matrix[100, N_lat-100]
        O_even_vec[n] = O_even  # O_even_matrix[100, N_lat-100]
        expect_E_vec[n] = expect_E
        # integ_M_k_vec[n] = sum(abs.(M_k_vec[(N_lat÷10):end]))
        c_vec[n] = c
        prob_Q_vec[n] = prob_Q
        # prob_Q_vec[n] = prob_Q_mod_bulk  # prob_Q
        # prob_P_vec[n] = prob_P_bulk

        println("************** $(n)/$(N_gx) DONE **************")
    end # for

    if Threads.threadid() == 1
        ##### PLOTTING #####
        plotN_order_parameters_vs_gx(gx_vec, δN_vec, g1_vec, g2_vec, integ_g1_vec, integ_g2_vec, integ_S_k_vec, O_odd_vec, O_even_vec, expect_E_vec, S_k0_vec, M_k0_vec, S_kπ_vec, M_kπ_vec, c_vec, prob_Q_vec)
        ##### END OF PLOTTING #####
    end # if
    ##### END OF PHASE CURVE #####

    # ##### SINGLE SELECTED POINT #####
    # S_gx = -3.0

    # # S_n = 6
    # S_n = x2index(S_gx, gx_vec)

    # gx_n = gx_vec[S_n]

    # expectation_values = npzread("$(save_path)/$(file_names[findfirst(contains("gx=$(round(gx_n; digits=4))"), file_names)])")
    # expect_E, δN, n_arr, g1, g2, g1_vec, g2_vec, g1_matrix, g2_matrix, S_k_vec, M_k_vec, c, Δ1, Δ2 = expectation_values["expect_E"], expectation_values["δN"], expectation_values["n_arr"], expectation_values["g1"], expectation_values["g2"], expectation_values["g1_vec"], expectation_values["g2_vec"], expectation_values["g1_matrix"], expectation_values["g2_matrix"], expectation_values["S_k_vec"], expectation_values["M_k_vec"], expectation_values["c"], expectation_values["Δ1"], expectation_values["Δ2"]

    # println("***** SINGLE SELECTED POINT *****")
    # @show gx_n
    # @show expect_E
    # @show 2*1.0*0.1666*N_lat
    # @show δN
    # @show g1
    # @show g2
    # @show c
    # @show Δ1
    # @show Δ2
    # println("***** END OF SINGLE SELECTED POINT *****")

    # plotN_expectation_values_v1(N_lat, k_arr, n_arr, S_k_vec, g1_vec, g2_vec, g1_matrix, M_k_vec)
    # ##### END OF SINGLE SELECTED POINT #####
end # function

function routine_read_phase_curve_gaps_gx(N_lat, N_part, boson_dim, conserveParticleNumber, μ, N_repeat, N_sweeps, max_dim, cutoff, noise)
    ##### SPECIFIC PARAMETERS #####
    N_gx = 31  # 12  # 22
    gx_vec = LinRange(-5.0, 0.0, N_gx)
    ##### END OF SPECIFIC PARAMETERS #####

    k_arr = LinRange(0.0, 2π, N_lat+1)[1:N_lat]

    # save_path = "$(@__DIR__)/Selected NPY/Phase_Curve_Gaps_gx, 2025-09-02/Phase_Curve_Gaps_gx"  # boson_dim=6
    # save_path = "$(@__DIR__)/Selected NPY/Phase_Curve_Gaps_gx, 2025-09-11/Phase_Curve_Gaps_gx"  # boson_dim=3
    # save_path = "$(@__DIR__)/Selected NPY/Phase_Curve_Gaps_gx, 2025-09-27/Phase_Curve_Gaps_gx"  # boson_dim=3, N_gx=51, N_lat=60, gx_vec = LinRange(-5.0, 0.0, N_gx)
    # save_path = "$(@__DIR__)/Selected NPY/Phase_Curve_Gaps_gx, 2025-09-28/Phase_Curve_Gaps_gx"  # boson_dim=3, N_gx=31, N_lat=100, gx_vec = LinRange(-5.0, 0.0, N_gx)
    # save_path = "$(@__DIR__)/Selected NPY/Phase_Curve_Gaps_gx, 2025-10-23/Phase_Curve_Gaps_gx"  # boson_dim=3, N_gx=31, N_lat=400, gx_vec = LinRange(-5.0, 0.0, N_gx)
    # save_path = "$(@__DIR__)/Selected NPY/Phase_Curve_gx, 2025-11-16/Phase_Curve_gx"  # boson_dim=3, N_gx=31, N_lat=401, gx_vec = LinRange(-5.0, 0.0, N_gx)
    # save_path = "$(@__DIR__)/Selected NPY/Phase_Curve_gx, Half-Filling, 2025-11-17/Phase_Curve_gx"  # boson_dim=3, N_gx=31, N_lat=401, gx_vec = LinRange(-5.0, 0.0, N_gx)
    save_path = "$(@__DIR__)/Selected NPY/Phase_Curve_gx, 801 sites, 2025-11-19/Phase_Curve_gx"  # boson_dim=3, N_gx=31, N_lat=801, gx_vec = LinRange(-5.0, 0.0, N_gx)

    file_names = readdir(save_path)

    ##### PHASE CURVE #####
    δN_vec = zeros(Float64, N_gx)
    g1_vec = zeros(Float64, N_gx)
    g2_vec = zeros(Float64, N_gx)
    Δ1_vec = zeros(Float64, N_gx)
    Δ2_vec = zeros(Float64, N_gx)
    S_k0_vec = zeros(Float64, N_gx)
    M_k0_vec = zeros(Float64, N_gx)
    S_kπ_vec = zeros(Float64, N_gx)  # Assumes N_lat is even
    M_kπ_vec = zeros(Float64, N_gx)  # Assumes N_lat is even
    integ_g1_vec = zeros(Float64, N_gx)
    integ_g2_vec = zeros(Float64, N_gx)
    integ_S_k_vec = zeros(Float64, N_gx)
    O_odd_vec = zeros(Float64, N_gx)
    O_even_vec = zeros(Float64, N_gx)
    expect_E_vec = zeros(Float64, N_gx)
    # integ_M_k_vec = zeros(Float64, N_gx)
    c_vec = zeros(Float64, N_gx)
    prob_Q_vec = zeros(Float64, N_gx)
    avg_NN_g1_vec = zeros(Float64, N_gx)
    avg_NN_g2_vec = zeros(Float64, N_gx)
    avg_NN_n_vec = zeros(Float64, N_gx)
    avg_sep2_n_vec = zeros(Float64, N_gx)

    for n in 1:N_gx
        gx_n = gx_vec[n]

        expectation_values = npzread("$(save_path)/$(file_names[findfirst(contains("gx=$(round(gx_n; digits=4))"), file_names)])")

        expect_E, δN, g1, g2, g1_matrix, g2_matrix, ni_nj_matrix, S_k_vec, M_k_vec, O_odd, O_even, c, prob_Q, avg_NN_g1, avg_NN_g2, avg_NN_n = expectation_values["expect_E"], expectation_values["δN"], expectation_values["g1"], expectation_values["g2"], expectation_values["g1_matrix"], expectation_values["g2_matrix"], expectation_values["ni_nj_matrix"], expectation_values["S_k_vec"], expectation_values["M_k_vec"], expectation_values["O_odd"], expectation_values["O_even"], expectation_values["c"], expectation_values["prob_Q"], expectation_values["avg_NN_g1"], expectation_values["avg_NN_g2"], expectation_values["avg_NN_n"]
        # expect_E, δN, g1, g2, g1_matrix, g2_matrix, ni_nj_matrix, S_k_vec, M_k_vec, O_odd, O_even, c, prob_Q_mod_bulk, avg_NN_g1, avg_NN_g2, avg_NN_n = expectation_values["expect_E"], expectation_values["δN"], expectation_values["g1"], expectation_values["g2"], expectation_values["g1_matrix"], expectation_values["g2_matrix"], expectation_values["ni_nj_matrix"], expectation_values["S_k_vec"], expectation_values["M_k_vec"], expectation_values["O_odd"], expectation_values["O_even"], expectation_values["c"], expectation_values["prob_Q_mod_bulk"], expectation_values["avg_NN_g1"], expectation_values["avg_NN_g2"], expectation_values["avg_NN_n"]
        δN_vec[n], g1_vec[n], g2_vec[n] = δN, g1, g2
        S_k0_vec[n], M_k0_vec[n] = real(S_k_vec[1]), real(M_k_vec[1])
        S_kπ_vec[n], M_kπ_vec[n] = real(S_k_vec[N_lat÷2+1]), real(M_k_vec[N_lat÷2+1])
        integ_g1_vec[n] = sum(abs.(g1_matrix)) - tr(abs.(g1_matrix))
        integ_g2_vec[n] = sum(abs.(g2_matrix)) - tr(abs.(g2_matrix))
        integ_S_k_vec[n] = sum(abs.(S_k_vec[(N_lat÷2-10):(N_lat÷2+10)]))  # sum(abs.(S_k_vec[(N_lat÷10):end]))
        O_odd_vec[n] = abs(O_odd)  # O_odd_matrix[100, N_lat-100]
        O_even_vec[n] = abs(O_even)  # O_even_matrix[100, N_lat-100]
        expect_E_vec[n] = expect_E
        # integ_M_k_vec[n] = sum(abs.(M_k_vec[(N_lat÷10):end]))
        c_vec[n] = c
        prob_Q_vec[n] = prob_Q
        # prob_Q_vec[n] = prob_Q_mod_bulk  # prob_Q
        avg_NN_g1_vec[n] = avg_NN_g1
        avg_NN_g2_vec[n] = avg_NN_g2
        avg_NN_n_vec[n] = avg_NN_n
        avg_sep2_n_vec[n] = sum(diag(ni_nj_matrix, 2))

        Δ1, Δ2 = expectation_values["Δ1"], expectation_values["Δ2"]
        Δ1_vec[n] = Δ1
        Δ2_vec[n] = Δ2

        println("************** $(n)/$(N_gx) DONE **************")
    end # for

    if Threads.threadid() == 1
        ##### PLOTTING #####
        plotN_order_parameters_gaps_vs_gx(gx_vec, δN_vec, g1_vec, g2_vec, Δ1_vec, Δ2_vec, integ_g1_vec, integ_g2_vec, integ_S_k_vec, O_odd_vec, O_even_vec, expect_E_vec, S_k0_vec, M_k0_vec, S_kπ_vec, M_kπ_vec, c_vec, prob_Q_vec, avg_NN_g1_vec, avg_NN_g2_vec, avg_NN_n_vec, avg_sep2_n_vec)
        ##### END OF PLOTTING #####
    end # if
    ##### END OF PHASE CURVE #####

    # ##### SINGLE SELECTED POINT #####
    # S_gx = -3.0

    # # S_n = 6
    # S_n = x2index(S_gx, gx_vec)

    # gx_n = gx_vec[S_n]

    # expectation_values = npzread("$(save_path)/$(file_names[findfirst(contains("gx=$(round(gx_n; digits=4))"), file_names)])")
    # expect_E, δN, n_arr, g1, g2, g1_vec, g2_vec, g1_matrix, g2_matrix, S_k_vec, M_k_vec, c, Δ1, Δ2 = expectation_values["expect_E"], expectation_values["δN"], expectation_values["n_arr"], expectation_values["g1"], expectation_values["g2"], expectation_values["g1_vec"], expectation_values["g2_vec"], expectation_values["g1_matrix"], expectation_values["g2_matrix"], expectation_values["S_k_vec"], expectation_values["M_k_vec"], expectation_values["c"], expectation_values["Δ1"], expectation_values["Δ2"]

    # println("***** SINGLE SELECTED POINT *****")
    # @show gx_n
    # @show expect_E
    # @show 2*1.0*0.1666*N_lat
    # @show δN
    # @show g1
    # @show g2
    # @show c
    # @show Δ1
    # @show Δ2
    # println("***** END OF SINGLE SELECTED POINT *****")

    # plotN_expectation_values_v1(N_lat, k_arr, n_arr, S_k_vec, g1_vec, g2_vec, g1_matrix, M_k_vec)
    # ##### END OF SINGLE SELECTED POINT #####
end # function

function routine_read_phase_diagram_gx_Npart(N_lat, boson_dim, conserveParticleNumber, μ, N_sweeps, max_dim, cutoff, noise)
    k_arr = LinRange(0.0, 2π, N_lat+1)[1:N_lat]

    # save_path = "$(@__DIR__)/NPY/2024-12-24/18:49/Expectation_Values_Vec"
    save_path = "$(@__DIR__)/NPY/2024-12-27/23:11/Expectation_Values_Vec"
    file_names = readdir(save_path)

    ##### PHASE CURVE #####
    N_gx = 12  # 4  # 21
    N_N_part = 11  # 12  # 22

    gx_vec = LinRange(-10.0, 10.0, N_gx)
    N_part_vec = round.(Int, LinRange(N_lat, 2*N_lat, N_N_part))

    δN_arr = zeros(Float64, N_gx, N_N_part)
    g1_arr = zeros(Float64, N_gx, N_N_part)
    g2_arr = zeros(Float64, N_gx, N_N_part)
    integ_g1_arr = zeros(Float64, N_gx, N_N_part)
    integ_g2_arr = zeros(Float64, N_gx, N_N_part)
    integ_S_k_arr = zeros(Float64, N_gx, N_N_part)
    integ_M_k_arr = zeros(Float64, N_gx, N_N_part)
    abs_O_odd_arr = zeros(Float64, N_gx, N_N_part)

    for m in 1:N_gx
        for n in 1:N_N_part
            gx_m = gx_vec[m]
            N_part_n = N_part_vec[n]

            expectation_values = npzread("$(save_path)/$(file_names[findfirst(contains("m=$(m)!n=$(n)"), file_names)])")
            δN, g1, g2, g1_matrix, g2_matrix, S_k_vec, M_k_vec, O_odd, O_even = expectation_values["δN"], expectation_values["g1"], expectation_values["g2"], expectation_values["g1_matrix"], expectation_values["g2_matrix"], expectation_values["S_k_vec"], expectation_values["M_k_vec"], expectation_values["O_odd"], expectation_values["O_even"]
            integ_g1_arr[m, n] = sum(abs.(g1_matrix)) - tr(abs.(g1_matrix))
            integ_g2_arr[m, n] = sum(abs.(g2_matrix)) - tr(abs.(g2_matrix))
            δN_arr[m, n], g1_arr[m, n], g2_arr[m, n] = δN, g1, g2
            integ_S_k_arr[m, n] = sum(abs.(S_k_vec[(N_lat÷10):end]))
            integ_M_k_arr[m, n] = sum(abs.(M_k_vec[(N_lat÷10):end]))
            abs_O_odd_arr[m, n] = abs(O_odd)
        end # for
    end # for

    if Threads.threadid() == 1
        ##### PLOTTING #####
        plotN_order_parameters_vs_gx_Npart(gx_vec, N_part_vec, δN_arr, g1_arr, g2_arr, integ_g1_arr, integ_g2_arr, integ_S_k_arr, integ_M_k_arr, abs_O_odd_arr)
        ##### END OF PLOTTING #####
    end # if
    ##### END OF PHASE CURVE #####
end # function
##### END OF READING ROUTINES #####

function routine_test()
    ##### A #####
    # expectation_values = npzread("$(@__DIR__)/NPY/2024-08-15/Expectation_Values/Expectation_Values!2024-08-15!20.07.26.npz")
    # expectation_values = npzread("$(@__DIR__)/NPY/2024-08-16/Expectation_Values/Expectation_Values!2024-08-16!19.48.12.npz")
    # @show expectation_values["δN"]
    # @show expectation_values["g2"]
    # @show expectation_values["n2_arr"]
    ##### END OF A #####

    ##### B #####
    # N_lat = 100
    # N_part = 4*N_lat
    # boson_dim = 8
    # conserveParticleNumber = true
    # μ = 0.0
    # k_arr = LinRange(0.0, 2π, N_lat+1)[1:N_lat]
    # sites = siteinds("Boson", N_lat; dim=boson_dim, conserve_qns=true)
    # ψ_Mott = create_Mott_ψ0(sites, N_lat, N_part; createRandom=false)
    # Ham = calc_BH_Ham(sites, N_lat, conserveParticleNumber, μ; J=0.0, U=0.0)  # Not used
    # expectation_values = calc_expectation_values(ψ_Mott, Ham, sites, k_arr, N_lat)
    # g2_vec = expectation_values["g2_vec"]
    # plt.plot(g2_vec, color="cyan")
    ##### END OF B #####

    # ##### C #####
    # N_J2 = 21
    # J2_vec = LinRange(0.0, 0.100, N_J2)

    # # NOTE: Complex procedure because saved expectation value data in several
    # # files.  Couldn't have I just saved an array of dictionaries in a single
    # # file?
    # data_dir = string(@__DIR__, "/Handpicked/2024-08-20/SS Phase Curves/Expectation_Values_Vec")
    # data_files = readdir(data_dir)
    # @assert length(data_files) == N_J2
    # # display(data_files)
    # unordered_data_J2_vec = Vector{Float64}([])
    # for (i, data_file) in enumerate(data_files)
    #     # println(data_files[i])
    #     idx_start = findfirst('=', data_files[i])
    #     idx_end = findall(x -> x == '!', data_files[i])[2]
    #     J2_val = parse(Float64, data_files[i][(idx_start+1):(idx_end-1)])
    #     push!(unordered_data_J2_vec, J2_val)
    # end # for
    # data_ordering = sortperm(unordered_data_J2_vec)
    # ordered_data_files = data_files[data_ordering]

    # δN_arr = zeros(Float64, N_J2)
    # g1_arr = zeros(Float64, N_J2)
    # g2_arr = zeros(Float64, N_J2)
    # for (i, data_file) in enumerate(ordered_data_files)
    #     data_path = string(data_dir, "/", data_file)
    #     expectation_values = npzread(data_path)

    #     δN_arr[i] = expectation_values["δN"]

    #     i_start = expectation_values["i_start"]
    #     i_end = expectation_values["i_end"]

    #     # g1_arr[i] = expectation_values["g1"]
    #     g1_js = expectation_values["g1_vec"]
    #     g1_arr[i] = maximum(g1_js[(end-10):end])

    #     # g2_arr[i] = expectation_values["g2"]
    #     g2_js = expectation_values["g2_vec"]
    #     g2_arr[i] = maximum(g2_js[(end-10):end])
    # end # for

    # plotN_SS_phase_curves_v1(J2_vec, δN_arr, g1_arr, g2_arr)
    # ##### END OF C #####

    # ##### D #####
    # N_lat = 200
    # N_part = N_lat
    # boson_dim = 3
    # conserveParticleNumber = true

    # sites = siteinds("Boson", N_lat; dim=boson_dim, conserve_qns=conserveParticleNumber)

    # # ψ = create_ψ0(sites, N_lat, N_part, boson_dim; linkdims=100)
    # # ψ = create_pair_ψ0(sites, N_lat, N_part, boson_dim)
    # ψ = create_Mott_ψ0(sites, N_lat, N_part; createRandom=true)

    # ##### SAMPLING #####
    # N_samples = 100
    # orthogonalize!(ψ, 1)
    # ψ_samples = zeros(Int, N_samples, N_lat)
    # for i in 1:N_samples
    #     ψ_samples[i, :] = (sample(ψ) .- 1)
    # end # for
    # println("********** SAMPLES **********")
    # display([ψ_samples[i, :] for i in 1:N_samples])
    # println("********** END OF SAMPLES **********")
    # # display([ψ_samples[i, (N_lat÷2-15):(N_lat÷2+15)] for i in 1:N_samples])
    # ##### END OF SAMPLING #####
    # ##### END OF D #####

    ##### E #####
    N_lat = 601  # 201  # 75

    save_path = "$(@__DIR__)/Selected NPY/Phase_Diagram_J1_gx, chi=300, 2026-01-25, 1/Expectation_Values_Vec"  # N_lat = 601
    file_names = readdir(save_path)

    expectation_values = npzread("$(save_path)/$(file_names[findfirst(contains("m=1!n=1"), file_names)])")
    abs_J1_vec = expectation_values["abs_J1_vec"]
    gx_vec = expectation_values["gx_vec"]
    # N_lat = expectation_values["N_lat"]
    N_J1 = length(abs_J1_vec)
    N_gx = length(gx_vec)

    k_arr = LinRange(0.0, 2π, N_lat+1)[1:N_lat]

    S_J1 = 0.0
    S_m = x2index(S_J1, abs_J1_vec)
    abs_J1_m = abs_J1_vec[S_m]
    @show abs_J1_m

    # expect_list = ["O_even", "g1", "κ2"]  # MI-CSF
    # expect_list = ["g2", "δN", "S_kπ"]  # DW-PSF
    # expect_list = ["κ2", "g1", "O_odd", "δN", "S_kπ"]  # DW-CSF
    # expect_list = ["κ2", "g1", "g2", "O_odd"]  # PSF-CSF
    # expect_list = ["O_odd", "O_even", "δN", "S_kπ"]  # MI-DW
    expect_list = ["O_even", "O_odd", "g1", "g2", "κ2", "δN", "S_kπ", "ni_nj"]  # All
    N_expect = length(expect_list)

    expect_vec2 = []
    for (i, expect_name) in enumerate(expect_list)
        expect_vec = []
        for n in 1:N_gx
            gx_n = gx_vec[n]

            expectation_values = npzread("$(save_path)/$(file_names[findfirst(contains("m=$(S_m)!n=$(n)"), file_names)])")
            # if expect_name == "g2"
            #     g2_matrix = expectation_values["g2_matrix"]
            #     # expect_val = sum(abs.(g2_matrix[100, 300:400]))
            #     expect_val = sum(abs.(g2_matrix[10, 100]))
            # else
            #     expect_val = expectation_values[expect_name]
            # end # if
            # expect_val = expectation_values[expect_name]
            if expect_name == "ni_nj"
                i_start, i_end, ni_nj_matrix = expectation_values["i_start"], expectation_values["i_end"], expectation_values["ni_nj_matrix"]
                expect_val = ni_nj_matrix[i_start, i_end]
            else
                expect_val = abs(expectation_values[expect_name])
            end # if

            push!(expect_vec, expect_val)
        end # for

        push!(expect_vec2, expect_vec)
    end # for

    plotN_ys_vs_x(gx_vec, expect_vec2, @__DIR__;
      ax_title=raw"$|J_1|$ = "*"$(round(abs_J1_m, digits=4))",
      x_lims_vec=[[gx_vec[end], gx_vec[1]] for i in 1:N_expect],
      x_label_vec=vcat(["" for i in 1:(N_expect-1)], [raw"$g_x/g_0$"]),
      y_label_vec=expect_list, marker="o",
      saveDirectly=false, file_name="Expect_vs_gx")
    ##### END OF E #####
end # function

# TODO A much more reasonable and elegant architecture should be possible.
# TODO More elegant method of choosing model, probably using dictionaries for
# parameters and comparing keys to detect model.
function main()
    # PARAMETERS
    N_lat = 200  # 601  # 201  # 75
    N_part = N_lat  # 4*N_lat÷5  # 2*N_lat  # N_lat÷2
    boson_dim = 3  # 12  # 6
    conserveParticleNumber = true  # false

    μ = 1.5  # Considered to be 0, if conserveParticleNumber = true

    N_repeat = 25  # Repeat calculation N_repeat times and choose least energy solution.

    n_checkpoint = 5  # After n_checkpoint sweeps, save intermediate MPS.
    loadCheckpoint = false
    checkpoint_job_id = 121244124  # Only used if loadCheckpoint

    # TODO Adaptive by energy sweep stopping.
    N_sweeps = 50  # 100
    max_dim = 100  # 200  # [50, 100, 200]  # [10, 20, 50, 100, 250]  # , 100, 200]  # [100, 200, 300]
    cutoff = 1E-12  # 1E-6  # [1e-12]
    noise = vcat(repeat([1E-3], 10), repeat([1E-5], 10), repeat([0], max(1, N_sweeps-20)))
    # noise = vcat(repeat([1E-1], 10), repeat([1E-2], 10), repeat([1E-3], 10), repeat([1E-5], 10), repeat([1E-6], 10), repeat([0], max(1, N_sweeps-20)))
    # noise = vcat(repeat([1E-3], 10), repeat([0], 10), repeat([1E-3], 10), repeat([0], 10), repeat([1E-3], 10), repeat([0], max(1, N_sweeps-50)))
    # END OF PARAMETERS

    save_parameter_TXT(@__FILE__)
    save_script_TXT(@__FILE__)

    # ROUTINES
    # routine_single(N_lat, N_part, boson_dim, conserveParticleNumber, μ, N_sweeps, max_dim, cutoff, noise)
    # routine_single_repeat(N_lat, N_part, boson_dim, conserveParticleNumber, μ, N_repeat, N_sweeps, max_dim, cutoff, noise)
    # routine_multi_gaps(N_lat, N_part, boson_dim, conserveParticleNumber, μ, N_sweeps, max_dim, cutoff, noise)
    # routine_phase_curve_PSF_v1(N_lat, N_part, boson_dim, conserveParticleNumber, μ, N_sweeps, max_dim, cutoff, noise)
    # routine_phase_curve_SS_v1(N_lat, N_part, boson_dim, conserveParticleNumber, μ, N_sweeps, max_dim, cutoff, noise)
    # routine_phase_curve_SS_repeat_v1(N_lat, N_part, boson_dim, conserveParticleNumber, μ, N_sweeps, max_dim, cutoff, noise)
    # routine_phase_curve_gx(N_lat, N_part, boson_dim, conserveParticleNumber, μ, N_repeat, N_sweeps, max_dim, cutoff, noise)
    # routine_phase_curve_gaps_gx(N_lat, N_part, boson_dim, conserveParticleNumber, μ, N_repeat, N_sweeps, max_dim, cutoff, noise)
    # routine_brick_wall_compare_to_XXZ_gx(N_lat, N_part, boson_dim, conserveParticleNumber, μ, N_repeat, N_sweeps, max_dim, cutoff, noise)
    # routine_phase_diagram_J1_gx(N_lat, N_part, boson_dim, conserveParticleNumber, μ, N_sweeps, max_dim, cutoff, noise)
    # routine_phase_diagram_gaps_J1_gx(N_lat, N_part, boson_dim, conserveParticleNumber, μ, N_sweeps, max_dim, cutoff, noise)
    # routine_phase_diagram_gx_gz(N_lat, N_part, boson_dim, conserveParticleNumber, μ, N_sweeps, max_dim, cutoff, noise)
    # routine_phase_diagram_J1_gx_HPC(N_lat, N_part, boson_dim, conserveParticleNumber, μ, N_repeat, N_sweeps, max_dim, cutoff, noise)
    # routine_phase_diagram_J1_gx_gaps_HPC(N_lat, N_part, N_repeat, boson_dim, conserveParticleNumber, μ, N_sweeps, max_dim, cutoff, noise)
    # routine_phase_diagram_J1_gx_HPC_checkpoints(N_lat, N_part, boson_dim, conserveParticleNumber, μ, N_repeat, n_checkpoint, loadCheckpoint, checkpoint_job_id, N_sweeps, max_dim, cutoff, noise)
    # routine_phase_diagram_gx_Npart_HPC(N_lat, boson_dim, conserveParticleNumber, μ, N_sweeps, max_dim, cutoff, noise)
    # routine_phase_diagram_J1_Npart_HPC(N_lat, boson_dim, conserveParticleNumber, μ, N_sweeps, max_dim, cutoff, noise)
    # routine_read_phase_curve_gx(N_lat, N_part, boson_dim, conserveParticleNumber, μ, N_repeat, N_sweeps, max_dim, cutoff, noise)
    # routine_read_phase_curve_gaps_gx(N_lat, N_part, boson_dim, conserveParticleNumber, μ, N_repeat, N_sweeps, max_dim, cutoff, noise)
    # routine_read_phase_diagram_J1_gx(N_lat, N_part, boson_dim, conserveParticleNumber, μ, N_sweeps, max_dim, cutoff, noise)
    # routine_read_phase_diagram_J1_gx_v2(N_lat, N_part, boson_dim, conserveParticleNumber, μ, N_sweeps, max_dim, cutoff, noise)
    # routine_multiple_read_phase_diagram_J1_gx_v2(N_lat, N_part, boson_dim, conserveParticleNumber, μ, N_sweeps, max_dim, cutoff, noise)
    # routine_read_phase_diagram_gaps_J1_gx(N_lat, N_part, boson_dim, conserveParticleNumber, μ, N_sweeps, max_dim, cutoff, noise)
    routine_read_phase_diagram_gaps_J1_gx_v2(N_lat, N_part, boson_dim, conserveParticleNumber, μ, N_sweeps, max_dim, cutoff, noise)
    # routine_read_phase_diagram_gx_Npart(N_lat, boson_dim, conserveParticleNumber, μ, N_sweeps, max_dim, cutoff, noise)
    # routine_test()
    # END OF ROUTINES
end # function

@time main()
