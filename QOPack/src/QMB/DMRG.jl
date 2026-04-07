module DMRG

using ITensors
using ITensorMPS
using LinearAlgebra
using LsqFit

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

##### STATE CREATION #####
function try_add_particle(part_arr, part_dim)
    j = rand(1:length(part_arr))
    max_part = part_dim-1
    if part_arr[j] < max_part
        part_arr[j] += 1
        return part_arr
    else
        try_add_particle(part_arr, part_dim)
    end # if
end # function

function create_ψ0(sites, N_lat, N_part, part_dim; linkdims=250)
    part_arr = zeros(Int64, N_lat)
    ii = max(N_lat÷N_part, 1)
    for i in 1:N_part
        # part_arr[mod1(ii*i, N_lat)] += 1
        # part_arr[rand(1:N_lat)] += 1
        part_arr = try_add_particle(part_arr, part_dim)
    end # for
    state = string.(part_arr)
    # ψ0 = randomMPS(sites, state, 250)
    ψ0 = randomMPS(sites, state, linkdims)
    # ψ0 = random_mps(sites, state; linkdims=250)

    # state = [isodd(n) ? "1" : "0" for n in 1:N_lat]
    # ψ0 = MPS(sites, state)

    return ψ0
end # function

function create_homogeneous_ψ0(sites, N_lat, N_part, boson_dim; linkdims=250)
    part_arr = zeros(Int64, N_lat)
    ii = max(N_lat÷N_part, 1)
    for i in 1:N_part
        # part_arr[mod1(ii*i, N_lat)] += 1
        part_arr[rand(1:N_lat)] += 1
        # part_arr = try_add_particle(part_arr, boson_dim)
    end # for
    state = string.(part_arr)
    ψ0 = randomMPS(sites, state, linkdims)
    # ψ0 = random_mps(sites, state; linkdims=250)

    # state = [isodd(n) ? "1" : "0" for n in 1:N_lat]
    # ψ0 = MPS(sites, state)

    return ψ0
end # function

function create_pair_ψ0(sites, N_lat, N_part, boson_dim; linkdims=250)
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

function create_Mott_ψ0(sites, N_lat, N_part; createRandom=false, linkdims=250)
    part_arr = zeros(Int64, N_lat)
    for i in 1:N_part
        part_arr[mod1(i, N_lat)] += 1
    end # for
    state = string.(part_arr)
    if createRandom
        ψ_Mott = randomMPS(sites, state, linkdims)
    else
        ψ_Mott = MPS(sites, state)
    end # if

    return ψ_Mott
end # function
##### END OF STATE CREATION #####

##### EXPECTATION VALUES #####
function calc_Fock_amplitude(ψ, el, sites)
    return inner(productMPS(sites, el), ψ)
end # function

function calc_entanglement_entropy(ψ, j)
    # Taken from https://itensor.github.io/ITensors.jl/dev/examples/MPSandMPO.html#Computing-the-Entanglement-Entropy-of-an-MPS

    # ψ = ITensors.orthogonalize(ψ, j)
    orthogonalize!(ψ, j)
    U,S,V = svd(ψ[j], (linkinds(ψ, j-1)..., siteinds(ψ, j)...))
    SvN = 0.0
    for n in 1:dim(S, 1)
        p = S[n, n]^2
        SvN -= p * log(p)
    end # for

    return SvN
end # function

function calc_central_charge(entropy_S_vec, N_lat)
    ERR_VAL = -9999999.9999999

    l_vec = collect((N_lat÷5):1:(N_lat-N_lat÷5))
    S(l, p) = p[1]/6.0 * log.(2.0*N_lat/π * sin.(π*l/N_lat)) .+ p[2]
    p_0 = [1.0, 0.0]

    S_l_vec = entropy_S_vec[l_vec]

    try
        fit = curve_fit(S, l_vec, S_l_vec, p_0)
        c = fit.param[1]
        # @show fit.resid
        c_resid = sum(abs.(fit.resid))

        return c, c_resid
    catch e
        c = ERR_VAL
        c_resid = ERR_VAL
        println("WARNING: Error in central charge fit.")

        return c, c_resid
    end # try/catch
end # function

function full_calc_central_charge(ψ, N_lat)
    # l_vec = collect((N_lat÷10):(N_lat÷10):(N_lat-N_lat÷10))
    l_vec = collect((N_lat÷5):1:(N_lat-N_lat÷5))
    # l_vec = collect(10:1:(N_lat-10))
    # l_vec = 1:N_lat
    S(l, p) = p[1]/6.0 * log.(2.0*N_lat/π * sin.(π*l/N_lat)) .+ p[2]
    p_0 = [1.0, 0.0]

    S_l_vec = [calc_entanglement_entropy(ψ, l) for l in l_vec]
    # n_vec = expect(ψ, "n")
    # avg_n = sum(n_vec) / N_lat
    # S_l_vec = [2.0*calc_entanglement_entropy(ψ, l)*avg_n/(n_vec[l]+n_vec[l+1]) for l in l_vec]

    fit = curve_fit(S, l_vec, S_l_vec, p_0)
    c = fit.param[1]
    # @show fit.resid
    c_resid = sum(abs.(fit.resid))

    # plt.plot(l_vec, S_l_vec, color="black")
    # plt.plot(l_vec, [S(l, fit.param) for l in l_vec], color="red")

    return c, c_resid
end # function
##### END OF EXPECTATION VALUES #####

##### HAMILTONIANS #####
function create_BC_sum_ranges(N_lat, BC; max_dist=1)
    if BC == "OBC"
        sum_ranges = [1:(N_lat-j) for j in 1:max_dist]
    elseif BC == "PBC"
        sum_ranges = [1:N_lat for j in 1:max_dist]
    else
        throw(ArgumentError("Invalid boundary condition."))
    end # if

    return sum_ranges
end # function

function calc_BH_Ham(sites, N_lat, conserveParticleNumber, μ; J, U, BC="OBC")
    sum_ranges = create_BC_sum_ranges(N_lat, BC; max_dist=1)

    os = OpSum()
    for i in sum_ranges[1]
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

function calc_brick_wall_Ham(sites, N_lat, conserveParticleNumber, μ; J_1, J_2, g_0, g_x, g_z, G_000, G_001, G_011, BC="OBC")
    sum_ranges = create_BC_sum_ranges(N_lat, BC; max_dist=2)

    os = OpSum()
    for i in sum_ranges[1]
        os += -J_1, "adag", i, "a", mod1(i+1, N_lat)
        os += -J_1, "adag", mod1(i+1, N_lat), "a", i
    end # for
    for i in sum_ranges[2]
        os += -J_2, "adag", i, "a", mod1(i+2, N_lat)
        os += -J_2, "adag", mod1(i+2, N_lat), "a", i
    end # for
    for i in 1:N_lat
        os += (g_0 + 0.5*g_x)*G_000, "n", i, "n", i
        os += -(g_0 + 0.5*g_x)*G_000, "n", i
    end # for
    for i in sum_ranges[1]
        ip1 = mod1(i+1, N_lat)

        os += 2*g_0*G_011, "n", i, "n", ip1

        os += g_z*G_001, "adag", ip1, "adag", ip1, "a", ip1, "a", i
        os += g_z*G_001, "adag", i, "adag", i, "a", i, "a", ip1
        os += g_z*G_001, "adag", i, "adag", ip1, "a", ip1, "a", ip1
        os += g_z*G_001, "adag", ip1, "adag", i, "a", i, "a", i

        os += -0.5*g_x*G_011, "adag", ip1, "adag", ip1, "a", i, "a", i
        os += -0.5*g_x*G_011, "adag", i, "adag", i, "a", ip1, "a", ip1
    end # for
    if !conserveParticleNumber
        for i in 1:N_lat
            os += -μ, "n", i
        end # for
    end # if
    Ham = MPO(os, sites)

    return Ham
end # function

# NOTE: Zhou considered MFT.
function calc_Zhou_PSF_deep_lat_Ham(sites, N_lat, conserveParticleNumber, μ; J, U, BC="OBC")
    sum_ranges = create_BC_sum_ranges(N_lat, BC; max_dist=1)

    # ZhouPRA2009
    os = OpSum()
    for i in sum_ranges[1]
        ip1 = mod1(i+1, N_lat)

        # os += -1e-5, "adag", i, "a", i+1
        # os += -1e-5, "adag", i+1, "a", i
        os += -J, "adag", i, "adag", i, "a", ip1, "a", ip1
        os += -J, "adag", ip1, "adag", ip1, "a", i, "a", i
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

function calc_Huber_interacting_Creutz_ladder(sites, N_lat, conserveParticleNumber, μ; t, U, m, ϵ, δ, BC="OBC")
    sum_ranges = create_BC_sum_ranges(N_lat, BC; max_dist=2)

    # HuberPRB2013
    os = OpSum()
    for i in 1:N_lat
        os += U/4, "n", i, "n", i
        os += -U/4, "n", i
    end # for
    for i in sum_ranges[1]
        os += U/2, "n", i, "n", mod1(i+1, N_lat)
    end # for
    for i in sum_ranges[1]
        os += -U/8, "adag", i, "adag", i, "a", mod1(i+1,N_lat), "a", mod1(i+1,N_lat)
        os += -U/8, "adag", mod1(i+1,N_lat), "adag", mod1(i+1,N_lat), "a", i, "a", i
    end # for
    for i in sum_ranges[1]
        os += -(m+2δ)*t/2, "adag", i, "a", mod1(i+1,N_lat)
        os += -(m+2δ)*t/2, "adag", mod1(i+1,N_lat), "a", i
    end # for
    for i in sum_ranges[2]
        os += -(ϵ+0.5*δ^2)*t/2, "adag", i, "a", mod1(i+2,N_lat)
        os += -(ϵ+0.5*δ^2)*t/2, "adag", mod1(i+2,N_lat), "a", i
    end # for
    if !conserveParticleNumber
        for i in 1:N_lat
            os += -μ, "n", i
        end # for
    end # if
    Ham = MPO(os, sites)

    return Ham
end # function

function calc_extended_int_BH_Ham(sites, N_lat, conserveParticleNumber, μ; J_1, J_2, U, V, P, D, BC="OBC")
    sum_ranges = create_BC_sum_ranges(N_lat, BC; max_dist=2)

    os = OpSum()
    for i in sum_ranges[1]
        os += -J_1, "adag", i, "a", mod1(i+1, N_lat)
        os += -J_1, "adag", mod1(i+1, N_lat), "a", i
    end # for
    for i in sum_ranges[2]
        os += -J_2, "adag", i, "a", mod1(i+2, N_lat)
        os += -J_2, "adag", mod1(i+2, N_lat), "a", i
    end # for
    for i in 1:N_lat
        os += U/2, "n", i, "n", i
        os += -U/2, "n", i
    end # for
    for i in sum_ranges[1]
        ip1 = mod1(i+1, N_lat)

        os += V, "n", i, "n", ip1

        os += -D, "adag", ip1, "adag", ip1, "a", ip1, "a", i
        os += -D, "adag", i, "adag", i, "a", i, "a", ip1

        os += -P, "adag", ip1, "adag", ip1, "a", i, "a", i
        os += -P, "adag", i, "adag", i, "a", ip1, "a", ip1
    end # for
    if !conserveParticleNumber
        for i in 1:N_lat
            os += -μ, "n", i
        end # for
    end # if
    Ham = MPO(os, sites)

    return Ham
end # function
##### END OF HAMILTONIANS #####
end # module
