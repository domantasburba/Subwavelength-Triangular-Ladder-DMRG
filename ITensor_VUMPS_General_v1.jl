using LinearAlgebra
using ITensors, ITensorMPS
using ITensorInfiniteMPS
using NPZ

base_path = joinpath(pkgdir(ITensorInfiniteMPS), "examples", "vumps", "src")
src_files = ["vumps_subspace_expansion.jl", "entropy.jl"]
for f in src_files
    include(joinpath(base_path, f))
end # for

import PyPlot as plt
ENV["MPLBACKEND"] = "Qt5Agg"
rcParams = plt.PyDict(plt.matplotlib."rcParams")
# rcParams["figure.figsize"] = [12, 8]
rcParams["font.size"] = 16
set_interactive_plt = bool -> (bool ? plt.ion() : plt.ioff())
set_interactive_plt(true)  # true - shows plots; false - doesn't show plots

using QOPack

# Taken from https://itensor.discourse.group/t/vumps-for-multi-site-unit-cells/460/6
# ITensorInfiniteMPS.opsum_infinite(Model("heisenberg"), 3)
# ITensorInfiniteMPS.opsum_infinite(Model("Bose_Hubbard"), 4; J=1.0, U=10.0)

function ITensorInfiniteMPS.unit_cell_terms(::Model"Bose_Hubbard"; J, U)
    os = OpSum()
    os += -J, "adag", 1, "a", 2
    os += -J, "adag", 2, "a", 1
    os += U/2, "n", 1, "n", 1
    os += -U/2, "n", 1

    return os
end # function

function ITensorInfiniteMPS.unit_cell_terms(::Model"Zhou_PSF_deep_lat"; J, U)
    os = OpSum()
    # os += -1e-5, "adag", 1, "a", 2  # Without these terms, DMRG seems to be unstable
    # os += -1e-5, "adag", 2, "a", 1
    os += -J, "adag", 1, "adag", 1, "a", 2, "a", 2
    os += -J, "adag", 2, "adag", 2, "a", 1, "a", 1
    os += U/2, "n", 1, "n", 1
    os += -U/2, "n", 1

    return os
end # function

function ITensorInfiniteMPS.unit_cell_terms(::Model"Int_Bose_Hubbard"; J, U, V, P, D)
    os = OpSum()
    os += -J, "adag", 1, "a", 2
    os += -J, "adag", 2, "a", 1
    os += U/2, "n", 1, "n", 1
    os += -U/2, "n", 1
    os += V, "n", 1, "n", 2
    os += -P, "adag", 1, "adag", 1, "a", 2, "a", 2
    os += -P, "adag", 2, "adag", 2, "a", 1, "a", 1
    os += -D, "adag", 1, "adag", 1, "a", 1, "a", 2
    os += -D, "adag", 2, "adag", 2, "a", 2, "a", 1
    os += -D, "adag", 2, "adag", 1, "a", 1, "a", 1
    os += -D, "adag", 1, "adag", 2, "a", 2, "a", 2

    return os
end # function

function ITensorInfiniteMPS.unit_cell_terms(::Model"Int_Brick_Wall"; g_0, g_x, g_z, G_000, G_001, G_011)
    os = OpSum()

    os += (g_0 + 0.5*g_x)*G_000, "n", 1, "n", 1
    os += -(g_0 + 0.5*g_x)*G_000, "n", 1

    os += 2*g_0*G_011, "n", 1, "n", 2

    os += g_z*G_001, "adag", 1, "adag", 1, "a", 1, "a", 2
    os += g_z*G_001, "adag", 2, "adag", 2, "a", 2, "a", 1
    os += g_z*G_001, "adag", 2, "adag", 1, "a", 1, "a", 1
    os += g_z*G_001, "adag", 1, "adag", 2, "a", 2, "a", 2

    os += -0.5*g_x*G_011, "adag", 1, "adag", 1, "a", 2, "a", 2
    os += -0.5*g_x*G_011, "adag", 2, "adag", 2, "a", 1, "a", 1

    return os
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

function expect_two_site(ψ::InfiniteCanonicalMPS, h::ITensor, n1n2)
    n1, n2 = n1n2
    ϕ = ψ.AL[n1] * ψ.AL[n2] * ψ.C[n2]
    return (noprime(ϕ * h) * dag(ϕ))[]
end # function

function expect_two_site(ψ::InfiniteCanonicalMPS, h::MPO, n1n2)
    return expect_two_site(ψ, prod(h), n1n2)
end # function

function routine_single(model_params, boson_dim, maxdim, cutoff, max_vumps_iters, vumps_tol, outer_iters, localham_type, conserve_qns, eager, N)
    # initstate(n) = "1"
    # initstate(n) = "2"
    initstate(n) = isodd(n) ? "2" : "0"
    # initstate(n) = isodd(n) ? "2" : "2"
    sites = infsiteinds("Boson", N; initstate, conserve_qns, dim=boson_dim)
    ψ = InfMPS(sites, initstate)

    # display(Array(op("adag", sites[1]), sites[1]', sites[1])^2)
    # display(Array(op("adag2", sites[1]), sites[1]', sites[1]))

    # model = Model("Bose_Hubbard")
    # model = Model("Zhou_PSF_deep_lat")
    # model = Model("Int_Bose_Hubbard")
    model = Model("Int_Brick_Wall")

    Ham = InfiniteSum{localham_type}(model, sites; model_params...)

    # Check translational invariance
    println("\nCheck translational invariance of initial infinite MPS")
    @show norm(contract(ψ.AL[1:N]..., ψ.C[N]) - contract(ψ.C[0], ψ.AR[1:N]...))

    outputlevel = 1
    vumps_kwargs = (tol=vumps_tol, maxiter=max_vumps_iters, outputlevel, eager)
    subspace_expansion_kwargs = (cutoff=cutoff, maxdim=maxdim)

    println("\nRun VUMPS on initial product state, unit cell size $N")
    ψ = vumps_subspace_expansion(Ham, ψ; outer_iters, subspace_expansion_kwargs, vumps_kwargs)

    # Check translational invariance
    println("\nCheck translational invariance of optimized infinite MPS")
    @show norm(contract(ψ.AL[1:N]..., ψ.C[N]) - contract(ψ.C[0], ψ.AR[1:N]...))

    ##### EXPECTATION VALUES #####
    println("\n***** EXPECTATION VALUES *****")
    bs = [(1, 2), (2, 3)]
    energy_infinite = map(b -> expect_two_site(ψ, Ham[b], b), bs)

    expect_n = [expect(ψ, "n", n) for n in 1:N]
    @show expect_n

    range = 1:600
    finite_ψ = finite_mps(ψ, range)

    g1_matrix = correlation_matrix(finite_ψ, "adag", "a"; sites=(range .+ 1))
    # display(g1_matrix)
    @show g1_matrix[1, 600]

    g2_matrix = correlation_matrix(finite_ψ, "adag2", "a2"; sites=(range .+ 1))
    @show g2_matrix[1, 600]

    # J = model_params[1]
    # U = model_params[2]
    # @show calc_1D_BH_energy_per_site_strong_U_expansion(J, U)
    # @show calc_1D_BH_energy_per_site_weak_U_expansion(J, U)
    ##### END OF EXPECTATION VALUES #####
end # function

function main()
    # PARAMETERS
    # model_params = (J=1.0, U=-7.0)
    # model_params = (J=1.0, U=1.0)
    # model_params = (J=0.0, U=1.0, V=0.05, P=0.3, D=0.0)
    # model_params = (J=0.0, U=0.05, V=0.05, P=0.3, D=0.0)
    model_params = (g_0=1.0, g_x=-4.1, g_z=0.0, G_000=1.0000, G_001=-0.2122, G_011=0.1666)
    # model_params = (g_0=1.0, g_x=-2.0, g_z=0.0, G_000=1.0000, G_001=-0.5000, G_011=0.5000)

    # VUMPS parameters
    boson_dim = 3
    maxdim = 200  # 100  # 50 # Maximum bond dimension
    cutoff = 1e-12  # 1e-9  # 1e-6 # Singular value cutoff when increasing the bond dimension
    max_vumps_iters = 20  # Maximum number of iterations of the VUMPS algorithm at each bond dimension
    vumps_tol = 1e-9  # 1e-5
    outer_iters = 10  # 5 # Number of times to increase the bond dimension
    localham_type = MPO  # or ITensor
    conserve_qns = true
    eager = true
    N = 2  # Unit cell size
    # End of VUMPS parameters
    # END OF PARAMETERS

    save_parameter_TXT(@__FILE__)
    save_script_TXT(@__FILE__)

    routine_single(model_params, boson_dim, maxdim, cutoff, max_vumps_iters, vumps_tol, outer_iters, localham_type, conserve_qns, eager, N)
end # function

@time main()
