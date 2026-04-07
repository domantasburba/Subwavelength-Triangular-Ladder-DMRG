using ITensors
using ITensorMPS
using Dates
using NPZ
using Random

import PyPlot as plt
ENV["MPLBACKEND"] = "Qt5Agg"
rcParams = plt.PyDict(plt.matplotlib."rcParams")
# rcParams["figure.figsize"] = [12, 8]
rcParams["font.size"] = 16
set_interactive_plt = bool -> (bool ? plt.ion() : plt.ioff())
set_interactive_plt(true)  # true - shows plots; false - doesn't show plots

using QOPack
using QOPack.DMRG

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

# TODO Choose magnetization sector, for now always S_z=0
function create_ψ0(par, sites)
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

function calc_expectation_values(ψ, sites, N_lat)
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

function save_res(res, name)
    save_path = "$(@__DIR__)/NPY/$(Dates.today())/$(name)"
    date_string = """$(Dates.format(Dates.now(), "Y-mm-dd!HH.MM.SS"))"""
    mkpath(save_path)
    npzwrite("$save_path/$(name)!$(date_string).npz", res)
end # function

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
        plt.show()
    end # if
end # function

function plotN_single_correlations(expectation_values; file_name="Single_Correlations")
    fig, ax = plt.subplots(1, 2)
    fig.set_size_inches(12, 6)

    bulk_range, SpSm_vec, SzSz_vec = expectation_values["bulk_range"], expectation_values["SpSm_vec"], expectation_values["SzSz_vec"]
    plot_y_vs_x(bulk_range, SpSm_vec; x_name=raw"$i-j$", y_name=raw"$S_+S_-$", fig_ax=(fig, ax[1]))
    plot_y_vs_x(bulk_range, SzSz_vec; x_name=raw"$i-j$", y_name=raw"$S_zS_z$", fig_ax=(fig, ax[2]))

    set_plot_defaults(fig, ax, addGrid=false)
    plt.subplots_adjust(hspace=0.4)
    save_plot(file_name, @__DIR__, file_type="pdf")
    plt.show()
end # function

# TODO Why not make expectation_values_vec?
# TODO Could be generalized. Would need to somehow iterate through dictionary.
function plotN_phase_curve_Δ(phase_curve_Δ_res; file_name="Phase_Curve_Delta")
    fig, ax = plt.subplots(1, 4)
    fig.set_size_inches(22, 6)

    Δ_vec, mz_vec, SpSm_vec, mz_stag_vec, SzSz_vec = phase_curve_Δ_res["Δ_vec"], phase_curve_Δ_res["mz_vec"], phase_curve_Δ_res["SpSm_vec"], phase_curve_Δ_res["mz_stag_vec"], phase_curve_Δ_res["SzSz_vec"]
    plot_y_vs_x(Δ_vec, mz_vec; x_name=raw"$\Delta$", y_name=raw"$m_z$", fig_ax=(fig, ax[1]))
    plot_y_vs_x(Δ_vec, SpSm_vec; x_name=raw"$\Delta$", y_name=raw"$S_+S_-$", fig_ax=(fig, ax[2]))
    plot_y_vs_x(Δ_vec, mz_stag_vec; x_name=raw"$\Delta$", y_name=raw"$m_{\mathrm{stag}, z}$", fig_ax=(fig, ax[3]))
    plot_y_vs_x(Δ_vec, SzSz_vec; x_name=raw"$\Delta$", y_name=raw"$S_zS_z$", fig_ax=(fig, ax[4]))

    set_plot_defaults(fig, ax, addGrid=false)
    plt.subplots_adjust(hspace=0.4)
    save_plot(file_name, @__DIR__, file_type="pdf")
    plt.show()
end # function

function routine_single(par; showRes=true, showPlot=true)
    N_lat, conserve_Sz, h, J, Δ, N_sweeps, max_dim, cutoff, noise = par.N_lat, par.conserve_Sz, par.h, par.J, par.Δ, par.N_sweeps, par.max_dim, par.cutoff, par.noise

    sites = siteinds("S=1/2", N_lat; conserve_qns=conserve_Sz)

    Ham = calc_XXZ_Hamiltonian(par, sites)

    ψ0 = create_ψ0(par, sites)
    E, ψ = dmrg(Ham, ψ0; nsweeps=N_sweeps, maxdim=max_dim, cutoff=cutoff, noise=noise)
    expectation_values = calc_expectation_values(ψ, sites, N_lat)

    if showRes
        mz, SpSm, mz_stag, SzSz = expectation_values["mz"], expectation_values["SpSm"], expectation_values["mz_stag"], expectation_values["SzSz"]
        @show mz
        @show SpSm
        @show mz_stag
        @show SzSz

        # state = ["Up" for j in 1:N_lat]
        ### NOTE: DMRG doesn't converge to single domain wall solution for Δ ≫ 1
        ### and S_z = 0, not sure what to do about this. Results are very
        ### sensitive to initial state.
        # state = vcat(["Up" for j in 1:(N_lat÷2)], ["Dn" for j in (N_lat÷2+1):N_lat])
        # ψ1 = productMPS(sites, state)
        # @show inner(ψ1', Ham, ψ1)
    end # if

    if showPlot
        plotN_single_correlations(expectation_values)
    end # if

    return E, ψ, expectation_values
end # function

# TODO Could make model_params object, which would be more general. First
# argument could be name of model and this would allow for choosing of models.
# This could allow more sophisticated and general interface but it might be
# simpler to just keep separate scripts for different models.
# TODO There should be more general approach to make phase curves w.r.t. any
# parameter by inputing string matching name of parameter to be varied. Probably
# needs use of macros.
# TODO Phase curve could be used as building block for phase diagram, just like
# single point is used as building block for phase curve.
function routine_phase_curve_Δ(par, Δ_vec; saveRes=true, showPlot=true)
    N_Δ = length(Δ_vec)

    phase_curve_Δ_res = Dict{String, Any}()

    mz_vec = zeros(Float64, N_Δ)
    SpSm_vec = zeros(Float64, N_Δ)
    mz_stag_vec = zeros(Float64, N_Δ)
    SzSz_vec = zeros(Float64, N_Δ)
    for i in 1:N_Δ
    # Threads.@threads for i in 1:N_Δ
        Δ = Δ_vec[i]
        par.Δ = Δ
        E, ψ, expectation_values = routine_single(par; showRes=false, showPlot=false)

        mz, SpSm, mz_stag, SzSz = expectation_values["mz"], expectation_values["SpSm"], expectation_values["mz_stag"], expectation_values["SzSz"]
        mz_vec[i] = mz
        SpSm_vec[i] = SpSm
        mz_stag_vec[i] = mz_stag
        SzSz_vec[i] = SzSz

        println("******* $(i)/$(N_Δ) DONE *******")
    end # for

    if Threads.threadid() == 1
        merge!(phase_curve_Δ_res, Dict("Δ_vec" => Δ_vec))
        merge!(phase_curve_Δ_res, Dict("mz_vec" => mz_vec))
        merge!(phase_curve_Δ_res, Dict("SpSm_vec" => SpSm_vec))
        merge!(phase_curve_Δ_res, Dict("mz_stag_vec" => mz_stag_vec))
        merge!(phase_curve_Δ_res, Dict("SzSz_vec" => SzSz_vec))

        if saveRes
            save_res(phase_curve_Δ_res, "Phase_Curve_Delta")
        end # if

        # if showPlot
        #     plotN_phase_curve_Δ(phase_curve_Δ_res)
        # end # if
    end # if

    return phase_curve_Δ_res
end # function

function routine_read_phase_curve_Δ(data_path)
    phase_curve_Δ_res = npzread(data_path)

    plotN_phase_curve_Δ(phase_curve_Δ_res)
end # function

function main()
    # PARAMETERS
    N_lat = 600

    conserve_Sz = true
    h = 0.0  # Considered to be 0, if conserve_Sz = true

    # 1) J > 0 => Ferromagnet (FM);
    # 2) J < 0 => Anti-ferromagnet (AFM);
    J = 1.0

    # 1) |Δ| > 1, JΔ > 0 => Ferromagnet;
    # 2) |Δ| < 1 => XY;
    # 3) |Δ| > 1, JΔ < 0 => Antiferromagnet (Neel);
    Δ = -2.0

    N_sweeps = 100  # 100  # 100  # 250
    max_dim = 500  # [50, 100, 200]  # [10, 20, 50, 100, 250]  # , 100, 200]  # [100, 200, 300]
    cutoff = 1E-12  # 1E-6  # [1e-12]
    noise = vcat(repeat([1E-3], 10), repeat([1E-5], 10), repeat([0], max(1, N_sweeps-20)))

    # routine_phase_curve_Δ
    Δ_vec = LinRange(-2.0, 2.0, 50)

    # routine_read_phase_curve_Δ
    # data_path = "$(@__DIR__)/Selected NPY/XXZ_Phase_Curve_Delta, anySz, 2025-10-01/Phase_Curve_Delta!2025-10-01!02.33.05.npz"
    # data_path = "$(@__DIR__)/Selected NPY/XXZ_Phase_Curve_Delta, Sz0, 2025-10-02/Phase_Curve_Delta!2025-10-01!19.18.03.npz"
    data_path = "$(@__DIR__)/Selected NPY/XXZ_Phase_Curve_Delta, Sz0, N_lat=500, 2026-02-09/Phase_Curve_Delta!2026-02-07!16.39.15.npz"
    # END OF PARAMETERS

    save_parameter_TXT(@__FILE__)
    save_script_TXT(@__FILE__)

    par = XXZHamiltonianParameters(N_lat, conserve_Sz, h, J, Δ, N_sweeps, max_dim, cutoff, noise)

    # ROUTINES
    # routine_single(par)
    # routine_phase_curve_Δ(par, Δ_vec)
    routine_read_phase_curve_Δ(data_path)
    # END OF ROUTINES
end # function

@time main()
