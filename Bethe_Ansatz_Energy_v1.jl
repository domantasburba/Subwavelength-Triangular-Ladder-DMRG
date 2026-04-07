using NPZ
using Roots

import PyPlot as plt
ENV["MPLBACKEND"] = "Qt5Agg"
rcParams = plt.PyDict(plt.matplotlib."rcParams")
rcParams["figure.figsize"] = [9, 6]
rcParams["font.size"] = 16
set_interactive_plt = bool -> (bool ? plt.ion() : plt.ioff())
set_interactive_plt(false)  # true - shows plots; false - doesn't show plots
# plt.ioff()  # Doesn't show plots
# plt.ion()  # Shows plots

using QOPack

mutable struct BetheAnsatzParameters
    G011_G000_ratio::Real

    gx_g0_ratio_span::Vector{Real}
    N_gx_g0::Int
    gx_g0_ratio_vec::Vector{Real}

    N_sum::Int

    function BetheAnsatzParameters(G011_G000_ratio, gx_g0_ratio_span, N_gx_g0, N_sum)
        gx_g0_ratio_vec = LinRange(gx_g0_ratio_span..., N_gx_g0)

        new(G011_G000_ratio, gx_g0_ratio_span, N_gx_g0, gx_g0_ratio_vec, N_sum)
    end # function
end # function

function calc_e_MI(gx_g0_ratio)
    e_MI = -(1.0 + 0.5 * gx_g0_ratio)

    return e_MI
end # function

function full_calc_e_AFM(gx_g0_ratio, G011_G000_ratio, N_sum)
    ϕ = acosh(-4.0 / gx_g0_ratio)

    e_sum = 0.0
    for n in 1:N_sum
        e_sum += 1.0 / (exp(2*n*ϕ) + 1.0)
    end # for

    e_AFM = 2.0*G011_G000_ratio * (1.0 + gx_g0_ratio * sinh(ϕ) * (0.5 + 2.0 * e_sum))

    return e_AFM
end # function

function routine_Bethe_ansatz_energy(par)
    G011_G000_ratio, gx_g0_ratio_span, N_gx_g0, gx_g0_ratio_vec, N_sum = par.G011_G000_ratio, par.gx_g0_ratio_span, par.N_gx_g0, par.gx_g0_ratio_vec, par.N_sum

    e_MI_vec = [calc_e_MI(gx_g0_ratio) for gx_g0_ratio in gx_g0_ratio_vec]

    calc_e_AFM = gx_g0_ratio -> full_calc_e_AFM(gx_g0_ratio, G011_G000_ratio, N_sum)
    e_AFM_vec = [calc_e_AFM(gx_g0_ratio) for gx_g0_ratio in gx_g0_ratio_vec]

    calc_e_diff = gx_g0_ratio -> (calc_e_AFM(gx_g0_ratio) - calc_e_MI(gx_g0_ratio))
    crit_gx_g0_ratio = find_zero(calc_e_diff, -1.5)
    @show crit_gx_g0_ratio
    @show calc_e_MI(crit_gx_g0_ratio)
    @show calc_e_AFM(crit_gx_g0_ratio)

    g_0 = 1.0
    g_x = crit_gx_g0_ratio * g_0
    G_000 = 1.0
    G_011 = G011_G000_ratio * G_000
    U = (2.0*g_0 + g_x) * G_000
    V = 2.0*g_0 * G_011
    P = 0.5*g_x * G_011
    @show U
    @show V
    @show P

    Bethe_ansatz_E_res = Dict{String, Any}()
    merge!(Bethe_ansatz_E_res, Dict("gx_g0_ratio_vec" => Float64.(gx_g0_ratio_vec)))
    merge!(Bethe_ansatz_E_res, Dict("crit_gx_g0_ratio" => Float64.(crit_gx_g0_ratio)))
    merge!(Bethe_ansatz_E_res, Dict("crit_e" => Float64.(calc_e_MI(crit_gx_g0_ratio))))
    merge!(Bethe_ansatz_E_res, Dict("e_MI_vec" => Float64.(e_MI_vec)))
    merge!(Bethe_ansatz_E_res, Dict("e_AFM_vec" => Float64.(e_AFM_vec)))

    ### SAVING RESULTS ###
    save_path_Bethe = "$(@__DIR__)/NPY/$(Dates.today())/Bethe_Ansatz_E_Res"
    date_string = """$(Dates.format(Dates.now(), "Y-mm-dd!HH.MM.SS"))"""
    mkpath(save_path_Bethe)
    npzwrite("$(save_path_Bethe)/Bethe_Ansatz_E_Res!$(date_string).npz", Bethe_ansatz_E_res)
    ### END OF SAVING RESULTS ###

    plot_ys_vs_x(gx_g0_ratio_vec, [e_MI_vec, e_AFM_vec], @__DIR__;
                 x_label=raw"$g_x$ / $g_0$", y_label=raw"$e$ / $g_0G_{000}$",
                 color_vec=["blue", "red"],
                 saveDirectly=false, file_name="Bethe_ansatz_energy_vs_gx_g0_ratio")
end # function

function main()
    # PARAMETERS
    G011_G000_ratio = 0.1666

    gx_g0_ratio_span = [-3.0, -0.5]
    N_gx_g0 = 500

    N_sum = 50
    # END OF PARAMETERS

    save_parameter_TXT(@__FILE__)
    save_script_TXT(@__FILE__)

    par = BetheAnsatzParameters(G011_G000_ratio, gx_g0_ratio_span, N_gx_g0, N_sum)

    # ROUTINES
    routine_Bethe_ansatz_energy(par)
    # END OF ROUTINES
end # function

@time main()
