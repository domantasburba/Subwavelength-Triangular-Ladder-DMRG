module QOPack

# using Reexport
# using Jive

# Write your package code here.
include("./Utility.jl")
include("./Plots.jl")
include("./Math.jl")
include("./Fitting.jl")
# include("./TimeEvolution.jl")
include("./Polynomials.jl")
include("./PlaneWaves.jl")
include("./Operators.jl")

include("./QMB/DMRG.jl")
include("./QMB/PhaseDiagram.jl")
include("./QMB/ExactDiagonalizationBosons.jl")
include("./QMB/ExactDiagonalizationFermions.jl")
include("./QMB/KnownResults.jl")

# runtests("../test/")
include("../test/runtests.jl")

export sparse_display, remove_insignificant_terms!, remove_insignificant_terms, save_animation, save_parameter_TXT, save_script_TXT, save_stdout_TXT
export get_fig_ax, set_plot_defaults, save_plot, plot_y_vs_x, plot_ys_vs_x, plotN_ys_vs_x, plotN_grid_ys_vs_x, plot_u_vs_xy, plot_u_w_streamlines_vs_xy, plotN_us_vs_xy, plotN_us_w_streamlines_vs_xy
export ⊗, sgn, Kronecker_δ, Levi_Civita_ϵ, inBounds, avg, std_dev, hsum, meshgrid, sqrt_Gaussian, Gaussian, comm, anticomm, id, add_ijv!, Gram_Schmidt, invert_dict, sum_non_diag, split_block_diag, find_split_idxs, calc_Dx, calc_D2x, Pauli_x, Pauli_y, Pauli_z, Pauli_i, create_S_plus, create_S_minus, create_S_x, create_S_y, create_S_z, create_S_squared, calc_orthogonal_vectors_3D, normalize_state, dx_y_FD, d2x_y_FD, dx_y_FFT, d2x_y_FFT, toeplitz, create_periodic_diff_matrix, create_periodic_diff2_matrix, create_x_Cheb, create_Cheb_diff_matrix, trapz, simps, simps_2D, x2index, x2index_floor, x2index_ceil, translate_y, is_approx_equal, check_if_real, check_if_imag, check_if_int, check_if_Hermitian, check_if_close
export power_law_fit, power_law_w_const_fit, ln_power_law_fit, ln_power_law_w_const_fit, exp_law_fit, exp_law_w_const_fit, ln_exp_law_fit, ln_exp_law_w_const_fit, calc_fit_1D, calc_fit_1D_MC
# export solve_Schrodinger
export eval_poly, eval_poly_xs, eval_poly_Horner, add_polynomial_coeffs, mul_xp_polynomial_coeffs, Dx_polynomial_coeffs, calc_Hermite_polynomial_coeffs_vec
export PWRep, index_dict, add_repeating, PW_id, PW_id_D, PW_exp_inqx, PW_cos_nqx, PW_sin_nqx, PW_exp_iqx, PW_exp_imqx, PW_cos_qx, PW_sin_qx, convert_PW_to_nested_vec, convert_nested_vec_to_PW, convert_PW_to_matrix, convert_PW_to_vector, convert_vector_to_PW, convert_PW_to_function_r, convert_vector_to_function_r, convert_full_ket_to_function_r, approx_function_r_with_PW, approx_function_r_with_PW_vector, calc_function_r_PW_approx
export op_to_lattice_op, op_to_lattice_ops, lattice_ops_to_collective_op, op_to_collective_op

# export try_add_particle, create_ψ0, create_homogeneous_ψ0, create_pair_ψ0, create_Mott_ψ0, create_BC_sum_ranges, calc_BH_Ham, calc_brick_wall_Ham, calc_Zhou_PSF_deep_lat_Ham, calc_Huber_interacting_Creutz_ladder, calc_extended_int_BH_Ham
# export create_boson_basis, create_boson_ext_basis, create_boson_Ham_chem_pot, create_boson_Ham_int, _calc_open_boson_Ham_int_Vj_mat_el, create_boson_Ham_int, create_boson_Ham_tunneling, create_boson_Ham_tunneling, create_boson_n_j, create_boson_g1_ij, create_boson_κ_j, create_boson_κ2_ij, create_BH_Ham, create_EBH_Ham, create_Ham, calc_expect_n_arr, calc_expect_Δn_arr, calc_expect_δN
export calc_1D_BH_energy_per_site_strong_U_expansion, calc_1D_BH_energy_per_site_weak_U_expansion

# @reexport using Utility
# @reexport using Math
# @reexport using TimeEvolution
# println(Pauli_x())

# import .Utility as Util
# import .Math as Math
# import .TimeEvolution as TimeEvo
# export Util
# export Math
# export TimeEvo

end # module
