function calc_1D_BH_energy_per_site_strong_U_expansion(J, U)
    # DamskiZakrzewskiPRA2006, strong interaction expansion up to 14th order
    x = J/U

    return 4.0*U * (
        -x^2 + x^4 + 68.0/9.0*x^6 - 1267.0/81.0*x^8 +
        44171.0/1458.0*x^10 - 4902596.0/6561.0*x^12 -
        8020902135607.0/2645395200.0*x^14
    )
end # function

function calc_1D_BH_energy_per_site_weak_U_expansion(J, U)
    return J * (-2.0 + U/(2.0*J) - sqrt(2.0)*(U/J)^(3/2)/(3.0*π))
end # function
