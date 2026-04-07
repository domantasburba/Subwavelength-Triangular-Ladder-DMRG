using LsqFit

function power_law_fit(L, p)
    return p[1]./L.^p[2]
end # function

function power_law_w_const_fit(L, p)
    return p[3] .+ p[1]./L.^p[2]
end # function

function ln_power_law_fit(L, p)
    return p[1] .- p[2].*log.(L)
end # function

function ln_power_law_w_const_fit(L, p)
    return log.(p[3] .+ p[1]./L.^p[2])
end # function

function exp_law_fit(L, p)
    return p[1].*exp.(-L./p[2])
end # function

function exp_law_w_const_fit(L, p)
    return p[3] .+ p[1].*exp.(-L./p[2])
end # function

function ln_exp_law_fit(L, p)
    return p[1] .- L./p[2]
end # function

function ln_exp_law_w_const_fit(L, p)
    return log.(p[3] .+ p[1].*exp.(-L./p[2]))
end # function

function calc_fit_1D(y_L, x_vec, y_vec, p_0; lower=nothing, upper=nothing)
    @assert length(x_vec) == length(y_vec)
    ERR_VAL = -9999999.9999999

    try
        if (lower == nothing) || (upper == nothing)
            fit = curve_fit(y_L, x_vec, y_vec, p_0)
        else
            fit = curve_fit(y_L, x_vec, y_vec, p_0; lower=lower, upper=upper)
        end # if
        y_fit = fit.param
        y_resid = fit.resid
        y_fit_vec = y_L(x_vec, y_fit)

        return y_fit, y_resid, y_fit_vec
    catch e
        N_p = length(p_0)
        N_data = length(x_vec)

        y_fit = [ERR_VAL for i in 1:N_p]
        y_resid = [ERR_VAL for i in 1:N_data]
        y_fit_vec = [ERR_VAL for i in 1:N_data]

        println("WARNING: Error in curve fit.")

        return y_fit, y_resid, y_fit_vec
    end # try/catch
end # function

function calc_fit_1D_MC(y_L, x_vec, y_vec, lower, upper; N_p0=100)
    @assert length(lower) == length(upper)
    N_p = length(lower)

    calc_total_resid = y_resid -> sum(abs2.(y_resid))

    cur_y_fit, cur_y_resid, cur_y_fit_vec = calc_fit_1D(y_L, x_vec, y_vec, lower; lower=lower, upper=upper)
    cur_total_resid = calc_total_resid(cur_y_resid)
    for i in 1:N_p0
        p_0 = [lower[j] + (upper[j] - lower[j]) * rand() for j in 1:N_p]

        # y_fit, y_resid, y_fit_vec = calc_fit_1D(y_L, x_vec, y_vec, p_0)
        y_fit, y_resid, y_fit_vec = calc_fit_1D(y_L, x_vec, y_vec, p_0; lower=lower, upper=upper)
        total_resid = calc_total_resid(y_resid)

        if total_resid < cur_total_resid
            cur_y_fit = y_fit
            cur_y_resid = y_resid
            cur_y_fit_vec = y_fit_vec
            cur_total_resid = calc_total_resid(cur_y_resid)
        end # if
    end # for

    return cur_y_fit, cur_y_resid, cur_y_fit_vec
end # function
