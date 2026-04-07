# module TimeEvolution

using DifferentialEquations

# export solve_Schrodinger

function solve_Schrodinger(calc_Hamiltonian, ket_initial, time_arr, reltol=1e-8, abstol=1e-12)
    function Schrodinger!(ket, p, t)
        Ham = calc_Hamiltonian(t)
        dt_ket = -im .* Ham * ket
    end

    N_time = length(time_arr)
    N_ket = length(ket_initial)
    time_span = (time_arr[1], time_arr[end])

    problem = ODEProblem(Schrodinger!, ket_initial, time_span, reltol=reltol, abstol=abstol)
    # sol = solve(prob, saveat=time_arr)
    sol = solve(problem)

    ket = Array{typeof(sol[1, 1])}(undef, N_time, N_ket)
    for i in 1:N_time
        # ket[i, :] = sol[:, i]
        ket[i, :] = sol(time_arr[i])
    end

    return ket
end

# end # module
