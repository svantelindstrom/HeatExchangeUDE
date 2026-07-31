using DifferentialEquations
using Plots
using HeatExchangeUDE
using JLD2
using DrWatson

function Kern_Seaton(t,Th,Tc,p)
    R = p.R0*(1-exp(-p.beta*t))
    return R
end

function steady_state()
    no_fouling(t,Th,Tc,p) = 0.0

    #Parameters to find steady state operation with no fouling
    p_steady = pVecBuilder(R = no_fouling,τ=1.0)
    tspan_steady = (0.0, 5e3)
    u0_h_steady = fill(p_steady.T_h_in,p_steady.N)
    u0_c_steady = fill(p_steady.T_c_in,p_steady.N)
    u0_steady = vcat(u0_h_steady,u0_c_steady)

    prob_steady = ODEProblem(EnergyBalance!,u0_steady,tspan_steady, p_steady)
    solution_steady = solve(prob_steady, AutoTsit5(Rosenbrock23()))
    return solution_steady
end

function true_dataset(solution_steady)
    #Constructing the ground truth dataset
    u0_true = Array(solution_steady.u[end])

    p_true = pVecBuilder(R = Kern_Seaton,τ=1.0)
    tspan_true = (0.0, p_true.final_time)

    prob_true = ODEProblem(EnergyBalance!,u0_true,tspan_true,p_true)
    solution_true = solve(prob_true,AutoTsit5(Rosenbrock23()),saveat = p_true.dt)

    #Exporting final solution
    u_true = Array(solution_true)
    t_true = Array(solution_true.t)
    return u_true, t_true,p_true.N,p_true.L
end

function save_true(u_true,t_true,N,L)
    ground_truth_dict = Dict(
        "Th" => u_true[1:N,:],
        "Tc" => u_true[N+1:2*N,:],
        "tspan"=> (t_true[1],t_true[end]),
        "tsteps"=>t_true,
        "N"=>N,
        "L"=>L
    )

    file_path = datadir("exp_raw","ground_truth_data.jld2")
    safesave(file_path,ground_truth_dict)
end

function save_steady(solution_steady)
    u_steady = Array(solution_steady)
    u0 = u_steady[:,end]
    steady_dict = Dict(
        "u0"=>u0
    )
    
    file_path = datadir("exp_raw","steady_state_data.jld2")
    safesave(file_path,steady_dict)
end

function GenerateDataset()
    solution_steady = steady_state()
    u_true,t_true,N,L = true_dataset(solution_steady)
    save_steady(solution_steady)
    save_true(u_true,t_true,N,L)
end






