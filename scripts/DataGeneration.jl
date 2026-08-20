using DifferentialEquations
using Plots
using HeatExchangeUDE
using JLD2
using DrWatson
using StableRNGs, Random

struct NoNoise end
struct GaussianNoise{T<:Real,R<:AbstractRNG}
    std::T
    rng::R
end

GaussianNoise(std;rng) = GaussianNoise(std,rng)


add_noise(u,::NoNoise) = u
add_noise(u,n::GaussianNoise) = u .+ n.std .* randn(n.rng,size(u))

function Kern_Seaton(t,p)
    R = p.R0*(1-exp(-p.beta*t))
    return R
end

function steady_state()
    no_fouling(t,p) = 0.0

    #Parameters to find steady state operation with no fouling
    p_steady = pVecBuilder(R = no_fouling,τ=1.0)
    tspan_steady = (0.0, 5e3)
    u0_h_steady = fill(p_steady.T_h_in,p_steady.N)
    u0_c_steady = fill(p_steady.T_c_in,p_steady.N)
    u0_steady = vcat(u0_h_steady,u0_c_steady)

    prob_steady = ODEProblem(EnergyBalance!,u0_steady,tspan_steady, p_steady)
    solution_steady = solve(prob_steady, AutoTsit5(Rosenbrock23()))
    steady_sol_arr = Array(solution_steady)
    return (steady_sol_arr,p_steady.N,p_steady.L)
end

function true_dataset(solution_steady,prop_train,noise)
    #Constructing the ground truth dataset
    u_steady,_,_ = solution_steady
    u0_true = u_steady[:,end]

    p_true = pVecBuilder(R = Kern_Seaton,τ=1.0,final_time = 2e6*prop_train,time_points = round(Int,300*prop_train))
    tspan_true = (0.0, p_true.final_time)

    prob_true = ODEProblem(EnergyBalance!,u0_true,tspan_true,p_true)
    solution_true = solve(prob_true,AutoTsit5(Rosenbrock23()),saveat = p_true.dt)

    #Exporting final solution
    u_true = Array(solution_true)
    t_true = Array(solution_true.t)

    u_true = add_noise(u_true,noise)
    return u_true, t_true,p_true.N,p_true.L
end

function GenerateDataset(filename;prop_train::Float64=1.0,noise=NoNoise())
    solution_steady = steady_state()
    u_true,t_true,N,L = true_dataset(solution_steady,prop_train,noise)
    save_steady(solution_steady,"steady_state_data")
    save_true(u_true,t_true,N,L,filename)  
end

function GenerateTrainingTestData()
    proportion_train = 0.8
    rng = StableRNG(123)
    δ=1.0

    train_name = "1.0_noisy_training_set"
    test_name = "1.0_noisy_test_set"

    #Generate Training Data
    GenerateDataset(train_name;prop_train=proportion_train,noise=GaussianNoise(δ;rng))

    #Generate Test Data
    GenerateDataset(test_name;noise=GaussianNoise(δ;rng))
end






