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

"""
    add_noise(u,::NoNoise)
    add_noise(u,n::GaussianNoise)

Adds noise based on the type of the second argument.

This one line function takes an input and modifies it based on the noise type of the second
arguments defined by the NoNoise and GaussianNoise structs.

# Arguments
- `u::Matrix{Float64}`: Clean data inputted for noise to be added to.
- `::NoNoise`: Output the clean data.
- `n::GaussianNoise`: Adds gaussian noise with a given standard deviation and rng seed.

# Returns
- `AbstractArray`: Resulting dataset based on the noise model applied.
"""
add_noise(u,::NoNoise) = u
add_noise(u,n::GaussianNoise) = u .+ n.std .* randn(n.rng,size(u))

"""
    Kern_Seaton(t,p)

Kern Seaton fouling model.

# Arguments
- `t::Vector{Float64}`: Vector of timesteps.
- `p::NamedTuple`: Parameter tuple.

# Returns
- `R::Vector{Float64}`: Vector of amount of fouling over time.
"""
Kern_Seaton(t,p) = p.R0*(1-exp(-p.beta*t))

"""
    no_fouling(t,p)

No fouling option for parameter tuple and generation of steady state dataset.

This function takes t and p as inputs as required by the way EnergyBalance! treats the function.

# Arguments
- `t::Vector{Float64}`: Vector of timesteps.
- `p::NamedTuple`: Parameter tuple.

# Returns
- `Float64`: Returns 0.
"""
no_fouling(t,p) = 0.0

"""
    steady_state()

Generates the steady state dataset as an initial condition for the UDE.

# Arguments
- `nothing`: This function takes no inputs.

# Returns
- `steady_state_sol::Matrix{Float64}`: Steady state solution of the energy balance.
- `p_steady.N::Float64`: Number of spacial nodes.
- `p_steady.L::Float64`: Physical length.
"""
function steady_state()
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

"""
    true_dataset(solution_steady,prop_train,noise)

Generate the true dataset based on the initial condition.

This function has the option of adding noise and the proportion of training to test data.

# Arguments
- `solution_steady::Matrix{Float64}`: Steady state solution.
- `prop_train::Float64`: Proportion of entire dataset used for training.
- `noise::Function`: Function describing how noise is added to clean data.

# Returns
- `u_true::Matrix{Float64}`: True dataset temperatures.
- `t_true::Vector{Float64}`: True dataset timesteps.
- `p_true.N::Float64`: Number of spacial nodes.
- `p_true.L::Float64`: Physical length
"""
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

"""
    GenerateDataset(filename;prop_train::Float64=1.0,noise=NoNoise())

Function to generate the true dataset.

# Arguments
- `filename::AbstractString`: Name of file which solution is saved to.
- `prop_train::Float64`: Proportion of timespan used for training. Defaults to 1.0.
- `noise::Function`: Noise function applied to clean dataset. Defaults to NoNoise().

# Returns
- `nothing`: No return, solution saved to filename.jld2.
"""
function GenerateDataset(filename;prop_train::Float64=1.0,noise=NoNoise())
    solution_steady = steady_state()
    u_true,t_true,N,L = true_dataset(solution_steady,prop_train,noise)
    save_steady(solution_steady,"steady_state_data")
    save_true(u_true,t_true,N,L,filename)  
end

"""
    GenerateTrainingTestData()  

This function generates both the true training and  test data.

# Arguments
- `nothing`: This function takes no inputs.

# Returns
- `nothing`: This function returns nothing, files saved to train_name.jld2 and test_name.jld2.
"""
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






