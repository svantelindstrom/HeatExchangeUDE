#Training the Neural Network to predict the fouling 
using HeatExchangeUDE
using StableRNGs, Lux, ComponentArrays
using Optimization, OptimizationOptimisers, OptimizationOptimJL, Zygote, Enzyme
using Optimisers, Optim
using DrWatson, JLD2


function setup_NN(hidden_layers::Int,nodes_per_layer::Int)
    NN = define_NN(hidden_layers,nodes_per_layer)

    rng = StableRNG(123)

    θ,st = Lux.setup(rng,NN)
    θ = ComponentArray(θ) |> f64
    return NN,θ,st
end

function LossOptim(lossFunc,p;max_iters_adam::Int=1000,learn_rate=0.001,max_iters_BFGS::Int=500)

    losses = Float64[]
    cb = (θ,l) -> begin
        push!(losses,l)
        if mod(length(losses),1)==0
            println("Iteration:",length(losses),"|Loss:",l)
        end
        return false
    end

    optf = OptimizationFunction((x,_)->lossFunc(x),Optimization.AutoZygote())

    optprob_adam = OptimizationProblem(optf,p.θ)
    adam_result = solve(optprob_adam,Optimisers.Adam(learn_rate),maxiters=max_iters_adam,callback = cb)

    optprob_BFGS =remake(optprob_adam,u0=adam_result.u)
    LBFGS_result = solve(optprob_BFGS,Optim.LBFGS(),maxiters=max_iters_BFGS,callback = cb)

    return (adam=adam_result,LBFGS=LBFGS_result,history = losses)
end

function R_NN(t,Th,Tc,p)
    states = vcat(reshape(Th, 1, :), reshape(Tc, 1, :))
    t_arr = [t for _ in 1:1, _ in 1:length(Tc)]
    input_NN = vcat(states, t_arr)

    R_pred,_ = p.model(input_NN,p.θ,p.st)
    
    return vec(R_pred)
end


function run_training()
    #Defining hyperparameters of Neural Network
    hidden_layers = 5
    nodes_per_layer = 5
    #Defining Neural Network
    NN,θ,st = setup_NN(hidden_layers,nodes_per_layer)

    #Constructing parameter tuple
    p = pVecBuilder(
        R = R_NN,
        model = NN,
        θ=θ,
        st=st,
    )
    
    #Data Loading 

    #Load Steady State data 
    steady_path = datadir("exp_raw","steady_state_data.jld2")
    steady_data = load(steady_path)
    u0 = steady_data["u0"]

    #Load Ground Truth Data
    true_path = datadir("exp_raw","ground_truth_data.jld2")
    true_data = load(true_path)
    tspan = true_data["tspan"]
    tsteps = true_data["tsteps"]
    Th = true_data["Th"]
    Tc = true_data["Tc"]
    ground_truth_data = vcat(Th,Tc)

    #Defining the closure for loss to be able to call loss with θ as the only argument 
    loss_evaluation = loss_function(p,u0,tspan,tsteps,ground_truth_data)
    results = LossOptim(loss_evaluation,p,max_iters_adam=5,max_iters_BFGS=5)

    return results
end