#Training the Neural Network to predict the fouling 
using HeatExchangeUDE
using StableRNGs, Lux, ComponentArrays
using Optimization, OptimizationOptimisers, OptimizationOptimJL, Zygote, Enzyme
using Optimisers, Optim
using DrWatson, JLD2

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
    adam_weights = adam_result.u

    optprob_BFGS =remake(optprob_adam,u0=adam_result.u)
    LBFGS_result = solve(optprob_BFGS,Optim.LBFGS(),maxiters=max_iters_BFGS,callback = cb)
    LBFGS_weights = LBFGS_result.u

    return (adam=adam_weights,LBFGS=LBFGS_weights,history = losses)
end

function run_training()
    #Defining hyperparameters of Neural Network
    hidden_layers = 1
    nodes_per_layer = 16
    #Defining Neural Network
    NN,θ,st = setup_NN(hidden_layers,nodes_per_layer)

    #Load Ground Truth Data
    Th,Tc,tsteps,_,_,tspan = load_true()

    ground_truth_data = vcat(Th,Tc)

    #Constructing parameter tuple
    p = pVecBuilder(
        R = R_NN,
        model = NN,
        θ=θ,
        st=st,
        Th_max = maximum(Th),
        Tc_max = maximum(Tc)
    )
    
    #Data Loading 

    #Load Steady State data 
    T0,_,_ = load_steady()

    #Initial fouling is 0:
    R0 = fill(0.0,p.N)

    u0 = vcat(T0,R0)

    #Defining the closure for loss to be able to call loss with θ as the only argument 
    loss_evaluation = loss_function(p,u0,tspan,tsteps,ground_truth_data)
    results = LossOptim(loss_evaluation,p,max_iters_adam=100,max_iters_BFGS=50)
    save_NN_results(results)
end