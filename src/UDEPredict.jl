using HeatExchangeUDE
using DifferentialEquations
using Statistics

#Prediction as a continuation of the training data 
function UDE_predict(θ,tsteps,Th,Tc)

    #Load Steady State Data
    T0,_,_ = load_steady()

    #Define Neural Network
    hidden_layers = 1
    nodes_per_layer = 16
    NN,_,st = setup_NN(hidden_layers,nodes_per_layer)

    #Build Parameter Vector
    p = pVecBuilder(
        R=R_NN,
        model=NN,
        θ=θ,
        st=st,
        τ=1.0,
        Th_max = maximum(Th),
        Tc_max = maximum(Tc)
    )

    tspan = (0,tsteps[end])

    R0 = fill(0.0,p.N)
    u0 = vcat(T0,R0)

    prob = ODEProblem(EnergyBalance!,u0,tspan,p)
    sol = solve(prob,Rodas5P(),saveat=tsteps)
    return sol
end

#Prediction with end of training data as initial condition

#Prediction error calculator 
function prediction_error(Th_ground_truth,Tc_ground_truth,Th_model,Tc_model,tsteps)
    Th_error = 100*abs.(Th_model.-Th_ground_truth)./(Th_ground_truth.+273.15)
    Tc_error = 100*abs.(Tc_model.-Tc_ground_truth)./(Tc_ground_truth.+273.15)

    spacial_avg_errors = Float64[]

    for i in 1:length(tsteps)
        current_Th_error = Th_error[:,i]
        current_Tc_error = Tc_error[:,i]
        push!(spacial_avg_errors,mean(vcat(current_Th_error,current_Tc_error)))
    end

    error_mean = mean(vcat(Th_error,Tc_error))
    mean_accuracy = 100-error_mean
    return Th_error,Tc_error,mean_accuracy,spacial_avg_errors
end