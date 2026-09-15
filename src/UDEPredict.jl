using HeatExchangeUDE
using DifferentialEquations
using Statistics

"""
    UDE_predict(θ,tsteps,Th,Tc)

Calculate predicted temperatures using the trained neural network weights(θ).

This function allows the trained neural network weights to be used in the prediction of temperatures 
at specific time points(tsteps) allowing both the training set and extrapolation of the test set to be 
predicted. It sets up the neural network and an ODE problem using the EnergyBalance! function to solve 
the ODE problem with the fouling rate predicted by the trained weights. It requires the true temperatures
of the training set for normalisation within the R_NN fouling rate function.

# Arguments
- `θ::ComponentVector{Float64}`: Trained model weights and biases obtained from the optimisation problem in the LossOptim function
- `tsteps::Vector{Float64}`: Discrete time steps taken by the solver when generating the true datasets
- `Th:Matrix{Float64}`: Hot temperature of the true data
- `Tc:Matrix{Float64}`: Cold temperature of the true data

# Returns
- `sol::ODESolution`: Predicted temperatures and fouling obtained by solving the UDE
"""
function UDE_predict(θ,tsteps,Th,Tc)

    #Load Steady State Data
    T0,_,_ = load_steady()

    #Define Neural Network
    hidden_layers = 1
    nodes_per_layer = 16
    NN,_,st = setup_NN(hidden_layers,nodes_per_layer)

    #Build Parameter Vector
    p = pVecBuilder(
        R=R_NN!,
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

"""
    prediction_error(Th_ground_truth,Tc_ground_truth,Th_model,Tc_model,tsteps)

Calculates error between prediction and true data for plotting.

This function takes in the temperatures of the true and model datasets, using them 
to find the spacial average error for each time step. It also calculates the mean error
over all time steps to give the percentage accuracy of the model.

# Arguments
- `Th_ground_truth::Matrix{Float64}`: Hot temperature of the true data 
- `Tc_ground_truth::Matrix{Float64}`: Cold temperature of the true data
- `Th_model::Matrix{Float64}`: Hot temperature of the models prediction
- `Tc_model::Matrix{Float64}`: Cold temperature of the models prediction
- `tsteps::Vector{Float64}`: Time steps of the datasets

# Returns
- `Th_error::Matrix{Float64}`: Errors in the hot stream
- `Tc_error::Matrix{Float64}`: Errors in the cold stream
- `mean_accuracy::Float64`: average accuracy of the models prediction
- `spacial_avg_errors::Vector{Float64}`: Spacial average error at each time step
- `error_mean::Float64`: Average error of the model's prediction in Kelvin
"""
function prediction_error(Th_ground_truth,Tc_ground_truth,Th_model,Tc_model,tsteps)
    Th_error_abs = abs.(Th_model.-Th_ground_truth)
    Tc_error_abs = abs.(Tc_model.-Tc_ground_truth)

    Th_error = 100*Th_error_abs./(Th_ground_truth.+273.15)
    Tc_error = 100*Tc_error_abs./(Tc_ground_truth.+273.15)
    

    spacial_avg_errors = Float64[]

    for i in 1:length(tsteps)
        current_Th_error = Th_error[:,i]
        current_Tc_error = Tc_error[:,i]
        push!(spacial_avg_errors,mean(vcat(current_Th_error,current_Tc_error)))
    end

    error_mean = mean(vcat(Th_error_abs,Tc_error_abs))
    percentage_error_mean = mean(vcat(Th_error,Tc_error))
    mean_accuracy = 100-percentage_error_mean
    return Th_error,Tc_error,mean_accuracy,spacial_avg_errors,error_mean
end