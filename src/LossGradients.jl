using DifferentialEquations
using Zygote
using Statistics
using SciMLSensitivity
using SciMLBase
using ReverseDiff
using Enzyme

"""
    loss_function(p_base,u0,tspan,tsteps,ground_truth_data)

Function to define loss metric.

This function uses the information about the true data set to predict the UDE solution using the current
weights. It makes use of the closure `pseudo_huber_loss(current_θ)`  to ensure that the loss function
is defined to only take the neural network weights as an input. The second `ODE_Wrapper(du,u,θ,t)` closure
ensures that only the neural network weights inside the parameter tuple are updated to ensure that
the sensitivity algorithm does not try to calculate gradients of other parameters as these are unchanged.

# Arguments
- `p_base::NamedTuple`: Parameter tuple 
- `u0::Matrix{Float64}`: Initial steady state condition for ODE problem
- `tspan::NamedTuple`: Span of time values
- `tsteps::Vector{Float64}`: Time steps taken by data generation solver
- `ground_truth_data::Matrix{Float64}`: True data for error prediction 

# Returns
- `pseudo_huber_loss::Function`: Loss function for defining loss closure in optimisation script
"""
function loss_function(p_base,u0,tspan,tsteps,ground_truth_data)
    function pseudo_huber_loss(current_θ)

        function ODE_Wrapper!(du,u,θ,t)
            p_train = merge(p_base,(θ=θ,))

            EnergyBalance!(du,u,p_train,t)
        end

        prob = ODEProblem(ODE_Wrapper!, u0,tspan,current_θ)
        sol = solve(prob,Rodas5P(),saveat=tsteps, sensealg=InterpolatingAdjoint(autojacvec=EnzymeVJP()))

        if sol.retcode != SciMLBase.ReturnCode.Success
            return Inf
        end

        predictions = Array(sol)

        temp_predictions = predictions[1:2*p_base.N,:]

        error = temp_predictions .- ground_truth_data 

        loss = (p_base.δ^2)*mean(sqrt.(1.0 .+(error./p_base.δ).^2).-1.0) + p_base.λ*sum(abs2,current_θ)
        return loss
    end

    return pseudo_huber_loss
end