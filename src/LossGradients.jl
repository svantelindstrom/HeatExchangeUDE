using DifferentialEquations
using Zygote
using Statistics
using SciMLSensitivity
using SciMLBase
using ReverseDiff
using Enzyme

function loss_function(p_base,u0,tspan,tsteps,ground_truth_data)
    function MSE_loss(current_θ)

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

        loss = mean(abs2.(temp_predictions.-ground_truth_data))

        return loss
    end

    return MSE_loss
end