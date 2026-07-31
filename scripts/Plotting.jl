using DrWatson
using Plots
using JLD2  

#Plot for u0 of the synthetic dataset
#= 
    x = range(0, p_steady.L, length=p_steady.N)
    plot(x, u0_true[1:p_steady.N], label="Hot Fluid (Th)", xlabel="Distance (m)", ylabel="Temperature (°C)")
    plot!(x, u0_true[p_steady.N+1:2*p_steady.N], label="Cold Fluid (Tc)")
=#

#Creating Plot of Ground Truth Data Set
function ground_truth_gif()

    ground_truth_path = datadir("exp_raw","ground_truth_data.jld2")
    data = load(ground_truth_path)
    
    Th = data["Th"]
    Tc = data["Tc"]
    tsteps = data["tsteps"]
    N = data["N"]
    L = data["L"]

    Tc_in = minimum(Tc)
    Th_in = maximum(Th)

    frame_step = 20
    x = range(0, L, length=N)
    anim = @animate for i in 1:frame_step:length(tsteps)

        Th_current = Th[:,i]
        Tc_current = Tc[:,i]
        t_current = tsteps[i]

        days_passed = round(t_current/(24*3600),digits=1)

        plot(x,Th_current,
            label="Hot Fluid (Th)", 
            color=:red,
            linewidth=2,
            xlabel="Distance (m)", 
            ylabel="Temperature (°C)",
            ylim=(Tc_in, Th_in), 
            title="Heat Exchanger Fouling: Day $days_passed",
            legend=:right)

        plot!(x, Tc_current, 
            label="Cold Fluid (Tc)",
            color=:blue,
            linewidth=2)


    end

    #Saving gif showing ground truth dataset
    save_path = plotsdir("fouling_animation.gif")
    mkpath(plotsdir())
    
    gif(anim, save_path, fps=15)
end