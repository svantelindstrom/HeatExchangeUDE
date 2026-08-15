using DrWatson
using Plots
using JLD2  
using Statistics
using HeatExchangeUDE
using DifferentialEquations
using GLMakie: Figure, Axis, Slider, @lift, lines!, text!, axislegend,scatter!

#Plot for u0 of the synthetic dataset
function steady_state_plot()
    u0,L,N = load_steady()

    x = range(0, L, N)
    p = plot(x, u0[1:N], label="Hot Fluid (Th)", xlabel="Distance (m)", ylabel="Temperature (°C)",color=:red)
    plot!(p,x, u0[N+1:2*N], label="Cold Fluid (Tc)", xlabel="Distance (m)", ylabel="Temperature (°C)",color=:blue)

    display(p)
end

#Creating Plot of Ground Truth Data Set
function ground_truth_gif()
    Th,Tc,tsteps,N,L,_ = load_true()

    Tc_in = minimum(Tc)
    Th_in = maximum(Th)


    frame_no = 15
    frame_step = floor(Int,length(tsteps)/frame_no)
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

function loss_plot()
    _,_,LossHistory = load_NN_results()

    p = plot(log10.(LossHistory),legend = false,title="Loss",xlabel="epochs",ylabel="Loss")
    display(p)
end

function trained_model_heatmap()
    Th_ground_truth,Tc_ground_truth,tsteps,N,L,_ = load_true()

    _,optim_params,_ = load_NN_results()

    optim_solution = UDE_predict(optim_params,tsteps)

    Th_model = optim_solution[1:N,:]
    Tc_model = optim_solution[N+1:2*N,:]

    Th_error = 100*abs.(Th_model.-Th_ground_truth)./(Th_ground_truth.+273.15)
    Tc_error = 100*abs.(Tc_model.-Tc_ground_truth)./(Tc_ground_truth.+273.15)

    error_mean = mean(vcat(Th_error,Tc_error))
    accuracy = 100-error_mean
    println("Accuracy: ",accuracy,"%")

    Lvec = range(0,L,length=N)

    #Transpose the error for plotting 
    Transpose_err_h = transpose(Th_error)
    Transpose_err_c = transpose(Tc_error)

    #Hot Stream Error Heatmap
    Th_error_heat = heatmap(Lvec,tsteps,Transpose_err_h, title="Hot Stream Error Heatmap",xlabel="Length (m)",ylabel="Time (s)")

    #Cold Stream Error Heatmap
    Tc_error_heat = heatmap(Lvec,tsteps,Transpose_err_c, title="Cold Stream Error Heatmap",xlabel="Length (m)",ylabel="Time (s)")

    p = plot(Th_error_heat,Tc_error_heat,layout = (2,1),legend = false)

    plot_path = plotsdir("error_heatmaps.png")
    savefig(p,plot_path)

    data_path = datadir("sims","trained_data_temps.jld2")
    trained_data = Dict(
        "Tc" => Tc_model,
        "Th" => Th_model
    )
    safesave(data_path,trained_data)

end

function interactive_temperature_profile(Th_model, Tc_model, Th_truth, Tc_truth, Lvec, tsteps)
    # 1. Initialize the interactive figure
    fig = Figure(resolution = (800, 600))
    
    # 2. Add an Axis
    ax = Axis(fig[1, 1], 
        title = "Interactive Heat Exchanger Temperature Profile",
        xlabel = "Length (m)", 
        ylabel = "Temperature"
    )

    # 3. Create a Slider for time steps
    time_slider = Slider(fig[2, 1], range = 1:length(tsteps), startvalue = 1)
    
    # 4. Create an Observable that listens to the slider
    time_index = time_slider.value

    # 5. "Lift" the data matrices so they update whenever time_index changes
    Th_model_plot = @lift(Th_model[:, $time_index])
    Tc_model_plot = @lift(Tc_model[:, $time_index])
    Th_truth_plot = @lift(Th_truth[:, $time_index])
    Tc_truth_plot = @lift(Tc_truth[:, $time_index])
    
    current_time = @lift("Time: $(round(tsteps[$time_index], digits=2)) s")

    # 6. Plot the lifted data
    # Plot true data first as semi-transparent lines (0.4 opacity)
    # Using a slightly thicker linewidth helps it stand out behind the dots
    lines!(ax, Lvec, Th_truth_plot, color = (:red, 0.4), label = "Hot (Truth)", linewidth = 4)
    lines!(ax, Lvec, Tc_truth_plot, color = (:blue, 0.4), label = "Cold (Truth)", linewidth = 4)
    
    # Plot predicted data as solid large dots using scatter!
    scatter!(ax, Lvec, Th_model_plot, color = :red, markersize = 14, label = "Hot (Model)")
    scatter!(ax, Lvec, Tc_model_plot, color = :blue, markersize = 14, label = "Cold (Model)")
    
    # Add dynamic text for the time and a legend
    text!(ax, current_time, position = (Lvec[end]*0.7, maximum(Th_truth)*0.9))
    axislegend(ax, position = :rt) # :rt locks legend to Right-Top

    # 7. Launch the interactive window
    display(fig)
end

function run_interactive_plot()
    Th_ground_truth,Tc_ground_truth,tsteps,N,L,_ = load_true()

    Th_model,Tc_model = load_NN_prediction()

    Lvec = range(0,L,N)

    interactive_temperature_profile(Th_model,Tc_model,Th_ground_truth,Tc_ground_truth,Lvec,tsteps)

end

function UDE_predict(θ,tsteps)

    #Load Steady State Data
    path = datadir("exp_raw","steady_state_data.jld2")
    data = load(path)
    T0 = data["u0"]

    Th,Tc,_,_,_,_ = load_true()

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

    tspan = (0,p.final_time)

    R0 = fill(0.0,p.N)
    u0 = vcat(T0,R0)

    prob = ODEProblem(EnergyBalance!,u0,tspan,p)
    sol = solve(prob,Rodas5P(),saveat=tsteps)
    return sol
end