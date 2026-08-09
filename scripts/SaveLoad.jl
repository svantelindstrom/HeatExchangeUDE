using JLD2, DrWatson

function save_NN_results(results)
    adam, LBFGS , LossHistory = results
    NN_result_dict = Dict(
        "Adam Results" => adam,
        "LBFGS" => LBFGS,
        "Loss History" => LossHistory
    )

    file_path = datadir("sims","trained_model_prediction.jld2")
    safesave(file_path,NN_result_dict)
end

function save_true(u_true,t_true,N,L)
    ground_truth_dict = Dict(
        "Th" => u_true[1:N,:],
        "Tc" => u_true[N+1:2*N,:],
        "tspan"=> (t_true[1],t_true[end]),
        "tsteps"=>t_true,
        "N"=>N,
        "L"=>L
    )

    file_path = datadir("exp_raw","ground_truth_data.jld2")
    safesave(file_path,ground_truth_dict)
end

function save_steady(solution_steady)
    u_steady,N,L = solution_steady
    u0 = u_steady[:,end]
    steady_dict = Dict(
        "u0"=>u0,
        "N"=>N,
        "L"=>L
    )
    
    file_path = datadir("exp_raw","steady_state_data.jld2")
    safesave(file_path,steady_dict)
end

function load_steady()
    steady_state_path = datadir("exp_raw","steady_state_data.jld2")
    data = load(steady_state_path)
    
    u0 = data["u0"]
    L = data["L"]
    N = data["N"]

    return u0,L,N
end

function load_true()
    ground_truth_path = datadir("exp_raw","ground_truth_data.jld2")
    data = load(ground_truth_path)
    
    Th = data["Th"]
    Tc = data["Tc"]
    tsteps = data["tsteps"]
    N = data["N"]
    L = data["L"]
    tspan = data["tspan"]

    return Th,Tc,tsteps,N,L,tspan
end

function load_NN_results()
    trained_model_path = datadir("sims","trained_model_prediction.jld2")
    data = load(trained_model_path)

    adam = data["Adam Results"]
    LBFGS = data["LBFGS"]
    LossHistory = data["Loss History"]
    
    return adam,LBFGS,LossHistory
end

function load_NN_prediction()
    model_prediction_path = datadir("sims","trained_data_temps.jld2")
    data = load(model_prediction_path)

    Th = data["Th"]
    Tc = data["Tc"]
    return Th,Tc
end