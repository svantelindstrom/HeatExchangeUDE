using JLD2, DrWatson

# ---Data Saving functions---
function save_NN_prediction(Th,Tc,filename)
    trained_data = Dict(
        "Tc" => Tc,
        "Th" => Th
    )

    file = string(filename,".jld2")

    log_file = datadir("sims","NN_prediction_save_history.txt")
    open(log_file,"a") do io
        println(io,file)
    end

    file_path = datadir("sims",file)
    safesave(file_path,trained_data)
end

function save_NN_results(results, filename)
    adam, LBFGS , LossHistory = results
    NN_result_dict = Dict(
        "Adam Results" => adam,
        "LBFGS" => LBFGS,
        "Loss History" => LossHistory
    )

    file = string(filename,".jld2")

    log_file = datadir("sims","NN_results_save_history.txt")
    open(log_file,"a") do io
        println(io,file)
    end

    file_path = datadir("sims",file)
    safesave(file_path,NN_result_dict)
end

function save_true(u_true,t_true,N,L,filename)
    ground_truth_dict = Dict(
        "Th" => u_true[1:N,:],
        "Tc" => u_true[N+1:2*N,:],
        "tspan"=> (t_true[1],t_true[end]),
        "tsteps"=>t_true,
        "N"=>N,
        "L"=>L
    )

    file = string(filename,".jld2")

    log_file =  datadir("exp_raw","ground_truth_save_history.txt")
    open(log_file,"a") do io
        println(io,file)
    end

    file_path = datadir("exp_raw",file)

    safesave(file_path,ground_truth_dict)
end

function save_steady(solution_steady,filename)
    u_steady,N,L = solution_steady
    u0 = u_steady[:,end]
    steady_dict = Dict(
        "u0"=>u0,
        "N"=>N,
        "L"=>L
    )
    
    file = string(filename,".jld2")

    log_file = datadir("exp_raw","steady_state_save_history.txt")
    open(log_file,"a") do io
        println(io,file)
    end

    file_path = datadir("exp_raw",file)
    safesave(file_path,steady_dict)
end

# ---Data Loading functions---

#Steady State Solution - u0

#Loading a specific file
function load_steady(filename::AbstractString)
    file = string(filename,".jld2")
    file_path = datadir("exp_raw",file)
    data = load(file_path)

    u0 = data["u0"]
    L = data["L"]
    N = data["N"]

    return u0,L,N
end

#Loading the most recent file
function load_steady()
    log_file = datadir("exp_raw","steady_state_save_history.txt")
    most_recent_file = readlines(log_file)[end]
    base_name = replace(most_recent_file,".jld2"=>"")

    return load_steady(base_name)
end

#Ground Truth Data 

#Loading a specific file
function load_true(filename::AbstractString)
    file = string(filename,".jld2")
    file_path = datadir("exp_raw",file)
    data = load(file_path)
    
    Th = data["Th"]
    Tc = data["Tc"]
    tsteps = data["tsteps"]
    N = data["N"]
    L = data["L"]
    tspan = data["tspan"]

    return Th,Tc,tsteps,N,L,tspan
end

#Loading the most recent file
function load_true()
    log_file = datadir("exp_raw","ground_truth_save_history.txt")
    most_recent_file = readlines(log_file)[end]
    base_name = replace(most_recent_file,".jld2"=>"")

    return load_true(base_name)
end

#Weights and Loss History

#Loading a specific file
function load_NN_results(filename::AbstractString)
    file = string(filename,".jld2")
    file_path = datadir("sims",file)
    data = load(file_path)

    adam = data["Adam Results"]
    LBFGS = data["LBFGS"]
    LossHistory = data["Loss History"]
    
    return adam,LBFGS,LossHistory
end

#Loading the most recent file
function load_NN_results()
    log_file = datadir("sims","NN_results_save_history.txt")
    most_recent_file = readlines(log_file)[end]
    base_name = replace(most_recent_file,".jld2"=>"")

    return load_NN_results(base_name)
end

#Predicted temperatures

#Loading a specific file
function load_NN_prediction(filename::AbstractString)
    file = string(filename,".jld2")
    file_path = datadir("sims",file)
    data = load(file_path)

    Th = data["Th"]
    Tc = data["Tc"]
    return Th,Tc
end

#Loading the most recent file
function load_NN_prediction()
    log_file = datadir("sims","NN_prediction_save_history.txt")
    most_recent_file = readlines(log_file)[end]
    base_name = replace(most_recent_file,".jld2"=>"")

    return load_NN_prediction(base_name)
end