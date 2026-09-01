using JLD2, DrWatson

"""
    save_NN_prediction(Th,Tc,filename)

Saves predicted temperatures to filename.jld2 and adds filename.jld2 to a history of 
saved predictions such that it can be loaded as the most recent file by load function.

# Arguments
- `Th::Matrix{Float64}`: Predicted temperature of hot stream.
- `Tc::Matrix{Float64}`: Predicted temperature of cold stream.
- `filename::AbstractString`: Name of file for prediction to be saved to, does not include .jld2 extension.

# Returns 
- `nothing`: No returns from function, saves prediction to specified file.
"""
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

"""
    save_NN_results(results, filename)

Saves neural network weights and biases to filename.jld2 and adds filename.jld2 to a history of 
saved training results such that it can be loaded as the most recent file by load function.

# Arguments
- `results::NamedTuple`: Tuple of biases, weights and loss from optimisation.
- `filename::AbstractString`: Name of file for prediction to be saved to, does not include .jld2 extension.

# Returns 
- `nothing`: No returns from function, saves results to specified file.
"""
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

"""
    save_true(u_true,t_true,N,L,filename)

Saves true dataset to filename.jld2 and adds filename.jld2 to a history of 
saved true datasets such that it can be loaded as the most recent file by load function.

# Arguments
- `u_true::Matrix{Float64}`: Temperatures of true dataset.
- `t_true::Vector{Float64}`: Times of data points in true data.
- `N::Int`: Number of spacial points for each temperature.
- `L::Float64`: Physical length of heat exchanger.
- `filename::AbstractString`: Name of file for prediction to be saved to, does not include .jld2 extension.

# Returns 
- `nothing`: No returns from function, saves results to specified file.
"""
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

"""
    save_steady(solution_steady,filename)

Saves steady state solution to filename.jld2 and adds filename.jld2 to a history of 
saved steady state solutions such that it can be loaded as the most recent file by load function.

# Arguments
- `solution_steady::NamedTuple`: Steady state temperatures, spacial points and physical length.
- `filename::AbstractString`: Name of file for prediction to be saved to, does not include .jld2 extension.

# Returns 
- `nothing`: No returns from function, saves results to specified file.
"""
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

"""
    load_steady(filename::AbstractString)
    load_steady()

Load the steady-state initial conditions and geometric parameters from a saved `.jld2` dataset.

When provided with a specific `filename`, this function loads the corresponding data from the 
`exp_raw` directory. When called without arguments, it automatically reads the 
`steady_state_save_history.txt` log file to locate and load the most recently generated dataset.

# Arguments
- `filename::AbstractString`: (Optional) The name of the file to load, excluding the `.jld2` extension.

# Returns
- `u0::Vector{Float64}`: The steady-state temperature profile used as the initial condition for the transient solver.
- `L::Float64`: The total length of the heat exchanger (m).
- `N::Int`: The number of spatial discretization nodes.
"""
function load_steady(filename::AbstractString)
    file = string(filename,".jld2")
    file_path = datadir("exp_raw",file)
    data = load(file_path)

    u0 = data["u0"]
    L = data["L"]
    N = data["N"]

    return u0,L,N
end

function load_steady()
    log_file = datadir("exp_raw","steady_state_save_history.txt")
    most_recent_file = readlines(log_file)[end]
    base_name = replace(most_recent_file,".jld2"=>"")

    return load_steady(base_name)
end

#Ground Truth Data 

"""
    load_true(filename::AbstractString)
    load_true()

Load the true dataset from a saved .jld2 data file.

When provided with a specific `filename`, this function loads the corresponding data from the 
`exp_raw` directory. When called without arguments, it automatically reads the 
`ground_truth_save_history.txt` log file to locate and load the most recently generated dataset.

# Arguments
- `filename::AbstractString`: (Optional) The name of the file to load, excluding the `.jld2` extension.

# Returns
- `u_true::Matrix{Float64}`: Temperatures of true dataset.
- `t_true::Vector{Float64}`: Times of data points in true data.
- `N::Int`: Number of spacial points for each temperature.
- `L::Float64`: Physical length of heat exchanger.
"""
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

function load_true()
    log_file = datadir("exp_raw","ground_truth_save_history.txt")
    most_recent_file = readlines(log_file)[end]
    base_name = replace(most_recent_file,".jld2"=>"")

    return load_true(base_name)
end

#Weights and Loss History

"""
    load_NN_results(filename::AbstractString)
    load_NN_results()

Load the trained neural network results from a saved .jld2 data file.

When provided with a specific `filename`, this function loads the corresponding data from the 
`sims` directory. When called without arguments, it automatically reads the 
`NN_results_save_history.txt` log file to locate and load the most recently generated dataset.

# Arguments
- `filename::AbstractString`: (Optional) The name of the file to load, excluding the `.jld2` extension.

# Returns
- `adam::NamedTuple`: Neural network weights and biases after the adam optimisation step.
- `LBFGS::NamedTuple`: Final neural network weights and biases after the LBFGS optimisation step.
- `LossHistory::Vector{Float64}`: History of loss over the epochs of the optimisation.
"""
function load_NN_results(filename::AbstractString)
    file = string(filename,".jld2")
    file_path = datadir("sims",file)
    data = load(file_path)

    adam = data["Adam Results"]
    LBFGS = data["LBFGS"]
    LossHistory = data["Loss History"]
    
    return adam,LBFGS,LossHistory
end

function load_NN_results()
    log_file = datadir("sims","NN_results_save_history.txt")
    most_recent_file = readlines(log_file)[end]
    base_name = replace(most_recent_file,".jld2"=>"")

    return load_NN_results(base_name)
end

#Predicted temperatures

"""
    load_NN_prediction(filename::AbstractString)
    load_NN_prediction()

Load the neural networks prediction from a saved .jld2 data file.

When provided with a specific `filename`, this function loads the corresponding data from the 
`sims` directory. When called without arguments, it automatically reads the 
`ground_truth_save_history.txt` log file to locate and load the most recently generated dataset.

# Arguments
- `filename::AbstractString`: (Optional) The name of the file to load, excluding the `.jld2` extension.

# Returns
- `Th::Matrix{Float64}`: Predicted temperature of hot stream.
- `Tc::Matrix{Float64}`: Predicted temperature of cold stream.
"""
function load_NN_prediction(filename::AbstractString)
    file = string(filename,".jld2")
    file_path = datadir("sims",file)
    data = load(file_path)

    Th = data["Th"]
    Tc = data["Tc"]
    return Th,Tc
end


function load_NN_prediction()
    log_file = datadir("sims","NN_prediction_save_history.txt")
    most_recent_file = readlines(log_file)[end]
    base_name = replace(most_recent_file,".jld2"=>"")

    return load_NN_prediction(base_name)
end