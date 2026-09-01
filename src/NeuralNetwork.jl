using Lux, StableRNGs, ComponentArrays,NNlib

"""
    define_NN(hidden_layers::Int,nodes_per_layer::Int;input_dims::Int=3,output_dims::Int=1)

Defines the neural network object using Lux.jl.

Creates the neural network based on the nodes per layer and the hidden layer arguments. For stable
training the final layers weights and biases are zero initialised.

# Arguments
- `hidden_layers::Int`: The number of hidden layers in the network architecture
- `nodes_per_layer::Int`: The number of nodes in every hidden layer
- `input_dims::Int`: Input dimensions to the first layer. Defaults to 3 with hot and cold temperatures and the fouling 
- `output_dims::Int`: Dimensions outputted by final layer. Defaults to 1, predicting the fouling rate

# Returns
- `Chain(layers...)::Chain{NamedTuple}`: A Lux.jl chain storing information about the network architecture as the type NamedTuple
"""
function define_NN(hidden_layers::Int,nodes_per_layer::Int;input_dims::Int = 3,output_dims::Int = 1)
    layers = Any[]

    push!(layers,Dense(input_dims=>nodes_per_layer,tanh))

    for _ in 2:hidden_layers
        push!(layers,Dense(nodes_per_layer=>nodes_per_layer,tanh))
    end

    #Adding zero initialisation
    zero_init(rng,dims...) = zeros(Float64,dims...)

    push!(layers,Dense(nodes_per_layer=>output_dims,tanh;init_weight = zero_init,init_bias = zero_init))
    
    return Chain(layers...)
end

"""
    R_NN(Th,Tc,R,p)

Using Neural Network inside the energy balance ODE to predict fouling rate.

This function takes the current fouling, temperatures and the parameter tuple p and 
passes these to the neural network to obtain the predicted fouling rate. The input variables are normalised
and placed into a matrix for input into the neural network. To ensure positivity of the fouling 
without restriction of the fouling rate to be positive the derivative is split up to an accumulation
and removal rate. The removal rate is multiplied by a tanh term which approaches 0 as the fouling 
approaches 0. Hence if the fouling is 0 positivity of the derivative is ensured using the
softplus function.

# Arguments 
- `Th::Matrix{Float64}`: Hot temperature of true dataset
- `Tc::Matrix{Float64}`: Cold temperature of true dataset
- `R::Matrix{Float64}`: Fouling 
- `p::NamedTuple`: Parameter tuple containing physical parameters and neural network states and weights

# Returns
- `vec(dR_pred.*p.dR_max)::Vector{Float64}`: Normalised fouling rate prediction multiplied by maximum fouling rate to get physical rate
"""
function R_NN(Th,Tc,R,p)
    norm_Th = Th./p.Th_max
    norm_Tc = Tc./p.Tc_max
    norm_R = R./p.R_max 

    input_NN = vcat(reshape(norm_Th, 1, :), reshape(norm_Tc, 1, :),reshape(norm_R,1, :))

    dR_pred,_ = p.model(input_NN,p.θ,p.st)

    dR_pred = vec(dR_pred)
    
    pos_dR = softplus.(dR_pred)
    neg_dR = softplus.(-dR_pred)

    dR_pred = pos_dR .- neg_dR.*tanh.(max.(0.0,norm_R)./p.ϵ)
    
    return vec(dR_pred.*p.dR_max)
end

"""
    setup_NN(hidden_layers::Int,nodes_per_layer::Int)

Sets up neural network and converts weights to a ComponentArray

# Arguments 
- `hidden_layers::Int`: The number of hidden layers in the network architecture
- `nodes_per_layer::Int`: The number of nodes in every hidden layer

# Returns
- `NN::Chain{NamedTuple}`: A Lux.jl chain storing information about the network architecture as the type NamedTuple
- `θ::ComponentArray{Float64}`: Initialised neural network weights
- `st::NamedTuple`: NamedTuple conatining the neural network states
"""
function setup_NN(hidden_layers::Int,nodes_per_layer::Int)
    NN = define_NN(hidden_layers,nodes_per_layer)

    rng = StableRNG(123)

    θ,st = Lux.setup(rng,NN)
    θ = ComponentArray(θ) |> f64
    return NN,θ,st
end