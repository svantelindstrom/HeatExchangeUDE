using Lux

#Create NeuralNetwork definition based on size defined by modeltraining.jl
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