module HeatExchangeUDE

    include("DataSetSpecs.jl")
    export pVecBuilder

    include("EnergyBalance.jl")
    export EnergyBalance!

    include("NeuralNetwork.jl")
    export define_NN

    include("LossGradients.jl")
    export loss_function

end
