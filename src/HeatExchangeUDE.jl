module HeatExchangeUDE
    using DrWatson
    
    include("DataSetSpecs.jl")
    export pVecBuilder

    include("EnergyBalance.jl")
    export EnergyBalance!

    include("NeuralNetwork.jl")
    export define_NN
    export setup_NN
    export R_NN

    include("LossGradients.jl")
    export loss_function

    include(scriptsdir("SaveLoad.jl"))
    export save_NN_results
    export save_true
    export save_steady
    export load_steady
    export load_true
    export load_NN_prediction
    export load_NN_results

end
