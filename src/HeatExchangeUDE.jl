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

    include("UDEPredict.jl")
    export UDE_predict
    export prediction_error

    include(scriptsdir("SaveLoad.jl"))
    export save_NN_results
    export save_true
    export save_steady
    export save_NN_prediction
    export load_steady
    export load_true
    export load_NN_prediction
    export load_NN_results

    include(scriptsdir("Plotting.jl"))
    export training_vs_test_plot
    export interactive_temperature_profile
    export trained_model_heatmap

end
