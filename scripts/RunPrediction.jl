using HeatExchangeUDE
function run_prediction()
    _,optim_params,_ = load_NN_results()
    Th_ground_truth,Tc_ground_truth,tsteps,N,L,_ = load_true()

    optim_solution = UDE_predict(optim_params,tsteps,Th_ground_truth,Tc_ground_truth)
    Th_model = optim_solution[1:N,:]
    Tc_model = optim_solution[N+1:2*N,:]
    Lvec = range(0,L,N)
    
    save_NN_prediction(Th_model,Tc_model,"1.0_noisy_prediction_test_data")

    Th_error,Tc_error,accuracy,spacial_avg_errors = prediction_error(Th_ground_truth,Tc_ground_truth,Th_model,Tc_model,tsteps)
    
    println("Mean Accuracy: ",accuracy,"%")

    #trained_model_heatmap(Th_error,Tc_error,tsteps,N,L)
    interactive_temperature_profile(Th_model, Tc_model, Th_ground_truth, Tc_ground_truth, Lvec, tsteps)
    training_vs_test_plot(spacial_avg_errors,tsteps)

end