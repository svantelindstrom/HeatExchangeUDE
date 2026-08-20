function pVecBuilder(;
    R,
    model = nothing,
    θ = nothing,
    st = nothing,
    R0 = 0.002, #Arbitrarily chosen value
    beta = 1e-6, #Arbitrarily chosen value
    T_h_in = 90.0, #Arbitrarily chosen value
    T_c_in = 20.0, #Arbitrarily chosen value
    r = 0.025, #Arbitrarily chosen value
    N = 10, #Placeholder value (may need changing in future)
    L = 20.0, #Placeholder value (may need changing in future)
    final_time=3e6,
    time_points=30,
    τ=final_time,
    ϵ = 1e-3,
    Th_max = nothing,
    Tc_max = nothing,
    δ = 1.0,
    λ = 0.0
)
return (
    r = r,
    P = 2*pi*r, 
    S = pi*r^2, 
    U0 = 500.0, #Arbitrarily chosen value
    v_h = 0.4, #Arbitrarily chosen value
    v_c = 0.2, #Arbitrarily chosen value
    rho_h = 1000.0, #Density of water
    rho_c = 1000.0, #Density of water
    cp_h = 4184.0, #Heat Capacity of Water
    cp_c = 4184.0, #Heat Capacity of Water
    N = N, #Placeholder value (may need changing in future)
    L = L, #Placeholder value (may need changing in future)
    dx = L / N,
    final_time = final_time,
    time_points = time_points,
    dt = final_time/time_points,
    T_h_in = T_h_in, 
    T_c_in = T_c_in,
    R0 = R0,
    beta = beta,
    R = R,
    model = model,
    θ = θ,
    st = st,
    τ=τ,
    ϵ=ϵ,
    R_max = R0*(1-exp(-beta*final_time)),
    Th_max = Th_max,
    Tc_max = Tc_max,
    dR_max = R0*beta,
    δ = δ,
    λ = λ
)

end


