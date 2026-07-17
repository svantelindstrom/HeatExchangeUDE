function EnergyBalance!(du,u,p,t)
    #Defining Discretisation
    @views Th = u[1:p.N]
    @views Tc = u[p.N+1:2*p.N]

    R = p.R(t,Th,Tc,p)

    U = 1.0 ./ (1.0/p.U0 .+ R)
    Q = U.*p.P.*(Th .- Tc)


    #Defining the ODE's:
    @views du[2:p.N] .= -p.v_h.*((Th[2:p.N].-Th[1:p.N-1])./p.dx) .- Q[2:p.N] ./(p.rho_h*p.cp_h*p.S)
    @views du[p.N+1:2*p.N-1] .= p.v_c.*((Tc[2:p.N].-Tc[1:p.N-1])./p.dx) .+ Q[1:p.N-1] ./(p.rho_c*p.cp_c*p.S)
    
    du[1] = -p.v_h*((Th[1]-p.T_h_in)/p.dx) - Q[1]/(p.rho_h*p.cp_h*p.S)
    du[2*p.N] = p.v_c*((p.T_c_in-Tc[p.N])/p.dx) + Q[p.N]/(p.rho_c*p.cp_c*p.S)
end