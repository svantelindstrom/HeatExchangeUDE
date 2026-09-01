"""
    EnergyBalance!(du,u,p,t)

Function defining the energy balance equation used to model the heat exchanger.

This function acts as the engine behind all calculations made in the program. If u contains only
the hot and cold temperatures the algebraic definition of the fouling expression is used. This is
the case when the true dataset is generated. If it also contains the fouling, the fouling rate is instead
defined to be predicted by the neural network. The energy balance includes the convective and advective heat transfer
terms but neglects conduction.

# Arguments
- `du::AbstractVector{Float64}`: Pre-allocated derivative vector updated in-place
- `u::AbstractVector{Float64}`: Current state vector containing fouling and temperatures 
- `p::NamedTuple`: Parameter tuple
- `t::Float64`: Current time of the simulation

# Returns
- `nothing`: The function operates in place to avoid memory allocation
"""
function EnergyBalance!(du,u,p,t)
    #Defining Discretisation
    @views Th = u[1:p.N]
    @views Tc = u[p.N+1:2*p.N]

    #Using the input vector u0's length to determine whether R should be treated as a state variable or not
    if length(u) == 2*p.N
        #Algebraic definition of R for ground truth data generation
        R = p.R(t,p)
    else
        #R is a state variable and prediction of the fouling rate is based on the value of R
        @views R = u[2*p.N+1:3*p.N]

        dR_dt = p.R(Th,Tc,R,p)

        @views du[2*p.N+1:3*p.N] = dR_dt
    end

    U = 1.0 ./ (1.0/p.U0 .+ R)
    Q = U.*p.P.*(Th .- Tc)

    #Defining the ODE's:
    @views du[2:p.N] .= -p.v_h.*((Th[2:p.N].-Th[1:p.N-1])./p.dx) .- Q[2:p.N] ./(p.rho_h*p.cp_h*p.S)
    @views du[p.N+1:2*p.N-1] .= p.v_c.*((Tc[2:p.N].-Tc[1:p.N-1])./p.dx) .+ Q[1:p.N-1] ./(p.rho_c*p.cp_c*p.S)
    
    du[1] = -p.v_h*((Th[1]-p.T_h_in)/p.dx) - Q[1]/(p.rho_h*p.cp_h*p.S)
    du[2*p.N] = p.v_c*((p.T_c_in-Tc[p.N])/p.dx) + Q[p.N]/(p.rho_c*p.cp_c*p.S)
end