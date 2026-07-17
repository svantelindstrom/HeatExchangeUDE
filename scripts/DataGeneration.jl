using DifferentialEquations
using Plots
using HeatExchangeUDE

function Kern_Seaton(t,Th,Tc,p)
    R = p.R0*(1-exp(-p.beta*t))
    return R
end

no_fouling(t,Th,Tc,p) = 0.0

#Parameters to find steady state operation with no fouling
p_steady = pVecBuilder(R = no_fouling)
tspan_steady = (0.0, 5e3)
u0_h_steady = fill(p_steady.T_h_in,p_steady.N)
u0_c_steady = fill(p_steady.T_c_in,p_steady.N)
u0_steady = vcat(u0_h_steady,u0_c_steady)

prob_steady = ODEProblem(EnergyBalance!,u0_steady,tspan_steady, p_steady)
solution_steady = solve(prob_steady, Tsit5())

u0_true = Array(solution_steady.u[end])

x = range(0, p_steady.L, length=p_steady.N)
plot(x, u0_true[1:p_steady.N], label="Hot Fluid (Th)", xlabel="Distance (m)", ylabel="Temperature (°C)")
plot!(x, u0_true[p_steady.N+1:2*p_steady.N], label="Cold Fluid (Tc)")




