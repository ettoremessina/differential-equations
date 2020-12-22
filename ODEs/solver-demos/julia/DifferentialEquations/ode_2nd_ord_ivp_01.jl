#=Original problem
x′′ + x′ + 2x = 0
x(0)=1
x′(0)=0
=#

#=Esplicit form of equation
x′′ = -x′ - 2x
=#

#=Analytical solution
x(t) = e^(-t/2) (cos(√7 t / 2) + sin(√7 t / 2) / √7)
=#

using DifferentialEquations
using Plots

function ode_fn(dx,x,p,t)
    -dx -2.0 * x
end

an_sol(t) = exp(-t/2.0) *
            (cos(sqrt(7.0) * t / 2.0) + sin(sqrt(7.0) * t / 2.0)/sqrt(7.0))

t_begin=0.0
t_end=12.0
tspan = (t_begin,t_end)
x_init=1.0
dxdt_init=0.0

prob = SecondOrderODEProblem(ode_fn, dxdt_init, x_init, tspan)
num_sol = solve(prob, Tsit5(), reltol=1e-8, abstol=1e-8)
x_num_sol = [u[2] for u in num_sol.u]

plot(num_sol.t, an_sol.(num_sol.t),
    linewidth=2, ls=:dash,
    title="ODE 2nd order IVP solved by D.E. package",
    xaxis="t", yaxis="x",
    label="analytical",
    legend=true)
plot!(num_sol.t, x_num_sol,
    linewidth=1,
    label="numerical")
