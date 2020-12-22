#=Original problem
x′ + x = sin t + 3 cos 2t
x(0) = 0
=#

#=Esplicit form of equation
x′ = sin t  + 3 cos 2t − x
=#

#=Analytical solution
x(t) = 1/2 sin t - 1/2 cos t + 3/5 cos 2t + 6/5 sin 2t - 1/10 e^-t
=#

using DifferentialEquations
using Plots

ode_fn(x,p,t) = sin(t) + 3.0 * cos(2.0 * t) - x

an_sol(t) = (1.0/2.0) * sin(t) - (1.0/2.0) * cos(t) +
            (3.0/5.0) * cos(2.0*t) + (6.0/5.0) * sin(2.0*t) -
            (1.0/10.0) * exp(-t)

t_begin=0.0
t_end=10.0
tspan = (t_begin,t_end)
x_init=0.0

prob = ODEProblem(ode_fn, x_init, tspan)
num_sol = solve(prob, Tsit5(), reltol=1e-8, abstol=1e-8)

plot(num_sol.t, an_sol.(num_sol.t),
    linewidth=2, ls=:dash,
    title="ODE 1st order IVP solved by D.E. package",
    xaxis="t", yaxis="x",
    label="analytical",
    legend=true)
plot!(num_sol,
    linewidth=1,
    label="numerical")
