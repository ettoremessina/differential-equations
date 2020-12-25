#=
Please see
https://computationalmindset.com/en/neural-networks/ordinary-differential-equation-solvers.html#sys1
for details
=#

#=Original problem
x′ + x − y = 0
y′ − 4x + y = 0
x(0)=2
y(0)=0
=#

#=Esplicit form of equations
x′ = y - x
y′ = 4x - y
=#

#=Analytical solution
x(t) = e^t + e^−3t
y(t) = 2e^t − 2e^−3t
=#

using DifferentialEquations
using Plots

function ode_fn(du,u,p,t)
    x, y = u
    du[1] = y - x
    du[2] = 4.0 * x - y
end

an_sol_x(t) = exp(t) + exp(-3.0 * t)
an_sol_y(t) = 2.0 * exp(t) - 2.0 * exp(-3.0 * t)

t_begin=0.0
t_end=5.0
tspan = (t_begin,t_end)
x_init=2.0
y_init=0.0

prob = ODEProblem(ode_fn, [x_init, y_init], tspan)
num_sol = solve(prob, Tsit5(), reltol=1e-8, abstol=1e-8)
x_num_sol = [u[1] for u in num_sol.u]
y_num_sol = [u[2] for u in num_sol.u]

plot(num_sol.t, an_sol_x.(num_sol.t),
    linewidth=2, ls=:dash,
    title="System of 2 ODEs 1st order IVP solved by D.E. package",
    xaxis="t",
    label="analytical x",
    legend=true)
plot!(num_sol.t, an_sol_y.(num_sol.t),
    linewidth=2, ls=:dash,
    label="analytical y",
    legend=true)
plot!(num_sol.t, x_num_sol,
    linewidth=1,
    label="numerical x")
plot!(num_sol.t, y_num_sol,
    linewidth=1,
    label="numerical y")
