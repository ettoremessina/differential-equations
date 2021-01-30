#=
For details, please visit:
https://computationalmindset.com/en/neural-networks/experiments-with-neural-odes-in-julia.html#exp2
=#

using Flux, DiffEqFlux, DifferentialEquations, Plots

math_law(u) = sin.(2. * u) + cos.(2. * u)

function true_ode(du,u,p,t)
    true_A = [-0.15 2.10; -2.10 -0.10]
    du .= ((math_law(u))'true_A)'
end

tbegin = 0.0
tend = 4.0
datasize = 51
t = range(tbegin,tend,length=datasize)
u0 = [2.5; 0.5]
tspan = (tbegin,tend)
trange = range(tbegin,tend,length=datasize)
prob = ODEProblem(true_ode, u0, tspan)
dataset_ts = Array(solve(prob, Tsit5(), saveat=trange))

dudt = Chain(u -> math_law(u),
             Dense(2, 50, tanh),
             Dense(50, 2))

reltol = 1e-7
abstol = 1e-9
n_ode = NeuralODE(dudt, tspan, Tsit5(), saveat=trange, reltol=reltol,abstol=abstol)
ps = Flux.params(n_ode.p)

function loss_n_ode()
  pred = n_ode(u0)
  loss = sum(abs2, dataset_ts[1,:] .- pred[1,:]) +
         sum(abs2, dataset_ts[2,:] .- pred[2,:])
  loss
end

n_epochs = 400
learning_rate = 0.01
data = Iterators.repeated((), n_epochs)
opt = ADAM(learning_rate)

cb = function ()
  loss = loss_n_ode()
  println("Loss: ", loss)
end

println();
cb()

Flux.train!(loss_n_ode, ps, data, opt, cb=cb)

pl = plot(
  trange,
  dataset_ts[1,:],
  linewidth=2, ls=:dash,
  title="Neural ODE for forecasting",
  xaxis="t",
  label="original timeseries x(t)",
  legend=:right)
display(pl)

pl = plot!(
  trange,
  dataset_ts[2,:],
  linewidth=2, ls=:dash,
  label="original timeseries y(t)")
display(pl)

pred = n_ode(u0)

pl = plot!(
  trange,
  pred[1,:],
  linewidth=1,
  label="predicted timeseries x(t)")
display(pl)

pl = plot!(
    trange,
    pred[2,:],
    linewidth=1,
    label="predicted timeseries y(t)")
display(pl)

tbegin_forecast = tend
tend_forecast = tbegin_forecast + 5.0
tspan_forecast = (tbegin_forecast, tend_forecast)
datasize_forecast = 351
trange_forecast = range(tspan_forecast[1], tspan_forecast[2], length=datasize_forecast)
u0_forecast = [dataset_ts[1,datasize], dataset_ts[2,datasize]]

n_ode_forecast = NeuralODE(
  n_ode.model, tspan_forecast;
  p=n_ode.p, saveat=trange_forecast, reltol=reltol, abstol=abstol)
forecast = n_ode_forecast(u0_forecast)

pl = plot!(
  trange_forecast,
  forecast[1,:],
  linewidth=1,
  label="forecast x(t)")
display(pl)

pl = plot!(
    trange_forecast,
    forecast[2,:],
    linewidth=1,
    label="forecast y(t)")
display(pl)
