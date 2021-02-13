#=
For details, please visit:
https://computationalmindset.com/en/neural-networks/experiments-with-neural-odes-in-julia.html#exp3
=#

using Plots
using Flux
using DiffEqFlux
using OrdinaryDiffEq

tbegin = 0.0
tend = 4.0
datasize = 51
dataset_in = range(tbegin, tend, length=datasize)
dataset_out = log.(1 .+ dataset_in) .-  3 * sqrt.(dataset_in)

function neural_ode(data_dim; saveat = dataset_in)
    fc = FastChain(FastDense(data_dim, 64, swish),
                  FastDense(64, 32, swish),
                  FastDense(32, data_dim))

    n_ode = NeuralODE(
            fc,
            (minimum(dataset_in), maximum(dataset_in)),
            Tsit5(),
            saveat = saveat,
            abstol = 1e-9, reltol = 1e-9)
end

n_ode = neural_ode(1)
theta = n_ode.p

predict(p) = n_ode(dataset_out[1:1], p)'

loss(p) = begin
  yhat = predict(p)
  l = Flux.mse(yhat, dataset_out)
end

learning_rate=1e-2
opt = ADAMW(learning_rate)
epochs = 500

function cb_train(theta, loss)
    println("Loss: ", loss)
    false
end

res_train = DiffEqFlux.sciml_train(
    loss, theta, opt,
    maxiters = epochs,
    cb = cb_train)

y_pred = predict(res_train.minimizer)

pl = plot(
    dataset_in,
    dataset_out,
    linewidth=2, ls=:dash,
    title="MLP by FastChain with Neural ODEs",
    xaxis="t",
    label="original y(t)",
    legend=:topright)
display(pl)

pl = plot!(
    dataset_in,
    y_pred,
    linewidth=1,
    label="predicted y(t)")
display(pl)
