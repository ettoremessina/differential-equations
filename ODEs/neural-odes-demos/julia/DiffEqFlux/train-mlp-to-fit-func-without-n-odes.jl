#=
For details, please visit:
https://computationalmindset.com/en/neural-networks/experiments-with-neural-odes-in-julia.html#exp3
=#

using Plots
using Flux
using DiffEqFlux

tbegin = 0.0
tend = 4.0
datasize = 51
dataset_in = range(tbegin, tend, length=datasize)
dataset_out = log.(1 .+ dataset_in) .- sqrt.(dataset_in)

function neural_network(data_dim)
    fc = FastChain(FastDense(data_dim, 64, swish),
                  FastDense(64, 32, swish),
                  FastDense(32, data_dim))
end

nn = neural_network(1)
theta = initial_params(nn)

predict(t, p) = nn(t', p)'

loss(p) = begin
  yhat = predict(dataset_in, p)
  l = Flux.mse(yhat, dataset_out)
end

learning_rate=1e-2
opt = ADAMW(learning_rate)
epochs = 1500

function cb_train(theta, loss)
    println("Loss: ", loss)
    false
end

res_train = DiffEqFlux.sciml_train(
    loss, theta, opt,
    maxiters = epochs,
    cb = cb_train)

y_pred = predict(dataset_in, res_train.minimizer)

pl = plot(
    dataset_in,
    dataset_out,
    linewidth=2, ls=:dash,
    title="MLP by FastChain without Neural ODEs",
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
