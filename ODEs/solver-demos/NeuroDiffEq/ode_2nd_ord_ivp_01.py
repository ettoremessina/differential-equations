"""
Please see
https://computationalmindset.com/en/neural-networks/ordinary-differential-equation-solvers.html#sys1
for details
"""

# x'' + x' + 2x = 0

import numpy as np
import matplotlib.pyplot as plt
import torch

from neurodiffeq import diff
from neurodiffeq.ode import solve
from neurodiffeq.ode import IVP
from neurodiffeq.ode import Monitor
import neurodiffeq.networks as ndenw

ode_fn = lambda x, t: diff(x, t, order=2) + diff(x, t, order=1) + 2. * x

an_sol = lambda t : \
	np.exp(-t/2.) * (np.cos(np.sqrt(7) * t / 2.) + \
	np.sin(np.sqrt(7) * t / 2.)/np.sqrt(7.))

t_begin=0.
t_end=12.
t_nsamples=100
t_space = np.linspace(t_begin, t_end, t_nsamples)
x_init = IVP(t_0=t_begin, x_0=1.0, x_0_prime=0.0)

x_an_sol = an_sol(t_space)

net = ndenw.FCNN(n_hidden_layers=6, n_hidden_units=50, actv=torch.nn.Tanh)
optimizer = torch.optim.Adam(net.parameters(), lr=0.002)
num_sol, loss_sol = solve(ode_fn, x_init, t_min=t_begin, t_max=t_end,
	batch_size=200,
	max_epochs=500,
	return_best=True,
	net=net,
	optimizer=optimizer,
	monitor=Monitor(t_min=t_begin, t_max=t_end, check_every=10))
x_num_sol = num_sol(t_space, as_type='np')

plt.figure()
plt.plot(t_space, x_an_sol, '--', linewidth=2, label='analytical')
plt.plot(t_space, x_num_sol, linewidth=1, label='numerical')
plt.title('ODE 2nd order IVP solved by NeuroDiffEq by FCNN')
plt.xlabel('t')
plt.ylabel('x')
plt.legend()
plt.show()

