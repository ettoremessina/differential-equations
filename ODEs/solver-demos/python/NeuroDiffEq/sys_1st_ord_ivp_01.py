"""
Please see
https://computationalmindset.com/en/neural-networks/ordinary-differential-equation-solvers.html#sys1
for details
"""

import numpy as np
import matplotlib.pyplot as plt
import torch

from neurodiffeq import diff
from neurodiffeq.ode import solve_system
from neurodiffeq.ode import IVP
from neurodiffeq.ode import Monitor
import neurodiffeq.networks as ndenw

ode_sys = lambda x, y, t: [diff(x, t, order=1) + x - y, diff(y, t, order=1) - 4. * x + y ]

an_sol_x = lambda t : np.exp(t) + np.exp(-3. * t)
an_sol_y = lambda t : 2. * np.exp(t) - 2. * np.exp(-3. * t)

t_begin=0.
t_end=2.
t_nsamples=100
t_space = np.linspace(t_begin, t_end, t_nsamples)
x_init = IVP(t_0=t_begin, x_0=2.0)
y_init = IVP(t_0=t_begin, x_0=0.0)

x_an_sol = an_sol_x(t_space)
y_an_sol = an_sol_y(t_space)

batch_size=200

net = ndenw.FCNN(
	n_input_units=1,
        n_output_units=2,
	n_hidden_layers=3, 
	n_hidden_units=50, 
	actv=ndenw.SinActv)

optimizer = torch.optim.Adam(net.parameters(), lr=0.003)
 
num_sol, history = solve_system(
	ode_system=ode_sys,
	conditions=[x_init, y_init], 
	t_min=t_begin, 
	t_max=t_end,
	batch_size=batch_size,
	max_epochs=1200,
	return_best=True,
	single_net = net,
	optimizer=optimizer,
	monitor=Monitor(t_min=t_begin, t_max=t_end, check_every=10))
num_sol = num_sol(t_space, as_type='np')

plt.figure()
plt.plot(t_space, x_an_sol, '--', linewidth=2, label='analytical x')
plt.plot(t_space, y_an_sol, '--', linewidth=2, label='analytical y')
plt.plot(t_space, num_sol[0], linewidth=1, label='numerical x')
plt.plot(t_space, num_sol[1], linewidth=1, label='numerical y')
plt.title('System of two ODEs 1st order IVP solved by NeuroDiffEq')
plt.xlabel('t')
plt.legend()
plt.show()

