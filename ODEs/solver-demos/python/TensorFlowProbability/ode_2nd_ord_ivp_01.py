"""
Please see
https://computationalmindset.com/en/neural-networks/ordinary-differential-equation-solvers.html#sys1
for details
"""

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_probability as tfp

def ode_sys(t, X):
	x=X[0]
	dx_dt=X[1]
	d2x_dt2=-dx_dt - 2*x
	return [dx_dt, d2x_dt2]

an_sol_x = lambda t : \
	np.exp(-t/2.) * (np.cos(np.sqrt(7) * t / 2.) + \
	np.sin(np.sqrt(7) * t / 2.)/np.sqrt(7.))

t_begin=0.
t_end=12.
t_nsamples=100
t_space = np.linspace(t_begin, t_end, t_nsamples)
t_init = tf.constant(t_begin)
x_init = tf.constant(1.)
dxdt_init = tf.constant(0.)

x_an_sol = an_sol_x(t_space)

num_sol = tfp.math.ode.BDF().solve(ode_sys, t_init, [x_init, dxdt_init],
	solution_times=tfp.math.ode.ChosenBySolver(tf.constant(t_end)) )

plt.figure()
plt.plot(t_space, x_an_sol, '--', linewidth=2, label='analytical')
plt.plot(num_sol.times, num_sol.states[0], linewidth=1, label='numerical')
plt.title('ODE 2nd order IVP solved by TFP with BDF')
plt.xlabel('t')
plt.ylabel('x')
plt.legend()
plt.show()

