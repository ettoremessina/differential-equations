import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_probability as tfp

#https://computationalmindset.com/en/neural-networks/ordinary-differential-equation-solvers.html#ODE1stIVP

ode_fn = lambda t, x: tf.math.sin(t) + tf.constant(3.) * tf.math.cos(tf.constant(2.) * t) - x

an_sol = lambda t : (1./2.) * np.sin(t) - (1./2.) * np.cos(t) + \
                    (3./5.) * np.cos(2.*t) + (6./5.) * np.sin(2.*t) - \
                    (1./10.) * np.exp(-t)

t_begin=0.
t_end=10.
t_nsamples=100
t_space = np.linspace(t_begin, t_end, t_nsamples)
t_init = tf.constant(t_begin)
x_init = tf.constant(0.)

x_an_sol = an_sol(t_space)

num_sol = tfp.math.ode.BDF().solve(ode_fn, t_init, x_init,
	solution_times=tfp.math.ode.ChosenBySolver(tf.constant(t_end)) )

plt.figure()
plt.plot(t_space, x_an_sol, label='analytical')
plt.plot(num_sol.times, num_sol.states, label='numerical')
plt.title('ODE 1st order IVP solved by TFP with BDF')
plt.xlabel('t')
plt.ylabel('x')
plt.legend()
plt.show()

