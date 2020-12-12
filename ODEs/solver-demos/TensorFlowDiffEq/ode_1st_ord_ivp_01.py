"""
Please see
https://computationalmindset.com/en/neural-networks/ordinary-differential-equation-solvers.html#ode1
for details
"""

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tfdiffeq import odeint

ode_fn = lambda t, x: tf.math.sin(t) + 3. * tf.math.cos(2. * t) - x

an_sol = lambda t : (1./2.) * np.sin(t) - (1./2.) * np.cos(t) + \
                    (3./5.) * np.cos(2.*t) + (6./5.) * np.sin(2.*t) - \
                    (1./10.) * np.exp(-t)

t_begin=0.
t_end=10.
t_nsamples=100
t_space = np.linspace(t_begin, t_end, t_nsamples)
x_init = tf.constant([0.])

x_an_sol = an_sol(t_space)

x_num_sol = odeint(ode_fn, x_init, tf.constant(t_space))

plt.figure()
plt.plot(t_space, x_an_sol, '--', linewidth=2, label='analytical')
plt.plot(t_space, x_num_sol, linewidth=1, label='numerical')
plt.title('ODE 1st order IVP solved by TensorFlowDiffEq')
plt.xlabel('t')
plt.ylabel('x')
plt.legend()
plt.show()

