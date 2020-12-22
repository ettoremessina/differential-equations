"""
Please see
https://computationalmindset.com/en/neural-networks/ordinary-differential-equation-solvers.html#sys1
for details
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp

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
x_init = 1.
dxdt_init = 0.

x_an_sol = an_sol_x(t_space)

method = 'RK45' #available methods: 'RK45', 'RK23', 'DOP853', 'Radau', 'BDF', 'LSODA'
num_sol = solve_ivp(ode_sys, [t_begin, t_end], [x_init, dxdt_init], method=method, dense_output=True)
X_num_sol = num_sol.sol(t_space)
x_num_sol = X_num_sol[0].T

plt.figure()
plt.plot(t_space, x_an_sol, '--', linewidth=2, label='analytical')
plt.plot(t_space, x_num_sol, linewidth=1, label='numerical')
plt.title('ODE 2nd order IVP solved by SciPy with method=' + method)
plt.xlabel('t')
plt.ylabel('x')
plt.legend()
plt.show()

