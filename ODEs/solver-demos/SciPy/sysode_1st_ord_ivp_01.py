#Please see
#https://computationalmindset.com/en/neural-networks/ordinary-differential-equation-solvers.html#sys1
#for details

import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp

def ode_fn(t, XY):
	x=XY[0]
	y=XY[1]
	dx_dt= - x + y
	dy_dt= 4. * x - y
	return [dx_dt, dy_dt]

an_sol_x = lambda t : np.exp(t) + np.exp(-3. * t)
an_sol_y = lambda t : 2. * np.exp(t) - 2. * np.exp(-3. * t)

t_begin=0.
t_end=5.
t_nsamples=100
t_space = np.linspace(t_begin, t_end, t_nsamples)
x_init = 2.
y_init = 0.

x_an_sol = an_sol_x(t_space)
y_an_sol = an_sol_y(t_space)

method = 'RK45' #available methods: 'RK45', 'RK23', 'DOP853', 'Radau', 'BDF', 'LSODA'
num_sol = solve_ivp(ode_fn, [t_begin, t_end], [x_init, y_init], method=method, dense_output=True)
XY_num_sol = num_sol.sol(t_space)
x_num_sol = XY_num_sol[0].T
y_num_sol = XY_num_sol[1].T

plt.figure()
plt.plot(t_space, x_an_sol, label='analytical x')
plt.plot(t_space, y_an_sol, label='analytical y')
plt.plot(t_space, x_num_sol, label='numerical x')
plt.plot(t_space, y_num_sol, label='numerical y')
plt.title('System of 2 ODEs 1st order IVP solved by SciPy with method=' + method)
plt.xlabel('t')
plt.legend()
plt.show()

