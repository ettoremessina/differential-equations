import numpy as np
import matplotlib.pyplot as plt

from sympy import *
x, y, z, t = symbols('x y z t')
k, m, n = symbols('k m n', integer=True)
f, g, h = symbols('f g h', cls=Function)

eq = eq = Eq(f(t).diff(t), (t**2 + f(t)**2)/(t * f(t)))
an_sol = dsolve(eq, ics={f(2): 2}, hint='1st_homogeneous_coeff_best')
print('ODE class: ', classify_ode(eq)[0])
pprint(an_sol)

t_begin=2.
t_end=12.
t_nsamples=101
t_space = np.linspace(t_begin, t_end, t_nsamples)

lmbd_sol = lambdify(t, an_sol.rhs)
x_an_sol = lmbd_sol(t_space)

plt.figure()
plt.plot(t_space, x_an_sol, linewidth=1, label='analytical')
plt.title('homogeneous ODE 1st order')
plt.xlabel('t')
plt.ylabel('f(t)')
plt.legend()
plt.show()
