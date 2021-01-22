import numpy as np
import matplotlib.pyplot as plt

from sympy import *
x, y, z, t = symbols('x y z t')
k, m, n = symbols('k m n', integer=True)
f, g, h = symbols('f g h', cls=Function)

eq = Eq(f(t).diff(t), -(1/t) * f(t) - (1/t) * f(t)**4)
an_sol = dsolve(eq, ics={f(1): 1})
print('ODE class: ', classify_ode(eq)[0])
pprint(an_sol)

t_begin=1.
t_end=10.
t_nsamples=101
t_space = np.linspace(t_begin, t_end, t_nsamples)

lmbd_sol = lambdify(t, an_sol.rhs)
x_an_sol = lmbd_sol(t_space)

plt.figure()
plt.plot(t_space, x_an_sol, linewidth=1, label='analytical')
plt.title('bernoulli ODE 1st order')
plt.xlabel('t')
plt.ylabel('f(t)')
plt.legend()
plt.show()
