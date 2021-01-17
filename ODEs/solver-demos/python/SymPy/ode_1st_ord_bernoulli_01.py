import numpy as np
import matplotlib.pyplot as plt

from sympy import *
x, y, z, t = symbols('x y z t')
k, m, n = symbols('k m n', integer=True)
f, g, h = symbols('f g h', cls=Function)

eq = Eq(f(t).diff(t), -f(t)/t - f(t)**4/t)
an_sol = dsolve(eq)
print('ODE class: ', classify_ode(eq)[0])
pprint(an_sol)

t_begin=0.
t_end=1.
t_nsamples=11
t_space = np.linspace(t_begin, t_end, t_nsamples)

lmbd_sol = lambdify(t, an_sol.rhs)
x_an_sol = lmbd_sol(t_space)

print(lmbd_sol(t_space))

plt.figure()
plt.plot(t_space, x_an_sol, linewidth=1, label='analytical')
plt.title('linear variable ODE 1st order')
plt.xlabel('t')
plt.ylabel('f(t)')
plt.legend()
plt.show()
