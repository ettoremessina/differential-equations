import numpy as np
import matplotlib.pyplot as plt
from jitcdde import jitcdde, y, t

equation= [-y(0, t-2.)]
dde = jitcdde(equation)

def initial_history_func_exp_mt(t):
    return [np.exp(-t) - 1]

def initial_history_func_exp_pt(t):
    return [np.exp(t) - 1]

plt.rcParams['font.size'] = 8
fig, axs = plt.subplots(2, 2)
fig.tight_layout(rect=[0, 0, 1, 0.95], pad=3.0)
fig.suptitle("$y'(t)=-y(t-2)$ solved by jitcdde")

ts = np.linspace(0, 4, 2000)

dde.past_from_function(initial_history_func_exp_mt)
ys = []
for t in ts:
	ys.append(dde.integrate(t))
axs[0, 0].plot(ts, ys, color='red', linewidth=1)
axs[0, 0].set_title('$ihf(t)=e^{-t} - 1, t \in [0, 4]$')

dde.past_from_function(initial_history_func_exp_pt)
ys = []
for t in ts:
	ys.append(dde.integrate(t))
axs[0, 1].plot(ts, ys, color='red', linewidth=1)
axs[0, 1].set_title('$ihf(t)=e^{t} - 1, t \in [0, 4]$')

ts = np.linspace(0, 60, 2000)

dde.past_from_function(initial_history_func_exp_mt)
ys = []
for t in ts:
	ys.append(dde.integrate(t))
axs[1, 0].plot(ts, ys, color='red', linewidth=1)
axs[1, 0].set_title('$ihf(t)=e^{-t} - 1, t \in [0, 60]$')

dde.past_from_function(initial_history_func_exp_mt)
ys = []
for t in ts:
	ys.append(dde.integrate(t))
axs[1, 1].plot(ts, ys, color='red', linewidth=1)
axs[1, 1].set_title('$ihf(t)=e^{t} - 1, t \in [0, 60]$')

plt.show()
