import numpy as np
import matplotlib.pyplot as plt
from ddeint import ddeint

def equation(Y, t):
    return -Y(t - 2)

def initial_history_func_exp_mt(t):
    return np.exp(-t) - 1

def initial_history_func_exp_pt(t):
    return np.exp(t) - 1

plt.rcParams['font.size'] = 8
fig, axs = plt.subplots(2, 2)
fig.tight_layout(rect=[0, 0, 1, 0.95], pad=3.0)
fig.suptitle("$y'(t)=-y(t-2)$ solved by ddeint")

ts = np.linspace(0, 4, 2000)

ys = ddeint(equation, initial_history_func_exp_mt, ts)
axs[0, 0].plot(ts, ys, color='red', linewidth=1)
axs[0, 0].set_title('$ihf(t)=e^{-t} - 1, t \in [0, 4]$')

ys = ddeint(equation, initial_history_func_exp_pt, ts)
axs[0, 1].plot(ts, ys, color='red', linewidth=1)
axs[0, 1].set_title('$ihf(t)=e^{t} - 1, t \in [0, 4]$')

ts = np.linspace(0, 60, 2000)

ys = ddeint(equation, initial_history_func_exp_mt, ts)
axs[1, 0].plot(ts, ys, color='red', linewidth=1)
axs[1, 0].set_title('$ihf(t)=e^{-t} - 1, t \in [0, 60]$')

ys = ddeint(equation, initial_history_func_exp_pt, ts)
axs[1, 1].plot(ts, ys, color='red', linewidth=1)
axs[1, 1].set_title('$ihf(t)=e^{t} - 1, t \in [0, 60]$')

plt.show()
