import numpy as np
import matplotlib.pyplot as plt
from ddeint import ddeint

def equation(Y, t):
    return -Y(t - 1)

def initial_history_func_m1(t):
    return -1.

def initial_history_func_0(t):
    return 0.

def initial_history_func_p1(t):
    return 1.

plt.rcParams['font.size'] = 8
fig, axs = plt.subplots(3, 1)
fig.tight_layout(rect=[0, 0, 1, 0.95], pad=3.0)
fig.suptitle("$y'(t)=-y(t-1)$ solved by ddeint")

ts = np.linspace(0, 20, 2000)

ys = ddeint(equation, initial_history_func_m1, ts)
axs[0].plot(ts, ys, color='red', linewidth=1)
axs[0].set_title('$ihf(t)=-1$')

ys = ddeint(equation, initial_history_func_0, ts)
axs[1].plot(ts, ys, color='red', linewidth=1)
axs[1].set_title('$ihf(t)=0$')

ys = ddeint(equation, initial_history_func_p1, ts)
axs[2].plot(ts, ys, color='red', linewidth=1)
axs[2].set_title('$ihf(t)=1$')

plt.show()
