import numpy as np
import matplotlib.pyplot as plt
from ddeint import ddeint

def delay(Y, t):
    return np.abs(0.1 * t * Y(0.1 * t))

def equation(Y, t):
    return Y(t - delay(Y, t))

def initial_history_func_m1(t):
    return -1.

def initial_history_func_p1(t):
    return 1.

plt.rcParams['font.size'] = 8
fig, axs = plt.subplots(2, 1)
fig.tight_layout(rect=[0, 0, 1, 0.95], pad=3.0)
fig.suptitle("$y'(t)=y(t-delay(y, t))$ solved by ddeint")

ts = np.linspace(0, 50, 10000)

ys = ddeint(equation, initial_history_func_m1, ts)
axs[0].plot(ts, ys, color='red', linewidth=1)
axs[0].set_title('$ihf(t)=-1$')

ys = ddeint(equation, initial_history_func_p1, ts)
axs[1].plot(ts, ys, color='red', linewidth=1)
axs[1].set_title('$ihf(t)=1$')

plt.show()
