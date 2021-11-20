import numpy as np
import matplotlib.pyplot as plt
from ddeint import ddeint

def equation(Y, t):
    return -Y(t - 1) + 0.3 * Y(t - 2)

def initial_history_func(t):
    return 1.

plt.rcParams['font.size'] = 8
fig, axs = plt.subplots(1, 1)
fig.tight_layout(rect=[0, 0, 1, 0.95], pad=3.0)
fig.suptitle("$y'(t)=-y(t-1) + 0.3\ y(t-2)$ solved by ddeint")

ts = np.linspace(0, 10, 10000)

ys = ddeint(equation, initial_history_func, ts)
axs.plot(ts, ys, color='red', linewidth=1)
axs.set_title('$ihf(t)=1$')

plt.show()
