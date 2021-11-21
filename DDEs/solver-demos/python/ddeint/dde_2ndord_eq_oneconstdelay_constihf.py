import numpy as np
import matplotlib.pyplot as plt
from ddeint import ddeint

def equation(Y, t):
    y,dydt = Y(t)
    ydelay, dydt_delay = Y(t-1)
    return [dydt, -dydt - 2 * y - 0.5 * ydelay]

def initial_history_func(t):
    return [1., 0.]

plt.rcParams['font.size'] = 8
fig, axs = plt.subplots(1, 1)
fig.tight_layout(rect=[0, 0, 1, 0.95], pad=3.0)
fig.suptitle("$y''(t)=-y'(t) - 2 y(t) - 0.5 y(t-1)$ solved by ddeint")

ts = np.linspace(0, 16, 4000)

ys = ddeint(equation, initial_history_func, ts)
axs.plot(ts, ys[:,0], color='red', linewidth=1)
#axs.plot(ts, ys[:,1], color='green', linewidth=1)
axs.set_title('$ihf_y(t)=1; ihf_{dy/dt}(t)=0$')

plt.show()
