import numpy as np
import matplotlib.pyplot as plt
from ddeint import ddeint

def equation(Y, t, d):
    x,y = Y(t)
    xd,yd = Y(t-d)
    return [x * yd, y * xd]

def initial_history_func(t):
    return [1., -1.]

plt.rcParams['font.size'] = 8
fig, axs = plt.subplots(1, 1)
fig.tight_layout(rect=[0, 0, 1, 0.95], pad=3.0)
fig.suptitle("$x'(t)=x(t) y(t-d); y'(t)=y(t) x(t-d)$ solved by ddeint")

ts = np.linspace(0, 3, 2000)

ys = ddeint(equation, initial_history_func, ts, fargs=(0.5,))
axs.plot(ts, ys[:,0], color='red', linewidth=1)
axs.plot(ts, ys[:,1], color='green', linewidth=1)
axs.set_title('$ihf_x(t)=1; ihf_y(t)=-1$; d=0.5')

plt.show()
