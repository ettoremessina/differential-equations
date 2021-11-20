import numpy as np
import matplotlib.pyplot as plt
from jitcdde import jitcdde, y, t

equation = [-y(0, t-1.)]
dde = jitcdde(equation)

plt.rcParams['font.size'] = 8
fig, axs = plt.subplots(3, 1)
fig.tight_layout(rect=[0, 0, 1, 0.95], pad=3.0)
fig.suptitle("$y'(t)=-y(t-1)$ solved by jitcdde")

ts = np.linspace(0, 20, 2000)

dde.constant_past([-1.])
ys = []
for t in ts:
	ys.append(dde.integrate(t))
axs[0].plot(ts, ys, color='red', linewidth=1)
axs[0].set_title('$ihf(t)=-1$')

dde.constant_past([0])
ys = []
for t in ts:
	ys.append(dde.integrate(t))
axs[1].plot(ts, ys, color='red', linewidth=1)
axs[1].set_title('$ihf(t)=0$')

dde.constant_past([1.])
ys = []
for t in ts:
	ys.append(dde.integrate(t))
axs[2].plot(ts, ys, color='red', linewidth=1)
axs[2].set_title('$ihf(t)=1$')

plt.show()
