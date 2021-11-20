import numpy as np
import matplotlib.pyplot as plt
from jitcdde import jitcdde, y, t

equation=[-y(0, t - 1) + 0.3 * y(0, t - 2)]
dde = jitcdde(equation)

plt.rcParams['font.size'] = 8
fig, axs = plt.subplots(1, 1)
fig.tight_layout(rect=[0, 0, 1, 0.95], pad=3.0)
fig.suptitle("$y'(t)=-y(t-1) + 0.3\ y(t-2)$ solved by jitcdde")

ts = np.linspace(0, 10, 10000)

dde.constant_past([1.])
ys = []
for t in ts:
	ys.append(dde.integrate(t))
axs.plot(ts, ys, color='red', linewidth=1)
axs.set_title('$ihf(t)=1$')

plt.show()
