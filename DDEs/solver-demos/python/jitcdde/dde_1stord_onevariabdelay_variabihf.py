import numpy as np
import matplotlib.pyplot as plt
from jitcdde import jitcdde, y, t

def delay(y, t):
    return np.abs(0.1 * t * y(0, 0.1 * t))

equation = [y(0, t - delay(y, t))]
dde = jitcdde(equation, max_delay=1000)

def initial_history_func_exp_pt(t):
    return [np.exp(t)]

plt.rcParams['font.size'] = 8
fig, axs = plt.subplots(1, 1)
fig.tight_layout(rect=[0, 0, 1, 0.95], pad=3.0)
fig.suptitle("$y'(t)=y(t-delay(y, t))$ solved by jitcdde")

ts = np.linspace(0, 30, 10000)

dde.past_from_function(initial_history_func_exp_pt)
ys = []
for t in ts:
	ys.append(dde.integrate(t))
axs.plot(ts, ys, color='red', linewidth=1)
axs.set_title('$ihf(t)=1$')

plt.show()
