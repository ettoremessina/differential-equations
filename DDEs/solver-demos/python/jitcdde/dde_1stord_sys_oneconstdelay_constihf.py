import numpy as np
import matplotlib.pyplot as plt
import symengine
from jitcdde import jitcdde, y, t

d=symengine.symbols("d")
equations=[
	y(0, t) * y(1, t-d),
	y(1, t) * y(0, t-d)]
ddesys = jitcdde(equations, control_pars=[d], max_delay=100.)

plt.rcParams['font.size'] = 8
fig, axs = plt.subplots(1, 1)
fig.tight_layout(rect=[0, 0, 1, 0.95], pad=3.0)
fig.suptitle("$x'(t)=x(t) y(t-d); y'(t)=y(t) x(t-d)$ solved by jitcdde")

ts = np.linspace(0, 3, 2000)

ddesys.constant_past([1., -1.])
params=[0.5]
ddesys.set_parameters(*params)
ys = []
for t in ts:
	ys.append(ddesys.integrate(t))
ys=np.array(ys)
axs.plot(ts, ys[:,0], color='red', linewidth=1)
axs.plot(ts, ys[:,1], color='green', linewidth=1)
axs.set_title('$ihf_x(t)=1; ihf_y(t)=-1$; d=0.5')

plt.show()
