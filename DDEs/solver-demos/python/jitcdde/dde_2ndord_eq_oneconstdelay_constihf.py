import numpy as np
import matplotlib.pyplot as plt
from jitcdde import jitcdde, y, t

equations=[
    y(1, t),
    -y(1, t) - 2 * y(0, t) - 0.5 * y(0, t-1)
]
ddesys = jitcdde(equations)

plt.rcParams['font.size'] = 8
fig, axs = plt.subplots(1, 1)
fig.tight_layout(rect=[0, 0, 1, 0.95], pad=3.0)
fig.suptitle("$y''(t)=-y'(t) - 2 y(t) - 0.5 y(t-1)$ solved by jitcdde")

ts = np.linspace(0, 16, 4000)

ddesys.constant_past([1., 0.])
ys = []
for t in ts:
	ys.append(ddesys.integrate(t))
ys=np.array(ys)
axs.plot(ts, ys[:,0], color='red', linewidth=1)
#axs.plot(ts, ys[:,1], color='green', linewidth=1)
axs.set_title('$ihf_y(t)=1; ihf_{dy/dt}(t)=0$')

plt.show()
