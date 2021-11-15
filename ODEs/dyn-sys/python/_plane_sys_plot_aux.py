import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
from scipy.integrate import odeint

def init_phase_portrait(args):
    plt.xlabel('x', fontsize=args.font_size)
    plt.ylabel('y', fontsize=args.font_size)
    plt.tick_params(labelsize=args.font_size)
    plt.xlim(args.x0_begin, args.x0_end)
    plt.ylim(args.y0_begin, args.y0_end)

def plot_phase_portait(args, dX_dt, plot_neg_time_traj=True):
    icx = np.linspace(args.x0_begin, args.x0_end, args.x0_num_of_samples)
    icy = np.linspace(args.y0_begin, args.y0_end, args.y0_num_of_samples)

    ts = np.linspace(0, args.t_end, args.t_num_of_samples)
    plot_trajectory(plt, dX_dt, icx, icy, ts)

    if plot_neg_time_traj:
        ts = np.linspace(0, -args.t_end, args.t_num_of_samples)
        plot_trajectory(plt, dX_dt, icx, icy, ts)

def plot_trajectory(plt, dX_dt, icx, icy, ts):
    for r in icx:
        for s in icy:
            x0 = [r, s]
            xs = odeint(dX_dt, x0, ts)
            plt.plot(xs[:, 0], xs[:, 1], color='red', linewidth=1)

def plot_gradient_vector(args, dX_dt):
    grdX, grdY = np.mgrid[args.x0_begin:args.x0_end:10j, args.y0_begin:args.y0_end:10j]
    grdXt = dX_dt([grdX, grdY], 0.)
    grdU = grdXt[0]
    grdV = grdXt[1]
    pl.quiver(grdX, grdY, grdU, grdV, width=0.003, color='blue')

def plot_eigenvectors(cp, eigW1, eigW2, eigV1, eigV2):
    ox = np.array((cp[0]))
    oy = np.array((cp[1]))

    if not np.iscomplex(eigW1):
        v1x = np.array((eigV1[0]))
        v1y = np.array((eigV1[1]))
        if not np.isclose(eigW1, 0.):
            color = 'green' if eigW1 < 0 else 'magenta'
        else:
            color = 'black'
        plt.quiver(ox, oy, v1x, v1y, units='xy',scale=1, angles='xy', color=color)

    if not np.iscomplex(eigW2):
        v2x = np.array((eigV2[0]))
        v2y = np.array((eigV2[1]))
        if not np.isclose(eigW2, 0.):
            color = 'green' if eigW2 < 0 else 'magenta'
        else:
            color = 'black'
        plt.quiver(ox, oy, v2x, v2y, units='xy',scale=1, angles='xy', color=color)
