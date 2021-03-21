import argparse
from argparse import RawTextHelpFormatter
import os
import sympy as sp
import numpy as np
from numpy import linalg as npla
import matplotlib.pyplot as plt
import pylab as pl
from scipy.integrate import odeint

def init_phase_portrait():
    plt.xlabel('x', fontsize=args.font_size)
    plt.ylabel('y', fontsize=args.font_size)
    plt.tick_params(labelsize=args.font_size)
    plt.xlim(args.x0_begin, args.x0_end)
    plt.ylim(args.y0_begin, args.y0_end)

def plot_phase_portait():
    icx = np.linspace(args.x0_begin, args.x0_end, args.x0_num_of_samples)
    icy = np.linspace(args.y0_begin, args.y0_end, args.y0_num_of_samples)

    ts = np.linspace(0, args.t_end, args.t_num_of_samples)
    plot_trajectory(plt, icx, icy, ts)

    ts = np.linspace(0, -args.t_end, args.t_num_of_samples)
    plot_trajectory(plt, icx, icy, ts)

def plot_trajectory(plt, icx, icy, ts):
    for r in icx:
        for s in icy:
            x0 = [r, s]
            xs = odeint(dX_dt, x0, ts)
            plt.plot(xs[:, 0], xs[:, 1], color='red', linewidth=1)

def plot_gradient_vector():
    grdX, grdY = np.mgrid[args.x0_begin:args.x0_end:10j, args.y0_begin:args.y0_end:10j]
    grdXt = dX_dt([grdX, grdY], 0.)
    grdU = grdXt[0]
    grdV = grdXt[1]
    pl.quiver(grdX, grdY, grdU, grdV, width=0.003, color='blue')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='%(prog)s analyzes a dynamyc system modeled by a nonlinear planar system using Hartman theorem', formatter_class=RawTextHelpFormatter)

    parser.add_argument('--version', action='version', version='%(prog)s 1.0.0')

    parser.add_argument('--dX_dt',
                        type=str,
                        dest='func_dX_dt_body',
                        required=True,
                        help='dX_dt(X, t) body (lamba format)')

    parser.add_argument('--t_end',
                        type=float,
                        dest='t_end',
                        default=10.,
                        required=False,
                        help='In the phase portait diagram, it is the final value of the interval of variable t (starting value of t is 0).\nFor backward time trajectories, t goes from -t_end to 0;\nfor forward time trajectories, t goes from 0 to t_end.')

    parser.add_argument('--t_num_of_samples',
                        type=int,
                        dest='t_num_of_samples',
                        default=100,
                        required=False,
                        help='In the phase portait diagram, it is the number of samples of variable t between -t_end and 0 for backward time trajectories\nand also it is the number of samples of variable t between 0 and t_end for forward time trajectories')

    parser.add_argument('--x0_begin',
                        type=float,
                        dest='x0_begin',
                        default=-5.,
                        required=False,
                        help='In the phase portait diagram, it is the starting value of the interval of initial condition x0')

    parser.add_argument('--x0_end',
                        type=float,
                        dest='x0_end',
                        default=5.,
                        required=False,
                        help='In the phase portait diagram, it is the final value of the interval of initial condition x0')

    parser.add_argument('--x0_num_of_samples',
                        type=int,
                        dest='x0_num_of_samples',
                        default=6,
                        required=False,
                        help='In the phase portait diagram, it is the number of samples of initial condition x0 between x0_begin and x0_end')

    parser.add_argument('--y0_begin',
                        type=float,
                        dest='y0_begin',
                        default=-5.,
                        required=False,
                        help='In the phase portait diagram, it is the starting value of of interval for initial condition y0')

    parser.add_argument('--y0_end',
                        type=float,
                        dest='y0_end',
                        default=5.,
                        required=False,
                        help='In the phase portait diagram, it is the final value of of interval for initial condition y0')

    parser.add_argument('--y0_num_of_samples',
                        type=int,
                        dest='y0_num_of_samples',
                        default=6,
                        required=False,
                        help='In the phase portait diagram, it is the number of samples of initial condition y0 between y0_begin and y0_end')

    parser.add_argument('--plot_favourite_sol',
                        type=bool,
                        dest='plot_favourite_sol',
                        default=False,
                        required=False,
                        help="'yes' to plot the favourite solution;\nif it is 'yes' C1 and C2 are used to choose the favourite solution")

    parser.add_argument('--C1',
                        type=float,
                        dest='constant_of_integration_C1',
                        default=1.0,
                        required=False,
                        help='Value of constant of integration C1 to choose (together with C2) the favourite solution')

    parser.add_argument('--C2',
                        type=float,
                        dest='constant_of_integration_C2',
                        default=1.0,
                        required=False,
                        help='Value of constant of integration C2 to chhose (together with C1) the favourite solution')

    parser.add_argument('--font_size',
                        type=int,
                        dest='font_size',
                        default=10,
                        required=False,
                        help='font size')

    args = parser.parse_args()

    sp.init_printing()

    print("#### Started %s ####" % os.path.basename(__file__));

    dX_dt = eval('lambda X, t: ' + args.func_dX_dt_body)

    init_phase_portrait()
    plot_phase_portait()
    plot_gradient_vector()
    plt.show()

    print("#### Terminated %s ####" % os.path.basename(__file__));
