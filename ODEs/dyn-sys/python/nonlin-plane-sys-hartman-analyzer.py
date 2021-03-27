import argparse
from argparse import RawTextHelpFormatter
import os
import sympy as sp
import numpy as np
from numpy import linalg as npla
import matplotlib.pyplot as plt
import pylab as pl
from scipy.integrate import odeint
import warnings

def find_critical_points():
    x, y = sp.symbols('x, y')
    p = eval(args.func_dx_dt_body)
    q = eval(args.func_dy_dt_body)
    p_eq_0 = sp.Eq(p, 0)
    q_eq_0 = sp.Eq(q, 0)
    critical_points = sp.solve((p_eq_0, q_eq_0), x, y)
    return critical_points

def build_jacobian():
    x, y = sp.symbols('x, y')
    p = eval(args.func_dx_dt_body)
    q = eval(args.func_dy_dt_body)
    J = [[sp.diff(p, x), sp.diff(p, y)],
         [sp.diff(q, x), sp.diff(q, y)]]
    return sp.Matrix(J)

def analyze_critical_point(J, critical_point):
    x, y = sp.symbols('x, y')
    Jcp = sp.lambdify([x, y], J) (critical_point[0], critical_point[1])
    npJcp = np.matrix(Jcp)
    print('*************************   ')
    print('Critical point            : ', critical_point)
    spJcp = sp.Matrix(npJcp)
    print('Jacobian at c.p.          : ')
    print(sp.pretty(spJcp))
    detJcp = npla.det(npJcp)
    eigenvalues, eigenvectors = npla.eig(npJcp)
    if len(eigenvalues) == 2:
        eigenvalues, eigenvectors = npla.eig(npJcp)
        eigenvalue1 = eigenvalues[0]
        eigenvalue2 = eigenvalues[1]
        eigenvector1 = [eigenvectors[0, 0], eigenvectors[1, 0]]
        eigenvector2 = [eigenvectors[0, 1], eigenvectors[1, 1]]
        print('Determinant               : ', detJcp)
        print('Eigenvalues               : ', eigenvalue1, eigenvalue2)
        print('Eigenvector 1             : ', eigenvector1)
        print('Eigenvector 2             : ', eigenvector2)

        eigenvalue1r = eigenvalue1
        eigenvalue2r = eigenvalue2
        if np.iscomplex(eigenvalue1):
            eigenvalue1r = np.real(eigenvalue1)
        if np.iscomplex(eigenvalue2):
            eigenvalue2r = np.real(eigenvalue2)
        if not np.isclose(eigenvalue1r, 0) and not np.isclose(eigenvalue2r, 0):
            print('Type of c.p.              : ', 'Hyperbolic')
        else:
            print('Type of c.p.              : ', 'Nonhyperbolic')
            print('\tSo Hartman theorem cannot be applied to this critical point')
    else:
        print('Unexpected number of eigenvalues: they must be 2')

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

    parser.add_argument('--dx_dt',
                        type=str,
                        dest='func_dx_dt_body',
                        required=True,
                        help='dx_dt(x, y, t) body (lamba format)')

    parser.add_argument('--dy_dt',
                        type=str,
                        dest='func_dy_dt_body',
                        required=True,
                        help='dy_dt(x, y, t) body (lamba format)')

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

    warnings.filterwarnings("ignore")

    strEq = '[' + args.func_dx_dt_body + ', ' + args.func_dy_dt_body + ']'
    strEq = strEq.replace('x', 'X[0]')
    strEq = strEq.replace('y', 'X[1]')
    dX_dt = eval('lambda X, t: ' + strEq)
    J = build_jacobian()

    sym_critical_points = find_critical_points()
    string_critical_points = sp.pretty(sym_critical_points)
    print('Critical point(s)         : ', string_critical_points)
    print('Formal Jacobian           : ')
    print(sp.pretty(J))

    if not ('x' in string_critical_points) and not ('y' in string_critical_points):
        critical_points = sp.lambdify([], sym_critical_points)()
        for critical_point in critical_points:
            analyze_critical_point(J, critical_point)
    else:
        print("Critical points are infinite;\n\tthis program supports only finite and hyperbolic critical points")

    init_phase_portrait()
    plot_phase_portait()
    plot_gradient_vector()
    plt.show()

    print("#### Terminated %s ####" % os.path.basename(__file__));
