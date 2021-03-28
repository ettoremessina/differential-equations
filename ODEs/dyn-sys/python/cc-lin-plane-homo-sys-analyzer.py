import argparse
from argparse import RawTextHelpFormatter
import os
import sympy as sp
import numpy as np
from numpy import linalg as npla
import matplotlib.pyplot as plt
import pylab as pl
from scipy.integrate import odeint

from _plane_sys_args_aux import add_xyt_params, add_plot_params
from _homo_plane_sys_analyzer_aux import analyze_homo_sys_2x2
from _plane_sys_plot_aux import \
    init_phase_portrait, \
    plot_phase_portait, \
    plot_gradient_vector

def dX_dt(X, t):
    return [
        args.matrix[0] * X[0] + args.matrix[1] * X[1],
        args.matrix[2] * X[0] + args.matrix[3] * X[1]]

def compute_critical_points(spA):
    x, y = sp.symbols('x, y')
    return sp.linsolve((spA, [0, 0]), [x, y])

def plot_eigenvectors(eigW1, eigW2, eigV1, eigV2):
    ox = np.array((0))
    oy = np.array((0))

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

def plot_favourite_trajectory(sym_sol):
    def plot_favourite_trajectory_aux(lambda_favourite_sol, ts):
        xs = lambda_favourite_sol(args.constant_of_integration_C1, args.constant_of_integration_C2, ts)
        plt.plot(xs[0], xs[1], color='gold', linewidth=2)

    t = sp.symbols('t')
    C1, C2 = sp.symbols('C1 C2', real = True, constant = True)
    lambda_favourite_sol = sp.lambdify([C1, C2, t], sym_sol)

    ts = np.linspace(0, args.t_end, args.t_num_of_samples)
    plot_favourite_trajectory_aux(lambda_favourite_sol, ts)
    ts = np.linspace(0, -args.t_end, args.t_num_of_samples)
    plot_favourite_trajectory_aux(lambda_favourite_sol, ts)

def init_solution_graph():
    plt.xlabel('t', fontsize=args.font_size)
    #plt.ylabel('y', fontsize=args.font_size)
    plt.tick_params(labelsize=args.font_size)
    plt.xlim(0, args.t_end)

def plot_favourite_solution(sym_sol):
    t = sp.symbols('t')
    C1, C2 = sp.symbols('C1 C2', real = True, constant = True)
    lambda_favourite_sol = sp.lambdify([C1, C2, t], sym_sol)

    ts = np.linspace(0, args.t_end, args.t_num_of_samples)
    xs = lambda_favourite_sol(args.constant_of_integration_C1, args.constant_of_integration_C2, ts)
    plt.plot(ts, xs[0], color='blue', linewidth=1, label='x(t)')
    plt.plot(ts, xs[1], color='purple', linewidth=1, label='y(t)')
    plt.legend()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='%(prog)s analyzes a dynamyc system modeled by a linear planar system', formatter_class=RawTextHelpFormatter)

    parser.add_argument('--version', action='version', version='%(prog)s 1.0.2')

    parser.add_argument('--matrix',
                        type=float,
                        dest='matrix',
                        required=True,
                        nargs = '+',
                        default = [],
                        help='coefficents of matrix \n|a b|\n|c d| of the system')

    add_xyt_params(parser)

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

    add_plot_params(parser)

    args = parser.parse_args()

    print("#### Started %s ####" % os.path.basename(__file__));

    sp.init_printing()

    if len(args.matrix) != 4:
        raise Exception('Number of coefficient of matrix has to be 4, because the matrix of a planar system is 2x2')

    npA = np.matrix(
          [
              [args.matrix[0], args.matrix[1]],
              [args.matrix[2], args.matrix[3]]
          ])
    spA = sp.Matrix(npA)
    detA = npla.det(npA)

    eigenvalues, eigenvectors = npla.eig(npA)

    if len(eigenvalues) != 2:
        raise Exception('Unexpected number of eigenvalues: they must be 2')

    eigenvalue1 = eigenvalues[0]
    eigenvalue2 = eigenvalues[1]
    eigenvector1 = [eigenvectors[0, 0], eigenvectors[1, 0]]
    eigenvector2 = [eigenvectors[0, 1], eigenvectors[1, 1]]

    critical_points = compute_critical_points(spA)

    print('Critical point(s)         : ', sp.pretty(critical_points))
    print('Determinant               : ', detA)
    print('Eigenvalues               : ', eigenvalue1, eigenvalue2)
    print('Eigenvector 1             : ', eigenvector1)
    print('Eigenvector 2             : ', eigenvector2)

    cpKind, sym_sol = analyze_homo_sys_2x2(npA, spA, eigenvalue1, eigenvalue2, eigenvector1, eigenvector2)
    print('Kind of critical point(s) : ', cpKind)
    print('General solution          :')
    sp.pprint(sym_sol)

    init_phase_portrait(args)
    plot_phase_portait(args, dX_dt)
    if cpKind != 'whole plane':
        plot_gradient_vector(args, dX_dt)
    plot_eigenvectors(eigenvalue1, eigenvalue2, eigenvector1, eigenvector2)
    if (sym_sol != None and args.plot_favourite_sol and cpKind != 'degenerate line'):
        plot_favourite_trajectory(sym_sol)
    plt.show()

    if (sym_sol != None and args.plot_favourite_sol):
        init_solution_graph()
        plot_favourite_solution(sym_sol)
        plt.show()

    print("#### Terminated %s ####" % os.path.basename(__file__));
