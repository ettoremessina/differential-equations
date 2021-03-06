import argparse
from argparse import RawTextHelpFormatter
import os
import sympy as sp
import numpy as np
from numpy import linalg as npla
import matplotlib.pyplot as plt
import pylab as pl
from scipy.integrate import odeint

def dX_dt(X, t):
    return [
        args.matrix[0] * X[0] + args.matrix[1] * X[1],
        args.matrix[2] * X[0] + args.matrix[3] * X[1]]

def analyze_homo_sys_2x2(detA, eigW1, eigW2, eigV1, eigV2):
    t = sp.symbols('t')
    C1, C2 = sp.symbols('C1 C2', real = True, constant = True)

    cpKind = 'unknown'
    sym_sol = 'unknown'

    if not np.iscomplex(eigW1):
        eigV1r = sp.Array(eigV1)
        eigV2r = sp.Array(eigV2)

        sol1 = C1 * sp.exp(eigW1 * t) * eigV1r
        sol2 = C2 * sp.exp(eigW2 * t) * eigV2r

        if eigW1 > 0 and eigW2 > 0:
            cpKind = 'unstable node'
        elif (eigW1 > 0 and eigW2 < 0) or (eigW1 < 0 and eigW2 > 0):
            cpKind = 'saddle point'
        elif eigW1 < 0 and eigW2 < 0:
            cpKind = 'stable node'

    else:
        eigW1r = np.real(eigW1)
        eigW1i = np.imag(eigW1)
        eigW2r = np.real(eigW2)
        eigW2i = np.imag(eigW2)
        eigV1r = sp.Array(np.real(eigV1))
        eigV1i = sp.Array(np.imag(eigV1))
        eigV2r = sp.Array(np.real(eigV2))
        eigV2i = sp.Array(np.imag(eigV2))

        sol1 = C1 * sp.exp(eigW1r * t) * \
                    (eigV1r * sp.cos(eigW1i * t) - eigV1i * sp.sin(eigW1i * t))
        sol2 = C2 * sp.exp(eigW2r * t) * \
                    (eigV2r * sp.cos(eigW2i * t) - eigV2i * sp.sin(eigW2i * t))

        if eigW1r > 0:
            cpKind = 'unstable focus'
        elif eigW1r == 0:
            cpKind = 'center'
        elif eigW1r < 0:
            cpKind = 'stable focus'

    sym_sol = sol1 + sol2
    return cpKind, sym_sol

def init_plt():
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
    grdU = args.matrix[0] * grdX + args.matrix[1] * grdY
    grdV = args.matrix[2] * grdX + args.matrix[3] * grdY
    pl.quiver(grdX, grdY, grdU, grdV, width=0.003, color='blue')

def plot_eigenvectors(eigW1, eigW2, eigV1, eigV2):
    ox = np.array((0))
    oy = np.array((0))

    if not np.iscomplex(eigW1):
        v1x = np.array((eigV1[0]))
        v1y = np.array((eigV1[1]))
        plt.quiver(ox, oy, v1x, v1y, units='xy',scale=1, angles='xy', 
            color='green' if eigW1 < 0 else 'magenta')

    if not np.iscomplex(eigW2):
        v2x = np.array((eigV2[0]))
        v2y = np.array((eigV2[1]))
        plt.quiver(ox, oy, v2x, v2y, units='xy',scale=1, angles='xy',
            color='green' if eigW2 < 0 else 'magenta')

def plot_favourite_solution(sym_sol):
    def plot_favourite_solution_aux(lambda_favourite_sol, ts):
      xs = lambda_favourite_sol(args.constant_of_integration_C1, args.constant_of_integration_C2, ts)
      plt.plot(xs[0], xs[1], color='gold', linewidth=2)

    t = sp.symbols('t')
    C1, C2 = sp.symbols('C1 C2', real = True, constant = True)
    lambda_favourite_sol = sp.lambdify([C1, C2, t], sym_sol)

    ts = np.linspace(0, args.t_end, args.t_num_of_samples)
    plot_favourite_solution_aux(lambda_favourite_sol, ts)
    ts = np.linspace(0, -args.t_end, args.t_num_of_samples)
    plot_favourite_solution_aux(lambda_favourite_sol, ts)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='%(prog)s analyzes a dynamyc system modeled by a linear planar system', formatter_class=RawTextHelpFormatter)

    parser.add_argument('--version', action='version', version='%(prog)s 1.0.0')

    parser.add_argument('--matrix',
                        type=float,
                        dest='matrix',
                        required=True,
                        nargs = '+',
                        default = [],
                        help='coefficents of matrix \n|a b|\n|c d| of the system')

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

    if len(args.matrix) != 4:
        raise Exception('Number of coefficient of matrix has to be 4, because the matrix of a planar system is 2x2')

    matrixA = np.matrix(
          [
              [args.matrix[0], args.matrix[1]],
              [args.matrix[2], args.matrix[3]]
          ])

    detA = npla.det(matrixA)
    if (np.isclose(detA, 0.)):
        raise Exception('At moment this program supports only simple system; simple means det(A) != 0')

    eigenvalues, eigenvectors = npla.eig(matrixA)

    if len(eigenvalues) != 2:
        raise Exception('Unexpected number of eigenvalues: they must be 2')

    eigenvalue1 = eigenvalues[0]
    eigenvalue2 = eigenvalues[1]
    eigenvector1 = [eigenvectors[0, 0], eigenvectors[1, 0]] 
    eigenvector2 = [eigenvectors[0, 1], eigenvectors[1, 1]]

    print('Critical point   : ', [0, 0])
    print('Determinant      : ', detA)
    print('Eigenvalues      : ', eigenvalues)
    print('Eigenvector 1    : ', eigenvector1)
    print('Eigenvector 2    : ', eigenvector2)

    cpKind, sym_sol = analyze_homo_sys_2x2(detA, eigenvalue1, eigenvalue2, eigenvector1, eigenvector2)
    print('Kind of c.p.     : ', cpKind)
    print('General solution :')
    sp.pprint(sym_sol)

    init_plt()
    plot_phase_portait()
    plot_gradient_vector()
    plot_eigenvectors(eigenvalue1, eigenvalue2, eigenvector1, eigenvector2)
    if (args.plot_favourite_sol):
        plot_favourite_solution(sym_sol)

    plt.show()

    print("#### Terminated %s ####" % os.path.basename(__file__));