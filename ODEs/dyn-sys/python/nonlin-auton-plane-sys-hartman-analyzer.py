import argparse
from argparse import RawTextHelpFormatter
import os
import sympy as sp
import numpy as np
from numpy import linalg as npla
import matplotlib.pyplot as plt
import warnings

from _plane_sys_args_aux import add_xyt_params, add_plot_params
from _homo_plane_sys_analyzer_aux import analyze_homo_sys_2x2
from _plane_sys_plot_aux import \
    init_phase_portrait, \
    plot_phase_portait, \
    plot_gradient_vector, \
    plot_eigenvectors

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
            cpKind, sym_sol = analyze_homo_sys_2x2(
                                                   npJcp, spJcp,
                                                   eigenvalue1, eigenvalue2,
                                                   eigenvector1, eigenvector2)
            print('Kind of critical point(s) : ', cpKind)
            plot_eigenvectors(critical_point, eigenvalue1, eigenvalue2, eigenvector1, eigenvector2)
        else:
            print('Type of c.p.              : ', 'Nonhyperbolic')
            print('\tSo Hartman theorem cannot be applied to this critical point')
    else:
        print('Unexpected number of eigenvalues: they must be 2')

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

    add_xyt_params(parser)
    add_plot_params(parser)

    args = parser.parse_args()

    print("#### Started %s ####" % os.path.basename(__file__));

    sp.init_printing()
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

    init_phase_portrait(args)
    plot_phase_portait(args, dX_dt)
    plot_gradient_vector(args, dX_dt)
    plt.show()

    print("#### Terminated %s ####" % os.path.basename(__file__));
