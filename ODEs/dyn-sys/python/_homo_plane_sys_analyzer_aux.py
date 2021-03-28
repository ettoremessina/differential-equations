import sympy as sp
import numpy as np
from numpy import linalg as npla

def analyze_homo_sys_2x2(npA, spA, eigW1, eigW2, eigV1, eigV2):
    t = sp.symbols('t')
    C1, C2 = sp.symbols('C1 C2', real = True, constant = True)

    cpKind = ''
    sym_sol = None

    if not np.iscomplex(eigW1):
        eigV1r = sp.Array(eigV1)
        eigV2r = sp.Array(eigV2)

        sol1 = C1 * sp.exp(eigW1 * t) * eigV1r
        sol2 = C2 * sp.exp(eigW2 * t) * eigV2r

        if np.isclose(eigW1, eigW2):
            if not np.allclose(npA, [[0., 0.], [0., 0.]]):
                matrixVV = np.matrix([eigV1, eigV2])
                detVV = npla.det(matrixVV)
                if (not np.isclose(detVV, 0.)):
                    cpKind = ('stable' if eigW1 < 0 else 'unstable') + \
                              ' singular node (said also star point)'
                else:
                    if not np.isclose(eigW1, 0):
                        cpKind = ('stable' if eigW1 < 0 else 'unstable') + ' degenerate node'
                    else:
                        cpKind = 'degenerate line'
                    eigV1, eigV2 = compute_generalized_eigenvectors(spA)
                    if eigV1 != None and eigV2 != None:
                       print('Generalized Eigenvector 1 : ', eigV1)
                       print('Generalized Eigenvector 2 : ', eigV2)
                       eigV1r = sp.Array(eigV1)
                       eigV2r = sp.Array(eigV2)
                       sol1 = C1 * sp.exp(eigW1 * t) * eigV1r
                       sol2 = C2 * sp.exp(eigW1 * t) * (t * eigV1r + eigV2r)
            else:
                cpKind = 'whole plane'
                sol1 = None
                sol2 = None
        elif np.isclose(eigW1, 0):
            cpKind = ('stable' if eigW2 < 0 else 'unstable') + ' line'
        elif np.isclose(eigW2, 0):
            cpKind = ('stable' if eigW1 < 0 else 'unstable') + ' line'
        elif eigW1 > 0 and eigW2 > 0:
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

        if np.isclose(eigW1r, 0):
            cpKind = 'center'
        elif eigW1r > 0:
            cpKind = 'unstable focus'
        elif eigW1r < 0:
            cpKind = 'stable focus'

    if sol1 != None and sol2 != None:
        sym_sol = sol1 + sol2
    elif sol1 != None:
        sym_sol = sol1
    elif sol2 != None:
        sym_sol = sol2
    else:
        sym_sol = None
    return cpKind, sym_sol

def compute_generalized_eigenvectors(spA):
    P, blocks = spA.jordan_cells()
    basis = [P[:,i] for i in range(P.shape[1])]
    if (len(basis) == 2):
        return [basis[0][0], basis[0][1]], [basis[1][0], basis[1][1]]
    else:
        return None, None
