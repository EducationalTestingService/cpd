"""Solving the near-SPD problem using Higham's algorithm."""

import numpy as np
from numpy import linalg as la

import cpd.linalg
import cpd.optim_continuation


def near_pd(a, eig_tol: float = 1e-6):
    """Find the nearest symmetric positive-definite matrix to input.
    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6

    Code written by Ahmed Fasih and modified by Oren Livne to support 'eig_tol'. See
    https://stackoverflow.com/questions/43238173/python-convert-matrix-to-positive-semi-definite/43244194#43244194

    Args:
        a: input square matrix.
        eig_tol: defines relative positiveness of eigenvalues compared to largest one, lambda_1. Eigenvalues
        lambda_k are treated as if zero when lambda_k / lambda_1 le eig.tol).

    Returns:
        SPD matrix.
    """
    b = 0.5 * (a + a.T)
    _, s, v = la.svd(b)
    h = np.dot(v.T, np.dot(np.diag(s), v))
    a2 = 0.5 * (b + h)
    a3 = 0.5 * (a2 + a2.T)
    if cpd.linalg.is_spd(a3, eig_tol=eig_tol):
        return a3

    spacing = np.spacing(la.norm(a))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky decomposition will accept matrices
    # with exactly 0-eigenvalue, whereas Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing` will be much larger than [1]'s
    # `eps(mineig)`, since `mineig` is usually on the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34,
    # whereas `spacing` will, for Gaussian random matrices of small dimension, be on the order of 1e-16. In
    # practice, both ways converge, as the unit test suggests.
    identity = np.eye(a.shape[0])
    k = 1
    while not cpd.linalg.is_spd(a3, eig_tol=eig_tol):
        min_eigenvalue = np.min(np.real(la.eigvals(a3)))
        a3 += identity * (-min_eigenvalue * k ** 2 + spacing)
        k += 1
    return a3
