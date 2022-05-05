"""Method of moment functional. Appears in the EBLP problem and generalizes the Frobenius norm (near PD) case."""

import numpy as np
from numpy.linalg import norm

import cpd.linalg
from cpd.linalg import ArrayList


def f(g: np.ndarray, p: ArrayList) -> np.ma:
    """
    Returns the value of the moment equating functional f(G) = sum_{ps in p} pi*G*pi^T.

    Args:
        g: n x n matrix G.
        p: list of matrices, each is m x n.

    Returns:
        f(G) = sum_{p_i in p} pi * G * pi^T.
    """
    return sum(ps @ g @ ps.T for ps in p)


def frobenius_error_to_matrix(n: int):
    """Sets up the the functional f(G) = I*G*G^T = G."""
    p = ArrayList([np.eye(n)])
    return matrix_function_to_matrix_sum(p, p)


def matrix_function_to_matrix_sum(p: ArrayList, q: ArrayList, lower: bool = False) -> np.ndarray:
    """
    Returns H in h(G) = sum_i (P_i * G * Q_i^T) as the function H*g, where g is the flattened lower-triangular part
    of G.

    Args:
        p: list of m x n left matrix.
        q: list of m x n right matrix.
        lower: whether G is lower triangular (if True), or symmetric (if False).

    Returns:
        H = unravelled matrix functional h(.) a matrix.
    """
    m, n = p.shape
    k, l = cpd.linalg.lower_subscripts(n)
    dd, ii, ii2, jj, jj2, kk2, ll2 = _matrix_function_to_matrix_indices(m, n)
    h = np.zeros((m * (m + 1) // 2, n * (n + 1) // 2))

    # Fill in columns corresponding to the diagonal part of G.
    diagonal_part = sum(pi[ii, dd] * qi[jj, dd] for pi, qi in zip(p, q))
    diagonal_index = cpd.linalg.lower_subscripts_diagonal_index(n)
    h[:, diagonal_index] = diagonal_part

    # Fill in columns corresponding to the lower triangular part of G.
    if lower:
        triangular_part = sum(pi[ii2, ll2] * qi[jj2, kk2] for pi, qi in zip(p, q))
    else:
        triangular_part = sum(pi[ii2, kk2] * qi[jj2, ll2] + pi[ii2, ll2] * qi[jj2, kk2] for pi, qi in zip(p, q))
    lower_index = np.where(k != l)[0]
    h[:, lower_index] = triangular_part

    return h


def _matrix_function_to_matrix_indices(m, n):
    # 1D list of indices in the lower-triangular+diagonal part of an n x n matrix.
    # Divide this column index lisst into the diagonal part (d and dd below) and lower triangular part (k, l).
    k, l = cpd.linalg.lower_subscripts(n)
    col_lower = np.where(k != l)[0]
    k, l = k[col_lower], l[col_lower]

    # 1D list of indices a lower-triangular+diagonal part m x m matrix.
    i, j = cpd.linalg.lower_subscripts(m)

    # For diagonal part of G.
    d = np.arange(n)
    ii, dd = [x.T for x in np.meshgrid(i, d)]
    jj = np.meshgrid(j, d)[0].T

    # For lower-triangular part of G.
    ii2, kk2 = [x.T for x in np.meshgrid(i, k)]
    jj2, ll2 = [x.T for x in np.meshgrid(j, l)]

    return dd, ii, ii2, jj, jj2, kk2, ll2


def solution_metric(x):
    """Returns a solution metric cond(L*L^T) where L = the raveled x."""
    l = cpd.linalg.ravel_lower(x)
    return np.linalg.cond(l.T @ l)
