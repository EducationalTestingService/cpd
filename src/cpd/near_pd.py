"""Solving the nearest symmetric positive definite matrix problem (near PD) using different algorithms."""

from typing import Callable, Tuple

import numpy as np

import cpd._higham
import cpd._rco
import cpd.optim_continuation


def near_pd(a: np.ndarray, method: str, leeway_factor: float = 1.1,
            num_alpha: int = 5, alpha_init: float = 0.01, alpha_step: float = 0.01,
            tol: float = 1e-4, tol_curve: float = 1e-3, regularization: str = "all_scaled",
            residual_fun: Callable[[np.ndarray], float] = None,
            start_from_small_alpha: bool = True) -> \
        Tuple[np.ndarray, cpd.optim.OptimizeResult]:
    """
    Returns the symmetric positive definite matrix nearest to a given matrix in the Frobenius norm.

    Args:
        a: input square matrix to perturb (which should be near positive definite).
        method: optimization method ("higham": Higham's Near PD; "rco" for regularized Cholesky optimization;
            "tn" for Tanaka-Nataka).
        leeway_factor: maximum allowed ratio of f(x)/min_alpha f(x, alpha) in the produced solution x. That is,
            if leeway_factor=1.1, we allow 10% increase in f value to minimize solution_metric(x).
        tol: gradient tolerance in the final optimization step (should typically be smaller than tol_curve).
        tol_curve: gradient tolerance in producing ROC solutions.
        num_alpha: number of ROC curve points to generate.
        alpha_init: Initial alpha is alpha-init * |f(x0)|/|reg_term(x0)| where x0=initial guess for continuation.
        alpha_step: alpha decrease factor along the ROC continuation curve.
        regularization: regularization method ("diagonal" for diagonal elements, or "all" for all elements).
            For method="rco" only.
        residual_fun: optional function (residual) value to report; if Non, uses f(G)=||G-g||^2_2.
        start_from_small_alpha: if True, starts from alpha=0 and climbs up to alpha_init. If False, starts
            from alpha_init and decreases it 'num_alpha' times.

    Returns:
        g: nearest SPD matrix to g_init.
        result: OptimizeResult containing
            info: struct with info on x (alpha, relative moments equation error value, solution metric).
            curve: ROC curve (#alpha x 3: alpha, relative moments equation error value, solution metric).
            index: optimal index in the curve array corresponding to the final optimum.
            nfev, njev, nhev # function evaluations.
    """
    if method == "higham":
        return _near_pd_optimize_higham(a, leeway_factor=leeway_factor, num_alpha=num_alpha,
                                        alpha_init=alpha_init, alpha_step=alpha_step, tol=tol, tol_curve=tol_curve)
    elif method == "tn":
        # alpha_init, alpha_step have no effect here and are internally set.
        return _near_pd_optimize_tn(a, leeway_factor=leeway_factor, num_alpha=num_alpha, tol=tol, tol_curve=tol_curve)
    elif method == "rco":
        # alpha_init has no effect here.
        return cpd._rco.near_pd(a, leeway_factor=leeway_factor, num_alpha=num_alpha,
                                alpha_init=10 if alpha_init is None else alpha_init,
                                alpha_step=0.1, tol=tol,
                                tol_curve=tol_curve, regularization=regularization, residual_fun=residual_fun,
                                start_from_small_alpha=True, continuation=True, adaptive=True)
    else:
        raise Exception("Unsupported method {}".format(method))


def higham(a, eig_tol: float = 1e-6):
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
    return cpd._higham.near_pd(a, eig_tol=eig_tol)


def tn(a, kappa: float, p: float = 2, full_output: bool = False, method: str = "direct"):
    """
    Returns the nearest SPD matrix to a given matrix in the Frobenius norm whose condition number <= kappa.

    Args:
        a: original square matrix. If non-symmetric, the algorithm is run on its symmetric part.
        kappa: condition number bound. Must be > 1. In most cases, the returned matrix will have a condition numbeerr
            exactly equal to kappa.
        p: exponent of Lp norm (default: p=2: Frobenius/2-norm).
        full_output: bool, optional
            If True, returns an optimization result struct with # function calls, etc.
        method: eigenvalue computation method.
            "direct": O(n^2) direct computation.
            "binary":O(n log n) binary search within endpoints for the unique minimum + newton in the two adjacent
                intervals.
            "brent": O(n) with a large constant.

    Returns:
        conditioned near-SPD matrix.

    See:
        Tanaka, M. and Nakata, K., Positive definite matrix approximation with condition number constraint.
        Optim Lett (2014) 8:939–947 DOI: 10.1007/s11590-013-0632-7
    """
    return cpd._tn.tn(a, kappa, p=p, full_output=full_output, method=method)


def _near_pd_optimize_higham(g_init: np.ndarray, leeway_factor: float = 1.1,
                             num_alpha: int = 5, alpha_init: float = 0.01, alpha_step: float = 0.01,
                             tol: float = 1e-4, tol_curve: float = 1e-3) -> Tuple[np.ndarray, cpd.optim.OptimizeResult]:
    """
    Returns an optimal SPD matrix using the near PD algorithm. Specifically, given the initial matrix 'g_init' (not
    PD), finds the ROC curve of cond(G) vs. Frobenius_norm(G-g) vs. the eig_tol threshold parameter, and returns
    the point of best tradeoff between accuracy in fun_objective and condition number.

    Args:
        g_init: input square matrix to perturb (which should be near positive definite).
        leeway_factor: maximum allowed ratio of f(x)/min_alpha f(x, alpha) in the produced solution x. That is,
            if leeway_factor=1.1, we allow 10% increase in f value to minimize solution_metric(x).
        tol: gradient tolerance in the final optimization step (should typically be smaller than tol_curve).
        tol_curve: gradient tolerance in producing ROC solutions.
        num_alpha: number of ROC curve points to generate.
        alpha_init: Initial alpha is alpha-init * |f(x0)|/|reg_term(x0)| where x0=initial guess for continuation.
        alpha_step: alpha decrease factor along the ROC continuation curve.

    Returns:
        x: final optimized solution.
        result: OptimizeResult containing
            info: struct with info on x (alpha, relative moments equation error value, solution metric).
            curve: ROC curve (#alpha x 3: alpha, relative moments equation error value, solution metric).
            index: optimal index in the curve array corresponding to the final optimum.
            nfev, njev, nhev # function evaluations.
    """

    def residual_fun(g):
        return np.linalg.norm(g - g_init) / np.linalg.norm(g_init)

    def optimizer(x0, alpha, tol):
        return cpd._higham.near_pd(x0, eig_tol=alpha)

    return cpd.optim_continuation.optimize_cond(residual_fun, optimizer, g_init,
                                                leeway_factor=leeway_factor, num_alpha=num_alpha,
                                                alpha_init=alpha_init, alpha_step=alpha_step, tol=tol, tol_curve=tol_curve)


def _near_pd_optimize_tn(g_init: np.ndarray, leeway_factor: float = 1.1,
                         num_alpha: int = 5, tol: float = 1e-4, tol_curve: float = 1e-3) \
        -> Tuple[np.ndarray, cpd.optim.OptimizeResult]:
    """
    Returns an optimal SPD matrix using the TN algorithm. Specifically, given the initial matrix 'g_init' (not
    PD), finds the ROC curve of cond(G) vs. Frobenius_norm(G-g) vs. the eig_tol threshold parameter, and returns
    the point of best tradeoff between accuracy in fun_objective and condition number.

    Args:
        g_init: input square matrix to perturb (which should be near positive definite).
        leeway_factor: maximum allowed ratio of f(x)/min_alpha f(x, alpha) in the produced solution x. That is,
            if leeway_factor=1.1, we allow 10% increase in f value to minimize solution_metric(x).
        tol: gradient tolerance in the final optimization step (should typically be smaller than tol_curve).
        tol_curve: gradient tolerance in producing ROC solutions.
        num_alpha: number of ROC curve points to generate.
        alpha_init: Initial alpha is alpha-init * |f(x0)|/|reg_term(x0)| where x0=initial guess for continuation.
        alpha_step: alpha decrease factor along the ROC continuation curve.

    Returns:
        x: final optimized solution.
        result: OptimizeResult containing
            info: struct with info on x (alpha, relative moments equation error value, solution metric).
            curve: ROC curve (#alpha x 3: alpha, relative moments equation error value, solution metric).
            index: optimal index in the curve array corresponding to the final optimum.
            nfev, njev, nhev # function evaluations.
    """

    def residual_fun(g):
        return np.linalg.norm(g - g_init) / np.linalg.norm(g_init)

    def optimizer(x0, alpha, tol):
        return tn(x0, alpha)

    alpha_init = min(100 * np.linalg.cond(g_init), 1e15)
    alpha_step = (1.1 / alpha_init) ** (1 / num_alpha)
    return cpd.optim_continuation.optimize_cond(residual_fun, optimizer, g_init,
                                                leeway_factor=leeway_factor, num_alpha=num_alpha,
                                                alpha_init=alpha_init, alpha_step=alpha_step, tol=tol, tol_curve=tol_curve)
