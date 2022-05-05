"""Functions for estimating the covariance matrix G of the score growth EBLP model by Dan McCaffrey,
Katherine Castellano, JR."""
from typing import Callable, Optional

import numpy as np
from numpy.linalg import norm

import cpd.optim_continuation
from cpd.linalg import ArrayList


def create_optimizer(method: str, p: ArrayList, w: Optional[np.ndarray] = None):
    """
    Creates an EBLP covariance matrix optimizer for a given moments equation left-hand-side. This forms
    the function f(G) = sum_i p[i] @ G @ p[i]^T, where the moments equation is min |f(G) - C|/|C|, |.| = matrix
    Frobenius norm. This is an expensive overhead, so if you solve for fixed p and many c's, it saves times to
    reuse this object and call optimize() multiple times with different RHS's c, especially if p is a large list.

    Args:
        method: optimization method ("higham" for baseline, or "rco" for Cholesky)
        p: an ArrayList of m x n matrices P
        w: optional m x m least-squares SYMMETRIC weight matrix. If None, unit weights are used. Only the diagonal and
            lower-triangular part of w are used; it is assumed to be symmetrically continued to its upper half.
    Returns:
        Optimizer object with an optimize(c) method.
    """
    if method == "higham":
        def optimizer(x0, alpha, tol):
            return cpd._higham.near_pd(x0, eig_tol=alpha)

        return _EblpNearPdOptimizer(optimizer, p, w=w)
    elif method == "tn":
        return _EblpNearPdTnOptimizer(p, w=w)
    elif method == "rco":
        return _EblpRcoOptimizer(p, w=w)
    else:
        raise Exception("Unsupported method {}".format(method))


class _EblpOptimizerBase:
    """
    A base class for solving the EBLP problem by starting from the LS solution and applying a near PD algorithm to it
    with different thresholds.
    """

    def __init__(self, p: ArrayList, w: Optional[np.ndarray] = None):
        """
        Creates an EBLP covariance matrix optimizer for a given moments equation left-hand-side. This forms
        the function f(G) = sum_i p[i] @ G @ p[i]^T, where the moments equation is min |f(G) - C|/|C|, |.| = matrix
        Frobenius norm. This is an expensive overhead, so if you solve for fixed p and many c's, it saves times to
        reuse this object and call optimize() multiple times with different RHS's c, especially if p is a large list.

        Args:
            p: an ArrayList of m x n matrices P
            w: optional m x m least-squares weight matrix. If None, unit weights are used. Only the diagonal and
                lower-triangular part of w are used; it is assumed to be symmetrically continued to its upper half.
        """
        if w is None:
            w = np.ones((p.shape[0], p.shape[0]))
        self._p = p
        self._w = w
        self._w_unraveled = np.diag(cpd.linalg.unravel_lower(w))
        self._h = self._w_unraveled @ cpd.moment_functional.matrix_function_to_matrix_sum(p, p)

    def optimize(self, c: np.ndarray, leeway_factor: float = 1.1, tol: float = 1e-4, tol_curve: float = 1e-3,
                 num_alpha: int = 5, alpha_init: float = 0.01, alpha_step: float = 0.01):
        """
        Performs a continuation to find the entire ROC curve of min fun_f(x) + alpha * fun_reg(x) vs. alpha
        and returns the minimizer.

        Args:
            c: EBLP moments equation RHS matrix.
            leeway_factor: maximum allowed ratio of f(x)/min_alpha f(x, alpha) in the produced solution x. That is,
                if leeway_factor=1.1, we allow 10% increase in f value to minimize solution_metric(x).
            tol: gradient tolerance in the final optimization step (should typically be smaller than tol_curve).
            tol_curve: gradient tolerance in producing ROC solutions.
            num_alpha: number of ROC curve points to generate.
            alpha_init: Initial alpha is alpha-init * |f(x0)|/|reg_term(x0)| where x0=initial guess for continuation.
            alpha_step: alpha decrease factor along the ROC continuation curve.

        Returns:
            x: final optimized solution.
            info: struct with info on x (alpha, relative moments equation error value, solution metric).
            curve: ROC curve (#alpha x 3: alpha, relative moments equation error value, solution metric).
            index: optimal index in the curve array corresponding to the final optimum.
        """
        # Initial guess: LS solution of the EBLP moments equation. Not PD.
        c_vector = self._w_unraveled @ cpd.linalg.unravel_lower(c)
        g_ls_vector = np.linalg.lstsq(self._h, c_vector, rcond=None)[0]
        g_init = cpd.linalg.ravel_lower(g_ls_vector, symmetric=True)

        return self._optimize(
            c, g_init, leeway_factor, num_alpha, alpha_init, alpha_step, tol, tol_curve)

    def _optimize(self, c: np.ndarray, g_init: np.ndarray,
                  leeway_factor: float,
                  num_alpha: int,
                  alpha_init: float,
                  alpha_step: float,
                  tol: float,
                  tol_curve: float):
        """A hook for different near PD algorithms."""
        raise Exception("Must be implemented by sub-classes")


class _EblpNearPdOptimizer(_EblpOptimizerBase):
    """
    A base class for solving the EBLP problem by starting from the LS solution and applying a near PD algorithm to it
    with different thresholds.
    """

    def __init__(self, optimizer: Callable[[float, np.ndarray, float], np.ndarray],
                 p: ArrayList, w: Optional[np.ndarray] = None):
        super().__init__(p, w)
        self._optimizer = optimizer

    def _optimize(self, c: np.ndarray, g_init: np.ndarray,
                  leeway_factor: float,
                  num_alpha: int,
                  alpha_init: float,
                  alpha_step: float,
                  tol: float,
                  tol_curve: float):
        # Set up the minimization functions.
        scale = 1 / norm(self._w * c)

        def fun_objective(g): return scale * norm(self._w * (cpd.moment_functional.f(g, self._p) - c))

        return cpd.optim_continuation.optimize_cond(fun_objective, self._optimizer, g_init,
                                                    leeway_factor=leeway_factor, num_alpha=num_alpha,
                                                    alpha_init=alpha_init, alpha_step=alpha_step, tol=tol, tol_curve=tol_curve)


class _EblpNearPdTnOptimizer(_EblpOptimizerBase):
    """
    A base class for solving the EBLP problem by starting from the LS solution and applying a near PD algorithm to it
    with different thresholds.
    """

    def __init__(self, p: ArrayList, w: Optional[np.ndarray] = None):
        super().__init__(p, w)

        def optimizer(x0, alpha, tol): return cpd._tn.tn(x0, alpha)

        self._optimizer = optimizer

    def _optimize(self, c: np.ndarray, g_init: np.ndarray,
                  leeway_factor: float,
                  num_alpha: int,
                  alpha_init: float,  # Not used
                  alpha_step: float,  # Not used
                  tol: float,
                  tol_curve: float):
        # Set up the minimization functions.
        scale = 1 / norm(self._w * c)

        def fun_objective(g): return scale * norm(self._w * (cpd.moment_functional.f(g, self._p) - c))

        alpha_init = min(100 * np.linalg.cond(g_init), 1e15)
        alpha_step = (1.1 / alpha_init) ** (1 / num_alpha)
        return cpd.optim_continuation.optimize_cond(fun_objective, self._optimizer, g_init,
                                                    leeway_factor=leeway_factor, num_alpha=num_alpha,
                                                    alpha_init=alpha_init, alpha_step=alpha_step, tol=tol, tol_curve=tol_curve)


class _EblpRcoOptimizer:
    """
    Finds a well-conditioned positive definite matrix G^* under the EBLP moments equation constraint, by optimizing
    its Cholesky factor.
    """

    def __init__(self, p: ArrayList, w: Optional[np.ndarray] = None):
        """
        Creates an EBLP covariance matrix optimizer for a given moments equation left-hand-side. This forms
        the function f(G) = sum_i p[i] @ G @ p[i]^T, where the moments equation is min |f(G) - C|/|C|, |.| = matrix
        Frobenius norm. This is an expensive overhead, so if you solve for fixed p and many c's, it saves times to
        reuse this object and call optimize() multiple times with different RHS's c, especially if p is a large list.

        Args:
            p: an ArrayList of m x n matrices P
            w: optional m x m least-squares weight matrix. If None, unit weights are used. Only the diagonal and
                lower-triangular part of w are used; it is assumed to be symmetrically continued to its upper half.
        """
        # Encode a into a list of matrices q appearing in quadratic-form terms of the LS functional of the Cholesky
        # factor. Save the weight matrix d since we multiply the RHS by it in optimize().
        self._n = p.shape[1]
        self._w = w
        self._a = cpd.moment_functional.matrix_function_to_matrix_sum(p, p)

    def optimize(self, c: np.ndarray,
                 leeway_factor: float = 1.1, tol: float = 1e-4, tol_curve: float = 1e-3,
                 num_alpha: int = 5, alpha_init: float = 100, alpha_step: float = 0.01,
                 regularization: str = "all_scaled"):
        """
        Performs a continuation to find the entire ROC curve of min fun_f(x) + alpha * fun_reg(x) vs. alpha
        and returns the minimizer.

        Args:
            c: EBLP moments equation RHS matrix.
            leeway_factor: maximum allowed ratio of f(x)/min_alpha f(x, alpha) in the produced solution x. That is,
                if leeway_factor=1.1, we allow 10% increase in f value to minimize solution_metric(x).
            tol: gradient tolerance in the final optimization step (should typically be smaller than tol_curve).
            tol_curve: gradient tolerance in producing ROC solutions.
            num_alpha: number of ROC curve points to generate.
            alpha_init: Initial alpha is alpha-init * |f(x0)|/|reg_term(x0)| where x0=initial guess for continuation.
            alpha_step: alpha decrease factor along the ROC continuation curve.
            regularization: regularization method ("diagonal" for diagonal elements, or "all" for all elements).

        Returns:
            x: final optimized solution.
            info: struct with info on x (alpha, relative moments equation error value, solution metric).
            curve: ROC curve (#alpha x 3: alpha, relative moments equation error value, solution metric).
            index: optimal index in the curve array corresponding to the final optimum.
        """
        # Set up the minimization functions.
        objective = cpd._rco.MomentCholeskyFunctionalExplicit(self._a, c, w=self._w)
        reg = cpd._rco_regularizer.create_regularizer(regularization)(self._n)
        # Start with an identity initial guess and find a large enough regularization parameter value 'alpha'
        # such that the regularization dominates the minimization.
        x = reg.identity()
        alpha_init_scaled = alpha_init * np.abs(objective.fun(x)) / np.abs(reg.fun(x))
        # Perform continuation in alpha.
        result = cpd.optim_continuation.optimize(objective, reg, cpd.moment_functional.solution_metric, x,
                                                 leeway_factor=leeway_factor, num_alpha=num_alpha, alpha_init=alpha_init_scaled,
                                                 alpha_step=alpha_step, tol=tol, tol_curve=tol_curve)

        # Translate function into relative error, not relative error squared.
        result.curve[:, 1] **= 0.5
        info = result.info
        result.info = (info[0], info[1] ** 0.5, info[2])
        # Translate x back into L and into G.
        l = cpd.linalg.ravel_lower(result.x)
        g = l @ l.T
        return g, result
