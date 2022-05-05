"""Optimization and ROC curve generation for a regularized minimization problem."""
from typing import Callable, Optional

import numpy as np
import scipy.optimize
import scipy.sparse
# from kneed import KneeLocator
from numpy.linalg import norm

import cpd._functional
import cpd.linalg


def optimize_cond(fun_objective: Callable[[np.ndarray], float],
                  optimizer: Callable[[float, np.ndarray, float], np.ndarray],
                  g_init: np.ndarray, leeway_factor: float = 1.1,
                  num_alpha: int = 5, alpha_init: float = 0.01, alpha_step: float = 0.01,
                  tol: float = 1e-4, tol_curve: float = 1e-3, method: str = "Newton-CG"):
    """
    Returns an optimal constrained SPD matrix using a near PD algorithm. Specifically, given the initial matrix 'g_init' (not PD)
    and an objective function 'fun_objective', finds the ROC curve of cond(g) vs. fun_objective(g) vs. the eig_tol
    threshold parameter, and returns the point of best tradeoff between accuracy in fun_objective and condition number.

    Args:
        fun_objective: objective function.
        optimizer: a functor(alpha, x0, tol); if supplied, used to solve the ROC problem at a fixed alpha to tolerance
            tol. If None, uses BFGS minimization to minimize fun_f(x) + alpha * fun_reg(x), starting from x0.
            Returns x = minimizer.
        g_init: input square matrix to perturb (which should be near positive definite).
        leeway_factor: maximum allowed ratio of f(x)/min_alpha f(x, alpha) in the produced solution x. That is,
            if leeway_factor=1.1, we allow 10% increase in f value to minimize solution_metric(x).
        tol: gradient tolerance in the final optimization step (should typically be smaller than tol_curve).
        tol_curve: gradient tolerance in producing ROC solutions.
        num_alpha: number of ROC curve points to generate.
        alpha_init: Initial alpha is alpha-init * |f(x0)|/|reg_term(x0)| where x0=initial guess for continuation.
        alpha_step: alpha decrease factor along the ROC continuation curve.
        method: minimization method. "Newton-CG" or "BFGS".

    Returns:
        x: final optimized solution.
        result: a struct containing
            info: struct with info on x (alpha, relative moments equation error value, solution metric).
            curve: ROC curve (#alpha x 3: alpha, relative moments equation error value, solution metric).
            index: optimal index in the curve array corresponding to the final optimum.
    """

    def near_pd_optimizer(alpha, x0, tol):
        x = optimizer(x0, alpha, tol)
        return cpd.optim.OptimizeResult(x=x, nfev=0, njev=0, nhev=0)

    result = cpd.optim_continuation.optimize(cpd._functional.as_functional(fun_objective), None, np.linalg.cond, g_init,
                                             leeway_factor=leeway_factor, num_alpha=num_alpha, alpha_init=alpha_init,
                                             alpha_step=alpha_step, optimizer=near_pd_optimizer, continuation=False,
                                             tol=tol, tol_curve=tol_curve, method=method)
    return result.x, result


def optimize(fun: cpd._functional.Functional,
             fun_reg: Optional[cpd._functional.Functional],
             solution_metric: Callable[[np.ndarray], float],
             x: np.ndarray,
             leeway_factor: float = 1.1,
             tol: float = 1e-4,
             tol_curve: float = 1e-3,
             num_alpha: int = 5,
             alpha_init: float = 100,
             alpha_step: float = 0.01,
             start_from_small_alpha: bool = False,
             optimizer: Callable[[float, np.ndarray, float], np.ndarray] = None,
             continuation: bool = True,
             residual_fun: Callable[[np.ndarray], np.ndarray] = None,
             method: str = "Newton-CG",
             adaptive: bool = False) -> cpd.optim.OptimizeResult:
    """
    Performs a continuation to find the entire ROC curve of min fun_f(x) + alpha * fun_reg(x) vs. alpha.

    Args:
        fun: objective functional to minimize.
        fun_reg: Optional regularization term. If None, 'optimizer' must be specified.
        solution_metric: solution metric (e.g., condition number of the matrix x represents; |x|).
        x: initial guess for continuation.
        leeway_factor: maximum allowed ratio of f(x)/min_alpha f(x, alpha) in the produced solution x. That is,
            if leeway_factor=1.1, we allow 10% increase in f value to minimize solution_metric(x).
        tol: gradient tolerance in the final optimization step (should typically be smaller than tol_curve).
        tol_curve: gradient tolerance in producing ROC solutions.
        num_alpha: number of ROC curve points to generate.
        alpha_init: Initial alpha value for continuation.
        alpha_step: alpha decrease factor along the ROC continuation curve.
        start_from_small_alpha: if True, starts from alpha=0 and climbs up to alpha_init. If False, starts
            from alpha_init and decreases it 'num_alpha' times.
        optimizer: a functor(alpha, x0, tol); if supplied, used to solve the ROC problem at a fixed alpha to tolerance
            tol. If None, uses BFGS minimization to minimize fun_f(x) + alpha * fun_reg(x), starting from x0.
            Returns x = minimizer.
        continuation: if True, performs continuation along the ROC curve, starting from alpha_init and using the
            initial guess from the previous (larger) alpha of the current alpha problem. If False, uses 'x' as
            the initial guess for all alpha problems.
        residual_fun: optional function (residual) value to report; if Non, uses f(G)=||G-g||.
        method: minimization method. "Newton-CG" or "BFGS".

    Returns:
        x: final optimized solution.
        info: struct with info on x (alpha, fun value, solution metric).
        curve: ROC curve (#alpha x 3: alpha, fun value, solution metric).
        index: optimal index in the curve array corresponding to the final optimum.
    """
    residual = residual_fun if residual_fun is not None else fun.fun
    # Functional evaluation counters.
    if optimizer is None:
        def optimizer(alpha, x0, tol):
            f = fun + alpha * fun_reg
            if method == "Newton-CG":
                return scipy.optimize.minimize(
                    f.fun, x0, method=method, jac=f.grad, hessp=f.hessian_action,
                    options={"xtol": tol, "disp": False, "maxiter": 50})
            elif method == "BFGS":
                return scipy.optimize.minimize(
                    f.fun, x0, method=method, jac=f.grad, options={"gtol": tol, "disp": False})
            else:
                raise Exception("Unsupported optimization method {}".format(method))

    # Start with the initial guess and find a large enough regularization parameter value 'alpha'
    # such that the regularization dominates the minimization.
    x0 = x.copy()
    # Calculate the ROC curve by continuation. Use low accuracy threshold for speed since we're only roughly
    # optimizing alpha here.
    # TODO(orenlivne): adaptively find points on the alpha curve until we reach the optimal point (elbow/leeway_factor
    # threshold) instead of the fixed set of values in 'alpha_values'.
    alpha_values = alpha_init * alpha_step ** np.arange(num_alpha)
    if start_from_small_alpha:
        # Include 0 in the set of alphas + reverse values.
        alpha_values = np.concatenate(([0], alpha_values[-1::-1]))
        #alpha_values = alpha_values[-1::-1]
    roc = [None] * len(alpha_values)
    n_eval = [0, 0, 0]

    def do_step(x, alpha):
        result = optimizer(alpha, x, tol=tol_curve)
        x = result.x
        n_eval[0] += result.nfev
        n_eval[1] += result.njev
        n_eval[2] += result.nhev
        return x

    if adaptive:
        alpha = 0
        x = do_step(x, alpha)
        roc0 = (x, (alpha, residual(x), solution_metric(x)))
        if not continuation:
            x = x0

        alpha = 1e-3
        alpha_values = []
        roc = []
        if not continuation:
            x = x0
        for i in range(100):
            x = do_step(x, alpha)
            alpha_values.append(alpha)
            roc.append((x, (alpha, residual(x), solution_metric(x))))
            metric = roc[-1][1][2]
            if metric < 10:
                break
            alpha *= 10
            alpha_values.append(alpha)
            if not continuation:
                x = x0
        # Do not add the 0 point.
        #roc.append(roc0)
    else:
        for i, alpha in enumerate(alpha_values):
            x = do_step(x, alpha)
            # Save the solution and solution info.
            roc[i] = (x, (alpha, residual(x), solution_metric(x)))
            alpha *= alpha_step
            # In continuation mode, each solution serves as the initial guess for the next ROC step, so do nothing here.
            if not continuation:
                x = x0

    # Find optimal tradeoff point on the ROC curve.
    roc_curve = np.array([row[1] for row in roc])
    roc_x = np.array([row[0] for row in roc])
    # Sort roc curve by descending alpha, since we assume this in 'curve_knee'.
    sort_index = np.argsort(-roc_curve[:, 0])
    roc_curve, roc_x = roc_curve[sort_index], roc_x[sort_index]

    index = _optimal_index_leeway_factor(roc_curve[:, 1], roc_curve[:, 2], leeway_factor)
    #    index = curve_knee(roc_curve[:, 1], np.log(roc_curve[:, 2]), leeway_factor)

    # Calculate the solution to higher accuracy 'tol' at the optimal point.
    alpha = roc_curve[index, 0]
    if np.abs(tol - tol_curve) < 1e-3 * tol:
        # Final tol = curve tol, reuse existing solution.
        x = roc_x[index]
    else:
        x = optimizer(alpha, roc_x[index], tol=tol).x
    return cpd.optim.OptimizeResult(x=x, info=(alpha, residual(x), solution_metric(x)), curve=roc_curve, index=index,
                          nfev=n_eval[0], njev=n_eval[1], nhev=n_eval[2])


def _optimal_index_leeway_factor(residual, cond, leeway_factor):
    """Returns the optimal index in an ROC curve (r, cond) by residual leeway factor.
    - If no index has a good accuracy, take the minimum alpha for best accuracy.
    - If there exist indices with good accuracy, take the maximum alpha among those for best stability.
    """
    index = np.where(residual < leeway_factor * min(residual))[0]
    if index.size == 0:
        index = np.argmin(residual)
    else:
        index = index[np.argmin(cond[index])]
    return index

# def curve_knee(x, y, leeway_factor):
#     """
#     Returns the index of the knee of the curve y(x). Assumes y(x) is a convex decreasing curve.
#     Args:
#         x: x values.
#         y: corresponding y values.
#         leeway_factor: maximum allowed ratio of x[index]/min(x).
#
#     Returns:
#         index of knee of the curve, i.e. the knee is at (x[index], y[index]).
#     """
#     warnings.filterwarnings('ignore', '.*No knee/elbow found.*', category=UserWarning)
#     # Fallback: pick optimal alpha such that the (unregularized) matrix functional (x) does not increase by more than
#     # leeway_factor. Use a more accurate tolerance for the final result.
#     min_x = min(x)
#     index_fallback = min(np.where(x < leeway_factor * min_x)[0])
#     knee_y = None
#     try:
#         kneedle = KneeLocator(x, y, S=1, curve="convex", direction="decreasing")
#         knee_y = kneedle.knee_y
#     except IndexError:
#         pass
#     if knee_y is not None:
#         index_knee = min(np.where(np.abs(y - knee_y) < 1e-15)[0])
#     else:
#         index_knee = 0
#     # If the knee is above the leeway factor, fall back.
#     return index_knee if x[index_knee] < leeway_factor * min_x else index_fallback
