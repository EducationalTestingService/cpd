"""Basic optimization functions."""
from typing import Callable, Optional

import numpy as np
import scipy.optimize
import scipy.sparse
# from kneed import KneeLocator
from numpy.linalg import norm

import cpd._functional
import cpd.linalg


class OptimizeResult(dict):
    """Represents the optimization result.

    Attributes
    ----------
    x : ndarray
        The solution of the optimization.
    info:
        An array containing the optimal point (alpha, residual(x), solution_metric(x))
    curve:
        An ROC curve (alpha, residual(x(alpha)), solution_metric(x(alpha))).
    nfev, njev, nhev : int
        Number of evaluations of the objective functions and of its
        Jacobian and Hessian.

    Notes
    -----
    There may be additional attributes not listed above depending of the
    specific solver. Since this class is essentially a subclass of dict
    with attribute accessors, one can see which attributes are available
    using the `keys()` method.
    """

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        if self.keys():
            m = max(map(len, list(self.keys()))) + 1
            return '\n'.join([k.rjust(m) + ': ' + repr(v)
                              for k, v in sorted(self.items())])
        else:
            return self.__class__.__name__ + "()"

    def __dir__(self):
        return list(self.keys())


def grad_check(fun, grad, n, delta: float = 1e-7, x=None):
    np.random.seed(0)
    x = np.random.random((n,)) if x is None else x
    # Second-order finite difference.
    grad_fd = np.array([(fun(x + delta * cpd.linalg.unit_vector(n, i)) -
                         fun(x - delta * cpd.linalg.unit_vector(n, i))) / (2 * delta)
                        for i in range(n)])
    g = grad(x)
    if scipy.sparse.issparse(g):
        g = g.todense()
    return norm(grad_fd - g) / max(1e-15, norm(g))


def minimize_newton(f, x0, jac, hess, tol: float = 1e-6, max_iter: int = 100, args=()) -> OptimizeResult:
    """
    Minimizes a function f using Newton's method.
    Args:
        f:
        x0:
        jac:
        hess:
        tol:
        max_iter:
        args:

    Returns:

    """
    x = x0
    iter = 0
    success = False
    delta = 1
    while iter < max_iter:
        delta_prev = delta
        delta = jac(x, *args) / hess(x0, *args)
        x -= delta
        #print(x, delta, np.abs(delta) / np.abs(delta_prev), np.abs(jac(x, *args)))
        iter += 1
        if np.abs(delta) < tol * np.clip(np.abs(x), 1e-5, None):
            success = True
            break
    return OptimizeResult(x=x, fun=f(x, *args), nfev=1, njev=iter, nhev=iter, iter=iter, success=success)
