"""Cholesky optimization regularization term (part of RCO)."""
from typing import ClassVar

import numpy as np

import cpd.optim_continuation
from cpd._functional import Functional


def create_regularizer(method: str) -> ClassVar[Functional]:
    """
    Creates a regularization functional.

    Args:
        method: regularization method ("diagonal" for diagonal elements, or "all" for all elements).
    Returns:
        regularization Functional class.
    """
    if method == "all":
        return _CholeskyRegularizerAll
    elif method == "all_scaled":
        return _CholeskyRegularizerAllScaled
    elif method == "diagonal":
        return _CholeskyRegularizerDiagonal
    else:
        raise Exception("Unsupported regularization method {}".format(method))


class _CholeskyRegularizerAll(Functional):
    """
    Calculates the Cholesky functional regularization term and its derivatives. This version includes
    all off-diagonals, so it repersents -log det(G) + log tr(G).
    """

    def __init__(self, n: int):
        """
        Creates a Cholesky functional regularization term (all elements) for an n x n matrix.
        Args:
            n: matrix dimension.
        """
        self._n = n
        self._diagonal_index = cpd.linalg.lower_subscripts_diagonal_index(n)
        self._diagonal_submatrix_subscript = np.meshgrid(self._diagonal_index, self._diagonal_index)

    def identity(self):
        """Returns a flattened identity matrix, for an initial guess for very strong regularization."""
        n = self._n
        x = np.zeros((n * (n + 1) // 2,))
        x[self._diagonal_index] = 1
        return x

    def fun(self, x: np.ndarray):
        xd = x[self._diagonal_index]
        return - sum(np.log(xd ** 2)) + np.log(sum(x ** 2))

    def grad(self, x: np.ndarray):
        xd = x[self._diagonal_index]
        g = np.zeros_like(x)
        g[self._diagonal_index] -= 2 * 1 / xd
        g += 2 * x / sum(x ** 2)
        return g

    def hessian(self, x: np.ndarray):
        N = len(x)
        subscript = self._diagonal_submatrix_subscript
        xd = x[self._diagonal_index]
        s = sum(x ** 2)
        g = np.zeros((N, N))
        g[subscript[0], subscript[1]] += 2 * np.diag(1 / xd ** 2)
        g += 2 * (np.eye(N) / s - 2 * np.outer(x, x) / s ** 2)
        return g

    def hessian_action(self, x: np.ndarray, y: np.ndarray):
        d = self._diagonal_index
        xd = x[d]
        s = sum(x ** 2)
        action = np.zeros((len(x),))
        action[d] += 2 * y[d] / (xd ** 2)
        action += (2 / s) * y - (2 / s) ** 2 * x * (x.T @ y)
        return action


class _CholeskyRegularizerAllScaled(Functional):
    """
    Calculates the Cholesky functional regularization term and its derivatives. This version includes
    all off-diagonals, so it repersents -log det(G) + log tr(G).
    """

    def __init__(self, n: int):
        """
        Creates a Cholesky functional regularization term (all elements + scaled det term) for an n x n matrix.
        Args:
            n: matrix dimension.
        """
        self._n = n
        self._diagonal_index = cpd.linalg.lower_subscripts_diagonal_index(n)
        self._diagonal_submatrix_subscript = np.meshgrid(self._diagonal_index, self._diagonal_index)

    def identity(self):
        """Returns a flattened identity matrix, for an initial guess for very strong regularization."""
        n = self._n
        x = np.zeros((n * (n + 1) // 2,))
        x[self._diagonal_index] = 1
        return x

    def fun(self, x: np.ndarray):
        n = self._n
        xd = x[self._diagonal_index]
        return - sum(np.log(xd ** 2)) / n + np.log(sum(x ** 2))

    def grad(self, x: np.ndarray):
        n = self._n
        xd = x[self._diagonal_index]
        g = np.zeros_like(x)
        g[self._diagonal_index] -= (2 / n) / xd
        g += 2 * x / sum(x ** 2)
        return g

    def hessian(self, x: np.ndarray):
        n = self._n
        N = len(x)
        subscript = self._diagonal_submatrix_subscript
        xd = x[self._diagonal_index]
        s = sum(x ** 2)
        g = np.zeros((N, N))
        g[subscript[0], subscript[1]] += (2 / n) * np.diag(1 / xd ** 2)
        g += 2 * (np.eye(N) / s - 2 * np.outer(x, x) / s ** 2)
        return g

    def hessian_action(self, x: np.ndarray, y: np.ndarray):
        d = self._diagonal_index
        xd = x[d]
        s = sum(x ** 2)
        action = np.zeros((len(x),))
        action[d] += 2 * y[d] / (xd ** 2)
        action += (2 / s) * y - (2 / s) ** 2 * x * (x.T @ y)
        return action


class _CholeskyRegularizerDiagonal(Functional):
    """
    Calculates the Cholesky functional regularization term and its derivatives. This version includes
    only diagonal elements.
    """

    def __init__(self, n: int):
        """
        Creates an Cholesky functional regularization term (diagonal elements only) for an n x n matrix.
        Args:
            n: matrix dimension.
        """
        self._n = n
        self._diagonal_index = cpd.linalg.lower_subscripts_diagonal_index(n)
        self._diagonal_submatrix_subscript = np.meshgrid(self._diagonal_index, self._diagonal_index)

    def identity(self):
        """Returns a flattened identity matrix, for an initial guess for very strong regularization."""
        n = self._n
        x = np.zeros((n * (n + 1) // 2,))
        x[self._diagonal_index] = 1
        return x

    def fun(self, x: np.ndarray):
        xd = x[self._diagonal_index]
        return - sum(np.log(xd ** 2)) + np.log(sum(xd ** 2))

    def grad(self, x: np.ndarray):
        xd = x[self._diagonal_index]
        g = np.zeros_like(x)
        g[self._diagonal_index] -= 2 * (1 / xd - xd / sum(xd ** 2))
        return g

    def hessian(self, x: np.ndarray):
        n = self._n
        xd = x[self._diagonal_index]
        g = np.zeros((len(x), len(x)))
        s = sum(xd ** 2)
        subscript = self._diagonal_submatrix_subscript
        g[subscript[0], subscript[1]] += 2 * (np.eye(n) / s - 2 * np.outer(xd, xd) / s ** 2 + np.diag(1 / xd ** 2))
        return g
