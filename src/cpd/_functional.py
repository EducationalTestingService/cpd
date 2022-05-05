"""Defines an interface of an objective function in an optimization context and function calculus."""
from typing import Callable

import numpy as np


class Functional:
    """Defines a generic functional interface."""

    def fun(self, x: np.ndarray):
        raise Exception("Must be implemented by sub-classes")

    def grad(self, x: np.ndarray):
        raise Exception("Must be implemented by sub-classes")

    def hessian(self, x: np.ndarray):
        raise Exception("Must be implemented by sub-classes")

    def hessian_action(self, x: np.ndarray, y: np.ndarray):
        """Returns the action of the Hessian at x on a vector y. Optimization for CG."""
        return self.hessian(x) @ y

    def tri_function(self, x: np.ndarray, y: np.ndarray):
        """Returns (f(x), grad(x), hessian_action(x, y)). A potentially optimized version of calling the three
        separately."""
        return self.fun(x), self.grad(x), self.hessian_action(x, y)

    def __add__(self, other):
        """Returns scale * functional."""
        return _SumFunctional(self, other)

    def __mul__(self, scale: float):
        """Returns functional * scale."""
        return _ScaledFunctional(self, scale)

    def __rmul__(self, scale: float):
        """Returns scale * functional."""
        return _ScaledFunctional(self, scale)


class _SumFunctional(Functional):
    def __init__(self, f: Functional, g: Functional):
        self._f = f
        self._g = g

    def fun(self, x: np.ndarray):
        return self._f.fun(x) + self._g.fun(x)

    def grad(self, x: np.ndarray):
        return self._f.grad(x) + self._g.grad(x)

    def hessian(self, x: np.ndarray):
        return self._f.hessian(x) + self._g.hessian(x)

    def hessian_action(self, x: np.ndarray, y: np.ndarray):
        return self._f.hessian_action(x, y) + self._g.hessian_action(x, y)

    def tri_function(self, x: np.ndarray, y: np.ndarray):
        return tuple(fy + gy for fy, gy in zip(self._f.tri_function(x, y), self._g.tri_function(x, y)))


class _ScaledFunctional(Functional):
    def __init__(self, f: Functional, scale: float):
        self._f = f
        self._scale = scale

    def fun(self, x: np.ndarray):
        return self._scale * self._f.fun(x)

    def grad(self, x: np.ndarray):
        return self._scale * self._f.grad(x)

    def hessian(self, x: np.ndarray):
        return self._scale * self._f.hessian(x)

    def hessian_action(self, x: np.ndarray, y: np.ndarray):
        return self._scale * self._f.hessian_action(x, y)

    def tri_function(self, x: np.ndarray, y: np.ndarray):
        return tuple(self._scale * z for z in self._f.tri_function(x, y))


class _FunctionalOfFunction(Functional):
    """A decorator that wraps a callable in a Functional."""

    def __init__(self, f: Callable[[np.ndarray], float]):
        self._f = f

    def fun(self, x: np.ndarray):
        return self._f(x)


def as_functional(f: Callable[[np.ndarray], float]):
    """A decorator that wraps a callable in a Functional."""
    return _FunctionalOfFunction(f)
