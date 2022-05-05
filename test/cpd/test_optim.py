"""Linear algebra function unit tests."""
import numpy as np
import pytest
from numpy.linalg import norm

import cpd


class TestOptim:
    def test_minimize_newton(self):
        def f(x):
            return x ** 2 + np.exp(-x)

        def f_deriv1(x):
            return 2 * x - np.exp(-x)

        def f_deriv2(x):
            return 2 + np.exp(-x)

        result = cpd.optim.minimize_newton(f, 1.3, f_deriv1, f_deriv2, tol=1e-10)

        assert result.success
        assert result.x == pytest.approx(0.35173371, 1e-4)
        assert result.iter == 16