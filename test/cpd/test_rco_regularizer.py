"""RCO regularization term under-the-hood routine unit tests."""
import numpy as np
import pytest
from numpy.linalg import norm

import cpd
from util import create_random_spd, small_problem_data


class TestRcoRegularizer:
    @pytest.mark.parametrize("method", ("all", "diagonal"))
    def test_cholesky_regularizer(self, method):
        n = 5
        N = n * (n + 1) // 2
        _, l_vector, _ = create_random_spd(n)

        f = cpd._rco_regularizer.create_regularizer(method)(n)

        # Check that original function value = explicit impl function value (in two different ways).
        l_diag = np.diag(cpd.linalg.ravel_lower(l_vector, symmetric=False))
        assert f.fun(l_vector) == pytest.approx(
            - sum(np.log(l_diag ** 2)) + np.log(sum((l_diag if method == "diagonal" else l_vector) ** 2)))

        # Check derivatives are consistent with FD.
        assert cpd.optim.grad_check(f.fun, f.grad, N, delta=1e-6) < 1e-8
        assert cpd.optim.grad_check(lambda x: f.grad(x).copy(), lambda x: f.hessian(x).copy(), N, delta=1e-6) < 1e-8

        # Hessian action check.
        x = np.random.random((N,))
        y = np.random.random((N,))
        hxy = f.hessian(x) @ y
        assert np.allclose(f.hessian_action(x, y), hxy, 1e-15)

        # Test optimized integrated evaluation of f, grad, hessian action.
        y = np.random.random((N,))
        f_x, grad_x, hessian_x_y = f.tri_function(l_vector, y)
        assert f_x == pytest.approx(f.fun(l_vector))
        assert grad_x == pytest.approx(f.grad(l_vector))
        assert hessian_x_y == pytest.approx(np.array((f.hessian(l_vector) @ y)).flatten())

    def test_regularized_functional(self):
        c, w, p = small_problem_data(weighted=True)
        n = p.shape[1]
        N = n * (n + 1) // 2
        a = cpd.moment_functional.matrix_function_to_matrix_sum(p, p)
        _, l_vector, _ = create_random_spd(n)

        alpha = 0.1
        objective = cpd._rco.MomentCholeskyFunctionalExplicit(a, c, w=w)
        reg = cpd._rco_regularizer.create_regularizer("diagonal")(n)
        f = objective + alpha * reg

        l = cpd.linalg.ravel_lower(l_vector, symmetric=False)
        l_diag = np.diag(cpd.linalg.ravel_lower(l_vector, symmetric=False))
        objective_term = norm(w * (cpd.moment_functional.f(l @ l.T, p) - c)) / norm(w * c)
        reg_term = - sum(np.log(l_diag ** 2)) + np.log(sum(l_diag ** 2))
        assert f.fun(l_vector) == pytest.approx(objective_term ** 2 + alpha * reg_term)
        assert cpd.optim.grad_check(f.fun, f.grad, N, delta=1e-6) < 1e-8
        assert cpd.optim.grad_check(lambda x: f.grad(x).copy(), lambda x: f.hessian(x).copy(), N, delta=1e-6) < 1e-8

        # Hessian action check.
        x = np.random.random((N,))
        y = np.random.random((N,))
        hxy = f.hessian(x) @ y
        assert np.allclose(f.hessian_action(x, y), hxy, 1e-15)

        # Test optimized integrated evaluation of f, grad, hessian action.
        y = np.random.random((N,))
        f_x, grad_x, hessian_x_y = f.tri_function(l_vector, y)
        assert f_x == pytest.approx(f.fun(l_vector))
        assert grad_x == pytest.approx(f.grad(l_vector))
        assert hessian_x_y == pytest.approx(np.array((f.hessian(l_vector) @ y)).flatten())
