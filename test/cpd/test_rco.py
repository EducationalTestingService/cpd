"""RCO under-the-hood routine unit tests."""
import collections
import os

import numpy as np
import pytest
import scipy.optimize
import scipy.sparse
from numpy.linalg import norm

import cpd
from util import create_random_spd, small_problem_data, EBLP_SMALL_DATA_DIR


class TestRco:
    def test_unravel_matrix_functional_parts(self):
        """Tests unravelling f(G) into H*g. Diagonal, triangular parts separately checked."""
        m, n = 12, 11
        p = np.random.random((m, n))
        q = np.random.random((m, n))
        m, n = p.shape

        # 1D list of indices in the lower-triangular+diagonal part of an n x n matrix.
        # Divide this column index list into the diagonal part (d and dd below) and lower triangular part (k, l).
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

        # Sanity check of diagonal element part of matrix.
        diagonal_part = np.zeros((m * (m + 1) // 2, n))
        for I in range(m * (m + 1) // 2):
            for d in range(n):
                diagonal_part[I, d] = p[i[I], d] * q[j[I], d]
        assert np.array_equal(diagonal_part, p[ii, dd] * q[jj, dd])

        # Sanity check of diagonal element part of matrix.
        triangular_part = np.zeros((m * (m + 1) // 2, n * (n - 1) // 2))
        for I in range(m * (m + 1) // 2):
            for D in range(n * (n - 1) // 2):
                triangular_part[I, D] = p[i[I], k[D]] * q[j[I], l[D]] + p[i[I], l[D]] * q[j[I], k[D]]
        assert np.array_equal(triangular_part, p[ii2, kk2] * q[jj2, ll2] + p[ii2, ll2] * q[jj2, kk2])

    def test_unraveled_problem_matches_original(self):
        """Tests that f(G) = H*g (full sanity check of flattened (matrix * vector) vs. original matrix
        function. Data (G^*, P, C) loaded from files."""
        # Load original problem from files in matrix form.
        a_file_name = os.path.join(EBLP_SMALL_DATA_DIR, "A.txt")
        c_file_name = os.path.join(EBLP_SMALL_DATA_DIR, "C.txt")
        w_file_name = os.path.join(EBLP_SMALL_DATA_DIR, "Weights_BxB.txt")
        pi_file_name = os.path.join(EBLP_SMALL_DATA_DIR, "Pi_s.txt")
        _, c, _, _, p = cpd.data.load_data(a_file_name, c_file_name, w_file_name, pi_file_name)

        # Convert to unraveled form.
        H = cpd.moment_functional.matrix_function_to_matrix_sum(p, p)
        c_vector = cpd.linalg.unravel_lower(c)

        # Generate an SPD random solution matrix G.
        n = p.shape[1]
        g = cpd.linalg.random_spsd(n, 2)
        g_vector = cpd.linalg.unravel_lower(g)

        # Check that f(G) = H*g.
        f_unraveled = cpd.linalg.unravel_lower(cpd.moment_functional.f(g, p))
        n = len(c_vector)
        assert norm(f_unraveled.flatten() - (H @ g_vector).flatten()) < 1e-13 * n

    def test_cholesky_factor_functional(self):
        """Sanity check of creating explicit equations for the flattened Cholesky factor L."""
        m, n = 12, 11

        M, N = m * (m + 1) // 2, n * (n + 1) // 2
        a = np.random.random((M, N))

        k, l = cpd.linalg.lower_subscripts(n)
        ind = -np.ones((n, n), dtype=int)
        ind[k, l] = np.arange(n * (n + 1) // 2)

        # Each (r, s) = l[r] * l[s] combination appears in exactly one g[j] entry where l = flattened L and
        # g = flattened G. Not all (r, s) pairs are present (O(n^3) pairs for an G: n x n).
        pair_counter = collections.Counter(
            (r, s) for j in range(N) for r, s in zip(ind[k[j]], ind[l[j]]) if r >= 0 and s >= 0)
        assert all(v == 1 for v in pair_counter.values())

        w2 = np.zeros((M, N, N))
        for j in range(N):
            for (r, s) in ((r, s) for r, s in zip(ind[k[j]], ind[l[j]]) if r >= 0 and s >= 0):
                for i, hij in enumerate(a[:, j]):
                    w2[i, max(r, s), min(r, s)] += hij

        element_of_pair = dict(((r, s), j) for j in range(N) for r, s in zip(ind[k[j]], ind[l[j]]) if r >= 0 and s >= 0)
        w1 = np.zeros((M, N, N))
        for (r, s), j in element_of_pair.items():
            w1[:, r, s] = a[:, j]

        w = cpd._rco._cholesky_factor_functional(a)

        assert norm(w1 - w2) == pytest.approx(0)
        assert max(norm(w[i] - w1[i]) for i in range(M)) < 1e-15

    def test_cholesky_factor_functional_near_pd(self):
        """Sanity check of creating explicit equations for the flattened Cholesky factor L
        for the case of a single P, Q = I, I (near PD use case)."""
        n = 5
        p = cpd.linalg.ArrayList([np.eye(n)])
        a = cpd.moment_functional.matrix_function_to_matrix_sum(p, p)
        M, N = a.shape

        k, l = cpd.linalg.lower_subscripts(n)
        ind = -np.ones((n, n), dtype=int)
        ind[k, l] = np.arange(len(k))

        # Each (r, s) = l[r] * l[s] combination appears in exactly one g[j] entry where l = flattened L and
        # g = flattened G. Not all (r, s) pairs are present (O(n^3) pairs for an G: n x n).
        pair_counter = collections.Counter(
            (r, s) for j in range(N) for r, s in zip(ind[k[j]], ind[l[j]]) if r >= 0 and s >= 0)
        assert all(v == 1 for v in pair_counter.values())

        w2 = np.zeros((M, N, N))
        for j in range(N):
            for (r, s) in ((r, s) for r, s in zip(ind[k[j]], ind[l[j]]) if r >= 0 and s >= 0):
                for i, hij in enumerate(a[:, j]):
                    w2[i, max(r, s), min(r, s)] += hij

        element_of_pair = dict(
            ((r, s), j) for j in range(N) for r, s in zip(ind[k[j]], ind[l[j]]) if r >= 0 and s >= 0)
        w1 = np.zeros((M, N, N))
        for (r, s), j in element_of_pair.items():
            w1[:, r, s] = a[:, j]

        w = cpd._rco._cholesky_factor_functional(a)

        assert norm(w1 - w2) == pytest.approx(0)
        assert max(norm(w[i] - w1[i]) for i in range(M)) < 1e-15

    def test_create_llt_mapping(self):
        n = 5
        l = np.tril(np.random.random((n, n)))
        g = l @ l.T

        l_vector = cpd.linalg.unravel_lower(l)
        g_vector = cpd.linalg.unravel_lower(g)

        index = cpd._rco.create_llt_mapping(n)
        g_llt = np.array([np.sum(l_vector[k0] * l_vector[k1]) for k0, k1 in index])

        assert np.allclose(g_vector, g_llt, 1e-15)

    def test_moment_cholesky_functional_raveled_equals_original(self):
        n = 5
        g, l_vector, f_original = create_random_spd(n)

        # Check that raveled functional = Frobenius norm relative error.
        a = scipy.sparse.csr_matrix(cpd.moment_functional.frobenius_error_to_matrix(n))
        f = cpd._rco.MomentCholeskyFunctionalRaveled(a, g)

        assert f.fun(l_vector) == pytest.approx(f_original)
        # Check that raveled functional(a) = 0.
        assert f.fun(cpd.linalg.unravel_lower(scipy.linalg.cholesky(g, lower=True))) == pytest.approx(0)

    def test_moment_cholesky_functional_qf_equal_original(self):
        n = 5
        g, l_vector, f_original = create_random_spd(n)

        # Check that unraveled functional = Frobenius norm relative error.
        a = scipy.sparse.csr_matrix(cpd.moment_functional.frobenius_error_to_matrix(n))
        f = cpd._rco.MomentCholeskyFunctionalQf(a, g)

        assert f.fun(l_vector) ** 0.5 == pytest.approx(f_original)
        # Check that fun_objective(a) = 0 since a is SPD here.
        assert f.fun(cpd.linalg.unravel_lower(scipy.linalg.cholesky(g, lower=True))) == pytest.approx(0)

    @pytest.mark.parametrize("weighted", (False, True))
    def test_moment_cholesky_functional_explicit(self, weighted):
        c, w, p = small_problem_data(weighted=weighted)
        n = p.shape[1]
        N = n * (n + 1) // 2
        a = cpd.moment_functional.matrix_function_to_matrix_sum(p, p)
        _, l_vector, _ = create_random_spd(n)

        f = cpd._rco.MomentCholeskyFunctionalExplicit(a, c, w=w)

        l = cpd.linalg.ravel_lower(l_vector, symmetric=False)
        assert f.fun(l_vector) ** 0.5 == pytest.approx(
            norm(w * (cpd.moment_functional.f(l @ l.T, p) - c)) / norm(w * c))
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

    def test_moment_cholesky_functional_explicit_frobenius(self):
        n = 5
        N = n * (n + 1) // 2
        g, l_vector, f_original = create_random_spd(n)

        f = cpd._rco.MomentCholeskyFunctionalExplicit(None, g)

        # Check that fun_objective(a) = 0 since a is SPD here.
        assert f.fun(cpd.linalg.unravel_lower(scipy.linalg.cholesky(g, lower=True))) == pytest.approx(0)
        # Check that original function value = explicit impl function value (in two different ways).
        assert f.fun(l_vector) ** 0.5 == pytest.approx(f_original)
        l = cpd.linalg.ravel_lower(l_vector, symmetric=False)
        assert f.fun(l_vector) ** 0.5 == pytest.approx(norm(l @ l.T - g) / norm(g))
        # Check derivatives are consistent with FD.
        assert cpd.optim.grad_check(f.fun, f.grad, N, delta=1e-6) < 1e-8
        assert cpd.optim.grad_check(lambda x: f.grad(x).copy(), lambda x: f.hessian(x).copy(), N, delta=1e-6) < 1e-8

    def test_llt_functional(self):
        n = 5
        N = n * (n + 1) // 2

        h = cpd._rco.LltFunctional(n)

        # Gradient check.
        assert cpd.optim.grad_check(h.fun, h.grad, N, delta=1e-8) < 1e-8

        # Hessian check. Hessian is a 3D tensor in this case so can't use grad_check. Check each hessian h_r separately.
        x = np.random.random((N,))
        hx = h.hessian(x)
        for r in range(N):
            # Gradient of hr.
            def fun(x):
                return np.array(h.grad(x).copy()[:, r].todense()).flatten()

            # Hessian is constant in x.
            def grad(x):
                return hx[r * N:(r + 1) * N].todense()

            assert cpd.optim.grad_check(fun, grad, N, delta=1e-7) < 1e-8

        # Hessian action check.
        y = np.random.random((N,))
        hxy = (hx @ y).reshape(N, N)
        assert np.allclose(np.vstack(h.hessian_action(x, y)), hxy, 1e-15)

    def test_fmin_newton_near_pd(self):
        n = 10
        g, l_vector, f_original = create_random_spd(n, noise=1e-1)
        a = scipy.sparse.csr_matrix(cpd.moment_functional.frobenius_error_to_matrix(n))
        f = cpd._rco.MomentCholeskyFunctionalExplicit(a, g)

        result = scipy.optimize.minimize(f.fun, l_vector, method="Newton-CG", jac=f.grad,
                                         hessp=f.hessian_action, options={"xtol": 1e-3, "disp": False, "maxiter": 20})

        assert result.fun == pytest.approx(7.73e-7, 1e-2)
        assert result.success
        assert result.nit == 15
        assert result.nfev == 16
        assert result.njev == 16
        assert result.nhev == 197

    def test_fmin_newton_at_alpha(self):
        np.random.seed(0)
        c, w, p = small_problem_data(weighted=True)
        n = p.shape[1]
        a = cpd.moment_functional.matrix_function_to_matrix_sum(p, p)
        _, l_vector, _ = create_random_spd(n, noise=1e-13)

        alpha = 0.1
        objective = cpd._rco.MomentCholeskyFunctionalExplicit(a, c, w=w)
        reg = cpd._rco_regularizer.create_regularizer("all")(n)
        f = objective + alpha * reg

        result = scipy.optimize.minimize(f.fun, l_vector, method="Newton-CG", jac=f.grad,
                                         hessp=f.hessian_action, options={"xtol": 1e-3, "disp": False, "maxiter": 20})

        assert result.fun == pytest.approx(0.269, 1e-2)
        assert result.success
        assert result.nit == 14
        assert result.nfev == 17
        assert result.njev == 17
        assert result.nhev == 39
