import numpy as np
import pytest
from numpy.linalg import norm

import cpd


class TestNearPd:
    @pytest.mark.parametrize("method", ("higham", "tn", "rco"))
    def test_near_pd_is_spd(self, method):
        np.random.seed(0)
        for i in range(3):
            for j in range(2, 10):
                a = np.random.randn(j, j)
                b, result = cpd.near_pd.near_pd(a, method)
                assert cpd.linalg.is_spd(b)

    def test_near_pd_vs_baseline(self):
        np.random.seed(1)
        dim = 10
        leeway_factor = 1.2
        a = np.random.randn(dim, dim)
        kappa = np.linalg.cond(a)
        np.set_printoptions(precision=8)
        a_baseline = cpd.near_pd.higham(a, eig_tol=1e-6)
        a_rco, info = cpd.near_pd.near_pd(a, "rco", leeway_factor=leeway_factor)
        a_tn, info_tn = cpd.near_pd.near_pd(a, "tn", leeway_factor=leeway_factor)

    def test_reproduce_spd_matrix(self):
        # Create a matrix with eigenvalues from 1..cond, then set min eigenvalue to lam0.
        np.random.seed(0)
        dim = 10
        cond = 1000
        lam0 = -1e-3
        a = np.random.randn(dim, dim)
        a = a @ a.T
        _, q = np.linalg.eig(a)
        lam = np.logspace(0, np.log10(cond), num=dim, base=10)
        lam[0] = lam0
        a = q.T @ np.diag(lam) @ q

        methods = ("higham", "rco", "tn")
        curves = [None] * len(methods)
        for i, method in enumerate(methods):
            curves[i] = cpd.near_pd.near_pd(a, method, num_alpha=10, alpha_step=0.1)[1].curve

    def test_reproduce_near_spd_matrix(self):
        # Create a matrix with eigenvalues from 1..cond, then set min eigenvalue to lam0.
        dim = 10
        np.random.seed(1)
        a = np.random.randn(dim, dim)
        a = a @ a.T
        _, q = np.linalg.eig(a)
        cond = 10
        lam0 = -1e-3
        lam = np.concatenate(([lam0], 10 ** (cond * np.random.random((dim - 1,)))))
        a = q.T @ np.diag(lam) @ q

        methods = ("higham", "rco", "tn")
        curves = [None] * len(methods)
        for i, method in enumerate(methods):
            curves[i] = cpd.near_pd.near_pd(a, method, num_alpha=10)[1].curve