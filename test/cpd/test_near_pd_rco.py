"""Linear algebra function unit tests."""
import numpy as np
import pytest
from numpy.linalg import norm

import cpd


class TestNearPdRco:
    def test_near_pd_rco_is_spd(self):
        np.random.seed(0)
        for i in range(3):
            for j in range(2, 10):
                a = np.random.randn(j, j)
                b, result = cpd.near_pd.near_pd(a, "rco")
                assert cpd.linalg.is_spd(b)
                assert result.nfev <= 300

    def test_near_pd_rco_of_spd(self):
        # Generate an SPD matrix with condition number 'cond'.
        np.random.seed(0)
        dim = 10
        cond_values = 10 ** np.arange(5)
        r = np.zeros((len(cond_values), 2))
        # Assert that curve(alpha=0) = original matrix for small condition #. That doesn't mean we are
        # returning it from near_rco (when the condition # is large we might find a better tradeoff). Also, the
        # Cholesky factor might be numerically unstable then.
        for j, cond in enumerate(cond_values):
            a = np.random.randn(dim, dim)
            a = a @ a.T
            lam, q = np.linalg.eig(a)
            a = q.T @ np.diag(1 + (cond - 1) * np.linspace(0, 1, num=dim, endpoint=True)) @ q
            assert cpd.linalg.is_spd(a)
            a_near_rco, result = cpd.near_pd.near_pd(a, "rco")
            assert cpd.linalg.is_spd(a_near_rco)
            r[j] = [result.info[1], result.info[2]]
            # assert norm(result.curve[-1, 1]) < 1e-15 * norm(a)

    def test_regulaizations(self):
        # Generate an SPD matrix with condition number 'cond'.
        np.random.seed(0)
        dim = 10
        num_experiments = 5
        nhev1 = np.zeros((num_experiments,))
        nhev2 = np.zeros((num_experiments,))
        for i in range(num_experiments):
            a = np.random.randn(dim, dim)
            _, result1 = cpd.near_pd.near_pd(a, "rco", regularization="all")
            _, result2 = cpd.near_pd.near_pd(a, "rco", regularization="all_scaled")
            nhev1[i] = result1.nhev
            nhev2[i] = result2.nhev
        # Both methods have the same number of function evaluations, more or les.
        assert np.mean(nhev1) == pytest.approx(115, 1e-1)
        assert np.mean(nhev2) == pytest.approx(136, 1e-1)
