"""TN near-CPD Algorithm function unit tests."""
import numpy as np
import sortednp as snp
import pytest
import scipy.linalg
from numpy.testing import assert_array_almost_equal

import cpd


class TestTn:
    def test_zero_mu_example(self):
        """Here mu=0 in one of the intervals here (l=-1, u=n). Should not encounter a division by zero."""
        a = np.array([[1.76405235, 0.40015721], [0.97873798, 2.2408932]])
        b = cpd.near_pd.tn(a, kappa=10)
        assert_array_almost_equal(b, 0.5 * (a + a.T), decimal=8)

    def test_bad_case_4x4(self):
        a = np.array([[2.94264391, 0.21468729, 1.42300408, 1.60615651],
                      [0.21468729, -0.62395222, -1.63257706, -0.52571663],
                      [1.42300408, -1.63257706, -1.06321182, 3.1252039],
                      [1.60615651, -0.52571663, 3.1252039, -0.14933596],
                      ])
        kappa = 1.1
        b = cpd.near_pd.tn(a, kappa)
        assert np.linalg.cond(b) <= kappa + 1e-8
        distance = lambda x: np.linalg.norm(a - x, ord="fro")
        assert is_local_minimum(distance, b, kappa, num=1000)

    def test_all_negative_eigenvalues_zero_result(self):
        a = np.array([[-0.62715497, -0.91365594], [-0.91365594, -1.3437533]])
        b = cpd.near_pd.tn(a, kappa=1.5)
        assert np.linalg.norm(b) == 0

    def test_small_matrix_distinct_eigenvalues(self):
        assert_correct_cpd_solution(
            [
                [1, 3, 5, 7],
                [0, 1, 4, 5],
                [1, 8, 1, 1],
                [0, 0, 0, 3],
            ], 1.5,
            [[2.19836071, 0.13759577, 0.09493764, 0.39623029],
             [0.13759577, 2.34309451, 0.45188726, -0.00358883],
             [0.09493764, 0.45188726, 2.42908776, -0.09894921],
             [0.39623029, -0.00358883, -0.09894921, 2.62297492]]
            , "fro")

    def test_small_matrix_multiple_zero_eigenvalues(self):
        assert_correct_cpd_solution(
            np.arange(16).reshape(4, 4), 1.5,
            [[9.23417388, 0.47504999, 0.64052485, 0.80599972],
             [0.47504999, 9.65357365, 0.98289979, 1.23682469],
             [0.64052485, 0.98289979, 10.24987348, 1.66764966],
             [0.80599972, 1.23682469, 1.66764966, 11.02307339]], "fro")

    @pytest.mark.parametrize("kappa", (1.1, 10, 100, 1000))
    def test_near_pd_is_spd(self, kappa):
        np.random.seed(0)
        for i in range(5):
            for j in range(2, 20, 5):
                a = random_conditioned_spd_matrix(j, 10 ** 6)
                b = cpd.near_pd.tn(a, kappa)
                if np.linalg.norm(b) == 0:
                    continue
                assert np.linalg.cond(b) <= kappa * (1 + 1e-8)
                distance = lambda x: np.linalg.norm(a - x, ord="fro")
                assert is_local_minimum(distance, b, kappa) >= 0

    @pytest.mark.parametrize("kappa", (1.1, 10, 100, 1000))
    def test_frobenius_brent_result_equals_direct(self, kappa):
        np.random.seed(0)
        for j in range(2, 100, 10):
            a = np.random.randn(j, j)
            b = cpd.near_pd.tn(a, kappa, method="direct")
            b_min = cpd.near_pd.tn(a, kappa, method="brent")
            assert_array_almost_equal(b, b_min, decimal=7)

    @pytest.mark.parametrize("kappa", (1.1, 10, 100, 1000))
    def test_binary_result_equals_brent(self, kappa):
        for j in range(2, 100, 10):
            a = np.random.randn(j, j)
            b = cpd.near_pd.tn(a, kappa, method="brent")
            b_min = cpd.near_pd.tn(a, kappa, method="binary")
            assert_array_almost_equal(b, b_min, decimal=7)

    def test_brent_num_function_eval_is_bounded(self):
        """Tests that the number of function evaluations in minimization is bounded."""
        np.random.seed(0)
        num_experiments = 10
        nfev = []
        for kappa in (1.1, 10, 100, 1000):
            for j in (10, 50, 100):
                for _ in range(num_experiments):
                    a = np.random.randn(j, j)
                    _, result = cpd.near_pd.tn(a, kappa, method="brent", full_output=True)
                    if result is not None:
                        assert result.success
                        nfev.append(result.nfev)
        assert np.mean(nfev) == pytest.approx(25, 0.1)
        assert np.std(nfev) == pytest.approx(6.5, 0.1)
        assert np.max(nfev) <= 50

    def test_binary_num_function_eval_is_bounded(self):
        """Tests that the number of function evaluations in minimization is bounded."""
        np.random.seed(0)
        num_experiments = 10
        nfev = []
        for kappa in (1.1, 10, 100, 1000):
            for j in (10, 50, 100):
                for p in (2, ): #(1.5, 2, 2.5):
                    for _ in range(num_experiments):
                        a = np.random.randn(j, j)
                        _, result = cpd.near_pd.tn(a, kappa, method="binary", full_output=True, p=p)
                        if result is not None:
                            assert result.success
                            nfev.append(result.nfev)
        assert np.mean(nfev) == pytest.approx(1.9, 0.1)
        assert np.std(nfev) == pytest.approx(0.3, 0.1)
        assert np.max(nfev) == 2


def assert_correct_cpd_solution(a, kappa, b_expected, ord):
    """Asserts that the TN solution of the Lp norm CPD problem is a local minimum 7 has condition number <= kappa."""
    a = np.array(a)
    b = cpd.near_pd.tn(a, kappa, method="direct")
    # To print the result, use np.array2string(b, separator=",", precision=8))
    assert_array_almost_equal(b, b_expected, decimal=8)
    assert np.linalg.cond(b) <= kappa + 1e-8
    assert is_local_minimum(lambda x: np.linalg.norm(a - x, ord=ord), b, kappa)


def is_local_minimum(f, a, kappa: float, num: int = 100) -> bool:
    """Returns True iff the matrix a is a local minimum of the matrix functional f(a). Checks that
    f(b) > f(a), where b = SPD matrix with the same condition number as a. We generate 'num' b's with random
    eigenvalues such that cond(b) = kappa.
    """
    return min(f(random_conditioned_spd_matrix(a.shape[0], kappa)) for _ in range(num)) - f(a)


def random_conditioned_spd_matrix(n: int, kappa: float):
    a = np.random.random((n, n))
    a = a @ a.T
    q = scipy.linalg.eigh(a)[1]
    lam = np.sort(np.random.random((n,)))
    lam = np.linspace(1, (kappa * lam[0]) / lam[-1], num=n) * lam
    return q @ np.diag(lam) @ q.T
