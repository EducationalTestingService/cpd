"""Linear algebra function unit tests."""
import numpy as np
import pytest
from numpy.linalg import norm

import cpd


class TestLinalg:
    def test_array_list(self):
        m, n, p, q, num_arrays = 10, 5, 8, 7, 11
        b = np.random.random((n, p))
        c = np.random.random((q, m))

        a = cpd.linalg.ArrayList(np.random.random((m, n)) for _ in range(num_arrays))

        assert len(a) == num_arrays
        assert a.shape == (m, n)

        ab = a @ b
        assert len(ab) == num_arrays
        assert ab.shape == (m, p)

        ca = c @ a
        assert len(ca) == num_arrays
        assert ca.shape == (q, n)

        d = a.delete_element(1)
        assert len(d) == num_arrays - 1
        assert np.array_equal(d[0], a[0])
        assert np.array_equal(d[1], a[2])

        d = a.delete_elements(np.array([6, 1, 3]))
        assert len(d) == num_arrays - 3
        assert np.array_equal(d[0], a[0])
        assert np.array_equal(d[1], a[2])
        assert np.array_equal(d[2], a[4])
        assert np.array_equal(d[3], a[5])
        assert np.array_equal(d[4], a[7])
        assert np.array_equal(d[5], a[8])
        assert np.array_equal(d[6], a[9])
        assert np.array_equal(d[7], a[10])

    def test_unravel_lower(self):
        a = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ])
        a_vector = cpd.linalg.unravel_lower(a)
        assert np.array_equal(a_vector, [1, 4, 7, 5, 8, 9])

    def test_ravel_and_unravel_lower(self):
        g = np.array([
            [1, 4, 7],
            [4, 5, 8],
            [7, 8, 9],
        ])
        g_vector = cpd.linalg.unravel_lower(g)
        assert np.array_equal(g_vector, [1, 4, 7, 5, 8, 9])
        assert np.array_equal(cpd.linalg.ravel_lower(g_vector), np.tril(g))
        assert np.array_equal(cpd.linalg.ravel_lower(g_vector, symmetric=True), g)

    def test_norm_weighting(self):
        n = 10
        b = np.random.random((n, n))
        b = b + b.T
        d = cpd.linalg.norm_weight_matrix(b.shape[0])
        b_vector = cpd.linalg.unravel_lower(b)
        assert norm(d @ b_vector) - norm(b) == pytest.approx(1e-15)

    def test_matrix_norm_identity(self):
        """Tests that for any A=A^T, B, |A-B|^2 = |A-0.5*(B+B^T)|^2 + |0.5*(B-B^T)|^2 in the Frobenius norm."""
        n = 10
        a = 2 * np.random.random((n, n)) - 1
        a = a + a.T
        b = 2 * np.random.random((n, n)) - 1
        assert norm(a - b) ** 2 == pytest.approx(norm(a - 0.5 * (b + b.T)) ** 2 + norm(0.5 * (b - b.T)) ** 2)

    def test_local_min(self):
        # Create a uni-modal array.
        arr = random_unimodal_array(np.random.randint(10, 20))
        assert cpd.linalg.local_min(arr) == np.argmin(arr)

    @pytest.mark.parametrize("n", np.logspace(0, 5, base=10, num=6, dtype=int))
    def test_local_min_functor(self, n):
        # Create a uni-modal array and wrap it in a functor.
        np.random.seed(1)
        arr = random_unimodal_array(n)
        assert len(arr) == n
        f = ArrayFunctor(arr)

        index = cpd.linalg.local_min_functor(f, n)

        # Complexity is ~ 3*log2(n).
        assert index == np.argmin(arr)
        assert f.n_eval <= 3 * np.log2(n)


def random_unimodal_array(n: int) -> np.ndarray:
    """Creates a uni-modal array."""
    return np.concatenate((np.sort(np.random.random(n // 2))[::-1], np.sort(np.random.random(n - (n // 2)))))


class ArrayFunctor:
    def __init__(self, data):
        self._data = data
        self.n_eval = 0

    def __call__(self, i: int) -> float:
        self.n_eval += 1
        return self._data[i]