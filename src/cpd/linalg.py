"""Linear algebra utilities."""
import itertools
from typing import Callable, Tuple

import numpy as np
from scipy.linalg import qr

# Force our __rmatmul__ to be called. For the necessary lowest values of MAGIC_NUMBER look into the numpy docs.
MAGIC_NUMBER = 15.0


class ArrayList:
    """
    A container that holds a list of numpy arrays. Defines matrix multiplication ("@") as multiplying each
    element in the container by a matrix (on the left or right).

    All arrays in the list are assumed to have the same shape.
    """
    __array_priority__ = MAGIC_NUMBER

    def __init__(self, array_generator):
        self._list = np.array(list(array_generator))
        assert all(self._list[i].shape == self._list[0].shape for i in range(1, len(self._list)))

    def __len__(self) -> int:
        return len(self._list)

    def __getitem__(self, key):
        return self._list[key]

    def __iter__(self):
        return iter(self._list)

    def __str__(self) -> str:
        return "ArrayList[size={}]".format(len(self))

    def __matmul__(self, other: np.ndarray):
        return ArrayList([a @ other for a in self._list])

    def __rmatmul__(self, other: np.ndarray):
        return ArrayList([other @ a for a in self._list])

    @property
    def shape(self):
        """Returns the shape of each element in this matrix list."""
        return self._list[0].shape

    def delete_element(self, index: int):
        """Returns a copy of the ArrayList with index 'index' removed. For R-caller during jackniffing."""
        return ArrayList(np.concatenate((self._list[:index], self._list[index + 1:])))

    def delete_elements(self, index: np.ndarray):
        """Returns a copy of the ArrayList with indices in the list 'index' removed. 'index' is not assumed
        to be ordered. For R-caller during jackniffing."""
        keep = np.setdiff1d(np.arange(0, len(self._list)), index)
        return ArrayList(self._list[keep])


def unravel_lower(c: np.ndarray) -> np.ndarray:
    """
    Converts the diagonal+lower triangular part of an n x n matrix C into a vector.
    """
    i, j = lower_subscripts(c.shape[0])
    return c[i, j].T


def ravel_lower(g_vector: np.ndarray, symmetric: bool = False) -> np.ndarray:
    """
    Converts a vector into the diagonal+lower-triangular part of a symmetric solution matrix G.
    The diagonal part is contiguous and appears first in the vector.

    Args:
        g_vector: flattened solution matrix, shape=(1, n * (n + 1) / 2)).
        symmetric: if True, returns the symmetric matrix (i.e., A + A.T - diag(A) where A = raveled lower-triangular
        matrix).

    Returns:
        Original solution matrix, shape=(n, n).
    """
    n = triangular_number_to_base(len(g_vector))
    k, l = lower_subscripts(n)
    g = np.zeros((n, n))

    # Fill in diagonal part.
    diagonal_index = lower_subscripts_diagonal_index(n)
    np.fill_diagonal(g, g_vector[diagonal_index])

    # Fill in lower triangular part.
    lower_index = np.where(k != l)[0]
    g[k[lower_index], l[lower_index]] = g_vector[lower_index]

    if symmetric:
        g = g + g.T - np.diag(np.diag(g))
    return g


def lower_subscripts_diagonal_index(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns the index of the diagonal elements of the diagonal+lower triangular part of an n x n
    matrix in the flattened vectors of lower_subscripts().

    Args:
        n: matrix dimension.

    Returns:
        diagonal element indices in the flattened vector of the diagonal+lower triangular part of an n x n matrix.
    """
    i, j = lower_subscripts(n)
    return np.where(i == j)[0]


def lower_subscripts(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns i, j = row, column indices of the diagonal+lower triangular part of an n x n
    matrix.

    Args:
        n: matrix dimension.

    Returns:
        i, j = flattened subscript vectors.
    """
    row_index = np.array(list(itertools.combinations_with_replacement(range(n), 2)))
    return row_index[:, 1], row_index[:, 0]


def norm_weight_matrix(n: int) -> np.ndarray:
    """
    Returns the norm weighting diagonal matrix that matches 2-norm of the unraveled matrix with the Frobenius
    norm of the original matrix.

    Args:
        n: matrix dimension.

    Returns:
        Diagonal weight matrix.
    """
    i, j = lower_subscripts(n)
    weight = np.sqrt(2) * np.ones((len(i),))
    weight[i == j] = 1
    return np.diag(weight)


def random_spsd(n, k):
    """
    Generates a symmetric semi-PD matrix G.

    Args:
        n: matrix dimension.
        k: # zero eigenvalues.

    Returns:
        symmetric semi-PD G.
    """
    h = np.random.randn(n, n)
    q = qr(h)[0]
    d = np.diag([0] * k + np.random.random(n - k).tolist())
    g = q.T @ d @ q
    return g


def is_spd(a, eig_tol: float = 0):
    """Returns true when input is positive-definite, via Cholesky. If eig_tol > 0, also requires
    the condition number to be at least 1/eig_tol."""
    if eig_tol < 1e-15:
        return True
    try:
        np.linalg.cholesky(a)
        return np.linalg.cond(a) < 1 / eig_tol
    except np.linalg.LinAlgError:
        return False


def triangular_number_to_base(n_triangular: int):
    """
    Assuming n_triangular = n * (n + 1)/2, returns n.
    Args:
        n_triangular: triangular number.

    Returns:
        triangle bae.
    """
    return int(0.5 * ((8 * n_triangular + 1) ** 0.5 - 1))


def unit_vector(n, i):
    """
    Returns the ith unit vector of size n (ei in R^n).
    Args:
        n: vector dimension.
        i: index of nonzero.

    Returns:
        unit vector = (0,...,0,1,0,...0).

    """
    e = np.zeros((n,))
    e[i] = 1
    return e


def local_min(arr: np.ndarray) -> int:
    """
    Returns -an- index of a local minimum in the array. Uses binary search. Complexity: O(log n).
    Args:
        arr:

    Returns:
        index of local minimum.
    """
    n = len(arr)
    low, high = 0, n - 1
    while low < high:
        # Find index of middle element.
        mid = low + (high - low) // 2
        # Compare middle element with its neighbours (if they exist).
        if (mid == 0 or arr[mid - 1] > arr[mid]) and (mid == n - 1 or arr[mid] < arr[mid + 1]):
            return mid
        elif mid > 0 and arr[mid - 1] < arr[mid]:
            # If middle element is not minima and its left neighbor < it, then left half must have a local minima.
            high = mid - 1
        else:
            # If middle element is not minima and its right neighbor < it, then right half must have a local minima.
            low = mid + 1
    return low


def local_min(arr: np.ndarray) -> int:
    """
    Returns -an- index of a local minimum in the array. Uses binary search. Complexity: O(log n).
    Args:
        arr:

    Returns:
        index of local minimum.
    """
    n = len(arr)
    low, high = 0, n - 1
    while low < high:
        # Find index of middle element.
        mid = low + (high - low) // 2
        # Compare middle element with its neighbours (if they exist).
        if (mid == 0 or arr[mid - 1] > arr[mid]) and (mid == n - 1 or arr[mid] < arr[mid + 1]):
            return mid
        elif mid > 0 and arr[mid - 1] < arr[mid]:
            # If middle element is not minima and its left neighbor < it, then left half must have a local minima.
            high = mid - 1
        else:
            # If middle element is not minima and its right neighbor < it, then right half must have a local minima.
            low = mid + 1
    return low


def local_min_functor(arr: Callable[[int], float], n: int) -> int:
    """
    Returns -an- index of a local minimum in an array, only that the array is evaluated at each element using
    the function 'arr'. Uses binary search. Complexity: O(log n).
    Args:
        arr: function of an integer i that evaluates arr[i], i = 0..n-1.
        n: size of the array.

    Returns:
        index of local minimum.
    """
    low, high = 0, n - 1
    while low < high:
        # Find index of middle element.
        mid = low + (high - low) // 2
        # Compare middle element with its neighbours (if they exist).
        a_mid = arr(mid)
        if (mid == 0 or arr(mid - 1) > a_mid) and (mid == n - 1 or a_mid < arr(mid + 1)):
            return mid
        elif mid > 0 and arr(mid - 1) < a_mid:
            # If middle element is not minima and its left neighbor < it, then left half must have a local minima.
            high = mid - 1
        else:
            # If middle element is not minima and its right neighbor < it, then right half must have a local minima.
            low = mid + 1
    return low
