"""Solves the near-CPD (near-PD with bounded condition number) problem using the Tanaka-Nakata (TTN) algorithm for the
Lp norm. This includes the particular case of the Frobenius norm (p=2).

We introduce a new slight speed-up improvement in calculating the optimal eigenvalues in O(n) operations (instead of
O(n log n) in the T-N paper). O(n) algorithms already exist for the Frobenius case, and here we solve it in O(n) for a
general p using a simple bracketing method + Newton's method in two intervals.

Original paper: Tanaka, M. and Nakata, K., Positive definite matrix approximation with condition number constraint.
Optim Lett (2014) 8:939–947 DOI: 10.1007/s11590-013-0632-7
"""
import bisect
import itertools
import numpy as np
import scipy.optimize
import sortednp as snp
from typing import Tuple

import cpd.linalg


def tn(a, kappa: float, p: float = 2, full_output: bool = False, method: str = "direct"):
    """
    Returns the nearest SPD matrix to a given matrix in the Frobenius norm whose condition number <= kappa.

    Args:
        a: original square matrix. If non-symmetric, the algorithm is run on its symmetric part.
        kappa: condition number bound. Must be > 1. In most cases, the returned matrix will have a condition numbeerr
            exactly equal to kappa.
        p: exponent of Lp norm (default: p=2: Frobenius/2-norm).
        full_output: bool, optional
            If True, returns an optimization result struct with # function calls, etc.
        method: eigenvalue computation method.
            "direct": O(n^2) direct computation of minimizers. Works only for p == 2.
            "binary":O(n log n) binary search within endpoints for the unique minimum + newton in the two adjacent
                intervals. Note: Newton's method reduces to a direct method for p = 2, since f is piecewise quadratic.
            "brent": O(n) with a large constant. Brent's method to minimize f globally.

    Returns:
        conditioned near-SPD matrix.

    See:
        Tanaka, M. and Nakata, K., Positive definite matrix approximation with condition number constraint.
        Optim Lett (2014) 8:939–947 DOI: 10.1007/s11590-013-0632-7
    """
    assert method != "direct" or p == 2
    assert a.shape[0] == a.shape[1]
    assert kappa > 1
    b = 0.5 * (a + a.T)
    lam, q = np.linalg.eigh(b)
    lmin, lmax = min(lam), max(lam)
    if lmax <= 1e-8:
        # All eigenvalues are non-positive. mu's range is the point 0.
        z = np.zeros_like(a)
        return (z, None) if full_output else z
    if lmin > 0 and (kappa > lmax / lmin):
        return (b, None) if full_output else b

    if method in ("direct", "binary"):
        intervals = list(_get_intervals(lam, kappa))
        endpoint_index = [item[0] for item in intervals]
        endpoint = [item[1] for item in intervals]
        if method == "direct":
            l, u, mu = _minimize_explicit_frobenius(endpoint, endpoint_index, kappa, lam)
            result = None
        elif method == "binary":
            # Find the discrete minimum of f over all interval endpoints.
            endpoint_list = np.array([pair[0] for pair in endpoint] + [endpoint[-1]])
            ind = _minimize_binary(endpoint_list, endpoint_index, kappa, lam, p)
            # Minimize in left and right intervals, if exist.
            inds = []
            if ind > 0:
                inds.append(ind - 1)
            if ind < len(endpoint) - 1:
                inds.append(ind)
            results = [cpd.optim.minimize_newton(
                _f, 0.5 * (endpoint[ind][0] + endpoint[ind][1]),
                _f_deriv1, _f_deriv2,
                args=(lam, kappa, endpoint_index[ind], p))
                for ind in inds]
            def flatten(x):
                if isinstance(x, np.ndarray):
                    return x.item()
                return x
            interval_minimizers = [flatten(np.clip(result.x, endpoint[ind][0], endpoint[ind][1]))
                                   for ind, result in zip(inds, results)]
            # Global minimum = minimum over left and right intervals. Reevaluate f after clipping the
            # minimizers to ensure we have the right function values.
            mu, (l, u) = min((_f(mu, lam, kappa, endpoint_index[ind], p), (mu, endpoint_index[ind]))
                             for ind, mu in zip(inds, interval_minimizers))[1]
            # Minimum may be attained at the boundary of the domain mu >= 0. This should be taken care of
            # by the restriction of minimization to intervals where f is convex, but add a max() here just in case.
            mu = max(mu, 0)
            result = results[0]
            result.success = all(item.success for item in results)
            result.nfev = sum(item.nfev for item in results)
    elif method == "brent":
        f = TnFunction(lam, kappa)
        result = scipy.optimize.minimize_scalar(f.fun, bracket=(0, f.upper_bound), args=(p,), method="brent")
        # Minimum may be attained at the boundary of the domain mu >= 0.
        mu = max(result.x, 0)
        l, u = f.interval_indices(mu)
    else:
        raise Exception("Unsupported method {}".format(method))
    lam_modified = [mu] * (l + 1) + list(lam[l + 1:u]) + [kappa * mu] * (len(lam) - u)
    b = q @ np.diag(lam_modified) @ q.T
    if full_output:
        return b, result
    else:
        return b


def _merge_indices(i1, i2):
    """
    Given two index lists of a merge-sort operation, returns a merged list of indices characterizing the elements of
    the merged list.

    Let a, b be two sorted lists, and let c be the result of merge-sort. Let i1, i2 be the the lists of indices of
    elements in the lists a and b in their order of appearance in c, respectively. For example:

    import numpy as np
    import sortednp as snp

    >>> a = np.array([-3, 4, 8, 1])
    >>> b = np.array([0, 6, 1, -1])
    # Merge-sort + return indices.
    >>> c, (i1, i2) = snp.merge(a, b, indices=True)
   array([-3,  0,  4,  6,  1, -1,  8,  1])
    >>> i1
    array([0, 2, 6, 7])
    >>> i2
    array([1, 3, 4, 5])
    >>> list(_merge_indices(i1, i2))
    [(0, 0), (1, 0), (0, 1), (1, 1), (1, 2), (1, 3), (0, 2), (0, 3)]

    Args:
        i1: sorted index list (indices of elements in the list 'a' in their order of appearance in 'c').
        i2: sorted index list (indices of elements in the list 'b' in their order of appearance in 'c').

    Returns:
        List of tuples: (0 or 1 corresponding to whether element c[i] is from a or b, j where c[i] = a[j] (or b[j])).
    """
    j1, j2 = 0, 0
    n1, n2 = len(i1), len(i2)
    for i in range(n1 + n2):
        first = 0 if ((j1 < len(i1)) and (i1[j1] == i)) else 1
        yield first, (j1 if first == 0 else j2)
        if first == 0:
            j1 += 1
        else:
            j2 += 1


class TnFunction:
    """Defines the T-N minimization function for a set of eigenvalues lam and condition number bound kappa."""
    def __init__(self, lam: np.ndarray, kappa: float):
        merged_endpoints, interval_indices = TnFunction._get_interval_indices(lam, kappa)
        self._merged_endpoints = merged_endpoints
        self._interval_indices = interval_indices
        self._lam = lam
        self._kappa = kappa

    @property
    def upper_bound(self):
        return self._merged_endpoints[-1]

    def fun(self, mu: float, p: float) -> float:
        """Returns the T-N function of mu. Works for all mu, not just non-negative. Piecewise smooth
        on each interval of the indices (l, u). O(log n) for bounds search + O(n) for function evaluation."""
        return _f(mu, self._lam, self._kappa, self.interval_indices(mu), p)

    def deriv1(self, mu: float, p: float) -> float:
        """Returns the first derviative of the T-N function of mu."""
        return _f_deriv1(mu, self._lam, self._kappa, self.interval_indices(mu), p)

    def deriv2(self, mu: float, p: float) -> float:
        """Returns the first derviative of the T-N function of mu."""
        return _f_deriv2(mu, self._lam, self._kappa, self.interval_indices(mu), p)

    def interval_indices(self, mu):
        try:
            return self._interval_indices[bisect.bisect_left(self._merged_endpoints, mu)]
        except IndexError:
            return None

    @staticmethod
    def _get_interval_indices(lam: np.ndarray, kappa: float):
        """
        Returns the T-N function components: interval endpoints and (l, u) indices.
        Args:
            lam: sorted list of eigenvalues.
            kappa: condition number bound.

        Returns:
            List of merged endpoint list: lam union lam/kappa. Includes an endpoint past the last value
                in lam union lam/k, so a total of 2*n+1 where n=len(lam).
            List of tuples (l, u) corresponding to all intervals. Includes an interval past the last
                value in the merged endpoints, sso a total of 2*n+1 intervals where n=len(lam)."""
        a = [lam, lam / kappa]
        merged_endpoints, (i1, i2) = snp.merge(a[0], a[1], indices=True)

        limit = [-1, -1]  # limit = current value of (l, u).
        interval_indices = [None] * (len(merged_endpoints) + 1)
        # Loop over index pairs in _merge_indices plus an extra entry for the one-past-end interval.
        for i, ind in enumerate(itertools.chain(_merge_indices(i1, i2), [(0, len(lam))])):
            interval_indices[i] = limit[0], limit[1] + 1
            limit[ind[0]] = ind[1]
        return merged_endpoints, interval_indices


def _f(mu: float, lam: np.ndarray, kappa: float, endpoint_index: Tuple[int], p: float):
    """Function of mu for mu >= 0. Piecewise quadratic on each interval of the indices (l, u)."""
    l, u = endpoint_index
    return sum(np.abs(mu - lam[:l + 1]) ** p) + sum(np.abs(lam[u:] - kappa * mu) ** p)


def _f_deriv1(mu: float, lam: np.ndarray, kappa: float, endpoint_index: Tuple[int], p: float):
    """f'."""
    l, u = endpoint_index
    return p * (sum(np.abs(mu - lam[:l + 1]) ** (p - 1) * np.sign(mu - lam[:l + 1])) -
                kappa * sum(np.abs(lam[u:] - kappa * mu) ** (p - 1) * np.sign(lam[u:] - kappa * mu)))


def _f_deriv2(mu: float, lam: np.ndarray, kappa: float, endpoint_index: Tuple[int], p: float):
    """f''."""
    l, u = endpoint_index
    return p * (p - 1) * (sum((mu - lam[:l + 1]) ** (p - 2)) + kappa ** 2 * sum((lam[u:] - kappa * mu) ** (p - 2)))


def _mu_minimizer_in_interval_direct(lam, kappa, endpoint_index, endpoint):
    """O(n^2) direct computation."""
    for (l, u), (mu_min, mu_max) in zip(endpoint_index, endpoint):
        if l == -1 and u == len(lam):
            yield 0
        else:
            mu = (sum(lam[:l + 1]) + kappa * sum(lam[u:])) / (l + 1 + kappa ** 2 * (len(lam) - u))
            yield min(max(mu, mu_min), mu_max)


def _minimize_explicit_frobenius(endpoint, endpoint_index, kappa, lam):
    """Explicit O(n^2) computation of p=2 minimizers in all intervals."""
    mu_minimizer = np.array(list(_mu_minimizer_in_interval_direct(lam, kappa, endpoint_index, endpoint)))
    f_min = np.array([_f(mu, lam, kappa, ind_pair, 2) for mu, ind_pair in zip(mu_minimizer, endpoint_index)])
    ind = np.argmin(f_min)
    return endpoint_index[ind][0], endpoint_index[ind][1], mu_minimizer[ind]


def _minimize_binary(endpoint, endpoint_index, kappa, lam, p):
    """O(n log n) binary search of minimizer among  all interval endpoints. Works for any p."""
    def f_endpoint(ind):
        return _f(endpoint[ind], lam, kappa, endpoint_index[ind], p)
    return cpd.linalg.local_min_functor(f_endpoint, len(endpoint))


def _get_intervals(lam, kappa):
    """lam must be sorted."""
    a = [lam, lam / kappa]
    merged_endpoints, (i1, i2) = snp.merge(a[0], a[1], indices=True)
    merged_index = list(_merge_indices(i1, i2)) + [(0, len(lam))]
    # Right-pad so |one-past-last interval| = |last interval|.
    merged_endpoints = np.append(merged_endpoints, 2 * merged_endpoints[-1] - merged_endpoints[-2])

    limit, endpoint_prev = [-1, -1], -1  # limit = (l, u). Dummy negative number for 'endpoint_prev'.
    for ind, endpoint in zip(merged_index, merged_endpoints):
        if endpoint > 0:
            # Add 1 from u to match f's definition in terms of lam[l-1], lam[l]; lam[u-1] ,lam[u].
            yield (limit[0], limit[1] + 1), (max(0, endpoint_prev), endpoint)
        limit[ind[0]] = ind[1]
        endpoint_prev = endpoint
