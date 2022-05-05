"""Estimating a conditioned positive definite matrix G under general constraints using RCO
(Regularized Cholesky Optimization)."""
import itertools
from typing import Callable, Optional, Tuple

import numpy as np
import scipy.optimize
import scipy.sparse
from numpy.linalg import norm

import cpd.optim_continuation
from cpd._functional import Functional
from cpd.linalg import ArrayList


class _NearRcoOptimizer:
    """
    Finds a well-conditioned positive definite matrix G^* closest to a given matrix, by optimizing
    its Cholesky factor. A near-PD/RCO hybrid.
    """

    def __init__(self, n: int):
        """
        Creates an near-PD RCO optimizer for an n x n matrix.
        Args:
            n: matrix dimension.
        """
        # Encode a into a list of matrices q appearing in quadratic-form terms of the LS functional of the Cholesky
        # factor. Save the weight matrix d since we multiply the RHS by it in optimize().
        self._a = cpd.moment_functional.frobenius_error_to_matrix(n)

    def optimize(self, g, leeway_factor: float = 1.1, tol: float = 1e-4, tol_curve: float = 1e-3,
                 num_alpha: int = 5, alpha_init: float = 100, alpha_step: float = 0.01,
                 residual_fun: Callable[[np.ndarray], float] = None,
                 regularization: str = "all_scaled",
                 start_from_small_alpha: bool = True,
                 continuation: bool = True,
                 adaptive: bool = False) -> Tuple[np.ndarray, cpd.optim.OptimizeResult]:
        """
        Performs a continuation to find the entire ROC curve of min fun_f(x) + alpha * fun_reg(x) vs. alpha
        and returns the minimizer.

        Args:
            leeway_factor: maximum allowed ratio of f(x)/min_alpha f(x, alpha) in the produced solution x. That is,
                if leeway_factor=1.1, we allow 10% increase in f value to minimize solution_metric(x).
            tol: gradient tolerance in the final optimization step (should typically be smaller than tol_curve).
            tol_curve: gradient tolerance in producing ROC solutions.
            num_alpha: number of ROC curve points to generate.
            alpha_init: Initial alpha is alpha-init * |f(x0)|/|reg_term(x0)| where x0=initial guess for continuation.
            alpha_step: alpha decrease factor along the ROC continuation curve.
            residual_fun: optional function (residual) value to report; if Non, uses f(G)=||G-g||.
            regularization: regularization method ("diagonal" for diagonal elements, or "all" for all elements).
            start_from_small_alpha: if True, starts from alpha=0 and climbs up to alpha_init. If False, starts
                from alpha_init and decreases it 'num_alpha' times.
            continuation: if True, performs continuation along the ROC curve, starting from alpha_init and using the
                initial guess from the previous (larger) alpha of the current alpha problem. If False, uses 'x' as
                the initial guess for all alpha problems.

        Returns:
            x: final optimized solution.
            info: struct with info on x (alpha, relative moments equation error value, solution metric).
            curve: ROC curve (#alpha x 3: alpha, relative moments equation error value, solution metric).
            index: optimal index in the curve array corresponding to the final optimum.
        """
        # Set up the minimization functions.
        objective = MomentCholeskyFunctionalExplicit(self._a, g)
        reg = cpd._rco_regularizer.create_regularizer(regularization)(g.shape[0])
        # Using an identity initial guess and find a large enough regularization parameter value 'alpha'
        # such that the regularization dominates the minimization.
        if regularization == "all_scaled":
            diagonal_const = np.mean(np.diagonal(g))
        else:
            diagonal_const = 1
        x = diagonal_const * reg.identity()
        alpha_init_scaled = alpha_init * np.abs(objective.fun(x)) / np.abs(reg.fun(x))
        # To ensure we reproduce g at alpha=0 when it is already SPD, start from the near PD solution with
        # a small threshold so we can perform Cholesky, and perform the continuation starting from alpha=0
        # and going to infinity.
        g_near_pd = cpd._higham.near_pd(g, eig_tol=1e-6)
        l = np.linalg.cholesky(g_near_pd)
        x = cpd.linalg.unravel_lower(l)
        # Perform continuation in alpha.
        result = cpd.optim_continuation.optimize(objective, reg, cpd.moment_functional.solution_metric, x,
                                                 leeway_factor=leeway_factor, num_alpha=num_alpha, alpha_init=alpha_init_scaled,
                                                 alpha_step=alpha_step, tol=tol, tol_curve=tol_curve, residual_fun=residual_fun,
                                                 continuation=continuation, start_from_small_alpha=start_from_small_alpha,
                                                 adaptive=adaptive)
        if residual_fun is None:
            # Translate function value into relative error, not relative error squared.
            result.curve[:, 1] **= 0.5
            info = result.info
            result.info = (info[0], info[1] ** 0.5, info[2])
        # Translate x back into L and into G.
        l = cpd.linalg.ravel_lower(result.x)
        g_rco = l @ l.T
        return g_rco, result


class MomentCholeskyFunctionalRaveled(Functional):
    """
    Calculates the Cholesky functional f(l) = |sum_i(P_i L @ L^T P_i^T)  - G|_F^2/norm(G) and its derivatives.
    Unraveled form |A*l - l|/norm(g).
    """

    def __init__(self, a: ArrayList, c: np.ndarray):
        """
        Creates an RCO optimizer for an n x n matrix.
        Args:
            a: unraveled sum_i(P_i . P_i^T) functional matrix.
            c: RHS matrix of the functional.
        """
        n = cpd.linalg.triangular_number_to_base(a.shape[0])
        self._d = cpd.linalg.norm_weight_matrix(n)
        self._a = self._d @ a
        self._b = self._d @ cpd.linalg.unravel_lower(c)

    def fun(self, l_vector: np.ndarray) -> float:
        l = cpd.linalg.ravel_lower(l_vector)
        return norm(self._a @ cpd.linalg.unravel_lower(l @ l.T) - self._b) / norm(self._b)


class MomentCholeskyFunctionalQf(Functional):
    """
    Calculates the Cholesky functional f(l) = |sum_i(P_i L @ L^T P_i^T)  - G|_F^2 and its derivatives using
    quadratic form representation.
    """

    def __init__(self, a: ArrayList, g: np.ndarray):
        """
        Creates an RCO optimizer for an n x n matrix.
        Args:
            n: matrix dimension.
        """
        # Encode a into a list of matrices q appearing in quadratic-form terms of the LS functional of the Cholesky
        # factor. Save the weight matrix d since we multiply the RHS by it in optimize().
        n = g.shape[0]
        d = cpd.linalg.norm_weight_matrix(n)
        self._n = n
        self._q = _cholesky_factor_functional(d @ a)
        self._q_symmetric = [ai + ai.T for ai in self._q]

        # Set up the minimization functions.
        b = d @ cpd.linalg.unravel_lower(g)
        self._r = 2 * sum(bi * ai for ai, bi in zip(self._q, b))
        self._r_symmetric = self._r + self._r.T
        self._b2 = sum(b ** 2)
        self._scale = 1 / norm(b) ** 2
        self._diagonal_index = cpd.linalg.lower_subscripts_diagonal_index(n)

    def fun(self, x: np.ndarray):
        return self._scale * (sum((x.T @ ai @ x) ** 2 for ai in self._q) - x.T @ self._r @ x + self._b2)

    def grad(self, x: np.ndarray):
        return self._scale * (2 * sum((x.T @ ai @ x) * ai_symmetric @ x
                                      for ai, ai_symmetric in zip(self._q, self._q_symmetric)) - self._r_symmetric @ x)


class MomentCholeskyFunctionalExplicit(Functional):
    """
    Calculates the Cholesky functional f(l) = |sum_i(P_i L @ L^T P_i^T)  - G|_F^2 and its derivatives.
    """

    def __init__(self, a: Optional, g: np.ndarray, w: Optional[np.ndarray] = None):
        """
        Creates an RCO optimizer for an n x n matrix.
        Args:
            a: unraveled sum_i(P_i . P_i^T) functional matrix representing the LHS matrix in f. If None, P=[I]
                (Frobenius near PD use case).
            g: RHS matrix G in f.
            w: optional m x m least-squares SYMMETRIC weight matrix. If None, unit weights are used. Only the diagonal
                and lower-triangular part of w are used; it is assumed to be symmetrically continued to its upper half.
        """
        if a is None:
            # Frobenius near PD case.
            a = scipy.sparse.csr_matrix(cpd.moment_functional.frobenius_error_to_matrix(g.shape[0]))
        m = cpd.linalg.triangular_number_to_base(a.shape[0])
        n = cpd.linalg.triangular_number_to_base(a.shape[1])
        # Set up minimization function parts.
        self._n = n
        self._N = n * (n + 1) // 2
        self._a = a
        d = cpd.linalg.norm_weight_matrix(m)
        if w is not None:
            w_vector = cpd.linalg.unravel_lower(w)
            d = d * w_vector[:, None]
        self._a = d @ a
        self._b = d @ cpd.linalg.unravel_lower(g)
        self._scale = 1 / norm(self._b) ** 2
        self._llt = LltFunctional(n)

    def fun(self, x: np.ndarray):
        r = self._a @ self._llt.fun(x) - self._b
        return self._scale * norm(r) ** 2

    def grad(self, x: np.ndarray):
        r = self._a @ self._llt.fun(x) - self._b
        at_r = self._a.T @ r
        return 2 * self._scale * self._llt.grad(x) @ at_r

    def hessian(self, x: np.ndarray):
        """O(n^4) non-zeros because  of the llt.grad @ llt.grad^T term."""
        r = self._a @ self._llt.fun(x) - self._b
        at_r = self._a.T @ r
        a_dh = self._llt.grad(x) @ self._a.T
        term1 = a_dh @ a_dh.T
        h = self._llt.hessian(x)
        N = self._N
        term2 = sum(z * h[r * N:(r + 1) * N] for r, z in enumerate(at_r))
        return 2 * self._scale * (term1 + term2)

    def hessian_action(self, x: np.ndarray, y: np.ndarray):
        """Returns the action of the Hessian at x on a vector y. O(n^3)."""
        r = self._a @ self._llt.fun(x) - self._b
        at_r = self._a.T @ r
        a_dh = self._llt.grad(x) @ self._a.T
        term1 = a_dh @ (a_dh.T @ y)
        term2 = at_r @ self._llt.hessian_action(x, y)
        return 2 * self._scale * (term1 + term2)

    def tri_function(self, x: np.ndarray, y: np.ndarray):
        r = self._a @ self._llt.fun(x) - self._b
        f = self._scale * norm(r) ** 2

        at_r = self._a.T @ r
        df = 2 * self._scale * self._llt.grad(x) @ at_r

        a_dh = self._llt.grad(x) @ self._a.T
        d2f = 2 * self._scale * (a_dh @ (a_dh.T @ y) + at_r @ self._llt.hessian_action(x, y))

        return f, df, d2f


class FrobeniusCholeskyFunctionalExplicit(Functional):
    """
    Calculates the Cholesky functional f(l) = |L L^T  - G|_F^2 and its derivatives.
    This is an optimized version of MomentCholeskyFunctionalExplicit for the case of P = [I].
    """

    def __init__(self, g: np.ndarray):
        """
        Creates an RCO optimizer for an n x n matrix.
        Args:
            g: RHS matrix G in f.
        """
        n = g.shape[0]
        # Set up minimization function parts.
        self._n = n
        self._b = cpd.linalg.unravel_lower(g)
        self._scale = 1 / norm(self._b) ** 2
        self._llt = LltFunctional(n)

    def fun(self, x: np.ndarray):
        r = self._llt.fun(x) - self._b
        return self._scale * norm(r) ** 2

    def grad(self, x: np.ndarray):
        r = self._llt.fun(x) - self._b
        return 2 * self._scale * self._llt.grad(x) @ r

    def hessian(self, x: np.ndarray):
        """O(n^4) non-zeros because of the  dh @ dh.T term."""
        r = self._llt.fun(x) - self._b
        dh = self._llt.grad(x)
        return 2 * self._scale * (dh @ dh.T + np.hstack(tuple(hr @ r for hr in self._llt.hessian(x))))

    def hessian_action(self, x: np.ndarray, y: np.ndarray):
        """Returns the action of the Hessian at x on a vector y. O(n^3)."""
        r = self._llt.fun(x) - self._b
        dh = self._llt.grad(x)
        return 2 * self._scale * (dh @ (dh.T @ y) + (np.hstack(tuple(hr @ r for hr in self._llt.hessian(x)))) @ y)


class LltFunctional(Functional):
    """Evaluates h(l) = L*L^T for the flattened L matrix l, and its gradient and Hessian."""

    def __init__(self, n: int):
        # h-term.
        self._index = create_llt_mapping(n)

        # Gradient(h) terms.
        N = n * (n + 1) // 2
        self._n = n
        self._N = N
        # Index j for which (r,r) appears in index[j].
        diagonal = self._diagonal_index()
        # All r's have a cross-term for some h_j(x) except the last (lower-right diagonal entry of L).
        dh_nz_cross = self._cross_term_ij()

        # Grad Sparsity pattern remains fixed; reuse the matrix object and only set the data in each grad_h call.
        dh_nz_diag = np.stack((np.arange(N), diagonal)).T
        dh_ij = np.vstack((dh_nz_diag, dh_nz_cross[:, :2]))
        dh_row, dh_col = dh_ij[:, 0], dh_ij[:, 1]

        # Create a Jacobian matrix with dummy non-zero elements, which are however useful for saving the
        # non-zero element ordering in the CSR's data elements below in self._perm.
        self._dh = scipy.sparse.csr_matrix((np.arange(1, len(dh_row) + 1), (dh_row, dh_col)), shape=(N, N))
        self._cross_term_index = dh_nz_cross[:, 2]
        # Save the CSR matrix data vector ordering. 0-based index.
        self._perm = self._dh.data.copy() - 1

        # The Hessian is constant.
        self._hessian = self._build_hessian(dh_col, dh_row)

    def _build_hessian(self, dh_col, dh_row):
        # Hessian is constant in x.
        n = self._n
        N = self._N
        # Build a single non-zeros list of stacked Hessian.
        ti = np.cumsum(np.concatenate(([0], np.array([r * (r - 1) for r in range(n, 1, -1)]))))
        index = list((np.concatenate(([r], row + N)).astype(int), np.concatenate(([2], [1] * len(row))))
                     for r, row in
                     enumerate(
                         row for r in range(n - 1, -1, -1) for row in
                         np.array(list(itertools.chain.from_iterable(
                             itertools.chain.from_iterable((np.arange(r * (s + 1), (r + 1) * (s + 1)),
                                                            np.arange((r + 1) * s, (r + 1) * s + r - s))
                                                           for s in range(r))))).reshape(r, r + 1).T + ti[n - r - 1]
                     ))
        nz_list = [(dh_row[i] + r * N, dh_col[i], data) for r, (i, data) in enumerate(index)]
        row = np.concatenate(tuple(row[0] for row in nz_list))
        col = np.concatenate(tuple(row[1] for row in nz_list))
        data = np.concatenate(tuple(row[2] for row in nz_list))

        # Stack Hessians of h[0],...,h[N-1] horizontally (next to each other) to obtain a N^2 x N matrix, since creating
        # many sparse matrices is slow so we make just one.
        # Instead, we could do smart ordering of the 'index' expression above, but at this point it's too hard...
        idx = np.argsort(col)
        sorted_row = row[idx]
        sorted_col = col[idx]
        sorted_data = data[idx]
        endpoints = np.concatenate(([0], np.where(np.diff(sorted_col) != 0)[0] + 1, [len(col)]))
        hr_row = sorted_row // N
        hr_col = sorted_row % N
        hr_row = np.concatenate(tuple(hr_row[endpoints[r]:endpoints[r + 1]] + r * N for r in range(N)))
        hessian = scipy.sparse.csr_matrix((sorted_data, (hr_row, hr_col)), shape=(N * N, N))
        # Old, slow code.
        # hessian = [None for r in range(N)]
        # for r in range(N):
        #     data = np.concatenate(
        #         (2 * cpd.linalg.unit_vector(N, r),
        #          (self._cross_term_index == r).astype(float))
        #     )
        #     hessian[r] = scipy.sparse.csr_matrix((data, (dh_row, dh_col)), shape=(N, N))
        #     hessian[r].eliminate_zeros()
        # hessian = scipy.sparse.vstack(hessian)
        #
        # Also slow:
        # return [hessian[:, r].reshape(N, N) for r in range(N)]
        return hessian

    def _cross_term_ij(self):
        # List of non-zeros in h corresponding to cross-terms: (r, j, tj).
        n = self._n

        # r-index list.
        t = np.concatenate(([0], np.cumsum(np.arange(n, 0, -1))))
        r_list = np.array(list(itertools.chain.from_iterable(
            itertools.chain.from_iterable([r] * (stop - start - 1) for r in range(start, stop))
            for start, stop in zip(t[:-1], t[1:])
        )))

        # r-index list.
        a = np.concatenate(([0], np.cumsum(np.arange(n, 1, -1))))
        j_list = np.array(list(itertools.chain.from_iterable(
            itertools.chain.from_iterable(tuple(
                (range(a[r] + 1, a[r + 1]),
                 itertools.chain.from_iterable(itertools.chain.from_iterable(
                     (a[r: r + s] + s - np.arange(s), range(a[r + s] + 1, a[r + s + 1]) if r + s + 1 < n else []) for s
                     in range(1, n - r))))))
            for r in range(n - 1))))

        # tj-index list.
        tj_list = np.array(list(itertools.chain.from_iterable(
            itertools.chain.from_iterable(
                itertools.chain.from_iterable((range(start, r), range(r + 1, stop)) for r in range(start, stop)))
            for start, stop in zip(a[:-1], a[1:])
        )))

        # Original slow code.
        # N = self._N
        # t = [None] * (N - 1)
        # for r in range(N - 1):
        #     tr = [(k[(k == r) ^ (l == r)], l[(k == r) ^ (l == r)]) for k, l in self._index]
        #     tr = [(j, [ks if ks != r else ls for ks, ls in zip(*trj)]) for j, trj in enumerate(tr)]
        #     # Each T_{rj} is either empty or contains a single element.
        #     tr = [(j, trj[0]) for j, trj in filter(lambda item: item[1], tr)]
        #     t[r] = list(zip(*tr))
        # # List of non-zeros in h corresponding to cross-terms: (r, j, tj).
        # dh_nz_cross = np.concatenate(tuple(np.stack((np.tile(r, len(j)), j, tj)) for r, (j, tj) in enumerate(t)),
        #                              axis=1).T
        # return dh_nz_cross
        return np.stack((r_list, j_list, tj_list)).T

    def _diagonal_index(self):
        n = self._n
        a = np.concatenate(([0], np.cumsum(np.arange(n, 1, -1))))
        diagonal = np.concatenate(tuple(a[r:] for r in range(n)))
        # This is the original code of locating the squared element terms L[k,l]^2 in (L L^T) entries, but it's slow,
        # and we can infer the indices directly.
        # diagonal  = np.array([next(iter(filter(lambda x: x[1].size > 0,
        #                                       [(j, np.where((k == r) & (l == r))[0])
        #                                        for j, (k, l) in enumerate(self._index)])))[0]
        #                                        for r in range(self._N)])
        return diagonal

    def fun(self, x: np.ndarray):
        return np.array([np.sum(x[k0] * x[k1]) for k0, k1 in self._index])

    def grad(self, x: np.ndarray):
        """Returns grad_h(x). Note: reuses the same CSR matrix and only updates its data in every call."""
        self._dh.data = np.concatenate((2 * x, x[self._cross_term_index]))[self._perm]
        return self._dh

    def hessian(self, x: np.ndarray):
        """Returns hessian(x) = list of hessian matrices of h[0]..h_[N-1]. Constant in x."""
        return self._hessian

    def hessian_action(self, x: np.ndarray, y: np.ndarray):
        """Returns a matrix whose columns are hessian(h_r)(x) @ y for all r=0..N-1."""
        return (self._hessian @ y).reshape(self._N, self._N)


def _cholesky_factor_functional(a: np.ndarray) -> np.ndarray:
    """"
    Converts a raveled a moment function matrix H(G^*) = A*G^* (for the flattened symmetric matrix G^*) into
    a quadratic-form matrix term list that appear in the functional for the unknown Cholesky factor L of G^*.
    Args:
        a: raveled EBLP moment function matrix, shape = (m * (m + 1) / 2, n * (n + 1) / 2) where G^* is m x n.

    Returns:
        list of M = m * (m + 1) / 2 matrices A_0,...,A_{M-1} w_i such that

        H(G^*) - C = H(L L^T) = sum_i (l^T A_i l - c_i)^2,

        where c is the raveled vector corresponding to the m x m symmetric matrix C.
    """
    M, N = a.shape
    n = cpd.linalg.triangular_number_to_base(N)
    element_of_pair = create_pair_mapping(n)
    data = np.concatenate(tuple(
        np.concatenate((np.arange(M)[:, None], np.tile((r, s), (M, 1)), a[:, j][:, None]), axis=1)
        for (r, s), j in element_of_pair.items()
    ), axis=0)
    data = data[data[:, 0].argsort()]
    data = data[data[:, 3] != 0]
    endpoints = np.concatenate(([0], np.where(np.diff(data[:, 0]) != 0)[0] + 1, [data.shape[0]]))
    w = [scipy.sparse.csr_matrix(
        (data[endpoints[i]:endpoints[i + 1], 3],
         (data[endpoints[i]:endpoints[i + 1], 1], data[endpoints[i]:endpoints[i + 1], 2])),
        shape=(N, N))
        for i in range(M)]

    # Dense matrix-vector multiplication os more efficient than sparse MVM at the typical size of an EBLP matrix.
    # To trade speed for space, remove this densification call.
    if N < 10000:
        w = [wi.toarray() for wi in w]
    return w


def create_pair_mapping(n):
    # Create a mapping of (k, l) elements in lower triangular part to a 1D index.
    k, l = cpd.linalg.lower_subscripts(n)
    ind = -np.ones((n, n), dtype=int)
    ind[k, l] = np.arange(n * (n + 1) // 2)
    # Holds the element indices of each row in L.
    ind = [ind[k, :(k + 1)] for k in range(ind.shape[0])]
    return dict(((r, s), j) for j in range(n * (n + 1) // 2) for r, s in zip(ind[k[j]], ind[l[j]]))


def create_llt_mapping(n):
    """
    Returns a list of index list pairs that encodes G = L*L^T where L = lower triangular into flattened arrays l
    and g for the lower triangular parts of L and G, respectively. Element in the output s corresponds to g[i] and
    consists of two l indices, i.e.

    g[i]  = sum_{0 <= j < len(s[i]0} l[s[i][0][j]] * l[s[i][1][j]], for all i = 0..n*(n+1)/2-1.

    Args:
        n: size of L and G.

    Returns:
        list of lists of pairs of index arrays into l.
    """
    # Create a mapping of (k, l) elements in lower triangular part to a 1D index.
    k, l = cpd.linalg.lower_subscripts(n)
    ind = -np.ones((n, n), dtype=int)
    ind[k, l] = np.arange(n * (n + 1) // 2)
    # Holds the element indices of each row in L.
    ind = [ind[k, :(k + 1)] for k in range(ind.shape[0])]
    return [(ind[k[j]][:min(k[j], l[j]) + 1], ind[l[j]][:min(k[j], l[j]) + 1]) for j in range(n * (n + 1) // 2)]


def near_pd(g, leeway_factor: float = 1.1, tol: float = 1e-4, tol_curve: float = 1e-3,
            num_alpha: int = 5, alpha_init: float = 100, alpha_step: float = 0.01,
            residual_fun: Callable[[np.ndarray], float] = None,
            regularization: str = "all_scaled",
            start_from_small_alpha: bool = True,
            continuation: bool = True, adaptive: bool = False) -> Tuple[np.ndarray, cpd.optim.OptimizeResult]:
    """
    Finds a well-conditioned positive definite matrix G^* closest to a given matrix, by optimizing
    its Cholesky factor. A near-PD/RCO hybrid.

    Performs a continuation to find the entire ROC curve of min fun_f(x) + alpha * fun_reg(x) vs. alpha
    and returns the minimizer and ROC curve.

    Args:
        g: original square matrix.
        leeway_factor: maximum allowed ratio of f(x)/min_alpha f(x, alpha) in the produced solution x. That is,
            if leeway_factor=1.1, we allow 10% increase in f value to minimize solution_metric(x).
        tol: gradient tolerance in the final optimization step (should typically be smaller than tol_curve).
        tol_curve: gradient tolerance in producing ROC solutions.
        num_alpha: number of ROC curve points to generate.
        alpha_init: Initial alpha is alpha-init * |f(x0)|/|reg_term(x0)| where x0=initial guess for continuation.
        alpha_step: alpha decrease factor along the ROC continuation curve.
        residual_fun: optional function (residual) value to report; if Non, uses f(G)=||G-g||^2_2.
        regularization: regularization method ("diagonal" for diagonal elements, or "all" for all elements).
        start_from_small_alpha: if True, starts from alpha=0 and climbs up to alpha_init. If False, starts
            from alpha_init and decreases it 'num_alpha' times.
        continuation: if True, performs continuation along the ROC curve, starting from alpha_init and using the
            initial guess from the previous (larger) alpha of the current alpha problem. If False, uses 'x' as
            the initial guess for all alpha problems.

    Returns:
        x: final optimized solution.
        info: struct with info on x (alpha, relative moments equation error value, solution metric).
        curve: ROC curve (#alpha x 3: alpha, relative moments equation error value, solution metric).
        index: optimal index in the curve array corresponding to the final optimum.
    """
    assert g.shape[0] == g.shape[1]
    optimizer = _NearRcoOptimizer(g.shape[0])
    g_cpd, result = optimizer.optimize(
        0.5 * (g + g.T), leeway_factor=leeway_factor, tol=tol, tol_curve=tol_curve, num_alpha=num_alpha,
        alpha_init=alpha_init, alpha_step=alpha_step, residual_fun=residual_fun,
        regularization=regularization, start_from_small_alpha=start_from_small_alpha, continuation=continuation,
        adaptive=adaptive)
    if residual_fun is None:
        info = result.info
        result.info = (info[0], norm(g_cpd - g) / norm(g), info[2])
    return g_cpd, result
