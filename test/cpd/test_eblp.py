"""Pawn game solver unit tests."""
import os
import unittest

import numpy as np
import pytest
from numpy.linalg import norm

import cpd
from cpd.moment_functional import f
from util import EBLP_SMALL_DATA_DIR


class TestEblp(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)

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
        f_unraveled = cpd.linalg.unravel_lower(f(g, p))
        n = len(c_vector)
        assert norm(f_unraveled.flatten() - (H @ g_vector).flatten()) < 1e-13 * n

    def test_unraveled_problem_least_squares_solution_matches_original(self):
        """Tests that f(G) = H*g (full sanity check of flattened (matrix * vector) vs. original matrix
        function. Data (G^*, P, C) loaded from files."""
        # Load original problem from files in matrix form.s
        c, w, p = load_example_data(EBLP_SMALL_DATA_DIR)
        # Original LS solution G, lower triangular part, flattened.
        g = np.loadtxt(os.path.join(EBLP_SMALL_DATA_DIR, "Gstar.txt"))
        g_vector = cpd.linalg.unravel_lower(g)

        # Solve the LS problem in the unraveled form of g.
        g_ls = solve_least_squares_problem(p, w, c)

        d = cpd.linalg.norm_weight_matrix(p.shape[0])
        w_unraveled = np.diag(cpd.linalg.unravel_lower(w))
        h = w_unraveled @ cpd.moment_functional.matrix_function_to_matrix_sum(p, p)
        c_vector = w_unraveled @ cpd.linalg.unravel_lower(c)
        assert norm(np.array(g_vector).flatten() - np.array(g_ls).flatten()) < 1e-13 * len(c_vector)
        assert norm(d @ (h @ g_ls - c_vector)) / norm(d @ c_vector) == \
               pytest.approx(norm((f(g, p) - c) / norm(c)))

    def test_rco_optimizer_unweighted(self):
        """End-to-end Cholesky factor optimization test."""
        # Load original problem from files in matrix form.
        c, w, p = load_example_data(EBLP_SMALL_DATA_DIR)

        optimizer = cpd.eblp.create_optimizer("rco", p)
        g, result = optimizer.optimize(c)

        # Relative error |f(G) - C|/|C|.
        assert result.info[1] == pytest.approx(0.0844, 1e-2)
        assert result.info[2] == pytest.approx(60380, 1e-1)
        assert result.index == 3
        # To save the solution, call:
        #np.savetxt(os.path.join(EBLP_SMALL_DATA_DIR, "Gstar_optimized.txt"), g)
        g_expected = np.loadtxt(os.path.join(EBLP_SMALL_DATA_DIR, "Gstar_optimized.txt"))
        assert g == pytest.approx(g_expected)

    def test_rco_optimizer_weighted(self):
        """End-to-end Cholesky factor optimization test."""
        # Load original problem from files in matrix form.
        c, w, p = load_example_data(EBLP_SMALL_DATA_DIR)

        optimizer = cpd.eblp.create_optimizer("rco", p, w)
        g, result = optimizer.optimize(c)

        # Relative error |f(G) - C|/|C|.
        assert result.info[1] == pytest.approx(0.0844, 1e-2)
        # Condition number.
        assert result.info[2] == pytest.approx(60380, 1e-1)
        assert result.index == 3
        # To save the solution, use
        # np.savetxt(os.path.join(DATA_DIR, "Gstar_optimized.txt"), g)
        g_expected = np.loadtxt(os.path.join(EBLP_SMALL_DATA_DIR, "Gstar_optimized.txt"))
        assert norm(g - g_expected) < 1e-5 * norm(g)

    def test_near_pd_optimizer_unweighted(self):
        """End-to-end near SPD optimization test."""
        # Load original problem from files in matrix form.
        c, w, p = load_example_data(EBLP_SMALL_DATA_DIR)

        optimizer = cpd.eblp.create_optimizer("higham", p, w)
        g, result = optimizer.optimize(c)

        # Relative error |f(G) - C|/|C|.
        assert result.info[1] == pytest.approx(0.1118, 1e-3)
        # Condition number.
        assert result.info[2] == pytest.approx(92762, 1e-1)
        assert result.index == 2

    def test_near_pd_optimizer_weighted(self):
        """End-to-end near SPD optimization test."""
        # Load original problem from files in matrix form.
        c, w, p = load_example_data(EBLP_SMALL_DATA_DIR)

        optimizer = cpd.eblp.create_optimizer("higham", p, w)
        g, result = optimizer.optimize(c)

        # Relative error |f(G) - C|/|C|.
        assert result.info[1] == pytest.approx(0.1118, 1e-3)
        # Condition number.
        assert result.info[2] == pytest.approx(92782, 1e-1)
        assert result.index == 2

    def test_tn_optimizer_weighted(self):
        """End-to-end near SPD optimization test."""
        # Load original problem from files in matrix form.
        c, w, p = load_example_data(EBLP_SMALL_DATA_DIR)

        optimizer = cpd.eblp.create_optimizer("tn", p, w)
        g, result = optimizer.optimize(c)

        # Relative error |f(G) - C|/|C|.
        assert result.info[1] == pytest.approx(0.1133, 1e-3)
        # Condition number.
        assert result.info[2] == pytest.approx(1703, 1e-1)
        assert result.index == 2

    def test_rco_small_problem(self):
        """End-to-end Cholesky factor optimization test on hard-coded input data."""
        # Load original problem from files in matrix form.
        c, w, p = small_problem_data()

        optimizer = cpd.eblp.create_optimizer("rco", p, w)
        g, result = optimizer.optimize(c)

        # Relative error in h(G) = C.
        assert result.info[1] == pytest.approx(0.274, 1e-2)
        assert result.index == 2

    def test_create_pair_mapping(self):
        for n in (5, 10, 20):
            assert len(cpd._rco.create_pair_mapping(n)) == n * (n + 1) * (n + 2) // 6


def small_problem_data():
    c = np.random.random((4, 4))
    c = c + c.T
    w = np.ones_like(c)
    p = cpd.linalg.ArrayList([
        np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]),
        np.array([
            [0.5, 0.5, 0, 0],
            [0.5, 0.5, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]),
    ])
    return c, w, p


def load_example_data(data_dir):
    """Loads an example problem with 20 schooles from files in matrix form."""
    a_file_name = os.path.join(data_dir, "A.txt")
    c_file_name = os.path.join(data_dir, "C.txt")
    w_file_name = os.path.join(data_dir, "Weights_BxB.txt")
    pi_file_name = os.path.join(data_dir, "Pi_s.txt")
    _, c, w, _, p = cpd.data.load_data(a_file_name, c_file_name, w_file_name, pi_file_name)
    return c, w, p


def solve_least_squares_problem(p, w, c):
    """Solves the EBLP covariance weighted LS problem."""
    w_unraveled = np.diag(cpd.linalg.unravel_lower(w))
    h = w_unraveled @ cpd.moment_functional.matrix_function_to_matrix_sum(p, p)
    c_vector = w_unraveled @ cpd.linalg.unravel_lower(c)
    return np.linalg.lstsq(h, c_vector, rcond=None)[0]
