"""Test utilities."""
import os
import pathlib

import numpy as np
import pytest
import scipy.optimize
import scipy.sparse
from numpy.linalg import norm

import cpd

DATA_DIR = os.path.join(os.path.dirname(pathlib.Path(__file__).parent.parent.absolute()), "data")
EBLP_SMALL_DATA_DIR = os.path.join(DATA_DIR, "eblp_small")


def create_random_spd(n, noise: float = 1e-2):
    """Generates an SPD matrix and its Cholesky factor. Explicitly calculate Frobenius error."""
    np.random.seed(0)
    g = cpd.linalg.random_spsd(n, 2) + 0.001 * np.eye(n)
    g /= norm(g)
    b = g + noise * np.eye(n)
    frobenius_error = norm(b - g) / norm(g)
    assert frobenius_error == pytest.approx(noise * np.sqrt(n))
    l = scipy.linalg.cholesky(b, lower=True)
    l_vector = cpd.linalg.unravel_lower(l)
    f_original = pytest.approx(frobenius_error)
    return g, l_vector, f_original


def small_problem_data(weighted: bool = False):
    c = np.random.random((5, 5))
    c = c + c.T
    if weighted:
        w = np.random.random(c.shape)
        w = w + w.T
    else:
        w = np.ones_like(c)
    p = cpd.linalg.ArrayList([
        np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 0, 0, 0],
        ]),
        np.array([
            [0.5, 0.5, 0, 0],
            [0.5, 0.5, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 0, 0, 0],
        ]),
    ])
    return c, w, p
