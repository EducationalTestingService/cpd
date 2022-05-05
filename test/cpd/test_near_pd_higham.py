"""Higham near PD algorithm unit tests."""
import os

import numpy as np
import pytest

import cpd
from util import DATA_DIR

HIGHAM_DATA_DIR = os.path.join(DATA_DIR, "higham")


class TestNearPdHigham:
    @pytest.mark.parametrize("eig_tol", (1e-6, 1e-5, 1e-4))
    def test_near_pd_is_spd(self, eig_tol):
        for i in range(10):
            for j in range(2, 50):
                a = np.random.randn(j, j)
                b = cpd.near_pd.higham(a, eig_tol=eig_tol)
                assert cpd.linalg.is_spd(b, eig_tol=eig_tol)

    def test_near_pd_reproduces_higham_paper_result(self):
        a = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        b_expected = np.array([
            [0.17678, 0.25, 0.17678],
            [0.25, 0.35355, 0.25],
            [0.17678, 0.25, 0.17678],
        ])

        b = cpd.near_pd.higham(a, eig_tol=1e-10)
        assert np.linalg.norm(b - b_expected) < 1e-4 * np.linalg.norm(b)
