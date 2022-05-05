"""Load problem data from files or generate random data."""
from typing import Tuple

import numpy as np
import pandas as pd

import cpd.linalg
from cpd.linalg import ArrayList


def load_data(a_file_name: str, c_file_name: str, w_file_name: str, pi_file_name: str) -> \
        Tuple[np.ndarray, np.ndarray, np.ndarray, ArrayList, ArrayList]:
    """
    Loads H*g = c least-squares problem data from files. Includes data for the G-problem as well as the
    G^*-problem.

    Args:
        a_file_name: A (projection matrix) file name.
        c_file_name: RHS matrix C file name.
        w_file_name: LS weight matrix C file name.
        pi_file_name: school matrix lists file name.

    Returns:
        a: projection matrix. shape=(n, n - 1).
        c: RHS matrix. shape=(m, m).
        w: weight matrix. shape=(m, m).
        r: G functional matrix list. Each matrix in the list is m x n.
        p: G^* functional matrix list. Each matrix in the list is m x n.
    """
    # Projection B->B-1 dimensions.
    a = np.loadtxt(a_file_name)
    # Right-hand-side matrix.
    c = pd.read_csv(c_file_name, delimiter=" ").values
    # Weight matrix. The weights go inside the least-squares square terms, so we take the sqrt. These
    # correspond to W[i,j] ~ 1/stddev(C[i,j]) = 1/sqrt(#schools).
    w = np.loadtxt(w_file_name) ** 0.5
    # School matrices Pi_s, s = 1..S.
    r = data_frame_to_array_list(pd.read_csv(pi_file_name, delimiter=" "))
    # Remove zero terms.
    r = cpd.linalg.ArrayList([ri for ri in r if np.linalg.norm(ri) != 0])
    p = r @ a
    # Full matrix terms P=(I - PI)*A (as long as we're using the form G = A*G^*A').
    return a, c, w, r, p


def data_frame_to_array_list(df: pd.DataFrame, subtract_diagonal_sums: bool = True) -> ArrayList:
    """
    Converts a school data DataFrame into a list of school matrices.
    Args:
        df: School data frame.
        subtract_diagonal_sums: iff True, subtracts the diagonal sums, i.e. ((diagonal(sum(Pi_s,axis=1)) - Pi_s)).

    Returns:
        r  = [r[s]]_s where p[s] = (diagonal(sum(Pi_s,axis=1)) - Pi_s), Pi_s = school s's sub-DataFrame.
        subtract_diagonal_sums: if True, returns (diagonal(sum(Pi_s,axis=1)) - Pi_s), else Pi_s.
    """
    # Number of blocks.
    school_id = df["school"]
    # O(n) operation below, since we know the "school" column is already sorted.
    # O(n^2) where n = #schools, slow.
    # p_school = [df[df["school"] == school][[str(i) for i in range(1, b + 1)]].values
    #             for school in df["school"].unique()]
    endpoints = np.array([i for i, (first, second) in enumerate(zip(school_id, school_id[1:])) if first != second])
    p_school = np.split(df[df.columns[~df.columns.isin(["school"])]].values, endpoints + 1)
    # Convert everything to float to avoid string/float pandas read issues.
    p_school = np.array([pi.astype(float) for pi in p_school])
    # Full matrix terms R=(I - PI) (working with G as an unknown).
    if subtract_diagonal_sums:
        return cpd.linalg.ArrayList([(np.diag((pi.sum(axis=1) != 0).astype(int)) - pi) for pi in p_school])
    else:
        return cpd.linalg.ArrayList(p_school)


def data_frame_to_p_list(df: pd.DataFrame, a: np.ndarray) -> ArrayList:
    """
    Converts a school data DataFrame into a list of school matrices, times the B->B-1 projection matrix.
    for R-code calling.

    Args:
        df: School data frame.
        a: projection matrix.

    Returns:
        p = [[p_s]]_s where p[s] = (diagonal(sum(Pi_s,axis=1)) - Pi_s) * A , Pi_s = school s's sub-DataFrame.
    """
    return data_frame_to_array_list(df) @ a
