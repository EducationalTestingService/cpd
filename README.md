# CPD: Algorithms for Conditioned Symmetric Positive Definite Matrix Estimation Under Constraints
This code implements several methods for solving the problem of finding a symmetric positive definite (SPD) matrix that
minimizes an objective functional under the constraint of a bounded condition number. This work was motivated by a specific application: estimating the covariance matrix of the Empirical Best Linear
Prediction (EBLP) model. Specific API calls are provided for this use case, but also provided for the general
"near-PD" case.

**CPD**: the problem of finding a symmetric positive definite (SPD) matrix that satisfies a condition (e.g., method of
moments equations for a covariance matrix satisfied to a given tolerance) and is well-conditioned. Typically, one cannot
obtain both, and we need to find a point of optimal tradeoff between accuracy (small constraint residual) and
stability (small condition number).

**Near PD**: a special case of CPD where the functional is the Frobenius norm between a given matrix and the desired SPD
matrix with bounded condition number. That is, the task is to find a conditioned SPD matrix nearest to a matrix. The
regularization parameter is either the condition number bound or a trade-off parameter as in the CPD formulation.

**RCO (Regularized Cholesky Optimization)**: an algorithm for solving CPD that minimizes a regularized functional of the
matrix that balances minimizing constraint satisfaction plus the (approximate) condition number of the matrix. The
minimization is performed for the entries of the Cholesky factor. The algorithm:
* Using a continuation method to compute the entire ROC curve fast.
* Each continuation step optimization is solved by Newton-CG.
* The optimal regularization parameter is picked around the knee of the curve (the default is by a given residual
  tolerance).

** TN (Tanaka-Nakata): implementation of the paper Tanaka, M., Nakata, N., Positive definite matrix approximation with
condition number constraint", Opt. Let. 8(3), 2014 for finding a bounded condition number nearest SPD matrix.
Uses a new O(n) dynamic programming algorithm to solve the eigenvalue optimization problem.

## Installation
- Install conda.
- Create a conda environment from the attached environment.yml:
  `conda env create -f environment.yml.`
- If you'd also like to develop/run unit tests, use the full environment file instead:
  `conda env create -f environment.full.yml.`
- Add `src` to your PYTHONPATH.

### Full Environment for Testing & Development
* Complete the installation steps above, but create a conda environment using
`conda env create -f environment_full.yml.` instead of `environment.yml`.
* Install nodejs (e.g., on mac, `brew install nodejs`).
* Run `jupyter labextension install @jupyter-widgets/jupyterlab-manager plotlywidget`

## Testing

The project contains Pytest unit tests for the main modules. To run all tests, run `pytest test`.

## Examples

### Near PD
Given an n x n matrix ```a```, call ```near_pd()``` to get a conditioned matrix near ```a``` in
the Frobeius norm ```norm(b - a)/norm(a)```.
```python
  b, info = cpd.near_pd.near_pd(a, "rco")      # Regularized Cholesky Optimization
  b, info = cpd.near_pd.near_pd(a, "tn")       # Tanaka-Nakata
  b, info = cpd.near_pd.near_pd(a, "higham")   # Higham near-PD with diagonal perturbation.
  b, info = cpd.near_pd.near_pd(a, "tn", leeway_factor=1.05)       # Stay within 5% Frobenius norm error
```

### EBLP

```python
import cpd
import os

# Load data from files; or crate p, w, c directly.
DATA_DIR = "/path/to/input_files"
a_file_name = os.path.join(DATA_DIR, "A.txt")
c_file_name = os.path.join(DATA_DIR, "C.txt")
w_file_name = os.path.join(DATA_DIR, "Weights_BxB.txt")
pi_file_name = os.path.join(DATA_DIR, "Pi_s.txt")
_, c, w, _, p = cpd.data.load_data(a_file_name, c_file_name, w_file_name, pi_file_name)

# Create optimizer; pass in an ArrayList of arrays p and an array c if not loading the
# data from files as above. Only needs to be called once per RHS term list p.
# Note: p is an cpd.linalg.ArrayList, not a list of matrices.
optimizer = cpd.eblp.create_optimizer("rco", p, w)
# Now call the optimizer fo ra particular LHS matrix c.
g, info = optimizer.optimize(c)

# Objective function value^(1/2) = Relative error in satisfying the moment equations f(G) = C.
print(info[1] ** 0.5)  # 0.0855678
```

To use the T-N method instead, use
```python
optimizer = cpd.eblp.create_optimizer("tn", p, w)
g, info = optimizer.optimize(c)
```

### General Constraints

Inputs:

* Constraint functional.
* Gradient of the constraint functional (for BFGS minimization).
* Solution quality metric (e.g., the matrix condition number).

Output:

* Optimal covariance matrix.

## Contents

- `data` - test data.
- `notebooks`: Juypter notebooks.
- `src`: source code.
- `test`: unit tests.

## References

* Dan, Katherine, JR, Improving Accuracy and Stability of Aggregate Student Growth Measures Using Empirical Best Linear
  Prediction (JEBStats, in review).

## TODO

* Test another data set.
* Integrate into R simulation - Katherine.
* Write research memo/paper.
