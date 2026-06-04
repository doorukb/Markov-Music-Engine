from __future__ import annotations
import numpy as np
from config import CONVERGENCE_THRESHOLD, MAX_ITERATIONS

_EIGENVALUE_UNIT_TOL = 1e-6

__all__ = ["stationary_power_iteration", "stationary_eigenvector"]

# estimate the stationary distribution π with power iteration
# starts from a uniform distribution and iterates until the L1 change in π is below tol
def stationary_power_iteration(transition_matrix: np.ndarray, tol: float = CONVERGENCE_THRESHOLD, max_iter: int = MAX_ITERATIONS) -> np.ndarray:
    p = np.asarray(transition_matrix, dtype=np.float64)
    if p.ndim != 2 or p.shape[0] != p.shape[1]:
        raise ValueError(f"transition_matrix must be square 2D; got shape {p.shape}")

    n = p.shape[0]
    pi = np.full(n, 1.0 / n, dtype=np.float64)

    for _ in range(1, max_iter + 1):
        pi_new = pi @ p
        if np.linalg.norm(pi_new - pi, ord=1) < tol:
            pi = pi_new / pi_new.sum()
            return pi
        pi = pi_new
    raise RuntimeError(f"stationary distribution did not converge within {max_iter} iterations (L1 tolerance {tol}).")

# estimate the stationary distribution π with eigenvector decomposition
# solves π = π P via np.linalg.eig on P.T. Raises ValueError if no eigenvalue lies within 1e-6 of 1.0.
# compare with stationary_power_iteration as an independent estimate of π.
def stationary_eigenvector(transition_matrix: np.ndarray) -> np.ndarray:
    p = np.asarray(transition_matrix, dtype=np.float64)
    if p.ndim != 2 or p.shape[0] != p.shape[1]:
        raise ValueError(f"transition_matrix must be square 2D; got shape {p.shape}")

    eigenvalues, eigenvectors = np.linalg.eig(p.T)
    idx = int(np.argmin(np.abs(eigenvalues - 1.0)))
    if np.abs(eigenvalues[idx] - 1.0) > _EIGENVALUE_UNIT_TOL:
        raise ValueError(f"no eigenvalue within 1e-6 of 1.0 found; closest eigenvalue is {eigenvalues[idx]}")

    pi = np.real(eigenvectors[:, idx])
    if pi.sum() <= 0:
        pi = -pi
    pi = np.maximum(pi, 0.0)
    total = pi.sum()
    if total <= 0:
        raise ValueError("stationary eigenvector has no positive mass after normalization")
    return pi / total
