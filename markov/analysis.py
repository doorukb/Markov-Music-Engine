from __future__ import annotations
import math
from typing import Any, Sequence
import numpy as np
from config import CONVERGENCE_THRESHOLD, MAX_ITERATIONS
from markov.parser import ChordToken

_EIGENVALUE_UNIT_TOL = 1e-6

__all__ = [
    "stationary_power_iteration",
    "stationary_eigenvector",
    "chain_entropy",
    "spectral_gap",
    "mixing_time_estimate",
    "summarise",
]

# ensure the transition matrix is square and has at least min_size rows and columns
def _as_square_transition_matrix(transition_matrix: np.ndarray, *, min_size: int = 1) -> np.ndarray:
    p = np.asarray(transition_matrix, dtype=np.float64)
    if p.ndim != 2 or p.shape[0] != p.shape[1]:
        raise ValueError(f"transition_matrix must be square 2D; got shape {p.shape}")
    if p.shape[0] < min_size:
        raise ValueError(f"transition_matrix must be at least {min_size}×{min_size}; got {p.shape[0]}×{p.shape[0]}")
    return p

# estimate the stationary distribution π with power iteration
# starts from a uniform distribution and iterates until the L1 change in π is below tol
def stationary_power_iteration(transition_matrix: np.ndarray, tol: float = CONVERGENCE_THRESHOLD, max_iter: int = MAX_ITERATIONS) -> np.ndarray:
    p = _as_square_transition_matrix(transition_matrix)
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
    p = _as_square_transition_matrix(transition_matrix)

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

# compute the stationary-weighted average row Shannon entropy (bits)
# for each row, H(row) = -sum(p * log2(p)) over entries with p > 0. The result is sum_i pi_i * H(row_i) where pi is the stationary distribution.
# a uniform transition matrix (every row 1/n) attains the maximum entropy log2(n) per row and therefore the maximum chain entropy for a given size.
def chain_entropy(transition_matrix: np.ndarray) -> float:
    p = _as_square_transition_matrix(transition_matrix)
    pi = stationary_power_iteration(p)

    with np.errstate(divide="ignore", invalid="ignore"):
        row_entropy = -np.sum(np.where(p > 0, p * np.log2(p), 0.0), axis=1)
    # return the stationary-weighted average row Shannon entropy, in bits
    return float(pi @ row_entropy)

# compute the spectral gap
def spectral_gap(transition_matrix: np.ndarray) -> float:
    p = _as_square_transition_matrix(transition_matrix, min_size=2)
    eigenvalues = np.linalg.eigvals(p)
    magnitudes = np.sort(np.abs(eigenvalues))[::-1]
    lambda2 = magnitudes[1]
    return float(1.0 - lambda2)

# compute the rough mixing-time estimate in steps
def mixing_time_estimate(transition_matrix: np.ndarray) -> int:
    gap = spectral_gap(transition_matrix)
    if gap <= 0:
        raise ValueError(f"spectral gap must be positive for mixing time estimate; got {gap}")
    return math.ceil(1.0 / gap)

# dashboard-ready summary of a chord transition matrix
def summarise(transition_matrix: np.ndarray, index_to_chord: Sequence[ChordToken]) -> dict[str, Any]:
    p = _as_square_transition_matrix(transition_matrix)
    if len(index_to_chord) != p.shape[0]:
        raise ValueError(f"index_to_chord length {len(index_to_chord)} does not match transition matrix size {p.shape[0]}")

    pi = stationary_power_iteration(p)
    entropy_bits = chain_entropy(p)
    mixing_time_steps = mixing_time_estimate(p)

    dominant_index = int(np.argmax(pi))
    dominant_chord = index_to_chord[dominant_index]
    dominant_chord_pct = float(pi[dominant_index] * 100.0)

    stationary_distribution = {index_to_chord[i]: float(pi[i]) for i in range(p.shape[0])}

    # return the summary
    return {
        "dominant_chord": dominant_chord,
        "dominant_chord_pct": dominant_chord_pct,
        "entropy_bits": entropy_bits,
        "mixing_time_steps": mixing_time_steps,
        "stationary_distribution": stationary_distribution,
    }