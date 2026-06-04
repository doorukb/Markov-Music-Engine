from __future__ import annotations
import numpy as np
from config import SMOOTHING_ALPHA
__all__ = ["laplace_smooth"]

# laplace smoothing a count matrix and return a row stochastic matrix
def laplace_smooth(counts: np.ndarray, alpha: float = SMOOTHING_ALPHA) -> np.ndarray:
    if counts.ndim != 2:
        raise ValueError(f"counts must be a 2D array; got shape {counts.shape}")
    if alpha <= 0:
        raise ValueError(f"alpha must be positive for Laplace smoothing; got {alpha}")

    smoothed = counts.astype(np.float64, copy=True) + alpha
    row_sums = smoothed.sum(axis=1, keepdims=True)
    return smoothed / row_sums