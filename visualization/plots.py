"""
All matplotlib plotting functions
- Render chord transition matrix as an annotated heatmap
- Render stationary distribution as a labeled bar chart
- Display entropy and spectral gap as formatted metric panels
- Render side by side order-1 vs order-2 melody comparisons
- All functions return matplotlib Figure objects (Streamlit-compatible)
"""
from __future__ import annotations
from typing import Sequence
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.figure import Figure
from markov.parser import ChordToken

__all__ = ["plot_transition_matrix"]

_MAX_CHORDS = 20
_LABEL_MAX_LEN = 16

# truncate the label for a chord
def _truncate_label(chord: ChordToken, max_len: int = _LABEL_MAX_LEN) -> str:
    text = str(chord)
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"

# get the top active indices for a transition matrix
def _top_active_indices(transition_matrix: np.ndarray, cap: int = _MAX_CHORDS) -> np.ndarray:
    row_sums = np.asarray(transition_matrix, dtype=np.float64).sum(axis=1)
    n = min(cap, row_sums.shape[0])
    return np.argsort(row_sums)[::-1][:n]

# render a chord transition matrix as an annotated heatmap
# Axes show the top chords by row-sum (outgoing mass), capped at 20, with truncated chord-name labels for readability
def plot_transition_matrix(transition_matrix: np.ndarray, index_to_chord: Sequence[ChordToken], title: str) -> Figure:
    matrix = np.asarray(transition_matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"transition_matrix must be square 2D; got shape {matrix.shape}")
    if len(index_to_chord) != matrix.shape[0]:
        raise ValueError(f"index_to_chord length {len(index_to_chord)} does not match transition matrix size {matrix.shape[0]}")

    indices = _top_active_indices(matrix)
    sub = matrix[np.ix_(indices, indices)]
    labels = [_truncate_label(index_to_chord[i]) for i in indices]

    n = sub.shape[0]
    fig_size = max(6.0, 0.45 * n + 2.5)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.9))
    sns.heatmap(
        sub,
        xticklabels=labels,
        yticklabels=labels,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        vmin=0.0,
        vmax=max(sub.max(), 1e-9),
        square=True,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "P(next | current)"},
        annot_kws={"size": max(6, 10 - n // 4)},
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Next chord")
    ax.set_ylabel("Current chord")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0)
    fig.tight_layout()
    return fig