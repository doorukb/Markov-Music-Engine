"""
Plotting helpers for the Streamlit dashboard and CLI analysis views.

Matplotlib functions return Figure objects for st.pyplot(). Streamlit metric
panels render directly via st.metric().
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import streamlit as st
from matplotlib.figure import Figure

from markov.parser import ChordToken

__all__ = [
    "plot_transition_matrix",
    "plot_stationary_distribution",
    "plot_metrics_panel",
    "plot_metrics_panels",
]

SummaryDict = dict[str, Any]

_MAX_CHORDS = 20
_MAX_STATIONARY_BARS = 15
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

# render a dominant chord, chain entropy, and mixing time as a three-column st.metric row
# When baseline is provided (order-2 vs order-1 comparison), entropy and mixing time include a delta versus the baseline summary
def plot_metrics_panel(summary: SummaryDict, order: int, *, baseline: SummaryDict | None = None) -> None:
    entropy_delta: float | None = None
    mixing_delta: int | None = None
    if baseline is not None:
        entropy_delta = float(summary["entropy_bits"]) - float(baseline["entropy_bits"])
        mixing_delta = int(summary["mixing_time_steps"]) - int(baseline["mixing_time_steps"])

    col_dominant, col_entropy, col_mixing = st.columns(3)
    with col_dominant:
        st.metric("Dominant chord", str(summary["dominant_chord"]), f"{float(summary['dominant_chord_pct']):.1f}% stationary mass")
    with col_entropy:
        st.metric("Chain entropy (bits)", f"{float(summary['entropy_bits']):.3f}", delta=f"{entropy_delta:+.3f}" if entropy_delta is not None else None, delta_color="normal")
    with col_mixing:
        st.metric("Mixing time (steps)", str(int(summary["mixing_time_steps"])), delta=f"{mixing_delta:+d}" if mixing_delta is not None else None, delta_color="normal")

# render harmony metrics for one or more melody orders
# Displays one or more three-column metric rows, one per order, with order-2 deltas vs order-1 if provided
# Single-order : one three-column row. Comparison mode: one row per order; order 2 shows entropy and mixing-time deltas relative to order 1.
def plot_metrics_panels(summaries_by_order: Mapping[int, SummaryDict]) -> None:
    if not summaries_by_order:
        raise ValueError("summaries_by_order must not be empty")

    orders = sorted(summaries_by_order)
    compare = len(orders) > 1
    baseline = summaries_by_order.get(1) if compare and 1 in summaries_by_order else None

    for order in orders:
        if compare:
            st.markdown(f"**Order {order}**")
        plot_metrics_panel(
            summaries_by_order[order],
            order,
            baseline=baseline if compare and order != 1 and baseline is not None else None,
        )

# render the stationary distribution as a horizontal bar chart
# Shows the top chords by long-run probability (capped at 15), sorted descending, with percentage labels on each bar
def plot_stationary_distribution(stationary_dict: Mapping[ChordToken, float], title: str) -> Figure:
    if not stationary_dict:
        raise ValueError("stationary_dict must not be empty")

    ranked = sorted(stationary_dict.items(), key=lambda item: item[1], reverse=True)
    ranked = ranked[:_MAX_STATIONARY_BARS]
    labels = [_truncate_label(chord) for chord, _ in ranked]
    probs = [float(p) for _, p in ranked]

    n = len(labels)
    fig_height = max(4.0, 0.35 * n + 1.5)
    fig, ax = plt.subplots(figsize=(8.0, fig_height))

    y_pos = np.arange(n)
    bars = ax.barh(y_pos, probs, color=sns.color_palette("Blues", n_colors=n + 2)[1], edgecolor="white")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0.0, max(probs) * 1.15 if probs else 1.0)
    ax.set_xlabel("Stationary probability")
    ax.set_title(title)

    for bar, prob in zip(bars, probs):
        width = bar.get_width()
        ax.text(
            width + max(probs) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{prob * 100:.1f}%",
            va="center",
            ha="left",
            fontsize=9,
        )

    fig.tight_layout()
    return fig