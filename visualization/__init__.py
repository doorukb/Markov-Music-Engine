"""
everything about plotting and dashboard happens here
plots   : transition matrix heatmaps, stationary distribution charts,
          entropy displays, and order-comparison views
"""
from visualization.plots import (
    plot_metrics_panel,
    plot_stationary_distribution,
    plot_transition_matrix,
    shared_top_chord_indices,
)

__all__ = [
    "plot_transition_matrix",
    "plot_stationary_distribution",
    "plot_metrics_panel",
    "shared_top_chord_indices",
]