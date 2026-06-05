"""
everything about plotting and dashboard happens here
plots   : transition matrix heatmaps, stationary distribution charts,
          entropy displays, and order-comparison views
"""
from visualization.plots import (
    plot_metrics_panel,
    plot_metrics_panels,
    plot_stationary_distribution,
    plot_transition_matrix,
)

__all__ = [
    "plot_transition_matrix",
    "plot_stationary_distribution",
    "plot_metrics_panel",
    "plot_metrics_panels",
]