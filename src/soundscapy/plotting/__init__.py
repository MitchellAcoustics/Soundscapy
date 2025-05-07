"""
Soundscapy Plotting Module.

This module provides tools for creating circumplex plots for soundscape analysis.
It supports various plot types and backends, with a focus on flexibility and ease
of use.
"""

from soundscapy.plotting import likert
from soundscapy.plotting.backends import (
    Backend,
    CircumplexPlot,
    CircumplexPlotParams,
    PlotType,
    StyleOptions,
)
from soundscapy.plotting.iso_plot import ISOPlot
from soundscapy.plotting.plot_functions import (
    # density_plot,
    scatter,
    # density,
    # jointplot,
    scatter_plot,
)

__all__ = [
    "Backend",
    "CircumplexPlot",
    "CircumplexPlotParams",
    "ISOPlot",
    "PlotType",
    "StyleOptions",
    # "density_plot",
    "likert",
    "scatter",
    "scatter_plot",
]
