"""
Soundscapy Plotting Module.

This module provides tools for creating circumplex plots for soundscape analysis.
It supports various plot types and backends, with a focus on flexibility and ease
of use.
"""

from soundscapy.plotting import likert
from soundscapy.plotting.circumplex_plot import CircumplexPlot, CircumplexPlotParams
from soundscapy.plotting.plot_functions import (
    create_circumplex_subplots,
    density_plot,
    scatter_plot,
)
from soundscapy.plotting.plotting_utils import Backend, PlotType
from soundscapy.plotting.stylers import StyleOptions

__all__ = [
    "Backend",
    "CircumplexPlot",
    "CircumplexPlotParams",
    "PlotType",
    "StyleOptions",
    "create_circumplex_subplots",
    "density_plot",
    "likert",
    "scatter_plot",
]
