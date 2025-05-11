"""
Soundscapy Plotting Module.

This module provides tools for creating circumplex plots for soundscape analysis.
It supports various plot types and backends, with a focus on flexibility and ease
of use.
"""

from soundscapy.plotting import likert
from soundscapy.plotting.backends_deprecated import (
    Backend,
    CircumplexPlot,
    CircumplexPlotParams,
    PlotType,
    StyleOptions,
)
from soundscapy.plotting.iso_plot import ISOPlot
from soundscapy.plotting.plot_functions import (
    create_circumplex_subplots,
    density,
    density_plot,
    jointplot,
    scatter,
    scatter_plot,
)

__all__ = [
    "Backend",
    "CircumplexPlot",
    "CircumplexPlotParams",
    "ISOPlot",
    "PlotType",
    "StyleOptions",
    "create_circumplex_subplots",
    "density",
    "density_plot",
    "jointplot",
    "likert",
    "scatter",
    "scatter_plot",
]
