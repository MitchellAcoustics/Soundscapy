"""
Soundscapy Plotting Module

This module provides tools for creating circumplex plots for soundscape analysis.
It supports various plot types and backends, with a focus on flexibility and ease of use.

Main components:
- CircumplexPlot: Main class for creating customizable circumplex plots
- scatter_plot: Function to quickly create scatter plots
- density_plot: Function to quickly create density plots
- create_circumplex_subplots: Function to create multiple circumplex plots in subplots
- Backend: Enum for selecting the plotting backend (Seaborn or Plotly)
- PlotType: Enum for specifying the type of plot to create

Example usage:
    from soundscapy.plotting import scatter_plot, density_plot, Backend, PlotType

    # Create a scatter plot using Seaborn backend
    scatter_plot(data, x='ISOPleasant', y='ISOEventful', backend=Backend.SEABORN)

    # Create a density plot using Plotly backend
    density_plot(data, x='ISOPleasant', y='ISOEventful', backend=Backend.PLOTLY)
"""

from . import likert
from .circumplex_plot import CircumplexPlot, CircumplexPlotParams
from .plot_functions import create_circumplex_subplots, density_plot, scatter_plot
from .plotting_utils import Backend, PlotType
from .stylers import StyleOptions

__all__ = [
    "CircumplexPlot",
    "CircumplexPlotParams",
    "scatter_plot",
    "density_plot",
    "create_circumplex_subplots",
    "Backend",
    "PlotType",
    "StyleOptions",
    "likert",
]
