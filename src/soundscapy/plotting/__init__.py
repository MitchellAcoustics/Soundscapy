"""
Soundscapy Plotting Module

This module provides tools for creating circumplex plots for soundscape analysis.
It utilizes the Grammar of Graphics approach with Seaborn Objects API to create
flexible, composable visualizations.

Main components:
- CircumplexPlot: Builder class for creating customizable circumplex plots
- scatter_plot: Function to quickly create scatter plots
- density_plot: Function to quickly create density plots
- create_circumplex_subplots: Function to create multiple circumplex plots in subplots

Example usage:
    from soundscapy.plotting import scatter_plot, density_plot, CircumplexPlot

    # Create a basic scatter plot
    scatter_plot(data, x='ISOPleasant', y='ISOEventful')

    # Create a density plot with scatter points
    density_plot(data, x='ISOPleasant', y='ISOEventful', incl_scatter=True)

    # Build a customized plot layer by layer
    (CircumplexPlot(data)
     .add_density(simple=True)
     .add_scatter()
     .add_grid(diagonal_lines=True)
     .add_title("Custom Plot")
     .show())
"""

from . import likert
from .circumplex_plot import CircumplexPlot, add_annotation, apply_circumplex_grid
from .plot_functions import (
    create_circumplex_subplots,
    density_plot,
    joint_plot,
    scatter_plot,
)
from .plotting_utils import PlotType

__all__ = [
    "CircumplexPlot",
    "apply_circumplex_grid",
    "add_annotation",
    "scatter_plot",
    "density_plot",
    "joint_plot",
    "create_circumplex_subplots",
    "PlotType",
    "likert",
]
