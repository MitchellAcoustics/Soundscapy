# SPDX-License-Identifier: BSD-3-Clause
# Copyright (C) 2026  Andrew Mitchell

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
from soundscapy.plotting.isoplot import ISOPlot
from soundscapy.plotting.layers import (
    DensityLayer,
    Layer,
    ScatterLayer,
    SimpleDensityLayer,
    SPIDensityLayer,
    SPIScatterLayer,
    SPISimpleLayer,
)
from soundscapy.plotting.likert import paq_likert, paq_radar_plot, stacked_likert
from soundscapy.plotting.plot_functions import (
    create_circumplex_subplots,
    create_iso_subplots,
    density,
    density_plot,
    iso_plot,
    jointplot,
    scatter,
    scatter_plot,
)

__all__ = [
    "Backend",
    "CircumplexPlot",
    "CircumplexPlotParams",
    "DensityLayer",
    "ISOPlot",
    "Layer",
    "PlotType",
    "SPIDensityLayer",
    "SPIScatterLayer",
    "SPISimpleLayer",
    "ScatterLayer",
    "SimpleDensityLayer",
    "StyleOptions",
    "create_circumplex_subplots",
    "create_iso_subplots",
    "density",
    "density_plot",
    "iso_plot",
    "jointplot",
    "likert",
    "paq_likert",
    "paq_radar_plot",
    "scatter",
    "scatter_plot",
    "stacked_likert",
]
