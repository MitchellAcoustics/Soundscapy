# SPDX-License-Identifier: BSD-3-Clause
# Copyright (C) 2026  Andrew Mitchell

"""
Soundscapy Plotting Module.

This module provides tools for creating circumplex plots for soundscape analysis.
It supports various plot types and backends, with a focus on flexibility and ease
of use.
"""

import sys

from soundscapy._deprecated import moved_module
from soundscapy.plotting import likert
from soundscapy.plotting._warnings import ExperimentalWarning
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

# ``iso_plot`` is now a function, so a stub module can't keep the old
# ``from soundscapy.plotting.iso_plot import ISOPlot`` path alive; use sys.modules.
sys.modules[f"{__name__}.iso_plot"] = moved_module(
    f"{__name__}.iso_plot",
    {
        "soundscapy.plotting": ["ISOPlot", "ExperimentalWarning"],
    },
)

__all__ = [
    "Backend",
    "CircumplexPlot",
    "CircumplexPlotParams",
    "DensityLayer",
    "ExperimentalWarning",
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
