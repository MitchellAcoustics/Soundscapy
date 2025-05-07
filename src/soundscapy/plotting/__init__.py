"""
Soundscapy Plotting Module.

This module provides tools for creating circumplex plots for soundscape analysis.
It supports various plot types and backends, with a focus on flexibility and ease
of use.
"""

import warnings

from soundscapy.plotting import likert
from soundscapy.plotting.iso_plot import ISOPlot
# from soundscapy.plotting.plot_functions import scatter, density, jointplot


# Deprecation warnings for v0.7 APIs:
class CircumplexPlot:
    """Alias for v0.7 API to raise deprecation warnings."""

    def __init__(self) -> None:  # noqa: D107
        warnings.warn(
            "The v0.7 APIs are deprecated. Please update your code accordingly."
            "CircumplexPlot is deprecated. Use ISOPlot instead.",
            DeprecationWarning,
            stacklevel=2,
        )


class CircumplexPlotParams:
    """Alias for v0.7 API to raise deprecation warnings."""

    def __init__(self) -> None:  # noqa: D107
        warnings.warn(
            "The v0.7 APIs are deprecated. Please update your code accordingly. "
            "CircumplexPlotParams is deprecated. "
            "Use `ISOPlot` and `plotting_types` instead.",
            DeprecationWarning,
            stacklevel=2,
        )


class Backend:
    """Alias for v0.7 API to raise deprecation warnings."""

    def __init__(self) -> None:  # noqa: D107
        warnings.warn(
            "The v0.7 APIs are deprecated. Please update your code accordingly."
            "Backend is deprecated. Use ISOPlotBackend instead.",
            DeprecationWarning,
            stacklevel=2,
        )


class PlotType:
    """Alias for v0.7 API to raise deprecation warnings."""

    def __init__(self) -> None:  # noqa: D107
        warnings.warn(
            "The v0.7 APIs are deprecated. Please update your code accordingly."
            "PlotType is deprecated. Add plot types as layers in ISOPlot instead.",
            DeprecationWarning,
            stacklevel=2,
        )


class StyleOptions:
    """Alias for v0.7 API to raise deprecation warnings."""

    def __init__(self) -> None:  # noqa: D107
        warnings.warn(
            "The v0.7 APIs are deprecated. Please update your code accordingly."
            "StyleOptions is deprecated. Use ISOPlot StylingParamsTypes instead.",
            DeprecationWarning,
            stacklevel=2,
        )


def deprecated_function() -> None:
    """Alias for v0.7 API to raise deprecation warnings."""
    warnings.warn(
        "The v0.7 APIs are deprecated. Please update your code accordingly."
        "This function is deprecated. Use ISOPlot API instead.",
        DeprecationWarning,
        stacklevel=2,
    )


__all__ = [
    "Backend",
    "CircumplexPlot",
    "CircumplexPlotParams",
    "ISOPlot",
    "PlotType",
    "StyleOptions",
    "likert",
]
