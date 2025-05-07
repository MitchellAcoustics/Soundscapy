"""Deprecation warnings for v0.7 APIs."""
# ruff: noqa: ANN002, ANN003, ARG001, ARG002, D107, D102, ANN204

import warnings
from enum import Enum


class CircumplexPlot:
    """Alias for v0.7 API to raise deprecation warnings."""

    def __init__(self, *args, **kwargs) -> None:
        warnings.warn(
            "The v0.7 APIs are deprecated. Please update your code accordingly."
            "CircumplexPlot is deprecated. Use ISOPlot instead.",
            DeprecationWarning,
            stacklevel=2,
        )


class CircumplexPlotParams:
    """Alias for v0.7 API to raise deprecation warnings."""

    def __init__(self, *args, **kwargs) -> None:
        warnings.warn(
            "The v0.7 APIs are deprecated. Please update your code accordingly. "
            "CircumplexPlotParams is deprecated. "
            "Use `ISOPlot` and `plotting_types` instead.",
            DeprecationWarning,
            stacklevel=2,
        )


class Backend(Enum):
    """Alias for v0.7 API to raise deprecation warnings."""

    SEABORN = "seaborn"
    PLOTLY = "plotly"

    def __call__(self, *args, **kwargs) -> None:
        warnings.warn(
            "The v0.7 APIs are deprecated. Please update your code accordingly."
            "Backend is deprecated. Use ISOPlotBackend instead.",
            DeprecationWarning,
            stacklevel=2,
        )


class PlotType:
    """Alias for v0.7 API to raise deprecation warnings."""

    SCATTER = "scatter"
    DENSITY = "density"
    JOINTPLOT = "jointplot"
    SIMPLE_DENSITY = "simple_density"

    def __call__(self, *args, **kwargs) -> None:
        msg = (
            "The v0.7 APIs are deprecated. Please update your code accordingly."
            "PlotType is deprecated. Add plot types as layers in ISOPlot instead."
        )
        raise DeprecationWarning(
            msg,
        )


class StyleOptions:
    """Alias for v0.7 API to raise deprecation warnings."""

    def __init__(self, *args, **kwargs) -> None:
        msg = (
            "The v0.7 APIs are deprecated. Please update your code accordingly."
            "StyleOptions is deprecated. Use ISOPlot StylingParamsTypes instead."
        )
        raise DeprecationWarning(
            msg,
        )
