"""
Refactored plotting module for soundscapy.

This module provides a refactored implementation of the plotting functionality
in soundscapy, using composition instead of inheritance and with a cleaner
architecture. The main entry point is the ISOPlot class.

Examples
--------
Create a simple scatter plot:

>>> import pandas as pd
>>> import numpy as np
>>> from soundscapy.plotting.new import ISOPlot
>>> # Create some sample data
>>> rng = np.random.default_rng(42)
>>> data = pd.DataFrame(
...     rng.multivariate_normal([0.2, 0.15], [[0.1, 0], [0, 0.2]], 100),
...     columns=['ISOPleasant', 'ISOEventful']
... )
>>> # Create a plot with multiple layers
>>> plot = (ISOPlot(data=data)
...         .create_subplots()
...         .add_scatter()
...         .add_simple_density(fill=False)
...         .apply_styling(
...             xlim=(-1, 1),
...             ylim=(-1, 1),
...             primary_lines=True
...         ))
>>> isinstance(plot, ISOPlot)
True
>>> # plot.show()  # Uncomment to display the plot

"""

from soundscapy.plotting.new.iso_plot import ISOPlot
from soundscapy.plotting.new.layer import (
    DensityLayer,
    Layer,
    ScatterLayer,
    SimpleDensityLayer,
    SPISimpleLayer,
)
from soundscapy.plotting.new.parameter_models import (
    BaseParams,
    DensityParams,
    ScatterParams,
    SimpleDensityParams,
    SPISeabornParams,
    SPISimpleDensityParams,
    StyleParams,
    SubplotsParams,
)
from soundscapy.plotting.new.plot_context import PlotContext

__all__ = [
    # Parameter models
    "BaseParams",
    "DensityLayer",
    "DensityParams",
    # Main plotting class
    "ISOPlot",
    # Layer classes
    "Layer",
    # Context
    "PlotContext",
    "SPISeabornParams",
    "SPISimpleDensityParams",
    "SPISimpleLayer",
    "ScatterLayer",
    "ScatterParams",
    "SimpleDensityLayer",
    "SimpleDensityParams",
    "StyleParams",
    "SubplotsParams",
]
