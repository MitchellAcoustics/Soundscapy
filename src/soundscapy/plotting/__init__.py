"""
Soundscapy Plotting Module.

This module provides tools for creating circumplex plots for soundscape analysis.
It utilizes the Grammar of Graphics approach with Seaborn Objects API to create
flexible, composable visualizations.

Main components:
- Custom marks for soundscape plots (SoundscapeCircumplex, SoundscapeQuadrantLabels)
- Custom stats for coordinate calculations (SoundscapeCoordinates)
- Function-based API for creating plots (scatter_plot, density_plot)
- CircumplexPlot builder class for backward compatibility

Example usage:

```python
import pandas as pd
import seaborn.objects as so
from soundscapy.plotting import (
    scatter_plot, density_plot, SoundscapeCircumplex, SoundscapeQuadrantLabels
)

# Function-based API (recommended for new code)
scatter_plot(data, x='ISOPleasant', y='ISOEventful')
density_plot(data, x='ISOPleasant', y='ISOEventful', incl_scatter=True)

# Direct use of custom components with so.Plot
plot = (
    so.Plot(data, x='ISOPleasant', y='ISOEventful')
    .add(so.Dots(), color='LocationID')
    .add(SoundscapeCircumplex())
    .add(SoundscapeQuadrantLabels())
)

# Legacy builder API (maintained for backwards compatibility)
(CircumplexPlot(data)
 .add_density(simple=True)
 .add_scatter()
 .add_grid(diagonal_lines=True)
 .add_title("Custom Plot")
 .show())
```
"""

from soundscapy.plotting import likert
from soundscapy.plotting.circumplex_plot import (
    CircumplexPlot,
    add_annotation,
    apply_circumplex_grid,
)
from soundscapy.plotting.marks import (
    SoundscapeCircumplex,
    SoundscapePointAnnotation,
    SoundscapeQuadrantLabels,
)
from soundscapy.plotting.plotting_utils import DEFAULT_XLIM, DEFAULT_YLIM, PlotType
from soundscapy.plotting.soundscape_functions import (
    add_calculated_coords,
    create_circumplex_subplots,
    density_plot,
    joint_plot,
    scatter_plot,
    use_soundscapy_style,
)
from soundscapy.plotting.stats import SoundscapeCoordinates

__all__ = [
    # Public API - function based (recommended)
    "scatter_plot",
    "density_plot",
    "joint_plot",
    "create_circumplex_subplots",
    "add_calculated_coords",
    "use_soundscapy_style",
    # Custom components
    "SoundscapeCircumplex",
    "SoundscapeQuadrantLabels",
    "SoundscapePointAnnotation",
    "SoundscapeCoordinates",
    # Legacy API (maintained for backwards compatibility)
    "CircumplexPlot",
    "add_annotation",
    "apply_circumplex_grid",
    # Utility classes and constants
    "PlotType",
    "DEFAULT_XLIM",
    "DEFAULT_YLIM",
    # Sub-modules
    "likert",
]
