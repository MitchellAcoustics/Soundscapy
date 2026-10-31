"""Plots of soundscape results on matplotlib Axes.

Every function draws on the ``ax`` it is given, or on a new pyplot-free
``matplotlib.figure.Figure`` when ``ax`` is omitted, returns the Axes and never calls
``show()``. matplotlib is imported only when a plot function runs.
"""

from soundscapy.plot.circumplex import scatter

__all__ = ["scatter"]
