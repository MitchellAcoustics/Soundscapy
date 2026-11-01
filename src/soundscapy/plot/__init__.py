"""Plots of soundscape results on matplotlib Axes.

Every function draws on the ``ax`` it is given, or on a new pyplot figure when ``ax`` is
omitted, returns the Axes and never calls ``show()``. Group colours follow matplotlib's
colour cycle, so a style sheet the user applies sets them; otherwise seaborn's colour-blind
palette is used. matplotlib and seaborn are imported only when a plot function runs.
"""

from soundscapy.plot.circumplex import scatter

__all__ = ["scatter"]
