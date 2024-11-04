"""
Soundscapy is a Python library for soundscape analysis and visualisation.
"""

# ruff: noqa: E402
from loguru import logger

# https://loguru.readthedocs.io/en/latest/resources/recipes.html#configuring-loguru-to-be-used-by-a-library-or-an-application
logger.disable("soundscapy")

import importlib.metadata

__version__ = importlib.metadata.version("soundscapy")

from soundscapy.logging import setup_logging
from soundscapy import plotting
from soundscapy.databases import araus, isd, satp
from soundscapy.surveys import processing
from soundscapy.plotting import scatter_plot, density_plot

__all__ = [
    "plotting",
    "araus",
    "isd",
    "satp",
    "processing",
    "scatter_plot",
    "density_plot",
    "setup_logging",
]

try:
    from soundscapy import audio

    __all__.extend(["audio"])
    __all__.extend(audio.__all__)
except ImportError:
    logger.debug("Audio module not available")
