"""
Soundscapy is a Python library for soundscape analysis and visualisation.
"""

# ruff: noqa: E402
from typing import Any
from loguru import logger

# https://loguru.readthedocs.io/en/latest/resources/recipes.html#configuring-loguru-to-be-used-by-a-library-or-an-application
logger.disable("soundscapy")

import importlib.metadata

__version__ = importlib.metadata.version("soundscapy")

from soundscapy._optionals import import_optional

# Always available core modules
from soundscapy import surveys
from soundscapy import databases
from soundscapy import plotting
from soundscapy.logging import setup_logging
from soundscapy.databases import araus, isd, satp
from soundscapy.surveys import processing
from soundscapy.plotting import scatter_plot, density_plot

__all__ = [
    # Core modules
    "surveys",
    "databases",
    "plotting",
    "araus",
    "isd",
    "satp",
    "processing",
    "scatter_plot",
    "density_plot",
    "setup_logging",
    # Optional modules listed explicitly for IDE/typing support
    "Binaural",
    "AudioAnalysis",
    "AnalysisSettings",
    "ConfigManager",
    "process_all_metrics",
    "prep_multiindex_df",
    "add_results",
    "parallel_process",
]


def __getattr__(name: str) -> Any:
    """Lazy import handling for optional components."""
    return import_optional(name)
