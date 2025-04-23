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

# Always available core modules
from soundscapy import surveys
from soundscapy import databases
from soundscapy import plotting
from soundscapy.logging import (
    setup_logging,
    enable_debug,
    disable_logging,
    get_logger,
)
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
    # Logging functions
    "setup_logging",
    "enable_debug",
    "disable_logging",
    "get_logger",
]

# Try to import optional audio module
try:
    from soundscapy import audio
    from soundscapy.audio import (
        Binaural, AudioAnalysis, AnalysisSettings, ConfigManager,
        process_all_metrics, prep_multiindex_df, add_results, parallel_process,
    )
    __all__.extend([
        "audio", "Binaural", "AudioAnalysis", "AnalysisSettings", 
        "ConfigManager", "process_all_metrics", "prep_multiindex_df",
        "add_results", "parallel_process",
    ])
except ImportError:
    # Audio module not available - this is expected if dependencies aren't installed
    pass

# Try to import optional SPI module
try:
    from soundscapy import spi
    from soundscapy.spi import (
        SkewNormalDistribution, fit_skew_normal, calculate_spi, calculate_spi_from_data,
    )
    __all__.extend([
        "spi", "SkewNormalDistribution", "fit_skew_normal", 
        "calculate_spi", "calculate_spi_from_data",
    ])
except ImportError:
    # SPI module not available
    pass
