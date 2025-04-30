"""Soundscapy is a Python library for soundscape analysis and visualisation."""

# ruff: noqa: E402
from loguru import logger

# https://loguru.readthedocs.io/en/latest/resources/recipes.html#configuring-loguru-to-be-used-by-a-library-or-an-application
logger.disable("soundscapy")

# Always available core modules
from soundscapy import databases, plotting, surveys
from soundscapy.databases import isd, satp
from soundscapy.plotting import density_plot, scatter_plot
from soundscapy.sspylogging import (
    disable_logging,
    enable_debug,
    get_logger,
    setup_logging,
)
from soundscapy.surveys import processing

from ._version import __version__  # noqa: F401

__all__ = [
    "databases",
    "density_plot",
    "disable_logging",
    "enable_debug",
    "get_logger",
    "isd",
    "plotting",
    "processing",
    "satp",
    "scatter_plot",
    # Logging functions
    "setup_logging",
    # Core modules
    "surveys",
]

# Try to import optional audio module
try:
    from soundscapy import audio
    from soundscapy.audio import (
        AnalysisSettings,
        AudioAnalysis,
        Binaural,
        ConfigManager,
        add_results,
        parallel_process,
        prep_multiindex_df,
        process_all_metrics,
    )

    __all__ += [
        "AnalysisSettings",
        "AudioAnalysis",
        "Binaural",
        "ConfigManager",
        "add_results",
        "audio",
        "parallel_process",
        "prep_multiindex_df",
        "process_all_metrics",
    ]

except ImportError:
    # Audio module not available - this is expected if dependencies aren't installed
    pass

# Try to import optional SPI module
try:
    from soundscapy import spi
    from soundscapy.spi import (
        CentredParams,
        DirectParams,
        MultiSkewNorm,
        cp2dp,
        dp2cp,
    )

    __all__ += [
        "CentredParams",
        "DirectParams",
        "MultiSkewNorm",
        "cp2dp",
        "dp2cp",
        "spi",
    ]

except ImportError:
    # SPI module not available
    pass
