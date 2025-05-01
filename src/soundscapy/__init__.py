"""Soundscapy is a Python library for soundscape analysis and visualisation."""

# ruff: noqa: E402
from loguru import logger

# https://loguru.readthedocs.io/en/latest/resources/recipes.html#configuring-loguru-to-be-used-by-a-library-or-an-application
logger.disable("soundscapy")

# Always available core modules
from . import databases, plotting, surveys
from ._version import __version__  # noqa: F401
from .databases import isd, satp
from .plotting import density_plot, scatter_plot
from .sspylogging import (
    disable_logging,
    enable_debug,
    get_logger,
    setup_logging,
)
from .surveys import processing

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
    from . import audio
    from .audio import (
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
    from . import spi
    from .spi import (
        CentredParams,
        DirectParams,
        MultiSkewNorm,
        cp2dp,
        dp2cp,
        msn,
    )

    __all__ += [
        "CentredParams",
        "DirectParams",
        "MultiSkewNorm",
        "cp2dp",
        "dp2cp",
        "msn",
        "spi",
    ]

except ImportError:
    # SPI module not available
    pass
