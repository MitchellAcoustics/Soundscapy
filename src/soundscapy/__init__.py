"""Soundscapy is a Python library for soundscape analysis and visualisation."""

# ruff: noqa: E402
from loguru import logger

# https://loguru.readthedocs.io/en/latest/resources/recipes.html#configuring-loguru-to-be-used-by-a-library-or-an-application
logger.disable("soundscapy")

# Always available core modules
from soundscapy import databases, plotting, surveys
from soundscapy._version import __version__  # noqa: F401
from soundscapy.databases import isd, satp
from soundscapy.plotting import (
    ISOPlot,
    create_iso_subplots,
    density,
    iso_plot,
    jointplot,
    likert,
    scatter,
)
from soundscapy.plotting.likert import paq_likert, paq_radar_plot, stacked_likert
from soundscapy.sspylogging import (
    disable_logging,
    enable_debug,
    get_logger,
    setup_logging,
)
from soundscapy.surveys import add_iso_coords, processing, rename_paqs
from soundscapy.surveys.survey_utils import PAQ_IDS, PAQ_LABELS

__all__ = [
    "PAQ_IDS",
    "PAQ_LABELS",
    "ISOPlot",
    "add_iso_coords",
    "create_iso_subplots",
    "databases",
    "density",
    "disable_logging",
    "enable_debug",
    "get_logger",
    "isd",
    "iso_plot",
    "jointplot",
    "likert",
    "paq_likert",
    "paq_radar_plot",
    "plotting",
    "processing",
    "rename_paqs",
    "satp",
    "scatter",
    # Logging functions
    "setup_logging",
    "stacked_likert",
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
