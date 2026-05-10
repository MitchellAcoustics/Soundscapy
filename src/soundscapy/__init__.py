"""Soundscapy is a Python library for soundscape analysis and visualisation."""

# ruff: noqa: E402
from importlib.metadata import version

# import generalimport


# class MissingDependencyException(generalimport.exception.MissingDependencyException):
#     allow_module_level = False


# generalimport.MissingDependencyException = MissingDependencyException


from generalimport import generalimport

# Optional audio dependencies
generalimport("acoustic_toolbox", "mosqito", "maad", "tqdm")
# Optional R dependencies for SATP and SPI
generalimport("rpy2")

from loguru import logger

# https://loguru.readthedocs.io/en/latest/resources/recipes.html#configuring-loguru-to-be-used-by-a-library-or-an-application
logger.disable("soundscapy")

# Always available core modules
from soundscapy import databases, plotting, surveys
from soundscapy import databases as db
from soundscapy.databases import isd
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
from soundscapy.surveys import add_iso_coords, ipsatize, processing, rename_paqs
from soundscapy.surveys.survey_utils import PAQ_IDS, PAQ_LABELS

__version__ = version("soundscapy")

__all__ = [
    "PAQ_IDS",
    "PAQ_LABELS",
    "ISOPlot",
    "__version__",
    "add_iso_coords",
    "create_iso_subplots",
    "databases",
    "db",
    "density",
    "disable_logging",
    "enable_debug",
    "get_logger",
    "ipsatize",
    "isd",
    "iso_plot",
    "jointplot",
    "likert",
    "paq_likert",
    "paq_radar_plot",
    "plotting",
    "processing",
    "rename_paqs",
    "scatter",
    # Logging functions
    "setup_logging",
    "stacked_likert",
    # Core modules
    "surveys",
]

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

from soundscapy.satp import (
    CircE,
    CircEResults,
    CircModelE,
    fit_circe,
    normalize_polar_angles,
)
from soundscapy.spi import (
    CentredParams,
    DirectParams,
    MultiSkewNorm,
    cp2dp,
    dp2cp,
    msn,
    spi_score,
)

__all__ += [
    "CentredParams",
    "CircE",
    "CircEResults",
    "CircModelE",
    "DirectParams",
    "MultiSkewNorm",
    "cp2dp",
    "dp2cp",
    "fit_circe",
    "msn",
    "normalize_polar_angles",
    "spi_score",
]
