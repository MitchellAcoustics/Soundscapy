"""Soundscapy is a Python library for soundscape analysis and visualisation."""

# ruff: noqa: E402
import importlib

from loguru import logger

# https://loguru.readthedocs.io/en/latest/resources/recipes.html#configuring-loguru-to-be-used-by-a-library-or-an-application
logger.disable("soundscapy")

# Always available core modules
from soundscapy import databases, plotting, surveys
from soundscapy import databases as db
from soundscapy._version import __version__  # noqa: F401
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
from soundscapy.surveys import add_iso_coords, processing, rename_paqs
from soundscapy.surveys.survey_utils import PAQ_IDS, PAQ_LABELS

__all__ = [
    "PAQ_IDS",
    "PAQ_LABELS",
    "ISOPlot",
    "add_iso_coords",
    "create_iso_subplots",
    "databases",
    "db",
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

# Optional R-backed modules (spi, satp) are loaded lazily via __getattr__ so
# that `import soundscapy` does not start the R process.  R only starts when
# the user explicitly accesses one of these names.
_SPI_ATTRS: frozenset[str] = frozenset(
    {
        "spi",
        "CentredParams",
        "DirectParams",
        "MultiSkewNorm",
        "cp2dp",
        "dp2cp",
        "msn",
        "spi_score",
    }
)
_SATP_ATTRS: frozenset[str] = frozenset({"satp", "SATP", "CircModelE"})


def __getattr__(name: str):  # noqa: ANN202
    """
    Lazily import optional R-backed sub-modules on first access.

    R is not started until one of these names is explicitly accessed.
    After the first access each name is stored in the module's ``__dict__``,
    so subsequent lookups skip this function entirely.
    """
    if name in _SPI_ATTRS:
        try:
            _spi = importlib.import_module("soundscapy.spi")
            _g = globals()
            _g["spi"] = _spi
            # Pull the individual public names from the sub-module so callers
            # can do ``sspy.MultiSkewNorm`` as well as ``sspy.spi.MultiSkewNorm``.
            for _attr in _SPI_ATTRS - {"spi"}:
                _g[_attr] = getattr(_spi, _attr)
            return _g[name]
        except ImportError as e:
            msg = (
                f"soundscapy.{name} requires optional SPI dependencies. "
                "Install with: pip install 'soundscapy[spi]'"
            )
            raise ImportError(msg) from e

    if name in _SATP_ATTRS:
        try:
            _satp = importlib.import_module("soundscapy.satp")
            _g = globals()
            _g["satp"] = _satp
            for _attr in _SATP_ATTRS - {"satp"}:
                _g[_attr] = getattr(_satp, _attr)
            return _g[name]
        except ImportError as e:
            msg = (
                f"soundscapy.{name} requires optional SATP dependencies. "
                "Install with: pip install 'soundscapy[satp]'"
            )
            raise ImportError(msg) from e

    msg = f"module 'soundscapy' has no attribute {name!r}"
    raise AttributeError(msg)


def __dir__() -> list[str]:
    """Extend dir() to include lazily-loaded optional names (PEP 562)."""
    return sorted(list(globals()) + list(_SPI_ATTRS) + list(_SATP_ATTRS))
