"""Soundscapy is a Python library for soundscape analysis and visualisation."""

# ruff: noqa: E402
from importlib.metadata import version

import lazy_loader as _lazy
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

# Optional subpackages (audio, spi, satp) are exposed lazily via SPEC 1
# (https://scientific-python.org/specs/spec-0001/).  ``import soundscapy``
# does not import them; each submodule's gate fires on first access and
# raises a uniform ImportError if its extras aren't installed.
# The adjacent __init__.pyi stub drives both lazy_loader and static type checkers.
__getattr__, __dir__, _lazy_all = _lazy.attach_stub(__name__, __file__)

__all__ = [  # noqa: PLE0604  # _lazy_all is list[str] from lazy_loader.attach_stub
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
    "setup_logging",
    "stacked_likert",
    "surveys",
    *_lazy_all,
]
