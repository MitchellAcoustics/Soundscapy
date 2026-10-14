# SPDX-License-Identifier: BSD-3-Clause
# Copyright (C) 2026  Andrew Mitchell

# Single source of truth for soundscapy's top-level public API. Read by:
#   - lazy_loader.attach_stub (runtime): only the ``from . import`` lines are
#     parsed; names already imported eagerly in __init__.py never hit the lazy
#     __getattr__, so listing them here is harmless.
#   - static type checkers (mypy, pyright, pyrefly): a .pyi fully shadows the
#     .py, so eager re-exports must be declared here too.
#   - griffe / great-docs (API reference discovery): the literal ``__all__``
#     below replaces the runtime-computed one, which griffe cannot evaluate.
# Relative imports are required by lazy_loader's stub parser.
# The "as X" aliases mark these as public re-exports per PEP 484 convention.

# Names from audio/satp/spi are lazy (gated on optional extras); the rest are
# imported eagerly in __init__.py. Only flagship objects from the optional
# subpackages are promoted here; everything else is reached via its subpackage
# (e.g. ``sspy.spi.MultiSkewNorm``). Formerly promoted names are deprecated in
# __init__.py's ``moved_to`` call and must not be re-added here.
from . import audio as audio
from . import constants as constants
from . import databases as databases
from . import plotting as plotting
from . import satp as satp
from . import spi as spi
from . import surveys as surveys
from .audio import AnalysisSettings as AnalysisSettings
from .audio import AudioAnalysis as AudioAnalysis
from .audio import Binaural as Binaural
from .databases import isd as isd
from .plotting import ISOPlot as ISOPlot
from .plotting import create_iso_subplots as create_iso_subplots
from .plotting import density as density
from .plotting import iso_plot as iso_plot
from .plotting import jointplot as jointplot
from .plotting import likert as likert
from .plotting import scatter as scatter
from .plotting.likert import paq_likert as paq_likert
from .plotting.likert import paq_radar_plot as paq_radar_plot
from .plotting.likert import stacked_likert as stacked_likert
from .satp import fit_circe as fit_circe
from .sspylogging import disable_logging as disable_logging
from .sspylogging import enable_debug as enable_debug
from .sspylogging import get_logger as get_logger
from .sspylogging import setup_logging as setup_logging
from .surveys import add_iso_coords as add_iso_coords
from .surveys import ipsatize as ipsatize
from .surveys import rename_paqs as rename_paqs

__version__: str

__all__ = [
    "AnalysisSettings",
    "AudioAnalysis",
    "Binaural",
    "ISOPlot",
    "__version__",
    "add_iso_coords",
    "audio",
    "constants",
    "create_iso_subplots",
    "databases",
    "density",
    "disable_logging",
    "enable_debug",
    "fit_circe",
    "get_logger",
    "ipsatize",
    "isd",
    "iso_plot",
    "jointplot",
    "likert",
    "paq_likert",
    "paq_radar_plot",
    "plotting",
    "rename_paqs",
    "satp",
    "scatter",
    "setup_logging",
    "spi",
    "stacked_likert",
    "surveys",
]
