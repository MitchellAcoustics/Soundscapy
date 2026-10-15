# SPDX-License-Identifier: BSD-3-Clause
# Copyright (C) 2026  Andrew Mitchell

# Single source of truth for the top-level API. lazy_loader parses relative
# imports, type checkers use the stub as the full API, and griffe needs the
# literal ``__all__``. ``as X`` aliases mark public re-exports (PEP 484).

# Optional audio/satp/spi exports are lazy; only flagship objects are promoted.
# Access other names through their subpackage. Deprecated moved_to names stay
# excluded.
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
