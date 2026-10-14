# SPDX-License-Identifier: BSD-3-Clause
# Copyright (C) 2026  Andrew Mitchell

"""Soundscapy is a Python library for soundscape analysis and visualisation."""

# ruff: noqa: E402
from importlib.metadata import version

import lazy_loader as _lazy
from loguru import logger

# https://loguru.readthedocs.io/en/latest/resources/recipes.html#configuring-loguru-to-be-used-by-a-library-or-an-application
logger.disable("soundscapy")

# Always available core modules
from soundscapy import constants, databases, plotting, surveys
from soundscapy._deprecated import moved_to
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
from soundscapy.surveys import add_iso_coords, ipsatize, rename_paqs

# Kept importable as ``soundscapy.processing`` for backwards compatibility, but
# left out of ``__all__`` so the docs list these functions once, under ``surveys``.
from soundscapy.surveys import processing as processing

__version__ = version("soundscapy")

# Optional subpackages (audio, spi, satp) are exposed lazily via SPEC 1
# (https://scientific-python.org/specs/spec-0001/).  ``import soundscapy``
# does not import them; each submodule's gate fires on first access and
# raises a uniform ImportError if its extras aren't installed.
# The adjacent __init__.pyi stub drives both lazy_loader and static type checkers.
_lazy_getattr, __dir__, _lazy_all = _lazy.attach_stub(__name__, __file__)

# Deprecated top-level constants warn and forward; everything else is lazy.
__getattr__ = moved_to(
    __name__,
    "soundscapy.constants",
    ["PAQ_IDS", "PAQ_LABELS"],
    fallback=_lazy_getattr,
)


# __all__ includes the runtime-computed lazy names from the stub
# pyrefly: ignore [unresolvable-dunder-all]
__all__ = [  # noqa: PLE0604 # _lazy_all is list[str] from lazy_loader.attach_stub
    "ISOPlot",
    "__version__",
    "add_iso_coords",
    "constants",
    "create_iso_subplots",
    "databases",
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
    "rename_paqs",
    "scatter",
    "setup_logging",
    "stacked_likert",
    "surveys",
    *_lazy_all,
]
