# SPDX-License-Identifier: BSD-3-Clause
# Copyright (C) 2026  Andrew Mitchell

"""Soundscapy is a Python library for soundscape analysis and visualisation."""

# ruff: noqa: E402, F401  # F401: re-exports are declared via __init__.pyi, not a literal __all__
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

# Optional sub-packages (audio, spi, satp) are exposed lazily via SPEC 1
# (https://scientific-python.org/specs/spec-0001/).  ``import soundscapy``
# does not import them; each submodule's gate fires on first access and
# raises a uniform `ImportError` if its extras aren't installed.
# The adjacent __init__.pyi stub drives both lazy_loader and static type checkers.
_lazy_getattr, __dir__, _lazy_all = _lazy.attach_stub(__name__, __file__)

# Deprecated top-level names warn and forward to their subpackage; everything
# else is lazy. These are deliberately absent from __init__.pyi and __all__.
__getattr__ = moved_to(
    __name__,
    {
        "soundscapy.constants": ["PAQ_IDS", "PAQ_LABELS"],
        "soundscapy.audio": [
            "ConfigManager",
            "add_results",
            "parallel_process",
            "prep_multiindex_df",
            "process_all_metrics",
        ],
        "soundscapy.satp": [
            "CircE",
            "CircEResults",
            "CircModelE",
            "normalize_polar_angles",
        ],
        "soundscapy.spi": [
            "CentredParams",
            "DirectParams",
            "MultiSkewNorm",
            "cp2dp",
            "dp2cp",
            "msn",
            "spi_score",
        ],
    },
    fallback=_lazy_getattr,
    renamed_modules={"db": "soundscapy.databases"},
)


# The public API is declared in __init__.pyi; _lazy_all is every name it imports.
# pyrefly: ignore [unresolvable-dunder-all]
__all__ = ["__version__", *_lazy_all]  # noqa: PLE0604 # _lazy_all is list[str]
