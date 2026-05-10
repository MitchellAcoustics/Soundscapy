"""
Soundscape Perception Indices (SPI) calculation module.

This module provides functions and classes for calculating SPI,
based on the R implementation. Requires optional dependencies.
"""

# ruff: noqa: E402
from soundscapy._optional import require_deps

require_deps(["rpy2"], extra="r")

import lazy_loader as _lazy

__getattr__, __dir__, __all__ = _lazy.attach_stub(__name__, __file__)
