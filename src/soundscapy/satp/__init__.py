# SPDX-License-Identifier: BSD-3-Clause
# Copyright (C) 2026  Andrew Mitchell

"""
Soundscape Attributes Translation (SATP) calculation module.

This module provides functions and classes for conducting the SATP
analysis, based on the R implementation. Requires optional dependencies.
"""

# ruff: noqa: E402
from soundscapy._optional import require_deps as _require_deps

_require_deps(["rpy2"], extra="r")

import lazy_loader as _lazy

__getattr__, __dir__, __all__ = _lazy.attach_stub(__name__, __file__)
