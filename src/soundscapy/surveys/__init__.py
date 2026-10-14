# SPDX-License-Identifier: BSD-3-Clause
# Copyright (C) 2026  Andrew Mitchell

"""
Soundscapy Surveys Package.

This package handles the processing and analysis of soundscape surveys,
including PAQ (Perceived Affective Quality) data and ISO coordinate calculations.
"""

from soundscapy._deprecated import moved_to
from soundscapy.surveys import processing, survey_utils
from soundscapy.surveys.processing import (
    add_iso_coords,
    calculate_iso_coords,
    ipsatize,
    return_paqs,
    simulation,
)
from soundscapy.surveys.survey_utils import rename_paqs

__getattr__ = moved_to(
    __name__, "soundscapy.constants", ["LANGUAGE_ANGLES", "PAQ_IDS", "PAQ_LABELS"]
)

__all__ = [
    "add_iso_coords",
    "calculate_iso_coords",
    "ipsatize",
    "processing",
    "rename_paqs",
    "return_paqs",
    "simulation",
    "survey_utils",
]
