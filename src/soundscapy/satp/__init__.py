"""
Soundscape Attributes Translation (SATP) calculation module.

This module provides functions and classes for conducting the SATP
analysis,  based on the R implementation. Requires optional dependencies.
"""

# Now we can import our modules that depend on the optional packages
from soundscapy.satp import circe
from soundscapy.satp.circe import (
    CircE,
    CircEResults,
    CircModelE,
    fit_circe,
    normalize_polar_angles,
    person_center,
)

__all__ = [
    "CircE",
    "CircEResults",
    "CircModelE",
    "circe",
    "fit_circe",
    "normalize_polar_angles",
    "person_center",
]
