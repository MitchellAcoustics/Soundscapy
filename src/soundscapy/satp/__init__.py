"""
Soundscape Attributes Translation (SATP) calculation module.

This module provides functions and classes for conducting the SATP
analysis,  based on the R implementation. Requires optional dependencies.
"""

# Check for required dependencies directly
# This will raise ImportError if any dependency is missing
try:
    import rpy2  # noqa: F401

except ImportError as e:
    msg = (
        "SATP functionality requires additional dependencies. "
        "Install with: pip install soundscapy[satp]"
    )
    raise ImportError(msg) from e

# Now we can import our modules that depend on the optional packages
from soundscapy.satp import circe
from soundscapy.satp.circe import CircE, CircModelE, fit_circe, person_center

__all__ = ["CircE", "CircModelE", "circe", "fit_circe", "person_center"]
