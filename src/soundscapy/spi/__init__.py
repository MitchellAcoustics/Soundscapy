"""
Soundscape Perception Indices (SPI) calculation module.

This module provides functions and classes for calculating SPI,
based on the R implementation. Requires optional dependencies.
"""

import warnings

# Check for required dependencies directly
# This will raise ImportError if any dependency is missing
try:
    import rpy2  # noqa: F401

except ImportError as e:
    msg = (
        "SPI functionality requires additional dependencies. "
        "Install with: pip install soundscapy[spi]"
    )
    raise ImportError(msg) from e

# Now we can import our modules that depend on the optional packages
from soundscapy.spi import msn
from soundscapy.spi.msn import (
    CentredParams,
    DirectParams,
    MultiSkewNorm,
    cp2dp,
    dp2cp,
    spi_score,
)

__all__ = [
    "CentredParams",
    "DirectParams",
    "MultiSkewNorm",
    "cp2dp",
    "dp2cp",
    "msn",
    "spi_score",
]

msg = "The SPI analysis module is experimental. Use with caution."
warnings.warn(msg, UserWarning, stacklevel=2)
