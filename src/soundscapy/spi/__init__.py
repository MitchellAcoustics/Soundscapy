"""
Soundscapy Psychoacoustic Indicator (SPI) calculation module.

This module provides functions and classes for calculating SPI,
based on the R implementation. Requires optional dependencies.
"""
# ruff: noqa: E402
# ignore module level import order because we need to check dependencies first

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
