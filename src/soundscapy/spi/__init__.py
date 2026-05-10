"""
Soundscape Perception Indices (SPI) calculation module.

This module provides functions and classes for calculating SPI,
based on the R implementation. Requires optional dependencies.
"""

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
