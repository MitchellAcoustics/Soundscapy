"""
soundscapy.spi - Soundscape Perception Indices
==================================

This module provides functionality for calculating Soundscape Perception Indices (SPI),
based on multivariate skew-normal distributions using the R 'sn' package.

Key Components:
- SkewNormalDistribution: Class representing a multivariate skew-normal distribution
- fit_skew_normal: Function to fit a skew-normal distribution to data
- calculate_spi: Function to calculate SPI score between distributions

This module requires R with the 'sn' package installed, accessed via rpy2.

Example:
    >>> # xdoctest: +SKIP
    >>> from soundscapy.spi import fit_skew_normal
    >>> distribution = fit_skew_normal(data)
    >>> spi_score = calculate_spi(distribution, target_distribution)
"""
# ruff: noqa: E402
# ignore module level import order because we need to run require_dependencies first

from soundscapy._optionals import require_dependencies
from soundscapy.logging import get_logger

logger = get_logger()

# First, check the Python packages (rpy2) using the standard mechanism
required = require_dependencies("spi")

# Then, check for R and R package dependencies
try:
    from ._r_wrapper import check_dependencies

    # This will raise an ImportError if R or the 'sn' package is not available
    dependencies = check_dependencies()
    logger.debug(f"R dependencies verified: {dependencies}")
except ImportError as e:
    # Add information to the original error about required Python packages
    full_error = (
        f"{str(e)}\n\n"
        f"Python dependency rpy2 should be installed with: pip install soundscapy[spi]"
    )
    raise ImportError(full_error) from e

# Module structure will be implemented in subsequent phases
# For now, raise NotImplementedError for the public API functions


class SkewNormalDistribution:
    """Represents a multivariate skew-normal distribution."""

    def __init__(self):
        raise NotImplementedError("To be implemented in Phase 2")


def fit_skew_normal(data, *, initial_params=None):
    """Fit a multivariate skew-normal distribution to data."""
    raise NotImplementedError("To be implemented in Phase 2")


def calculate_spi(test_dist, target_dist):
    """Calculate SPI score between two distributions."""
    raise NotImplementedError("To be implemented in Phase 3")


def calculate_spi_from_data(test_data, target_data):
    """Calculate SPI score directly from data."""
    raise NotImplementedError("To be implemented in Phase 3")


__all__ = [
    "SkewNormalDistribution",
    "fit_skew_normal",
    "calculate_spi",
    "calculate_spi_from_data",
]
