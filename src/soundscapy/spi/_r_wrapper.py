"""
R integration for skew-normal distribution calculations.

This module provides functions for:
1. Checking R and R package dependencies
2. Initializing and managing R sessions
3. Converting data between R and Python
4. Executing R functions for skew-normal calculations

It is not intended to be used directly by end users.
"""

from typing import Dict, Any
import sys
from soundscapy.logging import get_logger

logger = get_logger()

# Cached values to avoid repeated checks
_r_checked = False
_sn_checked = False


def check_r_availability() -> None:
    """
    Check if R is installed and accessible through rpy2.

    Raises:
        ImportError: If R is not installed or cannot be accessed.
    """
    global _r_checked

    if _r_checked:
        return

    try:
        import rpy2.robjects as robjects

        # Basic check to ensure R is running by getting R version
        r_version = robjects.r("R.version.string")[0]
        logger.debug(f"R version: {r_version}")

        # Check if minimum R version requirements are met
        # The 'sn' package requires R >= 3.6.0
        r_version_num = robjects.r(
            "as.numeric(R.version$major) + as.numeric(R.version$minor)/10"
        )[0]
        if r_version_num < 3.6:
            raise ImportError(
                f"R version {r_version_num} is too old. The 'sn' package requires R >= 3.6.0. "
                "Please upgrade your R installation."
            )

        _r_checked = True
    except ImportError:
        raise ImportError(
            "rpy2 is installed but it cannot find an R installation. "
            "Please ensure R is installed and correctly configured. "
            "On Linux: Install R with your package manager (e.g., apt-get install r-base). "
            "On macOS: Install R from CRAN (https://cran.r-project.org/bin/macosx/). "
            "On Windows: Install R from CRAN (https://cran.r-project.org/bin/windows/base/)."
        )
    except Exception as e:
        raise ImportError(
            f"Error accessing R installation: {str(e)}. "
            "Please ensure R is installed and correctly configured."
        )


def check_sn_package() -> None:
    """
    Check if the R 'sn' package is installed.

    Raises:
        ImportError: If the 'sn' package is not installed.
    """
    global _sn_checked

    if _sn_checked:
        return

    # First ensure R is available
    check_r_availability()

    try:
        import rpy2.robjects.packages as rpackages

        # Check if 'sn' package is installed
        try:
            # Just importing to verify it exists
            _ = rpackages.importr("sn")

            # Get package version to verify compatibility
            version = rpackages.InstalledPackage("sn").version[0]
            logger.debug(f"R 'sn' package version: {version}")

            # Check if package version meets requirements
            # The SPI implementation requires 'sn' >= 2.0.0
            if version < "2.0.0":
                raise ImportError(
                    f"R 'sn' package version {version} is too old. "
                    "The SPI feature requires 'sn' >= 2.0.0. "
                    "Please upgrade the package by running in R: install.packages('sn')"
                )

            _sn_checked = True
        except rpackages.PackageNotInstalledError:
            raise ImportError(
                "R package 'sn' is not installed. "
                "Please install it by running in R: install.packages('sn')"
            )
    except Exception as e:
        if "sn" in str(e):
            # Already a more specific error about the sn package
            raise
        else:
            raise ImportError(
                f"Error checking for R 'sn' package: {str(e)}. "
                "Please ensure the package is installed by running in R: install.packages('sn')"
            )


def check_dependencies() -> Dict[str, Any]:
    """
    Check all required R dependencies for the SPI module.

    This function checks:
    1. R installation accessibility
    2. R version compatibility
    3. 'sn' package availability
    4. 'sn' package version compatibility

    Returns:
        Dict[str, Any]: Dictionary with dependency information.

    Raises:
        ImportError: If any dependency check fails.
    """
    # Check R availability first
    check_r_availability()

    # Then check for the sn package
    check_sn_package()

    # If we get here, all dependencies are available
    import rpy2.robjects as robjects
    import rpy2.robjects.packages as rpackages

    # Return information about the dependencies
    return {
        "rpy2_version": sys.modules["rpy2"].__version__,
        "r_version": robjects.r("R.version.string")[0],
        "sn_version": rpackages.InstalledPackage("sn").version[0],
    }


# Session management and data conversion functions will be added in Phase 1C and 1D
