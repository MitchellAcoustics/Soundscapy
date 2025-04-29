"""
R integration for skew-normal distribution calculations.

This module provides functions for:
1. Checking R and R package dependencies
2. Initializing and managing R sessions
3. Converting data between R and Python
4. Executing R functions for skew-normal calculations

It is not intended to be used directly by end users.
"""

import sys
from typing import Any
import warnings

import rpy2.robjects as robjects
from rpy2.robjects import numpy2ri, pandas2ri

# These are used in the docstring examples but not in the code
# They will be used by code that imports and uses this module
from soundscapy.logging import get_logger

logger = get_logger()

# Cached values to avoid repeated checks
_r_checked = False
_sn_checked = False

# Session state
_r_session = None
_sn_package = None
_stats_package = None
_base_package = None
_session_active = False


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
        r_version = robjects.r("R.version.string")[0]  # type: ignore
        logger.debug(f"R version: {r_version}")

        # Check if minimum R version requirements are met
        # The 'sn' package requires R >= 3.6.0
        r_version_num = robjects.r(
            "as.numeric(R.version$major) + as.numeric(R.version$minor)/10"
        )[0]  # type: ignore
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

            # Get package version using R to verify compatibility
            import rpy2.robjects as robjects

            # Use R code to get the package version
            version = robjects.r('as.character(packageVersion("sn"))')[0]  # type: ignore
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


def check_dependencies() -> dict[str, Any]:
    """
    Check all required R dependencies for the SPI module.

    This function checks:
    1. R installation accessibility
    2. R version compatibility
    3. 'sn' package availability
    4. 'sn' package version compatibility

    Returns:
        dict[str, Any]: Dictionary with dependency information.

    Raises:
        ImportError: If any dependency check fails.
    """
    # Check R availability first
    check_r_availability()

    # Then check for the sn package
    check_sn_package()

    # If we get here, all dependencies are available

    # Return information about the dependencies
    return {
        "rpy2_version": sys.modules["rpy2"].__version__,
        "r_version": robjects.r("R.version.string")[0],  # type: ignore
        "sn_version": robjects.r('as.character(packageVersion("sn"))')[0],  # type: ignore
    }


# === SESSION MANAGEMENT ===


def initialize_r_session() -> dict[str, Any]:
    """
    Initialize an R session for skew-normal distribution calculations.

    This function:
    1. Checks for R and package dependencies
    2. Imports required R packages
    3. Sets up the R environment
    4. Updates global session state

    Returns:
        dict[str, Any]: Session information including R and package versions

    Raises:
        ImportError: If dependencies are missing
        RuntimeError: If session initialization fails
    """
    global _r_session, _sn_package, _stats_package, _base_package, _session_active

    # If session is already active, just return the state
    if _session_active:
        logger.debug("R session already initialized")
        return {
            "r_session": "active",
            "sn_package": "loaded",
            "stats_package": "loaded",
            "base_package": "loaded",
        }

    # First check all dependencies
    dep_info = check_dependencies()
    logger.debug(f"Dependencies verified: {dep_info}")

    try:
        import rpy2.robjects as robjects
        import rpy2.robjects.packages as rpackages

        # Import required packages
        _sn_package = rpackages.importr("sn")
        _stats_package = rpackages.importr("stats")
        _base_package = rpackages.importr("base")
        logger.debug("Imported R packages: sn, stats, base")

        # Set R random seed for reproducibility
        robjects.r("set.seed(42)")

        # Store R session
        _r_session = robjects

        # Update session state
        _session_active = True
        logger.info("R session successfully initialized")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Activate numpy and pandas conversion
            logger.debug("Activating numpy and pandas conversion")
            logger.info(
                "rpy2 throws a DeprecationWarning about global activation, which we're ignoreing for now."
            )
            # TODO: Remove global conversion, as recommended by rpy2
            numpy2ri.activate()
            pandas2ri.activate()

        return {
            "r_session": "active",
            "sn_package": str(_sn_package),
            "stats_package": str(_stats_package),
            "base_package": str(_base_package),
            **dep_info,
        }

    except Exception as e:
        logger.error(f"Failed to initialize R session: {str(e)}")
        _session_active = False
        _r_session = None
        _sn_package = None
        _stats_package = None
        _base_package = None
        raise RuntimeError(f"Failed to initialize R session: {str(e)}")


def shutdown_r_session() -> bool:
    """
    Shutdown the R session and clean up resources.

    This function:
    1. Deactivates numpy conversion
    2. Resets global session state
    3. Performs garbage collection

    Returns:
        bool: True if successful, False otherwise
    """
    global _r_session, _sn_package, _stats_package, _base_package, _session_active

    if not _session_active:
        logger.debug("No active R session to shutdown")
        return True

    try:
        import gc

        # Clear references to R objects
        _r_session = None
        _sn_package = None
        _stats_package = None
        _base_package = None

        # Update session state
        _session_active = False

        # Force garbage collection to release R resources
        gc.collect()
        logger.info("R session successfully shutdown")
        return True

    except Exception as e:
        logger.error(f"Error during R session shutdown: {str(e)}")
        return False


def get_r_session() -> tuple[Any, Any, Any, Any]:
    """
    Get the current R session and package objects.

    This function:
    1. Initializes the session if not already active
    2. Returns the session and package references

    Returns:
        tuple[Any, Any, Any, Any]: (r_session, sn_package, stats_package, base_package)

    Raises:
        RuntimeError: If session initialization fails
    """
    global _r_session, _sn_package, _stats_package, _base_package, _session_active

    if not _session_active:
        logger.debug("R session not active, initializing")
        initialize_r_session()

    if (
        not _session_active
        or not _r_session
        or not _sn_package
        or not _stats_package
        or not _base_package
    ):
        raise RuntimeError("Failed to initialize R session")

    return _r_session, _sn_package, _stats_package, _base_package


def install_r_packages(packages: list[str] = ["sn", "tvtnorm"]) -> None:
    """
    Install R packages if not already installed.

    Args:
        packages (list[str]): List of R package names to install

    Raises:
        ImportError: If R is not available or package installation fails
    """
    check_r_availability()

    try:
        import rpy2.robjects.packages as rpackages
        from rpy2.robjects.vectors import StrVector

        utils = rpackages.importr("utils")
        utils.chooseCRANmirror(ind=1)

        # Check if packages are installed
        packnames_to_install = [x for x in packages if not rpackages.isinstalled(x)]
        logger.debug(f"Packages to install: {packnames_to_install}")

        # Install missing packages
        if len(packnames_to_install) > 0:
            utils.install_packages(StrVector(packnames_to_install))
            logger.info(f"Installed missing R packages: {packnames_to_install}")
        else:
            logger.debug("All required R packages are already installed")

    except Exception as e:
        raise ImportError(f"Failed to install R packages: {str(e)}")


def is_session_active() -> bool:
    """
    Check if the R session is currently active.

    Returns:
        bool: True if the session is active, False otherwise
    """
    global _session_active
    return _session_active
