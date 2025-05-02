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
import warnings
from typing import Any, NoReturn

from rpy2 import robjects
from rpy2.robjects import numpy2ri, pandas2ri

# These are used in the docstring examples but not in the code
# They will be used by code that imports and uses this module
from soundscapy.sspylogging import get_logger

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

REQUIRED_R_VERSION = 3.6


def check_r_availability() -> None:
    """
    Check if R is installed and accessible through rpy2.

    Raises
    ------
    ImportError
        If R is not installed or cannot be accessed.

    """
    global _r_checked  # noqa: PLW0603

    def _raise_r_not_found_error() -> NoReturn:
        msg = (
            "rpy2 is installed but it cannot find an R installation. "
            "Please ensure R is installed and correctly configured. "
            "On Linux: Install R with your package manager (e.g., apt-get install r-base)."  # noqa: E501
            "On macOS: Install R from CRAN (https://cran.r-project.org/bin/macosx/). "
            "On Windows: Install R from CRAN (https://cran.r-project.org/bin/windows/base/)."
        )
        raise ImportError(msg)

    def _raise_r_access_error(e: Exception) -> NoReturn:
        msg = (
            f"Error accessing R installation: {e!s}. "
            "Please ensure R is installed and correctly configured."
        )
        raise ImportError(msg)

    def _raise_r_version_too_old_error(r_version_num: float) -> NoReturn:
        msg = (
            f"R version {r_version_num} is too old."
            f"The 'sn' package requires R >= {REQUIRED_R_VERSION}."
            "Please upgrade your R installation."
        )
        raise ImportError(msg)

    if _r_checked:
        return

    try:
        from rpy2 import robjects

        # Basic check to ensure R is running by getting R version
        r_version = robjects.r("R.version.string")[0]  # type: ignore[index]
        logger.debug("R version: %s", r_version)

        # Check if minimum R version requirements are met
        # The 'sn' package requires R >= 3.6.0
        r_version_num = robjects.r(
            "as.numeric(R.version$major) + as.numeric(R.version$minor)/10"
        )[0]  # type: ignore[index]

        if r_version_num < REQUIRED_R_VERSION:
            _raise_r_version_too_old_error(r_version_num)

        _r_checked = True
    except ImportError:
        _raise_r_not_found_error()  # Call the handler
    except Exception as e:  # noqa: BLE001
        _raise_r_access_error(e)  # Call the handler


def check_sn_package() -> None:
    """
    Check if the R 'sn' package is installed.

    Raises
    ------
    ImportError
        If the 'sn' package is not installed.

    """
    global _sn_checked  # noqa: PLW0603

    def _raise_sn_version_too_old_error(version: str) -> NoReturn:
        msg = (
            f"R 'sn' package version {version} is too old. "
            "The SPI feature requires 'sn' >= 2.0.0. "
            "Please upgrade the package by running in R: install.packages('sn')"
        )
        raise ImportError(msg)

    def _raise_sn_not_installed_error() -> NoReturn:
        msg = (
            "R package 'sn' is not installed. "
            "Please install it by running in R: install.packages('sn')"
        )
        raise ImportError(msg)

    def _raise_sn_check_error(e: Exception) -> NoReturn:
        msg = (
            f"Error checking for R 'sn' package: {e!s}. "
            "Please ensure the package is installed by running in R: install.packages('sn')"  # noqa: E501
        )
        raise ImportError(msg)

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
            from rpy2 import robjects

            # Use R code to get the package version
            version = robjects.r('as.character(packageVersion("sn"))')[0]  # type: ignore[index]
            logger.debug("R 'sn' package version: %s", version)

            # Check if package version meets requirements
            # The SPI implementation requires 'sn' >= 2.0.0
            if version < "2.0.0":
                _raise_sn_version_too_old_error(version)

            _sn_checked = True
        except rpackages.PackageNotInstalledError:
            _raise_sn_not_installed_error()
    except Exception as e:
        if "sn" in str(e):
            # Already a more specific error about the sn package
            raise  # Re-raising is okay here
        _raise_sn_check_error(e)


def check_dependencies() -> dict[str, Any]:
    """
    Check all required R dependencies for the SPI module.

    This function checks:
    1. R installation accessibility
    2. R version compatibility
    3. 'sn' package availability
    4. 'sn' package version compatibility

    Returns
    -------
    dict[str, Any]
        Dictionary with dependency information.

    Raises
    ------
    ImportError
        If any dependency check fails.

    """
    # Check R availability first
    check_r_availability()

    # Then check for the sn package
    check_sn_package()

    # If we get here, all dependencies are available

    # Return information about the dependencies
    return {
        "rpy2_version": sys.modules["rpy2"].__version__,
        "r_version": robjects.r("R.version.string")[0],  # type: ignore[index]
        "sn_version": robjects.r('as.character(packageVersion("sn"))')[0],  # type: ignore[index]
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

    Returns
    -------
    dict[str, Any]
        Session information including R and package versions.

    Raises
    ------
    ImportError
        If dependencies are missing.
    RuntimeError
        If session initialization fails.

    """
    global _r_session, _sn_package, _stats_package, _base_package, _session_active  # noqa: PLW0603

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
    logger.debug("Dependencies verified: %s", dep_info)

    try:
        import rpy2.robjects.packages as rpackages
        from rpy2 import robjects

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
                "rpy2 throws a DeprecationWarning about global activation, which we're ignoring for now."  # noqa: E501
            )
            # TODO(MitchellAcoustics): Remove global conversion, as recommended by rpy2
            # https://github.com/MitchellAcoustics/Soundscapy/issues/111
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
        logger.exception("Failed to initialize R session")
        _session_active = False
        _r_session = None
        _sn_package = None
        _stats_package = None
        _base_package = None
        msg = f"Failed to initialize R session: {e!s}"
        raise RuntimeError(msg) from e


def shutdown_r_session() -> bool:
    """
    Shutdown the R session and clean up resources.

    This function:
    1. Deactivates numpy conversion
    2. Resets global session state
    3. Performs garbage collection

    Returns
    -------
    bool
        True if successful, False otherwise.

    """
    global _r_session, _sn_package, _stats_package, _base_package, _session_active  # noqa: PLW0603

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

    except Exception:
        logger.exception("Error during R session shutdown")
        return False
    else:
        return True


def get_r_session() -> tuple[Any, Any, Any, Any]:
    """
    Get the current R session and package objects.

    This function:
    1. Initializes the session if not already active
    2. Returns the session and package references

    Returns
    -------
    tuple[Any, Any, Any, Any]
        (r_session, sn_package, stats_package, base_package)

    Raises
    ------
    RuntimeError
        If session initialization fails.

    """
    global _r_session, _sn_package, _stats_package, _base_package, _session_active  # noqa: PLW0602

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
        msg = "Failed to initialize R session"
        raise RuntimeError(msg)

    return _r_session, _sn_package, _stats_package, _base_package


def install_r_packages(packages: list[str] | None = None) -> None:
    """
    Install R packages if not already installed.

    Parameters
    ----------
    packages : list[str] | None, optional
        List of R package names to install. Defaults to ["sn", "tvtnorm"].

    Raises
    ------
    ImportError
        If R is not available or package installation fails.

    """
    if packages is None:
        packages = ["sn", "tvtnorm"]

    check_r_availability()

    try:
        import rpy2.robjects.packages as rpackages
        from rpy2.robjects.vectors import StrVector

        utils = rpackages.importr("utils")
        utils.chooseCRANmirror(ind=1)

        # Check if packages are installed
        packnames_to_install = [x for x in packages if not rpackages.isinstalled(x)]
        logger.debug("Packages to install: %s", packnames_to_install)

        # Install missing packages
        if len(packnames_to_install) > 0:
            utils.install_packages(StrVector(packnames_to_install))
            logger.info("Installed missing R packages: %s", packnames_to_install)
        else:
            logger.debug("All required R packages are already installed")

    except Exception as e:
        msg = f"Failed to install R packages: {e!s}"
        raise ImportError(msg) from e


def is_session_active() -> bool:
    """
    Check if the R session is currently active.

    Returns
    -------
    bool
        True if the session is active, False otherwise.

    """
    global _session_active  # noqa: PLW0602
    return _session_active
