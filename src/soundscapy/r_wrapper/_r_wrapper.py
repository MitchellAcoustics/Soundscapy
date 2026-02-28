"""
R integration for skew-normal distribution calculations.

This module provides functions for:
1. Checking R and R package dependencies
2. Initializing and managing R sessions
3. Converting data between R and Python
4. Executing R functions for skew-normal calculations

It is not intended to be used directly by end users.
"""

import importlib.metadata
from enum import Enum
from typing import Any, NamedTuple, NoReturn

from rpy2 import robjects

from soundscapy.sspylogging import get_logger

logger = get_logger()

# Cached values to avoid repeated checks
_r_checked = False
_sn_checked = False
_circe_checked = False

# Session state
_r_session = None
_sn_package = None
_circe_package = None
_stats_package = None
_base_package = None
_session_active = False

REQUIRED_R_VERSION = 3.6
AUTO_INSTALL_R_PACKAGES = True


class RSession(NamedTuple):
    """Typed container for the active R session and loaded package references.

    Returned by :func:`get_r_session`.  Named-field access (``r.sn``,
    ``r.circe`` …) is preferred over positional unpacking.
    """

    session: Any
    sn: Any
    stats: Any
    base: Any
    circe: Any


def _ver(v: str) -> tuple[int, ...]:
    """Parse a dotted version string into a comparable integer tuple.

    Avoids lexicographic pitfalls where ``"1.10" < "1.2"`` is True.
    """
    return tuple(int(x) for x in v.split("."))


class PKG_SRC(str, Enum):
    CIRCE = "MitchellAcoustics/CircE-R"


def check_r_availability() -> None:
    """
    Check that R is accessible and meets the minimum version requirement.

    Note: importing this module (or any rpy2-dependent module) already starts
    the R process via ``from rpy2 import robjects``.  This function therefore
    cannot test whether R is *installed* — R is always already running by the
    time it is called.  Its purpose is to verify the R *version* and to cache
    that result so the version query runs at most once per session.

    Raises
    ------
    ImportError
        If the running R version is older than :data:`REQUIRED_R_VERSION`, or
        if the R version cannot be queried for any reason.

    """
    global _r_checked  # noqa: PLW0603

    def _raise_r_version_too_old_error(r_version_num: float) -> NoReturn:
        msg = (
            f"R version {r_version_num} is too old. "
            f"The 'sn' package requires R >= {REQUIRED_R_VERSION}. "
            "Please upgrade your R installation."
        )
        raise ImportError(msg)

    def _raise_r_access_error(e: Exception) -> NoReturn:
        msg = (
            f"Error querying R version: {e!s}. "
            "Please ensure R is installed and correctly configured."
        )
        raise ImportError(msg)

    if _r_checked:
        return

    try:
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
    except Exception as e:  # noqa: BLE001
        _raise_r_access_error(e)


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
            if _ver(version) < (2, 0, 0):
                _raise_sn_version_too_old_error(version)

            _sn_checked = True
        except rpackages.PackageNotInstalledError:
            _raise_sn_not_installed_error()
    except Exception as e:
        if "sn" in str(e):
            # Already a more specific error about the sn package
            raise  # Re-raising is okay here
        _raise_sn_check_error(e)


def check_circe_package() -> None:
    """
    Check if the R 'CircE' package is installed.

    Raises
    ------
    ImportError
        If the 'CircE' package is not installed.

    """
    global _circe_checked  # noqa: PLW0603

    def _raise_circe_not_installed_error() -> NoReturn:
        msg = (
            "R package 'CircE' is not installed. "
            f"Please install it by running in R: remotes::install_github({PKG_SRC.CIRCE.value})"  # noqa: E501
        )
        raise ImportError(msg)

    def _raise_circe_version_too_old_error(version: str) -> NoReturn:
        msg = (
            f"R 'CircE' package version {version} is too old. "
            "The SPI feature requires 'CircE' >= 1.1. "
            f"Please upgrade the package by running in R: remotes::install_github({PKG_SRC.CIRCE.value})"  # noqa: E501
        )
        raise ImportError(msg)

    def _raise_circe_check_error(e: Exception) -> NoReturn:
        msg = (
            f"Error checking for R 'CircE' package: {e!s}. "
            f"Please ensure the package is installed by running in R: remotes::install_github({PKG_SRC.CIRCE.value})"  # noqa: E501
        )
        raise ImportError(msg)

    if _circe_checked:
        return

    # First ensure R is available
    check_r_availability()

    try:
        import rpy2.robjects.packages as rpackages

        # Check if 'CircE' package is installed
        try:
            # Just importing to verify it exists
            _ = rpackages.importr("CircE")

            # Get package version using R to verify compatibility
            from rpy2 import robjects

            # Use R code to get the package version
            version = robjects.r('as.character(packageVersion("CircE"))')[0]  # type: ignore[index]
            logger.debug("R 'CircE' package version: %s", version)

            # Tuple comparison avoids lexicographic pitfalls ("1.10" > "1.2")
            if _ver(version) < (1, 1):
                _raise_circe_version_too_old_error(version)

            _circe_checked = True

        except rpackages.PackageNotInstalledError:
            _raise_circe_not_installed_error()

    except Exception as e:
        if "CircE" in str(e):
            # Already a more specific error about the CircE package
            raise  # Re-raising is okay here
        _raise_circe_check_error(e)


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

    try:
        # Then check for the sn package
        check_sn_package()

        # Then check for the CircE package
        check_circe_package()

    except ImportError:
        if AUTO_INSTALL_R_PACKAGES:
            logger.warning(
                "One or more R dependencies are missing. Attempting to auto-install required R packages..."  # noqa: E501
            )
            try:
                install_r_packages()
                # After installation, check again to confirm everything is now available
                check_r_availability()
                check_sn_package()
                check_circe_package()
            except Exception as install_e:
                msg = (
                    f"Auto-installation of R packages failed: {install_e!s}. "
                    "Please install the required R packages manually and ensure they are accessible."  # noqa: E501
                )
                raise ImportError(msg) from install_e
        else:
            raise  # Re-raise the original ImportError if auto-install is not enabled

    # If we get here, all dependencies are available

    # Return information about the dependencies
    return {
        "rpy2_version": importlib.metadata.version("rpy2"),
        "r_version": robjects.r("R.version.string")[0],  # type: ignore[index]
        "sn_version": robjects.r('as.character(packageVersion("sn"))')[0],  # type: ignore[index]
        "circe_version": robjects.r('as.character(packageVersion("CircE"))')[0],  # type: ignore[index]
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
    global _r_session, _sn_package, _stats_package, _base_package, _session_active, _circe_package  # noqa: E501, PLW0603

    # If session is already active, just return the state
    if _session_active:
        logger.debug("R session already initialized")
        return {
            "r_session": "active",
            "sn_package": "loaded",
            "stats_package": "loaded",
            "base_package": "loaded",
            "circe_package": "loaded",
        }

    # First check all dependencies
    dep_info = check_dependencies()
    logger.debug("Dependencies verified: %s", dep_info)

    try:
        import rpy2.robjects.packages as rpackages
        from rpy2 import robjects

        # Import required packages
        _sn_package = rpackages.importr("sn")
        _circe_package = rpackages.importr("CircE")
        _stats_package = rpackages.importr("stats")
        _base_package = rpackages.importr("base")
        logger.debug("Imported R packages: sn, CircE, stats, base")

        # Set R random seed for reproducibility
        robjects.r("set.seed(42)")

        # Store R session
        _r_session = robjects

        # Update session state
        _session_active = True
        logger.info("R session successfully initialized")

        return {
            "r_session": "active",
            "sn_package": str(_sn_package),
            "stats_package": str(_stats_package),
            "base_package": str(_base_package),
            "circe_package": str(_circe_package),
            **dep_info,
        }

    except Exception as e:
        logger.exception("Failed to initialize R session")
        _session_active = False
        _r_session = None
        _sn_package = None
        _stats_package = None
        _base_package = None
        _circe_package = None
        msg = f"Failed to initialize R session: {e!s}"
        raise RuntimeError(msg) from e


def reset_r_session() -> bool:
    """
    Unload R packages and reset session state.

    Clears all Python references to the loaded R package objects (``sn``,
    ``CircE``, ``stats``, ``base``) and resets :data:`_session_active` to
    ``False``.  The *R process itself continues running* — rpy2 does not
    support terminating the embedded R interpreter.  After calling this
    function the next call to :func:`get_r_session` will re-import the
    packages.

    Returns
    -------
    bool
        ``True`` if successful, ``False`` if an error occurred.

    """
    global _r_session, _sn_package, _stats_package, _base_package, _session_active, _circe_package  # noqa: E501, PLW0603

    if not _session_active:
        logger.debug("No active R session to reset")
        return True

    try:
        import gc

        # Clear references to R objects
        _r_session = None
        _sn_package = None
        _stats_package = None
        _base_package = None
        _circe_package = None

        # Update session state
        _session_active = False

        # Force garbage collection to release R resources
        gc.collect()
        logger.info("R session packages successfully unloaded")

    except Exception:
        logger.exception("Error during R session reset")
        return False
    else:
        return True


def get_r_session() -> RSession:
    """
    Get the current R session and package objects, initialising lazily if needed.

    Returns
    -------
    RSession
        Named tuple with fields ``session``, ``sn``, ``stats``, ``base``,
        ``circe``.  Access by name (``r.sn``, ``r.circe`` …) rather than
        position.

    Raises
    ------
    RuntimeError
        If session initialisation fails.

    """
    global _r_session, _sn_package, _stats_package, _base_package, _session_active, _circe_package  # noqa: E501, PLW0602

    if not _session_active:
        logger.debug("R session not active, initializing")
        initialize_r_session()

    if (
        not _session_active
        or not _r_session
        or not _sn_package
        or not _stats_package
        or not _base_package
        or not _circe_package
    ):
        msg = "Failed to initialize R session"
        raise RuntimeError(msg)

    return RSession(
        session=_r_session,
        sn=_sn_package,
        stats=_stats_package,
        base=_base_package,
        circe=_circe_package,
    )


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
        packages = ["sn", "tvtnorm", "CircE"]

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
            if "CircE" in packnames_to_install:
                # CircE and RTHORR are only available from GitHub
                remotes = rpackages.importr("remotes")
                remotes.install_github(PKG_SRC.CIRCE.value)
                packnames_to_install.remove("CircE")
                logger.info("Installed R package 'CircE' from GitHub")

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
