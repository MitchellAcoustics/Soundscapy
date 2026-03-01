"""
R integration for skew-normal distribution calculations.

This module provides functions for:
1. Checking R and R package dependencies
2. Initializing and managing R sessions
3. Converting data between R and Python
4. Executing R functions for skew-normal calculations

Session state is held in a single module-level :class:`RSession` dataclass
instance (``_state``) rather than nine scattered globals.  Functions that read
or write session fields do so directly — no ``global`` declarations are needed,
since mutating an object's attributes does not rebind the module-level name.
The sole exception is :func:`reset_r_session`, which creates a fresh
``RSession()`` and therefore does rebind ``_state``.

It is not intended to be used directly by end users.
"""

import importlib.metadata
import os
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Any, NoReturn

# NOTE: importing rpy2.robjects here unconditionally starts the embedded R
# process.  There is no way to defer this further — R begins as soon as this
# module is loaded.  The lazy __getattr__ in soundscapy/__init__.py ensures
# this module (and therefore R) is only loaded when the user first accesses
# soundscapy.spi or soundscapy.satp, not on a plain ``import soundscapy``.
from rpy2 import robjects

from soundscapy.sspylogging import get_logger

logger = get_logger()

REQUIRED_R_VERSION: str = "3.6"


class PKG_SRC(str, Enum):  # noqa: N801
    CIRCE = "MitchellAcoustics/CircE-R"


@dataclass
class RSession:
    """
    Unified state container for the R session, loaded packages, and check flags.

    A single module-level instance (``_state``) replaces the previous nine
    module-level globals.  Module functions read and write its fields directly —
    no ``global`` declarations are needed except in :func:`reset_r_session`,
    which rebinds the name.

    :func:`get_r_session` returns ``_state`` directly once the session is
    ready; callers access package objects via named fields
    (``r.sn``, ``r.circe``, …).

    Attributes
    ----------
    session : any
        Reference to ``rpy2.robjects`` (populated by
        :func:`initialize_r_session`).
    sn : any
        Loaded ``sn`` R package object.
    stats : any
        Loaded ``stats`` R package object.
    base : any
        Loaded ``base`` R package object.
    circe : any
        Loaded ``CircE`` R package object.
    active : bool
        ``True`` once :func:`initialize_r_session` completes successfully.
    r_checked : bool
        ``True`` once :func:`check_r_availability` has passed (cached to avoid
        re-querying the R version on every call).
    sn_checked : bool
        ``True`` once :func:`check_sn_package` has passed (cached).
    circe_checked : bool
        ``True`` once :func:`check_circe_package` has passed (cached).

    All fields are reset to their defaults when :func:`reset_r_session` is
    called, ensuring a clean re-verification on the next
    :func:`get_r_session` call.

    """

    # Package references (populated by initialize_r_session)
    session: Any = None
    sn: Any = None
    stats: Any = None
    base: Any = None
    circe: Any = None

    # Session status
    active: bool = False

    # One-time check flags (cleared on reset so the next call re-verifies)
    r_checked: bool = False
    sn_checked: bool = False
    circe_checked: bool = False

    @property
    def is_ready(self) -> bool:
        """``True`` when the session is active and all package refs are loaded."""
        return bool(
            self.active
            and self.session
            and self.sn
            and self.stats
            and self.base
            and self.circe
        )


# Single module-level state instance.  All session functions operate on this
# object; only reset_r_session() rebinds the name (via ``global _state``).
_state = RSession()


def _confirm_install_r_packages() -> bool:
    """
    Determine whether to auto-install missing R packages.

    Checks the ``SOUNDSCAPY_AUTO_INSTALL_R`` environment variable first:

    - ``"1"``, ``"true"``, or ``"yes"``  →  install without prompting (CI / scripts)
    - ``"0"``, ``"false"``, or ``"no"``  →  never install

    If the variable is unset the user is prompted interactively when stdin is a
    TTY.  In non-interactive environments the default is *not* to install.
    """
    env_val = os.environ.get("SOUNDSCAPY_AUTO_INSTALL_R", "").lower()
    if env_val in ("1", "true", "yes"):
        return True
    if env_val in ("0", "false", "no"):
        return False

    if sys.stdin.isatty():
        try:
            print(  # noqa: T201
                "\nsoundscapy: One or more R packages required for this feature "
                "are not installed.\n"
                "  sn     → install.packages('sn')\n"
                f"  CircE  → remotes::install_github('{PKG_SRC.CIRCE.value}')\n"
            )
            response = input("Install them now via soundscapy? [y/N] ").strip().lower()
        except EOFError:
            pass
        else:
            return response in ("y", "yes")

    return False


def _ver(v: str) -> tuple[int, ...]:
    """
    Parse a dotted version string into a comparable integer tuple.

    Avoids lexicographic pitfalls where ``"1.10" < "1.2"`` is True.
    """
    return tuple(int(x) for x in v.split("."))


def check_r_availability() -> None:
    """
    Check that R is accessible and meets the minimum version requirement.

    Note: importing this module (or any rpy2-dependent module) already starts
    the R process via ``from rpy2 import robjects``.  This function therefore
    cannot test whether R is *installed* — R is always already running by the
    time it is called.  Its purpose is to verify the R *version* and to cache
    that result (``_state.r_checked``) so the version query runs at most once
    per session.

    Raises
    ------
    ImportError
        If the running R version is older than :data:`REQUIRED_R_VERSION`, or
        if the R version cannot be queried for any reason.

    """

    def _raise_r_version_too_old_error(r_version_str: str) -> NoReturn:
        msg = (
            f"R version {r_version_str} is too old. "
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

    if _state.r_checked:
        return

    try:
        r_version = robjects.r("R.version.string")[0]  # type: ignore[index]
        logger.debug("R version: %s", r_version)

        # Check if minimum R version requirements are met.
        # Use _ver() tuple comparison to avoid float pitfalls (e.g. "2.1" minor
        # parsed as 2.1/10 = 0.21 instead of the intended major.minor.patch).
        # R's $minor field is like "6.0" for R 4.6.0 or "2.1" for R 4.2.1.
        r_version_str = robjects.r("paste(R.version$major, R.version$minor, sep='.')")[
            0
        ]  # type: ignore[index]

        if _ver(r_version_str) < _ver(REQUIRED_R_VERSION):
            _raise_r_version_too_old_error(r_version_str)

        _state.r_checked = True
    except ImportError:
        raise  # from _raise_r_version_too_old_error — don't wrap it
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

    if _state.sn_checked:
        return

    # First ensure R is available
    check_r_availability()

    try:
        import rpy2.robjects.packages as rpackages

        # Check if 'sn' package is installed
        try:
            # Just importing to verify it exists
            _ = rpackages.importr("sn")

            # Use R code to get the package version
            version = robjects.r('as.character(packageVersion("sn"))')[0]  # type: ignore[index]
            logger.debug("R 'sn' package version: %s", version)

            # Check if package version meets requirements
            # The SPI implementation requires 'sn' >= 2.0.0
            if _ver(version) < (2, 0, 0):
                _raise_sn_version_too_old_error(version)

            _state.sn_checked = True
        except rpackages.PackageNotInstalledError:
            _raise_sn_not_installed_error()
    except ImportError:
        raise  # Already a specific ImportError from our helpers — re-raise as-is
    except Exception as e:  # noqa: BLE001
        _raise_sn_check_error(e)


def check_circe_package() -> None:
    """
    Check if the R 'CircE' package is installed.

    Raises
    ------
    ImportError
        If the 'CircE' package is not installed.

    """

    def _raise_circe_not_installed_error() -> NoReturn:
        msg = (
            "R package 'CircE' is not installed. "
            f"Please install it by running in R: remotes::install_github('{PKG_SRC.CIRCE.value}')"  # noqa: E501
        )
        raise ImportError(msg)

    def _raise_circe_version_too_old_error(version: str) -> NoReturn:
        msg = (
            f"R 'CircE' package version {version} is too old. "
            "The SPI feature requires 'CircE' >= 1.1. "
            f"Please upgrade the package by running in R: remotes::install_github('{PKG_SRC.CIRCE.value}')"  # noqa: E501
        )
        raise ImportError(msg)

    def _raise_circe_check_error(e: Exception) -> NoReturn:
        msg = (
            f"Error checking for R 'CircE' package: {e!s}. "
            f"Please ensure the package is installed by running in R: remotes::install_github('{PKG_SRC.CIRCE.value}')"  # noqa: E501
        )
        raise ImportError(msg)

    if _state.circe_checked:
        return

    # First ensure R is available
    check_r_availability()

    try:
        import rpy2.robjects.packages as rpackages

        # Check if 'CircE' package is installed
        try:
            # Just importing to verify it exists
            _ = rpackages.importr("CircE")

            # Use R code to get the package version
            version = robjects.r('as.character(packageVersion("CircE"))')[0]  # type: ignore[index]
            logger.debug("R 'CircE' package version: %s", version)

            # Tuple comparison avoids lexicographic pitfalls ("1.10" > "1.2")
            if _ver(version) < (1, 1):
                _raise_circe_version_too_old_error(version)

            _state.circe_checked = True

        except rpackages.PackageNotInstalledError:
            _raise_circe_not_installed_error()

    except ImportError:
        raise  # Already a specific ImportError from our helpers — re-raise as-is
    except Exception as e:  # noqa: BLE001
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
        if _confirm_install_r_packages():
            logger.info("User confirmed: installing missing R packages...")
            try:
                install_r_packages()
                # Re-check to confirm everything is now available
                check_r_availability()
                check_sn_package()
                check_circe_package()
            except Exception as install_e:
                msg = (
                    f"Auto-installation of R packages failed: {install_e!s}. "
                    "Please install the required R packages manually.\n"
                    "  sn     → install.packages('sn')\n"
                    f"  CircE  → remotes::install_github('{PKG_SRC.CIRCE.value}')"
                )
                raise ImportError(msg) from install_e
        else:
            raise  # User declined or non-interactive; re-raise the original ImportError

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
    4. Updates the ``_state`` singleton

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
    # If session is already active, just return the state
    if _state.active:
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

        # Import required packages
        _state.sn = rpackages.importr("sn")
        _state.circe = rpackages.importr("CircE")
        _state.stats = rpackages.importr("stats")
        _state.base = rpackages.importr("base")
        logger.debug("Imported R packages: sn, CircE, stats, base")

        # Set R random seed for reproducibility
        robjects.r("set.seed(42)")

        # Store R session reference
        _state.session = robjects

        # Mark session as active
        _state.active = True
        logger.info("R session successfully initialized")

        return {
            "r_session": "active",
            "sn_package": str(_state.sn),
            "stats_package": str(_state.stats),
            "base_package": str(_state.base),
            "circe_package": str(_state.circe),
            **dep_info,
        }

    except Exception as e:
        logger.exception("Failed to initialize R session")
        # Reset to a clean state so the next call can retry from scratch.
        # reset_r_session() always clears _state regardless of active flag.
        reset_r_session()
        msg = f"Failed to initialize R session: {e!s}"
        raise RuntimeError(msg) from e


def reset_r_session() -> bool:
    """
    Unload R packages and reset all session state.

    Replaces ``_state`` with a fresh :class:`RSession` instance, clearing all
    package references, the active flag, and the package-check caches.  The
    next call to :func:`get_r_session` will therefore re-verify package
    availability and re-import everything from scratch.

    Note: the *R process itself continues running* — rpy2 does not support
    terminating the embedded R interpreter.

    Returns
    -------
    bool
        ``True`` if successful, ``False`` if an error occurred.

    """
    global _state  # noqa: PLW0603

    try:
        import gc

        was_active = _state.active
        _state = RSession()
        gc.collect()

        if was_active:
            logger.info("R session packages successfully unloaded")
        else:
            logger.debug("R session state cleared")

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
        The module-level ``_state`` instance once fully initialised.  Access
        package objects by name: ``r.sn``, ``r.circe``, ``r.base``, etc.

    Raises
    ------
    RuntimeError
        If session initialisation fails.

    """
    if not _state.active:
        logger.debug("R session not active, initializing")
        initialize_r_session()

    if not _state.is_ready:
        msg = "Failed to initialize R session"
        raise RuntimeError(msg)

    return _state


def install_r_packages(packages: list[str] | None = None) -> None:
    """
    Install R packages if not already installed.

    Parameters
    ----------
    packages : list[str] | None, optional
        List of R package names to install. Defaults to ["sn", "CircE"].

    Raises
    ------
    ImportError
        If R is not available or package installation fails.

    """
    if packages is None:
        packages = ["sn", "CircE"]

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
                # CircE is only available from GitHub
                remotes = rpackages.importr("remotes")
                remotes.install_github(PKG_SRC.CIRCE.value)
                packnames_to_install.remove("CircE")
                logger.info("Installed R package 'CircE' from GitHub")

            if packnames_to_install:
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
    return _state.active
