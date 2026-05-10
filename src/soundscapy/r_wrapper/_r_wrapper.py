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

from __future__ import annotations

import importlib.metadata
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn, cast

from rpy2 import robjects

from soundscapy.sspylogging import get_logger

logger = get_logger()

REQUIRED_R_VERSION: str = "3.6"

CIRCE_EMBEDDED_DIR = Path(__file__).parent.joinpath("r_circe")
CIRCE_EMBEDDED_FILES: tuple[str, ...] = (
    "bound.assign.R",
    "char.assign.R",
    "residual.CircE.R",
    "CircE.Plot.R",
    "CircE.BFGS.R",
)
CIRCE_REQUIRED_SYMBOLS: tuple[str, ...] = (
    "CircE.BFGS",
    "CircE.Plot",
    "bound.assign",
    "char.assign",
    "residual.CircE",
)


class EmbeddedRPackage:
    """Proxy object exposing sourced R functions through Python attributes."""

    def __init__(self, package_name: str) -> None:
        self.package_name = package_name

    def __getattr__(self, name: str) -> Any:
        for symbol_name in (name, name.replace("_", ".")):
            if _r_function_exists(symbol_name):
                return robjects.globalenv[symbol_name]

        msg = f"Embedded R package '{self.package_name}' has no symbol '{name}'"
        raise AttributeError(msg)

    def __repr__(self) -> str:
        return f"<EmbeddedRPackage {self.package_name}>"


@dataclass
class RSession:
    """
    Unified state container for the R session, loaded packages, and check flags.

    A single module-level instance (``_state``) replaces the previous nine
    module-level globals.  Module functions read and write its fields directly —
    no ``global`` declarations are needed except in `reset_r_session`,
    which rebinds the name.

    `get_r_session` returns ``_state`` directly once the session is
    ready; callers access package objects via named fields
    (``r.sn``, ``r.circe``, …).

    Attributes
    ----------
    session
        Reference to ``rpy2.robjects`` (populated by
        `initialize_r_session`).
    sn
        Loaded ``sn`` R package object.
    stats
        Loaded ``stats`` R package object.
    base
        Loaded ``base`` R package object.
    circe
        Proxy exposing sourced embedded ``CircE`` R functions.
    active
        ``True`` once :func:`initialize_r_session` completes successfully.
    r_checked
        ``True`` once :func:`check_r_availability` has passed (cached to avoid
        re-querying the R version on every call).
    sn_checked
        ``True`` once :func:`check_sn_package` has passed (cached).
    circe_checked
        ``True`` once :func:`check_circe_package` has passed (cached).

    Notes
    -----
    All fields are reset to their defaults when `reset_r_session` is
    called, ensuring a clean re-verification on the next
    `get_r_session` call.

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
            )
            response = input("Install them now via soundscapy? [y/N] ").strip().lower()
        except EOFError:
            pass
        else:
            return response in ("y", "yes")

    return False


def _get_circe_embedded_paths() -> list[Path]:
    """Return the expected embedded CircE script paths, validating existence."""
    if not CIRCE_EMBEDDED_DIR.is_dir():
        msg = f"Embedded CircE scripts directory is missing: {CIRCE_EMBEDDED_DIR}"
        raise ImportError(msg)

    script_paths = [CIRCE_EMBEDDED_DIR / filename for filename in CIRCE_EMBEDDED_FILES]
    missing_scripts = [path.name for path in script_paths if not path.is_file()]
    if missing_scripts:
        msg = "Embedded CircE scripts are missing: " + ", ".join(missing_scripts)
        raise ImportError(msg)

    return script_paths


def _r_function_exists(symbol_name: str) -> bool:
    """Return ``True`` when an R function symbol is available in the session."""
    return bool(robjects.r(f"exists('{symbol_name}', mode='function')")[0])  # type: ignore[index]


def _embedded_circe_symbols_loaded() -> bool:
    """Return ``True`` when all required embedded CircE symbols are available."""
    return all(
        _r_function_exists(symbol_name) for symbol_name in CIRCE_REQUIRED_SYMBOLS
    )


def _source_embedded_circe_scripts() -> None:
    """Source the bundled CircE R scripts into the embedded R session."""
    script_paths = _get_circe_embedded_paths()

    for script_path in script_paths:
        try:
            source_fn = cast("Any", robjects.r["source"])
            source_fn(script_path.as_posix())
        except Exception as e:
            msg = f"Failed to source embedded CircE script '{script_path.name}': {e!s}"
            raise ImportError(msg) from e

    missing_symbols = [
        symbol_name
        for symbol_name in CIRCE_REQUIRED_SYMBOLS
        if not _r_function_exists(symbol_name)
    ]
    if missing_symbols:
        msg = (
            "Embedded CircE scripts were sourced but required symbols are still "
            "missing: " + ", ".join(missing_symbols)
        )
        raise ImportError(msg)


def _ver(v: str) -> tuple[int, ...]:
    """
    Parse a dotted version string into a comparable integer tuple.

    Avoids lexicographic pitfalls where ``"1.10" < "1.2"`` is True.
    """
    return tuple(int(x) for x in v.split("."))


def check_r_availability() -> None:
    """
    Check that R is accessible and meets the minimum version requirement.

    Notes
    -----
    Importing this module (or any rpy2-dependent module) already starts
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
        r_version_str = robjects.r("paste(R.version$major, R.version$minor, sep='.')")[  # type: ignore[bad-index]
            0
        ]

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
        import rpy2.robjects.packages as rpackages  # noqa: PLC0415

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
    Check that the embedded CircE R scripts are available and sourceable.

    Raises
    ------
    ImportError
        If the bundled CircE scripts are missing or cannot be sourced.

    """

    def _raise_circe_check_error(e: Exception) -> NoReturn:
        msg = (
            f"Error checking embedded CircE scripts: {e!s}. "
            f"Please ensure the bundled scripts exist under {CIRCE_EMBEDDED_DIR}"
        )
        raise ImportError(msg)

    if _state.circe_checked:
        return

    # First ensure R is available
    check_r_availability()

    try:
        _get_circe_embedded_paths()
        if not _embedded_circe_symbols_loaded():
            _source_embedded_circe_scripts()

        _state.circe_checked = True

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
    :
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

    except ImportError:
        if _confirm_install_r_packages():
            logger.info("User confirmed: installing missing R packages...")
            try:
                install_r_packages(["sn"])
                # Re-check to confirm everything is now available
                check_r_availability()
                check_sn_package()
            except Exception as install_e:
                msg = (
                    f"Auto-installation of R packages failed: {install_e!s}. "
                    "Please install the required R packages manually.\n"
                    "  sn     → install.packages('sn')"
                )
                raise ImportError(msg) from install_e
        else:
            raise  # User declined or non-interactive; re-raise the original ImportError

    # CircE is embedded in the Python package, so it is checked separately.
    check_circe_package()

    # If we get here, all dependencies are available

    # Return information about the dependencies
    return {
        "rpy2_version": importlib.metadata.version("rpy2"),
        "r_version": robjects.r("R.version.string")[0],  # type: ignore[index]
        "sn_version": robjects.r('as.character(packageVersion("sn"))')[0],  # type: ignore[index]
        "circe_source": "embedded",
        "circe_source_dir": str(CIRCE_EMBEDDED_DIR),
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
    :
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
            "circe_package": "embedded",
        }

    # First check all dependencies
    dep_info = check_dependencies()
    logger.debug("Dependencies verified: %s", dep_info)

    try:
        import rpy2.robjects.packages as rpackages  # noqa: PLC0415

        # Import required packages
        _state.sn = rpackages.importr("sn")
        _state.stats = rpackages.importr("stats")
        _state.base = rpackages.importr("base")
        check_circe_package()
        _state.circe = EmbeddedRPackage("CircE")
        logger.debug("Imported R packages: sn, stats, base; sourced embedded CircE")

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
            "circe_package": "embedded",
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
    :
        ``True`` if successful, ``False`` if an error occurred.

    """
    global _state  # noqa: PLW0603

    try:
        import gc  # noqa: PLC0415

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
    :
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
    packages
        List of R package names to install. Defaults to ["sn"].

    Raises
    ------
    ImportError
        If R is not available or package installation fails.

    """
    if packages is None:
        packages = ["sn"]

    check_r_availability()

    try:
        import rpy2.robjects.packages as rpackages  # noqa: PLC0415
        from rpy2.robjects.vectors import StrVector  # noqa: PLC0415

        utils = rpackages.importr("utils")
        utils.chooseCRANmirror(ind=1)

        if "CircE" in packages:
            logger.info(
                "Skipping R package 'CircE' installation because CircE is embedded"
            )
            packages = [package for package in packages if package != "CircE"]

        if not packages:
            logger.debug("No external R packages require installation")
            return

        # Check if packages are installed
        packnames_to_install = [x for x in packages if not rpackages.isinstalled(x)]
        logger.debug("Packages to install: %s", packnames_to_install)

        # Install missing packages
        if len(packnames_to_install) > 0:
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
    :
        True if the session is active, False otherwise.

    """
    return _state.active
