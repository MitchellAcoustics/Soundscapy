"""
R integration for skew-normal distribution calculations.

This module provides functions for:

1. Checking R and R package dependencies
2. Initializing and managing R sessions
3. Converting data between R and Python
4. Executing R functions for skew-normal calculations

Session state is held in a single module-level :class:`RSession` dataclass
instance (``_state``).  Functions that read or write session fields do so
directly — no ``global`` declarations are needed, since mutating an object's
attributes does not rebind the module-level name.  The sole exception is
:func:`reset_r_session`, which creates a fresh ``RSession()`` and therefore
does rebind ``_state``.

It is not intended to be used directly by end users.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

# NOTE: importing rpy2.robjects here unconditionally starts the embedded R
# process.  There is no way to defer this further — R begins as soon as this
# module is loaded.  The lazy __getattr__ in soundscapy/__init__.py ensures
# this module (and therefore R) is only loaded when the user first accesses
# soundscapy.spi or soundscapy.satp, not on a plain ``import soundscapy``.
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


@dataclass
class RSession:
    """
    Unified state container for the R session, loaded packages, and check flags.

    A single module-level instance (``_state``) replaces scattered module-level
    globals.  Module functions read and write its fields directly — no ``global``
    declarations are needed except in `reset_r_session`, which rebinds the name.

    `get_r_session` returns ``_state`` directly once the session is ready;
    callers access package objects via named fields (``r.sn``, ``r.base``, …).

    Attributes
    ----------
    sn
        Loaded ``sn`` R package object.
    base
        Loaded ``base`` R package object.
    circe_sourced
        ``True`` once the embedded CircE R scripts have been sourced.
    active
        ``True`` once :func:`initialize_r_session` completes successfully.
    r_checked
        ``True`` once :func:`check_r_availability` has passed (cached).
    sn_checked
        ``True`` once :func:`check_sn_package` has passed (cached).
    circe_checked
        ``True`` once :func:`check_circe_package` has passed (cached).

    Notes
    -----
    All fields are reset to their defaults when `reset_r_session` is called,
    ensuring clean re-verification on the next `get_r_session` call.

    """

    sn: Any = None
    base: Any = None
    circe_sourced: bool = False
    active: bool = False
    r_checked: bool = False
    sn_checked: bool = False
    circe_checked: bool = False


# Single module-level state instance.  All session functions operate on this
# object; only reset_r_session() rebinds the name (via ``global _state``).
_state = RSession()


def _get_circe_embedded_paths() -> list[Path]:
    """Return the expected embedded CircE script paths, validating existence."""
    if not CIRCE_EMBEDDED_DIR.is_dir():
        raise ImportError(
            f"Embedded CircE scripts directory is missing: {CIRCE_EMBEDDED_DIR}"
        )
    script_paths = [CIRCE_EMBEDDED_DIR / filename for filename in CIRCE_EMBEDDED_FILES]
    missing = [p.name for p in script_paths if not p.is_file()]
    if missing:
        raise ImportError("Embedded CircE scripts are missing: " + ", ".join(missing))
    return script_paths


def _r_function_exists(symbol_name: str) -> bool:
    """Return ``True`` when an R function symbol is available in the session."""
    return bool(robjects.r(f"exists('{symbol_name}', mode='function')")[0])  # type: ignore[index]


def _embedded_circe_symbols_loaded() -> bool:
    """Return ``True`` when all required embedded CircE symbols are available."""
    return all(_r_function_exists(s) for s in CIRCE_REQUIRED_SYMBOLS)


def _source_embedded_circe_scripts() -> None:
    """Source the bundled CircE R scripts into the embedded R session."""
    for script_path in _get_circe_embedded_paths():
        try:
            robjects.r["source"](script_path.as_posix())
        except Exception as e:
            raise ImportError(
                f"Failed to source embedded CircE script '{script_path.name}': {e}"
            ) from e

    missing = [s for s in CIRCE_REQUIRED_SYMBOLS if not _r_function_exists(s)]
    if missing:
        raise ImportError(
            "Embedded CircE scripts were sourced but required symbols are still "
            "missing: " + ", ".join(missing)
        )


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
    Importing this module already starts the R process via
    ``from rpy2 import robjects``.  This function verifies the R *version*
    and caches the result so the version query runs at most once per session.

    Raises
    ------
    ImportError
        If the running R version is older than :data:`REQUIRED_R_VERSION`.

    """
    if _state.r_checked:
        return
    try:
        # R's $minor field is like "6.0" for R 4.6.0 or "2.1" for R 4.2.1 —
        # use _ver() tuple comparison to avoid float pitfalls.
        r_version_str = robjects.r("paste(R.version$major, R.version$minor, sep='.')")[
            0
        ]  # type: ignore[index]
        logger.debug("R version: %s", robjects.r("R.version.string")[0])  # type: ignore[index]
        if _ver(r_version_str) < _ver(REQUIRED_R_VERSION):
            raise ImportError(
                f"R version {r_version_str} is too old; "
                f"requires >= {REQUIRED_R_VERSION}. Please upgrade your R installation."
            )
        _state.r_checked = True
    except ImportError:
        raise
    except Exception as e:  # noqa: BLE001
        raise ImportError(
            f"Error querying R version: {e}. "
            "Please ensure R is installed and correctly configured."
        ) from e


def check_sn_package() -> None:
    """
    Check that the R ``sn`` package is installed and meets the minimum version.

    Raises
    ------
    ImportError
        If the ``sn`` package is missing or too old.

    """
    if _state.sn_checked:
        return
    try:
        import rpy2.robjects.packages as rpackages  # noqa: PLC0415

        try:
            rpackages.importr("sn")
        except rpackages.PackageNotInstalledError:
            raise ImportError(
                "R package 'sn' is not installed. Run in R: install.packages('sn')"
            )
        version = robjects.r('as.character(packageVersion("sn"))')[0]  # type: ignore[index]
        logger.debug("R 'sn' package version: %s", version)
        if _ver(version) < (2, 0, 0):
            raise ImportError(
                f"R 'sn' package version {version} is too old; requires >= 2.0.0. "
                "Run in R: install.packages('sn')"
            )
        _state.sn_checked = True
    except ImportError:
        raise
    except Exception as e:  # noqa: BLE001
        raise ImportError(
            f"Error checking for R 'sn' package: {e}. Run in R: install.packages('sn')"
        ) from e


def check_circe_package() -> None:
    """
    Check that the embedded CircE R scripts are available and sourceable.

    Raises
    ------
    ImportError
        If the bundled CircE scripts are missing or cannot be sourced.

    """
    if _state.circe_checked:
        return
    if not _embedded_circe_symbols_loaded():
        _source_embedded_circe_scripts()
    _state.circe_checked = True


def check_dependencies() -> None:
    """
    Check all required R dependencies.

    Verifies R version, the ``sn`` CRAN package, and the embedded CircE scripts.

    Raises
    ------
    ImportError
        If any dependency check fails.

    """
    check_r_availability()
    check_sn_package()
    check_circe_package()


# === SESSION MANAGEMENT ===


def initialize_r_session() -> None:
    """
    Initialize an R session for skew-normal distribution calculations.

    Raises
    ------
    ImportError
        If dependencies are missing.
    RuntimeError
        If session initialization fails.

    """
    if _state.active:
        logger.debug("R session already initialized")
        return

    check_dependencies()

    try:
        import rpy2.robjects.packages as rpackages  # noqa: PLC0415

        _state.sn = rpackages.importr("sn")
        _state.base = rpackages.importr("base")
        _state.circe_sourced = True
        logger.debug("Imported R packages: sn, base; sourced embedded CircE")

        robjects.r("set.seed(42)")
        _state.active = True
        logger.info("R session successfully initialized")

    except Exception as e:
        logger.exception("Failed to initialize R session")
        reset_r_session()
        raise RuntimeError(f"Failed to initialize R session: {e}") from e


def reset_r_session() -> bool:
    """
    Reset all session state, forcing re-verification on the next call.

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
    Return the active R session, initialising lazily on first call.

    Returns
    -------
    :
        The module-level ``_state`` instance.  Access package objects by name:
        ``r.sn``, ``r.base``, etc.

    Raises
    ------
    RuntimeError
        If session initialisation fails.

    """
    if not _state.active:
        initialize_r_session()
    return _state


def install_r_packages(packages: list[str] | None = None) -> None:
    """
    Install R packages if not already installed.

    Parameters
    ----------
    packages
        List of R package names to install. Defaults to ``["sn"]``.

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
            logger.info("Skipping 'CircE' — it is embedded, not a CRAN package")
            packages = [p for p in packages if p != "CircE"]

        if not packages:
            return

        to_install = [p for p in packages if not rpackages.isinstalled(p)]
        if to_install:
            utils.install_packages(StrVector(to_install))
            logger.info("Installed R packages: %s", to_install)
        else:
            logger.debug("All required R packages are already installed")

    except Exception as e:
        raise ImportError(f"Failed to install R packages: {e}") from e
