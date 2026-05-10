"""
Internal R integration for skew-normal and circumplex model calculations.

Wraps rpy2 to expose R's ``sn`` package and the bundled CircE BFGS scripts.
Session state is held in a single module-level :class:`RSession` dataclass
instance (``_state``).  Call :func:`get_r_session` to obtain it; the session
is initialised lazily on first access.

Not intended for direct use — all public names are re-exported from
``soundscapy.spi`` and ``soundscapy.satp``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

# NOTE: importing rpy2.robjects here unconditionally starts the embedded R
# process.  There is no way to defer this further — R begins as soon as this
# module is loaded.  The lazy __getattr__ in soundscapy/__init__.py ensures
# this module (and therefore R) is only loaded when the user first accesses
# soundscapy.spi or soundscapy.satp, not on a plain ``import soundscapy``.
from rpy2 import robjects
from rpy2.robjects import numpy2ri, pandas2ri
from rpy2.robjects.methods import RS4
from scipy.stats import chi2 as scipy_chi2

from soundscapy.sspylogging import get_logger
from soundscapy.surveys.survey_utils import PAQ_IDS

logger = get_logger()

REQUIRED_R_VERSION: str = "3.6"

CIRCE_EMBEDDED_DIR = Path(__file__).parent / "r_circe"
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


# ── Session state ──────────────────────────────────────────────────────────────


@dataclass
class RSession:
    """
    State container for the active R session.

    A single module-level instance (``_state``) holds the loaded R package
    objects.  :func:`get_r_session` initialises it lazily on first call;
    :func:`reset_r_session` clears it to force re-initialisation.

    Attributes
    ----------
    sn
        Loaded ``sn`` R package object.
    base
        Loaded ``base`` R package object.
    active
        ``True`` once initialisation has completed successfully.
    """

    sn: Any = None
    base: Any = None
    active: bool = False


# Single module-level state instance.  Only reset_r_session() rebinds the name.
_state = RSession()


# ── Private helpers ────────────────────────────────────────────────────────────


def _ver(v: str) -> tuple[int, ...]:
    """Parse a dotted version string into a comparable integer tuple.

    Avoids lexicographic pitfalls where ``"1.10" < "1.2"`` is True.
    """
    return tuple(int(x) for x in v.split("."))


def _r_function_exists(symbol_name: str) -> bool:
    """Return ``True`` when an R function symbol is available in the session."""
    return bool(robjects.r(f"exists('{symbol_name}', mode='function')")[0])  # type: ignore[index]


def _embedded_circe_symbols_loaded() -> bool:
    """Return ``True`` when all required embedded CircE symbols are available."""
    return all(_r_function_exists(s) for s in CIRCE_REQUIRED_SYMBOLS)


def _source_embedded_circe_scripts() -> None:
    if not CIRCE_EMBEDDED_DIR.is_dir():
        raise ImportError(
            f"Embedded CircE scripts directory is missing: {CIRCE_EMBEDDED_DIR}"
        )
    script_paths = [CIRCE_EMBEDDED_DIR / f for f in CIRCE_EMBEDDED_FILES]
    missing = [p.name for p in script_paths if not p.is_file()]
    if missing:
        raise ImportError("Embedded CircE scripts are missing: " + ", ".join(missing))
    for path in script_paths:
        try:
            robjects.r["source"](path.as_posix())
        except Exception as e:
            raise ImportError(
                f"Failed to source embedded CircE script '{path.name}': {e}"
            ) from e
    missing_syms = [s for s in CIRCE_REQUIRED_SYMBOLS if not _r_function_exists(s)]
    if missing_syms:
        raise ImportError(
            "Embedded CircE scripts were sourced but required symbols are still "
            "missing: " + ", ".join(missing_syms)
        )


def _r2np(r_obj: object) -> np.ndarray:
    """Convert an R numeric object to a numpy array."""
    with (robjects.default_converter + numpy2ri.converter).context():
        return robjects.conversion.get_conversion().rpy2py(r_obj)


def _np2rmat(arr: np.ndarray) -> Any:
    """Convert a 2-D numpy array to an R matrix."""
    return robjects.r.matrix(  # type: ignore[reportCallIssue]
        robjects.FloatVector(arr.flatten()), nrow=arr.shape[0], ncol=arr.shape[1]
    )


# ── Session management ─────────────────────────────────────────────────────────


def _initialize_r_session() -> None:
    """Lazily initialise the R session, checking all dependencies."""
    if _state.active:
        return

    import rpy2.robjects.packages as rpackages  # noqa: PLC0415

    # Verify R version
    try:
        r_version_str = robjects.r("paste(R.version$major, R.version$minor, sep='.')")[
            0
        ]  # type: ignore[index]
        logger.debug("R version: %s", robjects.r("R.version.string")[0])  # type: ignore[index]
    except Exception as e:
        raise ImportError(
            f"Error querying R version: {e}. "
            "Please ensure R is installed and correctly configured."
        ) from e
    if _ver(r_version_str) < _ver(REQUIRED_R_VERSION):
        raise ImportError(
            f"R version {r_version_str} is too old; "
            f"requires >= {REQUIRED_R_VERSION}. Please upgrade your R installation."
        )

    # Check and load the sn package
    try:
        sn_pkg = rpackages.importr("sn")
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

    # Source CircE scripts if not already loaded in the global environment
    if not _embedded_circe_symbols_loaded():
        _source_embedded_circe_scripts()

    try:
        _state.sn = sn_pkg
        _state.base = rpackages.importr("base")
        robjects.r("set.seed(42)")
        _state.active = True
        logger.info("R session successfully initialized")
    except Exception as e:
        logger.exception("Failed to initialize R session")
        reset_r_session()
        raise RuntimeError(f"Failed to initialize R session: {e}") from e


def get_r_session() -> RSession:
    """Return the active R session, initialising lazily on first call.

    Returns
    -------
    :
        The module-level ``_state`` instance.

    Raises
    ------
    RuntimeError
        If session initialisation fails.
    """
    if not _state.active:
        _initialize_r_session()
    return _state


def reset_r_session() -> bool:
    """Reset all session state, forcing re-initialisation on the next call.

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
            logger.info("R session successfully reset")
        else:
            logger.debug("R session state cleared")
    except Exception:
        logger.exception("Error during R session reset")
        return False
    else:
        return True


def install_r_packages(packages: list[str] | None = None) -> None:  # noqa: ARG001
    """
    .. deprecated::
        This function is a no-op and will be removed in a future release.
        Install the R ``sn`` package directly from an R session::

            install.packages('sn')
    """
    warnings.warn(
        "install_r_packages() is deprecated and will be removed in a future release. "
        "Install the R 'sn' package directly from R: install.packages('sn')",
        DeprecationWarning,
        stacklevel=2,
    )


# ── Skew-normal wrappers (sn package) ─────────────────────────────────────────


def selm(x: str, y: str, data: pd.DataFrame) -> RS4:
    r = get_r_session()
    formula = f"cbind({x}, {y}) ~ 1"
    with (robjects.default_converter + pandas2ri.converter).context():
        r_data = robjects.conversion.get_conversion().py2rpy(data)
    return r.sn.selm(formula, data=r_data, family="SN")


def extract_cp(selm_model: RS4) -> tuple:
    # param[[1]] in R (0-indexed in rpy2) is the CP list: {mean, Sigma, skew}
    cp_r = selm_model.slots["param"][1]
    mean = _r2np(cp_r[0]).flatten()
    sigma = _r2np(cp_r[1])
    skew = _r2np(cp_r[2]).flatten()
    return (mean, sigma, skew)


def extract_dp(selm_model: RS4) -> tuple:
    # param[[0]] in R (0-indexed in rpy2) is the DP list: {xi, Omega, alpha}
    dp_r = selm_model.slots["param"][0]
    xi = _r2np(dp_r[0]).flatten()
    omega = _r2np(dp_r[1])
    alpha = _r2np(dp_r[2]).flatten()
    return (xi, omega, alpha)


def sample_msn(
    selm_model: RS4 | None = None,
    xi: np.ndarray | None = None,
    omega: np.ndarray | None = None,
    alpha: np.ndarray | None = None,
    n: int = 1000,
) -> np.ndarray:
    r = get_r_session()
    if selm_model is not None:
        r_result = r.sn.rmsn(n, dp=selm_model.slots["param"][0])
    elif xi is not None and omega is not None and alpha is not None:
        r_result = r.sn.rmsn(
            n,
            xi=robjects.FloatVector(xi.T),
            Omega=_np2rmat(omega),
            alpha=robjects.FloatVector(alpha),
        )
    else:
        raise ValueError("Either selm_model or xi, omega, and alpha must be provided.")
    return _r2np(r_result)


def sample_mtsn(
    selm_model: RS4 | None = None,
    xi: np.ndarray | None = None,
    omega: np.ndarray | None = None,
    alpha: np.ndarray | None = None,
    a: float = -1,
    b: float = 1,
    n: int = 1000,
    max_iter: int = 100_000,
) -> np.ndarray:
    """
    Sample from a multivariate truncated skew-normal distribution.

    Uses rejection sampling to ensure samples fall within ``[a, b]`` for both
    dimensions.

    Parameters
    ----------
    selm_model
        Fitted SELM model from R's ``sn`` package.  If provided, ``xi``,
        ``omega``, and ``alpha`` are ignored.
    xi
        Location parameter (2×1 array).
    omega
        Scale matrix (2×2 array).
    alpha
        Skewness parameter (2×1 array).
    a
        Lower truncation bound for both dimensions.
    b
        Upper truncation bound for both dimensions.
    n
        Number of samples to generate.
    max_iter
        Maximum total candidate draws before raising ``RuntimeError``.

    Returns
    -------
    :
        Array of samples (n × 2).

    Raises
    ------
    ValueError
        If neither ``selm_model`` nor all of ``xi``, ``omega``, ``alpha`` are given.
    RuntimeError
        If ``max_iter`` draws are exhausted before ``n`` accepted samples are
        collected.
    """
    if selm_model is None and not (
        xi is not None and omega is not None and alpha is not None
    ):
        raise ValueError("Either selm_model or xi, omega, and alpha must be provided.")

    accepted: list[np.ndarray] = []
    total_drawn = 0
    batch_size = max(n, 64)

    while len(accepted) < n:
        remaining_budget = max_iter - total_drawn
        if remaining_budget <= 0:
            raise RuntimeError(
                f"sample_mtsn: reached max_iter={max_iter} without collecting "
                f"{n} accepted samples (got {len(accepted)}). "
                "The distribution may have negligible mass inside "
                f"[{a}, {b}]. Adjust the bounds or increase max_iter."
            )
        current_batch_size = min(batch_size, remaining_budget)
        candidates = sample_msn(
            selm_model=selm_model,
            xi=xi,
            omega=omega,
            alpha=alpha,
            n=current_batch_size,
        )
        total_drawn += current_batch_size
        in_bounds = (
            (candidates[:, 0] >= a)
            & (candidates[:, 0] <= b)
            & (candidates[:, 1] >= a)
            & (candidates[:, 1] <= b)
        )
        accepted.extend(candidates[in_bounds])

    return np.vstack(accepted[:n])


def dp2cp(
    xi: np.ndarray,
    omega: np.ndarray,
    alpha: np.ndarray,
    family: Literal["SN", "ESN", "ST", "SC"] = "SN",
) -> tuple:
    """Convert Direct Parameters (DP) to Centred Parameters (CP).

    Parameters
    ----------
    xi
        Location parameter (2×1 array).
    omega
        Scale matrix (2×2 array).
    alpha
        Skewness parameter (2×1 array).
    family
        Distribution family.

    Returns
    -------
    :
        Tuple of centred parameters ``(mean, sigma, skew)``.
    """
    r = get_r_session()
    dp_r = robjects.ListVector(
        {
            "xi": robjects.FloatVector(xi.T),
            "Omega": _np2rmat(omega),
            "alpha": robjects.FloatVector(alpha),
        }
    )
    cp_r = r.sn.dp2cp(dp_r, family=family)
    return tuple(_r2np(cp_r[i]) for i in range(len(cp_r)))


def cp2dp(
    mean: np.ndarray,
    sigma: np.ndarray,
    skew: np.ndarray,
    family: Literal["SN", "ESN", "ST", "SC"] = "SN",
) -> tuple:
    """Convert Centred Parameters (CP) to Direct Parameters (DP).

    Parameters
    ----------
    mean
        Mean vector (2×1 array).
    sigma
        Covariance matrix (2×2 array).
    skew
        Skewness vector (2×1 array).
    family
        Distribution family.

    Returns
    -------
    :
        Tuple of direct parameters ``(xi, omega, alpha)``.
    """
    r = get_r_session()
    cp_r = robjects.ListVector(
        {
            "mean": robjects.FloatVector(mean.T),
            "Sigma": _np2rmat(sigma),
            "skew": robjects.FloatVector(skew),
        }
    )
    dp_r = r.sn.cp2dp(cp_r, family=family)
    return tuple(_r2np(dp_r[i]) for i in range(len(dp_r)))


# ── CircE wrappers (embedded R scripts) ───────────────────────────────────────


def bfgs_fit(
    data_cor: pd.DataFrame,
    n: int,
    scales: list[str] = PAQ_IDS,
    m_val: int = 3,
    *,
    equal_ang: bool = True,
    equal_com: bool = True,
) -> dict[str, Any]:
    """Fit a circumplex model and return extracted fit statistics.

    Calls the embedded CircE BFGS implementation and converts the result to a
    Python dict with scalar normalisation and a scipy-computed p-value.

    Parameters
    ----------
    data_cor
        Correlation matrix of the data.
    n
        Number of observations used to compute ``data_cor``.
    scales
        List of scale names.
    m_val
        Number of Fourier dimensions.
    equal_ang
        Whether to enforce equal-angles constraint.
    equal_com
        Whether to enforce equal-communalities constraint.

    Returns
    -------
    :
        Dictionary of fit statistics.
    """
    r = get_r_session()

    with (robjects.default_converter + pandas2ri.converter).context():
        # Only the Python→R conversion needs the pandas2ri context.
        # Calling as_matrix() inside the context would cause its R-matrix
        # return value to be auto-converted back to numpy by the active
        # converter, producing a numpy array instead of an R matrix.
        r_data_cor = robjects.conversion.get_conversion().py2rpy(data_cor)

    r_cor_mat = r.base.as_matrix(r_data_cor)
    circe_bfgs = robjects.globalenv["CircE.BFGS"]

    bfgs_model = circe_bfgs(
        r_cor_mat,
        v_names=robjects.StrVector(scales),
        m=m_val,
        N=n,
        start_values="PFA",
        equal_ang=equal_ang,
        equal_com=equal_com,
        iterlim=1000,
        try_refit_BFGS=True,
        print_level=0,
        file=robjects.NULL,
    )

    with (robjects.default_converter + pandas2ri.converter).context():
        py_res = {
            key.lower(): robjects.conversion.get_conversion().rpy2py(val)  # type: ignore[missing-attribute]
            for key, val in bfgs_model.items()
        }

    # Normalise length-1 numpy arrays to Python scalars.
    py_res = {
        k: (v.item() if isinstance(v, np.ndarray) and v.shape == (1,) else v)
        for k, v in py_res.items()
    }

    # rpy2 may deliver degree-of-freedom fields as numpy floats.
    for key in ("m", "d", "dfnull"):
        if key in py_res and py_res[key] is not None:
            py_res[key] = int(py_res[key])

    # Use scipy instead of R's pchisq to avoid py2rpy conversion issues.
    # Use model df ("d"), NOT null-model df ("dfnull").
    _chisq, _d = py_res.get("chisq"), py_res.get("d")
    py_res["p"] = (
        float(scipy_chi2.sf(_chisq, _d))
        if _chisq is not None and _d is not None
        else None
    )

    return py_res
