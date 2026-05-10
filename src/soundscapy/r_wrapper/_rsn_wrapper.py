from typing import Any, Literal

import numpy as np
import pandas as pd

# NOTE: importing rpy2 here starts the embedded R process (see _r_wrapper.py).
from rpy2 import robjects
from rpy2.robjects import numpy2ri, pandas2ri
from rpy2.robjects.methods import RS4

from soundscapy.sspylogging import get_logger

from ._r_wrapper import get_r_session

logger = get_logger()


def _r2np(r_obj: object) -> np.ndarray:
    """Convert a single R numeric object to a numpy array via an explicit converter."""
    with (robjects.default_converter + numpy2ri.converter).context():
        return robjects.conversion.get_conversion().rpy2py(r_obj)


def _np2rmat(arr: np.ndarray) -> Any:
    """Convert a 2-D numpy array to an R matrix."""
    return robjects.r.matrix(  # type: ignore[reportCallIssue]
        robjects.FloatVector(arr.flatten()), nrow=arr.shape[0], ncol=arr.shape[1]
    )


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
        msg = "Either selm_model or xi, omega, and alpha must be provided."
        raise ValueError(msg)
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

    Uses rejection sampling to ensure that the samples are within the bounds [a, b]
    for both dimensions.

    Parameters
    ----------
    selm_model
        Fitted SELM model from R's 'sn' package. If provided, parameters `xi`,
        `omega`, and `alpha` are ignored.
    xi
        Location parameter (2x1 array).
    omega
        Scale matrix (2x2 array).
    alpha
        Skewness parameter (2x1 array).
    a
        Lower truncation bound for both dimensions, by default -1.
    b
        Upper truncation bound for both dimensions, by default 1.
    n
        Number of samples to generate, by default 1000.
    max_iter
        Maximum total candidate draws before raising ``RuntimeError``.
        Guards against an infinite loop when the distribution has negligible
        probability mass inside ``[a, b]``.  Default: 100 000.

    Returns
    -------
    :
        Array of samples (n x 2).

    Raises
    ------
    ValueError
        If neither `selm_model` nor all of `xi`, `omega`, and `alpha` are provided.
    RuntimeError
        If ``max_iter`` candidate draws are exhausted before ``n`` accepted
        samples are collected, which indicates the distribution is largely
        outside ``[a, b]``.

    """
    if selm_model is None and not (
        xi is not None and omega is not None and alpha is not None
    ):
        msg = "Either selm_model or xi, omega, and alpha must be provided."
        raise ValueError(msg)

    accepted: list[np.ndarray] = []
    total_drawn = 0
    batch_size = max(n, 64)

    while len(accepted) < n:
        if total_drawn >= max_iter:
            msg = (
                f"sample_mtsn: reached max_iter={max_iter} without collecting "
                f"{n} accepted samples (got {len(accepted)}). "
                "The distribution may have negligible mass inside "
                f"[{a}, {b}]. Adjust the bounds or increase max_iter."
            )
            raise RuntimeError(msg)

        candidates = sample_msn(
            selm_model=selm_model, xi=xi, omega=omega, alpha=alpha, n=batch_size
        )
        total_drawn += batch_size

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
    """
    Convert Direct Parameters (DP) to Centred Parameters (CP).

    Parameters
    ----------
    xi
        Location parameter (2x1 array).
    omega
        Scale matrix (2x2 array).
    alpha
        Skewness parameter (2x1 array).
    family
        Distribution family, by default "SN".

    Returns
    -------
    :
        Tuple containing the centred parameters (mean, sigma, skew).

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
    """
    Convert Centred Parameters (CP) to Direct Parameters (DP).

    Parameters
    ----------
    mean
        Mean vector (2x1 array).
    sigma
        Covariance matrix (2x2 array).
    skew
        Skewness vector (2x1 array).
    family
        Distribution family, by default "SN".

    Returns
    -------
    :
        Tuple containing the direct parameters (xi, omega, alpha).

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
