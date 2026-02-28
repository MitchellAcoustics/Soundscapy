from typing import Literal

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
        r_xi = robjects.FloatVector(xi.T)  # Transpose to make it a column vector
        r_omega = robjects.r.matrix(
            robjects.FloatVector(omega.flatten()),
            nrow=omega.shape[0],
            ncol=omega.shape[1],
        )  # type: ignore[reportCallIssue]
        r_alpha = robjects.FloatVector(alpha)
        r_result = r.sn.rmsn(n, xi=r_xi, Omega=r_omega, alpha=r_alpha)
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
    selm_model : optional
        Fitted SELM model from R's 'sn' package. If provided, parameters `xi`,
        `omega`, and `alpha` are ignored.
    xi : np.ndarray, optional
        Location parameter (2x1 array).
    omega : np.ndarray, optional
        Scale matrix (2x2 array).
    alpha : np.ndarray, optional
        Skewness parameter (2x1 array).
    a : float, optional
        Lower truncation bound for both dimensions, by default -1.
    b : float, optional
        Upper truncation bound for both dimensions, by default 1.
    n : int, optional
        Number of samples to generate, by default 1000.
    max_iter : int, optional
        Maximum total candidate draws before raising ``RuntimeError``.
        Guards against an infinite loop when the distribution has negligible
        probability mass inside ``[a, b]``.  Default: 100 000.

    Returns
    -------
    np.ndarray
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
    samples = np.array([[0, 0]])
    n_samples = 0
    n_iter = 0
    while n_samples < n:
        if n_iter >= max_iter:
            msg = (
                f"sample_mtsn: reached max_iter={max_iter} without collecting "
                f"{n} accepted samples (got {n_samples}). "
                "The distribution may have negligible mass inside "
                f"[{a}, {b}]. Adjust the bounds or increase max_iter."
            )
            raise RuntimeError(msg)
        if selm_model is not None:
            sample = sample_msn(selm_model, n=1)
        elif xi is not None and omega is not None and alpha is not None:
            sample = sample_msn(xi=xi, omega=omega, alpha=alpha, n=1)
        else:
            msg = "Either selm_model or xi, omega, and alpha must be provided."
            raise ValueError(msg)
        n_iter += 1
        if a <= sample[0][0] <= b and a <= sample[0][1] <= b:
            samples = np.append(samples, sample, axis=0)
            if n_samples == 0:
                samples = samples[1:]
            n_samples += 1

    # Ensure the sample is within the bounds [a, b] for both dimensions
    if not np.all((a <= samples[:, 0]) & (samples[:, 0] <= b)):
        msg = f"Sample x-values are out of bounds: [{a}, {b}]"
        raise ValueError(msg)
    if not np.all((a <= samples[:, 1]) & (samples[:, 1] <= b)):
        msg = f"Sample y-values are out of bounds: [{a}, {b}]"
        raise ValueError(msg)
    return samples


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
    xi : np.ndarray
        Location parameter (2x1 array).
    omega : np.ndarray
        Scale matrix (2x2 array).
    alpha : np.ndarray
        Skewness parameter (2x1 array).
    family : str, optional
        Distribution family, by default "SN".

    Returns
    -------
    tuple
        Tuple containing the centred parameters (mean, sigma, skew).

    """
    r = get_r_session()
    r_xi = robjects.FloatVector(xi.T)  # Transpose to make it a column vector
    r_omega = robjects.r.matrix(
        robjects.FloatVector(omega.flatten()),
        nrow=omega.shape[0],
        ncol=omega.shape[1],
    )  # type: ignore[reportCallIssue]
    r_alpha = robjects.FloatVector(alpha)

    dp_r = robjects.ListVector(
        {
            "xi": r_xi,
            "Omega": r_omega,
            "alpha": r_alpha,
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
    mean : np.ndarray
        Mean vector (2x1 array).
    sigma : np.ndarray
        Covariance matrix (2x2 array).
    skew : np.ndarray
        Skewness vector (2x1 array).
    family : str, optional
        Distribution family, by default "SN".

    Returns
    -------
    tuple
        Tuple containing the direct parameters (xi, omega, alpha).

    """
    r = get_r_session()
    r_mean = robjects.FloatVector(mean.T)  # Transpose to make it a column vector
    r_sigma = robjects.r.matrix(
        robjects.FloatVector(sigma.flatten()),
        nrow=sigma.shape[0],
        ncol=sigma.shape[1],
    )  # type: ignore[reportCallIssue]
    r_skew = robjects.FloatVector(skew)
    cp_r = robjects.ListVector(
        {
            "mean": r_mean,
            "Sigma": r_sigma,
            "skew": r_skew,
        }
    )
    dp_r = r.sn.cp2dp(cp_r, family=family)
    return tuple(_r2np(dp_r[i]) for i in range(len(dp_r)))
