from typing import Literal

import numpy as np
import pandas as pd
from rpy2 import robjects
from rpy2.robjects.methods import RS4

from soundscapy.spi._r_wrapper import get_r_session
from soundscapy.sspylogging import get_logger

logger = get_logger()

_, sn, _, _ = get_r_session()
logger.debug("R session and packages retrieved successfully.")


def selm(x: str, y: str, data: pd.DataFrame) -> RS4:
    formula = f"cbind({x}, {y}) ~ 1"
    return sn.selm(formula, data=data, family="SN")


def calc_cp(x: str, y: str, data: pd.DataFrame) -> tuple:
    selm_model = selm(x, y, data)
    return extract_cp(selm_model)


def calc_dp(x: str, y: str, data: pd.DataFrame) -> tuple:
    selm_model = selm(x, y, data)
    return extract_dp(selm_model)


def extract_cp(selm_model: RS4) -> tuple:
    cp = tuple(selm_model.slots["param"][1])
    return (cp[0].flatten(), cp[1], cp[2].flatten())


def extract_dp(selm_model: RS4) -> tuple:
    dp = tuple(selm_model.slots["param"][0])
    return (dp[0].flatten(), dp[1], dp[2].flatten())


def sample_msn(
    selm_model: RS4 | None = None,
    xi: np.ndarray | None = None,
    omega: np.ndarray | None = None,
    alpha: np.ndarray | None = None,
    n: int = 1000,
) -> np.ndarray:
    if selm_model is not None:
        return sn.rmsn(n, dp=selm_model.slots["param"][0])
    if xi is not None and omega is not None and alpha is not None:
        r_xi = robjects.FloatVector(xi.T)  # Transpose to make it a column vector
        r_omega = robjects.r.matrix(
            robjects.FloatVector(omega.flatten()),
            nrow=omega.shape[0],
            ncol=omega.shape[1],
        )  # type: ignore[reportCallIssue]
        r_alpha = robjects.FloatVector(alpha)  # Transpose to make it a column vector
        return sn.rmsn(n, xi=r_xi, Omega=r_omega, alpha=r_alpha)
    msg = "Either selm_model or xi, omega, and alpha must be provided."
    raise ValueError(msg)


def sample_mtsn(
    selm_model: RS4 | None = None,
    xi: np.ndarray | None = None,
    omega: np.ndarray | None = None,
    alpha: np.ndarray | None = None,
    a: float = -1,
    b: float = 1,
    n: int = 1000,
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

    Returns
    -------
    np.ndarray
        Array of samples (n x 2).

    Raises
    ------
    ValueError
        If neither `selm_model` nor all of `xi`, `omega`, and `alpha` are provided.

    """
    samples = np.array([[0, 0]])
    n_samples = 0
    while n_samples < n:
        if selm_model is not None:
            sample = sample_msn(selm_model, n=1)
        elif xi is not None and omega is not None and alpha is not None:
            sample = sample_msn(xi=xi, omega=omega, alpha=alpha, n=1)
        else:
            msg = "Either selm_model or xi, omega, and alpha must be provided."
            raise ValueError(msg)
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
    r_xi = robjects.FloatVector(xi.T)  # Transpose to make it a column vector
    r_omega = robjects.r.matrix(
        robjects.FloatVector(omega.flatten()),
        nrow=omega.shape[0],
        ncol=omega.shape[1],
    )  # type: ignore[reportCallIssue]
    r_alpha = robjects.FloatVector(alpha)  # Transpose to make it a column vector

    dp_r = robjects.ListVector(
        {
            "xi": r_xi,
            "Omega": r_omega,
            "alpha": r_alpha,
        }
    )

    cp_r = sn.dp2cp(dp_r, family=family)

    return tuple(cp_r)


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
    r_mean = robjects.FloatVector(mean.T)  # Transpose to make it a column vector
    r_sigma = robjects.r.matrix(
        robjects.FloatVector(sigma.flatten()),
        nrow=sigma.shape[0],
        ncol=sigma.shape[1],
    )  # type: ignore[reportCallIssue]
    r_skew = robjects.FloatVector(skew)  # Transpose to make it a column vector
    cp_r = robjects.ListVector(
        {
            "mean": r_mean,
            "Sigma": r_sigma,
            "skew": r_skew,
        }
    )
    dp_r = sn.cp2dp(cp_r, family=family)
    return tuple(dp_r)
