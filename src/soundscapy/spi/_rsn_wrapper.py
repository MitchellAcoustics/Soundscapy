import numpy as np
import pandas as pd
from rpy2 import robjects

from soundscapy import get_logger
from soundscapy.spi._r_wrapper import get_r_session

logger = get_logger()

_, sn, _, _ = get_r_session()
logger.debug("R session and packages retrieved successfully.")


def selm(x: str, y: str, data: pd.DataFrame):
    formula = f"cbind({x}, {y}) ~ 1"
    return sn.selm(formula, data=data, family="SN")


def calc_cp(x: str, y: str, data: pd.DataFrame):
    selm_model = selm(x, y, data)
    return extract_cp(selm_model)


def calc_dp(x: str, y: str, data: pd.DataFrame):
    selm_model = selm(x, y, data)
    return extract_dp(selm_model)


def extract_cp(selm_model):
    cp = tuple(selm_model.slots["param"][1])
    cp = (cp[0].flatten(), cp[1], cp[2].flatten())
    return cp


def extract_dp(selm_model):
    dp = tuple(selm_model.slots["param"][0])
    dp = (dp[0].flatten(), dp[1], dp[2].flatten())
    return dp


def sample_msn(selm_model=None, xi=None, omega=None, alpha=None, n=1000):
    if selm_model is not None:
        return sn.rmsn(n, dp=selm_model.slots["param"][0])
    if xi is not None and omega is not None and alpha is not None:
        xi = robjects.FloatVector(xi.T)  # Transpose to make it a column vector
        omega = robjects.r.matrix(
            robjects.FloatVector(omega.flatten()),
            nrow=omega.shape[0],
            ncol=omega.shape[1],
        )
        alpha = robjects.FloatVector(alpha)  # Transpose to make it a column vector
        return sn.rmsn(n, xi=xi, Omega=omega, alpha=alpha)
    raise ValueError("Either selm_model or xi, omega, and alpha must be provided.")


def sample_sn(selm_model, n=1000):
    return sn.rsn(n, dp=selm_model.slots["param"][0])


def sample_mtsn(selm_model=None, xi=None, omega=None, alpha=None, a=-1, b=1, n=1000):
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
            raise ValueError(
                "Either selm_model or xi, omega, and alpha must be provided."
            )
        if a <= sample[0][0] <= b and a <= sample[0][1] <= b:
            samples = np.append(samples, sample, axis=0)
            if n_samples == 0:
                samples = samples[1:]
            n_samples += 1
    return samples


def _dp2cp(xi, omega, alpha, family="SN"):
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
    xi = robjects.FloatVector(xi.T)  # Transpose to make it a column vector
    omega = robjects.r.matrix(
        robjects.FloatVector(omega.flatten()),
        nrow=omega.shape[0],
        ncol=omega.shape[1],
    )
    alpha = robjects.FloatVector(alpha)  # Transpose to make it a column vector

    dp_r = robjects.ListVector(
        {
            "xi": xi,
            "Omega": omega,
            "alpha": alpha,
        }
    )

    cp_r = sn.dp2cp(dp_r, family=family)

    return tuple(cp_r)


def _cp2dp(mean, sigma, skew, family="SN"):
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
    mean = robjects.FloatVector(mean.T)  # Transpose to make it a column vector
    sigma = robjects.r.matrix(
        robjects.FloatVector(sigma.flatten()),
        nrow=sigma.shape[0],
        ncol=sigma.shape[1],
    )
    skew = robjects.FloatVector(skew)  # Transpose to make it a column vector
    cp_r = robjects.ListVector(
        {
            "mean": mean,
            "Sigma": sigma,
            "skew": skew,
        }
    )
    dp_r = sn.cp2dp(cp_r, family=family)
    return tuple(dp_r)
