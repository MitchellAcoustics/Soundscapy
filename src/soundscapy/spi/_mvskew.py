"""
Functions for multi-dimensional skew-normal and skew-T distributions.

Adapted from the ``mvskew`` package by Sven Serneels
(https://github.com/SvenSerneels/mvskew). Embedded directly since the
upstream package is unmaintained.

Functions
---------
msn_dp2cp(xi, omega_mat, alpha, tau, aux)
    Convert direct parameters to centred parameters for the skew-normal family.
mst_dp2cp(xi, omega_mat, alpha, tau, nu, upto, cp_type, symmetr, aux)
    Convert direct parameters to centred parameters for the skew-T family.
delta_etc(alpha, *args)
    Compute delta and related auxiliary quantities for a skew-normal distribution.
cov2cor(sigma)
    Convert a covariance matrix to a correlation matrix.
zeta(k, x)
    Compute the k-th derivative of the log-Mills ratio.
bleat(nu)
    Compute b(nu) from the SN book, eq. (4.15).
st_cumulants(xi, omega, alpha, nu, n)
    Compute cumulants of the skew-T distribution up to order n.
st_gamma1(delta, nu)
    Compute the standardised third cumulant of the skew-T distribution.
mst_mardia(delta_sq, nu, d)
    Compute Mardia's multivariate skewness and kurtosis for the skew-T family.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import scipy.special as sps
import scipy.stats as spp

# Maximum valid derivative order for zeta()
_MAX_ZETA_ORDER = 5
# x-threshold below which the asymptotic expansion is used in zeta(1, x)
_ASYMPTOTIC_THRESHOLD = -50
# nu threshold above which the asymptotic expansion is used in bleat()
_BLEAT_ASYMPTOTIC_NU: float = 1e4


def _check_format(x: Any) -> np.ndarray:
    """
    Ensure *x* is a flat 1-D numpy array.

    Parameters
    ----------
    x
        Input array or matrix.

    Returns
    -------
    :
        A 1-D numpy array.

    """
    # Ensure x is an ndarray (handles plain Python scalars/lists)
    x = np.atleast_1d(np.asarray(x, dtype=float))

    # Convert np.matrix to a plain ndarray
    if isinstance(x, np.matrix):
        x = np.array(x)

    # Flatten to 1-D if multi-dimensional
    if len(x.shape) > 1:
        x = x.reshape(-1)

    return x


def msn_dp2cp(
    xi: np.ndarray,
    omega_mat: np.ndarray,
    alpha: np.ndarray,
    tau: float | np.ndarray = 0,
    *,
    aux: bool = False,
) -> tuple:
    """
    Convert direct parameters to centred parameters for the skew-normal family.

    Parameters
    ----------
    xi
        Location vector; shape ``(n,)``, ``(n, 1)``, or ``(1, n)``.
    omega_mat
        Scale matrix; shape ``(n, n)``.
    alpha
        Skewness vector; shape ``(n,)``, ``(n, 1)``, or ``(1, n)``.
    tau
        Tau parameter(s); shape ``(n,)``, ``(n, 1)``, or ``(1, n)``.
    aux
        Whether to include auxiliary estimates in the output, by default False.

    Returns
    -------
    location : np.ndarray
        Centred location vector.
    scale : np.ndarray
        Centred scale matrix.
    skewness : np.ndarray
    input_skewness : np.ndarray
    auxiliary_estimates (optional)

    Notes
    -----
    Works for univariate parameters, but they must be entered as an array or
    matrix, e.g. ``msn_dp2cp(np.array([1]), np.array([2]), np.array([1]))``.

    """
    # Ensure all inputs are flat 1-D arrays
    xi = _check_format(xi)
    alpha = _check_format(alpha)
    tau = _check_format(tau)

    # Number of dimensions
    d = alpha.shape[0]

    # Wrap omega_mat as a matrix for downstream linear-algebra operations
    omega_mat = np.matrix(omega_mat)  # maybe not necessary
    # Marginal standard deviations from the diagonal of omega_mat
    omega = np.sqrt(np.diag(omega_mat))

    # Compute delta (standardised skewness direction) and related quantities
    delta, alpha_star, delta_star, ocor = delta_etc(alpha, omega_mat)

    # Mean and std-dev of the latent truncated-normal variable z
    mu_z = np.multiply(zeta(1, tau), delta)
    sd_z = np.sqrt(1 + np.multiply(zeta(2, tau), np.square(delta)))

    # Centred covariance matrix
    sigma_mat = omega_mat + np.multiply(
        zeta(2, tau), np.outer(np.multiply(omega, delta), np.multiply(omega, delta))
    )
    # Symmetrise: copy upper triangle into lower triangle (Azzalini convention)
    sigma_mat[np.tril_indices(d, k=-1)] = sigma_mat[np.triu_indices(d, k=1)]

    # Third standardised cumulant (skewness) of the centred parameterisation
    gamma1 = np.multiply(zeta(3, tau), np.power(np.divide(delta, sd_z), 3))

    # Compute the centred location (beta) — same formula for uni- and multivariate
    if isinstance(alpha, np.ndarray):  # multivariate
        beta = xi + np.multiply(mu_z, omega)
        cp = (beta, sigma_mat, gamma1, tau)
    else:  # univariate
        beta = xi + np.multiply(mu_z, omega)
        cp = (beta, sigma_mat, gamma1, tau)

    if aux:
        # Flatten delta if it arrived as a row/column vector
        if len(delta.shape) > 1 and delta.shape[1] > delta.shape[0]:
            delta = np.array(delta).reshape(-1)

        # Auxiliary quantities derived from delta — see Azzalini SN book §5.x
        lambhdha = np.divide(delta, np.sqrt(1 - np.square(delta)))
        d_diag = np.diag(np.sqrt(1 + np.square(lambhdha)))
        ocor = cov2cor(omega_mat)
        # psi: shape matrix in the conditional representation
        psi = np.matmul(d_diag, np.matmul((ocor - np.outer(delta, delta)), d_diag))
        psi = (psi + psi.T) / 2  # enforce symmetry numerically
        o_inv = np.linalg.inv(omega_mat)
        o_pcor = -cov2cor(o_inv)
        o_pcor[np.diag_indices(d, ndim=2)] = 1
        r_mat = ocor + np.multiply(zeta(2, tau), np.outer(delta, delta))
        # Symmetrise r_mat
        r_mat[np.tril_indices(d, k=-1)] = r_mat[np.triu_indices(d, k=1)]
        ratio2 = np.divide(
            np.square(delta_star), 1 + np.multiply(zeta(2, tau), np.square(delta_star))
        )
        # Multivariate Mardia skewness and kurtosis — SN book (5.74), (5.75) p.153
        gamma1_m = np.multiply(np.square(zeta(3, tau)), np.power(ratio2, 3))
        gamma2_m = np.multiply(zeta(4, tau), np.square(ratio2))
        cp = (
            beta,
            sigma_mat,
            gamma1,
            tau,
            omega,
            r_mat,
            o_inv,
            ocor,
            o_pcor,
            lambhdha,
            psi,
            delta,
            delta_star,
            alpha_star,
            gamma1_m,
            gamma2_m,
        )
    return cp


def delta_etc(alpha: np.ndarray, *args: np.ndarray) -> tuple:
    """
    Compute delta and related quantities for a skew-normal distribution.

    Parameters
    ----------
    alpha
        Skewness (shape) parameter vector.
    *args
        Optional scale matrix ``omega_mat`` of shape ``(n, n)``. If omitted, the
        univariate (d=1) path is taken.

    Returns
    -------
    :
        A 4-tuple ``(delta, alpha_star, delta_star, ocor)``.

    """
    largs = len(args)

    # Detect infinite alpha components and record their locations
    if np.isinf(alpha).any():
        inf = np.where(np.isinf(np.abs(alpha)))
        inf_flag = True
    else:
        inf_flag = False

    # Normalise alpha to a flat ndarray
    if isinstance(alpha, np.matrix):
        alpha = np.array(alpha)

    if len(alpha.shape) > 1:
        alpha = alpha.reshape(-1)

    if largs == 0:  # univariate case (d=1)
        # Standard delta formula for SN: delta = alpha / sqrt(1 + alpha^2)
        delta = alpha / np.sqrt(1 + np.square(alpha))
        if inf_flag:
            # Infinite alpha -> delta = sign(alpha)
            delta[inf] = np.sign(alpha[inf])
        alpha_star = np.nan
        delta_star = np.nan
        ocor = np.nan
    else:  # multivariate case (d>1)
        omega_mat = args[0]
        if any(omega_mat.shape != np.repeat(len(alpha), 2)):
            msg = "Dimension mismatch"
            raise ValueError(msg)
        # Convert scale matrix to correlation matrix
        ocor = cov2cor(omega_mat)
        if not inf_flag:  # standard case: all alpha finite
            # Quadratic form alpha^T * ocor * alpha
            ocor_alpha = np.matmul(ocor, alpha)
            alpha_sq = np.sum(np.multiply(alpha, ocor_alpha))
            delta = ocor_alpha / np.sqrt(1 + alpha_sq)
            alpha_star = np.sqrt(alpha_sq)
            delta_star = np.sqrt(alpha_sq / (1 + alpha_sq))
        else:  # some |alpha| == Inf
            if len(inf) > 1:
                warnings.warn(
                    "Several abs(alpha)==Inf, I handle them as 'equal-rate Inf'",
                    UserWarning,
                    stacklevel=2,
                )
            # Replace infinite components with their sign; treat as a direction vector k
            k = np.repeat(0, alpha.shape[0])
            if inf_flag:
                k[inf] = np.sign(alpha[inf])
            ocor_k = np.matmul(ocor, k)
            delta = ocor_k / np.sqrt(np.sum(np.multiply(k, ocor_k)))
            delta_star = 1
            alpha_star = np.inf
    return (delta, alpha_star, delta_star, ocor)


def cov2cor(sigma: np.ndarray) -> np.ndarray:
    """
    Convert a covariance matrix to a correlation matrix.

    Parameters
    ----------
    sigma
        Symmetric positive-definite covariance matrix of shape ``(n, n)``.

    Returns
    -------
    :
        Correlation matrix of the same shape.

    """
    n, p = sigma.shape
    if p != n:
        msg = "'sigma' must be a square numeric matrix"
        raise ValueError(msg)

    # Reciprocal standard deviations (1 / sqrt(diag))
    inv_sd = np.sqrt(1 / np.diag(sigma))
    if not (np.isfinite(inv_sd)).any:
        warnings.warn(
            "diag(.) had 0 or NA entries; non-finite result is doubtful",
            UserWarning,
            stacklevel=2,
        )

    # Scale rows and columns: rho_ij = sigma_ij / (sd_i * sd_j)
    rho = sigma
    rho = np.multiply(np.multiply(inv_sd, sigma), np.repeat(inv_sd, p).reshape((p, p)))
    # Ensure exact 1.0 on the diagonal
    rho[np.diag_indices(p, ndim=2)] = 1
    return rho


def zeta(k: int, x: np.ndarray) -> np.ndarray:  # noqa: PLR0912
    """
    Compute the k-th derivative of the log-Mills ratio.

    The Mills ratio is ``phi(x) / Phi(x)`` where ``phi`` and ``Phi`` are the
    standard normal PDF and CDF respectively. ``zeta(k, x)`` returns the
    k-th derivative of ``log(Phi(x)) + log(2)``, with special handling for
    very negative *x* (asymptotic expansion) and boundary values.

    Parameters
    ----------
    k
        Derivative order; must be an integer in ``{0, 1, 2, 3, 4, 5}``.
    x
        Evaluation points.

    Returns
    -------
    :
        Array of the same shape as *x* containing the k-th derivative values.

    Raises
    ------
    ValueError
        If *k* is not an integer in the range 0-5.

    """
    # Normalise x to a flat ndarray
    if isinstance(x, np.matrix):
        x = np.array(x)

    if len(x.shape) > 1:
        x = x.reshape(-1)

    # Return an error for invalid derivative order
    if not (0 <= k <= _MAX_ZETA_ORDER):
        msg = f"k must be an integer in {{0, ..., {_MAX_ZETA_ORDER}}}, got {k}"
        raise ValueError(msg)

    # Replace NaN entries with 0 for safe arithmetic (boundary values applied below)
    na = np.isnan(x)
    if na.any():
        x[na] = 0
    x2 = np.square(x)

    # k=0: log(2 * Phi(x)) = log of the folded-normal CDF
    if k == 0:
        z = np.log(spp.norm.cdf(x)) + np.log(2)

    # k=1: zeta_1(x) = phi(x) / Phi(x) — the Mills ratio
    if k == 1:
        # Flag elements where direct log-space evaluation is numerically unstable
        ind_sm_neg_50 = x <= _ASYMPTOTIC_THRESHOLD
        z = x
        if ind_sm_neg_50.any():
            # Asymptotic continued-fraction expansion for very negative x
            xx = x[ind_sm_neg_50]
            xx2 = x2[ind_sm_neg_50]
            z[ind_sm_neg_50] = -np.divide(
                xx,
                1
                - np.divide(1, (xx2 + 2))
                + np.divide(1, np.multiply((xx2 + 2), (xx2 + 4)))
                - np.divide(5, np.multiply(np.multiply(xx2 + 2, xx2 + 4), (xx2 + 6)))
                + np.divide(
                    9,
                    np.multiply(
                        np.multiply(np.multiply(xx2 + 2, xx2 + 4), xx2 + 6), xx2 + 8
                    ),
                )
                - np.divide(
                    129,
                    np.multiply(
                        np.multiply(
                            np.multiply(np.multiply(xx2 + 2, xx2 + 4), xx2 + 6), xx2 + 8
                        ),
                        xx2 + 10,
                    ),
                ),
            )
            z[not (ind_sm_neg_50)] = np.exp(
                np.log(spp.norm.pdf(x[not (ind_sm_neg_50)]))
                - np.log(spp.norm.cdf(x[not (ind_sm_neg_50)]))
            )
        else:
            # Standard log-space evaluation to avoid underflow
            z = np.exp(np.log(spp.norm.pdf(x)) - np.log(spp.norm.cdf(x)))

    # Higher-order derivatives expressed as recurrences in terms of lower-order zeta
    if k == 2:  # noqa: PLR2004
        z = -np.multiply(zeta(1, x), x + zeta(1, x))
    if k == 3:  # noqa: PLR2004
        z = -np.multiply(zeta(2, x), x + zeta(1, x)) - np.multiply(
            zeta(1, x), 1 + zeta(2, x)
        )
    if k == 4:  # noqa: PLR2004
        z = -np.multiply(zeta(3, x), x + 2 * zeta(1, x)) - 2 * np.multiply(
            zeta(2, x), 1 + zeta(2, x)
        )
    if k == 5:  # noqa: PLR2004
        z = (
            -np.multiply(zeta(4, x), x + 2 * zeta(1, x))
            - np.multiply(zeta(3, x), 3 + 4 * zeta(2, x))
            - 2 * np.multiply(zeta(2, x), zeta(3, x))
        )

    # Apply boundary values at -inf
    neg_inf = x == -np.inf
    if neg_inf.any():
        if k == 1:
            z[neg_inf] = np.inf
        if k == 2:  # noqa: PLR2004
            z[neg_inf] = -1
        if k in (3, 4, 5):
            z[neg_inf] = 0

    # Apply boundary values at +inf (all higher derivatives vanish)
    pos_inf = x == np.inf
    if (k > 1) and pos_inf.any():
        z[pos_inf] = 0

    return z


def bleat(nu: float | np.ndarray) -> np.ndarray:
    """
    Compute b(nu) from the SN book, eq. (4.15).

    Parameters
    ----------
    nu
        Degrees of freedom; scalar or array. Values <= 1 yield NaN.

    Returns
    -------
    :
        Array of the same shape as *nu* containing b(nu) values.

    Notes
    -----
    Uses an asymptotic expansion for nu > 1e4 to avoid Gamma overflow
    (SN book, exercise 4.6).

    """
    if isinstance(nu, (int, float, np.integer, np.floating)):
        nu = np.array([float(nu)])
    else:
        nu = np.asarray(nu, dtype=float)

    out = np.full_like(nu, np.nan)
    big = nu > _BLEAT_ASYMPTOTIC_NU
    # Standard Gamma-ratio formula applies for 1 < nu <= threshold, nu not NaN
    ok = np.where((nu > 1) & (nu <= _BLEAT_ASYMPTOTIC_NU) & ~np.isnan(nu))[0]

    # Asymptotic expansion for large nu (avoids Gamma overflow)
    out[big] = np.sqrt(2 / np.pi) * (1 + 0.75 / nu[big] + 0.78125 / np.square(nu[big]))
    # Standard formula: sqrt(nu/pi) * Gamma((nu-1)/2) / Gamma(nu/2)
    out[ok] = np.sqrt(nu[ok] / np.pi) * np.exp(
        sps.gammaln((nu[ok] - 1) / 2) - sps.gammaln(nu[ok] / 2)
    )
    return out


def st_cumulants(  # noqa: PLR0912
    xi: float | np.ndarray,
    omega: float,
    alpha: float | np.ndarray,
    nu: float,
    n: int = 4,
) -> np.ndarray:
    """
    Compute cumulants of the skew-T distribution up to order n.

    Parameters
    ----------
    xi
        Location parameter; scalar or 1-D array of dimension d.
    omega
        Scale parameter (scalar).
    alpha
        Skewness parameter; scalar or 1-D array of dimension d.
    nu
        Degrees of freedom (single value).
    n
        Number of cumulants to compute (1-4), by default 4.

    Returns
    -------
    :
        Array of shape ``(d, n)`` containing the cumulants.

    Raises
    ------
    NotImplementedError
        If *nu* is infinite (the SN limiting case is not implemented here).
    ValueError
        If *nu* is not a single positive value or *alpha* has unexpected type.

    """
    if isinstance(nu, np.ndarray):
        if nu.shape[0] > 1:
            msg = "'nu' must be a single value"
            raise ValueError(msg)
        nu = float(nu[0])

    if isinstance(alpha, (int, float, np.integer, np.floating)):
        alpha = np.array([float(alpha)])
    elif isinstance(alpha, np.ndarray):
        alpha = alpha.ravel()
    else:
        msg = "Please provide alpha as a scalar or 1-D array"
        raise TypeError(msg)

    if np.isinf(nu):
        msg = "At nu=inf, SN cumulants should be returned; not yet implemented"
        raise NotImplementedError(msg)

    d = alpha.shape[0]
    n = min(n, 4)

    # delta = alpha / sqrt(1 + alpha^2), with delta = sign(alpha) at +/-inf
    delta = np.where(
        np.abs(alpha) < np.inf,
        alpha / np.sqrt(1 + np.square(alpha)),
        np.sign(alpha),
    )

    cumul = np.full((d, n), np.nan)

    # s(k) = nu/(nu-k) — moment scaling factor for cumulant order k
    def _s(k: float) -> float:
        return 1.0 / (1.0 - k / nu)

    # First cumulant: mean = b(nu) * delta (scale applied below)
    mu = bleat(nu) * delta
    cumul[:, 0] = mu

    # Second cumulant (requires nu > 2)
    if n > 1 and nu > 2:  # noqa: PLR2004
        cumul[:, 1] = _s(2) - np.square(mu)
        # else: stays NaN (variance undefined for nu <= 2)

    # Third cumulant (requires nu > 3; infinite at nu == 3)
    if n > 2:  # noqa: PLR2004
        if nu > 3:  # noqa: PLR2004
            cumul[:, 2] = mu * (
                (3 - np.square(delta)) * _s(3) - 3 * _s(2) + 2 * np.square(mu)
            )
        elif nu == 3:  # noqa: PLR2004
            cumul[:, 2] = np.sign(alpha) * np.inf

    # Fourth cumulant (requires nu > 4; infinite at nu == 4)
    if n > 3:  # noqa: PLR2004
        if nu > 4:  # noqa: PLR2004
            cumul[:, 3] = (
                3 * _s(2) * _s(4)
                - 4 * np.square(mu) * (3 - np.square(delta)) * _s(3)
                + 6 * np.square(mu) * _s(2)
                - 3 * np.power(mu, 4)
                - 3 * np.square(cumul[:, 1])
            )
        elif nu == 4:  # noqa: PLR2004
            cumul[:, 3] = np.inf

    # Scale each cumulant column k by omega^k, then shift column 0 by xi
    cumul = cumul * (omega ** np.arange(1, n + 1))
    cumul[:, 0] = cumul[:, 0] + xi

    return cumul


def st_gamma1(delta: np.ndarray, nu: float | np.ndarray) -> np.ndarray:
    """
    Compute the standardised third cumulant (gamma1) of the skew-T distribution.

    Vectorised over *delta*; takes a single value of *nu*.

    Parameters
    ----------
    delta
        Skewness direction vector; entries in [-1, 1].
    nu
        Degrees of freedom (single value).

    Returns
    -------
    :
        Array of the same shape as *delta*. NaN where nu < 3.

    Raises
    ------
    ValueError
        If *nu* is not a single positive value.

    """
    if isinstance(nu, np.ndarray):
        if nu.shape[0] > 1:
            msg = "'nu' must be a single value"
            raise ValueError(msg)
        nu = float(nu[0])
    if nu <= 0:
        msg = "'nu' must be positive"
        raise ValueError(msg)

    delta = np.asarray(delta, dtype=float).ravel()
    out = np.full(len(delta), np.nan)
    ok = np.abs(delta) <= 1

    if nu >= 3 and np.any(ok):  # noqa: PLR2004
        # Back-compute alpha from delta so st_cumulants can use it
        alpha_ok = delta[ok] / np.sqrt(1 - np.square(delta[ok]))
        # cumul has shape (n_ok, 3); columns = cumulant orders 1..3
        cumul = st_cumulants(0, 1, alpha_ok, nu, n=3)
        # gamma1 = third cumulant / (second cumulant)^(3/2)
        out[ok] = cumul[:, 2] / np.power(cumul[:, 1], 1.5)

    return out


def mst_mardia(delta_sq: float, nu: float, d: int) -> tuple[float, float]:
    """
    Compute Mardia's multivariate skewness and kurtosis for the skew-T family.

    SN book (6.31), (6.32), p. 178.

    Parameters
    ----------
    delta_sq
        Squared delta_star; must lie in [0, 1].
    nu
        Degrees of freedom; must be > 3.
    d
        Number of dimensions; must be a positive integer.

    Returns
    -------
    :
        A 2-tuple ``(gamma1M, gamma2M)`` of Mardia's multivariate skewness
        and kurtosis measures.

    Raises
    ------
    ValueError
        If *delta_sq* is outside [0, 1] or nu <= 3.

    """
    if not (0 <= delta_sq <= 1):
        msg = "delta_sq must lie in [0, 1]"
        raise ValueError(msg)
    if nu <= 3:  # noqa: PLR2004
        msg = "'nu > 3' is required"
        raise ValueError(msg)

    # Compute standardised ST cumulants for the corresponding univariate case
    # (small epsilon avoids division by zero when delta_sq == 1)
    alpha_1d = np.sqrt(delta_sq / (1 - delta_sq + 1e-300))
    cumul = st_cumulants(0, 1, alpha_1d, nu).reshape((4,))
    mu = cumul[0]
    sigma = np.sqrt(cumul[1])
    gamma1 = cumul[2] / np.power(sigma, 3)
    gamma2 = cumul[3] / np.power(sigma, 4)

    # Multivariate Mardia skewness -- SN book (6.31)
    gamma1_m = float(
        np.square(gamma1) + 3 * (d - 1) * np.square(mu) / ((nu - 3) * np.square(sigma))
    )

    # _are(k1, k2) = (nu - k1) / (nu - k2) -- helper for kurtosis formula (6.32)
    def _are(k1: float, k2: float) -> float:
        return (nu - k1) / (nu - k2)

    # Multivariate Mardia kurtosis -- SN book (6.32)
    if nu > 4:  # noqa: PLR2004
        gamma2_m = float(
            gamma2
            + 3
            + (d**2 - 1) * _are(2, 4)
            + 2 * (d - 1) * (_are(0, 4) - np.square(mu) * _are(1, 3)) / np.square(sigma)
            - d * (d + 2)
        )
    else:
        gamma2_m = np.inf

    return (gamma1_m, gamma2_m)


def mst_dp2cp(  # noqa: PLR0912
    xi: np.ndarray,
    omega_mat: np.ndarray,
    alpha: np.ndarray,
    tau: float | np.ndarray = 0,
    nu: float = 1,
    upto: int = 4,
    cp_type: str = "proper",
    *,
    symmetr: bool = False,
    aux: bool = False,
) -> tuple | None:
    """
    Convert direct parameters to centred parameters for the skew-T family.

    Covers the skew-T (ST) family and, as the special case nu=1,
    the skew-Cauchy (SC) family.

    Parameters
    ----------
    xi
        Location vector; shape ``(n,)``, ``(n, 1)``, or ``(1, n)``.
    omega_mat
        Scale matrix; shape ``(n, n)``.
    alpha
        Skewness vector; shape ``(n,)``, ``(n, 1)``, or ``(1, n)``.
    tau
        Tau parameter(s), by default 0.
    nu
        Degrees of freedom, by default 1.
    upto
        Number of moments to include in the correction (1-4), by default 4.
    cp_type
        Type of centred parameters: ``"proper"`` (requires nu > upto) or
        ``"approx"`` (pseudo-CP, always computable). By default ``"proper"``.
    symmetr
        Enforce symmetry by setting alpha to zero, by default False.
    aux
        Whether to include auxiliary estimates in the output, by default False.

    Returns
    -------
    :
        A tuple ``(beta, sigma_mat, gamma1, gamma2, nu)`` when ``aux=False``,
        or a longer tuple with auxiliary fields when ``aux=True``.
        Returns ``None`` if proper centred parameters cannot be computed.

    Raises
    ------
    ValueError
        If *upto* is not a positive integer.

    Notes
    -----
    Works for univariate parameters, but they must be entered as an array or
    matrix, e.g.
    ``mst_dp2cp(np.array([1]), np.array([[2]]), np.array([1]))``.

    """
    xi = _check_format(xi)
    alpha = _check_format(alpha)
    tau = _check_format(tau)

    if not isinstance(upto, int) or upto < 1:
        msg = "'upto' must be a positive integer"
        raise ValueError(msg)

    # "proper" CP cannot be computed when nu <= upto
    if cp_type == "proper" and nu <= upto:
        warnings.warn(
            f"Centred parameters are not defined at {nu} degrees of freedom; "
            "proper correction requires nu > upto",
            UserWarning,
            stacklevel=2,
        )
        return None

    # Moment-order shift: zeros for "proper", [1..upto] for "approx"
    if cp_type == "proper":
        a = np.zeros(upto, dtype=int)
    else:
        a = np.arange(1, upto + 1, dtype=int)

    d = omega_mat.shape[0]
    if symmetr:
        alpha = np.zeros(d)

    omega = np.sqrt(np.diag(omega_mat))
    delta, alpha_star, delta_star, ocor = delta_etc(alpha, omega_mat)

    # First moment correction: mean shift
    mu0 = bleat(nu + a[0]) * delta * omega
    beta = xi + mu0

    # Second moment correction: covariance
    if upto > 1:
        mu_2 = bleat(nu + a[1]) * delta * omega
        sigma_mat = omega_mat * float(nu + a[1]) / float(nu + a[1] - 2) - np.outer(
            mu_2, mu_2
        )
    else:
        sigma_mat = omega_mat

    # Third moment correction: standardised skewness
    gamma1 = None
    if upto > 2 and not symmetr:  # noqa: PLR2004
        gamma1 = st_gamma1(delta, float(nu + a[2]))

    # Fourth moment correction: Mardia kurtosis
    gamma2 = None
    if upto > 3:  # noqa: PLR2004
        nu_4 = float(nu + a[3])
        if nu_4 > 3:  # noqa: PLR2004
            gamma2 = mst_mardia(delta_star**2, nu_4, d)[1]

    if aux:
        if nu <= 3:  # noqa: PLR2004
            warnings.warn(
                "Mardia parameters can only be computed for df >= 4",
                UserWarning,
                stacklevel=2,
            )
            cp = (beta, sigma_mat, gamma1, gamma2, nu)
        else:
            mardia = mst_mardia(delta_star**2, float(nu), d)
            cp = (
                beta,
                sigma_mat,
                gamma1,
                gamma2,
                nu,
                omega,
                ocor,
                delta,
                delta_star,
                alpha_star,
                mardia,
            )
    else:
        cp = (beta, sigma_mat, gamma1, gamma2, nu)

    return cp
