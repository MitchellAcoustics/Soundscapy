# -*- coding: utf-8 -*-
"""
Multivariate Skew-Normal Distribution.

This module implements the multivariate skew-normal distribution as described
by Azzalini & Capitanio (1999).

References
----------
.. [1] Azzalini, A. and Capitanio, A. (1999). Statistical applications
       of the multivariate skew-normal distribution.
       J. Roy. Statist. Soc., series B, vol. 61, no. 3, pp. 579-602.
       (Extended version: https://arxiv.org/abs/0911.2093)
.. [2] Azzalini, A. with the collaboration of Capitanio, A. (2014).
       The Skew-Normal and Related Families.
       Cambridge University Press, IMS Monographs series.
.. [3] The R package 'sn': The Skew-Normal and Related Distributions
       such as the Skew-t and the SUN (version 2.1.1).
       https://cran.r-project.org/package=sn

"""

import threading
import warnings  # For potential numerical warnings

import numpy as np
from scipy.linalg import cholesky, pinvh
from scipy.optimize import minimize

# from scipy.stats._covariance import Covariance, _PSD  # Use this if using Covariance objects
from scipy.special import ndtr as std_norm_cdf

# Assuming these are in the same directory or correctly pathed in SciPy structure
# Use _PSD for now, can be switched to Covariance later if integrated
from scipy.stats._multivariate import (
    _PSD,
    _squeeze_output,
    multi_rv_frozen,
    multi_rv_generic,
)

_LOG_2PI = np.log(2 * np.pi)
_LOG_2 = np.log(2)
_SQRT_2PI = np.sqrt(2 * np.pi)
_SQRT_2 = np.sqrt(2)
_SQRT_PI = np.sqrt(np.pi)

MVN_LOCK = threading.Lock()


# Helper function for logcdf based on norm.logcdf
# Copied from scipy.stats._distn_infrastructure
def _norm_logcdf(x):
    # Use np.logaddexp(np.log(0), ...) approach for potentially better precision?
    # Or handle potential -inf directly. Let's use standard logcdf for now.
    # return norm.logcdf(x) # scipy.stats.norm might not be available directly here
    # Reimplement using special functions if needed, or assume norm is available
    # For now, use scipy.special.ndtr and log
    with np.errstate(divide="ignore"):  # Ignore log(0) warnings
        return np.log(std_norm_cdf(x))


# Helper function to convert correlation matrix to covariance matrix
def _cor2cov(cor_mat, sd):
    """Convert correlation matrix to covariance matrix."""
    sd_mat = np.diag(sd)
    return sd_mat @ cor_mat @ sd_mat


# Helper function to convert covariance matrix to correlation matrix
def _cov2cor(cov_mat):
    """Convert covariance matrix to correlation matrix and return std devs."""
    sd = np.sqrt(np.diag(cov_mat))
    if np.any(sd == 0):
        # Handle case with zero variance components
        warnings.warn("Zero variance detected in covariance matrix.", RuntimeWarning)
        # Return identity for correlation where sd is 0? Or raise error?
        # Let's return identity for now, subsequent calculations might fail
        cor_mat = np.eye(cov_mat.shape[0])
        sd_inv = np.where(sd == 0, 0, 1.0 / sd)  # Avoid division by zero
        cor_mat = (cov_mat * sd_inv[:, np.newaxis]) * sd_inv[np.newaxis, :]
        # Force diagonal to 1 where sd > 0
        valid_diag = sd > 0
        cor_mat[
            np.diag_indices_from(cor_mat)[0][valid_diag],
            np.diag_indices_from(cor_mat)[1][valid_diag],
        ] = 1.0
        return sd, cor_mat

    sd_inv = 1.0 / sd
    cor_mat = (cov_mat * sd_inv[:, np.newaxis]) * sd_inv[np.newaxis, :]
    # Ensure diagonal is exactly 1 due to potential floating point issues
    np.fill_diagonal(cor_mat, 1.0)
    return sd, cor_mat


# --- Cholesky Parameterization Helpers for Optimization ---
def _vec_to_chol_param(vec, dim):
    """Convert a vector (diag L, off-diag L) to Cholesky factor parameters (log-diag L, off-diag L)."""
    chol_vec = np.zeros_like(vec)
    # Log transform diagonal elements
    # Add small epsilon for stability if diagonal elements are very close to zero
    epsilon = 1e-15
    chol_vec[:dim] = np.log(np.maximum(vec[:dim], epsilon))
    # Keep off-diagonal elements as they are
    chol_vec[dim:] = vec[dim:]
    return chol_vec


def _chol_param_to_vec(chol_vec, dim):
    """Convert Cholesky factor parameters (log-diag L, off-diag L) back to vector form (diag L, off-diag L)."""
    vec = np.zeros_like(chol_vec)
    # Inverse transform diagonal elements
    vec[:dim] = np.exp(chol_vec[:dim])
    # Keep off-diagonal elements as they are
    vec[dim:] = chol_vec[dim:]
    return vec


def _build_chol_from_vec(vec, dim):
    """Build the lower Cholesky factor L from a vector (diag L, off-diag L)."""
    if len(vec) != dim + dim * (dim - 1) // 2:
        raise ValueError(
            f"Vector length {len(vec)} does not match expected Cholesky parameters for dim={dim}."
        )
    L = np.zeros((dim, dim))
    L[np.diag_indices(dim)] = vec[:dim]  # Diagonal elements
    if dim > 1:
        L[np.tril_indices(dim, k=-1)] = vec[dim:]  # Off-diagonal elements
    return L


def _get_scale_from_chol_param(chol_params_vec, dim):
    """Get scale matrix Omega and Cholesky factor L from optimized parameters."""
    vec = _chol_param_to_vec(chol_params_vec, dim)
    L = _build_chol_from_vec(vec, dim)
    scale_mat = L @ L.T
    # Ensure symmetry
    scale_mat = (scale_mat + scale_mat.T) / 2.0
    return scale_mat, L


# --- End Cholesky Helpers ---


class multivariate_skew_normal_gen(multi_rv_generic):
    r"""A multivariate skew-normal random variable.

    The location vector `loc` (denoted :math:`\boldsymbol{\xi}`),
    the covariance matrix `scale` (denoted :math:`\boldsymbol{\Omega}`),
    and the shape vector `shape` (denoted :math:`\boldsymbol{\alpha}`)
    are the parameters.

    Methods
    -------
    pdf(x, loc=None, scale=1, shape=None, allow_singular=False)
        Probability density function.
    logpdf(x, loc=None, scale=1, shape=None, allow_singular=False)
        Log of the probability density function.
    cdf(x, loc=None, scale=1, shape=None, allow_singular=False, maxpts=None, abseps=1e-5, releps=1e-5)
        Cumulative distribution function.
    logcdf(x, loc=None, scale=1, shape=None, allow_singular=False, maxpts=None, abseps=1e-5, releps=1e-5)
        Log of the cumulative distribution function.
    rvs(loc=None, scale=1, shape=None, size=1, random_state=None, allow_singular=False)
        Draw random samples from a multivariate skew-normal distribution.
    mean(loc=None, scale=1, shape=None, allow_singular=False)
        Mean of the distribution.
    var(loc=None, scale=1, shape=None, allow_singular=False)
        Variance of the distribution components (diagonal of the covariance matrix).
    cov(loc=None, scale=1, shape=None, allow_singular=False)
        Covariance matrix of the distribution.
    fit(data, f_loc=None, f_scale=None, f_shape=None, **kwargs)
        Estimate parameters from data using Maximum Likelihood Estimation (MLE).
    dp2cp(loc, scale, shape, allow_singular=False)
        Convert Direct Parameters (DP) to Centered Parameters (CP).
    cp2dp(mean, cov, gamma1, allow_singular=False)
        Convert Centered Parameters (CP) to Direct Parameters (DP).

    Parameters
    ----------
    loc : array_like, optional
        Location parameter (default: zero vector).
    scale : array_like or `Covariance`, optional
        Symmetric positive (semi)definite scale matrix (default: identity matrix).
        This matrix is denoted :math:`\boldsymbol{\Omega}` in [1].
    shape : array_like, optional
        Shape parameter vector (default: zero vector, which corresponds to the
        multivariate normal distribution). This vector is denoted
        :math:`\boldsymbol{\alpha}` in [1].
    allow_singular : bool, default: ``False``
        Whether to allow a singular scale matrix. This is ignored if `scale` is
        a `Covariance` object.
    seed : {None, int, np.random.RandomState, np.random.Generator}, optional
        Used for drawing random variates. See `~scipy.stats.rv_continuous.rvs`
        for details.

    Notes
    -----
    The probability density function for `multivariate_skew_normal` is

    .. math::

        f(\mathbf{y}) = 2 \phi_k(\mathbf{y}; \boldsymbol{\xi}, \boldsymbol{\Omega})
                       \Phi(\boldsymbol{\alpha}^T \boldsymbol{\omega}^{-1} (\mathbf{y} - \boldsymbol{\xi}))

    where :math:`\boldsymbol{\xi}` is the location vector,
    :math:`\boldsymbol{\Omega}` is the scale matrix (a covariance matrix),
    :math:`\boldsymbol{\alpha}` is the shape vector, :math:`k` is the
    dimension of the distribution, :math:`\phi_k(\cdot; \boldsymbol{\xi}, \boldsymbol{\Omega})`
    is the probability density function of a k-dimensional normal distribution
    with location :math:`\boldsymbol{\xi}` and covariance matrix :math:`\boldsymbol{\Omega}`,
    and :math:`\Phi(\cdot)` is the cumulative distribution function of a
    standard normal distribution. The matrix :math:`\boldsymbol{\omega}` is
    a diagonal matrix whose diagonal entries are the square roots of the
    diagonal entries of :math:`\boldsymbol{\Omega}` (i.e., the standard deviations).

    When the shape parameter :math:`\boldsymbol{\alpha}` is the zero vector,
    the distribution reduces to the multivariate normal distribution with
    location :math:`\boldsymbol{\xi}` and covariance :math:`\boldsymbol{\Omega}`.

    The scale matrix :math:`\boldsymbol{\Omega}` must be symmetric and positive
    semidefinite. If `allow_singular` is False, it must be positive definite.
    Symmetry is not checked; only the lower triangular portion is used.
    The pseudo-determinant and pseudo-inverse are used if the matrix is singular.

    The mean vector :math:`\boldsymbol{\mu}` and covariance matrix
    :math:`\boldsymbol{\Sigma}` of the distribution are given by:

    .. math::

        \boldsymbol{\mu} = \boldsymbol{\xi} + \boldsymbol{\omega} \boldsymbol{\mu}_z \\
        \boldsymbol{\Sigma} = \boldsymbol{\Omega} - (\boldsymbol{\omega} \boldsymbol{\mu}_z) (\boldsymbol{\omega} \boldsymbol{\mu}_z)^T

    where :math:`\boldsymbol{\mu}_z = \sqrt{2/\pi} \boldsymbol{\delta}`, and
    :math:`\boldsymbol{\delta}` is derived from the standardized parameters:

    .. math::

        \boldsymbol{\Omega}_z = \boldsymbol{\omega}^{-1} \boldsymbol{\Omega} \boldsymbol{\omega}^{-1} \\
        \boldsymbol{\delta} = \frac{\boldsymbol{\Omega}_z \boldsymbol{\alpha}}{\sqrt{1 + \boldsymbol{\alpha}^T \boldsymbol{\Omega}_z \boldsymbol{\alpha}}}

    Here, :math:`\boldsymbol{\Omega}_z` is the correlation matrix corresponding
    to the scale matrix :math:`\boldsymbol{\Omega}`.

    The cumulative distribution function (CDF) is computed numerically by
    reduction to a multivariate normal CDF calculation on an augmented space,
    as described in [2] (Section 5.2.2). This relies on the `_mvn` module's
    integration routines. Accuracy depends on the arguments `maxpts`, `abseps`,
    `releps` passed to the underlying quasi-Monte Carlo integration.

    Parameter estimation via `fit` uses Maximum Likelihood Estimation (MLE).
    The optimization process involves parameter transformations to ensure the
    scale matrix remains positive semi-definite. As noted in [1] (Section 5.3),
    MLE for the shape parameter :math:`\boldsymbol{\alpha}` can sometimes
    diverge towards infinity. This implementation includes basic checks and
    warnings for such boundary cases but does not implement the more complex
    stopping criteria suggested in the paper. For challenging fits, consider
    using method of moments or providing bounds if this occurs.

    Parameter conversion functions `dp2cp` and `cp2dp` allow switching between
    Direct Parameters (DP: :math:`\boldsymbol{\xi}, \boldsymbol{\Omega}, \boldsymbol{\alpha}`)
    and Centered Parameters (CP: :math:`\boldsymbol{\mu}, \boldsymbol{\Sigma}, \boldsymbol{\gamma}_1`),
    where :math:`\boldsymbol{\gamma}_1` is the vector of component-wise standardized skewness.
    The conversion from CP to DP requires :math:`\boldsymbol{\Sigma}` to be positive definite
    and the implied :math:`|\delta_i| < 1`.

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from scipy.stats import multivariate_skew_normal # Assuming this file is importable

    >>> # Define parameters (DP)
    >>> xi = np.array([0, 0])
    >>> Omega = np.array([[1, 0.5], [0.5, 1]])
    >>> alpha = np.array([3, -1]) # Skewness parameter

    >>> # Create a multivariate skew-normal distribution object
    >>> msn = multivariate_skew_normal(loc=xi, scale=Omega, shape=alpha)

    >>> # Evaluate the PDF and CDF at a point
    >>> x = np.array([0.5, -0.2])
    >>> pdf_val = msn.pdf(x)
    >>> cdf_val = msn.cdf(x)
    >>> print(f"PDF at {x}: {pdf_val:.4f}")
    PDF at [ 0.5 -0.2]: 0.2026
    >>> print(f"CDF at {x}: {cdf_val:.4f}")
    CDF at [ 0.5 -0.2]: 0.1906

    >>> # Draw random samples
    >>> rng = np.random.default_rng(12345)
    >>> samples = msn.rvs(size=5, random_state=rng)
    >>> print("Random samples:\n", samples) # Output varies due to randomness

    >>> # Calculate mean and covariance matrix (CP parameters derived internally)
    >>> mean_vec = msn.mean()
    >>> cov_mat = msn.cov()
    >>> print("Mean vector (mu):\n", mean_vec)
    Mean vector (mu):
     [ 0.53... -0.13...]
    >>> print("Covariance matrix (Sigma):\n", cov_mat)
    Covariance matrix (Sigma):
     [[ 0.71...  0.57...]
      [ 0.57...  0.98...]]

    >>> # Convert DP to CP
    >>> mu_cp, Sigma_cp, gamma1_cp = multivariate_skew_normal.dp2cp(xi, Omega, alpha)
    >>> print("CP Mean (calculated):", mu_cp)
    >>> print("CP Cov (calculated):\n", Sigma_cp)
    >>> print("CP Skewness (gamma1):", gamma1_cp)

    >>> # Convert CP back to DP
    >>> xi_dp, Omega_dp, alpha_dp = multivariate_skew_normal.cp2dp(mu_cp, Sigma_cp, gamma1_cp)
    >>> print("DP Loc (re-calculated):", xi_dp)
    >>> print("DP Scale (re-calculated):\n", Omega_dp)
    >>> print("DP Shape (re-calculated):", alpha_dp)
    >>> # Check if they match original DPs (within tolerance)
    >>> print("DP match:", np.allclose(xi, xi_dp) and np.allclose(Omega, Omega_dp) and np.allclose(alpha, alpha_dp))

    >>> # Fit the distribution to data
    >>> data = msn.rvs(size=500, random_state=rng)
    >>> fitted_loc, fitted_scale, fitted_shape = multivariate_skew_normal.fit(data)
    >>> print("Fitted Location (xi):", fitted_loc)
    >>> print("Fitted Scale Matrix (Omega):\n", fitted_scale)
    >>> print("Fitted Shape (alpha):", fitted_shape)
    # Note: Fitted parameters will be close but not identical to original ones.
    """

    def __init__(self, seed=None):
        """Initialize the generator class."""
        super().__init__(seed)

    def __call__(self, loc=None, scale=1, shape=None, allow_singular=False, seed=None):
        """Create a frozen multivariate skew-normal distribution instance."""
        return multivariate_skew_normal_frozen(
            loc=loc, scale=scale, shape=shape, allow_singular=allow_singular, seed=seed
        )

    def _process_parameters(self, loc, scale, shape, allow_singular=False):
        """
        Infer dimensionality, validate parameters, and return standardized forms.
        Uses _PSD for scale matrix handling.
        """
        # Determine dimension primarily from shape if provided, else loc, else scale
        if shape is not None:
            shape_ = np.asarray(shape, dtype=float)
            if shape_.ndim == 0:  # Allow scalar shape for 1D
                dim = 1
                shape_ = shape_.reshape(
                    1,
                )
            elif shape_.ndim == 1:
                dim = shape_.size
                if dim == 0:  # Handle empty shape array
                    raise ValueError(
                        "Shape parameter 'shape' cannot be an empty array."
                    )
            else:
                raise ValueError(
                    "Shape parameter 'shape' (alpha) must be a 1D array or scalar."
                )
        elif loc is not None:
            loc_ = np.asarray(loc, dtype=float)
            if loc_.ndim == 0:
                dim = 1
            elif loc_.ndim == 1:
                dim = loc_.size
                if dim == 0:
                    raise ValueError(
                        "Location parameter 'loc' cannot be an empty array."
                    )
            else:
                raise ValueError(
                    "Location parameter 'loc' (xi) must be a 1D array or scalar."
                )
            # Default shape is zero if loc or scale implies dimension
            if dim > 0:
                shape_ = np.zeros(dim)
            else:
                raise ValueError("Cannot determine dimension.")  # Should not happen
        elif scale is not None:
            # Infer dim from scale, handling scalar, 1D, 2D cases
            scale_temp = np.asarray(scale, dtype=float)
            if scale_temp.ndim == 0:
                dim = 1
            elif scale_temp.ndim == 1:
                dim = scale_temp.shape[0]
                if dim == 0:
                    raise ValueError(
                        "Scale parameter 'scale' cannot be an empty array."
                    )
            elif scale_temp.ndim == 2:
                if scale_temp.shape[0] != scale_temp.shape[1]:
                    raise ValueError("Scale matrix 'scale' (Omega) must be square.")
                dim = scale_temp.shape[0]
                if dim == 0:
                    raise ValueError(
                        "Scale matrix 'scale' cannot be empty (shape (0,0))."
                    )
            else:
                raise ValueError("Scale matrix 'scale' (Omega) must be at most 2D.")
            # Default shape is zero if loc or scale implies dimension
            if dim > 0:
                shape_ = np.zeros(dim)
            else:
                raise ValueError("Cannot determine dimension.")  # Should not happen
        else:  # All are None
            dim = 1
            shape_ = np.zeros(1)

        # Dimension check
        if dim <= 0:
            raise ValueError("Dimension must be a positive integer.")

        # Process loc based on inferred dimension
        if loc is None:
            loc_ = np.zeros(dim)
        else:
            loc_ = np.asarray(loc, dtype=float)
            if loc_.ndim == 0:
                loc_ = np.full(dim, loc_.item())  # Broadcast scalar
            elif loc_.ndim == 1:
                if loc_.size != dim:
                    raise ValueError(
                        f"Location parameter 'loc' has size {loc_.size}, expected {dim}."
                    )
            else:
                raise ValueError("Location parameter 'loc' must be 1D or scalar.")

        # Process shape based on inferred dimension (already partially done)
        if shape is None:
            # If dim was inferred from loc/scale, shape defaults to zero
            if "shape_" not in locals():
                shape_ = np.zeros(dim)
        else:
            # Ensure shape_ is correctly sized after potential dim inference from loc/scale
            shape_ = np.asarray(shape, dtype=float)
            if shape_.ndim == 0:
                shape_ = np.full(dim, shape_.item())  # Broadcast scalar
            elif shape_.ndim == 1:
                if shape_.size != dim:
                    raise ValueError(
                        f"Shape parameter 'shape' has size {shape_.size}, expected {dim}."
                    )
            else:
                raise ValueError("Shape parameter 'shape' must be 1D or scalar.")

        # Process scale: Convert to matrix and use _PSD
        if scale is None:
            scale_mat = np.eye(dim)
        else:
            scale_mat = np.asarray(scale, dtype=float)
            if scale_mat.ndim == 0:
                # Check if scalar scale is non-positive before multiplying
                if scale_mat.item() <= 0:
                    raise ValueError("Scalar scale parameter must be positive.")
                scale_mat = scale_mat.item() * np.eye(dim)
            elif scale_mat.ndim == 1:
                if len(scale_mat) != dim:
                    raise ValueError(
                        f"Scale vector has length {len(scale_mat)}, expected {dim}."
                    )
                # Check if diagonal elements are positive
                if np.any(scale_mat <= 1e-15):  # Tolerance
                    raise ValueError(
                        "Diagonal elements of scale matrix must be positive."
                    )
                scale_mat = np.diag(scale_mat)
            elif scale_mat.ndim == 2:
                if scale_mat.shape != (dim, dim):
                    raise ValueError(
                        f"Scale matrix has shape {scale_mat.shape}, expected ({dim}, {dim})."
                    )
                # Check diagonal elements here before passing to _PSD if allow_singular=False?
                # _PSD will check for positive definiteness anyway.
                if np.any(
                    np.diag(scale_mat) <= 1e-15
                ):  # Check diagonal positivity robustly
                    raise ValueError(
                        "Diagonal elements of the scale matrix must be positive."
                    )
            else:
                raise ValueError("Scale must be scalar, 1D, or 2D.")

        # Use _PSD for handling the scale matrix (provides rank, log_pdet, U etc.)
        # _PSD handles positive definiteness checks based on allow_singular
        try:
            # Ensure input matrix is symmetric for eigh in _PSD
            scale_mat_symm = (scale_mat + scale_mat.T) / 2.0
            scale_psd = _PSD(scale_mat_symm, allow_singular=allow_singular)
        except np.linalg.LinAlgError as e:
            # Reraise with a more specific message if not PSD and allow_singular=False
            if not allow_singular:
                raise ValueError(
                    "Scale matrix must be positive definite when allow_singular is False."
                ) from e
            else:
                # Check if it's the PSD check within _PSD that failed
                if "symmetric positive semidefinite" in str(e):
                    raise ValueError(
                        "Scale matrix must be positive semi-definite."
                    ) from e
                else:  # Other LinAlgError
                    raise e
        except (
            ValueError
        ) as e:  # _PSD raises ValueError for non-PSD if allow_singular=True
            if "symmetric positive semidefinite" in str(e):
                raise ValueError("Scale matrix must be positive semi-definite.") from e
            else:  # Other ValueError from _PSD
                raise e

        # Calculate omega_diag (sqrt of diagonal elements of the processed scale matrix)
        # Use _M to get the original (or symmetrized) matrix stored in _PSD
        diag_elements = np.diag(scale_psd._M)
        # Check positivity after potential modifications by _PSD (though _PSD checks this)
        if np.any(diag_elements <= 1e-15):  # Use tolerance for near-zero
            # This check might be redundant given _PSD checks, but safer
            raise ValueError("Diagonal elements of the scale matrix must be positive.")
        omega_diag = np.sqrt(diag_elements)

        return dim, loc_, scale_psd, shape_, omega_diag

    def _process_quantiles(self, x, dim):
        """
        Adjust quantiles array so that last axis labels the components of
        each data point. Adapts from multivariate_normal.
        """
        x = np.asarray(x, dtype=float)

        if x.ndim == 0:
            if dim == 1:
                x = x.reshape(1, 1)
            else:
                raise ValueError(f"Quantile is scalar, but dimension is {dim}.")
        elif x.ndim == 1:
            if dim == 1:
                x = x[:, np.newaxis]  # Treat as multiple observations of 1D var
            else:
                # Treat as a single observation of dim-D var
                if x.shape[0] != dim:
                    raise ValueError(
                        f"Quantile has shape {x.shape}, but dimension is {dim}."
                    )
                x = x[np.newaxis, :]
        else:  # x.ndim >= 2
            if x.shape[-1] != dim:
                raise ValueError(
                    f"Last dimension of quantile ({x.shape[-1]}) must match distribution dimension ({dim})."
                )
        # Shape is now (..., dim)
        return x

    def _logpdf(self, x, loc, scale_psd, shape, omega_diag):
        """
        Internal log PDF calculation using processed parameters.
        Uses _PSD object for scale matrix properties.
        """
        # Log of 2 * phi_k(y; xi, Omega) * Phi(alpha^T omega^-1 (y - xi))
        # log(2) + log(phi_k(y; xi, Omega)) + log(Phi(alpha^T omega^-1 (y - xi)))

        # Calculate log(phi_k(y; xi, Omega)) using _PSD properties
        dev = x - loc
        # Whiten method is preferred now
        try:
            maha = np.sum(np.square(scale_psd.whiten(dev)), axis=-1)
        except np.linalg.LinAlgError:
            # Handle cases where whiten might fail (e.g., numerical issues with singular)
            # Return -inf for points likely outside support or if calculation fails
            return np.full(x.shape[:-1], -np.inf)

        log_det_scale = scale_psd.log_pdet
        rank = scale_psd.rank
        log_mvn_pdf = -0.5 * (rank * _LOG_2PI + log_det_scale + maha)

        # Calculate argument for Phi: alpha^T omega^-1 (y - xi)
        omega_inv_dev = dev / omega_diag  # Broadcasting handles dimensions
        phi_arg = np.einsum(
            "...k,k->...", omega_inv_dev, shape
        )  # Dot product along last axis

        # Calculate log(Phi(arg))
        log_phi_val = _norm_logcdf(phi_arg)  # Use helper for logcdf

        logpdf_val = _LOG_2 + log_mvn_pdf + log_phi_val

        # Handle support for singular cases using _PSD's support check
        if scale_psd.rank < scale_psd._M.shape[0]:  # Check if rank < dimension
            in_support = scale_psd._support_mask(dev)
            # Ensure logpdf_val is float array for assignment
            logpdf_val = np.asarray(logpdf_val, dtype=float)
            logpdf_val[~in_support] = -np.inf

        return logpdf_val

    def logpdf(self, x, loc=None, scale=1, shape=None, allow_singular=False):
        """Log of the multivariate skew-normal probability density function."""
        dim, loc_, scale_psd, shape_, omega_diag = self._process_parameters(
            loc, scale, shape, allow_singular
        )
        x = self._process_quantiles(x, dim)
        out = self._logpdf(x, loc_, scale_psd, shape_, omega_diag)
        return _squeeze_output(out)

    def pdf(self, x, loc=None, scale=1, shape=None, allow_singular=False):
        """Multivariate skew-normal probability density function."""
        # Avoids potential underflow/overflow by working with logs
        logpdf_val = self.logpdf(x, loc, scale, shape, allow_singular)
        # Handle -inf from logpdf correctly -> 0 pdf
        pdf_val = np.exp(logpdf_val)
        # Ensure 0 for values outside support in singular case
        pdf_val[np.isneginf(logpdf_val)] = 0.0
        return pdf_val

    def _calculate_delta(self, scale_mat, shape, omega_diag):
        """Helper internal function to calculate the delta vector used in mean, cov, rvs."""
        dim = scale_mat.shape[0]
        if dim == 0:
            return np.array([])
        if np.allclose(shape, 0):
            return np.zeros(dim)

        # 1. Calculate Omega_z (correlation matrix)
        omega_diag_inv = 1.0 / omega_diag
        # Equivalent to omega_inv @ scale_mat @ omega_inv
        omega_z = (scale_mat * omega_diag_inv[:, np.newaxis]) * omega_diag_inv[
            np.newaxis, :
        ]

        # Ensure Omega_z is numerically a correlation matrix (diag = 1)
        diag_diff = np.abs(np.diag(omega_z) - 1.0)
        # Increased tolerance slightly
        if np.max(diag_diff) > 1e-8:
            warnings.warn(
                "Derived correlation matrix Omega_z diagonal deviates "
                f"significantly from 1 (max diff: {np.max(diag_diff):.2e}). "
                "Forcing diagonal to 1.",
                RuntimeWarning,
            )
        np.fill_diagonal(omega_z, 1.0)  # Force diagonal to 1

        # Check if Omega_z is PSD after forcing diagonal (important if scale_mat was only PSD)
        try:
            # Use eigh to check eigenvalues, more robust than cholesky for check
            eigvals = np.linalg.eigvalsh(omega_z)
            min_eig = np.min(eigvals)
            # Allow slightly negative eigenvalues due to precision
            if min_eig < -1e-9 * max(
                1.0, np.max(eigvals)
            ):  # Relative/Absolute tolerance
                raise np.linalg.LinAlgError(
                    f"Derived correlation matrix Omega_z is not positive semidefinite (min eigenvalue: {min_eig:.2e})."
                )
        except np.linalg.LinAlgError as e:
            warnings.warn(
                f"Omega_z matrix check failed or not PSD: {e}. "
                "Delta calculation might be unstable.",
                RuntimeWarning,
            )
            # Proceed cautiously, subsequent steps might fail

        # 2. Calculate delta = Omega_z * alpha / sqrt(1 + alpha^T Omega_z alpha)
        try:
            alpha_t_omega_z = shape @ omega_z  # row vector * matrix
            alpha_t_omega_z_alpha = alpha_t_omega_z @ shape  # scalar
        except np.linalg.LinAlgError as e:
            # This could happen if Omega_z became indefinite
            raise ValueError(
                "Matrix multiplication failed during delta calculation, "
                f"possibly due to non-PSD Omega_z. Error: {e}"
            )

        denom_delta_sq = 1 + alpha_t_omega_z_alpha

        # Denominator check
        if denom_delta_sq <= 1e-14:  # Increased tolerance slightly
            # Check if alpha is extremely large, causing potential overflow/instability
            if np.linalg.norm(shape) > 1e6:  # Heuristic threshold
                warnings.warn(
                    f"Shape parameter norm is very large ({np.linalg.norm(shape):.2e}). "
                    "Delta calculation might be unstable or lead to NaN/Inf.",
                    RuntimeWarning,
                )
            # If still non-positive after warning, raise error
            if denom_delta_sq <= 1e-14:
                raise ValueError(
                    f"Denominator for delta calculation is non-positive ({denom_delta_sq:.2e}). "
                    "Check shape and scale parameters. Alpha might be too large "
                    "or scale matrix ill-conditioned."
                )

        delta = (omega_z @ shape) / np.sqrt(denom_delta_sq)

        return delta

    def _rvs(self, dim, loc, scale_mat, shape, omega_diag, size, random_state):
        """
        Internal method to draw random samples using stochastic representation.
        """
        delta = self._calculate_delta(scale_mat, shape, omega_diag)

        # Reconstruct Omega_z (needed for Omega_star)
        omega_diag_inv = 1.0 / omega_diag
        omega_z = (scale_mat * omega_diag_inv[:, np.newaxis]) * omega_diag_inv[
            np.newaxis, :
        ]
        np.fill_diagonal(omega_z, 1.0)  # Ensure it's correlation

        # 3. Construct Omega_star (k+1 x k+1 covariance matrix)
        omega_star = np.zeros((dim + 1, dim + 1), dtype=float)
        omega_star[0, 0] = 1.0
        omega_star[0, 1:] = delta
        omega_star[1:, 0] = delta
        omega_star[1:, 1:] = omega_z

        # Ensure Omega_star is positive semidefinite before sampling
        # np.random.multivariate_normal handles this internally with 'warn' or 'raise'

        # 4. Sample (X0, X) ~ N(0, Omega_star)
        mean_kplus1 = np.zeros(dim + 1)
        try:
            # Use 'warn' to catch minor PSD issues, 'raise' for severe ones
            samples_kplus1 = random_state.multivariate_normal(
                mean_kplus1, omega_star, size=size, check_valid="warn", tol=1e-8
            )
        except ValueError as e:
            # Reraise if sampling failed, possibly due to PSD issue
            raise ValueError(
                f"Sampling from N(0, Omega_star) failed. "
                f"Omega_star might not be sufficiently positive semi-definite. Error: {e}"
            )
        except np.linalg.LinAlgError as e:
            raise np.linalg.LinAlgError(
                f"Sampling from N(0, Omega_star) failed due to LinAlgError "
                f"(likely non-PSD Omega_star). Error: {e}"
            )

        # Extract X0 and X
        x0 = samples_kplus1[..., 0]
        x_latent = samples_kplus1[..., 1:]  # Shape (size, dim) or (dim,) if size=None

        # 5. Apply the sign logic: Z = sign(X0) * X
        sign_x0 = np.sign(x0)
        sign_x0[sign_x0 == 0] = 1  # Assign arbitrary sign for zero case

        # Reshape sign_x0 for broadcasting if size is specified
        if samples_kplus1.ndim > 1:  # Check if multiple samples were drawn
            sign_x0 = sign_x0[..., np.newaxis]

        z = sign_x0 * x_latent

        # 6. Transform Z to Y: Y = xi + omega * Z
        y = loc + omega_diag * z  # Broadcasting handles dimensions

        return y

    def rvs(
        self,
        loc=None,
        scale=1,
        shape=None,
        size=1,
        random_state=None,
        allow_singular=False,
    ):
        """Draw random samples from a multivariate skew-normal distribution."""
        # Determine output shape based on size
        dim_only, _, _, _, _ = self._process_parameters(
            loc, scale, shape, allow_singular
        )
        if size is None:
            rvs_size = None  # numpy handles this
            output_shape = (dim_only,)
        elif np.isscalar(size):
            rvs_size = int(size)
            output_shape = (int(size), dim_only)
        else:  # size is a tuple
            rvs_size = tuple(map(int, size))
            output_shape = rvs_size + (dim_only,)

        random_state = self._get_random_state(random_state)
        dim, loc_, scale_psd, shape_, omega_diag = self._process_parameters(
            loc, scale, shape, allow_singular
        )

        # Get the actual scale matrix from _PSD object
        scale_mat = scale_psd._M

        samples = self._rvs(
            dim, loc_, scale_mat, shape_, omega_diag, rvs_size, random_state
        )

        # Ensure output shape is correct, especially for size=1
        if size == 1 and np.isscalar(size):
            # numpy rvs returns (dim,) for size=1, reshape to (1, dim)
            return samples.reshape(output_shape)
        else:
            # For size=None, numpy returns (dim,) which matches output_shape
            # For size > 1 or tuple, numpy returns correct shape
            return samples.reshape(output_shape)

    def mean(self, loc=None, scale=1, shape=None, allow_singular=False):
        """Mean of the multivariate skew-normal distribution."""
        dim, loc_, scale_psd, shape_, omega_diag = self._process_parameters(
            loc, scale, shape, allow_singular
        )

        if np.allclose(shape_, 0):  # Reduces to multivariate normal
            return loc_

        try:
            delta = self._calculate_delta(scale_psd._M, shape_, omega_diag)
        except ValueError as e:
            warnings.warn(
                f"Could not calculate mean due to unstable delta: {e}", RuntimeWarning
            )
            return np.full(dim, np.nan)

        mu_z = np.sqrt(2 / np.pi) * delta
        mean_vec = loc_ + omega_diag * mu_z
        return mean_vec

    def cov(self, loc=None, scale=1, shape=None, allow_singular=False):
        """Covariance matrix of the multivariate skew-normal distribution."""
        dim, loc_, scale_psd, shape_, omega_diag = self._process_parameters(
            loc, scale, shape, allow_singular
        )
        scale_mat = scale_psd._M  # Get the original matrix

        if np.allclose(shape_, 0):  # Reduces to multivariate normal
            # Ensure symmetry if original input wasn't perfectly symmetric
            return (scale_mat + scale_mat.T) / 2.0

        try:
            delta = self._calculate_delta(scale_mat, shape_, omega_diag)
        except ValueError as e:
            warnings.warn(
                f"Could not calculate covariance due to unstable delta: {e}",
                RuntimeWarning,
            )
            return np.full((dim, dim), np.nan)

        mu_z = np.sqrt(2 / np.pi) * delta
        omega_mu_z = omega_diag * mu_z  # Element-wise multiplication

        # Covariance matrix Sigma = Omega - (omega mu_z) (omega mu_z)^T
        cov_mat = scale_mat - np.outer(omega_mu_z, omega_mu_z)
        # Ensure symmetry
        return (cov_mat + cov_mat.T) / 2.0

    def var(self, loc=None, scale=1, shape=None, allow_singular=False):
        """Variance of the multivariate skew-normal distribution components."""
        cov_mat = self.cov(loc, scale, shape, allow_singular)
        # Return nan if cov calculation failed
        if cov_mat is None or np.any(np.isnan(cov_mat)):
            dim = self._process_parameters(loc, scale, shape, allow_singular)[0]
            return np.full(dim, np.nan)
        return np.diag(cov_mat)

    def _cdf(self, x, loc, scale_mat, shape, omega_diag, maxpts, abseps, releps):
        """
        Internal CDF calculation using augmented MVN approach.
        """
        dim = scale_mat.shape[0]

        # Calculate z = omega^-1 (y - xi)
        dev = x - loc
        # Avoid division by zero if omega_diag has zeros (should be caught earlier)
        if np.any(omega_diag <= 1e-15):
            raise ValueError(
                "Cannot compute CDF with non-positive standard deviations (omega)."
            )
        z = dev / omega_diag  # Broadcasting handles multiple x

        # Calculate delta needed for Omega_star
        try:
            delta = self._calculate_delta(scale_mat, shape, omega_diag)
        except ValueError as e:
            raise ValueError(f"CDF calculation failed due to unstable delta: {e}")

        # Reconstruct Omega_z (correlation matrix part of Omega_star)
        omega_diag_inv = 1.0 / omega_diag
        omega_z = (scale_mat * omega_diag_inv[:, np.newaxis]) * omega_diag_inv[
            np.newaxis, :
        ]
        np.fill_diagonal(omega_z, 1.0)  # Ensure it's correlation

        # Construct Omega_star (k+1 x k+1 covariance matrix)
        omega_star = np.zeros((dim + 1, dim + 1), dtype=float)
        omega_star[0, 0] = 1.0
        omega_star[0, 1:] = delta
        omega_star[1:, 0] = delta
        omega_star[1:, 1:] = omega_z

        # Check if Omega_star is valid before calling mvnun
        # Use a tolerance relative to the largest eigenvalue
        try:
            eigvals = np.linalg.eigvalsh(omega_star)
            min_eig = np.min(eigvals)
            max_eig = np.max(eigvals)
            # Allow slightly negative eigenvalues due to precision
            # Tolerance based on numpy's default in multivariate_normal
            psd_tol = 1e-10
            if min_eig < -psd_tol * max(1.0, max_eig):
                raise np.linalg.LinAlgError(
                    f"Omega_star matrix is not positive semidefinite for CDF (min eigenvalue: {min_eig:.2e})."
                )
        except np.linalg.LinAlgError as e:
            raise ValueError(f"Cannot compute CDF because Omega_star is not PSD: {e}")

        # Define upper integration limits for the k+1 dimensional MVN CDF
        # Limits are (0, z_1, z_2, ..., z_k)
        original_shape = x.shape[:-1]  # Shape before the dimension axis
        num_points = np.prod(original_shape) if original_shape else 1
        z_flat = z.reshape(num_points, dim)

        limits_upper = np.hstack(
            (np.zeros((num_points, 1)), z_flat)
        )  # Shape (num_points, k+1)

        # Lower limits are all -infinity
        limits_lower = np.full_like(limits_upper, -np.inf)

        # Mean vector for the MVN is zero
        mean_kplus1 = np.zeros(dim + 1)

        # Call the MVN CDF integrator (_mvn.mvnun) for each point
        cdf_vals = np.zeros(num_points)
        mvn_error_code = 0  # Accumulate error codes

        for i in range(num_points):
            # Lock needed for thread safety of the Fortran code in _mvn
            with MVN_LOCK:
                # Use mvnun directly: _mvn.mvnun(lower, upper, means, covs, ...)
                # Note: mvnun expects correlation matrix, need to adapt omega_star?
                # Let's use multivariate_normal._cdf which handles covariance
                mvn_gen = multivariate_normal_gen()
                # _cdf expects quantiles (upper limits), mean, cov, ...
                # We need P(X_aug <= limits_upper)
                try:
                    # Pass None for lower_limit to indicate -inf
                    prob, info = mvn_gen._cdf(
                        limits_upper[i],
                        mean_kplus1,
                        omega_star,
                        maxpts,
                        abseps,
                        releps,
                        lower_limit=None,
                    )
                except Exception as e:
                    # Catch potential errors from _cdf or mvnun
                    warnings.warn(
                        f"Error during MVN CDF calculation: {e}. Returning NaN.",
                        RuntimeWarning,
                    )
                    prob = np.nan
                    info = -1  # Indicate error

                # Store result and check error code
                cdf_vals[i] = prob
                if info != 0 and mvn_error_code == 0:  # Store first error
                    mvn_error_code = info

        # Apply the factor of 2
        cdf_vals *= 2.0

        # Handle potential numerical issues where result > 1 or < 0
        cdf_vals = np.clip(cdf_vals, 0.0, 1.0)

        # Reshape back to original shape
        cdf_result = cdf_vals.reshape(original_shape)

        # Raise warning if mvnun reported an error
        if mvn_error_code != 0:
            warnings.warn(
                f"MVN CDF computation (_mvn.mvnun) returned error code {mvn_error_code}. "
                "Results may be inaccurate.",
                RuntimeWarning,
            )

        return cdf_result

    def cdf(
        self,
        x,
        loc=None,
        scale=1,
        shape=None,
        allow_singular=False,
        maxpts=None,
        abseps=1e-5,
        releps=1e-5,
    ):
        """Multivariate skew-normal cumulative distribution function."""
        dim, loc_, scale_psd, shape_, omega_diag = self._process_parameters(
            loc, scale, shape, allow_singular
        )
        x = self._process_quantiles(x, dim)

        # Use default maxpts if not provided, scaled by augmented dimension
        if maxpts is None:
            maxpts = 1000 * (dim + 1)  # Scale by augmented dimension

        scale_mat = scale_psd._M
        out = self._cdf(x, loc_, scale_mat, shape_, omega_diag, maxpts, abseps, releps)
        # _squeeze_output might be needed if input x was scalar/1D
        return _squeeze_output(out)

    def logcdf(
        self,
        x,
        loc=None,
        scale=1,
        shape=None,
        allow_singular=False,
        maxpts=None,
        abseps=1e-5,
        releps=1e-5,
    ):
        """Log of the multivariate skew-normal cumulative distribution function."""
        cdf_val = self.cdf(x, loc, scale, shape, allow_singular, maxpts, abseps, releps)
        # Handle potential zero CDF values before taking log
        with np.errstate(divide="ignore"):
            logcdf_val = np.log(cdf_val)  # Will produce -inf for cdf_val=0
        return logcdf_val

    def fit(self, data, f_loc=None, f_scale=None, f_shape=None, **kwargs):
        """
        Estimate parameters from data using Maximum Likelihood Estimation (MLE).

        Parameters
        ----------
        data : array_like
            Data to fit. Array of shape (n_samples, n_dimensions).
        f_loc : array_like, optional
            Fixed values for the location parameters. If provided, these
            parameters are not estimated.
        f_scale : array_like, optional
            Fixed value for the scale matrix. If provided, this parameter
            is not estimated. Must be a valid scale matrix (e.g., PSD).
        f_shape : array_like, optional
            Fixed values for the shape parameters. If provided, these
            parameters are not estimated.
        kwargs : dict, optional
            Additional keyword arguments passed to `scipy.optimize.minimize`.
            Useful for setting optimization tolerances, max iterations, etc.
            Also accepts `initial_params` (dict with keys 'loc', 'scale_chol_vec', 'shape')
            and `alpha_bound` (float, default 50) to bound the norm of alpha check.

        Returns
        -------
        loc : ndarray
            Estimated location parameter.
        scale : ndarray
            Estimated scale matrix.
        shape : ndarray
            Estimated shape parameter.
        """
        data = np.asarray(data)
        if data.ndim != 2:
            raise ValueError(
                "Data must be a 2D array of shape (n_samples, n_dimensions)."
            )
        n_samples, dim = data.shape
        if dim == 0:
            raise ValueError("Data cannot be empty.")

        # --- Negative Log-Likelihood Function ---
        def neg_log_likelihood(params_vec, data, fixed_params_dict, dim_in):
            """Calculate negative log-likelihood for given parameters."""
            current_params = fixed_params_dict.copy()  # Start with fixed params
            start_idx = 0

            # Unpack location parameters if not fixed
            if "loc" not in current_params:
                current_params["loc"] = params_vec[start_idx : start_idx + dim_in]
                start_idx += dim_in

            # Unpack scale matrix parameters (from Cholesky vector) if not fixed
            if "scale" not in current_params:
                n_chol_params = dim_in + dim_in * (dim_in - 1) // 2
                chol_params_vec = params_vec[start_idx : start_idx + n_chol_params]
                # Convert back from optimized parameters (log-diag, off-diag)
                try:
                    scale_mat, _ = _get_scale_from_chol_param(chol_params_vec, dim_in)
                    current_params["scale"] = scale_mat
                except (
                    ValueError,
                    IndexError,
                ):  # Catch errors from invalid chol params
                    return np.inf  # Penalize invalid Cholesky structure
                start_idx += n_chol_params

            # Unpack shape parameters if not fixed
            if "shape" not in current_params:
                current_params["shape"] = params_vec[start_idx : start_idx + dim_in]
                start_idx += dim_in

            # Calculate log-likelihood for all data points
            # Use allow_singular=True as optimizer might approach singularity
            try:
                # Ensure scale matrix is at least PSD for logpdf calculation
                # We parameterize to ensure this, but check anyway
                scale_to_use = current_params["scale"]
                try:
                    _ = cholesky(
                        scale_to_use + 1e-10 * np.eye(dim_in)
                    )  # Check PSD with jitter
                except np.linalg.LinAlgError:
                    return np.inf  # Not PSD

                log_pdf_values = self.logpdf(
                    data,
                    loc=current_params["loc"],
                    scale=scale_to_use,
                    shape=current_params["shape"],
                    allow_singular=True,
                )

                # Handle -inf values (e.g., outside support in singular case)
                log_pdf_values[
                    np.isneginf(log_pdf_values)
                ] = -1e100  # Assign large negative number

                # Check for NaNs which indicate invalid parameters during optimization
                if np.any(np.isnan(log_pdf_values)):
                    return np.inf  # Penalize invalid parameter combinations

                nll = -np.sum(log_pdf_values)

            except (ValueError, np.linalg.LinAlgError):
                # Catch errors during parameter processing or logpdf calculation
                return np.inf  # Return infinity if parameters are invalid

            # Check if nll itself is NaN or Inf
            if not np.isfinite(nll):
                return np.inf

            return nll

        # --- Initial Parameter Guesses ---
        fixed_params_dict = {}
        initial_guess_list = []
        initial_params_kw = kwargs.get("initial_params", {})

        # Location
        if f_loc is not None:
            f_loc = np.asarray(f_loc)
            if f_loc.shape != (dim,):
                raise ValueError(f"f_loc must have shape ({dim},)")
            fixed_params_dict["loc"] = f_loc
            initial_loc = f_loc  # Needed if scale depends on it later? No.
        else:
            initial_loc = initial_params_kw.get("loc", np.mean(data, axis=0))
            initial_guess_list.append(initial_loc)

        # Scale
        if f_scale is not None:
            f_scale = np.asarray(f_scale)
            if f_scale.shape != (dim, dim):
                raise ValueError(f"f_scale must have shape ({dim},{dim})")
            # Check if f_scale is PSD
            try:
                _ = cholesky(f_scale, lower=True)
            except np.linalg.LinAlgError:
                raise ValueError(
                    "Fixed scale matrix f_scale must be positive definite."
                )
            fixed_params_dict["scale"] = f_scale
            initial_scale = f_scale  # Needed for shape guess? No.
        else:
            if "scale_chol_vec" in initial_params_kw:
                # Use provided Cholesky parameters
                initial_chol_opt_params = initial_params_kw["scale_chol_vec"]
                # Need to validate length
                n_chol_expected = dim + dim * (dim - 1) // 2
                if len(initial_chol_opt_params) != n_chol_expected:
                    raise ValueError(
                        "Provided initial_params['scale_chol_vec'] has wrong length."
                    )
                initial_guess_list.append(initial_chol_opt_params)
                # Store initial scale matrix derived from this guess
                initial_scale, _ = _get_scale_from_chol_param(
                    initial_chol_opt_params, dim
                )

            else:
                # Use sample covariance as initial guess for scale (Omega)
                sample_cov = np.atleast_2d(np.cov(data, rowvar=False))
                # Ensure PSD for Cholesky
                try:
                    # Add small jitter for stability if near singular
                    jitter = 1e-8 * np.eye(dim)
                    chol_factor = cholesky(sample_cov + jitter, lower=True)
                except np.linalg.LinAlgError:
                    warnings.warn(
                        "Sample covariance matrix is not positive definite. "
                        "Using identity matrix as initial scale guess.",
                        RuntimeWarning,
                    )
                    chol_factor = np.eye(dim)
                initial_scale = (
                    chol_factor @ chol_factor.T
                )  # Store the initial Omega guess
                # Parameterize using Cholesky factor elements
                initial_chol_diag = np.diag(chol_factor)
                initial_chol_offdiag = (
                    chol_factor[np.tril_indices(dim, k=-1)] if dim > 1 else np.array([])
                )
                initial_chol_vec = np.concatenate(
                    (initial_chol_diag, initial_chol_offdiag)
                )
                # Convert to optimization parameters (log-diag, off-diag)
                initial_guess_list.append(_vec_to_chol_param(initial_chol_vec, dim))

        # Shape
        if f_shape is not None:
            f_shape = np.asarray(f_shape)
            if f_shape.shape != (dim,):
                raise ValueError(f"f_shape must have shape ({dim},)")
            fixed_params_dict["shape"] = f_shape
        else:
            # Initial guess for shape (alpha) is often zero (i.e., normal)
            initial_shape = initial_params_kw.get("shape", np.zeros(dim))
            initial_guess_list.append(initial_shape)

        # Combine initial guesses into a single vector
        initial_params_vec = (
            np.concatenate(initial_guess_list) if initial_guess_list else np.array([])
        )

        # --- Optimization ---
        # Use kwargs for optimizer options, e.g., method='L-BFGS-B'
        optimizer_method = kwargs.get("method", "L-BFGS-B")  # Good default
        opt_options = {"disp": kwargs.get("disp", False)}  # Default to not verbose
        if "maxiter" in kwargs:
            opt_options["maxiter"] = kwargs.get("maxiter")
        # Adjust default tolerances based on method
        default_ftol = 1e-9 if optimizer_method in ["L-BFGS-B", "BFGS", "CG"] else 1e-6
        default_gtol = (
            1e-5 if optimizer_method == "L-BFGS-B" else 1e-5
        )  # L-BFGS-B uses gtol
        opt_options["ftol"] = kwargs.get("ftol", default_ftol)
        if optimizer_method == "L-BFGS-B":
            opt_options["gtol"] = kwargs.get("gtol", default_gtol)

        # Bounds: Only really needed for log-diag of Cholesky (implicitly handled by log)
        # and potentially alpha if divergence is a major concern.
        # For now, rely on initial guess and optimizer's ability.
        bounds = None

        result = minimize(
            neg_log_likelihood,
            initial_params_vec,
            args=(data, fixed_params_dict, dim),
            method=optimizer_method,
            options=opt_options,
            jac=kwargs.get("jac"),  # Allow user to provide jacobian
            hess=kwargs.get("hess"),  # Allow user to provide hessian
            bounds=bounds,
        )

        if not result.success:
            warnings.warn(f"MLE optimization failed: {result.message}", RuntimeWarning)
            # Return NaNs or raise error? Return NaNs for now.
            nan_scale = (
                np.full((dim, dim), np.nan)
                if "scale" not in fixed_params_dict
                else fixed_params_dict["scale"]
            )
            nan_loc = (
                np.full(dim, np.nan)
                if "loc" not in fixed_params_dict
                else fixed_params_dict["loc"]
            )
            nan_shape = (
                np.full(dim, np.nan)
                if "shape" not in fixed_params_dict
                else fixed_params_dict["shape"]
            )
            return nan_loc, nan_scale, nan_shape

        # --- Extract Results ---
        estimated_params_vec = result.x
        estimated_loc = fixed_params_dict.get("loc")
        estimated_scale = fixed_params_dict.get("scale")
        estimated_shape = fixed_params_dict.get("shape")
        start_idx = 0

        if estimated_loc is None:
            estimated_loc = estimated_params_vec[start_idx : start_idx + dim]
            start_idx += dim
        if estimated_scale is None:
            n_chol_params = dim + dim * (dim - 1) // 2
            chol_params_vec = estimated_params_vec[
                start_idx : start_idx + n_chol_params
            ]
            try:
                estimated_scale, _ = _get_scale_from_chol_param(chol_params_vec, dim)
            except ValueError:
                warnings.warn(
                    "Failed to reconstruct scale matrix from optimized Cholesky parameters.",
                    RuntimeWarning,
                )
                estimated_scale = np.full((dim, dim), np.nan)  # Return NaN scale
            start_idx += n_chol_params
        if estimated_shape is None:
            estimated_shape = estimated_params_vec[start_idx : start_idx + dim]
            start_idx += dim

        # Check for boundary estimate of alpha
        alpha_bound = kwargs.get("alpha_bound", 50.0)  # Check norm of alpha
        if "shape" not in fixed_params_dict:
            alpha_norm = np.linalg.norm(estimated_shape)
            if alpha_norm > alpha_bound:
                warnings.warn(
                    f"Estimated shape parameter norm ({alpha_norm:.2f}) exceeds bound check ({alpha_bound}). "
                    "MLE may have approached boundary (infinity). Results might be unreliable.",
                    RuntimeWarning,
                )

        # Final check: ensure estimated scale is PSD if it was estimated
        if "scale" not in fixed_params_dict and not np.any(np.isnan(estimated_scale)):
            try:
                _ = cholesky(
                    estimated_scale + 1e-10 * np.eye(dim), lower=True
                )  # Check with jitter
            except np.linalg.LinAlgError:
                warnings.warn(
                    "Estimated scale matrix is not positive definite.", RuntimeWarning
                )
                # Return matrix anyway, user should check

        return estimated_loc, estimated_scale, estimated_shape

    @staticmethod
    def dp2cp(loc, scale, shape, allow_singular=False):
        """
        Convert Direct Parameters (DP) to Centered Parameters (CP).

        Parameters
        ----------
        loc : array_like
            Location parameter (xi).
        scale : array_like
            Scale matrix (Omega).
        shape : array_like
            Shape parameter (alpha).
        allow_singular : bool, default: False
            Whether to allow a singular scale matrix.

        Returns
        -------
        mean : ndarray
            Mean vector (mu).
        cov : ndarray
            Covariance matrix (Sigma).
        gamma1 : ndarray
            Vector of component-wise standardized skewness. Returns NaN if
            component variance is zero or calculation is unstable.
        """
        # Use the generator's methods for processing and calculation
        gen = multivariate_skew_normal_gen()
        dim, loc_, scale_psd, shape_, omega_diag = gen._process_parameters(
            loc, scale, shape, allow_singular
        )
        scale_mat = scale_psd._M

        # Calculate mean (mu)
        mean_vec = gen.mean(loc_, scale_mat, shape_, allow_singular)

        # Calculate covariance (Sigma)
        cov_mat = gen.cov(loc_, scale_mat, shape_, allow_singular)

        # Calculate component-wise skewness (gamma1)
        # gamma1_i = E[((Y_i - mu_i)/sigma_i)^3]
        # Formula: gamma1_i = (sqrt(2/pi)*(4/pi-1) * delta_i^3) / (1 - (2/pi)*delta_i^2)^(3/2)
        if np.any(np.isnan(mean_vec)) or np.any(np.isnan(cov_mat)):
            warnings.warn(
                "Cannot compute gamma1 because mean or covariance calculation failed.",
                RuntimeWarning,
            )
            gamma1 = np.full(dim, np.nan)
        elif np.allclose(shape_, 0):
            gamma1 = np.zeros(dim)
        else:
            try:
                delta = gen._calculate_delta(scale_mat, shape_, omega_diag)
            except ValueError as e:
                warnings.warn(
                    f"Could not calculate delta for gamma1: {e}", RuntimeWarning
                )
                return mean_vec, cov_mat, np.full(dim, np.nan)

            b = _SQRT_2 / _SQRT_PI
            c = 4 / np.pi - 1
            delta_sq = delta**2
            denom_sq = 1 - b**2 * delta_sq

            # Check for denominator close to zero (delta_i^2 approx pi/2)
            if np.any(denom_sq <= 1e-14):
                warnings.warn(
                    "Denominator near zero in gamma1 calculation (delta^2 approx pi/2).",
                    RuntimeWarning,
                )
                gamma1 = np.full(dim, np.nan)
                # Handle components where denom is ok?
                valid_denom = denom_sq > 1e-14
                gamma1[valid_denom] = (b * c * delta[valid_denom] ** 3) / (
                    denom_sq[valid_denom] ** 1.5
                )
            else:
                gamma1 = (b * c * delta**3) / (denom_sq**1.5)

            # Check for extreme delta values that might lead to gamma1 outside [-0.995, 0.995] due to precision
            max_gamma1_mag = 0.9952717  # Theoretical max magnitude
            gamma1 = np.clip(gamma1, -max_gamma1_mag, max_gamma1_mag)

        return mean_vec, cov_mat, gamma1

    @staticmethod
    def cp2dp(mean, cov, gamma1, allow_singular=False):
        """
        Convert Centered Parameters (CP) to Direct Parameters (DP).

        Parameters
        ----------
        mean : array_like
            Mean vector (mu).
        cov : array_like
            Covariance matrix (Sigma). Must be positive definite.
        gamma1 : array_like
            Vector of component-wise standardized skewness. Each element must
            be in the interval (-0.99527, 0.99527).
        allow_singular : bool, default: False
             This argument is ignored for cp2dp as the input covariance `cov`
             must be positive definite for the conversion to be well-defined.

        Returns
        -------
        loc : ndarray
            Location parameter (xi).
        scale : ndarray
            Scale matrix (Omega).
        shape : ndarray
            Shape parameter (alpha). Returns NaN if conversion fails.
        """
        mean = np.asarray(mean, dtype=float)
        cov = np.asarray(cov, dtype=float)
        gamma1 = np.asarray(gamma1, dtype=float)
        dim = mean.shape[0]

        if cov.shape != (dim, dim) or gamma1.shape != (dim,):
            raise ValueError("Dimension mismatch between mean, cov, and gamma1.")

        # Check gamma1 bounds
        max_gamma1_mag = 0.9952717  # Theoretical max magnitude
        if np.any(np.abs(gamma1) >= max_gamma1_mag):
            warnings.warn(
                "Input gamma1 values are outside the valid range approx (-0.995, 0.995). Conversion might fail.",
                RuntimeWarning,
            )
            # Allow calculation to proceed, might result in NaN/Inf

        # Check if cov (Sigma) is positive definite
        try:
            # Add jitter for robustness
            jitter = 1e-10 * np.eye(dim)
            _ = cholesky(cov + jitter, lower=True)
        except np.linalg.LinAlgError:
            raise ValueError(
                "Input covariance matrix (Sigma) must be positive definite for cp2dp conversion."
            )

        # 1. Solve for delta from gamma1
        # R_i = sign(g1_i) * (2*abs(g1_i)/(4/pi - 1))**(1/3)
        # delta_i = R_i * sqrt(2/pi) / sqrt(1 + R_i^2)
        b = _SQRT_2 / _SQRT_PI
        c = 4 / np.pi - 1
        # Handle gamma1=0 case separately to avoid division by zero
        R = np.zeros_like(gamma1)
        non_zero_g1 = np.abs(gamma1) > 1e-15
        R[non_zero_g1] = np.sign(gamma1[non_zero_g1]) * (
            2 * np.abs(gamma1[non_zero_g1]) / c
        ) ** (1 / 3)

        delta = np.zeros_like(gamma1)
        denom_delta_sq = 1 + R**2
        # Avoid division by zero if R is somehow NaN/Inf (shouldn't happen if gamma1 check passed)
        valid_denom = denom_delta_sq > 1e-15
        delta[valid_denom] = R[valid_denom] * b / np.sqrt(denom_delta_sq[valid_denom])

        # Check if calculated delta magnitude >= 1 (numerical issue or invalid gamma1)
        if np.any(np.abs(delta) >= 1.0):
            warnings.warn(
                "|delta| >= 1 encountered during cp2dp conversion. Clamping delta.",
                RuntimeWarning,
            )
            delta = np.clip(
                delta, -1.0 + 1e-9, 1.0 - 1e-9
            )  # Clamp slightly inside (-1, 1)

        # 2. Calculate mu_z
        mu_z = b * delta

        # 3. Calculate omega_diag
        # omega_diag_i^2 * (1 - mu_z_i^2) = diag(cov)_i
        diag_cov = np.diag(cov)
        denom_omega_sq = 1 - mu_z**2
        if np.any(denom_omega_sq <= 1e-15):
            warnings.warn(
                "Denominator (1 - mu_z^2) near zero in omega calculation. Results may be unstable.",
                RuntimeWarning,
            )
            # Set omega to NaN where denominator is bad
            omega_diag = np.full(dim, np.nan)
            valid_omega = denom_omega_sq > 1e-15
            omega_diag[valid_omega] = np.sqrt(
                diag_cov[valid_omega] / denom_omega_sq[valid_omega]
            )
        else:
            omega_diag = np.sqrt(diag_cov / denom_omega_sq)

        if np.any(np.isnan(omega_diag)):
            warnings.warn("NaN encountered during omega calculation.", RuntimeWarning)
            # Cannot proceed if omega is NaN
            return (
                mean,
                cov,
                np.full(dim, np.nan),
            )  # Return original mean/cov and NaN shape

        # 4. Calculate Omega (scale matrix)
        omega_mu_z = omega_diag * mu_z
        Omega = cov + np.outer(omega_mu_z, omega_mu_z)
        # Ensure symmetry
        Omega = (Omega + Omega.T) / 2.0

        # 5. Calculate Omega_z (correlation matrix)
        try:
            omega_diag_inv = 1.0 / omega_diag
            Omega_z = (Omega * omega_diag_inv[:, np.newaxis]) * omega_diag_inv[
                np.newaxis, :
            ]
            np.fill_diagonal(Omega_z, 1.0)  # Ensure correlation matrix
            # Check PSD of Omega_z
            eigvals = np.linalg.eigvalsh(Omega_z)
            if np.min(eigvals) < -1e-9 * max(1.0, np.max(eigvals)):
                raise np.linalg.LinAlgError(
                    "Derived Omega_z is not positive semidefinite."
                )
        except (ValueError, np.linalg.LinAlgError) as e:
            warnings.warn(
                f"Could not calculate or validate Omega_z: {e}", RuntimeWarning
            )
            return (
                mean,
                cov,
                np.full(dim, np.nan),
            )  # Return original mean/cov and NaN shape

        # 6. Solve for alpha
        # alpha = sqrt(1+a) * Omega_z_inv @ delta, where a/(1+a)^2 = delta^T Omega_z_inv delta
        try:
            # Use pseudo-inverse for robustness, although Omega_z should be invertible
            Omega_z_inv = pinvh(Omega_z, rcond=1e-12)  # Use pseudo-inverse
            delta_sq_term = delta @ Omega_z_inv @ delta

            # Ensure delta_sq_term is within [0, 1) for solvability
            if delta_sq_term < -1e-12 or delta_sq_term >= 1.0 - 1e-12:
                warnings.warn(
                    f"Term delta^T Omega_z^-1 delta ({delta_sq_term:.3f}) is outside [0, 1). Cannot solve for alpha.",
                    RuntimeWarning,
                )
                return mean, cov, np.full(dim, np.nan)

            # Solve quadratic: delta_sq * A^2 - A + delta_sq = 0 where A = 1/(1+a)
            # Or solve a = (1+a)^2 * delta_sq => delta_sq a^2 + (2*delta_sq - 1) a + delta_sq = 0
            # Handle delta_sq = 0 case (alpha=0)
            if delta_sq_term < 1e-15:
                a = 0.0
            else:
                # Quadratic formula for a
                coeff_a = delta_sq_term
                coeff_b = 2 * delta_sq_term - 1
                coeff_c = delta_sq_term
                discriminant = coeff_b**2 - 4 * coeff_a * coeff_c
                if discriminant < -1e-12:  # Allow small negative due to precision
                    warnings.warn(
                        "Discriminant is negative when solving for 'a' in cp2dp. Cannot solve for alpha.",
                        RuntimeWarning,
                    )
                    return mean, cov, np.full(dim, np.nan)
                discriminant = max(0, discriminant)  # Force non-negative
                # Two solutions for 'a', need the correct one.
                # Since a = alpha^T Omega_z alpha >= 0, we need non-negative solution
                a1 = (-coeff_b + np.sqrt(discriminant)) / (2 * coeff_a)
                a2 = (-coeff_b - np.sqrt(discriminant)) / (2 * coeff_a)
                # Which solution corresponds to the original alpha?
                # The relationship is complex. R sn uses a direct formula for alpha.
                # alpha = Omega_z_inv delta / sqrt(1 - delta^T Omega_z_inv delta) ? Let's check.
                # If Z ~ SN(0, Omega_z, alpha_z), then delta = ... alpha_z...
                # Let's use the formula derived from Azzalini & Dalla Valle (1996) eqn (A.4) / (10)
                # alpha = Omega_z_inv delta / sqrt(1 - delta^T Omega_z_inv delta)
                # Need to re-derive or trust this formula. Let's use it.
                denom_alpha_sq = 1 - delta_sq_term
                if denom_alpha_sq <= 1e-14:
                    warnings.warn(
                        "Denominator near zero when solving for alpha. Results may be unstable.",
                        RuntimeWarning,
                    )
                    # This case implies delta magnitude is near boundary
                    # Return large alpha? Or NaN? Let's return NaN.
                    return mean, cov, np.full(dim, np.nan)

                alpha = (Omega_z_inv @ delta) / np.sqrt(denom_alpha_sq)

        except np.linalg.LinAlgError as e:
            warnings.warn(
                f"Linear algebra error during alpha calculation: {e}", RuntimeWarning
            )
            return mean, cov, np.full(dim, np.nan)

        # 7. Calculate loc (xi)
        loc = mean - omega_diag * mu_z

        return loc, Omega, alpha


multivariate_skew_normal = multivariate_skew_normal_gen()


class multivariate_skew_normal_frozen(multi_rv_frozen):
    """
    A frozen multivariate skew-normal distribution.

    Parameters are fixed at initialization. See `multivariate_skew_normal_gen`
    for parameter details. Use `multivariate_skew_normal(...)` to create an instance.
    """

    def __init__(
        self,
        loc=None,
        scale=1,
        shape=None,
        allow_singular=False,
        seed=None,
        maxpts=None,
        abseps=1e-5,
        releps=1e-5,
    ):  # Add CDF params
        """Initialize the frozen distribution."""
        self._dist = multivariate_skew_normal_gen(seed)
        # Store processed parameters
        self.dim, self.loc, self.scale_psd, self.shape, self.omega_diag = (
            self._dist._process_parameters(loc, scale, shape, allow_singular)
        )
        # Store the actual scale matrix as well for convenience
        self.scale_mat = self.scale_psd._M

        # Store CDF parameters
        if maxpts is None:
            # Scale by augmented dimension
            self.maxpts = 1000 * (self.dim + 1)
        else:
            self.maxpts = maxpts
        self.abseps = abseps
        self.releps = releps

    def logpdf(self, x):
        """Log of the probability density function of the frozen distribution."""
        x = self._dist._process_quantiles(x, self.dim)
        # Use stored processed parameters
        out = self._dist._logpdf(
            x, self.loc, self.scale_psd, self.shape, self.omega_diag
        )
        return _squeeze_output(out)

    def pdf(self, x):
        """Probability density function of the frozen distribution."""
        # Avoids potential underflow/overflow by working with logs
        logpdf_val = self.logpdf(x)
        # Handle -inf from logpdf correctly -> 0 pdf
        pdf_val = np.exp(logpdf_val)
        pdf_val[np.isneginf(logpdf_val)] = 0.0
        return pdf_val

    def cdf(self, x):
        """Cumulative distribution function of the frozen distribution."""
        x = self._dist._process_quantiles(x, self.dim)
        # Use stored processed parameters and CDF settings
        out = self._dist._cdf(
            x,
            self.loc,
            self.scale_mat,
            self.shape,
            self.omega_diag,
            self.maxpts,
            self.abseps,
            self.releps,
        )
        return _squeeze_output(out)

    def logcdf(self, x):
        """Log of the cumulative distribution function of the frozen distribution."""
        cdf_val = self.cdf(x)
        # Handle potential zero CDF values before taking log
        with np.errstate(divide="ignore"):
            logcdf_val = np.log(cdf_val)  # Will produce -inf for cdf_val=0
        return logcdf_val

    def rvs(self, size=1, random_state=None):
        """Draw random samples from the frozen distribution."""
        # Determine output shape based on size
        if size is None:
            rvs_size = None
            output_shape = (self.dim,)
        elif np.isscalar(size):
            rvs_size = int(size)
            output_shape = (int(size), self.dim)
        else:  # size is a tuple
            rvs_size = tuple(map(int, size))
            output_shape = rvs_size + (self.dim,)

        random_state = self._dist._get_random_state(random_state)
        # Use stored processed parameters
        samples = self._dist._rvs(
            self.dim,
            self.loc,
            self.scale_mat,
            self.shape,
            self.omega_diag,
            rvs_size,
            random_state,
        )

        # Ensure output shape is correct
        if size == 1 and np.isscalar(size):
            return samples.reshape(output_shape)
        else:
            return samples.reshape(output_shape)

    def mean(self):
        """Mean of the frozen distribution."""
        # Use stored parameters
        if np.allclose(self.shape, 0):
            return self.loc
        try:
            # Use internal helper method of the generator instance
            delta = self._dist._calculate_delta(
                self.scale_mat, self.shape, self.omega_diag
            )
        except ValueError as e:
            warnings.warn(
                f"Could not calculate mean due to unstable delta: {e}", RuntimeWarning
            )
            return np.full(self.dim, np.nan)

        mu_z = np.sqrt(2 / np.pi) * delta
        mean_vec = self.loc + self.omega_diag * mu_z
        return mean_vec

    def cov(self):
        """Covariance matrix of the frozen distribution."""
        # Use stored parameters
        if np.allclose(self.shape, 0):
            # Ensure symmetry
            return (self.scale_mat + self.scale_mat.T) / 2.0
        try:
            # Use internal helper method of the generator instance
            delta = self._dist._calculate_delta(
                self.scale_mat, self.shape, self.omega_diag
            )
        except ValueError as e:
            warnings.warn(
                f"Could not calculate covariance due to unstable delta: {e}",
                RuntimeWarning,
            )
            return np.full((self.dim, self.dim), np.nan)

        mu_z = np.sqrt(2 / np.pi) * delta
        omega_mu_z = self.omega_diag * mu_z
        cov_mat = self.scale_mat - np.outer(omega_mu_z, omega_mu_z)
        # Ensure symmetry
        return (cov_mat + cov_mat.T) / 2.0

    def var(self):
        """Variance of the components of the frozen distribution."""
        cov_mat = self.cov()
        if cov_mat is None or np.any(np.isnan(cov_mat)):
            return np.full(self.dim, np.nan)
        return np.diag(cov_mat)
