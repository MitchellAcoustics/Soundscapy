"""
Module for handling Multi-dimensional Skewed Normal (MSN) distributions.

Provides classes and functions for defining, fitting, sampling, and analyzing
MSN distributions, often used in soundscape analysis for modeling ISOPleasant
and ISOEventful ratings.
"""

from typing import Literal

import numpy as np
import pandas as pd

from soundscapy.plotting import density_plot

from . import _rsn_wrapper as rsn
from .ks2d import ks2d2s


class DirectParams:
    """
    Represents a set of direct parameters for a statistical model.

    Direct parameters are the parameters that are directly used in the model.
    They are the parameters that are used to define the distribution of the
    data. In the case of a skew normal distribution, the direct parameters
    are the xi, omega, and alpha values.

    Parameters
    ----------
    xi : np.ndarray
        The location of the distribution in 2D space, represented as a 2x1 array
        with the x and y coordinates.
    omega : np.ndarray
        The covariance matrix of the distribution, represented as a 2x2 array.
        The covariance matrix represents the measure of the relationship between
        different variables. It provides information about how changes in one
        variable are associated with changes in other variables.
    alpha : np.ndarray
        The shape parameters for the x and y dimensions, controlling the shape
        (skewness) of the distribution. It is represented as a 2x1 array.

    """

    def __init__(self, xi: np.ndarray, omega: np.ndarray, alpha: np.ndarray) -> None:
        """Initialize DirectParams instance."""
        self.xi = xi
        self.omega = omega
        self.alpha = alpha
        self.validate()

    def __repr__(self) -> str:
        """Return a string representation of the DirectParams object."""
        return f"DirectParams(xi={self.xi}, omega={self.omega}, alpha={self.alpha})"

    def __str__(self) -> str:
        """Return a user-friendly string representation of the DirectParams object."""
        return (
            f"Direct Parameters:"
            f"\nxi:    {self.xi.round(3)}"
            f"\nomega: {self.omega.round(3)}"
            f"\nalpha: {self.alpha.round(3)}"
        )

    def _omega_is_pos_def(self) -> bool:
        return bool(np.all(np.linalg.eigvals(self.omega) > 0))

    def _omega_is_symmetric(self) -> bool:
        return np.allclose(self.omega, self.omega.T)

    def _xi_is_in_range(self, xi_range: np.ndarray | tuple[float, float]) -> bool:
        if isinstance(xi_range, tuple):
            xi_range = np.array([xi_range, xi_range])
        return bool(np.all((xi_range[:, 0] <= self.xi) & (self.xi <= xi_range[:, 1])))

    def validate(self) -> None:
        """
        Validate the direct parameters.

        In a skew normal distribution, the covariance matrix, often denoted as
        Ω (Omega), represents the measure of the relationship between different
        variables. It provides information about how changes in one variable are
        associated with changes in other variables. The covariance matrix must
        be positive definite and symmetric.

        Raises
        ------
        ValueError
            If the direct parameters are not valid.

        Returns
        -------
        None

        """
        if not self._omega_is_pos_def():
            msg = "Omega must be positive definite"
            raise ValueError(msg)
        if not self._omega_is_symmetric():
            msg = "Omega must be symmetric"
            raise ValueError(msg)


class CentredParams:
    """
    Represents the centered parameters of a distribution.

    Parameters
    ----------
    mean : float
        The mean of the distribution.
    sigma : float
        The standard deviation of the distribution.
    skew : float
        The skewness of the distribution.

    Attributes
    ----------
    mean : float
        The mean of the distribution.
    sigma : float
        The standard deviation of the distribution.
    skew : float
        The skewness of the distribution.

    Methods
    -------
    from_dp(dp)
        Converts DirectParams object to CentredParams object.

    """

    def __init__(self, mean: np.ndarray, sigma: np.ndarray, skew: np.ndarray) -> None:
        """Initialize CentredParams instance."""
        self.mean = mean
        self.sigma = sigma
        self.skew = skew

    def __repr__(self) -> str:
        """Return a string representation of the CentredParams object."""
        return f"CentredParams(mean={self.mean}, sigma={self.sigma}, skew={self.skew})"

    def __str__(self) -> str:
        """Return a user-friendly string representation of the CentredParams object."""
        return (
            f"Centred Parameters:"
            f"\nmean:  {self.mean.round(3)}"
            f"\nsigma: {self.sigma.round(3)}"
            f"\nskew:  {self.skew.round(3)}"
        )

    @classmethod
    def from_dp(cls, dp: DirectParams) -> "CentredParams":
        """
        Convert a DirectParams object to a CentredParams object.

        Parameters
        ----------
        dp : DirectParams
            The DirectParams object to convert.

        Returns
        -------
        CentredParams
            A new CentredParams object with the converted parameters.

        """
        cp = dp2cp(dp)
        return cls(cp.mean, cp.sigma, cp.skew)


class MultiSkewNorm:
    """
    A class representing a multi-dimensional skewed normal distribution.

    Attributes
    ----------
    selm_model
        The fitted SELM model.
    cp : CentredParams
        The centred parameters of the fitted model.
    dp : DirectParams
        The direct parameters of the fitted model.
    sample_data : np.ndarray | None
        The generated sample data from the fitted model.
    data : pd.DataFrame | None
        The input data used for fitting the model.

    Methods
    -------
    summary()
        Prints a summary of the fitted model.
    fit(data=None, x=None, y=None)
        Fits the model to the provided data.
    define_dp(xi, omega, alpha)
        Defines the direct parameters of the model.
    sample(n=1000, return_sample=False)
        Generates a sample from the fitted model.
    sspy_plot(color='blue', title=None, n=1000)
        Plots the joint distribution of the generated sample.
    ks2ds(test)
        Computes the two-sample Kolmogorov-Smirnov statistic.
    spi(test)
        Computes the similarity percentage index.

    """

    def __init__(self) -> None:
        """Initialize the MultiSkewNorm object."""
        self.selm_model = None
        self.cp = None
        self.dp = None
        self.sample_data = None
        self.data: pd.DataFrame | None = None

    def __repr__(self) -> str:
        """Return a string representation of the MultiSkewNorm object."""
        if self.cp is None and self.dp is None and self.selm_model is None:
            return "MultiSkewNorm() (unfitted)"
        return f"MultiSkewNorm(dp={self.dp})"

    def summary(self) -> str | None:
        """
        Provide a summary of the fitted MultiSkewNorm model.

        Returns
        -------
        str or None
            A string summarizing the model parameters and data, or a message
            indicating the model is not fitted. Returns None if fitted but
            summary logic is not fully implemented yet.

        """
        if self.cp is None and self.dp is None and self.selm_model is None:
            return "MultiSkewNorm is not fitted."
        if self.data is not None:
            pass
        else:
            pass
        return None

    def fit(
        self,
        data: pd.DataFrame | np.ndarray | None = None,
        x: np.ndarray | pd.Series | None = None,
        y: np.ndarray | pd.Series | None = None,
    ) -> None:
        """
        Fit the multi-dimensional skewed normal model to the provided data.

        Parameters
        ----------
        data : pd.DataFrame or np.ndarray, optional
            The input data as a pandas DataFrame or numpy array.
        x : np.ndarray or pd.Series, optional
            The x-values of the input data as a numpy array or pandas Series.
        y : np.ndarray or pd.Series, optional
            The y-values of the input data as a numpy array or pandas Series.

        Raises
        ------
        ValueError
            If neither `data` nor both `x` and `y` are provided.

        """
        if data is None and (x is None or y is None):
            # Either data or x and y must be provided
            msg = "Either data or x and y must be provided"
            raise ValueError(msg)

        if data is not None:
            # If data is provided, convert it to a pandas DataFrame
            if isinstance(data, pd.DataFrame):
                # If data is already a DataFrame, no need to convert
                data.columns = ["x", "y"]

            elif isinstance(data, np.ndarray):
                # If data is a numpy array, convert it to a DataFrame
                if data.ndim == 2:  # noqa: PLR2004
                    # If data is 2D, assume it's two variables
                    data = pd.DataFrame(data, columns=["x", "y"])
                else:
                    msg = "Data must be a 2D numpy array or DataFrame"
                    raise ValueError(msg)
            else:
                # If data is neither a DataFrame nor a numpy array, raise an error
                msg = "Data must be a pandas DataFrame or 2D numpy array."
                raise ValueError(msg)

        elif x is not None and y is not None:
            # If x and y are provided, convert them to a pandas DataFrame
            data = pd.DataFrame({"x": x, "y": y})

        else:
            # This should never happen
            msg = "Either data or x and y must be provided"
            raise ValueError(msg)

        # Fit the model
        m = rsn.selm("x", "y", data)

        # Extract the parameters
        cp = rsn.extract_cp(m)
        dp = rsn.extract_dp(m)

        self.cp = CentredParams(*cp)
        self.dp = DirectParams(*dp)
        self.data = data
        self.selm_model = m

    def define_dp(
        self, xi: np.ndarray, omega: np.ndarray, alpha: np.ndarray
    ) -> "MultiSkewNorm":
        """
        Initiate a distribution from the direct parameters.

        Parameters
        ----------
        xi : np.ndarray
            The xi values of the direct parameters.
        omega : np.ndarray
            The omega values of the direct parameters.
        alpha : np.ndarray
            The alpha values of the direct parameters.

        Returns
        -------
        self

        """
        self.dp = DirectParams(xi, omega, alpha)
        self.cp = CentredParams.from_dp(self.dp)
        return self

    def sample(
        self, n: int = 1000, *, return_sample: bool = False
    ) -> None | np.ndarray:
        """
        Generate a sample from the fitted model.

        Parameters
        ----------
        n : int, optional
            The number of samples to generate, by default 1000.
        return_sample : bool, optional
            Whether to return the generated sample as an np.ndarray, by default False.

        Returns
        -------
        None or np.ndarray
            The generated sample if `return_sample` is True, otherwise None.

        Raises
        ------
        ValueError
            If the model is not fitted (i.e., `selm_model` is None) and direct
            parameters (`dp`) are also not defined.

        """
        if self.selm_model is not None:
            sample = rsn.sample_msn(selm_model=self.selm_model, n=n)
        elif self.dp is not None:
            sample = rsn.sample_msn(
                xi=self.dp.xi, omega=self.dp.omega, alpha=self.dp.alpha, n=n
            )
        else:
            msg = "Either selm_model or xi, omega, and alpha must be provided."
            raise ValueError(msg)

        self.sample_data = sample

        if return_sample:
            return sample
        return None

    def sample_mtsn(
        self, n: int = 1000, a: float = -1, b: float = 1, *, return_sample: bool = False
    ) -> None | np.ndarray:
        """
        Generate a sample from the multi-dimensional truncated skew-normal distribution.

        Uses rejection sampling to ensure that the samples are within the bounds [a, b]
        for both dimensions.

        Parameters
        ----------
        n : int, optional
            The number of samples to generate, by default 1000.
        a : float, optional
            Lower truncation bound for both dimensions, by default -1.
        b : float, optional
            Upper truncation bound for both dimensions, by default 1.
        return_sample : bool, optional
            Whether to return the generated sample as an np.ndarray, by default False.

        Returns
        -------
        None or np.ndarray
            The generated sample if `return_sample` is True, otherwise None.

        """
        if self.selm_model is not None:
            sample = rsn.sample_mtsn(
                selm_model=self.selm_model,
                n=n,
                a=a,
                b=b,
            )
        elif self.dp is not None:
            sample = rsn.sample_mtsn(
                xi=self.dp.xi,
                omega=self.dp.omega,
                alpha=self.dp.alpha,
                n=n,
                a=a,
                b=b,
            )
        else:
            msg = "Either selm_model or xi, omega, and alpha must be provided."
            raise ValueError(msg)

        # Store the sample data
        self.sample_data = sample

        if return_sample:
            return sample
        return None

    def sspy_plot(
        self, color: str = "blue", title: str | None = None, n: int = 1000
    ) -> None:
        """
        Plot the joint distribution of the generated sample using soundscapy.

        Parameters
        ----------
        color : str, optional
            Color for the density plot, by default "blue".
        title : str, optional
            Title for the plot, by default None.
        n : int, optional
            Number of samples to generate if `sample_data` is None, by default 1000.

        """
        if self.sample_data is None:
            self.sample(n=n)

        data = pd.DataFrame(self.sample_data, columns=["ISOPleasant", "ISOEventful"])
        plot_title = title if title is not None else "Soundscapy Density Plot"
        density_plot(data, color=color, title=plot_title)

    def ks2ds(self, test: pd.DataFrame | np.ndarray) -> tuple[float, float]:
        """
        Compute the two-sample, two-dimensional Kolmogorov-Smirnov statistic.

        Parameters
        ----------
        test : pd.DataFrame or np.ndarray
            The test data.

        Returns
        -------
        tuple
            The KS2D statistic and p-value.

        """
        if self.sample_data is None:
            self.sample()

        # Ensure sample_data is populated after calling self.sample()
        if self.sample_data is None:
            msg = "Failed to generate sample data."
            raise ValueError(msg)

        if isinstance(self.sample_data, pd.DataFrame):
            sample_data = self.sample_data.to_numpy()
        else:
            # Explicitly cast to ndarray to satisfy type checker,
            # although it should be one already
            sample_data = np.asarray(self.sample_data)

        if isinstance(test, pd.DataFrame):
            if test.shape[1] != 2:  # noqa: PLR2004
                msg = "Test data must have two columns."
                raise ValueError(msg)
            test = test.to_numpy()

        return ks2d2s(sample_data, test)

    def spi(self, test: pd.DataFrame | np.ndarray) -> int:
        """
        Compute the Soundscape Perception Index (SPI).

        Calculates the SPI for the test data against the target distribution
        represented by this MultiSkewNorm instance.

        Parameters
        ----------
        test : pd.DataFrame or np.ndarray
            The test data.

        Returns
        -------
        int
            The Soundscape Perception Index (SPI), ranging from 0 to 100.

        """
        return int((1 - self.ks2ds(test)[0]) * 100)


def cp2dp(
    cp: CentredParams, family: Literal["SN", "ESN", "ST", "SC"] = "SN"
) -> DirectParams:
    """
    Convert centred parameters to direct parameters.

    Parameters
    ----------
    cp : CentredParams
        The centred parameters object.
    family : str, optional
        The distribution family, by default "SN" (Skew Normal).

    Returns
    -------
    DirectParams
        The corresponding direct parameters object.

    """
    dp_r = rsn.cp2dp(cp.mean, cp.sigma, cp.skew, family=family)

    return DirectParams(*dp_r)


def dp2cp(
    dp: DirectParams, family: Literal["SN", "ESN", "ST", "SC"] = "SN"
) -> CentredParams:
    """
    Convert direct parameters to centred parameters.

    Parameters
    ----------
    dp : DirectParams
        The direct parameters object.
    family : str, optional
        The distribution family, by default "SN" (Skew Normal).

    Returns
    -------
    CentredParams
        The corresponding centred parameters object.

    """
    cp_r = rsn.dp2cp(dp.xi, dp.omega, dp.alpha, family=family)

    return CentredParams(*cp_r)
