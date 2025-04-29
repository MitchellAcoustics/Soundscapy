import numpy as np
import pandas as pd
import soundscapy.spi._rsn_wrapper as rsn
import soundscapy as sspy
from soundscapy.spi.ks2d import ks2d2s
from test.plotting.test_plotting import sample_data


class DirectParams:
    """
    Represents a set of direct parameters for a statistical model.

    Direct parameters are the parameters that are directly used in the model. They are the parameters that
    are used to define the distribution of the data. In the case of a skew normal distribution, the direct
    parameters are the xi, omega, and alpha values.

    Attributes:
        xi (np.ndarray): The location of the distribution in 2D space, represented as a 2x1 array
            with the x and y coordinates.
        omega (np.ndarray): The covariance matrix of the distribution, represented as a 2x2 array.
            The covariance matrix represents the measure of the relationship between different variables.
            It provides information about how changes in one variable are associated with changes in other variables.
        alpha (np.ndarray): The shape parameters for the x and y dimensions, controlling the shape (skewness) of the distribution.
            It is represented as a 2x1 array.
    """

    def __init__(self, xi: np.ndarray, omega: np.ndarray, alpha: np.ndarray):
        self.xi = xi
        self.omega = omega
        self.alpha = alpha
        self.validate()

    def __repr__(self) -> str:
        return f"DirectParams(xi={self.xi}, omega={self.omega}, alpha={self.alpha})"

    def __str__(self) -> str:
        return (
            f"Direct Parameters:"
            f"\nxi:    {self.xi.round(3)}"
            f"\nomega: {self.omega.round(3)}"
            f"\nalpha: {self.alpha.round(3)}"
        )

    def _omega_is_pos_def(self) -> bool:
        return np.all(np.linalg.eigvals(self.omega) > 0)

    def _omega_is_symmetric(self) -> bool:
        return np.allclose(self.omega, self.omega.T)

    def _xi_is_in_range(self, xi_range: np.ndarray | tuple[float, float]) -> bool:
        if isinstance(xi_range, tuple):
            xi_range = np.array([xi_range, xi_range])
        return np.all((xi_range[:, 0] <= self.xi) & (self.xi <= xi_range[:, 1]))

    def validate(self):
        """
        Validate the direct parameters.

        In a skew normal distribution, the covariance matrix, often denoted as Ω (Omega), represents the
        measure of the relationship between different variables. It provides information about how changes
        in one variable are associated with changes in other variables. The covariance matrix must be positive
        definite and symmetric.

        Raises:
            AssertionError: If the direct parameters are not valid.

        Returns:
            None
        """
        assert self._omega_is_pos_def(), "Omega must be positive definite"
        assert self._omega_is_symmetric(), "Omega must be symmetric"


class CentredParams:
    """
    Represents the centered parameters of a distribution.

    Attributes:
        mean (float): The mean of the distribution.
        sigma (float): The standard deviation of the distribution.
        skew (float): The skewness of the distribution.

    Methods:
        __init__(mean, sigma, skew): Initializes a new instance of the CentredParams class.
        __repr__(): Returns a string representation of the CentredParams object.
        __str__(): Returns a formatted string representation of the CentredParams object.
        from_dp(dp): Converts DirectParams object to CentredParams object.

    """

    def __init__(self, mean, sigma, skew):
        self.mean = mean
        self.sigma = sigma
        self.skew = skew

    def __repr__(self):
        return f"CentredParams(mean={self.mean}, sigma={self.sigma}, skew={self.skew})"

    def __str__(self) -> str:
        return (
            f"Centred Parameters:"
            f"\nmean:  {self.mean.round(3)}"
            f"\nsigma: {self.sigma.round(3)}"
            f"\nskew:  {self.skew.round(3)}"
        )

    @classmethod
    def from_dp(cls, dp: DirectParams):
        """
        Converts a DirectParams object to a CentredParams object.

        Args:
            dp (DirectParams): The DirectParams object to convert.

        Returns:
            CentredParams: A new CentredParams object with the converted parameters.

        """
        cp = dp2cp(dp)
        return cls(cp.mean, cp.sigma, cp.skew)


class MultiSkewNorm:
    """
    A class representing a multi-dimensional skewed normal distribution.

    Attributes:
        selm_model: The fitted SELM model.
        cp: The centred parameters of the fitted model.
        dp: The direct parameters of the fitted model.
        sample_data: The generated sample data from the fitted model.
        data: The input data used for fitting the model.

    Methods:
        __init__: Initializes an instance of the MultiSkewNorm class.
        __repr__: Returns a string representation of the MultiSkewNorm instance.
        summary: Prints a summary of the fitted model.
        fit: Fits the model to the provided data.
        define_dp: Defines the direct parameters of the model.
        sample: Generates a sample from the fitted model.
        sspy_plot: Plots the joint distribution of the generated sample.
        ks2ds: Computes the two-sample Kolmogorov-Smirnov statistic.
        spi: Computes the similarity percentage index.

    """

    def __init__(self):
        self.selm_model = None
        self.cp = None
        self.dp = None
        self.sample_data = None
        self.data: pd.DataFrame | None = None

    def __repr__(self):
        if self.cp is None and self.dp is None and self.selm_model is None:
            return "MultiSkewNorm() (unfitted)"
        return f"MultiSkewNorm(dp={self.dp})"

    def summary(self):
        if self.cp is None and self.dp is None and self.selm_model is None:
            return "MultiSkewNorm is not fitted."
        else:
            if self.data is not None:
                print(f"Fitted from data. n = {len(self.data)}")
            else:
                print("Fitted from direct parameters.")
            print(self.dp)
            print("\n")
            print(self.cp)

    def fit(
        self,
        data: pd.DataFrame | np.ndarray | None = None,
        x: np.ndarray | pd.Series | None = None,
        y: np.ndarray | pd.Series | None = None,
    ):
        """
        Fits the multi-dimensional skewed normal model to the provided data.

        Args:
            data: The input data as a pandas DataFrame or numpy array.
            x: The x-values of the input data as a numpy array or pandas Series.
            y: The y-values of the input data as a numpy array or pandas Series.

        Raises:
            ValueError: If either data or x and y are not provided.

        """

        if data is None and (x is None or y is None):
            # Either data or x and y must be provided
            raise ValueError("Either data or x and y must be provided")

        if data is not None:
            # If data is provided, convert it to a pandas DataFrame
            if isinstance(data, pd.DataFrame):
                # If data is already a DataFrame, no need to convert
                df = data
                df.columns = ["x", "y"]

            elif isinstance(data, np.ndarray):
                # If data is a numpy array, convert it to a DataFrame
                if data.ndim == 2:
                    # If data is 2D, assume it's two variables
                    df = pd.DataFrame(data, columns=["x", "y"])
                else:
                    raise ValueError("Data must be a 2D numpy array or DataFrame")
            else:
                # If data is neither a DataFrame nor a numpy array, raise an error
                raise ValueError("Data must be a pandas DataFrame or 2D numpy array.")

        elif x is not None and y is not None:
            # If x and y are provided, convert them to a pandas DataFrame
            df = pd.DataFrame({"x": x, "y": y})

        else:
            # This should never happen
            raise ValueError("Either data or x and y must be provided")

        # Fit the model
        m = rsn.selm("x", "y", df)

        # Extract the parameters
        cp = rsn.extract_cp(m)
        dp = rsn.extract_dp(m)

        self.cp = CentredParams(*cp)
        self.dp = DirectParams(*dp)
        self.data = df
        self.selm_model = m

        return None

    def define_dp(self, xi: np.ndarray, omega: np.ndarray, alpha: np.ndarray):
        """
        Initiate a distribution from the direct parameters.

        Args:
            xi: The xi values of the direct parameters as a numpy array.
            omega: The omega values of the direct parameters as a numpy array.
            alpha: The alpha values of the direct parameters as a numpy array.

        """

        self.dp = DirectParams(xi, omega, alpha)
        self.cp = CentredParams.from_dp(self.dp)
        return self

    def sample(self, n: int = 1000, return_sample: bool = False) -> None | np.ndarray:
        """
        Generates a sample from the fitted model.

        Args:
            n: The number of samples to generate.
            return_sample: Whether to return the generated sample as an np.ndarray.

        Returns:
            None or numpy array: The generated sample if return_sample is True.

        Raises:
            ValueError: If either selm_model or xi, omega, and alpha are not provided.

        """

        if self.selm_model is not None:
            sample = rsn.sample_msn(selm_model=self.selm_model, n=n)
        elif self.dp is not None:
            sample = rsn.sample_msn(
                xi=self.dp.xi, omega=self.dp.omega, alpha=self.dp.alpha, n=n
            )
        else:
            raise ValueError(
                "Either selm_model or xi, omega, and alpha must be provided."
            )

        self.sample_data = sample

        if return_sample:
            return sample

    def sspy_plot(self, color: str = "blue", title: str | None = None, n: int = 1000):
        """
        Plots the joint distribution of the generated sample.

        """

        if self.sample_data is None:
            self.sample(n=n)

        df = pd.DataFrame(self.sample_data, columns=["ISOPleasant", "ISOEventful"])
        sspy.density_plot(df, color=color, title=title)

    def ks2ds(self, test: pd.DataFrame | np.ndarray):
        """
        Computes the two-sample, two-dimensional Kolmogorov-Smirnov statistic.

        Args:
            test: The test data as a pandas DataFrame or numpy array.
            nboot: The number of bootstrap samples to use for computing the p-value.
            extra: Whether to compute the extra statistics.

        Returns:
            tuple: The KS2D statistic, p-value, and extra statistics (if extra=True).

        """

        if self.sample_data is None:
            self.sample()
        if isinstance(self.sample_data, pd.DataFrame):
            sample_data = self.sample_data.values
        else:
            sample_data = self.sample_data

        if isinstance(test, pd.DataFrame):
            assert test.shape[1] == 2, "Test data must have two columns."
            test = test.values

        return ks2d2s(sample_data, test)  # type: ignore

    def spi(self, test: pd.DataFrame | np.ndarray):
        """
        Computes the Soundscape Perception Index (SPI) for the test data against the target distribution.

        Args:
            test: The test data as a pandas DataFrame or numpy array.

        Returns:
            int: The Soundscape Perception Index

        """

        return int((1 - self.ks2ds(test)[0]) * 100)


def cp2dp(cp: CentredParams, family: str = "SN") -> DirectParams:
    """
    Converts centred parameters to direct parameters.

    Args:
        cp: The centred parameters as a CentredParams object.

    Returns:
        DirectParams: The direct parameters as a DirectParams object.

    """
    dp_r = rsn._dp2cp(cp.mean, cp.sigma, cp.skew, family=family)

    return DirectParams(*dp_r)


def dp2cp(dp: DirectParams, family: str = "SN") -> CentredParams:
    """
    Converts direct parameters to centred parameters.

    Args:
        dp: The direct parameters as a DirectParams object.

    Returns:
        CentredParams: The centred parameters as a CentredParams object.

    """
    cp_r = rsn._dp2cp(dp.xi, dp.omega, dp.alpha, family=family)

    return CentredParams(*cp_r)
