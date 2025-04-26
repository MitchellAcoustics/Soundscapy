from abc import ABC
from dataclasses import dataclass
import numpy as np
from soundscapy.spi._r_wrapper import numpy2R



# @dataclass
# class MSNParams_(ABC):
#     """
#     Base class for the parameters of the SELM model.
#     """
#     xi: np.ndarray | float | tuple[float, float]
#     omega: np.ndarray
#     alpha: np.ndarray | float | tuple[float, float]

#     def to_r(self) -> ListVector:
#         """Convert the parameters to R format."""
#         return ListVector({"xi": self.xi, "omega": self.omega, "alpha": self.alpha})

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
    
    @

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

    def to_r(self):
        """
        Convert the DirectParams object to an R-compatible format.

        Returns:
            tuple: A tuple containing xi, omega, and alpha in a format compatible with R.
        """
        r_xi = numpy2R(self.xi)
        r_omega = numpy2R(self.omega)
        r_alpha = numpy2R(self.alpha)
        return r_xi, r_omega, r_alpha


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

    def from_dp(self, dp: DirectParams):
        """
        Converts a DirectParams object to a CentredParams object.

        Args:
            dp (DirectParams): The DirectParams object to convert.

        Returns:
            CentredParams: A new CentredParams object with the converted parameters.

        Note:
            This method is not yet implemented.

        """
        # TODO: Implement this method with dp2cp function
        pass

    def to_r(self):
        """
        Convert the CentredParams object to an R-compatible format.

        Returns:
            tuple: A tuple containing mean, sigma, and skew in a format compatible with R.
        """
        r_mean = numpy2R(self.mean)
        r_sigma = numpy2R(self.sigma)
        r_skew = numpy2R(self.skew)
        return r_mean, r_sigma, r_skew
