from unittest.mock import patch  # Keep patch for plotting

import numpy as np
import pandas as pd
import pytest
from numpy.random import default_rng

from soundscapy.spi.msn import CentredParams, DirectParams, MultiSkewNorm, cp2dp, dp2cp

# Check for R and 'sn' package availability
try:
    from rpy2.rinterface_lib.embedded import RRuntimeError
    from rpy2.robjects.packages import importr

    # Try importing the 'sn' package in R
    try:
        importr("sn")
        r_sn_available = True
    except RRuntimeError:
        r_sn_available = False
except ImportError:
    r_sn_available = False

needs_r_sn = pytest.mark.skipif(
    not r_sn_available,
    reason="Requires R, rpy2, and the R 'sn' package to be installed.",
)

rng = default_rng(42)  # Set a random seed for reproducibility


class TestDirectParams:
    def test_direct_params_init_valid(self):
        """Test initialization with valid parameters."""
        xi = np.array([0.1, 0.2])
        omega = np.array([[1.0, 0.5], [0.5, 1.0]])  # Symmetric and positive definite
        alpha = np.array([0.3, 0.4])

        dp = DirectParams(xi, omega, alpha)

        assert np.array_equal(dp.xi, xi)
        assert np.array_equal(dp.omega, omega)
        assert np.array_equal(dp.alpha, alpha)

    def test_direct_params_init_not_pos_def(self):
        """Test initialization with omega that is not positive definite."""
        xi = np.array([0.1, 0.2])
        # Not positive definite matrix
        omega = np.array([[1.0, 2.0], [2.0, 1.0]])
        alpha = np.array([0.3, 0.4])

        with pytest.raises(ValueError, match="Omega must be positive definite"):
            DirectParams(xi, omega, alpha)

    def test_direct_params_init_not_symmetric(self):
        """Test initialization with omega that is not symmetric."""
        xi = np.array([0.1, 0.2])
        # Not symmetric matrix
        omega = np.array([[1.0, 0.5], [0.6, 1.0]])
        alpha = np.array([0.3, 0.4])

        with pytest.raises(ValueError, match="Omega must be symmetric"):
            DirectParams(xi, omega, alpha)

    def test_direct_params_repr(self):
        """Test the __repr__ method."""
        xi = np.array([0.1, 0.2])
        omega = np.array([[1.0, 0.5], [0.5, 1.0]])
        alpha = np.array([0.3, 0.4])

        dp = DirectParams(xi, omega, alpha)

        assert repr(dp) == f"DirectParams(xi={xi}, omega={omega}, alpha={alpha})"

    def test_direct_params_str(self):
        """Test the __str__ method."""
        xi = np.array([0.1, 0.2])
        omega = np.array([[1.0, 0.5], [0.5, 1.0]])
        alpha = np.array([0.3, 0.4])

        dp = DirectParams(xi, omega, alpha)

        expected_str = (
            f"Direct Parameters:"
            f"\nxi:    {xi.round(3)}"
            f"\nomega: {omega.round(3)}"
            f"\nalpha: {alpha.round(3)}"
        )

        assert str(dp) == expected_str

    def test_xi_is_in_range(self):
        """Test the _xi_is_in_range method with tuple input."""
        xi = np.array([0.1, 0.2])
        omega = np.array([[1.0, 0.5], [0.5, 1.0]])
        alpha = np.array([0.3, 0.4])

        dp = DirectParams(xi, omega, alpha)

        # Test with tuple input
        assert dp._xi_is_in_range((-1.0, 1.0))
        assert not dp._xi_is_in_range((0.2, 1.0))


class TestCentredParams:
    def test_centred_params_init(self):
        """Test initialization of CentredParams."""
        mean = np.array([0.5, 0.6])
        sigma = np.array([1.0, 1.2])
        skew = np.array([0.1, -0.1])

        cp = CentredParams(mean, sigma, skew)

        assert np.array_equal(cp.mean, mean)
        assert np.array_equal(cp.sigma, sigma)
        assert np.array_equal(cp.skew, skew)

    def test_centred_params_repr(self):
        """Test the __repr__ method."""
        mean = np.array([0.5, 0.6])
        sigma = np.array([1.0, 1.2])
        skew = np.array([0.1, -0.1])

        cp = CentredParams(mean, sigma, skew)

        expected_repr = f"CentredParams(mean={mean}, sigma={sigma}, skew={skew})"
        assert repr(cp) == expected_repr

    def test_centred_params_str(self):
        """Test the __str__ method."""
        mean = np.array([0.5123, 0.6789])
        sigma = np.array([1.0456, 1.2789])
        skew = np.array([0.1111, -0.1999])

        cp = CentredParams(mean, sigma, skew)

        expected_str = (
            f"Centred Parameters:"
            f"\nmean:  {mean.round(3)}"
            f"\nsigma: {sigma.round(3)}"
            f"\nskew:  {skew.round(3)}"
        )
        assert str(cp) == expected_str

    @needs_r_sn
    def test_centred_params_from_dp(self):
        """Test the from_dp class method."""
        # Create a dummy DirectParams object
        dp_xi = np.array([0.1, 0.2])
        dp_omega = np.array([[1.0, 0.5], [0.5, 1.0]])
        dp_alpha = np.array([0.3, 0.4])
        dp = DirectParams(dp_xi, dp_omega, dp_alpha)

        # Expected values calculated from a known R execution or previous test
        expected_cp = CentredParams(
            mean=np.array([0.44083939, 0.57492333]),
            sigma=np.array([[0.88382851, 0.37221136], [0.37221136, 0.8594325]]),
            skew=np.array([0.02045318, 0.02839051]),
        )

        # Call the class method (which internally calls dp2cp)
        cp_from_dp = CentredParams.from_dp(dp)

        # Assert that the returned object is an instance of CentredParams
        assert isinstance(cp_from_dp, CentredParams)

        # Assert that the attributes match the expected values
        np.testing.assert_allclose(cp_from_dp.mean, expected_cp.mean, atol=1e-5)
        # Assuming sigma in CentredParams now holds the covariance matrix from dp2cp
        np.testing.assert_allclose(cp_from_dp.sigma, expected_cp.sigma, atol=1e-5)
        np.testing.assert_allclose(cp_from_dp.skew, expected_cp.skew, atol=1e-5)


# Mock data and parameters for testing
# Use values consistent with TestCentredParams.test_centred_params_from_dp
MOCK_XI = np.array([0.1, 0.2])
MOCK_OMEGA = np.array([[1.0, 0.5], [0.5, 1.0]])
MOCK_ALPHA = np.array([0.3, 0.4])
# Corresponding CP values (mean, covariance, skew) from dp2cp
EXPECTED_MEAN = np.array([0.44083939, 0.57492333])
EXPECTED_SIGMA_COV = np.array([[0.88382851, 0.37221136], [0.37221136, 0.8594325]])
EXPECTED_SKEW = np.array([0.02045318, 0.02839051])

# Sample data for fitting tests
MOCK_DF = pd.DataFrame(
    rng.random((50, 2)) * 0.5 + 0.1, columns=["x", "y"]
)  # Smaller N for faster fit
MOCK_X = MOCK_DF["x"].to_numpy()
MOCK_Y = MOCK_DF["y"].to_numpy()
MOCK_SAMPLE_SIZE = 100


@needs_r_sn
class TestMultiSkewNorm:
    def test_init(self):
        """Test initialization of MultiSkewNorm."""
        msn = MultiSkewNorm()
        assert msn.selm_model is None
        assert msn.cp is None
        assert msn.dp is None
        assert msn.sample_data is None
        assert msn.data is None

    def test_repr_unfitted(self):
        """Test __repr__ when the model is not fitted."""
        msn = MultiSkewNorm()
        assert repr(msn) == "MultiSkewNorm() (unfitted)"

    def test_repr_fitted(self):
        """Test __repr__ when the model is fitted."""
        msn = MultiSkewNorm()
        # Define DP, which implicitly calculates CP via dp2cp
        msn.define_dp(MOCK_XI, MOCK_OMEGA, MOCK_ALPHA)
        # The repr should now show the DP
        assert repr(msn) == f"MultiSkewNorm(dp={msn.dp})"

    def test_summary_unfitted(self):  # No mock needed
        """Test summary when the model is not fitted."""
        msn = MultiSkewNorm()
        assert msn.summary() == "MultiSkewNorm is not fitted."

    @needs_r_sn  # Needs fit -> R
    def test_summary_fitted_from_data(self, capsys):
        """Test summary when the model is fitted from data."""
        msn = MultiSkewNorm()
        msn.fit(data=MOCK_DF.copy())
        msn.summary()
        captured = capsys.readouterr()
        assert f"Fitted from data. n = {len(MOCK_DF)}" in captured.out
        assert "Direct Parameters:" in captured.out
        assert "Centred Parameters:" in captured.out
        assert "xi:" in captured.out
        assert "mean:" in captured.out

    def test_summary_fitted_from_dp(self, capsys):
        """Test summary when the model is fitted from direct parameters."""
        msn = MultiSkewNorm()
        msn.define_dp(MOCK_XI, MOCK_OMEGA, MOCK_ALPHA)  # This calculates CP
        msn.summary()
        captured = capsys.readouterr()
        assert "Fitted from direct parameters." in captured.out
        assert str(msn.dp) in captured.out
        assert str(msn.cp) in captured.out

    def test_fit_with_dataframe(self):
        """Test fit method with pandas DataFrame."""
        msn = MultiSkewNorm()
        msn.fit(data=MOCK_DF.copy())

        assert msn.selm_model is not None  # Check R model object exists
        assert isinstance(msn.cp, CentredParams)
        assert isinstance(msn.dp, DirectParams)
        assert msn.data is not None  # Add assertion for type checker
        pd.testing.assert_frame_equal(msn.data, MOCK_DF)
        # Check dimensions of parameters
        assert msn.cp.mean.shape == (2,)
        assert msn.dp.xi.shape == (2,)
        assert msn.dp.omega.shape == (2, 2)
        assert msn.dp.alpha.shape == (2,)

    def test_fit_with_numpy_array(self):
        """Test fit method with numpy array."""
        msn = MultiSkewNorm()
        numpy_data = MOCK_DF.values
        msn.fit(data=numpy_data)

        expected_df = pd.DataFrame(numpy_data, columns=["x", "y"])

        assert msn.selm_model is not None
        assert isinstance(msn.cp, CentredParams)
        assert isinstance(msn.dp, DirectParams)
        assert msn.data is not None  # Add assertion for type checker
        pd.testing.assert_frame_equal(msn.data, expected_df)
        assert msn.cp.mean.shape == (2,)
        assert msn.dp.xi.shape == (2,)

    def test_fit_with_1d_numpy_array(self):
        """Test fit method raises error on 1D numpy array."""
        msn = MultiSkewNorm()
        one_d_array = np.array([0.1, 0.2, 0.3])

        with pytest.raises(
            ValueError, match="Data must be a 2D numpy array or DataFrame"
        ):
            msn.fit(data=one_d_array)

    def test_fit_with_x_y(self):
        """Test fit method with x and y arrays."""
        msn = MultiSkewNorm()
        msn.fit(x=MOCK_X, y=MOCK_Y)

        expected_df = pd.DataFrame({"x": MOCK_X, "y": MOCK_Y})

        assert msn.selm_model is not None
        assert isinstance(msn.cp, CentredParams)
        assert isinstance(msn.dp, DirectParams)
        assert msn.data is not None  # Add assertion for type checker
        pd.testing.assert_frame_equal(msn.data, expected_df)
        assert msn.cp.mean.shape == (2,)
        assert msn.dp.xi.shape == (2,)

    def test_fit_no_data(self):
        """Test fit method raises ValueError when no data is provided."""
        msn = MultiSkewNorm()
        with pytest.raises(ValueError, match="Either data or x and y must be provided"):
            msn.fit()

    def test_define_dp(self):
        """Test define_dp method."""
        msn = MultiSkewNorm()
        result = msn.define_dp(MOCK_XI, MOCK_OMEGA, MOCK_ALPHA)

        assert isinstance(msn.dp, DirectParams)
        np.testing.assert_array_equal(msn.dp.xi, MOCK_XI)
        np.testing.assert_array_equal(msn.dp.omega, MOCK_OMEGA)
        np.testing.assert_array_equal(msn.dp.alpha, MOCK_ALPHA)
        # Check CP was also calculated
        assert isinstance(msn.cp, CentredParams)
        np.testing.assert_allclose(msn.cp.mean, EXPECTED_MEAN, atol=1e-5)
        # Assuming CentredParams.sigma holds covariance matrix after dp2cp
        np.testing.assert_allclose(msn.cp.sigma, EXPECTED_SIGMA_COV, atol=1e-5)
        np.testing.assert_allclose(msn.cp.skew, EXPECTED_SKEW, atol=1e-5)
        assert result is msn  # Check if it returns self for chaining

    def test_sample_after_fit(self):
        """Test sample method after fitting the model."""
        msn = MultiSkewNorm()
        msn.fit(data=MOCK_DF)
        result = msn.sample(n=MOCK_SAMPLE_SIZE, return_sample=False)

        assert result is None
        assert isinstance(msn.sample_data, np.ndarray)
        assert msn.sample_data.shape == (MOCK_SAMPLE_SIZE, 2)

    def test_sample_after_define_dp(self):
        """Test sample method after defining direct parameters."""
        msn = MultiSkewNorm()
        msn.define_dp(MOCK_XI, MOCK_OMEGA, MOCK_ALPHA)
        result = msn.sample(n=MOCK_SAMPLE_SIZE, return_sample=False)

        assert result is None
        assert isinstance(msn.sample_data, np.ndarray)
        assert msn.sample_data.shape == (MOCK_SAMPLE_SIZE, 2)

    def test_sample_return_sample_true(self):
        """Test sample method with return_sample=True."""
        msn = MultiSkewNorm()
        msn.define_dp(MOCK_XI, MOCK_OMEGA, MOCK_ALPHA)
        sample = msn.sample(n=MOCK_SAMPLE_SIZE, return_sample=True)

        assert isinstance(sample, np.ndarray)
        assert sample.shape == (MOCK_SAMPLE_SIZE, 2)
        # Ensure sample_data is also set
        assert isinstance(msn.sample_data, np.ndarray)
        np.testing.assert_array_equal(msn.sample_data, sample)

    def test_sample_not_fitted_or_defined(self):  # No mock needed
        """Test sample method raises ValueError when not fitted or defined."""
        msn = MultiSkewNorm()
        with pytest.raises(
            ValueError,
            match="Either selm_model or xi, omega, and alpha must be provided.",
        ):
            msn.sample()

    @patch("soundscapy.spi.msn.scatter")  # Keep mocking the plotting call
    def test_sspy_plot_calls_sample_if_needed(self, mock_scatter):
        """Test sspy_plot calls sample if sample_data is None."""
        msn = MultiSkewNorm()
        msn.define_dp(MOCK_XI, MOCK_OMEGA, MOCK_ALPHA)  # Define DP so sample can run

        assert msn.sample_data is None
        msn.sspy_plot(
            n=MOCK_SAMPLE_SIZE, color="red", title="Test Plot"
        )  # Pass n to sample

        # Check sample was called implicitly and data was generated
        assert isinstance(msn.sample_data, np.ndarray)
        assert msn.sample_data.shape == (MOCK_SAMPLE_SIZE, 2)

        # Check plot was called with the sampled data
        mock_scatter.assert_called_once()
        call_args = mock_scatter.call_args[0]
        call_kwargs = mock_scatter.call_args[1]
        assert isinstance(call_args[0], pd.DataFrame)
        # Check the dataframe passed to plot matches the generated sample data
        expected_plot_df = pd.DataFrame(
            msn.sample_data, columns=["ISOPleasant", "ISOEventful"]
        )
        pd.testing.assert_frame_equal(call_args[0], expected_plot_df)
        assert call_kwargs["color"] == "red"
        assert call_kwargs["title"] == "Test Plot"

    @patch("soundscapy.spi.msn.scatter")  # Keep mocking the plotting call
    def test_sspy_plot_uses_existing_sample(self, mock_scatter):
        """Test sspy_plot uses existing sample_data if available."""
        msn = MultiSkewNorm()
        # Create some dummy sample data
        existing_sample = rng.random((50, 2))
        msn.sample_data = existing_sample

        # Store original sample_data reference to check it wasn't re-generated
        sample_data_before_plot = msn.sample_data

        msn.sspy_plot()  # Should use existing sample_data

        # Check sample was NOT called again (sample_data should be unchanged)
        assert msn.sample_data is sample_data_before_plot
        np.testing.assert_array_equal(msn.sample_data, existing_sample)

        # Check plot was called with the existing data
        mock_scatter.assert_called_once()
        call_args = mock_scatter.call_args[0]
        assert isinstance(call_args[0], pd.DataFrame)
        expected_plot_df = pd.DataFrame(
            existing_sample, columns=["ISOPleasant", "ISOEventful"]
        )
        pd.testing.assert_frame_equal(call_args[0], expected_plot_df)

    def test_ks2d2s_calls_sample_if_needed(self):
        """Test ks2d2s calls sample if sample_data is None and calls ks2d2s."""
        msn = MultiSkewNorm()
        msn.define_dp(MOCK_XI, MOCK_OMEGA, MOCK_ALPHA)  # Define DP so sample can run

        assert msn.sample_data is None
        test_data_df = pd.DataFrame(rng.random((40, 2)), columns=["col1", "col2"])

        result = msn.ks2d2s(test_data_df)
        # TODO: still need to implement check for actual result values

        # Check sample was called implicitly and data was generated
        assert isinstance(msn.sample_data, np.ndarray)
        assert (
            msn.sample_data.shape[1] == 2  # noqa: PLR2004
        )  # Check sample data has 2 columns

        assert isinstance(result, tuple)
        assert isinstance(result[0], float)
        assert isinstance(result[1], float)

    def test_ks2d2s(self):
        """Test ks2d2s converts DataFrame input to numpy array."""
        msn = MultiSkewNorm()
        msn.define_dp(MOCK_XI, MOCK_OMEGA, MOCK_ALPHA)
        msn.sample(n=50)  # Generate sample data beforehand

        test_data_df = pd.DataFrame(rng.random((40, 2)), columns=["col1", "col2"])
        test_data_np = test_data_df.to_numpy()

        df_result = msn.ks2d2s(test_data_df)
        np_result = msn.ks2d2s(test_data_np)
        # TODO(MitchellAcoustics): still need to implement check for actual result values  # noqa: E501

        assert df_result == np_result, (
            "Results from DataFrame and numpy array should match."
        )

        assert isinstance(df_result, tuple)
        assert isinstance(df_result[0], float)
        assert isinstance(df_result[1], float)

        assert isinstance(np_result, tuple)
        assert isinstance(np_result[0], float)
        assert isinstance(np_result[1], float)

    def test_ks2d2s_dataframe_wrong_shape(self):
        """Test ks2d2s raises ValueError for DataFrame with wrong shape."""
        msn = MultiSkewNorm()
        msn.define_dp(MOCK_XI, MOCK_OMEGA, MOCK_ALPHA)
        msn.sample(n=50)

        test_data_df_wrong = pd.DataFrame(rng.random((40, 3)))  # 3 columns

        with pytest.raises(ValueError, match="Test data must have two columns."):
            msn.ks2d2s(test_data_df_wrong)

    def test_spi(self):
        """Test spi method calculation."""
        msn = MultiSkewNorm.from_params(xi=MOCK_XI, omega=MOCK_OMEGA, alpha=MOCK_ALPHA)
        # No need to fit or define dp as we are mocking ks2ds
        test_data = rng.random((50, 2))

        spi_value = msn.spi_score(test_data)

        # Check the SPI calculation
        # TODO(MitchellAcoustics): Implement actual SPI calculation check
        assert isinstance(spi_value, int)

    def test_spi_with_dataframe(self):
        """Test spi method with DataFrame input."""
        msn = MultiSkewNorm.from_params(xi=MOCK_XI, omega=MOCK_OMEGA, alpha=MOCK_ALPHA)
        test_data_df = pd.DataFrame(rng.random((60, 2)), columns=["a", "b"])

        spi_value = msn.spi_score(test_data_df)

        # Check the SPI calculation
        # TODO(MitchellAcoustics): Implement actual SPI calculation check
        assert isinstance(spi_value, int)


@needs_r_sn
@pytest.mark.skip(
    reason="Cannot directly convert cp to dp. Need to come up with a reasonable test."
)
def test_cp2dp():
    """Test cp2dp function."""
    cp_input = CentredParams(EXPECTED_MEAN, EXPECTED_SIGMA_COV, EXPECTED_SKEW)

    # Perform the conversion
    dp_output = cp2dp(cp_input)

    assert isinstance(dp_output, DirectParams)
    # Check if the output DP matches the original MOCK_DP used to generate the CPs
    np.testing.assert_allclose(dp_output.xi, MOCK_XI, atol=1e-5)
    np.testing.assert_allclose(dp_output.omega, MOCK_OMEGA, atol=1e-5)
    np.testing.assert_allclose(dp_output.alpha, MOCK_ALPHA, atol=1e-5)


@needs_r_sn  # Needs R for conversion
def test_dp2cp():
    """Test dp2cp function."""
    # Use the known DP values
    dp_input = DirectParams(MOCK_XI, MOCK_OMEGA, MOCK_ALPHA)

    # Perform the conversion
    cp_output = dp2cp(dp_input)

    assert isinstance(cp_output, CentredParams)
    # Check if the output CP matches the expected values
    np.testing.assert_allclose(cp_output.mean, EXPECTED_MEAN, atol=1e-5)
    # Assuming CentredParams.sigma holds covariance matrix from dp2cp
    np.testing.assert_allclose(cp_output.sigma, EXPECTED_SIGMA_COV, atol=1e-5)
    np.testing.assert_allclose(cp_output.skew, EXPECTED_SKEW, atol=1e-5)
