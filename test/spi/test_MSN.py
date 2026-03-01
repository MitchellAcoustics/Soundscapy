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


rng = default_rng(42)  # Set a random seed for reproducibility


@pytest.mark.optional_deps("spi")
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


@pytest.mark.optional_deps("spi")
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


@pytest.mark.optional_deps("spi")
class TestMultiSkewNorm:
    def test_init(self):
        """Test initialization of MultiSkewNorm."""
        msn = MultiSkewNorm()
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

    def test_summary_fitted_from_data(self):
        """Test summary when the model is fitted from data."""
        msn = MultiSkewNorm()
        msn.fit(data=MOCK_DF.copy())
        summary = msn.summary()
        assert f"Fitted from data. n = {len(MOCK_DF)}" in summary
        assert "Direct Parameters:" in summary
        assert "Centred Parameters:" in summary
        assert "xi:" in summary
        assert "mean:" in summary

    def test_summary_fitted_from_dp(self):
        """Test summary when the model is fitted from direct parameters."""
        msn = MultiSkewNorm()
        msn.define_dp(MOCK_XI, MOCK_OMEGA, MOCK_ALPHA)  # This calculates CP
        summary = msn.summary()
        assert "Fitted from direct parameters." in summary
        assert str(msn.dp) in summary
        assert str(msn.cp) in summary

    def test_fit_with_dataframe(self):
        """Test fit method with pandas DataFrame."""
        msn = MultiSkewNorm()
        msn.fit(data=MOCK_DF.copy())

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
        numpy_data = MOCK_DF.to_numpy()
        msn.fit(data=numpy_data)

        expected_df = pd.DataFrame(numpy_data, columns=["x", "y"])

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

        assert isinstance(msn.cp, CentredParams)
        assert isinstance(msn.dp, DirectParams)
        assert msn.data is not None  # Add assertion for type checker
        pd.testing.assert_frame_equal(msn.data, expected_df)
        assert msn.cp.mean.shape == (2,)
        assert msn.dp.xi.shape == (2,)

    def test_fit_does_not_mutate_input_dataframe(self):
        """
        fit() must not rename columns on the caller's DataFrame.

        Uses non-default column names so a regression would be visible —
        MOCK_DF already has columns ["x", "y"] and would pass trivially.
        """
        input_df = pd.DataFrame(MOCK_DF.values, columns=["ISOPleasant", "ISOEventful"])
        msn = MultiSkewNorm()
        msn.fit(data=input_df)
        assert list(input_df.columns) == ["ISOPleasant", "ISOEventful"], (
            "fit() must not modify the caller's DataFrame columns"
        )

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
            match="Model is not fitted. Call fit\\(\\) or define_dp\\(\\) first.",
        ):
            msn.sample()

    # --- sample_mtsn tests ---

    def test_sample_mtsn_shape(self):
        """sample_mtsn returns an (n, 2) array."""
        msn = MultiSkewNorm()
        msn.define_dp(MOCK_XI, MOCK_OMEGA, MOCK_ALPHA)
        result = msn.sample_mtsn(n=5, return_sample=True)
        assert isinstance(result, np.ndarray)
        assert result.shape == (5, 2)

    def test_sample_mtsn_within_bounds(self):
        """All samples returned by sample_mtsn are within [a, b]."""
        msn = MultiSkewNorm()
        msn.define_dp(MOCK_XI, MOCK_OMEGA, MOCK_ALPHA)
        result = msn.sample_mtsn(n=10, a=-1, b=1, return_sample=True)
        assert result is not None
        assert np.all(result >= -1), "Some samples are below the lower bound"
        assert np.all(result <= 1), "Some samples are above the upper bound"

    def test_sample_mtsn_stores_sample(self):
        """sample_mtsn stores the result in sample_data when return_sample=False."""
        msn = MultiSkewNorm()
        msn.define_dp(MOCK_XI, MOCK_OMEGA, MOCK_ALPHA)
        assert msn.sample_data is None
        retval = msn.sample_mtsn(n=5, return_sample=False)
        assert retval is None
        assert isinstance(msn.sample_data, np.ndarray)
        assert msn.sample_data.shape == (5, 2)

    def test_sample_mtsn_not_fitted(self):
        """sample_mtsn raises ValueError when the model has no parameters."""
        msn = MultiSkewNorm()
        with pytest.raises(
            ValueError,
            match="Model is not fitted. Call fit\\(\\) or define_dp\\(\\) first.",
        ):
            msn.sample_mtsn()

    # --- from_params branch tests ---

    def test_from_params_with_direct_params_object(self):
        """from_params(params=DirectParams(...)) sets dp and computes cp."""
        dp = DirectParams(MOCK_XI, MOCK_OMEGA, MOCK_ALPHA)
        msn = MultiSkewNorm.from_params(params=dp)
        assert isinstance(msn.dp, DirectParams)
        np.testing.assert_array_equal(msn.dp.xi, MOCK_XI)
        assert isinstance(msn.cp, CentredParams)
        np.testing.assert_allclose(msn.cp.mean, EXPECTED_MEAN, atol=1e-5)

    def test_from_params_with_centred_params_object(self):
        """from_params(params=CentredParams(...)) sets cp and converts to dp."""
        cp = CentredParams(EXPECTED_MEAN, EXPECTED_SIGMA_COV, EXPECTED_SKEW)
        msn = MultiSkewNorm.from_params(params=cp)
        assert isinstance(msn.cp, CentredParams)
        assert isinstance(msn.dp, DirectParams)

    def test_from_params_with_xi_omega_alpha_kwargs_sets_cp(self):
        """from_params(xi=..., omega=..., alpha=...) must populate both dp and cp."""
        msn = MultiSkewNorm.from_params(xi=MOCK_XI, omega=MOCK_OMEGA, alpha=MOCK_ALPHA)
        assert isinstance(msn.dp, DirectParams)
        assert isinstance(msn.cp, CentredParams), (
            "cp must not be None when from_params is called with DP kwargs"
        )
        np.testing.assert_allclose(msn.cp.mean, EXPECTED_MEAN, atol=1e-5)

    def test_from_params_with_mean_sigma_skew_kwargs(self):
        """from_params(mean=..., sigma=..., skew=...) creates MultiSkewNorm from CP."""
        msn = MultiSkewNorm.from_params(
            mean=EXPECTED_MEAN, sigma=EXPECTED_SIGMA_COV, skew=EXPECTED_SKEW
        )
        assert isinstance(msn.cp, CentredParams)
        assert isinstance(msn.dp, DirectParams)

    def test_from_params_no_args_raises(self):
        """from_params() with no arguments raises ValueError."""
        with pytest.raises(ValueError, match="Either params object"):
            MultiSkewNorm.from_params()

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

        # Check sample was called implicitly and data was generated
        assert isinstance(msn.sample_data, np.ndarray)
        assert msn.sample_data.shape[1] == 2  # Check sample data has 2 columns

        assert isinstance(result, tuple)
        ks_stat, p_value = result
        assert isinstance(ks_stat, float)
        assert isinstance(p_value, float)
        assert 0.0 <= ks_stat <= 1.0, "KS statistic must be in [0, 1]"
        assert 0.0 <= p_value <= 1.0, "p-value must be in [0, 1]"

    def test_ks2d2s(self):
        """Test ks2d2s converts DataFrame input to numpy array."""
        msn = MultiSkewNorm()
        msn.define_dp(MOCK_XI, MOCK_OMEGA, MOCK_ALPHA)
        msn.sample(n=50)  # Generate sample data beforehand

        test_data_df = pd.DataFrame(rng.random((40, 2)), columns=["col1", "col2"])
        test_data_np = test_data_df.to_numpy()

        df_result = msn.ks2d2s(test_data_df)
        np_result = msn.ks2d2s(test_data_np)

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

        assert isinstance(spi_value, int)
        assert 0 <= spi_value <= 100, "SPI score must be in [0, 100]"

    def test_spi_with_dataframe(self):
        """Test spi method with DataFrame input."""
        msn = MultiSkewNorm.from_params(xi=MOCK_XI, omega=MOCK_OMEGA, alpha=MOCK_ALPHA)
        test_data_df = pd.DataFrame(rng.random((60, 2)), columns=["a", "b"])

        spi_value = msn.spi_score(test_data_df)

        assert isinstance(spi_value, int)
        assert 0 <= spi_value <= 100, "SPI score must be in [0, 100]"


@pytest.mark.optional_deps("spi")
def test_cp2dp():
    """Test cp2dp via a round-trip: dp → cp → dp2cp(dp) should reproduce the same CP."""
    # Convert known DP to CP
    dp_input = DirectParams(MOCK_XI, MOCK_OMEGA, MOCK_ALPHA)
    cp = dp2cp(dp_input)

    # Convert CP back to DP
    dp_recovered = cp2dp(cp)
    assert isinstance(dp_recovered, DirectParams)

    # Convert the recovered DP back to CP again; it must match the original CP.
    # (The cp2dp→dp2cp round-trip is the numerically stable direction to test.)
    cp_roundtrip = dp2cp(dp_recovered)
    assert isinstance(cp_roundtrip, CentredParams)
    np.testing.assert_allclose(cp_roundtrip.mean, cp.mean, atol=1e-4)
    np.testing.assert_allclose(cp_roundtrip.sigma, cp.sigma, atol=1e-4)
    np.testing.assert_allclose(cp_roundtrip.skew, cp.skew, atol=1e-4)


@pytest.mark.optional_deps("spi")
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
