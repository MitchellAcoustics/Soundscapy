import pytest
from soundscapy.spi.MSN import MultiSkewNorm, DirectParams, CentredParams
import numpy as np
from unittest.mock import patch, MagicMock
import pandas as pd
from soundscapy.spi.MSN import cp2dp
from soundscapy.spi.MSN import dp2cp


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

        with pytest.raises(AssertionError, match="Omega must be positive definite"):
            DirectParams(xi, omega, alpha)

    def test_direct_params_init_not_symmetric(self):
        """Test initialization with omega that is not symmetric."""
        xi = np.array([0.1, 0.2])
        # Not symmetric matrix
        omega = np.array([[1.0, 0.5], [0.6, 1.0]])
        alpha = np.array([0.3, 0.4])

        with pytest.raises(AssertionError, match="Omega must be symmetric"):
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

    def test_centred_params_from_dp(self):
        """Test the from_dp class method."""
        # Create a dummy DirectParams object
        dp_xi = np.array([0.1, 0.2])
        dp_omega = np.array([[1.0, 0.5], [0.5, 1.0]])
        dp_alpha = np.array([0.3, 0.4])
        dp = DirectParams(dp_xi, dp_omega, dp_alpha)

        expected_cp = CentredParams(
            mean=np.array([0.44083939, 0.57492333]),
            sigma=np.array([[0.88382851, 0.37221136], [0.37221136, 0.8594325]]),
            skew=np.array([0.02045318, 0.02839051]),
        )

        # Call the class method
        cp_from_dp = CentredParams.from_dp(dp)

        # Assert that the returned object is an instance of CentredParams
        assert isinstance(cp_from_dp, CentredParams)

        # Assert that the attributes match the expected values from the mock
        np.testing.assert_allclose(cp_from_dp.mean, expected_cp.mean, atol=1e-5)
        np.testing.assert_allclose(cp_from_dp.sigma, expected_cp.sigma, atol=1e-5)
        np.testing.assert_allclose(cp_from_dp.skew, expected_cp.skew, atol=1e-5)


# Mock data and parameters for testing
MOCK_XI = np.array([0.1, 0.2])
MOCK_OMEGA = np.array([[1.0, 0.5], [0.5, 1.0]])
MOCK_ALPHA = np.array([0.3, 0.4])
MOCK_MEAN = np.array([0.5, 0.6])
MOCK_SIGMA = np.array([1.0, 1.2])
MOCK_SKEW = np.array([0.1, -0.1])
MOCK_SAMPLE = np.random.rand(100, 2)
MOCK_DF = pd.DataFrame(np.random.rand(10, 2), columns=["x", "y"])
MOCK_X = MOCK_DF["x"].values
MOCK_Y = MOCK_DF["y"].values


@patch("soundscapy.spi.MSN.rsn")
class TestMultiSkewNorm:
    def test_init(self, mock_rsn):
        """Test initialization of MultiSkewNorm."""
        msn = MultiSkewNorm()
        assert msn.selm_model is None
        assert msn.cp is None
        assert msn.dp is None
        assert msn.sample_data is None
        assert msn.data is None

    def test_repr_unfitted(self, mock_rsn):
        """Test __repr__ when the model is not fitted."""
        msn = MultiSkewNorm()
        assert repr(msn) == "MultiSkewNorm() (unfitted)"

    def test_repr_fitted(self, mock_rsn):
        """Test __repr__ when the model is fitted."""
        msn = MultiSkewNorm()
        dp = DirectParams(MOCK_XI, MOCK_OMEGA, MOCK_ALPHA)
        msn.dp = dp
        assert repr(msn) == f"MultiSkewNorm(dp={dp})"

    def test_summary_unfitted(self, mock_rsn):
        """Test summary when the model is not fitted."""
        msn = MultiSkewNorm()
        assert msn.summary() == "MultiSkewNorm is not fitted."

    @patch("builtins.print")
    def test_summary_fitted_from_data(self, mock_print, mock_rsn):
        """Test summary when the model is fitted from data."""
        msn = MultiSkewNorm()
        msn.data = MOCK_DF
        msn.dp = DirectParams(MOCK_XI, MOCK_OMEGA, MOCK_ALPHA)
        msn.cp = CentredParams(MOCK_MEAN, MOCK_SIGMA, MOCK_SKEW)

        msn.summary()

        mock_print.assert_any_call(f"Fitted from data. n = {len(MOCK_DF)}")
        mock_print.assert_any_call(msn.dp)
        mock_print.assert_any_call("\n")
        mock_print.assert_any_call(msn.cp)

    @patch("builtins.print")
    def test_summary_fitted_from_dp(self, mock_print, mock_rsn):
        """Test summary when the model is fitted from direct parameters."""
        msn = MultiSkewNorm()
        msn.dp = DirectParams(MOCK_XI, MOCK_OMEGA, MOCK_ALPHA)
        msn.cp = CentredParams(MOCK_MEAN, MOCK_SIGMA, MOCK_SKEW)  # Need CP for summary

        msn.summary()

        mock_print.assert_any_call("Fitted from direct parameters.")
        mock_print.assert_any_call(msn.dp)
        mock_print.assert_any_call("\n")
        mock_print.assert_any_call(msn.cp)

    def test_fit_with_dataframe(self, mock_rsn):
        """Test fit method with pandas DataFrame."""
        mock_selm_model = MagicMock()
        mock_rsn.selm.return_value = mock_selm_model
        mock_rsn.extract_cp.return_value = (MOCK_MEAN, MOCK_SIGMA, MOCK_SKEW)
        mock_rsn.extract_dp.return_value = (MOCK_XI, MOCK_OMEGA, MOCK_ALPHA)

        msn = MultiSkewNorm()
        msn.fit(data=MOCK_DF.copy())  # Use copy to avoid modifying original

        mock_rsn.selm.assert_called_once_with("x", "y", MOCK_DF)
        mock_rsn.extract_cp.assert_called_once_with(mock_selm_model)
        mock_rsn.extract_dp.assert_called_once_with(mock_selm_model)

        assert isinstance(msn.cp, CentredParams)
        assert isinstance(msn.dp, DirectParams)
        assert msn.selm_model == mock_selm_model
        pd.testing.assert_frame_equal(msn.data, MOCK_DF)
        np.testing.assert_array_equal(msn.cp.mean, MOCK_MEAN)
        np.testing.assert_array_equal(msn.dp.xi, MOCK_XI)

    def test_fit_with_numpy_array(self, mock_rsn):
        """Test fit method with numpy array."""
        mock_selm_model = MagicMock()
        mock_rsn.selm.return_value = mock_selm_model
        mock_rsn.extract_cp.return_value = (MOCK_MEAN, MOCK_SIGMA, MOCK_SKEW)
        mock_rsn.extract_dp.return_value = (MOCK_XI, MOCK_OMEGA, MOCK_ALPHA)

        msn = MultiSkewNorm()
        numpy_data = MOCK_DF.values
        msn.fit(data=numpy_data)

        expected_df = pd.DataFrame(numpy_data, columns=["x", "y"])
        mock_rsn.selm.assert_called_once()
        # Check the DataFrame passed to selm
        pd.testing.assert_frame_equal(mock_rsn.selm.call_args[0][2], expected_df)

        assert isinstance(msn.cp, CentredParams)
        assert isinstance(msn.dp, DirectParams)
        assert msn.selm_model == mock_selm_model
        pd.testing.assert_frame_equal(msn.data, expected_df)

    def test_fit_with_x_y(self, mock_rsn):
        """Test fit method with x and y arrays."""
        mock_selm_model = MagicMock()
        mock_rsn.selm.return_value = mock_selm_model
        mock_rsn.extract_cp.return_value = (MOCK_MEAN, MOCK_SIGMA, MOCK_SKEW)
        mock_rsn.extract_dp.return_value = (MOCK_XI, MOCK_OMEGA, MOCK_ALPHA)

        msn = MultiSkewNorm()
        msn.fit(x=MOCK_X, y=MOCK_Y)

        expected_df = pd.DataFrame({"x": MOCK_X, "y": MOCK_Y})
        mock_rsn.selm.assert_called_once()
        pd.testing.assert_frame_equal(mock_rsn.selm.call_args[0][2], expected_df)

        assert isinstance(msn.cp, CentredParams)
        assert isinstance(msn.dp, DirectParams)
        assert msn.selm_model == mock_selm_model
        pd.testing.assert_frame_equal(msn.data, expected_df)

    def test_fit_no_data(self, mock_rsn):
        """Test fit method raises ValueError when no data is provided."""
        msn = MultiSkewNorm()
        with pytest.raises(ValueError, match="Either data or x and y must be provided"):
            msn.fit()

    def test_define_dp(self, mock_rsn):
        """Test define_dp method."""
        msn = MultiSkewNorm()
        result = msn.define_dp(MOCK_XI, MOCK_OMEGA, MOCK_ALPHA)

        assert isinstance(msn.dp, DirectParams)
        np.testing.assert_array_equal(msn.dp.xi, MOCK_XI)
        np.testing.assert_array_equal(msn.dp.omega, MOCK_OMEGA)
        np.testing.assert_array_equal(msn.dp.alpha, MOCK_ALPHA)
        assert result is msn  # Check if it returns self for chaining

    def test_sample_after_fit(self, mock_rsn):
        """Test sample method after fitting the model."""
        mock_selm_model = MagicMock()
        mock_rsn.selm.return_value = mock_selm_model
        mock_rsn.extract_cp.return_value = (MOCK_MEAN, MOCK_SIGMA, MOCK_SKEW)
        mock_rsn.extract_dp.return_value = (MOCK_XI, MOCK_OMEGA, MOCK_ALPHA)
        mock_rsn.sample_msn.return_value = MOCK_SAMPLE

        msn = MultiSkewNorm()
        msn.fit(data=MOCK_DF)
        result = msn.sample(n=100, return_sample=False)

        mock_rsn.sample_msn.assert_called_once_with(selm_model=mock_selm_model, n=100)
        assert result is None
        np.testing.assert_array_equal(msn.sample_data, MOCK_SAMPLE)

    def test_sample_after_define_dp(self, mock_rsn):
        """Test sample method after defining direct parameters."""
        mock_rsn.sample_msn.return_value = MOCK_SAMPLE

        msn = MultiSkewNorm()
        msn.define_dp(MOCK_XI, MOCK_OMEGA, MOCK_ALPHA)
        result = msn.sample(n=50, return_sample=False)

        mock_rsn.sample_msn.assert_called_once_with(
            xi=MOCK_XI, omega=MOCK_OMEGA, alpha=MOCK_ALPHA, n=50
        )
        assert result is None
        np.testing.assert_array_equal(msn.sample_data, MOCK_SAMPLE)

    def test_sample_return_sample_true(self, mock_rsn):
        """Test sample method with return_sample=True."""
        mock_rsn.sample_msn.return_value = MOCK_SAMPLE

        msn = MultiSkewNorm()
        msn.define_dp(MOCK_XI, MOCK_OMEGA, MOCK_ALPHA)
        sample = msn.sample(n=100, return_sample=True)

        assert isinstance(sample, np.ndarray)
        np.testing.assert_array_equal(sample, MOCK_SAMPLE)
        np.testing.assert_array_equal(
            msn.sample_data, MOCK_SAMPLE
        )  # Ensure sample_data is still set

    def test_sample_not_fitted_or_defined(self, mock_rsn):
        """Test sample method raises ValueError when not fitted or defined."""
        msn = MultiSkewNorm()
        with pytest.raises(
            ValueError,
            match="Either selm_model or xi, omega, and alpha must be provided.",
        ):
            msn.sample()

    @patch("soundscapy.spi.MSN.sspy")
    def test_sspy_plot_calls_sample_if_needed(self, mock_sspy, mock_rsn):
        """Test sspy_plot calls sample if sample_data is None."""
        mock_rsn.sample_msn.return_value = MOCK_SAMPLE
        msn = MultiSkewNorm()
        msn.define_dp(MOCK_XI, MOCK_OMEGA, MOCK_ALPHA)  # Define DP so sample can run

        assert msn.sample_data is None
        msn.sspy_plot(color="red", title="Test Plot")

        # Check sample was called implicitly
        mock_rsn.sample_msn.assert_called_once()
        # Check plot was called with the sampled data
        mock_sspy.density_plot.assert_called_once()
        call_args = mock_sspy.density_plot.call_args[0]
        call_kwargs = mock_sspy.density_plot.call_args[1]
        assert isinstance(call_args[0], pd.DataFrame)
        pd.testing.assert_frame_equal(
            call_args[0],
            pd.DataFrame(MOCK_SAMPLE, columns=["ISOPleasant", "ISOEventful"]),
        )
        assert call_kwargs["color"] == "red"
        assert call_kwargs["title"] == "Test Plot"

    @patch("soundscapy.spi.MSN.sspy")
    def test_sspy_plot_uses_existing_sample(self, mock_sspy, mock_rsn):
        """Test sspy_plot uses existing sample_data if available."""
        msn = MultiSkewNorm()
        msn.sample_data = MOCK_SAMPLE  # Pre-set sample data

        msn.sspy_plot()

        # Check sample was NOT called again
        mock_rsn.sample_msn.assert_not_called()
        # Check plot was called with the existing data
        mock_sspy.density_plot.assert_called_once()
        call_args = mock_sspy.density_plot.call_args[0]
        assert isinstance(call_args[0], pd.DataFrame)
        pd.testing.assert_frame_equal(
            call_args[0],
            pd.DataFrame(MOCK_SAMPLE, columns=["ISOPleasant", "ISOEventful"]),
        )


# Test standalone functions
@patch("soundscapy.spi.MSN.rsn")
def test_cp2dp(mock_rsn):
    """Test cp2dp function."""
    mock_rsn._dp2cp.return_value = (MOCK_XI, MOCK_OMEGA, MOCK_ALPHA)
    cp = CentredParams(MOCK_MEAN, MOCK_SIGMA, MOCK_SKEW)

    dp = cp2dp(cp)

    mock_rsn._dp2cp.assert_called_once_with(MOCK_MEAN, MOCK_SIGMA, MOCK_SKEW)
    assert isinstance(dp, DirectParams)
    np.testing.assert_array_equal(dp.xi, MOCK_XI)
    np.testing.assert_array_equal(dp.omega, MOCK_OMEGA)
    np.testing.assert_array_equal(dp.alpha, MOCK_ALPHA)


@patch("soundscapy.spi.MSN.rsn")
def test_dp2cp(mock_rsn):
    """Test dp2cp function."""
    mock_rsn._dp2cp.return_value = (MOCK_MEAN, MOCK_SIGMA, MOCK_SKEW)
    dp = DirectParams(MOCK_XI, MOCK_OMEGA, MOCK_ALPHA)

    cp = dp2cp(dp)

    mock_rsn._dp2cp.assert_called_once_with(MOCK_XI, MOCK_OMEGA, MOCK_ALPHA)
    assert isinstance(cp, CentredParams)
    np.testing.assert_array_equal(cp.mean, MOCK_MEAN)
    np.testing.assert_array_equal(cp.sigma, MOCK_SIGMA)
    np.testing.assert_array_equal(cp.skew, MOCK_SKEW)
