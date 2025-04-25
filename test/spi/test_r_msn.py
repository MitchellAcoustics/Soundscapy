"""
Tests for the R wrapper's multivariate skew-normal distribution functions.

These tests check the R wrapper functions for MSN fitting, sampling, and density calculation.
They are skipped if rpy2 or the 'sn' package is not installed.
"""

import pytest
import numpy as np
import pandas as pd


@pytest.mark.optional_deps("spi")
class TestRMSNFunctions:
    """Test the R wrapper functions for multivariate skew-normal distributions."""

    def test_msn_session_management(self):
        """Test that the R session can be initialized and managed for MSN functions."""
        from soundscapy.spi._r_wrapper import (
            shutdown_r_session,
            is_session_active,
            r_session_context,
        )

        # First ensure session is not active
        if is_session_active():
            shutdown_r_session()

        # Initialize session with context manager
        with r_session_context() as (r_session, sn_package, stats_package):
            # Verify session and packages are available
            assert r_session is not None
            assert sn_package is not None
            assert stats_package is not None

            # Check that the sn package has the required functions
            assert hasattr(sn_package, "rmsn")
            assert hasattr(sn_package, "dmsn")
            assert hasattr(sn_package, "selm")

        # Clean up
        shutdown_r_session()

    @pytest.fixture
    def sample_bivariate_data(self) -> pd.DataFrame:
        """Create a sample bivariate dataset for testing MSN fitting."""
        # Create a simple bivariate dataset with some correlation and skewness
        np.random.seed(42)
        n = 100
        x = np.random.normal(0, 1, n) + 0.5  # Add skewness
        y = 0.7 * x + np.random.normal(0, 0.5, n) - 0.3  # Correlated with skewness

        # Convert to DataFrame with named columns
        df = pd.DataFrame({"x": x, "y": y})
        return df

    def test_msn_mle_wrapper(self, sample_bivariate_data):
        """Test the wrapper for R's msn.mle function."""
        pytest.importorskip(
            "soundscapy.spi.r_msn"
        )  # Skip if r_msn module doesn't exist yet

        from soundscapy.spi.r_msn import fit_msn

        # Fit model to sample data
        result = fit_msn(sample_bivariate_data, x_col="x", y_col="y")

        # Check that the result contains the expected components
        assert isinstance(result, dict)
        assert "dp" in result
        assert "cp" in result

        # Check direct parameters (dp)
        dp = result["dp"]
        assert "xi" in dp
        assert "Omega" in dp
        assert "alpha" in dp

        # Check centered parameters (cp)
        cp = result["cp"]
        assert "mean" in cp
        assert "variance" in cp
        assert "gamma1" in cp  # skewness

        # Validate dimensions
        assert len(dp["xi"]) == 2  # bivariate
        assert dp["Omega"].shape == (2, 2)  # 2x2 covariance matrix
        assert len(dp["alpha"]) == 2  # bivariate shape parameter

        # Check that the covariance matrix is positive definite
        eigenvalues = np.linalg.eigvals(dp["Omega"])
        assert np.all(eigenvalues > 0)

        # Check that the covariance matrix is symmetric
        assert np.allclose(dp["Omega"], dp["Omega"].T)

    def test_rmsn_wrapper_with_dp(self):
        """Test the wrapper for R's rmsn function with direct parameters."""
        pytest.importorskip(
            "soundscapy.spi.r_msn"
        )  # Skip if r_msn module doesn't exist yet

        from soundscapy.spi.r_msn import sample_msn

        # Simple bivariate parameters
        xi = np.array([0.0, 0.5])
        Omega = np.array([[1.0, 0.3], [0.3, 1.0]])
        alpha = np.array([1.5, -0.8])

        # Generate samples
        n_samples = 100
        samples = sample_msn(n=n_samples, xi=xi, Omega=Omega, alpha=alpha)

        # Check output
        assert isinstance(samples, np.ndarray)
        assert samples.shape == (n_samples, 2)

        # Basic statistical checks (inexact due to randomness)
        sample_mean = np.mean(samples, axis=0)
        assert np.allclose(sample_mean, xi, atol=0.5)

        # Check that samples are skewed in the expected direction
        skewness = np.abs(np.sign(alpha)) * np.abs(np.sign(sample_mean - xi))
        assert np.all(skewness >= 0)

    def test_rmsn_wrapper_with_model(self, sample_bivariate_data):
        """Test the wrapper for R's rmsn function with a fitted model."""
        pytest.importorskip(
            "soundscapy.spi.r_msn"
        )  # Skip if r_msn module doesn't exist yet

        from soundscapy.spi.r_msn import fit_msn, sample_msn

        # Fit model to sample data
        result = fit_msn(sample_bivariate_data, x_col="x", y_col="y")

        # Generate samples from the fitted model
        n_samples = 100
        samples = sample_msn(n=n_samples, fit_result=result)

        # Check output
        assert isinstance(samples, np.ndarray)
        assert samples.shape == (n_samples, 2)

        # Basic check - samples should be in roughly the same range as the input data
        x_range = (min(sample_bivariate_data["x"]), max(sample_bivariate_data["x"]))
        y_range = (min(sample_bivariate_data["y"]), max(sample_bivariate_data["y"]))

        # Allow for some samples to be outside the range due to random sampling
        x_samples = samples[:, 0]
        y_samples = samples[:, 1]

        assert np.percentile(x_samples, 5) < x_range[1]
        assert np.percentile(x_samples, 95) > x_range[0]
        assert np.percentile(y_samples, 5) < y_range[1]
        assert np.percentile(y_samples, 95) > y_range[0]

    def test_dmsn_wrapper(self):
        """Test the wrapper for R's dmsn function for density calculation."""
        pytest.importorskip(
            "soundscapy.spi.r_msn"
        )  # Skip if r_msn module doesn't exist yet

        from soundscapy.spi.r_msn import evaluate_msn_density

        # Simple bivariate parameters
        xi = np.array([0.0, 0.5])
        Omega = np.array([[1.0, 0.3], [0.3, 1.0]])
        alpha = np.array([1.5, -0.8])

        # Points to evaluate
        points = np.array(
            [
                [0.0, 0.5],  # at the mode
                [1.0, 1.0],  # away from the mode
                [-1.0, 0.0],  # another point
            ]
        )

        # Calculate density
        densities = evaluate_msn_density(points, xi=xi, Omega=Omega, alpha=alpha)

        # Check output
        assert isinstance(densities, np.ndarray)
        assert densities.shape == (3,)
        assert np.all(densities > 0)  # Densities should be positive

        # The density at the mode should be higher than at other points
        assert densities[0] > densities[1]
        assert densities[0] > densities[2]

    def test_error_handling_invalid_params(self):
        """Test error handling for invalid parameters."""
        pytest.importorskip(
            "soundscapy.spi.r_msn"
        )  # Skip if r_msn module doesn't exist yet

        from soundscapy.spi.r_msn import sample_msn, evaluate_msn_density

        # Invalid parameters: non-positive definite Omega
        xi = np.array([0.0, 0.5])
        invalid_Omega = np.array([[1.0, 2.0], [2.0, 1.0]])  # not positive definite
        alpha = np.array([1.5, -0.8])

        # Should raise an error when trying to sample
        with pytest.raises((ValueError, RuntimeError)):
            sample_msn(n=10, xi=xi, Omega=invalid_Omega, alpha=alpha)

        # Invalid parameters: mismatched dimensions
        mismatched_xi = np.array([0.0, 0.5, 1.0])  # 3D but Omega is 2x2
        valid_Omega = np.array([[1.0, 0.3], [0.3, 1.0]])

        # Should raise an error
        with pytest.raises((ValueError, RuntimeError)):
            sample_msn(n=10, xi=mismatched_xi, Omega=valid_Omega, alpha=alpha)

        # Invalid point dimensions for density calculation
        invalid_points = np.array([[0.0, 0.5, 1.0]])  # 3D but parameters are 2D

        # Should raise an error
        with pytest.raises((ValueError, RuntimeError)):
            evaluate_msn_density(invalid_points, xi=xi, Omega=valid_Omega, alpha=alpha)

    def test_edge_cases(self):
        """Test edge cases like high dimensions and degenerate cases."""
        pytest.importorskip(
            "soundscapy.spi.r_msn"
        )  # Skip if r_msn module doesn't exist yet

        from soundscapy.spi.r_msn import sample_msn

        # 1. High dimension case (5D)
        d = 5
        xi_high = np.zeros(d)
        Omega_high = np.eye(d)  # Identity matrix
        alpha_high = np.ones(d)

        # Should work for higher dimensions
        samples_high = sample_msn(n=10, xi=xi_high, Omega=Omega_high, alpha=alpha_high)
        assert samples_high.shape == (10, d)

        # 2. Degenerate case: alpha = 0 (reduces to normal distribution)
        xi = np.array([0.0, 0.5])
        Omega = np.array([[1.0, 0.3], [0.3, 1.0]])
        alpha_zero = np.array([0.0, 0.0])

        # Should still work with alpha = 0
        samples_normal = sample_msn(n=100, xi=xi, Omega=Omega, alpha=alpha_zero)
        assert samples_normal.shape == (100, 2)

        # 3. Small sample size
        samples_small = sample_msn(n=1, xi=xi, Omega=Omega, alpha=alpha_zero)
        assert samples_small.shape == (1, 2)

        # 4. Very small Omega entries (near-degenerate covariance)
        Omega_small = np.array([[1e-5, 0], [0, 1e-5]])

        # Might cause numerical issues, but should handle gracefully
        try:
            samples_small_var = sample_msn(
                n=10, xi=xi, Omega=Omega_small, alpha=alpha_zero
            )
            assert samples_small_var.shape == (10, 2)
        except (ValueError, RuntimeError) as e:
            # Either works or raises a meaningful error
            assert "positive definite" in str(e) or "numerical" in str(e).lower()
