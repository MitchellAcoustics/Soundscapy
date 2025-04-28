# -*- coding: utf-8 -*-
"""
Unit tests for the Multivariate Skew-Normal distribution.

Based on tests for multivariate_normal and the provided commit example.
"""

import numpy as np
import pytest
from numpy.testing import assert_, assert_allclose, assert_equal
from pytest import raises as assert_raises
from scipy.stats._multivariate import multivariate_normal

from soundscapy.spi.multivariate_skew_normal import multivariate_skew_normal

# Set a seed for reproducibility in tests involving random numbers
RNG = np.random.default_rng(123456789)


class TestMultivariateSkewNormal:
    """Tests for the multivariate_skew_normal distribution."""

    def test_input_validation(self):
        """Test input dimension validation."""
        # Mismatched dimensions
        assert_raises(
            ValueError,
            multivariate_skew_normal.pdf,
            [0, 1],
            loc=[0, 0, 0],
            scale=np.eye(3),
            shape=[0, 0, 0],
        )
        assert_raises(
            ValueError,
            multivariate_skew_normal.pdf,
            [0, 1, 2],
            loc=[0, 0],
            scale=np.eye(2),
            shape=[0, 0],
        )
        assert_raises(
            ValueError,
            multivariate_skew_normal.cdf,
            [0, 1],
            loc=[0, 0, 0],
            scale=np.eye(3),
            shape=[0, 0, 0],
        )
        assert_raises(
            ValueError,
            multivariate_skew_normal.cdf,
            [0, 1, 2],
            loc=[0, 0],
            scale=np.eye(2),
            shape=[0, 0],
        )
        assert_raises(
            ValueError,
            multivariate_skew_normal.rvs,
            loc=[0, 0, 0],
            scale=np.eye(2),
            shape=[0, 0],
        )
        assert_raises(
            ValueError,
            multivariate_skew_normal.rvs,
            loc=[0, 0],
            scale=np.eye(3),
            shape=[0, 0, 0],
        )
        assert_raises(
            ValueError,
            multivariate_skew_normal.rvs,
            loc=[0, 0],
            scale=np.eye(2),
            shape=[0, 0, 0],
        )

        # Invalid scale matrix (not PSD)
        scale_not_psd = [[1, 2], [2, 1]]
        assert_raises(ValueError, multivariate_skew_normal, scale=scale_not_psd)
        # With allow_singular=True, it should still fail if not PSD
        assert_raises(
            ValueError,
            multivariate_skew_normal,
            scale=scale_not_psd,
            allow_singular=True,
        )

        # Invalid scale matrix (non-positive diagonal)
        scale_zero_diag = [[1, 0.5], [0.5, 0]]
        assert_raises(ValueError, multivariate_skew_normal, scale=scale_zero_diag)
        assert_raises(
            ValueError,
            multivariate_skew_normal,
            scale=scale_zero_diag,
            allow_singular=True,
        )

        # Invalid shape dimension
        assert_raises(
            ValueError,
            multivariate_skew_normal,
            loc=[0, 0],
            scale=np.eye(2),
            shape=[1, 2, 3],
        )

    def test_scalar_values(self):
        """Test PDF/CDF return type for 1D case."""
        # When evaluated on scalar data (1D case), pdf/cdf should return a scalar
        x, loc, scale, shape = 1.5, 1.7, 2.5, 0.7
        pdf = multivariate_skew_normal.pdf(x, loc, scale, shape)
        cdf = multivariate_skew_normal.cdf(x, loc, scale, shape)
        logpdf = multivariate_skew_normal.logpdf(x, loc, scale, shape)
        logcdf = multivariate_skew_normal.logcdf(x, loc, scale, shape)
        assert_equal(pdf.ndim, 0, "PDF ndim failed")
        assert_equal(cdf.ndim, 0, "CDF ndim failed")
        assert_equal(logpdf.ndim, 0, "LogPDF ndim failed")
        assert_equal(logcdf.ndim, 0, "LogCDF ndim failed")

        # Test frozen version
        msn_frozen = multivariate_skew_normal(loc=loc, scale=scale, shape=shape)
        pdf_f = msn_frozen.pdf(x)
        cdf_f = msn_frozen.cdf(x)
        logpdf_f = msn_frozen.logpdf(x)
        logcdf_f = msn_frozen.logcdf(x)
        assert_equal(pdf_f.ndim, 0, "Frozen PDF ndim failed")
        assert_equal(cdf_f.ndim, 0, "Frozen CDF ndim failed")
        assert_equal(logpdf_f.ndim, 0, "Frozen LogPDF ndim failed")
        assert_equal(logcdf_f.ndim, 0, "Frozen LogCDF ndim failed")

    def test_logpdf_pdf_consistency(self):
        """Check logpdf is log(pdf)."""
        loc = RNG.standard_normal(3)
        scale = RNG.standard_normal((3, 3))
        scale = scale @ scale.T + np.eye(3)  # Ensure PSD
        shape = RNG.standard_normal(3)
        x = RNG.standard_normal(3)

        logpdf = multivariate_skew_normal.logpdf(x, loc, scale, shape)
        pdf = multivariate_skew_normal.pdf(x, loc, scale, shape)

        # Handle pdf=0 case where logpdf should be -inf
        pdf_is_zero = np.abs(pdf) < 1e-300  # Check for effective zero
        logpdf_expected = np.log(
            np.where(pdf_is_zero, 1.0, pdf)
        )  # Avoid log(0) warning
        logpdf_expected[pdf_is_zero] = -np.inf

        assert_allclose(logpdf, logpdf_expected, rtol=1e-10, atol=1e-10)

        # Test with multiple points
        x_multi = RNG.standard_normal((10, 3))
        logpdf_multi = multivariate_skew_normal.logpdf(x_multi, loc, scale, shape)
        pdf_multi = multivariate_skew_normal.pdf(x_multi, loc, scale, shape)
        pdf_is_zero_multi = np.abs(pdf_multi) < 1e-300
        logpdf_expected_multi = np.log(np.where(pdf_is_zero_multi, 1.0, pdf_multi))
        logpdf_expected_multi[pdf_is_zero_multi] = -np.inf
        assert_allclose(logpdf_multi, logpdf_expected_multi, rtol=1e-10, atol=1e-10)

    def test_logcdf_cdf_consistency(self):
        """Check logcdf is log(cdf)."""
        loc = RNG.standard_normal(2)
        scale = RNG.standard_normal((2, 2))
        scale = scale @ scale.T + np.eye(2)  # Ensure PSD
        shape = RNG.standard_normal(2) * 2  # Moderate skew
        x = RNG.standard_normal(2)

        logcdf = multivariate_skew_normal.logcdf(x, loc, scale, shape)
        cdf = multivariate_skew_normal.cdf(x, loc, scale, shape)

        # Handle cdf=0 case where logcdf should be -inf
        cdf_is_zero = (
            np.abs(cdf) < 1e-15
        )  # CDF less likely to be exactly zero far from -inf
        logcdf_expected = np.log(
            np.where(cdf_is_zero, 1.0, cdf)
        )  # Avoid log(0) warning
        logcdf_expected[cdf_is_zero] = -np.inf

        # CDF calculation can have lower precision
        assert_allclose(logcdf, logcdf_expected, rtol=1e-6, atol=1e-6)

        # Test with multiple points
        x_multi = RNG.standard_normal((5, 2))
        logcdf_multi = multivariate_skew_normal.logcdf(x_multi, loc, scale, shape)
        cdf_multi = multivariate_skew_normal.cdf(x_multi, loc, scale, shape)
        cdf_is_zero_multi = np.abs(cdf_multi) < 1e-15
        logcdf_expected_multi = np.log(np.where(cdf_is_zero_multi, 1.0, cdf_multi))
        logcdf_expected_multi[cdf_is_zero_multi] = -np.inf
        assert_allclose(logcdf_multi, logcdf_expected_multi, rtol=1e-6, atol=1e-6)

    def test_default_values(self):
        """Check default parameters (loc=0, scale=I, shape=0 -> standard normal)."""
        dim = 3
        x = RNG.standard_normal(dim)

        # MSN with defaults
        pdf_msn_def = multivariate_skew_normal.pdf(x)
        logpdf_msn_def = multivariate_skew_normal.logpdf(x)
        cdf_msn_def = multivariate_skew_normal.cdf(x)
        logcdf_msn_def = multivariate_skew_normal.logcdf(x)
        mean_msn_def = multivariate_skew_normal.mean(
            shape=np.zeros(dim)
        )  # Need shape for dim
        cov_msn_def = multivariate_skew_normal.cov(shape=np.zeros(dim))

        # Explicit standard normal parameters
        loc0 = np.zeros(dim)
        scaleI = np.eye(dim)
        shape0 = np.zeros(dim)
        pdf_msn_std = multivariate_skew_normal.pdf(x, loc0, scaleI, shape0)
        logpdf_msn_std = multivariate_skew_normal.logpdf(x, loc0, scaleI, shape0)
        cdf_msn_std = multivariate_skew_normal.cdf(x, loc0, scaleI, shape0)
        logcdf_msn_std = multivariate_skew_normal.logcdf(x, loc0, scaleI, shape0)
        mean_msn_std = multivariate_skew_normal.mean(loc0, scaleI, shape0)
        cov_msn_std = multivariate_skew_normal.cov(loc0, scaleI, shape0)

        # MVN for comparison
        mvn = multivariate_normal(mean=loc0, cov=scaleI)
        pdf_mvn = mvn.pdf(x)
        logpdf_mvn = mvn.logpdf(x)
        cdf_mvn = mvn.cdf(x)
        logcdf_mvn = mvn.logcdf(x)
        mean_mvn = mvn.mean
        cov_mvn = mvn.cov

        # Assertions
        assert_allclose(pdf_msn_def, pdf_mvn, rtol=1e-10)
        assert_allclose(logpdf_msn_def, logpdf_mvn, rtol=1e-10)
        assert_allclose(cdf_msn_def, cdf_mvn, rtol=1e-7)  # CDF tolerance
        assert_allclose(logcdf_msn_def, logcdf_mvn, rtol=1e-7)
        assert_allclose(mean_msn_def, mean_mvn, rtol=1e-10)
        assert_allclose(cov_msn_def, cov_mvn, rtol=1e-10)

        assert_allclose(pdf_msn_def, pdf_msn_std, rtol=1e-10)
        assert_allclose(logpdf_msn_def, logpdf_msn_std, rtol=1e-10)
        assert_allclose(cdf_msn_def, cdf_msn_std, rtol=1e-7)
        assert_allclose(logcdf_msn_def, logcdf_msn_std, rtol=1e-7)
        assert_allclose(mean_msn_def, mean_msn_std, rtol=1e-10)
        assert_allclose(cov_msn_def, cov_msn_std, rtol=1e-10)

    def test_frozen_consistency(self):
        """Compare frozen distribution with generator methods."""
        loc = RNG.standard_normal(2)
        scale = RNG.standard_normal((2, 2))
        scale = scale @ scale.T + np.eye(2) * 0.5  # Ensure PSD
        shape = RNG.standard_normal(2) * 3  # Larger skew
        x = RNG.standard_normal(2)
        x_multi = RNG.standard_normal((5, 2))

        msn_frozen = multivariate_skew_normal(loc=loc, scale=scale, shape=shape)

        # PDF
        assert_allclose(
            msn_frozen.pdf(x), multivariate_skew_normal.pdf(x, loc, scale, shape)
        )
        assert_allclose(
            msn_frozen.pdf(x_multi),
            multivariate_skew_normal.pdf(x_multi, loc, scale, shape),
        )
        # LogPDF
        assert_allclose(
            msn_frozen.logpdf(x), multivariate_skew_normal.logpdf(x, loc, scale, shape)
        )
        assert_allclose(
            msn_frozen.logpdf(x_multi),
            multivariate_skew_normal.logpdf(x_multi, loc, scale, shape),
        )
        # CDF
        assert_allclose(
            msn_frozen.cdf(x),
            multivariate_skew_normal.cdf(x, loc, scale, shape),
            rtol=1e-6,
        )
        assert_allclose(
            msn_frozen.cdf(x_multi),
            multivariate_skew_normal.cdf(x_multi, loc, scale, shape),
            rtol=1e-6,
        )
        # LogCDF
        assert_allclose(
            msn_frozen.logcdf(x),
            multivariate_skew_normal.logcdf(x, loc, scale, shape),
            rtol=1e-6,
        )
        assert_allclose(
            msn_frozen.logcdf(x_multi),
            multivariate_skew_normal.logcdf(x_multi, loc, scale, shape),
            rtol=1e-6,
        )
        # Mean
        assert_allclose(
            msn_frozen.mean(), multivariate_skew_normal.mean(loc, scale, shape)
        )
        # Cov
        assert_allclose(
            msn_frozen.cov(), multivariate_skew_normal.cov(loc, scale, shape)
        )
        # Var
        assert_allclose(
            msn_frozen.var(), multivariate_skew_normal.var(loc, scale, shape)
        )

    def test_rvs_shape(self):
        """Test output shape of rvs method."""
        dim = 4
        loc = np.zeros(dim)
        scale = np.eye(dim)
        shape = np.ones(dim)

        # size=None
        sample_none = multivariate_skew_normal.rvs(
            loc, scale, shape, size=None, random_state=RNG
        )
        assert_equal(sample_none.shape, (dim,))

        # size=1
        sample_1 = multivariate_skew_normal.rvs(
            loc, scale, shape, size=1, random_state=RNG
        )
        assert_equal(sample_1.shape, (1, dim))

        # size=N
        N = 100
        sample_N = multivariate_skew_normal.rvs(
            loc, scale, shape, size=N, random_state=RNG
        )
        assert_equal(sample_N.shape, (N, dim))

        # size=(M, K)
        M, K = 10, 5
        sample_MK = multivariate_skew_normal.rvs(
            loc, scale, shape, size=(M, K), random_state=RNG
        )
        assert_equal(sample_MK.shape, (M, K, dim))

        # Frozen version
        msn_frozen = multivariate_skew_normal(loc, scale, shape)
        sample_f_none = msn_frozen.rvs(size=None, random_state=RNG)
        assert_equal(sample_f_none.shape, (dim,))
        sample_f_1 = msn_frozen.rvs(size=1, random_state=RNG)
        assert_equal(sample_f_1.shape, (1, dim))
        sample_f_N = msn_frozen.rvs(size=N, random_state=RNG)
        assert_equal(sample_f_N.shape, (N, dim))
        sample_f_MK = msn_frozen.rvs(size=(M, K), random_state=RNG)
        assert_equal(sample_f_MK.shape, (M, K, dim))

    def test_rvs_moments(self):
        """Compare sample moments with theoretical moments."""
        dim = 3
        loc = RNG.standard_normal(dim)
        scale_mat = RNG.standard_normal((dim, dim))
        scale = scale_mat @ scale_mat.T + np.eye(dim)  # Ensure PSD
        shape = RNG.standard_normal(dim) * 2  # Moderate skew

        n_samples = 100000  # Use a large number of samples

        samples = multivariate_skew_normal.rvs(
            loc, scale, shape, size=n_samples, random_state=RNG
        )

        theoretical_mean = multivariate_skew_normal.mean(loc, scale, shape)
        theoretical_cov = multivariate_skew_normal.cov(loc, scale, shape)

        sample_mean = np.mean(samples, axis=0)
        sample_cov = np.cov(samples, rowvar=False)

        # Use relatively loose tolerance due to sampling variability
        assert_allclose(sample_mean, theoretical_mean, rtol=1e-2, atol=1e-2)
        assert_allclose(
            sample_cov, theoretical_cov, rtol=5e-2, atol=5e-2
        )  # Covariance needs looser tolerance

    def test_dp_cp_conversion(self):
        """Test DP <-> CP conversions."""
        dim = 3
        loc_dp = RNG.standard_normal(dim)
        scale_dp_mat = RNG.standard_normal((dim, dim))
        scale_dp = scale_dp_mat @ scale_dp_mat.T + np.eye(dim)  # Ensure PSD
        shape_dp = RNG.standard_normal(dim) * 2  # Moderate skew

        # DP -> CP
        mean_cp, cov_cp, gamma1_cp = multivariate_skew_normal.dp2cp(
            loc_dp, scale_dp, shape_dp
        )

        # Check consistency of derived CP moments with direct calculation
        mean_direct = multivariate_skew_normal.mean(loc_dp, scale_dp, shape_dp)
        cov_direct = multivariate_skew_normal.cov(loc_dp, scale_dp, shape_dp)
        assert_allclose(mean_cp, mean_direct, rtol=1e-9, atol=1e-9)
        assert_allclose(cov_cp, cov_direct, rtol=1e-9, atol=1e-9)

        # CP -> DP
        loc_dp2, scale_dp2, shape_dp2 = multivariate_skew_normal.cp2dp(
            mean_cp, cov_cp, gamma1_cp
        )

        # Check if conversion is reversible (within tolerance)
        assert_allclose(loc_dp, loc_dp2, rtol=1e-7, atol=1e-7)
        assert_allclose(scale_dp, scale_dp2, rtol=1e-7, atol=1e-7)
        assert_allclose(shape_dp, shape_dp2, rtol=1e-7, atol=1e-7)

        # Test edge case: alpha = 0 (should match MVN)
        shape_zero = np.zeros(dim)
        mean_cp0, cov_cp0, gamma1_cp0 = multivariate_skew_normal.dp2cp(
            loc_dp, scale_dp, shape_zero
        )
        assert_allclose(mean_cp0, loc_dp)
        assert_allclose(cov_cp0, scale_dp)
        assert_allclose(gamma1_cp0, np.zeros(dim))

        loc_dp0, scale_dp0, shape_dp0 = multivariate_skew_normal.cp2dp(
            loc_dp, scale_dp, np.zeros(dim)
        )
        assert_allclose(loc_dp0, loc_dp)
        assert_allclose(scale_dp0, scale_dp)
        assert_allclose(shape_dp0, np.zeros(dim))

    def test_cp2dp_invalid_input(self):
        """Test cp2dp with invalid CP inputs."""
        dim = 2
        mean_cp = np.zeros(dim)
        cov_cp = np.eye(dim)
        # Invalid gamma1 (outside bounds)
        gamma1_invalid = np.array([1.0, 0.5])  # > 0.995...
        assert_raises(
            UserWarning, multivariate_skew_normal.cp2dp, mean_cp, cov_cp, gamma1_invalid
        )
        # Check if it returns NaN for shape
        with pytest.warns(RuntimeWarning):
            _, _, shape_nan = multivariate_skew_normal.cp2dp(
                mean_cp, cov_cp, gamma1_invalid
            )
        assert_(np.any(np.isnan(shape_nan)))

        # Invalid cov (not PSD)
        cov_not_psd = np.array([[1, 2], [2, 1]])
        gamma1_valid = np.array([0.5, -0.3])
        assert_raises(
            ValueError,
            multivariate_skew_normal.cp2dp,
            mean_cp,
            cov_not_psd,
            gamma1_valid,
        )

    def test_cdf_bounds_monotonicity(self):
        """Test CDF properties: bounds and monotonicity."""
        dim = 2
        loc = np.array([0, 1])
        scale = np.array([[1, 0.5], [0.5, 1.5]])
        shape = np.array([2, -1])
        msn = multivariate_skew_normal(loc=loc, scale=scale, shape=shape)

        # Bounds
        inf_point = np.full(dim, np.inf)
        neg_inf_point = np.full(dim, -np.inf)
        assert_allclose(msn.cdf(inf_point), 1.0, rtol=1e-6)
        assert_allclose(msn.cdf(neg_inf_point), 0.0, rtol=1e-6)

        # Monotonicity (check along one dimension)
        x_base = np.array([0.5, 0.5])
        x_inc1 = np.array([0.6, 0.5])
        x_inc2 = np.array([0.5, 0.6])
        cdf_base = msn.cdf(x_base)
        cdf_inc1 = msn.cdf(x_inc1)
        cdf_inc2 = msn.cdf(x_inc2)
        assert_(cdf_inc1 >= cdf_base - 1e-7)  # Allow for numerical tolerance
        assert_(cdf_inc2 >= cdf_base - 1e-7)

    def test_cdf_against_mvn(self):
        """Test CDF matches MVN when shape is zero."""
        dim = 3
        loc = RNG.standard_normal(dim)
        scale_mat = RNG.standard_normal((dim, dim))
        scale = scale_mat @ scale_mat.T + np.eye(dim)
        shape_zero = np.zeros(dim)
        x = RNG.standard_normal((10, dim))  # Test multiple points

        cdf_msn = multivariate_skew_normal.cdf(x, loc, scale, shape_zero)
        cdf_mvn = multivariate_normal.cdf(x, mean=loc, cov=scale)

        assert_allclose(
            cdf_msn, cdf_mvn, rtol=1e-6, atol=1e-7
        )  # Use tolerance suitable for CDF

    def test_singular_scale(self):
        """Test methods with a singular scale matrix."""
        dim = 3
        rank = 2
        loc = np.zeros(dim)
        # Create a singular matrix
        A = RNG.standard_normal((dim, rank))
        scale_singular = A @ A.T
        shape = np.array([1, -1, 0.5])  # Non-zero shape

        # Point within the support subspace (approx)
        # Project a point onto the column space of A
        p_random = RNG.standard_normal(dim)
        p_in_support = A @ np.linalg.lstsq(A, p_random, rcond=None)[0]

        # Point outside the support subspace (approx)
        # Find a vector orthogonal to column space of A
        q, _ = np.linalg.qr(A)
        orth_vec = RNG.standard_normal(dim)
        orth_vec = orth_vec - q @ (q.T @ orth_vec)
        orth_vec /= np.linalg.norm(orth_vec)
        p_outside_support = p_in_support + orth_vec * 0.1  # Add component outside

        # --- Test Generator Methods ---
        # PDF/LogPDF
        pdf_in = multivariate_skew_normal.pdf(
            p_in_support, loc, scale_singular, shape, allow_singular=True
        )
        logpdf_in = multivariate_skew_normal.logpdf(
            p_in_support, loc, scale_singular, shape, allow_singular=True
        )
        pdf_out = multivariate_skew_normal.pdf(
            p_outside_support, loc, scale_singular, shape, allow_singular=True
        )
        logpdf_out = multivariate_skew_normal.logpdf(
            p_outside_support, loc, scale_singular, shape, allow_singular=True
        )
        assert_(pdf_in > 0)
        assert_(np.isfinite(logpdf_in))
        assert_allclose(pdf_out, 0.0, atol=1e-15)
        assert_equal(logpdf_out, -np.inf)

        # RVS (check if samples lie approx in the subspace)
        n_samples = 50
        samples = multivariate_skew_normal.rvs(
            loc,
            scale_singular,
            shape,
            size=n_samples,
            allow_singular=True,
            random_state=RNG,
        )
        assert_equal(samples.shape, (n_samples, dim))
        # Project samples onto orthogonal complement and check norm
        # Use SVD of scale_singular to find null space basis
        U, s, Vh = np.linalg.svd(scale_singular)
        null_space_basis = U[:, rank:]
        projections_outside = samples @ null_space_basis
        norms_outside = np.linalg.norm(projections_outside, axis=1)
        assert_allclose(norms_outside, 0, atol=1e-7)  # Samples should be in subspace

        # Mean/Cov (should still be calculable, cov will be singular)
        mean_sing = multivariate_skew_normal.mean(
            loc, scale_singular, shape, allow_singular=True
        )
        cov_sing = multivariate_skew_normal.cov(
            loc, scale_singular, shape, allow_singular=True
        )
        assert_(np.isfinite(mean_sing).all())
        assert_(np.isfinite(cov_sing).all())
        assert_equal(np.linalg.matrix_rank(cov_sing, tol=1e-8), rank)

        # CDF (Expect NaN or error as CDF is ill-defined for singular?)
        # SciPy's MVN CDF raises error for singular cov. Let's check ours.
        # Note: Our CDF relies on MVN CDF internally.
        assert_raises(
            ValueError,
            multivariate_skew_normal.cdf,
            p_in_support,
            loc,
            scale_singular,
            shape,
            allow_singular=True,
        )
        assert_raises(
            ValueError,
            multivariate_skew_normal.logcdf,
            p_in_support,
            loc,
            scale_singular,
            shape,
            allow_singular=True,
        )

        # --- Test Frozen Methods ---
        msn_frozen_sing = multivariate_skew_normal(
            loc, scale_singular, shape, allow_singular=True
        )
        pdf_f_in = msn_frozen_sing.pdf(p_in_support)
        logpdf_f_in = msn_frozen_sing.logpdf(p_in_support)
        pdf_f_out = msn_frozen_sing.pdf(p_outside_support)
        logpdf_f_out = msn_frozen_sing.logpdf(p_outside_support)
        assert_allclose(pdf_f_in, pdf_in)
        assert_allclose(logpdf_f_in, logpdf_in)
        assert_allclose(pdf_f_out, pdf_out, atol=1e-15)
        assert_equal(logpdf_f_out, logpdf_out)

        samples_f = msn_frozen_sing.rvs(size=n_samples, random_state=RNG)
        projections_outside_f = samples_f @ null_space_basis
        norms_outside_f = np.linalg.norm(projections_outside_f, axis=1)
        assert_allclose(norms_outside_f, 0, atol=1e-7)

        assert_allclose(msn_frozen_sing.mean(), mean_sing)
        assert_allclose(msn_frozen_sing.cov(), cov_sing)

        assert_raises(ValueError, msn_frozen_sing.cdf, p_in_support)
        assert_raises(ValueError, msn_frozen_sing.logcdf, p_in_support)

    def test_fit_mle(self):
        """Test Maximum Likelihood Estimation."""
        dim = 2
        true_loc = np.array([0.5, -1.0])
        true_scale_mat = np.array([[1.5, 0.6], [0.6, 1.0]])
        true_shape = np.array([2.0, -1.5])

        # Generate data
        n_samples = 2000  # Need sufficient samples for good estimates
        data = multivariate_skew_normal.rvs(
            true_loc, true_scale_mat, true_shape, size=n_samples, random_state=RNG
        )

        # Fit the model
        fitted_loc, fitted_scale, fitted_shape = multivariate_skew_normal.fit(data)

        # Check if fitted parameters are close to true parameters
        # Tolerances need to be somewhat loose for MLE
        assert_allclose(fitted_loc, true_loc, rtol=0.1, atol=0.1)
        assert_allclose(fitted_scale, true_scale_mat, rtol=0.15, atol=0.15)
        assert_allclose(
            fitted_shape, true_shape, rtol=0.2, atol=0.2
        )  # Shape can be harder to estimate

    def test_fit_mle_fixed_params(self):
        """Test MLE with some parameters fixed."""
        dim = 2
        true_loc = np.array([0.5, -1.0])
        true_scale_mat = np.array([[1.5, 0.6], [0.6, 1.0]])
        true_shape = np.array([2.0, -1.5])
        n_samples = 2000
        data = multivariate_skew_normal.rvs(
            true_loc, true_scale_mat, true_shape, size=n_samples, random_state=RNG
        )

        # Fix location
        fitted_loc_f, fitted_scale_f, fitted_shape_f = multivariate_skew_normal.fit(
            data, f_loc=true_loc
        )
        assert_allclose(fitted_loc_f, true_loc)  # Should be exactly the fixed value
        assert_allclose(fitted_scale_f, true_scale_mat, rtol=0.15, atol=0.15)
        assert_allclose(fitted_shape_f, true_shape, rtol=0.2, atol=0.2)

        # Fix shape
        fitted_loc_f, fitted_scale_f, fitted_shape_f = multivariate_skew_normal.fit(
            data, f_shape=true_shape
        )
        assert_allclose(fitted_loc_f, true_loc, rtol=0.1, atol=0.1)
        assert_allclose(fitted_scale_f, true_scale_mat, rtol=0.15, atol=0.15)
        assert_allclose(fitted_shape_f, true_shape)  # Should be exactly the fixed value

        # Fix scale
        fitted_loc_f, fitted_scale_f, fitted_shape_f = multivariate_skew_normal.fit(
            data, f_scale=true_scale_mat
        )
        assert_allclose(fitted_loc_f, true_loc, rtol=0.1, atol=0.1)
        assert_allclose(
            fitted_scale_f, true_scale_mat
        )  # Should be exactly the fixed value
        assert_allclose(fitted_shape_f, true_shape, rtol=0.2, atol=0.2)

    # Test for boundary warning (might be sensitive to data/optimizer)
    # def test_fit_boundary_warning(self):
    #     """Test if fit warns when shape estimate is large."""
    #     dim = 1
    #     # Generate data likely to push alpha high (e.g., half-normal like)
    #     true_loc = 0
    #     true_scale = 1
    #     true_shape = 100 # Very large shape
    #     # Need rvs to work reasonably well for large alpha
    #     try:
    #         data = multivariate_skew_normal.rvs(true_loc, true_scale, true_shape,
    #                                             size=100, random_state=RNG)
    #         data = data[data > -1] # Truncate left tail slightly to encourage boundary
    #         with pytest.warns(RuntimeWarning, match="Estimated shape parameter norm"):
    #             multivariate_skew_normal.fit(data, alpha_bound=10) # Use lower bound for test
    #     except ValueError as e:
    #          pytest.skip(f"Skipping boundary test, rvs/fit failed for large alpha: {e}")
    #     except np.linalg.LinAlgError as e:
    #          pytest.skip(f"Skipping boundary test, rvs/fit failed for large alpha: {e}")

    # Placeholder for R value comparison - requires reference values
    # def test_R_values_pdf(self):
    #     """Compare PDF with values from R's sn package."""
    #     # Example from R sn::dmsn
    #     # dp <- list(xi=c(0,0), Omega=matrix(c(1, 0.7, 0.7, 1), 2, 2), alpha=c(3, -5))
    #     # x <- c(0.5, -0.2)
    #     # dmsn(x, dp=dp) # Output: 0.1530604
    #     xi = np.array([0,0])
    #     Omega = np.array([[1, 0.7], [0.7, 1]])
    #     alpha = np.array([3, -5])
    #     x = np.array([0.5, -0.2])
    #     expected_pdf = 0.1530604
    #     calculated_pdf = multivariate_skew_normal.pdf(x, xi, Omega, alpha)
    #     assert_allclose(calculated_pdf, expected_pdf, rtol=1e-6)

    # def test_R_values_cdf(self):
    #     """Compare CDF with values from R's sn package."""
    #     # Example from R sn::pmsn
    #     # dp <- list(xi=c(0,0), Omega=matrix(c(1, 0.7, 0.7, 1), 2, 2), alpha=c(3, -5))
    #     # x <- c(0.5, -0.2)
    #     # pmsn(x, dp=dp) # Output: 0.3241452 (using default alg)
    #     xi = np.array([0,0])
    #     Omega = np.array([[1, 0.7], [0.7, 1]])
    #     alpha = np.array([3, -5])
    #     x = np.array([0.5, -0.2])
    #     expected_cdf = 0.3241452
    #     # Need to potentially adjust maxpts/tolerances for comparison
    #     calculated_cdf = multivariate_skew_normal.cdf(x, xi, Omega, alpha, abseps=1e-7, releps=1e-7)
    #     assert_allclose(calculated_cdf, expected_cdf, rtol=1e-5, atol=1e-5) # Looser tolerance for CDF
