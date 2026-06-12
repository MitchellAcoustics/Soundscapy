"""
Pure-Python regression tests for dp2cp / msn_dp2cp.

These tests do NOT require R or rpy2 to be installed.  The expected values
(EXPECTED_MEAN, EXPECTED_SIGMA, EXPECTED_SKEW) were originally derived from
R's ``sn`` package and serve as the regression baseline.
"""

import numpy as np
import pytest

from soundscapy.spi._mvskew import bleat, msn_dp2cp, mst_dp2cp
from soundscapy.spi.msn import CentredParams, DirectParams, dp2cp

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
MOCK_XI = np.array([0.1, 0.2])
MOCK_OMEGA = np.array([[1.0, 0.5], [0.5, 1.0]])
MOCK_ALPHA = np.array([0.3, 0.4])

# Regression baseline — computed from R's sn::dp2cp and verified to agree
# with msn_dp2cp to within ~4e-9 (near machine precision).
EXPECTED_MEAN = np.array([0.44083939, 0.57492333])
EXPECTED_SIGMA = np.array([[0.88382851, 0.37221136], [0.37221136, 0.8594325]])
EXPECTED_SKEW = np.array([0.02045318, 0.02839051])

ATOL = 1e-5


# ---------------------------------------------------------------------------
# msn_dp2cp (low-level)
# ---------------------------------------------------------------------------
class TestMsnDp2cpRaw:
    """Direct tests of the _mvskew.msn_dp2cp function."""

    def test_mean(self):
        mean, *_ = msn_dp2cp(MOCK_XI, MOCK_OMEGA, MOCK_ALPHA, tau=0)
        np.testing.assert_allclose(np.asarray(mean).ravel(), EXPECTED_MEAN, atol=ATOL)

    def test_sigma(self):
        _, sigma, *_ = msn_dp2cp(MOCK_XI, MOCK_OMEGA, MOCK_ALPHA, tau=0)
        np.testing.assert_allclose(np.asarray(sigma), EXPECTED_SIGMA, atol=ATOL)

    def test_skew(self):
        _, _, skew, *_ = msn_dp2cp(MOCK_XI, MOCK_OMEGA, MOCK_ALPHA, tau=0)
        np.testing.assert_allclose(np.asarray(skew).ravel(), EXPECTED_SKEW, atol=ATOL)


# ---------------------------------------------------------------------------
# dp2cp (high-level)
# ---------------------------------------------------------------------------
class TestDp2cpPure:
    """Tests for msn.dp2cp using the pure-Python path."""

    def setup_method(self):
        self.dp = DirectParams(MOCK_XI, MOCK_OMEGA, MOCK_ALPHA)

    def test_returns_centred_params(self):
        cp = dp2cp(self.dp)
        assert isinstance(cp, CentredParams)

    def test_mean(self):
        cp = dp2cp(self.dp)
        np.testing.assert_allclose(cp.mean, EXPECTED_MEAN, atol=ATOL)

    def test_sigma(self):
        cp = dp2cp(self.dp)
        np.testing.assert_allclose(cp.sigma, EXPECTED_SIGMA, atol=ATOL)

    def test_skew(self):
        cp = dp2cp(self.dp)
        np.testing.assert_allclose(cp.skew, EXPECTED_SKEW, atol=ATOL)

    def test_outputs_are_1d_arrays(self):
        """Mean and skew must be 1-D ndarrays, not row matrices."""
        cp = dp2cp(self.dp)
        assert cp.mean.ndim == 1
        assert cp.skew.ndim == 1


# ---------------------------------------------------------------------------
# bleat
# ---------------------------------------------------------------------------
class TestBleat:
    """Tests for the bleat (b-function) helper."""

    def test_scalar_returns_array(self):
        result = bleat(10)
        assert isinstance(result, np.ndarray)

    def test_nu_leq_1_is_nan(self):
        result = bleat(1.0)
        assert np.isnan(result[0])

    def test_large_nu_approaches_sqrt_2_over_pi(self):
        """For large nu, b(nu) -> sqrt(2/pi)."""
        result = bleat(1e8)
        np.testing.assert_allclose(result[0], np.sqrt(2 / np.pi), rtol=1e-4)

    def test_known_value_nu2(self):
        """b(2) = sqrt(2/pi) * Gamma(1/2) / Gamma(1) = sqrt(2/pi) * sqrt(pi) = sqrt(2)."""
        result = bleat(2)
        np.testing.assert_allclose(result[0], np.sqrt(2), rtol=1e-6)


# ---------------------------------------------------------------------------
# mst_dp2cp (low-level)
# ---------------------------------------------------------------------------
class TestMstDp2cpRaw:
    """Direct tests of the _mvskew.mst_dp2cp function."""

    def test_returns_5_tuple(self):
        result = mst_dp2cp(MOCK_XI, MOCK_OMEGA, MOCK_ALPHA, nu=10)
        assert result is not None
        assert len(result) == 5

    def test_beta_shape(self):
        result = mst_dp2cp(MOCK_XI, MOCK_OMEGA, MOCK_ALPHA, nu=10)
        assert result is not None
        beta = result[0]
        assert np.asarray(beta).shape == (2,)

    def test_sigma_mat_shape(self):
        result = mst_dp2cp(MOCK_XI, MOCK_OMEGA, MOCK_ALPHA, nu=10)
        assert result is not None
        sigma_mat = result[1]
        assert np.asarray(sigma_mat).shape == (2, 2)

    def test_nu_in_result(self):
        result = mst_dp2cp(MOCK_XI, MOCK_OMEGA, MOCK_ALPHA, nu=10)
        assert result is not None
        assert result[4] == 10

    def test_approx_mode_sc(self):
        """SC (nu=1) works with approx mode."""
        result = mst_dp2cp(MOCK_XI, MOCK_OMEGA, MOCK_ALPHA, nu=1, cp_type="approx")
        assert result is not None

    def test_proper_mode_low_nu_returns_none(self):
        """Proper mode with nu <= upto should warn and return None."""
        with pytest.warns(UserWarning, match="degrees of freedom"):
            result = mst_dp2cp(MOCK_XI, MOCK_OMEGA, MOCK_ALPHA, nu=3, cp_type="proper")
        assert result is None


# ---------------------------------------------------------------------------
# dp2cp — ST and SC families
# ---------------------------------------------------------------------------
class TestDp2cpST:
    """dp2cp with family='ST' (skew-T)."""

    def setup_method(self):
        self.dp = DirectParams(MOCK_XI, MOCK_OMEGA, MOCK_ALPHA, nu=10)

    def test_returns_centred_params(self):
        cp = dp2cp(self.dp, family="ST")
        assert isinstance(cp, CentredParams)

    def test_mean_shape(self):
        cp = dp2cp(self.dp, family="ST")
        assert cp.mean.shape == (2,)

    def test_sigma_shape(self):
        cp = dp2cp(self.dp, family="ST")
        assert cp.sigma.shape == (2, 2)

    def test_nu_stored(self):
        cp = dp2cp(self.dp, family="ST")
        assert cp.nu == 10

    def test_gamma2_present(self):
        cp = dp2cp(self.dp, family="ST")
        assert cp.gamma2 is not None


class TestDp2cpSC:
    """dp2cp with family='SC' (skew-Cauchy, nu=1)."""

    def setup_method(self):
        self.dp = DirectParams(MOCK_XI, MOCK_OMEGA, MOCK_ALPHA)

    def test_returns_centred_params(self):
        cp = dp2cp(self.dp, family="SC")
        assert isinstance(cp, CentredParams)

    def test_mean_shape(self):
        cp = dp2cp(self.dp, family="SC")
        assert cp.mean.shape == (2,)

    def test_nu_is_1(self):
        cp = dp2cp(self.dp, family="SC")
        assert cp.nu == 1


class TestDp2cpESN:
    """dp2cp with family='ESN' (Extended Skew Normal, tau=0 reduces to SN)."""

    def setup_method(self):
        self.dp = DirectParams(MOCK_XI, MOCK_OMEGA, MOCK_ALPHA)

    def test_returns_centred_params(self):
        cp = dp2cp(self.dp, family="ESN")
        assert isinstance(cp, CentredParams)

    def test_esn_tau0_matches_sn(self):
        """ESN with tau=0 must give the same result as SN."""
        cp_sn = dp2cp(self.dp, family="SN")
        cp_esn = dp2cp(self.dp, family="ESN")
        np.testing.assert_allclose(cp_esn.mean, cp_sn.mean, atol=ATOL)
        np.testing.assert_allclose(cp_esn.skew, cp_sn.skew, atol=ATOL)


class TestDp2cpInvalidFamily:
    """dp2cp raises for unrecognised family strings."""

    def setup_method(self):
        self.dp = DirectParams(MOCK_XI, MOCK_OMEGA, MOCK_ALPHA)

    def test_unknown_family_raises(self):
        with pytest.raises(ValueError, match="Unknown family"):
            dp2cp(self.dp, family="XYZ")
