"""
Integration tests for the CircE wrapper.

Reference values are derived from the CircE R package itself
(https://github.com/MitchellAcoustics/CircE-R).  The vocational interests
example from CircE.BFGS.Rd is run directly via our wrapper and the output
is verified against the values printed by the R package.

Two datasets are used:
- ``VOCATIONAL_COR`` / ``VOCATIONAL_N``: 7-variable example from the CircE
  package docs — used for low-level ``bfgs()`` / ``extract_bfgs_fit()`` tests
  where we have exact reference values.
- ISD data: 8-PAQ SATP-format data — used for ``CircE.compute_bfgs_fit()``
  and ``SATP`` tests (which require PAQ_IDS columns).
"""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import chi2 as scipy_chi2

# ---------------------------------------------------------------------------
# Vocational interests correlation matrix from CircE.BFGS.Rd example
# (N=175, 7 variables, m=3)
# ---------------------------------------------------------------------------

_V_NAMES = [
    "Health",
    "Science",
    "Technology",
    "Trades",
    "Business Operations",
    "Business Contact",
    "Social",
]

_R_LOWER = np.array(
    [
        [1, 0, 0, 0, 0, 0, 0],
        [0.654, 1, 0, 0, 0, 0, 0],
        [0.453, 0.644, 1, 0, 0, 0, 0],
        [0.251, 0.440, 0.757, 1, 0, 0, 0],
        [0.122, 0.158, 0.551, 0.493, 1, 0, 0],
        [0.218, 0.210, 0.570, 0.463, 0.754, 1, 0],
        [0.496, 0.264, 0.366, 0.202, 0.471, 0.650, 1],
    ]
)
_R_SYM = _R_LOWER + _R_LOWER.T - np.diag(np.diag(_R_LOWER))
VOCATIONAL_COR = pd.DataFrame(_R_SYM, index=_V_NAMES, columns=_V_NAMES)
VOCATIONAL_N = 175


# ---------------------------------------------------------------------------
# ISD data fixture (8-PAQ format required by compute_bfgs_fit / SATP)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def isd_paqs():
    """Return the ISD PAQ data (dropna) for use in SATP tests."""
    import soundscapy as sspy
    from soundscapy.surveys.survey_utils import PAQ_IDS

    return sspy.isd.load()[PAQ_IDS].dropna()


@pytest.fixture(scope="module")
def isd_cor(isd_paqs):
    """Correlation matrix of ISD PAQ data."""
    return isd_paqs.corr()


@pytest.fixture(scope="module")
def isd_n(isd_paqs):
    """Sample size of ISD PAQ data."""
    return len(isd_paqs)


# ---------------------------------------------------------------------------
# Tests for bfgs() / extract_bfgs_fit() wrappers
# ---------------------------------------------------------------------------


@pytest.mark.optional_deps("satp")
class TestBfgsWrapper:
    """
    Direct tests of the bfgs() and extract_bfgs_fit() wrappers.

    All reference values come from running CircE.BFGS(R, v.names, m=3, N=175)
    in R and reading the printed output.
    """

    def test_bfgs_returns_list_vector(self):
        """bfgs() should return an rpy2 ListVector (the raw R model object)."""
        from rpy2.robjects import ListVector

        from soundscapy.r_wrapper._circe_wrapper import bfgs

        result = bfgs(
            VOCATIONAL_COR,
            n=VOCATIONAL_N,
            scales=_V_NAMES,
            m_val=3,
            equal_ang=False,
            equal_com=False,
        )
        assert isinstance(result, ListVector)

    def test_extract_bfgs_fit_returns_dict(self):
        """extract_bfgs_fit() should return a plain Python dict."""
        from soundscapy.r_wrapper._circe_wrapper import bfgs, extract_bfgs_fit

        fit = extract_bfgs_fit(
            bfgs(
                VOCATIONAL_COR,
                n=VOCATIONAL_N,
                scales=_V_NAMES,
                m_val=3,
                equal_ang=False,
                equal_com=False,
            )
        )
        assert isinstance(fit, dict)

    def test_bfgs_fit_keys_present(self):
        """extract_bfgs_fit() result must contain the expected fit-statistic keys."""
        from soundscapy.r_wrapper._circe_wrapper import bfgs, extract_bfgs_fit

        fit = extract_bfgs_fit(
            bfgs(
                VOCATIONAL_COR,
                n=VOCATIONAL_N,
                scales=_V_NAMES,
                m_val=3,
                equal_ang=False,
                equal_com=False,
            )
        )
        required = {
            "chisq",
            "d",
            "dfnull",
            "p",
            "rmsea",
            "rmsea.l",
            "rmsea.u",
            "cfi",
            "gfi",
            "agfi",
            "srmr",
            "mcsc",
            "m",
        }
        assert required.issubset(fit.keys())

    # ------------------------------------------------------------------
    # Reference values from the CircE R package:
    #   model <- CircE.BFGS(R, v.names, m=3, N=175)
    #
    # Output (printed by R, rounded to 3 d.p.):
    #   chi-sq = 11.598,  Model df = 5,  Null df = 21
    #   p = 0.041   (Ho: perfect fit)
    #   RMSEA = 0.087  [0.017, 0.154]
    #   CFI = 0.991, GFI = 0.989, AGFI = 0.938, SRMR = 0.038, MCSC = 0.29
    # ------------------------------------------------------------------

    def test_bfgs_unconstrained_chisq(self):
        """Chi-square statistic must match R package reference value (±0.01)."""
        from soundscapy.r_wrapper._circe_wrapper import bfgs, extract_bfgs_fit

        fit = extract_bfgs_fit(
            bfgs(
                VOCATIONAL_COR,
                n=VOCATIONAL_N,
                scales=_V_NAMES,
                m_val=3,
                equal_ang=False,
                equal_com=False,
            )
        )
        assert pytest.approx(fit["chisq"], abs=0.01) == 11.598

    def test_bfgs_unconstrained_model_df(self):
        """Model degrees of freedom must be 5 (not 21 = dfnull)."""
        from soundscapy.r_wrapper._circe_wrapper import bfgs, extract_bfgs_fit

        fit = extract_bfgs_fit(
            bfgs(
                VOCATIONAL_COR,
                n=VOCATIONAL_N,
                scales=_V_NAMES,
                m_val=3,
                equal_ang=False,
                equal_com=False,
            )
        )
        assert int(fit["d"]) == 5

    def test_bfgs_unconstrained_p_value(self):
        """
        p-value must be computed against model df (d=5), not null df (dfnull=21).

        R reference: p = 0.041.
        If dfnull=21 were (wrongly) used the result would be ~0.95.
        """
        from soundscapy.r_wrapper._circe_wrapper import bfgs, extract_bfgs_fit

        fit = extract_bfgs_fit(
            bfgs(
                VOCATIONAL_COR,
                n=VOCATIONAL_N,
                scales=_V_NAMES,
                m_val=3,
                equal_ang=False,
                equal_com=False,
            )
        )
        assert pytest.approx(fit["p"], abs=0.005) == 0.041
        assert fit["p"] < 0.1, "p ≈ 0.95 suggests wrong df was used"

    def test_bfgs_p_equals_scipy_chi2_against_model_df(self):
        """The stored p must equal scipy_chi2.sf(chisq, d) exactly."""
        from soundscapy.r_wrapper._circe_wrapper import bfgs, extract_bfgs_fit

        fit = extract_bfgs_fit(
            bfgs(
                VOCATIONAL_COR,
                n=VOCATIONAL_N,
                scales=_V_NAMES,
                m_val=3,
                equal_ang=False,
                equal_com=False,
            )
        )
        expected_p = scipy_chi2.sf(fit["chisq"], fit["d"])
        assert pytest.approx(fit["p"], rel=1e-6) == expected_p

    def test_bfgs_unconstrained_fit_indices(self):
        """Fit indices must match R package reference values (±0.001)."""
        from soundscapy.r_wrapper._circe_wrapper import bfgs, extract_bfgs_fit

        fit = extract_bfgs_fit(
            bfgs(
                VOCATIONAL_COR,
                n=VOCATIONAL_N,
                scales=_V_NAMES,
                m_val=3,
                equal_ang=False,
                equal_com=False,
            )
        )
        assert pytest.approx(float(fit["rmsea"]), abs=0.001) == 0.087
        assert pytest.approx(float(fit["rmsea.l"]), abs=0.001) == 0.017
        assert pytest.approx(float(fit["rmsea.u"]), abs=0.001) == 0.154
        assert pytest.approx(float(fit["cfi"]), abs=0.001) == 0.991
        assert pytest.approx(float(fit["gfi"]), abs=0.001) == 0.989
        assert pytest.approx(float(fit["agfi"]), abs=0.001) == 0.938
        assert pytest.approx(float(fit["srmr"]), abs=0.001) == 0.038
        assert pytest.approx(float(fit["mcsc"]), abs=0.001) == 0.290

    def test_bfgs_equal_com_reference(self):
        """
        Equal-communalities model must match R package output.

        R reference (equal_ang=False, equal_com=True):
          chi-sq = 50.409, Model df = 11, RMSEA = 0.143, CFI = 0.946, SRMR = 0.060
        """
        from soundscapy.r_wrapper._circe_wrapper import bfgs, extract_bfgs_fit

        fit = extract_bfgs_fit(
            bfgs(
                VOCATIONAL_COR,
                n=VOCATIONAL_N,
                scales=_V_NAMES,
                m_val=3,
                equal_ang=False,
                equal_com=True,
            )
        )
        assert pytest.approx(fit["chisq"], abs=0.01) == 50.409
        assert int(fit["d"]) == 11
        assert pytest.approx(float(fit["rmsea"]), abs=0.001) == 0.143
        assert pytest.approx(float(fit["cfi"]), abs=0.001) == 0.946
        assert pytest.approx(float(fit["srmr"]), abs=0.001) == 0.060
        # p-value must be very small for this over-constrained model
        assert fit["p"] < 0.001


# ---------------------------------------------------------------------------
# Tests for CircE dataclass (uses ISD 8-PAQ data)
# ---------------------------------------------------------------------------


@pytest.mark.optional_deps("satp")
class TestCircEDataclass:
    """Tests for CircE.compute_bfgs_fit() using ISD data (8-PAQ format)."""

    def test_compute_bfgs_fit_returns_circe(self, isd_cor, isd_n):
        """compute_bfgs_fit() must return a CircE instance."""
        from soundscapy.satp.circe import CircE, CircModelE

        result = CircE.compute_bfgs_fit(
            isd_cor, isd_n, "ISD", "EN", CircModelE.UNCONSTRAINED
        )
        assert isinstance(result, CircE)

    def test_circe_n_matches_input(self, isd_cor, isd_n):
        """CircE.n must equal the n passed to compute_bfgs_fit, not N-1 from R."""
        from soundscapy.satp.circe import CircE, CircModelE

        result = CircE.compute_bfgs_fit(
            isd_cor, isd_n, "ISD", "EN", CircModelE.UNCONSTRAINED
        )
        assert result.n == isd_n

    def test_circe_d_is_model_df_not_null_df(self, isd_cor, isd_n):
        """
        CircE.d must be the model degrees of freedom, not dfnull=28 (8-var null).

        For 8 variables, dfnull = 8*7/2 = 28.  The unconstrained model df is
        much smaller.
        """
        from soundscapy.satp.circe import CircE, CircModelE

        result = CircE.compute_bfgs_fit(
            isd_cor, isd_n, "ISD", "EN", CircModelE.UNCONSTRAINED
        )
        # Model df must be strictly less than null df (28 for 8 variables)
        assert result.d < 28, (
            f"CircE.d = {result.d} — looks like dfnull was used instead of d."
        )
        assert result.d > 0

    def test_circe_p_value_formula(self, isd_cor, isd_n):
        """CircE.p must equal scipy_chi2.sf(chisq, df) exactly."""
        from soundscapy.satp.circe import CircE, CircModelE

        result = CircE.compute_bfgs_fit(
            isd_cor, isd_n, "ISD", "EN", CircModelE.UNCONSTRAINED
        )
        expected_p = scipy_chi2.sf(result.chisq, result.d)
        assert pytest.approx(result.p, rel=1e-6) == expected_p

    def test_circe_fit_stats_plausible(self, isd_cor, isd_n):
        """All fit statistics must be in their valid ranges."""
        from soundscapy.satp.circe import CircE, CircModelE

        result = CircE.compute_bfgs_fit(
            isd_cor, isd_n, "ISD", "EN", CircModelE.UNCONSTRAINED
        )
        assert result.chisq >= 0
        assert 0.0 <= result.p <= 1.0
        assert result.rmsea >= 0
        assert 0.0 <= result.cfi <= 1.0
        assert 0.0 <= result.gfi <= 1.0
        assert result.srmr >= 0

    def test_polar_angles_present_for_free_angle_models(self, isd_cor, isd_n):
        """polar_angles must be a DataFrame for UNCONSTRAINED and EQUAL_COM models."""
        from soundscapy.satp.circe import CircE, CircModelE

        for model in (CircModelE.UNCONSTRAINED, CircModelE.EQUAL_COM):
            result = CircE.compute_bfgs_fit(isd_cor, isd_n, "ISD", "EN", model)
            got = type(result.polar_angles)
            assert isinstance(result.polar_angles, pd.DataFrame), (
                f"{model.value}: polar_angles should be a DataFrame, got {got}"
            )
            assert result.polar_angles.shape[1] == 8, (
                f"{model.value}: expected 8 variables in polar_angles columns"
            )

    def test_polar_angles_none_for_constrained_angle_models(self, isd_cor, isd_n):
        """polar_angles must be None for EQUAL_ANG and CIRCUMPLEX models."""
        from soundscapy.satp.circe import CircE, CircModelE

        for model in (CircModelE.EQUAL_ANG, CircModelE.CIRCUMPLEX):
            result = CircE.compute_bfgs_fit(isd_cor, isd_n, "ISD", "EN", model)
            assert result.polar_angles is None, (
                f"{model.value}: polar_angles should be None for constrained models"
            )


# ---------------------------------------------------------------------------
# Tests for SATP class
# ---------------------------------------------------------------------------


@pytest.mark.optional_deps("satp")
class TestSATP:
    """Integration tests for the full SATP analysis pipeline."""

    @pytest.fixture
    def satp_data(self):
        """
        PAQ data from ISD using SessionID as participant grouper.

        The ISD dataset has 66 sessions, each with ~54 rows.  Using SessionID
        ensures every participant has many rows so ipsatization produces a
        valid (non-degenerate) correlation matrix.
        """
        import soundscapy as sspy
        from soundscapy.surveys.survey_utils import PAQ_IDS

        raw = sspy.isd.load()
        return (
            raw[[*PAQ_IDS, "SessionID"]]
            .dropna(subset=PAQ_IDS)
            .copy()
            .rename(columns={"SessionID": "participant"})
        )

    def test_satp_init_validates_schema(self, satp_data):
        """SATP.__init__ must accept valid SATP-format data without raising."""
        from soundscapy.satp.circe import SATP

        satp = SATP(satp_data, language="EN", datasource="ISD")
        assert satp is not None

    def test_satp_ipsatize(self, satp_data):
        """
        After ipsatization each PAQ column must have zero mean per participant.

        The implementation uses groupby.transform, which centers each *column*
        within each participant group, not each row across columns.  The correct
        invariant is therefore: for every (participant, PAQ_col) pair, the mean
        of that column across the participant's rows is zero.
        """
        from soundscapy.satp.circe import SATP
        from soundscapy.surveys.survey_utils import PAQ_IDS

        # Test _ipsatize_df directly so we can still access the participant labels
        # (satp.data loses the participant column after the transform).
        ipsatized = SATP._ipsatize_df(satp_data, by="participant")
        # groupby.transform preserves the original index, so joining back is safe.
        check = ipsatized[PAQ_IDS].assign(participant=satp_data["participant"])
        group_means = check.groupby("participant")[PAQ_IDS].mean()
        np.testing.assert_allclose(group_means.to_numpy(), 0.0, atol=1e-10)

    def test_satp_run_single_model(self, satp_data):
        """SATP.run(circ_model=...) must populate exactly that model slot."""
        from soundscapy.satp.circe import SATP, CircE, CircModelE

        satp = SATP(satp_data, language="EN", datasource="ISD")
        satp.run(circ_model=CircModelE.UNCONSTRAINED)

        assert isinstance(satp.model_results[CircModelE.UNCONSTRAINED], CircE)
        for model in [
            CircModelE.EQUAL_ANG,
            CircModelE.EQUAL_COM,
            CircModelE.CIRCUMPLEX,
        ]:
            assert satp.model_results[model] is None

    def test_satp_run_captures_n_correctly(self, satp_data):
        """The n stored on the CircE result must equal len(satp.data)."""
        from soundscapy.satp.circe import SATP, CircModelE

        satp = SATP(satp_data, language="EN", datasource="ISD")
        n_data = len(satp.data)
        satp.run(circ_model=CircModelE.UNCONSTRAINED)

        result = satp.model_results[CircModelE.UNCONSTRAINED]
        assert result is not None
        assert result.n == n_data

    def test_satp_run_p_value_formula(self, satp_data):
        """CircE.p from a full SATP run must equal scipy_chi2.sf(chisq, df)."""
        from soundscapy.satp.circe import SATP, CircModelE

        satp = SATP(satp_data, language="EN", datasource="ISD")
        satp.run(circ_model=CircModelE.UNCONSTRAINED)

        result = satp.model_results[CircModelE.UNCONSTRAINED]
        assert result is not None
        expected_p = scipy_chi2.sf(result.chisq, result.d)
        assert pytest.approx(result.p, rel=1e-6) == expected_p

    def test_satp_run_all_models_errors_captured(self, satp_data):
        """
        SATP.run() runs all models; convergence failures are captured.

        Failures are stored in _errors and never propagate as exceptions.
        """
        from soundscapy.satp.circe import SATP

        satp = SATP(satp_data, language="EN", datasource="ISD")
        satp.run()  # must not raise

        n_results = sum(v is not None for v in satp.model_results.values())
        n_errors = len(satp._errors)
        assert n_results + n_errors == 4
