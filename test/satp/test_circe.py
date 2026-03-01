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
  and ``fit_circe()`` tests (which require PAQ_IDS columns).
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
# ISD data fixture (8-PAQ format required by compute_bfgs_fit / fit_circe)
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


@pytest.fixture(scope="module")
def isd_with_participant():
    """
    ISD PAQ data with SessionID as participant grouper.

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


# ---------------------------------------------------------------------------
# Tests for bfgs() / extract_bfgs_fit() wrappers  ← NUMERICAL REGRESSION ANCHORS
# These tests must not be weakened or removed — they verify the actual R computation.
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

    def test_circe_model_field_is_enum(self, isd_cor, isd_n):
        """CircE.model must be a CircModelE enum value."""
        from soundscapy.satp.circe import CircE, CircModelE

        result = CircE.compute_bfgs_fit(
            isd_cor, isd_n, "ISD", "EN", CircModelE.UNCONSTRAINED
        )
        assert result.model is CircModelE.UNCONSTRAINED

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
        """CircE.p must equal scipy_chi2.sf(chisq, d) exactly."""
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

    def test_polar_angles_is_series_for_free_angle_models(self, isd_cor, isd_n):
        """polar_angles must be a pd.Series with PAQ_IDS index for UNCONSTRAINED and EQUAL_COM."""
        from soundscapy.satp.circe import CircE, CircModelE
        from soundscapy.surveys.survey_utils import PAQ_IDS

        for model in (CircModelE.UNCONSTRAINED, CircModelE.EQUAL_COM):
            result = CircE.compute_bfgs_fit(isd_cor, isd_n, "ISD", "EN", model)
            assert isinstance(result.polar_angles, pd.Series), (
                f"{model.value}: polar_angles should be a Series, got {type(result.polar_angles)}"
            )
            assert list(result.polar_angles.index) == PAQ_IDS, (
                f"{model.value}: polar_angles index should be PAQ_IDS"
            )
            assert len(result.polar_angles) == 8

    def test_polar_angles_none_for_constrained_angle_models(self, isd_cor, isd_n):
        """polar_angles must be None for EQUAL_ANG and CIRCUMPLEX models."""
        from soundscapy.satp.circe import CircE, CircModelE

        for model in (CircModelE.EQUAL_ANG, CircModelE.CIRCUMPLEX):
            result = CircE.compute_bfgs_fit(isd_cor, isd_n, "ISD", "EN", model)
            assert result.polar_angles is None, (
                f"{model.value}: polar_angles should be None for constrained models"
            )

    def test_circe_to_dict_has_expected_keys(self, isd_cor, isd_n):
        """to_dict() must include all fit statistics and PAQ1-PAQ8 columns."""
        from soundscapy.satp.circe import CircE, CircModelE
        from soundscapy.surveys.survey_utils import PAQ_IDS

        result = CircE.compute_bfgs_fit(
            isd_cor, isd_n, "ISD", "EN", CircModelE.UNCONSTRAINED
        )
        d = result.to_dict()
        expected_keys = {
            "datasource", "language", "model", "n", "m", "chisq", "d", "p",
            "cfi", "gfi", "agfi", "srmr", "mcsc", "rmsea", "rmsea_l", "rmsea_u",
            "gdiff", *PAQ_IDS,
        }
        assert expected_keys.issubset(d.keys())

    def test_circe_to_dict_paq_values_match_polar_angles(self, isd_cor, isd_n):
        """PAQ values in to_dict() must match the polar_angles Series."""
        from soundscapy.satp.circe import CircE, CircModelE
        from soundscapy.surveys.survey_utils import PAQ_IDS

        result = CircE.compute_bfgs_fit(
            isd_cor, isd_n, "ISD", "EN", CircModelE.UNCONSTRAINED
        )
        d = result.to_dict()
        for paq in PAQ_IDS:
            assert d[paq] == pytest.approx(result.polar_angles[paq])

    def test_circe_to_dict_paq_none_for_constrained(self, isd_cor, isd_n):
        """PAQ1-PAQ8 must be None in to_dict() for constrained-angle models."""
        from soundscapy.satp.circe import CircE, CircModelE
        from soundscapy.surveys.survey_utils import PAQ_IDS

        for model in (CircModelE.EQUAL_ANG, CircModelE.CIRCUMPLEX):
            result = CircE.compute_bfgs_fit(isd_cor, isd_n, "ISD", "EN", model)
            d = result.to_dict()
            for paq in PAQ_IDS:
                assert d[paq] is None, (
                    f"{model.value}: {paq} should be None in to_dict()"
                )


# ---------------------------------------------------------------------------
# Tests for CircModelE enum properties
# ---------------------------------------------------------------------------


@pytest.mark.optional_deps("satp")
class TestCircModelEProperties:
    """Tests for the equal_ang / equal_com properties on CircModelE."""

    def test_unconstrained_has_no_constraints(self):
        from soundscapy.satp.circe import CircModelE

        assert CircModelE.UNCONSTRAINED.equal_ang is False
        assert CircModelE.UNCONSTRAINED.equal_com is False

    def test_equal_ang_constrains_angles_only(self):
        from soundscapy.satp.circe import CircModelE

        assert CircModelE.EQUAL_ANG.equal_ang is True
        assert CircModelE.EQUAL_ANG.equal_com is False

    def test_equal_com_constrains_communalities_only(self):
        from soundscapy.satp.circe import CircModelE

        assert CircModelE.EQUAL_COM.equal_ang is False
        assert CircModelE.EQUAL_COM.equal_com is True

    def test_circumplex_constrains_both(self):
        from soundscapy.satp.circe import CircModelE

        assert CircModelE.CIRCUMPLEX.equal_ang is True
        assert CircModelE.CIRCUMPLEX.equal_com is True


# ---------------------------------------------------------------------------
# Tests for fit_circe() function and ipsatize()
# ---------------------------------------------------------------------------


@pytest.mark.optional_deps("satp")
class TestFitCirce:
    """Integration tests for fit_circe() and ipsatize()."""

    def test_ipsatize_per_participant_mean_zero(self, isd_with_participant):
        """
        After ipsatization each PAQ column must have zero mean per participant.

        The implementation uses groupby.transform, which centers each column
        within each participant group.  The invariant is: for every
        (participant, PAQ_col) pair, the mean across the participant's rows is zero.
        """
        from soundscapy.satp.circe import ipsatize
        from soundscapy.surveys.survey_utils import PAQ_IDS

        ipsatized = ipsatize(isd_with_participant, by="participant")
        # groupby.transform preserves the original index, so joining back is safe.
        check = ipsatized[PAQ_IDS].assign(
            participant=isd_with_participant["participant"]
        )
        group_means = check.groupby("participant")[PAQ_IDS].mean()
        np.testing.assert_allclose(group_means.to_numpy(), 0.0, atol=1e-10)

    def test_fit_circe_returns_dataframe(self, isd_with_participant):
        """fit_circe() must return a pd.DataFrame."""
        from soundscapy.satp.circe import fit_circe

        result = fit_circe(isd_with_participant, language="EN", datasource="ISD")
        assert isinstance(result, pd.DataFrame)

    def test_fit_circe_returns_four_rows(self, isd_with_participant):
        """fit_circe() with default models must return 4 rows (one per model)."""
        from soundscapy.satp.circe import fit_circe

        result = fit_circe(isd_with_participant, language="EN", datasource="ISD")
        assert len(result) == 4

    def test_fit_circe_model_column_contains_all_variants(self, isd_with_participant):
        """The 'model' column must contain all four CircModelE string values."""
        from soundscapy.satp.circe import CircModelE, fit_circe

        result = fit_circe(isd_with_participant, language="EN", datasource="ISD")
        expected = {m.value for m in CircModelE}
        assert set(result["model"]) == expected

    def test_fit_circe_numeric_fit_indices(self, isd_with_participant):
        """chisq, cfi, rmsea must be numeric floats (not None or NaN) in all rows."""
        from soundscapy.satp.circe import fit_circe

        result = fit_circe(isd_with_participant, language="EN", datasource="ISD")
        for col in ("chisq", "cfi", "rmsea", "d"):
            assert result[col].notna().all(), f"Column '{col}' has NaN values"
            assert pd.api.types.is_numeric_dtype(result[col]), (
                f"Column '{col}' is not numeric"
            )

    def test_fit_circe_p_value_formula(self, isd_with_participant):
        """p in the results must equal scipy_chi2.sf(chisq, d) for each row."""
        from soundscapy.satp.circe import fit_circe

        result = fit_circe(isd_with_participant, language="EN", datasource="ISD")
        for _, row in result.iterrows():
            expected_p = scipy_chi2.sf(row["chisq"], row["d"])
            assert pytest.approx(row["p"], rel=1e-6) == expected_p

    def test_fit_circe_n_uses_listwise_deletion(self, isd_with_participant):
        """
        n in results must equal len(data[PAQ_IDS].dropna()) after ipsatization.

        Introducing NaN rows verifies listwise deletion is applied.
        """
        from soundscapy.satp.circe import fit_circe, ipsatize
        from soundscapy.surveys.survey_utils import PAQ_IDS

        # Introduce NaN in one PAQ column for a single participant's rows
        data_with_nan = isd_with_participant.copy()
        first_participant = data_with_nan["participant"].iloc[0]
        mask = data_with_nan["participant"] == first_participant
        data_with_nan.loc[mask, "PAQ1"] = np.nan

        result = fit_circe(data_with_nan, language="EN", datasource="ISD")

        # Manually compute expected n
        ipsatized = ipsatize(data_with_nan, by="participant")
        expected_n = len(ipsatized[PAQ_IDS].dropna())

        # All rows should report the same n
        assert (result["n"] == expected_n).all(), (
            f"n={result['n'].unique()} but expected {expected_n}"
        )

    def test_fit_circe_subset_of_models(self, isd_with_participant):
        """fit_circe() with models=[...] must fit only those models."""
        from soundscapy.satp.circe import CircModelE, fit_circe

        result = fit_circe(
            isd_with_participant,
            language="EN",
            datasource="ISD",
            models=[CircModelE.UNCONSTRAINED, CircModelE.CIRCUMPLEX],
        )
        assert len(result) == 2
        assert set(result["model"]) == {
            CircModelE.UNCONSTRAINED.value,
            CircModelE.CIRCUMPLEX.value,
        }

    def test_fit_circe_rmsea_bounds_ordering(self, isd_with_participant):
        """rmsea_l <= rmsea <= rmsea_u must hold for every row."""
        from soundscapy.satp.circe import fit_circe

        result = fit_circe(isd_with_participant, language="EN", datasource="ISD")
        for _, row in result.iterrows():
            assert row["rmsea_l"] <= row["rmsea"], (
                f"{row['model']}: rmsea_l ({row['rmsea_l']}) > rmsea ({row['rmsea']})"
            )
            assert row["rmsea"] <= row["rmsea_u"], (
                f"{row['model']}: rmsea ({row['rmsea']}) > rmsea_u ({row['rmsea_u']})"
            )

    def test_fit_circe_ipsatize_false(self, isd_with_participant):
        """ipsatize_data=False must run without error and return a 4-row DataFrame."""
        from soundscapy.satp.circe import fit_circe

        result = fit_circe(
            isd_with_participant,
            language="EN",
            datasource="ISD",
            ipsatize_data=False,
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 4

    def test_fit_circe_empty_models_returns_empty_df(self, isd_with_participant):
        """fit_circe() with models=[] must return an empty DataFrame."""
        from soundscapy.satp.circe import fit_circe

        result = fit_circe(
            isd_with_participant, language="EN", datasource="ISD", models=[]
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_fit_circe_error_row_structure(self, isd_with_participant):
        """When a model raises during fitting, the error row has the expected keys."""
        from unittest.mock import patch

        from soundscapy.satp.circe import CircE, CircModelE, fit_circe

        # Patch compute_bfgs_fit for one specific model to simulate convergence failure.
        original = CircE.compute_bfgs_fit

        def failing_fit(data_cor, n, datasource, language, circ_model):
            if circ_model is CircModelE.UNCONSTRAINED:
                raise RuntimeError("simulated convergence failure")
            return original(data_cor, n, datasource, language, circ_model)

        with patch.object(CircE, "compute_bfgs_fit", staticmethod(failing_fit)):
            result = fit_circe(isd_with_participant, language="EN", datasource="ISD")

        error_rows = result[result["model"] == CircModelE.UNCONSTRAINED.value]
        assert len(error_rows) == 1
        row = error_rows.iloc[0]
        assert "error" in row.index
        assert row["language"] == "EN"
        assert row["datasource"] == "ISD"
        assert row["model"] == CircModelE.UNCONSTRAINED.value
        assert "convergence failure" in row["error"]

    def test_fit_circe_n_zero_raises(self):
        """fit_circe() must raise ValueError when no complete cases remain.

        Uses a 0-row slice of valid ISD data, which passes schema validation
        but yields n=0 after listwise deletion.
        """
        import warnings

        import soundscapy as sspy
        from soundscapy.satp.circe import fit_circe

        data = sspy.isd.load().rename(columns={"SessionID": "participant"})
        empty_data = data.iloc[0:0].copy()  # 0 rows, correct column structure

        with pytest.raises(ValueError, match="No complete cases"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                fit_circe(empty_data, language="EN", datasource="ISD")

    def test_satp_schema_participant_case_insensitive(self):
        """SATPSchema must accept 'PARTICIPANT' and normalise it to 'participant'."""
        from soundscapy.satp.circe import SATPSchema
        from soundscapy.surveys.survey_utils import PAQ_IDS

        # Build a tiny valid DataFrame with an ALL-CAPS participant column.
        rng = np.random.default_rng(0)
        df = pd.DataFrame(
            rng.uniform(0, 100, size=(4, 8)), columns=PAQ_IDS
        )
        df["PARTICIPANT"] = ["A", "A", "B", "B"]

        validated = SATPSchema.validate(df, lazy=True)
        assert "participant" in validated.columns
        assert "PARTICIPANT" not in validated.columns


# ---------------------------------------------------------------------------
# Tests for gdiff property
# ---------------------------------------------------------------------------


@pytest.mark.optional_deps("satp")
class TestGdiff:
    """Tests for CircE.gdiff property."""

    def test_gdiff_is_none_for_constrained_models(self, isd_cor, isd_n):
        """gdiff must be None for EQUAL_ANG and CIRCUMPLEX (no free angles)."""
        from soundscapy.satp.circe import CircE, CircModelE

        for model in (CircModelE.EQUAL_ANG, CircModelE.CIRCUMPLEX):
            result = CircE.compute_bfgs_fit(isd_cor, isd_n, "ISD", "EN", model)
            assert result.gdiff is None, (
                f"{model.value}: gdiff should be None when polar_angles is None"
            )

    def test_gdiff_is_float_for_free_angle_models(self, isd_cor, isd_n):
        """gdiff must be a non-negative float for UNCONSTRAINED and EQUAL_COM."""
        from soundscapy.satp.circe import CircE, CircModelE

        for model in (CircModelE.UNCONSTRAINED, CircModelE.EQUAL_COM):
            result = CircE.compute_bfgs_fit(isd_cor, isd_n, "ISD", "EN", model)
            assert isinstance(result.gdiff, float), (
                f"{model.value}: gdiff should be float, got {type(result.gdiff)}"
            )
            assert 0.0 <= result.gdiff <= 180.0, (
                f"{model.value}: gdiff={result.gdiff} out of plausible range [0, 180]"
            )
