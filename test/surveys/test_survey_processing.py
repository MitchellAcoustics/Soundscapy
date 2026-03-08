import numpy as np
import pandas as pd
import pytest

from soundscapy.surveys.processing import (
    add_iso_coords,
    calculate_iso_coords,
    likert_data_quality,
    simulation,
    ssm_metrics,
)
from soundscapy.surveys.survey_utils import (
    LANGUAGE_ANGLES,
    PAQ_IDS,
    rename_paqs,
    return_paqs,
)


@pytest.fixture
def basic_test_df():
    return pd.DataFrame(
        {
            "RecordID": ["EX1", "EX2", "EX3", "EX4"],
            "GroupID": ["A", "A", "B", "B"],
            "PAQ1": [4, 2, 3, 5],
            "PAQ2": [4, 3, 5, 2],
            "PAQ3": [4, 5, 2, 3],
            "PAQ4": [2, 5, 3, 4],
            "PAQ5": [1, 5, 4, 2],
            "PAQ6": [3, 5, 2, 4],
            "PAQ7": [3, 3, 5, 1],
            "PAQ8": [4, 1, 2, 5],
        }
    )


@pytest.fixture
def name_test_df():
    return pd.DataFrame(
        {
            "RecordID": ["EX1", "EX2"],
            "pleasant": [4, 2],
            "vibrant": [4, 3],
            "eventful": [4, 5],
            "chaotic": [2, 5],
            "annoying": [1, 5],
            "monotonous": [3, 5],
            "uneventful": [3, 3],
            "calm": [4, 1],
        }
    )


class TestSurveyUtils:
    def test_return_paqs(self, basic_test_df):
        result = return_paqs(basic_test_df)
        assert list(result.columns) == ["RecordID", "GroupID", *PAQ_IDS]

        result = return_paqs(basic_test_df, incl_ids=False, other_cols=["GroupID"])
        assert list(result.columns) == [*PAQ_IDS, "GroupID"]

    def test_rename_paqs(self, name_test_df):
        result = rename_paqs(name_test_df)
        assert list(result.columns) == ["RecordID", *PAQ_IDS]

        custom_aliases = {
            "pl": "PAQ1",
            "ch": "PAQ4",
            "ca": "PAQ8",
            "v": "PAQ2",
            "ev": "PAQ3",
            "un": "PAQ7",
            "ann": "PAQ5",
            "m": "PAQ6",
        }
        data = pd.DataFrame(
            {
                "RecordID": ["EX1", "EX2"],
                "pl": [4, 2],
                "ch": [4, 3],
                "ca": [4, 5],
                "v": [2, 5],
                "ev": [1, 5],
                "un": [3, 5],
                "ann": [3, 3],
                "m": [4, 1],
            }
        )
        result = rename_paqs(data, paq_aliases=custom_aliases)
        assert list(result.columns) == ["RecordID", *list(custom_aliases.values())]


class TestISOCoordinates:
    def test_calculate_iso_coords(self, basic_test_df):
        iso_pleasant, iso_eventful = calculate_iso_coords(basic_test_df)
        assert isinstance(iso_pleasant, pd.Series)
        assert isinstance(iso_eventful, pd.Series)
        assert len(iso_pleasant) == len(basic_test_df)
        assert len(iso_eventful) == len(basic_test_df)
        assert iso_pleasant.iloc[0] == pytest.approx(0.53, abs=0.05)
        assert iso_eventful.iloc[0] == pytest.approx(0.03, abs=0.05)

    def test_add_iso_coords(self, basic_test_df):
        data = add_iso_coords(basic_test_df)
        assert "ISOPleasant" in data.columns
        assert "ISOEventful" in data.columns

        with pytest.raises(Warning):
            data = add_iso_coords(data, overwrite=False)

        data = add_iso_coords(data, names=("pl_test", "ev_test"), overwrite=True)
        assert "pl_test" in data.columns
        assert "ev_test" in data.columns

    @pytest.mark.xfail(
        reason="Implemented the new projection formula,"
        "Have not yet determined correct results"
    )
    def test_iso_coords_with_language_angles(self, basic_test_df):
        iso_pleasant_eng, iso_eventful_eng = calculate_iso_coords(
            basic_test_df, angles=LANGUAGE_ANGLES["eng"]
        )
        iso_pleasant_cmn, iso_eventful_cmn = calculate_iso_coords(
            basic_test_df, angles=LANGUAGE_ANGLES["cmn"]
        )

        assert iso_pleasant_eng.iloc[0] == pytest.approx(0.66, abs=0.01)
        assert iso_eventful_eng.iloc[0] == pytest.approx(0.14, abs=0.01)
        assert iso_pleasant_cmn.iloc[0] == pytest.approx(0.41, abs=0.01)
        assert iso_eventful_cmn.iloc[0] == pytest.approx(-0.09, abs=0.01)


class TestDataQuality:
    def test_likert_data_quality(self, basic_test_df):
        result = likert_data_quality(basic_test_df)
        assert result is None  # Assuming all data in basic_test_df is valid

    def test_likert_data_quality_with_invalid_data(self):
        invalid_df = pd.DataFrame(
            {
                "PAQ1": [6, 2, 3, 5],
                "PAQ2": [4, 3, 5, 2],
                "PAQ3": [4, 5, 2, 3],
                "PAQ4": [2, 5, 3, 4],
                "PAQ5": [1, 5, 4, 2],
                "PAQ6": [3, 5, 2, 4],
                "PAQ7": [3, 3, 5, 1],
                "PAQ8": [4, 1, 2, 5],
            }
        )
        result = likert_data_quality(invalid_df)
        assert result == [0]

    def test_likert_data_quality_with_nan(self):
        nan_df = pd.DataFrame(
            {
                "PAQ1": [4, 2, np.nan, 5],
                "PAQ2": [4, 3, 5, 2],
                "PAQ3": [4, 5, 2, 3],
                "PAQ4": [2, 5, 3, 4],
                "PAQ5": [1, 5, 4, 2],
                "PAQ6": [3, 5, 2, 4],
                "PAQ7": [3, 3, 5, 1],
                "PAQ8": [4, 1, 2, 5],
            }
        )
        result = likert_data_quality(nan_df)
        assert result == [2]

        result_allow_na = likert_data_quality(nan_df, allow_na=True)
        assert result_allow_na is None


class TestSimulation:
    def test_simulation(self):
        data = simulation(n=200)
        assert data.shape == (200, 8)
        assert list(data.columns) == PAQ_IDS
        assert data.min().min() >= 1
        assert data.max().max() <= 5

    def test_simulation_with_iso_coords(self):
        data = simulation(n=200, incl_iso_coords=True)
        assert data.shape == (200, 10)
        assert list(data.columns) == [*PAQ_IDS, "ISOPleasant", "ISOEventful"]
        assert not data["ISOPleasant"].isna().any()
        assert not data["ISOEventful"].isna().any()


class TestSSMMetrics:
    @pytest.fixture
    def ssm_test_df(self):
        return pd.DataFrame(
            {
                "PAQ1": [4, 2, 3, 5],
                "PAQ2": [4, 3, 5, 2],
                "PAQ3": [4, 5, 2, 3],
                "PAQ4": [2, 5, 3, 4],
                "PAQ5": [1, 5, 4, 2],
                "PAQ6": [3, 5, 2, 4],
                "PAQ7": [3, 3, 5, 1],
                "PAQ8": [4, 1, 2, 5],
            }
        )

    def test_ssm_metrics_cosine(self, ssm_test_df):
        with pytest.warns(PendingDeprecationWarning):
            result = ssm_metrics(ssm_test_df, method="cosine")
            assert isinstance(result, pd.DataFrame)
            assert list(result.columns) == [
                "amplitude",
                "angle",
                "elevation",
                "displacement",
                "r_squared",
            ]
            assert len(result) == len(ssm_test_df)
            assert (result["r_squared"] >= 0).all()
            assert (result["r_squared"] <= 1).all()

    def test_ssm_metrics_polar(self, ssm_test_df):
        with pytest.warns(PendingDeprecationWarning):
            _ = ssm_metrics(ssm_test_df, method="polar")
            # assert isinstance(result, pd.DataFrame)
            # assert list(result.columns) == [
            #     "amplitude",
            #     "angle",
            #     "elevation",
            #     "displacement",
            #     "r_squared",
            # ]
            # assert len(result) == len(ssm_test_df)
            # assert (result["displacement"] == 0).all()
            # assert (result["r_squared"] == 1).all()

    def test_ssm_metrics_invalid_method(self, ssm_test_df):
        with pytest.raises(ValueError, match="either 'polar' or 'cosine'"):
            ssm_metrics(ssm_test_df, method="invalid")

    def test_ssm_metrics_missing_columns(self):
        invalid_df = pd.DataFrame({"PAQ1": [1, 2, 3], "PAQ2": [4, 5, 6]})
        with pytest.raises(ValueError, match="not present in DataFrame"):
            ssm_metrics(invalid_df)


class TestIntegration:
    def test_end_to_end_workflow(self):
        # Simulate data
        data = simulation(n=100, incl_iso_coords=True)

        # Check data quality
        invalid_indices = likert_data_quality(data)
        assert invalid_indices is None

        # Calculate SSM metrics
        ssm_result = ssm_metrics(data)

        # Verify results
        assert data.shape == (100, 10)
        assert ssm_result.shape == (100, 5)
        assert not data["ISOPleasant"].isna().any()
        assert not data["ISOEventful"].isna().any()
        assert not ssm_result.isna().any().any()


class TestIpsatize:
    """Tests for soundscapy.surveys.ipsatize()."""

    @pytest.fixture
    def sample_data(self):
        """Small DataFrame with known values for centering checks."""
        return pd.DataFrame(
            {
                "PAQ1": [50.0, 60.0, 40.0, 30.0],
                "PAQ2": [50.0, 60.0, 40.0, 30.0],
                "PAQ3": [50.0, 60.0, 40.0, 30.0],
                "PAQ4": [50.0, 60.0, 40.0, 30.0],
                "PAQ5": [50.0, 60.0, 40.0, 30.0],
                "PAQ6": [50.0, 60.0, 40.0, 30.0],
                "PAQ7": [50.0, 60.0, 40.0, 30.0],
                "PAQ8": [50.0, 60.0, 40.0, 30.0],
                "participant": ["A", "A", "B", "B"],
            }
        )

    def test_grand_mean_one_scalar_per_participant(self, sample_data):
        """Grand-mean centering must produce zero grand mean per participant."""
        from soundscapy.surveys import ipsatize
        from soundscapy.surveys.survey_utils import PAQ_IDS

        result = ipsatize(
            sample_data, method="grand_mean", participant_col="participant"
        )
        check = result[PAQ_IDS].assign(participant=sample_data["participant"].values)
        flat_means = check.groupby("participant")[PAQ_IDS].apply(
            lambda df: float(df.values.mean())
        )
        np.testing.assert_allclose(flat_means.to_numpy(), 0.0, atol=1e-10)

    def test_column_wise_zero_per_scale_per_participant(self, sample_data):
        """Column-wise centering must produce zero mean per scale per participant."""
        from soundscapy.surveys import ipsatize
        from soundscapy.surveys.survey_utils import PAQ_IDS

        result = ipsatize(
            sample_data, method="column_wise", participant_col="participant"
        )
        check = result[PAQ_IDS].assign(participant=sample_data["participant"].values)
        group_means = check.groupby("participant")[PAQ_IDS].mean()
        np.testing.assert_allclose(group_means.to_numpy(), 0.0, atol=1e-10)

    def test_row_wise_zero_per_observation(self, sample_data):
        """Row-wise centering must produce zero mean across scales per row."""
        from soundscapy.surveys import ipsatize
        from soundscapy.surveys.survey_utils import PAQ_IDS

        result = ipsatize(sample_data, method="row_wise")
        row_means = result[PAQ_IDS].mean(axis=1)
        np.testing.assert_allclose(row_means.to_numpy(), 0.0, atol=1e-10)

    def test_returns_only_scale_columns(self, sample_data):
        """Ipsatize must return only the PAQ scale columns, not participant."""
        from soundscapy.surveys import ipsatize
        from soundscapy.surveys.survey_utils import PAQ_IDS

        result = ipsatize(
            sample_data, method="grand_mean", participant_col="participant"
        )
        assert set(result.columns) == set(PAQ_IDS)
        assert "participant" not in result.columns

    def test_invalid_method_raises(self, sample_data):
        """Ipsatize must raise ValueError for an unknown method string."""
        from soundscapy.surveys import ipsatize

        with pytest.raises(ValueError, match="method"):
            ipsatize(
                sample_data,
                method="bad_method",  # ty:ignore[invalid-argument-type]
            )

    def test_grand_mean_differs_from_column_wise(self, sample_data):
        """Grand-mean and column-wise results must differ for asymmetric data."""
        from soundscapy.surveys import ipsatize

        # Make data asymmetric: A has different values across scales
        asym = sample_data.copy()
        asym.loc[asym["participant"] == "A", "PAQ1"] = 80.0

        gm = ipsatize(asym, method="grand_mean", participant_col="participant")
        cw = ipsatize(asym, method="column_wise", participant_col="participant")
        # Results should differ
        assert not gm.equals(cw)


if __name__ == "__main__":
    pytest.main()
