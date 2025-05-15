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
        assert list(result.columns) == ["RecordID", "GroupID"] + PAQ_IDS

        result = return_paqs(basic_test_df, incl_ids=False, other_cols=["GroupID"])
        assert list(result.columns) == PAQ_IDS + ["GroupID"]

    def test_rename_paqs(self, name_test_df):
        result = rename_paqs(name_test_df)
        assert list(result.columns) == ["RecordID"] + PAQ_IDS

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
        df = pd.DataFrame(
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
        result = rename_paqs(df, paq_aliases=custom_aliases)
        assert list(result.columns) == ["RecordID"] + list(custom_aliases.values())


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
        df = add_iso_coords(basic_test_df)
        assert "ISOPleasant" in df.columns and "ISOEventful" in df.columns

        with pytest.raises(Warning):
            df = add_iso_coords(df, overwrite=False)

        df = add_iso_coords(df, names=("pl_test", "ev_test"), overwrite=True)
        assert "pl_test" in df.columns and "ev_test" in df.columns

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
        df = simulation(n=200)
        assert df.shape == (200, 8)
        assert list(df.columns) == PAQ_IDS
        assert df.min().min() >= 1 and df.max().max() <= 5

    def test_simulation_with_iso_coords(self):
        df = simulation(n=200, incl_iso_coords=True)
        assert df.shape == (200, 10)
        assert list(df.columns) == PAQ_IDS + ["ISOPleasant", "ISOEventful"]
        assert not df["ISOPleasant"].isna().any() and not df["ISOEventful"].isna().any()


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
            assert (result["r_squared"] >= 0).all() and (result["r_squared"] <= 1).all()

    def test_ssm_metrics_polar(self, ssm_test_df):
        with pytest.warns(PendingDeprecationWarning):
            result = ssm_metrics(ssm_test_df, method="polar")
            assert isinstance(result, pd.DataFrame)
            assert list(result.columns) == [
                "amplitude",
                "angle",
                "elevation",
                "displacement",
                "r_squared",
            ]
            assert len(result) == len(ssm_test_df)
            assert (result["displacement"] == 0).all()
            assert (result["r_squared"] == 1).all()

    def test_ssm_metrics_invalid_method(self, ssm_test_df):
        with pytest.warns(PendingDeprecationWarning):
            with pytest.raises(ValueError):
                ssm_metrics(ssm_test_df, method="invalid")

    def test_ssm_metrics_missing_columns(self):
        with pytest.warns(PendingDeprecationWarning):
            invalid_df = pd.DataFrame({"PAQ1": [1, 2, 3], "PAQ2": [4, 5, 6]})
            with pytest.raises(ValueError):
                ssm_metrics(invalid_df)


class TestIntegration:
    def test_end_to_end_workflow(self):
        # Simulate data
        df = simulation(n=100, incl_iso_coords=True)

        # Check data quality
        invalid_indices = likert_data_quality(df)
        assert invalid_indices is None

        # Calculate SSM metrics
        ssm_result = ssm_metrics(df)

        # Verify results
        assert df.shape == (100, 10)
        assert ssm_result.shape == (100, 5)
        assert not df["ISOPleasant"].isna().any() and not df["ISOEventful"].isna().any()
        assert not ssm_result.isna().any().any()


if __name__ == "__main__":
    pytest.main()
