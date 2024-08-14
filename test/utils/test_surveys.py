import pandas as pd
import pytest
from pytest import approx, raises

import soundscapy.databases.isd as isd
from soundscapy.utils import surveys
from soundscapy.utils.parameters import EQUAL_ANGLES, LANGUAGE_ANGLES, PAQ_IDS


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


@pytest.fixture
def isd_data():
    return isd.load()


class TestBasicFunctions:
    def test_return_paqs(self, basic_test_df):
        result = surveys.return_paqs(basic_test_df)
        assert list(result.columns) == ["RecordID", "GroupID"] + PAQ_IDS

        result = surveys.return_paqs(
            basic_test_df, incl_ids=False, other_cols=["GroupID"]
        )
        assert list(result.columns) == PAQ_IDS + ["GroupID"]

    def test_mean_responses(self, basic_test_df):
        result = surveys.mean_responses(basic_test_df, group="GroupID")
        assert isinstance(result, pd.DataFrame)
        assert all(col in result.columns for col in PAQ_IDS)
        assert len(result) == basic_test_df["GroupID"].nunique()
        assert result.loc["A", "PAQ1"] == approx(3, abs=0.05)
        assert result.loc["B", "PAQ5"] == approx(3, abs=0.05)

    def test_convert_column_to_index(self, basic_test_df):
        result = surveys.convert_column_to_index(basic_test_df, "RecordID", drop=True)
        assert result.index.name == "RecordID"
        assert "RecordID" not in result.columns

        result = surveys.convert_column_to_index(basic_test_df, "RecordID")
        assert result.index.name == "RecordID"
        assert "RecordID" in result.columns

    def test_rename_paqs(self, name_test_df):
        result = surveys.rename_paqs(name_test_df)
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
        result = surveys.rename_paqs(df, paq_aliases=custom_aliases)
        assert list(result.columns) == ["RecordID"] + list(custom_aliases.values())

    def test_renames_paq_columns_with_dict_aliases(self, name_test_df):
        paq_aliases = {
            "pleasant": "PAQ1",
            "vibrant": "PAQ2",
            "eventful": "PAQ3",
            "chaotic": "PAQ4",
            "annoying": "PAQ5",
            "monotonous": "PAQ6",
            "uneventful": "PAQ7",
            "calm": "PAQ8",
        }
        renamed_df = surveys.rename_paqs(name_test_df, paq_aliases)
        expected_columns = PAQ_IDS
        assert list(renamed_df.columns) == ["RecordID"] + expected_columns

    def test_returns_same_dataframe_if_paqs_already_named(self):
        df = pd.DataFrame(
            {
                "PAQ1": [1, 2, 3],
                "PAQ2": [4, 5, 6],
                "PAQ3": [7, 8, 9],
                "PAQ4": [10, 11, 12],
                "PAQ5": [13, 14, 15],
                "PAQ6": [16, 17, 18],
                "PAQ7": [19, 20, 21],
                "PAQ8": [22, 23, 24],
            }
        )
        renamed_df = surveys.rename_paqs(df)
        assert renamed_df.equals(df)

    def test_returns_same_dataframe_if_no_paq_columns(self, basic_test_df):
        renamed_df = surveys.rename_paqs(basic_test_df, verbose=1)
        assert renamed_df.equals(basic_test_df)


class TestISOCoordinates:
    def test_calculate_iso_coords(self, basic_test_df):
        coords = surveys.calculate_iso_coords(basic_test_df)
        assert isinstance(coords, tuple)
        assert len(coords) == 2
        assert len(coords[0]) == len(basic_test_df)
        assert len(coords[1]) == len(basic_test_df)
        assert coords[0][0] == approx(0.53, abs=0.05)
        assert coords[1][0] == approx(0.03, abs=0.05)

    def test_add_iso_coords(self, basic_test_df):
        df = surveys.add_iso_coords(basic_test_df)
        assert "ISOPleasant" in df.columns and "ISOEventful" in df.columns

        with raises(Warning):
            df = surveys.add_iso_coords(df, overwrite=False)

        df = surveys.add_iso_coords(df, names=("pl_test", "ev_test"), overwrite=True)
        assert "pl_test" in df.columns and "ev_test" in df.columns

    def test_adj_iso_pl(self, basic_test_df):
        assert surveys.adj_iso_pl(
            basic_test_df.iloc[0, 2:], EQUAL_ANGLES, scale=4
        ) == approx(0.53, abs=0.01)
        assert surveys.adj_iso_pl(
            basic_test_df.iloc[0, 2:], LANGUAGE_ANGLES["eng"], scale=4
        ) == approx(0.66, abs=0.01)
        assert surveys.adj_iso_pl(
            basic_test_df.iloc[0, 2:], LANGUAGE_ANGLES["cmn"], scale=4
        ) == approx(0.41, abs=0.01)

    def test_adj_iso_ev(self, basic_test_df):
        assert surveys.adj_iso_ev(
            basic_test_df.iloc[0, 2:], EQUAL_ANGLES, scale=4
        ) == approx(0.03, abs=0.01)
        assert surveys.adj_iso_ev(
            basic_test_df.iloc[0, 2:], LANGUAGE_ANGLES["eng"], scale=4
        ) == approx(0.14, abs=0.01)
        assert surveys.adj_iso_ev(
            basic_test_df.iloc[0, 2:], LANGUAGE_ANGLES["cmn"], scale=4
        ) == approx(-0.09, abs=0.01)


class TestDataQuality:
    def test_likert_data_quality(self, basic_test_df):
        result = surveys.likert_data_quality(basic_test_df)
        assert result is None  # Assuming all data in basic_test_df is valid

        # Test with invalid data
        invalid_df = basic_test_df.copy()
        invalid_df.loc[0, PAQ_IDS] = 6  # Invalid value
        result = surveys.likert_data_quality(invalid_df)
        assert result == [0]


class TestSimulation:
    def test_simulation(self):
        df = surveys.simulation(n=200)
        assert df.shape == (200, 8)

        df = surveys.simulation(n=200, add_iso_coords=True)
        assert df.shape == (200, 10)
        assert list(df.columns) == PAQ_IDS + ["ISOPleasant", "ISOEventful"]


if __name__ == "__main__":
    pytest.main()
