import numpy as np
import pandas as pd
import pytest
from pytest import raises
from soundscapy.utils import surveys

import soundscapy.databases.isd as isd


@pytest.fixture
def isd_data():
    return isd.load()


@pytest.fixture
def isd_data_with_iso():
    data = isd.load()
    return surveys.add_iso_coords(data)


class TestISDLoading:
    def test_load(self):
        df = isd.load()
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (3589, 142)  # Adjust these numbers if they change
        assert all(col in df.columns for col in isd._PAQ_ALIASES.values())

    @pytest.mark.slow
    def test_load_zenodo(self):
        with raises(ValueError, match="Version .* not recognised"):
            isd.load_zenodo(version="invalid_version")

        df_latest = isd.load_zenodo()
        assert isinstance(df_latest, pd.DataFrame)

        df_v1 = isd.load_zenodo(version="v1.0.1")
        assert df_v1.shape == df_latest.shape

        df_old = isd.load_zenodo(version="v0.2.3")
        assert df_old.shape != df_latest.shape


class TestISDValidation:
    def test_validate(self, isd_data):
        df_clean, df_excl = isd.validate(isd_data)
        assert isinstance(df_clean, pd.DataFrame)
        assert isinstance(df_excl, pd.DataFrame) or df_excl is None
        assert len(df_clean) + len(df_excl) == len(isd_data)
        assert all(col in df_clean.columns for col in isd._PAQ_ALIASES.values())

        df_clean, df_excl = isd.validate(df_clean)
        assert df_excl is None

    def test_validate_allow_na(self, isd_data):
        df_clean, df_excl = isd.validate(isd_data, allow_paq_na=True)
        assert df_clean.isna().sum().sum() > 0


class TestISDSelection:
    def test__isd_select(self, isd_data):
        result = isd._isd_select(isd_data, "LocationID", "CamdenTown")
        assert len(result) > 0
        assert all(result["LocationID"] == "CamdenTown")

        result = isd._isd_select(isd_data, "GroupID", ["CT101", "CT102"])
        assert len(result) > 0
        assert all(result["GroupID"].isin(["CT101", "CT102"]))

        with raises(TypeError, match="Should be either a str, int, list, or tuple"):
            isd._isd_select(isd_data, "LocationID", {"invalid": "type"})

    def test_select_record_ids(self, isd_data):
        record_ids = isd_data["RecordID"].sample(5).tolist()
        result = isd.select_record_ids(isd_data, record_ids)
        # assert len(result) == 5 # TODO: Issue with duplicate RecordIDs in the ISD data
        assert all(result["RecordID"].isin(record_ids))

    def test_select_group_ids(self, isd_data):
        group_ids = isd_data["GroupID"].unique()[:3].tolist()
        result = isd.select_group_ids(isd_data, group_ids)
        assert len(result) > 0
        assert all(result["GroupID"].isin(group_ids))

    def test_select_session_ids(self, isd_data):
        session_ids = isd_data["SessionID"].unique()[:2].tolist()
        result = isd.select_session_ids(isd_data, session_ids)
        assert len(result) > 0
        assert all(result["SessionID"].isin(session_ids))

    def test_select_location_ids(self, isd_data):
        location_ids = isd_data["LocationID"].unique()[:2].tolist()
        result = isd.select_location_ids(isd_data, location_ids)
        assert len(result) > 0
        assert all(result["LocationID"].isin(location_ids))


class TestISDDescription:
    def test_describe_location(self, isd_data_with_iso):
        location = isd_data_with_iso["LocationID"].unique()[0]
        result = isd.describe_location(isd_data_with_iso, location)
        assert isinstance(result, dict)
        assert "count" in result
        assert "ISOPleasant" in result
        assert "ISOEventful" in result
        assert "pleasant" in result
        assert "eventful" in result

        result_count = isd.describe_location(isd_data_with_iso, location, type="count")
        assert isinstance(result_count["pleasant"], int)

    @pytest.mark.slow
    def test_soundscapy_describe(self, isd_data_with_iso):
        result = isd.soundscapy_describe(isd_data_with_iso)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == isd_data_with_iso["LocationID"].nunique()
        assert "count" in result.columns
        assert "ISOPleasant" in result.columns

        result_count = isd.soundscapy_describe(isd_data_with_iso, type="count")
        assert isinstance(
            result_count.loc[result_count.index[0], "pleasant"], (int, np.integer)
        )


if __name__ == "__main__":
    pytest.main()
