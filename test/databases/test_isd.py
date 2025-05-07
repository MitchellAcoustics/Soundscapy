import numpy as np
import pandas as pd
import pytest

from soundscapy.databases import isd
from soundscapy.surveys.processing import add_iso_coords


@pytest.fixture
def isd_data():
    """Fixture to load ISD data for tests."""
    return isd.load()


@pytest.fixture
def isd_data_with_iso(isd_data):
    """Fixture to load ISD data with ISO coordinates for tests."""
    return add_iso_coords(isd_data)


class TestISDLoading:
    def test_load(self):
        """Test loading of ISD data from local file."""
        df = isd.load()
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (3589, 142)  # Adjust these numbers if they change
        assert all(col in df.columns for col in isd._PAQ_ALIASES.values())

    @pytest.mark.slow
    def test_load_zenodo(self):
        """Test loading of ISD data from Zenodo."""
        with pytest.raises(ValueError, match="Version .* not recognised"):
            isd.load_zenodo(version="invalid_version")

        df_latest = isd.load_zenodo()
        assert isinstance(df_latest, pd.DataFrame)

        df_v1 = isd.load_zenodo(version="v1.0.1")
        assert df_v1.shape == df_latest.shape

        df_old = isd.load_zenodo(version="v0.2.3")
        assert df_old.shape != df_latest.shape


class TestISDValidation:
    def test_validate(self, isd_data):
        """Test validation of ISD data."""
        df_clean, df_excl = isd.validate(isd_data)
        assert isinstance(df_clean, pd.DataFrame)
        assert isinstance(df_excl, pd.DataFrame) or df_excl is None
        assert len(df_clean) + len(df_excl) == len(isd_data)
        assert all(col in df_clean.columns for col in isd._PAQ_ALIASES.values())

        df_clean, df_excl = isd.validate(df_clean)
        assert df_excl is None

    def test_validate_allow_na(self, isd_data):
        """Test validation of ISD data allowing NaN values."""
        df_clean, df_excl = isd.validate(isd_data, allow_paq_na=True)
        assert df_clean.isna().sum().sum() > 0


class TestISDSelection:
    def test__isd_select(self, isd_data):
        """Test internal selection function."""
        result = isd._isd_select(isd_data, "LocationID", "CamdenTown")
        assert len(result) > 0
        assert all(result["LocationID"] == "CamdenTown")

        result = isd._isd_select(isd_data, "GroupID", ["CT101", "CT102"])
        assert len(result) > 0
        assert all(result["GroupID"].isin(["CT101", "CT102"]))

        with pytest.raises(
            TypeError, match="Should be either a str, int, list, or tuple"
        ):
            isd._isd_select(isd_data, "LocationID", {"invalid": "type"})

    def test_select_record_ids(self, isd_data):
        """Test selection by RecordID."""
        record_ids = isd_data["RecordID"].sample(5).tolist()
        result = isd.select_record_ids(isd_data, record_ids)
        assert all(result["RecordID"].isin(record_ids))

    def test_select_group_ids(self, isd_data):
        """Test selection by GroupID."""
        group_ids = isd_data["GroupID"].unique()[:3].tolist()
        result = isd.select_group_ids(isd_data, group_ids)
        assert len(result) > 0
        assert all(result["GroupID"].isin(group_ids))

    def test_select_session_ids(self, isd_data):
        """Test selection by SessionID."""
        session_ids = isd_data["SessionID"].unique()[:2].tolist()
        result = isd.select_session_ids(isd_data, session_ids)
        assert len(result) > 0
        assert all(result["SessionID"].isin(session_ids))

    def test_select_location_ids(self, isd_data):
        """Test selection by LocationID."""
        location_ids = isd_data["LocationID"].unique()[:2].tolist()
        result = isd.select_location_ids(isd_data, location_ids)
        assert len(result) > 0
        assert all(result["LocationID"].isin(location_ids))


class TestISDDescription:
    def test_describe_location(self, isd_data_with_iso):
        """Test description of a specific location."""
        location = isd_data_with_iso["LocationID"].unique()[0]
        result = isd.describe_location(isd_data_with_iso, location)
        assert isinstance(result, dict)
        assert "count" in result
        assert "ISOPleasant" in result
        assert "ISOEventful" in result
        assert "pleasant" in result
        assert "eventful" in result

        result_count = isd.describe_location(
            isd_data_with_iso, location, calc_type="count"
        )
        assert isinstance(result_count["pleasant"], (int, np.integer))

    @pytest.mark.slow
    def test_soundscapy_describe(self, isd_data_with_iso):
        """Test overall description of ISD data."""
        result = isd.soundscapy_describe(isd_data_with_iso)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == isd_data_with_iso["LocationID"].nunique()
        assert "count" in result.columns
        assert "ISOPleasant" in result.columns

        result_count = isd.soundscapy_describe(isd_data_with_iso, calc_type="count")
        assert isinstance(
            result_count.loc[result_count.index[0], "pleasant"], (int, np.integer)
        )


if __name__ == "__main__":
    pytest.main()
