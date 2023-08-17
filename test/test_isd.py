import pandas as pd
import pytest

import soundscapy.databases as db

name_test_df = pd.DataFrame(
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


@pytest.fixture()
def data():
    return db.isd.load()


def test_load():
    df = db.isd.load()
    assert df.shape == (1909, 78)


def test_validate(data):
    df, excl = db.isd.validate(data)
    assert len(excl) == 627
    assert "PAQ1" in df.columns and "PAQ5" in df.columns
    assert "pleasant" not in df.columns
    assert df["PAQ1"].isna().sum() == 0


def test__isd_select(data):
    assert len(db.isd._isd_select(data, "GroupID", "CT202")) == 2
    with pytest.raises(TypeError) as excinfo:
        obj = db.isd._isd_select(data, "GroupID", {22})
    assert "Should be either a str, int, list, or tuple." in str(excinfo.value)


def test_select_group_ids(data):
    assert len(db.isd.select_group_ids(data, "CT202")) == 2
    assert len(db.isd.select_group_ids(data, ["CT202", "PL105"])) == 3


def test_select_session_ids(data):
    assert len(db.isd.select_session_ids(data, "CamdenTown1")) == 37
    assert len(db.isd.select_session_ids(data, ["CamdenTown1", "CamdenTown2"])) == 45


def test_select_location_ids(data):
    assert len(db.isd.select_location_ids(data, "CamdenTown")) == 150
    assert len(db.isd.select_location_ids(data, ["CamdenTown", "PancrasLock"])) == 327


def test_select_record_ids(data):
    record = db.isd.select_record_ids(data, 525)
    assert len(record) == 1
    assert record["LocationID"][0] == "CamdenTown"


def test_remove_lockdown(data):
    assert len(db.isd.remove_lockdown(data)) == 1338


def test_describe_location(data):
    res = db.isd.describe_location(data, "CamdenTown")
    assert type(res) == dict
    assert res["count"] == 150
    assert res["pleasant"] == 0.247
    res = db.isd.describe_location(data, "CamdenTown", type="count")
    assert res["count"] == 150
    assert res["pleasant"] == 37


def test_soundscapy_describe(data):
    res = db.isd.soundscapy_describe(data)
    assert len(res) == len(data.LocationID.unique()) == 13


def test_accessor_validate_dataset_deprecated(data):
    with pytest.raises(DeprecationWarning) as excinfo:
        obj = data.isd.validate_dataset()
    assert (
        "The ISD accessor has been deprecated. Please use `soundscapy.isd.validate(data)` instead."
        in str(excinfo.value)
    )


def test_accessor_filter_group_ids(data):
    with pytest.raises(DeprecationWarning) as excinfo:
        obj = data.isd.filter_group_ids("CT202")
    assert (
        "The ISD accessor has been deprecated. Please use `soundscapy.isd.select_group_ids()` instead."
        in str(excinfo.value)
    )


def test_accessor_filter_session_ids(data):
    with pytest.raises(DeprecationWarning) as excinfo:
        obj = data.isd.filter_session_ids("CamdenTown1")
    assert (
        "The ISD accessor has been deprecated. Please use `soundscapy.isd.select_session_ids()` instead."
        in str(excinfo.value)
    )


def test_accessor_filter_location_ids(data):
    with pytest.raises(DeprecationWarning) as excinfo:
        obj = data.isd.filter_location_ids("CamdenTown")
    assert (
        "The ISD accessor has been deprecated. Please use `soundscapy.isd.select_location_ids()` instead."
        in str(excinfo.value)
    )


def test_accessor_location_describe(data):
    with pytest.raises(DeprecationWarning) as excinfo:
        obj = data.isd.location_describe("CamdenTown")
    assert (
        "The ISD accessor has been deprecated. Please use `soundscapy.isd.describe_location()` instead."
        in str(excinfo.value)
    )
