import pandas as pd
import pytest
from pytest import approx

import soundscapy as sspy
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
    data = db.isd.load()
    data = sspy.surveys.add_iso_coords(data)
    return data


@pytest.fixture()
def old_data():
    return db.isd.load_zenodo(version="v0.2.3")


def test_load():
    df = db.isd.load()
    assert df.shape == (3589, 142)


def test_validate(data):
    df, excl = db.isd.validate(data)
    assert len(excl) == 109
    assert "PAQ1" in df.columns and "PAQ5" in df.columns
    assert "pleasant" not in df.columns
    assert df["PAQ1"].isna().sum() == 0


def test__isd_select(data):
    assert len(db.isd._isd_select(data, "GroupID", "CT202")) == 2


def test_select_group_ids(data):
    assert len(db.isd.select_group_ids(data, "CT202")) == 2
    assert len(db.isd.select_group_ids(data, ["CT202", "PL105"])) == 3


def test_select_session_ids(data):
    assert len(db.isd.select_session_ids(data, "CamdenTown1")) == 37
    assert len(db.isd.select_session_ids(data, ["CamdenTown1", "CamdenTown2"])) == 45


def test_select_location_ids(data):
    assert len(db.isd.select_location_ids(data, "CamdenTown")) == 105
    assert len(db.isd.select_location_ids(data, ["CamdenTown", "PancrasLock"])) == 200


def test_remove_lockdown(old_data):
    assert len(db.isd.remove_lockdown(old_data)) == 1338


def test_describe_location(data):
    sim = sspy.surveys.simulation(1000)
    sim = sspy.surveys.add_iso_coords(sim)
    sim["LocationID"] = "Simulated"
    res = db.isd.describe_location(sim, "Simulated")
    assert type(res) == dict
    assert res["count"] == 1000
    assert res["pleasant"] == approx(0.5, abs=0.05)

    res = db.isd.describe_location(data, "CamdenTown")
    assert type(res) == dict
    assert res["count"] == 105
    assert res["pleasant"] == 0.352

    res = db.isd.describe_location(data, "CamdenTown", type="count")
    assert res["count"] == 105
    assert res["pleasant"] == 37
