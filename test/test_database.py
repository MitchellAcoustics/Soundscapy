import pandas as pd
from pytest import approx
import pytest
import soundscapy.database as db
from pytest import raises


def test_load_isd_dataset_wrong_version():
    with raises(ValueError):
        df = db.load_isd_dataset(version="v0.1.0")


def test_load_isd_dataset():
    df = db.load_isd_dataset(version="v0.2.1")
    assert df.shape == (1909, 77)


def test_calculate_paq_coords():
    vals = {
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
    df = pd.DataFrame(vals)
    coords = db.calculate_paq_coords(df)
    assert coords[0][0] == approx(0.53, abs=0.05) and coords[0][1] == approx(
        -0.75, abs=0.05
    )  # ISOPleasant coords
    assert coords[1][0] == approx(0.03, abs=0.05) and coords[1][1] == approx(
        0.35, abs=0.05
    )  # ISOEventful coords


def test_calculate_paq_coords_val_range():
    vals = {
        "RecordID": ["EX1", "EX2"],
        "pleasant": [40, 20],
        "vibrant": [40, 30],
        "eventful": [40, 50],
        "chaotic": [20, 50],
        "annoying": [10, 50],
        "monotonous": [30, 50],
        "uneventful": [30, 30],
        "calm": [40, 10],
    }
    df = pd.DataFrame(vals)
    coords = db.calculate_paq_coords(df, val_range=(0, 100))
    assert coords[0][0] == approx(0.21, abs=0.05) and coords[0][1] == approx(
        -0.30, abs=0.05
    )  # ISOPleasant coords
    assert coords[1][0] == approx(0.01, abs=0.05) and coords[1][1] == approx(
        0.14, abs=0.05
    )  # ISOEventful coords


def test_calculate_paq_coords_min_max():
    # first is max pleasant, second is max eventful, third is min pleasant, fourth is min eventful
    vals = {
        "RecordID": ["maxpl", "maxev", "minpl", "minev"],
        "pleasant": [5, 3, 1, 3],
        "vibrant": [5, 5, 1, 1],
        "eventful": [3, 5, 3, 1],
        "chaotic": [1, 5, 5, 1],
        "annoying": [1, 3, 5, 3],
        "monotonous": [1, 1, 5, 5],
        "uneventful": [3, 1, 3, 5],
        "calm": [5, 1, 1, 5],
    }
    df = pd.DataFrame(vals)
    ISOPl, ISOEv = db.calculate_paq_coords(df)
    assert ISOPl[0] == approx(1, abs=0.01) and ISOPl[2] == approx(
        -1, abs=0.01
    )  # ISOPleasant coords
    assert ISOEv[1] == approx(1, abs=0.01) and ISOEv[3] == approx(
        -1, abs=0.01
    )  # ISOEventful coords


def test_calculate_paq_coords_val_range_min_max():
    # first is max pleasant, second is max eventful, third is min pleasant, fourth is min eventful
    vals = {
        "RecordID": ["maxpl", "maxev", "minpl", "minev"],
        "pleasant": [50, 0, -50, 0],
        "vibrant": [50, 50, -50, -50],
        "eventful": [0, 50, 0, -50],
        "chaotic": [-50, 50, 50, -50],
        "annoying": [-50, 0, 50, 0],
        "monotonous": [-50, -50, 50, 50],
        "uneventful": [0, -50, 0, 50],
        "calm": [50, -50, -50, 50],
    }
    df = pd.DataFrame(vals)
    ISOPl, ISOEv = db.calculate_paq_coords(df, val_range=(-50, 50))
    assert ISOPl[0] == approx(1, abs=0.01) and ISOPl[2] == approx(
        -1, abs=0.01
    )  # ISOPleasant coords
    assert ISOEv[1] == approx(1, abs=0.01) and ISOEv[3] == approx(
        -1, abs=0.01
    )  # ISOEventful coords


def test_simulation():
    df = db.simulation(n=200)
    assert df.shape == (200, 8)

    df = db.simulation(n=200, add_paq_coords=True)
    assert df.shape == (200, 10)

    df = db.simulation(n=200, add_paq_coords=True)
    assert df.columns.tolist() == [
        "pleasant",
        "vibrant",
        "eventful",
        "chaotic",
        "annoying",
        "monotonous",
        "uneventful",
        "calm",
        "ISOPleasant",
        "ISOEventful",
    ]


if __name__ == "__main__":
    pytest.main()

# %%
