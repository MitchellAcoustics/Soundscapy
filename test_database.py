import ssidDatabase as db
from pathlib import Path
from pytest import raises

TEST_DIR = Path("test_DB")


def test_input_sanity():
    # Does the function raise appropriate input errors?
    with raises(TypeError) as exception:
        db.collect_param_dirs(["test"], ["LocationA"], ["LevelA"])
    with raises(FileNotFoundError) as exception:
        bad_path = Path("testy")
        db.collect_param_dirs(bad_path, ["LocationA"], ["LevelA"])


def test_spectrum_dirs_collection():
    bin_dirs = [
        TEST_DIR.joinpath(
            "OFF_LocationA1_FULL_2020-12-31", "OFF_LocationA1_BIN_2020-12-31"
        ),
        TEST_DIR.joinpath(
            "OFF_LocationA2_FULL_2021-01-01", "OFF_LocationA2_BIN_2021-01-01"
        ),
        TEST_DIR.joinpath(
            "OFF_LocationB1_FULL_2021-01-13", "OFF_LocationB1_BIN_2021-01-13"
        ),
    ]

    full_spectrum_list = db._spectrum_dirs(bin_dirs)
    assert len(full_spectrum_list) == 6


def test_ts_dirs_collection():
    bin_dirs = [
        TEST_DIR.joinpath(
            "OFF_LocationA1_FULL_2020-12-31", "OFF_LocationA1_BIN_2020-12-31"
        ),
        TEST_DIR.joinpath(
            "OFF_LocationA2_FULL_2021-01-01", "OFF_LocationA2_BIN_2021-01-01"
        ),
        TEST_DIR.joinpath(
            "OFF_LocationB1_FULL_2021-01-13", "OFF_LocationB1_BIN_2021-01-13"
        ),
    ]
    param_list = ["LevelA", "Loudness"]
    full_ts_list = db._ts_dirs(bin_dirs, param_list)
    assert len(full_ts_list) == 6

