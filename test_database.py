import ssidDatabase as db
from pathlib import Path
from pytest import raises
import random

TEST_DIR = Path("test_DB")


def test_input_sanity():
    # Does the function raise appropriate input errors?
    with raises(TypeError) as exception:
        db.collect_all_dirs(["test"], ["LocationA"], ["LevelA"])
    with raises(FileNotFoundError) as exception:
        bad_path = Path("testy")
        db.collect_all_dirs(bad_path, ["LocationA"], ["LevelA"])


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


def test_wav_dirs_collection():
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
    full_wav_list = db._wav_dirs(bin_dirs)
    assert len(full_wav_list) == 3


# Integrated test
def test_all_dirs_collection():
    locations = ["LocationA1", "LocationA2", "LocationB1"]
    sampled_locations = random.sample(locations, 2)
    sampled_params = random.sample(db.PARAM_LIST, 4)

    ts_list, spectrum_list, wav_list = db.collect_all_dirs(
        TEST_DIR, sampled_locations, sampled_params
    )
    # ts_list should have a param dir for each location dir
    assert len(ts_list) == 2 * 4
    # spectrum_list should have 2 dirs for each location dir
    assert len(spectrum_list) == 2 * 2
    # wav_list should have 1 dir for each location dir
    assert len(wav_list) == 2
