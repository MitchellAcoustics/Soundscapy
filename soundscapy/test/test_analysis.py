import os
import sys

myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + "/../../")

from pathlib import Path

from soundscapy.analysis import *
from pytest import raises


def test_spectrogram_2ch_input_sanity():
    filepath = Path(
        "test/test_DB/OFF_LocationA1_FULL_2020-12-31/OFF_LocationA1_BIN_2020-12-31/LocationA1_WAV/LA101.wav"
    )
    # Does the function raise appropriate input errors?
    with raises(ValueError) as exception:
        spectrogram_2ch(filepath, method="wrong")

    filepath = Path(
        "test/test_DB/OFF_LocationA1_FULL_2020-12-31/OFF_LocationA1_BIN_2020-12-31/LocationA1_WAV/NOT_A_FILE.wav"
    )
    with raises(FileNotFoundError) as exception:
        spectrogram_2ch(filepath)


def test_loudness_from_wav_input_sanity():
    filepath = Path("testfile/mctestfile.wav")
    with raises(ValueError) as exception:
        loudness_from_wav(filepath, method="stationary_levels")
        loudness_from_wav(filepath, method="unknown_method")
    with raises(FileNotFoundError) as exception:
        loudness_from_wav(filepath, method="time_varying")
    with raises(TypeError) as expection:
        loudness_from_wav(20)


def test_loudness_from_wav():
    iso_exe = "./soundscapy/bin/ISO_532-1"
    filepath = Path("soundscapy\\test\\iso_532-1\\sine 1kHz 40dB 16bit.wav")
    ref_file = Path(
        "soundscapy\\test\\iso_532-1\\calibration signal sine 1kHz 60dB.wav"
    )
    results = loudness_from_wav(
        filepath,
        ref_file,
        ref_level=60,
        method="stationary",
        relocate_iso_exe=iso_exe,
    )
    assert results["loudness_sone"] == 1.0
    assert results["loudness_lvl_phon"] == 40.0

    time_result = loudness_from_wav(
        filepath,
        ref_file,
        ref_level=60,
        method="time_varying",
        relocate_iso_exe=iso_exe,
    )
    assert time_result["loudness_Nmax"] == 1.0
    assert time_result["loudness_N5"] == 1.0
