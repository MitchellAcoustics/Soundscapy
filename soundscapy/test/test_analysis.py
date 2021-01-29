import os
import sys

myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + "/../../")

from pathlib import Path

import soundscapy.analysis as an
from pytest import raises


def test_input_sanity():
    filepath = Path(
        "test/test_DB/OFF_LocationA1_FULL_2020-12-31/OFF_LocationA1_BIN_2020-12-31/LocationA1_WAV/LA101.wav"
    )
    # Does the function raise appropriate input errors?
    with raises(ValueError) as exception:
        an.spectrogram_2ch(filepath, method="wrong")

    filepath = Path(
        "test/test_DB/OFF_LocationA1_FULL_2020-12-31/OFF_LocationA1_BIN_2020-12-31/LocationA1_WAV/NOT_A_FILE.wav"
    )
    with raises(FileNotFoundError) as exception:
        an.spectrogram_2ch(filepath)

