import sys, os

myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + "/../../")

from pathlib import Path
from soundscapy.database import collect_all_dirs
from soundscapy.parameters import (
    PARAM_LIST,
    LOCATION_IDS,
    IGNORE_LIST,
    CATEGORISED_VARS,
)

DATA_DIR = Path("R:\\UCL_SSID\\SSID_DATABASE\\SSID_London")
ts_dirs, spectrum_dirs, wav_dirs = collect_all_dirs(
    DATA_DIR,
    LOCATION_IDS["London"],
    PARAM_LIST,
    include_TS=False,
    include_spectrum=False,
    include_WAV=True,
)
wav_dirs
