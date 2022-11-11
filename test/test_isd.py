import pandas as pd
from pytest import approx
import pytest
import soundscapy.databases as db
from pytest import raises

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

if __name__ == "__main__":
    pytest.main()