import sys, os

myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, f'{myPath}/../../')

import soundscapy.ssid.parameters as par
from pytest import raises


def test_surveyvar_dict():
    for key, val in par.SURVEY_VARS.items():
        if key in ["Traffic", "Other", "Human", "Natural"]:
            assert min(val["levels"].keys()) == 1
            assert max(val["levels"].keys()) == 5
        elif key in [
            "pleasant",
            "chaotic",
            "vibrant",
            "uneventful",
            "calm",
            "annoying",
            "eventful",
            "monotonous",
        ]:
            assert min(val["levels"].keys()) == 1
            assert max(val["levels"].keys()) == 5
        elif key in ["overall", "appropriateness", "loudness", "visit_again"]:
            assert min(val["levels"].keys()) == 1
            assert max(val["levels"].keys()) == 5
        elif key in ["who01", "who02", "who03", "who04", "who05"]:
            assert min(val["levels"].keys()) == 0
            assert max(val["levels"].keys()) == 5


if __name__ == "__main__":
    test_surveyvar_dict()
