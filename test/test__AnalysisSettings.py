import pytest
import soundscapy
from soundscapy import AnalysisSettings
from pathlib import Path

def test_default():
    assert isinstance(AnalysisSettings.default(), dict)
    assert list(AnalysisSettings.default().keys()) == [
        "PythonAcoustics",
        "MoSQITo",
        "scikit-maad",
        "runtime",
    ]


def test_from_yaml():
    root = Path(soundscapy.__path__[0])

    assert isinstance(AnalysisSettings.from_yaml(Path(root, "analysis", "default_settings.yaml")), dict)
    assert (
        AnalysisSettings.from_yaml(Path(root, "analysis", "default_settings.yaml"))
        == AnalysisSettings.default()
    )


if __name__ == "__main__":
    pytest.main()