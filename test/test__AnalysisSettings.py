import pytest
from soundscapy import AnalysisSettings


def test_default():
    assert isinstance(AnalysisSettings.default(), dict)
    assert list(AnalysisSettings.default().keys()) == [
        "PythonAcoustics",
        "MoSQITo",
        "scikit-maad",
        "runtime",
    ]


def test_from_yaml():
    assert isinstance(AnalysisSettings.from_yaml("../soundscapy/analysis/default_settings.yaml"), dict)
    assert (
        AnalysisSettings.from_yaml("../soundscapy/analysis/default_settings.yaml")
        == AnalysisSettings.default()
    )


if __name__ == "__main__":
    pytest.main()