import pytest
from tempfile import TemporaryDirectory, TemporaryFile
import soundscapy
from soundscapy import AnalysisSettings, get_default_yaml
from pathlib import Path
import yaml


@pytest.fixture()
def config():
    return {
        "library1": {
            "metric1": {"option1": "value1", "option2": "value2"},
            "metric2": {"option1": "value1", "option2": "value2"},
        },
        "library2": {
            "metric1": {"option1": "value1", "option2": "value2"},
            "metric2": {"option1": "value1", "option2": "value2"},
        },
    }


@pytest.fixture()
def temp_yaml_file(config):
    with open("test_settings.yaml", "w") as f:
        yaml.dump(config, f)
    yield "test_settings.yaml"
    Path("test_settings.yaml").unlink()


@pytest.fixture
def example_settings():
    data = {
        "mosqito": {
            "loudness_zwicker": {
                "frequency_range": [20, 20000],
                "time_range": [0, 3.5],
                "block_size": 4096,
            },
            "sharpness_din_from_loudness": {"frequency_range": [20, 20000]},
        }
    }
    return AnalysisSettings(
        data, run_stats=True, force_run_all=False, filepath="example.yaml"
    )


def test_from_yaml(temp_yaml_file, config):
    settings = AnalysisSettings.from_yaml(temp_yaml_file)
    assert (
        settings["library1"]["metric1"]["option1"]
        == config["library1"]["metric1"]["option1"]
    )
    assert (
        settings["library1"]["metric1"]["option2"]
        == config["library1"]["metric1"]["option2"]
    )
    assert (
        settings["library2"]["metric2"]["option1"]
        == config["library2"]["metric2"]["option1"]
    )
    assert (
        settings["library2"]["metric2"]["option2"]
        == config["library2"]["metric2"]["option2"]
    )

    root = Path(soundscapy.__path__[0])


def test_default():
    settings = AnalysisSettings.default()
    assert isinstance(settings, AnalysisSettings)
    assert "runtime" in settings
    assert isinstance(AnalysisSettings.default(), dict)
    assert list(AnalysisSettings.default().keys()) == [
        "PythonAcoustics",
        "MoSQITo",
        "scikit-maad",
        "runtime",
    ]


def test_get_default_yaml():
    get_default_yaml(save_as="test_default_settings.yaml")
    assert Path("test_default_settings.yaml").exists()
    Path("test_default_settings.yaml").unlink()


def test_reload(example_settings):
    with TemporaryDirectory() as tempdir:
        # Save example_settings to file
        filename = Path(tempdir, "example.yaml")
        example_settings.filepath = filename
        example_settings.to_yaml(filename)

        # Modify example_settings data
        example_settings["mosqito"]["loudness_zwicker"]["frequency_range"] = [20, 8000]

        # Reload from file
        reloaded_settings = example_settings.reload()

        # Check that the data has been reloaded from the file
        assert reloaded_settings["mosqito"]["loudness_zwicker"]["frequency_range"] == [
            20,
            20000,
        ]


def test_to_yaml(example_settings):
    with TemporaryDirectory() as tempdir:
        # Save example_settings to file
        filename = Path(tempdir, "example.yaml")
        example_settings.to_yaml(filename)
        assert filename.exists()

        # Load saved settings from file
        saved_data = example_settings.from_yaml(filename)

        # Check that the saved data matches the original data
        assert saved_data == example_settings


def test_parse_maad_all_alpha_indices():
    settings = AnalysisSettings.default()
    maad_settings = settings.parse_maad_all_alpha_indices("all_temporal_alpha_indices")
    assert len(maad_settings) == 2
    with pytest.raises(AssertionError) as excinfo:
        obj = settings.parse_maad_all_alpha_indices("missing_key")
    assert (
        "metric must be all_temporal_alpha_indices or all_spectral_alpha_indices."
        in str(excinfo.value)
    )


def test_parse_mosqito():
    settings = AnalysisSettings.default()
    mosqito_settings = settings.parse_mosqito("loudness_zwtv")
    assert len(mosqito_settings) == 6
    with pytest.raises(AssertionError) as excinfo:
        obj = settings.parse_mosqito("missing_key")
    assert "Metric missing_key not found." in str(excinfo.value)

    settings["MoSQITo"]["loudness_zwtv"] = {
        "run": False,
        "main": 5,
        "channel": ["Left", "Right"],
        "statistics": [0, 50, 90],
        "label": "N5",
    }
    run, channel, statistics, label, parallel, func_args = settings.parse_mosqito(
        "loudness_zwtv"
    )
    assert parallel == False


def test_parse_pyacoustics():
    settings = AnalysisSettings.default()
    pyacoustics_settings = settings.parse_pyacoustics("LAeq")
    assert len(pyacoustics_settings) == 5
