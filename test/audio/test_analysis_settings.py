from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
import yaml

from soundscapy.audio.analysis_settings import (
    AnalysisSettings,
    MetricSettings,
    get_default_yaml,
)


@pytest.fixture
def sample_config():
    return {
        "version": "1.0",
        "PythonAcoustics": {
            "LAeq": {
                "run": True,
                "main": "avg",
                "statistics": ["5", "95", "avg", "max", "min"],
                "channel": ["Left", "Right"],
                "label": "LAeq",
                "func_args": {"time": 0.125, "method": "average"},
            }
        },
        "MoSQITo": {
            "loudness_zwtv": {
                "run": True,
                "main": 5,
                "statistics": ["10", "50", "90", "95", "min", "max", "avg"],
                "channel": ["Left", "Right"],
                "label": "N",
                "parallel": True,
                "func_args": {"field_type": "free"},
            }
        },
        "scikit-maad": {
            "all_temporal_alpha_indices": {"run": True, "channel": ["Left", "Right"]}
        },
    }


@pytest.fixture
def analysis_settings(sample_config):
    with TemporaryDirectory() as tempdir:
        config_path = Path(tempdir) / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(sample_config, f)
        return AnalysisSettings(config_path)


def test_analysis_settings_initialization(analysis_settings, sample_config):
    assert isinstance(analysis_settings, AnalysisSettings)
    assert (
        analysis_settings.settings["PythonAcoustics"]["LAeq"].run
        == sample_config["PythonAcoustics"]["LAeq"]["run"]
    )
    assert (
        analysis_settings.settings["MoSQITo"]["loudness_zwtv"].main
        == sample_config["MoSQITo"]["loudness_zwtv"]["main"]
    )


def test_get_metric_settings(analysis_settings):
    laeq_settings = analysis_settings.get_metric_settings("PythonAcoustics", "LAeq")
    assert isinstance(laeq_settings, MetricSettings)
    assert laeq_settings.run == True
    assert laeq_settings.main == "avg"


def test_parse_pyacoustics(analysis_settings):
    laeq_settings = analysis_settings.parse_pyacoustics("LAeq")
    assert isinstance(laeq_settings, MetricSettings)
    assert laeq_settings.label == "LAeq"


def test_parse_mosqito(analysis_settings):
    loudness_settings = analysis_settings.parse_mosqito("loudness_zwtv")
    assert isinstance(loudness_settings, MetricSettings)
    assert loudness_settings.parallel == True


def test_parse_maad_all_alpha_indices(analysis_settings):
    maad_settings = analysis_settings.parse_maad_all_alpha_indices(
        "all_temporal_alpha_indices"
    )
    assert isinstance(maad_settings, MetricSettings)
    assert maad_settings.run == True


def test_from_yaml(sample_config):
    with TemporaryDirectory() as tempdir:
        config_path = Path(tempdir) / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(sample_config, f)
        settings = AnalysisSettings.from_yaml(config_path)
    assert isinstance(settings, AnalysisSettings)
    assert (
        settings.settings["PythonAcoustics"]["LAeq"].run
        == sample_config["PythonAcoustics"]["LAeq"]["run"]
    )


def test_to_yaml(analysis_settings):
    with TemporaryDirectory() as tempdir:
        output_path = Path(tempdir) / "output_config.yaml"
        analysis_settings.to_yaml(output_path)
        assert output_path.exists()
        with open(output_path, "r") as f:
            loaded_config = yaml.safe_load(f)
    assert (
        loaded_config["PythonAcoustics"]["LAeq"]["run"]
        == analysis_settings.settings["PythonAcoustics"]["LAeq"].run
    )


def test_default():
    settings = AnalysisSettings.default()
    assert isinstance(settings, AnalysisSettings)
    assert "PythonAcoustics" in settings.settings
    assert "MoSQITo" in settings.settings
    assert "scikit-maad" in settings.settings


def test_get_default_yaml():
    with TemporaryDirectory() as tempdir:
        output_path = Path(tempdir) / "default_settings.yaml"
        get_default_yaml(str(output_path))
        assert output_path.exists()


def test_invalid_configuration():
    invalid_config = {
        "version": "1.0",
        "InvalidLibrary": {"InvalidMetric": {"run": "NotABoolean"}},
    }
    with TemporaryDirectory() as tempdir:
        config_path = Path(tempdir) / "invalid_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(invalid_config, f)
        with pytest.raises(ValueError):
            AnalysisSettings(config_path)


def test_raises_file_not_found_error():
    with pytest.raises(FileNotFoundError, match="Configuration file not found:"):
        AnalysisSettings("non_existent_file.yaml")


def test_raises_value_error_for_invalid_config():
    invalid_config = {"invalid_key": "invalid_value"}
    with pytest.raises(ValueError, match="Invalid configuration:"):
        AnalysisSettings(invalid_config)


def test_loads_valid_config_from_dict():
    valid_config = {
        "version": "1.0",
        "PythonAcoustics": {
            "LAeq": {
                "run": True,
                "main": "avg",
                "statistics": ["5", "95", "avg"],
                "channel": ["Left", "Right"],
                "label": "LAeq",
            }
        },
    }
    settings = AnalysisSettings(valid_config)
    assert settings.settings["PythonAcoustics"]["LAeq"].run is True


def test_loads_valid_config_from_yaml(tmp_path):
    config_content = """
    version: "1.0"
    PythonAcoustics:
      LAeq:
        run: true
        main: avg
        statistics: ["5", "95", "avg"]
        channel: ["Left", "Right"]
        label: LAeq
    """
    config_file = tmp_path / "valid_config.yaml"
    config_file.write_text(config_content)
    settings = AnalysisSettings(config_file)
    assert settings.settings["PythonAcoustics"]["LAeq"].run is True


def test_raises_key_error_for_missing_metric():
    config = {
        "version": "1.0",
        "PythonAcoustics": {
            "LAeq": {
                "run": True,
                "main": "avg",
                "statistics": ["5", "95", "avg"],
                "channel": ["Left", "Right"],
                "label": "LAeq",
            }
        },
    }
    settings = AnalysisSettings(config)
    with pytest.raises(
        KeyError,
        match="Metric 'NonExistentMetric' not found in library 'PythonAcoustics'",
    ):
        settings.get_metric_settings("PythonAcoustics", "NonExistentMetric")


def test_raises_key_error_for_missing_library():
    config = {
        "version": "1.0",
        "PythonAcoustics": {
            "LAeq": {
                "run": True,
                "main": "avg",
                "statistics": ["5", "95", "avg"],
                "channel": ["Left", "Right"],
                "label": "LAeq",
            }
        },
    }
    settings = AnalysisSettings(config)
    with pytest.raises(
        KeyError, match="Metric 'LAeq' not found in library 'NonExistentLibrary'"
    ):
        settings.get_metric_settings("NonExistentLibrary", "LAeq")


def test_does_not_run_loudness_zwtv_if_sharpness_din_from_loudness_is_present():
    config = {
        "version": "1.0",
        "MoSQITo": {
            "loudness_zwtv": {
                "run": True,
                "main": "avg",
                "statistics": ["avg"],
                "channel": ["Left", "Right"],
                "label": "loudness_zwtv",
                "parallel": False,
            },
            "sharpness_din_from_loudness": {
                "run": True,
                "main": "avg",
                "statistics": ["avg"],
                "channel": ["Left", "Right"],
                "label": "sharpness_din_from_loudness",
                "parallel": False,
            },
        },
    }
    settings = AnalysisSettings(config)
    metric_settings = settings.parse_mosqito("loudness_zwtv")
    assert metric_settings.run is False


def test_runs_loudness_zwtv_if_sharpness_din_from_loudness_is_not_present():
    config = {
        "version": "1.0",
        "MoSQITo": {
            "loudness_zwtv": {
                "run": True,
                "main": "avg",
                "statistics": ["avg"],
                "channel": ["Left", "Right"],
                "label": "loudness_zwtv",
                "parallel": False,
            },
        },
    }
    settings = AnalysisSettings(config)
    metric_settings = settings.parse_mosqito("loudness_zwtv")
    assert metric_settings.run is True


def test_runs_loudness_zwtv_if_force_run_all_is_true():
    config = {
        "version": "1.0",
        "MoSQITo": {
            "loudness_zwtv": {
                "run": True,
                "main": "avg",
                "statistics": ["avg"],
                "channel": ["Left", "Right"],
                "label": "loudness_zwtv",
                "parallel": False,
            },
            "sharpness_din_from_loudness": {
                "run": True,
                "main": "avg",
                "statistics": ["avg"],
                "channel": ["Left", "Right"],
                "label": "sharpness_din_from_loudness",
                "parallel": False,
            },
        },
    }
    settings = AnalysisSettings(config, force_run_all=True)
    metric_settings = settings.parse_mosqito("loudness_zwtv")
    assert metric_settings.run is True


def test_raises_value_error_for_invalid_maad_metric():
    config = {
        "version": "1.0",
        "scikit-maad": {
            "invalid_metric": {
                "run": True,
                "channel": ["Left", "Right"],
            }
        },
    }
    settings = AnalysisSettings(config)
    with pytest.raises(ValueError, match="Invalid MAAD metric: invalid_metric"):
        settings.parse_maad_all_alpha_indices("invalid_metric")


def test_parses_valid_maad_metric_all_temporal_alpha_indices():
    config = {
        "version": "1.0",
        "scikit-maad": {
            "all_temporal_alpha_indices": {
                "run": True,
                "channel": ["Left", "Right"],
            }
        },
    }
    settings = AnalysisSettings(config)
    metric_settings = settings.parse_maad_all_alpha_indices(
        "all_temporal_alpha_indices"
    )
    assert metric_settings.run is True
    assert metric_settings.channel == ["Left", "Right"]


def test_parses_valid_maad_metric_all_spectral_alpha_indices():
    config = {
        "version": "1.0",
        "scikit-maad": {
            "all_spectral_alpha_indices": {
                "run": True,
                "channel": ["Left", "Right"],
            }
        },
    }
    settings = AnalysisSettings(config)
    metric_settings = settings.parse_maad_all_alpha_indices(
        "all_spectral_alpha_indices"
    )
    assert metric_settings.run is True
    assert metric_settings.channel == ["Left", "Right"]


def test_creates_analysis_settings_from_valid_dict():
    config = {
        "version": "1.0",
        "PythonAcoustics": {
            "LAeq": {
                "run": True,
                "main": "avg",
                "statistics": ["5", "95", "avg"],
                "channel": ["Left", "Right"],
                "label": "LAeq",
            }
        },
    }
    settings = AnalysisSettings.from_dict(config)
    assert settings.settings["PythonAcoustics"]["LAeq"].run is True


def test_raises_value_error_for_invalid_dict():
    invalid_config = {"invalid_key": "invalid_value"}
    with pytest.raises(ValueError, match="Invalid configuration:"):
        AnalysisSettings.from_dict(invalid_config)


def test_creates_analysis_settings_with_run_stats_false():
    config = {
        "version": "1.0",
        "PythonAcoustics": {
            "LAeq": {
                "run": True,
                "main": "avg",
                "statistics": ["5", "95", "avg"],
                "channel": ["Left", "Right"],
                "label": "LAeq",
            }
        },
    }
    settings = AnalysisSettings.from_dict(config, run_stats=False)
    assert settings.run_stats is False


def test_creates_analysis_settings_with_force_run_all_true():
    config = {
        "version": "1.0",
        "PythonAcoustics": {
            "LAeq": {
                "run": False,
                "main": "avg",
                "statistics": ["5", "95", "avg"],
                "channel": ["Left", "Right"],
                "label": "LAeq",
            }
        },
    }
    settings = AnalysisSettings.from_dict(config, force_run_all=True)
    assert settings.force_run_all is True


if __name__ == "__main__":
    pytest.main()
