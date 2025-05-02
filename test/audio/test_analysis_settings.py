import pytest
import yaml
from pydantic import ValidationError

from soundscapy.audio.analysis_settings import (
    AnalysisSettings,
    ConfigManager,
    LibrarySettings,
    MetricSettings,
)


@pytest.fixture
def sample_config():
    return {
        "version": "1.1",
        "AcousticToolbox": {
            "LAeq": {
                "run": True,
                "main": "avg",
                "statistics": [5, 10, 50, 90, 95, "min", "max", "kurt", "skew"],
                "channel": ["Left", "Right"],
                "label": "LAeq",
                "func_args": {"time": 0.125, "method": "average"},
            }
        },
        "MoSQITo": {
            "loudness_zwtv": {
                "run": True,
                "main": 5,
                "statistics": [10, 50, 90, 95, "min", "max", "kurt", "skew", "avg"],
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
def temp_config_file(tmp_path, sample_config):
    config_file = tmp_path / "test_config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(sample_config, f)
    return config_file


class TestMetricSettings:
    def test_valid_metric_settings(self):
        settings = MetricSettings(
            run=True,
            main="avg",
            statistics=[5, 10, 50, 90, 95, "min", "max", "kurt", "skew"],
            channel=["Left", "Right"],
            label="LAeq",
            func_args={"time": 0.125, "method": "average"},
        )
        assert settings.run is True
        assert settings.main == "avg"
        assert "Left" in settings.channel and "Right" in settings.channel

    def test_invalid_metric_settings(self):
        with pytest.raises(ValidationError):
            MetricSettings(run="not_a_boolean")


class TestLibrarySettings:
    def test_valid_library_settings(self):
        settings = LibrarySettings(
            root={"LAeq": MetricSettings(run=True, main="avg", label="LAeq")}
        )
        assert "LAeq" in settings.root
        assert settings.root["LAeq"].run is True

    def test_invalid_library_settings(self):
        with pytest.raises(AttributeError):
            LibrarySettings(root={"InvalidMetric": "Not a MetricSettings object"})

    def test_get_existing_metric_settings(self, sample_config):
        settings = AnalysisSettings(**sample_config)
        library_settings = settings.AcousticToolbox
        result = library_settings.get_metric_settings("LAeq")
        assert result == library_settings.root["LAeq"]

    def test_get_non_existing_metric_settings(self):
        library_settings = LibrarySettings(root={})
        with pytest.raises(KeyError):
            library_settings.get_metric_settings("metric2")


class TestAnalysisSettings:
    def test_from_yaml(self, temp_config_file):
        settings = AnalysisSettings.from_yaml(temp_config_file)
        assert settings.version == "1.1"
        assert "LAeq" in settings.AcousticToolbox.root
        assert "loudness_zwtv" in settings.MoSQITo.root
        assert "all_temporal_alpha_indices" in settings.scikit_maad.root

    def test_to_yaml(self, tmp_path, sample_config):
        settings = AnalysisSettings(**sample_config)
        output_file = tmp_path / "output_config.yaml"
        settings.to_yaml(output_file)
        assert output_file.exists()

        # Read back and verify
        with open(output_file) as f:
            loaded_config = yaml.safe_load(f)
        assert loaded_config["version"] == "1.1"
        assert "LAeq" in loaded_config["AcousticToolbox"]

    def test_get_metric_settings(self, sample_config):
        settings = AnalysisSettings(**sample_config)
        laeq_settings = settings.get_metric_settings("AcousticToolbox", "LAeq")
        assert isinstance(laeq_settings, MetricSettings)
        assert laeq_settings.run is True
        assert laeq_settings.main == "avg"

    def test_get_metric_settings_invalid(self, sample_config):
        settings = AnalysisSettings(**sample_config)
        with pytest.raises(AttributeError):
            settings.get_metric_settings("InvalidLibrary", "InvalidMetric")

    def test_get_enabled_metrics(self, sample_config):
        settings = AnalysisSettings(**sample_config)
        enabled = settings.get_enabled_metrics()
        assert "LAeq" in enabled["AcousticToolbox"]
        assert "loudness_zwtv" in enabled["MoSQITo"]
        assert "all_temporal_alpha_indices" in enabled["scikit_maad"]

    def test_update_existing_metric_setting(self, sample_config):
        settings = AnalysisSettings(**sample_config)
        settings.update_setting("AcousticToolbox", "LAeq", run=False)
        updated_metric = settings.get_metric_settings("AcousticToolbox", "LAeq")
        assert updated_metric.run is False

    def test_update_non_existing_metric_setting(self, sample_config):
        settings = AnalysisSettings(**sample_config)
        with pytest.raises(KeyError):
            settings.update_setting("AcousticToolbox", "metric2", run=False)

    def test_update_metric_with_invalid_setting(self, sample_config):
        settings = AnalysisSettings(**sample_config)
        with pytest.raises(KeyError):
            settings.update_setting("AcousticToolbox", "metric1", invalid_setting=True)


class TestConfigManager:
    @pytest.fixture
    def config_manager(self, temp_config_file):
        return ConfigManager(temp_config_file)

    def test_load_config(self, config_manager):
        config = config_manager.load_config()
        assert isinstance(config, AnalysisSettings)
        assert config.version == "1.1"

    def test_save_config(self, config_manager, tmp_path):
        config_manager.load_config()
        new_file = tmp_path / "new_config.yaml"
        config_manager.save_config(new_file)
        assert new_file.exists()

    def test_merge_configs(self, config_manager):
        config_manager.load_config()
        override = {"AcousticToolbox": {"LAeq": {"run": False}}}
        merged = config_manager.merge_configs(override)
        assert merged.AcousticToolbox.root["LAeq"].run is False

    def test_generate_minimal_config(self, config_manager):
        config_manager.load_config()
        minimal = config_manager.generate_minimal_config()
        assert "version" not in minimal  # Assuming version is the same as default
        assert "MoSQITo" in minimal

    def test_load_default_config(self):
        manager = ConfigManager()
        config = manager.load_config()
        assert isinstance(config, AnalysisSettings)
        # Add more assertions based on your default config structure


def test_end_to_end(temp_config_file, tmp_path):
    # Load configuration
    manager = ConfigManager(temp_config_file)
    config = manager.load_config()
    assert config.AcousticToolbox.root["LAeq"].run is True

    # Modify configuration
    override = {"AcousticToolbox": {"LAeq": {"run": False}}}
    merged_config = manager.merge_configs(override)
    assert merged_config.AcousticToolbox.root["LAeq"].run is False
    assert manager.current_config.AcousticToolbox.root["LAeq"].run is False

    # Save modified configuration
    new_file = tmp_path / "modified_config.yaml"
    manager.save_config(new_file)

    # Load the saved configuration and verify changes
    new_manager = ConfigManager(new_file)
    new_config = new_manager.load_config()
    assert new_config.AcousticToolbox.root["LAeq"].run is False


if __name__ == "__main__":
    pytest.main()
