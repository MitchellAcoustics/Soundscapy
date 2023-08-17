from pathlib import Path
from time import localtime, strftime
from typing import Union
import yaml
import urllib.request


def get_default_yaml(save_as="default_settings.yaml"):
    """
    Retrieves the default settings for analysis from the GitHub repository
    and saves them to a file.

    Parameters
    ----------
    save_as : str, optional
        The name of the file to save the default settings to. Defaults to
        "default_settings.yaml".
    """
    print("Downloading default settings from GitHub...")
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/MitchellAcoustics/Soundscapy/main/soundscapy/analysis/default_settings.yaml",
        save_as,
    )


class AnalysisSettings(dict):
    """Dict of settings for analysis methods. Each library has a dict of metrics,
    each of which has a dict of settings.
    """

    def __init__(
        self,
        data,
        run_stats=True,
        force_run_all=False,
        filepath: Union[str, Path] = None,
    ):
        super().__init__(data)
        self.run_stats = run_stats
        self.force_run_all = force_run_all
        self.filepath = filepath
        runtime = strftime("%Y-%m-%d %H:%M:%S", localtime())
        super().__setitem__("runtime", runtime)

    @classmethod
    def from_yaml(cls, filename: Union[Path, str], run_stats=True, force_run_all=False):
        """Generate a settings object from a yaml file.

        Parameters
        ----------
        filename : Path object or str
            filename of the yaml file
        run_stats : bool, optional
            whether to include all stats listed or just return the main metric, by default True

            This can simplify the results dataframe if you only want the main metric.
            For example, rather than including L_5, L_50, etc. will only include LEq
        force_run_all : bool, optional
            whether to force all metrics to run regardless of what is set in their options, by default False

            Use Cautiously. This can be useful if you want to run all metrics, but don't want to change the yaml file.
            Warning: If both mosqito:loudness_zwtv and mosqitsharpness_din_from_loudness are present in the settings file, this will result in the loudness calc being run twice.
        Returns
        -------
        AnalysisSettings
            AnalysisSettings object
        """
        with open(filename, "r") as f:
            return cls(
                yaml.load(f, Loader=yaml.Loader), run_stats, force_run_all, filename
            )

    @classmethod
    def default(cls, run_stats=True, force_run_all=False):
        """Generate a default settings object.

        Parameters
        ----------
        run_stats : bool, optional
            whether to include all stats listed or just return the main metric, by default True

            This can simplify the results dataframe if you only want the main metric.
            For example, rather than including L_5, L_50, etc. will only include LEq
        force_run_all : bool, optional
            whether to force all metrics to run regardless of what is set in their options, by default False

            Use Cautiously. This can be useful if you want to run all metrics, but don't want to change the yaml file.
            Warning: If both mosqito:loudness_zwtv and mosqito:sharpness_din_from_loudness are present in the settings
            file, this will result in the loudness calc being run twice.

        Returns
        -------
        AnalysisSettings
            AnalysisSettings object
        """
        import soundscapy

        root = Path(soundscapy.__path__[0])
        return cls(
            AnalysisSettings.from_yaml(
                Path(root, "analysis", "default_settings.yaml"),
                run_stats,
                force_run_all,
            )
        )

    def reload(self):
        """Reload the settings from the yaml file."""
        return self.from_yaml(self.filepath, self.run_stats, self.force_run_all)

    def to_yaml(self, filename: Union[Path, str]):
        """Save settings to a yaml file.

        Parameters
        ----------
        filename : Path object or str
            filename of the yaml file
        """
        with open(filename, "w") as f:
            yaml.dump(self, f)

    def parse_maad_all_alpha_indices(self, metric: str):
        """Generate relevant settings for the maad all_alpha_indices methods.

        Parameters
        ----------
        metric : str
            metric to prepare for

        Returns
        -------
        run: bool
            Whether to run the metric
        channel: tuple or list of str, or str
            channel(s) to run the metric on
        """
        assert metric in [
            "all_temporal_alpha_indices",
            "all_spectral_alpha_indices",
        ], "metric must be all_temporal_alpha_indices or all_spectral_alpha_indices."

        lib_settings = self["scikit-maad"].copy()
        run = lib_settings[metric]["run"] or self.force_run_all
        channel = lib_settings[metric]["channel"].copy()
        return run, channel

    def parse_pyacoustics(self, metric: str):
        """Generate relevant settings for a pyacoustics metric.

        Parameters
        ----------
        metric : str
            metric to prepare for

        Returns
        -------
        run: bool
            Whether to run the metric
        channel: tuple or list of str, or str
            channel(s) to run the metric on
        statistics: tuple or list of str, or str
            statistics to run the metric on.
            If run_stats is False, will only return the main statistic
        label: str
            label to use for the metric
        func_args: dict
            arguments to pass to the underlying metric function from python acoustics
        """
        return self._parse_method("PythonAcoustics", metric)

    def parse_mosqito(self, metric: str):
        """Generate relevant settings for a mosqito metric.

        Parameters
        ----------
        metric : str
            metric to prepare for

        Returns
        -------
        run: bool
            Whether to run the metric
        channel: tuple or list of str, or str
            channel(s) to run the metric on
        statistics: tuple or list of str, or str
            statistics to run the metric on.
            If run_stats is False, will only return the main statistic
        label: str
            label to use for the metric
        func_args: dict
            arguments to pass to the underlying metric function from MoSQITo
        """
        assert metric in [
            "loudness_zwtv",
            "sharpness_din_from_loudness",
            "sharpness_din_perseg",
            "sharpness_din_tv",
            "roughness_dw",
        ], f"Metric {metric} not found."
        run, channel, statistics, label, func_args = self._parse_method(
            "MoSQITo", metric
        )
        try:
            parallel = self["MoSQITo"][metric]["parallel"]
        except KeyError:
            parallel = False
        # Check for sub metric
        # if sub metric is present, don't run this metric
        if (
            metric == "loudness_zwtv"
            and "sharpness_din_from_loudness" in self["MoSQITo"].keys()
            and self["MoSQITo"]["sharpness_din_from_loudness"]["run"]
            and self.force_run_all is False
        ):
            run = False
        return run, channel, statistics, label, parallel, func_args

    def _parse_method(self, library: str, metric: str):
        """Helper function to return relevant settings for a library

        Parameters
        ----------
        library : str
            Library containing the metric
        metric : str
            metric to prepare for

        Returns
        -------
        run: bool
            Whether to run the metric
        channel: tuple or list of str, or str
            channel(s) to run the metric on
        statistics: tuple or list of str, or str
            statistics to run the metric on.
            If run_stats is False, will only return the main statistic
        label: str
            label to use for the metric
        func_args: dict
            arguments to pass to the underlying metric function from the library
        """
        lib_settings = self[library].copy()
        channel = lib_settings[metric]["channel"].copy()
        statistics = lib_settings[metric]["statistics"].copy() if self.run_stats else []
        label = lib_settings[metric]["label"]
        func_args = (
            lib_settings[metric]["func_args"]
            if "func_args" in lib_settings[metric].keys()
            else {}
        )

        main_stat = lib_settings[metric]["main"]
        statistics = (
            (list(statistics) + [main_stat])
            if main_stat not in statistics
            else tuple(statistics)
        )

        # Override metric run settings if force_run_all is True
        run = lib_settings[metric]["run"] or self.force_run_all
        return run, channel, statistics, label, func_args


# %%
