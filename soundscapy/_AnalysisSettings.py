
# %%
# from WavAnalysis import *
from pathlib import Path
from time import localtime, strftime
from typing import Union

import yaml


# %%
class AnalysisSettings(dict):
    """Dict of settings for analysis methods. Each library has a dict of metrics,
    each of which has a dict of settings.
    """

    def __init__(self, data, run_stats=True, force_run_all=False):
        super().__init__(data)
        self.run_stats = run_stats
        self.force_run_all = force_run_all
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
            return cls(yaml.safe_load(f), run_stats)

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
            Warning: If both mosqito:loudness_zwtv and mosqitsharpness_din_from_loudness are present in the settings file, this will result in the loudness calc being run twice.

        Returns
        -------
        AnalysisSettings
            AnalysisSettings object
        """
        return cls(AnalysisSettings.from_yaml(Path("soundscapy", "default_settings.yaml"), run_stats, force_run_all))

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
        lib_settings = self["maad"].copy()
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
        run, channel, statistics, label, func_args = self._parse_method(
            "MoSQITO", metric
        )
        try:
            parallel = self["MoSQITO"][metric]["parallel"]
        except KeyError:
            parallel = False
        # Check for sub metric
        # if sub metric is present, don't run this metric
        if (
            metric == "loudness_zwtv"
            and "sharpness_din_from_loudness" in self["MoSQITO"].keys()
            and self["MoSQITO"]["sharpness_din_from_loudness"]["run"]
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

