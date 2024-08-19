from dataclasses import dataclass, field
from typing import Any, Dict, Tuple, Union

import h5py
import numpy as np
import pandas as pd
from loguru import logger
from mosqito.sq_metrics import loudness_zwtv
from soundscapy.audio.binaural_signal import BinauralSignal
from soundscapy.audio.metric_registry import Metric, stat_calcs
from soundscapy.audio.result_storage import AnalysisResult


# Zwicker Time Varying Loudness (Mosqito)


@dataclass
class LoudnessZWTVResult(AnalysisResult):
    channel: str | int = field(default=None)
    N: np.ndarray = field(default_factory=lambda: np.array([]))
    N_specific: np.ndarray = field(default_factory=lambda: np.array([]))
    bark_axis: np.ndarray = field(default_factory=lambda: np.array([]))
    time_axis: np.ndarray = field(default_factory=lambda: np.array([]))

    def get_time_series(self) -> pd.DataFrame:
        return pd.DataFrame({"Time": self.time_axis, "Loudness": self.N})

    def get_specific_loudness(self) -> pd.DataFrame:
        return pd.DataFrame(
            self.N_specific, index=self.bark_axis, columns=self.time_axis
        )

    def get_summary_statistics(
        self,
        label: str = "N",
        statistics: Tuple[str | int] = ("mean", "max", "min", 5, 10, 50, 90, 95),
    ) -> Dict[str, Any]:
        logger.debug(
            f"Calculating summary statistics for Zwicker Time Varying Loudness ({label}): {statistics}"
        )
        return stat_calcs(label, self.N, statistics)

    def get_default_primary_stat(self):
        logger.info(
            "Calculating default primary statistic for Zwicker Time Varying Loudness: N_5"
        )
        stats = stat_calcs()
        return stats["N_5"]

    def save(self, location: Union[str, h5py.Group]):
        if isinstance(location, str):
            with h5py.File(location, "w") as f:
                self._save_to_group(f)
        elif isinstance(location, h5py.Group):
            self._save_to_group(location)
        else:
            raise ValueError("Invalid save location. Must be a file path or h5py.Group")

    def load(self, location: Union[str, h5py.Group]):
        if isinstance(location, str):
            with h5py.File(location, "r") as f:
                self._load_from_group(f)
        elif isinstance(location, h5py.Group):
            self._load_from_group(location)
        else:
            raise ValueError("Invalid load location. Must be a file path or h5py.Group")

    def _save_to_group(self, group: h5py.Group):
        group.create_dataset("N", data=self.N)
        group.create_dataset("N_specific", data=self.N_specific)
        group.create_dataset("bark_axis", data=self.bark_axis)
        group.create_dataset("time_axis", data=self.time_axis)
        group.attrs["channel"] = self.channel

    def _load_from_group(self, group: h5py.Group):
        self.N = group["N"][:]
        self.N_specific = group["N_specific"][:]
        self.bark_axis = group["bark_axis"][:]
        self.time_axis = group["time_axis"][:]
        self.channel = group.attrs["channel"]

    def __repr__(self):
        return f"LoudnessZWTVResult(channel={self.channel}, N.shape={self.N.shape}, N_specific.shape={self.N_specific.shape})"


class LoudnessZWTV(Metric):
    def __init__(self):
        super().__init__()
        self.settings = {
            "field_type": "free",
            "statistics": ["mean", "max", "min", 5, 10, 50, 90, 95],
        }
        self.results = LoudnessZWTVResult()

    def calculate(self, audio_data: BinauralSignal) -> LoudnessZWTVResult:
        try:
            N, N_spec, bark_axis, time_axis = loudness_zwtv(
                audio_data, audio_data.fs, field_type=self.settings["field_type"]
            )

            self.results.N = N
            self.results.N_specific = N_spec
            self.results.bark_axis = bark_axis
            self.results.time_axis = time_axis

            return self.results

        except Exception as e:
            logger.error(f"Error calculating Zwicker Time Varying Loudness: {str(e)}")
            raise e
