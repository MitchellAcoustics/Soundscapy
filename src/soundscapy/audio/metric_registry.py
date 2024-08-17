from abc import ABC, abstractmethod
from typing import Any, Dict, List

import numpy as np
from loguru import logger
from mosqito.sq_metrics import loudness_zwtv
from scipy import stats

from binaural_signal import BinauralSignal


class Metric(ABC):
    def __init__(self):
        self.settings = {}

    @abstractmethod
    def calculate(self, audio_data: Any) -> Dict[str, Any]:
        pass

    def configure(self, settings: Dict[str, Any]):
        self.settings.update(settings)


class MetricRegistry:
    def __init__(self):
        self._metrics = {}

    def register(
        self, name: str, metric_class: type, default_settings: Dict[str, Any] = None
    ):
        if name in self._metrics:
            logger.warning(f"Overwriting existing metric: {name}")
        metric = metric_class()
        if default_settings:
            metric.configure(default_settings)
        self._metrics[name] = metric
        logger.info(f"Registered metric: {name}")

    def unregister(self, name: str):
        if name in self._metrics:
            del self._metrics[name]
            logger.info(f"Unregistered metric: {name}")
        else:
            logger.warning(f"Attempted to unregister non-existent metric: {name}")

    def get_metric(self, name: str) -> Metric:
        if name not in self._metrics:
            logger.error(f"Requested metric not found: {name}", "METRIC_NOT_FOUND")
            raise KeyError(f"Metric not found: {name}")
        return self._metrics[name]

    def configure_metric(self, name: str, settings: Dict[str, Any]):
        if name in self._metrics:
            self._metrics[name].configure(settings)
            logger.info(f"Configured metric: {name}")
        else:
            logger.error(
                f"Attempted to configure non-existent metric: {name}",
                "METRIC_NOT_FOUND",
            )
            raise KeyError(f"Metric not found: {name}")

    def configure_all(self, config: Dict[str, Dict[str, Any]]):
        for name, settings in config.items():
            self.configure_metric(name, settings)


class ZwickerTimeVaryingLoudness(Metric):
    def __init__(self):
        super().__init__()
        self.settings = {"field_type": "free", "statistics": ["mean", "max", "min"]}

    def calculate(self, audio_data: BinauralSignal) -> Dict[str, Any]:
        try:
            N, N_spec, bark_axis, time_axis = loudness_zwtv(
                audio_data, audio_data.fs, field_type=self.settings["field_type"]
            )

            results = {
                "time_varying": N.tolist(),
                "specific": N_spec.tolist(),
                "bark_axis": bark_axis.tolist(),
                "time_axis": time_axis.tolist(),
                "stats": _stat_calcs(
                    "N",
                    N,
                    {},
                    self.settings["statistics"],
                ),
            }

            return results
        except Exception as e:
            logger.error(f"Error calculating Zwicker Time Varying Loudness: {str(e)}")
            return {}


def _stat_calcs(
    label: str, ts_array: np.ndarray, res: dict, statistics: List[int | str]
) -> dict:
    """
    Calculate various statistics for a time series array and add them to a results dictionary.

    This function computes specified statistics (e.g., percentiles, mean, max, min, kurtosis, skewness)
    for a given time series and adds them to a results dictionary with appropriate labels.

    Args:
        label (str): Base label for the statistic in the results dictionary.
        ts_array (np.ndarray): 1D numpy array of the time series data.
        res (dict): Existing results dictionary to update with new statistics.
        statistics (List[Union[int, str]]): List of statistics to calculate. Can include percentiles
            (as integers) and string identifiers for other statistics (e.g., "avg", "max", "min", "kurt", "skew").

    Returns:
        dict: Updated results dictionary with newly calculated statistics.

    Example:
        >>> ts = np.array([1, 2, 3, 4, 5])
        >>> res = {}
        >>> updated_res = _stat_calcs("metric", ts, res, [50, "avg", "max"])
        >>> print(updated_res)
        {'metric_50': 3.0, 'metric_avg': 3.0, 'metric_max': 5}
    """

    for stat in statistics:
        if stat in ("avg", "mean"):
            res[f"{label}_{stat}"] = ts_array.mean()
        elif stat == "max":
            res[f"{label}_{stat}"] = ts_array.max()
        elif stat == "min":
            res[f"{label}_{stat}"] = ts_array.min()
        elif stat == "kurt":
            res[f"{label}_{stat}"] = stats.kurtosis(ts_array)
        elif stat == "skew":
            res[f"{label}_{stat}"] = stats.skew(ts_array)
        elif stat == "std":
            res[f"{label}_{stat}"] = np.std(ts_array)
        else:
            res[f"{label}_{stat}"] = np.percentile(ts_array, 100 - stat)
    return res


# Global instance
metric_registry = MetricRegistry()

# Register the Zwicker Time Varying Loudness metric
metric_registry.register("zwicker_time_varying_loudness", ZwickerTimeVaryingLoudness)
