# %%
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Generic, List, Type, TypeVar, Union

import h5py
import numpy as np
import pandas as pd


# Abstract base class for handling analysis results
@dataclass
class AnalysisResult:
    def get_time_series(self) -> pd.DataFrame:
        raise NotImplementedError

    def get_summary_statistics(self) -> Dict[str, float]:
        raise NotImplementedError

    @abstractmethod
    def save(self, location: Union[str, h5py.Group]):
        pass

    @abstractmethod
    def load(self, location: Union[str, h5py.Group]):
        pass


# Type variable for the generic single-channel result
T = TypeVar("T", bound=AnalysisResult)


@dataclass
class MultiChannelResult(Generic[T]):
    channels: Dict[str, T] = field(default_factory=dict)

    def add_channel_result(self, channel: str, result: T):
        self.channels[channel] = result

    def get_channel_result(self, channel: str) -> T:
        if channel not in self.channels:
            raise ValueError(f"Channel '{channel}' not found.")
        return self.channels[channel]

    def get_all_channel_results(self) -> Dict[str, T]:
        return self.channels

    def get_all_channel_stats(self) -> Dict[str, Dict[str, float]]:
        return {
            channel: result.get_summary_statistics()
            for channel, result in self.channels.items()
        }

    def save(self, location: Union[str, h5py.Group]):
        if isinstance(location, str):
            with h5py.File(location, "w") as f:
                self._save_to_group(f)
        elif isinstance(location, h5py.Group):
            self._save_to_group(location)
        else:
            raise ValueError("Invalid save location. Must be a file path or h5py.Group")

    def load(self, location: Union[str, h5py.Group], result_class: Type[T]):
        if isinstance(location, str):
            with h5py.File(location, "r") as f:
                self._load_from_group(f, result_class)
        elif isinstance(location, h5py.Group):
            self._load_from_group(location, result_class)
        else:
            raise ValueError("Invalid load location. Must be a file path or h5py.Group")

    def _save_to_group(self, group: h5py.Group):
        for channel, result in self.channels.items():
            channel_group = group.create_group(channel)
            result.save(channel_group)

    def _load_from_group(self, group: h5py.Group, result_class: Type[T]):
        for channel in group.keys():
            result = result_class()
            channel_group = group[channel]
            result.load(channel_group)
            self.channels[channel] = result

    def __repr__(self):
        return f"MultiChannelResult(channels={list(self.channels.keys())})"


@dataclass
class FileAnalysisResults:
    file_path: str
    metrics: Dict[str, MultiChannelResult] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    def list_metrics(self) -> List[str]:
        return list(self.metrics.keys())

    def add_metric_result(self, metric_name: str, result: MultiChannelResult):
        self.metrics[metric_name] = result

    def get_metric_result(self, metric_name: str) -> MultiChannelResult:
        if metric_name not in self.metrics:
            raise ValueError(f"Metric '{metric_name}' not found.")
        return self.metrics[metric_name]

    def get_all_metric_results(self) -> Dict[str, MultiChannelResult]:
        return self.metrics

    def get_stats(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        return {
            metric: result.get_all_channel_stats()
            for metric, result in self.metrics.items()
        }

    def add_error(self, error_message: str):
        self.errors.append(error_message)

    def has_errors(self) -> bool:
        return len(self.errors) > 0

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
        group.attrs["file_path"] = self.file_path
        for metric_name, result in self.metrics.items():
            metric_group = group.create_group(metric_name)
            result.save(metric_group)
        if self.errors:
            group.create_dataset("errors", data=self.errors)

    def _load_from_group(self, group: h5py.Group):
        from soundscapy.audio import metric_registry

        self.file_path = group.attrs["file_path"]
        for metric_name in group.keys():
            if metric_name != "errors":
                metric_group = group[metric_name]
                result_class = metric_registry.get_result_class(metric_name)
                multi_channel_result = MultiChannelResult()
                multi_channel_result.load(metric_group, result_class)
                self.metrics[metric_name] = multi_channel_result
        if "errors" in group:
            self.errors = list(group["errors"])

    def __repr__(self):
        return f"FileAnalysisResults(file_path={self.file_path}, metrics={list(self.metrics.keys())}, errors={len(self.errors)})"


@dataclass
class DirectoryAnalysisResults:
    directory_path: str
    file_results: Dict[str, FileAnalysisResults] = field(default_factory=dict)

    def add_file_result(self, file_path: str, result: FileAnalysisResults):
        self.file_results[file_path] = result

    def get_file_result(self, file_path: str) -> FileAnalysisResults:
        if file_path not in self.file_results:
            raise ValueError(f"Results for file '{file_path}' not found.")
        return self.file_results[file_path]

    def get_all_file_results(self) -> Dict[str, FileAnalysisResults]:
        return self.file_results

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
        group.attrs["directory_path"] = self.directory_path
        for file_path, file_result in self.file_results.items():
            file_group = group.create_group(file_path)
            file_result.save(file_group)

    def _load_from_group(self, group: h5py.Group):
        self.directory_path = group.attrs["directory_path"]
        for file_path in group.keys():
            file_group = group[file_path]
            file_result = FileAnalysisResults(file_path)
            file_result.load(file_group)
            self.file_results[file_path] = file_result

    @staticmethod
    def _load_channel_result(channel_group, metric_name: str):
        from soundscapy.audio.mosqito_metrics import LoudnessZWTVResult

        # This method should be the same as in FileAnalysisResults
        if metric_name == "loudness_zwtv":
            result = LoudnessZWTVResult(
                channel=channel_group.name.split("/")[-1],
                N=channel_group["N"][:],
                N_specific=channel_group["N_specific"][:],
                bark_axis=channel_group["bark_axis"][:],
                time_axis=channel_group["time_axis"][:],
            )
        else:
            raise ValueError(f"Unknown metric type: {metric_name}")
        return result


# %%

if __name__ == "__main__":
    # %%
    from soundscapy.audio.metric_registry import LoudnessZWTVResult

    # Create some sample data
    loudness_result = LoudnessZWTVResult(
        channel="left",
        N=np.random.rand(100),
        N_specific=np.random.rand(24, 100),
        bark_axis=np.arange(24),
        time_axis=np.linspace(0, 1, 100),
    )

    # Create a FileAnalysisResults object
    file_result = FileAnalysisResults("test_file.wav")
    multi_channel_result = MultiChannelResult()
    multi_channel_result.add_channel_result("left", loudness_result)
    file_result.add_metric_result("loudness_zwtv", multi_channel_result)

    # Save the results
    file_result.save("test_results.h5")

    # Load the results
    loaded_result = FileAnalysisResults("test_file.wav")
    loaded_result.load("test_results.h5")

    # Verify the loaded data
    loaded_loudness = loaded_result.get_metric_result(
        "loudness_zwtv"
    ).get_channel_result("left")
    assert np.allclose(loudness_result.N, loaded_loudness.N)
    assert np.allclose(loudness_result.N_specific, loaded_loudness.N_specific)
    assert np.allclose(loudness_result.bark_axis, loaded_loudness.bark_axis)
    assert np.allclose(loudness_result.time_axis, loaded_loudness.time_axis)
    assert loudness_result.channel == loaded_loudness.channel

    print("Test passed successfully!")
