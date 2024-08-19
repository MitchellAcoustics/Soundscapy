# %%

import os
from typing import Dict, List, Union

import matplotlib.pyplot as plt
import pandas as pd
from soundscapy.audio.processing_engine import ProcessingEngine, PsychoacousticProcessor
from soundscapy.audio.result_storage import (
    DirectoryAnalysisResults,
    FileAnalysisResults,
)
from soundscapy.audio.state_manager import StateManager


class SoundscapyAnalyzer:
    def __init__(self, config: Dict = None):
        self.config = config or {"metrics": {"loudness_zwtv": {"enabled": True}}}
        self.processor = PsychoacousticProcessor(self.config)
        self.processing_engine = ProcessingEngine(self.processor)
        self.state_manager = StateManager("soundscapy_state.json")
        self.results = None

    def analyze_file(self, file_path: str) -> FileAnalysisResults:
        """Analyze a single audio file."""
        self.results = self.processing_engine._process_file(
            file_path, self.processor, self.state_manager
        )
        return self.results

    def analyze_directory(self, directory_path: str) -> DirectoryAnalysisResults:
        """Analyze all audio files in a directory."""
        self.results = self.processing_engine.process_directory(
            directory_path, self.state_manager
        )
        return self.results

    def get_results(self) -> Union[FileAnalysisResults, DirectoryAnalysisResults]:
        """Get the results of the last analysis."""
        if self.results is None:
            raise ValueError("No analysis has been performed yet.")
        return self.results

    def get_summary_stats(self) -> pd.DataFrame:
        """Get summary statistics for all analyzed files."""
        if isinstance(self.results, FileAnalysisResults):
            return pd.DataFrame(self.results.get_stats())
        elif isinstance(self.results, DirectoryAnalysisResults):
            stats = {}
            for file_path, file_result in self.results.file_results.items():
                stats[os.path.basename(file_path)] = file_result.get_stats()
            return pd.DataFrame(stats)
        else:
            raise ValueError("No valid results found.")

    def plot_loudness(self, file_name: str = None):
        """Plot loudness over time for a specific file or all files if not specified."""
        if file_name:
            if isinstance(self.results, FileAnalysisResults):
                self._plot_single_file_loudness(self.results)
            elif isinstance(self.results, DirectoryAnalysisResults):
                file_result = self.results.get_file_result(file_name)
                self._plot_single_file_loudness(file_result)
        else:
            if isinstance(self.results, FileAnalysisResults):
                self._plot_single_file_loudness(self.results)
            elif isinstance(self.results, DirectoryAnalysisResults):
                fig, axes = plt.subplots(
                    len(self.results.file_results),
                    1,
                    figsize=(10, 5 * len(self.results.file_results)),
                )
                for i, (file_path, file_result) in enumerate(
                    self.results.file_results.items()
                ):
                    self._plot_single_file_loudness(file_result, ax=axes[i])
                    axes[i].set_title(os.path.basename(file_path))
                plt.tight_layout()
        plt.show()

    def _plot_single_file_loudness(self, file_result: FileAnalysisResults, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))
        loudness_result = file_result.get_metric_result("loudness_zwtv")
        for channel, result in loudness_result.channels.items():
            df = result.get_time_series()
            ax.plot(df["Time"], df["Loudness"], label=channel)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Loudness (sone)")
        ax.legend()
        ax.set_title("Loudness over Time")

    def export_results(self, output_path: str):
        """Export results to a CSV file."""
        self.get_summary_stats().to_csv(output_path)
        print(f"Results exported to {output_path}")

    def add_metric(self, metric_name: str, metric_config: Dict):
        """Add a new metric to the analysis configuration."""
        self.config["metrics"][metric_name] = metric_config
        self.processor = PsychoacousticProcessor(self.config)
        self.processing_engine = ProcessingEngine(self.processor)

    def remove_metric(self, metric_name: str):
        """Remove a metric from the analysis configuration."""
        if metric_name in self.config["metrics"]:
            del self.config["metrics"][metric_name]
            self.processor = PsychoacousticProcessor(self.config)
            self.processing_engine = ProcessingEngine(self.processor)
        else:
            print(f"Metric '{metric_name}' not found in the current configuration.")

    def list_available_metrics(self) -> List[str]:
        """List all available metrics."""
        from soundscapy.audio import metric_registry

        return list(metric_registry._metrics.keys())

    def get_current_config(self) -> Dict:
        """Get the current analysis configuration."""
        return self.config


# %%
if __name__ == "__main__":
    # %%
    # Create an analyzer with default configuration
    analyzer = SoundscapyAnalyzer()

    # Analyze a single file
    result = analyzer.analyze_file(
        "/Users/mitch/Documents/GitHub/Soundscapy/test/data/CT101.wav"
    )
    print(result)

    # Analyze a directory
    results = analyzer.analyze_directory(
        "/Users/mitch/Documents/GitHub/Soundscapy/test/data"
    )
    print(results)

    # Get summary statistics
    stats = analyzer.get_summary_stats()
    print(stats)

    # Plot loudness for all analyzed files
    analyzer.plot_loudness()

    # Export results to CSV
    analyzer.export_results("analysis_results.csv")

    # Add a new metric (assuming 'sharpness' is an available metric)
    analyzer.add_metric("sharpness", {"enabled": True})

    # Remove a metric
    analyzer.remove_metric("loudness_zwtv")

    # List available metrics
    print(analyzer.list_available_metrics())

    # Get current configuration
    print(analyzer.get_current_config())
