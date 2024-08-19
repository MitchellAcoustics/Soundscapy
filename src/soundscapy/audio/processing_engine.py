import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Any, Dict

from loguru import logger

from soundscapy.audio import metric_registry
from soundscapy.audio.binaural_signal import BinauralSignal
from soundscapy.audio.metric_registry import Metric
from soundscapy.audio.progress_tracking import ProgressTracker
from soundscapy.audio.result_storage import (
    DirectoryAnalysisResults,
    FileAnalysisResults,
    MultiChannelResult,
)
from soundscapy.audio.state_manager import StateManager


# Psychoacoustic Processor
class PsychoacousticProcessor:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.segment_length = config.get("segment_length", None)  # in seconds
        self.overlap = config.get("overlap", 0)  # in seconds
        self.result_detail = config.get("result_detail", "all")
        self.force_reprocess = config.get("force_reprocess", False)

    def process(
        self, audio_data: "BinauralSignal", parallel: bool = False
    ) -> FileAnalysisResults:
        file_results = FileAnalysisResults(file_path=audio_data.file_path)

        try:
            for metric_name, metric_config in self.config.get("metrics", {}).items():
                if metric_config.get("enabled", True):
                    metric = metric_registry.get_metric(metric_name)
                    metric.configure(metric_config)

                    multi_channel_result = MultiChannelResult()

                    if parallel and audio_data.channels == 2:
                        logger.debug("Setting up parallel processing for stereo audio")
                        with ThreadPoolExecutor(max_workers=2) as executor:
                            future_left = executor.submit(
                                self._process_channel, metric, audio_data.left, "left"
                            )
                            future_right = executor.submit(
                                self._process_channel, metric, audio_data.right, "right"
                            )
                            multi_channel_result.add_channel_result(
                                "left", future_left.result()
                            )
                            multi_channel_result.add_channel_result(
                                "right", future_right.result()
                            )
                    else:
                        for i in range(audio_data.channels):
                            channel_name = f"channel_{i+1}"
                            channel_result = self._process_channel(
                                metric, audio_data[i], channel_name
                            )
                            multi_channel_result.add_channel_result(
                                channel_name, channel_result
                            )

                    logger.debug(f"Adding {metric_name} results to file results")
                    file_results.add_metric_result(metric_name, multi_channel_result)

            return file_results

        except Exception as e:
            logger.error(f"Error processing audio data. Error: {str(e)}")
            return file_results

    def _process_channel(
        self, metric: Metric, channel_data: "BinauralSignal", channel_name: str
    ) -> Any:  # Return type depends on the specific metric result class
        logger.debug(f"Processing {channel_name} as a single segment")
        result = metric.calculate(channel_data)

        # Assuming the result is already an instance of the appropriate result class (e.g., LoudnessZWTVResult)
        result.channel = channel_name

        return result


# Example usage
# config = {'metrics': {'loudness': {'enabled': True}, 'sharpness': {'enabled': True}}}
# processor = PsychoacousticProcessor(config)
# result = processor.process('audio1.wav')


class ProcessingEngine:
    def __init__(self, processor, max_workers: int = 4):
        self.processor = processor
        self.max_workers = max_workers

    @staticmethod
    def _process_file(
        file_path: str,
        processor,
        state_manager: StateManager,
        parallel_channels: bool = False,
    ) -> FileAnalysisResults:
        if state_manager.is_processed(file_path) and not processor.force_reprocess:
            logger.info(f"Skipping already processed file: {file_path}")
            return FileAnalysisResults(file_path)

        try:
            audio_data = BinauralSignal.from_wav(file_path)
            result = processor.process(audio_data, parallel=parallel_channels)
            state_manager.mark_processed(file_path)
            return result
        except Exception as e:
            logger.error(f"Failed to process file {file_path}. Error: {str(e)}")
            return FileAnalysisResults(file_path)

    def process_directory(
        self, directory_path: str, state_manager: StateManager
    ) -> DirectoryAnalysisResults:
        file_paths = [
            os.path.join(directory_path, f)
            for f in os.listdir(directory_path)
            if f.endswith(".wav")
        ]
        progress_tracker = ProgressTracker(len(file_paths), "Processing files")

        directory_results = DirectoryAnalysisResults(directory_path)
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_file = {
                executor.submit(
                    self._process_file, file_path, self.processor, state_manager, False
                ): file_path
                for file_path in file_paths
            }
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    result = future.result()
                    directory_results.add_file_result(file_path, result)
                except Exception as e:
                    logger.error(
                        f"Exception occurred while processing {file_path}. Error: {str(e)}"
                    )
                finally:
                    progress_tracker.update()

        return directory_results


# Example usage
# processing_engine = ProcessingEngine(PsychoacousticProcessor())
# state_manager = StateManager('processing_state.json')
# results = processing_engine.process_directory('/path/to/audio/files', state_manager)
