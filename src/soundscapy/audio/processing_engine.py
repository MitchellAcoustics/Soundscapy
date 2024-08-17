import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

from loguru import logger

from binaural_signal import BinauralSignal
from metric_registry import metric_registry
from progress_tracking import ProgressTracker
from state_manager import StateManager


class PsychoacousticProcessor:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.segment_length = config.get("segment_length", None)  # in seconds
        self.overlap = config.get("overlap", 0)  # in seconds

    def process(
        self, audio_data: BinauralSignal, parallel: bool = False
    ) -> Dict[str, Any]:
        try:
            results = {}
            for metric_name, metric_config in self.config.get("metrics", {}).items():
                if metric_config.get("enabled", True):
                    metric = metric_registry.get_metric(metric_name)
                    metric.configure(metric_config)

                    if parallel and audio_data.channels == 2:
                        with ThreadPoolExecutor(max_workers=2) as executor:
                            future_left = executor.submit(
                                self._process_channel, metric, audio_data.left
                            )
                            future_right = executor.submit(
                                self._process_channel, metric, audio_data.right
                            )
                            channel_results = [
                                future_left.result(),
                                future_right.result(),
                            ]
                    else:
                        channel_results = [
                            self._process_channel(metric, audio_data.channel(i))
                            for i in range(audio_data.channels)
                        ]

                    results[metric_name] = self._combine_channel_results(
                        channel_results
                    )

            return results
        except Exception as e:
            logger.error(f"Error processing audio data. Error: {str(e)}")
            return {"error": str(e)}

    def _process_channel(
        self, metric, channel_data: BinauralSignal
    ) -> List[Dict[str, Any]]:
        if self.segment_length is None:
            return [metric.calculate(channel_data)]
        else:
            segment_samples = int(self.segment_length * channel_data.fs)
            overlap_samples = int(self.overlap * channel_data.fs)
            hop_size = segment_samples - overlap_samples

            segments = []
            for start in range(0, len(channel_data), hop_size):
                end = start + segment_samples
                segment = channel_data[start:end]
                if len(segment) == segment_samples:  # only process full segments
                    segments.append(metric.calculate(segment))

            return segments

    def _combine_channel_results(
        self, channel_results: List[List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        combined = {}
        for i, channel in enumerate(channel_results):
            combined[f"channel_{i + 1}"] = channel
        return combined


# Example usage
# config = {'metrics': {'loudness': {'enabled': True}, 'sharpness': {'enabled': True}}}
# processor = PsychoacousticProcessor(config)
# result = processor.process('audio1.wav')


class ProcessingEngine:
    def __init__(self, processor: PsychoacousticProcessor, max_workers: int = 4):
        self.processor = processor
        self.max_workers = max_workers

    def process_file(
        self,
        file_path: str,
        state_manager: StateManager,
        parallel_channels: bool = False,
    ) -> Dict[str, Any]:
        if state_manager.is_processed(file_path):
            logger.info(f"Skipping already processed file: {file_path}")
            return {}

        try:
            audio_data = BinauralSignal.from_wav(file_path)
            result = self.processor.process(audio_data, parallel=parallel_channels)
            state_manager.mark_processed(file_path)
            return {file_path: result}
        except Exception as e:
            logger.error(f"Failed to process file {file_path}. Error: {str(e)}")
            return {file_path: {"error": str(e)}}

    def process_directory(
        self, directory_path: str, state_manager: StateManager
    ) -> List[Dict[str, Any]]:
        file_paths = [
            os.path.join(directory_path, f)
            for f in os.listdir(directory_path)
            if f.endswith(".wav")
        ]
        progress_tracker = ProgressTracker(len(file_paths), "Processing files")

        results = []
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_file = {
                executor.submit(self.process_file, file_path, state_manager): file_path
                for file_path in file_paths
            }
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as e:
                    logger.error(
                        f"Exception occurred while processing {file_path}. Error: {str(e)}"
                    )
                finally:
                    progress_tracker.update()

        return results


# Example usage
# processing_engine = ProcessingEngine(PsychoacousticProcessor())
# state_manager = StateManager('processing_state.json')
# results = processing_engine.process_directory('/path/to/audio/files', state_manager)
