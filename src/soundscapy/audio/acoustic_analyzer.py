from loguru import logger

from config_loader import ConfigLoader
from processing_engine import ProcessingEngine, PsychoacousticProcessor
from result_storage import create_result_storage
from state_manager import StateManager


class AcousticAnalyzer:
    def __init__(self, config_path: str, state_file: str):
        self.config = ConfigLoader(config_path)
        self.state_manager = StateManager(state_file)
        self.processor = PsychoacousticProcessor(self.config.config)
        self.processing_engine = ProcessingEngine(
            self.processor, max_workers=self.config.get("max_workers", 4)
        )
        self.result_storage = create_result_storage(
            self.config.get("result_storage_type", "json"),
            self.config.get("result_storage_path", "results.json"),
        )

    def analyze_file(self, file_path: str, parallel_channels: bool = False):
        logger.info(f"Starting analysis of file: {file_path}")
        result = self.processing_engine.process_file(
            file_path, self.state_manager, parallel_channels
        )
        self.result_storage.store([result])
        logger.info(
            f"Analysis complete. Results stored in {self.config.get('result_storage_path', 'results.json')}"
        )

    def analyze_directory(self, directory_path: str):
        logger.info(f"Starting analysis of directory: {directory_path}")
        results = self.processing_engine.process_directory(
            directory_path, self.state_manager
        )
        self.result_storage.store(results)
        logger.info(
            f"Analysis complete. Results stored in {self.config.get('result_storage_path', 'results.json')}"
        )

    def reset_state(self):
        self.state_manager.reset()
        logger.info("Processing state has been reset.")


# Example usage
# analyzer = PsychoacousticAnalyzer('config.yaml', 'processing_state.json')
# analyzer.analyze_directory('/path/to/audio/files')
