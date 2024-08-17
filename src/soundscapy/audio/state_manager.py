import json
from typing import Any, Dict

from loguru import logger


class StateManager:
    def __init__(self, state_file: str):
        self.state_file = state_file
        self.state: Dict[str, Any] = self.load_state()

    def load_state(self) -> Dict[str, Any]:
        try:
            with open(self.state_file, "r") as f:
                return json.load(f)
        except (IOError, json.JSONDecodeError):
            logger.info(
                f"No valid state file found at {self.state_file}. Starting with empty state."
            )
            return {}

    def save_state(self):
        try:
            with open(self.state_file, "w") as f:
                json.dump(self.state, f)
        except IOError as e:
            logger.error(f"Failed to save state to {self.state_file}. Error: {str(e)}")

    def mark_processed(self, file_path: str):
        self.state[file_path] = True
        self.save_state()

    def is_processed(self, file_path: str) -> bool:
        return self.state.get(file_path, False)

    def reset(self):
        self.state = {}
        self.save_state()


# Example usage
# state_manager = StateManager('processing_state.json')
# if not state_manager.is_processed('audio1.wav'):
#     # Process the file
#     state_manager.mark_processed('audio1.wav')
