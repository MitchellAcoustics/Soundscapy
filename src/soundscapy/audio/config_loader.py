from typing import Any, Dict, Optional

import yaml
from loguru import logger


class ConfigLoader:
    def __init__(self, config_path: Optional[str] = None, **overrides):
        self.config: Dict[str, Any] = {}
        if config_path:
            self.load_from_yaml(config_path)
        self.apply_overrides(overrides)

        if self.get("force_reprocess"):
            logger.info("Force reprocess is enabled. All files will be reprocessed.")

    def load_from_yaml(self, config_path: str):
        try:
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f)
        except (IOError, yaml.YAMLError) as e:
            logger.error(
                f"Failed to load configuration from {config_path}. Error: {str(e)}"
            )
            raise

    def apply_overrides(self, overrides: Dict[str, Any]):
        self.config.update(overrides)

    def get(self, key: str, default: Any = None) -> Any:
        return self.config.get(key, default)

    def __getitem__(self, key: str) -> Any:
        return self.config[key]

    def __contains__(self, key: str) -> bool:
        return key in self.config


# Example usage
# config = ConfigLoader('config.yaml', segment_length=1.0, overlap=0.5)
# segment_length = config.get('segment_length', 0.5)  # Default to 0.5 if not set
