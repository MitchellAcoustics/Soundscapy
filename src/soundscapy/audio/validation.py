from typing import Any, Dict

from error_handling import error_handler


class ValidationError(Exception):
    pass


class ValidationLayer:
    @staticmethod
    def validate_audio_file(file_path: str):
        # Placeholder for audio file validation
        # This would typically check file format, sample rate, etc.
        if not file_path.endswith((".wav", ".flac")):
            error_handler.error(
                f"Unsupported audio file format: {file_path}", "INVALID_AUDIO_FORMAT"
            )
            raise ValidationError(f"Unsupported audio file format: {file_path}")

    @staticmethod
    def validate_config(config: Dict[str, Any]):
        required_keys = ["segment_length", "overlap", "max_workers"]
        for key in required_keys:
            if key not in config:
                error_handler.error(
                    f"Missing required configuration key: {key}", "INVALID_CONFIG"
                )
                raise ValidationError(f"Missing required configuration key: {key}")

    @staticmethod
    def validate_metric_input(audio_data: Any):
        # Placeholder for metric input validation
        # This would typically check the shape, dtype, etc. of the audio data
        if not isinstance(audio_data, (list, tuple)):
            error_handler.error("Invalid audio data format", "INVALID_METRIC_INPUT")
            raise ValidationError("Invalid audio data format")

    @staticmethod
    def validate_metric_output(metric_result: Any):
        # Placeholder for metric output validation
        # This would typically check the structure and values of the metric result
        if not isinstance(metric_result, dict):
            error_handler.error("Invalid metric result format", "INVALID_METRIC_OUTPUT")
            raise ValidationError("Invalid metric result format")
