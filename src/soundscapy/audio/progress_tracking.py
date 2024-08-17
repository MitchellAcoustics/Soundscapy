from typing import Callable, Optional

from loguru import logger


class ProgressTracker:
    def __init__(self, total: int, description: str = "Processing"):
        self.total = total
        self.current = 0
        self.description = description
        self._callback: Optional[Callable[[int, int, str], None]] = None

    def update(self, increment: int = 1):
        self.current += increment
        if self.current > self.total:
            logger.warning(f"Progress exceeded total: {self.current}/{self.total}")
            self.current = self.total
        self._report_progress()

    def set_total(self, total: int):
        self.total = total
        self._report_progress()

    def set_description(self, description: str):
        self.description = description
        self._report_progress()

    def reset(self):
        self.current = 0
        self._report_progress()

    def set_callback(self, callback: Callable[[int, int, str], None]):
        self._callback = callback

    def _report_progress(self):
        if self._callback:
            self._callback(self.current, self.total, self.description)
        else:
            percentage = (self.current / self.total) * 100 if self.total > 0 else 0
            logger.info(
                f"{self.description}: {self.current}/{self.total} ({percentage:.2f}%)"
            )


class NestedProgressTracker:
    def __init__(self, levels: int):
        self.trackers = [ProgressTracker(100, f"Level {i+1}") for i in range(levels)]

    def update(self, level: int, increment: int = 1):
        if 0 <= level < len(self.trackers):
            self.trackers[level].update(increment)
        else:
            logger.error(f"Invalid progress tracking level: {level}")

    def set_total(self, level: int, total: int):
        if 0 <= level < len(self.trackers):
            self.trackers[level].set_total(total)
        else:
            logger.error(f"Invalid progress tracking level: {level}")

    def set_description(self, level: int, description: str):
        if 0 <= level < len(self.trackers):
            self.trackers[level].set_description(description)
        else:
            logger.error(f"Invalid progress tracking level: {level}")

    def reset(self, level: int):
        if 0 <= level < len(self.trackers):
            self.trackers[level].reset()
        else:
            logger.error(f"Invalid progress tracking level: {level}")

    def set_callback(self, callback: Callable[[int, int, str], None]):
        for tracker in self.trackers:
            tracker.set_callback(callback)


# Example usage
def progress_callback(current: int, total: int, description: str):
    percentage = (current / total) * 100 if total > 0 else 0
    print(f"{description}: {current}/{total} ({percentage:.2f}%)")


# progress_tracker = ProgressTracker(100, "Processing files")
# progress_tracker.set_callback(progress_callback)
#
# nested_tracker = NestedProgressTracker(2)
# nested_tracker.set_callback(progress_callback)
