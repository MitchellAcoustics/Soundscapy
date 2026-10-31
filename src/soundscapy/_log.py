"""The ``soundscapy`` logger and the one call that silences it."""

import logging
import sys

__all__ = ["logger", "set_log_level"]

logger = logging.getLogger("soundscapy")
logger.setLevel(logging.INFO)


class _DefaultHandler(logging.StreamHandler):
    """Write to stderr unless the application configured logging."""

    def emit(self, record: logging.LogRecord) -> None:
        # Stream looked up at emit time, so notebooks capture it.
        if logging.getLogger().handlers:
            return  # the application configured logging: defer to it
        self.stream = sys.stderr
        super().emit(record)


# propagate stays True, so caplog and GUIs receive records
logger.addHandler(_DefaultHandler())


def set_log_level(level: int | str) -> None:
    """Set the level of the ``soundscapy`` logger.

    This is the one silencing call; there is no environment variable.

    Parameters
    ----------
    level : int or str
        A :mod:`logging` level number or name, for example ``"WARNING"``.
    """
    logger.setLevel(level)
