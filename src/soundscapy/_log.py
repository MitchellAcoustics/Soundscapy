"""The ``soundscapy`` logger and the one call that silences it.

The default is visible (NFR-13): with no logging configured, INFO lines go to stderr, so a
terminal or a notebook shows them. Once the application configures logging, its
configuration decides: records reach its handlers only at or above the root logger's level,
unless the level of the ``soundscapy`` logger was set deliberately, by :func:`set_log_level`
or by the application itself. Records always propagate, so ``caplog`` and a GUI's own
handler receive them.
"""

import logging
import sys

__all__ = ["logger", "set_log_level"]

_DEFAULT_LEVEL = logging.INFO

logger = logging.getLogger("soundscapy")
if logger.level == logging.NOTSET:  # never override a level the application already set
    logger.setLevel(_DEFAULT_LEVEL)


class _DefaultHandler(logging.StreamHandler):
    """Write to stderr unless the application configured logging."""

    def emit(self, record: logging.LogRecord) -> None:
        # Stream looked up at emit time, so notebooks capture it.
        if logging.getLogger().handlers:
            return  # the application configured logging: defer to it
        self.stream = sys.stderr
        super().emit(record)


class _DeferToApplication(logging.Filter):
    """Apply the root logger's level once the application has configured logging.

    A logger's own level is all that decides whether a record propagates; ancestor levels
    are never consulted (Python ``Logger.propagate``). Without this filter, the INFO default
    above would pass soundscapy's INFO lines to the application's handlers even after
    ``logging.basicConfig(level=logging.WARNING)``.
    """

    def __init__(self) -> None:
        super().__init__()
        self.level_set = False  # set_log_level was called

    def filter(self, record: logging.LogRecord) -> bool:
        root = logging.getLogger()
        if self.level_set or logger.level != _DEFAULT_LEVEL or not root.handlers:
            return True
        return record.levelno >= root.getEffectiveLevel()


_defer = _DeferToApplication()
logger.addFilter(_defer)
logger.addHandler(_DefaultHandler())


def set_log_level(level: int | str) -> None:
    """Set the level of the ``soundscapy`` logger.

    This is the one silencing call; there is no environment variable. Standard logging
    configuration works too: an application's root level applies to soundscapy's records
    unless this function, or the application, sets the ``soundscapy`` logger's level.

    Parameters
    ----------
    level : int or str
        A :mod:`logging` level number or name, for example ``"WARNING"``.
    """
    logger.setLevel(level)
    _defer.level_set = True
