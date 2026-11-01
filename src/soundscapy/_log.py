"""The ``soundscapy`` logger and the one call that silences it.

The default is visible (NFR-13): with no logging configured, INFO lines go to stderr, so a
terminal or a notebook shows them. Once the application configures logging, its
configuration decides: records reach its handlers only at or above the root logger's level,
unless the level of the ``soundscapy`` logger was set deliberately, by ``set_log_level()``
or by the application itself. Records always propagate, so ``caplog`` and a GUI's own
handler receive them.

All of soundscapy logs through ``logger`` itself, never through a child logger such as
``soundscapy.survey``: the filter that applies the application's root level is attached to
this logger, and Python consults a logger's filters only for records created on it.
"""

import logging
import sys

__all__ = ["logger", "set_log_level"]

_DEFAULT_LEVEL = logging.INFO

logger = logging.getLogger("soundscapy")
if logger.level == logging.NOTSET:  # never override a level the application already set
    logger.setLevel(_DEFAULT_LEVEL)


def _application_configured() -> bool:
    """Return whether the application has configured logging (the root logger has handlers)."""
    return bool(logging.getLogger().handlers)


class _DefaultHandler(logging.StreamHandler):
    """Write to stderr unless the application configured logging."""

    def emit(self, record: logging.LogRecord) -> None:
        # Stream looked up at emit time, so notebooks capture it.
        if _application_configured():
            return  # the application's handlers show the record
        self.stream = sys.stderr
        super().emit(record)


def _defer_to_application(record: logging.LogRecord) -> bool:
    """Apply the root logger's level once the application has configured logging.

    A logger's own level is all that decides whether a record propagates; ancestor levels
    are never consulted (Python ``Logger.propagate``). Without this filter, the INFO default
    above would pass soundscapy's INFO lines to the application's handlers even after
    ``logging.basicConfig(level=logging.WARNING)``.

    An application that sets the ``soundscapy`` logger to any level but INFO keeps that
    level; one that sets it to INFO, or a GUI that adds its own handler to the
    ``soundscapy`` logger, calls ``set_log_level()`` to keep INFO lines whatever the
    root level is, because this filter cannot tell that choice from the default.
    """
    if logger.level != _DEFAULT_LEVEL or not _application_configured():
        return True
    return record.levelno >= logging.getLogger().getEffectiveLevel()


logger.addFilter(_defer_to_application)
logger.addHandler(_DefaultHandler())


def set_log_level(level: int | str) -> None:
    """Set the level of the ``soundscapy`` logger.

    This is the one silencing call; there is no environment variable. Standard logging
    configuration works too: an application's root level applies to soundscapy's records
    unless this function, or the application, sets the ``soundscapy`` logger's level.

    Parameters
    ----------
    level : int or str
        A ``logging`` level number or name, for example ``"WARNING"``.
    """
    logger.setLevel(level)
    logger.removeFilter(_defer_to_application)  # an explicit level is the user's decision
