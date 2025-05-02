"""
Logging configuration for Soundscapy.

This module provides simple functions to configure logging for both users and
developers. By default, Soundscapy logging is disabled to avoid unwanted output.
Users can enable logging with the setup_logging function.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import loguru
from loguru import logger

if TYPE_CHECKING:
    from pathlib import Path


def setup_logging(
    level: str = "INFO",
    log_file: str | Path | None = None,
    format_level: str = "basic",
) -> None:
    """
    Set up logging for Soundscapy with sensible defaults.

    Parameters
    ----------
    level : str, default="INFO"
        Logging level for console output.
        Options: "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
    log_file : str or Path, optional
        Path to a log file.
        If provided, all messages (including DEBUG) will be logged to this file.
    format_level : str, default="basic"
        Format complexity level. Options:
        - "basic": Simple format with timestamp, level, and message
        - "detailed": Adds module, function and line information
        - "developer": Adds exception details and diagnostics

    Examples
    --------
    >>> from soundscapy import setup_logging
    >>> # Basic usage - show INFO level and above in console
    >>> setup_logging()
    >>>
    >>> # Enable DEBUG level and log to file
    >>> setup_logging(level="DEBUG", log_file="soundscapy.log")
    >>>
    >>> # Use detailed format for debugging
    >>> setup_logging(level="DEBUG", format_level="detailed")

    """
    # Enable soundscapy logging (disabled by default in __init__.py)
    logger.enable("soundscapy")

    # Remove default handlers
    logger.remove()

    # Format configurations
    formats = {
        "basic": "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
        "detailed": (
            "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
            "{name}:{function}:{line} | {message}"
        ),
        "developer": (
            "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
            "{name}:{function}:{line} | {message}\n{exception}"
        ),
    }

    # Use the appropriate format
    if format_level not in formats:
        logger.warning(f"Unknown format_level '{format_level}'. Using 'basic' instead.")
        format_level = "basic"

    log_format = formats[format_level]

    # Configure console handler
    logger.add(
        sys.stderr,
        format=log_format,
        level=level,
        colorize=True,
        enqueue=True,
    )

    # Add file handler if specified
    if log_file:
        logger.add(
            log_file,
            format=log_format,
            level="DEBUG",  # Always log everything to file
            rotation="1 MB",
            compression="zip",
            enqueue=True,
        )

    logger.debug(f"Soundscapy logging configured - console:{level}, file:{log_file}")


def enable_debug() -> None:
    """
    Quickly enable DEBUG level logging to console.

    This is a convenience function for debugging during interactive sessions.

    Examples
    --------
    >>> from soundscapy import enable_debug
    >>> enable_debug()
    >>> # Now all debug messages will be shown

    """
    setup_logging(level="DEBUG", format_level="detailed")
    logger.info("Debug logging enabled")


def disable_logging() -> None:
    """
    Disable all Soundscapy logging.

    Examples
    --------
    >>> from soundscapy import disable_logging
    >>> disable_logging()
    >>> # No more logging messages will be shown

    """
    # First remove all handlers to ensure no output
    logger.remove()
    # Then disable the soundscapy namespace
    logger.disable("soundscapy")
    # Add a handler with an impossibly high level to ensure nothing is logged
    logger.add(sys.stderr, level=100)  # Level 100 is higher than any standard level


def get_logger() -> loguru.Logger:
    """
    Get the Soundscapy logger instance.

    Returns the loguru logger configured for Soundscapy. This is mainly for
    advanced users who want to configure logging themselves.

    Returns
    -------
    logger : loguru.logger
        The loguru logger instance

    Examples
    --------
    >>> from soundscapy import get_logger
    >>> logger = get_logger()
    >>> logger.debug("Custom debug message")

    """
    return logger


def is_notebook() -> bool:
    """
    Check if code is running in Jupyter notebook.

    Returns
    -------
    bool
        True if running in a Jupyter notebook, False otherwise

    """
    try:
        from IPython.core.getipython import get_ipython

        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":  # Jupyter notebook/lab
            return True
        if shell == "TerminalInteractiveShell":  # IPython
            return False
        return False  # noqa: TRY300
    except (NameError, ImportError):
        return False
