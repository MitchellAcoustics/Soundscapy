"""
Logging configuration for the soundscapy package.

This module sets up the logging system for soundscapy using loguru.
It provides functions to configure the logger based on environment variables
and to get the configured logger.
"""

import os
import sys
from pathlib import Path

from loguru import logger


def setup_logger():
    """
    Set up the logger for soundscapy.

    This function configures the logger based on environment variables:
    - SOUNDSCAPY_LOG_LEVEL: Set the logging level (default: INFO)
    - SOUNDSCAPY_LOG_FILE: Set a file to log to (optional)

    The function removes the default handler and adds new handlers based on
    the configuration.
    """
    # Remove the default handler
    logger.remove()

    # Get log level from environment variable, default to WARNING
    log_level = os.getenv("SOUNDSCAPY_LOG_LEVEL", "WARNING").upper()

    # Add a handler for stderr
    logger.add(sys.stderr, level=log_level)

    # If a log file is specified, add a handler for it
    log_file = os.getenv("SOUNDSCAPY_LOG_FILE")
    if log_file:
        log_path = Path(log_file).expanduser().resolve()
        logger.add(log_path, level=log_level, rotation="10 MB")

    logger.info(f"Logger initialized with level {log_level}")
    if log_file:
        logger.info(f"Logging to file: {log_path}")


def get_logger():
    """
    Get the configured logger.

    Returns:
        logger: The configured loguru logger instance.
    """
    return logger


# Set up the logger when this module is imported
setup_logger()
