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

# Global variable for log level
GLOBAL_LOG_LEVEL = "WARNING"


def setup_logger():
    global GLOBAL_LOG_LEVEL
    # Remove all existing handlers
    logger.remove()

    # Get log level from environment variable or use global variable
    log_level = os.getenv("SOUNDSCAPY_LOG_LEVEL", GLOBAL_LOG_LEVEL).upper()
    GLOBAL_LOG_LEVEL = log_level

    # Add a handler for stderr with the specified log level
    logger.add(sys.stderr, level=log_level, format="{time} {level} {message}")

    # If a log file is specified, add a handler for it
    log_file = os.getenv("SOUNDSCAPY_LOG_FILE")
    if log_file:
        log_path = Path(log_file).expanduser().resolve()
        logger.add(log_path, level=log_level, rotation="10 MB")

    logger.info(f"Logger initialized with level {log_level}")
    if log_file:
        logger.info(f"Logging to file: {log_path}")


def get_logger():
    return logger


def set_log_level(level: str):
    global GLOBAL_LOG_LEVEL
    GLOBAL_LOG_LEVEL = level.upper()
    logger.remove()
    logger.add(sys.stderr, level=GLOBAL_LOG_LEVEL)
    logger.info(f"Log level set to {GLOBAL_LOG_LEVEL}")


# Set up the logger when this module is imported
setup_logger()
