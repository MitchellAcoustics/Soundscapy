"""
Logging configuration for the soundscapy package.

This module sets up the logging system for soundscapy using loguru.
It provides functions to configure the logger based on environment variables
and to get the configured logger.
"""

import sys
from functools import wraps

from loguru import logger

# Global variable for log level
GLOBAL_LOG_LEVEL = "DEBUG"


class LogFormatter:
    """Unified formatter for both console and file output."""

    CONSOLE_FORMAT = (
        "<green>{time:HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "{extra[padding]}<level>{message}</level>\n"
        "{exception}"
    )

    FILE_FORMAT = (
        "{time:YYYY-MM-DD HH:mm:ss} | "
        "{level: <8} | "
        "{name}:{function}:{line}{extra[padding]} | "
        "{message}\n"
        "{exception}"
    )

    def __init__(self, fmt_type: str = "console"):
        self.padding = 0
        self.fmt_type = fmt_type

    def format(self, record):
        if self.fmt_type == "console":
            if "padding" not in record["extra"]:
                record["extra"]["padding"] = ""
            return self.CONSOLE_FORMAT
        else:
            metadata_length = len(
                f"{record['name']}:{record['function']}:{record['line']}"
            )
            self.padding = max(self.padding, metadata_length)
            record["extra"]["padding"] = " " * (self.padding - metadata_length)
            return self.FILE_FORMAT


def setup_logging(console_level: str = "WARNING", log_file: str | None = None) -> None:
    """Configure logging with optional file output."""
    global console_level_setting
    console_level_setting = console_level

    logger.remove()

    console_formatter = LogFormatter("console")
    logger.add(
        sys.stderr, format=console_formatter.format, level=console_level, colorize=True
    )

    if log_file:
        file_formatter = LogFormatter("file")
        logger.add(
            log_file,
            format=file_formatter.format,
            level="DEBUG",
            rotation="1 MB",
            enqueue=True,
        )


def stage(name: str):
    """Decorator to mark and log processing stages."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create stage header
            stage_header = f" STAGE: {name} "
            padding = "═" * (40 - len(stage_header) // 2)

            # Log stage start
            with logger.contextualize(padding=""):
                logger.info(f"{padding}{stage_header}{padding}")

            # Execute function with indented logging
            with logger.contextualize(padding=""):
                result = func(*args, **kwargs)

            # Log stage completion
            with logger.contextualize(padding=""):
                logger.info(f"{'═' * (len(stage_header) + 2 * len(padding))}\n")

            return result

        return wrapper

    return decorator


def substage(name: str):
    """Decorator to mark and log processing substages."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Log substage header with consistent indentation
            with logger.contextualize(padding=""):
                logger.info(f"▶ {name}")

            # Execute function with conditional indentation based on log level
            if console_level_setting == "SUCCESS":
                result = func(*args, **kwargs)
            else:
                with logger.contextualize(padding="  "):
                    result = func(*args, **kwargs)

            return result

        return wrapper

    return decorator
