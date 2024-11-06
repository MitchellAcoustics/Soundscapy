"""
Logging configuration using loguru.
Provides functions to configure the logger based on environment variables.
"""

import sys

from loguru import logger


class LogFormatter:
    """Unified formatter for both console and file output."""

    CONSOLE_FORMAT = (
        "<green>{time:HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
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
    """Configure logging with optional file output.

    Args:
        console_level: Logging level for console output
        log_file: Optional path to log file
    """
    try:
        logger.enable("soundscapy")
        # Remove all existing handlers
        logger.remove()

        # Configure console handler with custom formatter
        console_formatter = LogFormatter("console")
        logger.add(
            sys.stderr,
            format=console_formatter.format,
            level=console_level,
            colorize=True,
            enqueue=True,
            catch=True,
            backtrace=True,
            diagnose=True,
        )

        # Add file handler if specified
        if log_file:
            file_formatter = LogFormatter("file")
            logger.add(
                log_file,
                format=file_formatter.format,
                level="DEBUG",
                rotation="1 MB",
                compression="zip",
                enqueue=True,
                catch=True,
                backtrace=True,
            )

        logger.debug(f"Logging configured - console:{console_level}, file:{log_file}")
    except Exception as e:
        print(f"Failed to setup logging: {e}")
        raise


def is_notebook() -> bool:
    """Check if code is running in Jupyter notebook."""

    try:
        from IPython import get_ipython

        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":  # Jupyter notebook/lab
            return True
        elif shell == "TerminalInteractiveShell":  # IPython
            return False
        else:
            return False
    except NameError:
        return False
