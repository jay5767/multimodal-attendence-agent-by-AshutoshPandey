"""Logging configuration for Multimodal Attendance & Engagement Agent.

Ensures structured, informative logging without logging sensitive biometric data or raw images.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

_LOGGER_INITIALIZED = False


def setup_logger(
    name: str = "attendance_agent",
    log_file: Optional[Path] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """Configures and returns a standardized logger."""
    global _LOGGER_INITIALIZED
    logger = logging.getLogger(name)

    if not _LOGGER_INITIALIZED or not logger.handlers:
        logger.setLevel(level)

        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(level)
        logger.addHandler(console_handler)

        # Optional file handler
        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(str(log_file), encoding="utf-8")
            file_handler.setFormatter(formatter)
            file_handler.setLevel(level)
            logger.addHandler(file_handler)

        logger.propagate = False
        _LOGGER_INITIALIZED = True

    return logger


# Default application-wide logger instance
logger = setup_logger()
