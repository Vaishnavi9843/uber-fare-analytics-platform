"""logger.py

Application-wide logger configuration.

This module centralizes logger creation so other modules can do:

    from src.logger import logger

and then use logger.info / warning / error.
"""

from __future__ import annotations

import logging


def _configure_logger() -> logging.Logger:
    logger_ = logging.getLogger("uber_fare_prediction")

    # Avoid duplicate handlers if this module is imported multiple times.
    if logger_.handlers:
        return logger_

    logger_.setLevel(logging.INFO)

    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)

    logger_.addHandler(handler)
    logger_.propagate = False
    return logger_


logger = _configure_logger()

