import logging
import os
from typing import Optional


def setup_logger(name: Optional[str] = None) -> logging.Logger:
    """Simple logger setup used across the project.

    This is a lightweight stand-in for the more advanced logging described
    in the README. It configures a module-level logger with a console handler.
    """
    logger_name = name or __name__
    logger = logging.getLogger(logger_name)

    if logger.handlers:
        # Logger already configured
        return logger

    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    logger.setLevel(level)

    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger.propagate = False
    return logger
