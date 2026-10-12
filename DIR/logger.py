"""Centralized logger. Logs go to logs/YYYY-MM-DD/YYYY-MM-DD_HH-MM-SS.log."""

import logging
from datetime import datetime
from pathlib import Path

LOG_ROOT: Path = Path(__file__).resolve().parent / "logs"
LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(module)s:%(lineno)d | %(message)s"


def _build_logger() -> logging.Logger:
    """Create the project logger with console and timestamped file handlers.

    Returns:
        The configured logger.
    """
    now = datetime.now()
    log_dir = LOG_ROOT / now.strftime("%Y-%m-%d")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{now.strftime('%Y-%m-%d_%H-%M-%S')}.log"

    project_logger = logging.getLogger("dir")
    project_logger.setLevel(logging.DEBUG)
    project_logger.propagate = False

    formatter = logging.Formatter(LOG_FORMAT)
    for handler in (logging.FileHandler(log_file, mode="x", encoding="utf-8"), logging.StreamHandler()):
        handler.setFormatter(formatter)
        project_logger.addHandler(handler)
    return project_logger


logger: logging.Logger = _build_logger()
