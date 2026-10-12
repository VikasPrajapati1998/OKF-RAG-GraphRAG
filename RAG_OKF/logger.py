"""Centralized logging for the whole project.

Import ``logger`` from this module everywhere; never configure loggers
elsewhere. Log files are written to ``logs/YYYY-MM-DD/YYYY-MM-DD_HH-MM-SS.log``
(one new file per run, never overwritten). The console only shows warnings and
above so the chat interface stays clean.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

from exception import ConfigurationError

BASE_DIR: Path = Path(__file__).resolve().parent
LOG_ROOT: Path = BASE_DIR / "logs"
LOGGER_NAME = "chatbot"
_FORMAT = "%(asctime)s | %(levelname)-8s | %(module)s:%(funcName)s:%(lineno)d | %(message)s"


def _build_logger() -> logging.Logger:
    """Create and configure the single project logger.

    Returns:
        The configured logger.

    Raises:
        ConfigurationError: If the log directory or file cannot be created.
    """
    project_logger = logging.getLogger(LOGGER_NAME)
    if project_logger.handlers:
        return project_logger

    now = datetime.now()
    day_dir = LOG_ROOT / now.strftime("%Y-%m-%d")
    log_file = day_dir / f"{now.strftime('%Y-%m-%d_%H-%M-%S')}.log"
    try:
        day_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    except OSError as exc:
        raise ConfigurationError(f"Cannot create log file in {day_dir}: {exc}") from exc

    formatter = logging.Formatter(_FORMAT)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.ERROR)
    console_handler.setFormatter(formatter)

    project_logger.setLevel(logging.DEBUG)
    project_logger.addHandler(file_handler)
    project_logger.addHandler(console_handler)
    project_logger.propagate = False
    project_logger.info("Logging started: %s", log_file)
    return project_logger


logger: logging.Logger = _build_logger()
