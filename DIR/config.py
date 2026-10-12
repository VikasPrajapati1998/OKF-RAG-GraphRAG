"""Configuration management: loads settings from the .env file."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from exception import ConfigurationError

BASE_DIR: Path = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")


@dataclass(frozen=True)
class Config:
    """Application settings.

    Attributes:
        root_path: Directory whose structure is listed.
        exclude_names: File/directory names skipped during the scan.
    """

    root_path: Path
    exclude_names: frozenset[str]


def load_config() -> Config:
    """Build a Config from environment variables.

    Returns:
        The populated Config instance.

    Raises:
        ConfigurationError: If ROOT_PATH is unset or not an existing directory.
    """
    raw_root = os.getenv("ROOT_PATH", "").strip()
    if not raw_root:
        raise ConfigurationError("ROOT_PATH is not set in .env")

    root_path = Path(raw_root).expanduser()
    if not root_path.is_dir():
        raise ConfigurationError(f"ROOT_PATH is not an existing directory: {root_path}")

    raw_exclude = os.getenv("EXCLUDE_NAMES", "")
    exclude_names = frozenset(name.strip() for name in raw_exclude.split(",") if name.strip())
    return Config(root_path=root_path, exclude_names=exclude_names)
