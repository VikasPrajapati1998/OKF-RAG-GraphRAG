"""Configuration management: reads ``.env`` / environment variables.

Secrets and tunables live in ``.env`` (see ``.env.example``); nothing is
hardcoded here except safe defaults.
"""

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

from exception import ConfigurationError
from logger import BASE_DIR, logger


def _get_int(name: str, default: int, minimum: int = 0) -> int:
    """Read an integer environment variable.

    Args:
        name: Environment variable name.
        default: Value used when the variable is unset.
        minimum: Smallest accepted value.

    Returns:
        The parsed integer.

    Raises:
        ConfigurationError: If the value is not an integer or below ``minimum``.
    """
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be an integer, got {raw!r}") from exc
    if value < minimum:
        raise ConfigurationError(f"{name} must be >= {minimum}, got {value}")
    return value


def _get_float(name: str, default: float) -> float:
    """Read a float environment variable.

    Args:
        name: Environment variable name.
        default: Value used when the variable is unset.

    Returns:
        The parsed float.

    Raises:
        ConfigurationError: If the value is not a number.
    """
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be a number, got {raw!r}") from exc


@dataclass(frozen=True)
class Settings:
    """Immutable application settings."""

    ollama_host: str
    chat_model: str
    embed_model: str
    embed_query_prefix: str
    embed_doc_prefix: str
    request_timeout: int
    temperature: float
    num_ctx: int
    max_answer_tokens: int
    chunk_size: int
    chunk_overlap: int
    top_k: int
    max_history_turns: int
    data_dir: Path
    db_path: Path


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load settings once from ``.env`` and the environment.

    Returns:
        The cached :class:`Settings` instance.

    Raises:
        ConfigurationError: If any value is invalid.
    """
    load_dotenv(BASE_DIR / ".env")
    data_dir = BASE_DIR / os.getenv("DATA_DIR", "data")
    try:
        data_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ConfigurationError(f"Cannot create data directory {data_dir}: {exc}") from exc

    chunk_size = _get_int("CHUNK_SIZE", 800, minimum=100)
    chunk_overlap = _get_int("CHUNK_OVERLAP", 100, minimum=0)
    if chunk_overlap >= chunk_size:
        raise ConfigurationError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")

    settings = Settings(
        ollama_host=os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/"),
        chat_model=os.getenv("CHAT_MODEL", "granite3.3:2b"),
        embed_model=os.getenv("EMBED_MODEL", "nomic-embed-text:v1.5"),
        embed_query_prefix=os.getenv("EMBED_QUERY_PREFIX", "search_query: "),
        embed_doc_prefix=os.getenv("EMBED_DOC_PREFIX", "search_document: "),
        request_timeout=_get_int("REQUEST_TIMEOUT", 300, minimum=1),
        temperature=_get_float("TEMPERATURE", 0.2),
        num_ctx=_get_int("NUM_CTX", 4096, minimum=512),
        max_answer_tokens=_get_int("MAX_ANSWER_TOKENS", 512, minimum=16),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        top_k=_get_int("TOP_K", 4, minimum=1),
        max_history_turns=_get_int("MAX_HISTORY_TURNS", 6, minimum=0),
        data_dir=data_dir,
        db_path=data_dir / os.getenv("DB_NAME", "store.db"),
    )
    logger.info(
        "Settings loaded: chat=%s embed=%s host=%s db=%s",
        settings.chat_model,
        settings.embed_model,
        settings.ollama_host,
        settings.db_path,
    )
    return settings
