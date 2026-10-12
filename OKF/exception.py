"""Project-wide exception hierarchy.

Every custom exception inherits from :class:`ChatbotError` so callers can
handle all application failures with a single ``except ChatbotError``.
"""


class ChatbotError(Exception):
    """Base class for all chatbot errors."""


class ConfigurationError(ChatbotError):
    """Raised when configuration or environment values are missing or invalid."""


class DocumentError(ChatbotError):
    """Raised when a document cannot be found, read, parsed or is empty."""


class EmbeddingError(ChatbotError):
    """Raised when embeddings cannot be generated."""


class LLMError(ChatbotError):
    """Raised when the language model request fails or returns bad output."""


class DatabaseError(ChatbotError):
    """Raised when a SQLite operation fails."""


class RetrievalError(ChatbotError):
    """Raised when context retrieval fails."""
