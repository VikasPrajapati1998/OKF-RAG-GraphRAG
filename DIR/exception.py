"""Project-wide exception hierarchy."""


class DirError(Exception):
    """Base class for all errors raised by this project."""


class ConfigurationError(DirError):
    """Raised when configuration is missing or invalid."""


class DirectoryScanError(DirError):
    """Raised when a directory cannot be scanned."""
