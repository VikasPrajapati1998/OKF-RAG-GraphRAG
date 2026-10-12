"""Print the directory tree of ROOT_PATH, skipping excluded names."""

import os
from pathlib import Path

from config import Config, load_config
from exception import DirError, DirectoryScanError
from logger import logger


def build_tree(directory: Path, exclude_names: frozenset[str], prefix: str = "") -> list[str]:
    """Recursively build tree lines for a directory.

    Args:
        directory: Directory to list.
        exclude_names: Names to skip.
        prefix: Indentation prefix for the current depth.

    Returns:
        Formatted lines, directories first, then files, each sorted by name.

    Raises:
        DirectoryScanError: If the directory cannot be read.
    """
    try:
        with os.scandir(directory) as it:
            entries = [e for e in it if e.name not in exclude_names]
    except OSError as exc:
        raise DirectoryScanError(f"Cannot scan {directory}: {exc}") from exc

    entries.sort(key=lambda e: (not e.is_dir(follow_symlinks=False), e.name.lower()))
    lines: list[str] = []
    for index, entry in enumerate(entries):
        last = index == len(entries) - 1
        lines.append(f"{prefix}{'└── ' if last else '├── '}{entry.name}")
        if entry.is_dir(follow_symlinks=False):
            child_prefix = prefix + ("    " if last else "│   ")
            lines.extend(build_tree(Path(entry.path), exclude_names, child_prefix))
    return lines


def render(config: Config) -> str:
    """Render the full tree for the configured root.

    Args:
        config: Application settings.

    Returns:
        The tree as a single string.
    """
    lines = [config.root_path.name or str(config.root_path)]
    lines.extend(build_tree(config.root_path, config.exclude_names))
    return "\n".join(lines)


def main() -> None:
    """Load configuration and print the directory tree."""
    try:
        config = load_config()
        logger.info("Scanning %s (excluding %s)", config.root_path, sorted(config.exclude_names))
        print(render(config))  # CLI output, not application logging
    except DirError:
        logger.exception("Directory listing failed")
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
