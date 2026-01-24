"""
Utility functions for services

Common utilities shared across service classes.
"""

from pathlib import Path


def make_path_relative(absolute_path: Path) -> str:
    """
    Convert an absolute path to be relative to the current working directory.
    Uses forward slashes for consistency across platforms.
    Falls back to absolute path if relativization fails (e.g., different drives).

    Args:
        absolute_path: Absolute Path object

    Returns:
        Relative path string with forward slashes, or absolute path if conversion fails
    """
    try:
        cwd = Path.cwd()
        relative = absolute_path.relative_to(cwd)
        # Use forward slashes for consistency across platforms
        return relative.as_posix()
    except (ValueError, RuntimeError):
        # Fall back to absolute path if relative_to fails
        # (e.g., different drives on Windows)
        return absolute_path.as_posix()
