"""
Utility functions for services

Common utilities shared across service classes.
"""

from fnmatch import fnmatch
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


def matches_pattern(path: str, pattern: str) -> bool:
    """
    Simple glob pattern matching.

    Supports:
    - * (any characters within a path component)
    - ** (any characters across path components)
    - ? (single character)

    Args:
        path: Path to check
        pattern: Glob pattern

    Returns:
        True if path matches pattern
    """
    # Convert ** to match across directories
    if "**" in pattern:
        # For patterns like "test/**" or "**/test/*"
        parts = pattern.split("**")
        if len(parts) == 2:
            prefix, suffix = parts
            prefix = prefix.rstrip("/")
            suffix = suffix.lstrip("/")

            if prefix and not path.startswith(prefix):
                return False
            if suffix and not fnmatch(path, "*" + suffix):
                return False
            return True
        elif len(parts) >= 3:
            # Handle patterns with multiple ** like "**/vcpkg*/**"
            # Check if all non-empty parts match somewhere in the path
            prefix = parts[0].rstrip("/")
            suffix = parts[-1].lstrip("/")
            middle_parts = [p.strip("/") for p in parts[1:-1] if p.strip("/")]

            # Prefix must match start (if non-empty)
            if prefix and not path.startswith(prefix):
                return False

            # Suffix must match end (if non-empty)
            if suffix and not path.endswith(suffix) and not fnmatch(path.split("/")[-1], suffix):
                return False

            # Middle parts must appear somewhere in the path
            for middle in middle_parts:
                # Check if middle pattern matches any path component
                matched = False
                for component in path.split("/"):
                    if fnmatch(component, middle):
                        matched = True
                        break
                if not matched:
                    return False

            return True

    # Regular fnmatch
    return fnmatch(path, pattern)
