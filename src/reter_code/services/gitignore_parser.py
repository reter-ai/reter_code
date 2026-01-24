"""
Gitignore Parser Service

Parses .gitignore files and provides methods to check if paths should be ignored.
Supports:
- Standard .gitignore patterns (*, **, ?, [])
- Negation patterns (!)
- Directory-only patterns (trailing /)
- Comments (#)
- Nested .gitignore files
"""

import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from fnmatch import fnmatch


class GitignorePattern:
    """Represents a single gitignore pattern."""

    def __init__(self, pattern: str, base_dir: Path, negation: bool = False):
        """
        Initialize a gitignore pattern.

        Args:
            pattern: The pattern string (without leading !)
            base_dir: Directory containing the .gitignore file
            negation: Whether this is a negation pattern
        """
        self.original = pattern
        self.base_dir = base_dir
        self.negation = negation
        self.directory_only = pattern.endswith('/')

        # Remove trailing slash for matching
        if self.directory_only:
            pattern = pattern[:-1]

        # Determine if pattern is anchored (starts with / or contains / except at end)
        self.anchored = pattern.startswith('/') or '/' in pattern[:-1] if len(pattern) > 1 else pattern.startswith('/')

        # Remove leading slash
        if pattern.startswith('/'):
            pattern = pattern[1:]

        self.pattern = pattern

        # Convert gitignore pattern to regex
        self._regex = self._compile_pattern(pattern)

    def _compile_pattern(self, pattern: str) -> re.Pattern:
        """Convert gitignore pattern to regex."""
        # Escape special regex chars except gitignore wildcards
        result = ""
        i = 0
        while i < len(pattern):
            c = pattern[i]
            if c == '*':
                if i + 1 < len(pattern) and pattern[i + 1] == '*':
                    # ** matches any path
                    if i + 2 < len(pattern) and pattern[i + 2] == '/':
                        # **/ matches any directory path
                        result += "(?:.*/)?)"
                        i += 3
                        continue
                    else:
                        # ** at end matches everything
                        result += ".*"
                        i += 2
                        continue
                else:
                    # * matches anything except /
                    result += "[^/]*"
            elif c == '?':
                result += "[^/]"
            elif c == '[':
                # Character class - find closing ]
                j = i + 1
                if j < len(pattern) and pattern[j] in '!^':
                    j += 1
                if j < len(pattern) and pattern[j] == ']':
                    j += 1
                while j < len(pattern) and pattern[j] != ']':
                    j += 1
                result += pattern[i:j + 1]
                i = j
            elif c in '.^$+{}|()\\':
                result += '\\' + c
            else:
                result += c
            i += 1

        # Anchor pattern
        if self.anchored:
            result = "^" + result
        else:
            result = "(?:^|/)" + result

        # Match to end or directory
        result += "(?:/.*)?$"

        return re.compile(result)

    def matches(self, path: str, is_dir: bool = False, project_root: Path = None) -> bool:
        """
        Check if path matches this pattern.

        Patterns from nested .gitignore files only apply to files within
        their directory subtree.

        Args:
            path: Relative path to check (forward slashes, relative to project root)
            is_dir: Whether the path is a directory
            project_root: Project root directory (required for scoping nested patterns)

        Returns:
            True if path matches pattern
        """
        # Directory-only patterns only match directories
        if self.directory_only and not is_dir:
            return False

        # Scope nested .gitignore patterns to their base directory
        # Patterns from a .gitignore only apply to files in that directory or subdirectories
        if project_root is not None:
            try:
                base_rel = self.base_dir.relative_to(project_root)
                base_prefix = str(base_rel).replace('\\', '/')
                if base_prefix and base_prefix != '.':
                    # This pattern is from a nested .gitignore
                    # Check if path is under the base directory
                    if not path.startswith(base_prefix + '/') and path != base_prefix:
                        return False
                    # Strip base prefix for matching
                    path = path[len(base_prefix) + 1:] if path.startswith(base_prefix + '/') else ''
            except ValueError:
                # base_dir is not under project_root, shouldn't happen
                pass

        return bool(self._regex.search(path))


class GitignoreParser:
    """
    Parses and caches .gitignore patterns for a project.

    Usage:
        parser = GitignoreParser(project_root)
        if parser.is_ignored(file_path):
            # File should be ignored
    """

    def __init__(self, project_root: Path, progress_callback: Optional[Any] = None):
        """
        Initialize gitignore parser.

        Args:
            project_root: Root directory of the project
            progress_callback: Optional progress callback with update_gitignore_progress(path) method
        """
        self.project_root = project_root
        self._patterns: List[GitignorePattern] = []
        self._loaded_gitignores: Set[Path] = set()
        self._progress_callback = progress_callback

        # Load root .gitignore
        self._load_gitignore(project_root)

    def _load_gitignore(self, directory: Path) -> None:
        """
        Load .gitignore file from a directory.

        Args:
            directory: Directory to load .gitignore from
        """
        gitignore_path = directory / ".gitignore"

        if gitignore_path in self._loaded_gitignores:
            return

        if not gitignore_path.exists():
            return

        self._loaded_gitignores.add(gitignore_path)

        # Report progress when nested .gitignore files are discovered
        try:
            rel_dir = directory.relative_to(self.project_root)
            rel_str = str(rel_dir)
            if rel_str != '.':
                if self._progress_callback and hasattr(self._progress_callback, 'update_gitignore_progress'):
                    self._progress_callback.update_gitignore_progress(rel_str)
        except ValueError:
            pass

        try:
            with open(gitignore_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line = line.rstrip('\n\r')

                    # Skip empty lines and comments
                    if not line or line.startswith('#'):
                        continue

                    # Handle trailing spaces (escaped with \)
                    if line.endswith('\\ '):
                        line = line[:-2] + ' '
                    else:
                        line = line.rstrip()

                    if not line:
                        continue

                    # Check for negation
                    negation = line.startswith('!')
                    if negation:
                        line = line[1:]

                    if not line:
                        continue

                    pattern = GitignorePattern(line, directory, negation)
                    self._patterns.append(pattern)
        except Exception:
            # Silently ignore parse errors
            pass

    def _ensure_gitignore_loaded(self, file_path: Path) -> None:
        """
        Ensure .gitignore files are loaded for all parent directories.

        Args:
            file_path: Path to check
        """
        # Load .gitignore from each parent directory
        try:
            rel_path = file_path.relative_to(self.project_root)
        except ValueError:
            return

        current = self.project_root
        for part in rel_path.parts[:-1]:  # Skip the file itself
            current = current / part
            self._load_gitignore(current)

    def is_ignored(self, file_path: Path, is_dir: Optional[bool] = None) -> bool:
        """
        Check if a path should be ignored based on .gitignore rules.

        Args:
            file_path: Path to check (can be absolute or relative to project_root)
            is_dir: Whether the path is a directory (auto-detected if None)

        Returns:
            True if path should be ignored
        """
        # Ensure path is relative to project root
        try:
            if file_path.is_absolute():
                rel_path = file_path.relative_to(self.project_root)
            else:
                rel_path = file_path
        except ValueError:
            # Path is not under project root
            return False

        # Convert to forward slashes
        path_str = str(rel_path).replace('\\', '/')

        # Load any nested .gitignore files
        if file_path.is_absolute():
            self._ensure_gitignore_loaded(file_path)
        else:
            self._ensure_gitignore_loaded(self.project_root / file_path)

        # Determine if directory
        if is_dir is None:
            full_path = self.project_root / rel_path if not file_path.is_absolute() else file_path
            is_dir = full_path.is_dir()

        # Check patterns in order (last match wins)
        ignored = False
        for pattern in self._patterns:
            if pattern.matches(path_str, is_dir, self.project_root):
                ignored = not pattern.negation

        return ignored

    def get_pattern_count(self) -> int:
        """Get the number of loaded patterns."""
        return len(self._patterns)

    def get_loaded_gitignores(self) -> List[Path]:
        """Get list of loaded .gitignore files."""
        return list(self._loaded_gitignores)

    def load_all_gitignores(self) -> None:
        """
        Pre-load all .gitignore files in the project tree.

        This is more efficient than lazy loading during file scans,
        as it loads all patterns upfront in a single pass.
        """
        import hashlib

        # Find all .gitignore files in project
        for gitignore_path in self.project_root.rglob(".gitignore"):
            try:
                directory = gitignore_path.parent
                self._load_gitignore(directory)
            except Exception:
                pass  # Skip unreadable gitignore files

        # Compute combined hash of all gitignore contents
        self._gitignore_files_hashes: Dict[str, str] = {}
        for gitignore_path in self._loaded_gitignores:
            try:
                content = gitignore_path.read_text(encoding='utf-8', errors='ignore')
                file_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
                rel_path = str(gitignore_path.relative_to(self.project_root))
                self._gitignore_files_hashes[rel_path] = file_hash
            except Exception:
                pass

        # Compute combined hash
        import json
        combined = json.dumps(self._gitignore_files_hashes, sort_keys=True)
        self._combined_hash = hashlib.md5(combined.encode('utf-8')).hexdigest()

    def get_gitignore_hash(self) -> str:
        """
        Get combined hash of all gitignore files.

        Use this to detect when gitignore patterns have changed.
        Must call load_all_gitignores() first.

        Returns:
            MD5 hash of combined gitignore contents, or empty string if not loaded
        """
        return getattr(self, '_combined_hash', '')

    def get_gitignore_files_hashes(self) -> Dict[str, str]:
        """
        Get individual hashes for each .gitignore file.

        Returns:
            Dict mapping relative gitignore path -> content hash
        """
        return getattr(self, '_gitignore_files_hashes', {})

    def is_ignored_fast(self, rel_path_str: str, is_dir: bool = False) -> bool:
        """
        Fast check if a path should be ignored (no lazy loading).

        Use this after calling load_all_gitignores() for maximum performance.
        Unlike is_ignored(), this doesn't lazily load nested gitignore files.

        Args:
            rel_path_str: Relative path string (forward slashes)
            is_dir: Whether the path is a directory

        Returns:
            True if path should be ignored
        """
        # Check patterns in order (last match wins)
        ignored = False
        for pattern in self._patterns:
            if pattern.matches(rel_path_str, is_dir, self.project_root):
                ignored = not pattern.negation

        return ignored


# Convenience function
def is_git_ignored(project_root: Path, file_path: Path) -> bool:
    """
    Quick check if a file is ignored by .gitignore.

    For bulk checks, use GitignoreParser directly for better performance.

    Args:
        project_root: Project root directory
        file_path: Path to check

    Returns:
        True if path should be ignored
    """
    parser = GitignoreParser(project_root)
    return parser.is_ignored(file_path)
