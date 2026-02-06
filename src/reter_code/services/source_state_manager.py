"""
Source State Manager

Manages the unified source state JSON file for tracking all loaded files.
This is the single source of truth for what's loaded in RETER and RAG.

Key optimizations:
1. mtime-first checking: Only compute MD5 when mtime/size changes
2. Single JSON file: Replaces both RETER source queries and RAG's separate JSON
3. Pre-cached gitignore: Patterns loaded once and cached

State file format (.default.sources.json):
{
    "version": "2.0",
    "updated_at": "2024-01-15T10:30:00Z",
    "gitignore_hash": "abc123...",
    "files": {
        "src/foo.py": {
            "md5": "abc123def...",
            "mtime": 1705312200.0,
            "size": 1234,
            "in_reter": true,
            "in_rag": true,
            "reter_source_id": "abc123def...|src/foo.py"
        }
    },
    "gitignore_files": {".gitignore": "hash1...", "src/.gitignore": "hash2..."}
}
"""

import hashlib
import json
import sys
import time
from datetime import datetime, UTC
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set, NamedTuple
from dataclasses import dataclass, field

from ..reter_wrapper import debug_log
from .utils import matches_pattern


@dataclass
class FileInfo:
    """
    Information about a single tracked file.

    ::: This is-in-layer Utility-Layer.
    ::: This is a value-object.
    """
    rel_path: str
    abs_path: str
    md5: str
    mtime: float
    size: int
    in_reter: bool = False
    in_rag: bool = False
    reter_source_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "md5": self.md5,
            "mtime": self.mtime,
            "size": self.size,
            "in_reter": self.in_reter,
            "in_rag": self.in_rag,
            "reter_source_id": self.reter_source_id,
        }

    @classmethod
    def from_dict(cls, rel_path: str, abs_path: str, data: Dict[str, Any]) -> "FileInfo":
        return cls(
            rel_path=rel_path,
            abs_path=abs_path,
            md5=data.get("md5", ""),
            mtime=data.get("mtime", 0.0),
            size=data.get("size", 0),
            in_reter=data.get("in_reter", False),
            in_rag=data.get("in_rag", False),
            reter_source_id=data.get("reter_source_id"),
        )


@dataclass
class SyncChanges:
    """
    Changes detected during sync.

    ::: This is-in-layer Utility-Layer.
    ::: This is a value-object.
    """
    to_add: List[FileInfo] = field(default_factory=list)
    to_modify: List[Tuple[FileInfo, FileInfo]] = field(default_factory=list)  # (new_info, old_info)
    to_delete: List[FileInfo] = field(default_factory=list)

    @property
    def has_changes(self) -> bool:
        return bool(self.to_add or self.to_modify or self.to_delete)

    @property
    def total_count(self) -> int:
        return len(self.to_add) + len(self.to_modify) + len(self.to_delete)


class SourceStateManager:
    """
    Manages the unified source state JSON file.

    ::: This is-in-layer Service-Layer.
    ::: This is a manager.
    ::: This is-in-process Main-Process.
    ::: This is stateful.

    This is the single source of truth for what's loaded in RETER and RAG.
    """

    VERSION = "2.0"

    # Supported file extensions
    PYTHON_EXTENSIONS = {".py"}
    JAVASCRIPT_EXTENSIONS = {".js", ".mjs", ".jsx", ".ts", ".tsx"}
    HTML_EXTENSIONS = {".html", ".htm"}
    CSHARP_EXTENSIONS = {".cs"}
    CPP_EXTENSIONS = {".cpp", ".cc", ".cxx", ".c++", ".hpp", ".hh", ".hxx", ".h++", ".h"}
    MARKDOWN_EXTENSIONS = {".md", ".markdown"}
    ALL_CODE_EXTENSIONS = (
        PYTHON_EXTENSIONS | JAVASCRIPT_EXTENSIONS | HTML_EXTENSIONS |
        CSHARP_EXTENSIONS | CPP_EXTENSIONS
    )

    def __init__(self, state_file_path: Path, project_root: Path):
        """
        Initialize the source state manager.

        Args:
            state_file_path: Path to the .default.sources.json file
            project_root: Project root directory
        """
        self._state_path = state_file_path
        self._project_root = project_root
        self._state: Dict[str, Any] = self._empty_state()
        self._loaded = False

        # Cached gitignore patterns (loaded once)
        self._gitignore_patterns: List[Any] = []  # GitignorePattern objects
        self._gitignore_hash: str = ""
        self._gitignore_files: Dict[str, str] = {}

    def _empty_state(self) -> Dict[str, Any]:
        """Return an empty state structure."""
        return {
            "version": self.VERSION,
            "updated_at": None,
            "gitignore_hash": "",
            "files": {},
            "gitignore_files": {},
        }

    def load(self) -> bool:
        """
        Load state from JSON file.

        Returns:
            True if state was loaded from file, False if starting fresh
        """
        if self._loaded:
            return True

        if self._state_path.exists():
            try:
                with open(self._state_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Check version compatibility
                if data.get("version", "1.0") != self.VERSION:
                    self._state = self._empty_state()
                else:
                    self._state = data

                self._loaded = True
                return True
            except Exception:
                self._state = self._empty_state()

        self._loaded = True
        return False

    def save(self) -> None:
        """Save state to JSON file."""
        self._state["updated_at"] = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        self._state["gitignore_hash"] = self._gitignore_hash
        self._state["gitignore_files"] = self._gitignore_files

        # Ensure directory exists
        self._state_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self._state_path, 'w', encoding='utf-8') as f:
            json.dump(self._state, f, indent=2)

    def get_file(self, rel_path: str) -> Optional[FileInfo]:
        """Get cached file info by relative path."""
        data = self._state.get("files", {}).get(rel_path)
        if data is None:
            return None

        abs_path = str(self._project_root / rel_path)
        return FileInfo.from_dict(rel_path, abs_path, data)

    def set_file(self, file_info: FileInfo) -> None:
        """Set file info in state."""
        if "files" not in self._state:
            self._state["files"] = {}
        self._state["files"][file_info.rel_path] = file_info.to_dict()

    def remove_file(self, rel_path: str) -> None:
        """Remove file from state."""
        if "files" in self._state and rel_path in self._state["files"]:
            del self._state["files"][rel_path]

    def get_all_files(self) -> Dict[str, FileInfo]:
        """Get all tracked files."""
        result = {}
        for rel_path, data in self._state.get("files", {}).items():
            abs_path = str(self._project_root / rel_path)
            result[rel_path] = FileInfo.from_dict(rel_path, abs_path, data)
        return result

    def set_gitignore_patterns(
        self,
        patterns: List[Any],
        gitignore_hash: str,
        gitignore_files: Dict[str, str]
    ) -> None:
        """
        Set the cached gitignore patterns.

        Args:
            patterns: List of GitignorePattern objects
            gitignore_hash: Hash of all gitignore files combined
            gitignore_files: Dict mapping gitignore path -> content hash
        """
        self._gitignore_patterns = patterns
        self._gitignore_hash = gitignore_hash
        self._gitignore_files = gitignore_files

    def gitignore_changed(self) -> bool:
        """Check if gitignore files have changed since last save."""
        stored_hash = self._state.get("gitignore_hash", "")
        return stored_hash != self._gitignore_hash

    def quick_check_file(self, file_path: Path) -> Tuple[str, Optional[FileInfo]]:
        """
        Quick check if a file has changed using mtime-first strategy.

        Returns:
            Tuple of (status, new_file_info) where status is:
            - "unchanged": File hasn't changed
            - "modified": File content changed
            - "new": File is new
            new_file_info is None for "unchanged", populated otherwise
        """
        rel_path = str(file_path.relative_to(self._project_root)).replace('\\', '/')

        try:
            stat = file_path.stat()
            current_mtime = stat.st_mtime
            current_size = stat.st_size
        except OSError:
            return "error", None

        cached = self.get_file(rel_path)

        # If not in cache, it's new
        if cached is None:
            md5 = self._compute_md5(file_path)
            new_info = FileInfo(
                rel_path=rel_path,
                abs_path=str(file_path),
                md5=md5,
                mtime=current_mtime,
                size=current_size,
            )
            return "new", new_info

        # If mtime and size unchanged, assume unchanged (skip MD5)
        if cached.mtime == current_mtime and cached.size == current_size:
            return "unchanged", None

        # mtime or size changed - compute MD5 to confirm
        md5 = self._compute_md5(file_path)

        if md5 == cached.md5:
            # Content same despite mtime change - update mtime in cache
            cached.mtime = current_mtime
            cached.size = current_size
            self.set_file(cached)
            return "unchanged", None

        # Content actually changed
        new_info = FileInfo(
            rel_path=rel_path,
            abs_path=str(file_path),
            md5=md5,
            mtime=current_mtime,
            size=current_size,
        )
        return "modified", new_info

    def _compute_md5(self, file_path: Path) -> str:
        """Compute MD5 hash of file contents."""
        try:
            content = file_path.read_text(encoding='utf-8')
            return hashlib.md5(content.encode('utf-8')).hexdigest()
        except Exception:
            return ""

    def scan_and_diff(
        self,
        include_patterns: List[str],
        exclude_patterns: List[str],
        is_excluded_func: callable
    ) -> SyncChanges:
        """
        Scan filesystem and compute differences from cached state.

        This uses mtime-first checking for maximum performance.

        Args:
            include_patterns: Glob patterns to include (empty = all)
            exclude_patterns: Glob patterns to exclude
            is_excluded_func: Function to check if path is excluded (gitignore + patterns)

        Returns:
            SyncChanges with files to add, modify, or delete
        """
        start = time.time()
        changes = SyncChanges()
        seen_files: Set[str] = set()

        # Scan filesystem
        js_ts_excluded_count = 0
        js_ts_found_count = 0
        for ext in self.ALL_CODE_EXTENSIONS:
            pattern = f"**/*{ext}"
            for file_path in self._project_root.glob(pattern):
                rel_path = str(file_path.relative_to(self._project_root)).replace('\\', '/')
                is_js_ts = ext in ('.js', '.ts', '.jsx', '.tsx', '.mjs')

                # Check include patterns first (if set, file MUST match at least one)
                if include_patterns:
                    matches_include = False
                    for incl_pattern in include_patterns:
                        if self._matches_pattern(rel_path, incl_pattern):
                            matches_include = True
                            break
                    if not matches_include:
                        if is_js_ts:
                            js_ts_excluded_count += 1
                        continue

                # Skip excluded files
                if is_excluded_func(file_path.relative_to(self._project_root)):
                    if is_js_ts:
                        js_ts_excluded_count += 1
                    continue

                # Skip common excluded directories
                if self._is_common_excluded(rel_path):
                    if is_js_ts:
                        js_ts_excluded_count += 1
                    continue

                if is_js_ts:
                    js_ts_found_count += 1

                seen_files.add(rel_path)

                # Quick check using mtime-first strategy
                status, new_info = self.quick_check_file(file_path)

                if status == "new":
                    changes.to_add.append(new_info)
                elif status == "modified":
                    old_info = self.get_file(rel_path)
                    changes.to_modify.append((new_info, old_info))

        # Find deleted files
        for rel_path, cached in self.get_all_files().items():
            if rel_path not in seen_files:
                # Check if it's a code file (not markdown)
                if any(rel_path.endswith(ext) for ext in self.ALL_CODE_EXTENSIONS):
                    changes.to_delete.append(cached)

        elapsed = time.time() - start
        debug_log(
            f"[SourceState] scan_and_diff: +{len(changes.to_add)} ~{len(changes.to_modify)} "
            f"-{len(changes.to_delete)} in {elapsed:.2f}s"
        )
        debug_log(f"[SourceState] JS/TS files: found={js_ts_found_count}, excluded={js_ts_excluded_count}")

        return changes

    def _is_common_excluded(self, rel_path: str) -> bool:
        """Check common excluded directories."""
        excluded_dirs = [
            "node_modules/", "__pycache__/", ".git/", ".venv/", "venv/",
            "/bin/", "/obj/", "CMakeFiles/", "build/", "cmake-build-",
            ".tox/", ".pytest_cache/", ".mypy_cache/", "dist/", "eggs/"
        ]
        for excl in excluded_dirs:
            if excl in rel_path or rel_path.startswith(excl.lstrip("/")):
                return True
        return False

    def _matches_pattern(self, path: str, pattern: str) -> bool:
        """Simple glob pattern matching - delegates to shared utility."""
        return matches_pattern(path, pattern)

    def mark_in_reter(self, rel_path: str, source_id: str) -> None:
        """Mark a file as loaded in RETER."""
        file_info = self.get_file(rel_path)
        if file_info:
            file_info.in_reter = True
            file_info.reter_source_id = source_id
            self.set_file(file_info)

    def mark_in_rag(self, rel_path: str) -> None:
        """Mark a file as indexed in RAG."""
        file_info = self.get_file(rel_path)
        if file_info:
            file_info.in_rag = True
            self.set_file(file_info)

    def get_reter_sources(self) -> Dict[str, str]:
        """
        Get all files loaded in RETER as {rel_path: source_id}.

        This replaces the need to query reter.get_all_sources().
        """
        result = {}
        for rel_path, data in self._state.get("files", {}).items():
            if data.get("in_reter") and data.get("reter_source_id"):
                result[rel_path] = data["reter_source_id"]
        return result

    def get_rag_sources(self) -> Dict[str, str]:
        """
        Get all files indexed in RAG as {rel_path: md5}.

        This replaces the separate .default.rag_files.json.
        """
        result = {}
        for rel_path, data in self._state.get("files", {}).items():
            if data.get("in_rag"):
                result[rel_path] = data.get("md5", "")
        return result

    def clear(self) -> None:
        """Clear all state (for full rebuild)."""
        self._state = self._empty_state()
        self._loaded = True

    def build_from_reter(self, all_sources: List[str]) -> None:
        """
        Build initial state from RETER sources (migration from old format).

        Args:
            all_sources: List of source IDs from reter.get_all_sources()
        """
        debug_log(f"[SourceState] Building initial state from {len(all_sources)} RETER sources")

        for source_id in all_sources:
            if "|" not in source_id:
                continue

            md5_hash, rel_path = source_id.split("|", 1)
            rel_path_normalized = rel_path.replace('\\', '/')

            # Check if it's a code file
            if not any(rel_path_normalized.endswith(ext) for ext in self.ALL_CODE_EXTENSIONS):
                continue

            abs_path = self._project_root / rel_path_normalized
            try:
                stat = abs_path.stat()
                mtime = stat.st_mtime
                size = stat.st_size
            except OSError:
                mtime = 0.0
                size = 0

            file_info = FileInfo(
                rel_path=rel_path_normalized,
                abs_path=str(abs_path),
                md5=md5_hash,
                mtime=mtime,
                size=size,
                in_reter=True,
                in_rag=False,
                reter_source_id=source_id,
            )
            self.set_file(file_info)

        debug_log(f"[SourceState] Built state with {len(self._state.get('files', {}))} files")
