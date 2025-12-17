"""
Default Instance Manager Service

Manages the special "default" RETER instance that auto-syncs with project files.

Configuration (in order of precedence):
1. RETER_PROJECT_ROOT env var: Explicit project directory
2. Auto-detection from CWD: If current working directory looks like a Python project
   (has pyproject.toml, setup.py, .git, or .py files), it's used automatically.
   Claude Code sets CWD to the project root when launching MCP servers.

Optional env vars:
- RETER_PROJECT_INCLUDE: Comma-separated glob patterns to include (if set, only matching files are loaded)
- RETER_PROJECT_EXCLUDE: Comma-separated glob patterns to exclude

File Exclusion (in order of checking):
1. .gitignore patterns: If .gitignore exists in project root, files matching patterns are excluded
2. RETER_PROJECT_EXCLUDE: Additional explicit exclude patterns

RAG Integration:
- When RAG is enabled, this manager also syncs markdown files for semantic search
- RAG indexing is triggered after RETER sync completes
"""

import sys
import os
import hashlib
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple, Any, TYPE_CHECKING
from ..reter_wrapper import ReterWrapper, generate_source_id, debug_log
from .gitignore_parser import GitignoreParser

if TYPE_CHECKING:
    from .state_persistence import StatePersistenceService
    from .rag_index_manager import RAGIndexManager


class DefaultInstanceManager:
    """
    Manages the "default" RETER instance with automatic file synchronization.

    The default instance:
    - Always appears in list_all_instances()
    - Lazy-loads on first access
    - Auto-syncs with project files based on MD5 checksums
    - Configured via RETER_PROJECT_ROOT env var, OR auto-detected from CWD
    - Optional: RETER_PROJECT_INCLUDE and RETER_PROJECT_EXCLUDE for filtering
    """

    INSTANCE_NAME = "default"

    def __init__(self, persistence: "StatePersistenceService"):
        """
        Initialize the default instance manager.

        Args:
            persistence: StatePersistenceService for snapshot management
        """
        self._persistence = persistence
        self._project_root: Optional[Path] = None
        self._include_patterns: List[str] = []
        self._exclude_patterns: List[str] = []
        self._initialized = False
        self._syncing = False  # Re-entrancy guard to prevent recursive sync calls

        # Gitignore support
        self._gitignore_parser: Optional[GitignoreParser] = None

        # RAG integration
        self._rag_manager: Optional["RAGIndexManager"] = None
        self._rag_config: Dict[str, Any] = {}

        # Modification tracking for compaction
        # RETE network bloats ~2.3x per modify cycle due to orphaned structures
        # in the C++ RETE layer (remove_source doesn't compact network)
        # After REBUILD_THRESHOLD modifications, force a full rebuild
        self._modification_count = 0
        self.REBUILD_THRESHOLD = 20  # Rebuild after 20 file modifications

        # Python package roots - directories containing __init__.py
        # Used to calculate correct module names for Python files
        self._package_roots: Optional[Set[str]] = None

        # Read configuration from environment
        self._load_config()

    def _load_config(self) -> None:
        """Load configuration from environment variables or auto-detect from CWD."""
        root = os.getenv("RETER_PROJECT_ROOT")

        # Auto-detect from current working directory if not explicitly set
        # Claude Code sets CWD to project root when launching MCP servers
        if not root:
            cwd = Path.cwd()
            # Only auto-detect if CWD looks like a Python project
            # (has .py files, pyproject.toml, setup.py, or .git)
            if self._looks_like_python_project(cwd):
                root = str(cwd)
                print(f"[default] Auto-detected project root from CWD: {root}", file=sys.stderr, flush=True)

        if root:
            self._project_root = Path(root).resolve()
            if not self._project_root.exists():
                raise ValueError(f"RETER_PROJECT_ROOT does not exist: {self._project_root}")
            if not self._project_root.is_dir():
                raise ValueError(f"RETER_PROJECT_ROOT is not a directory: {self._project_root}")

            # Initialize gitignore parser if .gitignore exists
            gitignore_path = self._project_root / ".gitignore"
            if gitignore_path.exists():
                self._gitignore_parser = GitignoreParser(self._project_root)
                pattern_count = self._gitignore_parser.get_pattern_count()
                debug_log(f"[default] Loaded .gitignore with {pattern_count} patterns")

        include = os.getenv("RETER_PROJECT_INCLUDE", "")
        if include:
            self._include_patterns = [p.strip() for p in include.split(",") if p.strip()]

        exclude = os.getenv("RETER_PROJECT_EXCLUDE", "")
        if exclude:
            self._exclude_patterns = [p.strip() for p in exclude.split(",") if p.strip()]

    def _looks_like_python_project(self, path: Path) -> bool:
        """
        Check if a directory looks like a Python project.

        Heuristics:
        - Has pyproject.toml, setup.py, or setup.cfg
        - Has .git directory (is a repo)
        - Has any .py files in root or immediate subdirs
        """
        # Check for common project files
        project_markers = ["pyproject.toml", "setup.py", "setup.cfg", "requirements.txt"]
        for marker in project_markers:
            if (path / marker).exists():
                return True

        # Check for .git directory
        if (path / ".git").exists():
            return True

        # Check for any .py files in root
        if list(path.glob("*.py")):
            return True

        # Check for src/ or common package directories with .py files
        common_dirs = ["src", "lib", "app", "tests"]
        for dirname in common_dirs:
            subdir = path / dirname
            if subdir.is_dir() and list(subdir.glob("*.py")):
                return True

        return False

    def is_configured(self) -> bool:
        """Check if default instance is configured (RETER_PROJECT_ROOT is set or auto-detected)."""
        return self._project_root is not None

    @property
    def project_root(self) -> Optional[Path]:
        """Get the project root directory."""
        return self._project_root

    def set_rag_manager(
        self,
        rag_manager: "RAGIndexManager",
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Set the RAG index manager for synchronization.

        RAG initialization is lazy - embedding model is NOT loaded here.
        It will be loaded on first search or reindex call.

        Args:
            rag_manager: RAGIndexManager instance
            config: Optional RAG configuration dict
        """
        self._rag_manager = rag_manager
        self._rag_config = config or {}
        # NOTE: RAG initialization is lazy - don't call rag_manager.initialize() here
        # The embedding model is heavy and will be loaded on first use

    def get_rag_manager(self) -> Optional["RAGIndexManager"]:
        """Get the RAG index manager if configured."""
        return self._rag_manager

    def get_status(self) -> str:
        """
        Get status of default instance.

        Returns:
            "loaded" if instance exists in memory
            "available" if snapshot exists
            "configured" if RETER_PROJECT_ROOT is set but not yet loaded
            "not_configured" if RETER_PROJECT_ROOT is not set
        """
        if not self.is_configured():
            return "not_configured"

        # Check if loaded in memory
        instances = self._persistence.instance_manager.get_all_instances()
        if self.INSTANCE_NAME in instances:
            return "loaded"

        # Check if snapshot exists
        snapshot_path = self._persistence.snapshots_dir / f"{self.INSTANCE_NAME}.reter"
        if snapshot_path.exists():
            return "available"

        return "configured"

    def ensure_default_instance_synced(self, reter: ReterWrapper) -> Optional[ReterWrapper]:
        """
        Ensure default instance is synced with project files.

        This method:
        1. Scans project directory for Python files
        2. Compares against existing sources in RETER
        3. Adds new files, reloads modified files, forgets deleted files
        4. Saves snapshot to disk if any changes were made
        5. Periodically rebuilds from scratch to compact RETE network

        Args:
            reter: The default ReterWrapper instance to sync

        Returns:
            None if sync completed in-place (no changes to instance)
            ReterWrapper if a full rebuild was performed (caller should replace instance)
        """
        if not self.is_configured():
            return None

        # Re-entrancy guard: prevent recursive sync calls
        # This can happen when RAG sync triggers REQL queries that somehow
        # trigger another sync request
        if self._syncing:
            print(f"[default] Sync already in progress, skipping recursive call", file=sys.stderr, flush=True)
            return None

        self._syncing = True
        try:
            return self._do_sync(reter)
        finally:
            self._syncing = False

    def _do_sync(self, reter: ReterWrapper) -> Optional[ReterWrapper]:
        """
        Internal sync implementation.

        Returns:
            None if no rebuild was needed (sync completed in-place)
            ReterWrapper if a full rebuild was performed (caller should replace instance)
        """
        import time
        start = time.time()
        print(f"[default] Syncing... (initialized={self._initialized})", file=sys.stderr, flush=True)
        if self._include_patterns:
            print(f"[default] Include patterns: {self._include_patterns}", file=sys.stderr, flush=True)
        if self._exclude_patterns:
            print(f"[default] Exclude patterns: {self._exclude_patterns}", file=sys.stderr, flush=True)

        # Get current files from filesystem
        print(f"[default] Scanning filesystem...", file=sys.stderr, flush=True)
        current_files = self._scan_project_files()
        print(f"[default] Found {len(current_files)} Python files in {time.time()-start:.2f}s", file=sys.stderr, flush=True)

        # Check if we should do a full rebuild instead of incremental sync
        # This compacts the RETE network which bloats ~20% per modify cycle
        if self._modification_count >= self.REBUILD_THRESHOLD:
            print(f"[default] Modification threshold exceeded ({self._modification_count} >= {self.REBUILD_THRESHOLD})", file=sys.stderr, flush=True)
            fresh_reter = self._force_rebuild(current_files)

            # Save the fresh (compacted) snapshot
            self._persistence.snapshots_dir.mkdir(parents=True, exist_ok=True)
            snapshot_path = self._persistence.snapshots_dir / ".default.reter"
            print(f"[default] Saving compacted snapshot to {snapshot_path}...", file=sys.stderr, flush=True)
            save_start = time.time()
            fresh_reter.save_network(str(snapshot_path))
            fresh_reter.mark_clean()
            print(f"[default] Compacted snapshot saved in {time.time()-save_start:.2f}s (total: {time.time()-start:.2f}s)", file=sys.stderr, flush=True)

            self._initialized = True
            return fresh_reter  # Caller should replace instance

        # Get existing sources from RETER
        print(f"[default] Querying existing sources...", file=sys.stderr, flush=True)
        existing_sources = self._get_existing_sources(reter)
        print(f"[default] Found {len(existing_sources)} sources already loaded in {time.time()-start:.2f}s", file=sys.stderr, flush=True)

        # Sync files
        print(f"[default] Syncing files...", file=sys.stderr, flush=True)
        changes_made = self._sync_files(reter, current_files, existing_sources)
        print(f"[default] Sync completed in {time.time()-start:.2f}s", file=sys.stderr, flush=True)

        # Auto-save snapshot if any changes were made
        if changes_made:
            self._persistence.snapshots_dir.mkdir(parents=True, exist_ok=True)
            snapshot_path = self._persistence.snapshots_dir / ".default.reter"  # Leading dot matches persistence convention
            print(f"[default] Saving snapshot to {snapshot_path}...", file=sys.stderr, flush=True)
            save_start = time.time()
            reter.save_network(str(snapshot_path))
            reter.mark_clean()
            print(f"[default] Snapshot saved in {time.time()-save_start:.2f}s (total: {time.time()-start:.2f}s)", file=sys.stderr, flush=True)
        else:
            print(f"[default] No changes detected, skipping snapshot save (total: {time.time()-start:.2f}s)", file=sys.stderr, flush=True)

        self._initialized = True

        # Initialize RAG index if configured (lazy initialization)
        # This ensures RAG is ready for semantic search after first use
        if self._rag_manager and self._rag_manager.is_enabled and not self._rag_manager.is_initialized:
            print(f"[default] Initializing RAG index...", file=sys.stderr, flush=True)
            try:
                rag_start = time.time()
                # sync_sources calls initialize() internally and indexes all sources
                rag_stats = self._rag_manager.sync_sources(
                    reter=reter,
                    project_root=self._project_root,
                )
                print(
                    f"[default] RAG initialized: {rag_stats.get('total_vectors', 0)} vectors "
                    f"in {time.time() - rag_start:.2f}s",
                    flush=True
                )
            except Exception as e:
                import traceback
                print(f"[default] RAG initialization error: {e}", file=sys.stderr, flush=True)
                debug_log(f"[default] RAG initialization error: {traceback.format_exc()}")

        return None  # No rebuild needed, sync completed in-place

    # Supported file extensions for code analysis
    PYTHON_EXTENSIONS = {".py"}
    JAVASCRIPT_EXTENSIONS = {".js", ".mjs", ".jsx", ".ts", ".tsx"}
    HTML_EXTENSIONS = {".html", ".htm"}
    CSHARP_EXTENSIONS = {".cs"}
    CPP_EXTENSIONS = {".cpp", ".cc", ".cxx", ".c++", ".hpp", ".hh", ".hxx", ".h++", ".h"}
    ALL_CODE_EXTENSIONS = PYTHON_EXTENSIONS | JAVASCRIPT_EXTENSIONS | HTML_EXTENSIONS | CSHARP_EXTENSIONS | CPP_EXTENSIONS

    def _scan_project_files(self) -> Dict[str, Tuple[str, str]]:
        """
        Scan project directory for Python, JavaScript, HTML, C#, and C++ files.

        Also scans for Python package roots (__init__.py files) to calculate
        correct module names for Python files.

        Returns:
            Dict mapping relative path to (absolute_path, md5_hash)
        """
        files: Dict[str, Tuple[str, str]] = {}

        if not self._project_root:
            return files

        # Scan for Python package roots (directories with __init__.py)
        # This must be done BEFORE loading files so we can calculate correct module names
        self._package_roots = ReterWrapper.scan_package_roots(str(self._project_root))
        if self._package_roots:
            debug_log(f"[default] Found {len(self._package_roots)} Python package roots: {sorted(self._package_roots)[:10]}...")

        # If include patterns are set, only scan those specific directories (fast path)
        if self._include_patterns:
            for pattern in self._include_patterns:
                # Extract the directory prefix from pattern (e.g., "src/*" -> "src")
                # Handle patterns like "src/**", "src/*", "src/foo/*"
                pattern_clean = pattern.replace('\\', '/')

                # Find the static prefix (everything before first wildcard)
                prefix_parts = []
                for part in pattern_clean.split('/'):
                    if '*' in part or '?' in part:
                        break
                    prefix_parts.append(part)

                if prefix_parts:
                    # Scan only the specified subdirectory
                    scan_dir = self._project_root / '/'.join(prefix_parts)
                    if scan_dir.exists() and scan_dir.is_dir():
                        self._scan_directory(scan_dir, files, pattern)
                else:
                    # Pattern starts with wildcard, must scan root
                    self._scan_directory(self._project_root, files, pattern)
        else:
            # No include patterns - scan everything
            self._scan_directory(self._project_root, files, None)

        return files

    def _scan_directory(self, scan_dir: Path, files: Dict[str, Tuple[str, str]], include_pattern: Optional[str]) -> None:
        """
        Scan a directory for Python, JavaScript, HTML, C#, and C++ files and add to files dict.

        Args:
            scan_dir: Directory to scan
            files: Dict to populate with results
            include_pattern: Optional pattern to filter files (if None, include all)
        """
        # Scan for all supported file types
        for ext in self.ALL_CODE_EXTENSIONS:
            pattern = f"*{ext}"
            for code_file in scan_dir.rglob(pattern):
                rel_path = code_file.relative_to(self._project_root)
                rel_path_str = str(rel_path).replace('\\', '/')

                # Skip if already processed (from another include pattern)
                if rel_path_str in files:
                    continue

                # If include pattern specified, verify file matches
                if include_pattern and not self._matches_pattern(rel_path_str, include_pattern):
                    continue

                # Skip excluded files
                if self._is_excluded(rel_path):
                    continue

                # Skip node_modules for JavaScript
                if "node_modules" in rel_path_str:
                    continue

                # Skip bin/obj directories for C#
                if "/bin/" in rel_path_str or "/obj/" in rel_path_str or rel_path_str.startswith("bin/") or rel_path_str.startswith("obj/"):
                    continue

                # Skip build directories for C++ (CMakeFiles, build, cmake-build-*, etc.)
                if "/CMakeFiles/" in rel_path_str or "/build/" in rel_path_str or "cmake-build-" in rel_path_str:
                    continue
                if rel_path_str.startswith("CMakeFiles/") or rel_path_str.startswith("build/"):
                    continue

                # Read file and compute MD5
                try:
                    content = code_file.read_text(encoding='utf-8')
                    md5_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
                    files[rel_path_str] = (str(code_file), md5_hash)
                except Exception as e:
                    print(f"Warning: Failed to read {code_file}: {e}", file=sys.stderr)

    def _is_excluded(self, rel_path: Path) -> bool:
        """
        Check if a file should be excluded based on patterns and .gitignore.

        Exclusion sources (in order of checking):
        1. .gitignore patterns (if .gitignore exists)
        2. RETER_PROJECT_EXCLUDE patterns (from reter.json or env var)

        Args:
            rel_path: Relative path to check

        Returns:
            True if file should be excluded
        """
        path_str = str(rel_path).replace('\\', '/')

        # Check gitignore patterns first
        if self._gitignore_parser is not None:
            if self._gitignore_parser.is_ignored(rel_path):
                return True

        # Check explicit exclude patterns
        for pattern in self._exclude_patterns:
            # Simple glob pattern matching
            if self._matches_pattern(path_str, pattern):
                return True

        return False

    def _matches_pattern(self, path: str, pattern: str) -> bool:
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
        from fnmatch import fnmatch

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

        # Regular fnmatch
        return fnmatch(path, pattern)

    def _get_existing_sources(self, reter: ReterWrapper) -> Dict[str, Tuple[str, str]]:
        """
        Get existing sources from RETER instance.

        Returns:
            Dict mapping normalized relative path (forward slashes) to (md5_hash, original_source_id)
        """
        sources: Dict[str, Tuple[str, str]] = {}

        # Get all sources
        all_sources, _ = reter.get_all_sources()
        debug_log(f"[default] get_all_sources returned {len(all_sources)} sources")

        for source_id in all_sources:
            # Parse source ID: format is "md5_hash|rel_path"
            if "|" in source_id:
                md5_hash, rel_path = source_id.split("|", 1)
                # Check if it's a supported code file type
                is_code_file = any(rel_path.endswith(ext) for ext in self.ALL_CODE_EXTENSIONS)
                if is_code_file:
                    # Normalize path to forward slashes (must match _scan_project_files)
                    rel_path_normalized = rel_path.replace('\\', '/')
                    # Store both md5 and original source_id for proper forgetting
                    sources[rel_path_normalized] = (md5_hash, source_id)
                    debug_log(f"[default]   existing: {rel_path_normalized} -> md5={md5_hash[:8]}...")

        debug_log(f"[default] Found {len(sources)} code sources in RETER")
        return sources

    def _load_code_file(self, reter: ReterWrapper, abs_path: str, rel_path: str) -> None:
        """
        Load a code file using the appropriate loader based on file extension.

        Args:
            reter: ReterWrapper instance
            abs_path: Absolute path to the file
            rel_path: Relative path (for logging)
        """
        # Determine file type by extension
        is_python = any(rel_path.endswith(ext) for ext in self.PYTHON_EXTENSIONS)
        is_javascript = any(rel_path.endswith(ext) for ext in self.JAVASCRIPT_EXTENSIONS)
        is_html = any(rel_path.endswith(ext) for ext in self.HTML_EXTENSIONS)
        is_csharp = any(rel_path.endswith(ext) for ext in self.CSHARP_EXTENSIONS)
        is_cpp = any(rel_path.endswith(ext) for ext in self.CPP_EXTENSIONS)

        if is_python:
            # Pass package_roots for proper Python module name calculation
            reter.load_python_file(abs_path, str(self._project_root), self._package_roots)
        elif is_javascript:
            reter.load_javascript_file(abs_path, str(self._project_root))
        elif is_html:
            reter.load_html_file(abs_path, str(self._project_root))
        elif is_csharp:
            reter.load_csharp_file(abs_path, str(self._project_root))
        elif is_cpp:
            reter.load_cpp_file(abs_path, str(self._project_root))
        else:
            raise ValueError(f"Unsupported file type: {rel_path}")

    def _sync_files(
        self,
        reter: ReterWrapper,
        current_files: Dict[str, Tuple[str, str]],
        existing_sources: Dict[str, Tuple[str, str]]
    ) -> bool:
        """
        Sync files between filesystem and RETER instance.

        Strategy:
        - New files: load with load_python_file
        - Modified files (MD5 changed): forget then reload
        - Deleted files: forget

        Also syncs with RAG index if a RAGIndexManager is configured.

        Args:
            reter: ReterWrapper instance
            current_files: Dict mapping rel_path to (abs_path, md5_hash)
            existing_sources: Dict mapping rel_path to (md5_hash, original_source_id)

        Returns:
            bool: True if any changes were made (requires snapshot save)
        """
        added_count = 0
        modified_count = 0
        deleted_count = 0

        # Track source IDs for RAG sync (separated by language)
        changed_python_sources: List[str] = []
        deleted_python_sources: List[str] = []
        changed_javascript_sources: List[str] = []
        deleted_javascript_sources: List[str] = []
        changed_html_sources: List[str] = []
        deleted_html_sources: List[str] = []
        changed_csharp_sources: List[str] = []
        deleted_csharp_sources: List[str] = []
        changed_cpp_sources: List[str] = []
        deleted_cpp_sources: List[str] = []

        errors = []  # Track errors but continue processing

        # Helper to classify source by file extension
        def track_changed_source(rel_path: str, source_id: str) -> None:
            """Add source to appropriate changed list based on file extension."""
            ext = Path(rel_path).suffix.lower()
            if ext in PYTHON_EXTENSIONS:
                changed_python_sources.append(source_id)
            elif ext in JAVASCRIPT_EXTENSIONS:
                changed_javascript_sources.append(source_id)
            elif ext in HTML_EXTENSIONS:
                changed_html_sources.append(source_id)
            elif ext in CSHARP_EXTENSIONS:
                changed_csharp_sources.append(source_id)
            elif ext in CPP_EXTENSIONS:
                changed_cpp_sources.append(source_id)

        def track_deleted_source(rel_path: str, source_id: str) -> None:
            """Add source to appropriate deleted list based on file extension."""
            ext = Path(rel_path).suffix.lower()
            if ext in PYTHON_EXTENSIONS:
                deleted_python_sources.append(source_id)
            elif ext in JAVASCRIPT_EXTENSIONS:
                deleted_javascript_sources.append(source_id)
            elif ext in HTML_EXTENSIONS:
                deleted_html_sources.append(source_id)
            elif ext in CSHARP_EXTENSIONS:
                deleted_csharp_sources.append(source_id)
            elif ext in CPP_EXTENSIONS:
                deleted_cpp_sources.append(source_id)

        # Debug: log first few current files for comparison
        debug_log(f"[default] _sync_files: {len(current_files)} current files, {len(existing_sources)} existing sources")
        for i, (rel_path, (abs_path, current_md5)) in enumerate(list(current_files.items())[:5]):
            debug_log(f"[default]   current[{i}]: {rel_path} -> md5={current_md5[:8]}...")

        # Find new and modified files
        for rel_path, (abs_path, current_md5) in current_files.items():
            if rel_path not in existing_sources:
                # New file - load it using the appropriate loader
                debug_log(f"[default] NEW file (not in existing): {rel_path}")
                print(f"[default] Adding new file: {rel_path}", file=sys.stderr, flush=True)
                try:
                    self._load_code_file(reter, abs_path, rel_path)
                    added_count += 1
                    print(f"[default]   ✓ Added {rel_path}", file=sys.stderr, flush=True)
                    # Track for RAG: generate source_id for new file
                    new_source_id = generate_source_id(abs_path, str(self._project_root))
                    track_changed_source(rel_path, new_source_id)
                except Exception as e:
                    # Log error but continue with other files (don't re-raise)
                    error_msg = f"Error loading {rel_path}: {type(e).__name__}: {e}"
                    print(f"[default]   ✗ {error_msg}", file=sys.stderr, flush=True)
                    errors.append(error_msg)
            else:
                old_md5, old_source_id = existing_sources[rel_path]
                if old_md5 != current_md5:
                    # Modified file - forget and reload
                    debug_log(f"[default] MD5 MISMATCH: {rel_path} old={old_md5[:8]}... new={current_md5[:8]}...")
                    print(f"[default] Reloading modified file: {rel_path}", file=sys.stderr, flush=True)
                    try:
                        reter.forget_source(old_source_id)
                        self._load_code_file(reter, abs_path, rel_path)
                        modified_count += 1
                        print(f"[default]   ✓ Reloaded {rel_path}", file=sys.stderr, flush=True)
                        # Track for RAG: old source deleted, new source added
                        track_deleted_source(rel_path, old_source_id)
                        new_source_id = generate_source_id(abs_path, str(self._project_root))
                        track_changed_source(rel_path, new_source_id)
                    except Exception as e:
                        # Log error but continue with other files (don't re-raise)
                        error_msg = f"Error reloading {rel_path}: {type(e).__name__}: {e}"
                        print(f"[default]   ✗ {error_msg}", file=sys.stderr, flush=True)
                        errors.append(error_msg)

        # Find deleted files
        for rel_path, (old_md5, old_source_id) in existing_sources.items():
            if rel_path not in current_files:
                # Deleted file - forget it
                print(f"[default] Forgetting deleted file: {rel_path}", file=sys.stderr, flush=True)
                try:
                    reter.forget_source(old_source_id)
                    deleted_count += 1
                    print(f"[default]   ✓ Forgot {rel_path}", file=sys.stderr, flush=True)
                    # Track for RAG
                    track_deleted_source(rel_path, old_source_id)
                except Exception as e:
                    # Log error but continue with other files (don't re-raise)
                    error_msg = f"Error forgetting {rel_path}: {type(e).__name__}: {e}"
                    print(f"[default]   ✗ {error_msg}", file=sys.stderr, flush=True)
                    errors.append(error_msg)

        if errors:
            print(f"[default] Sync completed with {len(errors)} errors", file=sys.stderr, flush=True)

        changes_made = added_count > 0 or modified_count > 0 or deleted_count > 0

        if changes_made:
            print(f"[default] Sync complete: +{added_count} ~{modified_count} -{deleted_count}", file=sys.stderr)

        # Sync with RAG index if configured
        # NOTE: RAG sync is skipped during initial load to avoid slowing down startup.
        # Use rag_reindex(force=True) to build the index after startup.
        if self._rag_manager and self._rag_manager.is_enabled and self._rag_manager.is_initialized:
            has_changes = (
                changed_python_sources or deleted_python_sources or
                changed_javascript_sources or deleted_javascript_sources or
                changed_html_sources or deleted_html_sources or
                changed_csharp_sources or deleted_csharp_sources or
                changed_cpp_sources or deleted_cpp_sources
            )
            if has_changes:
                print(f"[default] Syncing RAG index...", file=sys.stderr, flush=True)
                try:
                    rag_stats = self._rag_manager.sync(
                        reter=reter,
                        changed_python_sources=changed_python_sources,
                        deleted_python_sources=deleted_python_sources,
                        project_root=self._project_root,
                        changed_javascript_sources=changed_javascript_sources,
                        deleted_javascript_sources=deleted_javascript_sources,
                        changed_html_sources=changed_html_sources,
                        deleted_html_sources=deleted_html_sources,
                        changed_csharp_sources=changed_csharp_sources,
                        deleted_csharp_sources=deleted_csharp_sources,
                        changed_cpp_sources=changed_cpp_sources,
                        deleted_cpp_sources=deleted_cpp_sources,
                    )
                    total_added = (
                        rag_stats.get('python_vectors_added', 0) +
                        rag_stats.get('javascript_vectors_added', 0) +
                        rag_stats.get('html_vectors_added', 0) +
                        rag_stats.get('csharp_vectors_added', 0) +
                        rag_stats.get('cpp_vectors_added', 0)
                    )
                    print(
                        f"[default] RAG sync: +{total_added} vectors "
                        f"in {rag_stats.get('time_ms', 0)}ms",
                        flush=True
                    )
                except Exception as e:
                    print(f"[default] RAG sync error: {e}", file=sys.stderr, flush=True)

        # Track modifications for compaction (only forget operations cause bloat)
        modification_ops = modified_count + deleted_count
        if modification_ops > 0:
            self._modification_count += modification_ops
            debug_log(f"[default] Modification count: {self._modification_count} (threshold: {self.REBUILD_THRESHOLD})")

        if changes_made:
            return True  # Changes were made, need to save snapshot

        return False  # No changes, no save needed

    def _force_rebuild(
        self,
        current_files: Dict[str, Tuple[str, str]]
    ) -> ReterWrapper:
        """
        Force a complete rebuild of the RETER instance from scratch.

        This creates a fresh RETE network without accumulated garbage from
        remove_source operations. The C++ RETE layer doesn't compact network
        structures when sources are removed, causing ~20% bloat per modify cycle.

        Args:
            current_files: Dict mapping rel_path to (abs_path, md5_hash)

        Returns:
            Fresh ReterWrapper instance with all files loaded
        """
        import time
        start = time.time()
        print(f"[default] Force rebuilding to compact RETE network ({self._modification_count} modifications accumulated)...", file=sys.stderr, flush=True)

        # Create fresh ReterWrapper with new RETE network
        fresh_reter = ReterWrapper()

        # Load all current files
        loaded = 0
        errors = []
        for rel_path, (abs_path, _) in current_files.items():
            try:
                self._load_code_file(fresh_reter, abs_path, rel_path)
                loaded += 1
            except Exception as e:
                error_msg = f"Error loading {rel_path}: {type(e).__name__}: {e}"
                print(f"[default]   ✗ {error_msg}", file=sys.stderr, flush=True)
                errors.append(error_msg)

        # Reset modification count
        self._modification_count = 0

        print(f"[default] Rebuild complete: loaded {loaded} files in {time.time()-start:.2f}s", file=sys.stderr, flush=True)
        if errors:
            print(f"[default] Rebuild had {len(errors)} errors", file=sys.stderr, flush=True)

        return fresh_reter
