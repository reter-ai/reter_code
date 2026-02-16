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

import os
import hashlib
import threading
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple, Any, TYPE_CHECKING
from ..logging_config import configure_logger_for_debug_trace
from ..reter_wrapper import ReterWrapper
from .gitignore_parser import GitignoreParser
from .source_state_manager import SourceStateManager, SyncChanges, FileInfo
from .rag_index_manager import SyncChanges as RAGSyncChanges, LanguageSourceChanges
from .utils import matches_pattern

try:
    from watchdog.observers import Observer
    from watchdog.events import (
        FileSystemEventHandler,
        FileSystemEvent,
        FileCreatedEvent,
        FileDeletedEvent,
        FileModifiedEvent,
        FileMovedEvent,
    )
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    Observer = None
    FileSystemEventHandler = object
    FileSystemEvent = None
    FileCreatedEvent = None
    FileDeletedEvent = None
    FileModifiedEvent = None
    FileMovedEvent = None

if TYPE_CHECKING:
    from .state_persistence import StatePersistenceService
    from .rag_index_manager import RAGIndexManager

logger = configure_logger_for_debug_trace(__name__)


class _FileChangeHandler(FileSystemEventHandler):
    """
    Watchdog event handler that sets a dirty flag when files change.

    ::: This is-in-layer Service-Layer.
    ::: This is a handler.
    ::: This is-in-process Main-Process.
    ::: This is stateful.

    Only triggers on supported code file extensions and respects
    gitignore/exclude patterns.

    NOTE: We use specific event handlers (on_created, on_deleted, on_modified,
    on_moved) instead of on_any_event to avoid false positives from file access
    events on Windows. Reading a file can update atime which triggers events.

    We also track mtime to filter out false "modified" events that don't
    actually change file content (e.g., antivirus scanners, indexers).
    """

    def __init__(self, manager: "DefaultInstanceManager"):
        super().__init__()
        self._manager = manager
        self._last_mtime: Dict[str, float] = {}  # Track mtime to filter false positives

    def _handle_file_event(self, event: "FileSystemEvent", check_mtime: bool = False) -> None:
        """Common handler for file create/delete/modify/move events."""
        if event.is_directory:
            return

        # For move events, check destination path (e.g., atomic writes: tmp -> target)
        # This handles Claude Code's atomic write pattern: write to .tmp, rename to .py
        if hasattr(event, 'dest_path') and event.dest_path:
            file_path = Path(event.dest_path)
        else:
            file_path = Path(event.src_path)

        # Get relative path
        try:
            if not self._manager._project_root:
                return
            rel_path = file_path.relative_to(self._manager._project_root)
        except (ValueError, TypeError):
            return

        # Check if it's a supported code file or markdown
        ext = file_path.suffix.lower()
        if ext not in self._manager.ALL_CODE_EXTENSIONS and ext != ".md":
            return

        # Check if excluded
        if self._manager._is_excluded_for_scan(rel_path):
            return

        # For "modified" events, check if mtime actually changed
        # This filters out false positives from antivirus, indexers, etc.
        if check_mtime:
            try:
                current_mtime = file_path.stat().st_mtime
                path_key = str(file_path)
                last_mtime = self._last_mtime.get(path_key, 0)
                if current_mtime == last_mtime:
                    # mtime unchanged - likely a false positive (read, not write)
                    return
                self._last_mtime[path_key] = current_mtime
            except (OSError, IOError):
                # File might have been deleted, let it through
                pass

        # Track the file for targeted sync (just record that it changed, not the event type)
        rel_path_str = str(rel_path).replace('\\', '/')
        self._manager._pending_changes.add(rel_path_str)

        if not self._manager._dirty:
            # Get event type name for logging
            event_type_name = type(event).__name__
            logger.debug(f"[FileWatcher] Change detected: {rel_path} ({event_type_name})")
            self._manager._dirty = True

    def on_created(self, event: "FileSystemEvent") -> None:
        """Handle file creation."""
        self._handle_file_event(event)

    def on_deleted(self, event: "FileSystemEvent") -> None:
        """Handle file deletion."""
        self._handle_file_event(event)

    def on_modified(self, event: "FileSystemEvent") -> None:
        """Handle file modification - with mtime check to filter false positives."""
        self._handle_file_event(event, check_mtime=True)

    def on_moved(self, event: "FileSystemEvent") -> None:
        """Handle file move/rename."""
        self._handle_file_event(event)


class DefaultInstanceManager:
    """
    Manages the "default" RETER instance with automatic file synchronization.

    ::: This is-in-layer Service-Layer.
    ::: This is a manager.
    ::: This is-in-process Main-Process.
    ::: This is stateful.

    The default instance:
    - Always appears in list_all_instances()
    - Lazy-loads on first access
    - Auto-syncs with project files based on MD5 checksums
    - Configured via RETER_PROJECT_ROOT env var, OR auto-detected from CWD
    - Optional: RETER_PROJECT_INCLUDE and RETER_PROJECT_EXCLUDE for filtering
    """

    INSTANCE_NAME = "default"

    def __init__(self, persistence: "StatePersistenceService", progress_callback: Optional[Any] = None):
        """
        Initialize the default instance manager.

        Args:
            persistence: StatePersistenceService for snapshot management
            progress_callback: Optional ConsoleProgress for UI updates during initialization
        """
        self._persistence = persistence
        self._project_root: Optional[Path] = None
        self._include_patterns: List[str] = []
        self._exclude_patterns: List[str] = []
        self._initialized = False
        self._syncing = False  # Re-entrancy guard to prevent recursive sync calls
        self._progress_callback: Optional[Any] = None  # ConsoleProgress for UI updates
        self._init_progress_callback = progress_callback  # Used during __init__ for gitignore loading

        # Gitignore support (patterns pre-loaded for performance)
        self._gitignore_parser: Optional[GitignoreParser] = None

        # Source state manager (single source of truth for tracked files)
        self._source_state: Optional[SourceStateManager] = None

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

        # File watcher for detecting changes (avoids 10s scan on every tool call)
        # _dirty=True means we need to scan; False means no changes since last scan
        self._dirty = True  # Start dirty - need initial scan
        self._observer: Optional["Observer"] = None
        self._watcher_started = False

        # Pending changes from file watcher - avoids full rescan
        # Just stores rel_paths of files that changed (we detect add/modify/delete during sync)
        self._pending_changes: Set[str] = set()

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
            root = str(cwd)
            # Only log if no progress UI is active
            if not self._init_progress_callback:
                logger.info(f"[default] Auto-detected project root from CWD: {root}")

        if root:
            self._project_root = Path(root).resolve()
            if not self._project_root.exists():
                raise ValueError(f"RETER_PROJECT_ROOT does not exist: {self._project_root}")
            if not self._project_root.is_dir():
                raise ValueError(f"RETER_PROJECT_ROOT is not a directory: {self._project_root}")

            # Initialize gitignore parser and pre-load ALL .gitignore files
            # This is more efficient than lazy loading during file scans
            if self._init_progress_callback and hasattr(self._init_progress_callback, 'start_gitignore_loading'):
                self._init_progress_callback.start_gitignore_loading()

            self._gitignore_parser = GitignoreParser(self._project_root, progress_callback=self._init_progress_callback)
            self._gitignore_parser.load_all_gitignores()
            pattern_count = self._gitignore_parser.get_pattern_count()

            if self._init_progress_callback and hasattr(self._init_progress_callback, 'end_gitignore_loading'):
                self._init_progress_callback.end_gitignore_loading(pattern_count)

            if pattern_count > 0:
                logger.debug(f"[default] Pre-loaded {pattern_count} gitignore patterns from {len(self._gitignore_parser.get_loaded_gitignores())} files")

            # Initialize source state manager (single source of truth)
            state_file = self._persistence.snapshots_dir / ".default.sources.json"
            self._source_state = SourceStateManager(state_file, self._project_root)
            self._source_state.set_gitignore_patterns(
                patterns=self._gitignore_parser._patterns,
                gitignore_hash=self._gitignore_parser.get_gitignore_hash(),
                gitignore_files=self._gitignore_parser.get_gitignore_files_hashes(),
            )

        include = os.getenv("RETER_PROJECT_INCLUDE", "")
        if include:
            self._include_patterns = [p.strip() for p in include.split(",") if p.strip()]
            logger.debug(f"[default] Include patterns: {self._include_patterns}")

        exclude = os.getenv("RETER_PROJECT_EXCLUDE", "")
        if exclude:
            self._exclude_patterns = [p.strip() for p in exclude.split(",") if p.strip()]
            logger.debug(f"[default] Exclude patterns: {self._exclude_patterns}")

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

    def get_source_state(self) -> Optional[SourceStateManager]:
        """
        Get the source state manager.

        The source state manager is the single source of truth for what files
        are loaded in RETER and RAG. Use this to query tracked files without
        making expensive RETER queries.

        Returns:
            SourceStateManager instance, or None if not configured
        """
        return self._source_state

    def start_file_watcher(self) -> bool:
        """
        Start the filesystem watcher to detect file changes.

        The watcher monitors the project root for changes to code files,
        setting a dirty flag when changes are detected. This allows
        ensure_default_instance_synced to skip expensive filesystem scans
        when no files have changed.

        Returns:
            True if watcher started successfully, False otherwise
        """
        if not WATCHDOG_AVAILABLE:
            if not self._progress_callback:
                logger.info("[FileWatcher] watchdog not installed, file watching disabled")
            return False

        if self._watcher_started:
            return True

        if not self._project_root:
            return False

        try:
            self._observer = Observer()
            handler = _FileChangeHandler(self)
            self._observer.schedule(handler, str(self._project_root), recursive=True)
            self._observer.start()
            self._watcher_started = True
            # Only log if no progress UI is active
            if not self._progress_callback:
                logger.info(f"[FileWatcher] Started watching {self._project_root}")
            return True
        except Exception as e:
            if not self._progress_callback:
                logger.info(f"[FileWatcher] Failed to start: {e}")
            self._observer = None
            return False

    def stop_file_watcher(self) -> None:
        """Stop the filesystem watcher."""
        if self._observer and self._watcher_started:
            try:
                self._observer.stop()
                self._observer.join(timeout=2.0)
                if not self._progress_callback:
                    logger.info("[FileWatcher] Stopped")
            except Exception as e:
                if not self._progress_callback:
                    logger.info(f"[FileWatcher] Error stopping: {e}")
            finally:
                self._observer = None
                self._watcher_started = False

    def mark_dirty(self) -> None:
        """Mark the instance as needing a sync (for manual triggering)."""
        self._dirty = True
        logger.debug("[FileWatcher] Manually marked dirty")

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

    def ensure_default_instance_synced(
        self,
        reter: ReterWrapper,
        progress_callback: Optional[Any] = None,
    ) -> Optional[ReterWrapper]:
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
            progress_callback: Optional ConsoleProgress instance for UI updates

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
            if progress_callback is None:
                logger.info("[default] Sync already in progress, skipping recursive call")
            return None

        self._syncing = True
        self._progress_callback = progress_callback
        try:
            return self._do_sync(reter)
        finally:
            self._syncing = False
            self._progress_callback = None

    def _do_sync(self, reter: ReterWrapper) -> Optional[ReterWrapper]:
        """
        Internal sync implementation using SourceStateManager for optimized tracking.

        Key optimizations:
        1. Uses mtime-first checking (only computes MD5 when mtime/size changes)
        2. JSON state file is the source of truth (no RETER query needed)
        3. Pre-loaded gitignore patterns (no lazy loading during scan)

        Returns:
            None if no rebuild was needed (sync completed in-place)
            ReterWrapper if a full rebuild was performed (caller should replace instance)
        """
        import time
        start = time.time()
        if self._progress_callback is None:
            logger.info(f"[default] Syncing... (initialized={self._initialized})")
            if self._include_patterns:
                logger.info(f"[default] Include patterns: {self._include_patterns}")
            if self._exclude_patterns:
                logger.info(f"[default] Exclude patterns: {self._exclude_patterns}")
        else:
            self._progress_callback.set_phase("Initializing sync...")

        # Load source state from JSON (single source of truth)
        state_loaded = self._source_state.load() if self._source_state else False

        # If no state file exists, build initial state from RETER (migration)
        if not state_loaded and self._source_state:
            if self._progress_callback is None:
                logger.info("[default] No state file found, building from RETER...")
            else:
                self._progress_callback.set_phase("Building initial state...")
            all_sources, _ = reter.get_all_sources()
            self._source_state.build_from_reter(all_sources)

        # Check if we should do a full rebuild instead of incremental sync
        # This compacts the RETE network which bloats ~20% per modify cycle
        if self._modification_count >= self.REBUILD_THRESHOLD:
            if self._progress_callback is None:
                logger.info(f"[default] Modification threshold exceeded ({self._modification_count} >= {self.REBUILD_THRESHOLD})")
            else:
                self._progress_callback.set_phase("Rebuilding (compacting)...")
            # For rebuild, we need to scan all files
            current_files = self._scan_project_files()
            fresh_reter = self._force_rebuild(current_files)

            # Update state with all loaded files
            if self._source_state:
                self._source_state.clear()
                for rel_path, (abs_path, md5_hash) in current_files.items():
                    source_id = f"{md5_hash}|{rel_path}"
                    file_info = FileInfo(
                        rel_path=rel_path,
                        abs_path=abs_path,
                        md5=md5_hash,
                        mtime=Path(abs_path).stat().st_mtime,
                        size=Path(abs_path).stat().st_size,
                        in_reter=True,
                        reter_source_id=source_id,
                    )
                    self._source_state.set_file(file_info)
                self._source_state.save()

            # Save the fresh (compacted) snapshot
            self._persistence.snapshots_dir.mkdir(parents=True, exist_ok=True)
            snapshot_path = self._persistence.snapshots_dir / ".default.reter"
            if self._progress_callback is None:
                logger.info(f"[default] Saving compacted snapshot to {snapshot_path}...")
            else:
                self._progress_callback.set_phase("Saving snapshot...")
            save_start = time.time()

            # Create progress callback for save operation
            def save_progress(percent: int) -> None:
                if self._progress_callback:
                    self._progress_callback.set_phase(f"Saving snapshot... {percent}%")
                else:
                    if percent % 10 == 0:
                        logger.debug(f"[default] Saving... {percent}%")

            fresh_reter.save_network(str(snapshot_path), save_progress)
            fresh_reter.mark_clean()
            if self._progress_callback is None:
                logger.info(f"[default] Compacted snapshot saved in {time.time()-save_start:.2f}s (total: {time.time()-start:.2f}s)")

            self._initialized = True
            self._dirty = False  # Just rebuilt, everything is clean

            # Start file watcher after rebuild
            if not self._watcher_started:
                self.start_file_watcher()

            return fresh_reter  # Caller should replace instance

        # Track changed files for RAG targeted sync (None = full scan)
        rag_changed_files: Optional[Set[str]] = None

        # Use SourceStateManager for fast mtime-first diff
        if self._source_state:
            # Check if we can skip the scan (no changes detected by watcher)
            if not self._dirty and self._watcher_started and self._initialized:
                logger.debug(f"[default] No changes detected by watcher, skipping scan")
                if self._progress_callback is None:
                    logger.info(f"[default] No changes (watcher), skipping scan (total: {time.time()-start:.2f}s)")
                return None

            # OPTIMIZATION: If we have pending changes from file watcher, use targeted sync
            # instead of scanning the entire filesystem

            if self._pending_changes and self._watcher_started and self._initialized:
                pending_count = len(self._pending_changes)
                if self._progress_callback is None:
                    logger.info(f"[default] Targeted sync for {pending_count} changed files...")
                else:
                    self._progress_callback.set_phase("Processing file changes...")
                    self._progress_callback.start_scan(f"Processing {pending_count} changes")

                # Save for RAG targeted sync before clearing
                rag_changed_files = self._pending_changes.copy()

                changes = self._build_changes_from_pending(self._progress_callback)

                # Clear pending changes after building
                self._pending_changes.clear()

                total_changes = len(changes.to_add) + len(changes.to_modify) + len(changes.to_delete)
                if self._progress_callback is None:
                    logger.info(f"[default] Targeted sync: +{len(changes.to_add)} ~{len(changes.to_modify)} -{len(changes.to_delete)} in {time.time()-start:.2f}s")
                else:
                    self._progress_callback.end_scan(total_changes)

            else:
                # Full scan (initial load or no pending changes tracked)
                if self._progress_callback is None:
                    logger.info("[default] Quick scanning with mtime-first check...")
                else:
                    self._progress_callback.set_phase("Scanning for changes...")
                    self._progress_callback.start_scan("Scanning files")

                changes = self._source_state.scan_and_diff(
                    include_patterns=self._include_patterns,
                    exclude_patterns=self._exclude_patterns,
                    is_excluded_func=self._is_excluded_for_scan,
                )

                total_changes = len(changes.to_add) + len(changes.to_modify) + len(changes.to_delete)
                if self._progress_callback is None:
                    logger.info(f"[default] Quick scan found +{len(changes.to_add)} ~{len(changes.to_modify)} -{len(changes.to_delete)} in {time.time()-start:.2f}s")
                else:
                    self._progress_callback.end_scan(total_changes)

                # Clear any pending changes accumulated during full scan (already handled by scan)
                self._pending_changes.clear()
                # Full scan means RAG should also do full scan (no targeted optimization)
                rag_changed_files = None

            # Clear dirty flag after successful scan/targeted sync
            self._dirty = False

            # Apply changes using the optimized method
            changes_made = self._apply_sync_changes(reter, changes, rag_changed_files)
        else:
            # Fallback to old method if no state manager
            if self._progress_callback is None:
                logger.info("[default] Scanning filesystem (fallback)...")
            current_files = self._scan_project_files()
            if self._progress_callback is None:
                logger.info(f"[default] Found {len(current_files)} code files in {time.time()-start:.2f}s")

            if self._progress_callback is None:
                logger.info("[default] Querying existing sources...")
            existing_sources = self._get_existing_sources(reter)
            if self._progress_callback is None:
                logger.info(f"[default] Found {len(existing_sources)} sources already loaded in {time.time()-start:.2f}s")

            if self._progress_callback is None:
                logger.info("[default] Syncing files...")
            changes_made = self._sync_files(reter, current_files, existing_sources)

            # Clear dirty flag after fallback scan too
            self._dirty = False

        if self._progress_callback is None:
            logger.info(f"[default] Sync completed in {time.time()-start:.2f}s")

        # Auto-save snapshot if any changes were made
        if changes_made:
            self._persistence.snapshots_dir.mkdir(parents=True, exist_ok=True)
            snapshot_path = self._persistence.snapshots_dir / ".default.reter"
            if self._progress_callback is None:
                logger.info(f"[default] Saving snapshot to {snapshot_path}...")
            else:
                self._progress_callback.set_phase("Saving snapshot...")
            save_start = time.time()

            # Create progress callback for save operation
            def save_progress(percent: int) -> None:
                if self._progress_callback:
                    self._progress_callback.set_phase(f"Saving snapshot... {percent}%")
                else:
                    if percent % 10 == 0:
                        logger.debug(f"[default] Saving... {percent}%")

            # CRITICAL: Use incremental save for hybrid mode, full save otherwise
            # Using save_network() in hybrid mode would overwrite the base with only
            # the materialized portion (which may be incomplete after sync)
            if reter.is_hybrid_mode():
                reter.save_incremental()
            else:
                reter.save_network(str(snapshot_path), save_progress)
                # Switch to hybrid mode after full save for faster subsequent saves
                try:
                    reter.open_hybrid(str(snapshot_path))
                    if self._progress_callback is None:
                        logger.info("[default] Switched to hybrid mode for incremental saves")
                except Exception as e:
                    logger.debug(f"[default] Failed to switch to hybrid mode: {e}")
            reter.mark_clean()
            # Also save source state
            if self._source_state:
                self._source_state.save()
            if self._progress_callback is None:
                logger.info(f"[default] Snapshot saved in {time.time()-save_start:.2f}s (total: {time.time()-start:.2f}s)")
        else:
            if self._progress_callback is None:
                logger.info(f"[default] No changes detected, skipping snapshot save (total: {time.time()-start:.2f}s)")

        self._initialized = True

        # Start file watcher after first successful sync
        if not self._watcher_started:
            self.start_file_watcher()

        # Initialize RAG index if configured (lazy initialization)
        if self._rag_manager and self._rag_manager.is_enabled and not self._rag_manager.is_initialized:
            if self._progress_callback is None:
                logger.info("[default] Initializing RAG index...")
            else:
                self._progress_callback.start_rag_indexing()

            # Create progress callback for RAG sync
            def rag_progress_callback(current: int, total: int, phase: str):
                if self._progress_callback:
                    self._progress_callback.update_rag_progress(current, total)

            try:
                rag_start = time.time()
                rag_stats = self._rag_manager.sync_sources(
                    reter=reter,
                    project_root=self._project_root,
                    progress_callback=rag_progress_callback,
                    changed_files=rag_changed_files,
                )
                total_vectors = rag_stats.get('total_vectors', 0)
                if self._progress_callback is None:
                    logger.info(
                        f"[default] RAG initialized: {total_vectors} vectors "
                        f"in {time.time() - rag_start:.2f}s"
                    )
                else:
                    self._progress_callback.end_rag_indexing(total_vectors)
            except Exception as e:
                import traceback
                if self._progress_callback is None:
                    logger.info(f"[default] RAG initialization error: {e}")
                logger.debug(f"[default] RAG initialization error: {traceback.format_exc()}")

        return None  # No rebuild needed, sync completed in-place

    def _build_changes_from_pending(self, progress_callback: Optional[Any] = None) -> SyncChanges:
        """
        Build SyncChanges from pending file events (targeted sync).

        Instead of scanning the entire filesystem, this processes only the
        files that were reported changed by the file watcher.

        Args:
            progress_callback: Optional ConsoleProgress for UI updates

        Returns:
            SyncChanges with files to add, modify, or delete
        """
        import time
        start = time.time()
        changes = SyncChanges()

        if not self._source_state or not self._pending_changes:
            return changes

        total_pending = len(self._pending_changes)
        logger.debug(f"[default] Building changes from {total_pending} pending files")

        for idx, rel_path in enumerate(self._pending_changes):
            # Update progress
            if progress_callback:
                progress_callback.update_progress("Processing changes", idx + 1, total_pending)

            # Skip non-code files (e.g. .md) — handled by RAG markdown sync separately
            ext = Path(rel_path).suffix.lower()
            if ext not in self.ALL_CODE_EXTENSIONS:
                continue

            abs_path = self._project_root / rel_path
            cached = self._source_state.get_file(rel_path)

            if not abs_path.exists():
                # File doesn't exist - if we had it cached, it's deleted
                if cached:
                    changes.to_delete.append(cached)
                    logger.debug(f"[default]   Deleted: {rel_path}")
                continue

            # File exists - check if new or modified
            status, new_info = self._source_state.quick_check_file(abs_path)

            if status == "new":
                changes.to_add.append(new_info)
                logger.debug(f"[default]   New: {rel_path}")
            elif status == "modified":
                if cached:
                    changes.to_modify.append((new_info, cached))
                    logger.debug(f"[default]   Modified: {rel_path}")
                else:
                    # No cache but modified status - treat as new
                    changes.to_add.append(new_info)
                    logger.debug(f"[default]   New (no cache): {rel_path}")
            # else: unchanged - skip

        logger.debug(f"[default] Built changes in {(time.time()-start)*1000:.1f}ms: "
                  f"+{len(changes.to_add)} ~{len(changes.to_modify)} -{len(changes.to_delete)}")

        return changes

    def _apply_sync_changes(
        self,
        reter: ReterWrapper,
        changes: SyncChanges,
        rag_changed_files: Optional[Set[str]] = None,
    ) -> bool:
        """
        Apply sync changes from SourceStateManager to RETER.

        Args:
            reter: ReterWrapper instance
            changes: SyncChanges with files to add, modify, or delete
            rag_changed_files: Optional set of changed file paths for targeted RAG/markdown sync

        Returns:
            True if any changes were made
        """
        has_code_changes = changes.has_changes
        has_md_in_watcher = rag_changed_files and any(f.endswith(".md") for f in rag_changed_files)

        if not has_code_changes and not has_md_in_watcher:
            return False

        added_count = 0
        modified_count = 0
        deleted_count = 0
        errors = []

        # Track source IDs for RAG sync (keyed by language name)
        changed_sources: Dict[str, List[str]] = {}
        deleted_sources: Dict[str, List[str]] = {}

        # Map extension sets to language names for RAG tracking
        _ext_to_lang = [
            (self.PYTHON_EXTENSIONS, "python"),
            (self.JAVASCRIPT_EXTENSIONS, "javascript"),
            (self.HTML_EXTENSIONS, "html"),
            (self.CSHARP_EXTENSIONS, "csharp"),
            (self.C_EXTENSIONS, "c"),
            (self.CPP_EXTENSIONS, "cpp"),
            (self.JAVA_EXTENSIONS, "java"),
            (self.GO_EXTENSIONS, "go"),
            (self.RUST_EXTENSIONS, "rust"),
            (self.ERLANG_EXTENSIONS, "erlang"),
            (self.PHP_EXTENSIONS, "php"),
            (self.OBJC_EXTENSIONS, "objc"),
            (self.SWIFT_EXTENSIONS, "swift"),
            (self.VB6_EXTENSIONS, "vb6"),
            (self.SCALA_EXTENSIONS, "scala"),
            (self.HASKELL_EXTENSIONS, "haskell"),
            (self.KOTLIN_EXTENSIONS, "kotlin"),
            (self.R_EXTENSIONS, "r"),
            (self.RUBY_EXTENSIONS, "ruby"),
            (self.DART_EXTENSIONS, "dart"),
            (self.DELPHI_EXTENSIONS, "delphi"),
            (self.ADA_EXTENSIONS, "ada"),
            (self.LUA_EXTENSIONS, "lua"),
            (self.XAML_EXTENSIONS, "xaml"),
            (self.BASH_EXTENSIONS, "bash"),
        ]

        def _lang_for_ext(rel_path: str) -> Optional[str]:
            ext = Path(rel_path).suffix.lower()
            for ext_set, lang_name in _ext_to_lang:
                if ext in ext_set:
                    return lang_name
            return None

        def track_changed_source(rel_path: str, source_id: str) -> None:
            """Add source to appropriate changed list based on file extension."""
            lang = _lang_for_ext(rel_path)
            if lang:
                changed_sources.setdefault(lang, []).append(source_id)

        def track_deleted_source(rel_path: str, source_id: str) -> None:
            """Add source to appropriate deleted list based on file extension."""
            lang = _lang_for_ext(rel_path)
            if lang:
                deleted_sources.setdefault(lang, []).append(source_id)

        # Scan for Python package roots if we have changes
        if changes.to_add or changes.to_modify:
            self._package_roots = ReterWrapper.scan_package_roots(str(self._project_root))

        # Process deletions first (before entity accumulation)
        for file_info in changes.to_delete:
            if file_info.reter_source_id:
                if self._progress_callback is None:
                    logger.info(f"[default] Forgetting deleted file: {file_info.rel_path}")
                try:
                    reter.forget_source(file_info.reter_source_id)
                    deleted_count += 1
                    track_deleted_source(file_info.rel_path, file_info.reter_source_id)
                    # Remove from state
                    if self._source_state:
                        self._source_state.remove_file(file_info.rel_path)
                except Exception as e:
                    error_msg = f"Error forgetting {file_info.rel_path}: {type(e).__name__}: {e}"
                    if self._progress_callback is None:
                        logger.info(f"[default]   {error_msg}")
                    errors.append(error_msg)

        # Enable entity accumulation for batch loading of new/modified files
        # This is especially important for C++ where same entity appears in .h and .cpp
        use_accumulation = len(changes.to_add) + len(changes.to_modify) > 1
        if use_accumulation:
            reter.begin_entity_accumulation()

        try:
            # Calculate total files for progress
            total_files = len(changes.to_modify) + len(changes.to_add)
            current_file_idx = 0

            # Start file loading progress if we have a progress callback
            if self._progress_callback is not None and total_files > 0:
                self._progress_callback.set_phase("Loading code files...")
                self._progress_callback.start_file_loading(total_files)

            # Process modifications (forget then reload)
            for new_info, old_info in changes.to_modify:
                current_file_idx += 1
                if self._progress_callback is None:
                    logger.info(f"[default] Reloading modified file: {new_info.rel_path}")
                else:
                    self._progress_callback.update_file_progress(current_file_idx, total_files, new_info.rel_path)
                try:
                    # Forget old version
                    if old_info.reter_source_id:
                        reter.forget_source(old_info.reter_source_id)
                        track_deleted_source(old_info.rel_path, old_info.reter_source_id)

                    # Load new version
                    self._load_code_file(reter, new_info.abs_path, new_info.rel_path)
                    modified_count += 1

                    # Update state
                    source_id = f"{new_info.md5}|{new_info.rel_path}"
                    new_info.in_reter = True
                    new_info.reter_source_id = source_id
                    if self._source_state:
                        self._source_state.set_file(new_info)
                    track_changed_source(new_info.rel_path, source_id)

                except Exception as e:
                    error_msg = f"Error reloading {new_info.rel_path}: {type(e).__name__}: {e}"
                    if self._progress_callback is None:
                        logger.info(f"[default]   {error_msg}")
                    errors.append(error_msg)

            # Process additions
            # Debug: Log JS/TS files in to_add
            js_ts_in_to_add = [f for f in changes.to_add if any(f.rel_path.endswith(ext) for ext in self.JAVASCRIPT_EXTENSIONS)]
            if js_ts_in_to_add:
                logger.debug(f"[default] Found {len(js_ts_in_to_add)} JS/TS files in to_add: {[f.rel_path for f in js_ts_in_to_add[:5]]}")

            for file_info in changes.to_add:
                current_file_idx += 1
                is_js_ts = any(file_info.rel_path.endswith(ext) for ext in self.JAVASCRIPT_EXTENSIONS)
                if is_js_ts:
                    logger.debug(f"[default] Processing JS/TS file from to_add: {file_info.rel_path}")

                if self._progress_callback is None:
                    logger.info(f"[default] Adding new file: {file_info.rel_path}")
                else:
                    self._progress_callback.update_file_progress(current_file_idx, total_files, file_info.rel_path)
                try:
                    self._load_code_file(reter, file_info.abs_path, file_info.rel_path)
                    added_count += 1

                    # Update state
                    source_id = f"{file_info.md5}|{file_info.rel_path}"
                    file_info.in_reter = True
                    file_info.reter_source_id = source_id
                    if self._source_state:
                        self._source_state.set_file(file_info)
                        if is_js_ts:
                            logger.debug(f"[default] Saved JS/TS file to state: {file_info.rel_path}")
                    track_changed_source(file_info.rel_path, source_id)

                except Exception as e:
                    error_msg = f"Error loading {file_info.rel_path}: {type(e).__name__}: {e}"
                    if self._progress_callback is None:
                        logger.info(f"[default]   {error_msg}")
                    if is_js_ts:
                        logger.debug(f"[default] ERROR loading JS/TS file {file_info.rel_path}: {error_msg}")
                    errors.append(error_msg)

            # End file loading progress
            if self._progress_callback is not None and total_files > 0:
                self._progress_callback.end_file_loading()

        finally:
            # Finalize entity accumulation - merges duplicate entities
            if use_accumulation:
                self._finalize_entity_accumulation_with_progress(reter)

        if errors and self._progress_callback is None:
            logger.info(f"[default] Sync completed with {len(errors)} errors")

        changes_made = added_count > 0 or modified_count > 0 or deleted_count > 0

        if changes_made and self._progress_callback is None:
            logger.info(f"[default] Sync complete: +{added_count} ~{modified_count} -{deleted_count}")

        # Track modifications for compaction
        modification_ops = modified_count + deleted_count
        if modification_ops > 0:
            self._modification_count += modification_ops
            logger.debug(f"[default] Modification count: {self._modification_count} (threshold: {self.REBUILD_THRESHOLD})")

        # Sync with RAG index if configured (includes markdown change detection)
        if self._rag_manager and self._rag_manager.is_enabled and self._rag_manager.is_initialized:
            rag_changes = self._build_rag_sync_changes(
                changed_sources, deleted_sources, rag_changed_files
            )
            if rag_changes.has_changes():
                if self._progress_callback is None:
                    logger.info("[default] Syncing RAG index...")
                try:
                    rag_stats = self._rag_manager.sync_with_changes(
                        reter=reter,
                        project_root=self._project_root,
                        changes=rag_changes,
                    )
                    total_added = (
                        sum(rag_stats.get(f'{lang}_vectors_added', 0)
                            for lang in ["python", "javascript", "html", "csharp", "c", "cpp",
                                         "java", "go", "rust", "erlang", "php",
                                         "objc", "swift", "vb6", "scala", "haskell",
                                         "kotlin", "r", "ruby", "dart", "delphi",
                                         "ada", "lua", "xaml"])
                        + rag_stats.get('markdown_vectors_added', 0)
                    )
                    if self._progress_callback is None:
                        logger.info(
                            f"[default] RAG sync: +{total_added} vectors "
                            f"in {rag_stats.get('time_ms', 0)}ms"
                        )

                    # Mark files as indexed in RAG
                    if self._source_state:
                        for lang_sources in changed_sources.values():
                            for source_id in lang_sources:
                                if "|" in source_id:
                                    _, rel_path = source_id.split("|", 1)
                                    self._source_state.mark_in_rag(rel_path)

                except Exception as e:
                    if self._progress_callback is None:
                        logger.info(f"[default] RAG sync error: {e}")

        return changes_made

    # Supported file extensions for code analysis
    PYTHON_EXTENSIONS = {".py"}
    JAVASCRIPT_EXTENSIONS = {".js", ".mjs", ".jsx", ".ts", ".tsx"}
    HTML_EXTENSIONS = {".html", ".htm"}
    CSHARP_EXTENSIONS = {".cs"}
    C_EXTENSIONS = {".c"}
    CPP_EXTENSIONS = {".cpp", ".cc", ".cxx", ".c++", ".hpp", ".hh", ".hxx", ".h++", ".h"}
    JAVA_EXTENSIONS = {".java"}
    GO_EXTENSIONS = {".go"}
    RUST_EXTENSIONS = {".rs"}
    ERLANG_EXTENSIONS = {".erl", ".hrl"}
    PHP_EXTENSIONS = {".php"}
    OBJC_EXTENSIONS = {".m", ".mm"}
    SWIFT_EXTENSIONS = {".swift"}
    VB6_EXTENSIONS = {".bas", ".cls", ".frm"}
    SCALA_EXTENSIONS = {".scala", ".sc"}
    HASKELL_EXTENSIONS = {".hs", ".lhs"}
    KOTLIN_EXTENSIONS = {".kt", ".kts"}
    R_EXTENSIONS = {".r", ".R"}
    RUBY_EXTENSIONS = {".rb", ".rake", ".gemspec"}
    DART_EXTENSIONS = {".dart"}
    DELPHI_EXTENSIONS = {".pas", ".dpr", ".dpk", ".inc"}
    ADA_EXTENSIONS = {".adb", ".ads", ".ada"}
    LUA_EXTENSIONS = {".lua"}
    XAML_EXTENSIONS = {".xaml"}
    BASH_EXTENSIONS = {".sh", ".bash", ".zsh", ".ksh"}
    ALL_CODE_EXTENSIONS = (
        PYTHON_EXTENSIONS | JAVASCRIPT_EXTENSIONS | HTML_EXTENSIONS |
        CSHARP_EXTENSIONS | C_EXTENSIONS | CPP_EXTENSIONS | JAVA_EXTENSIONS |
        GO_EXTENSIONS | RUST_EXTENSIONS | ERLANG_EXTENSIONS |
        PHP_EXTENSIONS | OBJC_EXTENSIONS | SWIFT_EXTENSIONS |
        VB6_EXTENSIONS | SCALA_EXTENSIONS | HASKELL_EXTENSIONS |
        KOTLIN_EXTENSIONS | R_EXTENSIONS | RUBY_EXTENSIONS |
        DART_EXTENSIONS | DELPHI_EXTENSIONS | ADA_EXTENSIONS |
        LUA_EXTENSIONS | XAML_EXTENSIONS | BASH_EXTENSIONS
    )

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
            logger.debug(f"[default] Found {len(self._package_roots)} Python package roots: {sorted(self._package_roots)[:10]}...")

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
                    if not self._progress_callback:
                        logger.info(f"Warning: Failed to read {code_file}: {e}")

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

        # Check gitignore patterns first (use fast method if available)
        if self._gitignore_parser is not None:
            if self._gitignore_parser.is_ignored_fast(path_str):
                return True

        # Check explicit exclude patterns
        for pattern in self._exclude_patterns:
            # Simple glob pattern matching
            if self._matches_pattern(path_str, pattern):
                return True

        return False

    def _is_excluded_for_scan(self, rel_path: Path) -> bool:
        """
        Fast exclusion check for use during filesystem scan.

        Uses pre-loaded gitignore patterns for maximum performance.

        Args:
            rel_path: Relative path to check

        Returns:
            True if file should be excluded
        """
        path_str = str(rel_path).replace('\\', '/')

        # Check gitignore patterns (pre-loaded, no lazy loading)
        if self._gitignore_parser is not None:
            if self._gitignore_parser.is_ignored_fast(path_str):
                return True

        # Check explicit exclude patterns
        for pattern in self._exclude_patterns:
            if self._matches_pattern(path_str, pattern):
                return True

        return False

    def is_excluded(self, rel_path_str: str) -> bool:
        """Check if a relative path should be skipped.

        Checks both include patterns (if set, file must match at least one)
        and exclude patterns (.gitignore + project_exclude).
        Same rules as code files.

        Args:
            rel_path_str: Forward-slash normalized relative path string

        Returns:
            True if the path should be excluded
        """
        # If include patterns are set, file must match at least one
        if self._include_patterns:
            if not any(self._matches_pattern(rel_path_str, p) for p in self._include_patterns):
                return True

        return self._is_excluded(Path(rel_path_str))

    def _detect_markdown_changes(
        self,
        changed_files: Optional[set] = None,
    ) -> Tuple[List[str], List[str]]:
        """Detect changed and deleted markdown files for RAG sync.

        Compares current markdown files on disk against the RAG index's
        tracked markdown files (via rag_files.json MD5 hashes).

        Args:
            changed_files: If provided, only check these files (targeted sync).
                           If None, full scan.

        Returns:
            (changed_rel_paths, deleted_rel_paths) for markdown files
        """
        if not self._rag_manager:
            return [], []

        # Get indexed markdown MD5s from RAG manager's tracked files
        indexed_markdown: Dict[str, str] = {}
        for key, md5 in self._rag_manager._indexed_files.items():
            if key.startswith("md:"):
                indexed_markdown[key[3:]] = md5

        # Scan current markdown files
        current_markdown: Dict[str, str] = {}  # rel_path -> md5
        md_files = self._rag_manager._scan_markdown_files(self._project_root)
        for rel_path in md_files:
            rel_path_normalized = rel_path.replace("\\", "/")

            # If targeted sync and file not in changed set, use cached MD5
            if changed_files and rel_path_normalized not in changed_files:
                cached_md5 = indexed_markdown.get(rel_path_normalized)
                if cached_md5:
                    current_markdown[rel_path_normalized] = cached_md5
                continue

            try:
                abs_path = self._project_root / rel_path
                content = abs_path.read_text(encoding='utf-8')
                md5_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
                current_markdown[rel_path_normalized] = md5_hash
            except Exception:
                pass

        # Diff
        changed: List[str] = []
        deleted: List[str] = []

        for rel_path, current_md5 in current_markdown.items():
            indexed_md5 = indexed_markdown.get(rel_path)
            if indexed_md5 is None or current_md5 != indexed_md5:
                changed.append(rel_path)

        for rel_path in indexed_markdown:
            if rel_path not in current_markdown:
                deleted.append(rel_path)

        if changed or deleted:
            logger.debug(f"[default] Markdown changes: +{len(changed)} -{len(deleted)}")

        return changed, deleted

    def _build_rag_sync_changes(
        self,
        changed_sources: Dict[str, List[str]],
        deleted_sources: Dict[str, List[str]],
        changed_files: Optional[set] = None,
    ) -> RAGSyncChanges:
        """Build RAGSyncChanges including markdown changes.

        Args:
            changed_sources: Dict of language -> list of changed source IDs
            deleted_sources: Dict of language -> list of deleted source IDs
            changed_files: Optional set of changed file paths for targeted sync

        Returns:
            RAGSyncChanges with code and markdown changes
        """
        _all_rag_langs = [
            "python", "javascript", "html", "csharp", "c", "cpp",
            "java", "go", "rust", "erlang", "php",
            "objc", "swift", "vb6", "scala", "haskell", "kotlin", "r", "ruby",
            "dart", "delphi", "ada", "lua", "xaml",
        ]

        # Detect markdown changes
        md_changed, md_deleted = self._detect_markdown_changes(changed_files)

        lang_kwargs = {
            lang: LanguageSourceChanges(
                changed_sources.get(lang, []),
                deleted_sources.get(lang, []),
            )
            for lang in _all_rag_langs
        }
        lang_kwargs["markdown"] = LanguageSourceChanges(md_changed, md_deleted)

        return RAGSyncChanges(**lang_kwargs)

    def _matches_pattern(self, path: str, pattern: str) -> bool:
        """Simple glob pattern matching - delegates to shared utility."""
        return matches_pattern(path, pattern)

    def _get_existing_sources(self, reter: ReterWrapper) -> Dict[str, Tuple[str, str]]:
        """
        Get existing sources from RETER instance.

        Returns:
            Dict mapping normalized relative path (forward slashes) to (md5_hash, original_source_id)
        """
        sources: Dict[str, Tuple[str, str]] = {}

        # Get all sources
        all_sources, _ = reter.get_all_sources()
        logger.debug(f"[default] get_all_sources returned {len(all_sources)} sources")

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
                    logger.debug(f"[default]   existing: {rel_path_normalized} -> md5={md5_hash[:8]}...")

        logger.debug(f"[default] Found {len(sources)} code sources in RETER")
        return sources

    def _load_code_file(self, reter: ReterWrapper, abs_path: str, rel_path: str) -> None:
        """
        Load a code file using the appropriate loader based on file extension.

        Args:
            reter: ReterWrapper instance
            abs_path: Absolute path to the file
            rel_path: Relative path (for logging)
        """
        ext = Path(rel_path).suffix.lower()
        base = str(self._project_root)

        if ext in self.PYTHON_EXTENSIONS:
            reter.load_python_file(abs_path, base, self._package_roots)
        elif ext in self.JAVASCRIPT_EXTENSIONS:
            reter.load_javascript_file(abs_path, base)
        elif ext in self.HTML_EXTENSIONS:
            reter.load_html_file(abs_path, base)
        elif ext in self.CSHARP_EXTENSIONS:
            reter.load_csharp_file(abs_path, base)
        elif ext in self.C_EXTENSIONS:
            reter.load_c_file(abs_path, base)
        elif ext in self.CPP_EXTENSIONS:
            reter.load_cpp_file(abs_path, base)
        elif ext in self.JAVA_EXTENSIONS:
            reter.load_java_file(abs_path, base)
        elif ext in self.GO_EXTENSIONS:
            reter.load_go_file(abs_path, base)
        elif ext in self.RUST_EXTENSIONS:
            reter.load_rust_file(abs_path, base)
        elif ext in self.ERLANG_EXTENSIONS:
            reter.load_erlang_file(abs_path, base)
        elif ext in self.PHP_EXTENSIONS:
            reter.load_php_file(abs_path, base)
        elif ext in self.OBJC_EXTENSIONS:
            reter.load_objc_file(abs_path, base)
        elif ext in self.SWIFT_EXTENSIONS:
            reter.load_swift_file(abs_path, base)
        elif ext in self.VB6_EXTENSIONS:
            reter.load_vb6_file(abs_path, base)
        elif ext in self.SCALA_EXTENSIONS:
            reter.load_scala_file(abs_path, base)
        elif ext in self.HASKELL_EXTENSIONS:
            reter.load_haskell_file(abs_path, base)
        elif ext in self.KOTLIN_EXTENSIONS:
            reter.load_kotlin_file(abs_path, base)
        elif ext in self.R_EXTENSIONS:
            reter.load_r_file(abs_path, base)
        elif ext in self.RUBY_EXTENSIONS:
            reter.load_ruby_file(abs_path, base)
        elif ext in self.DART_EXTENSIONS:
            reter.load_dart_file(abs_path, base)
        elif ext in self.DELPHI_EXTENSIONS:
            reter.load_delphi_file(abs_path, base)
        elif ext in self.ADA_EXTENSIONS:
            reter.load_ada_file(abs_path, base)
        elif ext in self.LUA_EXTENSIONS:
            reter.load_lua_file(abs_path, base)
        elif ext in self.XAML_EXTENSIONS:
            reter.load_xaml_file(abs_path, base)
        elif ext in self.BASH_EXTENSIONS:
            reter.load_bash_file(abs_path, base)
        else:
            raise ValueError(f"Unsupported file type: {rel_path}")

    def _finalize_entity_accumulation_with_progress(
        self,
        reter: ReterWrapper,
        batch_size: int = 1000
    ) -> int:
        """
        Finalize entity accumulation with progress reporting.

        Args:
            reter: ReterWrapper instance with active entity accumulation
            batch_size: Number of facts to process per batch (for progress updates)

        Returns:
            Total number of entities processed
        """
        accumulated = reter.accumulated_entity_count()
        if accumulated == 0:
            reter.end_entity_accumulation()
            return 0

        # Start entity processing progress
        if self._progress_callback:
            self._progress_callback.start_entity_processing(accumulated)
        else:
            logger.debug(f"[default] Inserting entities: 0/{accumulated} (0%)")

        def on_progress(processed: int, total: int) -> None:
            if self._progress_callback:
                self._progress_callback.update_entity_progress(processed, total)
            else:
                pct = 100 * processed // total if total > 0 else 0
                logger.debug(f"[default] Inserting entities: {processed}/{total} ({pct}%)")

        total = reter.end_entity_accumulation_with_progress(
            batch_size=batch_size,
            progress_callback=on_progress
        )

        # End entity processing progress
        if self._progress_callback:
            self._progress_callback.end_entity_processing()
        else:
            logger.info(f"[default] Entity accumulation: {total} unique entities merged")

        return total

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

        # Track source IDs for RAG sync (keyed by language name)
        changed_sources: Dict[str, List[str]] = {}
        deleted_sources: Dict[str, List[str]] = {}

        errors = []  # Track errors but continue processing

        # Map extension sets to language names for RAG tracking
        _ext_to_lang = [
            (self.PYTHON_EXTENSIONS, "python"),
            (self.JAVASCRIPT_EXTENSIONS, "javascript"),
            (self.HTML_EXTENSIONS, "html"),
            (self.CSHARP_EXTENSIONS, "csharp"),
            (self.C_EXTENSIONS, "c"),
            (self.CPP_EXTENSIONS, "cpp"),
            (self.JAVA_EXTENSIONS, "java"),
            (self.GO_EXTENSIONS, "go"),
            (self.RUST_EXTENSIONS, "rust"),
            (self.ERLANG_EXTENSIONS, "erlang"),
            (self.PHP_EXTENSIONS, "php"),
            (self.OBJC_EXTENSIONS, "objc"),
            (self.SWIFT_EXTENSIONS, "swift"),
            (self.VB6_EXTENSIONS, "vb6"),
            (self.SCALA_EXTENSIONS, "scala"),
            (self.HASKELL_EXTENSIONS, "haskell"),
            (self.KOTLIN_EXTENSIONS, "kotlin"),
            (self.R_EXTENSIONS, "r"),
            (self.RUBY_EXTENSIONS, "ruby"),
            (self.DART_EXTENSIONS, "dart"),
            (self.DELPHI_EXTENSIONS, "delphi"),
            (self.ADA_EXTENSIONS, "ada"),
            (self.LUA_EXTENSIONS, "lua"),
            (self.XAML_EXTENSIONS, "xaml"),
            (self.BASH_EXTENSIONS, "bash"),
        ]

        def _lang_for_ext(rel_path: str) -> Optional[str]:
            ext = Path(rel_path).suffix.lower()
            for ext_set, lang_name in _ext_to_lang:
                if ext in ext_set:
                    return lang_name
            return None

        def track_changed_source(rel_path: str, source_id: str) -> None:
            """Add source to appropriate changed list based on file extension."""
            lang = _lang_for_ext(rel_path)
            if lang:
                changed_sources.setdefault(lang, []).append(source_id)

        def track_deleted_source(rel_path: str, source_id: str) -> None:
            """Add source to appropriate deleted list based on file extension."""
            lang = _lang_for_ext(rel_path)
            if lang:
                deleted_sources.setdefault(lang, []).append(source_id)

        # Debug: log first few current files for comparison
        logger.debug(f"[default] _sync_files: {len(current_files)} current files, {len(existing_sources)} existing sources")
        for i, (rel_path, (abs_path, current_md5)) in enumerate(list(current_files.items())[:5]):
            logger.debug(f"[default]   current[{i}]: {rel_path} -> md5={current_md5[:8]}...")

        # Count files to add/modify for deciding on entity accumulation
        files_to_load = sum(1 for rel_path in current_files if rel_path not in existing_sources or
                           existing_sources[rel_path][0] != current_files[rel_path][1])
        use_accumulation = files_to_load > 1
        if use_accumulation:
            reter.begin_entity_accumulation()

        try:
            # Count files to process for progress reporting
            files_to_process = list(current_files.items())
            total_files = len(files_to_process)
            processed_files = 0

            # Start file loading progress if callback is provided
            if self._progress_callback:
                self._progress_callback.start_file_loading(total_files)

            # Find new and modified files
            for rel_path, (abs_path, current_md5) in files_to_process:
                processed_files += 1

                # Update progress callback
                if self._progress_callback:
                    self._progress_callback.update_file_progress(processed_files, total_files, rel_path)

                if rel_path not in existing_sources:
                    # New file - load it using the appropriate loader
                    logger.debug(f"[default] NEW file (not in existing): {rel_path}")
                    if self._progress_callback is None:
                        logger.info(f"[default] Adding new file: {rel_path}")
                    try:
                        self._load_code_file(reter, abs_path, rel_path)
                        added_count += 1
                        if self._progress_callback is None:
                            logger.info(f"[default]   Added {rel_path}")
                        # Track for RAG: construct source_id from md5 and rel_path
                        new_source_id = f"{current_md5}|{rel_path}"
                        track_changed_source(rel_path, new_source_id)
                    except Exception as e:
                        # Log error but continue with other files (don't re-raise)
                        error_msg = f"Error loading {rel_path}: {type(e).__name__}: {e}"
                        if self._progress_callback is None:
                            logger.info(f"[default]   {error_msg}")
                        errors.append(error_msg)
                else:
                    old_md5, old_source_id = existing_sources[rel_path]
                    if old_md5 != current_md5:
                        # Modified file - forget and reload
                        logger.debug(f"[default] MD5 MISMATCH: {rel_path} old={old_md5[:8]}... new={current_md5[:8]}...")
                        if self._progress_callback is None:
                            logger.info(f"[default] Reloading modified file: {rel_path}")
                        try:
                            reter.forget_source(old_source_id)
                            self._load_code_file(reter, abs_path, rel_path)
                            modified_count += 1
                            if self._progress_callback is None:
                                logger.info(f"[default]   Reloaded {rel_path}")
                            # Track for RAG: old source deleted, new source added
                            track_deleted_source(rel_path, old_source_id)
                            new_source_id = f"{current_md5}|{rel_path}"
                            track_changed_source(rel_path, new_source_id)
                        except Exception as e:
                            # Log error but continue with other files (don't re-raise)
                            error_msg = f"Error reloading {rel_path}: {type(e).__name__}: {e}"
                            if self._progress_callback is None:
                                logger.info(f"[default]   {error_msg}")
                            errors.append(error_msg)

            # End file loading progress
            if self._progress_callback:
                self._progress_callback.end_file_loading()

        finally:
            # Finalize entity accumulation - merges duplicate entities
            if use_accumulation:
                self._finalize_entity_accumulation_with_progress(reter)

        # Find deleted files (after accumulation finalized)
        for rel_path, (old_md5, old_source_id) in existing_sources.items():
            if rel_path not in current_files:
                # Deleted file - forget it
                if self._progress_callback is None:
                    logger.info(f"[default] Forgetting deleted file: {rel_path}")
                try:
                    reter.forget_source(old_source_id)
                    deleted_count += 1
                    if self._progress_callback is None:
                        logger.info(f"[default]   Forgot {rel_path}")
                    # Track for RAG
                    track_deleted_source(rel_path, old_source_id)
                except Exception as e:
                    # Log error but continue with other files (don't re-raise)
                    error_msg = f"Error forgetting {rel_path}: {type(e).__name__}: {e}"
                    if self._progress_callback is None:
                        logger.info(f"[default]   {error_msg}")
                    errors.append(error_msg)

        if errors and self._progress_callback is None:
            logger.info(f"[default] Sync completed with {len(errors)} errors")

        changes_made = added_count > 0 or modified_count > 0 or deleted_count > 0

        if changes_made and self._progress_callback is None:
            logger.info(f"[default] Sync complete: +{added_count} ~{modified_count} -{deleted_count}")

        # Sync with RAG index if configured (includes markdown change detection)
        if self._rag_manager and self._rag_manager.is_enabled and self._rag_manager.is_initialized:
            # Full scan — no targeted file set for markdown
            rag_changes = self._build_rag_sync_changes(
                changed_sources, deleted_sources, changed_files=None
            )
            if rag_changes.has_changes():
                if self._progress_callback is None:
                    logger.info("[default] Syncing RAG index...")
                try:
                    rag_stats = self._rag_manager.sync_with_changes(
                        reter=reter,
                        project_root=self._project_root,
                        changes=rag_changes,
                    )
                    total_added = (
                        sum(rag_stats.get(f'{lang}_vectors_added', 0)
                            for lang in ["python", "javascript", "html", "csharp", "c", "cpp",
                                         "java", "go", "rust", "erlang", "php",
                                         "objc", "swift", "vb6", "scala", "haskell",
                                         "kotlin", "r", "ruby", "dart", "delphi",
                                         "ada", "lua", "xaml"])
                        + rag_stats.get('markdown_vectors_added', 0)
                    )
                    if self._progress_callback is None:
                        logger.info(
                            f"[default] RAG sync: +{total_added} vectors "
                            f"in {rag_stats.get('time_ms', 0)}ms"
                        )
                except Exception as e:
                    if self._progress_callback is None:
                        logger.info(f"[default] RAG sync error: {e}")

        # Track modifications for compaction (only forget operations cause bloat)
        modification_ops = modified_count + deleted_count
        if modification_ops > 0:
            self._modification_count += modification_ops
            logger.debug(f"[default] Modification count: {self._modification_count} (threshold: {self.REBUILD_THRESHOLD})")

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

        Uses entity accumulation mode to deduplicate entities that appear in
        multiple files (e.g., C++ method declared in .h and defined in .cpp).

        Args:
            current_files: Dict mapping rel_path to (abs_path, md5_hash)

        Returns:
            Fresh ReterWrapper instance with all files loaded
        """
        import time
        start = time.time()
        if self._progress_callback is None:
            logger.info(f"[default] Force rebuilding to compact RETE network ({self._modification_count} modifications accumulated)...")

        # Create fresh ReterWrapper with new RETE network
        fresh_reter = ReterWrapper()

        # Enable entity accumulation for cross-file deduplication
        # This is especially important for C++ where methods are declared in .h
        # and defined in .cpp - both would otherwise create duplicate entities
        fresh_reter.begin_entity_accumulation()

        # Load all current files
        loaded = 0
        errors = []
        try:
            for rel_path, (abs_path, _) in current_files.items():
                try:
                    self._load_code_file(fresh_reter, abs_path, rel_path)
                    loaded += 1
                except Exception as e:
                    error_msg = f"Error loading {rel_path}: {type(e).__name__}: {e}"
                    if self._progress_callback is None:
                        logger.info(f"[default]   {error_msg}")
                    errors.append(error_msg)
        finally:
            # Finalize accumulated entities - creates merged facts
            self._finalize_entity_accumulation_with_progress(fresh_reter)

        # Reset modification count
        self._modification_count = 0

        if self._progress_callback is None:
            logger.info(f"[default] Rebuild complete: loaded {loaded} files in {time.time()-start:.2f}s")
            if errors:
                logger.info(f"[default] Rebuild had {len(errors)} errors")

        return fresh_reter
