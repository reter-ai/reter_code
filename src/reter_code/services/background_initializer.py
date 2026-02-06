"""
Background Initializer Service

Handles background initialization of the default RETER instance.
Runs in a separate thread to avoid blocking MCP server startup.

Note: The embedding model is NOT loaded here - it loads lazily on first use
during RAG search or reindex operations.
"""

import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, List, TYPE_CHECKING

from .initialization_progress import (
    get_instance_progress,
    InitStatus,
    InitPhase,
    SyncStatus,
    SyncPhase,
)
from ..reter_wrapper import debug_log

from .rag_index_manager import SyncChanges as RAGSyncChanges, LanguageSourceChanges

# Re-export BackgroundSyncTask for backward compatibility
from .background_sync_task import BackgroundSyncTask

__all__ = ["BackgroundInitializer", "BackgroundSyncTask"]

if TYPE_CHECKING:
    from .default_instance_manager import DefaultInstanceManager
    from .rag_index_manager import RAGIndexManager
    from ..reter_wrapper import ReterWrapper


class BackgroundInitializer:
    """
    Handles background initialization of the default RETER instance.

    ::: This is-in-layer Service-Layer.
    ::: This is a manager.
    ::: This is-in-process Main-Process.
    ::: This is stateful.

    Runs in a separate thread to avoid blocking MCP server startup.

    Phases:
    1. Loading Python files into RETER (50% progress)
    2. Building RAG index if enabled (45% progress)
       - Embedding model loads lazily during first generate_embedding() call
    3. Complete (5% progress)
    """

    def __init__(
        self,
        instance_manager: "DefaultInstanceManager",
        rag_manager: Optional["RAGIndexManager"],
        project_root: Path
    ):
        """
        Initialize the background initializer.

        Args:
            instance_manager: DefaultInstanceManager for file loading
            rag_manager: Optional RAGIndexManager for RAG indexing
            project_root: Project root directory
        """
        self._instance_manager = instance_manager
        self._rag_manager = rag_manager
        self._project_root = project_root
        self._thread: Optional[threading.Thread] = None
        self._progress = get_instance_progress()

    def start(self, blocking: bool = False) -> None:
        """
        Start initialization.

        Args:
            blocking: If True, run synchronously (blocks until complete).
                     If False, run in background thread (non-blocking).
        """
        if self._thread is not None and self._thread.is_alive():
            debug_log("[BackgroundInit] Already running, skipping")
            return

        debug_log(f"[BackgroundInit] start() called with blocking={blocking}")

        self._progress.update(
            init_status=InitStatus.INITIALIZING,
            init_phase=InitPhase.PENDING,
            init_progress=0.0,
            init_message="Starting initialization...",
            init_started_at=datetime.now(),
            init_error=None,
            init_completed_at=None
        )

        if blocking:
            # Run synchronously (blocking)
            debug_log("[BackgroundInit] Starting synchronous initialization...")
            self._run_initialization(blocking=True)
            debug_log("[BackgroundInit] Synchronous initialization complete")
        else:
            # Run in background thread (non-blocking)
            self._thread = threading.Thread(
                target=self._run_initialization,
                name="ReterBackgroundInit",
                daemon=True  # Don't block process exit
            )
            self._thread.start()
            debug_log("[BackgroundInit] Started background initialization thread")

    def is_running(self) -> bool:
        """Check if background initialization is currently running."""
        return self._thread is not None and self._thread.is_alive()

    def wait(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for initialization to complete.

        Args:
            timeout: Maximum seconds to wait (None = wait forever)

        Returns:
            True if initialization completed, False if timeout
        """
        if self._thread is None:
            return True
        self._thread.join(timeout=timeout)
        return not self._thread.is_alive()

    def _run_initialization(self, blocking: bool = False) -> None:
        """Main initialization logic (runs in background thread or synchronously)."""
        import traceback

        try:
            # Phase 1: Load Python files into RETER
            debug_log("[BackgroundInit] Phase 1: Loading Python files...")
            self._phase_load_python()
            debug_log("[BackgroundInit] Phase 1 complete")

            # Phase 2: Build RAG index (if enabled)
            # Note: Embedding model loads lazily on first generate_embedding() call
            if self._rag_manager and self._rag_manager.is_enabled:
                debug_log("[BackgroundInit] Phase 2: Building RAG index...")
                self._phase_build_rag_index()
                debug_log("[BackgroundInit] Phase 2 complete")

            # Complete
            self._progress.update(
                init_status=InitStatus.READY,
                init_phase=InitPhase.COMPLETE,
                init_progress=1.0,
                init_message="Initialization complete",
                init_completed_at=datetime.now()
            )
            elapsed = (datetime.now() - self._progress.init_started_at).total_seconds()
            debug_log(f"[BackgroundInit] Initialization complete in {elapsed:.1f}s!")

        except Exception as e:
            debug_log(f"[BackgroundInit] ERROR: {e}")
            debug_log(f"[BackgroundInit] Traceback: {traceback.format_exc()}")
            self._progress.update(
                init_status=InitStatus.ERROR,
                init_error=str(e),
                init_message=f"Initialization failed: {e}"
            )
            # Re-raise in blocking mode so caller knows initialization failed
            if blocking:
                raise

    def _phase_load_python(self) -> None:
        """Phase 1: Load Python files into RETER."""
        debug_log("[BackgroundInit] Phase 1: Loading Python files...")
        self._progress.update(
            init_phase=InitPhase.LOADING_PYTHON,
            init_progress=0.05,
            init_message="Scanning Python files..."
        )

        # Use the instance manager to sync files
        # This handles new, modified, and deleted files
        from .state_persistence import StatePersistenceService

        # Get or create the default RETER instance
        reter = self._instance_manager._persistence.instance_manager.get_or_create_instance("default")

        # Sync files with progress updates
        self._sync_python_files_with_progress(reter)

        debug_log("[BackgroundInit] Phase 1 complete")

    def _sync_python_files_with_progress(self, reter: "ReterWrapper") -> None:
        """Sync Python files with progress updates."""
        # Scan project files
        current_files = self._instance_manager._scan_project_files()
        total_files = len(current_files)

        self._progress.update(
            python_files_total=total_files,
            python_files_loaded=0,
            init_message=f"Found {total_files} Python files"
        )

        if total_files == 0:
            self._progress.update(
                init_progress=0.50,
                init_message="No Python files found"
            )
            return

        # Get existing sources
        existing_sources = self._instance_manager._get_existing_sources(reter)

        # Track counts
        loaded = 0
        added = 0
        modified = 0
        deleted = 0

        # Process new and modified files
        for rel_path, (abs_path, current_md5) in current_files.items():
            if rel_path not in existing_sources:
                # New file
                try:
                    reter.load_python_file(abs_path, str(self._project_root))
                    added += 1
                except Exception as e:
                    debug_log(f"[BackgroundInit] Error loading {rel_path}: {e}")
            else:
                old_md5, old_source_id = existing_sources[rel_path]
                if old_md5 != current_md5:
                    # Modified file
                    try:
                        reter.forget_source(old_source_id)
                        reter.load_python_file(abs_path, str(self._project_root))
                        modified += 1
                    except Exception as e:
                        debug_log(f"[BackgroundInit] Error reloading {rel_path}: {e}")

            loaded += 1

            # Update progress every 10 files or at end
            if loaded % 10 == 0 or loaded == total_files:
                # Phase 1 = 0-50% progress
                progress = 0.05 + (0.45 * loaded / total_files)
                self._progress.update(
                    python_files_loaded=loaded,
                    init_progress=progress,
                    init_message=f"Loaded {loaded}/{total_files} Python files"
                )

        # Process deleted files
        for rel_path, (old_md5, old_source_id) in existing_sources.items():
            if rel_path not in current_files:
                try:
                    reter.forget_source(old_source_id)
                    deleted += 1
                except Exception as e:
                    debug_log(f"[BackgroundInit] Error forgetting {rel_path}: {e}")

        # Save snapshot if changes were made
        if added > 0 or modified > 0 or deleted > 0:
            debug_log(f"[BackgroundInit] Changes: +{added} ~{modified} -{deleted}")
            snapshot_path = self._instance_manager._persistence.snapshots_dir / ".default.reter"

            # Create progress callback for save operation
            def save_progress(percent: int) -> None:
                self._progress.update(init_message=f"Saving snapshot... {percent}%")

            # CRITICAL: Use incremental save for hybrid mode
            if reter.is_hybrid_mode():
                reter.save_incremental()
            else:
                reter.save_network(str(snapshot_path), save_progress)
            reter.mark_clean()

        self._progress.update(
            init_progress=0.50,
            init_message=f"Loaded {total_files} Python files (+{added} ~{modified} -{deleted})"
        )

    def _phase_build_rag_index(self) -> None:
        """Phase 2: Sync RAG index with current files.

        Uses incremental sync (MD5-based) instead of full reindex:
        - Loads existing index from disk
        - Compares MD5 hashes to find changed/new/deleted files
        - Only reindexes what changed

        Note: The embedding model loads lazily on the first call to
        generate_embedding(). This happens automatically if any files
        need to be indexed.
        """
        debug_log("[BackgroundInit] Phase 2: Syncing RAG index...")
        self._progress.update(
            init_phase=InitPhase.BUILDING_RAG_INDEX,
            init_progress=0.50,
            init_message="Syncing RAG index (loading existing, checking for changes)..."
        )

        # Get RETER instance for querying entities
        reter = self._instance_manager._persistence.instance_manager.get_or_create_instance("default")

        # Define progress callback for indexing new/modified files
        def progress_callback(current: int, total: int, phase: str):
            self._progress.update(
                rag_vectors_total=total,
                rag_vectors_indexed=current,
                # Phase 2 = 50-95% progress
                init_progress=0.50 + (0.45 * current / max(total, 1)),
                init_message=f"Indexing: {current}/{total} ({phase})"
            )

        # Sync sources - loads existing index, only indexes what changed
        try:
            stats = self._rag_manager.sync_sources(
                reter=reter,
                project_root=self._project_root,
                progress_callback=progress_callback
            )

            total_vectors = stats.get("total_vectors", 0)
            unchanged = stats.get("python_unchanged", 0) + stats.get("markdown_unchanged", 0)
            added = stats.get("python_added", 0) + stats.get("markdown_added", 0)
            removed = stats.get("python_removed", 0) + stats.get("markdown_removed", 0)

            if added == 0 and removed == 0:
                msg = f"RAG index ready: {total_vectors} vectors (no changes)"
            else:
                msg = f"RAG index ready: {total_vectors} vectors (+{added} -{removed} ={unchanged})"

            self._progress.update(
                init_progress=0.95,
                rag_vectors_indexed=total_vectors,
                init_message=msg
            )
            debug_log(f"[BackgroundInit] Phase 2 complete: {stats}")

        except Exception as e:
            debug_log(f"[BackgroundInit] RAG sync failed: {e}")
            import traceback
            debug_log(f"[BackgroundInit] Traceback: {traceback.format_exc()}")
            # Don't fail the whole initialization for RAG errors
            self._progress.update(
                init_progress=0.95,
                init_message=f"RAG sync failed: {e} (continuing without RAG)"
            )

