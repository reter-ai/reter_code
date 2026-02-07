"""
Background Sync Task Service

Handles background file synchronization when file changes are detected.
Triggered via file watcher or manual check.
"""

import threading
from datetime import datetime
from pathlib import Path
from typing import Optional, TYPE_CHECKING

from .initialization_progress import (
    get_instance_progress,
    SyncStatus,
    SyncPhase,
)
from ..logging_config import configure_logger_for_debug_trace
from .rag_types import SyncChanges as RAGSyncChanges, LanguageSourceChanges

logger = configure_logger_for_debug_trace(__name__)

if TYPE_CHECKING:
    from .default_instance_manager import DefaultInstanceManager
    from .rag_index_manager import RAGIndexManager
    from ..reter_wrapper import ReterWrapper


class BackgroundSyncTask:
    """
    Handles background file synchronization.

    ::: This is-in-layer Service-Layer.
    ::: This is a task.
    ::: This is-in-process Main-Process.
    ::: This is stateful.

    Triggered when file changes are detected (via file watcher or manual check).
    """

    def __init__(
        self,
        instance_manager: "DefaultInstanceManager",
        rag_manager: Optional["RAGIndexManager"],
        reter: "ReterWrapper",
        project_root: Path,
        changes: dict  # {new: [...], modified: [...], deleted: [...]}
    ):
        """
        Initialize the background sync task.

        Args:
            instance_manager: DefaultInstanceManager for file operations
            rag_manager: Optional RAGIndexManager for RAG sync
            reter: ReterWrapper instance to sync
            project_root: Project root directory
            changes: Dict with new, modified, deleted file lists
        """
        self._instance_manager = instance_manager
        self._rag_manager = rag_manager
        self._reter = reter
        self._project_root = project_root
        self._changes = changes
        self._thread: Optional[threading.Thread] = None
        self._progress = get_instance_progress()

    def start(self) -> None:
        """Start background sync (non-blocking)."""
        if self._thread is not None and self._thread.is_alive():
            logger.debug("[BackgroundSync] Already running, skipping")
            return

        total_changes = (
            len(self._changes.get("new", [])) +
            len(self._changes.get("modified", [])) +
            len(self._changes.get("deleted", []))
        )

        self._progress.update(
            sync_status=SyncStatus.SYNCING,
            sync_phase=SyncPhase.SCANNING,
            sync_progress=0.0,
            sync_message=f"Starting sync of {total_changes} files...",
            sync_started_at=datetime.now(),
            sync_error=None,
            sync_completed_at=None,
            files_to_sync=total_changes,
            files_synced=0
        )

        self._thread = threading.Thread(
            target=self._run_sync,
            name="ReterBackgroundSync",
            daemon=True
        )
        self._thread.start()
        logger.debug(f"[BackgroundSync] Started sync of {total_changes} files")

    def is_running(self) -> bool:
        """Check if sync is currently running."""
        return self._thread is not None and self._thread.is_alive()

    def _run_sync(self) -> None:
        """Main sync logic (runs in background thread)."""
        try:
            files_synced = 0
            total = self._progress.files_to_sync

            # Phase 1: Load new files
            new_files = self._changes.get("new", [])
            if new_files:
                self._progress.update(
                    sync_phase=SyncPhase.LOADING,
                    sync_message=f"Loading {len(new_files)} new files..."
                )
                for f in new_files:
                    try:
                        self._reter.load_python_file(f["path"], str(self._project_root))
                    except Exception as e:
                        logger.debug(f"[BackgroundSync] Error loading {f['rel_path']}: {e}")
                    files_synced += 1
                    self._update_sync_progress(files_synced, total)

            # Phase 2: Reload modified files
            modified_files = self._changes.get("modified", [])
            if modified_files:
                self._progress.update(
                    sync_phase=SyncPhase.LOADING,
                    sync_message=f"Reloading {len(modified_files)} modified files..."
                )
                for f in modified_files:
                    try:
                        self._reter.forget_source(f["old_source_id"])
                        self._reter.load_python_file(f["path"], str(self._project_root))
                    except Exception as e:
                        logger.debug(f"[BackgroundSync] Error reloading {f['rel_path']}: {e}")
                    files_synced += 1
                    self._update_sync_progress(files_synced, total)

            # Phase 3: Forget deleted files
            deleted_files = self._changes.get("deleted", [])
            if deleted_files:
                self._progress.update(
                    sync_phase=SyncPhase.FORGETTING,
                    sync_message=f"Forgetting {len(deleted_files)} deleted files..."
                )
                for f in deleted_files:
                    try:
                        self._reter.forget_source(f["source_id"])
                    except Exception as e:
                        logger.debug(f"[BackgroundSync] Error forgetting {f['rel_path']}: {e}")
                    files_synced += 1
                    self._update_sync_progress(files_synced, total)

            # Phase 4: Sync RAG index
            if self._rag_manager and self._rag_manager.is_enabled and self._rag_manager.is_initialized:
                self._progress.update(
                    sync_phase=SyncPhase.RAG_SYNC,
                    sync_message="Syncing RAG index..."
                )
                # Build source lists for RAG sync
                from ..reter_wrapper import generate_source_id
                changed_sources = [
                    generate_source_id(f["path"], str(self._project_root))
                    for f in new_files + modified_files
                ]
                deleted_sources = [f["source_id"] for f in deleted_files]

                try:
                    rag_changes = RAGSyncChanges(
                        python=LanguageSourceChanges(changed_sources, deleted_sources),
                    )
                    self._rag_manager.sync_with_changes(
                        reter=self._reter,
                        project_root=self._project_root,
                        changes=rag_changes,
                    )
                except Exception as e:
                    logger.debug(f"[BackgroundSync] RAG sync error: {e}")

            # Phase 5: Save snapshot
            self._progress.update(
                sync_phase=SyncPhase.SAVING,
                sync_progress=0.95,
                sync_message="Saving snapshot..."
            )
            snapshot_path = self._instance_manager._persistence.snapshots_dir / ".default.reter"
            # CRITICAL: Use incremental save for hybrid mode
            if self._reter.is_hybrid_mode():
                self._reter.save_incremental()
            else:
                self._reter.save_network(str(snapshot_path))
            self._reter.mark_clean()

            # Complete
            self._progress.update(
                sync_status=SyncStatus.READY,
                sync_phase=SyncPhase.COMPLETE,
                sync_progress=1.0,
                sync_message=f"Sync complete: {total} files processed",
                sync_completed_at=datetime.now()
            )
            logger.debug(f"[BackgroundSync] Complete: {total} files synced")

        except Exception as e:
            logger.debug(f"[BackgroundSync] ERROR: {e}")
            import traceback
            traceback.print_exc()
            self._progress.update(
                sync_status=SyncStatus.ERROR,
                sync_error=str(e),
                sync_message=f"Sync failed: {e}"
            )

    def _update_sync_progress(self, done: int, total: int) -> None:
        """Update sync progress based on files processed."""
        progress = 0.1 + (0.75 * done / max(total, 1))  # 10-85%
        self._progress.update(
            files_synced=done,
            sync_progress=progress,
            sync_message=f"Processed {done}/{total} files"
        )
