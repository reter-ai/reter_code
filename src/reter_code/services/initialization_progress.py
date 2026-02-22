"""
Initialization Progress Tracking Service

Thread-safe shared state for tracking initialization and sync progress.
Used by BackgroundInitializer and tools to coordinate non-blocking startup.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from functools import wraps
from typing import Optional, Callable, Any, Dict


class InitStatus(Enum):
    """
    Status of initial server startup.

    ::: This is-in-layer Utility-Layer.
    ::: This is a value-object.
    """
    PENDING = "pending"           # Not started
    INITIALIZING = "initializing" # In progress
    READY = "ready"               # Complete, tools available
    ERROR = "error"               # Failed


class InitPhase(Enum):
    """
    Current phase of initialization.

    ::: This is-in-layer Utility-Layer.
    ::: This is a value-object.
    """
    PENDING = "pending"
    LOADING_PYTHON = "loading_python"
    BUILDING_RAG_INDEX = "building_rag_index"  # Embedding model loads lazily here
    COMPLETE = "complete"


class SyncStatus(Enum):
    """
    Status of file sync operations.

    ::: This is-in-layer Utility-Layer.
    ::: This is a value-object.
    """
    IDLE = "idle"           # No sync in progress
    SYNCING = "syncing"     # Sync in progress
    READY = "ready"         # Sync complete
    ERROR = "error"         # Sync failed


class SyncPhase(Enum):
    """
    Current phase of file synchronization.

    ::: This is-in-layer Utility-Layer.
    ::: This is a value-object.
    """
    IDLE = "idle"
    SCANNING = "scanning"
    LOADING = "loading"
    FORGETTING = "forgetting"
    RAG_SYNC = "rag_sync"
    SAVING = "saving"
    COMPLETE = "complete"


@dataclass
class InstanceProgress:
    """
    Shared state for initialization and sync progress.

    ::: This is-in-layer Utility-Layer.
    ::: This is a value-object.

    The instance goes through two types of blocking operations:
    1. Initial startup (loading all files for the first time)
    2. Incremental sync (processing file changes)

    Both use background threads and both block tool usage until complete.

    Note: We don't use threading.Lock here because:
    1. Python's GIL makes simple attribute reads/writes atomic
    2. Locks in async context block the entire event loop
    3. The values are set by one thread (background) and read by another (async)
    """

    # Initialization state (startup)
    init_status: InitStatus = InitStatus.PENDING
    init_phase: InitPhase = InitPhase.PENDING
    init_progress: float = 0.0
    init_message: str = "Waiting to start"
    init_error: Optional[str] = None
    init_started_at: Optional[datetime] = None
    init_completed_at: Optional[datetime] = None

    # Sync state (file changes)
    sync_status: SyncStatus = SyncStatus.IDLE
    sync_phase: SyncPhase = SyncPhase.IDLE
    sync_progress: float = 0.0
    sync_message: str = ""
    sync_error: Optional[str] = None
    sync_started_at: Optional[datetime] = None
    sync_completed_at: Optional[datetime] = None

    # Detailed counts
    python_files_total: int = 0
    python_files_loaded: int = 0
    files_to_sync: int = 0
    files_synced: int = 0
    rag_vectors_total: int = 0
    rag_vectors_indexed: int = 0

    def update(self, **kwargs) -> None:
        """Update progress fields (GIL provides atomicity for individual assignments)."""
        for key, value in kwargs.items():
            if hasattr(self, key) and not key.startswith('_'):
                setattr(self, key, value)

    def is_busy(self) -> bool:
        """Check if instance is busy (initializing OR syncing)."""
        if self.init_status == InitStatus.INITIALIZING:
            return True
        if self.sync_status == SyncStatus.SYNCING:
            return True
        return False

    def is_ready(self) -> bool:
        """Check if instance is ready (initialized AND not syncing)."""
        if self.init_status != InitStatus.READY:
            return False
        if self.sync_status == SyncStatus.SYNCING:
            return False
        return True

    def get_blocking_reason(self) -> Optional[Dict[str, Any]]:
        """
        Get reason why instance is not ready.

        Returns None if ready, otherwise returns progress dict.
        """
        # Snapshot values to avoid inconsistency during reads
        init_status = self.init_status
        sync_status = self.sync_status

        if init_status == InitStatus.INITIALIZING:
            return {
                "reason": "initializing",
                "phase": self.init_phase.value,
                "progress": round(self.init_progress, 2),
                "message": self.init_message,
                "elapsed_seconds": (
                    (datetime.now() - self.init_started_at).total_seconds()
                    if self.init_started_at else 0
                ),
                "python_files": {
                    "total": self.python_files_total,
                    "loaded": self.python_files_loaded,
                },
                "rag_index": {
                    "total": self.rag_vectors_total,
                    "indexed": self.rag_vectors_indexed,
                },
            }
        if sync_status == SyncStatus.SYNCING:
            return {
                "reason": "syncing",
                "phase": self.sync_phase.value,
                "progress": round(self.sync_progress, 2),
                "message": self.sync_message,
                "files_changed": self.files_to_sync,
                "files_processed": self.files_synced,
                "elapsed_seconds": (
                    (datetime.now() - self.sync_started_at).total_seconds()
                    if self.sync_started_at else 0
                ),
            }
        if init_status == InitStatus.ERROR:
            return {
                "reason": "init_error",
                "error": self.init_error,
            }
        if sync_status == SyncStatus.ERROR:
            return {
                "reason": "sync_error",
                "error": self.sync_error,
            }
        if init_status == InitStatus.PENDING:
            return {
                "reason": "not_started",
                "message": "Initialization has not started yet",
            }
        return None  # Ready

    def get_init_snapshot(self) -> Dict[str, Any]:
        """Get snapshot of initialization state."""
        # Snapshot all values to get consistent view
        init_started_at = self.init_started_at
        init_completed_at = self.init_completed_at
        return {
            "status": self.init_status.value,
            "phase": self.init_phase.value,
            "progress": round(self.init_progress, 2),
            "message": self.init_message,
            "error": self.init_error,
            "started_at": init_started_at.isoformat() if init_started_at else None,
            "completed_at": init_completed_at.isoformat() if init_completed_at else None,
            "elapsed_seconds": (
                (datetime.now() - init_started_at).total_seconds()
                if init_started_at else 0
            ),
            "python_files": {
                "total": self.python_files_total,
                "loaded": self.python_files_loaded,
            },
            "rag_index": {
                "total": self.rag_vectors_total,
                "indexed": self.rag_vectors_indexed,
            },
        }

    def get_sync_snapshot(self) -> Dict[str, Any]:
        """Get snapshot of sync state."""
        # Snapshot all values to get consistent view
        sync_started_at = self.sync_started_at
        sync_completed_at = self.sync_completed_at
        return {
            "status": self.sync_status.value,
            "phase": self.sync_phase.value,
            "progress": round(self.sync_progress, 2),
            "message": self.sync_message,
            "error": self.sync_error,
            "started_at": sync_started_at.isoformat() if sync_started_at else None,
            "completed_at": sync_completed_at.isoformat() if sync_completed_at else None,
            "elapsed_seconds": (
                (datetime.now() - sync_started_at).total_seconds()
                if sync_started_at else 0
            ),
            "files": {
                "total": self.files_to_sync,
                "processed": self.files_synced,
            },
        }


class InstanceNotReadyError(Exception):
    """
    Raised when a tool is called while the instance is busy.

    ::: This is-in-layer Utility-Layer.
    ::: This is a exception.

    Covers both:
    - Initial startup (initialization in progress)
    - File sync (reindexing after file changes)
    """

    def __init__(self, blocking_reason: Dict[str, Any]):
        self.blocking_reason = blocking_reason
        reason = blocking_reason.get("reason", "unknown")

        if reason == "initializing":
            phase = blocking_reason.get("phase", "unknown")
            pct = int(blocking_reason.get("progress", 0) * 100)
            elapsed = int(blocking_reason.get("elapsed_seconds", 0))
            message = blocking_reason.get("message", "Initializing...")

            super().__init__(
                f"Default instance is still initializing ({pct}% complete, {elapsed}s elapsed). "
                f"Phase: {phase}. {message}. Please wait."
            )

        elif reason == "syncing":
            phase = blocking_reason.get("phase", "unknown")
            files = blocking_reason.get("files_changed", 0)
            processed = blocking_reason.get("files_processed", 0)
            elapsed = int(blocking_reason.get("elapsed_seconds", 0))

            super().__init__(
                f"File sync in progress ({processed}/{files} files, {elapsed}s elapsed). "
                f"Phase: {phase}. Please wait for sync to complete."
            )

        elif reason == "init_error":
            error = blocking_reason.get("error", "Unknown error")
            super().__init__(f"Initialization failed: {error}. Please restart the server.")

        elif reason == "sync_error":
            error = blocking_reason.get("error", "Unknown error")
            super().__init__(f"File sync failed: {error}. Some files may be stale.")

        elif reason == "not_started":
            super().__init__(
                "Default instance initialization has not started yet. Please wait."
            )

        else:
            super().__init__(f"Instance not ready: {reason}")


# Global singleton instance - created at module load (thread-safe via import lock)
_instance_progress: InstanceProgress = InstanceProgress()


def get_instance_progress() -> InstanceProgress:
    """Get the global instance progress tracker."""
    return _instance_progress


def reset_instance_progress() -> None:
    """Reset the global instance progress tracker (for testing)."""
    global _instance_progress
    _instance_progress = InstanceProgress()


def require_ready(func: Callable) -> Callable:
    """
    Decorator that blocks tool execution until instance is ready.

    Checks for BOTH:
    - Initialization complete
    - No sync in progress

    Usage:
        @require_ready
        def my_tool_handler(...):
            ...
    """
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        progress = get_instance_progress()

        blocking = progress.get_blocking_reason()
        if blocking is not None:
            raise InstanceNotReadyError(blocking)

        return func(*args, **kwargs)

    return wrapper


def require_ready_async(func: Callable) -> Callable:
    """Async version of require_ready decorator."""
    @wraps(func)
    async def wrapper(*args, **kwargs) -> Any:
        progress = get_instance_progress()

        blocking = progress.get_blocking_reason()
        if blocking is not None:
            raise InstanceNotReadyError(blocking)

        return await func(*args, **kwargs)

    return wrapper


def get_initializing_response(**extra_fields) -> Dict[str, Any]:
    """
    Return a consistent initialization status response for tools.

    Use this when a tool is called while the server is still initializing.

    Args:
        **extra_fields: Additional fields to include (e.g., results=[], clusters=[])

    Returns:
        Consistent response dict matching init_status() format
    """
    progress = get_instance_progress()
    init_snapshot = progress.get_init_snapshot()
    component_status = get_component_readiness().get_status()

    response = {
        "is_ready": False,
        "success": False,
        "error": "Server is still initializing. The embedding model and code index are being loaded in the background.",
        "hint": "Please try again in 30 seconds. Use init_status() to check progress.",
        "blocking_reason": {
            "reason": "initializing",
            "phase": init_snapshot.get("phase", "unknown"),
            "progress": init_snapshot.get("progress", 0),
            "message": init_snapshot.get("message", "Initializing..."),
        },
        "init": init_snapshot,
        "components": component_status,
    }

    # Add any extra fields (e.g., results=[], clusters=[], pairs=[])
    response.update(extra_fields)

    return response


# =============================================================================
# Component-Based Initialization
# =============================================================================

@dataclass
class ComponentReadiness:
    """
    Component readiness tracking.

    ::: This is-in-layer Utility-Layer.
    ::: This is a value-object.

    Components initialize sequentially:
    1. reter_ready - Default RETER instance with Python files loaded
    2. embedding_ready - SentenceTransformers model loaded
    3. rag_code_ready - RAG Python code entities indexed
    4. rag_docs_ready - RAG Markdown documents indexed

    Tools check appropriate flags before executing:
    - code_inspection, reql, natural_language_query → require_default_instance()
    - semantic_search (code) → require_rag_code_index()
    - semantic_search (docs) → require_rag_document_index()

    Note: No locks used - Python's GIL makes attribute access atomic.
    """

    # Component readiness flags
    sql_ready: bool = False
    reter_ready: bool = False
    embedding_ready: bool = False
    rag_code_ready: bool = False
    rag_docs_ready: bool = False

    # Timestamps for when each component became ready
    sql_ready_at: Optional[datetime] = None
    reter_ready_at: Optional[datetime] = None
    embedding_ready_at: Optional[datetime] = None
    rag_code_ready_at: Optional[datetime] = None
    rag_docs_ready_at: Optional[datetime] = None

    # Error tracking per component
    reter_error: Optional[str] = None
    rag_error: Optional[str] = None

    def set_sql_ready(self, value: bool = True) -> None:
        """Mark SQLite as ready."""
        if value:
            self.sql_ready_at = datetime.now()
        self.sql_ready = value

    def set_reter_ready(self, value: bool = True, error: Optional[str] = None) -> None:
        """Mark default RETER instance as ready."""
        self.reter_error = error
        if value:
            self.reter_ready_at = datetime.now()
        self.reter_ready = value

    def set_embedding_ready(self, value: bool = True) -> None:
        """Mark embedding model as ready."""
        if value:
            self.embedding_ready_at = datetime.now()
        self.embedding_ready = value

    def set_rag_code_ready(self, value: bool = True, error: Optional[str] = None) -> None:
        """Mark RAG code index as ready."""
        if error:
            self.rag_error = error
        if value:
            self.rag_code_ready_at = datetime.now()
        self.rag_code_ready = value

    def set_rag_docs_ready(self, value: bool = True, error: Optional[str] = None) -> None:
        """Mark RAG document index as ready."""
        if error:
            self.rag_error = error
        if value:
            self.rag_docs_ready_at = datetime.now()
        self.rag_docs_ready = value

    def is_fully_ready(self) -> bool:
        """Check if all components are ready."""
        return (
            self.sql_ready and
            self.reter_ready and
            self.rag_code_ready and
            self.rag_docs_ready
        )

    def get_status(self) -> Dict[str, Any]:
        """Get snapshot of all component statuses."""
        # Snapshot timestamps for consistency
        sql_ready_at = self.sql_ready_at
        reter_ready_at = self.reter_ready_at
        embedding_ready_at = self.embedding_ready_at
        rag_code_ready_at = self.rag_code_ready_at
        rag_docs_ready_at = self.rag_docs_ready_at

        return {
            "sql": {
                "ready": self.sql_ready,
                "ready_at": sql_ready_at.isoformat() if sql_ready_at else None,
            },
            "reter": {
                "ready": self.reter_ready,
                "ready_at": reter_ready_at.isoformat() if reter_ready_at else None,
                "error": self.reter_error,
            },
            "embedding": {
                "ready": self.embedding_ready,
                "ready_at": embedding_ready_at.isoformat() if embedding_ready_at else None,
            },
            "rag_code": {
                "ready": self.rag_code_ready,
                "ready_at": rag_code_ready_at.isoformat() if rag_code_ready_at else None,
                "error": self.rag_error if not self.rag_code_ready else None,
            },
            "rag_docs": {
                "ready": self.rag_docs_ready,
                "ready_at": rag_docs_ready_at.isoformat() if rag_docs_ready_at else None,
                "error": self.rag_error if not self.rag_docs_ready else None,
            },
        }


# Global singleton for component readiness - created at module load
_component_readiness: ComponentReadiness = ComponentReadiness()


def get_component_readiness() -> ComponentReadiness:
    """Get the global component readiness tracker."""
    return _component_readiness


def reset_component_readiness() -> None:
    """Reset the global component readiness tracker (for testing)."""
    global _component_readiness
    _component_readiness = ComponentReadiness()


class ComponentNotReadyError(Exception):
    """
    Raised when a specific component is not ready.

    ::: This is-in-layer Utility-Layer.
    ::: This is a exception.

    Allows tools to check for specific component readiness and return
    appropriate error messages indicating which component is still initializing.
    """

    def __init__(self, component: str, message: str):
        self.component = component
        self.message = message
        super().__init__(f"{component}: {message}")

    def to_response(self, **extra_fields) -> Dict[str, Any]:
        """Convert to a tool response dict."""
        progress = get_instance_progress()
        init_snapshot = progress.get_init_snapshot()
        component_status = get_component_readiness().get_status()

        response = {
            "is_ready": False,
            "success": False,
            "error": self.message,
            "hint": "Please try again in 30 seconds. Use init_status() to check progress.",
            "blocking_reason": {
                "reason": "component_not_ready",
                "component": self.component,
                "message": self.message,
            },
            "init": init_snapshot,
            "components": component_status,
        }
        response.update(extra_fields)
        return response


def require_sql() -> None:
    """
    Check that SQLite is ready.

    Raises:
        ComponentNotReadyError: If SQLite is not initialized
    """
    state = get_component_readiness()
    if not state.sql_ready:
        raise ComponentNotReadyError(
            "sql",
            "SQLite is not initialized yet. Please wait for server startup."
        )


def require_default_instance() -> None:
    """
    Check that the default RETER instance is ready.

    This means Python files have been loaded into the RETER engine.

    Raises:
        ComponentNotReadyError: If default RETER instance is not ready
    """
    state = get_component_readiness()
    if not state.reter_ready:
        if state.reter_error:
            raise ComponentNotReadyError(
                "reter",
                f"Default RETER instance failed to initialize: {state.reter_error}"
            )
        raise ComponentNotReadyError(
            "reter",
            "Default RETER instance is still loading Python files. Please wait."
        )


def require_rag_code_index() -> None:
    """
    Check that RAG code index is ready.

    This means Python code entities have been embedded and indexed.

    Raises:
        ComponentNotReadyError: If RAG code index is not ready
    """
    state = get_component_readiness()
    if not state.rag_code_ready:
        if state.rag_error:
            raise ComponentNotReadyError(
                "rag_code",
                f"RAG code index failed to build: {state.rag_error}"
            )
        raise ComponentNotReadyError(
            "rag_code",
            "RAG code index is still building. Please wait for code indexing to complete."
        )


def require_rag_document_index() -> None:
    """
    Check that RAG document index is ready.

    This means Markdown files have been embedded and indexed.

    Raises:
        ComponentNotReadyError: If RAG document index is not ready
    """
    state = get_component_readiness()
    if not state.rag_docs_ready:
        if state.rag_error:
            raise ComponentNotReadyError(
                "rag_docs",
                f"RAG document index failed to build: {state.rag_error}"
            )
        raise ComponentNotReadyError(
            "rag_docs",
            "RAG document index is still indexing Markdown files. Please wait."
        )
