"""SQLite Session Store for RETER session tools.

Provides a SQLite-based storage backend for session data including:
- Recommendations and activity tracking
- Gantt chart tasks and dependencies
- Requirements and traceability

This replaces the RETER-based .current.reter storage with .current.sqlite
for better performance on CRUD-heavy workloads.
"""

import os
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


class SQLiteSessionStore:
    """
    Base class for SQLite-based session storage.

    Features:
    - Connection pooling with thread-local connections
    - WAL mode for concurrent read/write access
    - Automatic schema creation and migration
    - Transaction support via context manager

    Usage:
        store = SQLiteSessionStore()
        with store.transaction() as conn:
            conn.execute("INSERT INTO ...")
    """

    # Schema version for migrations
    SCHEMA_VERSION = 2  # v2: Added logical_sessions and logical_thoughts tables

    # Default database filename
    DEFAULT_DB_NAME = ".current.sqlite"

    def __init__(self, db_path: Optional[Union[str, Path]] = None):
        """
        Initialize the SQLite session store.

        Args:
            db_path: Path to the SQLite database file. If None, uses
                     RETER_SNAPSHOTS_DIR/.current.sqlite
        """
        if db_path is None:
            snapshots_dir = os.getenv(
                "RETER_SNAPSHOTS_DIR",
                os.path.join(os.getcwd(), ".reter")
            )
            self._db_path = Path(snapshots_dir) / self.DEFAULT_DB_NAME
        else:
            self._db_path = Path(db_path)

        # Ensure parent directory exists
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        # Thread-local storage for connections
        self._local = threading.local()

        # Lock for schema initialization
        self._schema_lock = threading.Lock()
        self._schema_initialized = False

        # Initialize schema on first access
        self._ensure_schema()

    @property
    def db_path(self) -> Path:
        """Return the database file path."""
        return self._db_path

    def _get_connection(self) -> sqlite3.Connection:
        """
        Get a thread-local database connection.

        Returns:
            sqlite3.Connection: Database connection for current thread
        """
        if not hasattr(self._local, 'connection') or self._local.connection is None:
            conn = sqlite3.connect(
                str(self._db_path),
                check_same_thread=False,
                timeout=30.0
            )
            # Enable WAL mode for better concurrency
            conn.execute("PRAGMA journal_mode=WAL")
            # Enable foreign keys
            conn.execute("PRAGMA foreign_keys=ON")
            # Return rows as dictionaries
            conn.row_factory = sqlite3.Row
            self._local.connection = conn
        return self._local.connection

    @contextmanager
    def transaction(self):
        """
        Context manager for database transactions.

        Usage:
            with store.transaction() as conn:
                conn.execute("INSERT INTO ...")
                conn.execute("UPDATE ...")
            # Auto-commits on success, rolls back on exception

        Yields:
            sqlite3.Connection: Database connection
        """
        conn = self._get_connection()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def execute(
        self,
        sql: str,
        params: Optional[Union[Tuple, Dict]] = None
    ) -> sqlite3.Cursor:
        """
        Execute a SQL statement.

        Args:
            sql: SQL statement to execute
            params: Parameters for the SQL statement

        Returns:
            sqlite3.Cursor: Cursor with results
        """
        conn = self._get_connection()
        if params is None:
            return conn.execute(sql)
        return conn.execute(sql, params)

    def execute_many(
        self,
        sql: str,
        params_list: List[Union[Tuple, Dict]]
    ) -> sqlite3.Cursor:
        """
        Execute a SQL statement with multiple parameter sets.

        Args:
            sql: SQL statement to execute
            params_list: List of parameter sets

        Returns:
            sqlite3.Cursor: Cursor with results
        """
        conn = self._get_connection()
        return conn.executemany(sql, params_list)

    def fetch_one(
        self,
        sql: str,
        params: Optional[Union[Tuple, Dict]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Execute a query and fetch one result.

        Args:
            sql: SQL query
            params: Query parameters

        Returns:
            Dict or None: Result row as dictionary, or None if no results
        """
        cursor = self.execute(sql, params)
        row = cursor.fetchone()
        return dict(row) if row else None

    def fetch_all(
        self,
        sql: str,
        params: Optional[Union[Tuple, Dict]] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute a query and fetch all results.

        Args:
            sql: SQL query
            params: Query parameters

        Returns:
            List[Dict]: List of result rows as dictionaries
        """
        cursor = self.execute(sql, params)
        return [dict(row) for row in cursor.fetchall()]

    def commit(self) -> None:
        """Commit the current transaction."""
        self._get_connection().commit()

    def rollback(self) -> None:
        """Roll back the current transaction."""
        self._get_connection().rollback()

    def close(self) -> None:
        """Close the database connection for the current thread."""
        if hasattr(self._local, 'connection') and self._local.connection:
            self._local.connection.close()
            self._local.connection = None

    def _ensure_schema(self) -> None:
        """Ensure the database schema is initialized."""
        with self._schema_lock:
            if self._schema_initialized:
                return

            conn = self._get_connection()

            # Check if schema_version table exists
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_version'"
            )
            if cursor.fetchone() is None:
                # Fresh database - create all tables
                self._create_schema(conn)
            else:
                # Check version and migrate if needed
                cursor = conn.execute("SELECT version FROM schema_version")
                row = cursor.fetchone()
                current_version = row['version'] if row else 0

                if current_version < self.SCHEMA_VERSION:
                    self._migrate_schema(conn, current_version)

            conn.commit()
            self._schema_initialized = True

    def _create_schema(self, conn: sqlite3.Connection) -> None:
        """Create the initial database schema."""
        schema_sql = self._get_schema_sql()
        conn.executescript(schema_sql)

    def _migrate_schema(
        self,
        conn: sqlite3.Connection,
        from_version: int
    ) -> None:
        """
        Migrate the schema from an older version.

        Args:
            conn: Database connection
            from_version: Current schema version in database
        """
        # Migration v1 -> v2: Add logical_sessions and logical_thoughts tables
        if from_version < 2:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS logical_sessions (
                    session_id TEXT PRIMARY KEY,
                    instance_name TEXT NOT NULL DEFAULT 'default',
                    goal TEXT,
                    context TEXT,
                    status TEXT DEFAULT 'active' CHECK (status IN ('active', 'completed', 'paused', 'failed')),
                    knowledge_stats TEXT,
                    loaded_sources TEXT,
                    created_at TEXT NOT NULL DEFAULT (datetime('now')),
                    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
                );

                CREATE INDEX IF NOT EXISTS idx_logical_sessions_instance ON logical_sessions(instance_name);
                CREATE INDEX IF NOT EXISTS idx_logical_sessions_status ON logical_sessions(status);

                CREATE TABLE IF NOT EXISTS logical_thoughts (
                    thought_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL REFERENCES logical_sessions(session_id) ON DELETE CASCADE,
                    thought TEXT NOT NULL,
                    thought_number INTEGER NOT NULL,
                    total_thoughts INTEGER NOT NULL,
                    next_thought_needed INTEGER DEFAULT 1,
                    thought_type TEXT DEFAULT 'reasoning',
                    logic_operation TEXT,
                    query_results TEXT,
                    inferences TEXT,
                    contradictions TEXT,
                    confidence REAL DEFAULT 1.0,
                    justification TEXT,
                    assumptions TEXT,
                    is_revision INTEGER DEFAULT 0,
                    revises_thought INTEGER,
                    branch_from_thought INTEGER,
                    branch_id TEXT,
                    needs_more_thoughts INTEGER DEFAULT 0,
                    timestamp TEXT NOT NULL DEFAULT (datetime('now'))
                );

                CREATE INDEX IF NOT EXISTS idx_logical_thoughts_session ON logical_thoughts(session_id);
                CREATE INDEX IF NOT EXISTS idx_logical_thoughts_branch ON logical_thoughts(branch_id);
                CREATE INDEX IF NOT EXISTS idx_logical_thoughts_number ON logical_thoughts(thought_number);
            """)

        # Update version
        conn.execute(
            "UPDATE schema_version SET version = ?, updated_at = ?",
            (self.SCHEMA_VERSION, datetime.now().isoformat())
        )

    def _get_schema_sql(self) -> str:
        """
        Get the SQL schema definition.

        Returns:
            str: SQL script to create all tables
        """
        return """
-- Schema version tracking
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

INSERT INTO schema_version (version, created_at, updated_at)
VALUES (1, datetime('now'), datetime('now'));

-- ============================================================================
-- RECOMMENDATIONS TABLES
-- ============================================================================

CREATE TABLE IF NOT EXISTS recommendations (
    rec_id TEXT PRIMARY KEY,
    text TEXT NOT NULL,
    category TEXT DEFAULT 'general',
    severity TEXT DEFAULT 'medium' CHECK (severity IN ('critical', 'high', 'medium', 'low')),
    status TEXT DEFAULT 'pending' CHECK (status IN ('pending', 'in_progress', 'completed', 'skipped')),
    source_tool TEXT,
    source_activity_id TEXT,
    gantt_task_id TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_recommendations_status ON recommendations(status);
CREATE INDEX IF NOT EXISTS idx_recommendations_severity ON recommendations(severity);
CREATE INDEX IF NOT EXISTS idx_recommendations_category ON recommendations(category);

CREATE TABLE IF NOT EXISTS recommendation_files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    rec_id TEXT NOT NULL REFERENCES recommendations(rec_id) ON DELETE CASCADE,
    file_path TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_recommendation_files_rec_id ON recommendation_files(rec_id);

CREATE TABLE IF NOT EXISTS recommendation_entities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    rec_id TEXT NOT NULL REFERENCES recommendations(rec_id) ON DELETE CASCADE,
    entity_name TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_recommendation_entities_rec_id ON recommendation_entities(rec_id);

-- ============================================================================
-- ACTIVITY TRACKING TABLES
-- ============================================================================

CREATE TABLE IF NOT EXISTS activities (
    activity_id TEXT PRIMARY KEY,
    tool_name TEXT NOT NULL,
    params_summary TEXT,
    result_summary TEXT,
    reter_instance TEXT DEFAULT 'default',
    issues_found INTEGER DEFAULT 0,
    duration_ms REAL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_activities_tool_name ON activities(tool_name);
CREATE INDEX IF NOT EXISTS idx_activities_created_at ON activities(created_at);

CREATE TABLE IF NOT EXISTS activity_files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    activity_id TEXT NOT NULL REFERENCES activities(activity_id) ON DELETE CASCADE,
    file_path TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_activity_files_activity_id ON activity_files(activity_id);

-- ============================================================================
-- ARTIFACT TRACKING TABLES
-- ============================================================================

CREATE TABLE IF NOT EXISTS artifacts (
    artifact_id TEXT PRIMARY KEY,
    file_path TEXT NOT NULL UNIQUE,
    artifact_type TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    checksum TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_artifacts_file_path ON artifacts(file_path);

CREATE TABLE IF NOT EXISTS artifact_sources (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    artifact_id TEXT NOT NULL REFERENCES artifacts(artifact_id) ON DELETE CASCADE,
    source_file TEXT NOT NULL,
    source_checksum TEXT
);

CREATE INDEX IF NOT EXISTS idx_artifact_sources_artifact_id ON artifact_sources(artifact_id);

-- ============================================================================
-- GANTT CHART TABLES
-- ============================================================================

CREATE TABLE IF NOT EXISTS gantt_tasks (
    task_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    status TEXT DEFAULT 'planned' CHECK (status IN ('planned', 'in_progress', 'completed', 'blocked')),
    progress_percent INTEGER DEFAULT 0 CHECK (progress_percent >= 0 AND progress_percent <= 100),
    start_date TEXT,
    end_date TEXT,
    duration_days INTEGER,
    actual_start_date TEXT,
    actual_end_date TEXT,
    phase TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_gantt_tasks_status ON gantt_tasks(status);
CREATE INDEX IF NOT EXISTS idx_gantt_tasks_phase ON gantt_tasks(phase);

CREATE TABLE IF NOT EXISTS gantt_dependencies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT NOT NULL REFERENCES gantt_tasks(task_id) ON DELETE CASCADE,
    depends_on_task_id TEXT NOT NULL REFERENCES gantt_tasks(task_id) ON DELETE CASCADE,
    UNIQUE(task_id, depends_on_task_id)
);

CREATE INDEX IF NOT EXISTS idx_gantt_dependencies_task_id ON gantt_dependencies(task_id);
CREATE INDEX IF NOT EXISTS idx_gantt_dependencies_depends_on ON gantt_dependencies(depends_on_task_id);

CREATE TABLE IF NOT EXISTS gantt_resources (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT NOT NULL REFERENCES gantt_tasks(task_id) ON DELETE CASCADE,
    resource_name TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_gantt_resources_task_id ON gantt_resources(task_id);
CREATE INDEX IF NOT EXISTS idx_gantt_resources_resource_name ON gantt_resources(resource_name);

CREATE TABLE IF NOT EXISTS gantt_milestones (
    milestone_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    target_date TEXT NOT NULL,
    status TEXT DEFAULT 'planned' CHECK (status IN ('planned', 'achieved', 'missed')),
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS milestone_dependencies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    milestone_id TEXT NOT NULL REFERENCES gantt_milestones(milestone_id) ON DELETE CASCADE,
    depends_on_task_id TEXT NOT NULL REFERENCES gantt_tasks(task_id) ON DELETE CASCADE
);

-- ============================================================================
-- REQUIREMENTS TABLES
-- ============================================================================

CREATE TABLE IF NOT EXISTS requirements (
    req_id TEXT PRIMARY KEY,
    text TEXT NOT NULL,
    req_type TEXT DEFAULT 'requirement' CHECK (req_type IN (
        'requirement', 'functionalRequirement', 'performanceRequirement',
        'interfaceRequirement', 'physicalRequirement', 'designConstraint'
    )),
    risk TEXT DEFAULT 'medium' CHECK (risk IN ('low', 'medium', 'high')),
    verifymethod TEXT DEFAULT 'test' CHECK (verifymethod IN ('test', 'inspection', 'demonstration', 'analysis')),
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_requirements_req_type ON requirements(req_type);
CREATE INDEX IF NOT EXISTS idx_requirements_risk ON requirements(risk);

CREATE TABLE IF NOT EXISTS elements (
    element_id TEXT PRIMARY KEY,
    element_type TEXT NOT NULL,
    docref TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS relationships (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id TEXT NOT NULL,
    relationship_type TEXT NOT NULL CHECK (relationship_type IN (
        'traces', 'contains', 'derives', 'satisfies', 'verifies', 'refines', 'copies'
    )),
    target_id TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(source_id, relationship_type, target_id)
);

CREATE INDEX IF NOT EXISTS idx_relationships_source ON relationships(source_id);
CREATE INDEX IF NOT EXISTS idx_relationships_target ON relationships(target_id);

-- ============================================================================
-- SESSION MANAGEMENT
-- ============================================================================

CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL DEFAULT (datetime('now')),
    ended_at TEXT,
    summary TEXT
);

-- ============================================================================
-- LOGICAL THINKING TABLES
-- ============================================================================

CREATE TABLE IF NOT EXISTS logical_sessions (
    session_id TEXT PRIMARY KEY,
    instance_name TEXT NOT NULL DEFAULT 'default',
    goal TEXT,
    context TEXT,
    status TEXT DEFAULT 'active' CHECK (status IN ('active', 'completed', 'paused', 'failed')),
    knowledge_stats TEXT,  -- JSON
    loaded_sources TEXT,   -- JSON array
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_logical_sessions_instance ON logical_sessions(instance_name);
CREATE INDEX IF NOT EXISTS idx_logical_sessions_status ON logical_sessions(status);

CREATE TABLE IF NOT EXISTS logical_thoughts (
    thought_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES logical_sessions(session_id) ON DELETE CASCADE,
    thought TEXT NOT NULL,
    thought_number INTEGER NOT NULL,
    total_thoughts INTEGER NOT NULL,
    next_thought_needed INTEGER DEFAULT 1,  -- BOOLEAN
    thought_type TEXT DEFAULT 'reasoning',
    logic_operation TEXT,   -- JSON
    query_results TEXT,     -- JSON
    inferences TEXT,        -- JSON
    contradictions TEXT,    -- JSON array
    confidence REAL DEFAULT 1.0,
    justification TEXT,     -- JSON array
    assumptions TEXT,       -- JSON array
    is_revision INTEGER DEFAULT 0,  -- BOOLEAN
    revises_thought INTEGER,
    branch_from_thought INTEGER,
    branch_id TEXT,
    needs_more_thoughts INTEGER DEFAULT 0,  -- BOOLEAN
    timestamp TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_logical_thoughts_session ON logical_thoughts(session_id);
CREATE INDEX IF NOT EXISTS idx_logical_thoughts_branch ON logical_thoughts(branch_id);
CREATE INDEX IF NOT EXISTS idx_logical_thoughts_number ON logical_thoughts(thought_number);
"""

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def now_iso(self) -> str:
        """Return current timestamp in ISO format."""
        return datetime.now().isoformat()

    def table_exists(self, table_name: str) -> bool:
        """Check if a table exists in the database."""
        result = self.fetch_one(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,)
        )
        return result is not None

    def count(self, table_name: str, where: str = "", params: Tuple = ()) -> int:
        """
        Count rows in a table.

        Args:
            table_name: Name of the table
            where: Optional WHERE clause (without 'WHERE' keyword)
            params: Parameters for WHERE clause

        Returns:
            int: Number of rows
        """
        sql = f"SELECT COUNT(*) as cnt FROM {table_name}"
        if where:
            sql += f" WHERE {where}"
        result = self.fetch_one(sql, params)
        return result['cnt'] if result else 0
