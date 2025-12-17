"""
Unified Store for Thinking System

Single SQLite-based storage for all session data:
- Sessions (thinking sessions with goals and project info)
- Items (thoughts, requirements, recommendations, tasks, milestones, activities)
- Relations (traceability links between items)
- Artifacts (generated files with freshness tracking)
"""

import logging
import sqlite3
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Literal
from contextlib import contextmanager

logger = logging.getLogger(__name__)


# Item types supported by the unified store
ItemType = Literal[
    "thought",
    "requirement",
    "recommendation",
    "task",
    "milestone",
    "activity",
    "decision",
    "element"
]

# Relation types for traceability
RelationType = Literal[
    "traces",      # Generic traceability
    "derives",     # Requirement derives from another
    "satisfies",   # Element satisfies requirement
    "verifies",    # Test verifies requirement
    "implements",  # Code implements requirement
    "depends_on",  # Task depends on another task
    "affects",     # Item affects file/entity
    "creates"      # Thought creates item
]

# Status values
Status = Literal[
    "planned",
    "pending",
    "in_progress",
    "completed",
    "verified",
    "blocked",
    "skipped"
]

# Priority values
Priority = Literal["critical", "high", "medium", "low"]


UNIFIED_SCHEMA = """
-- Sessions table
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    instance_name TEXT NOT NULL,
    goal TEXT,
    context TEXT,
    status TEXT DEFAULT 'active',
    project_start TEXT,
    project_end TEXT,
    knowledge_stats TEXT,  -- JSON
    loaded_sources TEXT,   -- JSON array
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_sessions_instance ON sessions(instance_name);
CREATE INDEX IF NOT EXISTS idx_sessions_status ON sessions(status);

-- Unified items table
CREATE TABLE IF NOT EXISTS items (
    item_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    item_type TEXT NOT NULL,  -- thought, requirement, recommendation, task, milestone, activity, decision, element
    content TEXT NOT NULL,

    -- Common fields
    status TEXT DEFAULT 'pending',
    priority TEXT DEFAULT 'medium',
    category TEXT,
    source_tool TEXT,

    -- Thought-specific
    thought_number INTEGER,
    total_thoughts INTEGER,
    next_thought_needed INTEGER,
    thought_type TEXT,
    logic_operation TEXT,    -- JSON
    query_results TEXT,      -- JSON
    inferences TEXT,         -- JSON
    contradictions TEXT,     -- JSON
    confidence REAL DEFAULT 1.0,
    justification TEXT,      -- JSON array
    assumptions TEXT,        -- JSON array
    is_revision INTEGER DEFAULT 0,
    revises_thought INTEGER,
    branch_from_thought INTEGER,
    branch_id TEXT,
    needs_more_thoughts INTEGER DEFAULT 0,

    -- Requirement-specific
    risk TEXT,
    verify_method TEXT,

    -- Task/Milestone-specific
    start_date TEXT,
    end_date TEXT,
    duration_days INTEGER,
    actual_start TEXT,
    actual_end TEXT,
    progress INTEGER DEFAULT 0,
    assigned_to TEXT,        -- JSON array
    phase TEXT,

    -- Recommendation-specific
    description TEXT,
    affected_files TEXT,     -- JSON array
    affected_entities TEXT,  -- JSON array

    -- Activity-specific
    params_summary TEXT,
    result_summary TEXT,
    duration_ms REAL,
    issues_found INTEGER,
    files_analyzed TEXT,     -- JSON array
    reter_instance TEXT,

    -- Element-specific (for requirements traceability)
    element_type TEXT,
    docref TEXT,

    -- Metadata
    metadata TEXT,           -- JSON for extensibility
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,

    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

CREATE INDEX IF NOT EXISTS idx_items_session ON items(session_id);
CREATE INDEX IF NOT EXISTS idx_items_type ON items(item_type);
CREATE INDEX IF NOT EXISTS idx_items_status ON items(status);
CREATE INDEX IF NOT EXISTS idx_items_priority ON items(priority);
CREATE INDEX IF NOT EXISTS idx_items_thought_number ON items(thought_number);
CREATE INDEX IF NOT EXISTS idx_items_category ON items(category);

-- Relations table for traceability
CREATE TABLE IF NOT EXISTS relations (
    relation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    target_type TEXT NOT NULL,  -- 'item', 'file', 'entity'
    relation_type TEXT NOT NULL, -- traces, derives, satisfies, verifies, implements, depends_on, affects, creates
    metadata TEXT,              -- JSON for additional info
    created_at TEXT NOT NULL,

    UNIQUE(source_id, target_id, relation_type)
);

CREATE INDEX IF NOT EXISTS idx_relations_source ON relations(source_id);
CREATE INDEX IF NOT EXISTS idx_relations_target ON relations(target_id);
CREATE INDEX IF NOT EXISTS idx_relations_type ON relations(relation_type);

-- Artifacts table for generated files
CREATE TABLE IF NOT EXISTS artifacts (
    artifact_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    file_path TEXT NOT NULL,
    artifact_type TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    source_files TEXT,         -- JSON array
    source_hashes TEXT,        -- JSON dict of file -> hash
    created_at TEXT NOT NULL,

    FOREIGN KEY (session_id) REFERENCES sessions(session_id),
    UNIQUE(session_id, file_path)
);

CREATE INDEX IF NOT EXISTS idx_artifacts_session ON artifacts(session_id);

-- ID counters table for auto-incrementing IDs per session
CREATE TABLE IF NOT EXISTS id_counters (
    session_id TEXT NOT NULL,
    prefix TEXT NOT NULL,
    counter INTEGER DEFAULT 0,
    PRIMARY KEY (session_id, prefix)
);
"""


class UnifiedStore:
    """
    Unified SQLite store for the thinking system.

    Manages sessions, items, relations, and artifacts in a single database.
    """

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize the unified store.

        Args:
            db_path: Path to SQLite database. If None, uses default location.
        """
        if db_path is None:
            # Default: .reter/.unified.sqlite in project root or CWD
            project_root = os.getenv("RETER_PROJECT_ROOT", os.getcwd())
            reter_dir = os.path.join(project_root, ".reter")
            os.makedirs(reter_dir, exist_ok=True)
            db_path = os.path.join(reter_dir, ".unified.sqlite")

        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize the database schema."""
        with self._get_connection() as conn:
            conn.executescript(UNIFIED_SCHEMA)

    @contextmanager
    def _get_connection(self):
        """Get a database connection with proper cleanup."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _now(self) -> str:
        """Get current timestamp as ISO string."""
        return datetime.utcnow().isoformat()

    # =========================================================================
    # Session Management
    # =========================================================================

    def get_or_create_session(
        self,
        instance_name: str,
        goal: Optional[str] = None,
        project_start: Optional[str] = None,
        project_end: Optional[str] = None
    ) -> str:
        """
        Get existing active session or create new one for instance.

        Returns:
            session_id
        """
        with self._get_connection() as conn:
            # Check for existing active session
            row = conn.execute(
                "SELECT session_id FROM sessions WHERE instance_name = ? AND status = 'active'",
                (instance_name,)
            ).fetchone()

            if row:
                # Update goal if provided
                if goal:
                    conn.execute(
                        "UPDATE sessions SET goal = ?, updated_at = ? WHERE session_id = ?",
                        (goal, self._now(), row["session_id"])
                    )
                return row["session_id"]

            # Create new session
            import uuid
            session_id = str(uuid.uuid4())
            now = self._now()

            conn.execute(
                """INSERT INTO sessions
                   (session_id, instance_name, goal, status, project_start, project_end, created_at, updated_at)
                   VALUES (?, ?, ?, 'active', ?, ?, ?, ?)""",
                (session_id, instance_name, goal, project_start, project_end, now, now)
            )

            return session_id

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session by ID."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM sessions WHERE session_id = ?",
                (session_id,)
            ).fetchone()

            if not row:
                return None

            return self._row_to_dict(row)

    def get_session_by_instance(self, instance_name: str) -> Optional[Dict[str, Any]]:
        """Get active session for instance."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM sessions WHERE instance_name = ? AND status = 'active'",
                (instance_name,)
            ).fetchone()

            if not row:
                return None

            return self._row_to_dict(row)

    def update_session(self, session_id: str, **updates) -> bool:
        """Update session fields."""
        if not updates:
            return True

        updates["updated_at"] = self._now()

        # Handle JSON fields
        json_fields = ["knowledge_stats", "loaded_sources"]
        for field in json_fields:
            if field in updates and not isinstance(updates[field], str):
                updates[field] = json.dumps(updates[field])

        set_clause = ", ".join(f"{k} = ?" for k in updates.keys())
        values = list(updates.values()) + [session_id]

        with self._get_connection() as conn:
            conn.execute(
                f"UPDATE sessions SET {set_clause} WHERE session_id = ?",
                values
            )

        return True

    def end_session(self, session_id: str) -> bool:
        """Mark session as completed."""
        return self.update_session(session_id, status="completed")

    def clear_session(self, session_id: str) -> Dict[str, int]:
        """Delete all data for a session."""
        with self._get_connection() as conn:
            # Get item IDs for relation cleanup
            item_ids = [row["item_id"] for row in conn.execute(
                "SELECT item_id FROM items WHERE session_id = ?",
                (session_id,)
            ).fetchall()]

            # Delete relations
            relations_deleted = 0
            if item_ids:
                placeholders = ",".join("?" * len(item_ids))
                cursor = conn.execute(
                    f"DELETE FROM relations WHERE source_id IN ({placeholders})",
                    item_ids
                )
                relations_deleted = cursor.rowcount

            # Delete items
            cursor = conn.execute(
                "DELETE FROM items WHERE session_id = ?",
                (session_id,)
            )
            items_deleted = cursor.rowcount

            # Delete artifacts
            cursor = conn.execute(
                "DELETE FROM artifacts WHERE session_id = ?",
                (session_id,)
            )
            artifacts_deleted = cursor.rowcount

            # Delete ID counters
            conn.execute(
                "DELETE FROM id_counters WHERE session_id = ?",
                (session_id,)
            )

            # Delete session
            conn.execute(
                "DELETE FROM sessions WHERE session_id = ?",
                (session_id,)
            )

        return {
            "items_deleted": items_deleted,
            "relations_deleted": relations_deleted,
            "artifacts_deleted": artifacts_deleted
        }

    # =========================================================================
    # Item Management
    # =========================================================================

    def generate_id(self, session_id: str, prefix: str) -> str:
        """
        Generate auto-incrementing ID for session.

        Args:
            session_id: Session ID
            prefix: ID prefix (e.g., "REQ", "TASK", "REC")

        Returns:
            Generated ID like "REQ-001", "TASK-002" (globally unique via session hash)
        """
        # Use first 4 chars of session_id for global uniqueness
        session_prefix = session_id[:4].upper()

        with self._get_connection() as conn:
            # Get and increment counter
            row = conn.execute(
                "SELECT counter FROM id_counters WHERE session_id = ? AND prefix = ?",
                (session_id, prefix)
            ).fetchone()

            if row:
                counter = row["counter"] + 1
                conn.execute(
                    "UPDATE id_counters SET counter = ? WHERE session_id = ? AND prefix = ?",
                    (counter, session_id, prefix)
                )
            else:
                counter = 1
                conn.execute(
                    "INSERT INTO id_counters (session_id, prefix, counter) VALUES (?, ?, ?)",
                    (session_id, prefix, counter)
                )

            return f"{session_prefix}-{prefix}-{counter:03d}"

    def add_item(
        self,
        session_id: str,
        item_type: str,
        content: str,
        item_id: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Add an item to the store.

        Args:
            session_id: Session ID
            item_type: Type of item (thought, requirement, task, etc.)
            content: Main content/text of the item
            item_id: Optional custom ID, auto-generated if not provided
            **kwargs: Additional fields based on item type

        Returns:
            item_id
        """
        # Generate ID if not provided
        if not item_id:
            prefix_map = {
                "thought": "THOUGHT",
                "requirement": "REQ",
                "recommendation": "REC",
                "task": "TASK",
                "milestone": "MS",
                "activity": "ACT",
                "decision": "DEC",
                "element": "ELM"
            }
            prefix = prefix_map.get(item_type, item_type.upper()[:3])
            item_id = self.generate_id(session_id, prefix)

        now = self._now()

        # Build insert statement
        fields = ["item_id", "session_id", "item_type", "content", "created_at", "updated_at"]
        values = [item_id, session_id, item_type, content, now, now]

        # Handle JSON fields
        json_fields = [
            "logic_operation", "query_results", "inferences", "contradictions",
            "justification", "assumptions", "assigned_to", "affected_files",
            "affected_entities", "files_analyzed", "metadata"
        ]

        for key, value in kwargs.items():
            if value is not None:
                fields.append(key)
                if key in json_fields and not isinstance(value, str):
                    values.append(json.dumps(value))
                else:
                    values.append(value)

        placeholders = ",".join("?" * len(values))
        fields_str = ",".join(fields)

        with self._get_connection() as conn:
            conn.execute(
                f"INSERT INTO items ({fields_str}) VALUES ({placeholders})",
                values
            )

        return item_id

    def get_item(self, item_id: str) -> Optional[Dict[str, Any]]:
        """Get item by ID."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM items WHERE item_id = ?",
                (item_id,)
            ).fetchone()

            if not row:
                return None

            return self._row_to_dict(row)

    def get_items(
        self,
        session_id: str,
        item_type: Optional[str] = None,
        status: Optional[str] = None,
        priority: Optional[str] = None,
        category: Optional[str] = None,
        phase: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Get items with optional filters.
        """
        query = "SELECT * FROM items WHERE session_id = ?"
        params = [session_id]

        if item_type:
            query += " AND item_type = ?"
            params.append(item_type)

        if status:
            query += " AND status = ?"
            params.append(status)

        if priority:
            query += " AND priority = ?"
            params.append(priority)

        if category:
            query += " AND category = ?"
            params.append(category)

        if phase:
            query += " AND phase = ?"
            params.append(phase)

        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [self._row_to_dict(row) for row in rows]

    def update_item(self, item_id: str, **updates) -> bool:
        """Update item fields."""
        if not updates:
            return True

        updates["updated_at"] = self._now()

        # Handle JSON fields
        json_fields = [
            "logic_operation", "query_results", "inferences", "contradictions",
            "justification", "assumptions", "assigned_to", "affected_files",
            "affected_entities", "files_analyzed", "metadata"
        ]
        for field in json_fields:
            if field in updates and not isinstance(updates[field], str):
                updates[field] = json.dumps(updates[field])

        set_clause = ", ".join(f"{k} = ?" for k in updates.keys())
        values = list(updates.values()) + [item_id]

        with self._get_connection() as conn:
            conn.execute(
                f"UPDATE items SET {set_clause} WHERE item_id = ?",
                values
            )

        return True

    def delete_item(self, item_id: str) -> Dict[str, int]:
        """Delete item and its relations."""
        with self._get_connection() as conn:
            # Delete relations
            cursor = conn.execute(
                "DELETE FROM relations WHERE source_id = ? OR target_id = ?",
                (item_id, item_id)
            )
            relations_deleted = cursor.rowcount

            # Delete item
            conn.execute("DELETE FROM items WHERE item_id = ?", (item_id,))

        return {"relations_deleted": relations_deleted}

    def delete_items_by_filter(
        self,
        session_id: str,
        item_type: Optional[str] = None,
        status: Optional[str] = None,
        priority: Optional[str] = None,
        category: Optional[str] = None,
        source_tool: Optional[str] = None
    ) -> Dict[str, int]:
        """
        Delete multiple items matching filters.

        Args:
            session_id: Session ID
            item_type: Filter by type (thought, requirement, recommendation, task, etc.)
            status: Filter by status
            priority: Filter by priority
            category: Filter by category
            source_tool: Filter by source_tool (e.g., 'refactoring_improving:find_large_classes')

        Returns:
            Dict with items_deleted and relations_deleted counts
        """
        # Build WHERE clause
        conditions = ["session_id = ?"]
        params = [session_id]

        if item_type:
            conditions.append("item_type = ?")
            params.append(item_type)

        if status:
            conditions.append("status = ?")
            params.append(status)

        if priority:
            conditions.append("priority = ?")
            params.append(priority)

        if category:
            conditions.append("category = ?")
            params.append(category)

        if source_tool:
            # Support partial matching with LIKE for source_tool
            if "%" in source_tool:
                conditions.append("source_tool LIKE ?")
            else:
                conditions.append("source_tool = ?")
            params.append(source_tool)

        where_clause = " AND ".join(conditions)

        with self._get_connection() as conn:
            # First, get item IDs to delete relations
            rows = conn.execute(
                f"SELECT item_id FROM items WHERE {where_clause}",
                params
            ).fetchall()
            item_ids = [row[0] for row in rows]

            if not item_ids:
                return {"items_deleted": 0, "relations_deleted": 0}

            # Delete relations for these items
            placeholders = ",".join("?" * len(item_ids))
            cursor = conn.execute(
                f"DELETE FROM relations WHERE source_id IN ({placeholders}) OR target_id IN ({placeholders})",
                item_ids + item_ids
            )
            relations_deleted = cursor.rowcount

            # Delete items
            cursor = conn.execute(
                f"DELETE FROM items WHERE {where_clause}",
                params
            )
            items_deleted = cursor.rowcount

        return {"items_deleted": items_deleted, "relations_deleted": relations_deleted}

    # =========================================================================
    # Relation Management
    # =========================================================================

    def add_relation(
        self,
        source_id: str,
        target_id: str,
        target_type: str,
        relation_type: str,
        metadata: Optional[Dict] = None
    ) -> int:
        """
        Add a relation between items or to external entities.

        Args:
            source_id: Source item ID
            target_id: Target item ID, file path, or entity name
            target_type: 'item', 'file', or 'entity'
            relation_type: Type of relation
            metadata: Optional additional data

        Returns:
            relation_id
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                """INSERT OR IGNORE INTO relations
                   (source_id, target_id, target_type, relation_type, metadata, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (source_id, target_id, target_type, relation_type,
                 json.dumps(metadata) if metadata else None, self._now())
            )
            return cursor.lastrowid

    def get_relations(
        self,
        item_id: str,
        direction: str = "both",
        relation_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get relations for an item.

        Args:
            item_id: Item ID
            direction: 'outgoing', 'incoming', or 'both'
            relation_type: Optional filter by relation type
        """
        relations = []

        with self._get_connection() as conn:
            if direction in ("outgoing", "both"):
                query = "SELECT * FROM relations WHERE source_id = ?"
                params = [item_id]
                if relation_type:
                    query += " AND relation_type = ?"
                    params.append(relation_type)

                rows = conn.execute(query, params).fetchall()
                relations.extend([self._row_to_dict(row) for row in rows])

            if direction in ("incoming", "both"):
                query = "SELECT * FROM relations WHERE target_id = ? AND target_type = 'item'"
                params = [item_id]
                if relation_type:
                    query += " AND relation_type = ?"
                    params.append(relation_type)

                rows = conn.execute(query, params).fetchall()
                relations.extend([self._row_to_dict(row) for row in rows])

        return relations

    def get_items_by_relation(
        self,
        target_id: str,
        relation_type: str,
        target_type: str = "item"
    ) -> List[Dict[str, Any]]:
        """Get items that have a relation to target."""
        with self._get_connection() as conn:
            rows = conn.execute(
                """SELECT i.* FROM items i
                   JOIN relations r ON i.item_id = r.source_id
                   WHERE r.target_id = ? AND r.target_type = ? AND r.relation_type = ?""",
                (target_id, target_type, relation_type)
            ).fetchall()

            return [self._row_to_dict(row) for row in rows]

    def delete_relation(self, relation_id: int) -> bool:
        """Delete a relation by ID."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM relations WHERE relation_id = ?", (relation_id,))
        return True

    # =========================================================================
    # Artifact Management
    # =========================================================================

    def add_artifact(
        self,
        session_id: str,
        file_path: str,
        artifact_type: str,
        tool_name: str,
        source_files: Optional[List[str]] = None
    ) -> int:
        """Record a generated artifact."""
        import hashlib

        # Calculate source file hashes
        source_hashes = {}
        if source_files:
            for f in source_files:
                if os.path.exists(f):
                    with open(f, 'rb') as fh:
                        source_hashes[f] = hashlib.md5(fh.read()).hexdigest()

        with self._get_connection() as conn:
            cursor = conn.execute(
                """INSERT OR REPLACE INTO artifacts
                   (session_id, file_path, artifact_type, tool_name, source_files, source_hashes, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (session_id, file_path, artifact_type, tool_name,
                 json.dumps(source_files) if source_files else None,
                 json.dumps(source_hashes) if source_hashes else None,
                 self._now())
            )
            return cursor.lastrowid

    def get_artifacts(self, session_id: str) -> List[Dict[str, Any]]:
        """Get all artifacts for session with freshness check."""
        import hashlib

        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM artifacts WHERE session_id = ?",
                (session_id,)
            ).fetchall()

        artifacts = []
        for row in rows:
            artifact = self._row_to_dict(row)

            # Check freshness
            artifact["fresh"] = True
            artifact["changed_sources"] = []

            if artifact.get("source_hashes"):
                source_hashes = json.loads(artifact["source_hashes"]) if isinstance(artifact["source_hashes"], str) else artifact["source_hashes"]
                for f, old_hash in source_hashes.items():
                    if os.path.exists(f):
                        with open(f, 'rb') as fh:
                            new_hash = hashlib.md5(fh.read()).hexdigest()
                        if new_hash != old_hash:
                            artifact["fresh"] = False
                            artifact["changed_sources"].append(f)
                    else:
                        artifact["fresh"] = False
                        artifact["changed_sources"].append(f)

            artifacts.append(artifact)

        return artifacts

    # =========================================================================
    # Summary Methods
    # =========================================================================

    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Get summary counts for a session."""
        with self._get_connection() as conn:
            # Count items by type
            type_counts = {}
            for row in conn.execute(
                "SELECT item_type, COUNT(*) as count FROM items WHERE session_id = ? GROUP BY item_type",
                (session_id,)
            ).fetchall():
                type_counts[row["item_type"]] = row["count"]

            # Count items by status
            status_counts = {}
            for row in conn.execute(
                "SELECT status, COUNT(*) as count FROM items WHERE session_id = ? GROUP BY status",
                (session_id,)
            ).fetchall():
                status_counts[row["status"]] = row["count"]

            # Get thought chain info
            thought_info = conn.execute(
                """SELECT MAX(thought_number) as max_thought, COUNT(*) as total_thoughts
                   FROM items WHERE session_id = ? AND item_type = 'thought'""",
                (session_id,)
            ).fetchone()

            return {
                "by_type": type_counts,
                "by_status": status_counts,
                "total_items": sum(type_counts.values()),
                "thought_chain": {
                    "max_number": thought_info["max_thought"] or 0,
                    "total": thought_info["total_thoughts"] or 0
                }
            }

    def get_thought_chain(
        self,
        session_id: str,
        branch_id: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get thought chain for session."""
        query = "SELECT * FROM items WHERE session_id = ? AND item_type = 'thought'"
        params = [session_id]

        if branch_id:
            query += " AND branch_id = ?"
            params.append(branch_id)
        else:
            query += " AND (branch_id IS NULL OR branch_id = '')"

        query += " ORDER BY thought_number ASC LIMIT ?"
        params.append(limit)

        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [self._row_to_dict(row) for row in rows]

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def _row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        """Convert SQLite row to dict with JSON parsing."""
        result = dict(row)

        # Parse JSON fields
        json_fields = [
            "knowledge_stats", "loaded_sources", "logic_operation", "query_results",
            "inferences", "contradictions", "justification", "assumptions",
            "assigned_to", "affected_files", "affected_entities", "files_analyzed",
            "metadata", "source_files", "source_hashes"
        ]

        for field in json_fields:
            if field in result and result[field]:
                try:
                    result[field] = json.loads(result[field])
                except (json.JSONDecodeError, TypeError) as e:
                    logger.debug(f"Could not parse JSON field '{field}': {e}")

        return result
