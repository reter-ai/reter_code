"""Logical Thinking Store for SQLite-based persistence.

Provides session and thought persistence for the logical_thinking tool.
Sessions are stored per RETER instance, allowing continuation across runs.

UPDATED: Now uses UnifiedStore as backend for single database consolidation.
"""

import json
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

from ..unified.store import UnifiedStore
from ..dataclasses import ThoughtStoreData, LogicData, ReasoningMeta, BranchingInfo


class LogicalThinkingStore:
    """
    SQLite-based store for logical thinking sessions and thoughts.

    @reter-cnl: This is-in-layer Service-Layer.
    @reter-cnl: This is a repository.

    Features:
    - Session persistence per RETER instance
    - Thought history with branch support
    - JSON serialization for complex fields
    - Automatic session creation/resumption

    UPDATED: Uses UnifiedStore for single database (.unified.sqlite)
    """

    def __init__(self, store: Optional[UnifiedStore] = None):
        """
        Initialize the logical thinking store.

        Args:
            store: Existing UnifiedStore instance. If None, creates a new one.
        """
        self._store = store or UnifiedStore()

    # =========================================================================
    # Session Management (delegated to UnifiedStore)
    # =========================================================================

    def get_or_create_session(
        self,
        instance_name: str,
        goal: str = "Logical reasoning session"
    ) -> Dict[str, Any]:
        """
        Get the active session for an instance, or create a new one.

        Args:
            instance_name: RETER instance name
            goal: Session goal description

        Returns:
            Dict with session data
        """
        # Use UnifiedStore's session management
        session_id = self._store.get_or_create_session(instance_name, goal)

        # Get session details
        session = self._store.get_session(session_id)
        if session:
            # Get thought count
            thoughts = self._store.get_thought_chain(session_id)
            branches = self._get_branches(session_id)

            return {
                "session_id": session_id,
                "instance_name": instance_name,
                "goal": session.get("goal"),
                "context": session.get("context"),
                "status": session.get("status", "active"),
                "knowledge_stats": json.loads(session.get("knowledge_stats") or "{}"),
                "loaded_sources": json.loads(session.get("loaded_sources") or "[]"),
                "branches": branches,
                "thought_history_length": len(thoughts),
                "created_at": session.get("created_at"),
                "updated_at": session.get("updated_at")
            }

        # Fallback if session not found
        return {
            "session_id": session_id,
            "instance_name": instance_name,
            "goal": goal,
            "context": None,
            "status": "active",
            "knowledge_stats": {},
            "loaded_sources": [],
            "branches": [],
            "thought_history_length": 0,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a session by ID.

        Args:
            session_id: Session UUID

        Returns:
            Session dict or None if not found
        """
        session = self._store.get_session(session_id)
        if not session:
            return None

        thoughts = self._store.get_thought_chain(session_id)
        branches = self._get_branches(session_id)

        return {
            "session_id": session_id,
            "instance_name": session.get("instance_name"),
            "goal": session.get("goal"),
            "context": session.get("context"),
            "status": session.get("status", "active"),
            "knowledge_stats": json.loads(session.get("knowledge_stats") or "{}"),
            "loaded_sources": json.loads(session.get("loaded_sources") or "[]"),
            "branches": branches,
            "thought_history_length": len(thoughts),
            "created_at": session.get("created_at"),
            "updated_at": session.get("updated_at")
        }

    def update_session(
        self,
        session_id: str,
        status: Optional[str] = None,
        knowledge_stats: Optional[Dict] = None,
        loaded_sources: Optional[List[str]] = None
    ) -> bool:
        """
        Update session fields.

        Args:
            session_id: Session UUID
            status: New status (active, completed, paused, failed)
            knowledge_stats: Updated knowledge statistics
            loaded_sources: Updated list of loaded sources

        Returns:
            True if updated, False if not found
        """
        updates = {}
        if status is not None:
            updates["status"] = status
        if knowledge_stats is not None:
            updates["knowledge_stats"] = json.dumps(knowledge_stats)
        if loaded_sources is not None:
            updates["loaded_sources"] = json.dumps(loaded_sources)

        if not updates:
            return True

        try:
            self._store.update_session(session_id, **updates)
            return True
        except (sqlite3.Error, OSError):
            # Database or file system error
            return False

    def add_loaded_source(self, session_id: str, source: str) -> bool:
        """
        Add a source to the session's loaded sources list.

        Args:
            session_id: Session UUID
            source: Source identifier to add

        Returns:
            True if updated
        """
        session = self._store.get_session(session_id)
        if not session:
            return False

        sources = json.loads(session.get('loaded_sources') or '[]')
        if source not in sources:
            sources.append(source)
            self._store.update_session(session_id, loaded_sources=json.dumps(sources))
        return True

    def remove_loaded_source(self, session_id: str, source: str) -> bool:
        """
        Remove a source from the session's loaded sources list.

        Args:
            session_id: Session UUID
            source: Source identifier to remove

        Returns:
            True if updated
        """
        session = self._store.get_session(session_id)
        if not session:
            return False

        sources = json.loads(session.get('loaded_sources') or '[]')
        if source in sources:
            sources.remove(source)
            self._store.update_session(session_id, loaded_sources=json.dumps(sources))
        return True

    # =========================================================================
    # Thought Management (using UnifiedStore items table)
    # =========================================================================

    def add_thought(
        self,
        session_id: str,
        data: ThoughtStoreData,
    ) -> str:
        """
        Add a thought to a session.

        Args:
            session_id: Session UUID
            data: ThoughtStoreData containing all thought parameters

        Returns:
            thought_id: UUID of the created thought

        Example:
            # Using structured data
            data = ThoughtStoreData(
                thought="Analysis complete",
                thought_number=1,
                total_thoughts=3,
                next_thought_needed=True,
                logic=LogicData(query_results={"count": 10}),
                reasoning=ReasoningMeta(confidence=0.95),
            )
            thought_id = store.add_thought(session_id, data)

            # Using from_params for backward compatibility
            data = ThoughtStoreData.from_params(
                thought="Analysis",
                thought_number=1,
                total_thoughts=3,
                next_thought_needed=True,
                confidence=0.9,
            )
            thought_id = store.add_thought(session_id, data)
        """
        thought_id = self._store.add_item(
            session_id=session_id,
            item_type="thought",
            content=data.thought,
            status="completed",
            metadata=data.to_metadata()
        )

        return thought_id

    def add_thought_from_params(
        self,
        session_id: str,
        thought: str,
        thought_number: int,
        total_thoughts: int,
        next_thought_needed: bool,
        thought_type: str = "reasoning",
        logic_operation: Optional[Dict] = None,
        query_results: Optional[Dict] = None,
        inferences: Optional[List] = None,
        contradictions: Optional[List[str]] = None,
        confidence: float = 1.0,
        justification: Optional[List[str]] = None,
        assumptions: Optional[List[str]] = None,
        is_revision: bool = False,
        revises_thought: Optional[int] = None,
        branch_from_thought: Optional[int] = None,
        branch_id: Optional[str] = None,
        needs_more_thoughts: bool = False
    ) -> str:
        """
        Add a thought using individual parameters (backward compatibility).

        This method preserves the original 19-parameter signature for
        backward compatibility. New code should use add_thought() with
        ThoughtStoreData instead.

        Returns:
            thought_id: UUID of the created thought
        """
        data = ThoughtStoreData.from_params(
            thought=thought,
            thought_number=thought_number,
            total_thoughts=total_thoughts,
            next_thought_needed=next_thought_needed,
            thought_type=thought_type,
            logic_operation=logic_operation,
            query_results=query_results,
            inferences=inferences,
            contradictions=contradictions,
            confidence=confidence,
            justification=justification,
            assumptions=assumptions,
            is_revision=is_revision,
            revises_thought=revises_thought,
            branch_from_thought=branch_from_thought,
            branch_id=branch_id,
            needs_more_thoughts=needs_more_thoughts,
        )
        return self.add_thought(session_id, data)

    def get_thought(self, thought_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a thought by ID.

        Args:
            thought_id: Thought UUID

        Returns:
            Thought dict or None if not found
        """
        item = self._store.get_item(thought_id)
        if not item or item.get("item_type") != "thought":
            return None
        return self._item_to_thought(item)

    def get_session_thoughts(
        self,
        session_id: str,
        branch_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Get all thoughts for a session.

        Args:
            session_id: Session UUID
            branch_id: Optional branch filter (None = main branch)
            limit: Maximum number of thoughts to return
            offset: Number of thoughts to skip

        Returns:
            List of thought dicts
        """
        thoughts = self._store.get_thought_chain(session_id)

        # Filter by branch
        if branch_id is not None:
            thoughts = [t for t in thoughts if t.get("branch_id") == branch_id]
        else:
            thoughts = [t for t in thoughts if not t.get("branch_id")]

        # Apply pagination
        thoughts = thoughts[offset:offset + limit]

        return [self._item_to_thought(t) for t in thoughts]

    def get_session_branches(self, session_id: str) -> List[str]:
        """
        Get all branch IDs for a session.

        Args:
            session_id: Session UUID

        Returns:
            List of branch IDs
        """
        return self._get_branches(session_id)

    def _get_branches(self, session_id: str) -> List[str]:
        """Get distinct branch IDs from thoughts."""
        thoughts = self._store.get_thought_chain(session_id)
        branches = set()
        for t in thoughts:
            branch_id = t.get("branch_id")
            if branch_id:
                branches.add(branch_id)
        return list(branches)

    def count_session_thoughts(self, session_id: str) -> int:
        """
        Count total thoughts in a session (all branches).

        Args:
            session_id: Session UUID

        Returns:
            Number of thoughts
        """
        thoughts = self._store.get_thought_chain(session_id)
        return len(thoughts)

    def count_main_branch_thoughts(self, session_id: str) -> int:
        """
        Count thoughts in the main branch (no branch_id).

        Args:
            session_id: Session UUID

        Returns:
            Number of thoughts in main branch
        """
        thoughts = self._store.get_thought_chain(session_id)
        return len([t for t in thoughts if not t.get("branch_id")])

    # =========================================================================
    # Cleanup
    # =========================================================================

    def clear_session(self, session_id: str) -> bool:
        """
        Delete a session and all its thoughts.

        Args:
            session_id: Session UUID

        Returns:
            True if deleted
        """
        result = self._store.clear_session(session_id)
        return result.get("items_deleted", 0) >= 0

    def clear_instance_sessions(self, instance_name: str) -> int:
        """
        Delete all sessions for a RETER instance.

        Args:
            instance_name: RETER instance name

        Returns:
            Number of sessions deleted
        """
        # Find all sessions for this instance and clear them
        session = self._store.get_session_by_instance(instance_name)
        count = 0
        while session:
            self._store.clear_session(session["session_id"])
            count += 1
            session = self._store.get_session_by_instance(instance_name)
        return count

    # =========================================================================
    # Helpers
    # =========================================================================

    def _item_to_thought(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a UnifiedStore item to a thought dict."""
        # UnifiedStore already parses JSON fields, so we just map the fields
        metadata = item.get("metadata") or {}
        if isinstance(metadata, str):
            metadata = json.loads(metadata)

        return {
            "thought_id": item.get("item_id"),
            "session_id": item.get("session_id"),
            "thought": item.get("content"),
            "thought_number": item.get("thought_number") or metadata.get("thought_number"),
            "total_thoughts": item.get("total_thoughts") or metadata.get("total_thoughts"),
            "next_thought_needed": bool(item.get("next_thought_needed", 1) or metadata.get("next_thought_needed", True)),
            "thought_type": item.get("thought_type") or metadata.get("thought_type", "reasoning"),
            "logic_operation": item.get("logic_operation") or metadata.get("logic_operation"),
            "query_results": item.get("query_results") or metadata.get("query_results"),
            "inferences": item.get("inferences") or metadata.get("inferences"),
            "contradictions": item.get("contradictions") or metadata.get("contradictions"),
            "confidence": item.get("confidence", 1.0) or metadata.get("confidence", 1.0),
            "justification": item.get("justification") or metadata.get("justification"),
            "assumptions": item.get("assumptions") or metadata.get("assumptions"),
            "is_revision": bool(item.get("is_revision", 0) or metadata.get("is_revision", False)),
            "revises_thought": item.get("revises_thought") or metadata.get("revises_thought"),
            "branch_from_thought": item.get("branch_from_thought") or metadata.get("branch_from_thought"),
            "branch_id": item.get("branch_id") or metadata.get("branch_id"),
            "needs_more_thoughts": bool(item.get("needs_more_thoughts", 0) or metadata.get("needs_more_thoughts", False)),
            "timestamp": item.get("created_at")
        }
