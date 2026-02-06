"""
RETER Session Handler.

Handles session and thinking operations for Agent SDK integration.
Uses UnifiedStore and ThinkingSession directly.

::: This is-in-layer Handler-Layer.
::: This is-in-component System-Handlers.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

from . import BaseHandler
from ..protocol import METHOD_SESSION, METHOD_THINKING, METHOD_ITEMS


class SessionHandler(BaseHandler):
    """Handler for session and thinking operations.

    ::: This is-in-layer Service-Layer.
    ::: This is a handler.
    ::: This is stateful.
    """

    def __init__(self, context):
        super().__init__(context)
        self._sessions = {}  # instance_name -> ThinkingSession
        self._store = None

    def _get_store(self):
        """Get or create the UnifiedStore."""
        if self._store is None:
            from ...tools.unified import UnifiedStore

            # Use project root for store location
            project_root = os.getenv("RETER_PROJECT_ROOT")
            if project_root:
                db_path = Path(project_root) / ".reter_code" / ".unified.sqlite"
            else:
                db_path = Path.cwd() / ".reter_code" / ".unified.sqlite"

            db_path.parent.mkdir(parents=True, exist_ok=True)
            self._store = UnifiedStore(str(db_path))

        return self._store

    def _get_session(self, instance_name: str = "default"):
        """Get or create a ThinkingSession for the instance."""
        if instance_name not in self._sessions:
            from ...tools.unified import ThinkingSession

            store = self._get_store()

            # Get RETER engine (optional)
            reter_engine = self.reter if hasattr(self, 'reter') else None

            self._sessions[instance_name] = ThinkingSession(store, reter_engine)

        return self._sessions[instance_name]

    def _register_methods(self) -> None:
        """Register session method handlers."""
        self._methods = {
            METHOD_SESSION: self._handle_session,
            METHOD_THINKING: self._handle_thinking,
            METHOD_ITEMS: self._handle_items,
        }

    def can_handle(self, method: str) -> bool:
        """Check if this handler can process the method."""
        return method in self._methods

    def _handle_session(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle session lifecycle operations.

        Params:
            action: Session action (start, context, end, clear)
            goal: Session goal (for start action)
            project_start: Project start date
            project_end: Project end date

        Returns:
            Session state and context
        """
        action = params.get("action", "context")
        instance_name = "default"
        session = self._get_session(instance_name)

        if action == "start":
            return session.start_session(
                instance_name=instance_name,
                goal=params.get("goal"),
                project_start=params.get("project_start"),
                project_end=params.get("project_end")
            )
        elif action == "context":
            return session.get_context(instance_name)
        elif action == "end":
            return session.end_session(instance_name)
        elif action == "clear":
            return session.clear_session(instance_name)
        else:
            raise ValueError(f"Unknown session action: {action}")

    def _handle_thinking(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle thinking/reasoning operations.

        Params:
            thought: The reasoning step content
            thought_number: Current step number
            total_thoughts: Estimated total steps
            thought_type: Type (reasoning, analysis, decision, etc.)
            section: Design doc section
            operations: Operations to execute
            branch_id: Branch identifier
            is_revision: Whether this revises a previous thought

        Returns:
            Thought creation result
        """
        instance_name = "default"
        session = self._get_session(instance_name)

        return session.think(
            instance_name=instance_name,
            thought=params.get("thought", ""),
            thought_number=params.get("thought_number", 1),
            total_thoughts=params.get("total_thoughts", 1),
            thought_type=params.get("thought_type", "reasoning"),
            section=params.get("section"),
            operations=params.get("operations"),
            branch_id=params.get("branch_id"),
            branch_from=params.get("branch_from"),
            is_revision=params.get("is_revision", False),
            revises_thought=params.get("revises_thought"),
            next_thought_needed=params.get("next_thought_needed", True),
            needs_more_thoughts=params.get("needs_more_thoughts", False)
        )

    def _handle_items(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle item operations (query, get, update, delete).

        Params:
            action: Item action (list, get, delete, update, clear)
            item_id: Item ID for get/delete/update
            updates: Fields to update
            item_type: Filter by type
            status: Filter by status
            priority: Filter by priority
            ... other filters

        Returns:
            Item operation result
        """
        action = params.get("action", "list")
        instance_name = "default"
        store = self._get_store()

        # Get session ID
        session_id = store.get_or_create_session(instance_name)

        if action == "list":
            # Build query params
            query_params = {}
            for key in ["item_type", "status", "priority", "phase", "category", "source_tool", "classification"]:
                if params.get(key):
                    query_params[key] = params[key]
            query_params["limit"] = params.get("limit", 100)
            query_params["offset"] = params.get("offset", 0)

            items = store.get_items(session_id, **query_params)

            return {
                "success": True,
                "items": items,
                "count": len(items),
                "has_more": len(items) >= query_params["limit"]
            }

        elif action == "get":
            item_id = params.get("item_id")
            if not item_id:
                raise ValueError("item_id is required for get action")

            item = store.get_item(item_id)
            if not item:
                return {"success": False, "error": f"Item {item_id} not found"}

            result = {"success": True, "item": item}

            if params.get("include_relations", True):
                relations = store.get_relations(item_id)
                result["relations"] = relations

            return result

        elif action == "update":
            item_id = params.get("item_id")
            if not item_id:
                raise ValueError("item_id is required for update action")

            updates = params.get("updates", {})

            # Handle special update fields
            if params.get("status"):
                updates["status"] = params["status"]
            if params.get("subject"):
                updates["subject"] = params["subject"]
            if params.get("description"):
                updates["description"] = params["description"]
            if params.get("activeForm"):
                updates["active_form"] = params["activeForm"]
            if params.get("owner"):
                updates["owner"] = params["owner"]

            success = store.update_item(item_id, **updates)
            if success:
                item = store.get_item(item_id)
                return {"success": True, "item": item}
            else:
                return {"success": False, "error": f"Failed to update item {item_id}"}

        elif action == "delete":
            item_id = params.get("item_id")
            if not item_id:
                raise ValueError("item_id is required for delete action")

            result = store.delete_item(item_id)
            return {
                "success": True,
                "deleted_relations": result.get("relations_deleted", 0)
            }

        elif action == "clear":
            result = store.delete_items_by_filter(
                session_id=session_id,
                item_type=params.get("item_type"),
                status=params.get("status")
            )
            return {
                "success": True,
                "items_deleted": result.get("items_deleted", 0),
                "relations_deleted": result.get("relations_deleted", 0)
            }

        elif action == "classify":
            # Classify a task as TP or FP
            # For TP classifications, optionally create a follow-up implementation task
            item_id = params.get("item_id")
            if not item_id:
                raise ValueError("item_id is required for classify action")

            classification = params.get("classification")
            if not classification:
                raise ValueError("classification is required for classify action")

            valid_classifications = [
                "TP-EXTRACT", "TP-PARAMETERIZE", "PARTIAL-TP",
                "FP-INTERFACE", "FP-LAYERS", "FP-STRUCTURAL", "FP-TRIVIAL"
            ]
            if classification not in valid_classifications:
                raise ValueError(f"Invalid classification: {classification}. "
                               f"Valid values: {valid_classifications}")

            # Get existing item
            item = store.get_item(item_id)
            if not item:
                return {"success": False, "error": f"Item {item_id} not found"}

            # Update metadata with classification
            from datetime import datetime, timezone
            metadata = item.get("metadata", {}) or {}
            metadata["classification"] = classification
            metadata["classification_notes"] = params.get("notes", "")
            metadata["classified_at"] = datetime.now(timezone.utc).isoformat()

            success = store.update_item(item_id, metadata=metadata)
            if not success:
                return {"success": False, "error": f"Failed to classify item {item_id}"}

            item = store.get_item(item_id)
            result = {"success": True, "item": item}

            # For TP classifications, optionally create follow-up task
            if classification.startswith("TP") or classification == "PARTIAL-TP":
                if params.get("create_followup", False):
                    # Generate default follow-up name based on classification
                    default_names = {
                        "TP-EXTRACT": f"Implement extraction: {item.get('content', '')}",
                        "TP-PARAMETERIZE": f"Implement parameterization: {item.get('content', '')}",
                        "PARTIAL-TP": f"Review and implement: {item.get('content', '')}",
                    }
                    followup_name = params.get("followup_name", default_names.get(classification, f"Implement: {item.get('content', '')}"))

                    # Get prompt from original task metadata or use custom prompt
                    source_metadata = item.get("metadata", {}) or {}
                    prompt = params.get("followup_prompt") or source_metadata.get("prompt", "")

                    # Build follow-up metadata
                    followup_metadata = {
                        "derived_from": item_id,
                        "original_classification": classification,
                    }
                    if prompt:
                        followup_metadata["prompt"] = prompt

                    # Copy relevant source metadata
                    for key in ["avg_similarity", "cluster_id", "member_count", "members", "affected_files"]:
                        if key in source_metadata:
                            followup_metadata[key] = source_metadata[key]

                    # Create follow-up task
                    followup_id = store.add_item(
                        session_id=session_id,
                        item_type="task",
                        content=followup_name,
                        description=params.get("followup_description") or prompt or f"Follow-up from {item_id}",
                        category=params.get("followup_category", "refactor"),
                        priority=params.get("followup_priority", item.get("priority", "medium")),
                        status="pending",
                        source_tool=item.get("source_tool"),
                        metadata=followup_metadata
                    )

                    # Link follow-up to original via derives relation
                    store.add_relation(
                        source_id=followup_id,
                        target_id=item_id,
                        target_type="item",
                        relation_type="derives"
                    )

                    # Add affects relations for files from source metadata
                    members = source_metadata.get("members", [])
                    affected_files = set()
                    for m in members:
                        if isinstance(m, dict):
                            f = m.get("file") or m.get("source_file")
                            if f:
                                affected_files.add(f)
                    for f in affected_files:
                        store.add_relation(
                            source_id=followup_id,
                            target_id=f,
                            target_type="file",
                            relation_type="affects"
                        )

                    # Optionally mark original as completed
                    if params.get("complete_original", False):
                        store.update_item(item_id, status="completed")
                        item = store.get_item(item_id)
                        result["item"] = item

                    result["followup_task"] = store.get_item(followup_id)

            return result

        elif action == "verify":
            # Mark a task as verified
            item_id = params.get("item_id")
            if not item_id:
                raise ValueError("item_id is required for verify action")

            # Get existing item
            item = store.get_item(item_id)
            if not item:
                return {"success": False, "error": f"Item {item_id} not found"}

            # Update metadata with verification status
            from datetime import datetime, timezone
            metadata = item.get("metadata", {}) or {}
            metadata["verified"] = True
            metadata["verified_at"] = datetime.now(timezone.utc).isoformat()
            metadata["verified_by"] = params.get("verified_by", "user")

            # Optionally update status to verified
            updates = {"metadata": metadata}
            if params.get("update_status", False):
                updates["status"] = "verified"

            success = store.update_item(item_id, **updates)
            if success:
                item = store.get_item(item_id)
                return {"success": True, "item": item}
            else:
                return {"success": False, "error": f"Failed to verify item {item_id}"}

        else:
            raise ValueError(f"Unknown items action: {action}")


__all__ = ["SessionHandler"]
