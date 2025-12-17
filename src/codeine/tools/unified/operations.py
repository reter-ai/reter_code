"""
Operations Handler for Unified Thinking System

Handles all operations that can be performed within a thinking step:
- RETER knowledge operations (assert, query, python_file, forget)
- Item creation (requirement, recommendation, task, milestone, etc.)
- Traceability operations (traces, derives, satisfies, etc.)
- Activity recording
"""

from typing import Any, Dict, List, Optional, Tuple
from .store import UnifiedStore


class OperationsHandler:
    """
    Handles operations dict from thinking() calls.

    Operations are processed in order:
    1. RETER knowledge operations
    2. Item creation operations
    3. Traceability operations
    """

    # Item type to ID prefix mapping
    TYPE_PREFIXES = {
        "requirement": "REQ",
        "constraint": "CON",
        "assumption": "ASM",
        "recommendation": "REC",
        "task": "TASK",
        "milestone": "MS",
        "activity": "ACT",
        "decision": "DEC",
        "element": "EL",
    }

    # Traceability relation keys and their target types
    TRACEABILITY_OPS = {
        "traces": ("item", "traces"),
        "derives": ("item", "derives"),
        "satisfies": ("item", "satisfies"),
        "verifies": ("item", "verifies"),
        "implements": ("item", "implements"),
        "depends_on": ("item", "depends_on"),
        "blocks": ("item", "blocks"),
        "affects": ("file", "affects"),
        "affects_entity": ("entity", "affects"),
    }

    def __init__(self, store: UnifiedStore, reter_engine: Optional[Any] = None):
        """
        Initialize handler.

        Args:
            store: UnifiedStore instance for item/relation operations
            reter_engine: Optional RETER engine for knowledge operations
        """
        self.store = store
        self.reter_engine = reter_engine

    def execute(
        self,
        session_id: str,
        thought_id: str,
        operations: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute all operations in the dict.

        Args:
            session_id: Current session ID
            thought_id: ID of the thought these operations belong to
            operations: Operations dict from thinking() call

        Returns:
            Dict with results for each operation type
        """
        results = {
            "success": True,
            "reter": {},
            "items_created": [],
            "items_updated": [],
            "relations_created": [],
            "errors": []
        }

        # 1. Execute RETER operations
        reter_results = self._execute_reter_ops(operations)
        results["reter"] = reter_results

        # 2. Execute item creation operations
        items_results = self._execute_item_ops(session_id, thought_id, operations)
        results["items_created"] = items_results.get("created", [])
        results["items_updated"] = items_results.get("updated", [])
        if items_results.get("errors"):
            results["errors"].extend(items_results["errors"])

        # 3. Execute traceability operations (link thought to targets)
        trace_results = self._execute_traceability_ops(thought_id, operations)
        results["relations_created"] = trace_results.get("created", [])
        if trace_results.get("errors"):
            results["errors"].extend(trace_results["errors"])

        # Set success based on errors
        if results["errors"]:
            results["success"] = False

        return results

    def _execute_reter_ops(self, operations: Dict[str, Any]) -> Dict[str, Any]:
        """Execute RETER knowledge operations."""
        results = {}

        # Assert ontology
        if "assert" in operations:
            if self.reter_engine:
                try:
                    result = self.reter_engine.add_knowledge(
                        source=operations["assert"],
                        type="ontology"
                    )
                    results["assert"] = {"success": True, "items_added": result.get("items_added", 0)}
                except Exception as e:
                    results["assert"] = {"success": False, "error": str(e)}
            else:
                results["assert"] = {"success": False, "error": "RETER engine not available"}

        # Execute query
        if "query" in operations:
            if self.reter_engine:
                try:
                    result = self.reter_engine.quick_query(
                        query=operations["query"],
                        type="reql"
                    )
                    results["query"] = {"success": True, "count": result.get("count", 0), "results": result.get("results", [])}
                except Exception as e:
                    results["query"] = {"success": False, "error": str(e)}
            else:
                results["query"] = {"success": False, "error": "RETER engine not available"}

        # Add Python file
        if "python_file" in operations:
            if self.reter_engine:
                try:
                    result = self.reter_engine.add_knowledge(
                        source=operations["python_file"],
                        type="python"
                    )
                    results["python_file"] = {"success": True, "items_added": result.get("items_added", 0)}
                except Exception as e:
                    results["python_file"] = {"success": False, "error": str(e)}
            else:
                results["python_file"] = {"success": False, "error": "RETER engine not available"}

        # Forget source
        if "forget_source" in operations:
            if self.reter_engine:
                try:
                    result = self.reter_engine.forget_source(source=operations["forget_source"])
                    results["forget_source"] = {"success": True}
                except Exception as e:
                    results["forget_source"] = {"success": False, "error": str(e)}
            else:
                results["forget_source"] = {"success": False, "error": "RETER engine not available"}

        return results

    def _execute_item_ops(
        self,
        session_id: str,
        thought_id: str,
        operations: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute item creation/update operations."""
        results = {"created": [], "updated": [], "errors": []}

        # Requirement
        if "requirement" in operations:
            item = self._create_requirement(session_id, thought_id, operations["requirement"])
            if item.get("error"):
                results["errors"].append(item["error"])
            else:
                results["created"].append(item)

        # Constraint
        if "constraint" in operations:
            item = self._create_constraint(session_id, thought_id, operations["constraint"])
            if item.get("error"):
                results["errors"].append(item["error"])
            else:
                results["created"].append(item)

        # Assumption
        if "assumption" in operations:
            item = self._create_assumption(session_id, thought_id, operations["assumption"])
            if item.get("error"):
                results["errors"].append(item["error"])
            else:
                results["created"].append(item)

        # Recommendation
        if "recommendation" in operations:
            item = self._create_recommendation(session_id, thought_id, operations["recommendation"])
            if item.get("error"):
                results["errors"].append(item["error"])
            else:
                results["created"].append(item)

        # Task
        if "task" in operations:
            item = self._create_task(session_id, thought_id, operations["task"])
            if item.get("error"):
                results["errors"].append(item["error"])
            else:
                results["created"].append(item)

        # Milestone
        if "milestone" in operations:
            item = self._create_milestone(session_id, thought_id, operations["milestone"])
            if item.get("error"):
                results["errors"].append(item["error"])
            else:
                results["created"].append(item)

        # Activity
        if "activity" in operations:
            item = self._create_activity(session_id, thought_id, operations["activity"])
            if item.get("error"):
                results["errors"].append(item["error"])
            else:
                results["created"].append(item)

        # Decision
        if "decision" in operations:
            item = self._create_decision(session_id, thought_id, operations["decision"])
            if item.get("error"):
                results["errors"].append(item["error"])
            else:
                results["created"].append(item)

        # Element
        if "element" in operations:
            item = self._create_element(session_id, thought_id, operations["element"])
            if item.get("error"):
                results["errors"].append(item["error"])
            else:
                results["created"].append(item)

        # Update item
        if "update_item" in operations:
            update = self._update_item(operations["update_item"])
            if update.get("error"):
                results["errors"].append(update["error"])
            else:
                results["updated"].append(update)

        # Update task
        if "update_task" in operations:
            update = self._update_task(operations["update_task"])
            if update.get("error"):
                results["errors"].append(update["error"])
            else:
                results["updated"].append(update)

        # Complete task
        if "complete_task" in operations:
            update = self._complete_task(operations["complete_task"])
            if update.get("error"):
                results["errors"].append(update["error"])
            else:
                results["updated"].append(update)

        return results

    def _execute_traceability_ops(
        self,
        source_id: str,
        operations: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute traceability operations."""
        results = {"created": [], "errors": []}

        for op_key, (target_type, relation_type) in self.TRACEABILITY_OPS.items():
            if op_key in operations:
                targets = operations[op_key]
                if isinstance(targets, str):
                    targets = [targets]

                for target_id in targets:
                    try:
                        relation_id = self.store.add_relation(
                            source_id=source_id,
                            target_id=target_id,
                            target_type=target_type,
                            relation_type=relation_type
                        )
                        results["created"].append({
                            "relation_id": relation_id,
                            "source": source_id,
                            "target": target_id,
                            "type": relation_type
                        })
                    except Exception as e:
                        results["errors"].append(f"Failed to create {relation_type} relation: {str(e)}")

        return results

    # =========================================================================
    # Item Creation Methods
    # =========================================================================

    def _create_requirement(
        self,
        session_id: str,
        thought_id: str,
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create a requirement item."""
        try:
            item_id = data.get("id") or self.store.generate_id(session_id, "REQ")

            created_id = self.store.add_item(
                session_id=session_id,
                item_type="requirement",
                content=data.get("text", ""),
                item_id=item_id,
                status="pending",
                priority=data.get("priority", "medium"),
                category=data.get("type", "functional"),
                verify_method=data.get("verify_method", "test"),
                risk=data.get("risk", "medium")
            )

            # Link requirement to source thought
            self.store.add_relation(created_id, thought_id, "item", "derives")

            return {"id": created_id, "type": "requirement"}
        except Exception as e:
            return {"error": f"Failed to create requirement: {str(e)}"}

    def _create_constraint(
        self,
        session_id: str,
        thought_id: str,
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create a constraint item."""
        try:
            item_id = data.get("id") or self.store.generate_id(session_id, "CON")

            created_id = self.store.add_item(
                session_id=session_id,
                item_type="constraint",
                content=data.get("text", ""),
                item_id=item_id,
                status="active",
                priority=data.get("priority", "medium"),
                category="constraint"
            )

            self.store.add_relation(created_id, thought_id, "item", "derives")

            return {"id": created_id, "type": "constraint"}
        except Exception as e:
            return {"error": f"Failed to create constraint: {str(e)}"}

    def _create_assumption(
        self,
        session_id: str,
        thought_id: str,
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create an assumption item."""
        try:
            item_id = data.get("id") or self.store.generate_id(session_id, "ASM")

            created_id = self.store.add_item(
                session_id=session_id,
                item_type="assumption",
                content=data.get("text", ""),
                item_id=item_id,
                status="active",
                priority=data.get("priority", "info"),
                category="assumption"
            )

            self.store.add_relation(created_id, thought_id, "item", "derives")

            return {"id": created_id, "type": "assumption"}
        except Exception as e:
            return {"error": f"Failed to create assumption: {str(e)}"}

    def _create_recommendation(
        self,
        session_id: str,
        thought_id: str,
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create a recommendation item."""
        try:
            item_id = data.get("id") or self.store.generate_id(session_id, "REC")

            created_id = self.store.add_item(
                session_id=session_id,
                item_type="recommendation",
                content=data.get("text", ""),
                item_id=item_id,
                status="pending",
                priority=data.get("priority", "medium"),
                category=data.get("category", "general")
            )

            self.store.add_relation(created_id, thought_id, "item", "derives")

            # Add affects relations for files
            affects = data.get("affects", [])
            if isinstance(affects, str):
                affects = [affects]
            for file_path in affects:
                self.store.add_relation(created_id, file_path, "file", "affects")

            return {"id": created_id, "type": "recommendation"}
        except Exception as e:
            return {"error": f"Failed to create recommendation: {str(e)}"}

    # Testing guidance to append to task content
    TASK_TESTING_GUIDANCE = (
        "\n\n[Testing Checklist]:\n"
        "- Use `code_inspection(action=\"find_tests\", target=\"<entity>\")` to find existing tests\n"
        "- If no tests exist, create unit tests before making changes\n"
        "- Run tests after completion to verify no regressions"
    )

    def _create_task(
        self,
        session_id: str,
        thought_id: str,
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create a task item with testing guidance."""
        try:
            item_id = data.get("id") or self.store.generate_id(session_id, "TASK")

            # Calculate end_date if start_date and duration provided
            start_date = data.get("start_date")
            duration_days = data.get("duration_days")
            end_date = data.get("end_date")

            if start_date and duration_days and not end_date:
                from datetime import datetime, timedelta
                start = datetime.strptime(start_date, "%Y-%m-%d")
                end = start + timedelta(days=duration_days)
                end_date = end.strftime("%Y-%m-%d")

            # Handle assigned_to as JSON
            import json
            assigned_to = data.get("assigned_to")
            if isinstance(assigned_to, list):
                assigned_to = json.dumps(assigned_to)

            # Add testing guidance to task content unless explicitly disabled
            task_name = data.get("name", "")
            if not data.get("skip_testing_guidance", False):
                task_content = task_name + self.TASK_TESTING_GUIDANCE
            else:
                task_content = task_name

            created_id = self.store.add_item(
                session_id=session_id,
                item_type="task",
                content=task_content,
                item_id=item_id,
                status="pending",
                priority=data.get("priority", "medium"),
                phase=data.get("phase"),
                start_date=start_date,
                end_date=end_date,
                duration_days=duration_days,
                assigned_to=assigned_to
            )

            self.store.add_relation(created_id, thought_id, "item", "derives")

            # Handle depends_on
            depends_on = data.get("depends_on", [])
            if isinstance(depends_on, str):
                depends_on = [depends_on]
            for dep_id in depends_on:
                self.store.add_relation(created_id, dep_id, "item", "depends_on")

            return {"id": created_id, "type": "task"}
        except Exception as e:
            return {"error": f"Failed to create task: {str(e)}"}

    def _create_milestone(
        self,
        session_id: str,
        thought_id: str,
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create a milestone item."""
        try:
            item_id = data.get("id") or self.store.generate_id(session_id, "MS")

            created_id = self.store.add_item(
                session_id=session_id,
                item_type="milestone",
                content=data.get("name", ""),
                item_id=item_id,
                status="pending",
                priority="high",
                end_date=data.get("target_date"),
                duration_days=0
            )

            self.store.add_relation(created_id, thought_id, "item", "derives")

            # Handle depends_on
            depends_on = data.get("depends_on", [])
            if isinstance(depends_on, str):
                depends_on = [depends_on]
            for dep_id in depends_on:
                self.store.add_relation(created_id, dep_id, "item", "depends_on")

            return {"id": created_id, "type": "milestone"}
        except Exception as e:
            return {"error": f"Failed to create milestone: {str(e)}"}

    def _create_activity(
        self,
        session_id: str,
        thought_id: str,
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create an activity item."""
        try:
            item_id = self.store.generate_id(session_id, "ACT")

            import json
            metadata = {
                "tool": data.get("tool", ""),
                "params": data.get("params", ""),
                "result": data.get("result", ""),
                "files_analyzed": data.get("files_analyzed", []),
                "issues_found": data.get("issues_found", 0)
            }

            created_id = self.store.add_item(
                session_id=session_id,
                item_type="activity",
                content=f"{data.get('tool', 'unknown')}: {data.get('params', '')}",
                item_id=item_id,
                status="completed",
                source_tool=data.get("tool"),
                metadata=json.dumps(metadata)
            )

            self.store.add_relation(created_id, thought_id, "item", "derives")

            return {"id": created_id, "type": "activity"}
        except Exception as e:
            return {"error": f"Failed to create activity: {str(e)}"}

    def _create_decision(
        self,
        session_id: str,
        thought_id: str,
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create a decision item."""
        try:
            item_id = data.get("id") or self.store.generate_id(session_id, "DEC")

            import json
            metadata = {"rationale": data.get("rationale", "")}

            created_id = self.store.add_item(
                session_id=session_id,
                item_type="decision",
                content=data.get("text", ""),
                item_id=item_id,
                status="active",
                priority="high",
                metadata=json.dumps(metadata)
            )

            self.store.add_relation(created_id, thought_id, "item", "derives")

            return {"id": created_id, "type": "decision"}
        except Exception as e:
            return {"error": f"Failed to create decision: {str(e)}"}

    def _create_element(
        self,
        session_id: str,
        thought_id: str,
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create a design element item."""
        try:
            item_id = data.get("id") or self.store.generate_id(session_id, "EL")

            import json
            metadata = {
                "element_type": data.get("type", ""),
                "docref": data.get("docref", "")
            }

            created_id = self.store.add_item(
                session_id=session_id,
                item_type="element",
                content=data.get("id", item_id),
                item_id=item_id,
                status="active",
                category=data.get("type", "component"),
                metadata=json.dumps(metadata)
            )

            self.store.add_relation(created_id, thought_id, "item", "derives")

            # Add file reference if docref provided
            if data.get("docref"):
                self.store.add_relation(created_id, data["docref"], "file", "affects")

            return {"id": created_id, "type": "element"}
        except Exception as e:
            return {"error": f"Failed to create element: {str(e)}"}

    # =========================================================================
    # Item Update Methods
    # =========================================================================

    def _update_item(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Update any item by ID."""
        try:
            item_id = data.get("id")
            if not item_id:
                return {"error": "update_item requires 'id' field"}

            # Extract update fields
            updates = {k: v for k, v in data.items() if k != "id"}

            self.store.update_item(item_id, **updates)

            return {"id": item_id, "updated_fields": list(updates.keys())}
        except Exception as e:
            return {"error": f"Failed to update item: {str(e)}"}

    def _update_task(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Update task-specific fields."""
        try:
            item_id = data.get("id")
            if not item_id:
                return {"error": "update_task requires 'id' field"}

            # Map task-specific fields
            updates = {}
            if "status" in data:
                updates["status"] = data["status"]
            if "progress" in data:
                updates["progress"] = data["progress"]
            if "actual_start" in data:
                updates["actual_start"] = data["actual_start"]
            if "actual_end" in data:
                updates["actual_end"] = data["actual_end"]
            if "assigned_to" in data:
                import json
                assigned = data["assigned_to"]
                updates["assigned_to"] = json.dumps(assigned) if isinstance(assigned, list) else assigned

            self.store.update_item(item_id, **updates)

            return {"id": item_id, "type": "task", "updated_fields": list(updates.keys())}
        except Exception as e:
            return {"error": f"Failed to update task: {str(e)}"}

    def _complete_task(self, task_id: str) -> Dict[str, Any]:
        """Mark a task as complete."""
        try:
            from datetime import datetime

            self.store.update_item(
                task_id,
                status="completed",
                progress=100,
                actual_end=datetime.now().strftime("%Y-%m-%d")
            )

            return {"id": task_id, "type": "task", "status": "completed"}
        except Exception as e:
            return {"error": f"Failed to complete task: {str(e)}"}
