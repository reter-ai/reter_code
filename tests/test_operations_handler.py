"""
Tests for OperationsHandler

Uses shared fixtures from conftest.py:
- store: Temporary UnifiedStore instance
- handler: OperationsHandler instance
- session_with_thought: Pre-created session with thought (session_id, thought_id)
"""

import pytest


class TestItemCreation:
    """Test item creation operations."""

    def test_create_requirement(self, handler, store, session_with_thought):
        """Test creating a requirement."""
        session_id, thought_id = session_with_thought

        result = handler.execute(
            session_id=session_id,
            thought_id=thought_id,
            operations={
                "requirement": {
                    "text": "System shall support OAuth2",
                    "risk": "medium",
                    "verify_method": "test"
                }
            }
        )

        assert result["success"]
        assert len(result["items_created"]) == 1
        assert result["items_created"][0]["type"] == "requirement"

        # Verify item in store
        req_id = result["items_created"][0]["id"]
        item = store.get_item(req_id)
        assert item["content"] == "System shall support OAuth2"
        assert item["verify_method"] == "test"
        assert item["risk"] == "medium"

    def test_create_requirement_with_custom_id(self, handler, store, session_with_thought):
        """Test creating requirement with custom ID."""
        session_id, thought_id = session_with_thought

        result = handler.execute(
            session_id=session_id,
            thought_id=thought_id,
            operations={
                "requirement": {
                    "id": "REQ-CUSTOM",
                    "text": "Custom requirement"
                }
            }
        )

        assert result["success"]
        assert result["items_created"][0]["id"] == "REQ-CUSTOM"

    def test_create_constraint(self, handler, store, session_with_thought):
        """Test creating a constraint."""
        session_id, thought_id = session_with_thought

        result = handler.execute(
            session_id=session_id,
            thought_id=thought_id,
            operations={
                "constraint": {
                    "text": "Must use existing database"
                }
            }
        )

        assert result["success"]
        assert result["items_created"][0]["type"] == "constraint"
        assert "-CON-" in result["items_created"][0]["id"]

    def test_create_assumption(self, handler, store, session_with_thought):
        """Test creating an assumption."""
        session_id, thought_id = session_with_thought

        result = handler.execute(
            session_id=session_id,
            thought_id=thought_id,
            operations={
                "assumption": {
                    "text": "Assuming high traffic load"
                }
            }
        )

        assert result["success"]
        assert result["items_created"][0]["type"] == "assumption"
        assert "-ASM-" in result["items_created"][0]["id"]

    def test_create_recommendation(self, handler, store, session_with_thought):
        """Test creating a recommendation."""
        session_id, thought_id = session_with_thought

        result = handler.execute(
            session_id=session_id,
            thought_id=thought_id,
            operations={
                "recommendation": {
                    "text": "Add error handling to auth module",
                    "priority": "high",
                    "category": "reliability",
                    "affects": ["auth/module.py"]
                }
            }
        )

        assert result["success"]
        assert result["items_created"][0]["type"] == "recommendation"

        # Verify file relation
        rec_id = result["items_created"][0]["id"]
        relations = store.get_relations(rec_id, direction="outgoing")
        file_relations = [r for r in relations if r["target_type"] == "file"]
        assert len(file_relations) == 1
        assert file_relations[0]["target_id"] == "auth/module.py"

    def test_create_task(self, handler, store, session_with_thought):
        """Test creating a task."""
        session_id, thought_id = session_with_thought

        result = handler.execute(
            session_id=session_id,
            thought_id=thought_id,
            operations={
                "task": {
                    "name": "Implement OAuth",
                    "start_date": "2024-01-15",
                    "duration_days": 3,
                    "assigned_to": ["developer"],
                    "phase": "implementation"
                }
            }
        )

        assert result["success"]
        assert result["items_created"][0]["type"] == "task"

        # Verify task details
        task_id = result["items_created"][0]["id"]
        item = store.get_item(task_id)
        # Task content includes the name plus testing guidance
        assert item["content"].startswith("Implement OAuth")
        assert "[Testing Checklist]" in item["content"]
        assert item["start_date"] == "2024-01-15"
        assert item["end_date"] == "2024-01-18"  # Calculated
        assert item["phase"] == "implementation"

    def test_create_milestone(self, handler, store, session_with_thought):
        """Test creating a milestone."""
        session_id, thought_id = session_with_thought

        result = handler.execute(
            session_id=session_id,
            thought_id=thought_id,
            operations={
                "milestone": {
                    "name": "Auth Complete",
                    "target_date": "2024-01-20",
                    "depends_on": []
                }
            }
        )

        assert result["success"]
        assert result["items_created"][0]["type"] == "milestone"
        assert "-MS-" in result["items_created"][0]["id"]

    def test_create_activity(self, handler, store, session_with_thought):
        """Test creating an activity."""
        session_id, thought_id = session_with_thought

        result = handler.execute(
            session_id=session_id,
            thought_id=thought_id,
            operations={
                "activity": {
                    "tool": "refactoring",
                    "params": "detector=god_class",
                    "result": "Found 3 issues",
                    "issues_found": 3
                }
            }
        )

        assert result["success"]
        assert result["items_created"][0]["type"] == "activity"
        assert "-ACT-" in result["items_created"][0]["id"]

    def test_create_decision(self, handler, store, session_with_thought):
        """Test creating a decision."""
        session_id, thought_id = session_with_thought

        result = handler.execute(
            session_id=session_id,
            thought_id=thought_id,
            operations={
                "decision": {
                    "text": "Use Auth0 for OAuth",
                    "rationale": "Best integration support"
                }
            }
        )

        assert result["success"]
        assert result["items_created"][0]["type"] == "decision"

    def test_create_element(self, handler, store, session_with_thought):
        """Test creating a design element."""
        session_id, thought_id = session_with_thought

        result = handler.execute(
            session_id=session_id,
            thought_id=thought_id,
            operations={
                "element": {
                    "id": "AuthProvider",
                    "type": "class",
                    "docref": "auth/provider.py"
                }
            }
        )

        assert result["success"]
        assert result["items_created"][0]["type"] == "element"


class TestTraceabilityOperations:
    """Test traceability relation operations."""

    def test_traces_relation(self, handler, store, session_with_thought):
        """Test creating traces relations."""
        session_id, thought_id = session_with_thought

        # Create a requirement first
        req_id = store.add_item(session_id, "requirement", "Test req", item_id="REQ-001")

        result = handler.execute(
            session_id=session_id,
            thought_id=thought_id,
            operations={
                "traces": ["REQ-001"]
            }
        )

        assert result["success"]
        assert len(result["relations_created"]) == 1
        assert result["relations_created"][0]["type"] == "traces"
        assert result["relations_created"][0]["target"] == "REQ-001"

    def test_multiple_traces(self, handler, store, session_with_thought):
        """Test creating multiple trace relations."""
        session_id, thought_id = session_with_thought

        store.add_item(session_id, "requirement", "Req 1", item_id="REQ-001")
        store.add_item(session_id, "requirement", "Req 2", item_id="REQ-002")

        result = handler.execute(
            session_id=session_id,
            thought_id=thought_id,
            operations={
                "traces": ["REQ-001", "REQ-002"]
            }
        )

        assert result["success"]
        assert len(result["relations_created"]) == 2

    def test_satisfies_relation(self, handler, store, session_with_thought):
        """Test creating satisfies relation."""
        session_id, thought_id = session_with_thought

        store.add_item(session_id, "requirement", "Req", item_id="REQ-001")

        result = handler.execute(
            session_id=session_id,
            thought_id=thought_id,
            operations={
                "satisfies": ["REQ-001"]
            }
        )

        assert result["success"]
        assert result["relations_created"][0]["type"] == "satisfies"

    def test_depends_on_relation(self, handler, store, session_with_thought):
        """Test creating depends_on relation."""
        session_id, thought_id = session_with_thought

        store.add_item(session_id, "task", "Task 1", item_id="TASK-001")

        result = handler.execute(
            session_id=session_id,
            thought_id=thought_id,
            operations={
                "depends_on": ["TASK-001"]
            }
        )

        assert result["success"]
        assert result["relations_created"][0]["type"] == "depends_on"

    def test_affects_files_relation(self, handler, store, session_with_thought):
        """Test creating affects relations for files."""
        session_id, thought_id = session_with_thought

        result = handler.execute(
            session_id=session_id,
            thought_id=thought_id,
            operations={
                "affects": ["auth/module.py", "auth/utils.py"]
            }
        )

        assert result["success"]
        assert len(result["relations_created"]) == 2
        for rel in result["relations_created"]:
            assert rel["type"] == "affects"


class TestUpdateOperations:
    """Test item update operations."""

    def test_update_item(self, handler, store, session_with_thought):
        """Test updating any item."""
        session_id, thought_id = session_with_thought

        req_id = store.add_item(
            session_id, "requirement", "Test req",
            item_id="REQ-001", status="pending"
        )

        result = handler.execute(
            session_id=session_id,
            thought_id=thought_id,
            operations={
                "update_item": {
                    "id": "REQ-001",
                    "status": "verified",
                    "priority": "high"
                }
            }
        )

        assert result["success"]
        assert len(result["items_updated"]) == 1
        assert "status" in result["items_updated"][0]["updated_fields"]

        # Verify update
        item = store.get_item("REQ-001")
        assert item["status"] == "verified"
        assert item["priority"] == "high"

    def test_update_task(self, handler, store, session_with_thought):
        """Test updating task-specific fields."""
        session_id, thought_id = session_with_thought

        store.add_item(
            session_id, "task", "Test task",
            item_id="TASK-001", status="pending", progress=0
        )

        result = handler.execute(
            session_id=session_id,
            thought_id=thought_id,
            operations={
                "update_task": {
                    "id": "TASK-001",
                    "status": "in_progress",
                    "progress": 50,
                    "actual_start": "2024-01-15"
                }
            }
        )

        assert result["success"]

        item = store.get_item("TASK-001")
        assert item["status"] == "in_progress"
        assert item["progress"] == 50
        assert item["actual_start"] == "2024-01-15"

    def test_complete_task(self, handler, store, session_with_thought):
        """Test completing a task."""
        session_id, thought_id = session_with_thought

        store.add_item(
            session_id, "task", "Test task",
            item_id="TASK-001", status="in_progress", progress=80
        )

        result = handler.execute(
            session_id=session_id,
            thought_id=thought_id,
            operations={
                "complete_task": "TASK-001"
            }
        )

        assert result["success"]

        item = store.get_item("TASK-001")
        assert item["status"] == "completed"
        assert item["progress"] == 100
        assert item["actual_end"] is not None


class TestMultipleOperations:
    """Test executing multiple operations together."""

    def test_combined_operations(self, handler, store, session_with_thought):
        """Test creating multiple items and relations in one call."""
        session_id, thought_id = session_with_thought

        result = handler.execute(
            session_id=session_id,
            thought_id=thought_id,
            operations={
                # Create items
                "requirement": {
                    "text": "System shall authenticate users",
                    "id": "REQ-AUTH"
                },
                "recommendation": {
                    "text": "Implement OAuth2 flow",
                    "priority": "high"
                },
                "task": {
                    "name": "Create auth module",
                    "id": "TASK-AUTH",
                    "start_date": "2024-01-15",
                    "duration_days": 5
                },
                # Create relations
                "affects": ["auth/module.py"],
                # Decision
                "decision": {
                    "text": "Use JWT tokens",
                    "rationale": "Stateless authentication"
                }
            }
        )

        assert result["success"]
        assert len(result["items_created"]) == 4  # req, rec, task, decision
        assert len(result["relations_created"]) == 1  # affects

        # Verify all items exist
        assert store.get_item("REQ-AUTH") is not None
        assert store.get_item("TASK-AUTH") is not None


class TestRETEROperationsSkipped:
    """Test RETER operations are handled gracefully without engine."""

    def test_reter_ops_without_engine(self, handler, store, session_with_thought):
        """Test RETER operations fail gracefully without engine."""
        session_id, thought_id = session_with_thought

        result = handler.execute(
            session_id=session_id,
            thought_id=thought_id,
            operations={
                "assert": "Individual(x, SomeClass)",
                "query": "FIND ?x WHERE Class(?x, Module)"
            }
        )

        # Operations fail but don't crash
        assert "reter" in result
        assert result["reter"]["assert"]["success"] is False
        assert result["reter"]["query"]["success"] is False
        assert "RETER engine not available" in result["reter"]["assert"]["error"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
