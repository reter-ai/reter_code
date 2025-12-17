"""
Tests for UnifiedStore

Uses shared fixtures from conftest.py:
- store: Temporary UnifiedStore instance
"""

import pytest


class TestSessionManagement:
    """Test session CRUD operations."""

    def test_create_session(self, store):
        """Test creating a new session."""
        session_id = store.get_or_create_session("test-instance", goal="Test goal")

        assert session_id is not None
        session = store.get_session(session_id)
        assert session["instance_name"] == "test-instance"
        assert session["goal"] == "Test goal"
        assert session["status"] == "active"

    def test_get_existing_session(self, store):
        """Test getting existing active session."""
        session_id1 = store.get_or_create_session("test-instance")
        session_id2 = store.get_or_create_session("test-instance")

        assert session_id1 == session_id2

    def test_end_session(self, store):
        """Test ending a session."""
        session_id = store.get_or_create_session("test-instance")
        store.end_session(session_id)

        session = store.get_session(session_id)
        assert session["status"] == "completed"

    def test_clear_session(self, store):
        """Test clearing a session."""
        session_id = store.get_or_create_session("test-instance")
        store.add_item(session_id, "thought", "Test thought")

        result = store.clear_session(session_id)

        assert result["items_deleted"] == 1
        assert store.get_session(session_id) is None


class TestItemManagement:
    """Test item CRUD operations."""

    def test_add_item(self, store):
        """Test adding an item."""
        session_id = store.get_or_create_session("test")
        item_id = store.add_item(session_id, "thought", "Test thought content")

        # ID format: XXXX-THOUGHT-NNN (session prefix + type prefix + number)
        assert "-THOUGHT-" in item_id
        item = store.get_item(item_id)
        assert item["content"] == "Test thought content"
        assert item["item_type"] == "thought"

    def test_add_item_with_custom_id(self, store):
        """Test adding item with custom ID."""
        session_id = store.get_or_create_session("test")
        item_id = store.add_item(session_id, "requirement", "Test req", item_id="REQ-CUSTOM")

        assert item_id == "REQ-CUSTOM"

    def test_add_various_item_types(self, store):
        """Test adding different item types."""
        session_id = store.get_or_create_session("test")

        types_and_prefixes = [
            ("thought", "THOUGHT"),
            ("requirement", "REQ"),
            ("recommendation", "REC"),
            ("task", "TASK"),
            ("milestone", "MS"),
            ("activity", "ACT"),
            ("decision", "DEC"),
        ]

        for item_type, prefix in types_and_prefixes:
            item_id = store.add_item(session_id, item_type, f"Test {item_type}")
            # ID format: XXXX-PREFIX-NNN (session prefix + type prefix + number)
            assert f"-{prefix}-" in item_id

    def test_get_items_with_filter(self, store):
        """Test getting items with filters."""
        session_id = store.get_or_create_session("test")
        store.add_item(session_id, "task", "Task 1", status="pending", priority="high")
        store.add_item(session_id, "task", "Task 2", status="completed", priority="low")
        store.add_item(session_id, "recommendation", "Rec 1", status="pending")

        # Filter by type
        tasks = store.get_items(session_id, item_type="task")
        assert len(tasks) == 2

        # Filter by status
        pending = store.get_items(session_id, status="pending")
        assert len(pending) == 2

        # Filter by type and status
        pending_tasks = store.get_items(session_id, item_type="task", status="pending")
        assert len(pending_tasks) == 1

    def test_update_item(self, store):
        """Test updating an item."""
        session_id = store.get_or_create_session("test")
        item_id = store.add_item(session_id, "task", "Test task", status="pending")

        store.update_item(item_id, status="completed", progress=100)

        item = store.get_item(item_id)
        assert item["status"] == "completed"
        assert item["progress"] == 100

    def test_delete_item(self, store):
        """Test deleting an item."""
        session_id = store.get_or_create_session("test")
        item_id = store.add_item(session_id, "task", "Test task")

        store.delete_item(item_id)

        assert store.get_item(item_id) is None


class TestRelationManagement:
    """Test relation CRUD operations."""

    def test_add_relation(self, store):
        """Test adding a relation."""
        session_id = store.get_or_create_session("test")
        req_id = store.add_item(session_id, "requirement", "Test requirement")
        task_id = store.add_item(session_id, "task", "Test task")

        relation_id = store.add_relation(task_id, req_id, "item", "traces")

        assert relation_id > 0

    def test_get_outgoing_relations(self, store):
        """Test getting outgoing relations."""
        session_id = store.get_or_create_session("test")
        req_id = store.add_item(session_id, "requirement", "Test requirement")
        task_id = store.add_item(session_id, "task", "Test task")

        store.add_relation(task_id, req_id, "item", "traces")
        store.add_relation(task_id, "auth/service.py", "file", "affects")

        relations = store.get_relations(task_id, direction="outgoing")
        assert len(relations) == 2

    def test_get_items_by_relation(self, store):
        """Test getting items by relation."""
        session_id = store.get_or_create_session("test")
        req_id = store.add_item(session_id, "requirement", "Test requirement")
        task1_id = store.add_item(session_id, "task", "Task 1")
        task2_id = store.add_item(session_id, "task", "Task 2")

        store.add_relation(task1_id, req_id, "item", "traces")
        store.add_relation(task2_id, req_id, "item", "traces")

        tasks = store.get_items_by_relation(req_id, "traces")
        assert len(tasks) == 2


class TestIDGeneration:
    """Test ID generation."""

    def test_auto_increment_ids(self, store):
        """Test auto-incrementing IDs."""
        session_id = store.get_or_create_session("test")
        session_prefix = session_id[:4].upper()

        id1 = store.generate_id(session_id, "REQ")
        id2 = store.generate_id(session_id, "REQ")
        id3 = store.generate_id(session_id, "TASK")

        # IDs now include session prefix for global uniqueness
        assert id1 == f"{session_prefix}-REQ-001"
        assert id2 == f"{session_prefix}-REQ-002"
        assert id3 == f"{session_prefix}-TASK-001"


class TestThoughtChain:
    """Test thought chain operations."""

    def test_get_thought_chain(self, store):
        """Test getting thought chain."""
        session_id = store.get_or_create_session("test")

        store.add_item(session_id, "thought", "Thought 1", thought_number=1, total_thoughts=3)
        store.add_item(session_id, "thought", "Thought 2", thought_number=2, total_thoughts=3)
        store.add_item(session_id, "thought", "Thought 3", thought_number=3, total_thoughts=3)

        chain = store.get_thought_chain(session_id)
        assert len(chain) == 3
        assert chain[0]["thought_number"] == 1
        assert chain[2]["thought_number"] == 3


class TestSessionSummary:
    """Test session summary."""

    def test_get_session_summary(self, store):
        """Test getting session summary."""
        session_id = store.get_or_create_session("test")

        store.add_item(session_id, "thought", "T1", thought_number=1, status="completed")
        store.add_item(session_id, "thought", "T2", thought_number=2, status="completed")
        store.add_item(session_id, "task", "Task", status="pending")
        store.add_item(session_id, "recommendation", "Rec", status="pending")

        summary = store.get_session_summary(session_id)

        assert summary["by_type"]["thought"] == 2
        assert summary["by_type"]["task"] == 1
        assert summary["by_status"]["completed"] == 2
        assert summary["by_status"]["pending"] == 2
        assert summary["thought_chain"]["max_number"] == 2


class TestGetSessionByInstance:
    """Test get_session_by_instance."""

    def test_get_session_by_instance_exists(self, store):
        """Test getting active session by instance name."""
        session_id = store.get_or_create_session("test-instance", goal="Test")

        found = store.get_session_by_instance("test-instance")

        assert found is not None
        assert found["session_id"] == session_id
        assert found["instance_name"] == "test-instance"

    def test_get_session_by_instance_not_found(self, store):
        """Test getting session for non-existent instance."""
        found = store.get_session_by_instance("nonexistent-instance")

        assert found is None

    def test_get_session_by_instance_after_end(self, store):
        """Test getting session after ending it (should return None)."""
        session_id = store.get_or_create_session("test-instance")
        store.end_session(session_id)

        found = store.get_session_by_instance("test-instance")

        # Should return None because session is no longer active
        assert found is None


class TestUpdateSession:
    """Test update_session."""

    def test_update_session_goal(self, store):
        """Test updating session goal."""
        session_id = store.get_or_create_session("test")

        result = store.update_session(session_id, goal="Updated goal")

        assert result is True
        session = store.get_session(session_id)
        assert session["goal"] == "Updated goal"

    def test_update_session_multiple_fields(self, store):
        """Test updating multiple session fields."""
        session_id = store.get_or_create_session("test")

        store.update_session(
            session_id,
            goal="New goal",
            project_start="2024-01-01",
            project_end="2024-12-31"
        )

        session = store.get_session(session_id)
        assert session["goal"] == "New goal"
        assert session["project_start"] == "2024-01-01"
        assert session["project_end"] == "2024-12-31"

    def test_update_session_nonexistent(self, store):
        """Test updating non-existent session."""
        result = store.update_session("nonexistent-id", goal="Test")

        # Should handle gracefully - returns True even if no rows affected
        # (SQL UPDATE on non-existent row succeeds without error)
        assert result is True


class TestDeleteRelation:
    """Test delete_relation."""

    def test_delete_relation(self, store):
        """Test deleting a relation."""
        session_id = store.get_or_create_session("test")
        item1_id = store.add_item(session_id, "task", "Task 1")
        item2_id = store.add_item(session_id, "task", "Task 2")

        relation_id = store.add_relation(item1_id, item2_id, "item", "depends_on")

        result = store.delete_relation(relation_id)

        assert result is True
        # Verify relation is gone
        relations = store.get_relations(item1_id, direction="outgoing")
        assert len(relations) == 0

    def test_delete_relation_nonexistent(self, store):
        """Test deleting non-existent relation."""
        result = store.delete_relation(99999)

        # Should handle gracefully - returns True even if no rows affected
        # (SQL DELETE on non-existent row succeeds without error)
        assert result is True


class TestGetItemsPagination:
    """Test get_items pagination."""

    def test_get_items_with_limit(self, store):
        """Test getting items with limit."""
        session_id = store.get_or_create_session("test")

        for i in range(10):
            store.add_item(session_id, "task", f"Task {i}")

        items = store.get_items(session_id, item_type="task", limit=5)

        assert len(items) == 5

    def test_get_items_with_offset(self, store):
        """Test getting items with offset."""
        session_id = store.get_or_create_session("test")

        for i in range(10):
            store.add_item(session_id, "task", f"Task {i}")

        items_page1 = store.get_items(session_id, item_type="task", limit=5, offset=0)
        items_page2 = store.get_items(session_id, item_type="task", limit=5, offset=5)

        assert len(items_page1) == 5
        assert len(items_page2) == 5
        # Pages should have different items
        page1_ids = {i["item_id"] for i in items_page1}
        page2_ids = {i["item_id"] for i in items_page2}
        assert page1_ids.isdisjoint(page2_ids)


class TestGetRelationsDirection:
    """Test get_relations with different directions."""

    def test_get_relations_incoming(self, store):
        """Test getting incoming relations."""
        session_id = store.get_or_create_session("test")
        source_id = store.add_item(session_id, "task", "Source task")
        target_id = store.add_item(session_id, "requirement", "Target requirement")

        store.add_relation(source_id, target_id, "item", "traces")

        incoming = store.get_relations(target_id, direction="incoming")

        assert len(incoming) == 1
        assert incoming[0]["source_id"] == source_id

    def test_get_relations_both(self, store):
        """Test getting both incoming and outgoing relations."""
        session_id = store.get_or_create_session("test")
        item1 = store.add_item(session_id, "task", "Task 1")
        item2 = store.add_item(session_id, "task", "Task 2")
        item3 = store.add_item(session_id, "task", "Task 3")

        # item1 -> item2, item3 -> item1
        store.add_relation(item1, item2, "item", "depends_on")
        store.add_relation(item3, item1, "item", "depends_on")

        both = store.get_relations(item1, direction="both")

        assert len(both) == 2

    def test_get_relations_with_type_filter(self, store):
        """Test getting relations filtered by type."""
        session_id = store.get_or_create_session("test")
        task_id = store.add_item(session_id, "task", "Task")
        req_id = store.add_item(session_id, "requirement", "Requirement")

        store.add_relation(task_id, req_id, "item", "traces")
        store.add_relation(task_id, "file.py", "file", "affects")

        traces = store.get_relations(task_id, direction="outgoing", relation_type="traces")

        assert len(traces) == 1
        assert traces[0]["relation_type"] == "traces"


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_get_nonexistent_item(self, store):
        """Test getting non-existent item returns None."""
        result = store.get_item("nonexistent-id")

        assert result is None

    def test_update_nonexistent_item(self, store):
        """Test updating non-existent item."""
        result = store.update_item("nonexistent-id", status="completed")

        # SQL UPDATE on non-existent row succeeds without error
        assert result is True

    def test_delete_nonexistent_item(self, store):
        """Test deleting non-existent item."""
        result = store.delete_item("nonexistent-id")

        # delete_item returns {"relations_deleted": N}
        # Even if item doesn't exist, it returns successfully
        assert "relations_deleted" in result
        assert result["relations_deleted"] == 0

    def test_get_items_empty_session(self, store):
        """Test getting items from empty session."""
        session_id = store.get_or_create_session("test")

        items = store.get_items(session_id)

        assert items == []

    def test_get_thought_chain_empty(self, store):
        """Test getting thought chain from session with no thoughts."""
        session_id = store.get_or_create_session("test")

        chain = store.get_thought_chain(session_id)

        assert chain == []

    def test_get_session_summary_empty(self, store):
        """Test getting summary from empty session."""
        session_id = store.get_or_create_session("test")

        summary = store.get_session_summary(session_id)

        assert summary["total_items"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
