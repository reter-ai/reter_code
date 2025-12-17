"""
Tests for ThinkingSession

Uses shared fixtures from conftest.py:
- store: Temporary UnifiedStore instance
- session: ThinkingSession instance
"""

import pytest


class TestSessionLifecycle:
    """Test session lifecycle operations."""

    def test_start_session(self, session):
        """Test starting a new session."""
        result = session.start_session(
            instance_name="test",
            goal="Test goal",
            project_start="2024-01-15",
            project_end="2024-02-15"
        )

        assert result["success"]
        assert result["session_id"] is not None
        assert result["goal"] == "Test goal"
        assert result["project_start"] == "2024-01-15"

    def test_start_existing_session(self, session):
        """Test getting existing session."""
        result1 = session.start_session("test", goal="Goal 1")
        result2 = session.start_session("test", goal="Goal 2")

        # Should return same session
        assert result1["session_id"] == result2["session_id"]

    def test_end_session(self, session):
        """Test ending a session."""
        session.start_session("test")
        result = session.end_session("test")

        assert result["success"]
        assert result["status"] == "completed"
        assert "summary" in result

    def test_clear_session(self, session):
        """Test clearing a session."""
        session.start_session("test")
        session.think("test", "Thought 1", 1, 3)
        session.think("test", "Thought 2", 2, 3)

        result = session.clear_session("test")

        assert result["success"]
        assert result["items_deleted"] >= 2

    def test_get_context(self, session):
        """Test getting full context."""
        session.start_session("test", goal="Test context")
        session.think("test", "First thought", 1, 3)

        result = session.get_context("test")

        assert result["success"]
        assert "session" in result
        assert "thoughts" in result
        assert "requirements" in result
        assert "recommendations" in result
        assert "project" in result
        assert "suggestions" in result
        assert "mcp_guide" in result

    def test_context_has_mcp_guide(self, session):
        """Test that context includes MCP guide."""
        session.start_session("test")
        result = session.get_context("test")

        assert "tools" in result["mcp_guide"]
        assert "recommended_workflow" in result["mcp_guide"]
        assert "thinking" in result["mcp_guide"]["tools"]


class TestThinking:
    """Test thinking functionality."""

    def test_basic_think(self, session):
        """Test creating a basic thought."""
        session.start_session("test")

        result = session.think(
            instance_name="test",
            thought="Analyzing the problem",
            thought_number=1,
            total_thoughts=5,
            thought_type="analysis"
        )

        assert result["success"]
        assert result["thought_number"] == 1
        assert result["total_thoughts"] == 5
        assert result["thought_type"] == "analysis"

    def test_think_with_operations(self, session):
        """Test thinking with operations."""
        session.start_session("test")

        result = session.think(
            instance_name="test",
            thought="Creating requirements",
            thought_number=1,
            total_thoughts=3,
            operations={
                "requirement": {
                    "text": "System shall authenticate users",
                    "risk": "medium"
                },
                "task": {
                    "name": "Implement auth",
                    "start_date": "2024-01-15",
                    "duration_days": 5
                }
            }
        )

        assert result["success"]
        assert len(result["items_created"]) == 2

        # Verify types
        types = [item["type"] for item in result["items_created"]]
        assert "requirement" in types
        assert "task" in types

    def test_think_with_traceability(self, session, store):
        """Test thinking with traceability operations."""
        sess_result = session.start_session("test")
        session_id = sess_result["session_id"]

        # Create a requirement first
        store.add_item(session_id, "requirement", "Test req", item_id="REQ-001")

        result = session.think(
            instance_name="test",
            thought="Implementing requirement",
            thought_number=1,
            total_thoughts=3,
            operations={
                "traces": ["REQ-001"],
                "affects": ["auth/module.py"]
            }
        )

        assert result["success"]
        assert len(result["relations_created"]) == 2

    def test_thought_chain(self, session):
        """Test creating a chain of thoughts."""
        session.start_session("test")

        for i in range(1, 4):
            result = session.think(
                instance_name="test",
                thought=f"Thought {i}",
                thought_number=i,
                total_thoughts=3,
                next_thought_needed=(i < 3)
            )
            assert result["success"]
            assert result["thought_number"] == i

        # Check context shows thought chain
        context = session.get_context("test")
        assert context["thoughts"]["total"] == 3
        assert context["thoughts"]["max_number"] == 3

    def test_branching_thought(self, session):
        """Test creating a branching thought."""
        session.start_session("test")
        session.think("test", "Main thought", 1, 5)

        result = session.think(
            instance_name="test",
            thought="Branch thought",
            thought_number=2,
            total_thoughts=5,
            branch_id="alternative",
            branch_from=1
        )

        assert result["success"]
        assert result["thought_number"] == 2


class TestProjectAnalytics:
    """Test project analytics functionality."""

    def test_project_health_empty(self, session):
        """Test project health with no tasks."""
        session.start_session("test")
        result = session.get_project_health("test")

        assert result["success"]
        assert result["tasks"]["total"] == 0
        assert result["tasks"]["percent_complete"] == 0

    def test_project_health_with_tasks(self, session, store):
        """Test project health with tasks."""
        sess_result = session.start_session("test", project_end="2024-12-31")
        session_id = sess_result["session_id"]

        # Add tasks directly
        store.add_item(session_id, "task", "Task 1", status="completed")
        store.add_item(session_id, "task", "Task 2", status="in_progress")
        store.add_item(session_id, "task", "Task 3", status="pending")

        result = session.get_project_health("test")

        assert result["tasks"]["total"] == 3
        assert result["tasks"]["completed"] == 1
        assert result["tasks"]["in_progress"] == 1
        assert result["tasks"]["pending"] == 1
        assert result["tasks"]["percent_complete"] == pytest.approx(33.3, rel=0.1)

    def test_critical_path_empty(self, session):
        """Test critical path with no tasks."""
        session.start_session("test")
        result = session.get_critical_path("test")

        assert result["success"]
        assert result["critical_tasks"] == []
        assert result["total_duration"] == 0

    def test_critical_path_with_dependencies(self, session, store):
        """Test critical path calculation with dependencies."""
        sess_result = session.start_session("test")
        session_id = sess_result["session_id"]

        # Create tasks with dependencies
        store.add_item(session_id, "task", "Task A", item_id="TASK-A", duration_days=3)
        store.add_item(session_id, "task", "Task B", item_id="TASK-B", duration_days=2)
        store.add_item(session_id, "task", "Task C", item_id="TASK-C", duration_days=4)

        # B depends on A
        store.add_relation("TASK-B", "TASK-A", "item", "depends_on")
        # C depends on B
        store.add_relation("TASK-C", "TASK-B", "item", "depends_on")

        result = session.get_critical_path("test")

        assert result["success"]
        assert result["total_duration"] == 9  # 3 + 2 + 4
        assert len(result["critical_tasks"]) == 3  # All tasks on critical path

    def test_overdue_tasks(self, session, store):
        """Test getting overdue tasks."""
        sess_result = session.start_session("test")
        session_id = sess_result["session_id"]

        # Add overdue task
        store.add_item(
            session_id, "task", "Overdue task",
            end_date="2020-01-01",  # Past date
            status="pending"
        )
        # Add future task
        store.add_item(
            session_id, "task", "Future task",
            end_date="2030-01-01",
            status="pending"
        )

        result = session.get_overdue_tasks("test")

        assert result["success"]
        assert result["total_overdue"] == 1
        assert result["overdue_tasks"][0]["name"] == "Overdue task"

    def test_impact_analysis(self, session, store):
        """Test delay impact analysis."""
        sess_result = session.start_session("test")
        session_id = sess_result["session_id"]

        # Create task chain
        store.add_item(
            session_id, "task", "Task A",
            item_id="TASK-A", end_date="2024-01-20"
        )
        store.add_item(
            session_id, "task", "Task B",
            item_id="TASK-B", end_date="2024-01-25"
        )
        store.add_relation("TASK-B", "TASK-A", "item", "depends_on")

        result = session.analyze_impact("test", "TASK-A", 5)

        assert result["success"]
        assert result["delayed_task"]["id"] == "TASK-A"
        assert result["delayed_task"]["delay_days"] == 5
        assert len(result["affected_tasks"]) == 1
        assert result["affected_tasks"][0]["id"] == "TASK-B"


class TestContextGeneration:
    """Test context generation for session restoration."""

    def test_context_includes_all_sections(self, session):
        """Test that context has all required sections."""
        session.start_session("test", goal="Complete test")

        context = session.get_context("test")

        required_sections = [
            "session", "thoughts", "requirements", "recommendations",
            "project", "artifacts", "recent_activities", "reter",
            "suggestions", "mcp_guide"
        ]
        for section in required_sections:
            assert section in context, f"Missing section: {section}"

    def test_context_with_data(self, session, store):
        """Test context with actual data."""
        sess_result = session.start_session("test", goal="Rich context")
        session_id = sess_result["session_id"]

        # Add various items
        session.think("test", "First thought", 1, 3)
        store.add_item(session_id, "requirement", "Req 1", item_id="REQ-001")
        store.add_item(session_id, "recommendation", "Rec 1", status="pending", priority="high")
        store.add_item(session_id, "task", "Task 1", status="in_progress")
        store.add_item(session_id, "decision", "Decision 1")

        context = session.get_context("test")

        assert context["thoughts"]["total"] >= 1
        assert context["requirements"]["total"] >= 1
        assert context["recommendations"]["total"] >= 1
        assert context["project"]["total_tasks"] >= 1

    def test_suggestions_generation(self, session, store):
        """Test that suggestions are generated based on state."""
        sess_result = session.start_session("test")
        session_id = sess_result["session_id"]

        # Add pending recommendation
        store.add_item(session_id, "recommendation", "Do something", status="pending")

        # Add blocked task
        store.add_item(session_id, "task", "Blocked task", status="blocked")

        context = session.get_context("test")

        # Should have suggestions
        assert len(context["suggestions"]) > 0
        assert any("pending" in s.lower() for s in context["suggestions"])
        assert any("blocked" in s.lower() for s in context["suggestions"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
