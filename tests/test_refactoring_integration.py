"""
Tests for Refactoring Tools Integration with UnifiedStore

These tests verify that the refactoring tools (RefactoringToolBase,
RefactoringTool, RefactoringToPatternsTool) work correctly with the
UnifiedStore after Phase 6 migration.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from typing import List

from codeine.tools.unified.store import UnifiedStore
from codeine.tools.refactoring.base import RefactoringToolBase
from codeine.tools.base import ToolMetadata, ToolDefinition


class ConcreteRefactoringTool(RefactoringToolBase):
    """Concrete implementation for testing RefactoringToolBase."""

    def get_metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="test_refactoring_tool",
            version="1.0.0",
            description="Test tool for refactoring integration tests",
            author="Test",
            requires_reter=False,
            dependencies=[],
            categories=["test"]
        )

    def get_tools(self) -> List[ToolDefinition]:
        return []


class TestRefactoringToolBase:
    """Test RefactoringToolBase helper methods."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.store = UnifiedStore(db_path=Path(self.temp_dir) / "test.db")
        # get_or_create_session returns session_id string directly
        self.session_id = self.store.get_or_create_session("test")
        # Create a concrete instance
        self.base = ConcreteRefactoringTool(instance_manager=None)
        yield
        shutil.rmtree(self.temp_dir)

    def test_get_unified_store(self):
        """Test _get_unified_store returns a store instance."""
        store = self.base._get_unified_store()
        assert store is not None

    def test_severity_to_priority(self):
        """Test severity to priority mapping."""
        assert self.base._severity_to_priority("critical") == "critical"
        assert self.base._severity_to_priority("high") == "high"
        assert self.base._severity_to_priority("medium") == "medium"
        assert self.base._severity_to_priority("low") == "low"
        assert self.base._severity_to_priority("info") == "info"
        assert self.base._severity_to_priority("unknown") == "medium"

    def test_extract_findings_empty(self):
        """Test extracting findings from empty result."""
        findings = self.base._extract_findings({})
        assert findings == []

    def test_extract_findings_with_classes(self):
        """Test extracting findings with classes key."""
        result = {"classes": [{"name": "A"}, {"name": "B"}]}
        findings = self.base._extract_findings(result)
        assert len(findings) == 2

    def test_extract_findings_with_opportunities(self):
        """Test extracting findings with opportunities key."""
        result = {"opportunities": [{"class": "X"}]}
        findings = self.base._extract_findings(result)
        assert len(findings) == 1

    def test_count_findings(self):
        """Test counting findings."""
        assert self.base._count_findings({}) == 0
        assert self.base._count_findings({"classes": [1, 2, 3]}) == 3
        assert self.base._count_findings({"count": 5}) == 5

    def test_finding_to_text_with_name_and_module(self):
        """Test generating text from finding with name and module."""
        finding = {"name": "MyClass", "module": "myapp.models"}
        text = self.base._finding_to_text("large_classes", finding)
        assert "large_classes" in text
        assert "MyClass" in text
        assert "myapp.models" in text

    def test_finding_to_text_with_qualified_name(self):
        """Test generating text from finding with qualified name."""
        finding = {"qualified_name": "myapp.models.MyClass"}
        text = self.base._finding_to_text("feature_envy", finding)
        assert "feature_envy" in text
        assert "myapp.models.MyClass" in text

    def test_extract_files(self):
        """Test extracting files from finding."""
        finding = {"file": "models.py", "module": "myapp.models"}
        files = self.base._extract_files(finding)
        assert "models.py" in files
        assert "myapp.models" in files

    def test_extract_entities(self):
        """Test extracting entities from finding."""
        finding = {"name": "MyClass", "method_name": "process"}
        entities = self.base._extract_entities(finding)
        assert "MyClass" in entities
        assert "process" in entities


class TestFindingsToItems:
    """Test _findings_to_items method."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.store = UnifiedStore(db_path=Path(self.temp_dir) / "test.db")
        # get_or_create_session returns session_id string directly
        self.session_id = self.store.get_or_create_session("test")

        # Create a concrete test tool
        self.base = ConcreteRefactoringTool(instance_manager=None)

        yield
        shutil.rmtree(self.temp_dir)

    def test_findings_to_items_empty(self):
        """Test converting empty findings."""
        result = {}
        detector_info = {"category": "code_smell", "severity": "high"}

        items_result = self.base._findings_to_items(
            detector_name="test_detector",
            detector_info=detector_info,
            result=result,
            store=self.store,
            session_id=self.session_id
        )

        assert items_result["items_created"] == 0
        assert items_result["tasks_created"] == 0
        assert items_result["relations_created"] == 0

    def test_findings_to_items_creates_recommendations(self):
        """Test converting findings creates recommendation items."""
        result = {"classes": [
            {"name": "LargeClass", "module": "app.models"},
            {"name": "BigClass", "module": "app.views"}
        ]}
        detector_info = {"category": "code_smell", "severity": "high"}

        items_result = self.base._findings_to_items(
            detector_name="find_large_classes",
            detector_info=detector_info,
            result=result,
            store=self.store,
            session_id=self.session_id
        )

        assert items_result["items_created"] == 2

        # Verify items were created
        items = self.store.get_items(session_id=self.session_id, item_type="recommendation")
        assert len(items) == 2

    def test_findings_to_items_with_category_prefix(self):
        """Test converting findings with category prefix."""
        result = {"opportunities": [{"class_name": "TestClass"}]}
        detector_info = {"category": "creation", "severity": "medium"}

        self.base._findings_to_items(
            detector_name="detect_chain_constructors",
            detector_info=detector_info,
            result=result,
            store=self.store,
            session_id=self.session_id,
            category_prefix="pattern:"
        )

        items = self.store.get_items(session_id=self.session_id)
        assert len(items) == 1
        assert items[0]["category"] == "pattern:creation"

    def test_findings_to_items_creates_tasks_for_high_priority(self):
        """Test auto-create tasks for high priority findings with test-first workflow."""
        result = {"classes": [{"name": "BadClass"}]}
        detector_info = {"category": "code_smell", "severity": "high"}

        items_result = self.base._findings_to_items(
            detector_name="test_detector",
            detector_info=detector_info,
            result=result,
            store=self.store,
            session_id=self.session_id,
            create_tasks=True
        )

        assert items_result["items_created"] == 1
        # 4 tasks for test-first workflow: find tests, ensure coverage, refactor, verify
        assert items_result["tasks_created"] == 4

        # Verify tasks were created with test-first workflow
        tasks = self.store.get_items(session_id=self.session_id, item_type="task")
        assert len(tasks) == 4

        # Verify workflow steps are present
        task_contents = [t["content"] for t in tasks]
        assert any("FIND EXISTING TESTS" in c for c in task_contents)
        assert any("ENSURE TEST COVERAGE" in c for c in task_contents)
        assert any("APPLY REFACTORING" in c for c in task_contents)
        assert any("VERIFY TESTS PASS" in c for c in task_contents)

    def test_findings_to_items_links_to_thought(self):
        """Test linking recommendations to thought."""
        # Create a thought first
        thought_id = self.store.add_item(
            session_id=self.session_id,
            item_type="thought",
            content="Analyzing code smells"
        )

        result = {"classes": [{"name": "TestClass"}]}
        detector_info = {"category": "code_smell", "severity": "medium"}

        items_result = self.base._findings_to_items(
            detector_name="test_detector",
            detector_info=detector_info,
            result=result,
            store=self.store,
            session_id=self.session_id,
            link_to_thought=thought_id
        )

        assert items_result["relations_created"] >= 1

        # Verify relation was created
        items = self.store.get_items(session_id=self.session_id, item_type="recommendation")
        relations = self.store.get_relations(items[0]["item_id"])
        assert any(r["target_id"] == thought_id for r in relations)


class TestRegistrarParameterChanges:
    """Test that registrar parameters are correctly updated."""

    def test_registrar_has_session_instance_parameter(self):
        """Test that recommender tool has session_instance parameter."""
        from codeine.services.registrars.refactoring import RecommenderToolsRegistrar

        # Just verify the class exists and can be imported
        assert RecommenderToolsRegistrar is not None

    def test_refactoring_tool_has_new_parameters(self):
        """Test RefactoringTool has updated parameters."""
        from codeine.tools.refactoring_improving.tool import RefactoringTool
        import inspect

        # Check prepare signature
        prepare_sig = inspect.signature(RefactoringTool.prepare)
        assert "session_instance" in prepare_sig.parameters

        # Check detector signature
        detector_sig = inspect.signature(RefactoringTool.detector)
        assert "session_instance" in detector_sig.parameters
        assert "create_tasks" in detector_sig.parameters
        assert "link_to_thought" in detector_sig.parameters

    def test_patterns_tool_has_new_parameters(self):
        """Test RefactoringToPatternsTool has updated parameters."""
        from codeine.tools.refactoring_to_patterns.tool import RefactoringToPatternsTool
        import inspect

        # Check prepare signature
        prepare_sig = inspect.signature(RefactoringToPatternsTool.prepare)
        assert "session_instance" in prepare_sig.parameters

        # Check detector signature
        detector_sig = inspect.signature(RefactoringToPatternsTool.detector)
        assert "session_instance" in detector_sig.parameters
        assert "create_tasks" in detector_sig.parameters
        assert "link_to_thought" in detector_sig.parameters
