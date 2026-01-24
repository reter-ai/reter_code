"""
Unit tests for class_hierarchy CADSL tool.

Verifies generation of class inheritance hierarchy diagrams.
"""

import pytest
from unittest.mock import Mock
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'src'))

from reter_code.dsl.tools.diagrams.class_hierarchy import class_hierarchy
from reter_code.dsl.core import Pipeline, Context


class TestClassHierarchyStructure:
    """Test class_hierarchy tool structure."""

    def test_has_cadsl_spec(self):
        """Tool should have CADSL spec attached."""
        assert hasattr(class_hierarchy, '_cadsl_spec')

    def test_spec_has_correct_name(self):
        """Tool spec should have correct name."""
        spec = class_hierarchy._cadsl_spec
        assert spec.name == "class_hierarchy"

    def test_has_root_class_param(self):
        """Should have root_class parameter."""
        spec = class_hierarchy._cadsl_spec
        assert 'root_class' in spec.params

    def test_has_format_param(self):
        """Should have format parameter."""
        spec = class_hierarchy._cadsl_spec
        assert 'format' in spec.params


class TestClassHierarchyPipeline:
    """Test class_hierarchy pipeline structure."""

    @pytest.fixture
    def pipeline(self):
        """Get the pipeline."""
        spec = class_hierarchy._cadsl_spec
        ctx = Context(reter=None, params={"root_class": None, "format": "markdown"}, language="oo", instance_name="default")
        return spec.pipeline_factory(ctx)

    def test_pipeline_is_pipeline_object(self, pipeline):
        """Should return a Pipeline."""
        assert isinstance(pipeline, Pipeline)

    def test_has_emit_key(self, pipeline):
        """Should emit 'diagram'."""
        assert pipeline._emit_key == "diagram"

    def test_reql_has_class_type(self, pipeline):
        """REQL should query Class type."""
        query = pipeline._source.query
        assert "Class" in query

    def test_reql_has_inheritance(self, pipeline):
        """REQL should query inheritsFrom."""
        query = pipeline._source.query
        assert "inheritsFrom" in query


class TestClassHierarchyExecution:
    """Test execution with mock context."""

    @pytest.fixture
    def mock_context(self):
        """Create mock context."""
        ctx = Context(
            reter=Mock(),
            params={"root_class": None, "format": "markdown"},
            language="oo",
            instance_name="default"
        )

        mock_table = Mock()
        mock_table.num_rows = 2
        mock_table.to_pylist.return_value = [
            {"?c": "Child", "?className": "Child", "?base": "Parent", "?baseName": "Parent"},
            {"?c": "Parent", "?className": "Parent", "?base": None, "?baseName": None}
        ]
        ctx.reter.reql.return_value = mock_table

        return ctx

    def test_execute_returns_success(self, mock_context):
        """Execution should return success."""
        result = class_hierarchy(ctx=mock_context)
        assert "success" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
