"""
Unit tests for class_diagram CADSL tool.

Verifies generation of Mermaid class diagrams.
"""

import pytest
from unittest.mock import Mock
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'src'))

from reter_code.dsl.tools.diagrams.class_diagram import class_diagram
from reter_code.dsl.core import Pipeline, Context


class TestClassDiagramStructure:
    """Test class_diagram tool structure."""

    def test_has_cadsl_spec(self):
        """Tool should have CADSL spec attached."""
        assert hasattr(class_diagram, '_cadsl_spec')

    def test_spec_has_correct_name(self):
        """Tool spec should have correct name."""
        spec = class_diagram._cadsl_spec
        assert spec.name == "class_diagram"

    def test_has_classes_param(self):
        """Should have classes parameter."""
        spec = class_diagram._cadsl_spec
        assert 'classes' in spec.params

    def test_has_include_methods_param(self):
        """Should have include_methods parameter."""
        spec = class_diagram._cadsl_spec
        assert 'include_methods' in spec.params

    def test_has_include_attributes_param(self):
        """Should have include_attributes parameter."""
        spec = class_diagram._cadsl_spec
        assert 'include_attributes' in spec.params


class TestClassDiagramPipeline:
    """Test class_diagram pipeline structure."""

    @pytest.fixture
    def pipeline(self):
        """Get the pipeline."""
        spec = class_diagram._cadsl_spec
        ctx = Context(reter=None, params={"classes": ["MyClass"], "include_methods": True, "include_attributes": True, "format": "mermaid"}, language="oo", instance_name="default")
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


class TestClassDiagramExecution:
    """Test execution with mock context."""

    @pytest.fixture
    def mock_context(self):
        """Create mock context."""
        ctx = Context(
            reter=Mock(),
            params={"classes": ["MyClass"], "include_methods": True, "include_attributes": True, "format": "mermaid"},
            language="oo",
            instance_name="default"
        )

        mock_table = Mock()
        mock_table.num_rows = 1
        mock_table.to_pylist.return_value = [
            {"?c": "MyClass", "?className": "MyClass", "?base": None, "?baseName": None, "?m": "process", "?methodName": "process", "?a": "data", "?attrName": "data"}
        ]
        ctx.reter.reql.return_value = mock_table

        return ctx

    def test_execute_returns_success(self, mock_context):
        """Execution should return success."""
        result = class_diagram(ctx=mock_context)
        assert "success" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
