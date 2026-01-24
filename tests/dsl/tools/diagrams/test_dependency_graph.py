"""
Unit tests for dependency_graph CADSL tool.

Verifies generation of module dependency graph with circular dependency detection.
"""

import pytest
from unittest.mock import Mock
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'src'))

from reter_code.dsl.tools.diagrams.dependency_graph import dependency_graph
from reter_code.dsl.core import Pipeline, Context


class TestDependencyGraphStructure:
    """Test dependency_graph tool structure."""

    def test_has_cadsl_spec(self):
        """Tool should have CADSL spec attached."""
        assert hasattr(dependency_graph, '_cadsl_spec')

    def test_spec_has_correct_name(self):
        """Tool spec should have correct name."""
        spec = dependency_graph._cadsl_spec
        assert spec.name == "dependency_graph"

    def test_has_show_external_param(self):
        """Should have show_external parameter."""
        spec = dependency_graph._cadsl_spec
        assert 'show_external' in spec.params

    def test_has_module_filter_param(self):
        """Should have module_filter parameter."""
        spec = dependency_graph._cadsl_spec
        assert 'module_filter' in spec.params

    def test_has_highlight_circular_param(self):
        """Should have highlight_circular parameter."""
        spec = dependency_graph._cadsl_spec
        assert 'highlight_circular' in spec.params

    def test_has_format_param(self):
        """Should have format parameter."""
        spec = dependency_graph._cadsl_spec
        assert 'format' in spec.params

    def test_has_limit_param(self):
        """Should have limit parameter."""
        spec = dependency_graph._cadsl_spec
        assert 'limit' in spec.params


class TestDependencyGraphPipeline:
    """Test dependency_graph pipeline structure."""

    @pytest.fixture
    def pipeline(self):
        """Get the pipeline."""
        spec = dependency_graph._cadsl_spec
        ctx = Context(reter=None, params={"show_external": False, "module_filter": None, "highlight_circular": True, "format": "mermaid", "limit": 100}, language="oo", instance_name="default")
        return spec.pipeline_factory(ctx)

    def test_pipeline_is_pipeline_object(self, pipeline):
        """Should return a Pipeline."""
        assert isinstance(pipeline, Pipeline)

    def test_has_emit_key(self, pipeline):
        """Should emit 'diagram'."""
        assert pipeline._emit_key == "diagram"

    def test_reql_has_module_type(self, pipeline):
        """REQL should query Module type."""
        query = pipeline._source.query
        assert "Module" in query

    def test_reql_has_import_type(self, pipeline):
        """REQL should query Import type."""
        query = pipeline._source.query
        assert "Import" in query

    def test_reql_has_imports_relationship(self, pipeline):
        """REQL should query imports relationship."""
        query = pipeline._source.query
        assert "imports" in query


class TestDependencyGraphExecution:
    """Test execution with mock context."""

    @pytest.fixture
    def mock_context(self):
        """Create mock context."""
        ctx = Context(
            reter=Mock(),
            params={"show_external": False, "module_filter": None, "highlight_circular": True, "format": "mermaid", "limit": 100},
            language="oo",
            instance_name="default"
        )

        mock_table = Mock()
        mock_table.num_rows = 2
        mock_table.to_pylist.return_value = [
            {"?moduleName": "myapp.service", "?imported": "myapp.utils"},
            {"?moduleName": "myapp.main", "?imported": "myapp.service"}
        ]
        ctx.reter.reql.return_value = mock_table

        return ctx

    def test_execute_returns_success(self, mock_context):
        """Execution should return success."""
        result = dependency_graph(ctx=mock_context)
        assert "success" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
