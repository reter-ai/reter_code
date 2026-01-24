"""
Unit tests for call_graph CADSL tool.

Verifies generation of function/method call graph visualizations.
"""

import pytest
from unittest.mock import Mock
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'src'))

from reter_code.dsl.tools.diagrams.call_graph import call_graph
from reter_code.dsl.core import Pipeline, Context


class TestCallGraphStructure:
    """Test call_graph tool structure."""

    def test_has_cadsl_spec(self):
        """Tool should have CADSL spec attached."""
        assert hasattr(call_graph, '_cadsl_spec')

    def test_spec_has_correct_name(self):
        """Tool spec should have correct name."""
        spec = call_graph._cadsl_spec
        assert spec.name == "call_graph"

    def test_has_focus_function_param(self):
        """Should have focus_function parameter."""
        spec = call_graph._cadsl_spec
        assert 'focus_function' in spec.params

    def test_focus_function_is_required(self):
        """focus_function should be required."""
        spec = call_graph._cadsl_spec
        param_spec = spec.params['focus_function']
        assert param_spec.required is True

    def test_has_direction_param(self):
        """Should have direction parameter."""
        spec = call_graph._cadsl_spec
        assert 'direction' in spec.params

    def test_has_max_depth_param(self):
        """Should have max_depth parameter."""
        spec = call_graph._cadsl_spec
        assert 'max_depth' in spec.params

    def test_has_format_param(self):
        """Should have format parameter."""
        spec = call_graph._cadsl_spec
        assert 'format' in spec.params


class TestCallGraphPipeline:
    """Test call_graph pipeline structure."""

    @pytest.fixture
    def pipeline(self):
        """Get the pipeline."""
        spec = call_graph._cadsl_spec
        ctx = Context(reter=None, params={"focus_function": "main", "direction": "both", "max_depth": 3, "format": "mermaid"}, language="oo", instance_name="default")
        return spec.pipeline_factory(ctx)

    def test_pipeline_is_pipeline_object(self, pipeline):
        """Should return a Pipeline."""
        assert isinstance(pipeline, Pipeline)

    def test_has_emit_key(self, pipeline):
        """Should emit 'diagram'."""
        assert pipeline._emit_key == "diagram"

    def test_reql_has_calls(self, pipeline):
        """REQL should query calls relationship."""
        query = pipeline._source.query
        assert "calls" in query

    def test_reql_has_name(self, pipeline):
        """REQL should query name."""
        query = pipeline._source.query
        assert "name" in query


class TestCallGraphExecution:
    """Test execution with mock context."""

    @pytest.fixture
    def mock_context(self):
        """Create mock context."""
        ctx = Context(
            reter=Mock(),
            params={"focus_function": "main", "direction": "both", "max_depth": 3, "format": "mermaid"},
            language="oo",
            instance_name="default"
        )

        mock_table = Mock()
        mock_table.num_rows = 2
        mock_table.to_pylist.return_value = [
            {"?callerName": "main", "?calleeName": "helper"},
            {"?callerName": "helper", "?calleeName": "process"}
        ]
        ctx.reter.reql.return_value = mock_table

        return ctx

    def test_execute_returns_success(self, mock_context):
        """Execution should return success."""
        result = call_graph(ctx=mock_context)
        assert "success" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
