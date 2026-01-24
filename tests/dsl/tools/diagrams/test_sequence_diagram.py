"""
Unit tests for sequence_diagram CADSL tool.

Verifies generation of Mermaid sequence diagrams.
"""

import pytest
from unittest.mock import Mock
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'src'))

from reter_code.dsl.tools.diagrams.sequence_diagram import sequence_diagram
from reter_code.dsl.core import Pipeline, Context


class TestSequenceDiagramStructure:
    """Test sequence_diagram tool structure."""

    def test_has_cadsl_spec(self):
        """Tool should have CADSL spec attached."""
        assert hasattr(sequence_diagram, '_cadsl_spec')

    def test_spec_has_correct_name(self):
        """Tool spec should have correct name."""
        spec = sequence_diagram._cadsl_spec
        assert spec.name == "sequence_diagram"

    def test_has_classes_param(self):
        """Should have classes parameter."""
        spec = sequence_diagram._cadsl_spec
        assert 'classes' in spec.params

    def test_has_entry_point_param(self):
        """Should have entry_point parameter."""
        spec = sequence_diagram._cadsl_spec
        assert 'entry_point' in spec.params


class TestSequenceDiagramPipeline:
    """Test sequence_diagram pipeline structure."""

    @pytest.fixture
    def pipeline(self):
        """Get the pipeline."""
        spec = sequence_diagram._cadsl_spec
        ctx = Context(reter=None, params={"classes": ["A", "B"], "entry_point": None, "max_depth": 10, "format": "mermaid"}, language="oo", instance_name="default")
        return spec.pipeline_factory(ctx)

    def test_pipeline_is_pipeline_object(self, pipeline):
        """Should return a Pipeline."""
        assert isinstance(pipeline, Pipeline)

    def test_has_emit_key(self, pipeline):
        """Should emit 'diagram'."""
        assert pipeline._emit_key == "diagram"

    def test_reql_has_method_type(self, pipeline):
        """REQL should query Method type."""
        query = pipeline._source.query
        assert "Method" in query

    def test_reql_has_calls(self, pipeline):
        """REQL should query calls relationship."""
        query = pipeline._source.query
        assert "calls" in query


class TestSequenceDiagramExecution:
    """Test execution with mock context."""

    @pytest.fixture
    def mock_context(self):
        """Create mock context."""
        ctx = Context(
            reter=Mock(),
            params={"classes": ["ServiceA", "ServiceB"], "entry_point": None, "max_depth": 10, "format": "mermaid"},
            language="oo",
            instance_name="default"
        )

        mock_table = Mock()
        mock_table.num_rows = 1
        mock_table.to_pylist.return_value = [
            {"?caller": "m1", "?callerName": "doWork", "?callee": "m2", "?calleeName": "process", "?callerClassName": "ServiceA", "?calleeClassName": "ServiceB"}
        ]
        ctx.reter.reql.return_value = mock_table

        return ctx

    def test_execute_returns_success(self, mock_context):
        """Execution should return success."""
        result = sequence_diagram(ctx=mock_context)
        assert "success" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
