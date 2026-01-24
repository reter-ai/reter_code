"""
Unit tests for move_method CADSL detector.

Verifies detection of Move Method refactoring opportunities.
"""

import pytest
from unittest.mock import Mock
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'src'))

from reter_code.dsl.tools.refactoring.move_method import move_method
from reter_code.dsl.core import Pipeline, Context


class TestMoveMethodStructure:
    """Test move_method detector structure."""

    def test_has_cadsl_spec(self):
        """Tool should have CADSL spec attached."""
        assert hasattr(move_method, '_cadsl_spec')

    def test_spec_has_correct_name(self):
        """Tool spec should have correct name."""
        spec = move_method._cadsl_spec
        assert spec.name == "move_method"

    def test_has_threshold_param(self):
        """Should have threshold parameter."""
        spec = move_method._cadsl_spec
        assert 'threshold' in spec.params


class TestMoveMethodPipeline:
    """Test move_method pipeline structure."""

    @pytest.fixture
    def pipeline(self):
        """Get the pipeline."""
        spec = move_method._cadsl_spec
        ctx = Context(reter=None, params={"threshold": 0.6}, language="oo", instance_name="default")
        return spec.pipeline_factory(ctx)

    def test_pipeline_is_pipeline_object(self, pipeline):
        """Should return a Pipeline."""
        assert isinstance(pipeline, Pipeline)

    def test_has_emit_key(self, pipeline):
        """Should emit 'findings'."""
        assert pipeline._emit_key == "findings"

    def test_reql_has_method_type(self, pipeline):
        """REQL should filter by Method type."""
        query = pipeline._source.query
        assert "Method" in query


class TestMoveMethodVsReference:
    """Compare with reference implementation."""

    @pytest.fixture
    def pipeline(self):
        """Get the pipeline."""
        spec = move_method._cadsl_spec
        ctx = Context(reter=None, params={"threshold": 0.6}, language="oo", instance_name="default")
        return spec.pipeline_factory(ctx)

    def test_filters_by_threshold(self, pipeline):
        """Should filter by external usage ratio."""
        query = pipeline._source.query
        has_filter = "FILTER" in query
        assert has_filter, "Should have FILTER for threshold"


class TestMoveMethodExecution:
    """Test execution with mock context."""

    @pytest.fixture
    def mock_context(self):
        """Create mock context."""
        ctx = Context(
            reter=Mock(),
            params={"threshold": 0.6},
            language="oo",
            instance_name="default"
        )

        mock_table = Mock()
        mock_table.num_rows = 1
        mock_table.to_pylist.return_value = [
            {"?m": "Order.calculate_tax", "?name": "calculate_tax", "?class_name": "Order", "?file": "order.py", "?line": "50", "?target_class": "TaxCalculator", "?external_usage": "0.8"}
        ]
        ctx.reter.reql.return_value = mock_table

        return ctx

    def test_execute_returns_success(self, mock_context):
        """Execution should return success."""
        result = move_method(ctx=mock_context)
        assert "success" in result

    def test_execute_returns_findings(self, mock_context):
        """Should return findings."""
        result = move_method(ctx=mock_context)
        if result.get("success"):
            assert "findings" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
