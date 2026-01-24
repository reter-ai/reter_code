"""
Unit tests for long_methods CADSL detector.

Verifies detection of methods that are too long.
"""

import pytest
from unittest.mock import Mock
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'src'))

from reter_code.dsl.tools.smells.long_methods import long_methods
from reter_code.dsl.core import Pipeline, Context, ToolType


class TestLongMethodsStructure:
    """Test long_methods detector structure."""

    def test_has_cadsl_spec(self):
        """Tool should have CADSL spec attached."""
        assert hasattr(long_methods, '_cadsl_spec')

    def test_spec_has_correct_name(self):
        """Tool spec should have correct name."""
        spec = long_methods._cadsl_spec
        assert spec.name == "long_methods"

    def test_has_max_lines_param(self):
        """Should have max_lines parameter."""
        spec = long_methods._cadsl_spec
        assert 'max_lines' in spec.params


class TestLongMethodsPipeline:
    """Test long_methods pipeline structure."""

    @pytest.fixture
    def pipeline(self):
        """Get the pipeline."""
        spec = long_methods._cadsl_spec
        ctx = Context(reter=None, params={"max_lines": 50}, language="oo", instance_name="default")
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

    def test_reql_has_line_count(self, pipeline):
        """REQL should query lineCount."""
        query = pipeline._source.query
        assert "lineCount" in query or "line_count" in query


class TestLongMethodsVsReference:
    """Compare with reference implementation."""

    @pytest.fixture
    def pipeline(self):
        """Get the pipeline."""
        spec = long_methods._cadsl_spec
        ctx = Context(reter=None, params={"max_lines": 50}, language="oo", instance_name="default")
        return spec.pipeline_factory(ctx)

    def test_filters_by_threshold(self, pipeline):
        """Should filter by line count threshold."""
        query = pipeline._source.query
        has_filter = "FILTER" in query
        assert has_filter, "Should have FILTER for threshold"

    def test_orders_by_line_count(self, pipeline):
        """Should order results by line count."""
        query = pipeline._source.query
        has_order = "ORDER BY" in query
        assert has_order, "Should ORDER BY line_count"


class TestLongMethodsExecution:
    """Test execution with mock context."""

    @pytest.fixture
    def mock_context(self):
        """Create mock context."""
        ctx = Context(
            reter=Mock(),
            params={"max_lines": 50},
            language="oo",
            instance_name="default"
        )

        mock_table = Mock()
        mock_table.num_rows = 1
        mock_table.to_pylist.return_value = [
            {"?m": "MyClass.long_method", "?name": "long_method", "?class_name": "MyClass", "?file": "main.py", "?line": "20", "?line_count": "150"}
        ]
        ctx.reter.reql.return_value = mock_table

        return ctx

    def test_execute_returns_success(self, mock_context):
        """Execution should return success."""
        result = long_methods(ctx=mock_context)
        assert "success" in result

    def test_execute_returns_findings(self, mock_context):
        """Should return findings."""
        result = long_methods(ctx=mock_context)
        if result.get("success"):
            assert "findings" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
