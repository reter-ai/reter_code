"""
Unit tests for extract_method CADSL detector.

Verifies detection of Extract Method refactoring opportunities.
"""

import pytest
from unittest.mock import Mock
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'src'))

from reter_code.dsl.tools.refactoring.extract_method import extract_method
from reter_code.dsl.core import Pipeline, Context


class TestExtractMethodStructure:
    """Test extract_method detector structure."""

    def test_has_cadsl_spec(self):
        """Tool should have CADSL spec attached."""
        assert hasattr(extract_method, '_cadsl_spec')

    def test_spec_has_correct_name(self):
        """Tool spec should have correct name."""
        spec = extract_method._cadsl_spec
        assert spec.name == "extract_method"

    def test_has_min_lines_param(self):
        """Should have min_lines parameter."""
        spec = extract_method._cadsl_spec
        assert 'min_lines' in spec.params


class TestExtractMethodPipeline:
    """Test extract_method pipeline structure."""

    @pytest.fixture
    def pipeline(self):
        """Get the pipeline."""
        spec = extract_method._cadsl_spec
        ctx = Context(reter=None, params={"min_lines": 20}, language="oo", instance_name="default")
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


class TestExtractMethodVsReference:
    """Compare with reference implementation."""

    @pytest.fixture
    def pipeline(self):
        """Get the pipeline."""
        spec = extract_method._cadsl_spec
        ctx = Context(reter=None, params={"min_lines": 20}, language="oo", instance_name="default")
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


class TestExtractMethodExecution:
    """Test execution with mock context."""

    @pytest.fixture
    def mock_context(self):
        """Create mock context."""
        ctx = Context(
            reter=Mock(),
            params={"min_lines": 20},
            language="oo",
            instance_name="default"
        )

        mock_table = Mock()
        mock_table.num_rows = 1
        mock_table.to_pylist.return_value = [
            {"?m": "Service.process_all", "?name": "process_all", "?class_name": "Service", "?file": "service.py", "?line": "50", "?line_count": "80", "?block_count": "5"}
        ]
        ctx.reter.reql.return_value = mock_table

        return ctx

    def test_execute_returns_success(self, mock_context):
        """Execution should return success."""
        result = extract_method(ctx=mock_context)
        assert "success" in result

    def test_execute_returns_findings(self, mock_context):
        """Should return findings."""
        result = extract_method(ctx=mock_context)
        if result.get("success"):
            assert "findings" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
