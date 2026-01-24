"""
Unit tests for complex_untested CADSL detector.

Verifies detection of complex (long) code without tests.
Uses lineCount as a proxy for complexity.
"""

import pytest
from unittest.mock import Mock
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'src'))

from reter_code.dsl.tools.testing.complex_untested import complex_untested
from reter_code.dsl.core import Pipeline, Context


class TestComplexUntestedStructure:
    """Test complex_untested detector structure."""

    def test_has_cadsl_spec(self):
        """Tool should have CADSL spec attached."""
        assert hasattr(complex_untested, '_cadsl_spec')

    def test_spec_has_correct_name(self):
        """Tool spec should have correct name."""
        spec = complex_untested._cadsl_spec
        assert spec.name == "complex_untested"

    def test_has_min_lines_param(self):
        """Should have min_lines parameter (proxy for complexity)."""
        spec = complex_untested._cadsl_spec
        assert 'min_lines' in spec.params


class TestComplexUntestedPipeline:
    """Test complex_untested pipeline structure."""

    @pytest.fixture
    def pipeline(self):
        """Get the pipeline."""
        spec = complex_untested._cadsl_spec
        ctx = Context(reter=None, params={"min_lines": 30}, language="oo", instance_name="default")
        return spec.pipeline_factory(ctx)

    def test_pipeline_is_pipeline_object(self, pipeline):
        """Should return a Pipeline."""
        assert isinstance(pipeline, Pipeline)

    def test_has_emit_key(self, pipeline):
        """Should emit 'findings'."""
        assert pipeline._emit_key == "findings"

    def test_reql_uses_line_count(self, pipeline):
        """REQL should use lineCount as complexity proxy."""
        query = pipeline._source.query
        assert "lineCount" in query or "line_count" in query


class TestComplexUntestedVsReference:
    """Compare with reference implementation."""

    @pytest.fixture
    def pipeline(self):
        """Get the pipeline."""
        spec = complex_untested._cadsl_spec
        ctx = Context(reter=None, params={"min_lines": 30}, language="oo", instance_name="default")
        return spec.pipeline_factory(ctx)

    def test_filters_by_line_count(self, pipeline):
        """Should filter by line count threshold."""
        query = pipeline._source.query
        has_filter = "FILTER" in query
        assert has_filter, "Should filter by line count"

    def test_excludes_test_files(self, pipeline):
        """Should exclude test files."""
        query = pipeline._source.query
        assert "test" in query.lower(), "Should filter out test files"


class TestComplexUntestedExecution:
    """Test execution with mock context."""

    @pytest.fixture
    def mock_context(self):
        """Create mock context."""
        ctx = Context(
            reter=Mock(),
            params={"min_lines": 30},
            language="oo",
            instance_name="default"
        )

        mock_table = Mock()
        mock_table.num_rows = 1
        mock_table.to_pylist.return_value = [
            {"?e": "Parser.parse_complex", "?name": "parse_complex", "?class_name": "Parser", "?file": "parser.py", "?line": "50", "?line_count": "75"}
        ]
        ctx.reter.reql.return_value = mock_table

        return ctx

    def test_execute_returns_success(self, mock_context):
        """Execution should return success."""
        result = complex_untested(ctx=mock_context)
        assert "success" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
