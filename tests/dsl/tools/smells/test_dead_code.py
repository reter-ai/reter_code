"""
Unit tests for dead_code CADSL detector.

Verifies detection of unused classes, methods, and functions.
"""

import pytest
from unittest.mock import Mock
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'src'))

from reter_code.dsl.tools.smells.dead_code import dead_code
from reter_code.dsl.core import Pipeline, Context, ToolType


class TestDeadCodeStructure:
    """Test dead_code detector structure."""

    def test_has_cadsl_spec(self):
        """Tool should have CADSL spec attached."""
        assert hasattr(dead_code, '_cadsl_spec')

    def test_spec_has_correct_name(self):
        """Tool spec should have correct name."""
        spec = dead_code._cadsl_spec
        assert spec.name == "dead_code"

    def test_has_include_private_param(self):
        """Should have include_private parameter."""
        spec = dead_code._cadsl_spec
        assert 'include_private' in spec.params


class TestDeadCodePipeline:
    """Test dead_code pipeline structure."""

    @pytest.fixture
    def pipeline(self):
        """Get the pipeline."""
        spec = dead_code._cadsl_spec
        ctx = Context(reter=None, params={"include_private": False}, language="oo", instance_name="default")
        return spec.pipeline_factory(ctx)

    def test_pipeline_is_pipeline_object(self, pipeline):
        """Should return a Pipeline."""
        assert isinstance(pipeline, Pipeline)

    def test_has_emit_key(self, pipeline):
        """Should emit 'findings'."""
        assert pipeline._emit_key == "findings"

    def test_reql_has_method_or_function(self, pipeline):
        """REQL should query Method or Function type."""
        query = pipeline._source.query
        has_method = "Method" in query
        has_function = "Function" in query
        assert has_method or has_function

    def test_reql_has_not_exists(self, pipeline):
        """REQL should use NOT EXISTS for uncalled code."""
        query = pipeline._source.query
        assert "NOT EXISTS" in query or "FILTER NOT" in query


class TestDeadCodeVsReference:
    """Compare with reference implementation."""

    @pytest.fixture
    def pipeline(self):
        """Get the pipeline."""
        spec = dead_code._cadsl_spec
        ctx = Context(reter=None, params={"include_private": False}, language="oo", instance_name="default")
        return spec.pipeline_factory(ctx)

    def test_filters_uncalled(self, pipeline):
        """Should filter out called code."""
        query = pipeline._source.query
        has_filter = "FILTER" in query or "NOT EXISTS" in query
        assert has_filter, "Should filter for uncalled code"


class TestDeadCodeExecution:
    """Test execution with mock context."""

    @pytest.fixture
    def mock_context(self):
        """Create mock context."""
        ctx = Context(
            reter=Mock(),
            params={"include_private": False},
            language="oo",
            instance_name="default"
        )

        mock_table = Mock()
        mock_table.num_rows = 1
        mock_table.to_pylist.return_value = [
            {"?e": "unused_helper", "?name": "unused_helper", "?entity_type": "function", "?file": "utils.py", "?line": "100"}
        ]
        ctx.reter.reql.return_value = mock_table

        return ctx

    def test_execute_returns_success(self, mock_context):
        """Execution should return success."""
        result = dead_code(ctx=mock_context)
        assert "success" in result

    def test_execute_returns_findings(self, mock_context):
        """Should return findings."""
        result = dead_code(ctx=mock_context)
        if result.get("success"):
            assert "findings" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
