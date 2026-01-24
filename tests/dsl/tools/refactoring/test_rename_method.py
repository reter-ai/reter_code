"""
Unit tests for rename_method CADSL detector.

Verifies detection of Rename Method refactoring opportunities.
"""

import pytest
from unittest.mock import Mock
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'src'))

from reter_code.dsl.tools.refactoring.rename_method import rename_method
from reter_code.dsl.core import Pipeline, Context


class TestRenameMethodStructure:
    """Test rename_method detector structure."""

    def test_has_cadsl_spec(self):
        """Tool should have CADSL spec attached."""
        assert hasattr(rename_method, '_cadsl_spec')

    def test_spec_has_correct_name(self):
        """Tool spec should have correct name."""
        spec = rename_method._cadsl_spec
        assert spec.name == "rename_method"

    def test_has_min_length_param(self):
        """Should have min_length parameter."""
        spec = rename_method._cadsl_spec
        assert 'min_length' in spec.params

    def test_has_max_length_param(self):
        """Should have max_length parameter."""
        spec = rename_method._cadsl_spec
        assert 'max_length' in spec.params


class TestRenameMethodPipeline:
    """Test rename_method pipeline structure."""

    @pytest.fixture
    def pipeline(self):
        """Get the pipeline."""
        spec = rename_method._cadsl_spec
        ctx = Context(reter=None, params={"min_length": 1, "max_length": 50}, language="oo", instance_name="default")
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


class TestRenameMethodVsReference:
    """Compare with reference implementation."""

    @pytest.fixture
    def pipeline(self):
        """Get the pipeline."""
        spec = rename_method._cadsl_spec
        ctx = Context(reter=None, params={"min_length": 1, "max_length": 50}, language="oo", instance_name="default")
        return spec.pipeline_factory(ctx)

    def test_filters_by_name_criteria(self, pipeline):
        """Should filter by name length or pattern."""
        query = pipeline._source.query
        has_filter = "FILTER" in query
        assert has_filter, "Should have FILTER for name criteria"


class TestRenameMethodExecution:
    """Test execution with mock context."""

    @pytest.fixture
    def mock_context(self):
        """Create mock context."""
        ctx = Context(
            reter=Mock(),
            params={"min_length": 1, "max_length": 50},
            language="oo",
            instance_name="default"
        )

        mock_table = Mock()
        mock_table.num_rows = 1
        mock_table.to_pylist.return_value = [
            {"?m": "Service.do", "?name": "do", "?class_name": "Service", "?file": "service.py", "?line": "20", "?name_length": "2"}
        ]
        ctx.reter.reql.return_value = mock_table

        return ctx

    def test_execute_returns_success(self, mock_context):
        """Execution should return success."""
        result = rename_method(ctx=mock_context)
        assert "success" in result

    def test_execute_returns_findings(self, mock_context):
        """Should return findings."""
        result = rename_method(ctx=mock_context)
        if result.get("success"):
            assert "findings" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
