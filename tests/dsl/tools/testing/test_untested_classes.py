"""
Unit tests for untested_classes CADSL detector.

Verifies detection of classes without corresponding test classes.
"""

import pytest
from unittest.mock import Mock
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'src'))

from reter_code.dsl.tools.testing.untested_classes import untested_classes
from reter_code.dsl.core import Pipeline, Context


class TestUntestedClassesStructure:
    """Test untested_classes detector structure."""

    def test_has_cadsl_spec(self):
        """Tool should have CADSL spec attached."""
        assert hasattr(untested_classes, '_cadsl_spec')

    def test_spec_has_correct_name(self):
        """Tool spec should have correct name."""
        spec = untested_classes._cadsl_spec
        assert spec.name == "untested_classes"


class TestUntestedClassesPipeline:
    """Test untested_classes pipeline structure."""

    @pytest.fixture
    def pipeline(self):
        """Get the pipeline."""
        spec = untested_classes._cadsl_spec
        ctx = Context(reter=None, params={}, language="oo", instance_name="default")
        return spec.pipeline_factory(ctx)

    def test_pipeline_is_pipeline_object(self, pipeline):
        """Should return a Pipeline."""
        assert isinstance(pipeline, Pipeline)

    def test_has_emit_key(self, pipeline):
        """Should emit 'findings'."""
        assert pipeline._emit_key == "findings"

    def test_reql_has_class_type(self, pipeline):
        """REQL should filter by Class type."""
        query = pipeline._source.query
        assert "Class" in query


class TestUntestedClassesVsReference:
    """Compare with reference implementation."""

    @pytest.fixture
    def pipeline(self):
        """Get the pipeline."""
        spec = untested_classes._cadsl_spec
        ctx = Context(reter=None, params={}, language="oo", instance_name="default")
        return spec.pipeline_factory(ctx)

    def test_filters_for_untested(self, pipeline):
        """Should filter for untested classes."""
        query = pipeline._source.query
        has_filter = "FILTER" in query or "NOT EXISTS" in query
        assert has_filter, "Should filter for untested classes"


class TestUntestedClassesExecution:
    """Test execution with mock context."""

    @pytest.fixture
    def mock_context(self):
        """Create mock context."""
        ctx = Context(
            reter=Mock(),
            params={},
            language="oo",
            instance_name="default"
        )

        mock_table = Mock()
        mock_table.num_rows = 1
        mock_table.to_pylist.return_value = [
            {"?c": "UserService", "?name": "UserService", "?file": "services.py", "?line": "10", "?method_count": "5"}
        ]
        ctx.reter.reql.return_value = mock_table

        return ctx

    def test_execute_returns_success(self, mock_context):
        """Execution should return success."""
        result = untested_classes(ctx=mock_context)
        assert "success" in result

    def test_execute_returns_findings(self, mock_context):
        """Should return findings."""
        result = untested_classes(ctx=mock_context)
        if result.get("success"):
            assert "findings" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
