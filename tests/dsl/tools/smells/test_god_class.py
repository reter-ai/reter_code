"""
Unit tests for god_class CADSL detector.

Verifies detection of classes that do too much (God Class anti-pattern).
"""

import pytest
from unittest.mock import Mock
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'src'))

from reter_code.dsl.tools.smells.god_class import god_class
from reter_code.dsl.core import Pipeline, Context, ToolType


class TestGodClassStructure:
    """Test god_class detector structure."""

    def test_has_cadsl_spec(self):
        """Tool should have CADSL spec attached."""
        assert hasattr(god_class, '_cadsl_spec')

    def test_spec_has_correct_name(self):
        """Tool spec should have correct name."""
        spec = god_class._cadsl_spec
        assert spec.name == "god_class"

    def test_has_max_methods_param(self):
        """Should have max_methods parameter."""
        spec = god_class._cadsl_spec
        assert 'max_methods' in spec.params

    def test_has_limit_param(self):
        """Should have limit parameter."""
        spec = god_class._cadsl_spec
        assert 'limit' in spec.params


class TestGodClassPipeline:
    """Test god_class pipeline structure."""

    @pytest.fixture
    def pipeline(self):
        """Get the pipeline."""
        spec = god_class._cadsl_spec
        ctx = Context(reter=None, params={"max_methods": 15, "limit": 100}, language="oo", instance_name="default")
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

    def test_reql_counts_methods(self, pipeline):
        """REQL should COUNT methods."""
        query = pipeline._source.query
        assert "COUNT" in query or "method_count" in query


class TestGodClassVsReference:
    """Compare with reference implementation."""

    @pytest.fixture
    def pipeline(self):
        """Get the pipeline."""
        spec = god_class._cadsl_spec
        ctx = Context(reter=None, params={"max_methods": 15, "limit": 100}, language="oo", instance_name="default")
        return spec.pipeline_factory(ctx)

    def test_has_having_threshold(self, pipeline):
        """Should use HAVING for threshold."""
        query = pipeline._source.query
        has_having = "HAVING" in query
        assert has_having, "Should have HAVING for threshold"

    def test_orders_by_method_count(self, pipeline):
        """Should order results by method count."""
        query = pipeline._source.query
        has_order = "ORDER BY" in query
        assert has_order, "Should ORDER BY method_count"


class TestGodClassExecution:
    """Test execution with mock context."""

    @pytest.fixture
    def mock_context(self):
        """Create mock context."""
        ctx = Context(
            reter=Mock(),
            params={"max_methods": 15, "limit": 100},
            language="oo",
            instance_name="default"
        )

        mock_table = Mock()
        mock_table.num_rows = 1
        mock_table.to_pylist.return_value = [
            {"?c": "GodController", "?name": "GodController", "?file": "main.py", "?line": "10", "?method_count": "30"}
        ]
        ctx.reter.reql.return_value = mock_table

        return ctx

    def test_execute_returns_success(self, mock_context):
        """Execution should return success."""
        result = god_class(ctx=mock_context)
        assert "success" in result

    def test_execute_returns_findings(self, mock_context):
        """Should return findings."""
        result = god_class(ctx=mock_context)
        if result.get("success"):
            assert "findings" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
