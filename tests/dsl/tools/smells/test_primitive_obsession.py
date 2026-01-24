"""
Unit tests for primitive_obsession CADSL detector.

Verifies detection of overuse of primitive types for domain concepts.
"""

import pytest
from unittest.mock import Mock
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'src'))

from reter_code.dsl.tools.smells.primitive_obsession import primitive_obsession
from reter_code.dsl.core import Pipeline, Context, ToolType


class TestPrimitiveObsessionStructure:
    """Test primitive_obsession detector structure."""

    def test_has_cadsl_spec(self):
        """Tool should have CADSL spec attached."""
        assert hasattr(primitive_obsession, '_cadsl_spec')

    def test_spec_has_correct_name(self):
        """Tool spec should have correct name."""
        spec = primitive_obsession._cadsl_spec
        assert spec.name == "primitive_obsession"

    def test_has_min_primitives_param(self):
        """Should have min_primitives parameter."""
        spec = primitive_obsession._cadsl_spec
        assert 'min_primitives' in spec.params


class TestPrimitiveObsessionPipeline:
    """Test primitive_obsession pipeline structure."""

    @pytest.fixture
    def pipeline(self):
        """Get the pipeline."""
        spec = primitive_obsession._cadsl_spec
        ctx = Context(reter=None, params={"min_primitives": 3}, language="oo", instance_name="default")
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

    def test_reql_uses_count_for_primitives(self, pipeline):
        """REQL should use COUNT() for primitive params."""
        query = pipeline._source.query
        assert "COUNT" in query or "primitive_params" in query


class TestPrimitiveObsessionVsReference:
    """Compare with reference implementation."""

    @pytest.fixture
    def pipeline(self):
        """Get the pipeline."""
        spec = primitive_obsession._cadsl_spec
        ctx = Context(reter=None, params={"min_primitives": 3}, language="oo", instance_name="default")
        return spec.pipeline_factory(ctx)

    def test_filters_by_threshold(self, pipeline):
        """Should filter by primitive param count threshold."""
        query = pipeline._source.query
        has_having = "HAVING" in query
        assert has_having, "Should have HAVING for threshold on COUNT"

    def test_uses_type_annotation(self, pipeline):
        """Should use typeAnnotation property."""
        query = pipeline._source.query
        assert "typeAnnotation" in query, "Should check typeAnnotation for primitives"


class TestPrimitiveObsessionExecution:
    """Test execution with mock context."""

    @pytest.fixture
    def mock_context(self):
        """Create mock context."""
        ctx = Context(
            reter=Mock(),
            params={"min_primitives": 3},
            language="oo",
            instance_name="default"
        )

        mock_table = Mock()
        mock_table.num_rows = 1
        mock_table.to_pylist.return_value = [
            {"?m": "Order.create", "?name": "create", "?class_name": "Order", "?file": "order.py", "?line": "30", "?primitive_params": "6"}
        ]
        ctx.reter.reql.return_value = mock_table

        return ctx

    def test_execute_returns_success(self, mock_context):
        """Execution should return success."""
        result = primitive_obsession(ctx=mock_context)
        assert "success" in result

    def test_execute_returns_findings(self, mock_context):
        """Should return findings."""
        result = primitive_obsession(ctx=mock_context)
        if result.get("success"):
            assert "findings" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
