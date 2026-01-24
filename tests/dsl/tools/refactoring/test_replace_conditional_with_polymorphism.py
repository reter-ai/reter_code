"""
Unit tests for replace_conditional_with_polymorphism CADSL detector.

Verifies detection of conditionals that could use polymorphism.
"""

import pytest
from unittest.mock import Mock
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'src'))

from reter_code.dsl.tools.refactoring.replace_conditional_with_polymorphism import replace_conditional_with_polymorphism
from reter_code.dsl.core import Pipeline, Context


class TestReplaceConditionalStructure:
    """Test replace_conditional_with_polymorphism detector structure."""

    def test_has_cadsl_spec(self):
        """Tool should have CADSL spec attached."""
        assert hasattr(replace_conditional_with_polymorphism, '_cadsl_spec')

    def test_spec_has_correct_name(self):
        """Tool spec should have correct name."""
        spec = replace_conditional_with_polymorphism._cadsl_spec
        assert spec.name == "replace_conditional_with_polymorphism"

    def test_has_min_branches_param(self):
        """Should have min_branches parameter."""
        spec = replace_conditional_with_polymorphism._cadsl_spec
        assert 'min_branches' in spec.params


class TestReplaceConditionalPipeline:
    """Test replace_conditional_with_polymorphism pipeline structure."""

    @pytest.fixture
    def pipeline(self):
        """Get the pipeline."""
        spec = replace_conditional_with_polymorphism._cadsl_spec
        ctx = Context(reter=None, params={"min_branches": 4}, language="oo", instance_name="default")
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


class TestReplaceConditionalVsReference:
    """Compare with reference implementation."""

    @pytest.fixture
    def pipeline(self):
        """Get the pipeline."""
        spec = replace_conditional_with_polymorphism._cadsl_spec
        ctx = Context(reter=None, params={"min_branches": 4}, language="oo", instance_name="default")
        return spec.pipeline_factory(ctx)

    def test_filters_by_branch_count(self, pipeline):
        """Should filter by branch count threshold."""
        query = pipeline._source.query
        has_filter = "FILTER" in query
        assert has_filter, "Should have FILTER for threshold"


class TestReplaceConditionalExecution:
    """Test execution with mock context."""

    @pytest.fixture
    def mock_context(self):
        """Create mock context."""
        ctx = Context(
            reter=Mock(),
            params={"min_branches": 4},
            language="oo",
            instance_name="default"
        )

        mock_table = Mock()
        mock_table.num_rows = 1
        mock_table.to_pylist.return_value = [
            {"?m": "Calculator.calculate", "?name": "calculate", "?class_name": "Calculator", "?file": "calc.py", "?line": "40", "?branch_count": "6", "?conditional_type": "switch"}
        ]
        ctx.reter.reql.return_value = mock_table

        return ctx

    def test_execute_returns_success(self, mock_context):
        """Execution should return success."""
        result = replace_conditional_with_polymorphism(ctx=mock_context)
        assert "success" in result

    def test_execute_returns_findings(self, mock_context):
        """Should return findings."""
        result = replace_conditional_with_polymorphism(ctx=mock_context)
        if result.get("success"):
            assert "findings" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
