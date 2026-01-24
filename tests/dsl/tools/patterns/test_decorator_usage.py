"""
Unit tests for find_decorator_usage CADSL tool.

Verifies detection of decorator usages in the codebase.
"""

import pytest
from unittest.mock import Mock
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'src'))

from reter_code.dsl.tools.patterns.decorator_usage import find_decorator_usage
from reter_code.dsl.core import Pipeline, Context


class TestDecoratorUsageStructure:
    """Test find_decorator_usage tool structure."""

    def test_has_cadsl_spec(self):
        """Tool should have CADSL spec attached."""
        assert hasattr(find_decorator_usage, '_cadsl_spec')

    def test_spec_has_correct_name(self):
        """Tool spec should have correct name."""
        spec = find_decorator_usage._cadsl_spec
        assert spec.name == "find_decorator_usage"

    def test_has_decorator_name_param(self):
        """Should have decorator_name parameter."""
        spec = find_decorator_usage._cadsl_spec
        assert 'decorator_name' in spec.params

    def test_has_limit_param(self):
        """Should have limit parameter."""
        spec = find_decorator_usage._cadsl_spec
        assert 'limit' in spec.params

    def test_has_category_metadata(self):
        """Should have patterns category."""
        spec = find_decorator_usage._cadsl_spec
        assert spec.meta.get('category') == 'patterns'

    def test_has_severity_metadata(self):
        """Should have info severity."""
        spec = find_decorator_usage._cadsl_spec
        assert spec.meta.get('severity') == 'info'


class TestDecoratorUsagePipeline:
    """Test find_decorator_usage pipeline structure."""

    @pytest.fixture
    def pipeline(self):
        """Get the pipeline."""
        spec = find_decorator_usage._cadsl_spec
        ctx = Context(reter=None, params={"decorator_name": None, "limit": 100}, language="oo", instance_name="default")
        return spec.pipeline_factory(ctx)

    def test_pipeline_is_pipeline_object(self, pipeline):
        """Should return a Pipeline."""
        assert isinstance(pipeline, Pipeline)

    def test_has_emit_key(self, pipeline):
        """Should emit 'findings'."""
        assert pipeline._emit_key == "findings"

    def test_reql_has_decorator(self, pipeline):
        """REQL should query hasDecorator."""
        query = pipeline._source.query
        assert "hasDecorator" in query


class TestDecoratorUsageExecution:
    """Test execution with mock context."""

    @pytest.fixture
    def mock_context(self):
        """Create mock context."""
        ctx = Context(
            reter=Mock(),
            params={"decorator_name": None, "limit": 100},
            language="oo",
            instance_name="default"
        )

        mock_table = Mock()
        mock_table.num_rows = 1
        mock_table.to_pylist.return_value = [
            {"?decorator_name": "staticmethod", "?target_name": "helper", "?target_type": "method", "?file": "utils.py", "?line": 15}
        ]
        ctx.reter.reql.return_value = mock_table

        return ctx

    def test_execute_returns_success(self, mock_context):
        """Execution should return success."""
        result = find_decorator_usage(ctx=mock_context)
        assert "success" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
