"""
Unit tests for find_factory_pattern CADSL tool.

Verifies detection of factory pattern implementations.
"""

import pytest
from unittest.mock import Mock
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'src'))

from reter_code.dsl.tools.patterns.factory import find_factory_pattern
from reter_code.dsl.core import Pipeline, Context


class TestFactoryPatternStructure:
    """Test find_factory_pattern tool structure."""

    def test_has_cadsl_spec(self):
        """Tool should have CADSL spec attached."""
        assert hasattr(find_factory_pattern, '_cadsl_spec')

    def test_spec_has_correct_name(self):
        """Tool spec should have correct name."""
        spec = find_factory_pattern._cadsl_spec
        assert spec.name == "find_factory_pattern"

    def test_has_limit_param(self):
        """Should have limit parameter."""
        spec = find_factory_pattern._cadsl_spec
        assert 'limit' in spec.params

    def test_has_category_metadata(self):
        """Should have patterns category."""
        spec = find_factory_pattern._cadsl_spec
        assert spec.meta.get('category') == 'patterns'

    def test_has_severity_metadata(self):
        """Should have info severity."""
        spec = find_factory_pattern._cadsl_spec
        assert spec.meta.get('severity') == 'info'


class TestFactoryPatternPipeline:
    """Test find_factory_pattern pipeline structure."""

    @pytest.fixture
    def pipeline(self):
        """Get the pipeline."""
        spec = find_factory_pattern._cadsl_spec
        ctx = Context(reter=None, params={"limit": 100}, language="oo", instance_name="default")
        return spec.pipeline_factory(ctx)

    def test_pipeline_is_pipeline_object(self, pipeline):
        """Should return a Pipeline."""
        assert isinstance(pipeline, Pipeline)

    def test_has_emit_key(self, pipeline):
        """Should emit 'findings'."""
        assert pipeline._emit_key == "findings"

    def test_reql_has_method_type(self, pipeline):
        """REQL should query Method type."""
        query = pipeline._source.query
        assert "Method" in query

    def test_reql_has_factory_method(self, pipeline):
        """REQL should check isFactoryMethod."""
        query = pipeline._source.query
        assert "isFactoryMethod" in query


class TestFactoryPatternExecution:
    """Test execution with mock context."""

    @pytest.fixture
    def mock_context(self):
        """Create mock context."""
        ctx = Context(
            reter=Mock(),
            params={"limit": 100},
            language="oo",
            instance_name="default"
        )

        mock_table = Mock()
        mock_table.num_rows = 1
        mock_table.to_pylist.return_value = [
            {"?name": "create_widget", "?entity_type": "factory_method", "?file": "factory.py", "?line": 25, "?creates_type": "Widget"}
        ]
        ctx.reter.reql.return_value = mock_table

        return ctx

    def test_execute_returns_success(self, mock_context):
        """Execution should return success."""
        result = find_factory_pattern(ctx=mock_context)
        assert "success" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
