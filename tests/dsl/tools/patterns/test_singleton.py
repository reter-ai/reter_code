"""
Unit tests for find_singleton_pattern CADSL tool.

Verifies detection of singleton pattern implementations.
"""

import pytest
from unittest.mock import Mock
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'src'))

from reter_code.dsl.tools.patterns.singleton import find_singleton_pattern
from reter_code.dsl.core import Pipeline, Context


class TestSingletonPatternStructure:
    """Test find_singleton_pattern tool structure."""

    def test_has_cadsl_spec(self):
        """Tool should have CADSL spec attached."""
        assert hasattr(find_singleton_pattern, '_cadsl_spec')

    def test_spec_has_correct_name(self):
        """Tool spec should have correct name."""
        spec = find_singleton_pattern._cadsl_spec
        assert spec.name == "find_singleton_pattern"

    def test_has_limit_param(self):
        """Should have limit parameter."""
        spec = find_singleton_pattern._cadsl_spec
        assert 'limit' in spec.params

    def test_has_category_metadata(self):
        """Should have patterns category."""
        spec = find_singleton_pattern._cadsl_spec
        assert spec.meta.get('category') == 'patterns'

    def test_has_severity_metadata(self):
        """Should have info severity."""
        spec = find_singleton_pattern._cadsl_spec
        assert spec.meta.get('severity') == 'info'


class TestSingletonPatternPipeline:
    """Test find_singleton_pattern pipeline structure."""

    @pytest.fixture
    def pipeline(self):
        """Get the pipeline."""
        spec = find_singleton_pattern._cadsl_spec
        ctx = Context(reter=None, params={"limit": 100}, language="oo", instance_name="default")
        return spec.pipeline_factory(ctx)

    def test_pipeline_is_pipeline_object(self, pipeline):
        """Should return a Pipeline."""
        assert isinstance(pipeline, Pipeline)

    def test_has_emit_key(self, pipeline):
        """Should emit 'findings'."""
        assert pipeline._emit_key == "findings"

    def test_reql_has_class_type(self, pipeline):
        """REQL should query Class type."""
        query = pipeline._source.query
        assert "Class" in query

    def test_reql_has_is_singleton(self, pipeline):
        """REQL should check isSingleton."""
        query = pipeline._source.query
        assert "isSingleton" in query


class TestSingletonPatternExecution:
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
            {"?name": "Database", "?file": "db.py", "?line": 10, "?implementation_type": "metaclass"}
        ]
        ctx.reter.reql.return_value = mock_table

        return ctx

    def test_execute_returns_success(self, mock_context):
        """Execution should return success."""
        result = find_singleton_pattern(ctx=mock_context)
        assert "success" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
