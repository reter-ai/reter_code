"""
Unit tests for find_test_fixtures CADSL detector.

Verifies detection of pytest fixtures in the codebase.
"""

import pytest
from unittest.mock import Mock
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'src'))

from reter_code.dsl.tools.testing.test_fixtures import find_test_fixtures
from reter_code.dsl.core import Pipeline, Context


class TestFindTestFixturesStructure:
    """Test find_test_fixtures detector structure."""

    def test_has_cadsl_spec(self):
        """Tool should have CADSL spec attached."""
        assert hasattr(find_test_fixtures, '_cadsl_spec')

    def test_spec_has_correct_name(self):
        """Tool spec should have correct name."""
        spec = find_test_fixtures._cadsl_spec
        assert spec.name == "find_test_fixtures"


class TestFindTestFixturesPipeline:
    """Test find_test_fixtures pipeline structure."""

    @pytest.fixture
    def pipeline(self):
        """Get the pipeline."""
        spec = find_test_fixtures._cadsl_spec
        ctx = Context(reter=None, params={}, language="oo", instance_name="default")
        return spec.pipeline_factory(ctx)

    def test_pipeline_is_pipeline_object(self, pipeline):
        """Should return a Pipeline."""
        assert isinstance(pipeline, Pipeline)

    def test_has_emit_key(self, pipeline):
        """Should emit 'findings'."""
        assert pipeline._emit_key == "findings"

    def test_reql_has_function_type(self, pipeline):
        """REQL should filter by Function type."""
        query = pipeline._source.query
        assert "Function" in query

    def test_reql_has_fixture_decorator(self, pipeline):
        """REQL should look for fixture decorator."""
        query = pipeline._source.query
        assert "fixture" in query.lower() or "hasDecorator" in query


class TestFindTestFixturesVsReference:
    """Compare with reference implementation."""

    @pytest.fixture
    def pipeline(self):
        """Get the pipeline."""
        spec = find_test_fixtures._cadsl_spec
        ctx = Context(reter=None, params={}, language="oo", instance_name="default")
        return spec.pipeline_factory(ctx)

    def test_queries_for_fixtures(self, pipeline):
        """Should query for fixture decorator."""
        query = pipeline._source.query
        has_fixture = "fixture" in query.lower()
        assert has_fixture, "Should look for fixture decorator"


class TestFindTestFixturesExecution:
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
            {"?f": "mock_database", "?name": "mock_database", "?file": "conftest.py", "?line": "10", "?scope": "function", "?usage_count": "25"}
        ]
        ctx.reter.reql.return_value = mock_table

        return ctx

    def test_execute_returns_success(self, mock_context):
        """Execution should return success."""
        result = find_test_fixtures(ctx=mock_context)
        assert "success" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
