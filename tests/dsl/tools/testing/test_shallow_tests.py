"""
Unit tests for shallow_tests CADSL detector.

Verifies detection of test classes with too few test methods.
"""

import pytest
from unittest.mock import Mock
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'src'))

from reter_code.dsl.tools.testing.shallow_tests import shallow_tests
from reter_code.dsl.core import Pipeline, Context


class TestShallowTestsStructure:
    """Test shallow_tests detector structure."""

    def test_has_cadsl_spec(self):
        """Tool should have CADSL spec attached."""
        assert hasattr(shallow_tests, '_cadsl_spec')

    def test_spec_has_correct_name(self):
        """Tool spec should have correct name."""
        spec = shallow_tests._cadsl_spec
        assert spec.name == "shallow_tests"

    def test_has_min_tests_param(self):
        """Should have min_tests parameter."""
        spec = shallow_tests._cadsl_spec
        assert 'min_tests' in spec.params


class TestShallowTestsPipeline:
    """Test shallow_tests pipeline structure."""

    @pytest.fixture
    def pipeline(self):
        """Get the pipeline."""
        spec = shallow_tests._cadsl_spec
        ctx = Context(reter=None, params={"min_tests": 3}, language="oo", instance_name="default")
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


class TestShallowTestsVsReference:
    """Compare with reference implementation."""

    @pytest.fixture
    def pipeline(self):
        """Get the pipeline."""
        spec = shallow_tests._cadsl_spec
        ctx = Context(reter=None, params={"min_tests": 3}, language="oo", instance_name="default")
        return spec.pipeline_factory(ctx)

    def test_filters_by_test_count(self, pipeline):
        """Should filter by test count."""
        query = pipeline._source.query
        has_filter = "FILTER" in query
        assert has_filter, "Should filter by test count"


class TestShallowTestsExecution:
    """Test execution with mock context."""

    @pytest.fixture
    def mock_context(self):
        """Create mock context."""
        ctx = Context(
            reter=Mock(),
            params={"min_tests": 3},
            language="oo",
            instance_name="default"
        )

        mock_table = Mock()
        mock_table.num_rows = 1
        mock_table.to_pylist.return_value = [
            {"?c": "TestService", "?name": "TestService", "?file": "test_service.py", "?line": "5", "?test_count": "1", "?tested_class": "Service"}
        ]
        ctx.reter.reql.return_value = mock_table

        return ctx

    def test_execute_returns_success(self, mock_context):
        """Execution should return success."""
        result = shallow_tests(ctx=mock_context)
        assert "success" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
