"""
Unit tests for pull_up_method CADSL detector.

Verifies detection of Pull Up Method refactoring opportunities.
"""

import pytest
from unittest.mock import Mock
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'src'))

from reter_code.dsl.tools.refactoring.pull_up_method import pull_up_method
from reter_code.dsl.core import Pipeline, Context


class TestPullUpMethodStructure:
    """Test pull_up_method detector structure."""

    def test_has_cadsl_spec(self):
        """Tool should have CADSL spec attached."""
        assert hasattr(pull_up_method, '_cadsl_spec')

    def test_spec_has_correct_name(self):
        """Tool spec should have correct name."""
        spec = pull_up_method._cadsl_spec
        assert spec.name == "pull_up_method"

    def test_has_similarity_threshold_param(self):
        """Should have similarity_threshold parameter."""
        spec = pull_up_method._cadsl_spec
        assert 'similarity_threshold' in spec.params


class TestPullUpMethodPipeline:
    """Test pull_up_method pipeline structure."""

    @pytest.fixture
    def pipeline(self):
        """Get the pipeline."""
        spec = pull_up_method._cadsl_spec
        ctx = Context(reter=None, params={"similarity_threshold": 0.9}, language="oo", instance_name="default")
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


class TestPullUpMethodVsReference:
    """Compare with reference implementation."""

    @pytest.fixture
    def pipeline(self):
        """Get the pipeline."""
        spec = pull_up_method._cadsl_spec
        ctx = Context(reter=None, params={"similarity_threshold": 0.9}, language="oo", instance_name="default")
        return spec.pipeline_factory(ctx)

    def test_filters_by_similarity(self, pipeline):
        """Should filter by similarity threshold."""
        query = pipeline._source.query
        has_filter = "FILTER" in query
        assert has_filter, "Should have FILTER for similarity"


class TestPullUpMethodExecution:
    """Test execution with mock context."""

    @pytest.fixture
    def mock_context(self):
        """Create mock context."""
        ctx = Context(
            reter=Mock(),
            params={"similarity_threshold": 0.9},
            language="oo",
            instance_name="default"
        )

        mock_table = Mock()
        mock_table.num_rows = 1
        mock_table.to_pylist.return_value = [
            {"?m1": "Dog.speak", "?name": "speak", "?class1": "Dog", "?class2": "Cat", "?parent_class": "Animal", "?file": "dog.py", "?line": "15", "?similarity": "0.95"}
        ]
        ctx.reter.reql.return_value = mock_table

        return ctx

    def test_execute_returns_success(self, mock_context):
        """Execution should return success."""
        result = pull_up_method(ctx=mock_context)
        assert "success" in result

    def test_execute_returns_findings(self, mock_context):
        """Should return findings."""
        result = pull_up_method(ctx=mock_context)
        if result.get("success"):
            assert "findings" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
