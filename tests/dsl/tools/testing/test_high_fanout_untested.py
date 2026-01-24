"""
Unit tests for high_fanout_untested CADSL detector.

Verifies detection of high fan-out functions without tests.
"""

import pytest
from unittest.mock import Mock
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'src'))

from reter_code.dsl.tools.testing.high_fanout_untested import high_fanout_untested
from reter_code.dsl.core import Pipeline, Context


class TestHighFanoutUntestedStructure:
    """Test high_fanout_untested detector structure."""

    def test_has_cadsl_spec(self):
        """Tool should have CADSL spec attached."""
        assert hasattr(high_fanout_untested, '_cadsl_spec')

    def test_spec_has_correct_name(self):
        """Tool spec should have correct name."""
        spec = high_fanout_untested._cadsl_spec
        assert spec.name == "high_fanout_untested"

    def test_has_min_fanout_param(self):
        """Should have min_fanout parameter."""
        spec = high_fanout_untested._cadsl_spec
        assert 'min_fanout' in spec.params


class TestHighFanoutUntestedPipeline:
    """Test high_fanout_untested pipeline structure."""

    @pytest.fixture
    def pipeline(self):
        """Get the pipeline."""
        spec = high_fanout_untested._cadsl_spec
        ctx = Context(reter=None, params={"min_fanout": 5}, language="oo", instance_name="default")
        return spec.pipeline_factory(ctx)

    def test_pipeline_is_pipeline_object(self, pipeline):
        """Should return a Pipeline."""
        assert isinstance(pipeline, Pipeline)

    def test_has_emit_key(self, pipeline):
        """Should emit 'findings'."""
        assert pipeline._emit_key == "findings"


class TestHighFanoutUntestedVsReference:
    """Compare with reference implementation."""

    @pytest.fixture
    def pipeline(self):
        """Get the pipeline."""
        spec = high_fanout_untested._cadsl_spec
        ctx = Context(reter=None, params={"min_fanout": 5}, language="oo", instance_name="default")
        return spec.pipeline_factory(ctx)

    def test_filters_by_fanout(self, pipeline):
        """Should filter by fanout count."""
        query = pipeline._source.query
        has_filter = "FILTER" in query
        assert has_filter, "Should filter by fanout"


class TestHighFanoutUntestedExecution:
    """Test execution with mock context."""

    @pytest.fixture
    def mock_context(self):
        """Create mock context."""
        ctx = Context(
            reter=Mock(),
            params={"min_fanout": 5},
            language="oo",
            instance_name="default"
        )

        mock_table = Mock()
        mock_table.num_rows = 1
        mock_table.to_pylist.return_value = [
            {"?e": "Coordinator.run", "?name": "run", "?class_name": "Coordinator", "?file": "coordinator.py", "?line": "30", "?fanout": "12"}
        ]
        ctx.reter.reql.return_value = mock_table

        return ctx

    def test_execute_returns_success(self, mock_context):
        """Execution should return success."""
        result = high_fanout_untested(ctx=mock_context)
        assert "success" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
