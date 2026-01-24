"""
Unit tests for coupling_matrix CADSL tool.

Verifies generation of class coupling/cohesion matrix visualizations.
"""

import pytest
from unittest.mock import Mock
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'src'))

from reter_code.dsl.tools.diagrams.coupling_matrix import coupling_matrix
from reter_code.dsl.core import Pipeline, Context


class TestCouplingMatrixStructure:
    """Test coupling_matrix tool structure."""

    def test_has_cadsl_spec(self):
        """Tool should have CADSL spec attached."""
        assert hasattr(coupling_matrix, '_cadsl_spec')

    def test_spec_has_correct_name(self):
        """Tool spec should have correct name."""
        spec = coupling_matrix._cadsl_spec
        assert spec.name == "coupling_matrix"

    def test_has_classes_param(self):
        """Should have classes parameter."""
        spec = coupling_matrix._cadsl_spec
        assert 'classes' in spec.params

    def test_has_max_classes_param(self):
        """Should have max_classes parameter."""
        spec = coupling_matrix._cadsl_spec
        assert 'max_classes' in spec.params

    def test_has_threshold_param(self):
        """Should have threshold parameter."""
        spec = coupling_matrix._cadsl_spec
        assert 'threshold' in spec.params

    def test_has_include_inheritance_param(self):
        """Should have include_inheritance parameter."""
        spec = coupling_matrix._cadsl_spec
        assert 'include_inheritance' in spec.params

    def test_has_format_param(self):
        """Should have format parameter."""
        spec = coupling_matrix._cadsl_spec
        assert 'format' in spec.params


class TestCouplingMatrixPipeline:
    """Test coupling_matrix pipeline structure."""

    @pytest.fixture
    def pipeline(self):
        """Get the pipeline."""
        spec = coupling_matrix._cadsl_spec
        ctx = Context(reter=None, params={"classes": None, "max_classes": 20, "threshold": 0, "include_inheritance": True, "format": "markdown"}, language="oo", instance_name="default")
        return spec.pipeline_factory(ctx)

    def test_pipeline_is_pipeline_object(self, pipeline):
        """Should return a Pipeline."""
        assert isinstance(pipeline, Pipeline)

    def test_has_emit_key(self, pipeline):
        """Should emit 'diagram'."""
        assert pipeline._emit_key == "diagram"

    def test_reql_has_method_type(self, pipeline):
        """REQL should query Method type."""
        query = pipeline._source.query
        assert "Method" in query

    def test_reql_has_calls(self, pipeline):
        """REQL should query calls relationship."""
        query = pipeline._source.query
        assert "calls" in query

    def test_reql_has_class_type(self, pipeline):
        """REQL should query Class type."""
        query = pipeline._source.query
        assert "Class" in query

    def test_reql_has_inherits_from(self, pipeline):
        """REQL should query inheritsFrom relationship."""
        query = pipeline._source.query
        assert "inheritsFrom" in query


class TestCouplingMatrixExecution:
    """Test execution with mock context."""

    @pytest.fixture
    def mock_context(self):
        """Create mock context."""
        ctx = Context(
            reter=Mock(),
            params={"classes": None, "max_classes": 20, "threshold": 0, "include_inheritance": True, "format": "markdown"},
            language="oo",
            instance_name="default"
        )

        mock_table = Mock()
        mock_table.num_rows = 2
        mock_table.to_pylist.return_value = [
            {"?callerClassName": "ServiceA", "?calleeClassName": "ServiceB", "?className": None, "?baseName": None},
            {"?callerClassName": None, "?calleeClassName": None, "?className": "Child", "?baseName": "Parent"}
        ]
        ctx.reter.reql.return_value = mock_table

        return ctx

    def test_execute_returns_success(self, mock_context):
        """Execution should return success."""
        result = coupling_matrix(ctx=mock_context)
        assert "success" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
