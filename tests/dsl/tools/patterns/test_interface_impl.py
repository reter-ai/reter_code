"""
Unit tests for find_interface_implementations CADSL tool.

Verifies detection of interface/ABC implementations.
"""

import pytest
from unittest.mock import Mock
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'src'))

from reter_code.dsl.tools.patterns.interface_impl import find_interface_implementations
from reter_code.dsl.core import Pipeline, Context


class TestInterfaceImplStructure:
    """Test find_interface_implementations tool structure."""

    def test_has_cadsl_spec(self):
        """Tool should have CADSL spec attached."""
        assert hasattr(find_interface_implementations, '_cadsl_spec')

    def test_spec_has_correct_name(self):
        """Tool spec should have correct name."""
        spec = find_interface_implementations._cadsl_spec
        assert spec.name == "find_interface_implementations"

    def test_has_interface_name_param(self):
        """Should have interface_name parameter."""
        spec = find_interface_implementations._cadsl_spec
        assert 'interface_name' in spec.params

    def test_has_limit_param(self):
        """Should have limit parameter."""
        spec = find_interface_implementations._cadsl_spec
        assert 'limit' in spec.params

    def test_has_category_metadata(self):
        """Should have patterns category."""
        spec = find_interface_implementations._cadsl_spec
        assert spec.meta.get('category') == 'patterns'

    def test_has_severity_metadata(self):
        """Should have info severity."""
        spec = find_interface_implementations._cadsl_spec
        assert spec.meta.get('severity') == 'info'


class TestInterfaceImplPipeline:
    """Test find_interface_implementations pipeline structure."""

    @pytest.fixture
    def pipeline(self):
        """Get the pipeline."""
        spec = find_interface_implementations._cadsl_spec
        ctx = Context(reter=None, params={"interface_name": None, "limit": 100}, language="oo", instance_name="default")
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

    def test_reql_has_inherits_from(self, pipeline):
        """REQL should query inheritsFrom."""
        query = pipeline._source.query
        assert "inheritsFrom" in query

    def test_reql_has_is_abstract(self, pipeline):
        """REQL should check isAbstract."""
        query = pipeline._source.query
        assert "isAbstract" in query


class TestInterfaceImplExecution:
    """Test execution with mock context."""

    @pytest.fixture
    def mock_context(self):
        """Create mock context."""
        ctx = Context(
            reter=Mock(),
            params={"interface_name": None, "limit": 100},
            language="oo",
            instance_name="default"
        )

        mock_table = Mock()
        mock_table.num_rows = 1
        mock_table.to_pylist.return_value = [
            {"?class_name": "FileHandler", "?interface_name": "Handler", "?file": "handlers.py", "?line": 10, "?implemented_methods": 3}
        ]
        ctx.reter.reql.return_value = mock_table

        return ctx

    def test_execute_returns_success(self, mock_context):
        """Execution should return success."""
        result = find_interface_implementations(ctx=mock_context)
        assert "success" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
