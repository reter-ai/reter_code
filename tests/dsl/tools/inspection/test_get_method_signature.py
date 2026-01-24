"""
Unit tests for get_method_signature CADSL tool.

Verifies behavior matches reference implementation at:
d:/ROOT/reter_code/src/reter_code/tools/python_basic/python_tools.py:658-708

Reference behavior:
- Finds method by name or qualified name (CONTAINS filter)
- Returns: name, qualifiedName, returnType, line, parameters
"""

import pytest
from unittest.mock import Mock
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'src'))

from reter_code.dsl.tools.inspection.get_method_signature import get_method_signature
from reter_code.dsl.core import Pipeline, Context, ToolType


class TestGetMethodSignatureStructure:
    """Test get_method_signature tool structure."""

    def test_has_cadsl_spec(self):
        """Tool should have CADSL spec attached."""
        assert hasattr(get_method_signature, '_cadsl_spec')

    def test_spec_has_correct_name(self):
        """Tool spec should have correct name."""
        spec = get_method_signature._cadsl_spec
        assert spec.name == "get_method_signature"

    def test_has_target_param(self):
        """Should have target parameter."""
        spec = get_method_signature._cadsl_spec
        assert 'target' in spec.params

    def test_target_param_is_required(self):
        """Target param should be required."""
        spec = get_method_signature._cadsl_spec
        assert spec.params['target'].required == True


class TestGetMethodSignaturePipeline:
    """Test get_method_signature pipeline structure."""

    @pytest.fixture
    def pipeline(self):
        """Get the pipeline."""
        spec = get_method_signature._cadsl_spec
        ctx = Context(reter=None, params={"target": "test_method"}, language="oo", instance_name="default")
        return spec.pipeline_factory(ctx)

    def test_pipeline_is_pipeline_object(self, pipeline):
        """Should return a Pipeline."""
        assert isinstance(pipeline, Pipeline)

    def test_reql_has_return_type(self, pipeline):
        """REQL should query returnType."""
        query = pipeline._source.query
        assert "returnType" in query or "return_type" in query

    def test_reql_supports_methods(self, pipeline):
        """REQL should support Method type."""
        query = pipeline._source.query
        assert "Method" in query

    def test_reql_has_parameter_info(self, pipeline):
        """REQL should query parameter info."""
        query = pipeline._source.query
        has_params = "param" in query.lower() or "Parameter" in query
        assert has_params, "Should query parameter information"


class TestGetMethodSignatureVsReference:
    """Compare with reference implementation."""

    @pytest.fixture
    def pipeline(self):
        """Get the pipeline."""
        spec = get_method_signature._cadsl_spec
        ctx = Context(reter=None, params={"target": "test_method"}, language="oo", instance_name="default")
        return spec.pipeline_factory(ctx)

    def test_has_contains_filter(self, pipeline):
        """
        Should use CONTAINS for flexible name matching.

        Uses FILTER(?name = target || CONTAINS(STR(?m), target))
        to match by entity ID (which is the qualified name)
        """
        query = pipeline._source.query

        has_contains = "CONTAINS" in query

        if not has_contains:
            pytest.fail(
                "MISSING: CONTAINS filter for flexible name matching.\n"
                "Should use: FILTER(?name = target || CONTAINS(STR(?m), target))"
            )

    def test_uses_entity_id_as_qualified_name(self, pipeline):
        """
        Entity IDs are the qualified names.

        The entity ID (?m) is used directly as the qualified name instead of
        querying a separate qualifiedName attribute. This is achieved via
        the select step: `qualified_name: m`
        """
        # Check that the select step maps entity ID to qualified_name
        select_step = None
        for step in pipeline._steps:
            if hasattr(step, 'fields'):
                select_step = step
                break

        assert select_step is not None, "Should have select step"
        assert "qualified_name" in select_step.fields, \
            "Select should include qualified_name mapped from entity ID"


class TestGetMethodSignatureExecution:
    """Test execution with mock context."""

    @pytest.fixture
    def mock_context(self):
        """Create mock context."""
        ctx = Context(
            reter=Mock(),
            params={"target": "test_method"},
            language="oo",
            instance_name="default"
        )

        mock_table = Mock()
        mock_table.num_rows = 1
        mock_table.to_pylist.return_value = [
            {
                "?m": "test.TestClass.test_method",
                "?name": "test_method",
                "?file": "test.py",
                "?line": "15",
                "?return_type": "str",
                "?param_name": "arg1",
                "?param_type": "int",
                "?param_default": None,
                "?param_order": "1"
            }
        ]
        ctx.reter.reql.return_value = mock_table

        return ctx

    def test_execute_returns_success(self, mock_context):
        """Execution should return success."""
        result = get_method_signature(ctx=mock_context)
        assert "success" in result

    def test_execute_returns_signature(self, mock_context):
        """Should return signature."""
        result = get_method_signature(ctx=mock_context)
        if result.get("success"):
            assert "signature" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
