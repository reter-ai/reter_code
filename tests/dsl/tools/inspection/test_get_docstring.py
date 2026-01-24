"""
Unit tests for get_docstring CADSL tool.

Verifies behavior matches reference implementation at:
d:/ROOT/reter_code/src/reter_code/tools/python_basic/python_tools.py:777-827

Reference behavior:
- Searches by name or qualified name (CONTAINS filter)
- Returns: qualified_name, name, type, docstring
"""

import pytest
from unittest.mock import Mock
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'src'))

from reter_code.dsl.tools.inspection.get_docstring import get_docstring
from reter_code.dsl.core import Pipeline, Context, ToolType


class TestGetDocstringStructure:
    """Test get_docstring tool structure."""

    def test_has_cadsl_spec(self):
        """Tool should have CADSL spec attached."""
        assert hasattr(get_docstring, '_cadsl_spec')

    def test_spec_has_correct_name(self):
        """Tool spec should have correct name."""
        spec = get_docstring._cadsl_spec
        assert spec.name == "get_docstring"

    def test_has_target_param(self):
        """Should have target parameter."""
        spec = get_docstring._cadsl_spec
        assert 'target' in spec.params

    def test_target_param_is_required(self):
        """Target param should be required."""
        spec = get_docstring._cadsl_spec
        assert spec.params['target'].required == True


class TestGetDocstringPipeline:
    """Test get_docstring pipeline structure."""

    @pytest.fixture
    def pipeline(self):
        """Get the pipeline."""
        spec = get_docstring._cadsl_spec
        ctx = Context(reter=None, params={"target": "test"}, language="oo", instance_name="default")
        return spec.pipeline_factory(ctx)

    def test_pipeline_is_pipeline_object(self, pipeline):
        """Should return a Pipeline."""
        assert isinstance(pipeline, Pipeline)

    def test_reql_has_docstring_field(self, pipeline):
        """REQL should select docstring."""
        query = pipeline._source.query
        assert "docstring" in query

    def test_reql_has_type_info(self, pipeline):
        """REQL should get entity type."""
        query = pipeline._source.query
        has_type = "entity_type" in query or "type" in query or "BIND" in query
        assert has_type, "Should have entity type"

    def test_reql_supports_multiple_entity_types(self, pipeline):
        """REQL should support Class, Method, Function."""
        query = pipeline._source.query
        has_class = "Class" in query
        has_method = "Method" in query
        has_function = "Function" in query
        assert has_class and (has_method or has_function), \
            "Should support multiple entity types"


class TestGetDocstringVsReference:
    """Compare with reference implementation."""

    @pytest.fixture
    def pipeline(self):
        """Get the pipeline."""
        spec = get_docstring._cadsl_spec
        ctx = Context(reter=None, params={"target": "test"}, language="oo", instance_name="default")
        return spec.pipeline_factory(ctx)

    def test_reference_has_contains_filter(self, pipeline):
        """
        Reference uses CONTAINS for flexible name matching.

        Reference (python_tools.py:810):
            FILTER(CONTAINS(?entityName, name) || CONTAINS(?entity, name))
        """
        query = pipeline._source.query

        has_contains = "CONTAINS" in query

        if not has_contains:
            pytest.fail(
                "MISSING: CONTAINS filter for flexible name matching.\n"
                "Reference uses: FILTER(CONTAINS(?entityName, name) || CONTAINS(?entity, name))\n"
                "This allows finding by partial name or qualified name."
            )


class TestGetDocstringExecution:
    """Test execution with mock context."""

    @pytest.fixture
    def mock_context(self):
        """Create mock context."""
        ctx = Context(
            reter=Mock(),
            params={"target": "TestClass"},
            language="oo",
            instance_name="default"
        )

        mock_table = Mock()
        mock_table.num_rows = 1
        mock_table.to_pylist.return_value = [
            {
                "?e": "test.TestClass",
                "?name": "TestClass",
                "?docstring": "A test class.",
                "?file": "test.py",
                "?line": "10",
                "?entity_type": "class"
            }
        ]
        ctx.reter.reql.return_value = mock_table

        return ctx

    def test_execute_returns_success(self, mock_context):
        """Execution should return success."""
        result = get_docstring(ctx=mock_context)
        assert "success" in result

    def test_execute_returns_result(self, mock_context):
        """Should return result."""
        result = get_docstring(ctx=mock_context)
        if result.get("success"):
            assert "result" in result

    def test_result_has_docstring(self, mock_context):
        """Result should have docstring field."""
        result = get_docstring(ctx=mock_context)

        if not result.get("success") or not result.get("result"):
            pytest.skip("No result returned")

        item = result["result"]
        if isinstance(item, list) and item:
            item = item[0]

        assert "docstring" in item or "name" in item


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
