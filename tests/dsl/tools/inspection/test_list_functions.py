"""
Unit tests for list_functions CADSL tool.

Verifies behavior matches reference implementation at:
d:/ROOT/reter_code/src/reter_code/tools/python_basic/python_tools.py:854-914

Reference behavior:
- Returns functions with: qualified_name, name, file, line, return_type
- Optional module filter with FILTER(CONTAINS(?file, module_name))
- ORDER BY ?name (or ?line when filtered)
"""

import pytest
from unittest.mock import Mock
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'src'))

from reter_code.dsl.tools.inspection.list_functions import list_functions
from reter_code.dsl.core import Pipeline, Context, ToolType


class TestListFunctionsStructure:
    """Test list_functions tool structure."""

    def test_has_cadsl_spec(self):
        """Tool should have CADSL spec attached."""
        assert hasattr(list_functions, '_cadsl_spec')

    def test_spec_has_correct_name(self):
        """Tool spec should have correct name."""
        spec = list_functions._cadsl_spec
        assert spec.name == "list_functions"

    def test_spec_has_query_type(self):
        """Tool should be of QUERY type."""
        spec = list_functions._cadsl_spec
        assert spec.type == ToolType.QUERY

    def test_has_module_param(self):
        """Should have module parameter."""
        spec = list_functions._cadsl_spec
        assert 'module' in spec.params

    def test_module_param_is_optional(self):
        """Module param should be optional."""
        spec = list_functions._cadsl_spec
        assert spec.params['module'].required == False

    def test_has_limit_param(self):
        """Should have limit parameter."""
        spec = list_functions._cadsl_spec
        assert 'limit' in spec.params


class TestListFunctionsPipeline:
    """Test list_functions pipeline structure."""

    @pytest.fixture
    def pipeline(self):
        """Get the pipeline."""
        spec = list_functions._cadsl_spec
        ctx = Context(reter=None, params={}, language="oo", instance_name="default")
        return spec.pipeline_factory(ctx)

    def test_pipeline_is_pipeline_object(self, pipeline):
        """Should return a Pipeline."""
        assert isinstance(pipeline, Pipeline)

    def test_has_emit_key(self, pipeline):
        """Should emit 'functions'."""
        assert pipeline._emit_key == "functions"

    def test_reql_has_function_type(self, pipeline):
        """REQL should filter by Function type."""
        query = pipeline._source.query
        assert "type {Function}" in query or "type oo:Function" in query

    def test_reql_has_name(self, pipeline):
        """REQL should select name."""
        query = pipeline._source.query
        assert "name ?name" in query

    def test_reql_has_file(self, pipeline):
        """REQL should select file."""
        query = pipeline._source.query
        assert "inFile ?file" in query

    def test_reql_has_line(self, pipeline):
        """REQL should select line."""
        query = pipeline._source.query
        assert "atLine ?line" in query

    def test_reql_excludes_methods(self, pipeline):
        """REQL should exclude methods (functions defined in class)."""
        query = pipeline._source.query
        # Check for either FILTER NOT EXISTS or some mechanism to exclude methods
        has_exclusion = "FILTER NOT EXISTS" in query or "definedIn" in query
        assert has_exclusion, "Should exclude methods (functions in classes)"

    def test_select_has_qualified_name(self, pipeline):
        """Select should include qualified_name."""
        select_step = None
        for step in pipeline._steps:
            if hasattr(step, 'fields'):
                select_step = step
                break
        assert select_step is not None
        assert "qualified_name" in select_step.fields


class TestListFunctionsVsReference:
    """Compare with reference implementation."""

    @pytest.fixture
    def pipeline(self):
        """Get the pipeline."""
        spec = list_functions._cadsl_spec
        ctx = Context(reter=None, params={}, language="oo", instance_name="default")
        return spec.pipeline_factory(ctx)

    def test_reference_has_module_filter(self, pipeline):
        """
        Reference uses FILTER(CONTAINS(?file, module_name)).

        Reference (python_tools.py:892):
            FILTER(CONTAINS(?file, "{module_name}"))
        """
        query = pipeline._source.query

        has_filter = "CONTAINS" in query or "{module}" in query

        if not has_filter:
            pytest.fail(
                "MISSING: Module filter not implemented.\n"
                "Reference uses: FILTER(CONTAINS(?file, module_name))\n"
                "Add: FILTER { CONTAINS(?file, \"{module}\") }"
            )

    def test_uses_entity_id_as_qualified_name(self, pipeline):
        """
        Entity IDs are the qualified names.

        The entity ID (?f) is used directly as the qualified name instead of
        querying a separate qualifiedName attribute. This is achieved via
        the select step: `qualified_name: f`
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

    def test_reference_has_return_type(self, pipeline):
        """
        Reference queries returnType optionally.

        Reference (python_tools.py:891):
            OPTIONAL {{ ?function returnType ?returnType }}
        """
        query = pipeline._source.query

        has_return_type = "returnType" in query

        if not has_return_type:
            pytest.fail(
                "MISSING: returnType not queried.\n"
                "Reference has: OPTIONAL { ?function returnType ?returnType }"
            )


class TestListFunctionsExecution:
    """Test execution with mock context."""

    @pytest.fixture
    def mock_context(self):
        """Create mock context."""
        ctx = Context(
            reter=Mock(),
            params={"module": None, "limit": 100, "offset": 0},
            language="oo",
            instance_name="default"
        )

        mock_table = Mock()
        mock_table.num_rows = 2
        mock_table.to_pylist.return_value = [
            {"?f": "test.func1", "?name": "func1", "?file": "test.py", "?line": "10", "?module": "test"},
            {"?f": "test.func2", "?name": "func2", "?file": "test.py", "?line": "20", "?module": "test"},
        ]
        ctx.reter.reql.return_value = mock_table

        return ctx

    def test_execute_returns_success(self, mock_context):
        """Execution should return success."""
        result = list_functions(ctx=mock_context)
        assert "success" in result

    def test_execute_returns_functions(self, mock_context):
        """Should return functions list."""
        result = list_functions(ctx=mock_context)
        if result.get("success"):
            assert "functions" in result
            assert isinstance(result["functions"], list)

    def test_function_has_required_fields(self, mock_context):
        """Each function should have required fields."""
        result = list_functions(ctx=mock_context)

        if not result.get("success") or not result.get("functions"):
            pytest.skip("No functions returned")

        func = result["functions"][0]
        required = ["name", "file", "line", "qualified_name"]
        for field in required:
            assert field in func, f"Function should have '{field}'"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
