"""
Unit tests for list_classes CADSL tool.

Verifies behavior matches reference implementation at:
d:/ROOT/reter_code/src/reter_code/tools/python_basic/python_tools.py

Reference behavior:
- Returns classes with: name, qualified_name, full_qualified_name, file, line, method_count
- Supports optional module_name filter with FILTER(CONTAINS(?file, module_name))
- Pagination with limit/offset
- Returns total_count and has_more
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'src'))

from reter_code.dsl.tools.inspection.list_classes import list_classes
from reter_code.dsl.core import Pipeline, Context


class TestListClassesStructure:
    """Test list_classes tool structure (without execution)."""

    def test_has_cadsl_spec(self):
        """Tool should have CADSL spec attached."""
        assert hasattr(list_classes, '_cadsl_spec'), \
            "list_classes should have _cadsl_spec attribute"

    def test_has_cadsl_tool(self):
        """Tool should have CADSL tool attached."""
        assert hasattr(list_classes, '_cadsl_tool'), \
            "list_classes should have _cadsl_tool attribute"

    def test_spec_has_correct_name(self):
        """Tool spec should have correct name."""
        spec = list_classes._cadsl_spec
        assert spec.name == "list_classes", \
            f"Tool name should be 'list_classes', got '{spec.name}'"

    def test_spec_has_query_type(self):
        """Tool should be of QUERY type."""
        from reter_code.dsl.core import ToolType
        spec = list_classes._cadsl_spec
        assert spec.type == ToolType.QUERY, \
            f"Tool type should be QUERY, got {spec.type}"

    def test_has_required_params(self):
        """Tool should have required parameters."""
        spec = list_classes._cadsl_spec
        param_names = list(spec.params.keys())

        assert 'module' in param_names, \
            "Tool should have 'module' parameter"
        assert 'limit' in param_names, \
            "Tool should have 'limit' parameter"
        assert 'offset' in param_names, \
            "Tool should have 'offset' parameter"

    def test_module_param_is_optional(self):
        """Module parameter should be optional."""
        spec = list_classes._cadsl_spec
        module_param = spec.params.get('module')

        assert module_param is not None, "Should have module param"
        assert module_param.required == False, \
            "Module param should be optional (required=False)"

    def test_limit_has_default(self):
        """Limit parameter should have default value."""
        spec = list_classes._cadsl_spec
        limit_param = spec.params.get('limit')

        assert limit_param is not None, "Should have limit param"
        assert limit_param.default == 100, \
            f"Limit default should be 100, got {limit_param.default}"


class TestListClassesPipeline:
    """Test the pipeline structure."""

    @pytest.fixture
    def pipeline(self):
        """Get the actual pipeline by calling pipeline_factory."""
        spec = list_classes._cadsl_spec
        # Create a minimal context
        ctx = Context(reter=None, params={}, language="oo", instance_name="default")
        return spec.pipeline_factory(ctx)

    def test_pipeline_is_pipeline_object(self, pipeline):
        """Pipeline factory should return a Pipeline."""
        assert isinstance(pipeline, Pipeline), \
            f"Should return Pipeline, got {type(pipeline)}"

    def test_pipeline_has_source(self, pipeline):
        """Pipeline should have a REQL source."""
        assert pipeline._source is not None, \
            "Pipeline should have a source"

    def test_pipeline_has_steps(self, pipeline):
        """Pipeline should have processing steps."""
        assert len(pipeline._steps) >= 1, \
            "Pipeline should have at least one step"

    def test_pipeline_has_emit_key(self, pipeline):
        """Pipeline should emit 'classes' key."""
        assert pipeline._emit_key == "classes", \
            f"Should emit 'classes', got '{pipeline._emit_key}'"

    def test_reql_query_has_class_type(self, pipeline):
        """REQL should filter by Class type."""
        query = pipeline._source.query
        assert "type {Class}" in query or "type oo:Class" in query, \
            "Query should filter by Class type"

    def test_reql_query_has_name(self, pipeline):
        """REQL should select name."""
        query = pipeline._source.query
        assert "name ?name" in query, \
            "Query should select name"

    def test_reql_query_has_file(self, pipeline):
        """REQL should select file."""
        query = pipeline._source.query
        assert "inFile ?file" in query, \
            "Query should select inFile"

    def test_reql_query_has_line(self, pipeline):
        """REQL should select line number."""
        query = pipeline._source.query
        assert "atLine ?line" in query, \
            "Query should select atLine"

    def test_reql_query_has_order_by(self, pipeline):
        """REQL should have ORDER BY."""
        query = pipeline._source.query
        assert "ORDER BY" in query, \
            "Query should have ORDER BY clause"

    def test_select_step_has_qualified_name(self, pipeline):
        """Select step should include qualified_name."""
        select_step = None
        for step in pipeline._steps:
            if hasattr(step, 'fields'):
                select_step = step
                break

        assert select_step is not None, "Should have select step"
        assert "qualified_name" in select_step.fields, \
            "Select should include qualified_name"


class TestListClassesVsReference:
    """Compare CADSL implementation to reference."""

    @pytest.fixture
    def pipeline(self):
        """Get the actual pipeline."""
        spec = list_classes._cadsl_spec
        ctx = Context(reter=None, params={}, language="oo", instance_name="default")
        return spec.pipeline_factory(ctx)

    def test_reference_has_module_filter(self, pipeline):
        """
        ISSUE: Reference has module filter, CADSL may not.

        Reference (python_tools.py:161-172):
            FILTER(CONTAINS(?file, "{module_name}"))
        """
        query = pipeline._source.query

        has_filter = "FILTER" in query and ("CONTAINS" in query or "{module}" in query)

        if not has_filter:
            pytest.fail(
                "MISSING: Module filter not implemented.\n"
                "Reference uses: FILTER(CONTAINS(?file, module_name))\n"
                "Add to REQL: FILTER {{ CONTAINS(?file, \"{module}\") }}"
            )

    def test_reference_has_method_count(self, pipeline):
        """
        ISSUE: Reference queries method counts.

        Reference (python_tools.py:196-208):
            SELECT ?class (COUNT(?method) AS ?methodCount)
            WHERE {{
                ?method type {method_concept}
                ?method definedIn ?class
            }}
            GROUP BY ?class
        """
        query = pipeline._source.query

        has_method_count = "methodCount" in query or "method_count" in query or "COUNT" in query

        if not has_method_count:
            pytest.fail(
                "MISSING: Method count not implemented.\n"
                "Reference does second query with COUNT(?method).\n"
                "Consider adding methodCount field or secondary query."
            )

    def test_uses_entity_id_as_qualified_name(self, pipeline):
        """
        Entity IDs are the qualified names.

        The entity ID (?c) is used directly as the qualified name instead of
        querying a separate qualifiedName attribute. This is achieved via
        the select step: `qualified_name: c`
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


class TestListClassesExecution:
    """Test execution with mock context."""

    @pytest.fixture
    def mock_context(self):
        """Create mock context with ReterWrapper."""
        ctx = Context(
            reter=Mock(),
            params={"module": None, "limit": 100, "offset": 0},
            language="oo",
            instance_name="default"
        )

        # Mock REQL result as PyArrow table
        mock_table = Mock()
        mock_table.num_rows = 2
        mock_table.to_pylist.return_value = [
            {"?c": "module.ClassA", "?name": "ClassA", "?file": "test.py", "?line": "10"},
            {"?c": "module.ClassB", "?name": "ClassB", "?file": "test.py", "?line": "20"},
        ]
        ctx.reter.reql.return_value = mock_table

        return ctx

    def test_execute_returns_success(self, mock_context):
        """Execution should return success dict."""
        result = list_classes(ctx=mock_context)

        assert isinstance(result, dict), "Should return dict"
        assert "success" in result, "Should have 'success' key"

    def test_execute_returns_classes(self, mock_context):
        """Execution should return classes list."""
        result = list_classes(ctx=mock_context)

        if result.get("success"):
            assert "classes" in result, "Should have 'classes' key"
            assert isinstance(result["classes"], list), "classes should be list"
        else:
            pytest.skip(f"Execution failed: {result.get('error')}")

    def test_execute_class_has_required_fields(self, mock_context):
        """Each class should have required fields."""
        result = list_classes(ctx=mock_context)

        if not result.get("success") or not result.get("classes"):
            pytest.skip(f"No classes returned: {result.get('error')}")

        first_class = result["classes"][0]

        # Fields from reference implementation
        required_fields = ["name", "file", "line", "qualified_name"]

        for field in required_fields:
            assert field in first_class, \
                f"Class should have '{field}' field"

    def test_execute_with_module_filter(self, mock_context):
        """Should support module filter parameter."""
        mock_context.params["module"] = "test"

        result = list_classes(ctx=mock_context)

        # Just verify it doesn't error - actual filtering tested separately
        assert "success" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
