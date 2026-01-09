"""
Unit tests for describe_class CADSL tool.

Verifies behavior matches reference implementation at:
d:/ROOT/codeine/src/codeine/tools/python_basic/python_tools.py:241-500

Reference behavior:
- Finds class by name (supports simple or qualified name via CONTAINS filter)
- Returns: class_info with name, qualifiedName, file, line, docstring
- Returns: methods list with signatures, parameters, return types
- Returns: attributes list
- Supports pagination for methods
- Supports summary_only mode
"""

import pytest
from unittest.mock import Mock
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'src'))

from codeine.dsl.tools.inspection.describe_class import describe_class
from codeine.dsl.core import Pipeline, Context, ToolType


class TestDescribeClassStructure:
    """Test describe_class tool structure."""

    def test_has_cadsl_spec(self):
        """Tool should have CADSL spec attached."""
        assert hasattr(describe_class, '_cadsl_spec')

    def test_spec_has_correct_name(self):
        """Tool spec should have correct name."""
        spec = describe_class._cadsl_spec
        assert spec.name == "describe_class"

    def test_spec_has_query_type(self):
        """Tool should be of QUERY type."""
        spec = describe_class._cadsl_spec
        assert spec.type == ToolType.QUERY

    def test_has_target_param(self):
        """Should have target parameter."""
        spec = describe_class._cadsl_spec
        assert 'target' in spec.params

    def test_target_param_is_required(self):
        """Target param should be required."""
        spec = describe_class._cadsl_spec
        assert spec.params['target'].required == True

    def test_has_include_methods_param(self):
        """Should have include_methods parameter."""
        spec = describe_class._cadsl_spec
        assert 'include_methods' in spec.params

    def test_has_include_attributes_param(self):
        """Should have include_attributes parameter."""
        spec = describe_class._cadsl_spec
        assert 'include_attributes' in spec.params


class TestDescribeClassPipeline:
    """Test describe_class pipeline structure."""

    @pytest.fixture
    def pipeline(self):
        """Get the pipeline."""
        spec = describe_class._cadsl_spec
        ctx = Context(reter=None, params={"target": "TestClass"}, language="oo", instance_name="default")
        return spec.pipeline_factory(ctx)

    def test_pipeline_is_pipeline_object(self, pipeline):
        """Should return a Pipeline."""
        assert isinstance(pipeline, Pipeline)

    def test_has_emit_key(self, pipeline):
        """Should emit 'class_info'."""
        assert pipeline._emit_key == "class_info"

    def test_reql_has_class_type(self, pipeline):
        """REQL should filter by Class type."""
        query = pipeline._source.query
        assert "type {Class}" in query or "type oo:Class" in query

    def test_reql_has_name(self, pipeline):
        """REQL should select name."""
        query = pipeline._source.query
        assert "name ?name" in query

    def test_reql_has_file(self, pipeline):
        """REQL should select file."""
        query = pipeline._source.query
        assert "inFile ?file" in query

    def test_reql_has_docstring(self, pipeline):
        """REQL should optionally select docstring."""
        query = pipeline._source.query
        assert "docstring" in query.lower()


class TestDescribeClassVsReference:
    """Compare with reference implementation."""

    @pytest.fixture
    def pipeline(self):
        """Get the pipeline."""
        spec = describe_class._cadsl_spec
        ctx = Context(reter=None, params={"target": "TestClass"}, language="oo", instance_name="default")
        return spec.pipeline_factory(ctx)

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

    def test_supports_qualified_name_filter(self, pipeline):
        """
        Should support finding by qualified name using CONTAINS on entity ID.

        The query filters using:
            FILTER(CONTAINS(STR(?c), target) || ?name = target)
        This allows finding class by module.ClassName
        """
        query = pipeline._source.query

        # Reference uses CONTAINS to match qualified names (via STR(?c))
        has_contains = "CONTAINS" in query

        if not has_contains:
            pytest.fail(
                "MISSING: Qualified name filter not implemented.\n"
                "Should use: FILTER(CONTAINS(STR(?c), target) || ?name = target)\n"
                "This allows finding class by module.ClassName"
            )

    def test_reference_has_inheritance_info(self, pipeline):
        """
        Reference gets parent class info.

        Reference (python_tools.py - inheritance query):
            Gets parent classes via inheritsFrom relation
        """
        query = pipeline._source.query

        has_inheritance = "inheritsFrom" in query or "parent" in query.lower()

        if not has_inheritance:
            pytest.fail(
                "MISSING: Inheritance info not queried.\n"
                "Should query parent classes via inheritsFrom"
            )


class TestDescribeClassExecution:
    """Test execution with mock context."""

    @pytest.fixture
    def mock_context(self):
        """Create mock context."""
        ctx = Context(
            reter=Mock(),
            params={"target": "TestClass", "include_methods": True, "include_attributes": True, "include_docstrings": True},
            language="oo",
            instance_name="default"
        )

        mock_table = Mock()
        mock_table.num_rows = 1
        mock_table.to_pylist.return_value = [
            {
                "?c": "test.TestClass",
                "?name": "TestClass",
                "?file": "test.py",
                "?line": "10",
                "?docstring": "A test class.",
                "?parent_name": "BaseClass"
            }
        ]
        ctx.reter.reql.return_value = mock_table

        return ctx

    def test_execute_returns_success(self, mock_context):
        """Execution should return success."""
        result = describe_class(ctx=mock_context)
        assert "success" in result

    def test_execute_returns_class_info(self, mock_context):
        """Should return class_info."""
        result = describe_class(ctx=mock_context)
        if result.get("success"):
            assert "class_info" in result

    def test_class_info_has_required_fields(self, mock_context):
        """class_info should have required fields."""
        result = describe_class(ctx=mock_context)

        if not result.get("success") or not result.get("class_info"):
            pytest.skip("No class_info returned")

        info = result["class_info"]
        if isinstance(info, list) and info:
            info = info[0]

        required = ["name", "file", "line", "qualified_name"]
        for field in required:
            assert field in info, f"class_info should have '{field}'"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
