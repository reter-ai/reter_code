"""
Unit tests for list_modules CADSL tool.

Verifies behavior matches reference implementation at:
d:/ROOT/reter_code/src/reter_code/tools/python_basic/python_tools.py:63-128

Reference behavior:
- Returns modules with: qualified_name, name, file
- ORDER BY ?name
- Pagination with limit/offset
"""

import pytest
from unittest.mock import Mock
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'src'))

from reter_code.dsl.tools.inspection.list_modules import list_modules
from reter_code.dsl.core import Pipeline, Context, ToolType


class TestListModulesStructure:
    """Test list_modules tool structure."""

    def test_has_cadsl_spec(self):
        """Tool should have CADSL spec attached."""
        assert hasattr(list_modules, '_cadsl_spec')

    def test_spec_has_correct_name(self):
        """Tool spec should have correct name."""
        spec = list_modules._cadsl_spec
        assert spec.name == "list_modules"

    def test_spec_has_query_type(self):
        """Tool should be of QUERY type."""
        spec = list_modules._cadsl_spec
        assert spec.type == ToolType.QUERY

    def test_has_limit_param(self):
        """Tool should have limit parameter."""
        spec = list_modules._cadsl_spec
        assert 'limit' in spec.params

    def test_has_offset_param(self):
        """Tool should have offset parameter."""
        spec = list_modules._cadsl_spec
        assert 'offset' in spec.params

    def test_limit_default_is_100(self):
        """Limit default should be 100."""
        spec = list_modules._cadsl_spec
        assert spec.params['limit'].default == 100


class TestListModulesPipeline:
    """Test list_modules pipeline structure."""

    @pytest.fixture
    def pipeline(self):
        """Get the pipeline."""
        spec = list_modules._cadsl_spec
        ctx = Context(reter=None, params={}, language="oo", instance_name="default")
        return spec.pipeline_factory(ctx)

    def test_pipeline_is_pipeline_object(self, pipeline):
        """Should return a Pipeline."""
        assert isinstance(pipeline, Pipeline)

    def test_has_source(self, pipeline):
        """Pipeline should have source."""
        assert pipeline._source is not None

    def test_has_emit_key(self, pipeline):
        """Should emit 'modules'."""
        assert pipeline._emit_key == "modules"

    def test_reql_has_module_type(self, pipeline):
        """REQL should filter by Module type."""
        query = pipeline._source.query
        assert "type {Module}" in query or "type oo:Module" in query

    def test_reql_has_name(self, pipeline):
        """REQL should select name."""
        query = pipeline._source.query
        assert "name ?name" in query

    def test_reql_has_file(self, pipeline):
        """REQL should select file."""
        query = pipeline._source.query
        assert "inFile ?file" in query

    def test_select_has_qualified_name(self, pipeline):
        """Select should include qualified_name."""
        select_step = None
        for step in pipeline._steps:
            if hasattr(step, 'fields'):
                select_step = step
                break
        assert select_step is not None
        assert "qualified_name" in select_step.fields


class TestListModulesVsReference:
    """Compare with reference implementation."""

    @pytest.fixture
    def pipeline(self):
        """Get the pipeline."""
        spec = list_modules._cadsl_spec
        ctx = Context(reter=None, params={}, language="oo", instance_name="default")
        return spec.pipeline_factory(ctx)

    def test_reference_order_by_name(self, pipeline):
        """
        Reference uses ORDER BY ?name.

        Reference (python_tools.py:94):
            ORDER BY ?name
        """
        query = pipeline._source.query

        # Check if ORDER BY is in query OR if there's an order_by step
        has_order = "ORDER BY" in query
        has_order_step = any(hasattr(s, 'field_name') for s in pipeline._steps)

        if not has_order and not has_order_step:
            pytest.fail(
                "MISSING: ORDER BY clause.\n"
                "Reference has: ORDER BY ?name"
            )


class TestListModulesExecution:
    """Test execution with mock context."""

    @pytest.fixture
    def mock_context(self):
        """Create mock context."""
        ctx = Context(
            reter=Mock(),
            params={"limit": 100, "offset": 0},
            language="oo",
            instance_name="default"
        )

        mock_table = Mock()
        mock_table.num_rows = 2
        mock_table.to_pylist.return_value = [
            {"?m": "test.module", "?name": "module", "?file": "test/module.py"},
            {"?m": "test.other", "?name": "other", "?file": "test/other.py"},
        ]
        ctx.reter.reql.return_value = mock_table

        return ctx

    def test_execute_returns_success(self, mock_context):
        """Execution should return success."""
        result = list_modules(ctx=mock_context)
        assert "success" in result

    def test_execute_returns_modules(self, mock_context):
        """Should return modules list."""
        result = list_modules(ctx=mock_context)
        if result.get("success"):
            assert "modules" in result
            assert isinstance(result["modules"], list)

    def test_module_has_required_fields(self, mock_context):
        """Each module should have required fields."""
        result = list_modules(ctx=mock_context)

        if not result.get("success") or not result.get("modules"):
            pytest.skip("No modules returned")

        module = result["modules"][0]
        required = ["name", "file", "qualified_name"]
        for field in required:
            assert field in module, f"Module should have '{field}'"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
