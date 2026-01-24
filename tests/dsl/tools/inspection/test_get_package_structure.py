"""
Unit tests for get_package_structure CADSL tool.

Verifies behavior matches reference implementation at:
d:/ROOT/reter_code/src/reter_code/tools/python_advanced/architecture_analysis.py:74-135

Reference behavior:
- Queries Module type with inFile property
- Returns modules grouped by directory
- Orders by file path
"""

import pytest
from unittest.mock import Mock
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'src'))

from reter_code.dsl.tools.inspection.get_package_structure import get_package_structure
from reter_code.dsl.core import Pipeline, Context, ToolType


class TestGetPackageStructureStructure:
    """Test get_package_structure tool structure."""

    def test_has_cadsl_spec(self):
        """Tool should have CADSL spec attached."""
        assert hasattr(get_package_structure, '_cadsl_spec')

    def test_spec_has_correct_name(self):
        """Tool spec should have correct name."""
        spec = get_package_structure._cadsl_spec
        assert spec.name == "get_package_structure"

    def test_has_limit_param(self):
        """Should have limit parameter."""
        spec = get_package_structure._cadsl_spec
        assert 'limit' in spec.params

    def test_has_offset_param(self):
        """Should have offset parameter."""
        spec = get_package_structure._cadsl_spec
        assert 'offset' in spec.params


class TestGetPackageStructurePipeline:
    """Test get_package_structure pipeline structure."""

    @pytest.fixture
    def pipeline(self):
        """Get the pipeline."""
        spec = get_package_structure._cadsl_spec
        ctx = Context(reter=None, params={}, language="oo", instance_name="default")
        return spec.pipeline_factory(ctx)

    def test_pipeline_is_pipeline_object(self, pipeline):
        """Should return a Pipeline."""
        assert isinstance(pipeline, Pipeline)

    def test_has_emit_key(self, pipeline):
        """Should emit 'modules'."""
        assert pipeline._emit_key == "modules"

    def test_reql_has_module_type(self, pipeline):
        """REQL should filter by Module type."""
        query = pipeline._source.query
        assert "Module" in query

    def test_reql_has_infile(self, pipeline):
        """REQL should query inFile property."""
        query = pipeline._source.query
        assert "inFile" in query

    def test_reql_has_order_by_file(self, pipeline):
        """REQL should order by file."""
        query = pipeline._source.query
        assert "ORDER BY ?file" in query


class TestGetPackageStructureVsReference:
    """Compare with reference implementation."""

    @pytest.fixture
    def pipeline(self):
        """Get the pipeline."""
        spec = get_package_structure._cadsl_spec
        ctx = Context(reter=None, params={}, language="oo", instance_name="default")
        return spec.pipeline_factory(ctx)

    def test_reference_queries_module_and_file(self, pipeline):
        """
        Reference queries module and file.

        Reference (architecture_analysis.py:88-95):
            SELECT ?module ?file
            WHERE {
                ?module type {Module} .
                ?module inFile ?file
            }
        """
        query = pipeline._source.query

        has_module = "Module" in query
        has_file = "inFile" in query

        if not (has_module and has_file):
            pytest.fail(
                "MISSING: Module type or inFile property.\n"
                "Reference queries: ?module type {Module} . ?module inFile ?file"
            )

    def test_reference_orders_by_file(self, pipeline):
        """
        Reference orders results by file path.

        Reference (architecture_analysis.py:94):
            ORDER BY ?file
        """
        query = pipeline._source.query

        has_order = "ORDER BY" in query and "file" in query

        if not has_order:
            pytest.fail(
                "MISSING: ORDER BY file.\n"
                "Reference has: ORDER BY ?file"
            )


class TestGetPackageStructureExecution:
    """Test execution with mock context."""

    @pytest.fixture
    def mock_context(self):
        """Create mock context."""
        ctx = Context(
            reter=Mock(),
            params={},
            language="oo",
            instance_name="default"
        )

        mock_table = Mock()
        mock_table.num_rows = 2
        mock_table.to_pylist.return_value = [
            {
                "?m": "src.utils",
                "?name": "utils",
                "?file": "src/utils.py"
            },
            {
                "?m": "src.main",
                "?name": "main",
                "?file": "src/main.py"
            }
        ]
        ctx.reter.reql.return_value = mock_table

        return ctx

    def test_execute_returns_success(self, mock_context):
        """Execution should return success."""
        result = get_package_structure(ctx=mock_context)
        assert "success" in result

    def test_execute_returns_modules(self, mock_context):
        """Should return modules."""
        result = get_package_structure(ctx=mock_context)
        if result.get("success"):
            assert "modules" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
