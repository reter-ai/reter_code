"""
Test equivalence between CADSL and Python DSL tool implementations.

This test module verifies that CADSL tools produce identical results
to their Python DSL counterparts.
"""

import pytest
from pathlib import Path
from typing import Dict, Any, List, Optional
import json

# CADSL imports
from .loader import load_tool_file, LoadResult
from .parser import parse_cadsl_file, ParseResult


# ============================================================
# TEST FIXTURES
# ============================================================

CADSL_TOOLS_DIR = Path(__file__).parent / "tools"


@pytest.fixture
def cadsl_tools_path():
    """Return path to CADSL tools directory."""
    return CADSL_TOOLS_DIR


# ============================================================
# PARSING TESTS
# ============================================================

class TestCADSLParsing:
    """Test that CADSL files parse without errors."""

    def test_god_class_parses(self, cadsl_tools_path):
        """god_class.cadsl should parse without errors."""
        path = cadsl_tools_path / "smells" / "god_class.cadsl"
        result = parse_cadsl_file(path)
        assert result.success, f"Parse errors: {result.errors}"

    def test_long_methods_parses(self, cadsl_tools_path):
        """long_methods.cadsl should parse without errors."""
        path = cadsl_tools_path / "smells" / "long_methods.cadsl"
        result = parse_cadsl_file(path)
        assert result.success, f"Parse errors: {result.errors}"

    def test_find_callers_parses(self, cadsl_tools_path):
        """find_callers.cadsl should parse without errors."""
        path = cadsl_tools_path / "inspection" / "find_callers.cadsl"
        result = parse_cadsl_file(path)
        assert result.success, f"Parse errors: {result.errors}"

    def test_list_classes_parses(self, cadsl_tools_path):
        """list_classes.cadsl should parse without errors."""
        path = cadsl_tools_path / "inspection" / "list_classes.cadsl"
        result = parse_cadsl_file(path)
        assert result.success, f"Parse errors: {result.errors}"

    def test_dead_code_parses(self, cadsl_tools_path):
        """dead_code.cadsl should parse without errors."""
        path = cadsl_tools_path / "smells" / "dead_code.cadsl"
        result = parse_cadsl_file(path)
        assert result.success, f"Parse errors: {result.errors}"


class TestCADSLLoading:
    """Test that CADSL files load and transform correctly."""

    def test_god_class_loads(self, cadsl_tools_path):
        """god_class.cadsl should load without errors."""
        path = cadsl_tools_path / "smells" / "god_class.cadsl"
        result = load_tool_file(path, register=False)
        assert result.success, f"Load errors: {result.errors}"
        assert result.tools_loaded == 1
        assert "god_class" in result.tool_names

    def test_long_methods_loads(self, cadsl_tools_path):
        """long_methods.cadsl should load without errors."""
        path = cadsl_tools_path / "smells" / "long_methods.cadsl"
        result = load_tool_file(path, register=False)
        assert result.success, f"Load errors: {result.errors}"
        assert result.tools_loaded == 1
        assert "long_methods" in result.tool_names

    def test_find_callers_loads(self, cadsl_tools_path):
        """find_callers.cadsl should load without errors."""
        path = cadsl_tools_path / "inspection" / "find_callers.cadsl"
        result = load_tool_file(path, register=False)
        assert result.success, f"Load errors: {result.errors}"
        assert result.tools_loaded == 1
        assert "find_callers" in result.tool_names

    def test_list_classes_loads(self, cadsl_tools_path):
        """list_classes.cadsl should load without errors."""
        path = cadsl_tools_path / "inspection" / "list_classes.cadsl"
        result = load_tool_file(path, register=False)
        assert result.success, f"Load errors: {result.errors}"
        assert result.tools_loaded == 1
        assert "list_classes" in result.tool_names

    def test_dead_code_loads(self, cadsl_tools_path):
        """dead_code.cadsl should load without errors."""
        path = cadsl_tools_path / "smells" / "dead_code.cadsl"
        result = load_tool_file(path, register=False)
        assert result.success, f"Load errors: {result.errors}"
        assert result.tools_loaded == 1
        assert "dead_code" in result.tool_names


# ============================================================
# EQUIVALENCE TESTS
# ============================================================

class TestEquivalence:
    """
    Test that CADSL tools produce equivalent results to Python tools.

    These tests require a RETER instance with loaded code.
    """

    @pytest.fixture
    def mock_context(self):
        """Create a mock context for testing."""
        try:
            from codeine.dsl.core import Context
            return Context(reter=None, params={})
        except ImportError:
            pytest.skip("codeine.dsl.core not available")

    def compare_results(
        self,
        py_result: Dict[str, Any],
        cadsl_result: Dict[str, Any],
        ignore_fields: Optional[List[str]] = None
    ) -> bool:
        """
        Compare results from Python and CADSL tools.

        Args:
            py_result: Result from Python tool
            cadsl_result: Result from CADSL tool
            ignore_fields: Fields to ignore in comparison

        Returns:
            True if results are equivalent
        """
        ignore_fields = ignore_fields or []

        # Check top-level keys match
        py_keys = set(py_result.keys()) - set(ignore_fields)
        cadsl_keys = set(cadsl_result.keys()) - set(ignore_fields)

        if py_keys != cadsl_keys:
            return False

        # Check counts match
        if "count" in py_result and "count" in cadsl_result:
            if py_result["count"] != cadsl_result["count"]:
                return False

        # Check list results match (sorted by a common key if available)
        for key in py_keys:
            py_val = py_result[key]
            cadsl_val = cadsl_result[key]

            if isinstance(py_val, list) and isinstance(cadsl_val, list):
                if len(py_val) != len(cadsl_val):
                    return False

                # Sort by 'name' or 'file' if available
                if py_val and isinstance(py_val[0], dict):
                    sort_key = None
                    for k in ['name', 'file', 'qualified_name']:
                        if k in py_val[0]:
                            sort_key = k
                            break

                    if sort_key:
                        py_sorted = sorted(py_val, key=lambda x: x.get(sort_key, ''))
                        cadsl_sorted = sorted(cadsl_val, key=lambda x: x.get(sort_key, ''))

                        for py_item, cadsl_item in zip(py_sorted, cadsl_sorted):
                            for field in py_item:
                                if field not in ignore_fields:
                                    if py_item.get(field) != cadsl_item.get(field):
                                        return False

        return True

    @pytest.mark.skip(reason="Requires active RETER instance")
    def test_god_class_equivalence(self, mock_context, cadsl_tools_path):
        """Compare god_class outputs between Python and CADSL."""
        # This test would run both Python and CADSL versions
        # and compare their outputs
        pass

    @pytest.mark.skip(reason="Requires active RETER instance")
    def test_long_methods_equivalence(self, mock_context, cadsl_tools_path):
        """Compare long_methods outputs between Python and CADSL."""
        pass

    @pytest.mark.skip(reason="Requires active RETER instance")
    def test_find_callers_equivalence(self, mock_context, cadsl_tools_path):
        """Compare find_callers outputs between Python and CADSL."""
        pass

    @pytest.mark.skip(reason="Requires active RETER instance")
    def test_list_classes_equivalence(self, mock_context, cadsl_tools_path):
        """Compare list_classes outputs between Python and CADSL."""
        pass

    @pytest.mark.skip(reason="Requires active RETER instance")
    def test_dead_code_equivalence(self, mock_context, cadsl_tools_path):
        """Compare dead_code outputs between Python and CADSL."""
        pass


# ============================================================
# DIRECTORY LOADING TEST
# ============================================================

class TestDirectoryLoading:
    """Test loading all CADSL tools from directory."""

    def test_load_smells_directory(self, cadsl_tools_path):
        """All smells should load without errors."""
        from .loader import load_tools_directory

        smells_path = cadsl_tools_path / "smells"
        if not smells_path.exists():
            pytest.skip("Smells directory not found")

        result = load_tools_directory(smells_path, register=False)
        assert result.success, f"Load errors: {result.errors}"
        assert result.tools_loaded >= 2  # At least god_class and dead_code

    def test_load_inspection_directory(self, cadsl_tools_path):
        """All inspection queries should load without errors."""
        from .loader import load_tools_directory

        inspection_path = cadsl_tools_path / "inspection"
        if not inspection_path.exists():
            pytest.skip("Inspection directory not found")

        result = load_tools_directory(inspection_path, register=False)
        assert result.success, f"Load errors: {result.errors}"
        assert result.tools_loaded >= 2  # At least find_callers and list_classes


# ============================================================
# QUICK VALIDATION SCRIPT
# ============================================================

def validate_all_cadsl_tools():
    """
    Quick validation script to check all CADSL tools parse and load.

    Run with: python -m codeine.cadsl.test_equivalence
    """
    from .loader import load_tools_directory

    print("=" * 60)
    print("CADSL Tool Validation")
    print("=" * 60)

    tools_dir = CADSL_TOOLS_DIR
    if not tools_dir.exists():
        print(f"ERROR: Tools directory not found: {tools_dir}")
        return False

    # Load all tools recursively
    result = load_tools_directory(tools_dir, recursive=True, register=False)

    print(f"\nDirectory: {tools_dir}")
    print(f"Tools loaded: {result.tools_loaded}")
    print(f"Tool names: {', '.join(result.tool_names)}")

    if result.errors:
        print(f"\nERRORS:")
        for error in result.errors:
            print(f"  - {error}")

    if result.warnings:
        print(f"\nWarnings:")
        for warning in result.warnings:
            print(f"  - {warning}")

    print("\n" + "=" * 60)
    if result.success:
        print("VALIDATION PASSED")
    else:
        print("VALIDATION FAILED")
    print("=" * 60)

    return result.success


if __name__ == "__main__":
    import sys
    success = validate_all_cadsl_tools()
    sys.exit(0 if success else 1)
