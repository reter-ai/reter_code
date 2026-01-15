"""
Test script for CADSL Loader.

Run with: python -m codeine.cadsl.test_loader
"""

import sys
import tempfile
from pathlib import Path

# Add src to path for development
src_path = Path(__file__).parent.parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from codeine.cadsl.loader import (
    LoadResult,
    load_tool,
    load_tool_file,
    load_tools_directory,
    load_cadsl,
    load_cadsl_file,
    load_cadsl_directory,
)
from codeine.cadsl.python_executor import SecurityLevel


# ============================================================
# TEST CASES
# ============================================================

def test_load_simple_query():
    """Test loading a simple query from string."""
    print("\n" + "=" * 60)
    print("TEST: Load Simple Query")
    print("=" * 60)

    source = '''
    query list_items() {
        """List all items."""

        param limit: int = 100;

        reql {
            SELECT ?x ?name WHERE { ?x type oo:Item . ?x name ?name }
        }
        | select { name }
        | limit { {limit} }
        | emit { items }
    }
    '''

    result = load_tool(source, register=False)
    print(f"  Success: {result.success}")
    print(f"  Tools loaded: {result.tools_loaded}")
    print(f"  Tool names: {result.tool_names}")
    print(f"  Errors: {result.errors}")
    print(f"  Warnings: {result.warnings}")

    if result.success and result.tools_loaded == 1 and "list_items" in result.tool_names:
        print("Load simple query: PASSED")
        return True
    else:
        print("Load simple query: FAILED")
        return False


def test_load_detector_with_python():
    """Test loading a detector with Python block."""
    print("\n" + "=" * 60)
    print("TEST: Load Detector with Python Block")
    print("=" * 60)

    source = '''
    detector find_issues(category="code_smell", severity="medium") {
        """Find code issues."""

        param threshold: int = 10;

        reql {
            SELECT ?c ?name WHERE { ?c type oo:Class . ?c name ?name }
        }
        | python {
            # Process results
            result = [
                {**row, "issue": "detected", "count": len(row["name"])}
                for row in rows
            ]
        }
        | filter { count > {threshold} }
        | emit { findings }
    }
    '''

    result = load_tool(source, register=False, security_level=SecurityLevel.STANDARD)
    print(f"  Success: {result.success}")
    print(f"  Tools loaded: {result.tools_loaded}")
    print(f"  Tool names: {result.tool_names}")

    if result.success and "find_issues" in result.tool_names:
        print("Load detector with Python: PASSED")
        return True
    else:
        print(f"Load detector with Python: FAILED - {result.errors}")
        return False


def test_load_multiple_tools():
    """Test loading multiple tools from one source."""
    print("\n" + "=" * 60)
    print("TEST: Load Multiple Tools")
    print("=" * 60)

    source = '''
    query get_modules() {
        """Get all modules."""
        reql { SELECT ?m WHERE { ?m type oo:Module } }
        | emit { modules }
    }

    query get_classes() {
        """Get all classes."""
        reql { SELECT ?c WHERE { ?c type oo:Class } }
        | emit { classes }
    }

    detector find_god_classes(category="design", severity="high") {
        """Find god classes."""
        param max_methods: int = 15;
        reql { SELECT ?c WHERE { ?c type oo:Class } }
        | filter { method_count > {max_methods} }
        | emit { findings }
    }
    '''

    result = load_tool(source, register=False)
    print(f"  Success: {result.success}")
    print(f"  Tools loaded: {result.tools_loaded}")
    print(f"  Tool names: {result.tool_names}")

    if result.success and result.tools_loaded == 3:
        expected = {"get_modules", "get_classes", "find_god_classes"}
        if set(result.tool_names) == expected:
            print("Load multiple tools: PASSED")
            return True

    print("Load multiple tools: FAILED")
    return False


def test_load_from_file():
    """Test loading tools from a .cadsl file."""
    print("\n" + "=" * 60)
    print("TEST: Load From File")
    print("=" * 60)

    source = '''
    query file_test() {
        """Tool loaded from file."""
        reql { SELECT ?x WHERE { ?x type oo:Test } }
        | emit { results }
    }
    '''

    # Create temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.cadsl', delete=False) as f:
        f.write(source)
        temp_path = f.name

    try:
        result = load_tool_file(temp_path, register=False)
        print(f"  Success: {result.success}")
        print(f"  Tools loaded: {result.tools_loaded}")
        print(f"  Tool names: {result.tool_names}")
        print(f"  Source: {result.source}")

        if result.success and "file_test" in result.tool_names:
            print("Load from file: PASSED")
            return True
        else:
            print(f"Load from file: FAILED - {result.errors}")
            return False
    finally:
        Path(temp_path).unlink(missing_ok=True)


def test_load_from_directory():
    """Test loading tools from a directory."""
    print("\n" + "=" * 60)
    print("TEST: Load From Directory")
    print("=" * 60)

    tool1 = '''
    query dir_tool1() {
        """First tool in directory."""
        reql { SELECT ?x WHERE { ?x type oo:A } }
        | emit { results }
    }
    '''

    tool2 = '''
    query dir_tool2() {
        """Second tool in directory."""
        reql { SELECT ?x WHERE { ?x type oo:B } }
        | emit { results }
    }

    detector dir_detector() {
        """Detector in directory."""
        reql { SELECT ?x WHERE { ?x type oo:C } }
        | emit { findings }
    }
    '''

    # Create temp directory with files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Write files
        (temp_path / "tool1.cadsl").write_text(tool1)
        (temp_path / "tool2.cadsl").write_text(tool2)
        (temp_path / "not_cadsl.txt").write_text("ignored")

        result = load_tools_directory(temp_path, register=False)
        print(f"  Success: {result.success}")
        print(f"  Tools loaded: {result.tools_loaded}")
        print(f"  Tool names: {result.tool_names}")
        print(f"  Errors: {result.errors}")

        if result.success and result.tools_loaded == 3:
            expected = {"dir_tool1", "dir_tool2", "dir_detector"}
            if set(result.tool_names) == expected:
                print("Load from directory: PASSED")
                return True

    print("Load from directory: FAILED")
    return False


def test_load_directory_recursive():
    """Test loading tools from directory recursively."""
    print("\n" + "=" * 60)
    print("TEST: Load From Directory (Recursive)")
    print("=" * 60)

    tool1 = '''
    query top_level() {
        """Top level tool."""
        reql { SELECT ?x WHERE { ?x type oo:A } }
        | emit { results }
    }
    '''

    tool2 = '''
    query nested() {
        """Nested tool."""
        reql { SELECT ?x WHERE { ?x type oo:B } }
        | emit { results }
    }
    '''

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create subdirectory
        subdir = temp_path / "subdir"
        subdir.mkdir()

        # Write files
        (temp_path / "top.cadsl").write_text(tool1)
        (subdir / "nested.cadsl").write_text(tool2)

        # Non-recursive should only find top level
        result_flat = load_tools_directory(temp_path, recursive=False, register=False)
        print(f"  Non-recursive: {result_flat.tools_loaded} tools")

        # Recursive should find both
        result_recursive = load_tools_directory(temp_path, recursive=True, register=False)
        print(f"  Recursive: {result_recursive.tools_loaded} tools")
        print(f"  Tool names: {result_recursive.tool_names}")

        if result_flat.tools_loaded == 1 and result_recursive.tools_loaded == 2:
            print("Load directory recursive: PASSED")
            return True

    print("Load directory recursive: FAILED")
    return False


def test_file_not_found():
    """Test error handling for missing file."""
    print("\n" + "=" * 60)
    print("TEST: File Not Found Error")
    print("=" * 60)

    result = load_tool_file("/nonexistent/path/tool.cadsl", register=False)
    print(f"  Success: {result.success}")
    print(f"  Errors: {result.errors}")

    if not result.success and any("not found" in e.lower() for e in result.errors):
        print("File not found error: PASSED")
        return True
    else:
        print("File not found error: FAILED")
        return False


def test_parse_error():
    """Test error handling for parse errors."""
    print("\n" + "=" * 60)
    print("TEST: Parse Error Handling")
    print("=" * 60)

    source = '''
    query broken( {
        # Missing closing paren and body
        reql
    '''

    result = load_tool(source, register=False)
    print(f"  Success: {result.success}")
    print(f"  Errors: {result.errors[:2] if result.errors else []}")  # Show first 2

    if not result.success and len(result.errors) > 0:
        print("Parse error handling: PASSED")
        return True
    else:
        print("Parse error handling: FAILED")
        return False


def test_validation_error():
    """Test error handling for validation errors."""
    print("\n" + "=" * 60)
    print("TEST: Validation Error Handling")
    print("=" * 60)

    source = '''
    query invalid_tool() {
        """Tool with invalid param type."""
        param x: unknown_type = 10;

        reql { SELECT ?x WHERE { ?x type oo:A } }
        | emit { results }
    }
    '''

    result = load_tool(source, register=False)
    print(f"  Success: {result.success}")
    print(f"  Errors: {result.errors}")

    # Should fail due to invalid type
    if not result.success:
        print("Validation error handling: PASSED")
        return True
    else:
        print("Validation error handling: FAILED")
        return False


def test_empty_source():
    """Test handling of empty source."""
    print("\n" + "=" * 60)
    print("TEST: Empty Source")
    print("=" * 60)

    result = load_tool("", register=False)
    print(f"  Success: {result.success}")
    print(f"  Tools loaded: {result.tools_loaded}")

    # Empty source should fail (no tools found)
    if not result.success:
        print("Empty source: PASSED")
        return True
    else:
        print("Empty source: FAILED")
        return False


def test_convenience_functions():
    """Test convenience function aliases."""
    print("\n" + "=" * 60)
    print("TEST: Convenience Functions")
    print("=" * 60)

    source = '''
    query test_alias() {
        """Test alias."""
        reql { SELECT ?x WHERE { ?x type oo:A } }
        | emit { results }
    }
    '''

    # Test load_cadsl (alias for load_tool)
    result1 = load_cadsl(source)
    print(f"  load_cadsl: {result1.success}")

    # Create temp file for load_cadsl_file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.cadsl', delete=False) as f:
        f.write(source)
        temp_path = f.name

    try:
        result2 = load_cadsl_file(temp_path)
        print(f"  load_cadsl_file: {result2.success}")
    finally:
        Path(temp_path).unlink(missing_ok=True)

    # Test load_cadsl_directory
    with tempfile.TemporaryDirectory() as temp_dir:
        (Path(temp_dir) / "test.cadsl").write_text(source)
        result3 = load_cadsl_directory(temp_dir)
        print(f"  load_cadsl_directory: {result3.success}")

    if result1.success and result2.success and result3.success:
        print("Convenience functions: PASSED")
        return True
    else:
        print("Convenience functions: FAILED")
        return False


def test_load_result_bool():
    """Test LoadResult bool conversion."""
    print("\n" + "=" * 60)
    print("TEST: LoadResult Bool")
    print("=" * 60)

    source = '''
    query test() {
        reql { SELECT ?x WHERE { ?x type oo:A } }
        | emit { results }
    }
    '''

    result = load_tool(source, register=False)

    # Test bool conversion
    if result:
        print("  Result is truthy: True")
    else:
        print("  Result is truthy: False")

    # Test in if statement
    if result and result.tools_loaded == 1:
        print("LoadResult bool: PASSED")
        return True
    else:
        print("LoadResult bool: FAILED")
        return False


# ============================================================
# TEST RUNNER
# ============================================================

def run_all_tests() -> int:
    """Run all test cases."""
    print("CADSL Loader Test Suite")
    print("=" * 60)

    tests = [
        test_load_simple_query,
        test_load_detector_with_python,
        test_load_multiple_tools,
        test_load_from_file,
        test_load_from_directory,
        test_load_directory_recursive,
        test_file_not_found,
        test_parse_error,
        test_validation_error,
        test_empty_source,
        test_convenience_functions,
        test_load_result_bool,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"\nEXCEPTION: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"RESULTS: {passed} passed, {failed} failed")
    print(f"{'=' * 60}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
