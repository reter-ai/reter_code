"""
Test script for CADSL parser and validator.

Run with: python -m codeine.cadsl.test_parser
"""

import sys
from pathlib import Path

# Add src to path for development
src_path = Path(__file__).parent.parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from codeine.cadsl import (
    parse_cadsl,
    validate_cadsl,
    pretty_print_tree,
    get_tool_names,
    count_tools,
    Severity,
)


# ============================================================
# TEST CASES
# ============================================================

SIMPLE_QUERY = '''
query list_modules() {
    """List all modules in the codebase."""

    param limit: int = 100;

    reql {
        SELECT ?m ?name ?file
        WHERE {
            ?m type oo:Module .
            ?m name ?name .
            ?m inFile ?file
        }
        ORDER BY ?name
        LIMIT {limit}
    }
    | select { name, file }
    | emit { modules }
}
'''

DETECTOR_WITH_FILTER = '''
detector god_class(category="design", severity="high") {
    """Finds classes with too many methods."""

    param max_methods: int = 15;
    param limit: int = 100;

    reql {
        SELECT ?c ?name (COUNT(?m) AS ?method_count)
        WHERE {
            ?c type oo:Class .
            ?c name ?name .
            ?m type oo:Method .
            ?m definedIn ?c
        }
        GROUP BY ?c ?name
        ORDER BY DESC(?method_count)
        LIMIT {limit}
    }
    | filter { method_count > {max_methods} }
    | select { name, method_count }
    | map {
        ...row,
        issue: "god_class",
        message: "Class has too many methods"
    }
    | emit { findings }
}
'''

DETECTOR_WITH_PYTHON = '''
detector circular_imports(category="dependencies", severity="high", security="standard") {
    """Detect circular import dependencies."""

    reql {
        SELECT ?m1 ?m2 ?name1 ?name2
        WHERE {
            ?m1 type oo:Module . ?m1 name ?name1 .
            ?m2 type oo:Module . ?m2 name ?name2 .
            ?m1 imports ?m2
        }
    }
    | python {
        from collections import defaultdict

        graph = defaultdict(set)
        for row in rows:
            graph[row["name1"]].add(row["name2"])

        cycles = []
        result = [{"cycle": c, "issue": "circular_import"} for c in cycles]
    }
    | emit { findings }
}
'''

DIAGRAM_TOOL = '''
diagram class_hierarchy() {
    """Generate class inheritance hierarchy diagram."""

    param root_class: str = null;
    param format: str = "mermaid";

    reql {
        SELECT ?c ?className ?base ?baseName
        WHERE {
            ?c type oo:Class .
            ?c name ?className .
            OPTIONAL {
                ?c inheritsFrom ?base .
                ?base name ?baseName
            }
        }
        ORDER BY ?className
    }
    | select { className, baseName, c as class_iri }
    | render { format: {format}, renderer: hierarchy_renderer }
    | emit { diagram }
}
'''

MULTIPLE_TOOLS = '''
query list_classes() {
    param limit: int = 50;
    reql { SELECT ?c WHERE { ?c type oo:Class } LIMIT {limit} }
    | emit { classes }
}

detector empty_class(category="code_smell", severity="low") {
    reql { SELECT ?c WHERE { ?c type oo:Class } }
    | filter { method_count == 0 }
    | emit { findings }
}
'''

COMPLEX_CONDITIONS = '''
detector complex_filter(category="test", severity="medium") {
    param threshold: int = 10;

    reql { SELECT ?x ?name ?count WHERE { ?x type oo:Class } }
    | filter { count > {threshold} and not (name starts_with "_") }
    | filter { name matches "^[A-Z]" or count >= 100 }
    | filter { status in ["active", "pending"] }
    | emit { findings }
}
'''

INVALID_SYNTAX = '''
detector broken {
    reql { SELECT ?x }
    emit findings
}
'''

INVALID_TYPE = '''
query bad_types() {
    param count: integer = 10;
    reql { SELECT ?x WHERE { ?x type oo:Class } }
    | emit { results }
}
'''


# ============================================================
# TEST RUNNER
# ============================================================

def run_test_case(name: str, source: str, expect_success: bool = True) -> bool:
    """Run a single test case (helper function, not a pytest test)."""
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"{'='*60}")

    # Parse
    result = parse_cadsl(source)

    if not result.success:
        print(f"Parse {'FAILED' if expect_success else 'FAILED (expected)'}:")
        for error in result.errors:
            print(f"  {error}")
        return not expect_success

    print("Parse: SUCCESS")
    print(f"Tools found: {get_tool_names(result.tree)}")
    print(f"Tool counts: {count_tools(result.tree)}")

    # Validate
    validation = validate_cadsl(result.tree)

    if validation.errors:
        print(f"\nValidation errors:")
        for issue in validation.errors:
            print(f"  {issue}")

    if validation.warnings:
        print(f"\nValidation warnings:")
        for issue in validation.warnings:
            print(f"  {issue}")

    if validation.valid:
        print("\nValidation: PASSED")
        print(f"Tool info: {validation.tool_info}")
    else:
        print(f"\nValidation: {'FAILED' if expect_success else 'FAILED (expected)'}")

    # Print tree for debugging (first 30 lines)
    if result.tree and expect_success:
        print(f"\nParse tree (truncated):")
        tree_str = pretty_print_tree(result.tree)
        lines = tree_str.split('\n')[:30]
        for line in lines:
            print(f"  {line}")
        if len(tree_str.split('\n')) > 30:
            print("  ... (truncated)")

    success = result.success == expect_success
    if expect_success:
        success = success and validation.valid

    return success


def run_all_tests() -> int:
    """Run all test cases."""
    print("CADSL Parser Test Suite")
    print("=" * 60)

    tests = [
        ("Simple Query", SIMPLE_QUERY, True),
        ("Detector with Filter", DETECTOR_WITH_FILTER, True),
        ("Detector with Python", DETECTOR_WITH_PYTHON, True),
        ("Diagram Tool", DIAGRAM_TOOL, True),
        ("Multiple Tools", MULTIPLE_TOOLS, True),
        ("Complex Conditions", COMPLEX_CONDITIONS, True),
        ("Invalid Syntax", INVALID_SYNTAX, False),
        ("Invalid Type", INVALID_TYPE, False),
    ]

    passed = 0
    failed = 0

    for name, source, expect_success in tests:
        try:
            if run_test_case(name, source, expect_success):
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"\nEXCEPTION: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*60}")
    print(f"RESULTS: {passed} passed, {failed} failed")
    print(f"{'='*60}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
