"""
Test script for CADSL transformer.

Run with: python -m reter_code.cadsl.test_transformer
"""

import sys
from pathlib import Path

# Add src to path for development
src_path = Path(__file__).parent.parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from reter_code.cadsl import (
    parse_cadsl,
    validate_cadsl,
    transform_cadsl,
    ToolSpec,
    compile_condition,
    compile_expression,
    compile_object_expr,
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
detector circular_imports(category="dependencies", severity="high") {
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


# ============================================================
# COMPILER TESTS
# ============================================================

def test_expression_compiler():
    """Test expression compilation."""
    print("\n" + "=" * 60)
    print("TEST: Expression Compiler")
    print("=" * 60)

    # Test field reference
    result = parse_cadsl('''
    query test() {
        reql { SELECT ?x WHERE { ?x type oo:Class } }
        | filter { count > 10 }
        | emit { results }
    }
    ''')

    if not result.success:
        print(f"Parse failed: {result.errors}")
        return False

    # Get the filter condition
    tree = result.tree
    tool_def = tree.children[0]
    tool_body = None
    for child in tool_def.children:
        if hasattr(child, 'data') and child.data == "tool_body":
            tool_body = child
            break

    if tool_body:
        pipeline = None
        for child in tool_body.children:
            if hasattr(child, 'data') and child.data == "pipeline":
                pipeline = child
                break

        if pipeline:
            # Find filter step
            for child in pipeline.children:
                if hasattr(child, 'data') and child.data == "step":
                    for step_child in child.children:
                        if hasattr(step_child, 'data') and step_child.data == "filter_step":
                            # Get condition
                            for cond in step_child.children:
                                if hasattr(cond, 'data'):
                                    predicate = compile_condition(cond)

                                    # Test predicate
                                    row1 = {"count": 15}
                                    row2 = {"count": 5}

                                    r1 = predicate(row1)
                                    r2 = predicate(row2)

                                    print(f"  count=15 > 10: {r1} (expected True)")
                                    print(f"  count=5 > 10: {r2} (expected False)")

                                    if r1 == True and r2 == False:
                                        print("Expression compiler: PASSED")
                                        return True
                                    else:
                                        print("Expression compiler: FAILED")
                                        return False

    print("Could not find filter condition")
    return False


def test_object_expression():
    """Test object expression compilation."""
    print("\n" + "=" * 60)
    print("TEST: Object Expression Compiler")
    print("=" * 60)

    result = parse_cadsl('''
    query test() {
        reql { SELECT ?x WHERE { ?x type oo:Class } }
        | map { name: name, extra: "value", ...row }
        | emit { results }
    }
    ''')

    if not result.success:
        print(f"Parse failed: {result.errors}")
        return False

    # Find the map step object expression
    tree = result.tree
    tool_def = tree.children[0]

    for child in tool_def.children:
        if hasattr(child, 'data') and child.data == "tool_body":
            for body_child in child.children:
                if hasattr(body_child, 'data') and body_child.data == "pipeline":
                    for step in body_child.children:
                        if hasattr(step, 'data') and step.data == "step":
                            for step_child in step.children:
                                if hasattr(step_child, 'data') and step_child.data == "map_step":
                                    for obj in step_child.children:
                                        if hasattr(obj, 'data') and obj.data == "object_expr":
                                            transform = compile_object_expr(obj)

                                            # Test transform
                                            row = {"name": "TestClass", "file": "test.py"}
                                            result = transform(row)

                                            print(f"  Input: {row}")
                                            print(f"  Output: {result}")

                                            # Check result has spread fields + new fields
                                            if result.get("name") == "TestClass" and result.get("extra") == "value":
                                                print("Object expression compiler: PASSED")
                                                return True

    print("Object expression compiler: FAILED")
    return False


# ============================================================
# TRANSFORMER TESTS
# ============================================================

def test_transform_simple_query():
    """Test transforming a simple query."""
    print("\n" + "=" * 60)
    print("TEST: Transform Simple Query")
    print("=" * 60)

    result = parse_cadsl(SIMPLE_QUERY)
    if not result.success:
        print(f"Parse failed: {result.errors}")
        return False

    tools = transform_cadsl(result.tree)

    if len(tools) != 1:
        print(f"Expected 1 tool, got {len(tools)}")
        return False

    tool = tools[0]
    print(f"  Name: {tool.name}")
    print(f"  Type: {tool.tool_type}")
    print(f"  Description: {tool.description[:50]}...")
    print(f"  Params: {[p.name for p in tool.params]}")
    print(f"  Source type: {tool.source_type}")
    print(f"  Steps: {len(tool.steps)}")
    print(f"  Emit key: {tool.steps[-1].get('key') if tool.steps else None}")

    # Verify structure
    checks = [
        tool.name == "list_modules",
        tool.tool_type == "query",
        len(tool.params) == 1,
        tool.params[0].name == "limit",
        tool.params[0].type == "int",
        tool.params[0].default == 100,
        tool.source_type == "reql",
        "SELECT ?m ?name ?file" in tool.source_content,
        len(tool.steps) == 2,  # select, emit
        tool.steps[-1].get("key") == "modules",
    ]

    if all(checks):
        print("Transform simple query: PASSED")
        return True
    else:
        print(f"Transform simple query: FAILED (checks: {checks})")
        return False


def test_transform_detector():
    """Test transforming a detector with filter."""
    print("\n" + "=" * 60)
    print("TEST: Transform Detector")
    print("=" * 60)

    result = parse_cadsl(DETECTOR_WITH_FILTER)
    if not result.success:
        print(f"Parse failed: {result.errors}")
        return False

    tools = transform_cadsl(result.tree)

    if len(tools) != 1:
        print(f"Expected 1 tool, got {len(tools)}")
        return False

    tool = tools[0]
    print(f"  Name: {tool.name}")
    print(f"  Type: {tool.tool_type}")
    print(f"  Metadata: {tool.metadata}")
    print(f"  Params: {[(p.name, p.type, p.default) for p in tool.params]}")
    print(f"  Steps: {[s.get('type') for s in tool.steps]}")

    # Verify structure
    checks = [
        tool.name == "god_class",
        tool.tool_type == "detector",
        tool.metadata.get("category") == "design",
        tool.metadata.get("severity") == "high",
        len(tool.params) == 2,
        tool.source_type == "reql",
    ]

    # Check steps
    step_types = [s.get("type") for s in tool.steps]
    expected_steps = ["filter", "select", "map", "emit"]
    checks.append(step_types == expected_steps)

    # Check filter predicate
    filter_step = tool.steps[0]
    if filter_step.get("type") == "filter" and "predicate" in filter_step:
        predicate = filter_step["predicate"]
        # Test with mock context
        class MockCtx:
            params = {"max_methods": 15}

        # Test predicate with sample data
        test_row = {"method_count": 20}
        try:
            result = predicate(test_row, MockCtx())
            print(f"  Filter predicate (method_count=20 > 15): {result}")
            checks.append(result == True)
        except Exception as e:
            print(f"  Filter predicate error: {e}")
            checks.append(False)

    if all(checks):
        print("Transform detector: PASSED")
        return True
    else:
        print(f"Transform detector: FAILED (checks: {checks})")
        return False


def test_transform_python_step():
    """Test transforming a tool with Python step."""
    print("\n" + "=" * 60)
    print("TEST: Transform Python Step")
    print("=" * 60)

    result = parse_cadsl(DETECTOR_WITH_PYTHON)
    if not result.success:
        print(f"Parse failed: {result.errors}")
        return False

    tools = transform_cadsl(result.tree)

    if len(tools) != 1:
        print(f"Expected 1 tool, got {len(tools)}")
        return False

    tool = tools[0]
    print(f"  Name: {tool.name}")
    print(f"  Steps: {[s.get('type') for s in tool.steps]}")

    # Find python step
    python_step = None
    for step in tool.steps:
        if step.get("type") == "python":
            python_step = step
            break

    if python_step is None:
        print("Python step not found")
        return False

    code = python_step.get("code", "")
    print(f"  Python code length: {len(code)} chars")
    print(f"  Code preview: {code[:100]}...")

    # Verify code contains expected elements
    checks = [
        "from collections import defaultdict" in code,
        "graph = defaultdict(set)" in code,
        "result = " in code,
    ]

    if all(checks):
        print("Transform Python step: PASSED")
        return True
    else:
        print(f"Transform Python step: FAILED (checks: {checks})")
        return False


# ============================================================
# INTEGRATION TEST
# ============================================================

def test_full_pipeline():
    """Test full parse -> validate -> transform flow."""
    print("\n" + "=" * 60)
    print("TEST: Full Pipeline Integration")
    print("=" * 60)

    sources = [
        ("Simple Query", SIMPLE_QUERY),
        ("Detector with Filter", DETECTOR_WITH_FILTER),
        ("Detector with Python", DETECTOR_WITH_PYTHON),
    ]

    passed = 0
    for name, source in sources:
        print(f"\n  Testing: {name}")

        # Parse
        result = parse_cadsl(source)
        if not result.success:
            print(f"    Parse FAILED: {result.errors[0]}")
            continue

        # Validate
        validation = validate_cadsl(result.tree)
        if not validation.valid:
            print(f"    Validation FAILED: {validation.errors[0]}")
            continue

        # Transform
        try:
            tools = transform_cadsl(result.tree)
            if tools:
                print(f"    Transform OK: {len(tools)} tool(s)")
                for t in tools:
                    print(f"      - {t.name} ({t.tool_type}): {len(t.steps)} steps")
                passed += 1
            else:
                print("    Transform FAILED: no tools")
        except Exception as e:
            print(f"    Transform ERROR: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n  Integration: {passed}/{len(sources)} passed")
    return passed == len(sources)


# ============================================================
# TEST RUNNER
# ============================================================

def run_all_tests() -> int:
    """Run all test cases."""
    print("CADSL Transformer Test Suite")
    print("=" * 60)

    tests = [
        test_expression_compiler,
        test_object_expression,
        test_transform_simple_query,
        test_transform_detector,
        test_transform_python_step,
        test_full_pipeline,
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
