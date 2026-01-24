"""
Test script for CADSL Python security sandbox.

Run with: python -m reter_code.cadsl.test_security
"""

import sys
from pathlib import Path

# Add src to path for development
src_path = Path(__file__).parent.parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from reter_code.cadsl.python_executor import (
    PythonExecutor,
    SecurityContext,
    SecurityLevel,
    ASTValidator,
    validate_python_code,
    execute_python_safely,
    validate_imports,
)
from reter_code.cadsl.builtins import CADSL_BUILTINS


# ============================================================
# TEST CASES
# ============================================================

def test_safe_code_execution():
    """Test that safe code executes correctly."""
    print("\n" + "=" * 60)
    print("TEST: Safe Code Execution")
    print("=" * 60)

    code = """
# Simple data transformation
data = [{"name": "foo", "count": 10}, {"name": "bar", "count": 20}]
filtered = [x for x in data if x["count"] > 15]
result = filtered
"""

    result = execute_python_safely(code)
    print(f"  Success: {result.success}")
    print(f"  Result: {result.result}")
    print(f"  Time: {result.execution_time_ms:.2f}ms")

    if result.success and len(result.result) == 1:
        print("Safe code execution: PASSED")
        return True
    else:
        print("Safe code execution: FAILED")
        return False


def test_blocked_imports_restricted():
    """Test that imports are blocked in restricted mode."""
    print("\n" + "=" * 60)
    print("TEST: Blocked Imports (Restricted)")
    print("=" * 60)

    code = """
import os
result = os.getcwd()
"""

    result = execute_python_safely(code, security_level=SecurityLevel.RESTRICTED)
    print(f"  Success: {result.success}")
    print(f"  Error: {result.error}")

    # In restricted mode, Import is a blocked construct
    if not result.success and ("blocked" in result.error.lower() or "not allowed" in result.error.lower()):
        print("Blocked imports (restricted): PASSED")
        return True
    else:
        print("Blocked imports (restricted): FAILED")
        return False


def test_blocked_dangerous_imports():
    """Test that dangerous imports are blocked even in standard mode."""
    print("\n" + "=" * 60)
    print("TEST: Blocked Dangerous Imports")
    print("=" * 60)

    dangerous_imports = [
        "import os",
        "import subprocess",
        "import socket",
        "from os import system",
        "import pickle",
        "import ctypes",
    ]

    all_blocked = True
    for imp in dangerous_imports:
        code = f"{imp}\nresult = 'success'"
        result = execute_python_safely(code, security_level=SecurityLevel.STANDARD)
        blocked = not result.success
        print(f"  {imp}: {'BLOCKED' if blocked else 'ALLOWED!'}")
        if not blocked:
            all_blocked = False

    if all_blocked:
        print("Blocked dangerous imports: PASSED")
        return True
    else:
        print("Blocked dangerous imports: FAILED")
        return False


def test_blocked_dangerous_calls():
    """Test that dangerous function calls are blocked."""
    print("\n" + "=" * 60)
    print("TEST: Blocked Dangerous Calls")
    print("=" * 60)

    dangerous_calls = [
        ("eval('1+1')", "eval"),
        ("exec('x=1')", "exec"),
        ("compile('x=1', '', 'exec')", "compile"),
        ("__import__('os')", "__import__"),
    ]

    all_blocked = True
    for call, name in dangerous_calls:
        code = f"result = {call}"
        result = execute_python_safely(code, security_level=SecurityLevel.STANDARD)
        blocked = not result.success
        print(f"  {name}: {'BLOCKED' if blocked else 'ALLOWED!'}")
        if not blocked:
            all_blocked = False

    if all_blocked:
        print("Blocked dangerous calls: PASSED")
        return True
    else:
        print("Blocked dangerous calls: FAILED")
        return False


def test_blocked_attribute_access():
    """Test that dangerous attribute access is blocked."""
    print("\n" + "=" * 60)
    print("TEST: Blocked Attribute Access")
    print("=" * 60)

    dangerous_attrs = [
        ("x = ().__class__.__bases__", "__bases__"),
        ("x = ''.__class__.__mro__", "__mro__"),
        ("x = (lambda:0).__globals__", "__globals__"),
        ("x = (lambda:0).__code__", "__code__"),
    ]

    all_blocked = True
    for code_snippet, attr_name in dangerous_attrs:
        code = f"{code_snippet}\nresult = 'accessed'"
        is_valid, errors, _ = validate_python_code(code, SecurityLevel.STANDARD)
        blocked = not is_valid
        print(f"  {attr_name}: {'BLOCKED' if blocked else 'ALLOWED!'}")
        if not blocked:
            all_blocked = False

    if all_blocked:
        print("Blocked attribute access: PASSED")
        return True
    else:
        print("Blocked attribute access: FAILED")
        return False


def test_timeout():
    """Test that infinite loops are terminated."""
    print("\n" + "=" * 60)
    print("TEST: Timeout Protection")
    print("=" * 60)

    code = """
# Infinite loop
while True:
    pass
result = "never reached"
"""

    # Use short timeout for testing
    context = SecurityContext(
        level=SecurityLevel.STANDARD,
        timeout_seconds=1.0,  # 1 second timeout
    )
    executor = PythonExecutor(context)
    result = executor.execute(code)

    print(f"  Success: {result.success}")
    print(f"  Error: {result.error}")
    print(f"  Time: {result.execution_time_ms:.2f}ms")

    if not result.success and "timeout" in result.error.lower():
        print("Timeout protection: PASSED")
        return True
    else:
        print("Timeout protection: FAILED")
        return False


def test_safe_imports_allowed():
    """Test that safe imports are allowed in standard mode."""
    print("\n" + "=" * 60)
    print("TEST: Safe Imports Allowed")
    print("=" * 60)

    safe_code = """
import json
import re
from collections import defaultdict
from datetime import datetime

data = {"key": "value"}
result = json.dumps(data)
"""

    result = execute_python_safely(code=safe_code, security_level=SecurityLevel.STANDARD)
    print(f"  Success: {result.success}")
    print(f"  Result: {result.result}")

    if result.success and result.result == '{"key": "value"}':
        print("Safe imports allowed: PASSED")
        return True
    else:
        print(f"Safe imports allowed: FAILED - {result.error}")
        return False


def test_rows_and_ctx_available():
    """Test that rows and ctx are available in namespace."""
    print("\n" + "=" * 60)
    print("TEST: Rows and Ctx Available")
    print("=" * 60)

    code = """
# rows and ctx should be available
total = sum(row["count"] for row in rows)
multiplier = ctx.get("multiplier", 1)
result = {"total": total * multiplier}
"""

    class MockCtx:
        def get(self, key, default=None):
            return {"multiplier": 2}.get(key, default)

    namespace = {
        "rows": [{"count": 10}, {"count": 20}, {"count": 30}],
        "ctx": MockCtx(),
    }

    result = execute_python_safely(code, namespace=namespace)
    print(f"  Success: {result.success}")
    print(f"  Result: {result.result}")

    if result.success and result.result.get("total") == 120:  # 60 * 2
        print("Rows and ctx available: PASSED")
        return True
    else:
        print(f"Rows and ctx available: FAILED - {result.error}")
        return False


def test_cadsl_builtins_available():
    """Test that CADSL builtins are available."""
    print("\n" + "=" * 60)
    print("TEST: CADSL Builtins Available")
    print("=" * 60)

    code = """
# Test CADSL builtins
items = [{"name": "foo", "count": 10}, {"name": "bar", "count": 20}]
grouped = group_by(items, "name")
values = pluck(items, "count")
avg = average(values)
result = {"grouped_keys": list(grouped.keys()), "avg": avg}
"""

    result = execute_python_safely(code, security_level=SecurityLevel.STANDARD)
    print(f"  Success: {result.success}")
    print(f"  Result: {result.result}")

    if result.success:
        r = result.result
        if r.get("avg") == 15.0 and set(r.get("grouped_keys", [])) == {"foo", "bar"}:
            print("CADSL builtins available: PASSED")
            return True

    print(f"CADSL builtins available: FAILED - {result.error}")
    return False


def test_restricted_mode():
    """Test restricted mode is most restrictive."""
    print("\n" + "=" * 60)
    print("TEST: Restricted Mode")
    print("=" * 60)

    # Even safe-looking imports should fail
    code = """
from collections import Counter
result = Counter([1,2,3])
"""

    result = execute_python_safely(code, security_level=SecurityLevel.RESTRICTED)
    print(f"  Import blocked: {not result.success}")

    # Basic operations should work
    code2 = """
data = [1, 2, 3, 4, 5]
result = sum(data)
"""
    result2 = execute_python_safely(code2, security_level=SecurityLevel.RESTRICTED)
    print(f"  Basic ops work: {result2.success}")
    print(f"  Result: {result2.result}")

    if not result.success and result2.success and result2.result == 15:
        print("Restricted mode: PASSED")
        return True
    else:
        print("Restricted mode: FAILED")
        return False


def test_ast_validator():
    """Test AST validator directly."""
    print("\n" + "=" * 60)
    print("TEST: AST Validator")
    print("=" * 60)

    # Safe code
    is_valid, errors, warnings = validate_python_code("x = 1 + 2", SecurityLevel.STANDARD)
    print(f"  'x = 1 + 2': valid={is_valid}")

    # Dangerous eval (as a name access, not just call)
    is_valid2, errors2, _ = validate_python_code("y = eval", SecurityLevel.STANDARD)
    print(f"  'y = eval': valid={is_valid2}, errors={errors2}")

    # Dangerous attribute
    is_valid3, errors3, _ = validate_python_code("x.__globals__", SecurityLevel.STANDARD)
    print(f"  'x.__globals__': valid={is_valid3}, errors={errors3}")

    # Exec call
    is_valid4, errors4, _ = validate_python_code("exec('x=1')", SecurityLevel.STANDARD)
    print(f"  'exec(...)': valid={is_valid4}, errors={errors4}")

    if is_valid and not is_valid2 and not is_valid3 and not is_valid4:
        print("AST validator: PASSED")
        return True
    else:
        print("AST validator: FAILED")
        return False


def test_import_whitelist():
    """Test import whitelist validation."""
    print("\n" + "=" * 60)
    print("TEST: Import Whitelist")
    print("=" * 60)

    # Safe imports
    valid, blocked = validate_imports({"json", "re", "collections"}, SecurityLevel.STANDARD)
    print(f"  Safe imports: valid={valid}")

    # Dangerous imports
    valid2, blocked2 = validate_imports({"os", "subprocess"}, SecurityLevel.STANDARD)
    print(f"  Dangerous imports: valid={valid2}, blocked={set(blocked2)}")

    # Mixed
    valid3, blocked3 = validate_imports({"json", "os"}, SecurityLevel.STANDARD)
    print(f"  Mixed imports: valid={valid3}, blocked={blocked3}")

    # Check blocked2 contains both os and subprocess (order doesn't matter)
    blocked2_set = set(blocked2)
    if valid and not valid2 and not valid3 and blocked2_set == {"os", "subprocess"}:
        print("Import whitelist: PASSED")
        return True
    else:
        print("Import whitelist: FAILED")
        return False


def test_syntax_error_handling():
    """Test syntax error handling."""
    print("\n" + "=" * 60)
    print("TEST: Syntax Error Handling")
    print("=" * 60)

    code = """
def broken(
    # Missing closing paren
result = 1
"""

    result = execute_python_safely(code)
    print(f"  Success: {result.success}")
    print(f"  Error type: {result.error_type}")
    print(f"  Error: {result.error[:60]}...")

    # Syntax errors are caught during AST validation, so they come back as SecurityError
    if not result.success and ("syntax" in result.error.lower() or "SecurityError" in result.error_type):
        print("Syntax error handling: PASSED")
        return True
    else:
        print("Syntax error handling: FAILED")
        return False


def test_runtime_error_handling():
    """Test runtime error handling."""
    print("\n" + "=" * 60)
    print("TEST: Runtime Error Handling")
    print("=" * 60)

    code = """
x = 1 / 0
result = x
"""

    result = execute_python_safely(code)
    print(f"  Success: {result.success}")
    print(f"  Error type: {result.error_type}")
    print(f"  Error: {result.error}")

    if not result.success and result.error_type == "ZeroDivisionError":
        print("Runtime error handling: PASSED")
        return True
    else:
        print("Runtime error handling: FAILED")
        return False


# ============================================================
# TEST RUNNER
# ============================================================

def run_all_tests() -> int:
    """Run all test cases."""
    print("CADSL Python Security Sandbox Test Suite")
    print("=" * 60)

    tests = [
        test_safe_code_execution,
        test_blocked_imports_restricted,
        test_blocked_dangerous_imports,
        test_blocked_dangerous_calls,
        test_blocked_attribute_access,
        test_timeout,
        test_safe_imports_allowed,
        test_rows_and_ctx_available,
        test_cadsl_builtins_available,
        test_restricted_mode,
        test_ast_validator,
        test_import_whitelist,
        test_syntax_error_handling,
        test_runtime_error_handling,
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
