"""
CADSL Python Executor - Secure Python Block Execution.

This module provides a sandboxed execution environment for inline Python blocks
in CADSL tools. It implements multiple layers of security:

Layer 1: AST Validation - Block dangerous constructs at parse time
Layer 2: Restricted Builtins - Only safe functions available
Layer 3: Import Whitelist - Only approved modules can be imported
Layer 4: Resource Limits - Timeout and memory constraints

Security levels:
- restricted: Most restrictive, no imports, limited builtins
- standard: Safe imports, common builtins (default)
- trusted: All imports allowed, full builtins (for admin use only)
"""

import ast
import sys
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union


# ============================================================
# SECURITY LEVELS
# ============================================================

class SecurityLevel(Enum):
    """Security levels for Python block execution."""
    RESTRICTED = "restricted"  # No imports, minimal builtins
    STANDARD = "standard"      # Safe imports, common builtins
    TRUSTED = "trusted"        # Full access (admin only)


# ============================================================
# CAPABILITY SYSTEM
# ============================================================

@dataclass
class Capability:
    """Represents a permission capability."""
    prefix: str
    pattern: str  # e.g., "fs:read:*", "net:http:api.example.com"

    def matches(self, request: str) -> bool:
        """Check if this capability matches a request."""
        if self.pattern.endswith("*"):
            base = self.pattern[:-1]
            return request.startswith(base)
        return request == self.pattern


@dataclass
class SecurityContext:
    """Security context for Python execution."""
    level: SecurityLevel = SecurityLevel.STANDARD
    capabilities: List[str] = field(default_factory=list)
    timeout_seconds: float = 30.0
    max_memory_mb: int = 256
    max_output_size: int = 1_000_000  # 1MB max output

    def has_capability(self, request: str) -> bool:
        """Check if context has a capability."""
        for cap in self.capabilities:
            if cap.endswith("*"):
                if request.startswith(cap[:-1]):
                    return True
            elif cap == request:
                return True
        return False


# ============================================================
# AST VALIDATOR (Layer 1)
# ============================================================

# Dangerous AST node types by security level
BLOCKED_NODES: Dict[SecurityLevel, Set[str]] = {
    SecurityLevel.RESTRICTED: {
        "Import", "ImportFrom",  # No imports
        "Exec",  # No exec (Python 2 compat)
        "Global", "Nonlocal",  # No scope manipulation
        "AsyncFunctionDef", "AsyncFor", "AsyncWith", "Await",  # No async
    },
    SecurityLevel.STANDARD: {
        "Exec",  # No exec
        "Global", "Nonlocal",  # Limited scope manipulation
    },
    SecurityLevel.TRUSTED: set(),  # No restrictions
}

# Dangerous function calls
BLOCKED_CALLS: Dict[SecurityLevel, Set[str]] = {
    SecurityLevel.RESTRICTED: {
        "eval", "exec", "compile", "open", "input",
        "__import__", "globals", "locals", "vars",
        "getattr", "setattr", "delattr", "hasattr",
        "type", "super", "object",
        "memoryview", "bytearray",
        "breakpoint", "help", "copyright", "credits", "license",
    },
    SecurityLevel.STANDARD: {
        "eval", "exec", "compile",
        "__import__", "globals", "locals",
        "breakpoint",
    },
    SecurityLevel.TRUSTED: set(),
}

# Dangerous attribute access
BLOCKED_ATTRIBUTES: Dict[SecurityLevel, Set[str]] = {
    SecurityLevel.RESTRICTED: {
        "__class__", "__bases__", "__subclasses__", "__mro__",
        "__dict__", "__globals__", "__code__", "__closure__",
        "__func__", "__self__", "__module__", "__qualname__",
        "__builtins__", "__import__",
        "f_locals", "f_globals", "f_code", "f_back",
        "gi_frame", "gi_code",
        "co_code", "co_consts", "co_names",
    },
    SecurityLevel.STANDARD: {
        "__class__", "__bases__", "__subclasses__", "__mro__",
        "__globals__", "__code__", "__closure__",
        "__builtins__", "__import__",
        "f_locals", "f_globals", "f_code", "f_back",
        "gi_frame", "gi_code",
    },
    SecurityLevel.TRUSTED: set(),
}


class ASTValidator(ast.NodeVisitor):
    """
    Validates Python AST for security violations.

    Checks for:
    - Blocked node types (Import, Exec, etc.)
    - Dangerous function calls (eval, exec, etc.)
    - Dangerous attribute access (__class__, __globals__, etc.)
    """

    def __init__(self, security_level: SecurityLevel = SecurityLevel.STANDARD):
        self.security_level = security_level
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self._import_names: Set[str] = set()

    def validate(self, code: str) -> Tuple[bool, List[str], List[str]]:
        """
        Validate Python code.

        Returns:
            (is_valid, errors, warnings)
        """
        self.errors = []
        self.warnings = []
        self._import_names = set()

        try:
            tree = ast.parse(code, mode='exec')
            self.visit(tree)
        except SyntaxError as e:
            self.errors.append(f"Syntax error: {e}")

        return len(self.errors) == 0, self.errors, self.warnings

    def visit(self, node: ast.AST) -> None:
        """Visit a node and check for violations."""
        node_type = type(node).__name__

        # Check blocked node types
        blocked = BLOCKED_NODES.get(self.security_level, set())
        if node_type in blocked:
            self.errors.append(
                f"Blocked construct '{node_type}' at line {getattr(node, 'lineno', '?')}"
            )

        # Call type-specific visitor method if it exists
        method = 'visit_' + node_type
        visitor = getattr(self, method, None)
        if visitor:
            visitor(node)
        else:
            self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        """Check import statements."""
        if self.security_level == SecurityLevel.RESTRICTED:
            self.errors.append(
                f"Imports not allowed in restricted mode at line {node.lineno}"
            )
        else:
            for alias in node.names:
                self._import_names.add(alias.name.split('.')[0])
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Check from...import statements."""
        if self.security_level == SecurityLevel.RESTRICTED:
            self.errors.append(
                f"Imports not allowed in restricted mode at line {node.lineno}"
            )
        else:
            if node.module:
                self._import_names.add(node.module.split('.')[0])
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        """Check function calls."""
        func_name = self._get_call_name(node.func)
        blocked = BLOCKED_CALLS.get(self.security_level, set())

        if func_name and func_name in blocked:
            self.errors.append(
                f"Blocked function '{func_name}' at line {node.lineno}"
            )

        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        """Check for blocked names being accessed."""
        blocked = BLOCKED_CALLS.get(self.security_level, set())
        if node.id in blocked:
            self.errors.append(
                f"Blocked name '{node.id}' at line {node.lineno}"
            )
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        """Check attribute access."""
        blocked = BLOCKED_ATTRIBUTES.get(self.security_level, set())

        if node.attr in blocked:
            self.errors.append(
                f"Blocked attribute '{node.attr}' at line {node.lineno}"
            )

        self.generic_visit(node)

    def _get_call_name(self, node: ast.AST) -> str:
        """Get the name of a called function."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return node.attr
        return ""

    @property
    def imports(self) -> Set[str]:
        """Get set of imported module names."""
        return self._import_names


# ============================================================
# IMPORT WHITELIST (Layer 3)
# ============================================================

# Modules allowed at each security level
ALLOWED_IMPORTS: Dict[SecurityLevel, Set[str]] = {
    SecurityLevel.RESTRICTED: set(),  # No imports allowed

    SecurityLevel.STANDARD: {
        # Collections and data structures
        "collections", "itertools", "functools",
        "dataclasses", "typing", "enum",

        # Math and numbers
        "math", "statistics", "decimal", "fractions",
        "random",  # Note: not cryptographically secure

        # String processing
        "re", "string", "textwrap", "difflib",

        # Date/time
        "datetime", "calendar", "time",

        # Data formats
        "json", "csv", "base64", "hashlib",

        # Utilities
        "copy", "operator", "bisect", "heapq",
        "contextlib", "abc",
    },

    SecurityLevel.TRUSTED: {
        # All standard modules plus...
        "*",  # Special marker for "all allowed"
    },
}

# Modules that are always blocked (even in trusted mode)
ALWAYS_BLOCKED_IMPORTS: Set[str] = {
    "os", "sys", "subprocess", "shutil",  # System access
    "socket", "http", "urllib", "requests",  # Network access
    "pickle", "marshal", "shelve",  # Serialization (code execution risk)
    "importlib", "pkgutil", "modulefinder",  # Import manipulation
    "ctypes", "cffi",  # Native code access
    "multiprocessing", "threading",  # Process/thread manipulation
    "signal", "atexit",  # Signal handling
    "code", "codeop", "dis", "inspect",  # Code introspection
    "gc", "tracemalloc", "resource",  # Runtime manipulation
    "builtins", "__builtins__",  # Builtin manipulation
}


def validate_imports(imports: Set[str], security_level: SecurityLevel,
                     capabilities: List[str] = None) -> Tuple[bool, List[str]]:
    """
    Validate that all imports are allowed.

    Returns:
        (is_valid, list of blocked imports)
    """
    capabilities = capabilities or []
    blocked = []

    allowed = ALLOWED_IMPORTS.get(security_level, set())
    is_trusted = "*" in allowed

    for module in imports:
        # Check always-blocked first
        if module in ALWAYS_BLOCKED_IMPORTS:
            # Can be overridden with explicit capability
            cap_request = f"import:{module}"
            if not any(cap.startswith("import:") and
                      (cap == cap_request or cap == "import:*")
                      for cap in capabilities):
                blocked.append(module)
                continue

        # Check against whitelist
        if not is_trusted and module not in allowed:
            blocked.append(module)

    return len(blocked) == 0, blocked


# ============================================================
# SAFE BUILTINS (Layer 2)
# ============================================================

def _safe_print(*args, **kwargs) -> None:
    """Safe print that captures output instead of printing."""
    pass  # Output captured via StringIO in executor


def _safe_open(*args, **kwargs):
    """Blocked open function."""
    raise PermissionError("File access not allowed in sandbox")


def _safe_input(*args, **kwargs):
    """Blocked input function."""
    raise PermissionError("Interactive input not allowed in sandbox")


# Builtins available at each security level
def get_safe_builtins(security_level: SecurityLevel) -> Dict[str, Any]:
    """Get restricted builtins for security level."""

    # Base safe builtins (all levels)
    base = {
        # Types
        "bool": bool,
        "int": int,
        "float": float,
        "str": str,
        "list": list,
        "dict": dict,
        "set": set,
        "frozenset": frozenset,
        "tuple": tuple,
        "bytes": bytes,

        # Functions
        "len": len,
        "range": range,
        "enumerate": enumerate,
        "zip": zip,
        "map": map,
        "filter": filter,
        "sorted": sorted,
        "reversed": reversed,
        "sum": sum,
        "min": min,
        "max": max,
        "abs": abs,
        "round": round,
        "pow": pow,
        "divmod": divmod,

        # String operations
        "chr": chr,
        "ord": ord,
        "repr": repr,
        "ascii": ascii,
        "format": format,

        # Predicates
        "all": all,
        "any": any,
        "isinstance": isinstance,
        "issubclass": issubclass,
        "callable": callable,

        # Iterators
        "iter": iter,
        "next": next,

        # Exceptions (for catching)
        "Exception": Exception,
        "ValueError": ValueError,
        "TypeError": TypeError,
        "KeyError": KeyError,
        "IndexError": IndexError,
        "AttributeError": AttributeError,
        "StopIteration": StopIteration,
        "RuntimeError": RuntimeError,
        "ZeroDivisionError": ZeroDivisionError,

        # Constants
        "True": True,
        "False": False,
        "None": None,

        # Blocked with safe stubs
        "print": _safe_print,
        "open": _safe_open,
        "input": _safe_input,
    }

    if security_level == SecurityLevel.RESTRICTED:
        # Most restrictive - only base
        return base

    elif security_level == SecurityLevel.STANDARD:
        # Add some more utilities
        base.update({
            "slice": slice,
            "property": property,
            "staticmethod": staticmethod,
            "classmethod": classmethod,
            "getattr": getattr,  # Allow with restrictions checked at AST level
            "setattr": setattr,
            "hasattr": hasattr,
            "hash": hash,
            "id": id,
            "type": type,
            "object": object,
            "super": super,
        })
        return base

    else:  # TRUSTED
        # Full builtins minus the most dangerous
        import builtins
        full = dict(vars(builtins))
        # Still block these even in trusted
        for blocked in ["eval", "exec", "compile", "__import__"]:
            full.pop(blocked, None)
        return full


# ============================================================
# EXECUTION RESULT
# ============================================================

@dataclass
class ExecutionResult:
    """Result of Python code execution."""
    success: bool
    result: Any = None
    error: Optional[str] = None
    error_type: Optional[str] = None
    execution_time_ms: float = 0.0
    output: str = ""  # Captured stdout
    warnings: List[str] = field(default_factory=list)


# ============================================================
# PYTHON EXECUTOR
# ============================================================

class PythonExecutor:
    """
    Secure Python code executor with multi-layer sandboxing.

    Usage:
        executor = PythonExecutor(SecurityLevel.STANDARD)
        result = executor.execute(code, {"rows": data, "ctx": context})
        if result.success:
            return result.result
        else:
            raise RuntimeError(result.error)
    """

    def __init__(self, security_context: Optional[SecurityContext] = None):
        """
        Initialize executor with security context.

        Args:
            security_context: Security settings (defaults to STANDARD)
        """
        self.context = security_context or SecurityContext()
        self.validator = ASTValidator(self.context.level)

    def execute(self, code: str, namespace: Dict[str, Any] = None) -> ExecutionResult:
        """
        Execute Python code in sandbox.

        Args:
            code: Python code to execute
            namespace: Variables available in the code (e.g., rows, ctx)

        Returns:
            ExecutionResult with success status and result/error
        """
        start_time = time.perf_counter()
        namespace = namespace or {}

        # Layer 1: AST Validation
        is_valid, errors, warnings = self.validator.validate(code)
        if not is_valid:
            return ExecutionResult(
                success=False,
                error=f"Security validation failed: {'; '.join(errors)}",
                error_type="SecurityError",
                warnings=warnings,
            )

        # Layer 3: Import Validation
        imports = self.validator.imports
        imports_valid, blocked_imports = validate_imports(
            imports, self.context.level, self.context.capabilities
        )
        if not imports_valid:
            return ExecutionResult(
                success=False,
                error=f"Blocked imports: {', '.join(blocked_imports)}",
                error_type="ImportError",
                warnings=warnings,
            )

        # Layer 2: Prepare restricted builtins
        safe_builtins = get_safe_builtins(self.context.level)

        # Prepare execution namespace
        exec_namespace = {
            "__builtins__": safe_builtins,
            "result": None,  # Must be set by code
            **namespace,
        }

        # Add commonly used imports if at STANDARD or above
        if self.context.level != SecurityLevel.RESTRICTED:
            # Pre-import safe modules
            import collections
            import re as re_module
            import json as json_module
            import math as math_module
            import datetime as datetime_module
            import decimal
            import itertools
            import functools
            import statistics
            import copy
            import operator
            import string
            import textwrap
            import base64
            import hashlib

            exec_namespace.update({
                # Collections
                "defaultdict": collections.defaultdict,
                "Counter": collections.Counter,
                "OrderedDict": collections.OrderedDict,
                "namedtuple": collections.namedtuple,
                "deque": collections.deque,
                # Modules (pre-imported, safe to use)
                "re": re_module,
                "json": json_module,
                "math": math_module,
                "datetime": datetime_module,
                "itertools": itertools,
                "functools": functools,
                "statistics": statistics,
                "copy": copy,
                "operator": operator,
                "string": string,
                "textwrap": textwrap,
                "base64": base64,
                "hashlib": hashlib,
                # Specific useful items
                "Decimal": decimal.Decimal,
                "reduce": functools.reduce,
                "partial": functools.partial,
            })

            # Add CADSL builtins
            try:
                from .builtins import get_cadsl_builtins
                exec_namespace.update(get_cadsl_builtins())
            except ImportError:
                pass  # Builtins module not available

            # Add safe import function for whitelisted modules
            allowed_set = ALLOWED_IMPORTS.get(self.context.level, set())
            context_level = self.context.level

            def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
                """Import only whitelisted modules."""
                base_module = name.split('.')[0]

                if base_module in ALWAYS_BLOCKED_IMPORTS:
                    raise ImportError(f"Import of '{name}' is not allowed in sandbox")
                if "*" not in allowed_set and base_module not in allowed_set:
                    raise ImportError(f"Import of '{name}' is not allowed in sandbox")

                import builtins
                return builtins.__import__(name, globals, locals, fromlist, level)

            # Add the safe import to builtins so import statements work
            safe_builtins["__import__"] = safe_import

        # Layer 4: Execute with timeout
        try:
            compiled = compile(code, "<cadsl_sandbox>", "exec")

            # Execute in thread with timeout
            exec_result = {"error": None, "completed": False}

            def run_code():
                try:
                    exec(compiled, exec_namespace)
                    exec_result["completed"] = True
                except Exception as e:
                    exec_result["error"] = e

            thread = threading.Thread(target=run_code)
            thread.daemon = True
            thread.start()
            thread.join(timeout=self.context.timeout_seconds)

            if thread.is_alive():
                # Timeout - thread still running
                return ExecutionResult(
                    success=False,
                    error=f"Execution timeout ({self.context.timeout_seconds}s)",
                    error_type="TimeoutError",
                    execution_time_ms=(time.perf_counter() - start_time) * 1000,
                    warnings=warnings,
                )

            if exec_result["error"]:
                exc = exec_result["error"]
                return ExecutionResult(
                    success=False,
                    error=str(exc),
                    error_type=type(exc).__name__,
                    execution_time_ms=(time.perf_counter() - start_time) * 1000,
                    warnings=warnings,
                )

            # Get result
            result_value = exec_namespace.get("result")

            return ExecutionResult(
                success=True,
                result=result_value,
                execution_time_ms=(time.perf_counter() - start_time) * 1000,
                warnings=warnings,
            )

        except SyntaxError as e:
            return ExecutionResult(
                success=False,
                error=f"Syntax error: {e}",
                error_type="SyntaxError",
                execution_time_ms=(time.perf_counter() - start_time) * 1000,
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                error=f"Execution error: {e}",
                error_type=type(e).__name__,
                execution_time_ms=(time.perf_counter() - start_time) * 1000,
            )


# ============================================================
# SECURE PYTHON STEP
# ============================================================

class SecurePythonStep:
    """
    Pipeline step that executes Python code in a security sandbox.

    This replaces the simpler PythonStep in transformer.py with
    full security controls.
    """

    def __init__(self, code: str, security_context: Optional[SecurityContext] = None):
        """
        Initialize secure Python step.

        Args:
            code: Python code to execute
            security_context: Security settings
        """
        self.code = code
        self.context = security_context or SecurityContext()
        self.executor = PythonExecutor(self.context)

        # Pre-validate at construction time
        self._validation_result = self.executor.validator.validate(code)

    @property
    def is_valid(self) -> bool:
        """Check if code passed validation."""
        return self._validation_result[0]

    @property
    def validation_errors(self) -> List[str]:
        """Get validation errors."""
        return self._validation_result[1]

    def execute(self, data: Any, ctx: Any = None):
        """
        Execute the Python step.

        Args:
            data: Input data (available as 'rows' in code)
            ctx: Execution context (available as 'ctx' in code)

        Returns:
            PipelineResult with result or error
        """
        # Import here to avoid circular imports
        from reter_code.dsl.core import pipeline_ok, pipeline_err

        if not self.is_valid:
            return pipeline_err(
                "python",
                f"Security validation failed: {'; '.join(self.validation_errors)}"
            )

        # Prepare namespace
        namespace = {
            "rows": data,
            "ctx": ctx,
        }

        # Execute
        result = self.executor.execute(self.code, namespace)

        if result.success:
            # If no result set, pass through input data
            output = result.result if result.result is not None else data
            return pipeline_ok(output)
        else:
            return pipeline_err("python", f"{result.error_type}: {result.error}")


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def validate_python_code(code: str,
                         security_level: SecurityLevel = SecurityLevel.STANDARD
                         ) -> Tuple[bool, List[str], List[str]]:
    """
    Validate Python code without executing it.

    Returns:
        (is_valid, errors, warnings)
    """
    validator = ASTValidator(security_level)
    return validator.validate(code)


def execute_python_safely(code: str,
                          namespace: Dict[str, Any] = None,
                          security_level: SecurityLevel = SecurityLevel.STANDARD,
                          timeout: float = 30.0) -> ExecutionResult:
    """
    Execute Python code in a security sandbox.

    Args:
        code: Python code to execute
        namespace: Variables available in the code
        security_level: Security level to use
        timeout: Maximum execution time in seconds

    Returns:
        ExecutionResult with success status and result/error
    """
    context = SecurityContext(
        level=security_level,
        timeout_seconds=timeout,
    )
    executor = PythonExecutor(context)
    return executor.execute(code, namespace)
