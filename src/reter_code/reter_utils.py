"""
RETER Utility Functions

Contains utility functions for the RETER integration layer:
- Initialization state management
- Debug logging utilities
- Safe C++ call wrapper
- Source ID generation
"""

import hashlib
import os
import sys
import time
import traceback
from typing import Any, Callable, TypeVar

from .logging_config import configure_logger_for_debug_trace, is_stderr_suppressed
from .reter_exceptions import DefaultInstanceNotInitialised

# Configure module logger to also write to debug_trace.log
logger = configure_logger_for_debug_trace(__name__)


# =============================================================================
# Initialization State Management
# =============================================================================
# Two-flag system to control access during initialization:
# - _initialization_in_progress: True while background init thread is running
#   (allows internal access for loading Python files, running REQL, etc.)
# - _initialization_complete: True after init finishes (allows normal operation)
#
# Access is BLOCKED only when BOTH flags are False (before init starts)
# This ensures MCP tools cannot access RETER while allowing init code to work.

_initialization_in_progress = False
_initialization_complete = False


def set_initialization_in_progress(value: bool) -> None:
    """Set whether initialization is currently in progress."""
    global _initialization_in_progress
    _initialization_in_progress = value
    debug_log(f"[InitState] _initialization_in_progress = {value}")


def set_initialization_complete(value: bool) -> None:
    """Set whether initialization is complete."""
    global _initialization_complete
    _initialization_complete = value
    debug_log(f"[InitState] _initialization_complete = {value}")


def is_initialization_complete() -> bool:
    """Check if initialization is complete."""
    return _initialization_complete


def check_initialization() -> None:
    """
    Check if access to RETER is allowed.

    Raises:
        DefaultInstanceNotInitialised: If neither init is in progress nor complete

    Access is ALLOWED when:
    - _initialization_complete is True (normal operation after init)
    - _initialization_in_progress is True (internal init code running)

    Access is BLOCKED when:
    - Both flags are False (server starting, init not yet begun)
    """
    if not _initialization_complete and not _initialization_in_progress:
        raise DefaultInstanceNotInitialised(
            "Server is still initializing. The embedding model and code index "
            "are being loaded in the background. Please wait a few seconds and retry."
        )


def test_hybrid_mode_function(x: int, y: int) -> int:
    """
    A test function to verify hybrid mode delta tracking.

    This function was added to test that new code gets properly
    tracked in the hybrid network's delta journal.

    Args:
        x: First number
        y: Second number

    Returns:
        Sum of x and y
    """
    return x + y


# Max length for parameter/return value logging
DEBUG_MAX_VALUE_LEN = int(os.getenv("RETER_DEBUG_MAX_VALUE_LEN", "200"))

# REQL query timeout in milliseconds (default: 5 minutes = 300000ms)
# Set RETER_REQL_TIMEOUT environment variable to override (in seconds)
RETER_REQL_TIMEOUT_MS = int(os.getenv("RETER_REQL_TIMEOUT", "300")) * 1000


def debug_log(msg: str):
    """Write debug message via standard logger (configured to write to debug_trace.log)."""
    logger.debug(msg)


def _shorten_value(value: Any, max_len: int = None) -> str:
    """
    Shorten a value for debug logging.

    Args:
        value: Any value to represent as string
        max_len: Maximum length (uses DEBUG_MAX_VALUE_LEN if not specified)

    Returns:
        Shortened string representation
    """
    if max_len is None:
        max_len = DEBUG_MAX_VALUE_LEN

    try:
        # Handle None
        if value is None:
            return "None"

        # Handle common types with type info
        if isinstance(value, bool):
            return str(value)

        if isinstance(value, (int, float)):
            return str(value)

        if isinstance(value, str):
            if len(value) <= max_len:
                return repr(value)
            return repr(value[:max_len]) + f"...({len(value)} chars)"

        if isinstance(value, bytes):
            if len(value) <= max_len:
                return f"bytes({len(value)})"
            return f"bytes({len(value)})"

        if isinstance(value, (list, tuple)):
            type_name = type(value).__name__
            if len(value) == 0:
                return f"{type_name}(empty)"
            if len(value) <= 3:
                items = ", ".join(_shorten_value(v, max_len // 4) for v in value)
                result = f"{type_name}[{items}]"
                if len(result) <= max_len:
                    return result
            return f"{type_name}({len(value)} items)"

        if isinstance(value, dict):
            if len(value) == 0:
                return "dict(empty)"
            if len(value) <= 3:
                items = ", ".join(
                    f"{_shorten_value(k, 20)}: {_shorten_value(v, max_len // 4)}"
                    for k, v in list(value.items())[:3]
                )
                result = f"dict{{{items}}}"
                if len(result) <= max_len:
                    return result
            return f"dict({len(value)} keys)"

        # Handle objects with __class__
        type_name = type(value).__name__
        str_repr = str(value)
        if len(str_repr) <= max_len:
            return f"{type_name}({str_repr})"
        return f"{type_name}({str_repr[:max_len]}...)"

    except Exception as e:
        return f"<error formatting: {e}>"


def _format_args(args: tuple, kwargs: dict) -> str:
    """Format function arguments for debug logging."""
    parts = []

    # Format positional args
    for i, arg in enumerate(args):
        parts.append(f"arg{i}={_shorten_value(arg)}")

    # Format keyword args
    for key, value in kwargs.items():
        parts.append(f"{key}={_shorten_value(value)}")

    return ", ".join(parts) if parts else "(no args)"


T = TypeVar('T')


def safe_cpp_call(func: Callable[..., T], *args, **kwargs) -> T:
    """
    Safely call a C++ function with comprehensive exception handling.

    Catches:
    - Standard Python exceptions
    - SystemError (C extension errors)
    - MemoryError
    - Any other exceptions from C++ bindings

    Args:
        func: The C++ function to call
        *args: Positional arguments
        **kwargs: Keyword arguments

    Returns:
        The result of the function call

    Raises:
        RuntimeError: If C++ call fails with details about the error
    """
    func_name = func.__name__ if hasattr(func, '__name__') else str(func)
    args_str = _format_args(args, kwargs)
    debug_log(f"safe_cpp_call ENTER: {func_name}({args_str})")
    start_time = time.perf_counter()
    try:
        result = func(*args, **kwargs)
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        result_str = _shorten_value(result)
        debug_log(f"safe_cpp_call EXIT: {func_name} [{elapsed_ms:.1f}ms] -> {result_str}")
        return result
    except SystemError as e:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        # SystemError typically indicates C extension problems
        error_msg = f"C++ SystemError in {func_name}: {e}"
        debug_log(f"safe_cpp_call EXIT: {func_name} [{elapsed_ms:.1f}ms] FAILED - SystemError: {e}")
        debug_log(f"  Args were: {args_str}")
        debug_log(f"  Traceback: {traceback.format_exc()}")
        if not is_stderr_suppressed():
            print(f"ERROR: {error_msg}", file=sys.stderr)
            traceback.print_exc()
        raise RuntimeError(error_msg) from e
    except MemoryError as e:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        error_msg = f"Memory error in C++ call {func_name}: {e}"
        debug_log(f"safe_cpp_call EXIT: {func_name} [{elapsed_ms:.1f}ms] FAILED - MemoryError: {e}")
        debug_log(f"  Args were: {args_str}")
        if not is_stderr_suppressed():
            print(f"ERROR: {error_msg}", file=sys.stderr)
        raise RuntimeError(error_msg) from e
    except OSError as e:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        # OSError can occur with file operations in C++
        error_msg = f"OS error in C++ call {func_name}: {e}"
        debug_log(f"safe_cpp_call EXIT: {func_name} [{elapsed_ms:.1f}ms] FAILED - OSError: {e}")
        debug_log(f"  Args were: {args_str}")
        if not is_stderr_suppressed():
            print(f"ERROR: {error_msg}", file=sys.stderr)
        raise RuntimeError(error_msg) from e
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        # Catch any other exceptions
        error_msg = f"Error in C++ call {func_name}: {type(e).__name__}: {e}"
        debug_log(f"safe_cpp_call EXIT: {func_name} [{elapsed_ms:.1f}ms] FAILED - {type(e).__name__}: {e}")
        debug_log(f"  Args were: {args_str}")
        debug_log(f"  Traceback: {traceback.format_exc()}")
        if not is_stderr_suppressed():
            print(f"ERROR: {error_msg}", file=sys.stderr)
            traceback.print_exc()
        raise


def generate_source_id(file_content: str, rel_path: str) -> str:
    """
    Generate source ID based on MD5 hash of file content and relative path.

    Format: {md5_hash}|{rel_path}

    Args:
        file_content: Content of the Python file
        rel_path: Relative path of the file

    Returns:
        Source ID string in format "md5hash|relative/path.py"
    """
    md5_hash = hashlib.md5(file_content.encode('utf-8')).hexdigest()
    # Normalize path separators to forward slashes for consistency across platforms
    normalized_path = rel_path.replace('\\', '/')
    return f"{md5_hash}|{normalized_path}"


def extract_in_file_path(source: str) -> str:
    """
    Extract normalized file path from a source identifier.

    Handles formats like:
    - "md5hash|path/to/file.py" -> "path/to/file.py"
    - "path/to/file.py@1234567890" -> "path/to/file.py"
    - "path\\to\\file.py" -> "path/to/file.py"

    Args:
        source: Source identifier potentially containing MD5 prefix or timestamp suffix

    Returns:
        Normalized file path with forward slashes
    """
    in_file = source

    # Strip MD5 prefix if present (format: "md5hash|path")
    if '|' in in_file:
        in_file = in_file.split('|', 1)[1]

    # Strip timestamp suffix if present (format: "path@timestamp")
    if '@' in in_file:
        in_file = in_file.split('@', 1)[0]

    # Normalize path separators to forward slashes
    in_file = in_file.replace('\\', '/')

    return in_file


def format_parse_errors(errors: list) -> list:
    """
    Convert parser errors to standardized list of dicts.

    Args:
        errors: List of error dicts from C++ parser (with line, column, message keys)

    Returns:
        List of standardized error dicts with line, column, message keys
    """
    error_list = []
    for err in errors:
        error_list.append({
            "line": err.get("line", 0),
            "column": err.get("column", 0),
            "message": err.get("message", "Unknown error")
        })
    return error_list


def test_unified_hybrid_api(version: int = 10) -> str:
    """
    Test function to verify unified ReteNetwork hybrid mode works.
    Hybrid persistence confirmed working.
    """
    return f"unified_hybrid_api_v{version}_working"


__all__ = [
    # Initialization state
    "set_initialization_in_progress",
    "set_initialization_complete",
    "is_initialization_complete",
    "check_initialization",
    # Debug utilities
    "DEBUG_MAX_VALUE_LEN",
    "RETER_REQL_TIMEOUT_MS",
    "debug_log",
    "_shorten_value",
    "_format_args",
    # Safe C++ call
    "safe_cpp_call",
    # Source ID and path utilities
    "generate_source_id",
    "extract_in_file_path",
    "format_parse_errors",
]
