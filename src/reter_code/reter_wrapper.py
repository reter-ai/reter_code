"""
RETER Integration Wrapper

Provides a clean Python interface to RETER's C++ reasoning engine
using the AI lexer variant for natural language DL syntax.
"""

from pathlib import Path
from typing import List, Optional, Callable, TypeVar, Any, Tuple, Dict, Set
import logging
import time
import hashlib
import traceback
import sys
import functools
import os

from reter import Reter

from .logging_config import configure_logger_for_debug_trace

# Configure module logger to also write to debug_trace.log
logger = configure_logger_for_debug_trace(__name__)


# =============================================================================
# RETER Exception Hierarchy
# =============================================================================
class ReterError(Exception):
    """
    Base exception for all RETER operations.

    @reter: UtilityLayer(self)
    @reter: Exception(self)
    """
    pass


class ReterFileError(ReterError):
    """
    Exception for file-related RETER operations (save/load).

    @reter: UtilityLayer(self)
    @reter: Exception(self)
    """
    pass


class ReterFileNotFoundError(ReterFileError):
    """
    Raised when a RETER snapshot file is not found.

    @reter: UtilityLayer(self)
    @reter: Exception(self)
    """
    pass


class ReterSaveError(ReterFileError):
    """
    Raised when saving RETER network fails.

    @reter: UtilityLayer(self)
    @reter: Exception(self)
    """
    pass


class ReterLoadError(ReterFileError):
    """
    Raised when loading RETER network fails.

    @reter: UtilityLayer(self)
    @reter: Exception(self)
    """
    pass


class ReterQueryError(ReterError):
    """
    Exception for query-related RETER operations.

    @reter: UtilityLayer(self)
    @reter: Exception(self)
    """
    pass


class ReterOntologyError(ReterError):
    """
    Exception for ontology/knowledge loading errors.

    @reter: UtilityLayer(self)
    @reter: Exception(self)
    """
    pass


class DefaultInstanceNotInitialised(ReterError):
    """
    Raised when attempting to access RETER before initialization is complete.

    @reter: UtilityLayer(self)
    @reter: Exception(self)

    This exception is thrown by ReterWrapper and RAGIndexManager when:
    - Server is starting up and background initialization hasn't completed
    - Embedding model is still loading
    - Default instance Python files are still being indexed

    MCP tools should catch this exception and return an appropriate error
    message to the client indicating they should wait and retry.
    """
    pass


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
        print(f"ERROR: {error_msg}", file=sys.stderr)
        traceback.print_exc()
        raise RuntimeError(error_msg) from e
    except MemoryError as e:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        error_msg = f"Memory error in C++ call {func_name}: {e}"
        debug_log(f"safe_cpp_call EXIT: {func_name} [{elapsed_ms:.1f}ms] FAILED - MemoryError: {e}")
        debug_log(f"  Args were: {args_str}")
        print(f"ERROR: {error_msg}", file=sys.stderr)
        raise RuntimeError(error_msg) from e
    except OSError as e:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        # OSError can occur with file operations in C++
        error_msg = f"OS error in C++ call {func_name}: {e}"
        debug_log(f"safe_cpp_call EXIT: {func_name} [{elapsed_ms:.1f}ms] FAILED - OSError: {e}")
        debug_log(f"  Args were: {args_str}")
        print(f"ERROR: {error_msg}", file=sys.stderr)
        raise RuntimeError(error_msg) from e
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        # Catch any other exceptions
        error_msg = f"Error in C++ call {func_name}: {type(e).__name__}: {e}"
        debug_log(f"safe_cpp_call EXIT: {func_name} [{elapsed_ms:.1f}ms] FAILED - {type(e).__name__}: {e}")
        debug_log(f"  Args were: {args_str}")
        debug_log(f"  Traceback: {traceback.format_exc()}")
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
    return f"{md5_hash}|{rel_path}"


class ReterWrapper:
    """
    Wrapper for RETER - Incremental Semantic Reasoning Engine

    RETER is a forward-chaining incremental reasoner combining:
    - RETE algorithm for efficient pattern matching
    - OWL 2 RL Description Logic reasoning
    - SWRL rules for inference
    - Python code semantic analysis

    Key Design Principle: INCREMENTAL KNOWLEDGE ACCUMULATION
    - Knowledge ADDS, never replaces
    - SWRL rules automatically fire on new facts
    - Selective forgetting by source ID
    - Persistent semantic memory for AI agents

    @reter: InfrastructureLayer(self)
    @reter: ReasoningEngine(self)
    @reter: dependsOn(self, reter.Reter)
    """

    def __init__(self, load_ontology: bool = True) -> None:
        """
        Initialize RETER reasoner with AI-friendly lexer

        The AI lexer provides natural language-style syntax:
        - Case-insensitive keywords (is_a, IS_A, Is_A all work)
        - Natural language operators (is_subclass_of, some, all, not)
        - Manchester syntax: hasChild some Person
        - Prefix syntax: some hasChild that_is Person
        - SWRL rules: if Person(object x) also hasParent(object x, object y) then hasAncestor(object x, object y)
        - Standard ASCII: parentheses (), commas ,
        - Cardinality: symbols (>=, <=, =) or keywords (at_least, at_most, exactly)

        Args:
            load_ontology: If True, automatically load Python ontology.
                          Set to False when loading from snapshot (ontology already included).
        """
        debug_log(f"ReterWrapper.__init__ starting... (load_ontology={load_ontology})")
        debug_log("Creating Reter(variant='ai')...")
        try:
            self.reasoner = Reter(variant="ai")
            debug_log("Reter created successfully")
        except Exception as e:
            debug_log(f"ERROR creating Reter: {type(e).__name__}: {e}")
            raise
        self._session_stats = {"total_wmes": 0, "total_sources": 0}

        # Change tracking for auto-save
        self._dirty = False  # True if instance has unsaved changes
        self._last_save_time = time.time()  # Timestamp of last save

        # Automatically load ontologies for code analysis (skip if loading from snapshot)
        # Load order: OO meta-ontology first (defines generic concepts), then language-specific
        if load_ontology:
            self._load_oo_ontology()  # Must be first - defines generic OO concepts
            self._load_python_ontology()
            self._load_javascript_ontology()
            self._load_html_ontology()
            self._load_csharp_ontology()
            self._load_cpp_ontology()
        else:
            debug_log("Skipping ontology load (will be loaded from snapshot)")

    def _load_oo_ontology(self) -> None:
        """
        Load the Object-Oriented meta-ontology first.

        This ontology defines language-independent OO concepts (class, function,
        method, etc.) using CNL naming conventions. Must be loaded before
        Python and JavaScript ontologies.

        Loaded with source "oo_ontology" for potential selective forgetting.
        """
        debug_log("_load_oo_ontology starting...")
        try:
            # Get path to CNL ontology file relative to this module
            # Resources are now inside the package: src/reter_code/resources/
            ontology_path = Path(__file__).parent / "resources" / "oo_ontology.cnl"
            debug_log(f"OO ontology path: {ontology_path}")

            if ontology_path.exists():
                debug_log("Reading OO CNL ontology file...")
                with open(ontology_path, 'r', encoding='utf-8') as f:
                    ontology_content = f.read()
                debug_log(f"OO ontology content length: {len(ontology_content)} chars")

                # Load the CNL ontology into RETER - wrap with safe call for C++ protection
                debug_log("Calling safe_cpp_call(load_cnl) for OO...")
                wme_count = safe_cpp_call(self.reasoner.load_cnl, ontology_content, "oo_ontology")
                debug_log(f"load_cnl returned: {wme_count} WMEs")
                self._session_stats["total_wmes"] += wme_count
                self._session_stats["total_sources"] += 1

                debug_log(f"OO meta-ontology loaded successfully ({wme_count} WMEs)")
            else:
                # Log warning but don't fail - ontology is optional
                debug_log(f"WARNING: OO ontology not found at {ontology_path}")
        except Exception as e:
            # Log error but don't fail initialization - ontology is optional
            debug_log(f"ERROR loading OO ontology: {type(e).__name__}: {e}")

    def _load_python_ontology(self) -> None:
        """
        Automatically load Python code analysis ontology on initialization

        This ontology provides inference rules for Python code understanding:
        - Class hierarchy (Module, Class, Function, Method, etc.)
        - Transitive relationships (calls, imports, inheritance)
        - Magic method recognition (__init__, __str__, etc.)
        - Decorator-based inference (@property, @dataclass, etc.)

        Loaded with source "python_ontology" for potential selective forgetting.
        """
        debug_log("_load_python_ontology starting...")
        try:
            # Get path to CNL ontology file relative to this module
            # Resources are now inside the package: src/reter_code/resources/
            ontology_path = Path(__file__).parent / "resources" / "py_ontology.cnl"
            debug_log(f"Ontology path: {ontology_path}")

            if ontology_path.exists():
                debug_log("Reading CNL ontology file...")
                with open(ontology_path, 'r', encoding='utf-8') as f:
                    ontology_content = f.read()
                debug_log(f"Ontology content length: {len(ontology_content)} chars")

                # Load the CNL ontology into RETER - wrap with safe call for C++ protection
                debug_log("Calling safe_cpp_call(load_cnl)...")
                wme_count = safe_cpp_call(self.reasoner.load_cnl, ontology_content, "python_ontology")
                debug_log(f"load_cnl returned: {wme_count} WMEs")
                self._session_stats["total_wmes"] += wme_count
                self._session_stats["total_sources"] += 1

                debug_log(f"Python ontology loaded successfully ({wme_count} WMEs)")
            else:
                # Log warning but don't fail - ontology is optional
                debug_log(f"WARNING: Python ontology not found at {ontology_path}")
        except Exception as e:
            # Log error but don't fail initialization - ontology is optional
            debug_log(f"ERROR loading Python ontology: {type(e).__name__}: {e}")

    def _load_javascript_ontology(self) -> None:
        """
        Automatically load JavaScript code analysis ontology on initialization

        This ontology provides inference rules for JavaScript code understanding:
        - Class hierarchy (Module, Class, Function, Method, ArrowFunction, etc.)
        - Transitive relationships (calls, imports, inheritance)
        - Async/generator function recognition
        - Error handling patterns (try/catch/finally)
        - ES6+ features (private fields, getters/setters, etc.)

        Loaded with source "javascript_ontology" for potential selective forgetting.
        """
        debug_log("_load_javascript_ontology starting...")
        try:
            # Get path to CNL ontology file relative to this module
            # Resources are now inside the package: src/reter_code/resources/
            ontology_path = Path(__file__).parent / "resources" / "js_ontology.cnl"
            debug_log(f"JavaScript ontology path: {ontology_path}")

            if ontology_path.exists():
                debug_log("Reading JavaScript CNL ontology file...")
                with open(ontology_path, 'r', encoding='utf-8') as f:
                    ontology_content = f.read()
                debug_log(f"JavaScript ontology content length: {len(ontology_content)} chars")

                # Load the CNL ontology into RETER - wrap with safe call for C++ protection
                debug_log("Calling safe_cpp_call(load_cnl) for JavaScript...")
                wme_count = safe_cpp_call(self.reasoner.load_cnl, ontology_content, "javascript_ontology")
                debug_log(f"load_cnl returned: {wme_count} WMEs")
                self._session_stats["total_wmes"] += wme_count
                self._session_stats["total_sources"] += 1

                debug_log(f"JavaScript ontology loaded successfully ({wme_count} WMEs)")
            else:
                # Log warning but don't fail - ontology is optional
                debug_log(f"WARNING: JavaScript ontology not found at {ontology_path}")
        except Exception as e:
            # Log error but don't fail initialization - ontology is optional
            debug_log(f"ERROR loading JavaScript ontology: {type(e).__name__}: {e}")

    def _load_html_ontology(self) -> None:
        """
        Automatically load HTML document analysis ontology on initialization

        This ontology provides inference rules for HTML document understanding:
        - Document structure (documents, elements, forms, links)
        - Script detection (inline and external)
        - Event handler recognition (onclick, onsubmit, etc.)
        - Framework detection (Vue, Angular, HTMX, Alpine)
        - Security considerations (CSRF, script integrity)

        Loaded with source "html_ontology" for potential selective forgetting.
        """
        debug_log("_load_html_ontology starting...")
        try:
            # Get path to CNL ontology file relative to this module
            # Resources are now inside the package: src/reter_code/resources/
            ontology_path = Path(__file__).parent / "resources" / "html_ontology.cnl"
            debug_log(f"HTML ontology path: {ontology_path}")

            if ontology_path.exists():
                debug_log("Reading HTML CNL ontology file...")
                with open(ontology_path, 'r', encoding='utf-8') as f:
                    ontology_content = f.read()
                debug_log(f"HTML ontology content length: {len(ontology_content)} chars")

                # Load the CNL ontology into RETER - wrap with safe call for C++ protection
                debug_log("Calling safe_cpp_call(load_cnl) for HTML...")
                wme_count = safe_cpp_call(self.reasoner.load_cnl, ontology_content, "html_ontology")
                debug_log(f"load_cnl returned: {wme_count} WMEs")
                self._session_stats["total_wmes"] += wme_count
                self._session_stats["total_sources"] += 1

                debug_log(f"HTML ontology loaded successfully ({wme_count} WMEs)")
            else:
                # Log warning but don't fail - ontology is optional
                debug_log(f"WARNING: HTML ontology not found at {ontology_path}")
        except Exception as e:
            # Log error but don't fail initialization - ontology is optional
            debug_log(f"ERROR loading HTML ontology: {type(e).__name__}: {e}")

    def _load_csharp_ontology(self) -> None:
        """
        Automatically load C# code analysis ontology on initialization

        This ontology provides inference rules for C# code understanding:
        - Class hierarchy (CompilationUnit, Namespace, Class, Struct, Interface, etc.)
        - Transitive relationships (calls, inheritance)
        - Method and property recognition
        - Attribute/decorator support
        - Exception handling (try/catch/finally)

        Loaded with source "csharp_ontology" for potential selective forgetting.
        """
        debug_log("_load_csharp_ontology starting...")
        try:
            # Get path to CNL ontology file relative to this module
            # Resources are now inside the package: src/reter_code/resources/
            ontology_path = Path(__file__).parent / "resources" / "cs_ontology.cnl"
            debug_log(f"C# ontology path: {ontology_path}")

            if ontology_path.exists():
                debug_log("Reading C# CNL ontology file...")
                with open(ontology_path, 'r', encoding='utf-8') as f:
                    ontology_content = f.read()
                debug_log(f"C# ontology content length: {len(ontology_content)} chars")

                # Load the CNL ontology into RETER - wrap with safe call for C++ protection
                debug_log("Calling safe_cpp_call(load_cnl) for C#...")
                wme_count = safe_cpp_call(self.reasoner.load_cnl, ontology_content, "csharp_ontology")
                debug_log(f"load_cnl returned: {wme_count} WMEs")
                self._session_stats["total_wmes"] += wme_count
                self._session_stats["total_sources"] += 1

                debug_log(f"C# ontology loaded successfully ({wme_count} WMEs)")
            else:
                # Log warning but don't fail - ontology is optional
                debug_log(f"WARNING: C# ontology not found at {ontology_path}")
        except Exception as e:
            # Log error but don't fail initialization - ontology is optional
            debug_log(f"ERROR loading C# ontology: {type(e).__name__}: {e}")

    def _load_cpp_ontology(self) -> None:
        """
        Automatically load C++ code analysis ontology on initialization

        This ontology provides inference rules for C++ code understanding:
        - Class hierarchy (TranslationUnit, Namespace, Class, Struct, etc.)
        - Transitive relationships (calls, inheritance)
        - Method and field recognition
        - Template support
        - Exception handling (try/catch/throw)
        - Enums and enumerators

        Loaded with source "cpp_ontology" for potential selective forgetting.
        """
        debug_log("_load_cpp_ontology starting...")
        try:
            # Get path to CNL ontology file relative to this module
            # Resources are now inside the package: src/reter_code/resources/
            ontology_path = Path(__file__).parent / "resources" / "cpp_ontology.cnl"
            debug_log(f"C++ ontology path: {ontology_path}")

            if ontology_path.exists():
                debug_log("Reading C++ CNL ontology file...")
                with open(ontology_path, 'r', encoding='utf-8') as f:
                    ontology_content = f.read()
                debug_log(f"C++ ontology content length: {len(ontology_content)} chars")

                # Load the CNL ontology into RETER - wrap with safe call for C++ protection
                debug_log("Calling safe_cpp_call(load_cnl) for C++...")
                wme_count = safe_cpp_call(self.reasoner.load_cnl, ontology_content, "cpp_ontology")
                debug_log(f"load_cnl returned: {wme_count} WMEs")
                self._session_stats["total_wmes"] += wme_count
                self._session_stats["total_sources"] += 1

                debug_log(f"C++ ontology loaded successfully ({wme_count} WMEs)")
            else:
                # Log warning but don't fail - ontology is optional
                debug_log(f"WARNING: C++ ontology not found at {ontology_path}")
        except Exception as e:
            # Log error but don't fail initialization - ontology is optional
            debug_log(f"ERROR loading C++ ontology: {type(e).__name__}: {e}")

    def _run_with_lock(self, func: Callable[..., T], *args: Any) -> T:
        """
        Run a function directly - no lock needed in synchronous version.
        """
        return func(*args)

    def _load_directory_generic(
        self,
        directory: str,
        extensions: List[str],
        default_excludes: List[str],
        load_file_func: Callable[[str, str], Tuple[int, str, float, List[Any]]],
        recursive: bool = True,
        exclude_patterns: Optional[List[str]] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> Tuple[int, Dict[str, List[Any]], float]:
        """
        Generic directory loader for any language.

        Args:
            directory: Path to directory containing source files
            extensions: List of glob patterns for file extensions (e.g., ["*.py"], ["*.js", "*.jsx"])
            default_excludes: List of directory/path substrings to exclude by default
            load_file_func: Function to load a single file, signature: (filepath, base_path) -> (wme_count, source_id, time_ms, errors)
            recursive: If True, recursively scan subdirectories
            exclude_patterns: Additional patterns to exclude (e.g., ["test_*.py"])
            progress_callback: Optional callback function(items_processed, total_items, message)

        Returns:
            Tuple[int, dict, float]: (total_wmes, all_errors, time_ms)
        """
        check_initialization()
        start_time = time.time()

        if not exclude_patterns:
            exclude_patterns = []

        import fnmatch

        # Find all files matching extensions
        directory_path = Path(directory)
        files: List[Path] = []

        for ext in extensions:
            if recursive:
                files.extend(directory_path.rglob(ext))
            else:
                files.extend(directory_path.glob(ext))

        # Filter out default excludes (check if any exclude substring is in the path)
        filtered_files = []
        for filepath in files:
            filepath_str = str(filepath)
            excluded = False
            for exclude in default_excludes:
                # Handle both substring matches and path segment matches
                if exclude in filepath_str or exclude in filepath_str.split(os.sep):
                    excluded = True
                    break
            if not excluded:
                filtered_files.append(filepath)
        files = filtered_files

        # Apply user exclude patterns
        filtered_files = []
        for filepath in files:
            try:
                rel_path = filepath.relative_to(directory_path)
            except ValueError:
                rel_path = filepath

            excluded = False
            for pattern in exclude_patterns:
                if fnmatch.fnmatch(str(rel_path), pattern) or fnmatch.fnmatch(filepath.name, pattern):
                    excluded = True
                    break

            if not excluded:
                filtered_files.append(filepath)
        files = filtered_files

        # Load each file
        total_wmes = 0
        all_errors: Dict[str, List[Any]] = {}

        for i, filepath in enumerate(files):
            try:
                # Progress callback
                if progress_callback:
                    try:
                        rel_path = filepath.relative_to(directory_path)
                    except ValueError:
                        rel_path = filepath
                    progress_callback(i, len(files), str(rel_path))

                wme_count, source_id, _, errors = load_file_func(str(filepath), str(directory_path))
                total_wmes += wme_count

                if errors:
                    all_errors[str(filepath)] = errors

            except Exception as e:
                all_errors[str(filepath)] = [{"line": 0, "message": str(e)}]

        time_ms = (time.time() - start_time) * 1000
        return total_wmes, all_errors, time_ms

    @staticmethod
    def _path_to_module_name(file_path: str, package_roots: Optional[Set[str]] = None) -> str:
        """
        Convert a file path to a Python module name, respecting package structure.

        Python module names are determined by the package hierarchy:
        - A directory is a package only if it contains __init__.py
        - Module names are relative to the nearest package root
        - Files outside packages use just their filename

        Args:
            file_path: Relative file path with forward slashes (e.g., "src/utils/helpers.py")
            package_roots: Set of directory paths that contain __init__.py files.
                          If None, falls back to simple path-to-dots conversion.

        Returns:
            Python module name using dot notation

        Examples with package_roots={"src", "src/utils", "src/utils/sub"}:
            "src/utils/helpers.py" -> "utils.helpers"  (src is package root)
            "src/utils/__init__.py" -> "utils"
            "src/utils/sub/mod.py" -> "utils.sub.mod"
            "tests/test_foo.py" -> "test_foo"  (tests/ has no __init__.py)
            "script.py" -> "script"

        Examples with package_roots=None (legacy behavior):
            "src/utils/helpers.py" -> "src.utils.helpers"
        """
        # Remove .py extension
        if file_path.endswith('.py'):
            module_path = file_path[:-3]
        else:
            module_path = file_path

        # Handle __init__.py - use parent directory as module name
        if module_path.endswith('/__init__'):
            module_path = module_path[:-9]  # Remove "/__init__"

        # If no package structure info, fall back to simple conversion
        if package_roots is None:
            module_name = module_path.replace('/', '.')
            return module_name if module_name else "__init__"

        # Find the package root for this file
        # Walk up from the file's directory to find the topmost package
        parts = module_path.split('/')

        if len(parts) == 1:
            # File at root level (e.g., "script.py")
            return parts[0] if parts[0] else "__init__"

        # Find the deepest directory that starts a package chain
        # A valid package root is where:
        # 1. The directory has __init__.py (is in package_roots)
        # 2. All ancestors up to that point also have __init__.py

        # Build list of ancestor directories
        # For "src/utils/sub/mod.py" -> ["src", "src/utils", "src/utils/sub"]
        ancestors = []
        for i in range(len(parts) - 1):  # -1 to exclude the file itself
            ancestors.append('/'.join(parts[:i+1]))

        # Find the topmost package root (the first ancestor that is a package)
        package_start_idx = None
        for i, ancestor in enumerate(ancestors):
            if ancestor in package_roots:
                # Check if this starts a valid package chain
                # All directories from here to the file must be packages
                is_valid_chain = True
                for j in range(i, len(ancestors)):
                    if ancestors[j] not in package_roots:
                        is_valid_chain = False
                        break

                if is_valid_chain:
                    package_start_idx = i
                    break

        if package_start_idx is not None:
            # Module name starts from the package root's name
            # e.g., if "src" is package root and file is "src/utils/helpers.py"
            # then module name is "utils.helpers" (relative to src's parent)
            # BUT if we want the full qualified name from the package root:
            # "src.utils.helpers"
            module_parts = parts[package_start_idx:]
            module_name = '.'.join(module_parts)
        else:
            # No valid package chain - just use the filename
            module_name = parts[-1]

        return module_name if module_name else "__init__"

    @staticmethod
    def scan_package_roots(project_root: str) -> Set[str]:
        """
        Scan a project directory to find all Python package roots.

        A package root is a directory containing __init__.py.

        Args:
            project_root: Absolute path to the project root

        Returns:
            Set of relative paths (with forward slashes) that are Python packages
        """
        from pathlib import Path

        package_roots: Set[str] = set()
        project_path = Path(project_root)

        # Find all __init__.py files
        for init_file in project_path.rglob("__init__.py"):
            # Get the directory containing __init__.py
            package_dir = init_file.parent

            # Skip common non-package directories
            rel_path = str(package_dir.relative_to(project_path)).replace('\\', '/')
            if any(skip in rel_path for skip in ['__pycache__', 'node_modules', '.git', 'venv', '.venv', 'env', '.env']):
                continue

            # Add this directory as a package root
            if rel_path == '.':
                # Root directory has __init__.py (rare but valid)
                package_roots.add('')
            else:
                package_roots.add(rel_path)

        return package_roots

    def add_ontology(self, ontology: str, source: Optional[str] = None) -> Tuple[int, Optional[str], float]:
        """
        Incrementally add DL ontology facts/rules to RETER

        IMPORTANT: This ADDS knowledge, doesn't replace!
        - Facts accumulate across calls
        - SWRL rules automatically apply to existing facts
        - Use forget_logics(source) to selectively remove

        Args:
            ontology: DL/SWRL ontology text (supports multiple syntaxes)
            source: Optional source identifier for selective forgetting

        Returns:
            Tuple[int, str, float]: (wme_count, source, time_ms)

        Raises:
            Exception: If ontology loading fails

        Example:
            # Step 1: Add base classes
            wme_count, source, time_ms = add_ontology("Person is_a Thing", source="base")

            # Step 2: Add inference rule (applies to existing facts!)
            add_ontology("if Person(object x) also age(object x, var y) then Adult(object x)", source="rules")

            # Step 3: Add facts (rule auto-fires!)
            add_ontology("Person(Alice)\\nage(Alice, 25)", source="data")
            # → RETER infers: Adult(Alice)
        """
        check_initialization()
        start_time = time.time()
        wme_count = safe_cpp_call(self.reasoner.load_ontology, ontology, source)
        self._session_stats["total_wmes"] += wme_count
        self._session_stats["total_sources"] += 1
        time_ms = (time.time() - start_time) * 1000
        self._dirty = True  # Mark instance as modified
        return wme_count, source, time_ms

    def add_ontology_file(self, source: str) -> Tuple[int, str, float]:
        """
        Incrementally add DL ontology from file to RETER

        Same incremental behavior as add_ontology() but reads from file.

        Args:
            source: Path to ontology file to load

        Returns:
            Tuple[int, str, float]: (wme_count, source, time_ms)

        Raises:
            Exception: If file loading fails

        Note: The parameter is named 'source' to match add_ontology(source, source_id).
        This 'source' is the file path, not a source tracking identifier.
        """
        check_initialization()
        start_time = time.time()
        wme_count = safe_cpp_call(self.reasoner.load_ontology_file, source)
        self._session_stats["total_wmes"] += wme_count
        self._session_stats["total_sources"] += 1
        time_ms = (time.time() - start_time) * 1000
        self._dirty = True  # Mark instance as modified
        return wme_count, source, time_ms

    def reql(self, query: str, timeout_ms: Optional[int] = None) -> Any:
        """
        Execute REQL query and return PyArrow Table.

        REQL = RETE Query Language (SPARQL-inspired for DL)

        Supports all REQL query types:
        - SELECT: Returns tabular results with columns and rows
        - ASK: Returns boolean result (true/false)
        - DESCRIBE: Returns RDF-style triples

        Returns PyArrow Table directly. Server.py should handle conversion
        to dicts if needed for API responses.

        Args:
            query: REQL query string
            timeout_ms: Query timeout in milliseconds. If None, uses RETER_REQL_TIMEOUT_MS
                       (default 300000ms = 5 minutes). Set to 0 for no timeout.

        Returns:
            PyArrow Table with query results

        Raises:
            DefaultInstanceNotInitialised: If server initialization not complete
            RuntimeError: If query execution times out
            Exception: If query execution fails
        """
        check_initialization()
        # Use default timeout if not specified
        if timeout_ms is None:
            timeout_ms = RETER_REQL_TIMEOUT_MS
        # Returns PyArrow table directly - wrap with safe call for C++ protection
        return safe_cpp_call(self.reasoner.reql, query, timeout_ms)

    def load_python_file(
        self,
        filepath: str,
        base_path: Optional[str] = None,
        package_roots: Optional[Set[str]] = None
    ) -> Tuple[int, str, float, List[str]]:
        """
        Load Python source file and add semantic facts to RETER

        Args:
            filepath: Path to Python file to load
            base_path: Optional base path for calculating relative path (defaults to filepath's parent)
            package_roots: Optional set of Python package root directories (containing __init__.py).
                          If provided, enables proper Python module name calculation.

        Returns:
            Tuple[int, str, float, list]: (wme_count, source_id, time_ms, errors)

        Raises:
            Exception: If loading fails

        Extracts and adds facts about:
        - Classes, methods, functions
        - Inheritance relationships
        - Function calls
        - Imports and dependencies
        - Decorators
        - Parameters and return types

        Source ID is generated as MD5 hash of content | relative path.
        INCREMENTAL: Adds to existing knowledge, doesn't replace.
        """
        check_initialization()
        start_time = time.time()

        # Read file content
        filepath_obj = Path(filepath)
        with open(filepath_obj, 'r', encoding='utf-8') as f:
            code = f.read()

        # Calculate relative path
        if base_path:
            try:
                rel_path = filepath_obj.relative_to(Path(base_path))
            except ValueError:
                rel_path = filepath_obj
        else:
            rel_path = filepath_obj.name

        # Generate MD5-based source ID
        source_id = generate_source_id(code, str(rel_path))

        # Use relative path with forward slashes for inFile (e.g., "path/to/file.py")
        in_file = str(rel_path).replace('\\', '/')

        # Calculate Python module name from relative path
        # With package_roots: respects __init__.py to calculate proper import paths
        # Without: falls back to simple path-to-dots conversion
        module_name = self._path_to_module_name(in_file, package_roots)

        # Load Python code with in_file, module_name, and source_id - wrap with safe call for C++ protection
        wme_count, errors = safe_cpp_call(self.reasoner.load_python_code, code, in_file, module_name, source_id)

        self._session_stats["total_wmes"] += wme_count
        time_ms = (time.time() - start_time) * 1000
        self._dirty = True  # Mark instance as modified

        # Return errors in the response
        return wme_count, source_id, time_ms, errors

    def load_python_code(
        self,
        code: str,
        source: str = "module",
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        package_roots: Optional[Set[str]] = None
    ) -> Tuple[int, str, float, List[str]]:
        """
        Load Python code string and add semantic facts to RETER

        Args:
            code: Python source code as string
            source: Source ID for tracking (can be module name, path, or path@timestamp)
            progress_callback: Optional callback function(items_processed, total_items, message)
            package_roots: Optional set of Python package root directories (containing __init__.py).
                          If provided, enables proper Python module name calculation.

        Returns:
            Tuple[int, str, float, list]: (wme_count, source, time_ms, errors)

        Raises:
            Exception: If loading fails

        Same as load_python_file() but takes code as string.
        Useful for loading code snippets or dynamically generated code.

        INCREMENTAL: Adds to existing knowledge.
        """
        check_initialization()
        start_time = time.time()

        # Extract file path from source (strip timestamp, MD5)
        # Source formats: "module", "path/file.py", "path/file.py@timestamp", "md5|path/file.py"
        in_file = source

        # Strip MD5 prefix if present (format: "md5hash|path")
        if '|' in in_file:
            in_file = in_file.split('|', 1)[1]

        # Strip timestamp suffix if present (format: "path@timestamp")
        if '@' in in_file:
            in_file = in_file.split('@', 1)[0]

        # Normalize path separators to forward slashes (C++ visitor expects this)
        in_file = in_file.replace('\\', '/')

        # Calculate Python module name from file path
        # With package_roots: respects __init__.py to calculate proper import paths
        # Without: falls back to simple path-to-dots conversion
        module_name = self._path_to_module_name(in_file, package_roots)

        # load_python_code signature: (code, in_file, module_name, source_id)
        wme_count, errors = safe_cpp_call(self.reasoner.load_python_code, code, in_file, module_name, source)

        self._session_stats["total_wmes"] += wme_count
        time_ms = (time.time() - start_time) * 1000
        self._dirty = True  # Mark instance as modified

        # Return errors in the response
        return wme_count, source, time_ms, errors

    def load_python_directory(self, directory: str, recursive: bool = True, exclude_patterns: Optional[List[str]] = None, progress_callback: Optional[Callable[[int, int, str], None]] = None) -> Tuple[int, Dict[str, List[str]], float]:
        """
        Load all Python files from a directory with optional exclusion patterns

        Args:
            directory: Path to directory containing Python files
            recursive: If True, recursively scan subdirectories
            exclude_patterns: List of patterns to exclude (e.g., ["test_*.py", "tests/**/*.py"])
            progress_callback: Optional callback function(items_processed, total_items, message)

        Returns:
            Tuple[int, dict, float]: (total_wmes, all_errors, time_ms)
            where all_errors is a dict mapping filepath -> list of errors

        Raises:
            Exception: If loading fails

        INCREMENTAL: Adds to existing knowledge.
        """
        return self._load_directory_generic(
            directory=directory,
            extensions=["*.py"],
            default_excludes=["__pycache__"],
            load_file_func=self.load_python_file,
            recursive=recursive,
            exclude_patterns=exclude_patterns,
            progress_callback=progress_callback
        )

    def load_javascript_file(self, filepath: str, base_path: Optional[str] = None) -> Tuple[int, str, float, List[str]]:
        """
        Load JavaScript source file and add semantic facts to RETER

        Args:
            filepath: Path to JavaScript file to load
            base_path: Optional base path for calculating relative path (defaults to filepath's parent)

        Returns:
            Tuple[int, str, float, list]: (wme_count, source_id, time_ms, errors)

        Raises:
            Exception: If loading fails

        Extracts and adds facts about:
        - Classes, methods, functions
        - Inheritance relationships
        - Function calls
        - Imports and exports
        - Arrow functions
        - Parameters

        Source ID is generated as MD5 hash of content | relative path.
        INCREMENTAL: Adds to existing knowledge, doesn't replace.
        """
        check_initialization()
        start_time = time.time()

        # Read file content
        filepath_obj = Path(filepath)
        with open(filepath_obj, 'r', encoding='utf-8') as f:
            code = f.read()

        # Calculate relative path
        if base_path:
            try:
                rel_path = filepath_obj.relative_to(Path(base_path))
            except ValueError:
                rel_path = filepath_obj
        else:
            rel_path = filepath_obj.name

        # Generate MD5-based source ID
        source_id = generate_source_id(code, str(rel_path))

        # Use relative path with forward slashes for inFile (e.g., "path/to/file.js")
        in_file = str(rel_path).replace('\\', '/')

        # Load JavaScript code - use the C++ bindings
        from reter import owl_rete_cpp
        facts, errors = owl_rete_cpp.parse_javascript_code(code, in_file)

        # Add facts to the network with source tracking
        wme_count = 0
        for fact in facts:
            self.reasoner.network.add_fact_with_source(
                owl_rete_cpp.Fact(fact),
                source_id
            )
            wme_count += 1

        time_ms = (time.time() - start_time) * 1000
        self._dirty = True  # Mark instance as modified

        # Convert errors to list of dicts for consistency
        error_list = []
        for err in errors:
            error_list.append({
                "line": err.get("line", 0),
                "column": err.get("column", 0),
                "message": err.get("message", "Unknown error")
            })

        return wme_count, source_id, time_ms, error_list

    def load_javascript_code(self, code: str, source: str = "module", progress_callback: Optional[Callable[[int, int, str], None]] = None) -> Tuple[int, str, float, List[str]]:
        """
        Load JavaScript code string and add semantic facts to RETER

        Args:
            code: JavaScript source code as string
            source: Source ID for tracking (can be module name, path, or path@timestamp)
            progress_callback: Optional callback function(items_processed, total_items, message)

        Returns:
            Tuple[int, str, float, list]: (wme_count, source, time_ms, errors)

        Raises:
            Exception: If loading fails

        Same as load_javascript_file() but takes code as string.
        Useful for loading code snippets or dynamically generated code.

        INCREMENTAL: Adds to existing knowledge.
        """
        check_initialization()
        start_time = time.time()

        # Extract file path from source (strip timestamp, MD5)
        in_file = source

        # Strip MD5 prefix if present (format: "md5hash|path")
        if '|' in in_file:
            in_file = in_file.split('|', 1)[1]

        # Strip timestamp suffix if present (format: "path@timestamp")
        if '@' in in_file:
            in_file = in_file.split('@', 1)[0]

        # Normalize path separators to forward slashes
        in_file = in_file.replace('\\', '/')

        # Load JavaScript code - use the C++ bindings (C++ derives module name from in_file)
        from reter import owl_rete_cpp
        facts, errors = owl_rete_cpp.parse_javascript_code(code, in_file)

        # Add facts to the network with source tracking
        wme_count = 0
        for fact in facts:
            self.reasoner.network.add_fact_with_source(
                owl_rete_cpp.Fact(fact),
                source
            )
            wme_count += 1

        time_ms = (time.time() - start_time) * 1000
        self._dirty = True  # Mark instance as modified

        # Convert errors to list of dicts for consistency
        error_list = []
        for err in errors:
            error_list.append({
                "line": err.get("line", 0),
                "column": err.get("column", 0),
                "message": err.get("message", "Unknown error")
            })

        return wme_count, source, time_ms, error_list

    def load_javascript_directory(self, directory: str, recursive: bool = True, exclude_patterns: Optional[List[str]] = None, progress_callback: Optional[Callable[[int, int, str], None]] = None) -> Tuple[int, Dict[str, List[str]], float]:
        """
        Load all JavaScript files from a directory with optional exclusion patterns

        Args:
            directory: Path to directory containing JavaScript files
            recursive: If True, recursively scan subdirectories
            exclude_patterns: List of patterns to exclude (e.g., ["test_*.js", "node_modules/**/*.js"])
            progress_callback: Optional callback function(items_processed, total_items, message)

        Returns:
            Tuple[int, dict, float]: (total_wmes, all_errors, time_ms)
            where all_errors is a dict mapping filepath -> list of errors

        Raises:
            Exception: If loading fails

        INCREMENTAL: Adds to existing knowledge.
        """
        return self._load_directory_generic(
            directory=directory,
            extensions=["*.js", "*.mjs", "*.jsx"],
            default_excludes=["node_modules"],
            load_file_func=self.load_javascript_file,
            recursive=recursive,
            exclude_patterns=exclude_patterns,
            progress_callback=progress_callback
        )

    def load_html_file(self, filepath: str, base_path: Optional[str] = None) -> Tuple[int, str, float, List[str]]:
        """
        Load HTML file and add semantic facts to RETER

        Args:
            filepath: Path to HTML file to load
            base_path: Optional base path for calculating relative path (defaults to filepath's parent)

        Returns:
            Tuple[int, str, float, list]: (wme_count, source_id, time_ms, errors)

        Raises:
            Exception: If loading fails

        Extracts and adds facts about:
        - Document structure (title, language, charset)
        - Elements (forms, inputs, links, etc.)
        - Scripts (inline and external references)
        - Event handlers (onclick, onsubmit, etc.)
        - Framework usage (Vue, Angular, HTMX, Alpine)
        - Embedded JavaScript (parsed and extracted)

        Source ID is generated as MD5 hash of content | relative path.
        INCREMENTAL: Adds to existing knowledge, doesn't replace.
        """
        check_initialization()
        start_time = time.time()

        # Read file content
        filepath_obj = Path(filepath)
        with open(filepath_obj, 'r', encoding='utf-8') as f:
            code = f.read()

        # Calculate relative path
        if base_path:
            try:
                rel_path = filepath_obj.relative_to(Path(base_path))
            except ValueError:
                rel_path = filepath_obj
        else:
            rel_path = filepath_obj.name

        # Generate MD5-based source ID
        source_id = generate_source_id(code, str(rel_path))

        # Use relative path with forward slashes for inDocument (e.g., "path/to/file.html")
        in_file = str(rel_path).replace('\\', '/')

        # Load HTML code - use load_html_from_string directly like Python extractor
        from reter import owl_rete_cpp

        # First parse to get errors
        _, errors = owl_rete_cpp.parse_html_code(code, in_file)

        # Then load directly into network with source tracking
        wme_count = owl_rete_cpp.load_html_from_string(self.reasoner.network, code, in_file, source_id)

        time_ms = (time.time() - start_time) * 1000
        self._dirty = True  # Mark instance as modified

        # Convert errors to list of dicts for consistency
        error_list = []
        for err in errors:
            error_list.append({
                "line": err.get("line", 0),
                "column": err.get("column", 0),
                "message": err.get("message", "Unknown error")
            })

        return wme_count, source_id, time_ms, error_list

    def load_html_code(self, code: str, source: str = "document", progress_callback: Optional[Callable[[int, int, str], None]] = None) -> Tuple[int, str, float, List[str]]:
        """
        Load HTML code string and add semantic facts to RETER

        Args:
            code: HTML source code as string
            source: Source ID for tracking (can be document name, path, or path@timestamp)
            progress_callback: Optional callback function(items_processed, total_items, message)

        Returns:
            Tuple[int, str, float, list]: (wme_count, source, time_ms, errors)

        Raises:
            Exception: If loading fails

        Same as load_html_file() but takes code as string.
        Useful for loading HTML snippets or dynamically generated content.

        INCREMENTAL: Adds to existing knowledge.
        """
        check_initialization()
        start_time = time.time()

        # Extract file path from source (strip timestamp, MD5)
        in_file = source

        # Strip MD5 prefix if present (format: "md5hash|path")
        if '|' in in_file:
            in_file = in_file.split('|', 1)[1]

        # Strip timestamp suffix if present (format: "path@timestamp")
        if '@' in in_file:
            in_file = in_file.split('@', 1)[0]

        # Normalize path separators to forward slashes
        in_file = in_file.replace('\\', '/')

        # Load HTML code - use load_html_from_string directly (C++ derives module name from in_file)
        from reter import owl_rete_cpp

        # First parse to get errors
        _, errors = owl_rete_cpp.parse_html_code(code, in_file)

        # Then load directly into network with source tracking
        wme_count = owl_rete_cpp.load_html_from_string(self.reasoner.network, code, in_file, source)

        time_ms = (time.time() - start_time) * 1000
        self._dirty = True  # Mark instance as modified

        # Convert errors to list of dicts for consistency
        error_list = []
        for err in errors:
            error_list.append({
                "line": err.get("line", 0),
                "column": err.get("column", 0),
                "message": err.get("message", "Unknown error")
            })

        return wme_count, source, time_ms, error_list

    def load_html_directory(self, directory: str, recursive: bool = True, exclude_patterns: Optional[List[str]] = None, progress_callback: Optional[Callable[[int, int, str], None]] = None) -> Tuple[int, Dict[str, List[str]], float]:
        """
        Load all HTML files from a directory with optional exclusion patterns

        Args:
            directory: Path to directory containing HTML files
            recursive: If True, recursively scan subdirectories
            exclude_patterns: List of patterns to exclude (e.g., ["test_*.html", "node_modules/**/*.html"])
            progress_callback: Optional callback function(items_processed, total_items, message)

        Returns:
            Tuple[int, dict, float]: (total_wmes, all_errors, time_ms)
            where all_errors is a dict mapping filepath -> list of errors

        Raises:
            Exception: If loading fails

        INCREMENTAL: Adds to existing knowledge.
        """
        return self._load_directory_generic(
            directory=directory,
            extensions=["*.html", "*.htm"],
            default_excludes=["node_modules"],
            load_file_func=self.load_html_file,
            recursive=recursive,
            exclude_patterns=exclude_patterns,
            progress_callback=progress_callback
        )

    def load_csharp_file(self, filepath: str, base_path: Optional[str] = None) -> Tuple[int, str, float, List[str]]:
        """
        Load C# source file and add semantic facts to RETER

        Args:
            filepath: Path to C# file to load
            base_path: Optional base path for calculating relative path (defaults to filepath's parent)

        Returns:
            Tuple[int, str, float, list]: (wme_count, source_id, time_ms, errors)

        Raises:
            Exception: If loading fails

        Extracts and adds facts about:
        - Classes, interfaces, structs, enums
        - Methods, properties, fields, events
        - Inheritance relationships
        - Method calls
        - Using directives (imports)
        - Attributes (decorators)
        - Parameters and return types
        - Try/catch/finally blocks
        - Throw and return statements

        Source ID is generated as MD5 hash of content | relative path.
        INCREMENTAL: Adds to existing knowledge, doesn't replace.
        """
        check_initialization()
        start_time = time.time()

        # Read file content
        filepath_obj = Path(filepath)
        with open(filepath_obj, 'r', encoding='utf-8') as f:
            code = f.read()

        # Calculate relative path
        if base_path:
            try:
                rel_path = filepath_obj.relative_to(Path(base_path))
            except ValueError:
                rel_path = filepath_obj
        else:
            rel_path = filepath_obj.name

        # Generate MD5-based source ID
        source_id = generate_source_id(code, str(rel_path))

        # Use relative path with forward slashes for inFile (e.g., "path/to/file.cs")
        in_file = str(rel_path).replace('\\', '/')

        # Load C# code - use the C++ bindings
        from reter import owl_rete_cpp

        # Use load_csharp_from_string with in_file and source_id
        wme_count = owl_rete_cpp.load_csharp_from_string(
            self.reasoner.network,
            code,
            in_file,
            source_id
        )

        time_ms = (time.time() - start_time) * 1000
        self._dirty = True  # Mark instance as modified

        # C# parser doesn't return errors in the same format as Python/JS
        # Return empty list for now
        error_list: List[str] = []

        return wme_count, source_id, time_ms, error_list

    def load_csharp_code(self, code: str, source: str = "namespace", progress_callback: Optional[Callable[[int, int, str], None]] = None) -> Tuple[int, str, float, List[str]]:
        """
        Load C# code string and add semantic facts to RETER

        Args:
            code: C# source code as string
            source: Source ID for tracking (can be namespace name, path, or path@timestamp)
            progress_callback: Optional callback function(items_processed, total_items, message)

        Returns:
            Tuple[int, str, float, list]: (wme_count, source, time_ms, errors)

        Raises:
            Exception: If loading fails

        Same as load_csharp_file() but takes code as string.
        Useful for loading code snippets or dynamically generated code.

        INCREMENTAL: Adds to existing knowledge.
        """
        check_initialization()
        start_time = time.time()

        # Extract file path from source (strip timestamp, MD5)
        in_file = source

        # Strip MD5 prefix if present (format: "md5hash|path")
        if '|' in in_file:
            in_file = in_file.split('|', 1)[1]

        # Strip timestamp suffix if present (format: "path@timestamp")
        if '@' in in_file:
            in_file = in_file.split('@', 1)[0]

        # Normalize path separators to forward slashes
        in_file = in_file.replace('\\', '/')

        # Load C# code - use the C++ bindings (C++ derives namespace name from in_file)
        from reter import owl_rete_cpp

        # Use load_csharp_from_string which now supports source_id
        wme_count = owl_rete_cpp.load_csharp_from_string(
            self.reasoner.network,
            code,
            in_file,
            source
        )

        time_ms = (time.time() - start_time) * 1000
        self._dirty = True  # Mark instance as modified

        # C# parser doesn't return errors in the same format as Python/JS
        # Return empty list for now
        error_list: List[str] = []

        return wme_count, source, time_ms, error_list

    def load_csharp_directory(self, directory: str, recursive: bool = True, exclude_patterns: Optional[List[str]] = None, progress_callback: Optional[Callable[[int, int, str], None]] = None) -> Tuple[int, Dict[str, List[str]], float]:
        """
        Load all C# files from a directory with optional exclusion patterns

        Args:
            directory: Path to directory containing C# files
            recursive: If True, recursively scan subdirectories
            exclude_patterns: List of patterns to exclude (e.g., ["*.Designer.cs", "obj/**/*.cs"])
            progress_callback: Optional callback function(items_processed, total_items, message)

        Returns:
            Tuple[int, dict, float]: (total_wmes, all_errors, time_ms)
            where all_errors is a dict mapping filepath -> list of errors

        Raises:
            Exception: If loading fails

        INCREMENTAL: Adds to existing knowledge.
        """
        return self._load_directory_generic(
            directory=directory,
            extensions=["*.cs"],
            default_excludes=["bin", "obj"],
            load_file_func=self.load_csharp_file,
            recursive=recursive,
            exclude_patterns=exclude_patterns,
            progress_callback=progress_callback
        )

    def load_cpp_file(self, filepath: str, base_path: Optional[str] = None) -> Tuple[int, str, float, List[str]]:
        """
        Load C++ source file and add semantic facts to RETER

        Args:
            filepath: Path to C++ file to load
            base_path: Optional base path for calculating relative path (defaults to filepath's parent)

        Returns:
            Tuple[int, str, float, list]: (wme_count, source_id, time_ms, errors)

        Raises:
            Exception: If loading fails

        Extracts and adds facts about:
        - Classes, structs, namespaces
        - Methods, functions, constructors, destructors
        - Inheritance relationships
        - Function calls
        - Using directives (imports)
        - Templates
        - Parameters and return types
        - Try/catch/throw blocks
        - Enums and enumerators
        - Literals (for magic number detection)

        Source ID is generated as MD5 hash of content | relative path.
        INCREMENTAL: Adds to existing knowledge, doesn't replace.
        """
        check_initialization()
        start_time = time.time()

        # Read file content
        filepath_obj = Path(filepath)
        with open(filepath_obj, 'r', encoding='utf-8') as f:
            code = f.read()

        # Calculate relative path
        if base_path:
            try:
                rel_path = filepath_obj.relative_to(Path(base_path))
            except ValueError:
                rel_path = filepath_obj
        else:
            rel_path = filepath_obj.name

        # Generate MD5-based source ID
        source_id = generate_source_id(code, str(rel_path))

        # Use relative path with forward slashes for inFile (e.g., "path/to/file.cpp")
        in_file = str(rel_path).replace('\\', '/')

        # Load C++ code - use the C++ bindings
        from reter import owl_rete_cpp

        # Use load_cpp_from_string with in_file and source_id
        wme_count = owl_rete_cpp.load_cpp_from_string(
            self.reasoner.network,
            code,
            in_file,
            source_id
        )

        time_ms = (time.time() - start_time) * 1000
        self._dirty = True  # Mark instance as modified

        # C++ parser doesn't return errors in the same format as Python/JS
        # Return empty list for now
        error_list: List[str] = []

        return wme_count, source_id, time_ms, error_list

    def load_cpp_code(self, code: str, source: str = "namespace", progress_callback: Optional[Callable[[int, int, str], None]] = None) -> Tuple[int, str, float, List[str]]:
        """
        Load C++ code string and add semantic facts to RETER

        Args:
            code: C++ source code as string
            source: Source ID for tracking (can be namespace name, path, or path@timestamp)
            progress_callback: Optional callback function(items_processed, total_items, message)

        Returns:
            Tuple[int, str, float, list]: (wme_count, source, time_ms, errors)

        Raises:
            Exception: If loading fails

        Same as load_cpp_file() but takes code as string.
        Useful for loading code snippets or dynamically generated code.

        INCREMENTAL: Adds to existing knowledge.
        """
        check_initialization()
        start_time = time.time()

        # Extract file path from source (strip timestamp, MD5)
        in_file = source

        # Strip MD5 prefix if present (format: "md5hash|path")
        if '|' in in_file:
            in_file = in_file.split('|', 1)[1]

        # Strip timestamp suffix if present (format: "path@timestamp")
        if '@' in in_file:
            in_file = in_file.split('@', 1)[0]

        # Normalize path separators (use forward slashes)
        in_file = in_file.replace('\\', '/')

        # Load C++ code - use the C++ bindings
        from reter import owl_rete_cpp

        # Use load_cpp_from_string which supports source_id
        wme_count = owl_rete_cpp.load_cpp_from_string(
            self.reasoner.network,
            code,
            in_file,
            source
        )

        time_ms = (time.time() - start_time) * 1000
        self._dirty = True  # Mark instance as modified

        # C++ parser doesn't return errors in the same format as Python/JS
        # Return empty list for now
        error_list: List[str] = []

        return wme_count, source, time_ms, error_list

    def load_cpp_directory(self, directory: str, recursive: bool = True, exclude_patterns: Optional[List[str]] = None, progress_callback: Optional[Callable[[int, int, str], None]] = None) -> Tuple[int, Dict[str, List[str]], float]:
        """
        Load all C++ files from a directory with optional exclusion patterns

        Args:
            directory: Path to directory containing C++ files
            recursive: If True, recursively scan subdirectories
            exclude_patterns: List of patterns to exclude (e.g., ["test_*.cpp", "build/**/*.cpp"])
            progress_callback: Optional callback function(items_processed, total_items, message)

        Returns:
            Tuple[int, dict, float]: (total_wmes, all_errors, time_ms)
            where all_errors is a dict mapping filepath -> list of errors

        Raises:
            Exception: If loading fails

        INCREMENTAL: Adds to existing knowledge.
        """
        return self._load_directory_generic(
            directory=directory,
            extensions=["*.cpp", "*.cc", "*.cxx", "*.c++", "*.hpp", "*.hh", "*.hxx", "*.h++", "*.h"],
            default_excludes=["CMakeFiles", "build", "cmake-build-"],
            load_file_func=self.load_cpp_file,
            recursive=recursive,
            exclude_patterns=exclude_patterns,
            progress_callback=progress_callback
        )

    def get_all_sources(self) -> Tuple[List[str], float]:
        """
        Get list of all source identifiers currently loaded in RETER

        Returns all source IDs that have been used with add_ontology(),
        add_ontology_file(), or load_python_file(). Useful for:
        - Tracking what knowledge fragments are loaded
        - Deciding which sources to forget
        - Understanding knowledge base composition

        Returns:
            Tuple[List[str], float]: (sources, time_ms)

        Raises:
            DefaultInstanceNotInitialised: If server initialization not complete
            Exception: If retrieval fails
        """
        check_initialization()
        start_time = time.time()
        sources = safe_cpp_call(self.reasoner.get_all_sources)
        time_ms = (time.time() - start_time) * 1000
        return sources, time_ms

    def get_facts_from_source(self, source: str) -> Tuple[List[str], str, float]:
        """
        Get all fact IDs associated with a specific source

        Returns the internal fact IDs (WME identifiers) that were added
        from a particular source. Useful for debugging and understanding
        what knowledge came from which source.

        Args:
            source: Source identifier to query

        Returns:
            Tuple[List[str], str, float]: (fact_ids, source, time_ms)

        Raises:
            DefaultInstanceNotInitialised: If server initialization not complete
            Exception: If retrieval fails
        """
        check_initialization()
        start_time = time.time()
        fact_ids = safe_cpp_call(self.reasoner.get_facts_from_source, source)
        time_ms = (time.time() - start_time) * 1000
        return fact_ids, source, time_ms

    def forget_source(self, source: str) -> Tuple[str, float]:
        """
        Remove all facts from a source (selective forgetting)

        This is RETER's key memory management feature - allows selective
        forgetting of knowledge fragments by their source identifier.

        Args:
            source: Source identifier to forget (all facts from this source will be removed)

        Returns:
            Tuple[str, float]: (source, time_ms)

        Raises:
            DefaultInstanceNotInitialised: If server initialization not complete
            Exception: If removal fails
        """
        check_initialization()
        start_time = time.time()
        safe_cpp_call(self.reasoner.remove_source, source)
        time_ms = (time.time() - start_time) * 1000
        self._dirty = True  # Mark instance as modified
        return source, time_ms

    def save_network(self, filename: str) -> Tuple[bool, str, float]:
        """
        Save entire RETER network state to binary file

        Serializes the complete in-memory network including:
        - All facts (WMEs)
        - All rules
        - Inference network structure
        - Source mappings

        Args:
            filename: Path to save file

        Returns:
            Tuple[bool, str, float]: (success, filename, time_ms)

        Raises:
            DefaultInstanceNotInitialised: If server initialization not complete
            ReterSaveError: If save operation fails
        """
        check_initialization()
        start_time = time.time()
        # Use network.save() method from C++ bindings - wrap with safe call for C++ protection
        success = safe_cpp_call(self.reasoner.network.save, filename)
        time_ms = (time.time() - start_time) * 1000

        if not success:
            raise ReterSaveError(f"Failed to save network to {filename} - file may be read-only or path invalid")

        # Mark instance as clean and update save time on successful save
        self._dirty = False
        self._last_save_time = time.time()

        return success, filename, time_ms

    def load_network(self, filename: str) -> Tuple[bool, str, float]:
        """
        Load RETER network state from binary file

        Restores a previously saved network state, replacing
        the current in-memory network completely

        Args:
            filename: Path to saved file

        Returns:
            Tuple[bool, str, float]: (success, filename, time_ms)

        Raises:
            DefaultInstanceNotInitialised: If server initialization not complete
            ReterFileNotFoundError: If file does not exist
            ReterLoadError: If load operation fails
        """
        check_initialization()
        from pathlib import Path

        start_time = time.time()

        try:
            # Check if file exists
            if not Path(filename).exists():
                time_ms = (time.time() - start_time) * 1000
                raise ReterFileNotFoundError(f"File not found: {filename}")

            # Use network.load() method from C++ bindings - wrap with safe call for C++ protection
            success = safe_cpp_call(self.reasoner.network.load, filename)
            time_ms = (time.time() - start_time) * 1000

            if not success:
                raise ReterLoadError(f"Failed to load network from {filename} - file may be corrupted")

            # Mark instance as clean after successful load
            self._dirty = False
            self._last_save_time = time.time()

            return success, filename, time_ms

        except ReterFileError:
            # Re-raise our own exceptions without wrapping
            raise
        except Exception as e:
            time_ms = (time.time() - start_time) * 1000
            raise ReterLoadError(f"Load network error: {str(e)}") from e

    def check_consistency(self) -> Tuple[bool, List[str], float]:
        """
        Check for inconsistencies in the ontology

        Returns:
            Tuple[bool, List[str], float]: (is_consistent, inconsistencies, time_ms)

        Raises:
            DefaultInstanceNotInitialised: If server initialization not complete
            Exception: If consistency check fails
        """
        check_initialization()
        start_time = time.time()
        # Wrap with safe call for C++ protection
        result = safe_cpp_call(self.reasoner.check_consistency)
        is_consistent, inconsistencies = result
        time_ms = (time.time() - start_time) * 1000
        return is_consistent, inconsistencies, time_ms

    def is_dirty(self) -> bool:
        """
        Check if this instance has unsaved changes.

        Returns:
            True if instance has been modified since last save, False otherwise
        """
        return self._dirty

    def mark_clean(self) -> None:
        """
        Mark this instance as having no unsaved changes.
        Called after successful save operations.
        """
        self._dirty = False
        self._last_save_time = time.time()

    def get_last_save_time(self) -> float:
        """
        Get the timestamp of the last save operation.

        Returns:
            Unix timestamp of last save
        """
        return self._last_save_time

    # =========================================================================
    # Entity Accumulation API (for cross-file deduplication)
    # =========================================================================

    def begin_entity_accumulation(self) -> None:
        """
        Begin entity accumulation mode for deduplicating entities across multiple files.

        When loading multiple files (especially C++ header/source pairs), the same
        entity (method, class, etc.) may appear multiple times with different attributes.
        Entity accumulation mode collects these and merges them into single facts.

        Usage:
            reter.begin_entity_accumulation()
            try:
                for file in files:
                    reter.load_cpp_file(file, base_path)
            finally:
                reter.end_entity_accumulation()  # Finalizes and adds merged facts

        While active:
        - Entity facts (instance_of facts) are accumulated instead of added directly
        - Attributes are merged using intelligent strategies (prefer longer docs, OR booleans, etc.)
        - Non-entity facts (role assertions) are added directly as usual

        Call end_entity_accumulation() to finalize and add all merged entity facts.
        """
        safe_cpp_call(self.reasoner.network.begin_entity_accumulation)

    def end_entity_accumulation(self) -> None:
        """
        End entity accumulation mode and add all merged entity facts to the network.

        This finalizes the accumulated entities by:
        1. Merging collected attribute values (e.g., comma-separated file lists)
        2. Creating one fact per unique entity with merged attributes
        3. Adding these facts to the network

        Merge strategies applied:
        - Documentation: Prefer longer (more complete)
        - Boolean flags: OR operation (true if any occurrence is true)
        - Type info: Prefer definition over declaration
        - File location: Collect all unique files as comma-separated list
        - Start line: Keep first seen
        - End line: Keep last seen (definition has body)

        Must be called after begin_entity_accumulation(), even if no files were loaded.
        """
        safe_cpp_call(self.reasoner.network.end_entity_accumulation)
        # Resolve pending maybeCalls after all files are loaded
        safe_cpp_call(self.reasoner.network.resolve_maybe_calls)
        # Mark dirty since we've modified the network
        self._dirty = True

    def end_entity_accumulation_with_progress(
        self,
        batch_size: int = 1000,
        progress_callback: callable = None
    ) -> int:
        """
        End entity accumulation mode with progress reporting.

        Processes facts in batches and calls progress_callback after each batch.

        Args:
            batch_size: Number of facts to process per batch (default 1000)
            progress_callback: Optional callback(processed, total) called after each batch

        Returns:
            Total number of facts processed

        Example:
            def on_progress(processed, total):
                print(f"Inserting facts: {processed}/{total} ({100*processed//total}%)")

            total = reter.end_entity_accumulation_with_progress(
                batch_size=1000,
                progress_callback=on_progress
            )
        """
        total = 0
        while True:
            total_count, processed_count = safe_cpp_call(
                self.reasoner.network.end_entity_accumulation_batched,
                batch_size
            )

            # First call sets total
            if total == 0:
                total = total_count

            # Report progress
            if progress_callback and total > 0:
                progress_callback(processed_count, total)

            # Done when all processed
            if processed_count >= total_count:
                break

        # Resolve pending maybeCalls after all files are loaded
        safe_cpp_call(self.reasoner.network.resolve_maybe_calls)
        # Mark dirty since we've modified the network
        self._dirty = True
        return total

    def is_entity_accumulation_active(self) -> bool:
        """
        Check if entity accumulation mode is currently active.

        Returns:
            True if begin_entity_accumulation() has been called without
            a corresponding end_entity_accumulation()
        """
        return safe_cpp_call(self.reasoner.network.is_entity_accumulation_active)

    def accumulated_entity_count(self) -> int:
        """
        Get number of entities currently accumulated (before finalization).

        Returns:
            Number of unique entities collected so far
        """
        return safe_cpp_call(self.reasoner.network.accumulated_entity_count)

    def shutdown(self):
        """
        Gracefully shutdown this RETER instance.

        Call this before destroying the instance to prevent resource leaks.
        """
        # No async resources to clean up in synchronous version
        # The operation lock is automatically released when operations complete
        print(f"  ReterWrapper shutdown complete", file=sys.stderr)
