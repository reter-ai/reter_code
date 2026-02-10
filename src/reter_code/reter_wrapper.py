"""
RETER Integration Wrapper

Provides a clean Python interface to RETER's C++ reasoning engine
using the AI lexer variant for natural language DL syntax.
"""

import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Any, Tuple, Dict, Set, Callable, TypeVar

T = TypeVar('T')
import functools

from reter import Reter

from .logging_config import configure_logger_for_debug_trace

# Import exceptions from separate module (re-export for backward compatibility)
from .reter_exceptions import (
    ReterError,
    ReterFileError,
    ReterFileNotFoundError,
    ReterSaveError,
    ReterLoadError,
    ReterQueryError,
    ReterOntologyError,
    DefaultInstanceNotInitialised,
)

# Import utilities from separate module (re-export for backward compatibility)
from .reter_utils import (
    set_initialization_in_progress,
    set_initialization_complete,
    is_initialization_complete,
    check_initialization,
    DEBUG_MAX_VALUE_LEN,
    RETER_REQL_TIMEOUT_MS,
    _shorten_value,
    _format_args,
    safe_cpp_call,
    generate_source_id,
)

# Import loader mixins from separate module
from .reter_loaders import ReterLoaderMixin

# Configure module logger to also write to debug_trace.log
logger = configure_logger_for_debug_trace(__name__)

# Re-export for backward compatibility
__all__ = [
    # Exceptions
    "ReterError",
    "ReterFileError",
    "ReterFileNotFoundError",
    "ReterSaveError",
    "ReterLoadError",
    "ReterQueryError",
    "ReterOntologyError",
    "DefaultInstanceNotInitialised",
    # Utilities
    "set_initialization_in_progress",
    "set_initialization_complete",
    "is_initialization_complete",
    "check_initialization",
    "safe_cpp_call",
    "generate_source_id",
    # Main class
    "ReterWrapper",
]


class ReterWrapper(ReterLoaderMixin):
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

    ::: This is-in-layer Infrastructure-Layer.
    ::: This is a reasoning-engine.
    ::: This depends-on `reter.Reter`.
    ::: This is-in-process Main-Process.
    ::: This is stateful.
    ::: This holds-expensive-resource "rete-network".
    ::: This has-startup-order = 1.
    ::: This is not-serializable.
    ::: This has-singleton-scope.
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
            load_ontology: If True, automatically load code ontology.
                          Set to False when loading from snapshot (ontology already included).
        """
        logger.debug(f"ReterWrapper.__init__ starting... (load_ontology={load_ontology})")
        logger.debug("Creating Reter(variant='ai')...")
        try:
            self.reasoner = Reter(variant="ai")
            logger.debug("Reter created successfully")
        except Exception as e:
            logger.debug(f"ERROR creating Reter: {type(e).__name__}: {e}")
            raise
        self._session_stats = {"total_wmes": 0, "total_sources": 0}

        # Change tracking for auto-save
        self._dirty = False  # True if instance has unsaved changes
        self._last_save_time = time.time()  # Timestamp of last save

        # Automatically load code ontology for analysis (skip if loading from snapshot)
        if load_ontology:
            self._load_code_ontology()
        else:
            logger.debug("Skipping ontology load (will be loaded from snapshot)")

    def _load_code_ontology(self) -> None:
        """
        Load the code analysis ontology.

        This ontology defines entity type hierarchy for all supported languages
        (e.g., Every method is a function, Every class is a type, etc.).
        Generated from FACTS_COMPARISON.csv for 1:1 correspondence with C++ facts.

        Loaded with source "code_ontology" for potential selective forgetting.
        """
        try:
            ontology_path = Path(__file__).parent / "resources" / "code_ontology.cnl"
            if ontology_path.exists():
                with open(ontology_path, 'r', encoding='utf-8') as f:
                    ontology_content = f.read()
                wme_count = safe_cpp_call(self.reasoner.load_cnl, ontology_content, "code_ontology")
                self._session_stats["total_wmes"] += wme_count
                self._session_stats["total_sources"] += 1
                logger.debug(f"Code ontology loaded successfully ({wme_count} WMEs)")
            else:
                logger.debug(f"WARNING: Code ontology not found at {ontology_path}")
        except Exception as e:
            logger.debug(f"ERROR loading code ontology: {type(e).__name__}: {e}")

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

        ::: This is-exposed-via-ipc.
        ::: This communicates-sync.
        """
        check_initialization()
        start_time = time.time()

        # Unified storage: ReteNetwork handles hybrid mode internally
        # When in hybrid mode, add_fact() writes to delta automatically
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

        ::: This is-exposed-via-ipc.
        ::: This has-ipc-timeout = 300000.
        ::: This communicates-sync.
        """
        check_initialization()
        # Use default timeout if not specified
        if timeout_ms is None:
            timeout_ms = RETER_REQL_TIMEOUT_MS

        # Unified storage: ReteNetwork handles hybrid mode internally
        return safe_cpp_call(self.reasoner.reql, query, timeout_ms)

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

        ::: This is-exposed-via-ipc.
        ::: This communicates-sync.
        """
        check_initialization()
        start_time = time.time()

        # Unified storage: ReteNetwork handles hybrid mode internally
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

        # Unified storage: ReteNetwork handles hybrid mode internally
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

        ::: This is-exposed-via-ipc.
        ::: This communicates-sync.
        """
        check_initialization()
        start_time = time.time()

        # Normalize path separators in source ID for cross-platform consistency
        # Try forward slashes first (new format), then backslashes (legacy format)
        normalized_source = source.replace('\\', '/')
        safe_cpp_call(self.reasoner.remove_source, normalized_source)

        # Also try backslash version in case RETER has legacy format
        if '/' in source:
            backslash_source = source.replace('/', '\\')
            try:
                safe_cpp_call(self.reasoner.remove_source, backslash_source)
            except Exception:
                pass  # Ignore if not found with backslashes

        time_ms = (time.time() - start_time) * 1000
        self._dirty = True  # Mark instance as modified
        return source, time_ms

    def save_network(self, filename: str, progress_callback: Optional[Callable[[int], None]] = None) -> Tuple[bool, str, float]:
        """
        Save entire RETER network state to binary file

        Serializes the complete in-memory network including:
        - All facts (WMEs)
        - All rules
        - Inference network structure
        - Source mappings

        Args:
            filename: Path to save file
            progress_callback: Optional callback receiving percent (0-100)

        Returns:
            Tuple[bool, str, float]: (success, filename, time_ms)

        Raises:
            DefaultInstanceNotInitialised: If server initialization not complete
            ReterSaveError: If save operation fails

        ::: This is-exposed-via-ipc.
        ::: This communicates-sync.
        """
        check_initialization()
        start_time = time.time()

        # CRITICAL: In hybrid mode, use incremental save to avoid corrupting the base
        # The ReteNetwork in hybrid mode only contains synced files (not the full base),
        # so using network.save() would write a partial/corrupt snapshot.
        if self.is_hybrid_mode():
            success, delta_path, time_ms = self.save_incremental()
            logger.debug(f"save_network: redirected to save_incremental in hybrid mode")
            return success, delta_path, time_ms

        # Use network.save() method from C++ bindings - wrap with safe call for C++ protection
        success = safe_cpp_call(self.reasoner.network.save, filename, progress_callback)
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

        # Unified storage: ReteNetwork handles hybrid mode internally
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
    # Unified Storage API (Hybrid mode integrated into ReteNetwork)
    # =========================================================================

    def open_hybrid(self, filename: str) -> Tuple[bool, str, float]:
        """
        Open a snapshot in hybrid mode for incremental saves.

        This enables delta-based persistence where modifications are
        written to a small delta journal file instead of rewriting the
        entire base snapshot. Much faster for frequent saves.

        Args:
            filename: Path to base snapshot file (.reter)

        Returns:
            Tuple[bool, str, float]: (success, filename, time_ms)

        After calling this:
        - save_incremental() writes only changes to .delta file (fast)
        - compact() merges delta into base when needed
        - The reasoner continues to work normally
        """
        check_initialization()
        from pathlib import Path
        import glob as glob_module

        start_time = time.time()

        try:
            # Check for base file OR versioned files (.v1, .v2, etc.)
            base_exists = Path(filename).exists()
            versioned_exists = bool(glob_module.glob(f"{filename}.v*"))
            if not base_exists and not versioned_exists:
                time_ms = (time.time() - start_time) * 1000
                raise ReterFileNotFoundError(f"File not found: {filename}")

            # Use unified ReteNetwork.open() API
            success = self.reasoner.network.open(filename)
            time_ms = (time.time() - start_time) * 1000

            if not success:
                raise ReterLoadError(f"Failed to open hybrid network from {filename}")

            self._dirty = False
            self._last_save_time = time.time()

            logger.debug(f"Opened hybrid network: base={self.reasoner.network.base_fact_count()}, "
                     f"delta={self.reasoner.network.delta_fact_count()}")

            return success, filename, time_ms

        except ReterFileError:
            raise
        except Exception as e:
            time_ms = (time.time() - start_time) * 1000
            raise ReterLoadError(f"Open hybrid error: {str(e)}") from e

    def is_hybrid_mode(self) -> bool:
        """Check if hybrid/incremental mode is enabled."""
        return self.reasoner.network.is_hybrid()

    def save_incremental(self) -> Tuple[bool, str, float]:
        """
        Save changes incrementally using delta journal (fast).

        Only available when hybrid mode is enabled via open_hybrid().
        Writes only the changes since last save to the delta file,
        which is much faster than full serialization.

        Returns:
            Tuple[bool, str, float]: (success, delta_path, time_ms)

        Raises:
            ReterSaveError: If not in hybrid mode or save fails
        """
        check_initialization()
        start_time = time.time()

        if not self.is_hybrid_mode():
            raise ReterSaveError("Not in hybrid mode. Call open_hybrid() first.")

        try:
            self.reasoner.network.save()
            time_ms = (time.time() - start_time) * 1000

            self._dirty = False
            self._last_save_time = time.time()

            delta_path = self.reasoner.network.delta_path()
            logger.debug(f"Incremental save: delta={self.reasoner.network.delta_fact_count()} facts, "
                     f"size={self.reasoner.network.delta_file_size()} bytes")

            return True, delta_path, time_ms

        except Exception as e:
            time_ms = (time.time() - start_time) * 1000
            raise ReterSaveError(f"Incremental save error: {str(e)}") from e

    def compact(self, progress_callback: Optional[Callable[[int], None]] = None) -> Tuple[bool, float]:
        """
        Compact delta journal into base snapshot.

        Merges all delta changes into the base file and resets the delta.
        Call this periodically when delta grows large.

        Args:
            progress_callback: Optional callback receiving percent (0-100)

        Returns:
            Tuple[bool, float]: (success, time_ms)

        Raises:
            ReterSaveError: If not in hybrid mode or compact fails
        """
        check_initialization()
        start_time = time.time()

        if not self.is_hybrid_mode():
            raise ReterSaveError("Not in hybrid mode. Call open_hybrid() first.")

        try:
            self.reasoner.network.compact(progress_callback)
            time_ms = (time.time() - start_time) * 1000

            logger.debug(f"Compacted: base now has {self.reasoner.network.base_fact_count()} facts")

            return True, time_ms

        except Exception as e:
            time_ms = (time.time() - start_time) * 1000
            raise ReterSaveError(f"Compact error: {str(e)}") from e

    def needs_compaction(self, threshold_ratio: float = 0.2) -> bool:
        """
        Check if delta journal should be compacted.

        Args:
            threshold_ratio: Compact when delta_facts > base_facts * threshold_ratio

        Returns:
            True if compaction is recommended
        """
        if not self.is_hybrid_mode():
            return False

        # Check threshold in Python since C++ uses built-in thresholds
        base_facts = self.reasoner.network.base_fact_count()
        delta_facts = self.reasoner.network.delta_fact_count()

        # Compact when delta exceeds threshold ratio of base
        return delta_facts > base_facts * threshold_ratio

    def get_hybrid_stats(self) -> Dict[str, Any]:
        """
        Get statistics about hybrid network state.

        Returns:
            Dict with base_facts, delta_facts, deleted_facts, delta_file_size
        """
        if not self.is_hybrid_mode():
            return {"hybrid_mode": False}

        return {
            "hybrid_mode": True,
            "base_facts": self.reasoner.network.base_fact_count(),
            "delta_facts": self.reasoner.network.delta_fact_count(),
            "deleted_facts": self.reasoner.network.deleted_fact_count(),
            "delta_file_size": self.reasoner.network.delta_file_size(),
            "delta_path": self.reasoner.network.delta_path(),
        }

    def close_hybrid(self) -> None:
        """
        Close hybrid network and disable incremental mode.

        After calling this, use regular save_network()/load_network().
        """
        if self.is_hybrid_mode():
            try:
                self.reasoner.network.close()
            except Exception as e:
                logger.debug(f"Error closing hybrid network: {e}")

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
        # In hybrid mode WITHOUT materialization, entity accumulation works on empty ReteNetwork
        # New facts will be synced to hybrid network's delta after accumulation ends
        # This avoids slow materialization - facts are queryable via Arrow without RETE
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
        logger.debug("ReterWrapper shutdown complete")
