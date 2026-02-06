"""
RETER Operations Service

Handles all core RETER knowledge operations including:
- Adding knowledge (ontology and Python code)
- Querying knowledge
- Source management (checking validity, forgetting, reloading)
- Consistency checking

Extracted from LogicalThinkingServer as part of God Class refactoring.
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

from .instance_manager import InstanceManager
from .utils import make_path_relative
from ..logging_config import is_stderr_suppressed

logger = logging.getLogger(__name__)


class ReterOperations:
    """
    Service for core RETER knowledge operations.

    ::: This is-in-layer Service-Layer.
    ::: This is a service.
    ::: This is-in-process Main-Process.
    ::: This is stateless.

    This service handles:
    - Adding knowledge incrementally (ontology, Python code)
    - Querying knowledge (REQL, DL)
    - Source validity checking and management
    - Selective forgetting
    - Knowledge base consistency checking
    - Reloading modified sources

    Responsibilities:
    - Manage RETER knowledge operations with thread safety
    - Track file-based source modifications
    - Provide incremental reasoning capabilities
    """

    def __init__(self, instance_manager: InstanceManager):
        """
        Initialize the RETER operations service.

        Args:
            instance_manager: InstanceManager for accessing RETER instances
        """
        self.instance_manager = instance_manager

    # DISABLED: This functionality caused queue flooding deadlock
    # When multiple concurrent quick_query calls each called check_source_validity,
    # they all queued get_all_sources() operations, causing massive queue contention
    """
    def check_source_validity(self, instance_name: str) -> Dict[str, Any]:
        '''
        Check validity of file-based sources in RETER instance.

        Validates that files referenced in source IDs still exist and haven't been
        modified since they were loaded into RETER.

        Source ID format for files: `<filepath>@<iso_timestamp>`
        Example: `/path/to/file.py@2025-01-12T10:30:45.123456`

        Args:
            instance_name: RETER instance name to check

        Returns:
            Dict containing:
                valid: bool - True if all sources are up-to-date
                outdated_sources: List of sources with newer file modifications
                deleted_sources: List of sources where files no longer exist
                warnings: List of human-readable warning messages
        '''
        try:
            # Lazy load instance if snapshot is available
            self.instance_manager.ensure_instance_loaded(instance_name)

            # Get or create the RETER instance
            reter = self.instance_manager.get_or_create_instance(instance_name)

            # Get all sources from RETER (thread-safe at C++ level)
            try:
                sources_result = reter.get_all_sources()

                # get_all_sources() returns (sources_list, time_ms) tuple
                # where sources_list is the list of source IDs and time_ms is execution time
                if isinstance(sources_result, tuple) and len(sources_result) == 2:
                    sources = sources_result[0]
                    time_ms = sources_result[1]

                    # If sources is not a list, we can't iterate
                    if not isinstance(sources, list):
                        return {
                            "valid": True,
                            "outdated_sources": [],
                            "deleted_sources": [],
                            "warnings": [f"get_all_sources returned non-list: sources={sources!r} (type={type(sources).__name__}), time_ms={time_ms!r}"]
                        }
                else:
                    return {
                        "valid": True,
                        "outdated_sources": [],
                        "deleted_sources": [],
                        "warnings": [f"get_all_sources returned unexpected format: {sources_result!r}"]
                    }

            except Exception as e:
                import traceback
                traceback.print_exc()
                return {
                    "valid": True,
                    "outdated_sources": [],
                    "deleted_sources": [],
                    "warnings": [f"Exception in get_all_sources: {str(e)}"]
                }

            outdated_sources = []
            deleted_sources = []
            warnings = []

            # Double-check that sources is actually a list before iterating
            if not isinstance(sources, list):
                return {
                    "valid": True,
                    "outdated_sources": [],
                    "deleted_sources": [],
                    "warnings": [f"Sources is unexpectedly not a list at iteration point: {type(sources).__name__} = {sources!r}"]
                }

            try:
                for source_id in sources:
                    # Check if this is a file-based source (contains @ separator)
                    if "@" not in source_id:
                        continue  # Not a file-based source, skip

                    try:
                        # Parse the source ID
                        parts = source_id.rsplit("@", 1)  # Split from right to handle @ in filepath
                        if len(parts) != 2:
                            continue

                        filepath_str, timestamp_str = parts
                        filepath = Path(filepath_str)

                        # Parse the stored timestamp
                        stored_timestamp = datetime.fromisoformat(timestamp_str)

                        # Check if file exists
                        if not filepath.exists():
                            deleted_sources.append({
                                "source_id": source_id,
                                "filepath": str(filepath),
                                "stored_timestamp": timestamp_str
                            })
                            warnings.append(f"Source file deleted: {filepath}")
                            continue

                        # Get current file modification time
                        current_mtime = filepath.stat().st_mtime
                        current_timestamp = datetime.fromtimestamp(current_mtime)

                        # Compare timestamps (with small tolerance for float precision)
                        time_diff = (current_timestamp - stored_timestamp).total_seconds()
                        if time_diff > 0.001:  # File is newer than stored version
                            outdated_sources.append({
                                "source_id": source_id,
                                "filepath": str(filepath),
                                "stored_timestamp": timestamp_str,
                                "current_timestamp": current_timestamp.isoformat()
                            })
                            warnings.append(
                                f"Source file modified: {filepath} "
                                f"(stored: {stored_timestamp.strftime('%Y-%m-%d %H:%M:%S')}, "
                                f"current: {current_timestamp.strftime('%Y-%m-%d %H:%M:%S')})"
                            )

                    except Exception as e:
                        # Error parsing or checking this source, skip it
                        if not is_stderr_suppressed():
                            print(f"Error checking source '{source_id}': {e}", file=sys.stderr)
                        continue

            except TypeError as e:
                # Specific error for iteration problems
                return {
                    "valid": True,
                    "outdated_sources": [],
                    "deleted_sources": [],
                    "warnings": [f"Type error during iteration: {str(e)}. sources type: {type(sources).__name__}, sources value: {sources!r}"]
                }

            # Determine overall validity
            is_valid = len(outdated_sources) == 0 and len(deleted_sources) == 0

            # Add summary warning if needed
            if not is_valid:
                summary = f"Warning: RETER instance '{instance_name}' contains outdated information. "
                if outdated_sources:
                    summary += f"{len(outdated_sources)} file(s) have been modified. "
                if deleted_sources:
                    summary += f"{len(deleted_sources)} file(s) have been deleted."
                warnings.insert(0, summary)

            return {
                "valid": is_valid,
                "outdated_sources": outdated_sources,
                "deleted_sources": deleted_sources,
                "warnings": warnings
            }

        except Exception as e:
            return {
                "valid": True,  # Assume valid on error
                "outdated_sources": [],
                "deleted_sources": [],
                "warnings": [f"Error during source validation: {str(e)}"]
            }
    """

    def add_knowledge(
        self,
        instance_name: str,
        source: str,
        type: str = "ontology",
        source_id: str = None,
        ctx = None
    ) -> Dict[str, Any]:
        """
        Incrementally add knowledge to RETER's semantic memory.

        RETER is an incremental forward-chaining reasoner - knowledge accumulates!
        Each call ADDS facts/rules to the existing knowledge base (doesn't replace).

        Args:
            instance_name: RETER instance name
            source: Ontology content, file path, or single Python file path
            type: 'ontology' (DL/SWRL) or 'python' (single .py file analysis)
            source_id: Optional identifier for selective forgetting later
            ctx: Optional MCP Context for progress reporting

        Returns:
            success: Whether knowledge was successfully added
            items_added: Number of WMEs (facts/rules) added to RETER
            execution_time_ms: Time taken to add and process knowledge
            source_id: The source ID used (includes timestamp for files)
        """
        try:
            # Lazy load instance if snapshot is available
            self.instance_manager.ensure_instance_loaded(instance_name)

            # Get or create the RETER instance (thread-safe at C++ level)
            reter = self.instance_manager.get_or_create_instance(instance_name)

            # Determine the source_id based on whether source is a file
            final_source_id = None
            is_file = False

            if type == "ontology":
                source_path = Path(source)
                if source_path.exists() and source_path.is_file():
                    is_file = True
                    # Get file modification timestamp
                    mtime = source_path.stat().st_mtime
                    timestamp = datetime.fromtimestamp(mtime).isoformat()
                    # Create source ID with timestamp (relative path)
                    relative_path = make_path_relative(source_path.absolute())
                    final_source_id = f"{relative_path}@{timestamp}"
                else:
                    # Not a file, use provided source_id or generate one
                    final_source_id = source_id or f"source_{datetime.utcnow().timestamp()}"

            elif type == "python":
                source_path = Path(source)
                if source_path.is_dir():
                    return {
                        "success": False,
                        "items_added": 0,
                        "errors": ["Directory analysis not supported. Provide a single .py file path."],
                        "source_id": None
                    }
                if source_path.exists() and source_path.is_file():
                    is_file = True
                    # Get file modification timestamp
                    mtime = source_path.stat().st_mtime
                    timestamp = datetime.fromtimestamp(mtime).isoformat()
                    # Create source ID with timestamp (relative path)
                    relative_path = make_path_relative(source_path.absolute())
                    final_source_id = f"{relative_path}@{timestamp}"
                else:
                    return {
                        "success": False,
                        "items_added": 0,
                        "errors": [f"Python file not found: {source}"],
                        "source_id": None
                    }

            elif type in ("javascript", "html", "csharp", "cpp"):
                source_path = Path(source)
                if source_path.is_dir():
                    return {
                        "success": False,
                        "items_added": 0,
                        "errors": [f"Directory analysis not supported. Provide a single {type} file path."],
                        "source_id": None
                    }
                if source_path.exists() and source_path.is_file():
                    is_file = True
                    # Get file modification timestamp
                    mtime = source_path.stat().st_mtime
                    timestamp = datetime.fromtimestamp(mtime).isoformat()
                    # Create source ID with timestamp (relative path)
                    relative_path = make_path_relative(source_path.absolute())
                    final_source_id = f"{relative_path}@{timestamp}"
                else:
                    return {
                        "success": False,
                        "items_added": 0,
                        "errors": [f"{type.upper()} file not found: {source}"],
                        "source_id": None
                    }

            # Execute RETER operations (thread-safe at C++ level)
            if type == "ontology":
                if is_file:
                    # Load from file with timestamped source ID
                    with open(source, 'r', encoding='utf-8') as f:
                        content = f.read()
                    wme_count, returned_source, time_ms = reter.add_ontology(content, final_source_id)
                else:
                    # Treat source as ontology content
                    wme_count, returned_source, time_ms = reter.add_ontology(source, final_source_id)
            elif type == "python":
                # Analyze Python file with timestamped source ID
                with open(source, 'r', encoding='utf-8') as f:
                    code = f.read()

                # Create progress callback that reports to MCP client
                progress_callback = None
                if ctx:
                    def progress_callback(processed, total, msg):
                        # Report progress synchronously
                        try:
                            ctx.report_progress(processed, total)
                        except Exception as progress_error:
                            # Log if context reporting fails but don't interrupt
                            logger.debug(f"Progress reporting failed: {progress_error}")

                wme_count, returned_source, time_ms, parse_errors = reter.load_python_code(
                    code,
                    final_source_id,
                    progress_callback
                )

                # Format errors for response
                error_list = []
                if parse_errors:
                    for err in parse_errors:
                        error_list.append(f"Line {err.get('line', '?')}:{err.get('column', '?')} - {err.get('message', '?')}")

            elif type == "javascript":
                # Analyze JavaScript file with timestamped source ID
                with open(source, 'r', encoding='utf-8') as f:
                    code = f.read()

                wme_count, returned_source, time_ms, parse_errors = reter.load_javascript_code(
                    code,
                    final_source_id
                )

                # Format errors for response
                error_list = []
                if parse_errors:
                    for err in parse_errors:
                        if isinstance(err, dict):
                            error_list.append(f"Line {err.get('line', '?')}:{err.get('column', '?')} - {err.get('message', '?')}")
                        else:
                            error_list.append(str(err))

            elif type == "html":
                # Analyze HTML file with timestamped source ID
                with open(source, 'r', encoding='utf-8') as f:
                    code = f.read()

                wme_count, returned_source, time_ms, parse_errors = reter.load_html_code(
                    code,
                    final_source_id
                )

                # Format errors for response
                error_list = []
                if parse_errors:
                    for err in parse_errors:
                        if isinstance(err, dict):
                            error_list.append(f"Line {err.get('line', '?')}:{err.get('column', '?')} - {err.get('message', '?')}")
                        else:
                            error_list.append(str(err))

            elif type == "csharp":
                # Analyze C# file with timestamped source ID
                with open(source, 'r', encoding='utf-8') as f:
                    code = f.read()

                wme_count, returned_source, time_ms, parse_errors = reter.load_csharp_code(
                    code,
                    final_source_id
                )

                # Format errors for response
                error_list = []
                if parse_errors:
                    for err in parse_errors:
                        if isinstance(err, dict):
                            error_list.append(f"Line {err.get('line', '?')}:{err.get('column', '?')} - {err.get('message', '?')}")
                        else:
                            error_list.append(str(err))

            elif type == "cpp":
                # Analyze C++ file with timestamped source ID
                with open(source, 'r', encoding='utf-8') as f:
                    code = f.read()

                wme_count, returned_source, time_ms, parse_errors = reter.load_cpp_code(
                    code,
                    final_source_id
                )

                # Format errors for response
                error_list = []
                if parse_errors:
                    for err in parse_errors:
                        if isinstance(err, dict):
                            error_list.append(f"Line {err.get('line', '?')}:{err.get('column', '?')} - {err.get('message', '?')}")
                        else:
                            error_list.append(str(err))

            else:
                return {
                    "success": False,
                    "items_added": 0,
                    "errors": [f"Unknown knowledge type: {type}. Use 'ontology', 'python', 'javascript', 'html', 'csharp', or 'cpp'"],
                    "source_id": None
                }

            # Auto-save after successful knowledge addition
            # Ensure snapshot directory exists before saving
            self.instance_manager._persistence.snapshots_dir.mkdir(parents=True, exist_ok=True)
            snapshot_path = self.instance_manager._persistence.snapshots_dir / f"{instance_name}.reter"
            reter.save_network(str(snapshot_path))
            reter.mark_clean()

            return {
                "success": True,
                "items_added": wme_count,
                "execution_time_ms": time_ms,
                "errors": error_list if 'error_list' in locals() else [],
                "source_id": final_source_id
            }
        except Exception as e:
            return {
                "success": False,
                "items_added": 0,
                "errors": [str(e)],
                "source_id": None
            }

    # Supported file extensions (matching DefaultInstanceManager)
    PYTHON_EXTENSIONS = {".py"}
    JAVASCRIPT_EXTENSIONS = {".js", ".mjs", ".jsx", ".ts", ".tsx"}
    HTML_EXTENSIONS = {".html", ".htm"}
    CSHARP_EXTENSIONS = {".cs"}
    ALL_CODE_EXTENSIONS = PYTHON_EXTENSIONS | JAVASCRIPT_EXTENSIONS | HTML_EXTENSIONS | CSHARP_EXTENSIONS

    def add_external_directory(
        self,
        instance_name: str,
        directory: str,
        recursive: bool = True,
        exclude_patterns = None,
        ctx = None
    ) -> Dict[str, Any]:
        """
        Load all supported code files from a directory into RETER.

        Supports: Python (.py), JavaScript (.js, .mjs, .jsx, .ts, .tsx), HTML (.html, .htm), C# (.cs)

        Args:
            instance_name: RETER instance name
            directory: Path to directory containing code files
            recursive: Whether to recursively search subdirectories (default: True)
            exclude_patterns: List of patterns to exclude (e.g., ["test_*.py", "*/tests/*", "node_modules/*"])
            ctx: Optional MCP Context for progress reporting

        Returns:
            success: Whether operation succeeded
            files_loaded: Number of files successfully loaded (by type)
            total_files: Total number of files found (by type)
            total_wmes: Total WMEs added across all files
            errors: List of any errors encountered
            files_with_errors: List of files that failed to load
            execution_time_ms: Total time taken
        """
        import time
        import fnmatch
        start_time = time.time()

        # Reject "default" instance - it auto-syncs with RETER_PROJECT_ROOT
        if instance_name == "default":
            return {
                "success": False,
                "files_loaded": {"python": 0, "javascript": 0, "html": 0, "csharp": 0},
                "total_files": {"python": 0, "javascript": 0, "html": 0, "csharp": 0},
                "total_wmes": 0,
                "errors": [
                    "Cannot use 'default' instance with add_external_directory. "
                    "The 'default' instance auto-syncs with RETER_PROJECT_ROOT environment variable. "
                    "Use a different instance name (e.g., 'analysis', 'external') for loading external code."
                ],
                "files_with_errors": [],
                "execution_time_ms": 0
            }

        try:
            # Lazy load instance if snapshot is available
            self.instance_manager.ensure_instance_loaded(instance_name)

            # Get or create the RETER instance
            reter = self.instance_manager.get_or_create_instance(instance_name)

            directory_path = Path(directory)
            if not directory_path.exists():
                return {
                    "success": False,
                    "files_loaded": {"python": 0, "javascript": 0, "html": 0, "csharp": 0},
                    "total_files": {"python": 0, "javascript": 0, "html": 0, "csharp": 0},
                    "total_wmes": 0,
                    "errors": [f"Directory not found: {directory}"],
                    "files_with_errors": [],
                    "execution_time_ms": 0
                }

            if not directory_path.is_dir():
                return {
                    "success": False,
                    "files_loaded": {"python": 0, "javascript": 0, "html": 0, "csharp": 0},
                    "total_files": {"python": 0, "javascript": 0, "html": 0, "csharp": 0},
                    "total_wmes": 0,
                    "errors": [f"Path is not a directory: {directory}"],
                    "files_with_errors": [],
                    "execution_time_ms": 0
                }

            # Collect all supported files
            all_files = {"python": [], "javascript": [], "html": [], "csharp": []}
            exclude_patterns = exclude_patterns or []

            def should_exclude(file_path: Path) -> bool:
                """Check if file matches any exclude pattern."""
                rel_path = str(file_path.relative_to(directory_path))
                # Always exclude __pycache__, node_modules, bin, and obj directories
                if "__pycache__" in rel_path or "node_modules" in rel_path:
                    return True
                # Exclude C# build output directories
                if "/bin/" in rel_path or "/obj/" in rel_path or rel_path.startswith("bin/") or rel_path.startswith("obj/"):
                    return True
                for pattern in exclude_patterns:
                    if fnmatch.fnmatch(rel_path, pattern) or fnmatch.fnmatch(file_path.name, pattern):
                        return True
                return False

            # Scan for all file types
            for ext in self.ALL_CODE_EXTENSIONS:
                if recursive:
                    files = list(directory_path.rglob(f"*{ext}"))
                else:
                    files = list(directory_path.glob(f"*{ext}"))

                # Filter excluded files
                files = [f for f in files if not should_exclude(f)]

                # Categorize by type
                if ext in self.PYTHON_EXTENSIONS:
                    all_files["python"].extend(files)
                elif ext in self.JAVASCRIPT_EXTENSIONS:
                    all_files["javascript"].extend(files)
                elif ext in self.HTML_EXTENSIONS:
                    all_files["html"].extend(files)
                elif ext in self.CSHARP_EXTENSIONS:
                    all_files["csharp"].extend(files)

            # Load files and track results
            total_wmes = 0
            all_errors = {}
            files_loaded = {"python": 0, "javascript": 0, "html": 0, "csharp": 0}
            total_files_count = {"python": len(all_files["python"]),
                                 "javascript": len(all_files["javascript"]),
                                 "html": len(all_files["html"]),
                                 "csharp": len(all_files["csharp"])}

            # Progress tracking
            total_file_count = sum(len(files) for files in all_files.values())
            processed_count = 0

            # Enable entity accumulation for cross-file deduplication
            # This handles C++ header/source pairs, C# partial classes, etc.
            reter.begin_entity_accumulation()

            try:
                # Load Python files
                for py_file in all_files["python"]:
                    try:
                        wme_count, source_id, time_ms, file_errors = reter.load_python_file(str(py_file), str(directory_path))
                        total_wmes += wme_count
                        files_loaded["python"] += 1
                        if file_errors:
                            all_errors[str(py_file)] = [{"line": 0, "message": err} for err in file_errors]
                    except Exception as e:
                        all_errors[str(py_file)] = [{"line": 0, "message": str(e)}]
                    processed_count += 1
                    if ctx:
                        ctx.report_progress(processed_count, total_file_count)

                # Load JavaScript files
                for js_file in all_files["javascript"]:
                    try:
                        wme_count, source_id, time_ms, file_errors = reter.load_javascript_file(str(js_file), str(directory_path))
                        total_wmes += wme_count
                        files_loaded["javascript"] += 1
                        if file_errors:
                            all_errors[str(js_file)] = [{"line": 0, "message": err} for err in file_errors]
                    except Exception as e:
                        all_errors[str(js_file)] = [{"line": 0, "message": str(e)}]
                    processed_count += 1
                    if ctx:
                        ctx.report_progress(processed_count, total_file_count)

                # Load HTML files
                for html_file in all_files["html"]:
                    try:
                        wme_count, source_id, time_ms, file_errors = reter.load_html_file(str(html_file), str(directory_path))
                        total_wmes += wme_count
                        files_loaded["html"] += 1
                        if file_errors:
                            all_errors[str(html_file)] = [{"line": 0, "message": err} for err in file_errors]
                    except Exception as e:
                        all_errors[str(html_file)] = [{"line": 0, "message": str(e)}]
                    processed_count += 1
                    if ctx:
                        ctx.report_progress(processed_count, total_file_count)

                # Load C# files
                for cs_file in all_files["csharp"]:
                    try:
                        wme_count, source_id, time_ms, file_errors = reter.load_csharp_file(str(cs_file), str(directory_path))
                        total_wmes += wme_count
                        files_loaded["csharp"] += 1
                        if file_errors:
                            all_errors[str(cs_file)] = [{"line": 0, "message": err} for err in file_errors]
                    except Exception as e:
                        all_errors[str(cs_file)] = [{"line": 0, "message": str(e)}]
                    processed_count += 1
                    if ctx:
                        ctx.report_progress(processed_count, total_file_count)

            finally:
                # Finalize entity accumulation - merges duplicate entities
                accumulated_count = reter.accumulated_entity_count()
                reter.end_entity_accumulation()
                logger.info(f"Entity accumulation: {accumulated_count} unique entities merged")

            files_with_errors = list(all_errors.keys())

            # Format errors for response
            errors = []
            for filepath, file_errors in all_errors.items():
                for err in file_errors:
                    errors.append(f"{filepath} - Line {err.get('line', '?')}: {err.get('message', '?')}")

            # Save snapshot once at the end
            self.instance_manager._persistence.snapshots_dir.mkdir(parents=True, exist_ok=True)
            snapshot_path = self.instance_manager._persistence.snapshots_dir / f"{instance_name}.reter"
            reter.save_network(str(snapshot_path))
            reter.mark_clean()

            execution_time_ms = (time.time() - start_time) * 1000

            return {
                "success": True,
                "files_loaded": files_loaded,
                "total_files": total_files_count,
                "total_wmes": total_wmes,
                "errors": errors,
                "files_with_errors": files_with_errors,
                "execution_time_ms": execution_time_ms
            }

        except Exception as e:
            return {
                "success": False,
                "files_loaded": {"python": 0, "javascript": 0, "html": 0, "csharp": 0},
                "total_files": {"python": 0, "javascript": 0, "html": 0, "csharp": 0},
                "total_wmes": 0,
                "errors": [str(e)],
                "files_with_errors": [],
                "execution_time_ms": (time.time() - start_time) * 1000
            }

    def quick_query(
        self,
        instance_name: str,
        query: str,
        type: str = "reql"
    ) -> Dict[str, Any]:
        """
        Execute a quick query outside of reasoning flow.

        Args:
            instance_name: RETER instance name
            query: Query string (REQL syntax)
            type: Query type (only 'reql' is supported)

        Returns:
            results: Query results
            count: Number of matches
        """
        try:
            import traceback

            # Get or create the RETER instance (thread-safe at C++ level)
            try:
                reter = self.instance_manager.get_or_create_instance(instance_name)
            except Exception as inst_error:
                return {
                    "success": False,
                    "error": f"Failed to get instance: {str(inst_error)}",
                    "traceback": traceback.format_exc(),
                    "count": 0,
                    "results": [],
                    "execution_time_ms": 0
                }

            # Execute query (thread-safe at C++ level)
            if type == "reql":
                try:
                    # reql returns naked PyArrow table
                    table = reter.reql(query)

                    # Convert to dict format
                    if table is None:
                        results = []
                        count = 0
                    else:
                        results = table.to_pylist()
                        count = len(results)
                        # Get all column names from schema
                        # PyArrow's to_pylist() may not include keys for null values
                        column_names = table.column_names
                        # Ensure all columns are present in each row with "null" sentinel
                        # MCP protocol uses exclude_none=True which strips None values
                        # This preserves OPTIONAL clause null results
                        for row in results:
                            for col_name in column_names:
                                if col_name not in row or row[col_name] is None:
                                    row[col_name] = "null"

                    execution_time_ms = 0  # reql doesn't return time yet
                except Exception as query_error:
                    return {
                        "success": False,
                        "error": f"Query execution failed: {str(query_error)}",
                        "traceback": traceback.format_exc(),
                        "count": 0,
                        "results": [],
                        "execution_time_ms": 0
                    }
            else:
                return {
                    "success": False,
                    "error": f"Unknown query type: {type}. Only 'reql' is supported.",
                    "count": 0,
                    "results": [],
                }

            return {
                "success": True,
                "results": results,
                "count": count,
                "execution_time_ms": execution_time_ms,
            }
        except Exception as e:
            import traceback
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc(),
                "count": 0,
                "results": [],
                "execution_time_ms": 0
            }

    def forget_source(self, instance_name: str, source: str) -> Dict[str, Any]:
        """
        Remove all facts from a specific source (selective forgetting).

        Intelligently handles both exact source IDs and file paths:
        - If source contains "|", treats as exact source ID (format: {md5_hash}|{rel_path})
        - If source is a file path, finds and forgets all versions (any content hash)

        Args:
            instance_name: RETER instance name
            source: Source ID or file path to forget

        Returns:
            success: Whether forgetting succeeded
            message: Confirmation message
            forgotten_sources: List of source IDs that were forgotten
            count: Number of sources forgotten
        """
        try:
            # Lazy load instance if snapshot is available
            self.instance_manager.ensure_instance_loaded(instance_name)

            # Get or create the RETER instance (thread-safe at C++ level)
            reter = self.instance_manager.get_or_create_instance(instance_name)

            sources_to_forget = []

            # Check if this is an exact source ID (contains |) or a file path
            # Source ID format: {md5_hash}|{rel_path}
            if "|" in source:
                # Exact source ID provided
                sources_to_forget = [source]
            else:
                # Treat as file path - find all sources with this path
                # First, check if it might be a file path
                source_path = Path(source)

                # Get all sources to find matches (thread-safe at C++ level)
                all_sources, _ = reter.get_all_sources()

                if all_sources:
                    # Find all sources that match this file path
                    for source_id in all_sources:
                        if "|" in source_id:
                            # Parse file path from source ID (format: hash|path)
                            file_part = source_id.split("|", 1)[1] if "|" in source_id else source_id
                            # Compare paths (handle both absolute and relative, forward/backward slashes)
                            try:
                                path_matches = (
                                    Path(file_part).resolve() == source_path.resolve() or
                                    file_part == source or
                                    Path(file_part) == source_path or
                                    file_part.replace('\\', '/') == source.replace('\\', '/') or
                                    Path(file_part).as_posix() == Path(source).as_posix()
                                )
                                if path_matches:
                                    sources_to_forget.append(source_id)
                            except (OSError, ValueError):
                                # Path comparison might fail, try string comparison
                                if file_part == source or file_part.replace('\\', '/') == source.replace('\\', '/'):
                                    sources_to_forget.append(source_id)

                    # If no hash-prefixed sources found, try exact match
                    if not sources_to_forget and source in all_sources:
                        sources_to_forget = [source]

            # Forget all matching sources (thread-safe at C++ level)
            results = []
            forgotten = []
            failed = []

            for source_id in sources_to_forget:
                try:
                    returned_source, time_ms = reter.forget_source(source_id)
                    forgotten.append(source_id)
                    results.append({"time_ms": time_ms})
                except (RuntimeError, KeyError, ValueError):
                    # RuntimeError: RETER operation failed
                    # KeyError: Source not found
                    # ValueError: Invalid source ID
                    failed.append(source_id)
                    results.append({"time_ms": 0})

            # Build response message
            if len(sources_to_forget) == 0:
                message = f"No sources found matching: {source}"
                success = False
            elif len(sources_to_forget) == 1:
                if forgotten:
                    message = f"Successfully forgot source: {forgotten[0]}"
                else:
                    message = f"Failed to forget source: {sources_to_forget[0]}"
                success = len(forgotten) > 0
            else:
                message = f"Forgot {len(forgotten)} out of {len(sources_to_forget)} matching sources"
                if forgotten:
                    message += f". Sources forgotten: {', '.join(forgotten)}"
                success = len(forgotten) > 0

            # Auto-save after successful forgetting
            if forgotten:
                # Ensure snapshot directory exists before saving
                self.instance_manager._persistence.snapshots_dir.mkdir(parents=True, exist_ok=True)
                snapshot_path = self.instance_manager._persistence.snapshots_dir / f"{instance_name}.reter"
                reter.save_network(str(snapshot_path))
                reter.mark_clean()

            return {
                "success": success,
                "message": message,
                "forgotten_sources": forgotten,
                "count": len(forgotten),
                "execution_time_ms": sum(r.get("time_ms", 0) for r in results)
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to forget source: {str(e)}",
                "forgotten_sources": [],
                "count": 0,
                "execution_time_ms": 0
            }

    def check_consistency(self, instance_name: str) -> Dict[str, Any]:
        """
        Quick consistency check of current knowledge base.

        Args:
            instance_name: RETER instance name

        Returns:
            consistent: Whether KB is consistent
            inconsistencies: List of contradictions if any
        """
        try:
            # Lazy load instance if snapshot is available
            self.instance_manager.ensure_instance_loaded(instance_name)

            # Get or create the RETER instance (thread-safe at C++ level)
            reter = self.instance_manager.get_or_create_instance(instance_name)

            # Check consistency (thread-safe at C++ level)
            is_consistent, inconsistencies, time_ms = reter.check_consistency()

            return {
                "consistent": is_consistent,
                "contradictions": inconsistencies,
                "unsatisfiable": [],  # TODO: Extract from inconsistencies
                "check_time_ms": time_ms
            }
        except Exception as e:
            return {
                "consistent": True,  # Assume consistent on error
                "contradictions": [],
                "unsatisfiable": [],
                "check_time_ms": 0
            }

    def reload_sources(self, instance_name: str) -> Dict[str, Any]:
        """
        Reload all modified file-based sources in RETER instance.

        Checks all sources for modifications, forgets outdated ones, and reloads
        them from disk with updated timestamps. Warns about deleted sources that
        cannot be reloaded.

        Args:
            instance_name: RETER instance name

        Returns:
            success: Whether reload operation succeeded overall
            reloaded: List of successfully reloaded sources
            failed: List of sources that failed to reload
            deleted: List of deleted sources that couldn't be reloaded
            warnings: Human-readable warnings
            summary: Summary of the operation
        """
        try:
            # Source validation functionality has been removed to fix deadlock issues
            return {
                "success": False,
                "error": "reload_sources functionality has been disabled (source validation removed)",
                "reloaded": [],
                "failed": [],
                "deleted": [],
                "warnings": ["Source validation has been removed to fix queue flooding deadlock"],
                "summary": "This functionality is no longer available."
            }

            # DISABLED CODE BELOW - keeping for reference
            """

            # Get RETER instance (thread-safe at C++ level)
            reter = self.instance_manager.get_or_create_instance(instance_name)

            reloaded = []
            failed = []
            deleted = validity_check["deleted_sources"]
            warnings = []

            # Process outdated sources
            for outdated in validity_check["outdated_sources"]:
                source_id = outdated["source_id"]
                filepath_str = outdated["filepath"]

                try:
                    # Forget the old source (thread-safe at C++ level)
                    forget_result = reter.forget_source(source_id)

                    if not forget_result.get("success"):
                        failed.append({
                            "source_id": source_id,
                            "filepath": filepath_str,
                            "error": f"Failed to forget old source: {forget_result.get('error', 'Unknown error')}"
                        })
                        continue

                    # Reload from file with new timestamp
                    filepath = Path(filepath_str)
                    if filepath.exists():
                        # Determine file type based on extension
                        is_python = filepath.suffix == '.py'

                        # Get new timestamp
                        mtime = filepath.stat().st_mtime
                        timestamp = datetime.fromtimestamp(mtime).isoformat()
                        # Create source ID with timestamp (relative path)
                        relative_path = make_path_relative(filepath.absolute())
                        new_source_id = f"{relative_path}@{timestamp}"

                        # Read file content
                        with open(filepath, 'r', encoding='utf-8') as f:
                            content = f.read()

                        # Add back to RETER (thread-safe at C++ level)
                        if is_python:
                            wme_count, _, _, parse_errors = reter.load_python_code(content, new_source_id)
                            # Convert to dict format for consistency
                            result = {
                                "success": True,  # If we got here, loading succeeded even if there were parse errors
                                "wme_count": wme_count,
                                "errors": parse_errors
                            }
                        else:
                            # Assume ontology file
                            result = reter.add_ontology(content, new_source_id)

                        if result.get("success"):
                            reloaded.append({
                                "old_source_id": source_id,
                                "new_source_id": new_source_id,
                                "filepath": filepath_str,
                                "items_added": result.get("wme_count", 0)
                            })
                        else:
                            failed.append({
                                "source_id": source_id,
                                "filepath": filepath_str,
                                "error": f"Failed to reload: {result.get('error', 'Unknown error')}"
                            })
                    else:
                        # File disappeared between validity check and reload
                        deleted.append({
                            "source_id": source_id,
                            "filepath": filepath_str
                        })

                except Exception as e:
                    failed.append({
                        "source_id": source_id,
                        "filepath": filepath_str,
                        "error": str(e)
                    })

            # Build warnings
            if deleted:
                warnings.append(
                    f"Warning: {len(deleted)} source file(s) have been moved or deleted. "
                    "Knowledge from these files remains incomplete and cannot be updated."
                )
                for d in deleted:
                    warnings.append(f"  - Missing: {d['filepath']}")

            if failed:
                warnings.append(f"Warning: {len(failed)} source(s) failed to reload.")
                for f in failed:
                    warnings.append(f"  - Failed: {f['filepath']} ({f['error']})")

            # Build summary
            summary_parts = []
            if reloaded:
                summary_parts.append(f"{len(reloaded)} source(s) successfully reloaded")
            if failed:
                summary_parts.append(f"{len(failed)} source(s) failed to reload")
            if deleted:
                summary_parts.append(f"{len(deleted)} source(s) deleted/missing")

            summary = ". ".join(summary_parts) if summary_parts else "No sources needed reloading"

            return {
                "success": len(failed) == 0,
                "reloaded": reloaded,
                "failed": failed,
                "deleted": deleted,
                "warnings": warnings,
                "summary": summary
            }
            """

        except Exception as e:
            return {
                "success": False,
                "reloaded": [],
                "failed": [],
                "deleted": [],
                "warnings": [f"Error during reload: {str(e)}"],
                "summary": f"Reload failed: {str(e)}"
            }
