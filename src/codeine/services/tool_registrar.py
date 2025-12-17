"""
Tool Registrar Service

Handles registration and management of MCP tools.
Extracted from LogicalThinkingServer as part of Extract Class refactoring (Fowler Ch. 7).
"""

import asyncio
from typing import Dict, Any, Optional, TYPE_CHECKING
from fastmcp import FastMCP, Context  # Use fastmcp for proper Context injection

from ..logging_config import nlq_debug_logger as debug_log
from ..reter_wrapper import is_initialization_complete
from .initialization_progress import (
    get_initializing_response,
    require_default_instance,
    ComponentNotReadyError,
)

from .reter_operations import ReterOperations
from .state_persistence import StatePersistenceService
from .instance_manager import InstanceManager
from .tools_service import ToolsRegistrar
from .nlq_constants import REQL_SYSTEM_PROMPT
from .nlq_helpers import (
    query_instance_schema,
    build_nlq_prompt,
    extract_reql_from_response,
    execute_reql_query,
    is_retryable_error
)
from .response_truncation import truncate_response

if TYPE_CHECKING:
    from .default_instance_manager import DefaultInstanceManager


class ToolRegistrar:
    """
    Manages MCP tool registration.
    Single Responsibility: Register and configure MCP tools.
    """

    def __init__(
        self,
        reter_ops: ReterOperations,
        persistence: StatePersistenceService,
        instance_manager: InstanceManager,
        default_manager: "DefaultInstanceManager"
    ):
        """
        Initialize ToolRegistrar with required services.

        Args:
            reter_ops: Service for RETER operations
            persistence: Service for state persistence
            instance_manager: Service for managing RETER instances
            default_manager: Service for managing default instance
        """
        self.reter_ops = reter_ops
        self.persistence = persistence
        self.instance_manager = instance_manager
        self.default_manager = default_manager

        # Direct tools registration (pass default_manager for RAG)
        self.tools_registrar = ToolsRegistrar(instance_manager, persistence, default_manager)

    def register_all_tools(self, app: FastMCP) -> None:
        """
        Register all MCP tools with the application.

        Args:
            app: FastMCP application instance
        """
        self._register_knowledge_tools(app)
        self._register_query_tools(app)
        self._register_instance_manager_tool(app)
        self._register_domain_tools(app)
        self._register_experimental_tools(app)
        self._register_info_tools(app)

    def _register_knowledge_tools(self, app: FastMCP) -> None:
        """Register knowledge management tools."""

        @app.tool()
        def add_knowledge(
            instance_name: str,
            source: str,
            type: str = "ontology",
            source_id: Optional[str] = None,
            ctx: Context = None
        ) -> Dict[str, Any]:
            """
            Incrementally add knowledge to RETER's semantic memory.

            RETER is an incremental forward-chaining reasoner - knowledge accumulates!
            Each call ADDS facts/rules to the existing knowledge base (doesn't replace).

            Use cases:
            - Add domain ontology (classes, properties, individuals)
            - Add SWRL inference rules (automatically apply to existing facts)
            - Analyze Python code (extract semantic facts from single .py file)
            - Assert new facts (incremental reasoning)

            Args:
                instance_name: RETER instance name (auto-created if doesn't exist)
                    Special instance "default": Auto-syncs with RETER_PROJECT_ROOT.
                    Files are automatically loaded/reloaded/forgotten based on MD5 changes.
                    Best for project-wide analysis - changes detected automatically.
                source: Ontology content, file path, or single Python file path
                type: 'ontology' (DL/SWRL), 'python' (.py), 'javascript' (.js/.ts), 'html', 'csharp' (.cs), or 'cpp' (.cpp/.hpp/.h)
                source_id: Optional identifier for selective forgetting later

            Returns:
                success: Whether knowledge was successfully added
                items_added: Number of WMEs (facts/rules) added to RETER
                execution_time_ms: Time taken to add and process knowledge
                source_id: The source ID used (includes timestamp for files)
            """
            try:
                require_default_instance()
            except ComponentNotReadyError as e:
                return e.to_response()
            return self.reter_ops.add_knowledge(instance_name, source, type, source_id, ctx)

        @app.tool()
        def add_external_directory(
            instance_name: str,
            directory: str,
            recursive: bool = True,
            exclude_patterns: list[str] = None,
            ctx: Context = None
        ) -> Dict[str, Any]:
            """
            Load EXTERNAL code files from a directory into a NAMED RETER instance.

            Supports: Python (.py), JavaScript (.js, .mjs, .jsx, .ts, .tsx), HTML (.html, .htm),
                      C# (.cs), C++ (.cpp, .cc, .cxx, .hpp, .h)

            Use this tool to load code from external libraries, dependencies,
            or other codebases that are NOT your main project.

            IMPORTANT: Cannot use 'default' instance - it auto-syncs with RETER_PROJECT_ROOT.
            Your main project code should be loaded via RETER_PROJECT_ROOT env var instead.

            Args:
                instance_name: RETER instance name - MUST NOT be 'default'.
                    Use descriptive names like 'external', 'dependencies', 'library_name'.
                directory: Path to directory containing external code files
                recursive: Whether to recursively search subdirectories (default: True)
                exclude_patterns: List of glob patterns to exclude (e.g., ["test_*.py", "*/tests/*", "node_modules/*"])

            Returns:
                success: Whether operation succeeded
                files_loaded: Number of files successfully loaded (by type: python, javascript, html, csharp, cpp)
                total_files: Total number of files found (by type)
                total_wmes: Total WMEs (facts) added across all files
                errors: List of any errors encountered
                files_with_errors: List of files that failed to load
                execution_time_ms: Total time taken
            """
            try:
                require_default_instance()
            except ComponentNotReadyError as e:
                return e.to_response()
            return self.reter_ops.add_external_directory(instance_name, directory, recursive, exclude_patterns, ctx)

    def _register_query_tools(self, app: FastMCP) -> None:
        """Register query execution tools."""

        @app.tool()
        def quick_query(
            instance_name: str,
            query: str,
            type: str = "reql"
        ) -> Dict[str, Any]:
            """
            Execute a quick query outside of reasoning flow.

            NOTE: For most use cases, prefer using `natural_language_query` instead!
            It translates plain English questions into REQL automatically using LLM.
            Use `quick_query` only when you need precise control over the REQL syntax.

            Automatically checks source validity before executing queries and includes
            warnings if any sources are outdated or deleted.

            Args:
                instance_name: RETER instance name (auto-created if doesn't exist)
                    Special instance "default": Auto-syncs with RETER_PROJECT_ROOT.
                    Ideal for querying project-wide code patterns and relationships.
                query: Query string in REQL syntax
                type: 'reql', 'dl', or 'pattern'

            Returns:
                results: Query results
                count: Number of matches
                source_validity: Information about outdated/deleted sources
                warnings: Any warnings about source validity
            """
            try:
                require_default_instance()
            except ComponentNotReadyError as e:
                return e.to_response()
            result = self.reter_ops.quick_query(instance_name, query, type)
            return truncate_response(result)

    def _register_instance_manager_tool(self, app: FastMCP) -> None:
        """Register unified instance manager tool."""

        @app.tool()
        def instance_manager(
            action: str,
            instance_name: str = "default",
            source: Optional[str] = None
        ) -> Dict[str, Any]:
            """
            Unified tool for managing RETER instances and sources.

            **See: system://reter/multiple-instances for complete documentation**

            Actions:
            - list: List all RETER instances (loaded and available snapshots)
            - list_sources: List all sources loaded in an instance
            - get_facts: Get fact IDs for a specific source
            - forget: Remove all facts from a source (selective forgetting)
            - reload: Reload all modified file-based sources
            - check: Quick consistency check of knowledge base

            Args:
                action: One of: list, list_sources, get_facts, forget, reload, check
                instance_name: RETER instance name (default: "default")
                source: Source ID or file path (required for get_facts, forget)

            Returns:
                Action-specific results with success status

            Examples:
                instance_manager("list")  # List all instances
                instance_manager("list_sources", "default")  # List sources in default
                instance_manager("get_facts", "default", "path/to/file.py")  # Get facts
                instance_manager("forget", "default", "path/to/file.py")  # Forget source
                instance_manager("reload", "default")  # Reload modified sources
                instance_manager("check", "default")  # Check consistency
            """
            try:
                require_default_instance()
            except ComponentNotReadyError as e:
                return e.to_response()

            if action == "list":
                # List all instances
                loaded_instances_dict = self.instance_manager.get_all_instances()
                loaded_instances = list(loaded_instances_dict.keys())
                available_snapshots = self.persistence.get_available_snapshot_names()

                all_instances = {}
                default_status = self.default_manager.get_status()
                all_instances["default"] = default_status

                for inst_name in loaded_instances:
                    if inst_name != "default":
                        all_instances[inst_name] = "loaded"

                for inst_name in available_snapshots:
                    if inst_name not in all_instances:
                        all_instances[inst_name] = "available"

                return {
                    "success": True,
                    "action": "list",
                    "instances": all_instances,
                    "total_count": len(all_instances),
                    "loaded_count": len([s for s in all_instances.values() if s == "loaded"]),
                    "available_count": len([s for s in all_instances.values() if s == "available"]),
                    "_resources": {
                        "system://reter/multiple-instances": "Multiple Instances Guide",
                        "system://reter/source-management": "Source Management Guide"
                    }
                }

            elif action == "list_sources":
                result = self.persistence.list_sources(instance_name)
                result["action"] = "list_sources"
                return result

            elif action == "get_facts":
                if not source:
                    return {"success": False, "error": "source parameter required for get_facts action"}
                result = self.persistence.get_source_facts(instance_name, source)
                result["action"] = "get_facts"
                return result

            elif action == "forget":
                if not source:
                    return {"success": False, "error": "source parameter required for forget action"}
                result = self.reter_ops.forget_source(instance_name, source)
                result["action"] = "forget"
                return result

            elif action == "reload":
                result = self.reter_ops.reload_sources(instance_name)
                result["action"] = "reload"
                return result

            elif action == "check":
                result = self.reter_ops.check_consistency(instance_name)
                result["action"] = "check"
                return result

            else:
                return {
                    "success": False,
                    "error": f"Unknown action: {action}",
                    "available_actions": ["list", "list_sources", "get_facts", "forget", "reload", "check"],
                    "_resources": {
                        "system://reter/multiple-instances": "Multiple Instances Guide"
                    }
                }

    def _register_domain_tools(self, app: FastMCP) -> None:
        """Register all RETER domain-specific tools (Python analysis, UML, Gantt, etc.)."""
        self.tools_registrar.register_all_tools(app)

    def _register_experimental_tools(self, app: FastMCP) -> None:
        """Register experimental tools for testing new features."""
        self._register_nlq_tool(app)

    def _register_nlq_tool(self, app: FastMCP) -> None:
        """Register the natural language query tool."""

        @app.tool()
        async def natural_language_query(
            question: str,
            instance_name: str = "default",
            max_retries: int = 5,
            timeout: int = 30,
            max_results: int = 500,
            ctx: Context = None
        ) -> Dict[str, Any]:
            """
            Query CODE STRUCTURE using natural language (translates to REQL).

            **PURPOSE**: Ask questions about code structure, relationships, and patterns.
            This tool translates your natural language question into a REQL query that
            searches the parsed Python codebase (classes, methods, functions, imports, etc.).

            **NOT FOR**: General knowledge questions, documentation content, or semantic
            code search. Use `semantic_search` for finding code by meaning/similarity.

            **See: python://reter/query-patterns for REQL examples if needed**

            Examples:
                - "What classes inherit from BaseTool?"
                - "Find all methods that call the save function"
                - "List modules with more than 5 classes"
                - "Which functions have the most parameters?"
                - "Find functions with magic number 100"
                - "Show string literals containing 'error'"

            Translates natural language questions into REQL queries using LLM,
            executes them against the code knowledge graph, and returns results.

            Args:
                question: Natural language question about code structure (plain English)
                instance_name: RETER instance to query (default: "default")
                max_retries: Maximum retry attempts on syntax errors (default: 5)
                timeout: Query timeout in seconds (default: 30)
                max_results: Maximum results to return (default: 500)
                ctx: MCP context (injected automatically)

            Returns:
                success: Whether query succeeded
                results: Query results as list of dicts
                count: Number of results
                reql_query: The REQL query that was executed (useful for learning REQL)
                attempts: Number of attempts made
                error: Error message if failed
                truncated: Whether results were truncated (if count > max_results)
            """
            try:
                require_default_instance()
            except ComponentNotReadyError as e:
                return e.to_response()

            debug_log.debug(f"\n{'#'*60}\nNEW NLQ REQUEST\n{'#'*60}")
            debug_log.debug(f"Question: {question}, Instance: {instance_name}")

            if ctx is None:
                return self._nlq_error_response("Context not available for LLM sampling")

            try:
                reter = self.instance_manager.get_or_create_instance(instance_name)
            except Exception as e:
                return self._nlq_error_response(f"Failed to get RETER instance: {str(e)}")

            schema_info = query_instance_schema(reter)

            # Execute with timeout protection
            try:
                async with asyncio.timeout(timeout):
                    return await self._execute_nlq_with_retries(
                        reter, question, schema_info, max_retries, max_results, ctx
                    )
            except asyncio.TimeoutError:
                debug_log.debug(f"Query timed out after {timeout} seconds")
                return self._nlq_error_response(
                    f"Query timed out after {timeout} seconds. Try a more specific question.",
                    attempts=0
                )

    def _nlq_error_response(self, error: str, query: str = None, attempts: int = 0) -> Dict[str, Any]:
        """Create a standardized error response for NLQ tool."""
        return {
            "success": False,
            "results": [],
            "count": 0,
            "reql_query": query,
            "attempts": attempts,
            "error": error
        }

    async def _execute_nlq_with_retries(
        self,
        reter,
        question: str,
        schema_info: str,
        max_retries: int,
        max_results: int,
        ctx
    ) -> Dict[str, Any]:
        """Execute NLQ with retry logic for syntax errors."""
        attempts = 0
        last_error = None
        generated_query = None

        while attempts < max_retries:
            attempts += 1
            try:
                user_prompt = build_nlq_prompt(
                    question, schema_info, attempts, generated_query, last_error
                )
                debug_log.debug(f"\n{'='*60}\nNLQ ATTEMPT {attempts}/{max_retries}\n{'='*60}")

                response = await ctx.sample(user_prompt, system_prompt=REQL_SYSTEM_PROMPT)
                response_text = response.text if hasattr(response, 'text') else str(response)
                debug_log.debug(f"LLM RAW RESPONSE:\n{response_text}")

                generated_query = extract_reql_from_response(response_text)
                debug_log.debug(f"EXTRACTED REQL QUERY:\n{generated_query}")

                results = execute_reql_query(reter, generated_query)
                total_count = len(results)
                debug_log.debug(f"QUERY SUCCESS: {total_count} results")

                # Check for potential cross-join (excessive results)
                cross_join_threshold = 1000
                if total_count > cross_join_threshold:
                    debug_log.debug(f"CROSS-JOIN DETECTED: {total_count} results exceeds threshold")
                    return {
                        "success": False,
                        "results": [],
                        "count": total_count,
                        "reql_query": generated_query,
                        "attempts": attempts,
                        "error": f"Query returned {total_count} results which suggests a cross-join error. "
                                 f"Please rephrase your question to be more specific.",
                        "warning": "Possible cross-join detected - query may be missing proper join conditions"
                    }

                # Apply result truncation
                truncated = False
                if total_count > max_results:
                    results = results[:max_results]
                    truncated = True
                    debug_log.debug(f"Results truncated from {total_count} to {max_results}")

                response_dict = {
                    "success": True,
                    "results": results,
                    "count": total_count,
                    "reql_query": generated_query,
                    "attempts": attempts,
                    "error": None
                }

                if truncated:
                    response_dict["truncated"] = True
                    response_dict["warning"] = f"Results truncated. Showing {max_results} of {total_count}. Use more specific queries for full results."

                return truncate_response(response_dict)

            except Exception as e:
                last_error = str(e)
                debug_log.debug(f"QUERY ERROR: {last_error}")

                if is_retryable_error(last_error):
                    debug_log.debug(f"Retryable error (attempt {attempts}/{max_retries})")
                    continue
                else:
                    debug_log.debug("Non-retryable error, aborting")
                    return self._nlq_error_response(last_error, generated_query, attempts)

        debug_log.debug(f"MAX RETRIES EXHAUSTED after {attempts} attempts")
        return self._nlq_error_response(
            f"Failed after {max_retries} attempts. Last error: {last_error}",
            generated_query,
            attempts
        )

    def _register_info_tools(self, app: FastMCP) -> None:
        """Register info/diagnostic tools."""

        @app.tool()
        def reter_info() -> Dict[str, Any]:
            """
            Get version and diagnostic information about RETER components.

            Returns version info for:
            - MCP server (reter-logical-thinking-server)
            - Python reter package
            - C++ RETE binding (owl_rete_cpp)

            Useful for debugging deployment issues and verifying correct module versions.

            Returns:
                Dictionary with version information for all components.
            """
            import reter
            from reter import owl_rete_cpp

            # Get MCP server version
            try:
                from codeine import __version__ as mcp_version
            except ImportError:
                mcp_version = "unknown"

            # Get reter Python package version
            try:
                reter_version = reter.__version__
            except AttributeError:
                reter_version = "unknown"

            # Get C++ binding version info
            try:
                cpp_version = getattr(owl_rete_cpp, "__version__", "unknown")
                cpp_build_timestamp = getattr(owl_rete_cpp, "__build_timestamp__", "unknown")
                cpp_info = owl_rete_cpp.get_version_info() if hasattr(owl_rete_cpp, "get_version_info") else {}
            except Exception as e:
                cpp_version = f"error: {e}"
                cpp_build_timestamp = "unknown"
                cpp_info = {}

            # Get reter module file location
            try:
                reter_location = reter.__file__
            except AttributeError:
                reter_location = "unknown"

            # Get owl_rete_cpp module location
            try:
                cpp_location = owl_rete_cpp.__file__
            except AttributeError:
                cpp_location = "unknown"

            return {
                "success": True,
                "mcp_server": {
                    "name": "reter-logical-thinking-server",
                    "version": mcp_version,
                },
                "reter_package": {
                    "version": reter_version,
                    "location": reter_location,
                },
                "cpp_binding": {
                    "version": cpp_version,
                    "build_timestamp": cpp_build_timestamp,
                    "location": cpp_location,
                    "info": cpp_info,
                },
                "optional_fix_present": cpp_info.get("optional_fix", False),
            }