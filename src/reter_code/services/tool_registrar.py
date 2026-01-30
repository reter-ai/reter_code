"""
Tool Registrar Service

Handles registration and management of MCP tools.
Uses Claude Agent SDK for query generation (no sampling fallback).

Tool Availability Modes (via TOOLS_AVAILABLE env var):
- "default": Only essential tools: reql, system, thinking, session, items, semantic_search, natural_language_query
- "full": All tools including code_inspection, recommender, diagram, execute_cadsl, etc.
"""

import asyncio
import os
from typing import Dict, Any, Optional, TYPE_CHECKING, Set
from fastmcp import FastMCP, Context

from ..logging_config import nlq_debug_logger as debug_log, ensure_nlq_logger_configured
from .initialization_progress import (
    require_default_instance,
    ComponentNotReadyError,
)

from .reter_operations import ReterOperations
from .state_persistence import StatePersistenceService
from .instance_manager import InstanceManager
from .tools_service import ToolsRegistrar
from .registrars.system_tools import SystemToolsRegistrar
from .nlq_helpers import query_instance_schema
from .response_truncation import truncate_response
from .hybrid_query_engine import (
    build_rag_query_params,
    build_similar_tools_section,
)
from .agent_sdk_client import (
    is_agent_sdk_available,
    generate_cadsl_query,
    retry_cadsl_query,
)

if TYPE_CHECKING:
    from .default_instance_manager import DefaultInstanceManager


# Default tools available in "default" mode
DEFAULT_TOOLS: Set[str] = {
    "reql",
    "system",
    "thinking",
    "session",
    "items",
    "execute_cadsl",
    "generate_cadsl",
    "semantic_search",
    "natural_language_query",
}


class ToolRegistrar:
    """
    Manages MCP tool registration.
    Single Responsibility: Register and configure MCP tools.

    @reter: ServiceLayer(self)
    @reter: MCPToolProvider(self)
    @reter: dependsOn(self, reter_code.services.ReterOperations)
    @reter: dependsOn(self, reter_code.services.StatePersistenceService)
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

        # Read tools availability mode from environment
        # "default" = essential tools only, "full" = all tools
        self._tools_mode = os.environ.get("TOOLS_AVAILABLE", "default").lower()

        # Direct tools registration (pass default_manager for RAG)
        # Pass tools_filter for selective registration
        tools_filter = DEFAULT_TOOLS if self._tools_mode == "default" else None
        self.tools_registrar = ToolsRegistrar(
            instance_manager, persistence, default_manager, tools_filter=tools_filter
        )

        # System tools registrar (unified system management)
        self.system_registrar = SystemToolsRegistrar(
            instance_manager, persistence, default_manager, reter_ops
        )

    def register_all_tools(self, app: FastMCP) -> None:
        """
        Register all MCP tools with the application.

        Respects TOOLS_AVAILABLE environment variable:
        - "default": Only essential tools (reql, system, thinking, session, items, semantic_search, natural_language_query)
        - "full": All tools

        Args:
            app: FastMCP application instance
        """
        import logging
        logger = logging.getLogger(__name__)

        is_full_mode = self._tools_mode == "full"

        # Log which tools mode is active
        if is_full_mode:
            logger.info("Tools mode: FULL (all tools available)")
        else:
            logger.info("Tools mode: DEFAULT (limited to: %s)", ", ".join(sorted(DEFAULT_TOOLS)))

        # Knowledge tools (add_knowledge, add_external_directory) - full mode only
        if is_full_mode:
            self._register_knowledge_tools(app)

        # Query tools (reql) - always registered in default mode
        if is_full_mode or "reql" in DEFAULT_TOOLS:
            self._register_query_tools(app)

        # System tool - always registered in default mode
        if is_full_mode or "system" in DEFAULT_TOOLS:
            self.system_registrar.register(app)

        # Domain tools (code_inspection, recommender, thinking, session, items, diagram, semantic_search)
        # ToolsRegistrar handles filtering internally based on tools_filter
        self._register_domain_tools(app)

        # Experimental tools (natural_language_query, execute_cadsl)
        self._register_experimental_tools(app, is_full_mode)

    def _register_knowledge_tools(self, app: FastMCP) -> None:
        """Register knowledge management tools."""

        @app.tool()
        def add_knowledge(
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
            return self.reter_ops.add_knowledge("default", source, type, source_id, ctx)

        @app.tool()
        def add_external_directory(
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

            Args:
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
            return self.reter_ops.add_external_directory("default", directory, recursive, exclude_patterns, ctx)

    def _register_query_tools(self, app: FastMCP) -> None:
        """Register query execution tools."""

        @app.tool()
        def reql(
            query: str,
            type: str = "reql"
        ) -> Dict[str, Any]:
            """
            Execute a REQL query against the code knowledge graph.

            REQL (RETER Query Language) allows structural queries over parsed code:
            classes, methods, functions, imports, inheritance, etc.

            Automatically checks source validity before executing queries and includes
            warnings if any sources are outdated or deleted.

            Args:
                query: Query string in REQL syntax
                type: 'reql', 'dl', or 'pattern'

            Returns:
                results: Query results
                count: Number of matches
                source_validity: Information about outdated/deleted sources
                warnings: Any warnings about source validity

            Examples:
                - "SELECT ?c ?name WHERE { ?c type class . ?c has-name ?name }"
                - "SELECT ?m WHERE { ?m type method . ?m is-defined-in ?c . ?c has-name 'MyClass' }"
            """
            try:
                require_default_instance()
            except ComponentNotReadyError as e:
                return e.to_response()
            result = self.reter_ops.quick_query("default", query, type)
            return truncate_response(result)

    def _register_domain_tools(self, app: FastMCP) -> None:
        """Register all RETER domain-specific tools (Python analysis, UML, Gantt, etc.)."""
        self.tools_registrar.register_all_tools(app)

    def _register_experimental_tools(self, app: FastMCP, is_full_mode: bool = True) -> None:
        """Register experimental tools for testing new features.

        Args:
            app: FastMCP application instance
            is_full_mode: If True, register all experimental tools. If False, only natural_language_query.
        """
        # natural_language_query - always available in default mode
        if is_full_mode or "natural_language_query" in DEFAULT_TOOLS:
            self._register_nlq_tool(app)

        # execute_cadsl - available in default mode
        if is_full_mode or "execute_cadsl" in DEFAULT_TOOLS:
            self._register_cadsl_tool(app)

        # generate_cadsl - generate CADSL without executing
        if is_full_mode or "generate_cadsl" in DEFAULT_TOOLS:
            self._register_generate_cadsl_tool(app)

    async def _execute_cadsl_query(
        self,
        question: str,
        schema_info: str,
        max_retries: int,
        similar_tools: Optional[list] = None
    ) -> Dict[str, Any]:
        """Execute a CADSL query using Agent SDK for generation with retry on empty/error."""
        from ..cadsl.parser import parse_cadsl
        from ..cadsl.transformer import CADSLTransformer
        from ..cadsl.loader import build_pipeline_factory

        debug_log.debug(f"\n{'='*60}\nCADSL EXECUTION (Agent SDK)\n{'='*60}")

        if not is_agent_sdk_available():
            return {
                "success": False,
                "results": [],
                "count": 0,
                "query_type": "cadsl",
                "error": "Claude Agent SDK not available. Install with: pip install claude-agent-sdk"
            }

        # Build similar tools context if provided
        similar_tools_context = None
        if similar_tools:
            similar_tools_context = build_similar_tools_section(similar_tools)

        # Get reter instance and rag_manager for query tools
        try:
            reter = self.instance_manager.get_or_create_instance("default")
        except Exception:
            reter = None

        rag_manager = self.default_manager.get_rag_manager() if self.default_manager else None

        # Get project root for file path context in Agent SDK
        project_root = None
        if hasattr(self.instance_manager, '_project_root') and self.instance_manager._project_root:
            project_root = str(self.instance_manager._project_root)

        # Track total attempts across retries
        total_attempts = 0
        all_tools_used = []
        max_empty_retries = 2  # Max times to retry on empty results

        async def execute_single_cadsl(query: str, attempt_info: dict) -> Dict[str, Any]:
            """Execute a single CADSL query and return results."""
            nonlocal total_attempts, all_tools_used
            total_attempts += attempt_info.get("attempts", 1)
            all_tools_used.extend(attempt_info.get("tools_used", []))

            # Auto-fix bare reql blocks (Rule 10 violation)
            fixed_query = self._fix_bare_reql_block(query)

            # Parse CADSL
            parse_result = parse_cadsl(fixed_query)
            if not parse_result.success:
                return {
                    "success": False,
                    "results": [],
                    "count": 0,
                    "cadsl_query": fixed_query,
                    "query_type": "cadsl",
                    "attempts": total_attempts,
                    "tools_used": all_tools_used,
                    "error": f"Parse error: {parse_result.errors}"
                }

            transformer = CADSLTransformer()
            tool_specs = transformer.transform(parse_result.tree)

            if not tool_specs:
                return {
                    "success": False,
                    "results": [],
                    "count": 0,
                    "cadsl_query": fixed_query,
                    "query_type": "cadsl",
                    "attempts": total_attempts,
                    "tools_used": all_tools_used,
                    "error": "No tool spec generated from CADSL"
                }

            # Build and execute pipeline
            nonlocal reter, rag_manager
            reter = self.instance_manager.get_or_create_instance("default")
            rag_manager = self.default_manager.get_rag_manager() if self.default_manager else None

            tool_spec = tool_specs[0]

            # Extract default parameter values from tool spec
            params = {"rag_manager": rag_manager}
            for param in tool_spec.params:
                if param.default is not None:
                    params[param.name] = param.default
            debug_log.debug(f"CADSL pipeline params: {params}")

            from ..dsl.core import Context as PipelineContext
            pipeline_ctx = PipelineContext(reter=reter, params=params)

            pipeline_factory = build_pipeline_factory(tool_spec)
            pipeline = pipeline_factory(pipeline_ctx)

            pipeline_result = pipeline.execute(pipeline_ctx)

            # Check for Result monad errors (Err type)
            from ..dsl.catpy import Err
            if isinstance(pipeline_result, Err):
                error = pipeline_result.value
                return {
                    "success": False,
                    "results": [],
                    "count": 0,
                    "cadsl_query": fixed_query,
                    "query_type": "cadsl",
                    "attempts": total_attempts,
                    "tools_used": all_tools_used,
                    "error": str(error)  # PipelineError has __str__
                }

            # Unwrap Ok result if needed
            from ..dsl.catpy import Ok
            if isinstance(pipeline_result, Ok):
                pipeline_result = pipeline_result.value

            # Normalize result
            if isinstance(pipeline_result, dict) and "success" in pipeline_result:
                pipeline_result["cadsl_query"] = fixed_query
                pipeline_result["query_type"] = "cadsl"
                pipeline_result["attempts"] = total_attempts
                pipeline_result["tools_used"] = all_tools_used
                return pipeline_result
            else:
                results = pipeline_result if isinstance(pipeline_result, list) else [pipeline_result]
                return {
                    "success": True,
                    "results": results,
                    "count": len(results),
                    "cadsl_query": fixed_query,
                    "query_type": "cadsl",
                    "attempts": total_attempts,
                    "tools_used": all_tools_used,
                    "error": None
                }

        try:
            # Generate initial CADSL using Agent SDK
            result = await generate_cadsl_query(
                question=question,
                schema_info=schema_info,
                max_iterations=max_retries,
                similar_tools_context=similar_tools_context,
                reter_instance=reter,
                rag_manager=rag_manager,
                project_root=project_root
            )

            if not result.success:
                return {
                    "success": False,
                    "results": [],
                    "count": 0,
                    "cadsl_query": result.query,
                    "query_type": "cadsl",
                    "attempts": result.attempts,
                    "tools_used": result.tools_used,
                    "error": result.error or "Query generation failed"
                }

            generated_query = result.query
            debug_log.debug(f"GENERATED CADSL QUERY:\n{generated_query}")

            # Execute the query
            exec_result = await execute_single_cadsl(generated_query, {
                "attempts": result.attempts,
                "tools_used": result.tools_used
            })

            # Check if we need to retry (empty results or error)
            empty_retry_count = 0
            current_query = generated_query

            while empty_retry_count < max_empty_retries:
                result_count = exec_result.get("count", 0)
                has_error = not exec_result.get("success", False)
                error_msg = exec_result.get("error")

                # If we have results and no error, we're done
                if result_count > 0 and not has_error:
                    debug_log.debug(f"CADSL query returned {result_count} results, no retry needed")
                    return exec_result

                # Ask agent if it wants to retry
                debug_log.debug(f"CADSL query returned {result_count} results, error={error_msg}. Asking agent to retry...")

                retry_result = await retry_cadsl_query(
                    question=question,
                    previous_query=current_query,
                    result_count=result_count,
                    error_message=error_msg if has_error else None,
                    reter_instance=reter,
                    rag_manager=rag_manager
                )

                # Check if agent confirmed empty is correct
                if retry_result.error == "CONFIRM_EMPTY":
                    debug_log.debug("Agent confirmed empty results are correct")
                    exec_result["agent_confirmed_empty"] = True
                    return exec_result

                # Check if agent provided a new query
                if retry_result.success and retry_result.query:
                    debug_log.debug(f"Agent provided retry query: {retry_result.query[:100]}...")
                    current_query = retry_result.query
                    exec_result = await execute_single_cadsl(current_query, {
                        "attempts": retry_result.attempts,
                        "tools_used": retry_result.tools_used
                    })
                    empty_retry_count += 1
                else:
                    # Agent didn't provide a new query, return current result
                    debug_log.debug("Agent did not provide retry query, returning current result")
                    return exec_result

            # Max retries reached
            debug_log.debug(f"Max empty retries ({max_empty_retries}) reached")
            return exec_result

        except Exception as e:
            debug_log.debug(f"CADSL ERROR: {e}")
            return {
                "success": False,
                "results": [],
                "count": 0,
                "query_type": "cadsl",
                "attempts": total_attempts,
                "tools_used": all_tools_used,
                "error": str(e)
            }

    def _fix_bare_reql_block(self, cadsl_query: str) -> str:
        """
        Auto-fix bare reql blocks by wrapping them in a query definition.

        This fixes Rule 10 violations where the LLM generates:
            reql { ... } | select { ... }
        Instead of:
            query auto_generated() { reql { ... } | select { ... } | emit { results } }
        """
        import re

        stripped = cadsl_query.strip()

        # Check if it starts with 'reql' (bare reql block)
        if stripped.startswith('reql') and not stripped.startswith('reql_'):
            debug_log.debug("AUTO-FIX: Detected bare reql block, wrapping in query definition")

            # Check if there's already an emit step
            has_emit = bool(re.search(r'\|\s*emit\s*\{', stripped))

            if has_emit:
                # Already has emit, just wrap
                fixed = f'query auto_generated() {{\n    """{stripped[:50]}..."""\n\n    {stripped}\n}}'
            else:
                # Need to add emit step - find the last pipeline step or the reql block end
                # Look for pattern like "| select { ... }" or just the reql block
                if '|' in stripped:
                    # Has pipeline steps, add emit at the end
                    fixed = f'query auto_generated() {{\n    """{stripped[:50]}..."""\n\n    {stripped}\n    | emit {{ results }}\n}}'
                else:
                    # Just a reql block, add select and emit
                    fixed = f'query auto_generated() {{\n    """{stripped[:50]}..."""\n\n    {stripped}\n    | emit {{ results }}\n}}'

            debug_log.debug(f"AUTO-FIX: Wrapped query:\n{fixed[:200]}...")
            return fixed

        return cadsl_query

    def _execute_rag_query(self, question: str) -> Dict[str, Any]:
        """Execute a RAG query - semantic search, duplicate detection, or clustering."""
        rag_manager = self.default_manager.get_rag_manager() if self.default_manager else None

        if rag_manager is None:
            return {
                "success": False,
                "results": [],
                "count": 0,
                "query_type": "rag",
                "error": "RAG is not configured. Set RETER_PROJECT_ROOT to enable."
            }

        if not rag_manager.is_enabled:
            return {
                "success": False,
                "results": [],
                "count": 0,
                "query_type": "rag",
                "error": "RAG is disabled via configuration."
            }

        try:
            params = build_rag_query_params(question)
            analysis_type = params.get("analysis_type", "search")
            debug_log.debug(f"RAG QUERY PARAMS: {params} (type: {analysis_type})")

            if analysis_type == "duplicates":
                # Execute duplicate detection
                result = rag_manager.find_duplicate_candidates(
                    similarity_threshold=params.get("similarity_threshold", 0.85),
                    max_results=params.get("max_results", 50),
                    exclude_same_file=params.get("exclude_same_file", True),
                    exclude_same_class=params.get("exclude_same_class", True),
                    entity_types=params.get("entity_types"),
                )
                return {
                    "success": True,
                    "results": result.get("pairs", []),
                    "count": result.get("count", 0),
                    "query_type": "rag",
                    "analysis_type": "duplicates",
                    "rag_params": params,
                    "stats": result.get("stats", {}),
                    "error": None
                }

            elif analysis_type == "clusters":
                # Execute cluster detection
                result = rag_manager.find_similar_clusters(
                    n_clusters=params.get("n_clusters", 50),
                    min_cluster_size=params.get("min_size", 2),  # param name is min_cluster_size
                    exclude_same_file=params.get("exclude_same_file", True),
                    exclude_same_class=params.get("exclude_same_class", True),
                    entity_types=params.get("entity_types"),
                )
                return {
                    "success": True,
                    "results": result.get("clusters", []),
                    "count": result.get("total_clusters", 0),  # return key is total_clusters
                    "query_type": "rag",
                    "analysis_type": "clusters",
                    "rag_params": params,
                    "stats": result.get("stats", {}),
                    "error": None
                }

            else:
                # Default: semantic search
                results, stats = rag_manager.search(
                    query=params.get("query", question),
                    top_k=params.get("top_k", 20),
                    entity_types=params.get("entity_types"),
                    search_scope=params.get("search_scope", "code"),
                    include_content=params.get("include_content", False),
                )

                return {
                    "success": True,
                    "results": [r.to_dict() for r in results],
                    "count": len(results),
                    "query_type": "rag",
                    "analysis_type": "search",
                    "rag_params": params,
                    "stats": stats,
                    "error": None
                }

        except Exception as e:
            return {
                "success": False,
                "results": [],
                "count": 0,
                "query_type": "rag",
                "error": str(e)
            }

    def _register_nlq_tool(self, app: FastMCP) -> None:
        """Register the natural language query tool."""

        @app.tool()
        async def natural_language_query(
            question: str,
            max_retries: int = 5,
            timeout: int = 1800,
            max_results: int = 500,
            ctx: Context = None
        ) -> Dict[str, Any]:
            """
            Query code using natural language - translates to CADSL automatically.

            Ask questions about code structure, patterns, and relationships in plain English.
            The query is translated to CADSL (Code Analysis DSL) and executed against the
            code knowledge graph.

            Args:
                question: Natural language question about code
                max_retries: Maximum retry attempts on errors (default: 5)
                timeout: Query timeout in seconds (default: 1800)
                max_results: Maximum results to return (default: 500)
                ctx: MCP context (injected automatically)

            Returns:
                success: Whether query succeeded
                results: Query results as list of dicts
                count: Number of results
                query_type: Always "cadsl"
                generated_query: The CADSL query that was generated
                error: Error message if failed

            Examples:
                - "Find all classes with more than 10 methods"
                - "Show circular import dependencies"
                - "List methods that call the save function"
                - "Generate a class diagram for the services module"
                - "Find god classes in the codebase"
            """
            import time
            start_time = time.time()

            try:
                require_default_instance()
            except ComponentNotReadyError as e:
                return e.to_response()

            # Ensure logger is configured with correct directory
            ensure_nlq_logger_configured()

            debug_log.debug(f"\n{'#'*60}\nNEW NLQ REQUEST\n{'#'*60}")
            debug_log.debug(f"Question: {question}")

            if ctx is None:
                return {
                    "success": False,
                    "results": [],
                    "count": 0,
                    "error": "Context not available for LLM sampling"
                }

            try:
                reter = self.instance_manager.get_or_create_instance("default")
            except Exception as e:
                return {
                    "success": False,
                    "results": [],
                    "count": 0,
                    "error": f"Failed to get RETER instance: {str(e)}"
                }

            schema_info = query_instance_schema(reter)

            # Find similar CADSL tools for context
            from .hybrid_query_engine import find_similar_cadsl_tools
            similar_tools = find_similar_cadsl_tools(question, max_results=5)

            debug_log.debug(f"Similar tools: {[t.name for t in similar_tools]}")

            try:
                async with asyncio.timeout(timeout):
                    result = await self._execute_cadsl_query(
                        question, schema_info, max_retries,
                        similar_tools=similar_tools
                    )

                    if similar_tools:
                        result["similar_tools"] = [t.to_dict() for t in similar_tools]

                    result["execution_time_ms"] = (time.time() - start_time) * 1000
                    return truncate_response(result)

            except asyncio.TimeoutError:
                return {
                    "success": False,
                    "results": [],
                    "count": 0,
                    "query_type": "cadsl",
                    "error": f"Query timed out after {timeout} seconds",
                }

    def _register_cadsl_tool(self, app: FastMCP) -> None:
        """Register the CADSL script execution tool."""

        @app.tool()
        def execute_cadsl(
            script: str,
            params: Optional[Dict[str, Any]] = None,
            timeout_ms: int = 300000
        ) -> Dict[str, Any]:
            """
            Execute a CADSL script for code analysis.

            CADSL (Code Analysis DSL) is a pipeline-based language for complex code analysis
            including code smell detection, diagram generation, and semantic queries.

            Args:
                script: Either a CADSL script string or a path to a .cadsl file.
                        If it looks like a file path (ends with .cadsl or contains path separators),
                        it will be read from disk.
                params: Optional dictionary of parameters to pass to the CADSL tool.
                        These override default parameter values defined in the script.
                timeout_ms: Query timeout in milliseconds (default: 300000 = 5 minutes).
                        Set to 0 for no timeout. For heavy queries like call_graph, use 600000 (10 min).

            Returns:
                success: Whether execution succeeded
                results: Query/detector/diagram results
                count: Number of results
                tool_name: Name of the executed CADSL tool
                tool_type: Type of tool (query, detector, diagram)
                error: Error message if failed

            Examples:
                # Execute inline CADSL
                execute_cadsl('''
                    query find_large_classes() {
                        reql {
                            SELECT ?c ?name (COUNT(?m) AS ?method_count)
                            WHERE {
                                ?c type class . ?c has-name ?name .
                                ?m type method . ?m is-defined-in ?c
                            }
                            GROUP BY ?c ?name
                            HAVING (?method_count > 10)
                        }
                        | emit { results }
                    }
                ''')

                # Execute from file
                execute_cadsl("path/to/detector.cadsl", params={"threshold": 15})

                # Execute with longer timeout for heavy queries
                execute_cadsl("path/to/call_graph.cadsl", params={"target": "main"}, timeout_ms=600000)
            """
            import os
            import time
            from pathlib import Path
            from ..cadsl.parser import parse_cadsl
            from ..cadsl.transformer import CADSLTransformer
            from ..cadsl.loader import build_pipeline_factory

            try:
                require_default_instance()
            except ComponentNotReadyError as e:
                return e.to_response()

            start_time = time.time()

            # Determine if script is a file path or inline CADSL
            cadsl_content = script
            source_file = None

            # Check if it looks like a file path (not inline CADSL)
            # Inline CADSL typically starts with 'query', 'detector', 'diagram' keywords
            script_stripped = script.strip()
            is_inline_cadsl = (
                script_stripped.startswith('query ') or
                script_stripped.startswith('detector ') or
                script_stripped.startswith('diagram ') or
                script_stripped.startswith('//')  # Comment
            )

            if not is_inline_cadsl and (script_stripped.endswith('.cadsl') or os.path.sep in script_stripped):
                path = Path(script_stripped)
                if path.exists() and path.is_file():
                    source_file = str(path)
                    with open(path, 'r', encoding='utf-8') as f:
                        cadsl_content = f.read()
                elif not path.exists():
                    return {
                        "success": False,
                        "results": [],
                        "count": 0,
                        "error": f"CADSL file not found: {script}"
                    }

            try:
                # Parse CADSL
                parse_result = parse_cadsl(cadsl_content)
                if not parse_result.success:
                    return {
                        "success": False,
                        "results": [],
                        "count": 0,
                        "cadsl_script": cadsl_content[:500] + "..." if len(cadsl_content) > 500 else cadsl_content,
                        "source_file": source_file,
                        "error": f"Parse error: {parse_result.errors}"
                    }

                # Transform to tool spec
                transformer = CADSLTransformer()
                tool_specs = transformer.transform(parse_result.tree)

                if not tool_specs:
                    return {
                        "success": False,
                        "results": [],
                        "count": 0,
                        "cadsl_script": cadsl_content[:500] + "..." if len(cadsl_content) > 500 else cadsl_content,
                        "source_file": source_file,
                        "error": "No tool spec generated from CADSL"
                    }

                tool_spec = tool_specs[0]

                # Build pipeline context
                reter = self.instance_manager.get_or_create_instance("default")
                rag_manager = self.default_manager.get_rag_manager() if self.default_manager else None

                from ..dsl.core import Context as PipelineContext

                # Start with default param values from CADSL file
                pipeline_params = {"rag_manager": rag_manager, "timeout_ms": timeout_ms}
                for param_spec in tool_spec.params:
                    if param_spec.default is not None:
                        pipeline_params[param_spec.name] = param_spec.default

                # Override with user-provided params
                if params:
                    pipeline_params.update(params)

                pipeline_ctx = PipelineContext(reter=reter, params=pipeline_params)

                # Build and execute pipeline
                pipeline_factory = build_pipeline_factory(tool_spec)
                pipeline = pipeline_factory(pipeline_ctx)
                result = pipeline.execute(pipeline_ctx)

                execution_time = (time.time() - start_time) * 1000

                # Check for Result monad errors (Err type)
                from ..dsl.catpy import Err, Ok
                if isinstance(result, Err):
                    error = result.value
                    return {
                        "success": False,
                        "results": [],
                        "count": 0,
                        "tool_name": tool_spec.name,
                        "tool_type": tool_spec.tool_type,
                        "source_file": source_file,
                        "execution_time_ms": execution_time,
                        "error": str(error)
                    }

                # Unwrap Ok result if needed
                if isinstance(result, Ok):
                    result = result.value

                # Result is already {"success": True/False, "results": [...], ...}
                # Add our metadata without double-wrapping
                if isinstance(result, dict) and "success" in result:
                    result["tool_name"] = tool_spec.name
                    result["tool_type"] = tool_spec.tool_type
                    result["source_file"] = source_file
                    result["execution_time_ms"] = execution_time
                    return truncate_response(result)
                else:
                    # Fallback for unexpected result format
                    return truncate_response({
                        "success": True,
                        "results": result if isinstance(result, list) else [result],
                        "count": len(result) if isinstance(result, list) else 1,
                        "tool_name": tool_spec.name,
                        "tool_type": tool_spec.tool_type,
                        "source_file": source_file,
                        "execution_time_ms": execution_time,
                        "error": None
                    })

            except Exception as e:
                return {
                    "success": False,
                    "results": [],
                    "count": 0,
                    "source_file": source_file,
                    "error": str(e)
                }

    def _register_generate_cadsl_tool(self, app: FastMCP) -> None:
        """Register the generate CADSL tool (returns query without executing)."""

        @app.tool()
        async def generate_cadsl(
            question: str,
            max_retries: int = 5,
            timeout: int = 300,
            ctx: Context = None
        ) -> Dict[str, Any]:
            """
            Generate a CADSL query from natural language without executing it.

            Translates a natural language question into a CADSL (Code Analysis DSL) query
            and returns the generated query as a string. Does NOT execute the query.

            Use this when you want to:
            - See what CADSL query would be generated for a question
            - Modify the generated query before execution
            - Learn CADSL syntax by example
            - Debug query generation

            Args:
                question: Natural language question about code
                max_retries: Maximum retry attempts on generation errors (default: 5)
                timeout: Generation timeout in seconds (default: 300)
                ctx: MCP context (injected automatically)

            Returns:
                success: Whether generation succeeded
                cadsl_query: The generated CADSL query string
                similar_tools: List of similar existing CADSL tools for reference
                error: Error message if failed

            Examples:
                - "Find all classes with more than 10 methods"
                - "Show circular import dependencies"
                - "List methods that call the save function"
                - "Generate a class diagram for the services module"
            """
            import time
            start_time = time.time()

            try:
                require_default_instance()
            except ComponentNotReadyError as e:
                return e.to_response()

            # Ensure logger is configured with correct directory
            ensure_nlq_logger_configured()

            debug_log.debug(f"\n{'#'*60}\nGENERATE CADSL REQUEST\n{'#'*60}")
            debug_log.debug(f"Question: {question}")

            if ctx is None:
                return {
                    "success": False,
                    "cadsl_query": None,
                    "error": "Context not available for LLM sampling"
                }

            if not is_agent_sdk_available():
                return {
                    "success": False,
                    "cadsl_query": None,
                    "error": "Claude Agent SDK not available. Install with: pip install claude-agent-sdk"
                }

            try:
                reter = self.instance_manager.get_or_create_instance("default")
            except Exception as e:
                return {
                    "success": False,
                    "cadsl_query": None,
                    "error": f"Failed to get RETER instance: {str(e)}"
                }

            schema_info = query_instance_schema(reter)

            # Find similar CADSL tools for context
            from .hybrid_query_engine import find_similar_cadsl_tools
            similar_tools = find_similar_cadsl_tools(question, max_results=5)

            debug_log.debug(f"Similar tools: {[t.name for t in similar_tools]}")

            # Build similar tools context if provided
            similar_tools_context = None
            if similar_tools:
                similar_tools_context = build_similar_tools_section(similar_tools)

            # Get reter instance and rag_manager for query tools
            rag_manager = self.default_manager.get_rag_manager() if self.default_manager else None

            # Get project root for file path context
            project_root = None
            if hasattr(self.instance_manager, '_project_root') and self.instance_manager._project_root:
                project_root = str(self.instance_manager._project_root)

            try:
                async with asyncio.timeout(timeout):
                    # Generate CADSL using Agent SDK (without execution)
                    result = await generate_cadsl_query(
                        question=question,
                        schema_info=schema_info,
                        max_iterations=max_retries,
                        similar_tools_context=similar_tools_context,
                        project_root=project_root,
                        reter_instance=reter,
                        rag_manager=rag_manager
                    )

                    execution_time = (time.time() - start_time) * 1000

                    if result.success and result.query:
                        return {
                            "success": True,
                            "cadsl_query": result.query,
                            "attempts": result.attempts,
                            "tools_used": result.tools_used,
                            "similar_tools": [t.to_dict() for t in similar_tools] if similar_tools else [],
                            "execution_time_ms": execution_time,
                            "error": None
                        }
                    else:
                        return {
                            "success": False,
                            "cadsl_query": None,
                            "attempts": result.attempts,
                            "tools_used": result.tools_used,
                            "similar_tools": [t.to_dict() for t in similar_tools] if similar_tools else [],
                            "execution_time_ms": execution_time,
                            "error": result.error or "Failed to generate CADSL query"
                        }

            except asyncio.TimeoutError:
                return {
                    "success": False,
                    "cadsl_query": None,
                    "error": f"Query generation timed out after {timeout} seconds"
                }
            except Exception as e:
                debug_log.error(f"Generate CADSL failed: {e}", exc_info=True)
                return {
                    "success": False,
                    "cadsl_query": None,
                    "error": str(e)
                }
