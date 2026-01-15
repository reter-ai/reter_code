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
from .registrars.system_tools import SystemToolsRegistrar
from .nlq_helpers import (
    query_instance_schema,
    execute_reql_query,
    is_retryable_error
)
from .response_truncation import truncate_response
from .hybrid_query_engine import (
    classify_query_with_llm,
    QueryType,
    QueryClassification,
    SimilarTool,
    build_cadsl_prompt,
    extract_cadsl_from_response,
    extract_query_from_response,
    build_rag_query_params,
    build_similar_tools_section,
    HybridQueryResult,
    generate_query_with_tools,
    QUERY_TOOLS,
    handle_tool_call,
    REQL_GENERATION_PROMPT,
    CADSL_GENERATION_PROMPT,
)

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

        # System tools registrar (unified system management)
        self.system_registrar = SystemToolsRegistrar(
            instance_manager, persistence, default_manager, reter_ops
        )

    def register_all_tools(self, app: FastMCP) -> None:
        """
        Register all MCP tools with the application.

        Args:
            app: FastMCP application instance
        """
        self._register_knowledge_tools(app)
        self._register_query_tools(app)
        self.system_registrar.register(app)  # Unified system tool
        self._register_domain_tools(app)
        self._register_experimental_tools(app)

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
        def quick_query(
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
            result = self.reter_ops.quick_query("default", query, type)
            return truncate_response(result)

    def _register_domain_tools(self, app: FastMCP) -> None:
        """Register all RETER domain-specific tools (Python analysis, UML, Gantt, etc.)."""
        self.tools_registrar.register_all_tools(app)

    def _register_experimental_tools(self, app: FastMCP) -> None:
        """Register experimental tools for testing new features."""
        self._register_nlq_tool(app)
        self._register_hybrid_query_tool(app)
        self._register_cadsl_tool(app)

    def _register_nlq_tool(self, app: FastMCP) -> None:
        """Register the natural language query tool."""

        @app.tool()
        async def natural_language_query(
            question: str,
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
            debug_log.debug(f"Question: {question}, Instance: default")

            if ctx is None:
                return self._nlq_error_response("Context not available for LLM sampling")

            try:
                reter = self.instance_manager.get_or_create_instance("default")
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
        ctx,
        similar_tools: Optional[list] = None
    ) -> Dict[str, Any]:
        """Execute NLQ with retry logic for syntax errors using tool-augmented generation."""
        attempts = 0
        last_error = None
        generated_query = None
        tools_used = []

        while attempts < max_retries:
            attempts += 1
            try:
                debug_log.debug(f"\n{'='*60}\nREQL ATTEMPT {attempts}/{max_retries}\n{'='*60}")

                # Build prompt - include error feedback if retrying
                if last_error and attempts > 1:
                    prompt = f"""Previous REQL query failed with error:
{last_error}

Previous query was:
```reql
{generated_query}
```

Please fix the syntax error and regenerate. Original question: {question}

Schema info: {schema_info}"""
                else:
                    prompt = f"{schema_info}\n\nQuestion: {question}"

                # Use tool-augmented generation with case-based reasoning
                # Only pass similar tools on first attempt
                tools_for_attempt = similar_tools if attempts == 1 else None
                generated_query, new_tools = await self._generate_with_tools(
                    prompt, QueryType.REQL, ctx, max_tool_iterations=5,
                    similar_tools=tools_for_attempt
                )
                tools_used.extend(new_tools)
                debug_log.debug(f"GENERATED REQL QUERY:\n{generated_query}")
                debug_log.debug(f"TOOLS USED: {tools_used}")

                results = execute_reql_query(reter, generated_query)
                total_count = len(results)
                debug_log.debug(f"QUERY SUCCESS: {total_count} results")

                # Check for potential cross-join (excessive results)
                # Only warn if results exceed threshold AND query lacks aggregation
                # Aggregation queries (GROUP BY, COUNT) legitimately return many results
                cross_join_threshold = 5000
                has_aggregation = any(kw in generated_query.upper() for kw in ['GROUP BY', 'COUNT(', 'SUM(', 'AVG(', 'MIN(', 'MAX('])

                if total_count > cross_join_threshold and not has_aggregation:
                    debug_log.debug(f"POSSIBLE CROSS-JOIN: {total_count} results exceeds threshold (no aggregation)")
                    # Return results with warning instead of failing - let user decide
                    results_sample = results[:max_results]
                    return truncate_response({
                        "success": True,  # Changed from False - return results with warning
                        "results": results_sample,
                        "count": total_count,
                        "reql_query": generated_query,
                        "attempts": attempts,
                        "tools_used": tools_used,
                        "error": None,
                        "warning": f"Large result set ({total_count} rows). If unexpected, the query may have a cross-join. "
                                   f"Showing first {len(results_sample)} results.",
                        "truncated": True
                    })

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
                    "tools_used": tools_used,
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
                    return {
                        "success": False,
                        "results": [],
                        "count": 0,
                        "reql_query": generated_query,
                        "attempts": attempts,
                        "tools_used": tools_used,
                        "error": last_error
                    }

        debug_log.debug(f"MAX RETRIES EXHAUSTED after {attempts} attempts")
        return {
            "success": False,
            "results": [],
            "count": 0,
            "reql_query": generated_query,
            "attempts": attempts,
            "tools_used": tools_used,
            "error": f"Failed after {max_retries} attempts. Last error: {last_error}"
        }

    async def _execute_cadsl_query(
        self,
        question: str,
        schema_info: str,
        max_retries: int,
        ctx,
        similar_tools: Optional[list] = None
    ) -> Dict[str, Any]:
        """Execute a CADSL query using tool-augmented LLM generation."""
        from ..cadsl.parser import parse_cadsl
        from ..cadsl.transformer import CADSLTransformer
        from ..cadsl.loader import build_pipeline_factory

        attempts = 0
        last_error = None
        generated_query = None
        tools_used = []

        while attempts < max_retries:
            attempts += 1
            try:
                debug_log.debug(f"\n{'='*60}\nCADSL ATTEMPT {attempts}/{max_retries}\n{'='*60}")

                # Build prompt - include error feedback if retrying
                if last_error and attempts > 1:
                    prompt = f"""Previous CADSL query failed with error:
{last_error}

Previous query was:
```cadsl
{generated_query}
```

Please fix the syntax error and regenerate. Original question: {question}"""
                else:
                    prompt = question

                # Use tool-augmented generation with case-based reasoning
                # Only pass similar tools on first attempt
                tools_for_attempt = similar_tools if attempts == 1 else None
                generated_query, new_tools = await self._generate_with_tools(
                    prompt, QueryType.CADSL, ctx, max_tool_iterations=5,
                    similar_tools=tools_for_attempt
                )
                tools_used.extend(new_tools)
                debug_log.debug(f"GENERATED CADSL QUERY:\n{generated_query}")
                debug_log.debug(f"TOOLS USED: {tools_used}")

                # Parse and execute CADSL
                parse_result = parse_cadsl(generated_query)
                if not parse_result.success:
                    raise Exception(f"Parse error: {parse_result.errors}")

                transformer = CADSLTransformer()
                tool_specs = transformer.transform(parse_result.tree)

                if not tool_specs:
                    raise Exception("No tool spec generated from CADSL")

                # Build and execute pipeline
                reter = self.instance_manager.get_or_create_instance("default")
                rag_manager = self.default_manager.get_rag_manager() if self.default_manager else None

                from ..dsl.core import Context as PipelineContext
                pipeline_ctx = PipelineContext(reter=reter, params={"rag_manager": rag_manager})

                tool_spec = tool_specs[0]
                pipeline_factory = build_pipeline_factory(tool_spec)
                pipeline = pipeline_factory(pipeline_ctx)

                result = pipeline.execute(pipeline_ctx)

                # Result is already {"success": True/False, "results": [...], ...}
                # Add our metadata without double-wrapping
                if isinstance(result, dict) and "success" in result:
                    result["cadsl_query"] = generated_query
                    result["query_type"] = "cadsl"
                    result["attempts"] = attempts
                    result["tools_used"] = tools_used
                    return result
                else:
                    return {
                        "success": True,
                        "results": result if isinstance(result, list) else [result],
                        "count": len(result) if isinstance(result, list) else 1,
                        "cadsl_query": generated_query,
                        "query_type": "cadsl",
                        "attempts": attempts,
                        "tools_used": tools_used,
                        "error": None
                    }

            except Exception as e:
                last_error = str(e)
                debug_log.debug(f"CADSL ERROR: {last_error}")

                if is_retryable_error(last_error):
                    continue
                else:
                    return {
                        "success": False,
                        "results": [],
                        "count": 0,
                        "cadsl_query": generated_query,
                        "query_type": "cadsl",
                        "attempts": attempts,
                        "tools_used": tools_used,
                        "error": last_error
                    }

        return {
            "success": False,
            "results": [],
            "count": 0,
            "cadsl_query": generated_query,
            "query_type": "cadsl",
            "attempts": attempts,
            "tools_used": tools_used,
            "error": f"Failed after {max_retries} attempts. Last error: {last_error}"
        }

    async def _generate_with_tools(
        self,
        question: str,
        query_type: QueryType,
        ctx,
        max_tool_iterations: int = 5,
        similar_tools: Optional[list] = None
    ) -> tuple:
        """
        Generate a query using tool-augmented LLM with case-based reasoning.

        Uses a text-based tool calling pattern where LLM can request tools via
        structured output like: TOOL_CALL: get_reql_grammar()

        If similar tools are provided from case-based reasoning, they are included
        in the prompt as templates the LLM can adapt.

        Returns:
            Tuple of (generated_query, list_of_tools_used)
        """
        import re

        system_prompt = REQL_GENERATION_PROMPT if query_type == QueryType.REQL else CADSL_GENERATION_PROMPT

        # Add tool calling instructions
        tool_instructions = """
You have these tools available. To use a tool, output EXACTLY this format on its own line:
TOOL_CALL: tool_name(arg="value")

Available tools:
- TOOL_CALL: get_reql_grammar()  - Get REQL grammar
- TOOL_CALL: get_cadsl_grammar() - Get CADSL grammar
- TOOL_CALL: list_examples(category="smells") - List examples (categories: smells, diagrams, rag, testing, inspection, refactoring, patterns, exceptions, dependencies)
- TOOL_CALL: get_example(name="god_class") - Get a specific example

When you have enough information, output the final query in a code block.
"""
        full_system = system_prompt + "\n" + tool_instructions

        # Build user message with similar tools from case-based reasoning
        conversation = f"Generate a query for: {question}"

        if similar_tools:
            similar_section = build_similar_tools_section(similar_tools)
            conversation = f"{conversation}\n{similar_section}"
            debug_log.debug(f"CASE-BASED: Including {len(similar_tools)} similar tools as templates")
        tools_used = []

        for iteration in range(max_tool_iterations):
            debug_log.debug(f"Tool iteration {iteration + 1}")

            response = await ctx.sample(conversation, system_prompt=full_system)
            response_text = response.text if hasattr(response, 'text') else str(response)

            debug_log.debug(f"LLM response (first 500): {response_text[:500]}")

            # Check for tool calls
            tool_pattern = r'TOOL_CALL:\s*(\w+)\((.*?)\)'
            tool_matches = re.findall(tool_pattern, response_text)

            if not tool_matches:
                # No tool calls - extract the query
                query = extract_query_from_response(response_text, query_type)
                return query, tools_used

            # Process tool calls
            tool_results = []
            for tool_name, args_str in tool_matches:
                debug_log.debug(f"Tool call: {tool_name}({args_str})")
                tools_used.append(tool_name)

                # Parse arguments
                tool_input = {}
                if args_str:
                    # Parse key="value" patterns
                    arg_pattern = r'(\w+)\s*=\s*"([^"]*)"'
                    for key, value in re.findall(arg_pattern, args_str):
                        tool_input[key] = value

                # Execute tool
                result = handle_tool_call(tool_name, tool_input)
                tool_results.append(f"Result of {tool_name}:\n{result}")

            # Add tool results to conversation and continue
            conversation = response_text + "\n\n" + "\n\n".join(tool_results) + "\n\nNow generate the final query:"

        # Max iterations reached - try to extract query anyway
        query = extract_query_from_response(response_text, query_type)
        return query, tools_used

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
                    min_size=params.get("min_size", 2),
                    exclude_same_file=params.get("exclude_same_file", True),
                    exclude_same_class=params.get("exclude_same_class", True),
                    entity_types=params.get("entity_types"),
                )
                return {
                    "success": True,
                    "results": result.get("clusters", []),
                    "count": result.get("count", 0),
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

    def _register_hybrid_query_tool(self, app: FastMCP) -> None:
        """Register the hybrid natural language query tool."""

        @app.tool()
        async def hybrid_query(
            question: str,
            force_type: str = None,
            max_retries: int = 5,
            timeout: int = 60,
            max_results: int = 500,
            ctx: Context = None
        ) -> Dict[str, Any]:
            """
            Smart natural language query that routes to the best execution engine.

            This is an enhanced version of natural_language_query that automatically
            detects the query type and routes to:
            - REQL: For structural code queries (classes, methods, inheritance, etc.)
            - CADSL: For complex queries (graph algorithms, diagrams, joins)
            - RAG: For semantic/similarity queries (find similar code, duplicates)
            - Hybrid: For queries combining structure and semantics

            Args:
                question: Natural language question about code
                force_type: Force a specific query type: "reql", "cadsl", "rag", or None for auto
                max_retries: Maximum retry attempts on errors (default: 5)
                timeout: Query timeout in seconds (default: 60)
                max_results: Maximum results to return (default: 500)
                ctx: MCP context (injected automatically)

            Returns:
                success: Whether query succeeded
                results: Query results as list of dicts
                count: Number of results
                query_type: Type of query executed (reql/cadsl/rag)
                classification: How the query was classified
                generated_query: The REQL or CADSL query generated (if applicable)
                error: Error message if failed

            Examples:
                - "Find all classes with more than 10 methods" -> REQL
                - "Show circular import dependencies" -> CADSL (graph_cycles)
                - "Find code similar to authentication handlers" -> RAG
                - "Generate a class diagram for the services module" -> CADSL (diagram)
                - "Find duplicate methods across files" -> CADSL+RAG (hybrid)
            """
            import time
            start_time = time.time()

            try:
                require_default_instance()
            except ComponentNotReadyError as e:
                return e.to_response()

            if ctx is None:
                return {
                    "success": False,
                    "results": [],
                    "count": 0,
                    "error": "Context not available for LLM sampling"
                }

            # Override with forced type if specified, otherwise use LLM classification
            if force_type:
                type_map = {
                    "reql": QueryType.REQL,
                    "cadsl": QueryType.CADSL,
                    "rag": QueryType.RAG,
                }
                if force_type.lower() in type_map:
                    classification = QueryClassification(
                        query_type=type_map[force_type.lower()],
                        confidence=1.0,
                        reasoning=f"Forced to {force_type} by user"
                    )
                else:
                    classification = await classify_query_with_llm(question, ctx)
            else:
                # Use LLM to classify the query
                classification = await classify_query_with_llm(question, ctx)

            debug_log.debug(f"QUERY CLASSIFICATION: {classification}")

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

            try:
                async with asyncio.timeout(timeout):
                    if classification.query_type == QueryType.RAG:
                        # Direct RAG query
                        result = self._execute_rag_query(question)

                    elif classification.query_type == QueryType.CADSL:
                        # CADSL pipeline query with case-based reasoning
                        result = await self._execute_cadsl_query(
                            question, schema_info, max_retries, ctx,
                            similar_tools=classification.similar_tools
                        )

                    else:  # QueryType.REQL
                        # Standard REQL query with case-based reasoning
                        result = await self._execute_nlq_with_retries(
                            reter, question, schema_info, max_retries, max_results, ctx,
                            similar_tools=classification.similar_tools
                        )

                    # Add classification info to result
                    result["classification"] = {
                        "type": classification.query_type.value,
                        "confidence": classification.confidence,
                        "reasoning": classification.reasoning,
                    }
                    if classification.suggested_cadsl_tool:
                        result["suggested_tool"] = classification.suggested_cadsl_tool

                    # Add case-based reasoning info if similar tools were used
                    if classification.similar_tools:
                        result["similar_tools"] = [t.to_dict() for t in classification.similar_tools]

                    result["execution_time_ms"] = (time.time() - start_time) * 1000

                    return truncate_response(result)

            except asyncio.TimeoutError:
                return {
                    "success": False,
                    "results": [],
                    "count": 0,
                    "query_type": classification.query_type.value,
                    "error": f"Query timed out after {timeout} seconds",
                    "classification": {
                        "type": classification.query_type.value,
                        "confidence": classification.confidence,
                        "reasoning": classification.reasoning,
                    }
                }

    def _register_cadsl_tool(self, app: FastMCP) -> None:
        """Register the CADSL script execution tool."""

        @app.tool()
        def execute_cadsl(
            script: str,
            params: Optional[Dict[str, Any]] = None
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
                                ?c type oo:Class . ?c name ?name .
                                ?m type oo:Method . ?m definedIn ?c
                            }
                            GROUP BY ?c ?name
                            HAVING (?method_count > 10)
                        }
                        | emit { results }
                    }
                ''')

                # Execute from file
                execute_cadsl("path/to/detector.cadsl", params={"threshold": 15})
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

            # Check if it looks like a file path
            if script.strip().endswith('.cadsl') or os.path.sep in script or '/' in script:
                path = Path(script)
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
                pipeline_params = {"rag_manager": rag_manager}
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

