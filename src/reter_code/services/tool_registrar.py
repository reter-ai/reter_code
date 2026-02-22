"""
Tool Registrar Service

Handles registration and management of MCP tools.
All RETER operations go through ReterClient via ZeroMQ (remote-only mode).
"""

import os
from typing import Dict, Any, Optional, TYPE_CHECKING, Set
from fastmcp import FastMCP, Context

from ..logging_config import configure_logger_for_nlq_debug

nlq_logger = configure_logger_for_nlq_debug(__name__)
from .tools_service import ToolsRegistrar
from .registrars.system_tools import SystemToolsRegistrar
from .response_truncation import truncate_response
from .hybrid_query_engine import build_similar_tools_section, SimilarTool
from .nlq_prompts import (
    NLQ_AGENT_SYSTEM_PROMPT_TEMPLATE,
    CADSL_SYSTEM_PROMPT_TEMPLATE,
)
from ..server.reter_client import ReterClient

if TYPE_CHECKING:
    pass


# Minimal tools available in "minimal" mode
MINIMAL_TOOLS: Set[str] = {
    "reql",
    "system",
    "thinking",
    "session",
    "execute_cadsl",
    "prepare_cadsl_context",
    "semantic_search",
    "prepare_nlq_context",
}


class ToolRegistrar:
    """
    Manages MCP tool registration.

    All RETER operations go through ReterClient via ZeroMQ.

    ::: This is-in-layer Service-Layer.
    ::: This is a model-context-protocol-tool-provider.
    ::: This depends-on `reter_code.server.ReterClient`.
    ::: This is-in-process MCP-Server-Process.
    ::: This is stateless.
    """

    def __init__(
        self,
        reter_ops,
        persistence,
        instance_manager,
        default_manager,
        reter_client: Optional[ReterClient] = None
    ):
        """
        Initialize ToolRegistrar.

        Args:
            reter_ops: Kept for interface compatibility
            persistence: Kept for interface compatibility
            instance_manager: Kept for interface compatibility
            default_manager: Kept for interface compatibility
            reter_client: ReterClient for remote RETER access via ZeroMQ
        """
        self.reter_client = reter_client

        # Read tools availability mode from environment
        # "full" (default) = all tools; "minimal" = limited set
        # Legacy: "default" is treated as "full" for backward compatibility
        self._tools_mode = os.environ.get("TOOLS_AVAILABLE", "full").lower()
        if self._tools_mode == "default":
            self._tools_mode = "full"

        # Direct tools registration with reter_client
        tools_filter = MINIMAL_TOOLS if self._tools_mode == "minimal" else None
        self.tools_registrar = ToolsRegistrar(
            instance_manager, persistence, default_manager,
            tools_filter=tools_filter, reter_client=reter_client
        )

        # System tools registrar
        self.system_registrar = SystemToolsRegistrar(
            instance_manager, persistence, default_manager, reter_ops,
            reter_client=reter_client
        )

    def _query_instance_schema(self) -> str:
        """
        Query the actual schema (entity types and predicates) from RETER instance.

        Returns:
            Schema info as formatted string for inclusion in NLQ prompts.
        """
        if self.reter_client is None:
            return "Schema information unavailable (no RETER connection)"

        try:
            # Query entity types and their predicates with counts
            schema_query = """SELECT ?concept ?pred (COUNT(*) AS ?count)
                WHERE { ?s type ?concept . ?s ?pred ?o }
                GROUP BY ?concept ?pred
                ORDER BY ?concept DESC(?count)
                LIMIT 500"""

            result = self.reter_client.reql(schema_query)
            nlq_logger.debug(f"[NLQ_SCHEMA] Raw result keys: {result.keys() if isinstance(result, dict) else type(result)}")
            nlq_logger.debug(f"[NLQ_SCHEMA] Result count: {result.get('count', 'N/A') if isinstance(result, dict) else 'N/A'}")

            # REQL result format: {"columns": [...], "rows": [...], "count": N}
            rows = result.get("rows", [])
            if not rows:
                # Fallback to basic status info
                status = self.reter_client.get_status()
                return f"Codebase has {status.get('class_count', 0)} classes, {status.get('method_count', 0)} methods"

            # Group predicates by concept (entity type)
            concepts = {}
            for row in rows:
                concept = row.get("?concept", "")
                pred = row.get("?pred", "")
                count = row.get("?count", 0)
                if concept and pred:
                    if concept not in concepts:
                        concepts[concept] = []
                    concepts[concept].append(f"{pred} ({count})")

            nlq_logger.debug(f"[NLQ_SCHEMA] Found {len(concepts)} entity types from {len(rows)} rows")

            # Format schema info
            schema_lines = ["## Available Entity Types and Predicates\n"]
            for concept in sorted(concepts.keys()):
                preds = concepts[concept]
                schema_lines.append(f"### {concept}")
                # Show top 15 most common predicates
                schema_lines.append(f"Predicates: {', '.join(preds[:15])}")
                schema_lines.append("")

            schema_info = "\n".join(schema_lines)
            nlq_logger.debug(f"[NLQ_SCHEMA] Queried schema:\n{schema_info[:500]}...")
            return schema_info

        except Exception as e:
            nlq_logger.debug(f"[NLQ_SCHEMA] Failed to query schema: {e}")
            # Fallback to basic status
            try:
                status = self.reter_client.get_status()
                return f"Codebase has {status.get('class_count', 0)} classes, {status.get('method_count', 0)} methods"
            except Exception:
                return "Schema information unavailable"

    def register_all_tools(self, app: FastMCP) -> None:
        """Register all MCP tools with the application."""
        import logging
        logger = logging.getLogger(__name__)

        is_minimal_mode = self._tools_mode == "minimal"

        if is_minimal_mode:
            logger.info("Tools mode: MINIMAL (limited to: %s)", ", ".join(sorted(MINIMAL_TOOLS)))
        else:
            logger.info("Tools mode: FULL (all tools available)")

        # Knowledge tools - not in minimal mode
        if not is_minimal_mode:
            self._register_knowledge_tools(app)

        # Query tools (reql)
        if not is_minimal_mode or "reql" in MINIMAL_TOOLS:
            self._register_query_tools(app)

        # System tool
        if not is_minimal_mode or "system" in MINIMAL_TOOLS:
            self.system_registrar.register(app)

        # Domain tools
        self._register_domain_tools(app)

        # View tool (always available)
        self._register_view_tool(app)

        # CADSL examples tool (always available)
        self._register_cadsl_examples_tool(app)

        # Experimental tools
        self._register_experimental_tools(app, is_minimal_mode)

    def _register_knowledge_tools(self, app: FastMCP) -> None:
        """Register knowledge management tools."""
        registrar = self

        @app.tool()
        def add_knowledge(
            source: str,
            type: str = "ontology",
            source_id: Optional[str] = None,
            ctx: Context = None
        ) -> Dict[str, Any]:
            """
            Incrementally add knowledge to RETER's semantic memory.

            Args:
                source: Ontology content, file path, or code file path
                type: 'ontology', 'python', 'javascript', 'html', 'csharp', or 'cpp'
                source_id: Optional identifier for selective forgetting later

            Returns:
                success, items_added, execution_time_ms, source_id

            Examples:
                CNL string:  add_knowledge("Every cat is a mammal.", type="ontology")
                CNL annotations: add_knowledge("MyClass is-in-layer Core-Layer.", type="ontology")
                Code file:   add_knowledge("path/to/file.py", type="python")
            """
            if registrar.reter_client is None:
                return {"success": False, "error": "RETER server not connected"}

            try:
                return registrar.reter_client.add_knowledge(source, type, source_id)
            except Exception as e:
                return {"success": False, "error": str(e)}

        @app.tool()
        def add_external_directory(
            directory: str,
            recursive: bool = True,
            exclude_patterns: list[str] = None,
            ctx: Context = None
        ) -> Dict[str, Any]:
            """
            Load EXTERNAL code files from a directory.

            Args:
                directory: Path to directory containing code files
                recursive: Search subdirectories (default: True)
                exclude_patterns: Glob patterns to exclude

            Returns:
                success, files_loaded, total_files, total_wmes, errors
            """
            if registrar.reter_client is None:
                return {"success": False, "error": "RETER server not connected"}

            try:
                return registrar.reter_client.add_directory(directory, recursive, exclude_patterns)
            except Exception as e:
                return {"success": False, "error": str(e)}

        @app.tool()
        def validate_cnl(
            statement: str,
            context_entity: Optional[str] = None
        ) -> Dict[str, Any]:
            """
            Validate a CNL statement without adding it.

            Args:
                statement: CNL statement to validate
                context_entity: Entity name for "This" resolution

            Returns:
                success, errors, facts, resolved_statement
            """
            if registrar.reter_client is None:
                return {"success": False, "error": "RETER server not connected"}

            try:
                return registrar.reter_client.validate_cnl(statement, context_entity)
            except Exception as e:
                return {"success": False, "error": str(e)}

    def _register_query_tools(self, app: FastMCP) -> None:
        """Register query execution tools."""
        registrar = self

        @app.tool()
        def reql(
            query: str,
            type: str = "reql"
        ) -> Dict[str, Any]:
            """
            Execute a REQL query against the code knowledge graph.

            Args:
                query: Query string in REQL syntax
                type: 'reql', 'dl', or 'pattern'

            Returns:
                results, count, source_validity, warnings
            """
            if registrar.reter_client is None:
                return {"success": False, "error": "RETER server not connected"}

            try:
                if type == "reql":
                    return truncate_response(registrar.reter_client.reql(query))
                elif type == "dl":
                    instances = registrar.reter_client.dl(query)
                    return truncate_response({"success": True, "instances": instances, "count": len(instances)})
                elif type == "pattern":
                    return truncate_response(registrar.reter_client.pattern(query))
                else:
                    return {"success": False, "error": f"Unknown query type: {type}"}
            except Exception as e:
                return {"success": False, "error": str(e)}

    def _register_domain_tools(self, app: FastMCP) -> None:
        """Register all RETER domain-specific tools."""
        self.tools_registrar.register_all_tools(app)

    def _register_view_tool(self, app: FastMCP) -> None:
        """Register the view tool for pushing content to browser."""
        registrar = self

        @app.tool()
        def view(
            content: str,
            content_type: str = "auto",
        ) -> Dict[str, Any]:
            """
            Display markdown or mermaid content in a live browser view.

            Pushes content to the RETER ViewServer which renders it in a
            browser page with mermaid.js diagram support and dark theme.

            The content can be:
            - A file path (reads and displays the file)
            - Inline markdown (rendered with marked.js; mermaid code blocks auto-render)
            - Inline mermaid diagram source (rendered with mermaid.js)
            - Raw HTML

            Args:
                content: File path or inline content string to display.
                    If the string is a path to an existing file, the file is read.
                content_type: "auto" (default), "markdown", "mermaid", or "html".
                    With "auto", detects mermaid by leading keywords (graph, flowchart,
                    sequenceDiagram, classDiagram, stateDiagram, erDiagram, gantt, pie,
                    gitgraph, mindmap, timeline, block-beta, etc).

            Returns:
                success, content_type, content_length, view_url
            """
            if registrar.reter_client is None:
                return {"success": False, "error": "RETER server not connected"}

            actual_content = content
            source_file = None

            # Check if content is a file path
            import os
            stripped = content.strip()
            if (
                not stripped.startswith(("{", "<", "#", "graph ", "flowchart ", "sequenceDiagram"))
                and len(stripped) < 500
                and ("\n" not in stripped)
            ):
                # Could be a file path
                if os.path.isfile(stripped):
                    source_file = stripped
                    with open(stripped, "r", encoding="utf-8") as f:
                        actual_content = f.read()

            # Auto-detect content type
            actual_type = content_type
            if actual_type == "auto":
                first_line = actual_content.lstrip().split("\n", 1)[0].strip().lower()
                mermaid_keywords = (
                    "graph ", "graph\n", "flowchart ", "flowchart\n",
                    "sequencediagram", "classdiagram", "statediagram",
                    "erdiagram", "gantt", "pie", "gitgraph",
                    "mindmap", "timeline", "block-beta",
                    "journey", "quadrantchart", "requirementdiagram",
                    "c4context", "c4container", "c4component", "c4deployment",
                    "sankey-beta", "xychart-beta",
                )
                if any(first_line.startswith(kw) for kw in mermaid_keywords):
                    actual_type = "mermaid"
                elif stripped.lstrip().startswith("<"):
                    actual_type = "html"
                else:
                    actual_type = "markdown"

            # Validate mermaid syntax before pushing
            validation_info = None
            if actual_type == "mermaid":
                try:
                    from reter_code.mermaid.validator import validate_mermaid
                    vr = validate_mermaid(actual_content)
                    validation_info = vr.to_dict()
                    if not vr.valid:
                        return {
                            "success": False,
                            "error": "Mermaid syntax validation failed",
                            "validation": validation_info,
                        }
                except Exception:
                    pass
            elif actual_type == "markdown":
                try:
                    from reter_code.mermaid.markdown_validator import validate_markdown
                    vr = validate_markdown(actual_content)
                    validation_info = vr.to_dict()
                    if not vr.valid:
                        return {
                            "success": False,
                            "error": "Markdown validation failed (embedded mermaid errors)",
                            "validation": validation_info,
                        }
                except Exception:
                    pass

            try:
                result = registrar.reter_client.view_push(actual_content, actual_type)
                if source_file:
                    result["source_file"] = source_file
                if validation_info:
                    result["validation"] = validation_info
                # Include the view URL from discovery if available
                try:
                    discovery = registrar.reter_client.config.discover_server()
                    if discovery and discovery.view_url:
                        result["view_url"] = discovery.view_url
                except Exception:
                    pass
                return result
            except Exception as e:
                return {"success": False, "error": str(e)}

    def _register_cadsl_examples_tool(self, app: FastMCP) -> None:
        """Register the CADSL examples browser/search tool."""
        registrar = self

        @app.tool()
        def cadsl_examples(
            action: str = "list",
            query: str = "",
            category: str = "",
            name: str = "",
            max_results: int = 10,
        ) -> Dict[str, Any]:
            """
            Browse and search the CADSL tool library.

            Use this to discover available analysis tools, find examples by
            category, search for tools matching a description, or read a
            specific tool's source code.

            Args:
                action: One of:
                    - "list"   : List all tools grouped by category.
                                 Optionally filter by category.
                    - "search" : Semantic search for tools matching a query.
                    - "get"    : Get full source code of a specific tool.
                query: Search query (required for action="search").
                category: Category filter for "list", e.g. "diagrams", "smells".
                name: Tool name for "get", e.g. "diagrams/class_hierarchy"
                      or just "class_hierarchy".
                max_results: Max results for "search" (default: 10).

            Returns:
                success, content (formatted text)
            """
            if registrar.reter_client is None:
                return {"success": False, "error": "RETER server not connected"}

            try:
                return registrar.reter_client.cadsl_examples(
                    action=action,
                    query=query,
                    category=category,
                    name=name,
                    max_results=max_results,
                )
            except Exception as e:
                return {"success": False, "error": str(e)}

    def _register_experimental_tools(self, app: FastMCP, is_minimal_mode: bool = False) -> None:
        """Register experimental tools."""
        if not is_minimal_mode or "prepare_nlq_context" in MINIMAL_TOOLS:
            self._register_prepare_nlq_context_tool(app)

        if not is_minimal_mode or "execute_cadsl" in MINIMAL_TOOLS:
            self._register_cadsl_tool(app)

        if not is_minimal_mode or "prepare_cadsl_context" in MINIMAL_TOOLS:
            self._register_prepare_cadsl_context_tool(app)

    def _register_prepare_nlq_context_tool(self, app: FastMCP) -> None:
        """Register the prepare NLQ context tool."""
        registrar = self

        @app.tool()
        def prepare_nlq_context(
            question: str,
            max_similar_tools: int = 5
        ) -> Dict[str, Any]:
            """Prepare context for a natural language code query.

            Gathers schema info and similar CADSL examples, returns a formatted
            prompt for use with a Claude Code Task subagent.

            Args:
                question: Natural language question about code
                max_similar_tools: Maximum similar tools to include (default: 5)

            Returns:
                success, prompt
            """
            if registrar.reter_client is None:
                return {"success": False, "error": "RETER server not connected"}

            # 1. Get schema
            schema_info = registrar._query_instance_schema()

            # 2. Find similar CADSL tools
            similar_result = registrar.reter_client.similar_cadsl_tools(
                question, max_results=max_similar_tools
            )
            similar_tools_text = ""
            if similar_result.get("success"):
                tools = similar_result.get("similar_tools", [])
                if tools:
                    similar_tools_text = build_similar_tools_section(
                        [SimilarTool(**t) for t in tools]
                    )

            # 3. Format prompt
            import os
            project_root = os.environ.get("RETER_PROJECT_ROOT", os.getcwd())
            system_prompt = NLQ_AGENT_SYSTEM_PROMPT_TEMPLATE.format(
                project_root=project_root,
                schema_info=schema_info
            )

            user_prompt = f"Question: {question}"
            if similar_tools_text:
                user_prompt += f"\n\n{similar_tools_text}"

            return {
                "success": True,
                "prompt": f"{system_prompt}\n\n---\n\n{user_prompt}"
            }

    def _register_cadsl_tool(self, app: FastMCP) -> None:
        """Register the CADSL script execution tool."""
        registrar = self

        @app.tool()
        def execute_cadsl(
            script: str,
            params: Optional[Dict[str, Any]] = None,
            timeout_ms: int = 300000
        ) -> Dict[str, Any]:
            """
            Execute a CADSL script for code analysis.

            Args:
                script: CADSL script string or path to .cadsl file
                params: Optional parameters to pass to the script
                timeout_ms: Timeout in milliseconds (default: 300000)

            Returns:
                success, results, count, tool_name, tool_type, error
            """
            import os
            import time
            from pathlib import Path

            if registrar.reter_client is None:
                return {"success": False, "error": "RETER server not connected"}

            start_time = time.time()

            # Determine if script is a file path or inline CADSL
            cadsl_content = script
            source_file = None

            script_stripped = script.strip()
            is_inline_cadsl = (
                script_stripped.startswith('query ') or
                script_stripped.startswith('detector ') or
                script_stripped.startswith('diagram ') or
                script_stripped.startswith('//')
            )

            if not is_inline_cadsl and (script_stripped.endswith('.cadsl') or os.path.sep in script_stripped or '/' in script_stripped):
                path = Path(script_stripped)
                if path.exists() and path.is_file():
                    source_file = str(path)
                    with open(path, 'r', encoding='utf-8') as f:
                        cadsl_content = f.read()
                elif not path.exists():
                    # Try category/name format (e.g., "security/auth_handler_complexity")
                    cadsl_tools_dir = Path(__file__).parent.parent / "cadsl" / "tools"
                    cadsl_tool_path = cadsl_tools_dir / f"{script_stripped}.cadsl" if not script_stripped.endswith('.cadsl') else cadsl_tools_dir / script_stripped
                    if cadsl_tool_path.exists() and cadsl_tool_path.is_file():
                        source_file = str(cadsl_tool_path)
                        with open(cadsl_tool_path, 'r', encoding='utf-8') as f:
                            cadsl_content = f.read()
                    else:
                        return {"success": False, "error": f"CADSL file not found: {script}"}

            try:
                result = registrar.reter_client.execute_cadsl(cadsl_content, params, timeout_ms)
                result["source_file"] = source_file
                result["execution_time_ms"] = (time.time() - start_time) * 1000
                return truncate_response(result)
            except Exception as e:
                return {"success": False, "error": str(e), "source_file": source_file}

    def _register_prepare_cadsl_context_tool(self, app: FastMCP) -> None:
        """Register the prepare CADSL context tool."""
        registrar = self

        @app.tool()
        def prepare_cadsl_context(
            question: str,
            max_similar_tools: int = 5
        ) -> Dict[str, Any]:
            """Prepare context for generating a CADSL query from natural language.

            Gathers schema info and similar CADSL examples, returns a formatted
            prompt for use with a Claude Code Task subagent.

            Args:
                question: Natural language question about code
                max_similar_tools: Maximum similar tools to include (default: 5)

            Returns:
                success, prompt
            """
            if registrar.reter_client is None:
                return {"success": False, "error": "RETER server not connected"}

            schema_info = registrar._query_instance_schema()

            similar_result = registrar.reter_client.similar_cadsl_tools(
                question, max_results=max_similar_tools
            )
            similar_tools_text = ""
            if similar_result.get("success"):
                tools = similar_result.get("similar_tools", [])
                if tools:
                    similar_tools_text = build_similar_tools_section(
                        [SimilarTool(**t) for t in tools]
                    )

            import os
            project_root = os.environ.get("RETER_PROJECT_ROOT", os.getcwd())
            system_prompt = CADSL_SYSTEM_PROMPT_TEMPLATE.format(
                project_root=project_root
            )

            user_prompt = f"Question: {question}"
            if similar_tools_text:
                user_prompt += f"\n\n{similar_tools_text}"

            # Include schema as part of user prompt
            if schema_info:
                user_prompt = f"{schema_info}\n\n{user_prompt}"

            return {
                "success": True,
                "prompt": f"{system_prompt}\n\n---\n\n{user_prompt}"
            }
