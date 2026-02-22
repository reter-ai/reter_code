"""
Unified Tools Registrar

Registers the unified MCP tools:
- thinking: RETER knowledge operations (assert, query, python_file, forget_source)
- session: Session lifecycle (returns tool guide)
- diagram: Generate UML/code diagrams

All operations go through ReterClient via ZeroMQ (remote-only mode).
"""

from typing import Dict, Any, Optional, List, TYPE_CHECKING
from fastmcp import FastMCP
from .base import ToolRegistrarBase, truncate_mcp_response

if TYPE_CHECKING:
    from ...server.reter_client import ReterClient


class UnifiedToolsRegistrar(ToolRegistrarBase):
    """
    Registers unified thinking system tools with FastMCP.

    All operations go through ReterClient via ZeroMQ.

    ::: This is-in-layer Service-Layer.
    ::: This is a registrar.
    ::: This is-in-process MCP-Server-Process.
    ::: This is stateless.
    """

    def __init__(
        self,
        instance_manager,
        persistence_service,
        tools_filter=None,
        reter_client: Optional["ReterClient"] = None
    ):
        super().__init__(instance_manager, persistence_service, tools_filter, reter_client)

    def register(self, app: FastMCP) -> None:
        """Register all unified tools (respects tools_filter)."""
        if self._should_register("thinking"):
            self._register_thinking_tool(app)
        if self._should_register("session"):
            self._register_session_tool(app)
        if self._should_register("diagram"):
            self._register_diagram_tool(app)

    def _register_thinking_tool(self, app: FastMCP) -> None:
        """Register the main thinking tool."""
        registrar = self

        @app.tool()
        def thinking(
            thought: str,
            thought_number: int,
            total_thoughts: int,
            thought_type: str = "reasoning",
            section: Optional[str] = None,
            next_thought_needed: bool = True,
            branch_id: Optional[str] = None,
            branch_from: Optional[int] = None,
            is_revision: bool = False,
            revises_thought: Optional[int] = None,
            needs_more_thoughts: bool = False,
            operations: Optional[Dict[str, Any]] = None
        ) -> Dict[str, Any]:
            """
            Design doc thinking tool with integrated operations.

            **See: guide://logical-thinking/usage for complete documentation**

            Creates a thought (optionally in a design doc section) and executes operations:
            - RETER: assert, query, python_file, forget_source

            Args:
                thought: Your current reasoning step
                thought_number: Current step number (1-indexed)
                total_thoughts: Estimated total steps
                thought_type: reasoning, analysis, decision, planning, verification
                section: Design doc section (context, goals, non_goals, design, alternatives, risks, implementation, tasks)
                next_thought_needed: Whether more thoughts are needed
                branch_id: ID for branching
                branch_from: Thought number to branch from
                is_revision: Whether this revises a previous thought
                revises_thought: Which thought number is being revised
                needs_more_thoughts: Signal that more analysis is needed
                operations: Dict of operations to execute (see examples)

            Operations examples:
                {"assert": "Every cat is a mammal."}
                {"query": "SELECT ?c WHERE { ?c type class } LIMIT 5"}
                {"python_file": "path/to/file.py"}
                {"forget_source": "path/to/file.py"}

            Returns:
                thought_number, total_thoughts, next_thought_needed, reter_operations
            """
            if registrar.reter_client is None:
                return {
                    "success": False,
                    "error": "RETER server not connected",
                }

            try:
                return registrar.reter_client.thinking(
                    thought=thought,
                    thought_number=thought_number,
                    total_thoughts=total_thoughts,
                    thought_type=thought_type,
                    section=section,
                    next_thought_needed=next_thought_needed,
                    branch_id=branch_id,
                    branch_from=branch_from,
                    is_revision=is_revision,
                    revises_thought=revises_thought,
                    needs_more_thoughts=needs_more_thoughts,
                    operations=operations
                )
            except Exception as e:
                return {
                    "success": False,
                    "error": str(e),
                }

    def _register_session_tool(self, app: FastMCP) -> None:
        """Register session lifecycle tool."""
        registrar = self

        @app.tool()
        @truncate_mcp_response
        def session(
            action: str,
            goal: Optional[str] = None,
            project_start: Optional[str] = None,
            project_end: Optional[str] = None
        ) -> Dict[str, Any]:
            """
            Session lifecycle management.

            **See: guide://reter/session-context for complete documentation**

            Actions:
            - start: Begin new session
            - context: Returns available tools guide
            - end: End session
            - clear: Reset session

            Args:
                action: start, context, end, clear
                goal: Session goal (for start action)
                project_start: Project start date ISO format (for start action)
                project_end: Project end date ISO format (for start action)

            Returns:
                For context: {tools guide with available RETER tools}
                For start/end/clear: {success: True, action: "..."}
            """
            if registrar.reter_client is None:
                return {
                    "success": False,
                    "error": "RETER server not connected",
                }

            try:
                kwargs = {"action": action}
                if goal is not None:
                    kwargs["goal"] = goal
                if project_start is not None:
                    kwargs["project_start"] = project_start
                if project_end is not None:
                    kwargs["project_end"] = project_end
                return registrar.reter_client.session(**kwargs)
            except Exception as e:
                return {
                    "success": False,
                    "error": str(e),
                }

    def _register_diagram_tool(self, app: FastMCP) -> None:
        """Register diagram generation tool."""
        registrar = self

        @app.tool()
        def diagram(
            diagram_type: str,
            format: str = "mermaid",
            root_id: Optional[str] = None,
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            target: Optional[str] = None,
            classes: Optional[List[str]] = None,
            include_methods: bool = True,
            include_attributes: bool = True,
            max_depth: int = 10,
            show_external: bool = False,
            params: Optional[Dict[str, Any]] = None
        ) -> Dict[str, Any]:
            """
            Generate diagrams and visualizations.

            **See: python://reter/tools for UML diagram documentation**

            Diagram types:

            **UML/Code Diagrams:**
            - class_hierarchy: Class inheritance hierarchy
            - class_diagram: Class diagram with methods/attributes
            - sequence: Sequence diagram of method calls
            - dependencies: Module dependency graph
            - call_graph: Call graph from entry point
            - coupling: Coupling/cohesion matrix

            Args:
                diagram_type: One of the diagram types above
                format: mermaid, markdown, json (default: mermaid)
                root_id: Root item ID for tree diagrams
                start_date: Start date filter
                end_date: End date filter
                target: Target entity (class, function, module) for UML diagrams
                classes: List of class names for class/sequence/coupling diagrams
                include_methods: Include methods in class diagrams (default: True)
                include_attributes: Include attributes in class diagrams (default: True)
                max_depth: Max depth for sequence/call_graph diagrams (default: 10)
                show_external: Show external deps in dependency diagram (default: False)
                params: Additional diagram-specific parameters

            Returns:
                {success: True, diagram: "...", format: "mermaid", ...}
            """
            if registrar.reter_client is None:
                return {
                    "success": False,
                    "error": "RETER server not connected",
                }

            try:
                kwargs = {
                    "diagram_type": diagram_type,
                    "format": format,
                    "root_id": root_id,
                    "start_date": start_date,
                    "end_date": end_date,
                    "target": target,
                    "classes": classes,
                    "include_methods": include_methods,
                    "include_attributes": include_attributes,
                    "max_depth": max_depth,
                    "show_external": show_external,
                }
                if params:
                    kwargs["params"] = params
                return registrar.reter_client.diagram(**kwargs)
            except Exception as e:
                return {
                    "success": False,
                    "error": str(e),
                }
