"""
Unified Tools Registrar

Registers the 4 unified MCP tools:
- thinking: Main thinking tool with operations
- session: Session lifecycle (start, context, end, clear)
- items: Query and manage items
- diagram: Generate diagrams

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
        if self._should_register("items"):
            self._register_items_tool(app)
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
            - Create items: task (with category), milestone
            - Create relations: traces, implements, depends_on, affects
            - Update items: update_item, update_task, complete_task
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
                {"task": {"name": "Implement X", "category": "feature", "priority": "high"}}
                {"milestone": {"name": "MVP Release", "date": "2024-03-01"}}
                {"traces": ["TASK-001"], "affects": ["module.py"]}
                {"complete_task": "TASK-001"}

            Returns:
                thought_id, thought_number, items_created, relations_created, session_status
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
            - start: Begin new session (params: goal, project_start, project_end)
            - context: **CRITICAL** Restore full context after compactification
            - end: Archive session (preserves data)
            - clear: Reset session (deletes all data)

            IMPORTANT: Call action="context" at session start and after compactification!

            Args:
                action: start, context, end, clear
                goal: Session goal (for start action)
                project_start: Project start date ISO format (for start action)
                project_end: Project end date ISO format (for start action)

            Returns:
                For start: {session_id, goal, status, created_at}
                For context: {session, design_doc, tasks, project_health, milestones, suggestions, mcp_guide}
                For end: {session_id, status, summary}
                For clear: {success, items_deleted}
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

    def _register_items_tool(self, app: FastMCP) -> None:
        """Register items query and management tool."""
        registrar = self

        @app.tool()
        @truncate_mcp_response
        def items(
            action: str = "list",
            item_id: Optional[str] = None,
            updates: Optional[Dict[str, Any]] = None,
            item_type: Optional[str] = None,
            status: Optional[str] = None,
            priority: Optional[str] = None,
            phase: Optional[str] = None,
            category: Optional[str] = None,
            source_tool: Optional[str] = None,
            traces_to: Optional[str] = None,
            traced_by: Optional[str] = None,
            depends_on: Optional[str] = None,
            blocks: Optional[str] = None,
            affects: Optional[str] = None,
            start_after: Optional[str] = None,
            end_before: Optional[str] = None,
            include_relations: bool = False,
            limit: int = 100,
            offset: int = 0,
            classification: Optional[str] = None,
            notes: Optional[str] = None,
            verified_by: Optional[str] = None,
            update_status: bool = False,
            create_followup: bool = False,
            followup_name: Optional[str] = None,
            followup_description: Optional[str] = None,
            followup_prompt: Optional[str] = None,
            followup_category: Optional[str] = None,
            followup_priority: Optional[str] = None,
            complete_original: bool = False
        ) -> Dict[str, Any]:
            """
            Query and manage items (thoughts, tasks, milestones).

            Actions:
            - list: Query items with filters
            - get: Get single item by ID
            - delete: Delete item and relations
            - update: Update item fields
            - clear: Delete multiple items matching filters
            - classify: Classify task as TP or FP (requires item_id and classification)
            - verify: Mark task as verified (requires item_id)

            Args:
                action: list, get, delete, update, clear, classify, verify
                item_id: Item ID (required for get/delete/update/classify/verify)
                updates: Fields to update (for update action)
                item_type: Filter by type (thought, requirement, task, etc.)
                status: Filter by status (pending, in_progress, completed, etc.)
                priority: Filter by priority (critical, high, medium, low)
                phase: Filter by project phase
                category: Filter by category
                source_tool: Filter by source tool
                traces_to: Items that trace to this ID
                traced_by: Items traced by this ID
                depends_on: Items depending on this ID
                blocks: Items blocked by this ID
                affects: Items affecting this file/entity
                start_after: Tasks starting after this date
                end_before: Tasks ending before this date
                include_relations: Include related items in response
                limit: Maximum items to return
                offset: Pagination offset
                classification: Classification for classify action or list filter. Valid values:
                    TP-EXTRACT, TP-PARAMETERIZE, PARTIAL-TP,
                    FP-INTERFACE, FP-LAYERS, FP-STRUCTURAL, FP-TRIVIAL
                    For list: use "TP" or "FP" prefix to match all TP-* or FP-* classifications
                notes: Optional notes for classification
                verified_by: Who verified (for verify action, default: "user")
                update_status: Update status to "verified" (for verify action)
                create_followup: For TP classifications, create a follow-up implementation task
                followup_name: Custom name for follow-up task (auto-generated if not provided)
                followup_description: Description for follow-up task
                followup_prompt: Custom prompt for Claude Code (auto-generated based on classification)
                followup_category: Category for follow-up task (default: "refactor")
                followup_priority: Priority for follow-up task (default: same as original)
                complete_original: Mark original task as completed when creating follow-up

            Returns:
                For list: {items: [...], count, has_more}
                For get: {item: {...}, relations: {...}}
                For delete: {success, deleted_relations}
                For update: {item: {...}}
                For clear: {success, items_deleted, relations_deleted}
                For classify: {success, item: {...}, followup_task?: {...}}
                For verify: {success, item: {...}}
            """
            if registrar.reter_client is None:
                return {
                    "success": False,
                    "error": "RETER server not connected",
                }

            try:
                kwargs = {
                    "action": action,
                    "item_id": item_id,
                    "updates": updates,
                    "item_type": item_type,
                    "status": status,
                    "priority": priority,
                    "phase": phase,
                    "category": category,
                    "source_tool": source_tool,
                    "traces_to": traces_to,
                    "traced_by": traced_by,
                    "depends_on": depends_on,
                    "blocks": blocks,
                    "affects": affects,
                    "start_after": start_after,
                    "end_before": end_before,
                    "include_relations": include_relations,
                    "limit": limit,
                    "offset": offset,
                    "classification": classification,
                    "notes": notes,
                    "verified_by": verified_by,
                    "update_status": update_status,
                    "create_followup": create_followup,
                    "followup_name": followup_name,
                    "followup_description": followup_description,
                    "followup_prompt": followup_prompt,
                    "followup_category": followup_category,
                    "followup_priority": followup_priority,
                    "complete_original": complete_original,
                }
                # Remove None values
                kwargs = {k: v for k, v in kwargs.items() if v is not None}
                return registrar.reter_client.items(**kwargs)
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

            **Session/Project Diagrams:**
            - gantt: Gantt chart for tasks and milestones
            - thought_chain: Reasoning chain with branches
            - design_doc: Design doc structure with sections
            - traceability: Task traceability matrix

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
                start_date: Start date filter for gantt
                end_date: End date filter for gantt
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
