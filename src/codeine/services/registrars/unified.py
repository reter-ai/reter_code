"""
Unified Tools Registrar

Registers the 4 unified MCP tools:
- thinking: Main thinking tool with operations
- session: Session lifecycle (start, context, end, clear)
- items: Query and manage items
- diagram: Generate diagrams
"""

from typing import Dict, Any, Optional, List
from fastmcp import FastMCP
from .base import ToolRegistrarBase, truncate_mcp_response
from ...tools.dataclasses import ItemsQueryFilters
from ...reter_wrapper import DefaultInstanceNotInitialised, is_initialization_complete
from ..initialization_progress import (
    get_initializing_response,
    require_sql,
    require_default_instance,
    ComponentNotReadyError,
)


class UnifiedToolsRegistrar(ToolRegistrarBase):
    """Registers unified thinking system tools with FastMCP."""

    def __init__(self, instance_manager, persistence_service):
        super().__init__(instance_manager, persistence_service)
        self._sessions = {}  # instance_name -> ThinkingSession

    def _get_session(self, instance_name: str):
        """Get or create a ThinkingSession for the instance."""
        if instance_name not in self._sessions:
            from ...tools.unified import UnifiedStore, ThinkingSession
            from pathlib import Path

            # Use persistence directory for store
            if self.persistence:
                db_path = Path(self.persistence.snapshots_dir) / ".unified.sqlite"
            else:
                db_path = Path.cwd() / ".codeine" / ".unified.sqlite"

            db_path.parent.mkdir(parents=True, exist_ok=True)
            store = UnifiedStore(str(db_path))

            # Get RETER engine for the instance (optional)
            try:
                reter_engine = self._get_reter(instance_name)
            except (KeyError, RuntimeError, OSError):
                # KeyError: Instance not found
                # RuntimeError: RETER initialization failed
                # OSError: File system issues
                reter_engine = None

            self._sessions[instance_name] = ThinkingSession(store, reter_engine)

        return self._sessions[instance_name]

    def _query_items(self, store, session_id: str, filters: ItemsQueryFilters) -> Dict[str, Any]:
        """Query items with filters.

        Args:
            store: UnifiedStore instance
            session_id: Current session ID
            filters: ItemsQueryFilters with all query parameters

        Returns:
            Dict with items, count, total, and has_more
        """
        # Build query parameters from filters
        query_params = {}
        if filters.item_type:
            query_params["item_type"] = filters.item_type
        if filters.status:
            query_params["status"] = filters.status
        if filters.priority:
            query_params["priority"] = filters.priority
        if filters.phase:
            query_params["phase"] = filters.phase
        if filters.category:
            query_params["category"] = filters.category
        if filters.source_tool:
            query_params["source_tool"] = filters.source_tool

        items_list = store.get_items(session_id, **query_params)

        # Apply date filters
        if filters.start_after:
            items_list = [i for i in items_list if i.get("start_date") and i.get("start_date") >= filters.start_after]
        if filters.end_before:
            items_list = [i for i in items_list if i.get("end_date") and i.get("end_date") <= filters.end_before]

        # Apply traceability filters
        if filters.traces_to:
            traced_items = store.get_items_by_relation(filters.traces_to, "traces")
            traced_ids = {i["item_id"] for i in traced_items}
            items_list = [i for i in items_list if i["item_id"] in traced_ids]

        # Apply pagination
        total = len(items_list)
        items_list = items_list[filters.offset:filters.offset + filters.limit]

        return {
            "success": True,
            "items": items_list,
            "count": len(items_list),
            "total": total,
            "has_more": total > filters.offset + filters.limit
        }

    def register(self, app: FastMCP) -> None:
        """Register all unified tools."""
        self._register_thinking_tool(app)
        self._register_session_tool(app)
        self._register_items_tool(app)
        self._register_diagram_tool(app)

    def _register_thinking_tool(self, app: FastMCP) -> None:
        """Register the main thinking tool."""

        @app.tool()
        def thinking(
            thought: str,
            thought_number: int,
            total_thoughts: int,
            instance_name: str = "default",
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
                instance_name: Session instance name (default: "default")
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
            # Thinking only requires SQLite to be ready (RETER operations are optional)
            try:
                require_sql()
            except ComponentNotReadyError as e:
                return e.to_response()

            session = self._get_session(instance_name)
            return session.think(
                instance_name=instance_name,
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

    def _register_session_tool(self, app: FastMCP) -> None:
        """Register session lifecycle tool."""

        @app.tool()
        @truncate_mcp_response
        def session(
            action: str,
            instance_name: str = "default",
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
                instance_name: Session instance name (default: "default")
                goal: Session goal (for start action)
                project_start: Project start date ISO format (for start action)
                project_end: Project end date ISO format (for start action)

            Returns:
                For start: {session_id, goal, status, created_at}
                For context: {session, design_doc, tasks, project_health, milestones, suggestions, mcp_guide}
                For end: {session_id, status, summary}
                For clear: {success, items_deleted}
            """
            # Session only requires SQLite to be ready
            try:
                require_sql()
            except ComponentNotReadyError as e:
                return e.to_response()

            sess = self._get_session(instance_name)

            if action == "start":
                return sess.start_session(
                    instance_name=instance_name,
                    goal=goal,
                    project_start=project_start,
                    project_end=project_end
                )
            elif action == "context":
                return sess.get_context(instance_name)
            elif action == "end":
                return sess.end_session(instance_name)
            elif action == "clear":
                return sess.clear_session(instance_name)
            else:
                return {"success": False, "error": f"Unknown action: {action}"}

    def _register_items_tool(self, app: FastMCP) -> None:
        """Register items query and management tool."""

        @app.tool()
        @truncate_mcp_response
        def items(
            instance_name: str = "default",
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
            offset: int = 0
        ) -> Dict[str, Any]:
            """
            Query and manage items (thoughts, tasks, milestones).

            Actions:
            - list: Query items with filters
            - get: Get single item by ID
            - delete: Delete item and relations
            - update: Update item fields
            - clear: Delete multiple items matching filters (requires at least one filter)

            Args:
                instance_name: Session instance name
                action: list, get, delete, update, clear
                item_id: Item ID (required for get/delete/update)
                updates: Fields to update (for update action)
                item_type: Filter by type (thought, requirement, task, etc.)
                status: Filter by status (pending, in_progress, completed, etc.)
                priority: Filter by priority (critical, high, medium, low)
                phase: Filter by project phase
                category: Filter by category
                source_tool: Filter by source tool (e.g., 'refactoring_improving:find_large_classes')
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

            Returns:
                For list: {items: [...], count, has_more}
                For get: {item: {...}, relations: {...}}
                For delete: {success, deleted_relations}
                For update: {item: {...}}
                For clear: {success, items_deleted, relations_deleted}
            """
            # Items only requires SQLite to be ready
            try:
                require_sql()
            except ComponentNotReadyError as e:
                return e.to_response()

            sess = self._get_session(instance_name)
            store = sess.store

            # Start session if needed
            session_id = store.get_or_create_session(instance_name)

            if action == "list":
                # Create filters object to reduce parameter passing
                filters = ItemsQueryFilters(
                    item_type=item_type,
                    status=status,
                    priority=priority,
                    phase=phase,
                    category=category,
                    source_tool=source_tool,
                    traces_to=traces_to,
                    traced_by=traced_by,
                    depends_on=depends_on,
                    blocks=blocks,
                    affects=affects,
                    start_after=start_after,
                    end_before=end_before,
                    limit=limit,
                    offset=offset,
                    include_relations=include_relations
                )
                return self._query_items(store, session_id, filters)

            elif action == "get":
                if not item_id:
                    return {"success": False, "error": "item_id required for get action"}

                item = store.get_item(item_id)
                if not item:
                    return {"success": False, "error": f"Item {item_id} not found"}

                result = {"success": True, "item": item}

                if include_relations:
                    result["relations"] = {
                        "outgoing": store.get_relations(item_id, direction="outgoing"),
                        "incoming": store.get_relations(item_id, direction="incoming")
                    }

                return result

            elif action == "delete":
                if not item_id:
                    return {"success": False, "error": "item_id required for delete action"}

                store.delete_item(item_id)
                return {"success": True, "deleted": item_id}

            elif action == "update":
                if not item_id:
                    return {"success": False, "error": "item_id required for update action"}
                if not updates:
                    return {"success": False, "error": "updates required for update action"}

                store.update_item(item_id, **updates)
                item = store.get_item(item_id)
                return {"success": True, "item": item}

            elif action == "clear":
                # Require at least one filter to prevent accidental deletion of everything
                has_filter = any([item_type, status, priority, category, source_tool])
                if not has_filter:
                    return {
                        "success": False,
                        "error": "At least one filter required for clear action (item_type, status, priority, category, or source_tool)"
                    }

                result = store.delete_items_by_filter(
                    session_id=session_id,
                    item_type=item_type,
                    status=status,
                    priority=priority,
                    category=category,
                    source_tool=source_tool
                )
                return {
                    "success": True,
                    "items_deleted": result["items_deleted"],
                    "relations_deleted": result["relations_deleted"],
                    "filters_used": {
                        k: v for k, v in {
                            "item_type": item_type,
                            "status": status,
                            "priority": priority,
                            "category": category,
                            "source_tool": source_tool
                        }.items() if v is not None
                    }
                }

            else:
                return {"success": False, "error": f"Unknown action: {action}"}

    def _register_diagram_tool(self, app: FastMCP) -> None:
        """Register diagram generation tool."""
        instance_manager = self.instance_manager

        @app.tool()
        def diagram(
            diagram_type: str,
            instance_name: str = "default",
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
            - design_doc: Design doc structure with sections (context, goals, design, etc.)
            - traceability: Task traceability matrix

            **UML/Code Diagrams:**
            - class_hierarchy: Class inheritance hierarchy (target=root_class, optional)
            - class_diagram: Class diagram with methods/attributes (classes=[...])
            - sequence: Sequence diagram of method calls (classes=[...], target=entry_point)
            - dependencies: Module dependency graph (target=module_filter, optional)
            - call_graph: Call graph from entry point (target=focus_function)
            - coupling: Coupling/cohesion matrix (classes=[...], optional)

            Args:
                diagram_type: One of the diagram types above
                instance_name: Session/RETER instance name
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
            # Session diagrams only need SQLite, UML diagrams need RETER
            if diagram_type in ("gantt", "thought_chain", "traceability", "design_doc"):
                try:
                    require_sql()
                except ComponentNotReadyError as e:
                    return e.to_response()
            elif diagram_type in ("class_hierarchy", "class_diagram", "sequence", "dependencies", "call_graph", "coupling"):
                try:
                    require_default_instance()
                except ComponentNotReadyError as e:
                    return e.to_response()

            # Session/Project diagrams
            if diagram_type in ("gantt", "thought_chain", "traceability", "design_doc"):
                sess = self._get_session(instance_name)
                store = sess.store
                session_id = store.get_or_create_session(instance_name)

                if diagram_type == "gantt":
                    return self._generate_gantt_diagram(store, session_id, start_date, end_date, format)
                elif diagram_type == "thought_chain":
                    return self._generate_thought_chain_diagram(store, session_id, format)
                elif diagram_type == "traceability":
                    return self._generate_traceability_diagram(store, session_id, format)
                elif diagram_type == "design_doc":
                    return self._generate_design_doc_diagram(store, session_id, format)

            # UML/Code diagrams - delegate to UMLTool
            if diagram_type in ("class_hierarchy", "class_diagram", "sequence", "dependencies", "call_graph", "coupling"):
                try:
                    from ...tools.uml.tool import UMLTool
                    uml_tool = UMLTool(instance_manager)
                    extra = params or {}

                    if diagram_type == "class_hierarchy":
                        return uml_tool.get_class_hierarchy(instance_name, target, format)

                    elif diagram_type == "class_diagram":
                        if not classes:
                            return {"success": False, "error": "classes list is required for class_diagram"}
                        return uml_tool.get_class_diagram(
                            instance_name, classes, include_methods, include_attributes, format
                        )

                    elif diagram_type == "sequence":
                        if not classes:
                            return {"success": False, "error": "classes list is required for sequence diagram"}
                        return uml_tool.get_sequence_diagram(
                            instance_name=instance_name,
                            classes=classes,
                            entry_point=target,
                            max_depth=max_depth,
                            exclude_patterns=extra.get("exclude_patterns"),
                            include_only_classes=extra.get("include_only_classes"),
                            format=format
                        )

                    elif diagram_type == "dependencies":
                        return uml_tool.get_dependency_graph(
                            instance_name=instance_name,
                            format=format,
                            show_external=show_external,
                            group_by_package=extra.get("group_by_package", True),
                            highlight_circular=extra.get("highlight_circular", True),
                            module_filter=target,
                            summary_only=extra.get("summary_only", False),
                            limit=extra.get("limit", 100),
                            offset=extra.get("offset", 0),
                            circular_deps_limit=extra.get("circular_deps_limit", 10)
                        )

                    elif diagram_type == "call_graph":
                        if not target:
                            return {"success": False, "error": "target (focus_function) is required for call_graph"}
                        return uml_tool.get_call_graph(
                            focus_function=target,
                            instance_name=instance_name,
                            direction=extra.get("direction", "both"),
                            max_depth=max_depth,
                            format=format,
                            exclude_patterns=extra.get("exclude_patterns")
                        )

                    elif diagram_type == "coupling":
                        return uml_tool.get_coupling_matrix(
                            instance_name=instance_name,
                            classes=classes,
                            max_classes=extra.get("max_classes", 20),
                            threshold=extra.get("threshold", 0),
                            include_inheritance=extra.get("include_inheritance", True),
                            format=format
                        )

                except Exception as e:
                    import traceback
                    return {"success": False, "error": str(e), "traceback": traceback.format_exc()}

            return {"success": False, "error": f"Unknown diagram type: {diagram_type}"}

    def _generate_gantt_diagram(self, store, session_id, start_date, end_date, format):
        """Generate Gantt chart diagram."""
        tasks = store.get_items(session_id, item_type="task")
        milestones = store.get_items(session_id, item_type="milestone")

        if format == "mermaid":
            lines = ["gantt", "    title Project Schedule", "    dateFormat YYYY-MM-DD"]

            # Group tasks by phase
            phases = {}
            for task in tasks:
                phase = task.get("phase") or "Tasks"
                if phase not in phases:
                    phases[phase] = []
                phases[phase].append(task)

            for phase, phase_tasks in phases.items():
                lines.append(f"    section {phase}")
                for task in phase_tasks:
                    task_id = task["item_id"]
                    name = task["content"][:30]
                    status = task.get("status", "pending")

                    # Determine status for mermaid
                    mermaid_status = ""
                    if status == "completed":
                        mermaid_status = "done, "
                    elif status == "in_progress":
                        mermaid_status = "active, "

                    start = task.get("start_date", "2024-01-01")
                    duration = task.get("duration_days", 1)

                    lines.append(f"    {name} :{mermaid_status}{task_id}, {start}, {duration}d")

            # Add milestones
            if milestones:
                lines.append("    section Milestones")
                for ms in milestones:
                    name = ms["content"][:30]
                    target = ms.get("end_date", "2024-01-01")
                    ms_id = ms["item_id"]
                    lines.append(f"    {name} :milestone, {ms_id}, {target}, 0d")

            return {
                "success": True,
                "diagram": "\n".join(lines),
                "format": "mermaid",
                "tasks_count": len(tasks),
                "milestones_count": len(milestones)
            }

        return {"success": False, "error": f"Format {format} not supported for gantt"}

    def _generate_thought_chain_diagram(self, store, session_id, format):
        """Generate thought chain diagram."""
        chain = store.get_thought_chain(session_id)

        if format == "mermaid":
            lines = ["graph TD"]

            for thought in chain:
                num = thought.get("thought_number", 0)
                ttype = thought.get("thought_type", "reasoning")
                content = thought.get("content", "")[:30].replace('"', "'")

                node_id = f"T{num}"
                lines.append(f'    {node_id}["{num}. {content}..."]')

                # Connect to previous thought
                if num > 1:
                    prev_id = f"T{num-1}"
                    lines.append(f"    {prev_id} --> {node_id}")

                # Handle branches
                branch_from = thought.get("branch_from_thought")
                if branch_from:
                    branch_id = f"T{branch_from}"
                    lines.append(f"    {branch_id} -.-> {node_id}")

            return {
                "success": True,
                "diagram": "\n".join(lines),
                "format": "mermaid",
                "thoughts_count": len(chain)
            }

        return {"success": False, "error": f"Format {format} not supported for thought_chain"}

    def _generate_traceability_diagram(self, store, session_id, format):
        """Generate traceability matrix.

        Shows tasks grouped by category (Design Docs approach).
        """
        tasks = store.get_items(session_id, item_type="task")
        thoughts = store.get_items(session_id, item_type="thought")

        # Group tasks by category
        tasks_by_category = {}
        for task in tasks:
            cat = task.get("category") or "uncategorized"
            if cat not in tasks_by_category:
                tasks_by_category[cat] = []
            tasks_by_category[cat].append(task)

        if format == "mermaid":
            lines = ["flowchart LR"]

            # Define subgraphs for each category
            for cat, cat_tasks in tasks_by_category.items():
                safe_cat = cat.replace("-", "_").replace(" ", "_")
                lines.append(f"    subgraph {safe_cat}[{cat.title()}]")
                for task in cat_tasks:
                    task_id = task["item_id"]
                    safe_id = task_id.replace("-", "_")
                    lines.append(f"        {safe_id}[{task_id}]")
                lines.append("    end")

            # Add relations between tasks
            lines.append("")
            lines.append("    %% Traceability links")

            for task in tasks:
                task_id = task["item_id"]
                safe_task = task_id.replace("-", "_")

                # Task -> Task (depends_on)
                deps = store.get_items_by_relation(task_id, "depends_on")
                for dep in deps:
                    if dep.get("item_type") == "task":
                        safe_dep = dep["item_id"].replace("-", "_")
                        lines.append(f"    {safe_dep} -->|blocks| {safe_task}")

                # Task -> Thought (traces)
                traced = store.get_items_by_relation(task_id, "traces")
                for t in traced:
                    if t.get("item_type") == "thought":
                        safe_t = t["item_id"].replace("-", "_")
                        lines.append(f"    {safe_t} -.->|traces| {safe_task}")

            return {
                "success": True,
                "diagram": "\n".join(lines),
                "format": "mermaid",
                "tasks_count": len(tasks),
                "categories": list(tasks_by_category.keys())
            }

        elif format == "markdown":
            lines = ["# Traceability Matrix", ""]

            # Tasks by category
            lines.append("## Tasks by Category")
            lines.append("| Category | Task | Status | Dependencies |")
            lines.append("|----------|------|--------|--------------|")

            for cat, cat_tasks in tasks_by_category.items():
                for task in cat_tasks:
                    task_id = task["item_id"]
                    status = task.get("status", "pending")
                    deps = store.get_items_by_relation(task_id, "depends_on")
                    dep_ids = [d["item_id"] for d in deps if d.get("item_type") == "task"]
                    lines.append(f"| {cat} | {task_id} | {status} | {', '.join(dep_ids) or '-'} |")

            # Thought to Task traceability
            lines.append("")
            lines.append("## Thoughts → Tasks")
            lines.append("| Thought | Related Tasks |")
            lines.append("|---------|---------------|")

            for thought in thoughts[:20]:  # Limit to recent thoughts
                thought_id = thought["item_id"]
                traced = store.get_items_by_relation(thought_id, "traces")
                task_ids = [t["item_id"] for t in traced if t.get("item_type") == "task"]
                if task_ids:
                    lines.append(f"| {thought_id} | {', '.join(task_ids)} |")

            return {
                "success": True,
                "diagram": "\n".join(lines),
                "format": "markdown",
                "tasks_count": len(tasks),
                "categories": list(tasks_by_category.keys())
            }

        return {"success": False, "error": f"Format {format} not supported for traceability"}

    def _generate_requirements_diagram(self, store, session_id, root_id, format):
        """Generate requirements hierarchy diagram."""
        requirements = store.get_items(session_id, item_type="requirement")
        constraints = store.get_items(session_id, item_type="constraint")
        assumptions = store.get_items(session_id, item_type="assumption")

        if format == "mermaid":
            lines = ["graph TD"]

            for req in requirements:
                req_id = req["item_id"]
                content = req.get("content", "")[:25].replace('"', "'")
                status = req.get("status", "pending")

                # Style based on status
                style = ""
                if status == "verified":
                    style = ":::done"
                elif status == "rejected":
                    style = ":::rejected"

                lines.append(f'    {req_id}["{req_id}: {content}..."]{style}')

                # Add derives relations
                relations = store.get_relations(req_id, direction="outgoing")
                for rel in relations:
                    if rel["relation_type"] == "derives" and rel["target_type"] == "item":
                        lines.append(f"    {rel['target_id']} --> {req_id}")

            # Add constraints
            for con in constraints:
                con_id = con["item_id"]
                content = con.get("content", "")[:25].replace('"', "'")
                lines.append(f'    {con_id}("{con_id}: {content}..."):::constraint')

            # Styles
            lines.append("")
            lines.append("    classDef done fill:#9f6,stroke:#333")
            lines.append("    classDef rejected fill:#f66,stroke:#333")
            lines.append("    classDef constraint fill:#69f,stroke:#333")

            return {
                "success": True,
                "diagram": "\n".join(lines),
                "format": "mermaid",
                "requirements_count": len(requirements),
                "constraints_count": len(constraints)
            }

        return {"success": False, "error": f"Format {format} not supported for requirements"}

    def _generate_design_doc_diagram(self, store, session_id, format):
        """Generate design doc diagram showing sections and thoughts."""
        thoughts = store.get_items(session_id, item_type="thought")
        tasks = store.get_items(session_id, item_type="task")

        # Define section order and colors
        section_order = ["context", "goals", "non_goals", "design", "alternatives", "risks", "implementation", "tasks"]
        section_colors = {
            "context": "#e1f5fe",
            "goals": "#c8e6c9",
            "non_goals": "#ffecb3",
            "design": "#e1bee7",
            "alternatives": "#ffe0b2",
            "risks": "#ffcdd2",
            "implementation": "#b2dfdb",
            "tasks": "#d1c4e9"
        }

        # Organize thoughts by section
        by_section = {s: [] for s in section_order}
        by_section["other"] = []

        for t in thoughts:
            section = t.get("section") or "other"
            if section not in by_section:
                section = "other"
            by_section[section].append(t)

        if format == "mermaid":
            lines = ["flowchart TB"]

            # Add section subgraphs
            for section in section_order:
                section_thoughts = by_section.get(section, [])
                if section_thoughts:
                    section_title = section.replace("_", " ").title()
                    lines.append(f"    subgraph {section}[{section_title}]")
                    for t in section_thoughts:
                        t_id = t["item_id"].replace("-", "_")
                        num = t.get("thought_number", 0)
                        content = t.get("content", "")[:30].replace('"', "'").replace("\n", " ")
                        lines.append(f'        {t_id}["{num}. {content}..."]')
                    lines.append("    end")
                    lines.append("")

            # Add other/unstructured thoughts if any
            other = by_section.get("other", [])
            if other:
                lines.append("    subgraph other[Unstructured]")
                for t in other:
                    t_id = t["item_id"].replace("-", "_")
                    num = t.get("thought_number", 0)
                    content = t.get("content", "")[:30].replace('"', "'").replace("\n", " ")
                    lines.append(f'        {t_id}["{num}. {content}..."]')
                lines.append("    end")
                lines.append("")

            # Add tasks summary
            if tasks:
                # Group by category
                by_category = {}
                for task in tasks:
                    cat = task.get("category") or "uncategorized"
                    if cat not in by_category:
                        by_category[cat] = []
                    by_category[cat].append(task)

                lines.append("    subgraph work[Work Items]")
                for cat, cat_tasks in by_category.items():
                    completed = len([t for t in cat_tasks if t.get("status") == "completed"])
                    total = len(cat_tasks)
                    cat_id = cat.replace("-", "_")
                    lines.append(f'        {cat_id}["{cat}: {completed}/{total}"]')
                lines.append("    end")
                lines.append("")

            # Add flow between sections (design doc flow)
            prev_section = None
            for section in section_order:
                if by_section.get(section):
                    if prev_section:
                        lines.append(f"    {prev_section} --> {section}")
                    prev_section = section

            # Link tasks section to work items
            if by_section.get("tasks") and tasks:
                lines.append("    tasks --> work")

            # Add styles
            lines.append("")
            for section, color in section_colors.items():
                lines.append(f"    style {section} fill:{color}")

            return {
                "success": True,
                "diagram": "\n".join(lines),
                "format": "mermaid",
                "thoughts_count": len(thoughts),
                "sections_used": [s for s in section_order if by_section.get(s)],
                "tasks_count": len(tasks)
            }

        elif format == "markdown":
            lines = ["# Design Doc Structure", ""]

            for section in section_order:
                section_thoughts = by_section.get(section, [])
                if section_thoughts:
                    section_title = section.replace("_", " ").title()
                    lines.append(f"## {section_title}")
                    lines.append("")
                    for t in section_thoughts:
                        num = t.get("thought_number", 0)
                        content = t.get("content", "")[:100]
                        lines.append(f"- **#{num}**: {content}...")
                    lines.append("")

            # Other thoughts
            other = by_section.get("other", [])
            if other:
                lines.append("## Unstructured Thoughts")
                lines.append("")
                for t in other:
                    num = t.get("thought_number", 0)
                    content = t.get("content", "")[:100]
                    lines.append(f"- **#{num}**: {content}...")
                lines.append("")

            # Tasks
            if tasks:
                lines.append("## Work Items")
                lines.append("")
                lines.append("| Category | Total | Completed | Status |")
                lines.append("|----------|-------|-----------|--------|")

                by_category = {}
                for task in tasks:
                    cat = task.get("category") or "uncategorized"
                    if cat not in by_category:
                        by_category[cat] = {"total": 0, "completed": 0}
                    by_category[cat]["total"] += 1
                    if task.get("status") == "completed":
                        by_category[cat]["completed"] += 1

                for cat, stats in by_category.items():
                    pct = (stats["completed"] / stats["total"] * 100) if stats["total"] > 0 else 0
                    lines.append(f"| {cat} | {stats['total']} | {stats['completed']} | {pct:.0f}% |")

            return {
                "success": True,
                "diagram": "\n".join(lines),
                "format": "markdown",
                "thoughts_count": len(thoughts),
                "sections_used": [s for s in section_order if by_section.get(s)],
                "tasks_count": len(tasks)
            }

        return {"success": False, "error": f"Format {format} not supported for design_doc"}
