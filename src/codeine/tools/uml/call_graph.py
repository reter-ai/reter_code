"""Call graph generator.

Generates call graphs showing function/method call relationships.
"""

import re
from typing import Dict, Any, List, Optional
from codeine.tools.dataclasses import CallGraphOptions
from .base import UMLGeneratorBase


class CallGraphGenerator(UMLGeneratorBase):
    """Generates call graphs centered on a focus function."""

    def generate(
        self,
        focus_function: str,
        instance_name: str = "default",
        direction: str = "both",
        max_depth: int = 3,
        format: str = "markdown",
        exclude_patterns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Generate call graph showing function/method call relationships.

        Args:
            focus_function: Function/method name to focus on
            instance_name: RETER instance name
            direction: Traversal direction ('upstream', 'downstream', 'both')
            max_depth: Maximum depth to traverse
            format: Output format ('json', 'markdown', 'graphviz')
            exclude_patterns: List of regex patterns to exclude

        Returns:
            Dictionary with call graph data
        """
        options = CallGraphOptions(
            direction=direction,
            max_depth=max_depth,
            exclude_patterns=exclude_patterns
        )
        return self._generate_impl(focus_function, instance_name, format, options)

    def _generate_impl(
        self,
        focus_function: str,
        instance_name: str,
        format: str,
        options: CallGraphOptions
    ) -> Dict[str, Any]:
        """Implementation of call graph generation."""
        reter = self.instance_manager.get_or_create_instance(instance_name)

        # Compile exclude patterns
        exclude_regexes = []
        if options.exclude_patterns:
            exclude_regexes = [re.compile(pattern) for pattern in options.exclude_patterns]

        # Query for method/function calls
        query = """
        SELECT ?callerName ?calleeName WHERE {
            ?caller calls ?callee .
            ?caller name ?callerName .
            ?callee name ?calleeName
        }
        """

        result = reter.reql(query)

        if not result or (hasattr(result, 'num_rows') and result.num_rows == 0):
            return {
                "nodes": [],
                "edges": [],
                "diagram": "No function calls found",
                "format": format
            }

        rows = self._result_to_pylist(result)
        if not rows:
            return {
                "nodes": [],
                "edges": [],
                "diagram": "No call data available",
                "format": format
            }

        # Build call graphs
        call_graph = {}  # function -> [callees]
        reverse_graph = {}  # function -> [callers]

        for row in rows:
            caller = str(row.get('?callerName', ''))
            callee = str(row.get('?calleeName', ''))

            # Apply exclude patterns
            if exclude_regexes:
                if any(regex.search(caller) for regex in exclude_regexes):
                    continue
                if any(regex.search(callee) for regex in exclude_regexes):
                    continue

            if caller not in call_graph:
                call_graph[caller] = []
            call_graph[caller].append(callee)

            if callee not in reverse_graph:
                reverse_graph[callee] = []
            reverse_graph[callee].append(caller)

        # Traverse graph from focus_function
        nodes = set()
        edges = []
        max_depth = options.max_depth

        def traverse_downstream(func: str, depth: int):
            if depth > max_depth or func in nodes:
                return
            nodes.add(func)
            for callee in call_graph.get(func, []):
                edges.append({"from": func, "to": callee, "type": "calls"})
                traverse_downstream(callee, depth + 1)

        def traverse_upstream(func: str, depth: int):
            if depth > max_depth or func in nodes:
                return
            nodes.add(func)
            for caller in reverse_graph.get(func, []):
                edges.append({"from": caller, "to": func, "type": "calls"})
                traverse_upstream(caller, depth + 1)

        # Start traversal
        nodes.add(focus_function)

        if options.direction in ["downstream", "both"]:
            traverse_downstream(focus_function, 0)

        if options.direction in ["upstream", "both"]:
            traverse_upstream(focus_function, 0)

        # Format output
        if format == "json":
            diagram = None
        elif format == "graphviz":
            diagram = self._render_graphviz(list(nodes), edges, focus_function)
        else:  # markdown
            diagram = self._render_markdown(list(nodes), edges, focus_function, options.direction)

        return {
            "focus_function": focus_function,
            "nodes": list(nodes),
            "edges": edges,
            "depth_analyzed": options.max_depth,
            "direction": options.direction,
            "diagram": diagram,
            "format": format
        }

    def _render_markdown(
        self,
        nodes: List[str],
        edges: List[Dict[str, str]],
        focus_function: str,
        direction: str
    ) -> str:
        """Render call graph as markdown."""
        lines = [f"# Call Graph: `{focus_function}`", ""]
        lines.append(f"**Direction**: {direction}")
        lines.append(f"**Total Functions**: {len(nodes)}")
        lines.append("")

        # Group edges by caller
        callers = {}
        for edge in edges:
            caller = edge['from']
            callee = edge['to']
            if caller not in callers:
                callers[caller] = []
            callers[caller].append(callee)

        # Render as tree
        lines.append("## Call Relationships")
        lines.append("")

        for caller in sorted(callers.keys()):
            marker = "📍 " if caller == focus_function else ""
            lines.append(f"- {marker}`{caller}`")
            for callee in sorted(callers[caller]):
                lines.append(f"  - → `{callee}`")

        return "\n".join(lines)

    def _render_graphviz(
        self,
        nodes: List[str],
        edges: List[Dict[str, str]],
        focus_function: str
    ) -> str:
        """Render call graph as Graphviz DOT format."""
        lines = ["```dot", "digraph CallGraph {", "    rankdir=TB;", "    node [shape=box];", ""]

        # Highlight focus function
        focus_node = focus_function.replace('.', '_')
        lines.append(f'    {focus_node} [style=filled, fillcolor=yellow];')
        lines.append("")

        # Render edges
        for edge in edges:
            from_node = edge['from'].replace('.', '_')
            to_node = edge['to'].replace('.', '_')
            lines.append(f'    {from_node} -> {to_node};')

        lines.extend(["", "}", "```"])

        return "\n".join(lines)
