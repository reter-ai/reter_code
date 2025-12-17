"""Dependency graph generator.

Generates module dependency graphs showing import relationships.
"""

from typing import Dict, Any, List, Optional
from codeine.tools.dataclasses import DependencyGraphOptions
from .base import UMLGeneratorBase


class DependencyGraphGenerator(UMLGeneratorBase):
    """Generates module dependency graphs."""

    def generate(
        self,
        instance_name: str = "default",
        format: str = "markdown",
        show_external: bool = False,
        group_by_package: bool = True,
        highlight_circular: bool = True,
        module_filter: Optional[str] = None,
        summary_only: bool = False,
        limit: int = 100,
        offset: int = 0,
        circular_deps_limit: int = 10
    ) -> Dict[str, Any]:
        """Generate module dependency graph showing import relationships.

        Args:
            instance_name: RETER instance name
            format: Output format ('json', 'markdown', 'graphviz')
            show_external: Include external/stdlib imports
            group_by_package: Group modules by package
            highlight_circular: Highlight circular dependencies
            module_filter: Filter modules by prefix
            summary_only: Return summary only (10x smaller response)
            limit: Max dependencies to return
            offset: Pagination offset
            circular_deps_limit: Limit circular deps in response

        Returns:
            Dictionary with dependency graph data
        """
        options = DependencyGraphOptions(
            show_external=show_external,
            group_by_package=group_by_package,
            highlight_circular=highlight_circular,
            module_filter=module_filter,
            summary_only=summary_only,
            limit=limit,
            offset=offset,
            circular_deps_limit=circular_deps_limit
        )
        return self._generate_impl(instance_name, format, options)

    def _generate_impl(
        self,
        instance_name: str,
        format: str,
        options: DependencyGraphOptions
    ) -> Dict[str, Any]:
        """Implementation of dependency graph generation."""
        reter = self.instance_manager.get_or_create_instance(instance_name)

        module_concept = self._concept('Module')
        import_concept = self._concept('Import')
        # Query for module imports
        query = f"""
        SELECT ?moduleName ?imported WHERE {{
            ?module type {module_concept} .
            ?module name ?moduleName .
            ?import type {import_concept} .
            ?import inModule ?module .
            ?import imports ?imported
        }}
        """

        result = reter.reql(query)

        if not result or (hasattr(result, 'num_rows') and result.num_rows == 0):
            return {
                "dependencies": [],
                "circular_dependencies": [],
                "diagram": "No module imports found",
                "format": format
            }

        rows = self._result_to_pylist(result)
        if not rows:
            return {
                "dependencies": [],
                "circular_dependencies": [],
                "diagram": "No import data available",
                "format": format
            }

        # Build dependency graph
        dependencies = []
        graph = {}  # module -> [imported_modules]

        for row in rows:
            importer = str(row.get('?moduleName', ''))
            imported = str(row.get('?imported', ''))

            # Filter external imports if requested
            if not options.show_external and (imported.startswith('__builtin__') or '.' not in imported):
                continue

            # Apply module filter if specified
            if options.module_filter:
                if not (importer.startswith(options.module_filter) or imported.startswith(options.module_filter)):
                    continue

            dependencies.append({
                "from": importer,
                "to": imported
            })

            if importer not in graph:
                graph[importer] = []
            graph[importer].append(imported)

        # Calculate totals before pagination
        total_dependencies = len(dependencies)
        external_count = len([d for d in dependencies if '.' not in d['to']])
        internal_count = len([d for d in dependencies if '.' in d['to']])

        # Detect circular dependencies
        circular_deps = []
        total_circular_deps = 0
        if options.highlight_circular:
            circular_deps = self._detect_circular_dependencies(graph)
            total_circular_deps = len(circular_deps)
            if options.circular_deps_limit > 0:
                circular_deps = circular_deps[:options.circular_deps_limit]

        # Apply pagination
        paginated_dependencies = dependencies[options.offset:options.offset + options.limit]

        # Summary-only mode
        if options.summary_only:
            return {
                "summary": {
                    "total_dependencies": total_dependencies,
                    "external_count": external_count,
                    "internal_count": internal_count,
                    "circular_dependencies_count": total_circular_deps,
                    "circular_dependencies_returned": len(circular_deps),
                    "unique_modules": len(graph)
                },
                "circular_dependencies": circular_deps,
                "total_count": total_dependencies,
                "count_returned": len(paginated_dependencies),
                "limit": options.limit,
                "offset": options.offset,
                "has_more": (options.offset + options.limit) < total_dependencies,
                "next_offset": options.offset + options.limit if (options.offset + options.limit) < total_dependencies else None,
                "format": format,
                "summary_only": True
            }

        # Format output
        if format == "json":
            diagram = None
        elif format == "graphviz":
            diagram = self._render_graphviz(paginated_dependencies, circular_deps)
        else:  # markdown
            diagram = self._render_markdown(paginated_dependencies, circular_deps, options.group_by_package)

        return {
            "dependencies": paginated_dependencies,
            "circular_dependencies": circular_deps,
            "summary": {
                "total_dependencies": total_dependencies,
                "external_count": external_count,
                "internal_count": internal_count,
                "circular_dependencies_count": total_circular_deps,
                "circular_dependencies_returned": len(circular_deps)
            },
            "total_count": total_dependencies,
            "count_returned": len(paginated_dependencies),
            "limit": options.limit,
            "offset": options.offset,
            "has_more": (options.offset + options.limit) < total_dependencies,
            "next_offset": options.offset + options.limit if (options.offset + options.limit) < total_dependencies else None,
            "diagram": diagram,
            "format": format
        }

    def _detect_circular_dependencies(self, graph: Dict[str, List[str]]) -> List[List[str]]:
        """Detect circular dependencies using DFS cycle detection."""
        cycles = []
        visited = set()
        rec_stack = set()

        def dfs(node: str, path: List[str]):
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    dfs(neighbor, path.copy())
                elif neighbor in rec_stack:
                    # Found a cycle
                    cycle_start = path.index(neighbor)
                    cycle = path[cycle_start:] + [neighbor]
                    if cycle not in cycles:
                        cycles.append(cycle)

            rec_stack.remove(node)

        for node in graph:
            if node not in visited:
                dfs(node, [])

        return cycles

    def _render_markdown(
        self,
        dependencies: List[Dict[str, str]],
        circular_deps: List[List[str]],
        group_by_package: bool
    ) -> str:
        """Render dependency graph as markdown."""
        lines = ["# Module Dependency Graph", ""]

        # Show circular dependencies first
        if circular_deps:
            lines.append("## ⚠️ Circular Dependencies Detected")
            lines.append("")
            for i, cycle in enumerate(circular_deps, 1):
                cycle_str = " → ".join(cycle)
                lines.append(f"{i}. {cycle_str}")
            lines.append("")

        # Group dependencies
        if group_by_package:
            packages = {}
            for dep in dependencies:
                from_pkg = dep['from'].split('.')[0] if '.' in dep['from'] else dep['from']
                if from_pkg not in packages:
                    packages[from_pkg] = []
                packages[from_pkg].append(dep)

            lines.append("## Dependencies by Package")
            lines.append("")
            for pkg in sorted(packages.keys()):
                lines.append(f"### {pkg}")
                lines.append("")
                for dep in sorted(packages[pkg], key=lambda x: x['from']):
                    is_circular = any(dep['from'] in cycle and dep['to'] in cycle for cycle in circular_deps)
                    marker = "⚠️ " if is_circular else ""
                    lines.append(f"- {marker}`{dep['from']}` → `{dep['to']}`")
                lines.append("")
        else:
            lines.append("## All Dependencies")
            lines.append("")
            for dep in sorted(dependencies, key=lambda x: (x['from'], x['to'])):
                is_circular = any(dep['from'] in cycle and dep['to'] in cycle for cycle in circular_deps)
                marker = "⚠️ " if is_circular else ""
                lines.append(f"- {marker}`{dep['from']}` → `{dep['to']}`")

        return "\n".join(lines)

    def _render_graphviz(
        self,
        dependencies: List[Dict[str, str]],
        circular_deps: List[List[str]]
    ) -> str:
        """Render dependency graph as Graphviz DOT format."""
        lines = ["```dot", "digraph Dependencies {", "    rankdir=LR;", "    node [shape=box];", ""]

        # Collect circular edges
        circular_edges = set()
        for cycle in circular_deps:
            for i in range(len(cycle) - 1):
                circular_edges.add((cycle[i], cycle[i + 1]))

        # Render edges
        for dep in dependencies:
            from_node = dep['from'].replace('.', '_')
            to_node = dep['to'].replace('.', '_')

            if (dep['from'], dep['to']) in circular_edges:
                lines.append(f'    {from_node} -> {to_node} [color=red, penwidth=2.0];')
            else:
                lines.append(f'    {from_node} -> {to_node};')

        lines.extend(["", "}", "```"])

        return "\n".join(lines)
