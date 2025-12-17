"""
Dependency Analysis Tools

Provides tools for analyzing module dependencies, imports, and external packages.
"""

from typing import Dict, Any, List
import time
from .base import AdvancedToolsBase


class DependencyAnalysisTools(AdvancedToolsBase):
    """Dependency analysis tools."""

    def get_import_graph(self, instance_name: str) -> Dict[str, Any]:
        """
        Get complete module import dependency graph.

        Args:
            instance_name: RETER instance name

        Returns:
            dict with success, nodes, edges, queries
        """
        start_time = time.time()
        queries = []
        try:
            module_concept = self._concept('Module')
            query = f"""
                SELECT ?importer ?imported
                WHERE {{
                    ?importer type {module_concept} .
                    ?importer imports ?imported .
                    ?imported type {module_concept}
                }}
            """
            queries.append(query.strip())

            result = self.reter.reql(query)
            rows = self._query_to_list(result)

            nodes = set()
            edges = []
            for row in rows:
                importer, imported = row[0], row[1]
                nodes.add(importer)
                nodes.add(imported)
                edges.append({"from": importer, "to": imported})

            time_ms = (time.time() - start_time) * 1000
            return {
                "success": True,
                "nodes": list(nodes),
                "edges": edges,
                "node_count": len(nodes),
                "edge_count": len(edges),
                "queries": queries,
                "time_ms": time_ms
            }
        except Exception as e:
            time_ms = (time.time() - start_time) * 1000
            return {
                "success": False,
                "error": str(e),
                "nodes": [],
                "edges": [],
                "queries": queries,
                "time_ms": time_ms
            }

    def find_circular_imports(self, instance_name: str) -> Dict[str, Any]:
        """
        Find circular import dependencies using DFS-based cycle detection.

        Detects:
        - Self-references (A→A)
        - Direct cycles (A↔B)
        - Transitive cycles (A→B→C→A, etc.)

        Args:
            instance_name: RETER instance name

        Returns:
            dict with success, cycles list (categorized by type), queries
        """
        start_time = time.time()
        queries = []
        try:
            # Get all import relationships
            # Uses the same query structure as UML tool - going through Import entities
            module_concept = self._concept('Module')
            import_concept = self._concept('Import')
            query = f"""
                SELECT ?moduleName ?imported
                WHERE {{
                    ?module type {module_concept} .
                    ?module name ?moduleName .
                    ?import type {import_concept} .
                    ?import inModule ?module .
                    ?import imports ?imported
                }}
            """
            queries.append(query.strip())

            result = self.reter.reql(query)
            rows = self._query_to_list(result)

            # Build adjacency list graph
            graph: Dict[str, List[str]] = {}
            for row in rows:
                importer, imported = str(row[0]), str(row[1])
                if importer not in graph:
                    graph[importer] = []
                graph[importer].append(imported)
                # Ensure imported module is in graph even if it has no outgoing edges
                if imported not in graph:
                    graph[imported] = []

            # Detect cycles using DFS (same algorithm as UML tool)
            all_cycles = self._detect_cycles_dfs(graph)

            # Categorize cycles
            self_refs = []
            direct_cycles = []
            transitive_cycles = []

            for cycle in all_cycles:
                # Cycle format is [A, B, C, A] - last element repeats first
                cycle_length = len(cycle) - 1  # Unique nodes in cycle

                if cycle_length == 1:
                    # Self-reference: [A, A]
                    self_refs.append({
                        "modules": [cycle[0]],
                        "type": "self_reference"
                    })
                elif cycle_length == 2:
                    # Direct: [A, B, A]
                    direct_cycles.append({
                        "modules": cycle[:-1],  # [A, B]
                        "type": "direct"
                    })
                else:
                    # Transitive: [A, B, C, A] or longer
                    transitive_cycles.append({
                        "modules": cycle[:-1],  # [A, B, C]
                        "type": "transitive"
                    })

            # Combine all cycles
            all_categorized = self_refs + direct_cycles + transitive_cycles

            time_ms = (time.time() - start_time) * 1000
            return {
                "success": True,
                "cycles": all_categorized,
                "count": len(all_categorized),
                "self_references": len(self_refs),
                "direct_cycles": len(direct_cycles),
                "transitive_cycles": len(transitive_cycles),
                "queries": queries,
                "time_ms": time_ms
            }
        except Exception as e:
            time_ms = (time.time() - start_time) * 1000
            return {
                "success": False,
                "error": str(e),
                "cycles": [],
                "count": 0,
                "queries": queries,
                "time_ms": time_ms
            }

    def _detect_cycles_dfs(self, graph: Dict[str, List[str]]) -> List[List[str]]:
        """
        Detect all cycles in a directed graph using DFS.

        Uses the same optimized algorithm as UML tool's _detect_circular_dependencies.
        Key optimizations:
        - In-place path modification with append/pop (O(1) vs O(n) for copy)
        - Set-based deduplication with normalized tuple keys (O(1) vs O(m) for list search)

        Args:
            graph: Adjacency list {node: [neighbors]}

        Returns:
            List of cycles, each as [A, B, C, A] (last repeats first)
        """
        seen_cycles: set = set()  # Use set with tuple keys for O(1) deduplication
        cycles: List[List[str]] = []
        visited: set = set()
        rec_stack: set = set()

        def dfs(node: str, path: List[str]) -> None:
            visited.add(node)
            rec_stack.add(node)
            path.append(node)  # In-place modification - O(1)

            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    dfs(neighbor, path)
                elif neighbor in rec_stack:
                    # Found a cycle - extract it
                    if neighbor in path:
                        cycle_start = path.index(neighbor)
                        cycle = path[cycle_start:] + [neighbor]
                        # Normalize and use tuple for O(1) set lookup
                        normalized = self._normalize_cycle_tuple(cycle)
                        if normalized not in seen_cycles:
                            seen_cycles.add(normalized)
                            cycles.append(list(normalized))

            path.pop()  # In-place modification - O(1)
            rec_stack.remove(node)

        for node in graph:
            if node not in visited:
                dfs(node, [])

        return cycles

    def _normalize_cycle_tuple(self, cycle: List[str]) -> tuple:
        """
        Normalize a cycle to start with the lexicographically smallest node.
        Returns a tuple for efficient set-based deduplication.

        This helps deduplicate cycles like [A,B,C,A] == [B,C,A,B] == [C,A,B,C].

        Args:
            cycle: Cycle as [A, B, C, A]

        Returns:
            Normalized cycle as tuple starting with smallest node
        """
        if len(cycle) <= 1:
            return tuple(cycle)

        # Remove the duplicate last element for rotation
        nodes = cycle[:-1]
        if not nodes:
            return tuple(cycle)

        # Find index of minimum element
        min_idx = nodes.index(min(nodes))

        # Rotate to start with minimum and add closing node
        rotated = nodes[min_idx:] + nodes[:min_idx]
        return tuple(rotated + [rotated[0]])

    def get_external_dependencies(self, instance_name: str) -> Dict[str, Any]:
        """
        Get external package dependencies.

        Args:
            instance_name: RETER instance name

        Returns:
            dict with success, dependencies, queries
        """
        start_time = time.time()
        queries = []
        try:
            # Find imports that reference external packages
            module_concept = self._concept('Module')
            query = f"""
                SELECT ?module ?imported
                WHERE {{
                    ?module type {module_concept} .
                    ?module imports ?imported .
                    FILTER NOT EXISTS {{ ?imported type {module_concept} }}
                }}
            """
            queries.append(query.strip())

            result = self.reter.reql(query)
            rows = self._query_to_list(result)

            # Group by external package
            external_deps: Dict[str, List[str]] = {}
            for row in rows:
                module, imported = row[0], row[1]
                # Extract package name (first part before '.')
                pkg = imported.split('.')[0] if '.' in imported else imported
                if pkg not in external_deps:
                    external_deps[pkg] = []
                if module not in external_deps[pkg]:
                    external_deps[pkg].append(module)

            dependencies = [
                {"package": pkg, "used_by": modules, "usage_count": len(modules)}
                for pkg, modules in sorted(external_deps.items(), key=lambda x: -len(x[1]))
            ]

            time_ms = (time.time() - start_time) * 1000
            return {
                "success": True,
                "dependencies": dependencies,
                "count": len(dependencies),
                "queries": queries,
                "time_ms": time_ms
            }
        except Exception as e:
            time_ms = (time.time() - start_time) * 1000
            return {
                "success": False,
                "error": str(e),
                "dependencies": [],
                "count": 0,
                "queries": queries,
                "time_ms": time_ms
            }
