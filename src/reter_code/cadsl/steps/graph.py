"""
CADSL Graph Step Classes.

Graph-related pipeline steps for cycle detection, transitive closure,
and graph traversal.

Extracted from transformer.py to reduce file size.
"""

from collections import defaultdict, deque
import logging
from typing import Any, Callable, Optional


class GraphCyclesStep:
    """
    Detect cycles in a directed graph.

    Syntax: graph_cycles { from: field, to: field }

    Uses DFS to detect cycles and returns a list of cycles found.

    ::: This is-in-layer Domain-Specific-Language-Layer.
    ::: This is a step.
    ::: This is a pipeline-step.
    ::: This is-in-process Main-Process.
    ::: This is stateless.
    """

    def __init__(self, from_field, to_field):
        self.from_field = from_field
        self.to_field = to_field

    def execute(self, data, ctx=None):
        """Execute cycle detection using DFS."""
        from reter_code.dsl.core import pipeline_ok, pipeline_err

        try:
            # Convert to list if Arrow table
            if hasattr(data, 'to_pylist'):
                data = data.to_pylist()

            # Build adjacency list
            graph = defaultdict(list)
            for row in data:
                from_val = row.get(self.from_field)
                to_val = row.get(self.to_field)
                if from_val and to_val:
                    graph[from_val].append(to_val)

            # DFS for cycle detection
            cycles = []
            visited = set()
            rec_stack = set()

            def dfs(node, path):
                if node in rec_stack:
                    cycle_start = path.index(node)
                    cycle = path[cycle_start:]
                    cycles.append(tuple(cycle))
                    return
                if node in visited:
                    return

                visited.add(node)
                rec_stack.add(node)
                path.append(node)

                for neighbor in graph.get(node, []):
                    dfs(neighbor, path)

                path.pop()
                rec_stack.remove(node)

            for node in graph:
                if node not in visited:
                    dfs(node, [])

            # Convert cycles to result format
            result = [
                {"cycle": list(c), "length": len(c), "message": f"Cycle: {' -> '.join(map(str, c))} -> {c[0]}"}
                for c in cycles
            ]

            return pipeline_ok(result)
        except Exception as e:
            return pipeline_err("graph_cycles", f"Cycle detection failed: {e}", e)


class GraphClosureStep:
    """
    Compute transitive closure of a directed graph.

    Syntax: graph_closure { from: field, to: field, max_depth: 10 }

    Returns all reachable nodes from each source node.

    ::: This is-in-layer Domain-Specific-Language-Layer.
    ::: This is a step.
    ::: This is a pipeline-step.
    ::: This is-in-process Main-Process.
    ::: This is stateless.
    """

    def __init__(self, from_field, to_field, max_depth=10):
        self.from_field = from_field
        self.to_field = to_field
        self.max_depth = max_depth

    def execute(self, data, ctx=None):
        """Execute transitive closure computation."""
        from reter_code.dsl.core import pipeline_ok, pipeline_err

        try:
            # Convert to list if Arrow table
            if hasattr(data, 'to_pylist'):
                data = data.to_pylist()

            # Build adjacency list
            graph = defaultdict(set)
            for row in data:
                from_val = row.get(self.from_field)
                to_val = row.get(self.to_field)
                if from_val and to_val:
                    graph[from_val].add(to_val)

            # Compute closure using BFS
            result = []
            for start in graph:
                visited = set()
                queue = deque([(start, 0)])
                path = []

                while queue:
                    node, depth = queue.popleft()
                    if depth > self.max_depth:
                        continue
                    if node in visited:
                        continue

                    visited.add(node)
                    if node != start:
                        path.append(node)

                    for neighbor in graph.get(node, []):
                        if neighbor not in visited:
                            queue.append((neighbor, depth + 1))

                result.append({
                    "source": start,
                    "reachable": list(visited - {start}),
                    "count": len(visited) - 1,
                    "path": path[:self.max_depth],
                })

            return pipeline_ok(result)
        except Exception as e:
            return pipeline_err("graph_closure", f"Transitive closure failed: {e}", e)


class GraphTraverseStep:
    """
    Traverse a directed graph using BFS or DFS.

    Syntax: graph_traverse { from: field, to: field, algorithm: bfs, max_depth: 10 }

    ::: This is-in-layer Domain-Specific-Language-Layer.
    ::: This is a step.
    ::: This is a pipeline-step.
    ::: This is-in-process Main-Process.
    ::: This is stateless.
    """

    def __init__(self, from_field, to_field, algorithm="bfs", max_depth=10, root=None):
        self.from_field = from_field
        self.to_field = to_field
        self.algorithm = algorithm
        self.max_depth = max_depth
        self.root = root

    def execute(self, data, ctx=None):
        """Execute graph traversal and filter edges to reachable subgraph."""
        from reter_code.dsl.core import pipeline_ok, pipeline_err

        logger = logging.getLogger(__name__)

        try:
            # Convert to list if Arrow table, preserve original for filtering
            original_data = data
            if hasattr(data, 'to_pylist'):
                data = data.to_pylist()

            # Build adjacency list
            graph = defaultdict(list)
            nodes = set()
            for row in data:
                from_val = row.get(self.from_field)
                to_val = row.get(self.to_field)
                if from_val and to_val:
                    graph[from_val].append(to_val)
                    nodes.add(from_val)
                    nodes.add(to_val)

            # Determine root nodes
            # BUG-002 FIX: Handle string root parameter correctly
            if self.root:
                if callable(self.root):
                    # Root is a filter function
                    roots = [n for n in nodes if self.root({"node": n}, ctx)]
                elif isinstance(self.root, str):
                    # Root is a specific node name
                    if self.root in nodes:
                        roots = [self.root]
                    elif '::' in self.root:
                        # Try partial matching for qualified names only
                        # Match by suffix (last component) or full containment
                        suffix = self.root.split('::')[-1]
                        matching = [n for n in nodes if n.endswith('::' + suffix) or n == suffix or self.root in n]
                        if matching:
                            roots = matching[:1]  # Take first match
                            logger.debug(f"Root '{self.root}' matched to '{roots[0]}'")
                        else:
                            logger.warning(f"Root node '{self.root}' not found in graph with {len(nodes)} nodes")
                            return pipeline_ok([])
                    else:
                        # Non-qualified name - require exact match only
                        logger.warning(f"Root node '{self.root}' not found in graph with {len(nodes)} nodes")
                        return pipeline_ok([])
                else:
                    roots = [self.root] if self.root in nodes else []
            else:
                # No root specified - find nodes with no incoming edges
                has_incoming = set()
                for neighbors in graph.values():
                    has_incoming.update(neighbors)
                roots = [n for n in nodes if n not in has_incoming] or list(nodes)[:1]

            if not roots:
                logger.warning("No root nodes found for graph traversal")
                return pipeline_ok([])

            # Traverse to find reachable nodes
            visited = set()

            if self.algorithm == "bfs":
                queue = deque([(r, 0) for r in roots])
                while queue:
                    node, depth = queue.popleft()
                    if depth > self.max_depth or node in visited:
                        continue
                    visited.add(node)
                    for neighbor in graph.get(node, []):
                        if neighbor not in visited:
                            queue.append((neighbor, depth + 1))
            else:  # DFS
                def dfs(node, depth):
                    if depth > self.max_depth or node in visited:
                        return
                    visited.add(node)
                    for neighbor in graph.get(node, []):
                        dfs(neighbor, depth + 1)

                for root in roots:
                    dfs(root, 0)

            # BUG-002 FIX: Filter original data to only edges within the visited subgraph
            # Only include edges where BOTH endpoints are visited
            # This ensures max_depth is respected (unvisited to_nodes mean depth exceeded)
            filtered = []
            for row in data:
                from_val = row.get(self.from_field)
                to_val = row.get(self.to_field)
                # Include edge only if both from and to nodes are in visited set
                if from_val in visited and to_val in visited:
                    filtered.append(row)

            return pipeline_ok(filtered)
        except Exception as e:
            return pipeline_err("graph_traverse", f"Graph traversal failed: {e}", e)


class ParallelStep:
    """
    Execute multiple steps in parallel on the same input.

    Syntax: parallel { step1, step2, ... }

    Results from all steps are collected into a list.

    ::: This is-in-layer Domain-Specific-Language-Layer.
    ::: This is a step.
    ::: This is a pipeline-step.
    ::: This is-in-process Main-Process.
    ::: This is stateless.
    """

    def __init__(self, step_specs):
        self.step_specs = step_specs

    def execute(self, data, ctx=None):
        """Execute all steps on the same input and collect results."""
        from reter_code.dsl.core import (
            pipeline_ok, pipeline_err,
            FilterStep, SelectStep, MapStep, LimitStep, AggregateStep
        )

        results = []
        errors = []

        for spec in self.step_specs:
            step_type = spec.get("type")

            try:
                if step_type == "filter":
                    predicate = spec.get("predicate", lambda r, c=None: True)
                    step = FilterStep(predicate)
                elif step_type == "select":
                    fields = spec.get("fields", {})
                    step = SelectStep(fields)
                elif step_type == "map":
                    transform = spec.get("transform", lambda r, c=None: r)
                    step = MapStep(transform)
                elif step_type == "limit":
                    count = spec.get("count", 100)
                    step = LimitStep(count)
                elif step_type == "aggregate":
                    aggs = spec.get("aggregations", {})
                    step = AggregateStep(aggs)
                else:
                    # Unknown step type, skip
                    continue

                result = step.execute(data, ctx)
                if result.is_ok():
                    results.append(result.unwrap())
                else:
                    errors.append(result)
            except Exception as e:
                errors.append(pipeline_err("parallel", f"Step failed: {e}", e))

        if errors and not results:
            return errors[0]

        return pipeline_ok(results)


__all__ = [
    "GraphCyclesStep",
    "GraphClosureStep",
    "GraphTraverseStep",
    "ParallelStep",
]
