"""
CADSL Data Flow Step Classes.

Data aggregation and transformation pipeline steps.

Extracted from transformer.py to reduce file size.
"""

from collections import defaultdict
from typing import Callable, Optional


class CollectStep:
    """
    Aggregate rows by key, collecting fields into sets/lists.

    Syntax: collect { by: field, name: op(field) }

    Operations: set, list, first, last, count, sum, avg, min, max

    ::: This is-in-layer Domain-Specific-Language-Layer.
    ::: This is a step.
    ::: This is a pipeline-step.
    ::: This is-in-process Main-Process.
    ::: This is stateless.
    """

    def __init__(self, by: str, fields: dict):
        self.by = by
        self.fields = fields  # {output_name: (source_field, operation)}

    def execute(self, data, ctx=None):
        """Execute collection/aggregation."""
        from reter_code.dsl.core import pipeline_ok, pipeline_err

        try:
            # Convert to list if Arrow table
            if hasattr(data, 'to_pylist'):
                data = data.to_pylist()

            groups = {}
            for row in data:
                key = row.get(self.by)
                if key not in groups:
                    groups[key] = {self.by: key}
                    for name, (source, op) in self.fields.items():
                        if op in ('set', 'list'):
                            groups[key][f"_{name}_values"] = []
                        else:
                            groups[key][f"_{name}_values"] = []

                for name, (source, op) in self.fields.items():
                    value = row.get(source)
                    if value is not None:
                        groups[key][f"_{name}_values"].append(value)

            # Apply aggregation operations
            result = []
            for key, group in groups.items():
                out = {self.by: key}
                for name, (source, op) in self.fields.items():
                    values = group.get(f"_{name}_values", [])
                    if op == 'set':
                        out[name] = list(dict.fromkeys(values))  # Preserve order, remove dupes
                    elif op == 'list':
                        out[name] = values
                    elif op == 'first':
                        out[name] = values[0] if values else None
                    elif op == 'last':
                        out[name] = values[-1] if values else None
                    elif op == 'count':
                        out[name] = len(values)
                    elif op == 'sum':
                        out[name] = sum(v for v in values if isinstance(v, (int, float)))
                    elif op == 'avg':
                        nums = [v for v in values if isinstance(v, (int, float))]
                        out[name] = sum(nums) / len(nums) if nums else 0
                    elif op == 'min':
                        out[name] = min(values) if values else None
                    elif op == 'max':
                        out[name] = max(values) if values else None
                    else:
                        out[name] = values
                result.append(out)

            return pipeline_ok(result)
        except Exception as e:
            return pipeline_err("collect", f"Collect failed: {e}", e)


class NestStep:
    """
    Create nested/tree structure from flat data.

    Syntax: nest { parent: field, child: field, root: expr, max_depth: 10 }

    ::: This is-in-layer Domain-Specific-Language-Layer.
    ::: This is a step.
    ::: This is a pipeline-step.
    ::: This is-in-process Main-Process.
    ::: This is stateless.
    """

    def __init__(self, parent: str, child: str, root=None, max_depth=10, children_key="children"):
        self.parent = parent
        self.child = child
        self.root = root
        self.max_depth = max_depth
        self.children_key = children_key

    def execute(self, data, ctx=None):
        """Execute nesting."""
        from reter_code.dsl.core import pipeline_ok, pipeline_err

        try:
            # Convert to list if Arrow table
            if hasattr(data, 'to_pylist'):
                data = data.to_pylist()

            # Build parent->children map
            children_map = defaultdict(list)
            all_items = {}
            for row in data:
                child_id = row.get(self.child)
                parent_id = row.get(self.parent)
                if child_id:
                    all_items[child_id] = dict(row)
                    if parent_id:
                        children_map[parent_id].append(child_id)

            # Find roots
            if self.root and callable(self.root):
                roots = [cid for cid, item in all_items.items()
                        if self.root(item, ctx)]
            else:
                # Items with no parent are roots
                has_parent = set()
                for row in data:
                    parent_id = row.get(self.parent)
                    child_id = row.get(self.child)
                    if parent_id and child_id:
                        has_parent.add(child_id)
                roots = [cid for cid in all_items if cid not in has_parent]

            # Build tree recursively
            def build_tree(item_id, depth=0):
                if depth > self.max_depth or item_id not in all_items:
                    return None
                node = dict(all_items[item_id])
                child_ids = children_map.get(item_id, [])
                if child_ids:
                    node[self.children_key] = [
                        build_tree(cid, depth + 1)
                        for cid in child_ids
                        if build_tree(cid, depth + 1) is not None
                    ]
                return node

            result = [build_tree(r) for r in roots if build_tree(r) is not None]
            return pipeline_ok(result)
        except Exception as e:
            return pipeline_err("nest", f"Nest failed: {e}", e)


__all__ = [
    "CollectStep",
    "NestStep",
]
