"""
CADSL Join Steps.

Contains step classes for joining and merging data in CADSL pipelines:
- JoinStep: Join pipeline data with another source using PyArrow
- MergeSource: Merge multiple sources into one
- CrossJoinStep: Cross join (Cartesian product) for N² pairwise comparison
"""

from typing import Any, Dict, List, Optional


class JoinStep:
    """
    Join step - joins pipeline data with another source using PyArrow.

    Syntax: join { left: key, right: source, right_key: key, type: inner }

    Supports all PyArrow join types: inner, left, right, outer, semi, anti.

    ::: This is-in-layer Domain-Specific-Language-Layer.
    ::: This is a step.
    ::: This is-in-process Main-Process.
    ::: This is stateless.
    """

    def __init__(self, left_key, right_source_spec, right_key, join_type="inner"):
        self.left_key = left_key
        self.right_source_spec = right_source_spec
        self.right_key = right_key
        self.join_type = join_type

    def execute(self, data, ctx=None):
        """Execute join using PyArrow."""
        from reter_code.dsl.core import pipeline_ok, pipeline_err

        try:
            import pyarrow as pa

            # Convert left data to Arrow table
            if isinstance(data, pa.Table):
                left_table = data
            elif isinstance(data, list):
                if not data:
                    return pipeline_ok([])
                # Convert list of dicts to Arrow table
                left_table = pa.Table.from_pylist(data)
            else:
                return pipeline_err("join", "Left data must be a list or Arrow table")

            # Execute right source
            right_data = self._execute_right_source(ctx)
            if isinstance(right_data, list):
                if not right_data:
                    # Right side empty - return empty for inner, left data for left/outer
                    if self.join_type in ("inner", "semi"):
                        return pipeline_ok([])
                    elif self.join_type in ("left", "outer"):
                        return pipeline_ok(data if isinstance(data, list) else data.to_pylist())
                    elif self.join_type == "anti":
                        return pipeline_ok(data if isinstance(data, list) else data.to_pylist())
                    return pipeline_ok([])
                right_table = pa.Table.from_pylist(right_data)
            elif isinstance(right_data, pa.Table):
                right_table = right_data
            else:
                return pipeline_err("join", "Right source must return a list or Arrow table")

            # Resolve column names (handle ? prefix)
            left_col = self._resolve_column(left_table, self.left_key)
            right_col = self._resolve_column(right_table, self.right_key)

            if left_col is None:
                return pipeline_err("join", f"Left key column not found: {self.left_key}")
            if right_col is None:
                return pipeline_err("join", f"Right key column not found: {self.right_key}")

            # Map CADSL join types to PyArrow join types
            # PyArrow uses "left outer" not "left", "right outer" not "right", etc.
            pyarrow_join_type_map = {
                "inner": "inner",
                "left": "left outer",
                "right": "right outer",
                "outer": "full outer",
                "semi": "left semi",
                "anti": "left anti"
            }
            pa_join_type = pyarrow_join_type_map.get(self.join_type, self.join_type)

            # Perform join
            joined = left_table.join(
                right_table,
                keys=left_col,
                right_keys=right_col,
                join_type=pa_join_type
            )

            # PyArrow join drops the right key column (since it equals left key).
            # If the right key is different from left key name, add it back.
            if right_col != left_col and right_col not in joined.column_names:
                # Add right key column as copy of left key column
                left_col_data = joined.column(left_col)
                joined = joined.append_column(right_col, left_col_data)

            # Convert back to list of dicts
            return pipeline_ok(joined.to_pylist())
        except Exception as e:
            return pipeline_err("join", f"Join failed: {e}", e)

    def _resolve_column(self, table, col_name):
        """Resolve column name, handling ? prefix."""
        if col_name in table.column_names:
            return col_name
        if f"?{col_name}" in table.column_names:
            return f"?{col_name}"
        return None

    def _execute_right_source(self, ctx):
        """Execute the right source and return data."""
        from reter_code.dsl.core import (
            REQLSource, ValueSource,
            RAGSearchSource, RAGDuplicatesSource, RAGClustersSource, RAGDBScanSource
        )

        spec = self.right_source_spec
        if spec is None:
            return []

        source_type = spec.get("type")

        # Helper to resolve parameter placeholders like "{similarity}" to actual values
        def resolve_param(value, default=None):
            if isinstance(value, str) and value.startswith("{") and value.endswith("}"):
                param_name = value[1:-1]
                if ctx and hasattr(ctx, 'params') and param_name in ctx.params:
                    return ctx.params[param_name]
                return default
            return value if value is not None else default

        if source_type == "reql":
            source = REQLSource(spec.get("content", ""))
        elif source_type == "rag_search":
            params = spec.get("params", {})
            source = RAGSearchSource(
                query=resolve_param(params.get("query"), ""),
                top_k=resolve_param(params.get("top_k"), 10),
                entity_types=resolve_param(params.get("entity_types")),
            )
        elif source_type == "rag_duplicates":
            params = spec.get("params", {})
            source = RAGDuplicatesSource(
                similarity=resolve_param(params.get("similarity"), 0.85),
                limit=resolve_param(params.get("limit"), 50),
                exclude_same_file=resolve_param(params.get("exclude_same_file"), True),
                exclude_same_class=resolve_param(params.get("exclude_same_class"), True),
                entity_types=resolve_param(params.get("entity_types")),
            )
        elif source_type == "rag_clusters":
            params = spec.get("params", {})
            source = RAGClustersSource(
                n_clusters=resolve_param(params.get("n_clusters"), 50),
                min_size=resolve_param(params.get("min_size"), 2),
                exclude_same_file=resolve_param(params.get("exclude_same_file"), True),
                exclude_same_class=resolve_param(params.get("exclude_same_class"), True),
                entity_types=resolve_param(params.get("entity_types")),
            )
        elif source_type == "rag_dbscan":
            params = spec.get("params", {})
            source = RAGDBScanSource(
                eps=resolve_param(params.get("eps"), 0.5),
                min_samples=resolve_param(params.get("min_samples"), 3),
                min_size=resolve_param(params.get("min_size"), 2),
                exclude_same_file=resolve_param(params.get("exclude_same_file"), True),
                exclude_same_class=resolve_param(params.get("exclude_same_class"), True),
                entity_types=resolve_param(params.get("entity_types")),
            )
        elif source_type == "value":
            source = ValueSource(spec.get("content", []))
        else:
            return []

        result = source.execute(ctx)
        if result.is_ok():
            return result.unwrap()
        return []


class MergeSource:
    """
    Merge multiple sources into one.

    Syntax: merge { source1, source2, ... }
    Or with per-source steps: merge { source1 | step1 | step2, source2 | step3 }

    Executes all sources (with their steps) and concatenates their results.

    ::: This is-in-layer Domain-Specific-Language-Layer.
    ::: This is a source.
    ::: This is-in-process Main-Process.
    ::: This is stateless.
    """

    def __init__(self, source_specs):
        self.source_specs = source_specs

    def execute(self, ctx=None):
        """Execute all sources and merge results."""
        from reter_code.dsl.core import (
            pipeline_ok, pipeline_err,
            REQLSource, ValueSource,
            RAGSearchSource, RAGDuplicatesSource, RAGClustersSource, RAGDBScanSource,
            Pipeline, SelectStep, MapStep, FilterStep
        )

        merged = []
        errors = []

        for spec in self.source_specs:
            source_type = spec.get("type")

            try:
                if source_type == "reql":
                    source = REQLSource(spec.get("content", ""))
                elif source_type == "rag_search":
                    params = spec.get("params", {})
                    source = RAGSearchSource(
                        query=params.get("query", ""),
                        top_k=params.get("top_k", 10),
                        entity_types=params.get("entity_types"),
                    )
                elif source_type == "rag_duplicates":
                    params = spec.get("params", {})
                    source = RAGDuplicatesSource(
                        similarity=params.get("similarity", 0.85),
                        limit=params.get("limit", 50),
                        exclude_same_file=params.get("exclude_same_file", True),
                        exclude_same_class=params.get("exclude_same_class", True),
                        entity_types=params.get("entity_types"),
                    )
                elif source_type == "rag_clusters":
                    params = spec.get("params", {})
                    source = RAGClustersSource(
                        n_clusters=params.get("n_clusters", 50),
                        min_size=params.get("min_size", 2),
                        exclude_same_file=params.get("exclude_same_file", True),
                        exclude_same_class=params.get("exclude_same_class", True),
                        entity_types=params.get("entity_types"),
                    )
                elif source_type == "rag_dbscan":
                    params = spec.get("params", {})
                    source = RAGDBScanSource(
                        eps=params.get("eps", 0.5),
                        min_samples=params.get("min_samples", 3),
                        min_size=params.get("min_size", 2),
                        exclude_same_file=params.get("exclude_same_file", True),
                        exclude_same_class=params.get("exclude_same_class", True),
                        entity_types=params.get("entity_types"),
                    )
                elif source_type == "value":
                    source = ValueSource(spec.get("content", []))
                else:
                    continue

                # Execute source
                result = source.execute(ctx)
                if result.is_err():
                    errors.append(result)
                    continue

                data = result.unwrap()

                # Apply per-source steps if any
                steps = spec.get("steps", [])
                if steps:
                    data = self._apply_steps(data, steps, ctx)

                # Merge into results
                if isinstance(data, list):
                    merged.extend(data)
                elif hasattr(data, 'to_pylist'):  # PyArrow table
                    merged.extend(data.to_pylist())
                else:
                    merged.append(data)

            except Exception as e:
                errors.append(pipeline_err("merge", f"Source failed: {e}", e))

        if errors and not merged:
            return errors[0]

        return pipeline_ok(merged)

    def _apply_steps(self, data, steps, ctx):
        """Apply pipeline steps to data."""
        from reter_code.dsl.core import SelectStep, MapStep, FilterStep, pipeline_ok

        # Convert Arrow table to list if needed
        if hasattr(data, 'to_pylist'):
            data = data.to_pylist()

        for step_spec in steps:
            step_type = step_spec.get("type")

            if step_type == "select":
                fields = step_spec.get("fields", {})
                step = SelectStep(fields)
                result = step.execute(data, ctx)
                if result.is_ok():
                    data = result.unwrap()

            elif step_type == "map":
                transform = step_spec.get("transform", lambda r, ctx=None: r)
                step = MapStep(transform)
                result = step.execute(data, ctx)
                if result.is_ok():
                    data = result.unwrap()

            elif step_type == "filter":
                predicate = step_spec.get("predicate", lambda r, ctx=None: True)
                step = FilterStep(predicate)
                result = step.execute(data, ctx)
                if result.is_ok():
                    data = result.unwrap()

        # Convert back to list if it's Arrow
        if hasattr(data, 'to_pylist'):
            data = data.to_pylist()

        return data


class CrossJoinStep:
    """
    Cross join (Cartesian product) step for N² pairwise comparison.

    Syntax: cross_join { unique_pairs: true, exclude_self: true, left_prefix: "left_", right_prefix: "right_" }

    Creates all pairs from input rows. With unique_pairs=true, generates (n*(n-1))/2 pairs.
    Uses PyArrow for efficient vectorized operations.

    ::: This is-in-layer Domain-Specific-Language-Layer.
    ::: This is a step.
    ::: This is-in-process Main-Process.
    ::: This is stateless.
    """

    def __init__(self, unique_pairs=True, exclude_self=True, left_prefix="left_", right_prefix="right_"):
        self.unique_pairs = unique_pairs
        self.exclude_self = exclude_self
        self.left_prefix = left_prefix
        self.right_prefix = right_prefix

    def execute(self, data, ctx=None):
        """Execute cross join using PyArrow."""
        from reter_code.dsl.core import pipeline_ok, pipeline_err

        try:
            import numpy as np
            import pyarrow as pa
            import pyarrow.compute as pc

            # Convert to Arrow table
            if isinstance(data, pa.Table):
                table = data
            elif isinstance(data, list):
                if not data:
                    return pipeline_ok([])
                table = pa.Table.from_pylist(data)
            else:
                return pipeline_err("cross_join", "Input must be a list or Arrow table")

            n = table.num_rows
            if n < 2:
                return pipeline_ok([])

            # Create index arrays
            idx = np.arange(n, dtype=np.int64)

            if self.unique_pairs:
                # Generate (i, j) pairs where i < j
                i_grid, j_grid = np.meshgrid(idx, idx, indexing='ij')
                mask = i_grid < j_grid
                left_idx = pa.array(i_grid[mask])
                right_idx = pa.array(j_grid[mask])
            else:
                # Full Cartesian product
                i_grid, j_grid = np.meshgrid(idx, idx, indexing='ij')
                if self.exclude_self:
                    mask = i_grid != j_grid
                    left_idx = pa.array(i_grid[mask])
                    right_idx = pa.array(j_grid[mask])
                else:
                    left_idx = pa.array(i_grid.ravel())
                    right_idx = pa.array(j_grid.ravel())

            # Build result table with prefixed columns
            result = {}
            for col in table.column_names:
                result[f'{self.left_prefix}{col}'] = pc.take(table.column(col), left_idx)
                result[f'{self.right_prefix}{col}'] = pc.take(table.column(col), right_idx)

            result_table = pa.table(result)
            return pipeline_ok(result_table.to_pylist())

        except ImportError:
            return pipeline_err("cross_join", "PyArrow and NumPy are required for cross_join")
        except Exception as e:
            return pipeline_err("cross_join", f"Cross join failed: {e}", e)
