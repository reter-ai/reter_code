"""
Arrow-based Pipeline Implementation

This module provides PyArrow-native pipeline operations for efficient
data processing, joins, and aggregations.

Key features:
- Sources return pa.Table instead of List[Dict]
- Steps operate on Arrow tables using vectorized operations
- Native join support (inner, left, right, outer, semi, anti)
- Lazy conversion to List[Dict] only when needed (python blocks, emit)
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.compute as pc
from dataclasses import dataclass, field
from typing import (
    TypeVar, Generic, Callable, List, Dict, Any, Optional,
    Union, Literal, Tuple
)
from abc import ABC, abstractmethod

from .catpy import (
    PipelineResult,
    pipeline_ok, pipeline_err,
)
from .core import Context

T = TypeVar("T")
U = TypeVar("U")

# Join types supported by PyArrow
JoinType = Literal["inner", "left", "right", "outer", "semi", "anti"]


# =============================================================================
# Arrow Table Utilities
# =============================================================================

def dict_list_to_table(data: List[Dict[str, Any]]) -> pa.Table:
    """Convert list of dicts to PyArrow table."""
    if not data:
        return pa.table({})

    # Collect all keys from all rows
    all_keys = set()
    for row in data:
        all_keys.update(row.keys())

    # Build columns
    columns = {key: [row.get(key) for row in data] for key in all_keys}
    return pa.table(columns)


def table_to_dict_list(table: pa.Table, preserve_nulls: bool = True) -> List[Dict[str, Any]]:
    """Convert PyArrow table to list of dicts.

    Args:
        table: PyArrow table to convert
        preserve_nulls: If True, ensures all columns from schema are present in each row,
                       even if the value is None. This is important for OPTIONAL clause results
                       where PyArrow's to_pylist() may omit keys for null values.
    """
    if table is None or table.num_rows == 0:
        return []

    results = table.to_pylist()

    if preserve_nulls:
        # PyArrow's to_pylist() may not include keys for null values (e.g., from OPTIONAL)
        # Ensure all columns from the schema are present in each row
        column_names = table.column_names
        for row in results:
            for col_name in column_names:
                if col_name not in row:
                    row[col_name] = None

    return results


def ensure_table(data: Union[pa.Table, List[Dict[str, Any]]]) -> pa.Table:
    """Ensure data is a PyArrow table."""
    if isinstance(data, pa.Table):
        return data
    return dict_list_to_table(data)


def ensure_list(data: Union[pa.Table, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Ensure data is a list of dicts."""
    if isinstance(data, pa.Table):
        return table_to_dict_list(data)
    return data


# =============================================================================
# Arrow Sources
# =============================================================================

class ArrowSource(ABC):
    """Base class for sources that return Arrow tables.

    ::: This is-in-layer Domain-Specific-Language-Layer.
    ::: This is a source.
    ::: This is a data-source.
    ::: This is stateless.
    ::: This is-in-process Main-Process.
    """

    @abstractmethod
    def execute(self, ctx: Context) -> PipelineResult[pa.Table]:
        """Execute the source and return an Arrow table."""
        pass


@dataclass
class ArrowREQLSource(ArrowSource):
    """REQL query source that returns Arrow table directly.

    REQL queries use plain type names (CNL naming convention):
    - class, method, function (language-independent)
    - Predicates use hyphenated format: is-in-file, has-name, is-defined-in

    ::: This is-in-layer Domain-Specific-Language-Layer.
    ::: This is a source.
    ::: This is a data-source.
    ::: This is stateless.
    ::: This is-in-process Main-Process.
    """
    query: str

    def execute(self, ctx: Context) -> PipelineResult[pa.Table]:
        """Execute REQL query and return Arrow table directly."""
        try:
            query = self.query

            # Substitute parameters
            for key, value in ctx.params.items():
                placeholder = "{" + key + "}"
                if placeholder in query:
                    query = query.replace(placeholder, str(value))

            # Execute query - returns PyArrow table
            table = ctx.reter.reql(query)

            if table is None:
                return pipeline_ok(pa.table({}))

            return pipeline_ok(table)
        except Exception as e:
            return pipeline_err("reql", f"Query failed: {e}", e)


@dataclass
class ArrowRAGSource(ArrowSource):
    """RAG source that returns Arrow table.

    ::: This is-in-layer Domain-Specific-Language-Layer.
    ::: This is a source.
    ::: This is a data-source.
    ::: This is stateless.
    ::: This is-in-process Main-Process.
    """
    operation: Literal["search", "duplicates", "clusters"]
    params: Dict[str, Any] = field(default_factory=dict)

    def execute(self, ctx: Context) -> PipelineResult[pa.Table]:
        """Execute RAG operation and return Arrow table."""
        try:
            from reter_code.services.default_instance_manager import DefaultInstanceManager

            rag_manager = ctx.get("rag_manager")
            if rag_manager is None:
                default_mgr = DefaultInstanceManager.get_instance()
                if default_mgr:
                    rag_manager = default_mgr.get_rag_manager()

            if rag_manager is None:
                return pipeline_err("rag", "RAG manager not available")

            if self.operation == "search":
                result = rag_manager.semantic_search(
                    query=self.params.get("query", ""),
                    top_k=self.params.get("top_k", 10),
                    entity_types=self.params.get("entity_types")
                )
                data = result.get("results", [])

            elif self.operation == "duplicates":
                result = rag_manager.find_duplicate_candidates(
                    similarity_threshold=self.params.get("similarity", 0.85),
                    max_results=self.params.get("limit", 50),
                    exclude_same_file=self.params.get("exclude_same_file", True),
                    exclude_same_class=self.params.get("exclude_same_class", True),
                    entity_types=self.params.get("entity_types", ["method", "function"])
                )
                # Flatten pairs to rows
                data = []
                for pair in result.get("pairs", []):
                    e1, e2 = pair["entity1"], pair["entity2"]
                    data.append({
                        "similarity": pair["similarity"],
                        "entity1_name": e1["name"],
                        "entity1_file": e1["file"],
                        "entity1_line": e1["line"],
                        "entity2_name": e2["name"],
                        "entity2_file": e2["file"],
                        "entity2_line": e2["line"],
                    })

            elif self.operation == "clusters":
                result = rag_manager.find_similar_clusters(
                    n_clusters=self.params.get("n_clusters", 50),
                    min_cluster_size=self.params.get("min_size", 2),
                    exclude_same_file=self.params.get("exclude_same_file", True),
                    exclude_same_class=self.params.get("exclude_same_class", True),
                    entity_types=self.params.get("entity_types", ["method", "function"])
                )
                data = []
                for cluster in result.get("clusters", []):
                    unique_files = list(set(m["file"] for m in cluster["members"]))
                    data.append({
                        "cluster_id": cluster["cluster_id"],
                        "member_count": cluster["member_count"],
                        "unique_files": len(unique_files),  # Count for CADSL templates
                        "members": [m["name"] for m in cluster["members"]],
                        "files": unique_files,
                    })
            else:
                return pipeline_err("rag", f"Unknown RAG operation: {self.operation}")

            return pipeline_ok(dict_list_to_table(data))
        except Exception as e:
            return pipeline_err("rag", f"RAG operation failed: {e}", e)


@dataclass
class ArrowValueSource(ArrowSource):
    """Literal value source as Arrow table.

    ::: This is-in-layer Domain-Specific-Language-Layer.
    ::: This is a source.
    ::: This is a data-source.
    ::: This is stateless.
    ::: This is-in-process Main-Process.
    """
    value: Union[List[Dict], pa.Table]

    def execute(self, ctx: Context) -> PipelineResult[pa.Table]:
        return pipeline_ok(ensure_table(self.value))


@dataclass
class ArrowMergeSource(ArrowSource):
    """Merge multiple sources (UNION ALL).

    ::: This is-in-layer Domain-Specific-Language-Layer.
    ::: This is a source.
    ::: This is a data-source.
    ::: This is stateless.
    ::: This is-in-process Main-Process.
    """
    sources: List[ArrowSource]

    def execute(self, ctx: Context) -> PipelineResult[pa.Table]:
        """Execute all sources and concatenate results."""
        try:
            tables = []
            for source in self.sources:
                result = source.execute(ctx)
                if result.is_err():
                    return result
                table = result.unwrap()
                if table.num_rows > 0:
                    tables.append(table)

            if not tables:
                return pipeline_ok(pa.table({}))

            # Concatenate all tables
            combined = pa.concat_tables(tables, promote_options="default")
            return pipeline_ok(combined)
        except Exception as e:
            return pipeline_err("merge", f"Merge failed: {e}", e)


# =============================================================================
# Arrow Steps
# =============================================================================

class ArrowStep(ABC):
    """Base class for steps that operate on Arrow tables.

    ::: This is-in-layer Domain-Specific-Language-Layer.
    ::: This is a step.
    ::: This is a pipeline-step.
    ::: This is stateless.
    ::: This is-in-process Main-Process.
    """

    @abstractmethod
    def execute(self, table: pa.Table, ctx: Optional[Context] = None) -> PipelineResult[pa.Table]:
        """Transform the input table."""
        pass


@dataclass
class ArrowFilterStep(ArrowStep):
    """Filter rows using Arrow compute expressions.

    ::: This is-in-layer Domain-Specific-Language-Layer.
    ::: This is a step.
    ::: This is a pipeline-step.
    ::: This is stateless.
    ::: This is-in-process Main-Process.
    """
    condition: str  # Expression like "count > 10"

    def execute(self, table: pa.Table, ctx: Optional[Context] = None) -> PipelineResult[pa.Table]:
        try:
            if table.num_rows == 0:
                return pipeline_ok(table)

            # Parse and evaluate condition
            mask = self._evaluate_condition(table, self.condition, ctx)
            filtered = table.filter(mask)
            return pipeline_ok(filtered)
        except Exception as e:
            return pipeline_err("filter", f"Filter failed: {e}", e)

    def _evaluate_condition(self, table: pa.Table, condition: str, ctx: Optional[Context]) -> pa.Array:
        """Evaluate a condition expression to a boolean mask."""
        import re

        # Substitute parameters
        if ctx:
            for key, value in ctx.params.items():
                condition = condition.replace("{" + key + "}", str(value))

        # Parse simple conditions: field op value
        # Supports: ==, !=, >, <, >=, <=, and, or

        # Handle 'and' / 'or' first
        if ' and ' in condition:
            parts = condition.split(' and ', 1)
            left = self._evaluate_condition(table, parts[0].strip(), ctx)
            right = self._evaluate_condition(table, parts[1].strip(), ctx)
            return pc.and_(left, right)

        if ' or ' in condition:
            parts = condition.split(' or ', 1)
            left = self._evaluate_condition(table, parts[0].strip(), ctx)
            right = self._evaluate_condition(table, parts[1].strip(), ctx)
            return pc.or_(left, right)

        # Parse comparison: field op value
        match = re.match(r'(\w+)\s*(==|!=|>=|<=|>|<)\s*(.+)', condition.strip())
        if not match:
            raise ValueError(f"Cannot parse condition: {condition}")

        field_name, op, value_str = match.groups()

        # Get column (try with and without ? prefix)
        if field_name in table.column_names:
            column = table.column(field_name)
        elif f"?{field_name}" in table.column_names:
            column = table.column(f"?{field_name}")
        else:
            raise ValueError(f"Column not found: {field_name}")

        # Parse value
        value_str = value_str.strip()
        if value_str.startswith('"') and value_str.endswith('"'):
            value = value_str[1:-1]
        elif value_str.startswith("'") and value_str.endswith("'"):
            value = value_str[1:-1]
        elif value_str.lower() == 'true':
            value = True
        elif value_str.lower() == 'false':
            value = False
        elif '.' in value_str:
            value = float(value_str)
        else:
            try:
                value = int(value_str)
            except ValueError:
                value = value_str

        # Apply comparison
        if op == '==':
            return pc.equal(column, value)
        elif op == '!=':
            return pc.not_equal(column, value)
        elif op == '>':
            return pc.greater(column, value)
        elif op == '<':
            return pc.less(column, value)
        elif op == '>=':
            return pc.greater_equal(column, value)
        elif op == '<=':
            return pc.less_equal(column, value)
        else:
            raise ValueError(f"Unknown operator: {op}")


@dataclass
class ArrowSelectStep(ArrowStep):
    """Select and rename columns.

    ::: This is-in-layer Domain-Specific-Language-Layer.
    ::: This is a step.
    ::: This is a pipeline-step.
    ::: This is stateless.
    ::: This is-in-process Main-Process.
    """
    fields: Dict[str, str]  # output_name -> source_name

    def execute(self, table: pa.Table, ctx: Optional[Context] = None) -> PipelineResult[pa.Table]:
        try:
            if table.num_rows == 0:
                return pipeline_ok(pa.table({}))

            columns = {}
            for out_name, src_name in self.fields.items():
                clean_out = out_name.lstrip("?")

                # Find source column
                if src_name in table.column_names:
                    columns[clean_out] = table.column(src_name)
                elif f"?{src_name}" in table.column_names:
                    columns[clean_out] = table.column(f"?{src_name}")
                elif out_name in table.column_names:
                    columns[clean_out] = table.column(out_name)
                elif f"?{out_name}" in table.column_names:
                    columns[clean_out] = table.column(f"?{out_name}")

            return pipeline_ok(pa.table(columns))
        except Exception as e:
            return pipeline_err("select", f"Select failed: {e}", e)


@dataclass
class ArrowOrderByStep(ArrowStep):
    """Sort table by column.

    ::: This is-in-layer Domain-Specific-Language-Layer.
    ::: This is a step.
    ::: This is a pipeline-step.
    ::: This is stateless.
    ::: This is-in-process Main-Process.
    """
    field_name: str
    descending: bool = False

    def execute(self, table: pa.Table, ctx: Optional[Context] = None) -> PipelineResult[pa.Table]:
        try:
            if table.num_rows == 0:
                return pipeline_ok(table)

            # Find column
            col_name = self.field_name
            if col_name not in table.column_names:
                if f"?{col_name}" in table.column_names:
                    col_name = f"?{col_name}"
                else:
                    return pipeline_ok(table)  # Column not found, return as-is

            order = "descending" if self.descending else "ascending"
            indices = pc.sort_indices(table, sort_keys=[(col_name, order)])
            sorted_table = table.take(indices)
            return pipeline_ok(sorted_table)
        except Exception as e:
            return pipeline_err("order_by", f"Sort failed: {e}", e)


@dataclass
class ArrowLimitStep(ArrowStep):
    """Limit number of rows.

    ::: This is-in-layer Domain-Specific-Language-Layer.
    ::: This is a step.
    ::: This is a pipeline-step.
    ::: This is stateless.
    ::: This is-in-process Main-Process.
    """
    count: int

    def execute(self, table: pa.Table, ctx: Optional[Context] = None) -> PipelineResult[pa.Table]:
        count = self.count
        if ctx and isinstance(count, str) and count.startswith("{"):
            param_name = count[1:-1]
            count = ctx.params.get(param_name, 100)
        return pipeline_ok(table.slice(0, int(count)))


@dataclass
class ArrowOffsetStep(ArrowStep):
    """Skip first N rows.

    ::: This is-in-layer Domain-Specific-Language-Layer.
    ::: This is a step.
    ::: This is a pipeline-step.
    ::: This is stateless.
    ::: This is-in-process Main-Process.
    """
    count: int

    def execute(self, table: pa.Table, ctx: Optional[Context] = None) -> PipelineResult[pa.Table]:
        return pipeline_ok(table.slice(self.count))


@dataclass
class ArrowUniqueStep(ArrowStep):
    """Remove duplicate rows based on columns.

    ::: This is-in-layer Domain-Specific-Language-Layer.
    ::: This is a step.
    ::: This is a pipeline-step.
    ::: This is stateless.
    ::: This is-in-process Main-Process.
    """
    columns: Optional[List[str]] = None

    def execute(self, table: pa.Table, ctx: Optional[Context] = None) -> PipelineResult[pa.Table]:
        try:
            if table.num_rows == 0:
                return pipeline_ok(table)

            # Use all columns if none specified
            cols = self.columns or table.column_names

            # Find valid columns
            valid_cols = []
            for col in cols:
                if col in table.column_names:
                    valid_cols.append(col)
                elif f"?{col}" in table.column_names:
                    valid_cols.append(f"?{col}")

            if not valid_cols:
                return pipeline_ok(table)

            # Group by the columns and take first of each group
            unique = table.group_by(valid_cols).aggregate([])
            return pipeline_ok(unique)
        except Exception as e:
            return pipeline_err("unique", f"Unique failed: {e}", e)


@dataclass
class ArrowMapStep(ArrowStep):
    """Apply transformations to create new columns.

    Transforms is a dict of column_name -> expression or value.
    Special key '...row' means include all existing columns.

    ::: This is-in-layer Domain-Specific-Language-Layer.
    ::: This is a step.
    ::: This is a pipeline-step.
    ::: This is stateless.
    ::: This is-in-process Main-Process.
    """
    transforms: Dict[str, Any]

    def execute(self, table: pa.Table, ctx: Optional[Context] = None) -> PipelineResult[pa.Table]:
        try:
            columns = {}

            # Check if we should preserve existing columns
            preserve_existing = "...row" in self.transforms

            if preserve_existing:
                for name in table.column_names:
                    clean_name = name.lstrip("?")
                    columns[clean_name] = table.column(name)

            # Apply transforms
            for out_name, expr in self.transforms.items():
                if out_name == "...row":
                    continue

                if isinstance(expr, str):
                    # String interpolation: "Class '{name}' has {count} methods"
                    if "{" in expr and "}" in expr:
                        # Build interpolated strings
                        values = []
                        for i in range(table.num_rows):
                            row = {name.lstrip("?"): table.column(name)[i].as_py()
                                   for name in table.column_names}
                            if ctx:
                                row.update(ctx.params)
                            try:
                                values.append(expr.format(**row))
                            except (KeyError, IndexError):
                                # KeyError: missing named placeholder
                                # IndexError: positional placeholder {0} with no positional args
                                values.append(expr)
                        columns[out_name] = pa.array(values)
                    else:
                        # Literal string for all rows
                        columns[out_name] = pa.array([expr] * table.num_rows)
                elif callable(expr):
                    # Python function - need to convert to list, apply, convert back
                    rows = table.to_pylist()
                    values = [expr(row) for row in rows]
                    columns[out_name] = pa.array(values)
                else:
                    # Literal value
                    columns[out_name] = pa.array([expr] * table.num_rows)

            return pipeline_ok(pa.table(columns))
        except Exception as e:
            return pipeline_err("map", f"Map failed: {e}", e)


@dataclass
class ArrowAggregateStep(ArrowStep):
    """Aggregate table with functions.

    ::: This is-in-layer Domain-Specific-Language-Layer.
    ::: This is a step.
    ::: This is a pipeline-step.
    ::: This is stateless.
    ::: This is-in-process Main-Process.
    """
    aggregations: Dict[str, Tuple[str, str]]  # output -> (field, func)

    def execute(self, table: pa.Table, ctx: Optional[Context] = None) -> PipelineResult[pa.Table]:
        try:
            result = {"count": table.num_rows}

            for out_name, (field_name, func) in self.aggregations.items():
                # Find column
                col_name = field_name
                if col_name not in table.column_names:
                    if f"?{col_name}" in table.column_names:
                        col_name = f"?{col_name}"
                    else:
                        continue

                column = table.column(col_name)

                if func == "count":
                    result[out_name] = pc.count(column).as_py()
                elif func == "sum":
                    result[out_name] = pc.sum(column).as_py()
                elif func == "mean" or func == "avg":
                    result[out_name] = pc.mean(column).as_py()
                elif func == "min":
                    result[out_name] = pc.min(column).as_py()
                elif func == "max":
                    result[out_name] = pc.max(column).as_py()
                elif func == "stddev":
                    result[out_name] = pc.stddev(column).as_py()

            return pipeline_ok(pa.table({k: [v] for k, v in result.items()}))
        except Exception as e:
            return pipeline_err("aggregate", f"Aggregation failed: {e}", e)


@dataclass
class ArrowGroupByStep(ArrowStep):
    """Group by columns with aggregations.

    ::: This is-in-layer Domain-Specific-Language-Layer.
    ::: This is a step.
    ::: This is a pipeline-step.
    ::: This is stateless.
    ::: This is-in-process Main-Process.
    """
    group_columns: List[str]
    aggregations: List[Tuple[str, str, str]]  # (output_name, source_column, func)

    def execute(self, table: pa.Table, ctx: Optional[Context] = None) -> PipelineResult[pa.Table]:
        try:
            if table.num_rows == 0:
                return pipeline_ok(table)

            # Resolve column names
            cols = []
            for col in self.group_columns:
                if col in table.column_names:
                    cols.append(col)
                elif f"?{col}" in table.column_names:
                    cols.append(f"?{col}")

            if not cols:
                return pipeline_err("group_by", "No valid group columns found")

            # Build aggregation specs
            agg_specs = []
            for out_name, src_col, func in self.aggregations:
                # Find source column
                if src_col in table.column_names:
                    col = src_col
                elif f"?{src_col}" in table.column_names:
                    col = f"?{src_col}"
                else:
                    continue

                # Map func name to Arrow function
                func_map = {
                    "count": "count",
                    "sum": "sum",
                    "avg": "mean",
                    "mean": "mean",
                    "min": "min",
                    "max": "max",
                }
                arrow_func = func_map.get(func, func)
                agg_specs.append((col, arrow_func))

            grouped = table.group_by(cols).aggregate(agg_specs)
            return pipeline_ok(grouped)
        except Exception as e:
            return pipeline_err("group_by", f"Group by failed: {e}", e)


@dataclass
class ArrowJoinStep(ArrowStep):
    """Join two tables.

    ::: This is-in-layer Domain-Specific-Language-Layer.
    ::: This is a step.
    ::: This is a pipeline-step.
    ::: This is stateless.
    ::: This is-in-process Main-Process.
    """
    right_source: ArrowSource
    left_keys: List[str]
    right_keys: List[str]
    join_type: JoinType = "inner"

    def execute(self, table: pa.Table, ctx: Optional[Context] = None) -> PipelineResult[pa.Table]:
        """Execute join with right source."""
        try:
            # Execute right source
            right_result = self.right_source.execute(ctx)
            if right_result.is_err():
                return right_result
            right_table = right_result.unwrap()

            # Resolve column names
            left_cols = []
            for col in self.left_keys:
                if col in table.column_names:
                    left_cols.append(col)
                elif f"?{col}" in table.column_names:
                    left_cols.append(f"?{col}")
                else:
                    return pipeline_err("join", f"Left key column not found: {col}")

            right_cols = []
            for col in self.right_keys:
                if col in right_table.column_names:
                    right_cols.append(col)
                elif f"?{col}" in right_table.column_names:
                    right_cols.append(f"?{col}")
                else:
                    return pipeline_err("join", f"Right key column not found: {col}")

            # Perform join
            # PyArrow join syntax varies by version, using the hash_join approach
            joined = table.join(
                right_table,
                keys=left_cols,
                right_keys=right_cols,
                join_type=self.join_type
            )

            return pipeline_ok(joined)
        except Exception as e:
            return pipeline_err("join", f"Join failed: {e}", e)


@dataclass
class ArrowPythonStep(ArrowStep):
    """Execute Python code on data.

    Converts Arrow table to list of dicts for Python execution,
    then converts result back to Arrow table.

    ::: This is-in-layer Domain-Specific-Language-Layer.
    ::: This is a step.
    ::: This is a pipeline-step.
    ::: This is stateless.
    ::: This is-in-process Main-Process.
    """
    code: str

    def execute(self, table: pa.Table, ctx: Optional[Context] = None) -> PipelineResult[pa.Table]:
        try:
            # Convert to list of dicts for Python code
            rows = table_to_dict_list(table)
            params = ctx.params if ctx else {}

            # Create execution namespace
            namespace = {
                "rows": rows,
                "params": params,
                "ctx": ctx,
                "result": None,
            }

            # Execute code
            exec(self.code, namespace)

            # Get result
            result = namespace.get("result", rows)

            # Convert back to Arrow table
            if isinstance(result, list):
                return pipeline_ok(dict_list_to_table(result))
            elif isinstance(result, dict):
                return pipeline_ok(dict_list_to_table([result]))
            elif isinstance(result, pa.Table):
                return pipeline_ok(result)
            else:
                return pipeline_ok(dict_list_to_table([{"result": result}]))
        except Exception as e:
            return pipeline_err("python", f"Python execution failed: {e}", e)


# =============================================================================
# Arrow Pipeline
# =============================================================================

@dataclass
class ArrowPipeline:
    """
    PyArrow-native pipeline for efficient data processing.

    Uses Arrow tables throughout, only converting to list of dicts
    when explicitly needed (python blocks, emit).

    ::: This is-in-layer Domain-Specific-Language-Layer.
    ::: This is a monad.
    ::: This is a executor.
    ::: This is stateful.
    ::: This is-in-process Main-Process.
    """
    _source: ArrowSource
    _steps: List[ArrowStep] = field(default_factory=list)
    _emit_key: Optional[str] = None

    @classmethod
    def from_reql(cls, query: str) -> "ArrowPipeline":
        """Create pipeline from REQL query."""
        return cls(_source=ArrowREQLSource(query))

    @classmethod
    def from_rag(cls, operation: str, **params) -> "ArrowPipeline":
        """Create pipeline from RAG operation."""
        return cls(_source=ArrowRAGSource(operation=operation, params=params))

    @classmethod
    def from_value(cls, value: Union[List[Dict], pa.Table]) -> "ArrowPipeline":
        """Create pipeline from literal value."""
        return cls(_source=ArrowValueSource(value))

    @classmethod
    def from_merge(cls, *sources: ArrowSource) -> "ArrowPipeline":
        """Create pipeline by merging multiple sources."""
        return cls(_source=ArrowMergeSource(list(sources)))

    def __rshift__(self, step: ArrowStep) -> "ArrowPipeline":
        """Add step: pipeline >> step"""
        new_steps = self._steps + [step]
        return ArrowPipeline(_source=self._source, _steps=new_steps, _emit_key=self._emit_key)

    def __or__(self, step: ArrowStep) -> "ArrowPipeline":
        """Add step: pipeline | step"""
        return self >> step

    def filter(self, condition: str) -> "ArrowPipeline":
        """Filter rows by condition."""
        return self >> ArrowFilterStep(condition)

    def select(self, *fields: str, **renames: str) -> "ArrowPipeline":
        """Select and rename columns."""
        field_map = {f: f for f in fields}
        field_map.update(renames)
        return self >> ArrowSelectStep(field_map)

    def order_by(self, field: str) -> "ArrowPipeline":
        """Sort by column."""
        descending = field.startswith("-")
        if descending:
            field = field[1:]
        return self >> ArrowOrderByStep(field, descending)

    def limit(self, count: int) -> "ArrowPipeline":
        """Limit rows."""
        return self >> ArrowLimitStep(count)

    def offset(self, count: int) -> "ArrowPipeline":
        """Skip rows."""
        return self >> ArrowOffsetStep(count)

    def unique(self, *columns: str) -> "ArrowPipeline":
        """Remove duplicates."""
        return self >> ArrowUniqueStep(list(columns) if columns else None)

    def map(self, **transforms) -> "ArrowPipeline":
        """Transform/add columns."""
        return self >> ArrowMapStep(transforms)

    def aggregate(self, **aggregations: Tuple[str, str]) -> "ArrowPipeline":
        """Aggregate."""
        return self >> ArrowAggregateStep(aggregations)

    def group_by(self, *columns: str, **aggs) -> "ArrowPipeline":
        """Group and aggregate."""
        aggregations = [(k, v[0], v[1]) for k, v in aggs.items()]
        return self >> ArrowGroupByStep(list(columns), aggregations)

    def join(self, right: ArrowSource, on: List[str],
             right_on: Optional[List[str]] = None,
             how: JoinType = "inner") -> "ArrowPipeline":
        """Join with another source."""
        right_keys = right_on or on
        return self >> ArrowJoinStep(right, on, right_keys, how)

    def python(self, code: str) -> "ArrowPipeline":
        """Execute Python code."""
        return self >> ArrowPythonStep(code)

    def emit(self, key: str) -> "ArrowPipeline":
        """Set output key."""
        return ArrowPipeline(_source=self._source, _steps=self._steps, _emit_key=key)

    def run(self, ctx: Context) -> PipelineResult[pa.Table]:
        """Execute pipeline and return Arrow table."""
        # Execute source
        result = self._source.execute(ctx)
        if result.is_err():
            return result

        # Run through steps
        table = result.unwrap()
        for step in self._steps:
            result = step.execute(table, ctx)
            if result.is_err():
                return result
            table = result.unwrap()

        return pipeline_ok(table)

    def execute(self, ctx: Context) -> Dict[str, Any]:
        """Execute and return output dict."""
        result = self.run(ctx)

        if result.is_err():
            return {
                "success": False,
                "error": str(result.error) if hasattr(result, 'error') else str(result)
            }

        table = result.unwrap()
        data = table_to_dict_list(table)

        output = {"success": True}

        if self._emit_key:
            output[self._emit_key] = data
            output["count"] = len(data)
        else:
            output["results"] = data
            output["count"] = len(data)

        return output
