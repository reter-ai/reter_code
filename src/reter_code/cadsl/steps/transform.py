"""
CADSL Transform Steps.

Contains step classes for data transformation in CADSL pipelines:
- PivotStep: Create a pivot table from data
- ComputeStep: Compute new fields using expressions
"""

from typing import Any, Dict, List, Optional


class PivotStep:
    """
    Create a pivot table from data.

    Syntax: pivot { rows: field, cols: field, value: field, aggregate: sum }

    ::: This is-in-layer Domain-Specific-Language-Layer.
    ::: This is a step.
    ::: This is-in-process Main-Process.
    ::: This is stateless.
    """

    def __init__(self, rows, cols, value, aggregate="sum"):
        self.rows = rows
        self.cols = cols
        self.value = value
        self.aggregate = aggregate

    def execute(self, data, ctx=None):
        """Execute pivot table creation."""
        from reter_code.dsl.core import pipeline_ok, pipeline_err
        from collections import defaultdict

        try:
            # Convert to list if Arrow table
            if hasattr(data, 'to_pylist'):
                data = data.to_pylist()

            # Collect values by (row, col)
            pivot_data = defaultdict(list)
            all_cols = set()
            all_rows = set()

            for row in data:
                row_key = row.get(self.rows)
                col_key = row.get(self.cols)
                val = row.get(self.value, 0)

                if row_key is not None and col_key is not None:
                    pivot_data[(row_key, col_key)].append(val if val is not None else 0)
                    all_rows.add(row_key)
                    all_cols.add(col_key)

            # Aggregate
            def aggregate_values(values):
                if not values:
                    return 0
                if self.aggregate == "sum":
                    return sum(values)
                elif self.aggregate == "avg":
                    return sum(values) / len(values)
                elif self.aggregate == "count":
                    return len(values)
                elif self.aggregate == "min":
                    return min(values)
                elif self.aggregate == "max":
                    return max(values)
                elif self.aggregate == "first":
                    return values[0]
                elif self.aggregate == "last":
                    return values[-1]
                return sum(values)

            # Build result table
            result = []
            sorted_cols = sorted(all_cols, key=str)
            for row_key in sorted(all_rows, key=str):
                row_result = {self.rows: row_key}
                for col_key in sorted_cols:
                    values = pivot_data.get((row_key, col_key), [])
                    row_result[str(col_key)] = aggregate_values(values)
                result.append(row_result)

            return pipeline_ok(result)
        except Exception as e:
            return pipeline_err("pivot", f"Pivot failed: {e}", e)


class ComputeStep:
    """
    Compute new fields using expressions.

    Syntax: compute { ratio: a / b, pct: ratio * 100 }

    ::: This is-in-layer Domain-Specific-Language-Layer.
    ::: This is a step.
    ::: This is-in-process Main-Process.
    ::: This is stateless.
    """

    def __init__(self, computations):
        self.computations = computations

    def execute(self, data, ctx=None):
        """Execute field computation."""
        from reter_code.dsl.core import pipeline_ok, pipeline_err

        try:
            # Convert to list if Arrow table
            if hasattr(data, 'to_pylist'):
                data = data.to_pylist()

            result = []
            for row in data:
                new_row = dict(row)
                # Compute each field in order (so later fields can reference earlier ones)
                for name, expr in self.computations.items():
                    try:
                        new_row[name] = expr(new_row, ctx)
                    except Exception:
                        new_row[name] = None
                result.append(new_row)

            return pipeline_ok(result)
        except Exception as e:
            return pipeline_err("compute", f"Compute failed: {e}", e)
