"""
Type Analysis Tools

Provides tools for analyzing type hints and type safety.
"""

from typing import Dict, Any
import time
from .base import AdvancedToolsBase


class TypeAnalysisTools(AdvancedToolsBase):
    """Type analysis tools."""

    def get_type_hints(
        self,
        instance_name: str,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        Extract all type hints from parameters and return types.

        Args:
            instance_name: RETER instance name
            limit: Maximum results to return
            offset: Number of results to skip

        Returns:
            dict with success, type_hints list, queries
        """
        start_time = time.time()
        queries = []
        try:
            query = f"""
                SELECT ?func ?funcName ?param ?paramName ?typeHint
                WHERE {{
                    ?func name ?funcName .
                    ?param ofFunction ?func .
                    ?param name ?paramName .
                    ?param hasType ?typeHint .
                    FILTER(?typeHint != "")
                }}
                LIMIT {limit}
                OFFSET {offset}
            """
            queries.append(query.strip())

            result = self.reter.reql(query)
            rows = self._query_to_list(result)

            type_hints = [
                {
                    "function": row[0],
                    "function_name": row[1],
                    "parameter": row[2],
                    "parameter_name": row[3],
                    "type_hint": row[4]
                }
                for row in rows
            ]

            time_ms = (time.time() - start_time) * 1000
            return {
                "success": True,
                "type_hints": type_hints,
                "count": len(type_hints),
                "limit": limit,
                "offset": offset,
                "queries": queries,
                "time_ms": time_ms
            }
        except Exception as e:
            time_ms = (time.time() - start_time) * 1000
            return {
                "success": False,
                "error": str(e),
                "type_hints": [],
                "queries": queries,
                "time_ms": time_ms
            }

    def find_untyped_functions(self, instance_name: str) -> Dict[str, Any]:
        """
        Find functions/methods without type hints.

        Args:
            instance_name: RETER instance name

        Returns:
            dict with success, untyped_functions list, queries
        """
        start_time = time.time()
        queries = []
        try:
            func_concept = self._concept('Function')
            method_concept = self._concept('Method')
            # Find functions without return type hints
            # Use inFile (works for all languages) instead of inModule
            query = f"""
                SELECT ?func ?name ?file
                WHERE {{
                    ?func type ?type .
                    ?func name ?name .
                    ?func inFile ?file .
                    FILTER(?type = {func_concept} || ?type = {method_concept})
                    FILTER NOT EXISTS {{ ?func hasReturnType ?returnType }}
                }}
                ORDER BY ?file ?name
            """
            queries.append(query.strip())

            result = self.reter.reql(query)
            rows = self._query_to_list(result)

            untyped = [
                {
                    "qualified_name": row[0],
                    "name": row[1],
                    "file": row[2]
                }
                for row in rows
            ]

            time_ms = (time.time() - start_time) * 1000
            return {
                "success": True,
                "untyped_functions": untyped,
                "count": len(untyped),
                "queries": queries,
                "time_ms": time_ms
            }
        except Exception as e:
            time_ms = (time.time() - start_time) * 1000
            return {
                "success": False,
                "error": str(e),
                "untyped_functions": [],
                "queries": queries,
                "time_ms": time_ms
            }
