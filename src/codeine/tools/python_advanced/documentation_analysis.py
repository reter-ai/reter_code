"""
Documentation Analysis Tools

Provides tools for analyzing code documentation and finding undocumented code.
"""

from typing import Dict, Any
import time
from .base import AdvancedToolsBase


class DocumentationAnalysisTools(AdvancedToolsBase):
    """Documentation analysis tools."""

    def find_undocumented_code(self, instance_name: str) -> Dict[str, Any]:
        """
        Find undocumented classes and functions (missing docstrings).

        Args:
            instance_name: RETER instance name

        Returns:
            dict with success, undocumented entities, count, queries
        """
        start_time = time.time()
        queries = []
        try:
            class_concept = self._concept('Class')
            func_concept = self._concept('Function')
            method_concept = self._concept('Method')

            # Direct query for entities without docstrings (not relying on SWRL)
            # Use inFile (works for all languages) instead of inModule
            query = f"""
                SELECT ?entity ?name ?type ?file
                WHERE {{
                    ?entity type ?type .
                    ?entity name ?name .
                    ?entity inFile ?file .
                    FILTER(?type = {class_concept} || ?type = {func_concept} || ?type = {method_concept})
                    FILTER NOT EXISTS {{ ?entity hasDocstring ?docstring }}
                }}
                ORDER BY ?file ?type ?name
            """
            queries.append(query.strip())

            result = self.reter.reql(query)
            rows = self._query_to_list(result)

            entities = [
                {
                    "qualified_name": row[0],
                    "name": row[1],
                    "type": row[2],
                    "file": row[3]
                }
                for row in rows
            ]

            time_ms = (time.time() - start_time) * 1000
            return {
                "success": True,
                "entities": entities,
                "count": len(entities),
                "queries": queries,
                "time_ms": time_ms
            }
        except Exception as e:
            time_ms = (time.time() - start_time) * 1000
            return {
                "success": False,
                "error": str(e),
                "entities": [],
                "count": 0,
                "queries": queries,
                "time_ms": time_ms
            }

    def get_api_documentation(self, instance_name: str) -> Dict[str, Any]:
        """
        Extract all API documentation (classes and functions with docstrings).

        Args:
            instance_name: RETER instance name

        Returns:
            dict with success, documented entities, count, queries
        """
        start_time = time.time()
        queries = []
        try:
            class_concept = self._concept('Class')
            func_concept = self._concept('Function')

            # Use inFile (works for all languages) instead of inModule
            query = f"""
                SELECT ?entity ?name ?type ?file ?docstring
                WHERE {{
                    ?entity type ?type .
                    ?entity name ?name .
                    ?entity inFile ?file .
                    ?entity hasDocstring ?docstring .
                    FILTER(?type = {class_concept} || ?type = {func_concept})
                }}
                ORDER BY ?file ?type ?name
            """
            queries.append(query.strip())

            result = self.reter.reql(query)
            rows = self._query_to_list(result)

            entities = [
                {
                    "qualified_name": row[0],
                    "name": row[1],
                    "type": row[2],
                    "file": row[3],
                    "docstring": row[4]
                }
                for row in rows
            ]

            time_ms = (time.time() - start_time) * 1000
            return {
                "success": True,
                "entities": entities,
                "count": len(entities),
                "queries": queries,
                "time_ms": time_ms
            }
        except Exception as e:
            time_ms = (time.time() - start_time) * 1000
            return {
                "success": False,
                "error": str(e),
                "entities": [],
                "count": 0,
                "queries": queries,
                "time_ms": time_ms
            }
