"""
Change Impact Analysis Tools

Provides tools for analyzing the impact of code changes.
"""

from typing import Dict, Any
import time
from .base import AdvancedToolsBase


class ChangeImpactTools(AdvancedToolsBase):
    """Change impact analysis tools."""

    def predict_change_impact(
        self,
        instance_name: str,
        entity_name: str
    ) -> Dict[str, Any]:
        """
        Predict impact of changing a function/method/class.

        Args:
            instance_name: RETER instance name
            entity_name: Name of the entity to analyze

        Returns:
            dict with success, direct_callers, indirect_callers, queries
        """
        start_time = time.time()
        queries = []
        try:
            # Find direct callers
            # Use inFile (works for all languages) instead of inModule
            direct_query = f"""
                SELECT ?caller ?callerName ?callerFile
                WHERE {{
                    ?caller calls ?target .
                    ?caller name ?callerName .
                    ?caller inFile ?callerFile .
                    FILTER(CONTAINS(?target, "{entity_name}"))
                }}
            """
            queries.append(direct_query.strip())

            result = self.reter.reql(direct_query)
            direct_rows = self._query_to_list(result)

            direct_callers = [
                {
                    "caller": row[0],
                    "name": row[1],
                    "file": row[2]
                }
                for row in direct_rows
            ]

            time_ms = (time.time() - start_time) * 1000
            return {
                "success": True,
                "entity": entity_name,
                "direct_callers": direct_callers,
                "direct_caller_count": len(direct_callers),
                "queries": queries,
                "time_ms": time_ms
            }
        except Exception as e:
            time_ms = (time.time() - start_time) * 1000
            return {
                "success": False,
                "error": str(e),
                "entity": entity_name,
                "direct_callers": [],
                "queries": queries,
                "time_ms": time_ms
            }

    def find_callers_recursive(
        self,
        instance_name: str,
        target_name: str
    ) -> Dict[str, Any]:
        """
        Find all callers of a function/method recursively.

        Args:
            instance_name: RETER instance name
            target_name: Name of the target function/method

        Returns:
            dict with success, callers list, queries
        """
        start_time = time.time()
        queries = []
        try:
            # Direct callers
            query = f"""
                SELECT ?caller ?callerName
                WHERE {{
                    ?caller calls ?target .
                    ?caller name ?callerName .
                    FILTER(CONTAINS(?target, "{target_name}"))
                }}
            """
            queries.append(query.strip())

            result = self.reter.reql(query)
            rows = self._query_to_list(result)

            callers = [
                {"caller": row[0], "name": row[1], "depth": 1}
                for row in rows
            ]

            time_ms = (time.time() - start_time) * 1000
            return {
                "success": True,
                "target": target_name,
                "callers": callers,
                "count": len(callers),
                "queries": queries,
                "time_ms": time_ms
            }
        except Exception as e:
            time_ms = (time.time() - start_time) * 1000
            return {
                "success": False,
                "error": str(e),
                "target": target_name,
                "callers": [],
                "queries": queries,
                "time_ms": time_ms
            }

    def find_callees_recursive(
        self,
        instance_name: str,
        source_name: str
    ) -> Dict[str, Any]:
        """
        Find all functions/methods called by a function recursively.

        Args:
            instance_name: RETER instance name
            source_name: Name of the source function/method

        Returns:
            dict with success, callees list, queries
        """
        start_time = time.time()
        queries = []
        try:
            query = f"""
                SELECT ?callee ?calleeName
                WHERE {{
                    ?source calls ?callee .
                    ?callee name ?calleeName .
                    FILTER(CONTAINS(?source, "{source_name}"))
                }}
            """
            queries.append(query.strip())

            result = self.reter.reql(query)
            rows = self._query_to_list(result)

            callees = [
                {"callee": row[0], "name": row[1], "depth": 1}
                for row in rows
            ]

            time_ms = (time.time() - start_time) * 1000
            return {
                "success": True,
                "source": source_name,
                "callees": callees,
                "count": len(callees),
                "queries": queries,
                "time_ms": time_ms
            }
        except Exception as e:
            time_ms = (time.time() - start_time) * 1000
            return {
                "success": False,
                "error": str(e),
                "source": source_name,
                "callees": [],
                "queries": queries,
                "time_ms": time_ms
            }
