"""
Code Quality Analysis Tools

Provides code quality metrics like finding large classes and long parameter lists.
"""

from typing import Dict, Any, List
import time
from .base import AdvancedToolsBase


class CodeQualityTools(AdvancedToolsBase):
    """Code quality analysis tools."""

    def find_large_classes(
        self,
        instance_name: str,
        threshold: int = 20
    ) -> Dict[str, Any]:
        """
        Find classes with too many methods (God classes).

        Args:
            instance_name: RETER instance name
            threshold: Minimum number of methods (default: 20)

        Returns:
            dict with success, classes list, count, queries
        """
        start_time = time.time()
        queries = []
        try:
            class_concept = self._concept('Class')
            method_concept = self._concept('Method')
            # Use inFile (works for all languages) instead of inModule (Python-specific)
            query = f"""
                SELECT ?class ?name ?file (COUNT(?method) AS ?method_count)
                WHERE {{
                    ?class type {class_concept} .
                    ?class name ?name .
                    ?class inFile ?file .
                    ?method type {method_concept} .
                    ?method definedIn ?class
                }}
                GROUP BY ?class ?name ?file
                HAVING (?method_count >= {threshold})
                ORDER BY DESC(?method_count)
            """
            queries.append(query.strip())

            result = self.reter.reql(query)
            rows = self._query_to_list(result)

            classes = [
                {
                    "qualified_name": row[0],
                    "name": row[1],
                    "file": row[2],  # Use file (works for all languages)
                    "method_count": int(row[3]) if row[3] else 0
                }
                for row in rows
            ]

            time_ms = (time.time() - start_time) * 1000
            return {
                "success": True,
                "classes": classes,
                "count": len(classes),
                "threshold": threshold,
                "queries": queries,
                "time_ms": time_ms
            }
        except Exception as e:
            time_ms = (time.time() - start_time) * 1000
            return {
                "success": False,
                "error": str(e),
                "classes": [],
                "count": 0,
                "queries": queries,
                "time_ms": time_ms
            }

    def find_long_parameter_lists(
        self,
        instance_name: str,
        threshold: int = 5
    ) -> Dict[str, Any]:
        """
        Find functions/methods with too many parameters.

        Args:
            instance_name: RETER instance name
            threshold: Maximum acceptable parameter count (default: 5)

        Returns:
            dict with success, functions list, count, queries
        """
        start_time = time.time()
        queries = []
        try:
            param_concept = self._concept('Parameter')
            func_concept = self._concept('Function')
            method_concept = self._concept('Method')
            query = f"""
                SELECT ?func ?name ?type (COUNT(?param) AS ?param_count)
                WHERE {{
                    ?func type ?type .
                    ?func name ?name .
                    ?param type {param_concept} .
                    ?param ofFunction ?func .
                    FILTER(?type = {func_concept} || ?type = {method_concept})
                }}
                GROUP BY ?func ?name ?type
                HAVING (?param_count > {threshold})
                ORDER BY DESC(?param_count)
            """
            queries.append(query.strip())

            result = self.reter.reql(query)
            rows = self._query_to_list(result)

            functions = [
                {
                    "qualified_name": row[0],
                    "name": row[1],
                    "type": row[2],
                    "parameter_count": int(row[3]) if row[3] else 0
                }
                for row in rows
            ]

            time_ms = (time.time() - start_time) * 1000
            return {
                "success": True,
                "functions": functions,
                "count": len(functions),
                "threshold": threshold,
                "queries": queries,
                "time_ms": time_ms
            }
        except Exception as e:
            time_ms = (time.time() - start_time) * 1000
            return {
                "success": False,
                "error": str(e),
                "functions": [],
                "count": 0,
                "queries": queries,
                "time_ms": time_ms
            }

    def find_magic_numbers(
        self,
        instance_name: str,
        exclude_common: bool = True,
        min_occurrences: int = 1,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        Find magic numbers (numeric literals) in code.

        Magic numbers are numeric literals embedded directly in code that should
        typically be extracted to named constants for better maintainability.

        Args:
            instance_name: RETER instance name
            exclude_common: Exclude common non-magic numbers like 0, 1, 2 (default: True)
            min_occurrences: Minimum occurrences to report (default: 1)
            limit: Maximum results (default: 100)
            offset: Pagination offset (default: 0)

        Returns:
            dict with success, magic_numbers list, by_value grouping, count, queries
        """
        start_time = time.time()
        queries = []

        # Common numbers that are typically not "magic"
        common_numbers = {
            "0", "1", "-1", "2", "10", "100", "1000",
            "0.0", "1.0", "0.5", "2.0",
            "60", "24", "365",  # Time-related
            "8", "16", "32", "64", "128", "256", "512", "1024", "2048", "4096",  # Powers of 2
            "255", "65535",  # Max values
        }

        try:
            # Query all number literals with their context
            # Use inFile (works for all languages) instead of inModule
            query = """
                SELECT ?entity ?num ?file ?entityName ?entityType
                WHERE {
                    ?entity hasNumberLiteral ?num .
                    ?entity inFile ?file .
                    ?entity name ?entityName .
                    ?entity type ?entityType .
                }
                ORDER BY ?num ?entity
            """
            queries.append(query.strip())

            result = self.reter.reql(query)
            rows = self._query_to_list(result)

            # Group by number value
            by_value: Dict[str, list] = {}
            all_findings = []

            for entity, num, file, entity_name, entity_type in rows:
                # Filter common numbers if requested
                if exclude_common and num in common_numbers:
                    continue

                finding = {
                    "entity": entity,
                    "entity_name": entity_name,
                    "entity_type": entity_type,
                    "file": file,
                    "value": num
                }

                if num not in by_value:
                    by_value[num] = []
                by_value[num].append(finding)
                all_findings.append(finding)

            # Filter by min_occurrences and sort by occurrence count
            filtered_by_value = {
                k: v for k, v in by_value.items()
                if len(v) >= min_occurrences
            }

            # Sort by number of occurrences (most repeated first)
            sorted_values = sorted(
                filtered_by_value.items(),
                key=lambda x: (-len(x[1]), x[0])
            )

            # Apply pagination
            paginated_values = sorted_values[offset:offset + limit]

            # Build summary of repeated magic numbers
            repeated_magic = [
                {
                    "value": value,
                    "occurrences": len(locations),
                    "locations": locations[:5],  # Limit locations shown
                    "suggestion": f"Consider extracting '{value}' to a named constant"
                }
                for value, locations in paginated_values
                if len(locations) >= 2  # Only show repeated ones in summary
            ]

            time_ms = (time.time() - start_time) * 1000
            return {
                "success": True,
                "magic_numbers": all_findings[:limit],
                "by_value": {k: v for k, v in paginated_values},
                "repeated_magic": repeated_magic,
                "count": len(all_findings),
                "unique_values": len(by_value),
                "repeated_count": len([v for v in by_value.values() if len(v) >= 2]),
                "exclude_common": exclude_common,
                "min_occurrences": min_occurrences,
                "queries": queries,
                "time_ms": time_ms
            }
        except Exception as e:
            time_ms = (time.time() - start_time) * 1000
            return {
                "success": False,
                "error": str(e),
                "magic_numbers": [],
                "by_value": {},
                "repeated_magic": [],
                "count": 0,
                "queries": queries,
                "time_ms": time_ms
            }
