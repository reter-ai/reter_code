"""
Architecture Analysis Tools

Provides tools for analyzing codebase architecture, structure, and organization.
"""

from typing import Dict, Any
from collections import defaultdict
import os
import time
from .base import AdvancedToolsBase


class ArchitectureAnalysisTools(AdvancedToolsBase):
    """Architecture and structure analysis tools."""

    def get_exception_hierarchy(self, instance_name: str) -> Dict[str, Any]:
        """
        Get exception class hierarchy (classes inheriting from Exception).

        Args:
            instance_name: RETER instance name

        Returns:
            dict with success, exception classes, count, queries
        """
        start_time = time.time()
        queries = []
        try:
            class_concept = self._concept('Class')
            query = f"""
                SELECT ?exception ?name ?parent
                WHERE {{
                    ?exception type {class_concept} .
                    ?exception name ?name .
                    ?exception inheritsFrom ?parent .
                    FILTER(REGEX(?parent, "Exception"))
                }}
                ORDER BY ?parent ?name
            """
            queries.append(query.strip())

            result = self.reter.reql(query)
            rows = self._query_to_list(result)

            exceptions = [
                {
                    "qualified_name": row[0],
                    "name": row[1],
                    "parent": row[2]
                }
                for row in rows
            ]

            time_ms = (time.time() - start_time) * 1000
            return {
                "success": True,
                "exceptions": exceptions,
                "count": len(exceptions),
                "queries": queries,
                "time_ms": time_ms
            }
        except Exception as e:
            time_ms = (time.time() - start_time) * 1000
            return {
                "success": False,
                "error": str(e),
                "exceptions": [],
                "count": 0,
                "queries": queries,
                "time_ms": time_ms
            }

    def get_package_structure(self, instance_name: str) -> Dict[str, Any]:
        """
        Get package/module structure of the codebase.

        Args:
            instance_name: RETER instance name

        Returns:
            dict with success, modules organized by path, count, queries
        """
        start_time = time.time()
        queries = []
        try:
            module_concept = self._concept('Module')
            query = f"""
                SELECT ?module ?file
                WHERE {{
                    ?module type {module_concept} .
                    ?module inFile ?file
                }}
                ORDER BY ?file
            """
            queries.append(query.strip())

            result = self.reter.reql(query)
            rows = self._query_to_list(result)

            modules = [
                {
                    "module": row[0],
                    "file": row[1]
                }
                for row in rows
            ]

            # Group by directory
            by_directory = defaultdict(list)
            for module in modules:
                directory = os.path.dirname(module["file"]) or "."
                by_directory[directory].append(module)

            time_ms = (time.time() - start_time) * 1000
            return {
                "success": True,
                "modules": modules,
                "by_directory": dict(by_directory),
                "module_count": len(modules),
                "directory_count": len(by_directory),
                "queries": queries,
                "time_ms": time_ms
            }
        except Exception as e:
            time_ms = (time.time() - start_time) * 1000
            return {
                "success": False,
                "error": str(e),
                "modules": [],
                "by_directory": {},
                "module_count": 0,
                "queries": queries,
                "time_ms": time_ms
            }

    def find_duplicate_names(self, instance_name: str) -> Dict[str, Any]:
        """
        Find entities with duplicate names across modules.

        Useful for identifying naming conflicts and potential confusion.

        Args:
            instance_name: RETER instance name

        Returns:
            dict with success, duplicates list, count, queries
        """
        start_time = time.time()
        queries = []
        try:
            class_concept = self._concept('Class')
            func_concept = self._concept('Function')

            # Find duplicate class names
            class_query = f"""
                SELECT ?name (COUNT(?class) AS ?count)
                WHERE {{
                    ?class type {class_concept} .
                    ?class name ?name
                }}
                GROUP BY ?name
                HAVING (?count > 1)
                ORDER BY DESC(?count)
            """
            queries.append(class_query.strip())
            class_result = self.reter.reql(class_query)
            class_rows = self._query_to_list(class_result)

            # Find duplicate function names
            function_query = f"""
                SELECT ?name (COUNT(?func) AS ?count)
                WHERE {{
                    ?func type {func_concept} .
                    ?func name ?name
                }}
                GROUP BY ?name
                HAVING (?count > 1)
                ORDER BY DESC(?count)
            """
            queries.append(function_query.strip())
            func_result = self.reter.reql(function_query)
            func_rows = self._query_to_list(func_result)

            # BATCH: Get all class details in one query (fixes N+1 problem)
            # Use inFile (works for all languages) instead of inModule
            all_class_details_query = f"""
                SELECT ?name ?class ?file
                WHERE {{
                    ?class type {class_concept} .
                    ?class name ?name .
                    ?class inFile ?file
                }}
            """
            queries.append(all_class_details_query.strip())
            all_class_details_result = self.reter.reql(all_class_details_query)
            all_class_details = self._query_to_list(all_class_details_result)

            # Build lookup: name -> list of (class, file)
            class_details_map = {}
            for name, class_id, file in all_class_details:
                if name not in class_details_map:
                    class_details_map[name] = []
                class_details_map[name].append((class_id, file))

            # Build duplicate_classes from pre-fetched data
            duplicate_classes = []
            for row in class_rows:
                name, count = row[0], row[1]
                details = class_details_map.get(name, [])
                duplicate_classes.append({
                    "name": name,
                    "type": "class",
                    "count": count,
                    "locations": [
                        {"class": d[0], "file": d[1]}
                        for d in details
                    ]
                })

            # BATCH: Get all function details in one query (fixes N+1 problem)
            # Use inFile (works for all languages) instead of inModule
            all_func_details_query = f"""
                SELECT ?name ?func ?file
                WHERE {{
                    ?func type {func_concept} .
                    ?func name ?name .
                    ?func inFile ?file
                }}
            """
            queries.append(all_func_details_query.strip())
            all_func_details_result = self.reter.reql(all_func_details_query)
            all_func_details = self._query_to_list(all_func_details_result)

            # Build lookup: name -> list of (func, file)
            func_details_map = {}
            for name, func_id, file in all_func_details:
                if name not in func_details_map:
                    func_details_map[name] = []
                func_details_map[name].append((func_id, file))

            # Build duplicate_functions from pre-fetched data
            duplicate_functions = []
            for row in func_rows:
                name, count = row[0], row[1]
                details = func_details_map.get(name, [])
                duplicate_functions.append({
                    "name": name,
                    "type": "function",
                    "count": count,
                    "locations": [
                        {"function": d[0], "file": d[1]}
                        for d in details
                    ]
                })

            all_duplicates = duplicate_classes + duplicate_functions

            time_ms = (time.time() - start_time) * 1000
            return {
                "success": True,
                "duplicates": all_duplicates,
                "duplicate_classes": len(duplicate_classes),
                "duplicate_functions": len(duplicate_functions),
                "count": len(all_duplicates),
                "queries": queries,
                "time_ms": time_ms
            }
        except Exception as e:
            time_ms = (time.time() - start_time) * 1000
            return {
                "success": False,
                "error": str(e),
                "duplicates": [],
                "count": 0,
                "queries": queries,
                "time_ms": time_ms
            }
