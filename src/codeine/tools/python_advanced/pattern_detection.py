"""
Pattern Detection Tools

Provides tools for detecting code patterns like decorators, magic methods, interfaces.
"""

from typing import Dict, Any, Optional
import time
from .base import AdvancedToolsBase


class PatternDetectionTools(AdvancedToolsBase):
    """Pattern detection tools."""

    def find_decorators_usage(
        self,
        instance_name: str,
        decorator_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Find all uses of decorators, optionally filtered by name.

        Args:
            instance_name: RETER instance name
            decorator_name: Optional specific decorator to find

        Returns:
            dict with success, decorators list, queries
        """
        start_time = time.time()
        queries = []
        try:
            if decorator_name:
                query = f"""
                    SELECT ?func ?funcName ?decorator
                    WHERE {{
                        ?func hasDecorator ?decorator .
                        ?func name ?funcName .
                        FILTER(CONTAINS(?decorator, "{decorator_name}"))
                    }}
                """
            else:
                query = """
                    SELECT ?func ?funcName ?decorator
                    WHERE {
                        ?func hasDecorator ?decorator .
                        ?func name ?funcName
                    }
                """
            queries.append(query.strip())

            result = self.reter.reql(query)
            rows = self._query_to_list(result)

            decorators = [
                {
                    "function": row[0],
                    "function_name": row[1],
                    "decorator": row[2]
                }
                for row in rows
            ]

            time_ms = (time.time() - start_time) * 1000
            return {
                "success": True,
                "decorators": decorators,
                "count": len(decorators),
                "filter": decorator_name,
                "queries": queries,
                "time_ms": time_ms
            }
        except Exception as e:
            time_ms = (time.time() - start_time) * 1000
            return {
                "success": False,
                "error": str(e),
                "decorators": [],
                "count": 0,
                "queries": queries,
                "time_ms": time_ms
            }

    def get_magic_methods(self, instance_name: str) -> Dict[str, Any]:
        """
        Find all magic methods (__init__, __str__, etc.).

        Args:
            instance_name: RETER instance name

        Returns:
            dict with success, magic_methods list, queries
        """
        start_time = time.time()
        queries = []
        try:
            method_concept = self._concept('Method')
            query = f"""
                SELECT ?method ?name ?class ?className
                WHERE {{
                    ?method type {method_concept} .
                    ?method name ?name .
                    ?method definedIn ?class .
                    ?class name ?className .
                    FILTER(STRSTARTS(?name, "__") && STRENDS(?name, "__"))
                }}
                ORDER BY ?className ?name
            """
            queries.append(query.strip())

            result = self.reter.reql(query)
            rows = self._query_to_list(result)

            magic_methods = [
                {
                    "method": row[0],
                    "name": row[1],
                    "class": row[2],
                    "class_name": row[3]
                }
                for row in rows
            ]

            time_ms = (time.time() - start_time) * 1000
            return {
                "success": True,
                "magic_methods": magic_methods,
                "count": len(magic_methods),
                "queries": queries,
                "time_ms": time_ms
            }
        except Exception as e:
            time_ms = (time.time() - start_time) * 1000
            return {
                "success": False,
                "error": str(e),
                "magic_methods": [],
                "count": 0,
                "queries": queries,
                "time_ms": time_ms
            }

    def get_interface_implementations(
        self,
        instance_name: str,
        interface_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Find classes that implement abstract base classes/interfaces.

        Args:
            instance_name: RETER instance name
            interface_name: Optional specific interface to find implementations of

        Returns:
            dict with success, implementations list, queries
        """
        start_time = time.time()
        queries = []
        try:
            class_concept = self._concept('Class')
            if interface_name:
                query = f"""
                    SELECT ?class ?className ?base ?baseName
                    WHERE {{
                        ?class type {class_concept} .
                        ?class name ?className .
                        ?class inheritsFrom ?base .
                        ?base name ?baseName .
                        FILTER(CONTAINS(?baseName, "{interface_name}") || CONTAINS(?base, "ABC"))
                    }}
                """
            else:
                query = f"""
                    SELECT ?class ?className ?base ?baseName
                    WHERE {{
                        ?class type {class_concept} .
                        ?class name ?className .
                        ?class inheritsFrom ?base .
                        ?base name ?baseName .
                        FILTER(CONTAINS(?base, "ABC") || CONTAINS(?baseName, "Base") || CONTAINS(?baseName, "Interface"))
                    }}
                """
            queries.append(query.strip())

            result = self.reter.reql(query)
            rows = self._query_to_list(result)

            implementations = [
                {
                    "class": row[0],
                    "class_name": row[1],
                    "interface": row[2],
                    "interface_name": row[3]
                }
                for row in rows
            ]

            time_ms = (time.time() - start_time) * 1000
            return {
                "success": True,
                "implementations": implementations,
                "count": len(implementations),
                "filter": interface_name,
                "queries": queries,
                "time_ms": time_ms
            }
        except Exception as e:
            time_ms = (time.time() - start_time) * 1000
            return {
                "success": False,
                "error": str(e),
                "implementations": [],
                "count": 0,
                "queries": queries,
                "time_ms": time_ms
            }

    def get_public_api(self, instance_name: str) -> Dict[str, Any]:
        """
        Get all public classes and functions (not starting with _).

        Args:
            instance_name: RETER instance name

        Returns:
            dict with success, public_classes, public_functions, queries
        """
        start_time = time.time()
        queries = []
        try:
            class_concept = self._concept('Class')
            func_concept = self._concept('Function')
            # Get public classes
            # Use inFile (works for all languages) instead of inModule
            class_query = f"""
                SELECT ?class ?name ?file
                WHERE {{
                    ?class type {class_concept} .
                    ?class name ?name .
                    ?class inFile ?file .
                    FILTER(!STRSTARTS(?name, "_"))
                }}
                ORDER BY ?file ?name
            """
            queries.append(class_query.strip())

            result = self.reter.reql(class_query)
            class_rows = self._query_to_list(result)

            # Get public functions
            # Use inFile (works for all languages) instead of inModule
            func_query = f"""
                SELECT ?func ?name ?file
                WHERE {{
                    ?func type {func_concept} .
                    ?func name ?name .
                    ?func inFile ?file .
                    FILTER(!STRSTARTS(?name, "_"))
                }}
                ORDER BY ?file ?name
            """
            queries.append(func_query.strip())

            result = self.reter.reql(func_query)
            func_rows = self._query_to_list(result)

            public_classes = [
                {"qualified_name": row[0], "name": row[1], "file": row[2]}
                for row in class_rows
            ]

            public_functions = [
                {"qualified_name": row[0], "name": row[1], "file": row[2]}
                for row in func_rows
            ]

            time_ms = (time.time() - start_time) * 1000
            return {
                "success": True,
                "public_classes": public_classes,
                "public_functions": public_functions,
                "class_count": len(public_classes),
                "function_count": len(public_functions),
                "queries": queries,
                "time_ms": time_ms
            }
        except Exception as e:
            time_ms = (time.time() - start_time) * 1000
            return {
                "success": False,
                "error": str(e),
                "public_classes": [],
                "public_functions": [],
                "queries": queries,
                "time_ms": time_ms
            }
