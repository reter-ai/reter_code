"""
Code Quality Analysis Tools

Provides code quality metrics like finding large classes and long parameter lists.
"""

from typing import Dict, Any, List
import time
from .base import AdvancedToolsBase


class CodeQualityTools(AdvancedToolsBase):
    """Code quality analysis tools."""

    # Default patterns to exclude from refactoring analysis
    DEFAULT_EXCLUDE_PATTERNS = [
        # Test files and classes
        "**/test_*", "**/*_test.py", "**/tests/**",
        "Test*",  # Test class names
        # Known large-by-design patterns
        "*Visitor*", "*FactExtraction*",  # AST visitors are expected to be large
        "*ParserBase*", "*LexerBase*",  # Parser/Lexer base classes
    ]

    def find_large_classes(
        self,
        instance_name: str,
        threshold: int = 20,
        exclude_patterns: List[str] = None,
        exclude_test_files: bool = True
    ) -> Dict[str, Any]:
        """
        Find classes with too many methods (God classes).

        Args:
            instance_name: RETER instance name
            threshold: Minimum number of methods (default: 20)
            exclude_patterns: Glob patterns to exclude (files/classes)
            exclude_test_files: Exclude test files and classes (default: True)

        Returns:
            dict with success, classes list, count, queries
        """
        import fnmatch
        start_time = time.time()
        queries = []

        # Build exclusion patterns
        patterns = list(exclude_patterns or [])
        if exclude_test_files:
            patterns.extend([
                "**/test_*", "**/*_test.py", "**/tests/**",
                "Test*", "*Test", "*Tests"
            ])

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

            classes = []
            excluded_count = 0
            for row in rows:
                qualified_name = row[0]
                name = row[1]
                file_path = row[2]
                method_count = int(row[3]) if row[3] else 0

                # Check exclusion patterns
                excluded = False
                for pattern in patterns:
                    if fnmatch.fnmatch(file_path, pattern) or fnmatch.fnmatch(name, pattern):
                        excluded = True
                        excluded_count += 1
                        break

                if not excluded:
                    classes.append({
                        "qualified_name": qualified_name,
                        "name": name,
                        "file": file_path,
                        "method_count": method_count
                    })

            time_ms = (time.time() - start_time) * 1000
            return {
                "success": True,
                "classes": classes,
                "count": len(classes),
                "excluded_count": excluded_count,
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
        threshold: int = 5,
        exclude_patterns: List[str] = None,
        exclude_test_files: bool = True
    ) -> Dict[str, Any]:
        """
        Find functions/methods with too many parameters.

        Args:
            instance_name: RETER instance name
            threshold: Maximum acceptable parameter count (default: 5)
            exclude_patterns: Glob patterns to exclude (files/functions)
            exclude_test_files: Exclude test files and functions (default: True)

        Returns:
            dict with success, functions list, count, queries
        """
        import fnmatch
        start_time = time.time()
        queries = []

        # Build exclusion patterns
        patterns = list(exclude_patterns or [])
        if exclude_test_files:
            patterns.extend([
                "**/test_*", "**/*_test.py", "**/tests/**",
                "test_*", "Test*"
            ])

        try:
            param_concept = self._concept('Parameter')
            func_concept = self._concept('Function')
            method_concept = self._concept('Method')
            query = f"""
                SELECT ?func ?name ?type ?file (COUNT(?param) AS ?param_count)
                WHERE {{
                    ?func type ?type .
                    ?func name ?name .
                    ?func inFile ?file .
                    ?param type {param_concept} .
                    ?param ofFunction ?func .
                    FILTER(?type = {func_concept} || ?type = {method_concept})
                }}
                GROUP BY ?func ?name ?type ?file
                HAVING (?param_count > {threshold})
                ORDER BY DESC(?param_count)
            """
            queries.append(query.strip())

            result = self.reter.reql(query)
            rows = self._query_to_list(result)

            functions = []
            excluded_count = 0
            for row in rows:
                qualified_name = row[0]
                name = row[1]
                func_type = row[2]
                file_path = row[3] if len(row) > 3 else ""
                param_count = int(row[4]) if len(row) > 4 and row[4] else 0

                # Check exclusion patterns
                excluded = False
                for pattern in patterns:
                    if (file_path and fnmatch.fnmatch(file_path, pattern)) or fnmatch.fnmatch(name, pattern):
                        excluded = True
                        excluded_count += 1
                        break

                if not excluded:
                    functions.append({
                        "qualified_name": qualified_name,
                        "name": name,
                        "type": func_type,
                        "file": file_path,
                        "parameter_count": param_count
                    })

            time_ms = (time.time() - start_time) * 1000
            return {
                "success": True,
                "functions": functions,
                "count": len(functions),
                "excluded_count": excluded_count,
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
