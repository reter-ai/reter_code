"""
Test Analysis Tools

Provides tools for analyzing test files and fixtures in the codebase.
"""

from typing import Dict, Any
import time
from .base import AdvancedToolsBase


class TestAnalysisTools(AdvancedToolsBase):
    """Test file and fixture analysis tools."""

    def find_test_files(self, instance_name: str) -> Dict[str, Any]:
        """
        Find test files based on naming conventions.

        Detects files matching patterns: test_*.py and *_test.py

        Args:
            instance_name: RETER instance name

        Returns:
            dict with success, test files list, count, queries
        """
        start_time = time.time()
        queries = []
        try:
            module_concept = self._concept('Module')
            query = f"""
                SELECT ?module ?file
                WHERE {{
                    ?module type {module_concept} .
                    ?module inFile ?file .
                    FILTER(REGEX(?file, "test_.*\\\\.py$") || REGEX(?file, ".*_test\\\\.py$"))
                }}
                ORDER BY ?file
            """
            queries.append(query.strip())

            result = self.reter.reql(query)
            rows = self._query_to_list(result)

            test_files = [
                {
                    "module": row[0],
                    "file": row[1]
                }
                for row in rows
            ]

            time_ms = (time.time() - start_time) * 1000
            return {
                "success": True,
                "test_files": test_files,
                "count": len(test_files),
                "queries": queries,
                "time_ms": time_ms
            }
        except Exception as e:
            time_ms = (time.time() - start_time) * 1000
            return {
                "success": False,
                "error": str(e),
                "test_files": [],
                "count": 0,
                "queries": queries,
                "time_ms": time_ms
            }

    def find_test_fixtures(self, instance_name: str) -> Dict[str, Any]:
        """
        Find pytest fixtures (@pytest.fixture decorator).

        Args:
            instance_name: RETER instance name

        Returns:
            dict with success, fixtures list, count, queries
        """
        start_time = time.time()
        queries = []
        try:
            func_concept = self._concept('Function')
            # Use inFile (works for all languages) instead of inModule
            query = f"""
                SELECT ?func ?name ?file
                WHERE {{
                    ?func type {func_concept} .
                    ?func hasDecorator ?decorator .
                    ?func name ?name .
                    ?func inFile ?file .
                    FILTER(REGEX(?decorator, "pytest\\\\.fixture"))
                }}
                ORDER BY ?file ?name
            """
            queries.append(query.strip())

            result = self.reter.reql(query)
            rows = self._query_to_list(result)

            fixtures = [
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
                "fixtures": fixtures,
                "count": len(fixtures),
                "queries": queries,
                "time_ms": time_ms
            }
        except Exception as e:
            time_ms = (time.time() - start_time) * 1000
            return {
                "success": False,
                "error": str(e),
                "fixtures": [],
                "count": 0,
                "queries": queries,
                "time_ms": time_ms
            }
