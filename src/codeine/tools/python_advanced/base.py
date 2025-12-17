"""
Base class for advanced code analysis tools.

Supports multiple languages via the LanguageSupport module:
- "oo" (default): Language-independent queries (Python + JavaScript)
- "python" or "py": Python-specific queries
- "javascript" or "js": JavaScript-specific queries
- "html" or "htm": HTML document queries
"""

from typing import List
from codeine.services.language_support import LanguageSupport, LanguageType


class AdvancedToolsBase:
    """Base class with common functionality for language-aware code analysis."""

    def __init__(self, reter_wrapper, language: LanguageType = "oo"):
        """
        Initialize with ReterWrapper instance.

        Args:
            reter_wrapper: ReterWrapper instance with loaded code
            language: Programming language to analyze ("oo", "python", "javascript", "html")
        """
        self.reter = reter_wrapper
        self.language = language
        self._lang = LanguageSupport

    def _concept(self, entity: str) -> str:
        """Build concept string for current language (e.g., 'py:Class' or 'oo:Class')."""
        return self._lang.concept(entity, self.language)

    def _relation(self, rel: str) -> str:
        """Build relation string for current language (e.g., 'py:inheritsFrom')."""
        return self._lang.relation(rel, self.language)

    def _prefix(self) -> str:
        """Get the current language prefix."""
        return self._lang.get_prefix(self.language)

    def _query_to_list(self, result) -> List[tuple]:
        """
        Convert PyArrow Table to list of tuples.

        Args:
            result: PyArrow Table from REQL query

        Returns:
            List of tuples, one per row
        """
        if result.num_rows == 0:
            return []

        columns = [result.column(name).to_pylist() for name in result.column_names]
        return list(zip(*columns))

    def _query_to_list_padded(self, result, expected_columns: int) -> List[tuple]:
        """
        Convert PyArrow Table to list of tuples, padding with None for missing OPTIONAL columns.

        REQL OPTIONAL clauses may not return columns when they don't match.
        This method ensures consistent tuple length for unpacking.

        Args:
            result: PyArrow Table from REQL query
            expected_columns: Number of columns expected in SELECT clause

        Returns:
            List of tuples, each padded to expected_columns length with None
        """
        if result.num_rows == 0:
            return []

        columns = [result.column(name).to_pylist() for name in result.column_names]
        rows = list(zip(*columns))

        # Pad rows if fewer columns than expected (OPTIONAL columns missing)
        actual_columns = len(result.column_names)
        if actual_columns < expected_columns:
            padding = tuple([None] * (expected_columns - actual_columns))
            rows = [row + padding for row in rows]

        return rows
