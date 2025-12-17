"""Base class for UML diagram generators."""

from typing import List, Any
from codeine.services.language_support import LanguageSupport, LanguageType


class UMLGeneratorBase:
    """Base class with common functionality for UML generators."""

    def __init__(self, instance_manager, language: LanguageType = "oo"):
        """Initialize with instance manager.

        Args:
            instance_manager: RETER instance manager for accessing knowledge bases
            language: Programming language to analyze ("oo", "python", "javascript", "html")
        """
        self.instance_manager = instance_manager
        self.language = language
        self._lang = LanguageSupport

    def _concept(self, entity: str) -> str:
        """Build concept string for current language (e.g., 'py:Class' or 'oo:Class')."""
        return self._lang.concept(entity, self.language)

    def _relation(self, rel: str) -> str:
        """Build relation string for current language (e.g., 'py:inheritsFrom')."""
        return self._lang.relation(rel, self.language)

    def _query_to_list(self, result) -> List[tuple]:
        """Convert PyArrow Table to list of tuples.

        Args:
            result: PyArrow Table from REQL query

        Returns:
            List of tuples, one per row
        """
        if result.num_rows == 0:
            return []

        columns = [result.column(name).to_pylist() for name in result.column_names]
        return list(zip(*columns))

    def _result_to_pylist(self, result: Any) -> List[dict]:
        """Convert query result to list of dicts.

        Args:
            result: PyArrow table or similar result

        Returns:
            List of row dicts
        """
        if hasattr(result, 'to_pylist'):
            return result.to_pylist()
        return []
