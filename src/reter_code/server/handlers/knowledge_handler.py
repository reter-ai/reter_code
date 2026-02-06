"""
RETER Knowledge Handler.

Handles knowledge loading, forgetting, and reload operations.

::: This is-in-layer Handler-Layer.
::: This is-in-component System-Handlers.
::: This depends-on reter_code.reter_wrapper.
"""

from typing import Any, Dict

from . import BaseHandler
from ..protocol import (
    METHOD_ADD_KNOWLEDGE,
    METHOD_ADD_DIRECTORY,
    METHOD_FORGET,
    METHOD_RELOAD,
    METHOD_VALIDATE_CNL,
    KNOWLEDGE_ERROR,
)


class KnowledgeHandler(BaseHandler):
    """Handler for knowledge operations (add, forget, reload).

    ::: This is-in-layer Service-Layer.
    ::: This is a handler.
    ::: This is stateful.
    """

    def _register_methods(self) -> None:
        """Register knowledge method handlers."""
        self._methods = {
            METHOD_ADD_KNOWLEDGE: self._handle_add_knowledge,
            METHOD_ADD_DIRECTORY: self._handle_add_directory,
            METHOD_FORGET: self._handle_forget,
            METHOD_RELOAD: self._handle_reload,
            METHOD_VALIDATE_CNL: self._handle_validate_cnl,
        }

    def can_handle(self, method: str) -> bool:
        """Check if this handler can process the method."""
        return method in self._methods

    def _handle_add_knowledge(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Add knowledge from source.

        Params:
            source: Ontology content, file path, or code
            type: Source type (ontology, python, javascript, etc.)
            source_id: Optional identifier for selective forgetting

        Returns:
            Dictionary with items_added and execution time
        """
        source = params.get("source", "")
        source_type = params.get("type", "ontology")
        source_id = params.get("source_id")

        if not source:
            raise ValueError("Source is required")

        # Dispatch based on type
        if source_type == "ontology":
            items, time_ms = self.reter.add_ontology(source, source_id=source_id)
        elif source_type == "python":
            items, time_ms = self.reter.load_python_file(source, source_id=source_id)
        elif source_type == "javascript":
            items, time_ms = self.reter.load_javascript_file(source, source_id=source_id)
        elif source_type == "html":
            items, time_ms = self.reter.load_html_file(source, source_id=source_id)
        elif source_type == "csharp":
            items, time_ms = self.reter.load_csharp_file(source, source_id=source_id)
        elif source_type == "cpp":
            items, time_ms = self.reter.load_cpp_file(source, source_id=source_id)
        else:
            raise ValueError(f"Unknown source type: {source_type}")

        return {
            "success": True,
            "items_added": items,
            "execution_time_ms": time_ms,
            "source_id": source_id
        }

    def _handle_add_directory(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Add all code files from a directory.

        Params:
            directory: Path to directory
            recursive: Whether to search subdirectories
            exclude_patterns: List of glob patterns to exclude

        Returns:
            Dictionary with files loaded and total WMEs
        """
        directory = params.get("directory", "")
        recursive = params.get("recursive", True)
        exclude_patterns = params.get("exclude_patterns", [])

        if not directory:
            raise ValueError("Directory is required")

        # Use instance manager to scan directory
        result = self.instance_manager.scan_directory(
            directory,
            recursive=recursive,
            exclude_patterns=exclude_patterns
        )

        return {
            "success": True,
            "files_loaded": result.get("files_loaded", 0),
            "total_files": result.get("total_files", 0),
            "total_wmes": result.get("total_wmes", 0),
            "errors": result.get("errors", []),
            "execution_time_ms": result.get("execution_time_ms", 0)
        }

    def _handle_forget(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Forget knowledge from a source.

        Params:
            source: Source identifier to forget

        Returns:
            Dictionary with source and execution time
        """
        source = params.get("source", "")

        if not source:
            raise ValueError("Source identifier is required")

        source_id, time_ms = self.reter.forget_source(source)

        return {
            "success": True,
            "source": source_id,
            "execution_time_ms": time_ms
        }

    def _handle_reload(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Reload modified source files.

        Returns:
            Dictionary with reload statistics
        """
        # Use instance manager to check for changes and reload
        result = self.instance_manager.sync_changes()

        return {
            "success": True,
            "files_reloaded": result.get("reloaded", 0),
            "files_added": result.get("added", 0),
            "files_removed": result.get("removed", 0),
            "execution_time_ms": result.get("execution_time_ms", 0)
        }

    def _handle_validate_cnl(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a CNL statement without adding it.

        Params:
            statement: CNL statement to validate
            context_entity: Optional entity name for "This" resolution

        Returns:
            Dictionary with validation result and parsed facts
        """
        statement = params.get("statement", "")
        context_entity = params.get("context_entity")

        if not statement:
            raise ValueError("Statement is required")

        # Import CNL parser
        from ...cnl.parser import CNLParser

        parser = CNLParser()
        result = parser.validate(statement, context_entity=context_entity)

        return {
            "success": result.get("success", False),
            "errors": result.get("errors", []),
            "facts": result.get("facts", []),
            "resolved_statement": result.get("resolved_statement")
        }


__all__ = ["KnowledgeHandler"]
