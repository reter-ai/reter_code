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
            import time as _time
            from ...reter_utils import safe_cpp_call
            start = _time.time()
            items = safe_cpp_call(
                self.reter.reasoner.load_cnl, source, source_id or ""
            )
            time_ms = (_time.time() - start) * 1000
        elif source_type == "python":
            items, _source, time_ms, _errors = self.reter.load_python_file(source)
        elif source_type == "javascript":
            items, _source, time_ms, _errors = self.reter.load_javascript_file(source)
        elif source_type == "html":
            items, _source, time_ms, _errors = self.reter.load_html_file(source)
        elif source_type == "csharp":
            items, _source, time_ms, _errors = self.reter.load_csharp_file(source)
        elif source_type == "cpp":
            items, _source, time_ms, _errors = self.reter.load_cpp_file(source)
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

        Scans the directory for supported language files and loads them all.
        Auto-detects languages by file extension.

        Params:
            directory: Path to directory
            recursive: Whether to search subdirectories
            exclude_patterns: List of glob patterns to exclude

        Returns:
            Dictionary with files loaded and total WMEs
        """
        directory = params.get("directory", "")
        recursive = params.get("recursive", True)
        exclude_patterns = params.get("exclude_patterns") or []

        if not directory:
            raise ValueError("Directory is required")

        import time as _time
        from pathlib import Path
        start = _time.time()

        # Load each supported language from the directory
        from ...reter_loaders import LANGUAGE_CONFIGS
        total_wmes = 0
        total_files = 0
        all_errors = []

        for lang_name in LANGUAGE_CONFIGS:
            loader_method = getattr(self.reter, f"load_{lang_name}_directory", None)
            if loader_method is None:
                continue
            try:
                wmes, errors_dict, lang_time = loader_method(
                    directory,
                    recursive=recursive,
                    exclude_patterns=exclude_patterns,
                )
                total_wmes += wmes
                for src, errs in errors_dict.items():
                    total_files += 1
                    if errs:
                        all_errors.extend(errs)
                    else:
                        total_files += 0  # counted above
            except Exception as e:
                all_errors.append(f"{lang_name}: {e}")

        time_ms = (_time.time() - start) * 1000
        return {
            "success": True,
            "files_loaded": total_files,
            "total_files": total_files,
            "total_wmes": total_wmes,
            "errors": all_errors,
            "execution_time_ms": time_ms
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

    def _handle_validate_cnl(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a CNL statement without adding it.

        Uses the C++ CNL parser (owl_rete_cpp.parse_cnl) to parse and
        validate the statement without modifying the knowledge base.

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

        # Resolve "This" references if context_entity provided
        resolved = statement
        if context_entity:
            resolved = statement.replace("This", context_entity)

        from reter import owl_rete_cpp

        result = owl_rete_cpp.parse_cnl(resolved)

        # Convert Fact objects to dicts
        facts = [dict(f) for f in result.facts]

        return {
            "success": result.success,
            "errors": list(result.errors),
            "facts": facts,
            "resolved_statement": resolved
        }


__all__ = ["KnowledgeHandler"]
