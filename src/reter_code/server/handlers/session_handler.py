"""
RETER Session Handler.

Handles session and thinking operations.
Thinking tool provides RETER knowledge operations (assert, query, python_file, forget_source).
Session tool returns a static tool guide.

::: This is-in-layer Handler-Layer.
::: This is-in-component System-Handlers.
"""

from typing import Any, Dict

from . import BaseHandler
from ..protocol import METHOD_SESSION, METHOD_THINKING


class SessionHandler(BaseHandler):
    """Handler for session and thinking operations.

    Stateless — no SQLite, no UnifiedStore, no ThinkingSession.

    ::: This is-in-layer Service-Layer.
    ::: This is a handler.
    ::: This is stateless.
    """

    def _register_methods(self) -> None:
        """Register session method handlers."""
        self._methods = {
            METHOD_SESSION: self._handle_session,
            METHOD_THINKING: self._handle_thinking,
        }


    def _handle_session(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle session lifecycle operations — returns static tool guide."""
        action = params.get("action", "context")
        if action == "context":
            return {
                "success": True,
                "message": "Session context is managed by Claude Code. Use TaskList to see tasks.",
                "tools": {
                    "thinking": "RETER knowledge operations (assert, query, python_file, forget_source)",
                    "reql": "Query the code knowledge graph",
                    "code_inspection": "Explore code structure",
                    "semantic_search": "Find code by meaning",
                    "diagram": "Generate UML/code diagrams",
                }
            }
        elif action in ("start", "end", "clear"):
            return {"success": True, "action": action}
        else:
            raise ValueError(f"Unknown session action: {action}")

    def _handle_thinking(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle thinking/reasoning operations.

        Returns thought metadata and executes RETER knowledge operations if provided.
        No persistence — thoughts are conversation context in Claude Code.
        """
        result = {
            "success": True,
            "thought_number": params.get("thought_number", 1),
            "total_thoughts": params.get("total_thoughts", 1),
            "next_thought_needed": params.get("next_thought_needed", True),
            "thought_type": params.get("thought_type", "reasoning"),
            "section": params.get("section"),
        }

        operations = params.get("operations")
        if operations:
            reter_results = self._execute_reter_ops(operations)
            if reter_results:
                result["reter_operations"] = reter_results

        return result

    def _execute_reter_ops(self, operations: Dict[str, Any]) -> Dict[str, Any]:
        """Execute RETER knowledge operations (assert, query, python_file, forget_source)."""
        from ...reter_utils import safe_cpp_call

        results = {}
        reter = self.reter
        if not reter:
            return results

        if "assert" in operations:
            try:
                items = safe_cpp_call(
                    reter.reasoner.load_cnl, operations["assert"], ""
                )
                results["assert"] = {"success": True, "items_added": items}
            except Exception as e:
                results["assert"] = {"success": False, "error": str(e)}

        if "query" in operations:
            try:
                table = reter.reql(operations["query"])
                # Convert PyArrow table to list of dicts
                if hasattr(table, 'to_pylist'):
                    rows = table.to_pylist()
                elif hasattr(table, 'to_pydict'):
                    d = table.to_pydict()
                    cols = list(d.keys())
                    rows = [{c: d[c][i] for c in cols} for i in range(len(d[cols[0]]))] if cols else []
                else:
                    rows = []
                results["query"] = {"success": True, "count": len(rows), "results": rows}
            except Exception as e:
                results["query"] = {"success": False, "error": str(e)}

        if "python_file" in operations:
            try:
                items, _source, _time_ms, _errors = reter.load_python_file(operations["python_file"])
                results["python_file"] = {"success": True, "items_added": items}
            except Exception as e:
                results["python_file"] = {"success": False, "error": str(e)}

        if "forget_source" in operations:
            try:
                reter.forget_source(source=operations["forget_source"])
                results["forget_source"] = {"success": True}
            except Exception as e:
                results["forget_source"] = {"success": False, "error": str(e)}

        return results


__all__ = ["SessionHandler"]
