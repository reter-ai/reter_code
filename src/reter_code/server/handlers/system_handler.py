"""
RETER System Handler.

Handles system operations (status, info, health checks).

::: This is-in-layer Handler-Layer.
::: This is-in-component System-Handlers.
"""

import time
from typing import Any, Dict

from . import BaseHandler
from ..protocol import METHOD_SYSTEM, METHOD_STATUS, METHOD_INFO, METHOD_VIEW_PUSH


class SystemHandler(BaseHandler):
    """Handler for system operations (status, info, health).

    ::: This is-in-layer Service-Layer.
    ::: This is a handler.
    ::: This is stateful.
    """

    def _register_methods(self) -> None:
        """Register system method handlers."""
        self._methods = {
            METHOD_SYSTEM: self._handle_system,
            METHOD_STATUS: self._handle_status,
            METHOD_INFO: self._handle_info,
            METHOD_VIEW_PUSH: self._handle_view_push,
            "health": self._handle_health,
            "sources": self._handle_sources,
            "facts": self._handle_facts,
        }


    def _handle_system(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatch system action.

        Params:
            action: System action (status, info, sources, facts, forget, reload, etc.)
            source: Source ID for facts/forget actions
            force: Force flag for certain actions

        Returns:
            Action-specific results
        """
        action = params.get("action", "status")

        action_handlers = {
            "status": lambda: self._handle_status(params),
            "info": lambda: self._handle_info(params),
            "sources": lambda: self._handle_sources(params),
            "facts": lambda: self._handle_facts(params),
            "health": lambda: self._handle_health(params),
            "check": lambda: self._handle_consistency_check(params),
            "forget": lambda: self._handle_forget(params),
        }

        handler = action_handlers.get(action)
        if handler:
            return handler()
        else:
            raise ValueError(f"Unknown system action: {action}")

    def _handle_status(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get comprehensive system status.

        Returns:
            Dictionary with RETER and RAG status
        """
        # Get RETER status
        sources, _ = self.reter.get_all_sources()
        stats = self.reter._session_stats

        # Get RAG status
        rag_status = {}
        try:
            rag_status = self.rag_manager.get_status()
        except Exception:
            rag_status = {"initialized": False}

        return {
            "success": True,
            "reter": {
                "initialized": True,
                "total_sources": len(sources),
                "total_wmes": stats.get("total_wmes", 0),
            },
            "rag": rag_status,
            "uptime_seconds": time.time() - getattr(self, "_start_time", time.time())
        }

    def _handle_info(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get system version and diagnostic info.

        Returns:
            Dictionary with version and environment info
        """
        import sys
        import platform

        return {
            "success": True,
            "version": "0.1.0",
            "python_version": sys.version,
            "platform": platform.platform(),
            "reter_variant": "ai",
        }

    def _handle_health(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Health check for load balancers/monitoring.

        Returns:
            Dictionary with health status
        """
        try:
            # Quick check - can we execute a simple query?
            self.reter.reql("SELECT (1 AS ?one) WHERE {}", timeout_ms=1000)
            healthy = True
            message = "OK"
        except Exception as e:
            healthy = False
            message = str(e)

        return {
            "success": True,
            "healthy": healthy,
            "message": message,
            "timestamp": time.time()
        }

    def _handle_sources(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """List all loaded sources.

        Returns:
            Dictionary with source list
        """
        sources, time_ms = self.reter.get_all_sources()

        return {
            "success": True,
            "sources": sources,
            "count": len(sources),
            "execution_time_ms": time_ms
        }

    def _handle_facts(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get facts for a specific source.

        Params:
            source: Source ID to get facts for

        Returns:
            Dictionary with fact IDs
        """
        source = params.get("source", "")

        if not source:
            raise ValueError("Source ID is required")

        fact_ids, source_id, time_ms = self.reter.get_facts_from_source(source)

        return {
            "success": True,
            "source": source_id,
            "facts": fact_ids,
            "count": len(fact_ids),
            "execution_time_ms": time_ms
        }

    def _handle_consistency_check(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Run consistency check on knowledge base.

        Returns:
            Dictionary with check results
        """
        # Basic consistency checks
        issues = []

        try:
            sources, _ = self.reter.get_all_sources()

            # Check for orphaned sources
            for source in sources:
                facts, _, _ = self.reter.get_facts_from_source(source)
                if len(facts) == 0:
                    issues.append({
                        "type": "empty_source",
                        "source": source,
                        "message": "Source has no facts"
                    })

        except Exception as e:
            issues.append({
                "type": "check_error",
                "message": str(e)
            })

        return {
            "success": True,
            "consistent": len(issues) == 0,
            "issues": issues,
            "issues_count": len(issues)
        }


    def _handle_view_push(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Push content to connected browser viewers.

        Params:
            content_type: "markdown", "mermaid", or "html"
            content: Content string to render

        Returns:
            Dictionary with success status
        """
        content_type = params.get("content_type", "markdown")
        content = params.get("content", "")

        if not content:
            raise ValueError("content is required")

        self.push_view(content_type, content)

        return {
            "success": True,
            "content_type": content_type,
            "content_length": len(content),
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


__all__ = ["SystemHandler"]
