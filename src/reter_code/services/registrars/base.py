"""
Base Tool Registrar

Common functionality for all tool registrars.
"""

import sys
from typing import Dict, Any, Optional, Callable, TypeVar
from pathlib import Path
from functools import wraps

from ...reter_wrapper import DefaultInstanceNotInitialised
from ..response_truncation import truncate_response

T = TypeVar('T')


def handle_not_initialised(func: Callable[..., Dict[str, Any]]) -> Callable[..., Dict[str, Any]]:
    """
    Decorator to catch DefaultInstanceNotInitialised and return error response.

    Use this on MCP tool implementations to gracefully handle the case
    when the server is still initializing.

    Example:
        @app.tool()
        @handle_not_initialised
        def my_tool(...) -> Dict[str, Any]:
            ...
    """
    @wraps(func)
    def wrapper(*args, **kwargs) -> Dict[str, Any]:
        try:
            return func(*args, **kwargs)
        except DefaultInstanceNotInitialised as e:
            return {
                "success": False,
                "error": str(e),
                "status": "initializing",
            }
    return wrapper


def truncate_mcp_response(func: Callable[..., Dict[str, Any]]) -> Callable[..., Dict[str, Any]]:
    """
    Decorator to truncate MCP tool responses to fit within size limits.

    Applies truncate_response() to the result of any MCP tool function.
    The size limit is controlled by RETER_MCP_MAX_RESPONSE_SIZE env var.

    Example:
        @app.tool()
        @truncate_mcp_response
        def my_tool(...) -> Dict[str, Any]:
            ...
    """
    @wraps(func)
    def wrapper(*args, **kwargs) -> Dict[str, Any]:
        result = func(*args, **kwargs)
        if isinstance(result, dict):
            return truncate_response(result)
        return result
    return wrapper


def truncate_mcp_response_async(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Async version of truncate_mcp_response decorator.

    Example:
        @app.tool()
        @truncate_mcp_response_async
        async def my_tool(...) -> Dict[str, Any]:
            ...
    """
    @wraps(func)
    async def wrapper(*args, **kwargs) -> Dict[str, Any]:
        result = await func(*args, **kwargs)
        if isinstance(result, dict):
            return truncate_response(result)
        return result
    return wrapper


class ToolRegistrarBase:
    """Base class with common functionality for tool registrars."""

    def __init__(self, instance_manager, persistence_service):
        self.instance_manager = instance_manager
        self.persistence = persistence_service
        self.tools_dir = Path(__file__).parent.parent.parent / "tools"
        self._ontology_loaded: Dict[str, Dict[str, bool]] = {}

    def _get_reter(self, instance_name: str):
        """Get or create a RETER instance."""
        return self.instance_manager.get_or_create_instance(instance_name)

    def _save_snapshot(self, instance_name: str) -> None:
        """Save a snapshot of the RETER instance."""
        try:
            reter_ops = self._get_reter(instance_name)
            if self.persistence is None:
                return

            snapshots_dir = self.persistence.snapshots_dir
            snapshots_dir.mkdir(parents=True, exist_ok=True)

            snapshot_path = snapshots_dir / f".{instance_name}.reter"
            temp_path = snapshots_dir / f".{instance_name}.reter.tmp"

            reter_ops.save_network(str(temp_path))
            if temp_path.exists():
                temp_path.replace(snapshot_path)
        except Exception as e:
            print(f"WARNING: Failed to save snapshot for {instance_name}: {e}", file=sys.stderr)

    def _ensure_ontology_loaded(self, instance_name: str, tool_name: str) -> None:
        """Ensure ontology is loaded for the given tool and instance."""
        if instance_name not in self._ontology_loaded:
            self._ontology_loaded[instance_name] = {}

        if self._ontology_loaded[instance_name].get(tool_name):
            return

        try:
            tool_dir = self.tools_dir / tool_name
            reter_ops = self._get_reter(instance_name)

            # Load ontology
            ontology_path = tool_dir / "ontology" / f"{tool_name}_ontology.reol"
            if ontology_path.exists():
                with open(ontology_path, 'r', encoding='utf-8') as f:
                    reter_ops.add_ontology(ontology=f.read(), source=f"{tool_name}_ontology")

            # Load rules
            rules_path = tool_dir / "rules" / f"{tool_name}_rules.reol"
            if rules_path.exists():
                with open(rules_path, 'r', encoding='utf-8') as f:
                    reter_ops.add_ontology(ontology=f.read(), source=f"{tool_name}_rules")

            self._ontology_loaded[instance_name][tool_name] = True
        except Exception as e:
            self._ontology_loaded[instance_name][tool_name] = True
            print(f"Error loading ontology for {tool_name}: {e}", file=sys.stderr)

    def register(self, app) -> None:
        """Register tools with FastMCP. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement register()")
