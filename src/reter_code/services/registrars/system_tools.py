"""
System Tools Registrar

Unified system management tool that consolidates:
- instance_manager (list, list_sources, get_facts, forget, reload, check)
- init_status
- rag_status
- rag_reindex
- reter_info
- initialize_project

All operations go through ReterClient via ZeroMQ (remote-only mode).
"""

from typing import Dict, Any, Optional, TYPE_CHECKING
from fastmcp import FastMCP
from .base import ToolRegistrarBase

if TYPE_CHECKING:
    from ...server.reter_client import ReterClient


class SystemToolsRegistrar(ToolRegistrarBase):
    """
    Registers the unified system management tool.

    All operations go through ReterClient via ZeroMQ.

    ::: This is-in-layer Service-Layer.
    ::: This is a registrar.
    ::: This is-in-process MCP-Server-Process.
    ::: This is stateless.
    """

    def __init__(
        self,
        instance_manager,
        persistence_service,
        default_manager=None,
        reter_ops=None,
        reter_client: Optional["ReterClient"] = None
    ):
        super().__init__(instance_manager, persistence_service, reter_client=reter_client)

    def register(self, app: FastMCP) -> None:
        """Register the system tool."""
        registrar = self

        @app.tool()
        async def system(
            action: str,
            source: Optional[str] = None,
            force: bool = False
        ) -> Dict[str, Any]:
            """
            Unified system management tool for RETER knowledge base.

            Actions:
            - status: Combined initialization and RAG status
            - info: Version and diagnostic information
            - sources: List all loaded source files
            - facts: Get fact IDs for a source (requires `source`)
            - forget: Remove facts from a source (requires `source`)
            - check: Consistency check of knowledge base

            Args:
                action: One of: status, info, sources, facts, forget, check
                source: Source ID or file path (required for facts, forget)

            Returns:
                Action-specific results with success status

            Examples:
                system("status")                    # Comprehensive status
                system("info")                      # Version info
                system("sources")                   # List loaded files
                system("facts", "path/to/file.py")  # Get facts for file
                system("forget", "path/to/file.py") # Forget a file
                system("check")                     # Consistency check
            """
            if registrar.reter_client is None:
                return {
                    "success": False,
                    "error": "RETER server not connected",
                    "action": action,
                }

            try:
                kwargs = {}
                if source is not None:
                    kwargs["source"] = source
                if force:
                    kwargs["force"] = force
                return registrar.reter_client.system(action, **kwargs)
            except Exception as e:
                return {
                    "success": False,
                    "error": str(e),
                    "action": action,
                }
