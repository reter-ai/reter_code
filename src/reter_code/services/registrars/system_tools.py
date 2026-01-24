"""
System Tools Registrar

Unified system management tool that consolidates:
- instance_manager (list, list_sources, get_facts, forget, reload, check)
- init_status
- rag_status
- rag_reindex
- reter_info
- initialize_project

All functionality is now available via a single `system` tool.
"""

from typing import Dict, Any, Optional
from fastmcp import FastMCP
from .base import ToolRegistrarBase
from ..initialization_progress import (
    get_instance_progress,
    get_component_readiness,
    require_default_instance,
    ComponentNotReadyError,
)
from ...reter_wrapper import DefaultInstanceNotInitialised


class SystemToolsRegistrar(ToolRegistrarBase):
    """Registers the unified system management tool."""

    def __init__(self, instance_manager, persistence_service, default_manager=None, reter_ops=None):
        super().__init__(instance_manager, persistence_service)
        self._default_manager = default_manager
        self._reter_ops = reter_ops

    def _get_rag_manager(self):
        """Get the RAG manager from the default instance manager."""
        if self._default_manager is None:
            return None
        return self._default_manager.get_rag_manager()

    def register(self, app: FastMCP) -> None:
        """Register the system tool."""
        import asyncio
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
            - reload: Reload modified source files (incremental)
            - check: Consistency check of knowledge base
            - initialize: Full re-initialization (reloads everything)
            - reindex: RAG index rebuild (use `force=True` for full rebuild)
            - reset_parser: Force CADSL grammar reload (after grammar.lark changes)

            Args:
                action: One of: status, info, sources, facts, forget, reload, check, initialize, reindex, reset_parser
                source: Source ID or file path (required for facts, forget)
                force: Force full rebuild for reindex action (default: False)

            Returns:
                Action-specific results with success status

            Examples:
                system("status")                    # Comprehensive status
                system("info")                      # Version info
                system("sources")                   # List loaded files
                system("facts", "path/to/file.py")  # Get facts for file
                system("forget", "path/to/file.py") # Forget a file
                system("reload")                    # Reload modified files
                system("check")                     # Consistency check
                system("initialize")                # Full re-init
                system("reindex", force=True)       # Force RAG rebuild
                system("reset_parser")              # Reload CADSL grammar
            """
            # ═══════════════════════════════════════════════════════════════════
            # STATUS - Combined init + RAG status (always available)
            # ═══════════════════════════════════════════════════════════════════
            if action == "status":
                return registrar._action_status()

            # ═══════════════════════════════════════════════════════════════════
            # INFO - Version and diagnostic info (always available)
            # ═══════════════════════════════════════════════════════════════════
            elif action == "info":
                return registrar._action_info()

            # ═══════════════════════════════════════════════════════════════════
            # SOURCES - List loaded sources (requires RETER ready)
            # ═══════════════════════════════════════════════════════════════════
            elif action == "sources":
                try:
                    require_default_instance()
                except ComponentNotReadyError as e:
                    return e.to_response()
                result = registrar.persistence.list_sources("default")
                result["action"] = "sources"
                return result

            # ═══════════════════════════════════════════════════════════════════
            # FACTS - Get facts for a source (requires RETER ready)
            # ═══════════════════════════════════════════════════════════════════
            elif action == "facts":
                if not source:
                    return {"success": False, "error": "source parameter required for facts action"}
                try:
                    require_default_instance()
                except ComponentNotReadyError as e:
                    return e.to_response()
                result = registrar.persistence.get_source_facts("default", source)
                result["action"] = "facts"
                return result

            # ═══════════════════════════════════════════════════════════════════
            # FORGET - Remove facts from a source (requires RETER ready)
            # ═══════════════════════════════════════════════════════════════════
            elif action == "forget":
                if not source:
                    return {"success": False, "error": "source parameter required for forget action"}
                try:
                    require_default_instance()
                except ComponentNotReadyError as e:
                    return e.to_response()
                result = registrar._reter_ops.forget_source("default", source)
                result["action"] = "forget"
                return result

            # ═══════════════════════════════════════════════════════════════════
            # RELOAD - Reload modified sources (requires RETER ready)
            # ═══════════════════════════════════════════════════════════════════
            elif action == "reload":
                try:
                    require_default_instance()
                except ComponentNotReadyError as e:
                    return e.to_response()
                result = registrar._reter_ops.reload_sources("default")
                result["action"] = "reload"
                return result

            # ═══════════════════════════════════════════════════════════════════
            # CHECK - Consistency check (requires RETER ready)
            # ═══════════════════════════════════════════════════════════════════
            elif action == "check":
                try:
                    require_default_instance()
                except ComponentNotReadyError as e:
                    return e.to_response()
                result = registrar._reter_ops.check_consistency("default")
                result["action"] = "check"
                return result

            # ═══════════════════════════════════════════════════════════════════
            # INITIALIZE - Full re-initialization
            # ═══════════════════════════════════════════════════════════════════
            elif action == "initialize":
                return await registrar._action_initialize()

            # ═══════════════════════════════════════════════════════════════════
            # REINDEX - RAG index rebuild
            # ═══════════════════════════════════════════════════════════════════
            elif action == "reindex":
                return await registrar._action_reindex(force=force)

            # ═══════════════════════════════════════════════════════════════════
            # RESET_PARSER - Force CADSL grammar reload
            # ═══════════════════════════════════════════════════════════════════
            elif action == "reset_parser":
                return registrar._action_reset_parser()

            # ═══════════════════════════════════════════════════════════════════
            # UNKNOWN ACTION
            # ═══════════════════════════════════════════════════════════════════
            else:
                return {
                    "success": False,
                    "error": f"Unknown action: {action}",
                    "available_actions": [
                        "status", "info", "sources", "facts", "forget",
                        "reload", "check", "initialize", "reindex", "reset_parser"
                    ]
                }

    def _action_status(self) -> Dict[str, Any]:
        """Combined init + RAG status."""
        # Get component readiness
        components = get_component_readiness()
        component_status = components.get_status()
        is_ready = components.is_fully_ready()

        # Build blocking reason
        blocking = None
        if not is_ready:
            if not components.sql_ready:
                blocking = {"component": "sql", "message": "SQLite is not initialized yet."}
            elif not components.reter_ready:
                blocking = {"component": "reter", "message": "RETER is still loading Python files."}
            elif not components.rag_code_ready:
                blocking = {"component": "rag_code", "message": "RAG code index is still building."}
            elif not components.rag_docs_ready:
                blocking = {"component": "rag_docs", "message": "RAG document index is still indexing."}
            else:
                blocking = {"component": "unknown", "message": "Server is still initializing."}

        # Get progress info
        progress = get_instance_progress()

        # Get RAG status if available
        rag_info = {}
        rag_manager = self._get_rag_manager()
        if rag_manager is not None:
            try:
                rag_status = rag_manager.get_status()
                rag_info = {
                    "rag_enabled": True,
                    "rag_status": rag_status.get("status", "unknown"),
                    "total_vectors": rag_status.get("total_vectors", 0),
                    "embedding_model": rag_status.get("embedding_model", "unknown"),
                }
            except Exception:
                rag_info = {"rag_enabled": True, "rag_status": "error"}
        else:
            rag_info = {"rag_enabled": False}

        return {
            "success": True,
            "action": "status",
            "is_ready": is_ready,
            "blocking_reason": blocking,
            "components": component_status,
            "init": progress.get_init_snapshot(),
            "sync": progress.get_sync_snapshot(),
            **rag_info,
        }

    def _action_info(self) -> Dict[str, Any]:
        """Version and diagnostic information."""
        import reter
        from reter import owl_rete_cpp

        # Get MCP server version
        try:
            from reter_code import __version__ as mcp_version
        except ImportError:
            mcp_version = "unknown"

        # Get reter Python package version
        try:
            reter_version = reter.__version__
        except AttributeError:
            reter_version = "unknown"

        # Get C++ binding version info
        try:
            cpp_version = getattr(owl_rete_cpp, "__version__", "unknown")
            cpp_build_timestamp = getattr(owl_rete_cpp, "__build_timestamp__", "unknown")
            cpp_info = owl_rete_cpp.get_version_info() if hasattr(owl_rete_cpp, "get_version_info") else {}
        except Exception as e:
            cpp_version = f"error: {e}"
            cpp_build_timestamp = "unknown"
            cpp_info = {}

        # Get module locations
        try:
            reter_location = reter.__file__
        except AttributeError:
            reter_location = "unknown"

        try:
            cpp_location = owl_rete_cpp.__file__
        except AttributeError:
            cpp_location = "unknown"

        return {
            "success": True,
            "action": "info",
            "mcp_server": {
                "name": "reter",
                "version": mcp_version,
            },
            "reter_package": {
                "version": reter_version,
                "location": reter_location,
            },
            "cpp_binding": {
                "version": cpp_version,
                "build_timestamp": cpp_build_timestamp,
                "location": cpp_location,
                "info": cpp_info,
            },
        }

    async def _action_initialize(self) -> Dict[str, Any]:
        """Full re-initialization."""
        import asyncio

        if self._default_manager is None:
            return {
                "success": False,
                "action": "initialize",
                "error": "Default manager not configured.",
            }

        if not self._default_manager.is_configured():
            return {
                "success": False,
                "action": "initialize",
                "error": "No project root configured. Set RETER_PROJECT_ROOT.",
            }

        try:
            # Run initialization in thread pool
            stats = await asyncio.to_thread(self._default_manager.initialize)
            return {
                "success": True,
                "action": "initialize",
                **stats,
            }
        except Exception as e:
            return {
                "success": False,
                "action": "initialize",
                "error": str(e),
            }

    def _action_reset_parser(self) -> Dict[str, Any]:
        """Reset the CADSL parser to force grammar reload."""
        try:
            import importlib
            from ...cadsl import parser, compiler, transformer, loader

            # Reload the modules to pick up code changes
            # Order matters: compiler first, then transformer (uses compiler),
            # then loader (uses transformer), then parser
            importlib.reload(compiler)
            importlib.reload(transformer)
            importlib.reload(loader)
            importlib.reload(parser)

            # Reset parser cache and force re-initialization
            from ...cadsl.parser import CADSLParser
            CADSLParser.reset()
            CADSLParser()

            return {
                "success": True,
                "action": "reset_parser",
                "message": "CADSL parser, compiler, transformer, and loader modules reloaded.",
            }
        except Exception as e:
            return {
                "success": False,
                "action": "reset_parser",
                "error": str(e),
            }

    async def _action_reindex(self, force: bool = False) -> Dict[str, Any]:
        """RAG index rebuild."""
        import asyncio

        try:
            require_default_instance()
        except ComponentNotReadyError as e:
            return e.to_response()

        rag_manager = self._get_rag_manager()

        if rag_manager is None:
            return {
                "success": False,
                "action": "reindex",
                "error": "RAG is not configured. Set RETER_PROJECT_ROOT to enable.",
            }

        if not rag_manager.is_enabled:
            return {
                "success": False,
                "action": "reindex",
                "error": "RAG is disabled via configuration.",
            }

        try:
            reter = self.instance_manager.get_or_create_instance("default")
            project_root = self._default_manager.project_root

            if project_root is None:
                return {
                    "success": False,
                    "action": "reindex",
                    "error": "Project root is not set.",
                }

            # Run blocking reindex in thread pool
            stats = await asyncio.to_thread(rag_manager.reindex_all, reter, project_root)
            return {
                "success": True,
                "action": "reindex",
                "force": force,
                **stats,
            }
        except DefaultInstanceNotInitialised as e:
            return {
                "success": False,
                "action": "reindex",
                "error": str(e),
                "status": "initializing",
            }
        except Exception as e:
            return {
                "success": False,
                "action": "reindex",
                "error": str(e),
            }
