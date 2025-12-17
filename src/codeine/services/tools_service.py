"""
Tools Registration

Direct MCP tool registration for all RETER tools.
Refactored to use separate registrar classes for better maintainability.

Consolidation Updates:
- Phase 7: Removed GanttToolsRegistrar, RecommendationsToolsRegistrar,
  RequirementsToolsRegistrar -> replaced by UnifiedToolsRegistrar
- Code Inspection: Removed PythonBasicToolsRegistrar, PythonAdvancedToolsRegistrar,
  UMLToolsRegistrar -> replaced by CodeInspectionToolsRegistrar
- Recommender: Renamed RefactoringToolsRegistrar -> RecommenderToolsRegistrar
- RAG: Added RAGToolsRegistrar for semantic search over code and documentation
"""

from typing import Optional, TYPE_CHECKING
from fastmcp import FastMCP

from .registrars import (
    CodeInspectionToolsRegistrar,
    RecommenderToolsRegistrar,
    UnifiedToolsRegistrar,
    RAGToolsRegistrar,
)
from .registrars.base import ToolRegistrarBase

if TYPE_CHECKING:
    from .default_instance_manager import DefaultInstanceManager


class ToolsRegistrar(ToolRegistrarBase):
    """
    Registers all RETER tools directly with FastMCP.

    Delegates to specialized registrar classes for each tool category.
    Extends ToolRegistrarBase for common functionality (_get_reter,
    _save_snapshot, _ensure_ontology_loaded).
    """

    def __init__(
        self,
        instance_manager,
        persistence_service,
        default_manager: Optional["DefaultInstanceManager"] = None
    ):
        """
        Initialize the tools registrar.

        Args:
            instance_manager: RETER instance manager
            persistence_service: State persistence service
            default_manager: Default instance manager (for RAG integration)
        """
        super().__init__(instance_manager, persistence_service)
        self.default_manager = default_manager

        # Initialize sub-registrars
        # Consolidated: CodeInspectionToolsRegistrar replaces Python and UML tools
        # UnifiedToolsRegistrar replaces Session, Gantt, Recommendations, Requirements
        # RecommenderToolsRegistrar provides unified recommender("type", "detector") interface
        # RAGToolsRegistrar provides semantic search over code and documentation
        self._registrars = [
            CodeInspectionToolsRegistrar(instance_manager, persistence_service),
            RecommenderToolsRegistrar(instance_manager, persistence_service, default_manager),
            UnifiedToolsRegistrar(instance_manager, persistence_service),
            RAGToolsRegistrar(instance_manager, persistence_service, default_manager),
        ]

    def register(self, app: FastMCP) -> None:
        """
        Register all tools with the FastMCP application.

        Implements the abstract method from ToolRegistrarBase.

        Args:
            app: FastMCP application instance
        """
        for registrar in self._registrars:
            registrar.register(app)

    def register_all_tools(self, app: FastMCP) -> None:
        """
        Register all tools with the FastMCP application.

        Alias for register() for backwards compatibility.

        Args:
            app: FastMCP application instance
        """
        self.register(app)
