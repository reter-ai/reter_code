"""
Tool Registrars Module

Split tool registration into separate classes for better maintainability.

Consolidation Updates:
- Phase 7: Removed GanttToolsRegistrar, RecommendationsToolsRegistrar,
  RequirementsToolsRegistrar -> replaced by UnifiedToolsRegistrar
- Code Inspection: Removed PythonBasicToolsRegistrar, PythonAdvancedToolsRegistrar,
  UMLToolsRegistrar -> replaced by CodeInspectionToolsRegistrar
- Recommender: Renamed RefactoringToolsRegistrar -> RecommenderToolsRegistrar
- RAG: Added RAGToolsRegistrar for semantic search over code and documentation
"""

from .base import (
    ToolRegistrarBase,
    handle_not_initialised,
    truncate_mcp_response,
    truncate_mcp_response_async,
)
from .code_inspection import CodeInspectionToolsRegistrar
from .refactoring import RecommenderToolsRegistrar
from .unified import UnifiedToolsRegistrar
from .rag_tools import RAGToolsRegistrar

__all__ = [
    'ToolRegistrarBase',
    'handle_not_initialised',
    'truncate_mcp_response',
    'truncate_mcp_response_async',
    'CodeInspectionToolsRegistrar',
    'RecommenderToolsRegistrar',
    'UnifiedToolsRegistrar',
    'RAGToolsRegistrar',
]
