"""
Tool Base Classes

Defines the core interfaces for tool implementations:
- ToolMetadata: Metadata describing a tool module
- ToolDefinition: Definition of an MCP tool
- BaseTool: Base class for all tool implementations
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Callable, Optional
from dataclasses import dataclass, field


@dataclass
class ToolMetadata:
    """
    Metadata describing a tool module.

    Attributes:
        name: Unique identifier (e.g., "python_basic")
        version: Semantic version (e.g., "1.0.0")
        description: Human-readable description
        author: Author/maintainer
        requires_reter: Whether tool needs RETER instance access
        categories: Category tags for organization
    """
    name: str
    version: str
    description: str
    author: str
    requires_reter: bool = True
    dependencies: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)


# Alias for backwards compatibility
PluginMetadata = ToolMetadata


@dataclass
class ToolDefinition:
    """
    Definition of an MCP tool.

    Attributes:
        name: Tool name (e.g., "list_modules")
        description: Human-readable description
        handler: Callable that implements the tool
        parameters_schema: JSON Schema defining tool parameters
        return_schema: JSON Schema defining return value structure
        examples: Example invocations for documentation
    """
    name: str
    description: str
    handler: Callable
    parameters_schema: Dict[str, Any]
    return_schema: Optional[Dict[str, Any]] = None
    examples: List[Dict[str, Any]] = field(default_factory=list)


class BaseTool(ABC):
    """
    Base class for all tool implementations.

    Each tool module provides a collection of related MCP tools that
    operate on RETER instances for reasoning and knowledge extraction.
    """

    def __init__(self, instance_manager):
        """
        Initialize the tool with access to RETER instances.

        Args:
            instance_manager: InstanceManager for accessing/creating RETER instances
        """
        self.instance_manager = instance_manager
        self._config: Dict[str, Any] = {}
        self._ontology_loaded: Dict[str, bool] = {}  # Track ontology loading per instance

    @property
    def config(self) -> Dict[str, Any]:
        """Get tool configuration."""
        return self._config

    def is_ontology_loaded(self, instance_name: str) -> bool:
        """Check if ontology has been loaded for an instance."""
        return self._ontology_loaded.get(instance_name, False)

    def mark_ontology_loaded(self, instance_name: str) -> None:
        """Mark ontology as loaded for an instance."""
        self._ontology_loaded[instance_name] = True

    @abstractmethod
    def get_metadata(self) -> ToolMetadata:
        """Return metadata describing this tool module."""
        pass

    @abstractmethod
    def get_tools(self) -> List[ToolDefinition]:
        """Return list of MCP tools provided by this module."""
        pass

    def initialize(self) -> None:
        """Initialize the tool (called after instantiation)."""
        pass

    def shutdown(self) -> None:
        """Shutdown the tool (called before unloading)."""
        pass


# Alias for backwards compatibility
AnalysisPlugin = BaseTool
