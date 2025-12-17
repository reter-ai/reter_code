"""UML diagram generation plugin for RETER logical thinking server.

This plugin provides tools to generate UML diagrams from Python code loaded into RETER:
- Class hierarchy diagrams
- Class diagrams with relationships
- Sequence diagrams
- Dependency graphs
- Call graphs
- Coupling matrices
"""

from typing import List, Dict, Any, Optional

from codeine.tools.base import (
    BaseTool,
    ToolMetadata,
    ToolDefinition,
)

# Import specialized generators
from .class_hierarchy import ClassHierarchyGenerator
from .class_diagram import ClassDiagramGenerator
from .sequence_diagram import SequenceDiagramGenerator
from .dependency_graph import DependencyGraphGenerator
from .call_graph import CallGraphGenerator
from .coupling_matrix import CouplingMatrixGenerator


class UMLTool(BaseTool):
    """UML diagram generation plugin.

    Delegates to specialized generators for each diagram type.
    """

    def __init__(self, instance_manager):
        """Initialize the UML plugin.

        Args:
            instance_manager: RETER instance manager for accessing knowledge bases
        """
        super().__init__(instance_manager)

        # Instantiate specialized generators
        self._hierarchy_gen = ClassHierarchyGenerator(instance_manager)
        self._class_diagram_gen = ClassDiagramGenerator(instance_manager)
        self._sequence_gen = SequenceDiagramGenerator(instance_manager)
        self._dependency_gen = DependencyGraphGenerator(instance_manager)
        self._call_graph_gen = CallGraphGenerator(instance_manager)
        self._coupling_gen = CouplingMatrixGenerator(instance_manager)

    def get_metadata(self) -> ToolMetadata:
        """Get plugin metadata."""
        return ToolMetadata(
            name="uml",
            version="1.0.0",
            description="Generate UML diagrams (class hierarchy, class diagrams, sequence diagrams) from Python code",
            author="RETER Team",
            requires_reter=True,
            dependencies=[],
            categories=["visualization", "documentation", "uml", "diagrams"]
        )

    def get_tools(self) -> List[ToolDefinition]:
        """Register UML diagram generation tools."""
        return [
            ToolDefinition(
                name="get_class_hierarchy",
                description="Generate class hierarchy diagram showing inheritance relationships (class names only, no methods/attributes). Returns diagram in JSON or Markdown format.",
                handler=self._get_class_hierarchy,
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "instance_name": {
                            "type": "string",
                            "description": "RETER instance name containing loaded Python code",
                            "default": "default"
                        },
                        "format": {
                            "type": "string",
                            "enum": ["json", "markdown"],
                            "default": "markdown",
                            "description": "Output format: 'json' for structured data, 'markdown' for text diagram"
                        },
                        "root_class": {
                            "type": "string",
                            "description": "Optional root class name to start hierarchy from (if not specified, shows all classes)"
                        }
                    },
                }
            ),
            ToolDefinition(
                name="get_class_diagram",
                description="Generate class diagram for specified classes showing attributes, methods, and relationships. Returns diagram in JSON or Markdown format.",
                handler=self._get_class_diagram,
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "instance_name": {
                            "type": "string",
                            "description": "RETER instance name containing loaded Python code",
                            "default": "default"
                        },
                        "classes": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of class names to include in diagram"
                        },
                        "format": {
                            "type": "string",
                            "enum": ["json", "markdown"],
                            "default": "markdown",
                            "description": "Output format: 'json' for structured data, 'markdown' for text diagram"
                        },
                        "include_methods": {
                            "type": "boolean",
                            "default": True,
                            "description": "Include methods in the diagram"
                        },
                        "include_attributes": {
                            "type": "boolean",
                            "default": True,
                            "description": "Include attributes in the diagram"
                        }
                    },
                    "required": ["classes"]
                }
            ),
            ToolDefinition(
                name="get_sequence_diagram",
                description="Generate sequence diagram showing method calls between specified classes. Returns diagram in JSON or Markdown format.",
                handler=self._get_sequence_diagram,
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "instance_name": {
                            "type": "string",
                            "description": "RETER instance name containing loaded Python code",
                            "default": "default"
                        },
                        "classes": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of class names to include in sequence diagram"
                        },
                        "format": {
                            "type": "string",
                            "enum": ["json", "markdown"],
                            "default": "markdown",
                            "description": "Output format: 'json' for structured data, 'markdown' for Mermaid-compatible text"
                        },
                        "entry_point": {
                            "type": "string",
                            "description": "Optional method name to use as entry point for sequence"
                        },
                        "max_depth": {
                            "type": "integer",
                            "default": 10,
                            "description": "Maximum call depth to traverse (prevents infinite recursion)"
                        },
                        "exclude_patterns": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of regex patterns to exclude methods (e.g., '__init__', '.*_internal.*')"
                        },
                        "include_only_classes": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "If specified, only show calls involving these classes"
                        }
                    },
                    "required": ["classes"]
                }
            ),
            ToolDefinition(
                name="get_dependency_graph",
                description="Generate module dependency graph showing import relationships. Use summary_only=true for 10x smaller response.",
                handler=self._get_dependency_graph,
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "instance_name": {
                            "type": "string",
                            "description": "RETER instance name containing loaded Python code",
                            "default": "default"
                        },
                        "format": {
                            "type": "string",
                            "enum": ["json", "markdown", "graphviz"],
                            "default": "markdown",
                            "description": "Output format"
                        },
                        "show_external": {
                            "type": "boolean",
                            "default": False,
                            "description": "Include external/stdlib imports"
                        },
                        "group_by_package": {
                            "type": "boolean",
                            "default": True,
                            "description": "Group modules by package/directory"
                        },
                        "highlight_circular": {
                            "type": "boolean",
                            "default": True,
                            "description": "Highlight circular dependencies"
                        },
                        "limit": {
                            "type": "integer",
                            "default": 100,
                            "description": "Max dependencies to return (pagination)"
                        },
                        "offset": {
                            "type": "integer",
                            "default": 0,
                            "description": "Number of dependencies to skip"
                        },
                        "module_filter": {
                            "type": "string",
                            "description": "Filter modules by prefix (e.g., 'myapp.core')"
                        },
                        "summary_only": {
                            "type": "boolean",
                            "default": False,
                            "description": "Return only summary stats (10x smaller response)"
                        }
                    }
                }
            ),
            ToolDefinition(
                name="get_call_graph",
                description="Generate call graph showing function/method call relationships from a specific entry point.",
                handler=self._get_call_graph,
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "instance_name": {
                            "type": "string",
                            "description": "RETER instance name containing loaded Python code",
                            "default": "default"
                        },
                        "focus_function": {
                            "type": "string",
                            "description": "Function/method name to focus on (e.g., 'process_order' or 'BankAccount.withdraw')"
                        },
                        "direction": {
                            "type": "string",
                            "enum": ["upstream", "downstream", "both"],
                            "default": "both",
                            "description": "Show upstream callers, downstream callees, or both"
                        },
                        "max_depth": {
                            "type": "integer",
                            "default": 3,
                            "description": "Maximum depth to traverse"
                        },
                        "format": {
                            "type": "string",
                            "enum": ["json", "markdown", "graphviz"],
                            "default": "markdown",
                            "description": "Output format"
                        },
                        "exclude_patterns": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Regex patterns to exclude (e.g., '__.*', 'test_.*')"
                        }
                    },
                    "required": ["focus_function"]
                }
            ),
            ToolDefinition(
                name="get_coupling_matrix",
                description="Generate coupling/cohesion matrix showing coupling strength between classes. Identifies tight coupling, code smells, and refactoring opportunities. Heat map visualization with coupling metrics.",
                handler=self._get_coupling_matrix,
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "instance_name": {
                            "type": "string",
                            "description": "RETER instance name containing loaded Python code",
                            "default": "default"
                        },
                        "classes": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional list of class names to analyze (if not specified, analyzes all classes)"
                        },
                        "format": {
                            "type": "string",
                            "enum": ["json", "markdown", "heatmap"],
                            "default": "markdown",
                            "description": "Output format: 'json' for data, 'markdown' for table, 'heatmap' for visual matrix"
                        },
                        "threshold": {
                            "type": "integer",
                            "default": 0,
                            "description": "Minimum coupling strength to display (filters out weak coupling)"
                        },
                        "include_inheritance": {
                            "type": "boolean",
                            "default": True,
                            "description": "Include inheritance relationships in coupling calculation"
                        },
                        "max_classes": {
                            "type": "integer",
                            "default": 20,
                            "description": "Maximum number of classes to include in matrix (most coupled classes)"
                        }
                    }
                }
            )
        ]

    def _get_class_hierarchy(
        self,
        instance_name: str = "default",
        format: str = "markdown",
        root_class: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate class hierarchy diagram.

        Delegates to ClassHierarchyGenerator.
        """
        return self._hierarchy_gen.generate(
            instance_name=instance_name,
            format=format,
            root_class=root_class
        )

    # Legacy method kept for backwards compatibility - delegates to generator
    def _build_hierarchy_tree(self, query_result: Any, root_class: Optional[str] = None) -> Dict[str, Any]:
        """Build hierarchy tree from query results. Delegates to generator."""
        return self._hierarchy_gen._build_hierarchy_tree(query_result, root_class)

    def _render_hierarchy_markdown(self, hierarchy: Dict[str, Any]) -> str:
        """Render hierarchy as markdown. Delegates to generator."""
        return self._hierarchy_gen._render_markdown(hierarchy)

    def _get_class_diagram(
        self,
        classes: List[str],
        instance_name: str = "default",
        format: str = "markdown",
        include_methods: bool = True,
        include_attributes: bool = True
    ) -> Dict[str, Any]:
        """Generate class diagram for specified classes.

        Delegates to ClassDiagramGenerator.
        """
        return self._class_diagram_gen.generate(
            classes=classes,
            instance_name=instance_name,
            format=format,
            include_methods=include_methods,
            include_attributes=include_attributes
        )

    def _get_sequence_diagram(
        self,
        classes: List[str],
        instance_name: str = "default",
        format: str = "markdown",
        entry_point: Optional[str] = None,
        max_depth: int = 10,
        exclude_patterns: Optional[List[str]] = None,
        include_only_classes: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Generate sequence diagram showing method calls.

        Delegates to SequenceDiagramGenerator.
        """
        return self._sequence_gen.generate(
            classes=classes,
            instance_name=instance_name,
            format=format,
            entry_point=entry_point,
            max_depth=max_depth,
            exclude_patterns=exclude_patterns,
            include_only_classes=include_only_classes
        )

    # =========================================================================
    # Public API Methods (called by tools_service.py)
    # =========================================================================

    def get_class_hierarchy(self, instance_name: str = "default", root_class: Optional[str] = None, format: str = "markdown") -> Dict[str, Any]:
        """Generate class hierarchy diagram."""
        return self._get_class_hierarchy(instance_name, format, root_class)

    def get_class_diagram(self, instance_name: str, classes: List[str], include_methods: bool = True, include_attributes: bool = True, format: str = "markdown") -> Dict[str, Any]:
        """Generate class diagram."""
        return self._get_class_diagram(classes, instance_name, format, include_methods, include_attributes)

    def get_sequence_diagram(self, instance_name: str, classes: List[str], entry_point: Optional[str] = None, max_depth: int = 10, format: str = "markdown", exclude_patterns: Optional[List[str]] = None, include_only_classes: Optional[List[str]] = None) -> Dict[str, Any]:
        """Generate sequence diagram."""
        return self._get_sequence_diagram(classes, instance_name, format, entry_point, max_depth, exclude_patterns, include_only_classes)

    def get_dependency_graph(self, instance_name: str = "default", format: str = "markdown", show_external: bool = False, group_by_package: bool = True, highlight_circular: bool = True, module_filter: Optional[str] = None, summary_only: bool = False, limit: int = 100, offset: int = 0, circular_deps_limit: int = 10) -> Dict[str, Any]:
        """Generate module dependency graph. Delegates to DependencyGraphGenerator."""
        return self._dependency_gen.generate(
            instance_name=instance_name,
            format=format,
            show_external=show_external,
            group_by_package=group_by_package,
            highlight_circular=highlight_circular,
            module_filter=module_filter,
            summary_only=summary_only,
            limit=limit,
            offset=offset,
            circular_deps_limit=circular_deps_limit
        )

    def get_call_graph(self, focus_function: str, instance_name: str = "default", direction: str = "both", max_depth: int = 3, format: str = "markdown", exclude_patterns: Optional[List[str]] = None) -> Dict[str, Any]:
        """Generate call graph. Delegates to CallGraphGenerator."""
        return self._call_graph_gen.generate(
            focus_function=focus_function,
            instance_name=instance_name,
            direction=direction,
            max_depth=max_depth,
            format=format,
            exclude_patterns=exclude_patterns
        )

    def get_coupling_matrix(self, instance_name: str = "default", classes: Optional[List[str]] = None, max_classes: int = 20, threshold: int = 0, include_inheritance: bool = True, format: str = "markdown") -> Dict[str, Any]:
        """Generate coupling/cohesion matrix. Delegates to CouplingMatrixGenerator."""
        return self._coupling_gen.generate(
            instance_name=instance_name,
            classes=classes,
            format=format,
            threshold=threshold,
            include_inheritance=include_inheritance,
            max_classes=max_classes
        )

