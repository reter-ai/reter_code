"""
Code Inspection Tools Registrar

Consolidates code analysis tools into a single code_inspection MCP tool.
UML diagram tools have been moved to the unified 'diagram' tool.

Uses CADSL (Code Analysis DSL) tools from cadsl/tools/inspection.
"""

from typing import Dict, Any, Optional, Literal
from fastmcp import FastMCP
from .base import ToolRegistrarBase, truncate_mcp_response
from ...reter_wrapper import DefaultInstanceNotInitialised, is_initialization_complete
from ..initialization_progress import (
    get_initializing_response,
    require_default_instance,
    ComponentNotReadyError,
)
from ..response_truncation import truncate_response


# Action categories and their descriptions
ACTIONS = {
    # Structure/Navigation
    "list_modules": "List all modules in the codebase",
    "list_classes": "List classes in the codebase or a specific module",
    "list_functions": "List top-level functions in the codebase or a specific module",
    "describe_class": "Get detailed description of a class",
    "get_docstring": "Get the docstring of a class or method",
    "get_method_signature": "Get the signature of a method including parameters and return type",
    "get_class_hierarchy": "Get the class hierarchy showing parent and child classes",
    "get_package_structure": "Get package/module structure of the codebase",

    # Search/Find
    "find_usages": "Find where a class or method is used (called) in the codebase",
    "find_subclasses": "Find all subclasses of a specified class",
    "find_callers": "Find all callers of a function/method recursively",
    "find_callees": "Find all functions/methods called by a function recursively",
    "find_decorators": "Find all uses of decorators, optionally filtered by name",
    "find_tests": "Find tests for a specific module, class, method, or function",

    # Analysis
    "analyze_dependencies": "Analyze the dependency graph of the codebase",
    "get_imports": "Get complete module import dependency graph",
    "get_external_deps": "Get external package dependencies",
    "predict_impact": "Predict impact of changing a function/method/class",
    "get_complexity": "Calculate complexity metrics for the codebase",
    "get_magic_methods": "Find all magic methods (__init__, __str__, etc.)",
    "get_interfaces": "Find classes that implement abstract base classes/interfaces",
    "get_public_api": "Get all public classes and functions",
    "get_type_hints": "Extract all type hints from parameters and return types",
    "get_api_docs": "Extract all API documentation",
    "get_exceptions": "Get exception class hierarchy",
    "get_architecture": "Generate high-level architectural overview of the codebase",
}


class CodeInspectionToolsRegistrar(ToolRegistrarBase):
    """
    Registers a unified code_inspection tool with FastMCP.

    Uses CADSL tools from cadsl/tools/inspection.
    UML diagrams are available via the 'diagram' tool.
    """

    def register(self, app: FastMCP) -> None:
        """Register the code_inspection tool."""
        from ...dsl.core import Context
        from ...cadsl.tools_bridge import inspection

        instance_manager = self.instance_manager

        @app.tool()
        @truncate_mcp_response
        def code_inspection(
            action: str,
            target: Optional[str] = None,
            module: Optional[str] = None,
            limit: int = 100,
            offset: int = 0,
            format: str = "json",
            include_methods: bool = True,
            include_attributes: bool = True,
            include_docstrings: bool = True,
            summary_only: bool = False,
            params: Optional[Dict[str, Any]] = None,
        ) -> Dict[str, Any]:
            """
            Unified code inspection tool for Python analysis.

            **See: python://reter/tools for complete documentation**

            For UML diagrams, use the 'diagram' tool instead.

            Args:
                action: The operation to perform. Available actions:

                    **Structure/Navigation:**
                    - list_modules: List all Python modules in the codebase.
                      Returns: modules[], count, total_count
                    - list_classes: List all classes, optionally filtered by module.
                      Params: module (optional filter). Returns: classes[], count, has_more
                    - list_functions: List top-level functions, optionally filtered by module.
                      Params: module (optional filter). Returns: functions[], count, has_more
                    - describe_class: Get detailed class description with methods/attributes.
                      Params: target=class_name (REQUIRED). Returns: class_info, methods[], attributes[]
                    - get_docstring: Get the docstring of a class, method, or function.
                      Params: target=name (REQUIRED). Returns: docstring, entity_type
                    - get_method_signature: Get method signature with parameters and return type.
                      Params: target=method_name (REQUIRED). Returns: signature, parameters[], return_type
                    - get_class_hierarchy: Get inheritance hierarchy (parents and children).
                      Params: target=class_name (REQUIRED). Returns: parents[], children[], hierarchy_depth
                    - get_package_structure: Get package/module directory structure.
                      Returns: modules[], by_directory{}, module_count

                    **Search/Find:**
                    - find_usages: Find where a class/method/function is called in the codebase.
                      Params: target=name (REQUIRED). Returns: usages[], count
                    - find_subclasses: Find all subclasses of a class (direct and indirect).
                      Params: target=class_name (REQUIRED). Returns: subclasses[], count
                    - find_callers: Find all functions/methods that call the target (recursive).
                      Params: target=name (REQUIRED). Returns: callers[], count, call_depth
                    - find_callees: Find all functions/methods called by the target (recursive).
                      Params: target=name (REQUIRED). Returns: callees[], count, call_depth
                    - find_decorators: Find all decorator usages, optionally by name.
                      Params: target=decorator_name (optional). Returns: decorators[], count
                    - find_tests: Find test classes/functions for a module, class, or function.
                      Params: target=name, module (optional). Returns: tests_found[], suggestions[]

                    **Analysis:**
                    - analyze_dependencies: Analyze module dependency graph.
                      Returns: dependencies[], circular_dependencies[], summary
                    - get_imports: Get complete module import dependency graph.
                      Returns: imports[], import_graph{}, external_deps[]
                    - get_external_deps: Get external (pip) package dependencies.
                      Returns: external_packages[], by_module{}
                    - predict_impact: Predict impact of changing a function/method/class.
                      Params: target=entity_name (REQUIRED). Returns: affected_files[], affected_entities[], risk_level
                    - get_complexity: Calculate complexity metrics for the codebase.
                      Returns: class_complexity{}, parameter_complexity{}, inheritance_complexity{}, call_complexity{}
                    - get_magic_methods: Find all dunder methods (__init__, __str__, etc.).
                      Returns: magic_methods[], by_class{}, count
                    - get_interfaces: Find classes implementing abstract base classes/interfaces.
                      Params: target=interface_name (optional filter). Returns: implementations[], count
                    - get_public_api: Get all public (non-underscore) classes and functions.
                      Returns: entities[], count, by_type{}
                    - get_type_hints: Extract all type annotations from parameters and returns.
                      Returns: type_hints[], coverage_stats{}
                    - get_api_docs: Extract all API documentation from docstrings.
                      Returns: documentation{}, coverage_stats{}
                    - get_exceptions: Get exception class hierarchy.
                      Returns: exceptions[], hierarchy{}, custom_exceptions[]
                    - get_architecture: Generate high-level architectural overview.
                      Params: format="json"|"markdown"|"mermaid". Returns: overview, layers[], components[]

                target: Target entity name (class, method, function, decorator)
                module: Module name filter for list operations
                limit: Maximum results to return (default: 100)
                offset: Pagination offset (default: 0)
                format: Output format - "json", "markdown", or "mermaid" (default: "json")
                include_methods: Include methods in class descriptions (default: True)
                include_attributes: Include attributes in class descriptions (default: True)
                include_docstrings: Include docstrings (default: True)
                summary_only: Return summary only for smaller response (default: False)
                params: Additional action-specific parameters as dict

            Returns:
                Action-specific results with success status
            """
            # Code inspection requires RETER to be ready
            try:
                require_default_instance()
            except ComponentNotReadyError as e:
                return e.to_response()

            if action not in ACTIONS:
                return {
                    "success": False,
                    "error": f"Unknown action: '{action}'",
                    "available_actions": list(ACTIONS.keys()),
                    "hint": "Use one of the available actions. For UML diagrams, use the 'diagram' tool.",
                    "_resources": {
                        "python://reter/tools": "Python Analysis Tools Reference",
                        "python://reter/query-patterns": "Query Patterns"
                    }
                }

            # Get RETER instance
            try:
                reter = instance_manager.get_or_create_instance("default")
            except DefaultInstanceNotInitialised as e:
                return {"success": False, "error": str(e), "status": "initializing"}
            except Exception as e:
                return {"success": False, "error": f"Failed to get RETER instance: {str(e)}"}

            # Build context for CADSL tools
            extra = params or {}
            ctx = Context(
                reter=reter,
                params={
                    "target": target,
                    "module": module,
                    "limit": limit,
                    "offset": offset,
                    "format": format,
                    "include_methods": include_methods,
                    "include_attributes": include_attributes,
                    "include_docstrings": include_docstrings,
                    "summary_only": summary_only,
                    **extra
                },
                instance_name="default"
            )

            try:
                # ═══════════════════════════════════════════════════════════
                # STRUCTURE/NAVIGATION
                # ═══════════════════════════════════════════════════════════

                if action == "list_modules":
                    return inspection.list_modules(ctx)

                elif action == "list_classes":
                    return inspection.list_classes(ctx)

                elif action == "list_functions":
                    return inspection.list_functions(ctx)

                elif action == "describe_class":
                    if not target:
                        return {"success": False, "error": "target (class_name) is required for describe_class"}
                    return inspection.describe_class(ctx)

                elif action == "get_docstring":
                    if not target:
                        return {"success": False, "error": "target (name) is required for get_docstring"}
                    return inspection.get_docstring(ctx)

                elif action == "get_method_signature":
                    if not target:
                        return {"success": False, "error": "target (method_name) is required for get_method_signature"}
                    return inspection.get_method_signature(ctx)

                elif action == "get_class_hierarchy":
                    if not target:
                        return {"success": False, "error": "target (class_name) is required for get_class_hierarchy"}
                    return inspection.get_class_hierarchy(ctx)

                elif action == "get_package_structure":
                    return inspection.get_package_structure(ctx)

                # ═══════════════════════════════════════════════════════════
                # SEARCH/FIND
                # ═══════════════════════════════════════════════════════════

                elif action == "find_usages":
                    if not target:
                        return {"success": False, "error": "target (name) is required for find_usages"}
                    return inspection.find_usages(ctx)

                elif action == "find_subclasses":
                    if not target:
                        return {"success": False, "error": "target (class_name) is required for find_subclasses"}
                    return inspection.find_subclasses(ctx)

                elif action == "find_callers":
                    if not target:
                        return {"success": False, "error": "target (name) is required for find_callers"}
                    return inspection.find_callers(ctx)

                elif action == "find_callees":
                    if not target:
                        return {"success": False, "error": "target (name) is required for find_callees"}
                    return inspection.find_callees(ctx)

                elif action == "find_decorators":
                    return inspection.find_decorators(ctx)

                elif action == "find_tests":
                    return inspection.find_tests(ctx)

                # ═══════════════════════════════════════════════════════════
                # ANALYSIS
                # ═══════════════════════════════════════════════════════════

                elif action == "analyze_dependencies":
                    return inspection.analyze_dependencies(ctx)

                elif action == "get_imports":
                    return inspection.get_imports(ctx)

                elif action == "get_external_deps":
                    return inspection.get_external_deps(ctx)

                elif action == "predict_impact":
                    if not target:
                        return {"success": False, "error": "target (entity_name) is required for predict_impact"}
                    return inspection.predict_impact(ctx)

                elif action == "get_complexity":
                    return inspection.get_complexity(ctx)

                elif action == "get_magic_methods":
                    return inspection.get_magic_methods(ctx)

                elif action == "get_interfaces":
                    return inspection.get_interfaces(ctx)

                elif action == "get_public_api":
                    return inspection.get_public_api(ctx)

                elif action == "get_type_hints":
                    return inspection.get_type_hints(ctx)

                elif action == "get_api_docs":
                    return inspection.get_api_docs(ctx)

                elif action == "get_exceptions":
                    return inspection.get_exceptions(ctx)

                elif action == "get_architecture":
                    return inspection.get_architecture(ctx)

                else:
                    return {"success": False, "error": f"Action '{action}' not implemented"}

            except Exception as e:
                import traceback
                return {
                    "success": False,
                    "error": str(e),
                    "action": action,
                    "traceback": traceback.format_exc()
                }

