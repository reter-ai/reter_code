"""
Code Inspection Tools Registrar

Consolidates code analysis tools into a single code_inspection MCP tool.
Supports multiple languages: Python, JavaScript, HTML, C#, C++, and language-independent (oo).
UML diagram tools have been moved to the unified 'diagram' tool.

Replaces:
- PythonBasicToolsRegistrar (10 tools)
- PythonAdvancedToolsRegistrar (15 tools)

Total: 25 analysis operations -> 1 unified tool
(UML diagrams moved to 'diagram' tool)
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
from ..language_support import LanguageSupport, LanguageType


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

    Consolidates python_basic and python_advanced tools into one.
    UML diagrams are available via the 'diagram' tool.
    """

    def register(self, app: FastMCP) -> None:
        """Register the code_inspection tool."""
        from ...tools.python_basic.python_tools import PythonAnalysisTools
        from ...tools.python_advanced.advanced_python_tools import AdvancedPythonTools

        instance_manager = self.instance_manager

        @app.tool()
        @truncate_mcp_response
        def code_inspection(
            action: str,
            instance_name: str = "default",
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
            language: str = "oo"
        ) -> Dict[str, Any]:
            """
            Unified code inspection tool for Python analysis.

            **See: python://reter/tools for complete documentation**

            Supports multiple languages via the `language` parameter:
            - "oo" (default): Language-independent queries (matches all languages)
            - "python" or "py": Python-specific queries
            - "javascript" or "js": JavaScript-specific queries
            - "html" or "htm": HTML document queries (for documents, forms, scripts, event handlers)

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

                instance_name: RETER instance name (default: "default")
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
                language: Programming language to analyze (default: "oo")
                    - "oo": Language-independent (matches Python + JavaScript)
                    - "python" or "py": Python only
                    - "javascript" or "js": JavaScript only
                    - "html" or "htm": HTML documents only

            Returns:
                Action-specific results with success status
            """
            # Validate language parameter
            try:
                LanguageSupport.get_language(language)
            except ValueError as e:
                return {
                    "success": False,
                    "error": str(e),
                    "supported_languages": LanguageSupport.supported_languages()
                }

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
                reter = instance_manager.get_or_create_instance(instance_name)
            except DefaultInstanceNotInitialised as e:
                return {"success": False, "error": str(e), "status": "initializing"}
            except Exception as e:
                return {"success": False, "error": f"Failed to get RETER instance: {str(e)}"}

            # Initialize tool classes with language support
            basic_tools = PythonAnalysisTools(reter, language=language)
            advanced_tools = AdvancedPythonTools(reter, language=language)

            # Extra params
            extra = params or {}

            try:
                # ═══════════════════════════════════════════════════════════
                # STRUCTURE/NAVIGATION
                # ═══════════════════════════════════════════════════════════

                if action == "list_modules":
                    return basic_tools.list_modules(instance_name, limit, offset)

                elif action == "list_classes":
                    return basic_tools.list_classes(instance_name, module, limit, offset)

                elif action == "list_functions":
                    return basic_tools.list_functions(instance_name, module, limit, offset)

                elif action == "describe_class":
                    if not target:
                        return {"success": False, "error": "target (class_name) is required for describe_class"}
                    return basic_tools.describe_class(
                        instance_name=instance_name,
                        class_name=target,
                        include_methods=include_methods,
                        include_attributes=include_attributes,
                        include_parameters=extra.get("include_parameters", True),
                        include_docstrings=include_docstrings,
                        methods_limit=extra.get("methods_limit", 20),
                        methods_offset=extra.get("methods_offset", 0),
                        summary_only=summary_only,
                        max_docstring_length=extra.get("max_docstring_length", 200)
                    )

                elif action == "get_docstring":
                    if not target:
                        return {"success": False, "error": "target (name) is required for get_docstring"}
                    return basic_tools.get_docstring(instance_name, target, limit, offset)

                elif action == "get_method_signature":
                    if not target:
                        return {"success": False, "error": "target (method_name) is required for get_method_signature"}
                    return basic_tools.get_method_signature(instance_name, target)

                elif action == "get_class_hierarchy":
                    if not target:
                        return {"success": False, "error": "target (class_name) is required for get_class_hierarchy"}
                    return basic_tools.get_class_hierarchy(instance_name, target)

                elif action == "get_package_structure":
                    return advanced_tools.get_package_structure(instance_name)

                # ═══════════════════════════════════════════════════════════
                # SEARCH/FIND
                # ═══════════════════════════════════════════════════════════

                elif action == "find_usages":
                    if not target:
                        return {"success": False, "error": "target (name) is required for find_usages"}
                    return basic_tools.find_usages(instance_name, target, limit, offset)

                elif action == "find_subclasses":
                    if not target:
                        return {"success": False, "error": "target (class_name) is required for find_subclasses"}
                    return basic_tools.find_subclasses(instance_name, target, limit, offset)

                elif action == "find_callers":
                    if not target:
                        return {"success": False, "error": "target (name) is required for find_callers"}
                    return advanced_tools.find_callers_recursive(instance_name, target)

                elif action == "find_callees":
                    if not target:
                        return {"success": False, "error": "target (name) is required for find_callees"}
                    return advanced_tools.find_callees_recursive(instance_name, target)

                elif action == "find_decorators":
                    return advanced_tools.find_decorators_usage(instance_name, target, limit, offset)

                elif action == "find_tests":
                    return self._find_tests(
                        basic_tools, instance_name, target, module, limit, offset
                    )

                # ═══════════════════════════════════════════════════════════
                # ANALYSIS
                # ═══════════════════════════════════════════════════════════

                elif action == "analyze_dependencies":
                    return basic_tools.analyze_dependencies(instance_name, limit, offset)

                elif action == "get_imports":
                    return advanced_tools.get_import_graph(instance_name, limit, offset)

                elif action == "get_external_deps":
                    return advanced_tools.get_external_dependencies(instance_name, limit, offset)

                elif action == "predict_impact":
                    if not target:
                        return {"success": False, "error": "target (entity_name) is required for predict_impact"}
                    return advanced_tools.predict_change_impact(instance_name, target)

                elif action == "get_complexity":
                    return advanced_tools.get_complexity_metrics(instance_name)

                elif action == "get_magic_methods":
                    return advanced_tools.get_magic_methods(instance_name, limit, offset)

                elif action == "get_interfaces":
                    return advanced_tools.get_interface_implementations(instance_name, target, limit, offset)

                elif action == "get_public_api":
                    return advanced_tools.get_public_api(instance_name, limit, offset)

                elif action == "get_type_hints":
                    return advanced_tools.get_type_hints(instance_name, limit, offset)

                elif action == "get_api_docs":
                    return advanced_tools.get_api_documentation(instance_name, limit, offset)

                elif action == "get_exceptions":
                    return advanced_tools.get_exception_hierarchy(instance_name)

                elif action == "get_architecture":
                    return advanced_tools.get_architecture_overview(instance_name, format)

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

    def _find_tests(
        self,
        basic_tools,
        instance_name: str,
        target: Optional[str],
        module: Optional[str],
        limit: int,
        offset: int
    ) -> Dict[str, Any]:
        """
        Find tests for a specific module, class, method, or function.

        Args:
            basic_tools: PythonAnalysisTools instance
            instance_name: RETER instance name
            target: Target entity name (class, method, or function)
            module: Module name to find tests for
            limit: Maximum results
            offset: Pagination offset

        Returns:
            Dict with found tests and suggestions
        """
        from ...tools.test_coverage.matcher import TestMatcher

        # Get all test classes and functions
        all_classes = basic_tools.list_classes(instance_name, limit=500).get("classes", [])
        all_functions = basic_tools.list_functions(instance_name, limit=1000).get("functions", [])

        test_classes = [c for c in all_classes if c.get("name", "").startswith("Test")]
        test_functions = [f for f in all_functions if f.get("name", "").startswith("test_")]

        matcher = TestMatcher(test_classes, test_functions)

        results = {
            "success": True,
            "action": "find_tests",
            "target": target,
            "module": module,
            "tests_found": [],
            "suggestions": []
        }

        # If module is specified, find tests for the module
        if module:
            module_tests = self._find_tests_for_module(
                module, test_classes, test_functions, matcher
            )
            results["tests_found"].extend(module_tests["tests"])
            results["suggestions"].extend(module_tests["suggestions"])
            results["module_coverage"] = module_tests["coverage"]

        # If target is specified, find tests for the entity
        if target:
            # Try to determine entity type
            entity_type = self._determine_entity_type(target, all_classes, all_functions)

            if entity_type == "class":
                class_tests = self._find_tests_for_class(target, matcher, test_classes)
                results["tests_found"].extend(class_tests["tests"])
                results["suggestions"].extend(class_tests["suggestions"])
                results["entity_type"] = "class"

            elif entity_type == "function":
                func_tests = self._find_tests_for_function(target, matcher)
                results["tests_found"].extend(func_tests["tests"])
                results["suggestions"].extend(func_tests["suggestions"])
                results["entity_type"] = "function"

            elif entity_type == "method":
                # target is in format "ClassName.method_name"
                parts = target.split(".")
                if len(parts) >= 2:
                    class_name = parts[0]
                    method_name = parts[-1]
                    method_tests = self._find_tests_for_method(
                        class_name, method_name, matcher
                    )
                    results["tests_found"].extend(method_tests["tests"])
                    results["suggestions"].extend(method_tests["suggestions"])
                    results["entity_type"] = "method"
            else:
                # Unknown - try all approaches
                results["entity_type"] = "unknown"
                results["suggestions"].append({
                    "message": f"Could not determine entity type for '{target}'",
                    "hint": "Use format 'ClassName' for classes, 'function_name' for functions, or 'ClassName.method_name' for methods"
                })

        # Apply pagination
        results["total_tests"] = len(results["tests_found"])
        results["tests_found"] = results["tests_found"][offset:offset + limit]

        return results

    def _determine_entity_type(
        self,
        target: str,
        all_classes: list,
        all_functions: list
    ) -> str:
        """Determine if target is a class, function, or method."""
        # Check if it's a method (contains dot)
        if "." in target:
            return "method"

        # Check if it's a known class
        class_names = {c.get("name", "") for c in all_classes}
        if target in class_names:
            return "class"

        # Check if it's a known function
        func_names = {f.get("name", "") for f in all_functions}
        if target in func_names:
            return "function"

        # Default: assume it's a class if PascalCase, function otherwise
        if target and target[0].isupper():
            return "class"
        return "function"

    def _find_tests_for_module(
        self,
        module: str,
        test_classes: list,
        test_functions: list,
        matcher
    ) -> Dict[str, Any]:
        """Find all tests for a module."""
        tests = []
        suggestions = []

        # Find test modules that might test this module
        module_name = module.split(".")[-1]
        test_module_patterns = [
            f"test_{module_name}",
            f"{module_name}_test",
            f"tests.test_{module_name}",
            f"tests.{module_name}_test"
        ]

        for tc in test_classes:
            tc_module = tc.get("module", "")
            tc_name = tc.get("name", "")
            for pattern in test_module_patterns:
                if pattern in tc_module or pattern in tc_name.lower():
                    tests.append({
                        "type": "test_class",
                        "name": tc_name,
                        "module": tc_module,
                        "file": tc.get("file", ""),
                        "confidence": 0.9
                    })
                    break

        for tf in test_functions:
            tf_module = tf.get("module", "")
            tf_name = tf.get("name", "")
            for pattern in test_module_patterns:
                if pattern in tf_module:
                    tests.append({
                        "type": "test_function",
                        "name": tf_name,
                        "module": tf_module,
                        "file": tf.get("file", ""),
                        "confidence": 0.8
                    })
                    break

        if not tests:
            suggestions.append({
                "type": "create_test_module",
                "suggested_name": f"test_{module_name}.py",
                "suggested_location": f"tests/test_{module_name}.py"
            })

        return {
            "tests": tests,
            "suggestions": suggestions,
            "coverage": {
                "test_classes": len([t for t in tests if t["type"] == "test_class"]),
                "test_functions": len([t for t in tests if t["type"] == "test_function"])
            }
        }

    def _find_tests_for_class(
        self,
        class_name: str,
        matcher,
        test_classes: list
    ) -> Dict[str, Any]:
        """Find tests for a specific class."""
        tests = []
        suggestions = []

        # Use matcher to find test class
        match = matcher.find_test_for_class(class_name)

        if match.is_tested:
            # Find the actual test class info
            for tc in test_classes:
                if tc.get("name") == match.test_name:
                    tests.append({
                        "type": "test_class",
                        "name": match.test_name,
                        "module": tc.get("module", ""),
                        "file": tc.get("file", ""),
                        "confidence": match.confidence,
                        "match_pattern": match.pattern_used
                    })
                    break
        else:
            suggestions.append({
                "type": "create_test_class",
                "for_class": class_name,
                "suggested_name": matcher.suggest_test_class(class_name),
                "suggested_methods": ["test_init", "test_basic_functionality"]  # Generic suggestions
            })

        return {"tests": tests, "suggestions": suggestions}

    def _find_tests_for_function(
        self,
        func_name: str,
        matcher
    ) -> Dict[str, Any]:
        """Find tests for a specific function."""
        tests = []
        suggestions = []

        match = matcher.find_test_for_function(func_name)

        if match.is_tested:
            tests.append({
                "type": "test_function",
                "name": match.test_name,
                "confidence": match.confidence,
                "match_pattern": match.pattern_used
            })
        else:
            suggestions.append({
                "type": "create_test_function",
                "for_function": func_name,
                "suggested_name": f"test_{func_name}"
            })

        return {"tests": tests, "suggestions": suggestions}

    def _find_tests_for_method(
        self,
        class_name: str,
        method_name: str,
        matcher
    ) -> Dict[str, Any]:
        """Find tests for a specific method."""
        tests = []
        suggestions = []

        # First find the test class
        class_match = matcher.find_test_for_class(class_name)

        if class_match.is_tested:
            # Look for method test
            method_match = matcher.find_test_for_method(method_name, class_name)

            if method_match.is_tested:
                tests.append({
                    "type": "test_method",
                    "name": method_match.test_name,
                    "test_class": class_match.test_name,
                    "confidence": method_match.confidence,
                    "match_pattern": method_match.pattern_used
                })
            else:
                suggestions.append({
                    "type": "add_test_method",
                    "for_method": f"{class_name}.{method_name}",
                    "test_class": class_match.test_name,
                    "suggested_name": f"test_{method_name}"
                })
        else:
            suggestions.append({
                "type": "create_test_class",
                "for_class": class_name,
                "suggested_name": matcher.suggest_test_class(class_name),
                "then_add_test": f"test_{method_name}"
            })

        return {"tests": tests, "suggestions": suggestions}
