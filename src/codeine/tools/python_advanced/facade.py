"""
AdvancedPythonTools Facade

Provides a unified interface to specialized Python analysis tools.
This facade delegates to specialized tool classes instead of duplicating methods.

Usage:
    tools = AdvancedPythonToolsFacade(reter_wrapper)
    result = tools.find_large_classes("default", threshold=20)
"""

from typing import Dict, Any, List, Optional

from .base import AdvancedToolsBase
from .code_quality import CodeQualityTools
from .dependency_analysis import DependencyAnalysisTools
from .pattern_detection import PatternDetectionTools
from .change_impact import ChangeImpactTools
from .type_analysis import TypeAnalysisTools


class AdvancedPythonToolsFacade(AdvancedToolsBase):
    """
    Facade for advanced Python code analysis tools.

    Delegates to specialized tool classes for better maintainability
    while providing a unified interface for consumers.

    Tool Categories:
    - Code Quality: find_large_classes, find_long_parameter_lists
    - Dependencies: get_import_graph, find_circular_imports, get_external_dependencies
    - Patterns: find_decorators_usage, get_magic_methods, get_interface_implementations, get_public_api
    - Change Impact: predict_change_impact, find_callers_recursive, find_callees_recursive
    - Type Analysis: get_type_hints, find_untyped_functions
    """

    def __init__(self, reter_wrapper):
        """
        Initialize facade with ReterWrapper instance.

        Creates instances of all specialized tool classes.

        Args:
            reter_wrapper: ReterWrapper instance with loaded Python code
        """
        super().__init__(reter_wrapper)

        # Initialize specialized tool classes
        self._code_quality = CodeQualityTools(reter_wrapper)
        self._dependency = DependencyAnalysisTools(reter_wrapper)
        self._pattern = PatternDetectionTools(reter_wrapper)
        self._change_impact = ChangeImpactTools(reter_wrapper)
        self._type_analysis = TypeAnalysisTools(reter_wrapper)

    # =========================================================================
    # CODE QUALITY METRICS (delegated to CodeQualityTools)
    # =========================================================================

    def find_large_classes(
        self,
        instance_name: str,
        threshold: int = 20,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        Find classes with too many methods (God classes).

        Delegates to CodeQualityTools.

        Args:
            instance_name: RETER instance name
            threshold: Minimum number of methods (default: 20)
            limit: Maximum results to return (default: 100)
            offset: Results to skip (default: 0)

        Returns:
            dict with success, classes list, count, queries
        """
        return self._code_quality.find_large_classes(instance_name, threshold)

    def find_long_parameter_lists(
        self,
        instance_name: str,
        threshold: int = 5,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        Find functions/methods with too many parameters.

        Delegates to CodeQualityTools.

        Args:
            instance_name: RETER instance name
            threshold: Maximum acceptable parameter count (default: 5)
            limit: Maximum results to return (default: 100)
            offset: Results to skip (default: 0)

        Returns:
            dict with success, functions list, count, queries
        """
        return self._code_quality.find_long_parameter_lists(instance_name, threshold)

    # =========================================================================
    # DEPENDENCY ANALYSIS (delegated to DependencyAnalysisTools)
    # =========================================================================

    def get_import_graph(self, instance_name: str) -> Dict[str, Any]:
        """
        Get the import dependency graph.

        Delegates to DependencyAnalysisTools.

        Args:
            instance_name: RETER instance name

        Returns:
            dict with success, imports list, modules, import_graph
        """
        return self._dependency.get_import_graph(instance_name)

    def find_circular_imports(self, instance_name: str) -> Dict[str, Any]:
        """
        Detect circular import dependencies.

        Delegates to DependencyAnalysisTools.

        Args:
            instance_name: RETER instance name

        Returns:
            dict with success, cycles list, count
        """
        return self._dependency.find_circular_imports(instance_name)

    def get_external_dependencies(self, instance_name: str) -> Dict[str, Any]:
        """
        Get external (pip) package dependencies.

        Delegates to DependencyAnalysisTools.

        Args:
            instance_name: RETER instance name

        Returns:
            dict with success, external_packages list
        """
        return self._dependency.get_external_dependencies(instance_name)

    # =========================================================================
    # PATTERN DETECTION (delegated to PatternDetectionTools)
    # =========================================================================

    def find_decorators_usage(
        self,
        instance_name: str,
        decorator_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Find all decorator usages, optionally filtered by name.

        Delegates to PatternDetectionTools.

        Args:
            instance_name: RETER instance name
            decorator_name: Optional filter by decorator name

        Returns:
            dict with success, decorators list, count
        """
        return self._pattern.find_decorators_usage(instance_name, decorator_name)

    def get_magic_methods(self, instance_name: str) -> Dict[str, Any]:
        """
        Find all dunder methods (__init__, __str__, etc.).

        Delegates to PatternDetectionTools.

        Args:
            instance_name: RETER instance name

        Returns:
            dict with success, magic_methods list, by_class dict
        """
        return self._pattern.get_magic_methods(instance_name)

    def get_interface_implementations(
        self,
        instance_name: str,
        interface_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Find classes implementing abstract base classes/interfaces.

        Delegates to PatternDetectionTools.

        Args:
            instance_name: RETER instance name
            interface_name: Optional filter by interface name

        Returns:
            dict with success, implementations list, count
        """
        return self._pattern.get_interface_implementations(instance_name, interface_name)

    def get_public_api(self, instance_name: str) -> Dict[str, Any]:
        """
        Get all public (non-underscore) classes and functions.

        Delegates to PatternDetectionTools.

        Args:
            instance_name: RETER instance name

        Returns:
            dict with success, entities list, by_type dict
        """
        return self._pattern.get_public_api(instance_name)

    # =========================================================================
    # CHANGE IMPACT (delegated to ChangeImpactTools)
    # =========================================================================

    def predict_change_impact(
        self,
        instance_name: str,
        target: str
    ) -> Dict[str, Any]:
        """
        Predict impact of changing a function/method/class.

        Delegates to ChangeImpactTools.

        Args:
            instance_name: RETER instance name
            target: Entity name to analyze

        Returns:
            dict with success, affected_files list, affected_entities list
        """
        return self._change_impact.predict_change_impact(instance_name, target)

    def find_callers_recursive(
        self,
        instance_name: str,
        target: str,
        max_depth: int = 10
    ) -> Dict[str, Any]:
        """
        Find all functions/methods that call the target (recursive).

        Delegates to ChangeImpactTools.

        Args:
            instance_name: RETER instance name
            target: Entity name to find callers of
            max_depth: Maximum recursion depth (default: 10)

        Returns:
            dict with success, callers list, count
        """
        return self._change_impact.find_callers_recursive(instance_name, target, max_depth)

    def find_callees_recursive(
        self,
        instance_name: str,
        target: str,
        max_depth: int = 10
    ) -> Dict[str, Any]:
        """
        Find all functions/methods called by the target (recursive).

        Delegates to ChangeImpactTools.

        Args:
            instance_name: RETER instance name
            target: Entity name to find callees of
            max_depth: Maximum recursion depth (default: 10)

        Returns:
            dict with success, callees list, count
        """
        return self._change_impact.find_callees_recursive(instance_name, target, max_depth)

    # =========================================================================
    # TYPE ANALYSIS (delegated to TypeAnalysisTools)
    # =========================================================================

    def get_type_hints(
        self,
        instance_name: str,
        include_builtins: bool = False
    ) -> Dict[str, Any]:
        """
        Extract all type annotations from parameters and returns.

        Delegates to TypeAnalysisTools.

        Args:
            instance_name: RETER instance name
            include_builtins: Include builtin types (default: False)

        Returns:
            dict with success, type_hints list, coverage_stats dict
        """
        return self._type_analysis.get_type_hints(instance_name, include_builtins)

    def find_untyped_functions(self, instance_name: str) -> Dict[str, Any]:
        """
        Find functions/methods without type hints.

        Delegates to TypeAnalysisTools.

        Args:
            instance_name: RETER instance name

        Returns:
            dict with success, untyped_functions list, count
        """
        return self._type_analysis.find_untyped_functions(instance_name)
