"""
Advanced Code Analysis Tools - Facade

This facade delegates to specialized tool classes for code analysis.
Each specialized class handles a specific domain of analysis.

Supports multiple languages via the LanguageSupport module:
- "oo" (default): Language-independent queries (Python + JavaScript)
- "python" or "py": Python-specific queries
- "javascript" or "js": JavaScript-specific queries
"""

from typing import Dict, Any, Optional

from .base import AdvancedToolsBase
from .code_quality import CodeQualityTools
from .dependency_analysis import DependencyAnalysisTools
from .pattern_detection import PatternDetectionTools
from .change_impact import ChangeImpactTools
from .type_analysis import TypeAnalysisTools
from .exception_analysis import ExceptionAnalysisTools
from .test_analysis import TestAnalysisTools
from .documentation_analysis import DocumentationAnalysisTools
from .architecture_analysis import ArchitectureAnalysisTools
from codeine.services.language_support import LanguageType


class AdvancedPythonTools(AdvancedToolsBase):
    """
    Facade for advanced code analysis tools.

    Delegates to specialized classes:
    - CodeQualityTools: Code smells, large classes, magic numbers
    - DependencyAnalysisTools: Import graphs, circular imports
    - PatternDetectionTools: Decorators, magic methods, interfaces
    - ChangeImpactTools: Call graphs, impact prediction
    - TypeAnalysisTools: Type hints, untyped functions
    - ExceptionAnalysisTools: Exception handling patterns
    - TestAnalysisTools: Test files, fixtures
    - DocumentationAnalysisTools: Docstrings, API docs
    - ArchitectureAnalysisTools: Structure, complexity, overview
    """

    def __init__(self, reter_wrapper, language: LanguageType = "oo"):
        super().__init__(reter_wrapper, language)
        self._code_quality = CodeQualityTools(reter_wrapper, language)
        self._dependency = DependencyAnalysisTools(reter_wrapper, language)
        self._pattern = PatternDetectionTools(reter_wrapper, language)
        self._change_impact = ChangeImpactTools(reter_wrapper, language)
        self._type_analysis = TypeAnalysisTools(reter_wrapper, language)
        self._exception_analysis = ExceptionAnalysisTools(reter_wrapper, language)
        self._test_analysis = TestAnalysisTools(reter_wrapper, language)
        self._documentation_analysis = DocumentationAnalysisTools(reter_wrapper, language)
        self._architecture_analysis = ArchitectureAnalysisTools(reter_wrapper, language)

    # =========================================================================
    # CODE QUALITY (CodeQualityTools)
    # =========================================================================

    def find_large_classes(self, instance_name: str, threshold: int = 20,
                           limit: int = 100, offset: int = 0) -> Dict[str, Any]:
        """Find classes with too many methods (God classes)."""
        return self._code_quality.find_large_classes(instance_name, threshold)

    def find_long_parameter_lists(self, instance_name: str, threshold: int = 5,
                                   limit: int = 100, offset: int = 0) -> Dict[str, Any]:
        """Find functions/methods with too many parameters."""
        return self._code_quality.find_long_parameter_lists(instance_name, threshold)

    def find_magic_numbers(self, instance_name: str, exclude_common: bool = True,
                           min_occurrences: int = 1, limit: int = 100,
                           offset: int = 0) -> Dict[str, Any]:
        """Find magic numbers (numeric literals) in code."""
        return self._code_quality.find_magic_numbers(
            instance_name, exclude_common, min_occurrences, limit, offset
        )

    # =========================================================================
    # DEPENDENCY ANALYSIS (DependencyAnalysisTools)
    # =========================================================================

    def get_import_graph(self, instance_name: str, limit: int = 100,
                         offset: int = 0) -> Dict[str, Any]:
        """Get complete module import dependency graph."""
        return self._dependency.get_import_graph(instance_name)

    def find_circular_imports(self, instance_name: str) -> Dict[str, Any]:
        """Find circular import dependencies."""
        return self._dependency.find_circular_imports(instance_name)

    def get_external_dependencies(self, instance_name: str, limit: int = 100,
                                   offset: int = 0) -> Dict[str, Any]:
        """Get external package dependencies."""
        return self._dependency.get_external_dependencies(instance_name)

    # =========================================================================
    # PATTERN DETECTION (PatternDetectionTools)
    # =========================================================================

    def find_decorators_usage(self, instance_name: str,
                               decorator_name: Optional[str] = None,
                               limit: int = 100, offset: int = 0) -> Dict[str, Any]:
        """Find all uses of decorators, optionally filtered by name."""
        return self._pattern.find_decorators_usage(instance_name, decorator_name)

    def get_magic_methods(self, instance_name: str, limit: int = 100,
                          offset: int = 0) -> Dict[str, Any]:
        """Find all magic methods (__init__, __str__, etc.)."""
        return self._pattern.get_magic_methods(instance_name)

    def get_interface_implementations(self, instance_name: str,
                                       interface_name: Optional[str] = None,
                                       limit: int = 100, offset: int = 0) -> Dict[str, Any]:
        """Find classes implementing abstract base classes/interfaces."""
        return self._pattern.get_interface_implementations(instance_name, interface_name)

    def get_public_api(self, instance_name: str, limit: int = 100,
                       offset: int = 0) -> Dict[str, Any]:
        """Get all public classes and functions (not starting with _)."""
        return self._pattern.get_public_api(instance_name)

    # =========================================================================
    # TYPE ANALYSIS (TypeAnalysisTools)
    # =========================================================================

    def get_type_hints(self, instance_name: str, limit: int = 100,
                       offset: int = 0) -> Dict[str, Any]:
        """Extract all type hints from parameters and return types."""
        return self._type_analysis.get_type_hints(instance_name)

    def find_untyped_functions(self, instance_name: str) -> Dict[str, Any]:
        """Find functions/methods without type hints."""
        return self._type_analysis.find_untyped_functions(instance_name)

    # =========================================================================
    # TEST ANALYSIS (TestAnalysisTools)
    # =========================================================================

    def find_test_files(self, instance_name: str) -> Dict[str, Any]:
        """Find test files based on naming conventions."""
        return self._test_analysis.find_test_files(instance_name)

    def find_test_fixtures(self, instance_name: str) -> Dict[str, Any]:
        """Find pytest fixtures."""
        return self._test_analysis.find_test_fixtures(instance_name)

    # =========================================================================
    # CHANGE IMPACT (ChangeImpactTools)
    # =========================================================================

    def predict_change_impact(self, instance_name: str,
                               entity_name: str) -> Dict[str, Any]:
        """Predict impact of changing a function/method/class."""
        return self._change_impact.predict_change_impact(instance_name, entity_name)

    def find_callers_recursive(self, instance_name: str,
                                target_name: str) -> Dict[str, Any]:
        """Find all callers of a function/method (recursive, transitive)."""
        return self._change_impact.find_callers_recursive(instance_name, target_name)

    def find_callees_recursive(self, instance_name: str,
                                source_name: str) -> Dict[str, Any]:
        """Find all functions/methods called by a function (recursive)."""
        return self._change_impact.find_callees_recursive(instance_name, source_name)

    # =========================================================================
    # DOCUMENTATION ANALYSIS (DocumentationAnalysisTools)
    # =========================================================================

    def find_undocumented_code(self, instance_name: str) -> Dict[str, Any]:
        """Find undocumented classes, functions, and methods."""
        return self._documentation_analysis.find_undocumented_code(instance_name)

    def get_api_documentation(self, instance_name: str) -> Dict[str, Any]:
        """Extract all docstrings and generate API documentation."""
        return self._documentation_analysis.get_api_documentation(instance_name)

    # =========================================================================
    # ARCHITECTURE ANALYSIS (ArchitectureAnalysisTools)
    # =========================================================================

    def get_exception_hierarchy(self, instance_name: str) -> Dict[str, Any]:
        """Get exception class hierarchy."""
        return self._architecture_analysis.get_exception_hierarchy(instance_name)

    def get_package_structure(self, instance_name: str) -> Dict[str, Any]:
        """Get package/module structure."""
        return self._architecture_analysis.get_package_structure(instance_name)

    def find_duplicate_names(self, instance_name: str) -> Dict[str, Any]:
        """Find entities with duplicate names across modules."""
        return self._architecture_analysis.find_duplicate_names(instance_name)

    def get_complexity_metrics(self, instance_name: str) -> Dict[str, Any]:
        """Calculate complexity metrics for the codebase."""
        return self._architecture_analysis.get_complexity_metrics(instance_name)

    def get_architecture_overview(self, instance_name: str,
                                   output_format: str = "json") -> Dict[str, Any]:
        """Generate high-level architectural overview of the codebase."""
        return self._architecture_analysis.get_architecture_overview(
            instance_name, output_format
        )
