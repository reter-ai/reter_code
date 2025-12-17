"""
Advanced Python Analysis Tools

Modular implementation split into focused tool classes.

The AdvancedPythonToolsFacade provides a unified interface that delegates
to specialized tool classes, eliminating code duplication.
"""

from .advanced_python_tools import AdvancedPythonTools
from .facade import AdvancedPythonToolsFacade
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
from .data_clump_detection import DataClumpDetectionTools
from .function_analysis import FunctionAnalysisTools
from .inheritance_refactoring import InheritanceRefactoringTools

__all__ = [
    'AdvancedPythonTools',
    'AdvancedPythonToolsFacade',
    'AdvancedToolsBase',
    'CodeQualityTools',
    'DependencyAnalysisTools',
    'PatternDetectionTools',
    'ChangeImpactTools',
    'TypeAnalysisTools',
    'ExceptionAnalysisTools',
    'TestAnalysisTools',
    'DocumentationAnalysisTools',
    'ArchitectureAnalysisTools',
    'DataClumpDetectionTools',
    'FunctionAnalysisTools',
    'InheritanceRefactoringTools',
]
