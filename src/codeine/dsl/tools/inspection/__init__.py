"""
Code Inspection Queries - Read-only code analysis tools.

All tools in this package use @query decorator (no side effects).

Re-exports from other packages for code_inspection registrar:
- find_decorators -> patterns/decorator_usage
- get_external_deps -> dependencies/external_deps
- get_magic_methods -> patterns/magic_methods
- get_interfaces -> patterns/interface_impl
- get_public_api -> patterns/public_api
"""

from .list_modules import list_modules

# Re-exports from patterns and dependencies modules (for code_inspection registrar)
from ..patterns.decorator_usage import find_decorator_usage as find_decorators
from ..patterns.magic_methods import find_magic_methods as get_magic_methods
from ..patterns.interface_impl import find_interface_implementations as get_interfaces
from ..patterns.public_api import find_public_api as get_public_api
from ..dependencies.external_deps import find_external_dependencies as get_external_deps
from .list_classes import list_classes
from .list_functions import list_functions
from .describe_class import describe_class
from .get_docstring import get_docstring
from .get_method_signature import get_method_signature
from .get_class_hierarchy import get_class_hierarchy
from .get_package_structure import get_package_structure
from .find_usages import find_usages
from .find_subclasses import find_subclasses
from .find_callers import find_callers
from .find_callees import find_callees
from .find_tests import find_tests
from .analyze_dependencies import analyze_dependencies
from .get_imports import get_imports
from .predict_impact import predict_impact
from .get_complexity import get_complexity
from .get_type_hints import get_type_hints
from .get_api_docs import get_api_docs
from .get_exceptions import get_exceptions
from .get_architecture import get_architecture
from .untyped_functions import untyped_functions

__all__ = [
    # Structure/Navigation
    "list_modules",
    "list_classes",
    "list_functions",
    "describe_class",
    "get_docstring",
    "get_method_signature",
    "get_class_hierarchy",
    "get_package_structure",
    # Search/Find
    "find_usages",
    "find_subclasses",
    "find_callers",
    "find_callees",
    "find_decorators",  # re-exported from patterns
    "find_tests",
    # Analysis
    "analyze_dependencies",
    "get_imports",
    "get_external_deps",  # re-exported from dependencies
    "predict_impact",
    "get_complexity",
    "get_magic_methods",  # re-exported from patterns
    "get_interfaces",  # re-exported from patterns
    "get_public_api",  # re-exported from patterns
    "get_type_hints",
    "get_api_docs",
    "get_exceptions",
    "get_architecture",
    "untyped_functions",
]
