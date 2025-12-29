"""
CADSL Registry - Tool Registration and Discovery

This module provides a registry for CADSL tools, enabling:
- Tool registration and lookup
- Tool discovery by type and metadata
- Namespace management for tool collections
"""

from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, field
from enum import Enum


# Forward reference to avoid circular import
# Actual ToolSpec is in core.py
class ToolSpec:
    """Forward reference placeholder."""
    name: str
    type: Any
    meta: Dict[str, Any]


class ToolType(Enum):
    """Tool type enum (forward reference)."""
    QUERY = "query"
    DETECTOR = "detector"
    DIAGRAM = "diagram"


class Registry:
    """
    Global registry for CADSL tools.

    Tools are automatically registered when decorated with @query, @detector,
    or @diagram. Tools can be looked up by name, filtered by type, or
    discovered by metadata.

    Example:
        # Registration happens automatically via decorators
        @query("list_modules")
        def list_modules(p): ...

        # Lookup
        tool = Registry.get("list_modules")

        # Discovery
        queries = Registry.get_by_type(ToolType.QUERY)
        code_smells = Registry.get_by_category("code_smell")
    """

    # Class-level storage
    _tools: Dict[str, "ToolSpec"] = {}
    _by_type: Dict[str, List[str]] = {
        "query": [],
        "detector": [],
        "diagram": []
    }
    _by_category: Dict[str, List[str]] = {}
    _namespaces: Dict[str, "Registry"] = {}

    @classmethod
    def register(cls, spec: "ToolSpec") -> None:
        """
        Register a tool specification.

        Args:
            spec: Tool specification to register
        """
        name = spec.name

        # Prevent duplicate registration
        if name in cls._tools:
            # Update existing registration
            pass

        cls._tools[name] = spec

        # Index by type
        type_key = spec.type.value if hasattr(spec.type, "value") else str(spec.type)
        if type_key not in cls._by_type:
            cls._by_type[type_key] = []
        if name not in cls._by_type[type_key]:
            cls._by_type[type_key].append(name)

        # Index by category
        category = spec.meta.get("category") if hasattr(spec, "meta") else None
        if category:
            if category not in cls._by_category:
                cls._by_category[category] = []
            if name not in cls._by_category[category]:
                cls._by_category[category].append(name)

    @classmethod
    def get(cls, name: str) -> Optional["ToolSpec"]:
        """
        Get a tool by name.

        Args:
            name: Tool name

        Returns:
            ToolSpec if found, None otherwise
        """
        return cls._tools.get(name)

    @classmethod
    def get_by_type(cls, type_: "ToolType") -> List["ToolSpec"]:
        """
        Get all tools of a specific type.

        Args:
            type_: Tool type (QUERY, DETECTOR, DIAGRAM)

        Returns:
            List of matching tool specs
        """
        type_key = type_.value if hasattr(type_, "value") else str(type_)
        names = cls._by_type.get(type_key, [])
        return [cls._tools[n] for n in names if n in cls._tools]

    @classmethod
    def get_by_category(cls, category: str) -> List["ToolSpec"]:
        """
        Get all tools in a category.

        Args:
            category: Category name (e.g., "code_smell", "architecture")

        Returns:
            List of matching tool specs
        """
        names = cls._by_category.get(category, [])
        return [cls._tools[n] for n in names if n in cls._tools]

    @classmethod
    def get_by_metadata(cls, **kwargs) -> List["ToolSpec"]:
        """
        Get tools matching metadata criteria.

        Args:
            **kwargs: Metadata key-value pairs to match

        Returns:
            List of matching tool specs
        """
        results = []
        for spec in cls._tools.values():
            meta = getattr(spec, "meta", {})
            if all(meta.get(k) == v for k, v in kwargs.items()):
                results.append(spec)
        return results

    @classmethod
    def list_all(cls) -> List[str]:
        """
        List all registered tool names.

        Returns:
            List of tool names
        """
        return list(cls._tools.keys())

    @classmethod
    def list_by_type(cls, type_: "ToolType") -> List[str]:
        """
        List tool names of a specific type.

        Args:
            type_: Tool type

        Returns:
            List of tool names
        """
        type_key = type_.value if hasattr(type_, "value") else str(type_)
        return cls._by_type.get(type_key, []).copy()

    @classmethod
    def unregister(cls, name: str) -> bool:
        """
        Unregister a tool by name.

        Args:
            name: Tool name to unregister

        Returns:
            True if tool was found and removed, False otherwise
        """
        if name not in cls._tools:
            return False

        spec = cls._tools.pop(name)

        # Remove from type index
        type_key = spec.type.value if hasattr(spec.type, "value") else str(spec.type)
        if type_key in cls._by_type and name in cls._by_type[type_key]:
            cls._by_type[type_key].remove(name)

        # Remove from category index
        category = spec.meta.get("category") if hasattr(spec, "meta") else None
        if category and category in cls._by_category and name in cls._by_category[category]:
            cls._by_category[category].remove(name)

        return True

    @classmethod
    def clear(cls) -> None:
        """Clear all registered tools. Mainly for testing."""
        cls._tools.clear()
        cls._by_type = {"query": [], "detector": [], "diagram": []}
        cls._by_category.clear()

    @classmethod
    def stats(cls) -> Dict[str, Any]:
        """
        Get registry statistics.

        Returns:
            Dict with counts by type and category
        """
        return {
            "total": len(cls._tools),
            "by_type": {k: len(v) for k, v in cls._by_type.items()},
            "by_category": {k: len(v) for k, v in cls._by_category.items()},
            "categories": list(cls._by_category.keys())
        }


@dataclass
class Namespace:
    """
    A namespace for grouping related tools.

    Namespaces allow organizing tools into collections and loading
    them from different sources (files, packages, etc).

    Example:
        # Create a namespace for refactoring tools
        refactoring = Namespace("refactoring")

        @refactoring.detector("find_god_class")
        def find_god_class(p): ...

        # Load all tools from namespace
        refactoring.load()
    """
    name: str
    _tools: Dict[str, Any] = field(default_factory=dict)

    def query(self, name: str, description: str = "") -> Callable:
        """Register a query in this namespace."""
        from .decorators import query as query_decorator

        def wrapper(func):
            full_name = f"{self.name}:{name}"
            decorated = query_decorator(full_name, description)(func)
            self._tools[name] = decorated
            return decorated

        return wrapper

    def detector(self, name: str, description: str = "") -> Callable:
        """Register a detector in this namespace."""
        from .decorators import detector as detector_decorator

        def wrapper(func):
            full_name = f"{self.name}:{name}"
            decorated = detector_decorator(full_name, description)(func)
            self._tools[name] = decorated
            return decorated

        return wrapper

    def diagram(self, name: str, description: str = "") -> Callable:
        """Register a diagram in this namespace."""
        from .decorators import diagram as diagram_decorator

        def wrapper(func):
            full_name = f"{self.name}:{name}"
            decorated = diagram_decorator(full_name, description)(func)
            self._tools[name] = decorated
            return decorated

        return wrapper

    def get(self, name: str) -> Optional[Callable]:
        """Get a tool from this namespace."""
        return self._tools.get(name)

    def list_tools(self) -> List[str]:
        """List all tools in this namespace."""
        return list(self._tools.keys())


def namespace(name: str) -> Namespace:
    """
    Create or get a namespace.

    Args:
        name: Namespace name

    Returns:
        Namespace instance
    """
    if name not in Registry._namespaces:
        Registry._namespaces[name] = Namespace(name)
    return Registry._namespaces[name]


def load_tools_from_module(module_path: str) -> int:
    """
    Load tools from a Python module.

    The module should contain decorated tool functions.

    Args:
        module_path: Dotted module path (e.g., "codeine.tools.refactoring")

    Returns:
        Number of tools loaded
    """
    import importlib

    before = len(Registry._tools)
    module = importlib.import_module(module_path)
    after = len(Registry._tools)

    return after - before


def load_tools_from_directory(path: str, pattern: str = "*.py") -> int:
    """
    Load tools from Python files in a directory.

    Args:
        path: Directory path
        pattern: Glob pattern for Python files

    Returns:
        Number of tools loaded
    """
    import importlib.util
    from pathlib import Path

    before = len(Registry._tools)

    for file in Path(path).glob(pattern):
        if file.name.startswith("_"):
            continue

        spec = importlib.util.spec_from_file_location(file.stem, file)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

    after = len(Registry._tools)
    return after - before
