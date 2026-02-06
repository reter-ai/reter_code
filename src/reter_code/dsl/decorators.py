"""
CADSL Decorators - Tool Definition Decorators

This module provides decorators for defining CADSL tools:
- @query: Define a read-only code inspection tool
- @detector: Define a code smell/pattern detector
- @diagram: Define a visualization generator
- @param: Define a tool parameter
- @meta: Add metadata to a tool
"""

from functools import wraps
from typing import Callable, Any, Dict, List, Optional, Type, Union
from enum import Enum

from .core import (
    Pipeline, Context, ToolSpec, ToolType, ParamSpec,
    Query, Detector, Diagram
)
from .catpy import Ok, Err, PipelineResult
from .registry import Registry


def _make_tool_wrapper(tool, params: Dict[str, Any]) -> Callable:
    """Create a wrapper function for tool execution.

    This is the shared implementation for all tool decorator wrappers.
    """
    def wrapper(ctx: Optional["Context"] = None, **kwargs) -> Dict[str, Any]:
        if ctx is None:
            ctx = Context(
                reter=kwargs.pop("reter", None),
                params=kwargs,
                instance_name=kwargs.pop("instance_name", "default")
            )
        else:
            ctx.params.update(kwargs)

        for param_name, param_spec in params.items():
            value = ctx.params.get(param_name)
            result = param_spec.validate(value)
            if result.is_err():
                return {"success": False, "error": str(result)}
            ctx.params[param_name] = result.unwrap()

        return tool.execute(ctx)
    return wrapper


def query(name: str, description: str = "") -> Callable:
    """
    Decorator to define a query (read-only code inspection) tool.

    The decorated function should accept a Pipeline and return a Pipeline.

    Example:
        @query("list_modules")
        def list_modules(p: Pipeline) -> Pipeline:
            return (
                p.reql("SELECT ?m ?name WHERE { ?m type module }")
                .select("name", "file")
                .emit("modules")
            )

    Args:
        name: Unique tool name (used for lookup)
        description: Human-readable description

    Returns:
        Decorated function that can be executed as a query tool
    """
    def decorator(func: Callable[[Pipeline], Pipeline]) -> Callable:
        # Get description from docstring if not provided
        desc = description or (func.__doc__ or "").strip().split("\n")[0]

        # Extract params from function metadata
        params = getattr(func, "_cadsl_params", {})
        meta = getattr(func, "_cadsl_meta", {})

        def pipeline_factory(ctx: Context) -> Pipeline:
            """Create initial pipeline with context params."""
            # Call the function - it returns a Pipeline directly
            return func()

        spec = ToolSpec(
            name=name,
            type=ToolType.QUERY,
            description=desc,
            pipeline_factory=pipeline_factory,
            params=params,
            meta=meta
        )

        tool = Query(spec)

        # Register with global registry
        Registry.register(spec)

        wrapper = wraps(func)(_make_tool_wrapper(tool, params))
        wrapper._cadsl_spec = spec
        wrapper._cadsl_tool = tool
        return wrapper

    return decorator


def detector(name: str, description: str = "", *, category: str = "", severity: str = "") -> Callable:
    """
    Decorator to define a detector (code smell/pattern finder) tool.

    The decorated function should accept a Pipeline and return a Pipeline
    that produces findings.

    Example:
        @detector("find_large_classes", category="code_smell", severity="high")
        @param("threshold", int, default=20)
        def find_large_classes(p: Pipeline) -> Pipeline:
            return (
                p.reql("SELECT ?c ?count WHERE { ... }")
                .filter(lambda r: r["count"] > p.ctx.params["threshold"])
                .emit("findings")
            )

    Args:
        name: Unique tool name
        description: Human-readable description
        category: Tool category (e.g., "code_smell", "design", "complexity")
        severity: Finding severity (e.g., "low", "medium", "high", "critical", "info")

    Returns:
        Decorated function that can be executed as a detector tool
    """
    def decorator(func: Callable[[Pipeline], Pipeline]) -> Callable:
        desc = description or (func.__doc__ or "").strip().split("\n")[0]
        params = getattr(func, "_cadsl_params", {})
        meta = getattr(func, "_cadsl_meta", {})

        # Add category and severity to meta if provided
        if category:
            meta["category"] = category
        if severity:
            meta["severity"] = severity

        def pipeline_factory(ctx: Context) -> Pipeline:
            # Call the function - it returns a Pipeline directly
            return func()

        spec = ToolSpec(
            name=name,
            type=ToolType.DETECTOR,
            description=desc,
            pipeline_factory=pipeline_factory,
            params=params,
            meta=meta
        )

        tool = Detector(spec)
        Registry.register(spec)

        wrapper = wraps(func)(_make_tool_wrapper(tool, params))
        wrapper._cadsl_spec = spec
        wrapper._cadsl_tool = tool
        return wrapper

    return decorator


def diagram(name: str, description: str = "") -> Callable:
    """
    Decorator to define a diagram (visualization) tool.

    The decorated function should accept a Pipeline and return a Pipeline
    that produces visualization output.

    Example:
        @diagram("class_hierarchy")
        @param("root", str, required=False)
        @param("format", str, default="mermaid", choices=["mermaid", "json"])
        def class_hierarchy(p: Pipeline) -> Pipeline:
            return (
                p.reql("SELECT ?c ?parent WHERE { ... }")
                .emit("diagram")
            )

    Args:
        name: Unique tool name
        description: Human-readable description

    Returns:
        Decorated function that can be executed as a diagram tool
    """
    def decorator(func: Callable[[Pipeline], Pipeline]) -> Callable:
        desc = description or (func.__doc__ or "").strip().split("\n")[0]
        params = getattr(func, "_cadsl_params", {})
        meta = getattr(func, "_cadsl_meta", {})

        def pipeline_factory(ctx: Context) -> Pipeline:
            # Call the function - it returns a Pipeline directly
            return func()

        spec = ToolSpec(
            name=name,
            type=ToolType.DIAGRAM,
            description=desc,
            pipeline_factory=pipeline_factory,
            params=params,
            meta=meta
        )

        tool = Diagram(spec)
        Registry.register(spec)

        wrapper = wraps(func)(_make_tool_wrapper(tool, params))
        wrapper._cadsl_spec = spec
        wrapper._cadsl_tool = tool
        return wrapper

    return decorator


def param(
    name: str,
    type_: Type = str,
    *,
    required: bool = True,
    default: Any = None,
    description: str = "",
    choices: Optional[List[Any]] = None
) -> Callable:
    """
    Decorator to define a parameter for a tool.

    Must be applied BEFORE the tool decorator (@query, @detector, @diagram).

    Example:
        @query("list_classes")
        @param("module", str, required=False, description="Filter by module")
        @param("limit", int, default=100)
        def list_classes(p: Pipeline) -> Pipeline:
            ...

    Args:
        name: Parameter name
        type_: Parameter type (str, int, float, bool)
        required: Whether parameter is required
        default: Default value if not provided
        description: Human-readable description
        choices: List of valid values (for enum-like params)

    Returns:
        Decorator that adds parameter metadata to function
    """
    spec = ParamSpec(
        name=name,
        type=type_,
        required=required,
        default=default,
        description=description,
        choices=choices
    )

    def decorator(func: Callable) -> Callable:
        # Initialize params dict if needed
        if not hasattr(func, "_cadsl_params"):
            func._cadsl_params = {}
        func._cadsl_params[name] = spec
        return func

    return decorator


def meta(**kwargs) -> Callable:
    """
    Decorator to add metadata to a tool.

    Must be applied BEFORE the tool decorator (@query, @detector, @diagram).

    Common metadata keys:
    - category: Tool category (e.g., "code_smell", "architecture")
    - severity: Finding severity (e.g., "low", "medium", "high", "critical")
    - tags: List of tags for filtering
    - version: Tool version

    Example:
        @detector("find_god_class")
        @meta(category="code_smell", severity="high", tags=["oop", "design"])
        def find_god_class(p: Pipeline) -> Pipeline:
            ...

    Args:
        **kwargs: Metadata key-value pairs

    Returns:
        Decorator that adds metadata to function
    """
    def decorator(func: Callable) -> Callable:
        if not hasattr(func, "_cadsl_meta"):
            func._cadsl_meta = {}
        func._cadsl_meta.update(kwargs)
        return func

    return decorator


# =============================================================================
# Alternative Decorator API (for more complex tools)
# =============================================================================

class ToolBuilder:
    """
    Builder for creating tools with more control.

    Example:
        list_modules = (
            ToolBuilder("list_modules", ToolType.QUERY)
            .description("List all modules")
            .param("limit", int, default=100)
            .pipeline(lambda p: p.reql("...").emit("modules"))
            .build()
        )

    ::: This is-in-layer Domain-Specific-Language-Layer.
    ::: This is a builder.
    """

    def __init__(self, name: str, type_: ToolType):
        self._name = name
        self._type = type_
        self._description = ""
        self._params: Dict[str, ParamSpec] = {}
        self._meta: Dict[str, Any] = {}
        self._pipeline_fn: Optional[Callable[[Pipeline], Pipeline]] = None

    def description(self, desc: str) -> "ToolBuilder":
        """Set tool description."""
        self._description = desc
        return self

    def param(
        self,
        name: str,
        type_: Type = str,
        *,
        required: bool = True,
        default: Any = None,
        description: str = "",
        choices: Optional[List[Any]] = None
    ) -> "ToolBuilder":
        """Add a parameter."""
        self._params[name] = ParamSpec(
            name=name,
            type=type_,
            required=required,
            default=default,
            description=description,
            choices=choices
        )
        return self

    def meta(self, **kwargs) -> "ToolBuilder":
        """Add metadata."""
        self._meta.update(kwargs)
        return self

    def pipeline(self, fn: Callable[[Pipeline], Pipeline]) -> "ToolBuilder":
        """Set the pipeline factory function."""
        self._pipeline_fn = fn
        return self

    def build(self) -> Callable:
        """Build and register the tool."""
        if not self._pipeline_fn:
            raise ValueError("Pipeline function not set")

        def pipeline_factory(ctx: Context) -> Pipeline:
            p = Pipeline.from_value([])
            return self._pipeline_fn(p)

        spec = ToolSpec(
            name=self._name,
            type=self._type,
            description=self._description,
            pipeline_factory=pipeline_factory,
            params=self._params,
            meta=self._meta
        )

        if self._type == ToolType.QUERY:
            tool = Query(spec)
        elif self._type == ToolType.DETECTOR:
            tool = Detector(spec)
        else:
            tool = Diagram(spec)

        Registry.register(spec)

        wrapper = _make_tool_wrapper(tool, self._params)
        wrapper._cadsl_spec = spec
        wrapper._cadsl_tool = tool
        return wrapper


# Convenience builder functions
def query_builder(name: str) -> ToolBuilder:
    """Create a query tool builder."""
    return ToolBuilder(name, ToolType.QUERY)


def detector_builder(name: str) -> ToolBuilder:
    """Create a detector tool builder."""
    return ToolBuilder(name, ToolType.DETECTOR)


def diagram_builder(name: str) -> ToolBuilder:
    """Create a diagram tool builder."""
    return ToolBuilder(name, ToolType.DIAGRAM)
