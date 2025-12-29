"""
CADSL Loader - Load tools from CADSL source files.

This module provides functions to load CADSL tool definitions from:
- String sources
- Individual files
- Directories of .cadsl files

Loaded tools are automatically registered in the global Registry.
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import logging

from .parser import parse_cadsl, parse_cadsl_file, ParseResult
from .validator import validate_cadsl, ValidationResult
from .transformer import (
    CADSLTransformer,
    PipelineBuilder,
    ToolSpec as CADSLToolSpec,
)
from .python_executor import SecurityContext, SecurityLevel, SecurePythonStep


logger = logging.getLogger(__name__)


# ============================================================
# LOAD RESULT
# ============================================================

@dataclass
class LoadResult:
    """Result of loading CADSL tools."""
    success: bool
    tools_loaded: int = 0
    tool_names: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    source: Optional[str] = None  # File path or "string"

    def __bool__(self) -> bool:
        return self.success


# ============================================================
# REGISTRY-COMPATIBLE TOOL SPEC
# ============================================================

@dataclass
class RegisteredToolSpec:
    """
    Tool specification compatible with the DSL Registry.

    This wraps a CADSL ToolSpec and provides the interface expected
    by the Registry class.
    """
    name: str
    type: "ToolType"
    description: str
    pipeline_factory: Callable
    params: Dict[str, "ParamSpec"] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    # Source information
    source_file: Optional[str] = None
    source_type: str = "cadsl"


class ToolType:
    """Tool type enum compatible with registry."""
    QUERY = "query"
    DETECTOR = "detector"
    DIAGRAM = "diagram"

    def __init__(self, value: str):
        self.value = value

    @classmethod
    def from_string(cls, s: str) -> "ToolType":
        return cls(s.lower())


@dataclass
class ParamSpec:
    """Parameter specification compatible with registry."""
    name: str
    type: type
    required: bool = False
    default: Any = None
    description: str = ""
    choices: Optional[List[Any]] = None

    @classmethod
    def from_cadsl(cls, param) -> "ParamSpec":
        """Convert CADSL ParamSpec to registry ParamSpec."""
        type_map = {
            "int": int,
            "str": str,
            "float": float,
            "bool": bool,
            "list": list,
        }
        param_type = type_map.get(param.type, str)

        return cls(
            name=param.name,
            type=param_type,
            required=param.required,
            default=param.default,
            choices=param.choices,
        )


# ============================================================
# PARAMETER RESOLUTION HELPERS
# ============================================================

def _resolve_param_ref(value: Any, ctx: "Context", default: Any = None) -> Any:
    """Resolve a parameter reference like '{param_name}' from context.

    CADSL RAG params may contain parameter references like '{n_clusters}'
    which need to be resolved from ctx.params at runtime.

    Args:
        value: The value to resolve (may be a param ref string or literal)
        ctx: The pipeline context containing params
        default: Default value if param not found

    Returns:
        Resolved value from ctx.params, or original value if not a param ref
    """
    if isinstance(value, str) and value.startswith("{") and value.endswith("}"):
        param_name = value[1:-1]  # Extract "param_name" from "{param_name}"
        resolved = ctx.params.get(param_name, default)
        return resolved
    return value if value is not None else default


def _resolve_rag_params(params: Dict[str, Any], ctx: "Context") -> Dict[str, Any]:
    """Resolve all parameter references in RAG params dict.

    Args:
        params: RAG parameters from CADSL (may contain param refs)
        ctx: Pipeline context

    Returns:
        Dict with all param refs resolved to actual values
    """
    defaults = {
        "n_clusters": 50,
        "min_size": 2,
        "top_k": 10,
        "similarity": 0.85,
        "limit": 50,
        "exclude_same_file": True,
        "exclude_same_class": True,
        "query": "",
    }

    resolved = {}
    for key, value in params.items():
        default = defaults.get(key)
        resolved[key] = _resolve_param_ref(value, ctx, default)

    return resolved


# ============================================================
# PIPELINE FACTORY BUILDER
# ============================================================

def build_pipeline_factory(spec: CADSLToolSpec,
                           security_context: Optional[SecurityContext] = None
                           ) -> Callable:
    """
    Build a pipeline factory function from a CADSL ToolSpec.

    The factory returns a Pipeline when called with a Context.

    Args:
        spec: CADSL tool specification
        security_context: Security settings for Python blocks

    Returns:
        Callable that creates Pipeline objects
    """
    from codeine.dsl.core import (
        Pipeline, REQLSource, ValueSource,
        RAGSearchSource, RAGDuplicatesSource, RAGClustersSource,
        FilterStep, SelectStep, MapStep, FlatMapStep,
        OrderByStep, LimitStep, OffsetStep,
        GroupByStep, AggregateStep, FlattenStep, UniqueStep,
        TapStep, RenderStep, Context,
    )

    security_context = security_context or SecurityContext()

    def factory(ctx: Context) -> Pipeline:
        # Create source
        if spec.source_type == "reql":
            source = REQLSource(spec.source_content)
        elif spec.source_type == "rag_search":
            params = _resolve_rag_params(spec.rag_params, ctx)
            source = RAGSearchSource(
                query=params.get("query", ""),
                top_k=params.get("top_k", 10),
                entity_types=params.get("entity_types"),
            )
        elif spec.source_type == "rag_duplicates":
            params = _resolve_rag_params(spec.rag_params, ctx)
            source = RAGDuplicatesSource(
                similarity=params.get("similarity", 0.85),
                limit=params.get("limit", 50),
                exclude_same_file=params.get("exclude_same_file", True),
                exclude_same_class=params.get("exclude_same_class", True),
                entity_types=params.get("entity_types"),
            )
        elif spec.source_type == "rag_clusters":
            params = _resolve_rag_params(spec.rag_params, ctx)
            source = RAGClustersSource(
                n_clusters=params.get("n_clusters", 50),
                min_size=params.get("min_size", 2),
                exclude_same_file=params.get("exclude_same_file", True),
                exclude_same_class=params.get("exclude_same_class", True),
                entity_types=params.get("entity_types"),
            )
        elif spec.source_type == "value":
            source = ValueSource(spec.source_content)
        elif spec.source_type == "merge":
            from .transformer import MergeSource
            source = MergeSource(spec.merge_sources)
        else:
            source = ValueSource([])

        pipeline = Pipeline(_source=source)

        # Process each step
        emit_key = None
        for step_spec in spec.steps:
            step_type = step_spec.get("type")

            if step_type == "filter":
                predicate = step_spec.get("predicate", lambda r, ctx=None: True)
                # Wrap predicate to handle context
                def make_filter_pred(pred):
                    def wrapped(r):
                        try:
                            return pred(r, ctx)
                        except TypeError:
                            return pred(r)
                    return wrapped
                pipeline = pipeline.filter(make_filter_pred(predicate))

            elif step_type == "select":
                fields = step_spec.get("fields", {})
                pipeline = pipeline >> SelectStep(fields)

            elif step_type == "map":
                transform = step_spec.get("transform", lambda r, ctx=None: r)
                def make_map_fn(t):
                    def wrapped(r):
                        try:
                            return t(r, ctx)
                        except TypeError:
                            return t(r)
                    return wrapped
                pipeline = pipeline.map(make_map_fn(transform))

            elif step_type == "flat_map":
                transform = step_spec.get("transform", lambda r, ctx=None: [r])
                def make_flatmap_fn(t):
                    def wrapped(r):
                        try:
                            return t(r, ctx)
                        except TypeError:
                            return t(r)
                    return wrapped
                pipeline = pipeline.flat_map(make_flatmap_fn(transform))

            elif step_type == "order_by":
                orders = step_spec.get("orders", [])
                for field_name, desc in orders:
                    if desc:
                        pipeline = pipeline.order_by("-" + field_name.lstrip("-"))
                    else:
                        pipeline = pipeline.order_by(field_name)

            elif step_type == "limit":
                if "param" in step_spec:
                    param = step_spec["param"]
                    count = ctx.params.get(param, 100)
                else:
                    count = step_spec.get("count", 100)
                pipeline = pipeline.limit(count)

            elif step_type == "offset":
                if "param" in step_spec:
                    param = step_spec["param"]
                    count = ctx.params.get(param, 0)
                else:
                    count = step_spec.get("count", 0)
                pipeline = pipeline.offset(count)

            elif step_type == "group_by":
                field_name = step_spec.get("field")
                key_fn = step_spec.get("key_fn")
                if field_name == "_all":
                    pipeline = pipeline.group_by(field=None)
                else:
                    pipeline = pipeline.group_by(field=field_name, key=key_fn)

            elif step_type == "aggregate":
                aggs = step_spec.get("aggregations", {})
                pipeline = pipeline >> AggregateStep(aggs)

            elif step_type == "flatten":
                pipeline = pipeline.flatten()

            elif step_type == "unique":
                key = step_spec.get("key")
                pipeline = pipeline.unique(key)

            elif step_type == "python":
                code = step_spec.get("code", "result = rows")
                python_step = SecurePythonStep(code, security_context)
                pipeline = pipeline >> python_step

            elif step_type == "render":
                fmt = step_spec.get("format", "text")
                if "format_param" in step_spec:
                    fmt = ctx.params.get(step_spec["format_param"], fmt)
                renderer_name = step_spec.get("renderer")
                renderer = _get_renderer(renderer_name)
                pipeline = pipeline.render(fmt, renderer)

            elif step_type == "emit":
                emit_key = step_spec.get("key", "result")

            elif step_type == "when":
                # Conditional execution
                from .transformer import WhenStep
                condition = step_spec.get("condition", lambda r, ctx=None: True)
                inner_step = step_spec.get("inner_step")
                if inner_step:
                    pipeline = pipeline >> WhenStep(condition, inner_step)

            elif step_type == "unless":
                # Inverted conditional
                from .transformer import UnlessStep
                condition = step_spec.get("condition", lambda r, ctx=None: False)
                inner_step = step_spec.get("inner_step")
                if inner_step:
                    pipeline = pipeline >> UnlessStep(condition, inner_step)

            elif step_type == "branch":
                # Branching
                from .transformer import BranchStep
                condition = step_spec.get("condition", lambda r, ctx=None: True)
                then_step = step_spec.get("then_step")
                else_step = step_spec.get("else_step")
                pipeline = pipeline >> BranchStep(condition, then_step, else_step)

            elif step_type == "catch":
                # Error handling
                from .transformer import CatchStep
                default_fn = step_spec.get("default", lambda r, ctx=None: [])
                pipeline = pipeline >> CatchStep(default_fn)

            elif step_type == "parallel":
                # Execute multiple steps in parallel
                from .transformer import ParallelStep
                inner_steps = step_spec.get("steps", [])
                pipeline = pipeline >> ParallelStep(inner_steps)

            elif step_type == "collect":
                # Group and aggregate
                from .transformer import CollectStep
                by_field = step_spec.get("by")
                fields = step_spec.get("fields", {})
                pipeline = pipeline >> CollectStep(by_field, fields)

            elif step_type == "cross_join":
                # Cartesian product
                from .transformer import CrossJoinStep
                pipeline = pipeline >> CrossJoinStep(
                    unique_pairs=step_spec.get("unique_pairs", True),
                    exclude_self=step_spec.get("exclude_self", True),
                    left_prefix=step_spec.get("left_prefix", "left_"),
                    right_prefix=step_spec.get("right_prefix", "right_"),
                )

            elif step_type == "set_similarity":
                # Set similarity calculation
                from .transformer import SetSimilarityStep
                pipeline = pipeline >> SetSimilarityStep(
                    left_col=step_spec.get("left_col"),
                    right_col=step_spec.get("right_col"),
                    sim_type=step_spec.get("sim_type", "jaccard"),
                    output=step_spec.get("output", "similarity"),
                    intersection_output=step_spec.get("intersection_output"),
                    union_output=step_spec.get("union_output"),
                )

            elif step_type == "compute":
                # Add computed fields
                from .transformer import ComputeStep
                pipeline = pipeline >> ComputeStep(step_spec.get("fields", {}))

        if emit_key:
            pipeline = pipeline.emit(emit_key)

        return pipeline

    return factory


def _get_renderer(name: Optional[str]) -> Callable:
    """Get a renderer function by name."""
    import json

    def json_renderer(data, fmt):
        return json.dumps(data, indent=2, default=str)

    def text_renderer(data, fmt):
        return str(data)

    def markdown_renderer(data, fmt):
        if isinstance(data, list):
            return "\n".join(f"- {item}" for item in data)
        return str(data)

    def mermaid_renderer(data, fmt):
        # Basic mermaid support
        if isinstance(data, dict) and "diagram" in data:
            return data["diagram"]
        return str(data)

    renderers = {
        "json": json_renderer,
        "text": text_renderer,
        "markdown": markdown_renderer,
        "mermaid": mermaid_renderer,
        None: text_renderer,
    }

    return renderers.get(name, text_renderer)


# ============================================================
# LOADING FUNCTIONS
# ============================================================

def load_tool(source: str,
              security_level: SecurityLevel = SecurityLevel.STANDARD,
              register: bool = True) -> LoadResult:
    """
    Load a single tool from CADSL source string.

    Args:
        source: CADSL source code
        security_level: Security level for Python blocks
        register: Whether to register in global Registry

    Returns:
        LoadResult with loaded tool info
    """
    # Parse
    parse_result = parse_cadsl(source)
    if not parse_result.success:
        return LoadResult(
            success=False,
            errors=[str(e) for e in parse_result.errors],
            source="string"
        )

    # Validate
    validation = validate_cadsl(parse_result.tree)
    if not validation.valid:
        return LoadResult(
            success=False,
            errors=[str(i) for i in validation.errors],
            warnings=[str(i) for i in validation.warnings],
            source="string"
        )

    # Transform
    transformer = CADSLTransformer()
    cadsl_specs = transformer.transform(parse_result.tree)

    if not cadsl_specs:
        return LoadResult(
            success=False,
            errors=["No tools found in source"],
            source="string"
        )

    # Build and optionally register
    security_context = SecurityContext(level=security_level)
    tool_names = []

    for cadsl_spec in cadsl_specs:
        # Convert to registry-compatible spec
        reg_spec = _to_registry_spec(cadsl_spec, security_context)

        if register:
            _register_tool(reg_spec)

        tool_names.append(cadsl_spec.name)

    return LoadResult(
        success=True,
        tools_loaded=len(tool_names),
        tool_names=tool_names,
        warnings=[str(i) for i in validation.warnings],
        source="string"
    )


def load_tool_file(path: Union[str, Path],
                   security_level: SecurityLevel = SecurityLevel.STANDARD,
                   register: bool = True) -> LoadResult:
    """
    Load tools from a CADSL file.

    Args:
        path: Path to .cadsl file
        security_level: Security level for Python blocks
        register: Whether to register in global Registry

    Returns:
        LoadResult with loaded tool info
    """
    path = Path(path)

    if not path.exists():
        return LoadResult(
            success=False,
            errors=[f"File not found: {path}"],
            source=str(path)
        )

    if not path.suffix == ".cadsl":
        logger.warning(f"File {path} does not have .cadsl extension")

    # Parse
    parse_result = parse_cadsl_file(path)
    if not parse_result.success:
        return LoadResult(
            success=False,
            errors=[str(e) for e in parse_result.errors],
            source=str(path)
        )

    # Validate
    validation = validate_cadsl(parse_result.tree)
    if not validation.valid:
        return LoadResult(
            success=False,
            errors=[str(i) for i in validation.errors],
            warnings=[str(i) for i in validation.warnings],
            source=str(path)
        )

    # Transform
    transformer = CADSLTransformer()
    cadsl_specs = transformer.transform(parse_result.tree)

    if not cadsl_specs:
        return LoadResult(
            success=True,  # Empty file is valid
            tools_loaded=0,
            warnings=["No tools found in file"],
            source=str(path)
        )

    # Build and optionally register
    security_context = SecurityContext(level=security_level)
    tool_names = []

    for cadsl_spec in cadsl_specs:
        reg_spec = _to_registry_spec(cadsl_spec, security_context, str(path))

        if register:
            _register_tool(reg_spec)

        tool_names.append(cadsl_spec.name)

    return LoadResult(
        success=True,
        tools_loaded=len(tool_names),
        tool_names=tool_names,
        warnings=[str(i) for i in validation.warnings],
        source=str(path)
    )


def load_tools_directory(path: Union[str, Path],
                         pattern: str = "*.cadsl",
                         recursive: bool = False,
                         security_level: SecurityLevel = SecurityLevel.STANDARD,
                         register: bool = True) -> LoadResult:
    """
    Load tools from all CADSL files in a directory.

    Args:
        path: Directory path
        pattern: Glob pattern for CADSL files
        recursive: Whether to search subdirectories
        security_level: Security level for Python blocks
        register: Whether to register in global Registry

    Returns:
        LoadResult with aggregated results
    """
    path = Path(path)

    if not path.exists():
        return LoadResult(
            success=False,
            errors=[f"Directory not found: {path}"],
            source=str(path)
        )

    if not path.is_dir():
        return LoadResult(
            success=False,
            errors=[f"Not a directory: {path}"],
            source=str(path)
        )

    # Find files
    if recursive:
        files = list(path.rglob(pattern))
    else:
        files = list(path.glob(pattern))

    if not files:
        return LoadResult(
            success=True,
            tools_loaded=0,
            warnings=[f"No {pattern} files found in {path}"],
            source=str(path)
        )

    # Load each file
    all_errors = []
    all_warnings = []
    all_tool_names = []
    total_loaded = 0

    for file in sorted(files):
        result = load_tool_file(file, security_level, register)

        if result.success:
            total_loaded += result.tools_loaded
            all_tool_names.extend(result.tool_names)
        else:
            all_errors.extend([f"{file}: {e}" for e in result.errors])

        all_warnings.extend([f"{file}: {w}" for w in result.warnings])

    return LoadResult(
        success=len(all_errors) == 0,
        tools_loaded=total_loaded,
        tool_names=all_tool_names,
        errors=all_errors,
        warnings=all_warnings,
        source=str(path)
    )


# ============================================================
# REGISTRY INTEGRATION
# ============================================================

def _to_registry_spec(cadsl_spec: CADSLToolSpec,
                      security_context: SecurityContext,
                      source_file: Optional[str] = None) -> RegisteredToolSpec:
    """Convert CADSL ToolSpec to registry-compatible spec."""
    # Build pipeline factory
    factory = build_pipeline_factory(cadsl_spec, security_context)

    # Convert params
    params = {}
    for param in cadsl_spec.params:
        params[param.name] = ParamSpec.from_cadsl(param)

    # Build metadata
    meta = cadsl_spec.metadata.copy()
    meta["description"] = cadsl_spec.description

    return RegisteredToolSpec(
        name=cadsl_spec.name,
        type=ToolType.from_string(cadsl_spec.tool_type),
        description=cadsl_spec.description,
        pipeline_factory=factory,
        params=params,
        meta=meta,
        source_file=source_file,
        source_type="cadsl",
    )


def _register_tool(spec: RegisteredToolSpec) -> None:
    """Register a tool in the global Registry."""
    try:
        from codeine.dsl.registry import Registry
        Registry.register(spec)
        logger.debug(f"Registered CADSL tool: {spec.name}")
    except ImportError:
        logger.warning("Could not import Registry - tool not registered globally")


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def load_cadsl(source: str) -> LoadResult:
    """Convenience function to load tools from a CADSL string."""
    return load_tool(source)


def load_cadsl_file(path: Union[str, Path]) -> LoadResult:
    """Convenience function to load tools from a CADSL file."""
    return load_tool_file(path)


def load_cadsl_directory(path: Union[str, Path],
                         recursive: bool = False) -> LoadResult:
    """Convenience function to load tools from a directory."""
    return load_tools_directory(path, recursive=recursive)


# ============================================================
# TOOL EXECUTION
# ============================================================

def execute_tool(name: str, ctx: Any = None, **params) -> Dict[str, Any]:
    """
    Execute a registered tool by name.

    Args:
        name: Tool name
        ctx: Execution context (or will be created)
        **params: Tool parameters

    Returns:
        Execution result dict
    """
    try:
        from codeine.dsl.registry import Registry
        from codeine.dsl.core import Context

        spec = Registry.get(name)
        if spec is None:
            return {"success": False, "error": f"Tool not found: {name}"}

        # Create context if needed
        if ctx is None:
            ctx = Context(reter=None, params=params)
        elif params:
            ctx = ctx.with_params(**params)

        # Build and execute pipeline
        pipeline = spec.pipeline_factory(ctx)
        return pipeline.execute(ctx)

    except Exception as e:
        return {"success": False, "error": str(e)}
