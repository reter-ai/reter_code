"""
CADSL - Code Analysis DSL (Python Embedded Version)

A fluent Python DSL for defining code analysis tools, built on
category theory foundations (Functor, Applicative, Monad).

Tool Types:
- query: Code inspection (no side effects)
- detector: Recommender (creates findings/tasks)
- diagram: Visualization (renders to format)

Category Theory:
- Pipeline is a Monad with bind/fmap/pure
- Steps are Kleisli arrows: T -> Result[U, Error]
- Error propagation via Result monad (Ok/Err)

Example:
    @query("list_modules")
    def list_modules(p: Pipeline) -> Pipeline:
        return (
            p.reql('''
                SELECT ?m ?name ?file
                WHERE { ?m type oo:Module . ?m name ?name . ?m inFile ?file }
            ''')
            .select(name="name", file="file", qualified_name="m")
            .order_by("file")
            .emit("modules")
        )
"""

# Category theory foundations
from .catpy import (
    Functor, Applicative, Monad,
    Result, Ok, Err,
    Maybe, Just, Nothing,
    ListF,
    PipelineError, PipelineResult,
    pipeline_ok, pipeline_err,
    compose as cat_compose, identity as cat_identity
)

# Core pipeline types
from .core import (
    Pipeline, Query, Detector, Diagram,
    Context, ToolSpec, ParamSpec, ToolType,
    Step, Source,
    reql, rag, value
)

# Decorators for tool definition
from .decorators import (
    query, detector, diagram, param, meta,
    ToolBuilder, query_builder, detector_builder, diagram_builder
)

# Operators for flow control
from .operators import (
    when, unless, branch, merge, identity, tap, catch, parallel, compose
)

# Registry for tool discovery
from .registry import Registry, Namespace, namespace, load_tools_from_module

__all__ = [
    # Category Theory (from catpy)
    "Functor",
    "Applicative",
    "Monad",
    "Result",
    "Ok",
    "Err",
    "Maybe",
    "Just",
    "Nothing",
    "ListF",
    "PipelineError",
    "PipelineResult",
    "pipeline_ok",
    "pipeline_err",
    # Core Types
    "Pipeline",
    "Query",
    "Detector",
    "Diagram",
    "Context",
    "ToolSpec",
    "ParamSpec",
    "ToolType",
    "Step",
    "Source",
    # Source Builders
    "reql",
    "rag",
    "value",
    # Decorators
    "query",
    "detector",
    "diagram",
    "param",
    "meta",
    # Builder API
    "ToolBuilder",
    "query_builder",
    "detector_builder",
    "diagram_builder",
    # Operators
    "when",
    "unless",
    "branch",
    "merge",
    "identity",
    "tap",
    "catch",
    "parallel",
    "compose",
    # Registry
    "Registry",
    "Namespace",
    "namespace",
    "load_tools_from_module",
]
