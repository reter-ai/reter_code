"""
CADSL - Reter Code Analysis DSL

A text-based domain-specific language for defining code analysis tools
that compile to executable Python Pipeline objects.

Example::

    from reter_code.cadsl import parse_cadsl, validate_cadsl, transform_cadsl

    source = '''
    query list_modules() {
        param limit: int = 100;

        reql {
            SELECT ?m ?name WHERE { ?m type oo:Module . ?m name ?name }
        }
        | select { name }
        | emit { modules }
    }
    '''

    # Parse and validate
    result = parse_cadsl(source)
    if result.success:
        validation = validate_cadsl(result.tree)
        if validation.valid:
            # Transform to executable specs
            tools = transform_cadsl(result.tree)
            for tool in tools:
                print(f"Tool: {tool.name} ({tool.tool_type})")
"""

from .parser import (
    # Classes
    CADSLParser,
    ParseResult,
    ParseError,
    # Functions
    parse_cadsl,
    parse_cadsl_file,
    # Utilities
    pretty_print_tree,
    get_tool_names,
    count_tools,
)

from .validator import (
    # Classes
    CADSLValidator,
    ValidationResult,
    ValidationIssue,
    Severity,
    # Functions
    validate_cadsl,
    # Constants
    VALID_TYPES,
    VALID_SECURITY_LEVELS,
    VALID_METADATA_KEYS,
    VALID_SEVERITIES,
    VALID_CATEGORIES,
)

from .compiler import (
    # Classes
    ExpressionCompiler,
    ConditionCompiler,
    ObjectExprCompiler,
    # Functions
    compile_expression,
    compile_condition,
    compile_object_expr,
    # Constants
    BUILTINS,
)

from .transformer import (
    # Classes
    CADSLTransformer,
    PipelineBuilder,
    ToolSpec,
    ParamSpec,
    PythonStep,
    # Functions
    transform_cadsl,
    build_pipeline,
)

from .python_executor import (
    # Classes
    PythonExecutor,
    SecurityContext,
    SecurityLevel,
    ASTValidator,
    SecurePythonStep,
    ExecutionResult,
    # Functions
    validate_python_code,
    execute_python_safely,
    validate_imports,
    get_safe_builtins,
)

from .builtins import (
    # Functions registry
    CADSL_BUILTINS,
    get_cadsl_builtins,
    # Commonly used functions
    group_by,
    unique_by,
    flatten,
    pluck,
    average,
    percentage,
    is_private,
    is_dunder,
)

from .loader import (
    # Classes
    LoadResult,
    RegisteredToolSpec,
    ToolType,
    ParamSpec as LoaderParamSpec,
    # Functions
    load_tool,
    load_tool_file,
    load_tools_directory,
    load_cadsl,
    load_cadsl_file,
    load_cadsl_directory,
    execute_tool,
    build_pipeline_factory,
)

__all__ = [
    # Parser
    "CADSLParser",
    "ParseResult",
    "ParseError",
    "parse_cadsl",
    "parse_cadsl_file",
    "pretty_print_tree",
    "get_tool_names",
    "count_tools",
    # Validator
    "CADSLValidator",
    "ValidationResult",
    "ValidationIssue",
    "Severity",
    "validate_cadsl",
    "VALID_TYPES",
    "VALID_SECURITY_LEVELS",
    "VALID_METADATA_KEYS",
    "VALID_SEVERITIES",
    "VALID_CATEGORIES",
    # Compiler
    "ExpressionCompiler",
    "ConditionCompiler",
    "ObjectExprCompiler",
    "compile_expression",
    "compile_condition",
    "compile_object_expr",
    "BUILTINS",
    # Transformer
    "CADSLTransformer",
    "PipelineBuilder",
    "ToolSpec",
    "ParamSpec",
    "PythonStep",
    "transform_cadsl",
    "build_pipeline",
    # Python Executor (Security Sandbox)
    "PythonExecutor",
    "SecurityContext",
    "SecurityLevel",
    "ASTValidator",
    "SecurePythonStep",
    "ExecutionResult",
    "validate_python_code",
    "execute_python_safely",
    "validate_imports",
    "get_safe_builtins",
    # Builtins
    "CADSL_BUILTINS",
    "get_cadsl_builtins",
    "group_by",
    "unique_by",
    "flatten",
    "pluck",
    "average",
    "percentage",
    "is_private",
    "is_dunder",
    # Loader
    "LoadResult",
    "RegisteredToolSpec",
    "ToolType",
    "LoaderParamSpec",
    "load_tool",
    "load_tool_file",
    "load_tools_directory",
    "load_cadsl",
    "load_cadsl_file",
    "load_cadsl_directory",
    "execute_tool",
    "build_pipeline_factory",
]

__version__ = "0.4.0"
