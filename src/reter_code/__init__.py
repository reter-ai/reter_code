"""
Reter Code - AI-powered code reasoning MCP server

A Model Context Protocol server that provides intelligent code analysis
and reasoning capabilities powered by the RETER reasoning engine.
"""

# Suppress SWIG deprecation warnings from FAISS before any imports
# (FAISS uses SWIG bindings that trigger Python 3.12+ deprecation warnings)
import warnings
warnings.filterwarnings("ignore", message="builtin type Swig", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="builtin type swig", category=DeprecationWarning)

__version__ = "0.1.0"
__author__ = "Reter.AI"

from .server import create_server, ReterCodeServer, main
from .models import (
    LogicalThought,
    LogicalSession,
    ThoughtType,
    LogicType,
    LogicalThinkingInput,
    LogicalThinkingOutput
)
from .reter_wrapper import (
    ReterWrapper,
    # Exception hierarchy
    ReterError,
    ReterFileError,
    ReterFileNotFoundError,
    ReterSaveError,
    ReterLoadError,
    ReterQueryError,
    ReterOntologyError,
)

__all__ = [
    "create_server",
    "ReterCodeServer",
    "main",
    "LogicalThought",
    "LogicalSession",
    "ThoughtType",
    "LogicType",
    "LogicalThinkingInput",
    "LogicalThinkingOutput",
    "ReterWrapper",
    # Exceptions
    "ReterError",
    "ReterFileError",
    "ReterFileNotFoundError",
    "ReterSaveError",
    "ReterLoadError",
    "ReterQueryError",
    "ReterOntologyError",
]
