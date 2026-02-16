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

__version__ = "0.1.8"
__author__ = "Reter.AI"

from .mcp_server import create_server, ReterCodeServer, main
from .models import (
    LogicalThought,
    LogicalSession,
    ThoughtType,
    LogicType,
    LogicalThinkingInput,
    LogicalThinkingOutput
)

# ReterWrapper and exceptions are lazy-imported (they depend on reter/reter_core)
_RETER_WRAPPER_ATTRS = {
    "ReterWrapper", "ReterError", "ReterFileError", "ReterFileNotFoundError",
    "ReterSaveError", "ReterLoadError", "ReterQueryError", "ReterOntologyError",
}

def __getattr__(name):
    if name in _RETER_WRAPPER_ATTRS:
        from . import reter_wrapper
        return getattr(reter_wrapper, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

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
    "ReterError",
    "ReterFileError",
    "ReterFileNotFoundError",
    "ReterSaveError",
    "ReterLoadError",
    "ReterQueryError",
    "ReterOntologyError",
]
