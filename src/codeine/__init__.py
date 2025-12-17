"""
Codeine - AI-powered code reasoning MCP server

A Model Context Protocol server that provides intelligent code analysis
and reasoning capabilities powered by the RETER reasoning engine.
"""

__version__ = "0.1.0"
__author__ = "Codeine.AI"

from .server import create_server, CodeineServer, main
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
    "CodeineServer",
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
