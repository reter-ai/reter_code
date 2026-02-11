"""
RETER ZeroMQ Server Module.

This module provides the ZeroMQ-based architecture for running RETER
as a standalone server process with rich console output.

Architecture:
- ReterServer: Standalone process with ZeroMQ interface
- ReterClient: Client that connects to ReterServer via ZeroMQ
- ConsoleUI: Rich terminal output for server process
- Protocol: Message types and serialization

Usage:
    # Start server for a project
    reter --project /path/to/project

    # Client connects via discovery
    from reter_code.server import ReterClient
    client = ReterClient.for_project(Path("/path/to/project"))
    result = client.reql("SELECT ?c WHERE {?c type class}")

::: This is-in-layer Server-Layer.
::: This is-in-component ZeroMQ-Server.
::: This depends-on pyzmq.
::: This depends-on rich.
"""

from .protocol import (
    ReterMessage,
    ReterError,
    serialize,
    deserialize,
    # Error codes
    PARSE_ERROR,
    INVALID_REQUEST,
    METHOD_NOT_FOUND,
    INVALID_PARAMS,
    INTERNAL_ERROR,
    TIMEOUT_ERROR,
    CONNECTION_ERROR,
)
from .config import (
    ServerConfig,
    ClientConfig,
    ServerDiscovery,
)
from .reter_server import ReterServer
from .reter_client import (
    ReterClient,
    ReterClientError,
    ReterClientTimeoutError,
    ReterClientConnectionError,
)
from .console_ui import ConsoleUI
from .view_server import ViewServer

__all__ = [
    # Protocol
    "ReterMessage",
    "ReterError",
    "serialize",
    "deserialize",
    # Error codes
    "PARSE_ERROR",
    "INVALID_REQUEST",
    "METHOD_NOT_FOUND",
    "INVALID_PARAMS",
    "INTERNAL_ERROR",
    "TIMEOUT_ERROR",
    "CONNECTION_ERROR",
    # Config
    "ServerConfig",
    "ClientConfig",
    "ServerDiscovery",
    # Server
    "ReterServer",
    # Client
    "ReterClient",
    "ReterClientError",
    "ReterClientTimeoutError",
    "ReterClientConnectionError",
    # Console
    "ConsoleUI",
    # View server
    "ViewServer",
]
