"""
RETER ZeroMQ Message Protocol.

Defines message types, error codes, and serialization for ZeroMQ communication
between RETER server and clients.

::: This is-in-layer Protocol-Layer.
::: This is-in-component Message-Protocol.
::: This depends-on msgpack.
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Literal, Optional
import uuid
import msgpack


# =============================================================================
# Error Codes (JSON-RPC 2.0 compatible)
# =============================================================================

PARSE_ERROR = -32700       # Invalid message format
INVALID_REQUEST = -32600   # Not a valid request
METHOD_NOT_FOUND = -32601  # Method does not exist
INVALID_PARAMS = -32602    # Invalid method parameters
INTERNAL_ERROR = -32603    # Internal server error

# Application-specific error codes (-32000 to -32099)
TIMEOUT_ERROR = -32000     # Request timed out
CONNECTION_ERROR = -32001  # Connection lost
INSTANCE_ERROR = -32002    # RETER instance error
QUERY_ERROR = -32003       # Query execution error
KNOWLEDGE_ERROR = -32004   # Knowledge loading error
RAG_ERROR = -32005         # RAG operation error
FILE_SCAN_ERROR = -32006   # File scan error


# =============================================================================
# Error Descriptions
# =============================================================================

ERROR_MESSAGES = {
    PARSE_ERROR: "Parse error",
    INVALID_REQUEST: "Invalid request",
    METHOD_NOT_FOUND: "Method not found",
    INVALID_PARAMS: "Invalid params",
    INTERNAL_ERROR: "Internal error",
    TIMEOUT_ERROR: "Request timeout",
    CONNECTION_ERROR: "Connection error",
    INSTANCE_ERROR: "RETER instance error",
    QUERY_ERROR: "Query execution error",
    KNOWLEDGE_ERROR: "Knowledge loading error",
    RAG_ERROR: "RAG operation error",
    FILE_SCAN_ERROR: "File scan error",
}


# =============================================================================
# Message Types
# =============================================================================

MessageType = Literal["request", "response", "event"]


@dataclass
class ReterError:
    """Error information for failed requests.

    ::: This is-in-layer Core-Layer.
    ::: This is a value-object.
    ::: This is stateless.
    ::: This is serializable.
    """

    code: int
    message: str
    details: Optional[Dict[str, Any]] = None

    @classmethod
    def from_code(cls, code: int, details: Optional[Dict[str, Any]] = None) -> "ReterError":
        """Create error from error code."""
        return cls(
            code=code,
            message=ERROR_MESSAGES.get(code, "Unknown error"),
            details=details
        )

    @classmethod
    def from_exception(cls, exc: Exception, code: int = INTERNAL_ERROR) -> "ReterError":
        """Create error from exception."""
        return cls(
            code=code,
            message=str(exc),
            details={"type": type(exc).__name__}
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "code": self.code,
            "message": self.message,
            "details": self.details
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReterError":
        """Create from dictionary."""
        return cls(
            code=data["code"],
            message=data["message"],
            details=data.get("details")
        )


@dataclass
class ReterMessage:
    """Message format for RETER ZeroMQ protocol.

    ::: This is-in-layer Core-Layer.
    ::: This is a value-object.
    ::: This is stateless.
    ::: This is serializable.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: MessageType = "request"
    method: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    result: Optional[Any] = None
    error: Optional[ReterError] = None

    # Metadata fields
    timestamp: Optional[float] = None
    client_id: Optional[str] = None

    @classmethod
    def request(cls, method: str, params: Optional[Dict[str, Any]] = None,
                client_id: Optional[str] = None) -> "ReterMessage":
        """Create a request message."""
        import time
        return cls(
            type="request",
            method=method,
            params=params or {},
            timestamp=time.time(),
            client_id=client_id
        )

    @classmethod
    def response(cls, request_id: str, result: Any = None,
                 error: Optional[ReterError] = None) -> "ReterMessage":
        """Create a response message."""
        import time
        return cls(
            id=request_id,
            type="response",
            result=result,
            error=error,
            timestamp=time.time()
        )

    @classmethod
    def event(cls, method: str, data: Dict[str, Any]) -> "ReterMessage":
        """Create an event message for PUB/SUB."""
        import time
        return cls(
            type="event",
            method=method,
            params=data,
            timestamp=time.time()
        )

    def is_success(self) -> bool:
        """Check if response was successful."""
        return self.type == "response" and self.error is None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = {
            "id": self.id,
            "type": self.type,
            "method": self.method,
            "params": self.params,
            "result": self.result,
            "timestamp": self.timestamp,
            "client_id": self.client_id,
        }
        if self.error:
            data["error"] = self.error.to_dict()
        else:
            data["error"] = None
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReterMessage":
        """Create from dictionary."""
        error_data = data.get("error")
        error = ReterError.from_dict(error_data) if error_data else None

        return cls(
            id=data.get("id", str(uuid.uuid4())),
            type=data.get("type", "request"),
            method=data.get("method", ""),
            params=data.get("params", {}),
            result=data.get("result"),
            error=error,
            timestamp=data.get("timestamp"),
            client_id=data.get("client_id")
        )


# =============================================================================
# Serialization Helpers
# =============================================================================

def serialize(message: ReterMessage) -> bytes:
    """Serialize message to bytes using msgpack.

    msgpack is faster and more compact than JSON for binary protocols.
    """
    return msgpack.packb(message.to_dict(), use_bin_type=True)


def deserialize(data: bytes) -> ReterMessage:
    """Deserialize bytes to message using msgpack."""
    unpacked = msgpack.unpackb(data, raw=False)
    return ReterMessage.from_dict(unpacked)


def serialize_json(message: ReterMessage) -> str:
    """Serialize message to JSON string (for debugging/logging)."""
    import json
    return json.dumps(message.to_dict(), default=str)


def deserialize_json(data: str) -> ReterMessage:
    """Deserialize JSON string to message."""
    import json
    return ReterMessage.from_dict(json.loads(data))


# =============================================================================
# Method Constants
# =============================================================================

# Query methods
METHOD_REQL = "reql"
METHOD_DL = "dl"
METHOD_PATTERN = "pattern"
METHOD_NATURAL_LANGUAGE = "natural_language_query"
METHOD_EXECUTE_CADSL = "execute_cadsl"
METHOD_GENERATE_CADSL = "generate_cadsl"

# Knowledge methods
METHOD_ADD_KNOWLEDGE = "add_knowledge"
METHOD_ADD_DIRECTORY = "add_external_directory"
METHOD_FORGET = "forget"
METHOD_RELOAD = "reload"
METHOD_VALIDATE_CNL = "validate_cnl"

# RAG methods
METHOD_RAG_SEARCH = "semantic_search"
METHOD_RAG_REINDEX = "rag_reindex"

# Code inspection methods
METHOD_CODE_INSPECTION = "code_inspection"
METHOD_DIAGRAM = "diagram"
METHOD_RECOMMENDER = "recommender"

# System methods
METHOD_SYSTEM = "system"
METHOD_STATUS = "status"
METHOD_INFO = "info"

# File scan methods
METHOD_FILE_SCAN = "file_scan"

# Session/thinking methods (for Agent SDK)
METHOD_SESSION = "session"
METHOD_THINKING = "thinking"
METHOD_ITEMS = "items"

# CADSL tools similarity search
METHOD_SIMILAR_CADSL_TOOLS = "similar_cadsl_tools"

# View server methods
METHOD_VIEW_PUSH = "view_push"


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Error codes
    "PARSE_ERROR",
    "INVALID_REQUEST",
    "METHOD_NOT_FOUND",
    "INVALID_PARAMS",
    "INTERNAL_ERROR",
    "TIMEOUT_ERROR",
    "CONNECTION_ERROR",
    "INSTANCE_ERROR",
    "QUERY_ERROR",
    "KNOWLEDGE_ERROR",
    "RAG_ERROR",
    "FILE_SCAN_ERROR",
    "ERROR_MESSAGES",
    # Message types
    "MessageType",
    "ReterError",
    "ReterMessage",
    # Serialization
    "serialize",
    "deserialize",
    "serialize_json",
    "deserialize_json",
    # Method constants
    "METHOD_REQL",
    "METHOD_DL",
    "METHOD_PATTERN",
    "METHOD_NATURAL_LANGUAGE",
    "METHOD_EXECUTE_CADSL",
    "METHOD_GENERATE_CADSL",
    "METHOD_ADD_KNOWLEDGE",
    "METHOD_ADD_DIRECTORY",
    "METHOD_FORGET",
    "METHOD_RELOAD",
    "METHOD_VALIDATE_CNL",
    "METHOD_RAG_SEARCH",
    "METHOD_RAG_REINDEX",
    "METHOD_CODE_INSPECTION",
    "METHOD_DIAGRAM",
    "METHOD_RECOMMENDER",
    "METHOD_SYSTEM",
    "METHOD_STATUS",
    "METHOD_INFO",
    "METHOD_FILE_SCAN",
    "METHOD_SESSION",
    "METHOD_THINKING",
    "METHOD_ITEMS",
    "METHOD_VIEW_PUSH",
]
