"""
RETER Exception Hierarchy

Contains all exception classes used by the RETER integration layer.
"""


class ReterError(Exception):
    """
    Base exception for all RETER operations.

    ::: This is-in-layer Utility-Layer.
    ::: This is a exception.
    ::: This is-in-process Main-Process.
    ::: This is stateless.
    """
    pass


class ReterFileError(ReterError):
    """
    Exception for file-related RETER operations (save/load).

    ::: This is-in-layer Utility-Layer.
    ::: This is a exception.
    ::: This is-in-process Main-Process.
    ::: This is stateless.
    """
    pass


class ReterFileNotFoundError(ReterFileError):
    """
    Raised when a RETER snapshot file is not found.

    ::: This is-in-layer Utility-Layer.
    ::: This is a exception.
    ::: This is-in-process Main-Process.
    ::: This is stateless.
    """
    pass


class ReterSaveError(ReterFileError):
    """
    Raised when saving RETER network fails.

    ::: This is-in-layer Utility-Layer.
    ::: This is a exception.
    ::: This is-in-process Main-Process.
    ::: This is stateless.
    """
    pass


class ReterLoadError(ReterFileError):
    """
    Raised when loading RETER network fails.

    ::: This is-in-layer Utility-Layer.
    ::: This is a exception.
    ::: This is-in-process Main-Process.
    ::: This is stateless.
    """
    pass


class ReterQueryError(ReterError):
    """
    Exception for query-related RETER operations.

    ::: This is-in-layer Utility-Layer.
    ::: This is a exception.
    ::: This is-in-process Main-Process.
    ::: This is stateless.
    """
    pass


class ReterOntologyError(ReterError):
    """
    Exception for ontology/knowledge loading errors.

    ::: This is-in-layer Utility-Layer.
    ::: This is a exception.
    ::: This is-in-process Main-Process.
    ::: This is stateless.
    """
    pass


class DefaultInstanceNotInitialised(ReterError):
    """
    Raised when attempting to access RETER before initialization is complete.

    ::: This is-in-layer Utility-Layer.
    ::: This is a exception.
    ::: This is-in-process Main-Process.
    ::: This is stateless.

    This exception is thrown by ReterWrapper and RAGIndexManager when:
    - Server is starting up and background initialization hasn't completed
    - Embedding model is still loading
    - Default instance Python files are still being indexed

    MCP tools should catch this exception and return an appropriate error
    message to the client indicating they should wait and retry.
    """
    pass


__all__ = [
    "ReterError",
    "ReterFileError",
    "ReterFileNotFoundError",
    "ReterSaveError",
    "ReterLoadError",
    "ReterQueryError",
    "ReterOntologyError",
    "DefaultInstanceNotInitialised",
]
