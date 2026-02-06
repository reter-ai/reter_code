"""
RETER ZeroMQ Request Handlers.

This module provides the handler framework for processing ZeroMQ requests.
Each handler processes a specific category of operations.

::: This is-in-layer Handler-Layer.
::: This is-in-component Query-Handlers.
::: This depends-on reter_code.server.protocol.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, TYPE_CHECKING

from ..protocol import (
    ReterMessage,
    ReterError,
    INTERNAL_ERROR,
    INVALID_PARAMS,
    METHOD_NOT_FOUND,
)

if TYPE_CHECKING:
    from ...reter_wrapper import ReterWrapper
    from ...services.rag_index_manager import RAGIndexManager
    from ...services.instance_manager import DefaultInstanceManager


class HandlerContext:
    """Context object providing access to RETER components.

    ::: This is-in-layer Service-Layer.
    ::: This is a value-object.
    ::: This is stateful.
    """

    def __init__(
        self,
        reter: "ReterWrapper",
        rag_manager: "RAGIndexManager",
        instance_manager: "DefaultInstanceManager",
        event_publisher: Optional[Callable[[str, Dict[str, Any]], None]] = None
    ):
        self.reter = reter
        self.rag_manager = rag_manager
        self.instance_manager = instance_manager
        self.event_publisher = event_publisher

    def publish_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Publish an event via PUB socket."""
        if self.event_publisher:
            self.event_publisher(event_type, data)


class BaseHandler(ABC):
    """Abstract base class for request handlers.

    ::: This is-in-layer Service-Layer.
    ::: This is a handler.
    ::: This is stateful.

    Each handler:
    1. Receives a ReterMessage request
    2. Has access to RETER components via context
    3. Returns a ReterMessage response
    """

    def __init__(self, context: HandlerContext):
        self.context = context
        self._methods: Dict[str, Callable[[Dict[str, Any]], Any]] = {}
        self._register_methods()

    @property
    def reter(self) -> "ReterWrapper":
        """Access to ReterWrapper."""
        return self.context.reter

    @property
    def rag_manager(self) -> "RAGIndexManager":
        """Access to RAG index manager."""
        return self.context.rag_manager

    @property
    def instance_manager(self) -> "DefaultInstanceManager":
        """Access to instance manager."""
        return self.context.instance_manager

    @abstractmethod
    def _register_methods(self) -> None:
        """Register method handlers. Override in subclass."""
        pass

    @abstractmethod
    def can_handle(self, method: str) -> bool:
        """Check if this handler can process the given method."""
        pass

    def handle(self, message: ReterMessage) -> ReterMessage:
        """Process a request message and return response.

        Args:
            message: The incoming request message

        Returns:
            Response message with result or error
        """
        try:
            # Get the method handler
            handler = self._methods.get(message.method)
            if not handler:
                return self._method_not_found(message)

            # Validate params
            validation_error = self._validate_params(message)
            if validation_error:
                return ReterMessage.response(
                    message.id,
                    error=validation_error
                )

            # Execute handler
            result = handler(message.params)

            # Return success response
            return ReterMessage.response(message.id, result=result)

        except Exception as e:
            return self._internal_error(message, e)

    def _validate_params(self, message: ReterMessage) -> Optional[ReterError]:
        """Validate request parameters. Override for custom validation."""
        return None

    def _method_not_found(self, message: ReterMessage) -> ReterMessage:
        """Create method not found error response."""
        return ReterMessage.response(
            message.id,
            error=ReterError(
                code=METHOD_NOT_FOUND,
                message=f"Method '{message.method}' not found",
                details={"method": message.method}
            )
        )

    def _invalid_params(self, message: ReterMessage, reason: str) -> ReterMessage:
        """Create invalid params error response."""
        return ReterMessage.response(
            message.id,
            error=ReterError(
                code=INVALID_PARAMS,
                message=reason,
                details={"params": message.params}
            )
        )

    def _internal_error(self, message: ReterMessage, exc: Exception) -> ReterMessage:
        """Create internal error response."""
        return ReterMessage.response(
            message.id,
            error=ReterError.from_exception(exc)
        )

    def publish_progress(self, operation: str, current: int, total: int,
                         message: Optional[str] = None) -> None:
        """Publish progress event."""
        self.context.publish_event("progress", {
            "operation": operation,
            "current": current,
            "total": total,
            "percent": (current / total * 100) if total > 0 else 0,
            "message": message
        })


class HandlerRegistry:
    """Registry for request handlers.

    ::: This is-in-layer Service-Layer.
    ::: This is a registry.
    ::: This is stateful.
    """

    def __init__(self):
        self._handlers: list[BaseHandler] = []

    def register(self, handler: BaseHandler) -> None:
        """Register a handler."""
        self._handlers.append(handler)

    def get_handler(self, method: str) -> Optional[BaseHandler]:
        """Find handler for a method."""
        for handler in self._handlers:
            if handler.can_handle(method):
                return handler
        return None

    def handle(self, message: ReterMessage) -> ReterMessage:
        """Route message to appropriate handler."""
        handler = self.get_handler(message.method)
        if handler:
            return handler.handle(message)

        # No handler found
        return ReterMessage.response(
            message.id,
            error=ReterError(
                code=METHOD_NOT_FOUND,
                message=f"No handler for method '{message.method}'",
                details={"method": message.method}
            )
        )

    @property
    def methods(self) -> list[str]:
        """List all registered methods."""
        methods = []
        for handler in self._handlers:
            methods.extend(handler._methods.keys())
        return methods


# =============================================================================
# Handler Imports
# =============================================================================

from .query_handler import QueryHandler
from .knowledge_handler import KnowledgeHandler
from .rag_handler import RAGHandler
from .file_scan_handler import FileScanHandler
from .system_handler import SystemHandler
from .inspection_handler import InspectionHandler
from .session_handler import SessionHandler


def create_handler_registry(context: HandlerContext) -> HandlerRegistry:
    """Create and populate handler registry with all handlers.

    Args:
        context: Handler context with RETER components

    Returns:
        Configured HandlerRegistry
    """
    registry = HandlerRegistry()

    # Register all handlers
    registry.register(QueryHandler(context))
    registry.register(KnowledgeHandler(context))
    registry.register(RAGHandler(context))
    registry.register(FileScanHandler(context))
    registry.register(SystemHandler(context))
    registry.register(InspectionHandler(context))
    registry.register(SessionHandler(context))

    return registry


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Base classes
    "HandlerContext",
    "BaseHandler",
    "HandlerRegistry",
    # Factory
    "create_handler_registry",
    # Handlers
    "QueryHandler",
    "KnowledgeHandler",
    "RAGHandler",
    "FileScanHandler",
    "SystemHandler",
    "InspectionHandler",
    "SessionHandler",
]
