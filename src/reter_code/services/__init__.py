"""
Service Classes for RETER Logical Thinking Server

This package contains service classes extracted from LogicalThinkingServer
as part of the God Class refactoring (Fowler's "Extract Class" pattern).

Each service has a single, well-defined responsibility following the
Single Responsibility Principle.
"""

# Lightweight imports (no reter/reter_core dependency)
from .documentation_provider import DocumentationProvider
from .resource_registrar import ResourceRegistrar
from .tool_registrar import ToolRegistrar
from .config_loader import ConfigLoader, load_config, get_config_loader


def __getattr__(name):
    """Lazy-import server-only classes that depend on reter/reter_core."""
    if name == "InstanceManager":
        from .instance_manager import InstanceManager
        return InstanceManager
    if name == "ReterOperations":
        from .reter_operations import ReterOperations
        return ReterOperations
    if name == "StatePersistenceService":
        from .state_persistence import StatePersistenceService
        return StatePersistenceService
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "InstanceManager",
    "DocumentationProvider",
    "ReterOperations",
    "StatePersistenceService",
    "ResourceRegistrar",
    "ToolRegistrar",
    "ConfigLoader",
    "load_config",
    "get_config_loader",
]
