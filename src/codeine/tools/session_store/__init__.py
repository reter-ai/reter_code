"""Session Store module for SQLite-based session data persistence.

This module provides the LogicalThinkingStore which uses the unified
SQLite database (.reter/.unified.sqlite) for all session data.

DEPRECATED: SQLiteSessionStore is deprecated. Use UnifiedStore from
tools.unified.store instead.
"""

from .logical_thinking_store import LogicalThinkingStore

# Deprecated - for backward compatibility only
try:
    from .sqlite_store import SQLiteSessionStore
except ImportError:
    SQLiteSessionStore = None  # type: ignore

__all__ = ["LogicalThinkingStore"]
