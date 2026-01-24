"""
Shared pytest fixtures for RETER logical thinking server tests.

This module provides common fixtures used across multiple test modules,
eliminating duplicate fixture definitions and ensuring consistency.
"""

import os
import tempfile
import pytest
from reter_code.tools.unified.store import UnifiedStore
from reter_code.tools.unified.session import ThinkingSession
from reter_code.tools.unified.operations import OperationsHandler
from reter_code.reter_wrapper import (
    set_initialization_complete,
    set_initialization_in_progress
)
from reter_code.services.initialization_progress import (
    get_component_readiness,
    reset_component_readiness,
)


@pytest.fixture(autouse=True)
def set_initialization_flags():
    """
    Set initialization flags for all tests.

    This fixture runs automatically before each test to ensure that
    the initialization checks in ReterWrapper and RAGIndexManager
    don't block test execution.
    """
    # Set old initialization flags (for backward compatibility)
    set_initialization_complete(True)
    set_initialization_in_progress(False)

    # Set new component readiness flags
    components = get_component_readiness()
    components.set_sql_ready(True)
    components.set_reter_ready(True)
    components.set_embedding_ready(True)
    components.set_rag_code_ready(True)
    components.set_rag_docs_ready(True)

    yield

    # Reset flags after test
    set_initialization_complete(False)
    set_initialization_in_progress(False)
    reset_component_readiness()


@pytest.fixture
def store():
    """
    Create a temporary UnifiedStore for testing.

    Yields:
        UnifiedStore: A store instance backed by a temporary SQLite database.

    The database file is automatically cleaned up after the test completes.
    """
    with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as f:
        db_path = f.name

    store = UnifiedStore(db_path)
    yield store

    # Cleanup
    try:
        os.unlink(db_path)
    except Exception:
        pass


@pytest.fixture
def session(store):
    """
    Create a ThinkingSession with the test store.

    Args:
        store: The UnifiedStore fixture.

    Returns:
        ThinkingSession: A session instance for testing.
    """
    return ThinkingSession(store)


@pytest.fixture
def handler(store):
    """
    Create an OperationsHandler with the test store.

    Args:
        store: The UnifiedStore fixture.

    Returns:
        OperationsHandler: A handler instance for testing.
    """
    return OperationsHandler(store)


@pytest.fixture
def session_with_thought(store):
    """
    Create a session with a pre-existing thought for testing.

    Args:
        store: The UnifiedStore fixture.

    Returns:
        tuple: (session_id, thought_id) for the created session and thought.
    """
    session_id = store.get_or_create_session("test-instance")
    thought_id = store.add_item(
        session_id, "thought", "Test thought",
        item_id="THOUGHT-001"
    )
    return session_id, thought_id
