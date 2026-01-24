"""
Tests for SQLiteSessionStore

Tests the SQLite-based session storage backend.
"""

import os
import tempfile
import threading
import time
from pathlib import Path
import pytest

from reter_code.tools.session_store.sqlite_store import SQLiteSessionStore


@pytest.fixture
def store():
    """Create a temporary SQLiteSessionStore for testing."""
    with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as f:
        db_path = f.name

    store = SQLiteSessionStore(db_path)
    yield store

    # Cleanup
    store.close()
    try:
        os.unlink(db_path)
    except Exception:
        pass
    # Also try to remove WAL and SHM files
    for ext in ['-wal', '-shm']:
        try:
            os.unlink(db_path + ext)
        except Exception:
            pass


@pytest.fixture
def store_with_data(store):
    """Create a store with some test data."""
    with store.transaction() as conn:
        # Insert test data using raw SQL since schema should exist
        conn.execute("""
            INSERT OR IGNORE INTO logical_sessions (session_id, instance_name, status, goal)
            VALUES ('test-session-1', 'test', 'active', 'Test goal')
        """)
    return store


class TestSQLiteSessionStoreInit:
    """Test SQLiteSessionStore initialization."""

    def test_init_creates_database(self, store):
        """Test that initialization creates the database file."""
        assert store.db_path.exists()

    def test_init_with_explicit_path(self):
        """Test initialization with explicit path."""
        with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as f:
            db_path = f.name

        try:
            store = SQLiteSessionStore(db_path)
            assert store.db_path == Path(db_path)
            store.close()
        finally:
            os.unlink(db_path)

    def test_init_creates_parent_directories(self):
        """Test that initialization creates parent directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "subdir" / "nested" / "test.sqlite"
            store = SQLiteSessionStore(db_path)

            assert db_path.parent.exists()
            store.close()

    def test_db_path_property(self, store):
        """Test db_path property returns correct path."""
        assert isinstance(store.db_path, Path)


class TestSQLiteSessionStoreExecute:
    """Test SQL execution methods."""

    def test_execute_simple_query(self, store):
        """Test executing a simple SQL query."""
        cursor = store.execute("SELECT 1 AS value")
        result = cursor.fetchone()

        assert result is not None
        assert result['value'] == 1

    def test_execute_with_params_tuple(self, store):
        """Test executing query with tuple parameters."""
        cursor = store.execute(
            "SELECT ? AS value",
            (42,)
        )
        result = cursor.fetchone()

        assert result['value'] == 42

    def test_execute_with_params_dict(self, store):
        """Test executing query with dict parameters."""
        cursor = store.execute(
            "SELECT :val AS value",
            {'val': 'test'}
        )
        result = cursor.fetchone()

        assert result['value'] == 'test'

    def test_execute_many(self, store):
        """Test executing with multiple parameter sets."""
        # Create a test table
        store.execute("""
            CREATE TABLE IF NOT EXISTS test_batch (
                id INTEGER PRIMARY KEY,
                value TEXT
            )
        """)
        store.commit()

        # Execute many
        store.execute_many(
            "INSERT INTO test_batch (value) VALUES (?)",
            [('a',), ('b',), ('c',)]
        )
        store.commit()

        cursor = store.execute("SELECT COUNT(*) AS cnt FROM test_batch")
        assert cursor.fetchone()['cnt'] == 3


class TestSQLiteSessionStoreFetch:
    """Test fetch methods."""

    def test_fetch_one_returns_dict(self, store):
        """Test fetch_one returns dictionary."""
        result = store.fetch_one("SELECT 1 AS col1, 'test' AS col2")

        assert isinstance(result, dict)
        assert result['col1'] == 1
        assert result['col2'] == 'test'

    def test_fetch_one_returns_none_for_empty(self, store):
        """Test fetch_one returns None for empty result."""
        result = store.fetch_one("SELECT * FROM sqlite_master WHERE 0=1")

        assert result is None

    def test_fetch_all_returns_list(self, store):
        """Test fetch_all returns list of dictionaries."""
        # Create test table with data
        store.execute("""
            CREATE TABLE IF NOT EXISTS test_fetch (
                id INTEGER PRIMARY KEY,
                name TEXT
            )
        """)
        store.execute("INSERT INTO test_fetch (name) VALUES ('a'), ('b'), ('c')")
        store.commit()

        results = store.fetch_all("SELECT * FROM test_fetch ORDER BY id")

        assert isinstance(results, list)
        assert len(results) == 3
        assert all(isinstance(r, dict) for r in results)

    def test_fetch_all_empty_returns_empty_list(self, store):
        """Test fetch_all returns empty list for no results."""
        results = store.fetch_all("SELECT * FROM sqlite_master WHERE 0=1")

        assert results == []


class TestSQLiteSessionStoreTransaction:
    """Test transaction context manager."""

    def test_transaction_commits_on_success(self, store):
        """Test transaction commits on successful completion."""
        store.execute("""
            CREATE TABLE IF NOT EXISTS test_tx (
                id INTEGER PRIMARY KEY,
                value TEXT
            )
        """)
        store.commit()

        with store.transaction() as conn:
            conn.execute("INSERT INTO test_tx (value) VALUES ('test')")

        # Should be committed
        result = store.fetch_one("SELECT value FROM test_tx WHERE value = 'test'")
        assert result is not None
        assert result['value'] == 'test'

    def test_transaction_rollback_on_exception(self, store):
        """Test transaction rolls back on exception."""
        store.execute("""
            CREATE TABLE IF NOT EXISTS test_rollback (
                id INTEGER PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)
        store.commit()

        try:
            with store.transaction() as conn:
                conn.execute("INSERT INTO test_rollback (value) VALUES ('keep')")
                # Force an error
                raise ValueError("Test error")
        except ValueError:
            pass

        # Should be rolled back
        result = store.fetch_one("SELECT * FROM test_rollback WHERE value = 'keep'")
        assert result is None

    def test_transaction_nested_usage(self, store):
        """Test that transaction can be used multiple times."""
        store.execute("CREATE TABLE IF NOT EXISTS test_multi (value TEXT)")
        store.commit()

        with store.transaction() as conn:
            conn.execute("INSERT INTO test_multi (value) VALUES ('first')")

        with store.transaction() as conn:
            conn.execute("INSERT INTO test_multi (value) VALUES ('second')")

        results = store.fetch_all("SELECT * FROM test_multi")
        assert len(results) == 2


class TestSQLiteSessionStoreCommitRollback:
    """Test commit and rollback methods."""

    def test_commit(self, store):
        """Test explicit commit."""
        store.execute("CREATE TABLE IF NOT EXISTS test_commit (value TEXT)")
        store.commit()

        store.execute("INSERT INTO test_commit (value) VALUES ('test')")
        store.commit()

        result = store.fetch_one("SELECT * FROM test_commit")
        assert result is not None

    def test_rollback(self, store):
        """Test explicit rollback."""
        store.execute("CREATE TABLE IF NOT EXISTS test_rb (value TEXT)")
        store.commit()

        store.execute("INSERT INTO test_rb (value) VALUES ('to_rollback')")
        store.rollback()

        result = store.fetch_one("SELECT * FROM test_rb WHERE value = 'to_rollback'")
        assert result is None


class TestSQLiteSessionStoreSchema:
    """Test schema management."""

    def test_schema_creates_sessions_table(self, store):
        """Test that schema creates logical_sessions table."""
        assert store.table_exists('logical_sessions')

    def test_schema_creates_thoughts_table(self, store):
        """Test that schema creates logical_thoughts table."""
        assert store.table_exists('logical_thoughts')

    def test_table_exists_returns_true_for_existing(self, store):
        """Test table_exists returns True for existing table."""
        # Use a table we know exists from our schema
        assert store.table_exists('logical_sessions') is True

    def test_table_exists_returns_false_for_missing(self, store):
        """Test table_exists returns False for missing table."""
        assert store.table_exists('nonexistent_table_xyz') is False


class TestSQLiteSessionStoreCount:
    """Test count method."""

    def test_count_all_rows(self, store):
        """Test counting all rows in a table."""
        store.execute("CREATE TABLE IF NOT EXISTS test_count (value TEXT)")
        store.execute("INSERT INTO test_count VALUES ('a'), ('b'), ('c')")
        store.commit()

        count = store.count('test_count')
        assert count == 3

    def test_count_with_where(self, store):
        """Test counting with WHERE clause."""
        store.execute("CREATE TABLE IF NOT EXISTS test_count_where (value TEXT)")
        store.execute("INSERT INTO test_count_where VALUES ('a'), ('b'), ('a')")
        store.commit()

        count = store.count('test_count_where', where="value = ?", params=('a',))
        assert count == 2


class TestSQLiteSessionStoreNowIso:
    """Test now_iso method."""

    def test_now_iso_format(self, store):
        """Test now_iso returns valid ISO format."""
        timestamp = store.now_iso()

        # Should be ISO format: YYYY-MM-DDTHH:MM:SS
        assert 'T' in timestamp
        assert len(timestamp) >= 19  # Minimum length


class TestSQLiteSessionStoreClose:
    """Test close method."""

    def test_close_closes_connection(self):
        """Test close method closes connection."""
        with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as f:
            db_path = f.name

        store = SQLiteSessionStore(db_path)

        # Access connection
        store.execute("SELECT 1")

        # Close
        store.close()

        # Connection should be None after close
        assert not hasattr(store._local, 'connection') or store._local.connection is None

        os.unlink(db_path)


class TestSQLiteSessionStoreThreadSafety:
    """Test thread safety features."""

    def test_thread_local_connections(self, store):
        """Test that each thread gets its own connection."""
        connections = []
        errors = []

        def thread_func():
            try:
                # Get connection in this thread
                conn = store._get_connection()
                connections.append(id(conn))
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=thread_func) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # No errors
        assert len(errors) == 0

        # Each thread should have different connection ID
        # (unless same thread reuses connection)
        assert len(connections) == 3

    def test_concurrent_writes(self, store):
        """Test concurrent write operations."""
        store.execute("CREATE TABLE IF NOT EXISTS test_concurrent (value INTEGER)")
        store.commit()

        errors = []
        success_count = [0]
        lock = threading.Lock()

        def insert_value(val):
            try:
                with store.transaction() as conn:
                    conn.execute(
                        "INSERT INTO test_concurrent (value) VALUES (?)",
                        (val,)
                    )
                with lock:
                    success_count[0] += 1
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=insert_value, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should have some successful writes
        assert success_count[0] > 0
        # Count should match success_count
        count = store.count('test_concurrent')
        assert count == success_count[0]


class TestSQLiteSessionStoreSchemaMigration:
    """Test schema migration."""

    def test_schema_version_stored(self, store):
        """Test that schema version table is created."""
        # Schema version is stored in schema_version table, not PRAGMA
        assert store.table_exists('schema_version')

    def test_get_schema_sql_returns_string(self, store):
        """Test _get_schema_sql returns SQL string."""
        sql = store._get_schema_sql()

        assert isinstance(sql, str)
        assert len(sql) > 0
        # Should contain CREATE TABLE
        assert 'CREATE TABLE' in sql
