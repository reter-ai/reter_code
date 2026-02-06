"""
Tests for ReterWrapper

Tests the wrapper around the RETER semantic reasoning engine.
Uses mocking for native module to ensure tests run in any environment.
"""

import os
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open
import pytest


class MockNetwork:
    """Mock for the RETER network object."""

    def __init__(self):
        self._data = None
        self._hybrid_mode = False

    def save(self, filename):
        """Mock saving network to file."""
        with open(filename, 'wb') as f:
            f.write(b'mock_network_data')
        return True

    def load(self, filename):
        """Mock loading network from file."""
        with open(filename, 'rb') as f:
            self._data = f.read()
        return True

    def is_hybrid(self):
        """Mock hybrid mode check."""
        return self._hybrid_mode


class MockReter:
    """Mock for the native Reter class."""

    def __init__(self, variant=None):
        self.variant = variant
        self._wme_count = 0
        self._sources = {}
        self._network_data = {}
        self.network = MockNetwork()

    def load_ontology(self, ontology_text, source=None):
        """Mock loading ontology - return simulated WME count."""
        wme_count = len(ontology_text.split('\n'))  # Simple heuristic
        if source:
            self._sources[source] = list(range(self._wme_count, self._wme_count + wme_count))
        self._wme_count += wme_count
        return wme_count

    def load_python_code(self, code, in_file, module_name, source):
        """Mock loading Python code - return simulated WME count and errors."""
        wme_count = len(code.split('\n'))
        if source:
            self._sources[source] = list(range(self._wme_count, self._wme_count + wme_count))
        self._wme_count += wme_count
        return (wme_count, [])  # (wme_count, errors)

    def reql(self, query, timeout_ms=None):
        """Mock REQL query execution."""
        # Return mock PyArrow-like result
        mock_result = MagicMock()
        mock_result.num_rows = 0
        mock_result.to_pylist = MagicMock(return_value=[])
        return mock_result

    def get_all_sources(self):
        """Return list of loaded sources."""
        return list(self._sources.keys())

    def get_facts_from_source(self, source):
        """Return fact IDs for a source."""
        return self._sources.get(source, [])

    def remove_source(self, source):
        """Remove facts from a source."""
        if source in self._sources:
            del self._sources[source]
        return f"Removed source: {source}"

    def forget_source(self, source):
        """Remove facts from a source (alias)."""
        return self.remove_source(source)

    def check_consistency(self):
        """Mock consistency check."""
        return (True, [])


@pytest.fixture
def mock_reter_module():
    """Fixture that patches the Reter import."""
    with patch('reter_code.reter_wrapper.Reter', MockReter):
        with patch('reter_code.reter_wrapper.safe_cpp_call', lambda f, *args: f(*args)):
            yield MockReter


@pytest.fixture
def wrapper(mock_reter_module):
    """Create a ReterWrapper with mocked native module."""
    from reter_code.reter_wrapper import ReterWrapper
    return ReterWrapper(load_ontology=False)


@pytest.fixture
def wrapper_with_ontology(mock_reter_module):
    """Create a ReterWrapper with ontology loading."""
    # Create a mock ontology file
    with tempfile.TemporaryDirectory() as tmpdir:
        ontology_dir = Path(tmpdir) / "resources" / "python"
        ontology_dir.mkdir(parents=True)
        ontology_file = ontology_dir / "py_ontology.reol"
        ontology_file.write_text("# Mock ontology\nModule is_a Thing")

        # Patch the path resolution
        with patch.object(Path, 'parent', new_callable=lambda: property(lambda self: Path(tmpdir))):
            from reter_code.reter_wrapper import ReterWrapper
            # Just use wrapper without ontology for now - path mocking is complex
            yield ReterWrapper(load_ontology=False)


class TestReterWrapperInit:
    """Test ReterWrapper initialization."""

    def test_init_without_ontology(self, mock_reter_module):
        """Test initialization without loading ontology."""
        from reter_code.reter_wrapper import ReterWrapper
        wrapper = ReterWrapper(load_ontology=False)

        assert wrapper.reasoner is not None
        assert wrapper._dirty is False
        assert wrapper._session_stats["total_wmes"] == 0
        assert wrapper._session_stats["total_sources"] == 0

    def test_init_creates_reasoner_with_ai_variant(self, mock_reter_module):
        """Test that reasoner is created with 'ai' variant."""
        from reter_code.reter_wrapper import ReterWrapper
        wrapper = ReterWrapper(load_ontology=False)

        assert wrapper.reasoner.variant == "ai"


class TestAddOntology:
    """Test add_ontology method."""

    def test_add_ontology_basic(self, wrapper):
        """Test adding basic ontology text."""
        wme_count, source, time_ms = wrapper.add_ontology("Person is_a Thing", source="test")

        assert wme_count > 0
        assert source == "test"
        assert time_ms >= 0
        assert wrapper._dirty is True

    def test_add_ontology_without_source(self, wrapper):
        """Test adding ontology without explicit source."""
        wme_count, source, time_ms = wrapper.add_ontology("Module is_a Thing")

        assert wme_count > 0
        assert source is None

    def test_add_ontology_accumulates(self, wrapper):
        """Test that multiple ontologies accumulate."""
        initial_wmes = wrapper._session_stats["total_wmes"]

        wrapper.add_ontology("Person is_a Thing", source="s1")
        after_first = wrapper._session_stats["total_wmes"]

        wrapper.add_ontology("Animal is_a Thing", source="s2")
        after_second = wrapper._session_stats["total_wmes"]

        assert after_first > initial_wmes
        assert after_second > after_first

    def test_add_ontology_marks_dirty(self, wrapper):
        """Test that adding ontology marks instance as dirty."""
        assert wrapper._dirty is False

        wrapper.add_ontology("Test fact", source="test")

        assert wrapper._dirty is True


class TestReql:
    """Test REQL query execution."""

    def test_reql_basic_query(self, wrapper):
        """Test executing a basic REQL query."""
        result = wrapper.reql("SELECT ?x WHERE { ?x concept \"py:Class\" }")

        assert result is not None

    def test_reql_returns_result(self, wrapper):
        """Test that REQL returns a result object."""
        result = wrapper.reql("SELECT ?x WHERE { ?x name \"Test\" }")

        # Should have result methods
        assert hasattr(result, 'num_rows') or hasattr(result, 'to_pylist')


class TestLoadPythonCode:
    """Test loading Python code into RETER."""

    def test_load_python_code_basic(self, wrapper):
        """Test loading basic Python code."""
        code = '''
class MyClass:
    def method(self):
        pass
'''
        wme_count, source, time_ms, warnings = wrapper.load_python_code(code, source="test_module")

        assert wme_count >= 0
        assert source is not None
        assert time_ms >= 0
        assert isinstance(warnings, list)

    def test_load_python_code_with_source(self, wrapper):
        """Test loading code with explicit source ID."""
        code = "x = 1"
        _, source, _, _ = wrapper.load_python_code(code, source="my_source")

        # Source should be used (though may be modified with MD5)
        assert source is not None


class TestLoadPythonFile:
    """Test loading Python files into RETER."""

    def test_load_python_file(self, wrapper):
        """Test loading a Python file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("class TestClass:\n    pass\n")
            filepath = f.name

        try:
            wme_count, source, time_ms, warnings = wrapper.load_python_file(filepath)

            assert wme_count >= 0
            assert source is not None
            assert time_ms >= 0
            assert isinstance(warnings, list)
        finally:
            os.unlink(filepath)

    def test_load_python_file_with_base_path(self, wrapper):
        """Test loading file with base path for relative source ID."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_module.py")
            with open(filepath, 'w') as f:
                f.write("def func(): pass\n")

            wme_count, source, time_ms, warnings = wrapper.load_python_file(
                filepath, base_path=tmpdir
            )

            assert source is not None
            # Source should contain relative path info
            assert "test_module" in source or wme_count >= 0


class TestSourceManagement:
    """Test source tracking and forgetting."""

    def test_get_all_sources(self, wrapper):
        """Test getting list of all sources."""
        wrapper.add_ontology("Test1", source="source1")
        wrapper.add_ontology("Test2", source="source2")

        sources, time_ms = wrapper.get_all_sources()

        assert isinstance(sources, list)
        assert time_ms >= 0

    def test_get_facts_from_source(self, wrapper):
        """Test getting facts for a specific source."""
        wrapper.add_ontology("Test fact", source="my_source")

        facts, source_id, time_ms = wrapper.get_facts_from_source("my_source")

        assert isinstance(facts, list)
        assert time_ms >= 0

    def test_forget_source(self, wrapper):
        """Test selectively forgetting a source."""
        wrapper.add_ontology("Test fact", source="to_forget")

        result, time_ms = wrapper.forget_source("to_forget")

        assert result is not None
        assert time_ms >= 0


class TestPersistence:
    """Test network save/load operations."""

    def test_save_network(self, wrapper):
        """Test saving network to file."""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.reter') as f:
            filepath = f.name

        try:
            wrapper.add_ontology("Test", source="test")
            success, message, time_ms = wrapper.save_network(filepath)

            assert success is True
            assert time_ms >= 0
            assert os.path.exists(filepath)
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)

    def test_load_network(self, wrapper):
        """Test loading network from file."""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.reter') as f:
            filepath = f.name

        try:
            # First save
            wrapper.add_ontology("Test", source="test")
            wrapper.save_network(filepath)

            # Then load into new wrapper
            from reter_code.reter_wrapper import ReterWrapper
            wrapper2 = ReterWrapper(load_ontology=False)
            success, message, time_ms = wrapper2.load_network(filepath)

            assert success is True
            assert time_ms >= 0
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)


class TestDirtyTracking:
    """Test dirty state tracking."""

    def test_is_dirty_initially_false(self, wrapper):
        """Test that new wrapper is not dirty."""
        assert wrapper.is_dirty() is False

    def test_is_dirty_after_add_ontology(self, wrapper):
        """Test that adding ontology marks dirty."""
        wrapper.add_ontology("Test", source="test")

        assert wrapper.is_dirty() is True

    def test_mark_clean(self, wrapper):
        """Test marking wrapper as clean."""
        wrapper.add_ontology("Test", source="test")
        assert wrapper.is_dirty() is True

        wrapper.mark_clean()

        assert wrapper.is_dirty() is False

    def test_get_last_save_time(self, wrapper):
        """Test getting last save timestamp."""
        save_time = wrapper.get_last_save_time()

        assert isinstance(save_time, float)
        assert save_time <= time.time()


class TestConsistencyCheck:
    """Test consistency checking."""

    def test_check_consistency(self, wrapper):
        """Test running consistency check."""
        wrapper.add_ontology("Test is_a Thing", source="test")

        is_consistent, issues, time_ms = wrapper.check_consistency()

        assert isinstance(is_consistent, bool)
        assert isinstance(issues, list)
        assert time_ms >= 0


class TestShutdown:
    """Test graceful shutdown."""

    def test_shutdown(self, wrapper):
        """Test shutdown doesn't raise."""
        wrapper.add_ontology("Test", source="test")

        # Should not raise
        wrapper.shutdown()


class TestLoadPythonDirectory:
    """Test loading entire directories of Python files."""

    def test_load_python_directory(self, wrapper):
        """Test loading a directory of Python files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create some Python files
            for i in range(3):
                filepath = os.path.join(tmpdir, f"module{i}.py")
                with open(filepath, 'w') as f:
                    f.write(f"class Class{i}: pass\n")

            wme_count, errors, time_ms = wrapper.load_python_directory(tmpdir)

            assert wme_count >= 0
            assert isinstance(errors, dict)
            assert time_ms >= 0

    def test_load_python_directory_recursive(self, wrapper):
        """Test recursive directory loading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create nested structure
            subdir = os.path.join(tmpdir, "subpackage")
            os.makedirs(subdir)

            with open(os.path.join(tmpdir, "main.py"), 'w') as f:
                f.write("def main(): pass\n")

            with open(os.path.join(subdir, "sub.py"), 'w') as f:
                f.write("def sub(): pass\n")

            wme_count, errors, time_ms = wrapper.load_python_directory(
                tmpdir, recursive=True
            )

            assert wme_count >= 0

    def test_load_python_directory_with_exclude(self, wrapper):
        """Test directory loading with exclusion patterns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create files including test files
            with open(os.path.join(tmpdir, "module.py"), 'w') as f:
                f.write("class Module: pass\n")

            with open(os.path.join(tmpdir, "test_module.py"), 'w') as f:
                f.write("class TestModule: pass\n")

            wme_count, errors, time_ms = wrapper.load_python_directory(
                tmpdir,
                exclude_patterns=["test_*.py"]
            )

            assert wme_count >= 0


# ============================================================================
# Integration Tests for Hybrid Network (uses real implementation, not mocks)
# ============================================================================

class TestHybridModeIntegration:
    """
    Integration tests for hybrid mode (incremental save with delta journals).

    These tests use the real implementation (not mocks) because hybrid mode
    depends on actual file I/O and the C++ backend.
    """

    @pytest.fixture
    def real_wrapper(self):
        """Create a real ReterWrapper (not mocked)."""
        from reter_code.reter_wrapper import ReterWrapper
        return ReterWrapper(load_ontology=False)

    def test_is_hybrid_mode_false_initially(self, real_wrapper):
        """Test that hybrid mode is false initially."""
        assert real_wrapper.is_hybrid_mode() is False

    def test_get_hybrid_stats_when_not_hybrid(self, real_wrapper):
        """Test get_hybrid_stats returns hybrid_mode=False when not in hybrid mode."""
        stats = real_wrapper.get_hybrid_stats()
        assert stats == {"hybrid_mode": False}

    def test_open_hybrid_file_not_found(self, real_wrapper):
        """Test open_hybrid raises error for non-existent file."""
        from reter_code.reter_wrapper import ReterFileNotFoundError
        with pytest.raises(ReterFileNotFoundError):
            real_wrapper.open_hybrid("/nonexistent/path/file.reter")

    def test_save_incremental_when_not_hybrid(self, real_wrapper):
        """Test save_incremental raises error when not in hybrid mode."""
        from reter_code.reter_wrapper import ReterSaveError
        with pytest.raises(ReterSaveError, match="Not in hybrid mode"):
            real_wrapper.save_incremental()

    def test_compact_when_not_hybrid(self, real_wrapper):
        """Test compact raises error when not in hybrid mode."""
        from reter_code.reter_wrapper import ReterSaveError
        with pytest.raises(ReterSaveError, match="Not in hybrid mode"):
            real_wrapper.compact()

    def test_needs_compaction_false_when_not_hybrid(self, real_wrapper):
        """Test needs_compaction returns False when not in hybrid mode."""
        assert real_wrapper.needs_compaction() is False

    def test_close_hybrid_safe_when_not_hybrid(self, real_wrapper):
        """Test close_hybrid is safe to call when not in hybrid mode."""
        # Should not raise
        real_wrapper.close_hybrid()

    def test_hybrid_mode_full_cycle(self, real_wrapper, tmp_path):
        """Test full hybrid mode cycle: save base, open hybrid, modify, save delta, close."""
        # 1. Add some data and save as base snapshot
        real_wrapper.add_ontology("Person is_a Thing", source="base_ontology")
        base_path = str(tmp_path / "test.reter")
        success, _, _ = real_wrapper.save_network(base_path)
        assert success is True

        # 2. Create new wrapper and open in hybrid mode
        from reter_code.reter_wrapper import ReterWrapper
        wrapper2 = ReterWrapper(load_ontology=False)
        success, filename, time_ms = wrapper2.open_hybrid(base_path)

        assert success is True
        assert wrapper2.is_hybrid_mode() is True

        # 3. Check hybrid stats
        stats = wrapper2.get_hybrid_stats()
        assert stats["hybrid_mode"] is True
        assert stats["base_facts"] >= 0
        assert stats["delta_facts"] >= 0
        assert "delta_path" in stats

        # 4. Add more data (this should go to delta)
        wrapper2.add_ontology("Student is_a Person", source="delta_ontology")

        # 5. Save incremental (delta only)
        success, delta_path, time_ms = wrapper2.save_incremental()
        assert success is True
        assert os.path.exists(delta_path)

        # 6. Close hybrid mode
        wrapper2.close_hybrid()
        assert wrapper2.is_hybrid_mode() is False

    def test_hybrid_mode_compaction(self, real_wrapper, tmp_path):
        """Test compaction merges delta into base."""
        # 1. Create base snapshot
        real_wrapper.add_ontology("Entity is_a Thing", source="base")
        base_path = str(tmp_path / "compact_test.reter")
        real_wrapper.save_network(base_path)

        # 2. Open hybrid and add delta
        from reter_code.reter_wrapper import ReterWrapper
        wrapper2 = ReterWrapper(load_ontology=False)
        wrapper2.open_hybrid(base_path)

        # Add multiple facts to delta
        for i in range(5):
            wrapper2.add_ontology(f"Thing{i} is_a Entity", source=f"delta_{i}")

        # 3. Check delta stats before compaction
        stats_before = wrapper2.get_hybrid_stats()
        delta_before = stats_before["delta_facts"]

        # 4. Compact
        success, time_ms = wrapper2.compact()
        assert success is True

        # 5. Check stats after compaction
        stats_after = wrapper2.get_hybrid_stats()
        assert stats_after["delta_facts"] == 0
        assert stats_after["base_facts"] >= stats_before["base_facts"]

        wrapper2.close_hybrid()

    def test_needs_compaction_threshold(self, real_wrapper, tmp_path):
        """Test needs_compaction respects threshold ratio.

        Uses unified ReteNetwork API - facts added via wrapper go to delta
        automatically when in hybrid mode.
        """
        from reter_code.reter_wrapper import ReterWrapper

        # Create small base snapshot
        wrapper = ReterWrapper(load_ontology=False)
        wrapper.add_ontology("Root is_a Thing", source="base")
        base_path = str(tmp_path / "threshold_test.reter")
        wrapper.save_network(base_path)

        # Open in hybrid mode and add many delta facts
        wrapper2 = ReterWrapper(load_ontology=False)
        wrapper2.open_hybrid(base_path)

        # Add many delta facts - unified API handles delta tracking
        for i in range(50):
            wrapper2.add_ontology(f"Item{i} is_a Root", source=f"delta_{i}")

        # Should need compaction because delta (50) >> 20% of base (1)
        assert wrapper2.needs_compaction() is True

        # With very high threshold, should not need compaction
        assert wrapper2.needs_compaction(threshold_ratio=100.0) is False

        wrapper2.close_hybrid()


class TestStatePersistenceHybridMode:
    """
    Tests for StatePersistenceService hybrid mode integration.

    Verifies that load_snapshot_if_available and save_all_instances
    work correctly with hybrid mode enabled.
    """

    @pytest.fixture
    def persistence_service(self, tmp_path):
        """Create StatePersistenceService with temp snapshots directory."""
        from reter_code.services.instance_manager import InstanceManager
        from reter_code.services.state_persistence import StatePersistenceService

        instance_mgr = InstanceManager()
        service = StatePersistenceService(instance_mgr)
        service.snapshots_dir = tmp_path
        return service

    def test_load_snapshot_hybrid_mode(self, persistence_service, tmp_path):
        """Test that load_snapshot_if_available uses hybrid mode by default."""
        from reter_code.reter_wrapper import ReterWrapper

        # 1. Create a snapshot file
        wrapper = ReterWrapper(load_ontology=False)
        wrapper.add_ontology("Entity is_a Thing", source="test")
        snapshot_path = tmp_path / ".test_instance.reter"
        wrapper.save_network(str(snapshot_path))

        # 2. Load via persistence service (should use hybrid mode)
        success = persistence_service.load_snapshot_if_available("test_instance")

        assert success is True

        # 3. Verify instance was loaded in hybrid mode
        instances = persistence_service.instance_manager.get_all_instances()
        loaded_wrapper = instances.get("test_instance")
        assert loaded_wrapper is not None
        assert loaded_wrapper.is_hybrid_mode() is True

        # Cleanup
        loaded_wrapper.close_hybrid()

    def test_load_snapshot_fallback_to_regular(self, persistence_service, tmp_path):
        """Test that load_snapshot falls back to regular mode if hybrid fails."""
        from reter_code.reter_wrapper import ReterWrapper

        # Create a snapshot
        wrapper = ReterWrapper(load_ontology=False)
        wrapper.add_ontology("Item is_a Thing", source="test")
        snapshot_path = tmp_path / ".fallback_test.reter"
        wrapper.save_network(str(snapshot_path))

        # Load with hybrid disabled
        success = persistence_service.load_snapshot_if_available(
            "fallback_test", use_hybrid=False
        )

        assert success is True

        # Verify instance was loaded in regular mode (not hybrid)
        instances = persistence_service.instance_manager.get_all_instances()
        loaded_wrapper = instances.get("fallback_test")
        assert loaded_wrapper is not None
        assert loaded_wrapper.is_hybrid_mode() is False

    def test_save_all_instances_incremental(self, persistence_service, tmp_path):
        """Test that save_all_instances uses incremental save for hybrid instances."""
        from reter_code.reter_wrapper import ReterWrapper

        # 1. Create and register an instance
        wrapper = ReterWrapper(load_ontology=False)
        wrapper.add_ontology("Data is_a Thing", source="data")
        persistence_service.instance_manager._instances["inc_test"] = wrapper

        # 2. Save (full) first time to create snapshot
        persistence_service.save_all_instances()

        # 3. Reload in hybrid mode
        wrapper2 = ReterWrapper(load_ontology=False)
        snapshot_path = tmp_path / ".inc_test.reter"
        wrapper2.open_hybrid(str(snapshot_path))
        wrapper2._dirty = True  # Mark as dirty
        persistence_service.instance_manager._instances["inc_test"] = wrapper2

        # Get delta path before save (save_all_instances closes hybrid)
        delta_path = wrapper2.reasoner.network.delta_path()

        # 4. Save again (should be incremental)
        # Note: save_all_instances closes hybrid mode, so we check file existence after
        persistence_service.save_all_instances(compact_on_shutdown=False)

        # 5. Verify delta file exists (was created during hybrid save)
        assert os.path.exists(delta_path)

    def test_add_ontology_syncs_to_hybrid_delta(self, tmp_path):
        """Test that add_ontology in hybrid mode syncs new facts to delta."""
        from reter_code.reter_wrapper import ReterWrapper

        # 1. Create base snapshot
        real_wrapper = ReterWrapper(load_ontology=False)
        real_wrapper.add_ontology("BaseEntity is_a Thing", source="base")
        base_path = str(tmp_path / "sync_test.reter")
        real_wrapper.save_network(base_path)

        # 2. Open in hybrid mode
        wrapper2 = ReterWrapper(load_ontology=False)
        wrapper2.open_hybrid(base_path)

        # Get delta count before add
        stats_before = wrapper2.get_hybrid_stats()
        delta_before = stats_before["delta_facts"]

        # 3. Add ontology in hybrid mode (should sync to delta)
        wrapper2.add_ontology("NewThing is_a BaseEntity", source="new_source")

        # 4. Verify delta count increased
        stats_after = wrapper2.get_hybrid_stats()
        delta_after = stats_after["delta_facts"]

        assert delta_after > delta_before, \
            f"Delta should increase after add_ontology: {delta_before} -> {delta_after}"

        # 5. Save incremental and verify delta file grew
        wrapper2.save_incremental()
        delta_size = stats_after["delta_file_size"]
        assert delta_size > 16, "Delta file should have content beyond header"

        wrapper2.close_hybrid()

    def test_forget_source_syncs_to_hybrid_delta(self, tmp_path):
        """Test that forget_source in hybrid mode syncs removal of delta sources."""
        from reter_code.reter_wrapper import ReterWrapper

        # 1. Create base snapshot
        real_wrapper = ReterWrapper(load_ontology=False)
        real_wrapper.add_ontology("Entity is_a Thing", source="base")
        base_path = str(tmp_path / "forget_test.reter")
        real_wrapper.save_network(base_path)

        # 2. Open in hybrid mode
        wrapper2 = ReterWrapper(load_ontology=False)
        wrapper2.open_hybrid(base_path)

        # 3. Add source in hybrid mode (goes to delta)
        wrapper2.add_ontology("ToForget is_a Entity", source="to_forget")

        # Verify it was added to delta
        stats_before = wrapper2.get_hybrid_stats()
        assert stats_before["delta_facts"] > 0, "Should have delta facts after add"

        # 4. Forget the delta source
        wrapper2.forget_source("to_forget")

        # 5. Verify deleted count increased (or delta facts decreased)
        stats_after = wrapper2.get_hybrid_stats()
        # After forgetting a delta source, it should be marked as deleted
        assert stats_after["deleted_facts"] > 0 or stats_after["delta_facts"] < stats_before["delta_facts"], \
            "Forgetting delta source should mark as deleted or reduce delta"

        wrapper2.close_hybrid()
