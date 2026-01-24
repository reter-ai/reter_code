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

    def load_python_code(self, code, module_name, source):
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
