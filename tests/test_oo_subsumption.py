"""
Tests for OO Ontology Subsumption

Tests that py:Class instances are properly propagated to oo:Class
via the is_subclass_of relationship.
"""

import pytest
from pathlib import Path

# Try to import real RETER - skip tests if not available
try:
    from reter import Reter
    RETER_AVAILABLE = True
except ImportError:
    RETER_AVAILABLE = False


# Sample Python code to load
SAMPLE_PYTHON_CODE = '''
class BaseClass:
    """A base class for testing."""

    def base_method(self):
        pass


class DerivedClass(BaseClass):
    """A derived class for testing."""

    def derived_method(self):
        pass


def standalone_function():
    """A standalone function."""
    pass
'''


@pytest.fixture
def resources_dir():
    """Get the resources directory path."""
    return Path(__file__).parent.parent / "resources"


@pytest.fixture
def reter_with_ontologies(resources_dir):
    """Create a RETER instance with OO and Python ontologies loaded."""
    if not RETER_AVAILABLE:
        pytest.skip("RETER not available")

    reter = Reter(variant="ai")

    # Load OO ontology first
    oo_ontology_path = resources_dir / "oo_ontology.reol"
    with open(oo_ontology_path, 'r') as f:
        oo_ontology = f.read()
    oo_wmes = reter.load_ontology(oo_ontology, source="oo_ontology")
    print(f"Loaded OO ontology: {oo_wmes} WMEs")

    # Load Python ontology
    py_ontology_path = resources_dir / "python" / "py_ontology.reol"
    with open(py_ontology_path, 'r') as f:
        py_ontology = f.read()
    py_wmes = reter.load_ontology(py_ontology, source="py_ontology")
    print(f"Loaded Python ontology: {py_wmes} WMEs")

    return reter


@pytest.mark.skipif(not RETER_AVAILABLE, reason="RETER not available")
class TestOOSubsumption:
    """Test OO ontology subsumption works correctly."""

    def test_ontologies_load(self, reter_with_ontologies):
        """Test that both ontologies load successfully."""
        reter = reter_with_ontologies
        sources = reter.get_all_sources()
        assert "oo_ontology" in sources
        assert "py_ontology" in sources

    def test_python_code_loads(self, reter_with_ontologies):
        """Test that Python code loads and creates py:Class facts."""
        reter = reter_with_ontologies

        # Load sample Python code (args: code, module_name, source_id)
        wme_count, errors = reter.load_python_code(
            SAMPLE_PYTHON_CODE,
            "test_module",
            "test_code"
        )
        print(f"Loaded Python code: {wme_count} WMEs, {len(errors)} errors")
        assert wme_count > 0, "Should create WMEs from Python code"
        assert len(errors) == 0, f"Should not have errors: {errors}"

    def test_py_class_query(self, reter_with_ontologies):
        """Test that py:Class query finds Python classes."""
        reter = reter_with_ontologies

        # Load sample Python code (args: code, module_name, source_id)
        reter.load_python_code(
            SAMPLE_PYTHON_CODE,
            "test_module",
            "test_code"
        )

        # Query for py:Class using 'type' predicate (not 'concept')
        query = '''
            SELECT ?class ?name
            WHERE {
                ?class type py:Class .
                ?class name ?name
            }
        '''
        result = reter.reql(query)

        # Convert to list
        if result.num_rows > 0:
            columns = [result.column(name).to_pylist() for name in result.column_names]
            rows = list(zip(*columns))
        else:
            rows = []

        print(f"py:Class (type) query returned {len(rows)} rows: {rows}")

        # Should find BaseClass and DerivedClass
        class_names = [row[1] for row in rows]
        assert "BaseClass" in class_names, "Should find BaseClass"
        assert "DerivedClass" in class_names, "Should find DerivedClass"

    def test_oo_class_query_subsumption(self, reter_with_ontologies):
        """Test that oo:Class query finds Python classes via subsumption."""
        reter = reter_with_ontologies

        # Load sample Python code (args: code, module_name, source_id)
        reter.load_python_code(
            SAMPLE_PYTHON_CODE,
            "test_module",
            "test_code"
        )

        # Query for oo:Class using 'type' predicate (should find py:Class instances via subsumption)
        query = '''
            SELECT ?class ?name
            WHERE {
                ?class type oo:Class .
                ?class name ?name
            }
        '''
        result = reter.reql(query)

        # Convert to list
        if result.num_rows > 0:
            columns = [result.column(name).to_pylist() for name in result.column_names]
            rows = list(zip(*columns))
        else:
            rows = []

        print(f"oo:Class (type) query returned {len(rows)} rows: {rows}")

        # Should find BaseClass and DerivedClass via subsumption
        class_names = [row[1] for row in rows]
        assert "BaseClass" in class_names, "oo:Class should find BaseClass via subsumption"
        assert "DerivedClass" in class_names, "oo:Class should find DerivedClass via subsumption"

    def test_py_and_oo_class_same_count(self, reter_with_ontologies):
        """Test that py:Class and oo:Class return the same number of classes."""
        reter = reter_with_ontologies

        # Load sample Python code (args: code, module_name, source_id)
        reter.load_python_code(
            SAMPLE_PYTHON_CODE,
            "test_module",
            "test_code"
        )

        # Query py:Class using 'type' predicate
        py_query = '''
            SELECT ?class
            WHERE {
                ?class type py:Class
            }
        '''
        py_result = reter.reql(py_query)
        py_count = py_result.num_rows

        # Query oo:Class using 'type' predicate
        oo_query = '''
            SELECT ?class
            WHERE {
                ?class type oo:Class
            }
        '''
        oo_result = reter.reql(oo_query)
        oo_count = oo_result.num_rows

        print(f"py:Class (type) count: {py_count}, oo:Class (type) count: {oo_count}")

        assert oo_count == py_count, \
            f"oo:Class ({oo_count}) should have same count as py:Class ({py_count})"

    def test_subsumption_chain(self, reter_with_ontologies):
        """Test that is_subclass_of chain works: py:Class -> oo:Class -> oo:CodeEntity."""
        reter = reter_with_ontologies

        # Load sample Python code (args: code, module_name, source_id)
        reter.load_python_code(
            SAMPLE_PYTHON_CODE,
            "test_module",
            "test_code"
        )

        # Query for oo:CodeEntity using 'type' predicate (should include classes, functions, etc.)
        query = '''
            SELECT ?entity ?name
            WHERE {
                ?entity type oo:CodeEntity .
                ?entity name ?name
            }
        '''
        result = reter.reql(query)

        if result.num_rows > 0:
            columns = [result.column(name).to_pylist() for name in result.column_names]
            rows = list(zip(*columns))
        else:
            rows = []

        print(f"oo:CodeEntity (type) query returned {len(rows)} rows")

        entity_names = [row[1] for row in rows]

        # Should include both classes and functions
        assert "BaseClass" in entity_names, "oo:CodeEntity should include BaseClass"
        assert "DerivedClass" in entity_names, "oo:CodeEntity should include DerivedClass"
        # Functions might also be included depending on subsumption chain


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
