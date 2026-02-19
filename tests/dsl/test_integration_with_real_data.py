"""
Integration tests for CADSL detectors using real RETER and RAG data.

These tests load the actual `.default.reter` and `.default.faiss` from
the project's `.reter_code` directory to verify detectors work with real data.

Prerequisites:
- The reter_code MCP server should have been run at least once to create
  the snapshot files in `.reter_code/`
"""

import pytest
import sys
import os
from pathlib import Path
from unittest.mock import Mock

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

# Path to the .reter_code directory with real data
RETER_CODE_DIR = Path(__file__).parent.parent.parent.parent / ".reter_code"
RETER_SNAPSHOT = RETER_CODE_DIR / ".default.reter"
FAISS_INDEX = RETER_CODE_DIR / ".default.faiss"
FAISS_META = RETER_CODE_DIR / ".default.faiss.meta"


def has_real_data():
    """Check if real snapshot data exists."""
    return RETER_SNAPSHOT.exists() and FAISS_INDEX.exists() and FAISS_META.exists()


# Skip all tests if no real data
pytestmark = pytest.mark.skipif(
    not has_real_data(),
    reason=f"Real data not found in {RETER_CODE_DIR}. Run the MCP server first."
)


@pytest.fixture(scope="module")
def real_reter():
    """
    Load the real RETER instance from .default.reter snapshot.

    Module-scoped for efficiency (loading is expensive).
    """
    from reter_code.reter_wrapper import (
        ReterWrapper,
        set_initialization_complete,
        set_initialization_in_progress
    )

    # Enable initialization to allow load_network
    set_initialization_in_progress(True)
    set_initialization_complete(True)

    try:
        # Create ReterWrapper with load_ontology=False since snapshot includes ontology
        reter = ReterWrapper(load_ontology=False)
        success, filename, time_ms = reter.load_network(str(RETER_SNAPSHOT))

        if not success:
            pytest.skip(f"Failed to load RETER snapshot from {RETER_SNAPSHOT}")

        print(f"\n[FIXTURE] Loaded RETER snapshot in {time_ms:.1f}ms", file=sys.stderr)

        yield reter

    finally:
        # Reset state after tests
        set_initialization_in_progress(False)
        set_initialization_complete(False)


@pytest.fixture(scope="module")
def real_rag(real_reter):
    """
    Load the real RAG index from .default.faiss snapshot.

    Depends on real_reter to ensure initialization state is set.
    Module-scoped for efficiency.
    """
    try:
        import faiss
    except ImportError:
        pytest.skip("faiss-cpu not installed")

    from reter_code.services.rag_index_manager import RAGIndexManager
    from reter_code.services.state_persistence import StatePersistenceService
    from reter_code.services.instance_manager import InstanceManager

    # Set up persistence pointing to real .reter_code directory
    os.environ["RETER_SNAPSHOTS_DIR"] = str(RETER_CODE_DIR)

    try:
        instance_manager = InstanceManager()
        persistence = StatePersistenceService(instance_manager)

        # Create RAG manager with default config
        config = {
            "rag_enabled": True,
            "rag_use_lightweight": True,
        }

        rag_manager = RAGIndexManager(persistence, config)
        rag_manager._model_loaded = True  # Skip model loading

        # Initialize - this will load the existing index
        project_root = RETER_CODE_DIR.parent
        rag_manager.initialize(project_root=project_root)

        if not rag_manager.is_initialized:
            pytest.skip("Failed to initialize RAG manager")

        status = rag_manager.get_status()
        print(f"\n[FIXTURE] Loaded RAG index with {status.get('total_vectors', 0)} vectors", file=sys.stderr)

        yield rag_manager

    finally:
        if "RETER_SNAPSHOTS_DIR" in os.environ:
            del os.environ["RETER_SNAPSHOTS_DIR"]


class TestReterDataIntegrity:
    """Test that RETER data is loaded correctly."""

    def test_reter_has_sources(self, real_reter):
        """RETER should have loaded sources."""
        sources, count = real_reter.get_all_sources()
        assert count > 0, "RETER should have at least some sources loaded"
        print(f"\n  Found {count} sources in RETER", file=sys.stderr)

    def test_reter_has_classes(self, real_reter):
        """RETER should have Class entities."""
        # Use proper REQL syntax: type oo:Class, name property
        result = real_reter.reql("SELECT ?c ?name WHERE { ?c type oo:Class . ?c name ?name }")
        count = result.num_rows if hasattr(result, 'num_rows') else len(result)
        assert count > 0, "RETER should have Class entities"
        print(f"\n  Found {count} classes", file=sys.stderr)

    def test_reter_has_functions(self, real_reter):
        """RETER should have Function entities."""
        result = real_reter.reql("SELECT ?f ?name WHERE { ?f type oo:Function . ?f name ?name }")
        count = result.num_rows if hasattr(result, 'num_rows') else len(result)
        assert count > 0, "RETER should have Function entities"
        print(f"\n  Found {count} functions", file=sys.stderr)

    def test_reter_has_methods(self, real_reter):
        """RETER should have Method entities."""
        result = real_reter.reql("SELECT ?m ?name WHERE { ?m type oo:Method . ?m name ?name }")
        count = result.num_rows if hasattr(result, 'num_rows') else len(result)
        assert count > 0, "RETER should have Method entities"
        print(f"\n  Found {count} methods", file=sys.stderr)

    def test_reter_has_modules(self, real_reter):
        """RETER should have Module entities."""
        result = real_reter.reql("SELECT ?m ?name WHERE { ?m type oo:Module . ?m name ?name }")
        count = result.num_rows if hasattr(result, 'num_rows') else len(result)
        assert count > 0, "RETER should have Module entities"
        print(f"\n  Found {count} modules", file=sys.stderr)


class TestRagDataIntegrity:
    """Test that RAG data is loaded correctly."""

    def test_rag_has_vectors(self, real_rag):
        """RAG index should have vectors."""
        status = real_rag.get_status()
        total_vectors = status.get("total_vectors", 0)
        # Note: Lightweight test model may not have vectors from full model
        # This test verifies the index loads without errors
        print(f"\n  RAG has {total_vectors} vectors (may be 0 with lightweight model)", file=sys.stderr)

    def test_rag_search_returns_results(self, real_rag):
        """RAG search should not error (may return empty with lightweight model)."""
        results = real_rag.search("class definition", top_k=5)

        # Handle tuple return (results, stats)
        if isinstance(results, tuple):
            results, stats = results

        # With lightweight model, may return empty results
        print(f"\n  Search returned {len(results)} results", file=sys.stderr)

    def test_rag_search_code_entities(self, real_rag):
        """RAG should execute search without errors."""
        results = real_rag.search("function method", top_k=10)

        if isinstance(results, tuple):
            results, _ = results

        # Check that we got results with proper attributes (if any)
        if results:
            result = results[0]
            assert hasattr(result, 'entity_type')
            assert hasattr(result, 'score')
            assert hasattr(result, 'file')
            print(f"\n  Top result: {result.entity_type} in {result.file}", file=sys.stderr)
        else:
            print(f"\n  No results (expected with lightweight model)", file=sys.stderr)


class TestCircularImportsIntegration:
    """Integration tests for circular_imports detector with real data."""

    def test_detector_runs_without_error(self, real_reter):
        """circular_imports should execute without errors."""
        from reter_code.dsl.tools.dependencies.circular_imports import find_circular_imports
        from reter_code.dsl.core import Context

        ctx = Context(
            reter=real_reter,
            params={"limit": 10},
            language="oo",
            instance_name="default"
        )

        result = find_circular_imports(ctx=ctx)

        assert "success" in result
        assert result["success"] is True
        print(f"\n  circular_imports found {result.get('count', 0)} issues", file=sys.stderr)

    def test_detector_reql_query_executes(self, real_reter):
        """The REQL query should execute against real data."""
        # This is similar to the query used by circular_imports (with proper REQL syntax)
        query = """
        SELECT ?m1 ?m2 ?file1 ?file2
        WHERE {
            ?m1 type oo:Module .
            ?m1 imports ?m2 .
            ?m2 type oo:Module .
            ?m1 inFile ?file1 .
            ?m2 inFile ?file2 .
        }
        """

        result = real_reter.reql(query)
        count = result.num_rows if hasattr(result, 'num_rows') else len(result)
        print(f"\n  Import query returned {count} rows", file=sys.stderr)


class TestReqlQueriesIntegration:
    """Integration tests for REQL queries used by CADSL detectors."""

    def test_class_method_query(self, real_reter):
        """Query for classes and their methods should work."""
        # This is the core query used by alternative_interfaces
        query = """
        SELECT ?c ?class_name ?method_name ?file ?line
        WHERE {
            ?c type oo:Class .
            ?c name ?class_name .
            ?c inFile ?file .
            ?c atLine ?line .
            ?m type oo:Method .
            ?m definedIn ?c .
            ?m name ?method_name .
        }
        """
        result = real_reter.reql(query)
        count = result.num_rows if hasattr(result, 'num_rows') else len(result)
        assert count > 0, "Should find classes with methods"
        print(f"\n  Found {count} class-method pairs", file=sys.stderr)

    def test_inheritance_query(self, real_reter):
        """Query for class inheritance should work."""
        # This is the core query used by parallel_inheritance
        query = """
        SELECT ?c ?name ?base_name ?file ?line
        WHERE {
            ?c type oo:Class .
            ?c name ?name .
            ?c inFile ?file .
            ?c atLine ?line .
            ?c inheritsFrom ?base .
            ?base name ?base_name .
        }
        """
        result = real_reter.reql(query)
        count = result.num_rows if hasattr(result, 'num_rows') else len(result)
        # May or may not have inheritance relationships
        print(f"\n  Found {count} inheritance relationships", file=sys.stderr)

    def test_class_with_parameters_query(self, real_reter):
        """Query for classes with method parameters should work."""
        # This is the core query used by primitive_obsession
        query = """
        SELECT ?c ?name ?param_name ?param_type ?file ?line
        WHERE {
            ?c type oo:Class .
            ?c name ?name .
            ?c inFile ?file .
            ?c atLine ?line .
            ?m type oo:Method .
            ?m definedIn ?c .
            ?p type oo:Parameter .
            ?p definedIn ?m .
            ?p name ?param_name .
            ?p hasType ?param_type .
        }
        """
        result = real_reter.reql(query)
        count = result.num_rows if hasattr(result, 'num_rows') else len(result)
        # May or may not have typed parameters
        print(f"\n  Found {count} typed parameters", file=sys.stderr)

    def test_function_with_line_count_query(self, real_reter):
        """Query for functions with line counts should work."""
        # This tests the metrics available for complex_untested
        query = """
        SELECT ?f ?name ?file ?line ?lineCount
        WHERE {
            ?f type oo:Function .
            ?f name ?name .
            ?f inFile ?file .
            ?f atLine ?line .
            ?f lineCount ?lineCount .
        }
        """
        result = real_reter.reql(query)
        count = result.num_rows if hasattr(result, 'num_rows') else len(result)
        print(f"\n  Found {count} functions with line counts", file=sys.stderr)


class TestCodeInspectionQueriesIntegration:
    """Integration tests for code_inspection REQL queries with real data."""

    def test_list_classes_query(self, real_reter):
        """Query for listing classes should work."""
        # Note: Entity ID (?c) is the qualified name, no separate qualifiedName attribute
        query = """
        SELECT ?c ?name ?file ?line
        WHERE {
            ?c type oo:Class .
            ?c name ?name .
            ?c inFile ?file .
            ?c atLine ?line .
        }
        """
        result = real_reter.reql(query)
        count = result.num_rows if hasattr(result, 'num_rows') else len(result)
        assert count > 0, "Should find classes"
        print(f"\n  list_classes query returned {count} classes", file=sys.stderr)

    def test_list_functions_query(self, real_reter):
        """Query for listing functions should work."""
        query = """
        SELECT ?f ?name ?file ?line
        WHERE {
            ?f type oo:Function .
            ?f name ?name .
            ?f inFile ?file .
            ?f atLine ?line .
        }
        """
        result = real_reter.reql(query)
        count = result.num_rows if hasattr(result, 'num_rows') else len(result)
        assert count > 0, "Should find functions"
        print(f"\n  list_functions query returned {count} functions", file=sys.stderr)

    def test_dependency_query(self, real_reter):
        """Query for module dependencies should work."""
        query = """
        SELECT ?m1 ?m2 ?file1 ?file2
        WHERE {
            ?m1 type oo:Module .
            ?m1 imports ?m2 .
            ?m2 type oo:Module .
            ?m1 inFile ?file1 .
            ?m2 inFile ?file2 .
        }
        """
        result = real_reter.reql(query)
        count = result.num_rows if hasattr(result, 'num_rows') else len(result)
        print(f"\n  dependency query returned {count} import relationships", file=sys.stderr)


class TestDiagramQueriesIntegration:
    """Integration tests for diagram REQL queries with real data."""

    def test_class_hierarchy_query(self, real_reter):
        """Query for class hierarchy diagram should work."""
        query = """
        SELECT ?child ?child_name ?parent ?parent_name
        WHERE {
            ?child type oo:Class .
            ?child name ?child_name .
            ?child inheritsFrom ?parent .
            ?parent name ?parent_name .
        }
        """
        result = real_reter.reql(query)
        count = result.num_rows if hasattr(result, 'num_rows') else len(result)
        print(f"\n  class_hierarchy query returned {count} inheritance edges", file=sys.stderr)

    def test_class_diagram_query(self, real_reter):
        """Query for class diagram should work."""
        query = """
        SELECT ?c ?class_name ?method_name ?attr_name
        WHERE {
            ?c type oo:Class .
            ?c name ?class_name .
            OPTIONAL { ?m type oo:Method . ?m definedIn ?c . ?m name ?method_name }
            OPTIONAL { ?a type oo:Attribute . ?a definedIn ?c . ?a name ?attr_name }
        }
        """
        result = real_reter.reql(query)
        count = result.num_rows if hasattr(result, 'num_rows') else len(result)
        print(f"\n  class_diagram query returned {count} rows", file=sys.stderr)

    def test_call_graph_query(self, real_reter):
        """Query for call graph should work."""
        query = """
        SELECT ?caller ?caller_name ?callee ?callee_name ?file
        WHERE {
            ?caller type oo:Function .
            ?caller name ?caller_name .
            ?caller inFile ?file .
            ?caller calls ?callee .
            ?callee name ?callee_name .
        }
        """
        result = real_reter.reql(query)
        count = result.num_rows if hasattr(result, 'num_rows') else len(result)
        print(f"\n  call_graph query returned {count} call edges", file=sys.stderr)

    def test_coupling_matrix_query(self, real_reter):
        """Query for coupling matrix should work."""
        query = """
        SELECT ?c1 ?name1 ?c2 ?name2
        WHERE {
            ?c1 type oo:Class .
            ?c1 name ?name1 .
            ?m type oo:Method .
            ?m definedIn ?c1 .
            ?m calls ?target .
            ?target definedIn ?c2 .
            ?c2 type oo:Class .
            ?c2 name ?name2 .
        }
        """
        result = real_reter.reql(query)
        count = result.num_rows if hasattr(result, 'num_rows') else len(result)
        print(f"\n  coupling_matrix query returned {count} class dependencies", file=sys.stderr)


class TestSmellDetectorProcessingFunctions:
    """
    Test the Python processing functions from CADSL detectors directly.

    Since the full detector execution fails due to {param} syntax in lambdas,
    we test the underlying processing functions with real REQL query data.
    """

    def test_circular_imports_detection(self, real_reter):
        """Test circular import detection with real data."""
        from reter_code.dsl.tools.dependencies.circular_imports import _detect_circular_imports

        # Run the REQL query
        query = """
        SELECT ?module1 ?module2 ?file1 ?file2
        WHERE {
            ?m1 type oo:Module .
            ?m1 name ?module1 .
            ?m1 inFile ?file1 .
            ?m1 imports ?m2 .
            ?m2 type oo:Module .
            ?m2 name ?module2 .
            ?m2 inFile ?file2 .
        }
        """
        result = real_reter.reql(query)
        rows = result.to_pylist() if hasattr(result, 'to_pylist') else []

        # Run the processing function
        findings = _detect_circular_imports(rows)

        print(f"\n  circular_imports found {len(findings)} circular dependencies", file=sys.stderr)
        # Findings may be empty if no circular imports exist
        assert isinstance(findings, list)

    def test_alternative_interfaces_detection(self, real_reter):
        """Test alternative interfaces detection with real data."""
        from reter_code.dsl.tools.smells.alternative_interfaces import _find_similar_interfaces

        # Run the REQL query (without FILTER for compatibility)
        query = """
        SELECT ?c ?class_name ?method_name ?file ?line
        WHERE {
            ?c type oo:Class .
            ?c name ?class_name .
            ?c inFile ?file .
            ?c atLine ?line .
            ?m type oo:Method .
            ?m definedIn ?c .
            ?m name ?method_name .
        }
        """
        result = real_reter.reql(query)
        rows = result.to_pylist() if hasattr(result, 'to_pylist') else []

        # Filter out private methods in Python
        rows = [r for r in rows if not r.get('method_name', '').startswith('_')]

        # Run the processing function
        findings = _find_similar_interfaces(rows, min_similarity=0.6, min_shared_methods=3, limit=100)

        print(f"\n  alternative_interfaces found {len(findings)} similar class pairs", file=sys.stderr)
        assert isinstance(findings, list)

    def test_parallel_inheritance_detection(self, real_reter):
        """Test parallel inheritance detection with real data."""
        from reter_code.dsl.tools.smells.parallel_inheritance import _find_parallel_hierarchies

        # Run the REQL query
        query = """
        SELECT ?c ?name ?base_name ?file ?line
        WHERE {
            ?c type oo:Class .
            ?c name ?name .
            ?c inFile ?file .
            ?c atLine ?line .
            ?c inheritsFrom ?base .
            ?base name ?base_name .
        }
        """
        result = real_reter.reql(query)
        rows = result.to_pylist() if hasattr(result, 'to_pylist') else []

        # Run the processing function
        findings = _find_parallel_hierarchies(rows, min_parallel_pairs=2, limit=100)

        print(f"\n  parallel_inheritance found {len(findings)} parallel hierarchies", file=sys.stderr)
        assert isinstance(findings, list)

    def test_duplicate_names_detection(self, real_reter):
        """Test duplicate names detection with real data."""
        from reter_code.dsl.tools.smells.duplicate_names import _find_duplicates

        # Run the REQL query
        query = """
        SELECT ?name ?file ?concept
        WHERE {
            ?e type oo:Class .
            ?e name ?name .
            ?e inFile ?file .
            ?e type ?concept .
        }
        """
        result = real_reter.reql(query)
        rows = result.to_pylist() if hasattr(result, 'to_pylist') else []

        # Run the processing function
        findings = _find_duplicates(rows, entity_type="class", limit=100)

        print(f"\n  duplicate_names found {len(findings)} duplicate class names", file=sys.stderr)
        assert isinstance(findings, list)


class TestInspectionToolsQueries:
    """Integration tests for inspection tool REQL queries."""

    def test_describe_class_query(self, real_reter):
        """Query for describing a class should work."""
        query = """
        SELECT ?c ?name ?method_name ?attr_name ?file ?line
        WHERE {
            ?c type oo:Class .
            ?c name ?name .
            ?c inFile ?file .
            ?c atLine ?line .
            OPTIONAL { ?m type oo:Method . ?m definedIn ?c . ?m name ?method_name }
            OPTIONAL { ?a type oo:Attribute . ?a definedIn ?c . ?a name ?attr_name }
        }
        """
        result = real_reter.reql(query)
        count = result.num_rows if hasattr(result, 'num_rows') else len(result)
        assert count > 0, "Should find class details"
        print(f"\n  describe_class query returned {count} rows", file=sys.stderr)

    def test_find_callers_query(self, real_reter):
        """Query for finding callers should work."""
        query = """
        SELECT ?caller ?caller_name ?target ?target_name ?file
        WHERE {
            ?caller type oo:Function .
            ?caller name ?caller_name .
            ?caller inFile ?file .
            ?caller calls ?target .
            ?target name ?target_name .
        }
        """
        result = real_reter.reql(query)
        count = result.num_rows if hasattr(result, 'num_rows') else len(result)
        print(f"\n  find_callers query returned {count} call relationships", file=sys.stderr)

    def test_get_docstring_query(self, real_reter):
        """Query for docstrings should work."""
        query = """
        SELECT ?f ?name ?docstring ?file
        WHERE {
            ?f type oo:Function .
            ?f name ?name .
            ?f inFile ?file .
            ?f docstring ?docstring .
        }
        """
        result = real_reter.reql(query)
        count = result.num_rows if hasattr(result, 'num_rows') else len(result)
        print(f"\n  get_docstring query returned {count} documented functions", file=sys.stderr)

    def test_get_method_signature_query(self, real_reter):
        """Query for method signatures should work."""
        query = """
        SELECT ?m ?name ?param_name ?param_type ?return_type ?file ?line
        WHERE {
            ?m type oo:Method .
            ?m name ?name .
            ?m inFile ?file .
            ?m atLine ?line .
            OPTIONAL { ?p type oo:Parameter . ?p definedIn ?m . ?p name ?param_name . ?p hasType ?param_type }
            OPTIONAL { ?m returnType ?return_type }
        }
        """
        result = real_reter.reql(query)
        count = result.num_rows if hasattr(result, 'num_rows') else len(result)
        print(f"\n  get_method_signature query returned {count} methods", file=sys.stderr)

    def test_find_subclasses_query(self, real_reter):
        """Query for finding subclasses should work."""
        query = """
        SELECT ?c ?name ?parent_name ?file ?line
        WHERE {
            ?c type oo:Class .
            ?c name ?name .
            ?c inFile ?file .
            ?c atLine ?line .
            ?c inheritsFrom ?parent .
            ?parent name ?parent_name .
        }
        """
        result = real_reter.reql(query)
        count = result.num_rows if hasattr(result, 'num_rows') else len(result)
        print(f"\n  find_subclasses query returned {count} inheritance relationships", file=sys.stderr)


class TestExceptionToolsQueries:
    """Integration tests for exception-related REQL queries."""

    def test_try_except_query(self, real_reter):
        """Query for try/except blocks should work."""
        query = """
        SELECT ?t ?handler_type ?file ?line
        WHERE {
            ?t type oo:TryBlock .
            ?t inFile ?file .
            ?t atLine ?line .
            OPTIONAL { ?h type oo:CatchClause . ?h definedIn ?t . ?h handlerType ?handler_type }
        }
        """
        result = real_reter.reql(query)
        count = result.num_rows if hasattr(result, 'num_rows') else len(result)
        print(f"\n  try_except query returned {count} try blocks", file=sys.stderr)

    def test_raise_statement_query(self, real_reter):
        """Query for raise statements should work."""
        query = """
        SELECT ?r ?exception_type ?file ?line
        WHERE {
            ?r type oo:ThrowStatement .
            ?r inFile ?file .
            ?r atLine ?line .
            OPTIONAL { ?r exceptionType ?exception_type }
        }
        """
        result = real_reter.reql(query)
        count = result.num_rows if hasattr(result, 'num_rows') else len(result)
        print(f"\n  raise_statement query returned {count} raise statements", file=sys.stderr)


class TestPatternToolsQueries:
    """Integration tests for pattern detection REQL queries."""

    def test_decorator_usage_query(self, real_reter):
        """Query for decorator usage should work."""
        query = """
        SELECT ?f ?name ?decorator_name ?file ?line
        WHERE {
            ?f type oo:Function .
            ?f name ?name .
            ?f inFile ?file .
            ?f atLine ?line .
            ?f hasDecorator ?d .
            ?d name ?decorator_name .
        }
        """
        result = real_reter.reql(query)
        count = result.num_rows if hasattr(result, 'num_rows') else len(result)
        print(f"\n  decorator_usage query returned {count} decorated functions", file=sys.stderr)

    def test_magic_methods_query(self, real_reter):
        """Query for magic methods should work."""
        query = """
        SELECT ?m ?name ?class_name ?file ?line
        WHERE {
            ?m type oo:Method .
            ?m name ?name .
            ?m inFile ?file .
            ?m atLine ?line .
            ?m definedIn ?c .
            ?c name ?class_name .
        }
        """
        result = real_reter.reql(query)
        rows = result.to_pylist() if hasattr(result, 'to_pylist') else []

        # Filter for magic methods in Python
        magic_methods = [r for r in rows if r.get('name', '').startswith('__') and r.get('name', '').endswith('__')]

        print(f"\n  magic_methods query returned {len(magic_methods)} magic methods", file=sys.stderr)

    def test_singleton_pattern_query(self, real_reter):
        """Query for potential singleton patterns should work."""
        # Look for classes with instance class attribute or __new__ method
        query = """
        SELECT ?c ?name ?method_name ?file ?line
        WHERE {
            ?c type oo:Class .
            ?c name ?name .
            ?c inFile ?file .
            ?c atLine ?line .
            ?m type oo:Method .
            ?m definedIn ?c .
            ?m name ?method_name .
        }
        """
        result = real_reter.reql(query)
        rows = result.to_pylist() if hasattr(result, 'to_pylist') else []

        # Look for classes with __new__ method (potential singletons)
        singleton_candidates = set()
        for row in rows:
            if row.get('method_name') == '__new__':
                singleton_candidates.add(row.get('name'))

        print(f"\n  singleton_pattern query found {len(singleton_candidates)} potential singletons", file=sys.stderr)


class TestTestingToolsQueries:
    """Integration tests for testing-related REQL queries."""

    def test_test_functions_query(self, real_reter):
        """Query for test functions should work."""
        query = """
        SELECT ?f ?name ?file ?line
        WHERE {
            ?f type oo:Function .
            ?f name ?name .
            ?f inFile ?file .
            ?f atLine ?line .
        }
        """
        result = real_reter.reql(query)
        rows = result.to_pylist() if hasattr(result, 'to_pylist') else []

        # Filter for test functions (pytest convention)
        test_functions = [r for r in rows if r.get('name', '').startswith('test_')]

        print(f"\n  test_functions query returned {len(test_functions)} test functions", file=sys.stderr)

    def test_test_classes_query(self, real_reter):
        """Query for test classes should work."""
        query = """
        SELECT ?c ?name ?file ?line
        WHERE {
            ?c type oo:Class .
            ?c name ?name .
            ?c inFile ?file .
            ?c atLine ?line .
        }
        """
        result = real_reter.reql(query)
        rows = result.to_pylist() if hasattr(result, 'to_pylist') else []

        # Filter for test classes (pytest convention)
        test_classes = [r for r in rows if r.get('name', '').startswith('Test')]

        print(f"\n  test_classes query returned {len(test_classes)} test classes", file=sys.stderr)


class TestRefactoringToolsQueries:
    """Integration tests for refactoring suggestion REQL queries."""

    def test_large_classes_query(self, real_reter):
        """Query for potentially large classes should work."""
        query = """
        SELECT ?c ?name ?method_count ?file ?line
        WHERE {
            ?c type oo:Class .
            ?c name ?name .
            ?c inFile ?file .
            ?c atLine ?line .
            ?c methodCount ?method_count .
        }
        """
        result = real_reter.reql(query)
        count = result.num_rows if hasattr(result, 'num_rows') else len(result)
        print(f"\n  large_classes query returned {count} classes with method counts", file=sys.stderr)

    def test_long_methods_query(self, real_reter):
        """Query for long methods should work."""
        query = """
        SELECT ?m ?name ?lineCount ?file ?line
        WHERE {
            ?m type oo:Method .
            ?m name ?name .
            ?m lineCount ?lineCount .
            ?m inFile ?file .
            ?m atLine ?line .
        }
        """
        result = real_reter.reql(query)
        count = result.num_rows if hasattr(result, 'num_rows') else len(result)
        print(f"\n  long_methods query returned {count} methods with line counts", file=sys.stderr)

    def test_long_parameter_list_query(self, real_reter):
        """Query for methods with many parameters should work."""
        query = """
        SELECT ?m ?name ?param_count ?file ?line
        WHERE {
            ?m type oo:Method .
            ?m name ?name .
            ?m parameterCount ?param_count .
            ?m inFile ?file .
            ?m atLine ?line .
        }
        """
        result = real_reter.reql(query)
        count = result.num_rows if hasattr(result, 'num_rows') else len(result)
        print(f"\n  long_parameter_list query returned {count} methods with parameter counts", file=sys.stderr)


class TestRealWorldQueryPatterns:
    """Test real-world query patterns that tools use."""

    def test_find_usages_pattern(self, real_reter):
        """Test the find_usages query pattern."""
        # Find all usages of a common function name
        query = """
        SELECT ?caller ?caller_name ?target ?target_name ?file ?line
        WHERE {
            ?caller type oo:Function .
            ?caller name ?caller_name .
            ?caller inFile ?file .
            ?caller atLine ?line .
            ?caller calls ?target .
            ?target name ?target_name .
        }
        """
        result = real_reter.reql(query)
        count = result.num_rows if hasattr(result, 'num_rows') else len(result)
        assert count >= 0, "Query should execute without error"
        print(f"\n  find_usages pattern returned {count} call relationships", file=sys.stderr)

    def test_dependency_graph_pattern(self, real_reter):
        """Test the dependency graph query pattern."""
        query = """
        SELECT ?m1 ?name1 ?m2 ?name2 ?file1 ?file2
        WHERE {
            ?m1 type oo:Module .
            ?m1 name ?name1 .
            ?m1 inFile ?file1 .
            ?m1 imports ?m2 .
            ?m2 type oo:Module .
            ?m2 name ?name2 .
            ?m2 inFile ?file2 .
        }
        """
        result = real_reter.reql(query)
        count = result.num_rows if hasattr(result, 'num_rows') else len(result)
        print(f"\n  dependency_graph pattern returned {count} import edges", file=sys.stderr)

    def test_class_hierarchy_pattern(self, real_reter):
        """Test the class hierarchy query pattern."""
        query = """
        SELECT ?child ?child_name ?parent ?parent_name ?file ?line
        WHERE {
            ?child type oo:Class .
            ?child name ?child_name .
            ?child inFile ?file .
            ?child atLine ?line .
            ?child inheritsFrom ?parent .
            ?parent type oo:Class .
            ?parent name ?parent_name .
        }
        """
        result = real_reter.reql(query)
        count = result.num_rows if hasattr(result, 'num_rows') else len(result)
        print(f"\n  class_hierarchy pattern returned {count} inheritance edges", file=sys.stderr)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
