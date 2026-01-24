"""
Unit tests for circular_imports CADSL detector.

Verifies detection of circular import dependencies using DFS.
"""

import pytest
from unittest.mock import Mock
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'src'))

from reter_code.dsl.tools.dependencies.circular_imports import (
    find_circular_imports,
    _detect_circular_imports,
    _detect_cycles_dfs,
    _normalize_cycle
)
from reter_code.dsl.core import Pipeline, Context


class TestCircularImportsStructure:
    """Test circular_imports detector structure."""

    def test_has_cadsl_spec(self):
        """Tool should have CADSL spec attached."""
        assert hasattr(find_circular_imports, '_cadsl_spec')

    def test_spec_has_correct_name(self):
        """Tool spec should have correct name."""
        spec = find_circular_imports._cadsl_spec
        assert spec.name == "find_circular_imports"

    def test_has_limit_param(self):
        """Should have limit parameter."""
        spec = find_circular_imports._cadsl_spec
        assert 'limit' in spec.params


class TestCircularImportsPipeline:
    """Test circular_imports pipeline structure."""

    @pytest.fixture
    def pipeline(self):
        """Get the pipeline."""
        spec = find_circular_imports._cadsl_spec
        ctx = Context(reter=None, params={"limit": 100}, language="oo", instance_name="default")
        return spec.pipeline_factory(ctx)

    def test_pipeline_is_pipeline_object(self, pipeline):
        """Should return a Pipeline."""
        assert isinstance(pipeline, Pipeline)

    def test_has_emit_key(self, pipeline):
        """Should emit 'findings'."""
        assert pipeline._emit_key == "findings"

    def test_reql_queries_imports(self, pipeline):
        """REQL should query import relationships."""
        query = pipeline._source.query
        assert "imports" in query
        assert "Module" in query


class TestDFSCycleDetection:
    """Test the DFS cycle detection algorithm."""

    def test_detects_simple_cycle(self):
        """Should detect A -> B -> A cycle."""
        graph = {
            "A": ["B"],
            "B": ["A"]
        }
        cycles = _detect_cycles_dfs(graph)
        assert len(cycles) >= 1
        # Check that A and B are in a cycle together
        cycle_nodes = set()
        for cycle in cycles:
            cycle_nodes.update(cycle)
        assert "A" in cycle_nodes and "B" in cycle_nodes

    def test_detects_longer_cycle(self):
        """Should detect A -> B -> C -> A cycle."""
        graph = {
            "A": ["B"],
            "B": ["C"],
            "C": ["A"]
        }
        cycles = _detect_cycles_dfs(graph)
        assert len(cycles) >= 1

    def test_no_cycle_in_dag(self):
        """Should return empty for DAG."""
        graph = {
            "A": ["B", "C"],
            "B": ["D"],
            "C": ["D"],
            "D": []
        }
        cycles = _detect_cycles_dfs(graph)
        assert len(cycles) == 0

    def test_handles_self_loop(self):
        """Should detect self-loop A -> A."""
        graph = {
            "A": ["A"]
        }
        cycles = _detect_cycles_dfs(graph)
        assert len(cycles) >= 1


class TestCycleNormalization:
    """Test cycle normalization for deduplication."""

    def test_normalizes_to_min_element(self):
        """Should rotate to start with minimum element."""
        cycle = ["C", "A", "B", "C"]
        normalized = _normalize_cycle(cycle)
        assert normalized[0] == "A"

    def test_same_cycle_different_start(self):
        """Same cycle from different starts should normalize identically."""
        cycle1 = ["A", "B", "C", "A"]
        cycle2 = ["B", "C", "A", "B"]
        cycle3 = ["C", "A", "B", "C"]

        n1 = _normalize_cycle(cycle1)
        n2 = _normalize_cycle(cycle2)
        n3 = _normalize_cycle(cycle3)

        assert n1 == n2 == n3


class TestDetectCircularImports:
    """Test the full detection function."""

    def test_converts_rows_to_findings(self):
        """Should convert query rows to findings."""
        rows = [
            {"module1": "A", "module2": "B", "file1": "a.py", "file2": "b.py"},
            {"module1": "B", "module2": "A", "file1": "b.py", "file2": "a.py"}
        ]
        findings = _detect_circular_imports(rows)
        assert len(findings) >= 1
        assert findings[0]["issue"] == "circular_import"

    def test_includes_file_info(self):
        """Should include file information in findings."""
        rows = [
            {"module1": "A", "module2": "B", "file1": "a.py", "file2": "b.py"},
            {"module1": "B", "module2": "A", "file1": "b.py", "file2": "a.py"}
        ]
        findings = _detect_circular_imports(rows)
        if findings:
            assert "file1" in findings[0]
            assert "file2" in findings[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
