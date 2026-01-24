"""
Unit tests for parallel_inheritance CADSL detector.

Verifies detection of parallel inheritance hierarchies.
Uses Python pattern matching for class naming analysis.
"""

import pytest
from unittest.mock import Mock
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'src'))

from reter_code.dsl.tools.smells.parallel_inheritance import (
    parallel_inheritance,
    _find_parallel_hierarchies,
    _find_parallel_pairs,
    _names_are_parallel
)
from reter_code.dsl.core import Pipeline, Context


class TestParallelInheritanceStructure:
    """Test parallel_inheritance detector structure."""

    def test_has_cadsl_spec(self):
        """Tool should have CADSL spec attached."""
        assert hasattr(parallel_inheritance, '_cadsl_spec')

    def test_spec_has_correct_name(self):
        """Tool spec should have correct name."""
        spec = parallel_inheritance._cadsl_spec
        assert spec.name == "parallel_inheritance"

    def test_has_min_parallel_pairs_param(self):
        """Should have min_parallel_pairs parameter."""
        spec = parallel_inheritance._cadsl_spec
        assert 'min_parallel_pairs' in spec.params


class TestParallelInheritancePipeline:
    """Test parallel_inheritance pipeline structure."""

    @pytest.fixture
    def pipeline(self):
        """Get the pipeline."""
        spec = parallel_inheritance._cadsl_spec
        ctx = Context(reter=None, params={"min_parallel_pairs": 2}, language="oo", instance_name="default")
        return spec.pipeline_factory(ctx)

    def test_pipeline_is_pipeline_object(self, pipeline):
        """Should return a Pipeline."""
        assert isinstance(pipeline, Pipeline)

    def test_has_emit_key(self, pipeline):
        """Should emit 'findings'."""
        assert pipeline._emit_key == "findings"

    def test_reql_queries_inheritance(self, pipeline):
        """REQL should query inheritance relationships."""
        query = pipeline._source.query
        assert "Class" in query
        assert "inheritsFrom" in query


class TestNamePatternMatching:
    """Test the name pattern matching algorithm."""

    def test_matching_suffix(self):
        """Should match names with same suffix."""
        assert _names_are_parallel("FooHandler", "BarHandler")
        assert _names_are_parallel("FooService", "BarService")
        assert _names_are_parallel("UserController", "OrderController")

    def test_matching_prefix(self):
        """Should match names with same prefix."""
        assert _names_are_parallel("AbstractFoo", "AbstractBar")
        assert _names_are_parallel("BaseFoo", "BaseBar")

    def test_not_matching_different_patterns(self):
        """Should not match names with different patterns."""
        assert not _names_are_parallel("FooHandler", "BarService")
        assert not _names_are_parallel("Foo", "Bar")

    def test_not_matching_same_name(self):
        """Should not match identical names."""
        assert not _names_are_parallel("FooHandler", "FooHandler")

    def test_short_names_not_matched(self):
        """Should not match very short names."""
        assert not _names_are_parallel("Fo", "Ba")


class TestParallelPairFinding:
    """Test finding parallel pairs between hierarchies."""

    def test_finds_parallel_pairs(self):
        """Should find classes with matching name patterns."""
        children1 = [
            {"name": "FooHandler"},
            {"name": "BarHandler"},
        ]
        children2 = [
            {"name": "FooService"},
            {"name": "BarService"},
        ]
        pairs = _find_parallel_pairs(children1, children2)
        # FooHandler/FooService and BarHandler/BarService have matching prefix
        assert len(pairs) >= 2

    def test_no_parallel_pairs(self):
        """Should return empty for non-parallel classes."""
        children1 = [{"name": "Alpha"}]
        children2 = [{"name": "Beta"}]
        pairs = _find_parallel_pairs(children1, children2)
        assert len(pairs) == 0


class TestParallelHierarchyDetection:
    """Test the full hierarchy detection."""

    def test_finds_parallel_hierarchies(self):
        """Should find hierarchies with parallel naming patterns."""
        rows = [
            {"c": "c1", "name": "FooHandler", "base_name": "Handler", "file": "handlers.py", "line": 10},
            {"c": "c2", "name": "BarHandler", "base_name": "Handler", "file": "handlers.py", "line": 20},
            {"c": "c3", "name": "FooService", "base_name": "Service", "file": "services.py", "line": 10},
            {"c": "c4", "name": "BarService", "base_name": "Service", "file": "services.py", "line": 20},
        ]
        findings = _find_parallel_hierarchies(rows, min_parallel_pairs=2, limit=100)
        assert len(findings) >= 1
        assert findings[0]["parallel_pairs"] >= 2

    def test_ignores_single_child_hierarchies(self):
        """Should not find hierarchies with only one child each."""
        rows = [
            {"c": "c1", "name": "FooHandler", "base_name": "Handler", "file": "handlers.py", "line": 10},
            {"c": "c2", "name": "BarService", "base_name": "Service", "file": "services.py", "line": 10},
        ]
        findings = _find_parallel_hierarchies(rows, min_parallel_pairs=2, limit=100)
        assert len(findings) == 0


class TestParallelInheritanceExecution:
    """Test execution with mock context."""

    @pytest.fixture
    def mock_context(self):
        """Create mock context."""
        ctx = Context(
            reter=Mock(),
            params={"min_parallel_pairs": 2, "limit": 100},
            language="oo",
            instance_name="default"
        )

        mock_table = Mock()
        mock_table.num_rows = 4
        mock_table.to_pylist.return_value = [
            {"?c": "c1", "?name": "FooHandler", "?base_name": "Handler", "?file": "handlers.py", "?line": "10"},
            {"?c": "c2", "?name": "BarHandler", "?base_name": "Handler", "?file": "handlers.py", "?line": "20"},
            {"?c": "c3", "?name": "FooService", "?base_name": "Service", "?file": "services.py", "?line": "10"},
            {"?c": "c4", "?name": "BarService", "?base_name": "Service", "?file": "services.py", "?line": "20"},
        ]
        ctx.reter.reql.return_value = mock_table

        return ctx

    def test_execute_returns_success(self, mock_context):
        """Execution should return success."""
        result = parallel_inheritance(ctx=mock_context)
        assert "success" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
