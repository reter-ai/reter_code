"""
Unit tests for alternative_interfaces CADSL detector.

Verifies detection of classes with similar behavior but different interfaces.
Uses Python N² comparison for method similarity.
"""

import pytest
from unittest.mock import Mock
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'src'))

from reter_code.dsl.tools.smells.alternative_interfaces import (
    alternative_interfaces,
    _find_similar_interfaces
)
from reter_code.dsl.core import Pipeline, Context


class TestAlternativeInterfacesStructure:
    """Test alternative_interfaces detector structure."""

    def test_has_cadsl_spec(self):
        """Tool should have CADSL spec attached."""
        assert hasattr(alternative_interfaces, '_cadsl_spec')

    def test_spec_has_correct_name(self):
        """Tool spec should have correct name."""
        spec = alternative_interfaces._cadsl_spec
        assert spec.name == "alternative_interfaces"

    def test_has_min_similarity_param(self):
        """Should have min_similarity parameter."""
        spec = alternative_interfaces._cadsl_spec
        assert 'min_similarity' in spec.params

    def test_has_min_shared_methods_param(self):
        """Should have min_shared_methods parameter."""
        spec = alternative_interfaces._cadsl_spec
        assert 'min_shared_methods' in spec.params


class TestAlternativeInterfacesPipeline:
    """Test alternative_interfaces pipeline structure."""

    @pytest.fixture
    def pipeline(self):
        """Get the pipeline."""
        spec = alternative_interfaces._cadsl_spec
        ctx = Context(reter=None, params={"min_similarity": 0.6, "min_shared_methods": 3}, language="oo", instance_name="default")
        return spec.pipeline_factory(ctx)

    def test_pipeline_is_pipeline_object(self, pipeline):
        """Should return a Pipeline."""
        assert isinstance(pipeline, Pipeline)

    def test_has_emit_key(self, pipeline):
        """Should emit 'findings'."""
        assert pipeline._emit_key == "findings"

    def test_reql_queries_classes_and_methods(self, pipeline):
        """REQL should query classes and their methods."""
        query = pipeline._source.query
        assert "Class" in query
        assert "Method" in query
        assert "definedIn" in query


class TestSimilarityCalculation:
    """Test the N² similarity calculation."""

    def test_finds_similar_classes(self):
        """Should find classes with similar methods."""
        rows = [
            {"c": "c1", "class_name": "ClassA", "method_name": "foo", "file": "a.py", "line": 1},
            {"c": "c1", "class_name": "ClassA", "method_name": "bar", "file": "a.py", "line": 1},
            {"c": "c1", "class_name": "ClassA", "method_name": "baz", "file": "a.py", "line": 1},
            {"c": "c2", "class_name": "ClassB", "method_name": "foo", "file": "b.py", "line": 1},
            {"c": "c2", "class_name": "ClassB", "method_name": "bar", "file": "b.py", "line": 1},
            {"c": "c2", "class_name": "ClassB", "method_name": "qux", "file": "b.py", "line": 1},
        ]
        findings = _find_similar_interfaces(rows, min_similarity=0.5, min_shared_methods=2, limit=100)
        assert len(findings) >= 1
        assert findings[0]["shared_methods"] >= 2

    def test_ignores_dissimilar_classes(self):
        """Should not find classes with few shared methods."""
        rows = [
            {"c": "c1", "class_name": "ClassA", "method_name": "foo", "file": "a.py", "line": 1},
            {"c": "c1", "class_name": "ClassA", "method_name": "bar", "file": "a.py", "line": 1},
            {"c": "c2", "class_name": "ClassB", "method_name": "qux", "file": "b.py", "line": 1},
            {"c": "c2", "class_name": "ClassB", "method_name": "baz", "file": "b.py", "line": 1},
        ]
        findings = _find_similar_interfaces(rows, min_similarity=0.6, min_shared_methods=3, limit=100)
        assert len(findings) == 0

    def test_calculates_correct_similarity(self):
        """Should calculate Jaccard similarity correctly."""
        # ClassA: {a, b, c}, ClassB: {a, b, d}
        # Intersection: {a, b} = 2
        # Union: {a, b, c, d} = 4
        # Similarity: 2/4 = 0.5
        rows = [
            {"c": "c1", "class_name": "ClassA", "method_name": "a", "file": "a.py", "line": 1},
            {"c": "c1", "class_name": "ClassA", "method_name": "b", "file": "a.py", "line": 1},
            {"c": "c1", "class_name": "ClassA", "method_name": "c", "file": "a.py", "line": 1},
            {"c": "c2", "class_name": "ClassB", "method_name": "a", "file": "b.py", "line": 1},
            {"c": "c2", "class_name": "ClassB", "method_name": "b", "file": "b.py", "line": 1},
            {"c": "c2", "class_name": "ClassB", "method_name": "d", "file": "b.py", "line": 1},
        ]
        findings = _find_similar_interfaces(rows, min_similarity=0.4, min_shared_methods=2, limit=100)
        assert len(findings) == 1
        assert findings[0]["similarity"] == 0.5


class TestAlternativeInterfacesExecution:
    """Test execution with mock context."""

    @pytest.fixture
    def mock_context(self):
        """Create mock context."""
        ctx = Context(
            reter=Mock(),
            params={"min_similarity": 0.6, "min_shared_methods": 3, "limit": 100},
            language="oo",
            instance_name="default"
        )

        mock_table = Mock()
        mock_table.num_rows = 4
        mock_table.to_pylist.return_value = [
            {"?c": "c1", "?class_name": "Handler", "?method_name": "handle", "?file": "handler.py", "?line": "10"},
            {"?c": "c1", "?class_name": "Handler", "?method_name": "process", "?file": "handler.py", "?line": "10"},
            {"?c": "c2", "?class_name": "Processor", "?method_name": "handle", "?file": "processor.py", "?line": "20"},
            {"?c": "c2", "?class_name": "Processor", "?method_name": "process", "?file": "processor.py", "?line": "20"},
        ]
        ctx.reter.reql.return_value = mock_table

        return ctx

    def test_execute_returns_success(self, mock_context):
        """Execution should return success."""
        result = alternative_interfaces(ctx=mock_context)
        assert "success" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
