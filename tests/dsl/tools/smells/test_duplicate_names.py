"""
Unit tests for duplicate_names CADSL detector.

Verifies detection of entities with duplicate names across modules.
Uses Python grouping with defaultdict.
"""

import pytest
from unittest.mock import Mock
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'src'))

from reter_code.dsl.tools.smells.duplicate_names import (
    duplicate_names,
    _find_duplicates
)
from reter_code.dsl.core import Pipeline, Context


class TestDuplicateNamesStructure:
    """Test duplicate_names detector structure."""

    def test_has_cadsl_spec(self):
        """Tool should have CADSL spec attached."""
        assert hasattr(duplicate_names, '_cadsl_spec')

    def test_spec_has_correct_name(self):
        """Tool spec should have correct name."""
        spec = duplicate_names._cadsl_spec
        assert spec.name == "duplicate_names"

    def test_has_entity_type_param(self):
        """Should have entity_type parameter."""
        spec = duplicate_names._cadsl_spec
        assert 'entity_type' in spec.params


class TestDuplicateNamesPipeline:
    """Test duplicate_names pipeline structure."""

    @pytest.fixture
    def pipeline(self):
        """Get the pipeline."""
        spec = duplicate_names._cadsl_spec
        ctx = Context(reter=None, params={"entity_type": "class"}, language="oo", instance_name="default")
        return spec.pipeline_factory(ctx)

    def test_pipeline_is_pipeline_object(self, pipeline):
        """Should return a Pipeline."""
        assert isinstance(pipeline, Pipeline)

    def test_has_emit_key(self, pipeline):
        """Should emit 'findings'."""
        assert pipeline._emit_key == "findings"

    def test_reql_queries_class_and_function(self, pipeline):
        """REQL should query Class and Function types."""
        query = pipeline._source.query
        assert "Class" in query or "Function" in query


class TestDuplicateDetection:
    """Test the duplicate detection algorithm."""

    def test_finds_duplicates(self):
        """Should find names appearing in multiple files."""
        rows = [
            {"name": "Config", "concept": "oo:Class", "file": "a/config.py"},
            {"name": "Config", "concept": "oo:Class", "file": "b/config.py"},
            {"name": "Utils", "concept": "oo:Class", "file": "utils.py"},
        ]
        findings = _find_duplicates(rows, entity_type="class", limit=100)
        assert len(findings) == 1
        assert findings[0]["name"] == "Config"
        assert findings[0]["count"] == 2

    def test_ignores_unique_names(self):
        """Should not report unique names."""
        rows = [
            {"name": "Foo", "concept": "oo:Class", "file": "foo.py"},
            {"name": "Bar", "concept": "oo:Class", "file": "bar.py"},
            {"name": "Baz", "concept": "oo:Class", "file": "baz.py"},
        ]
        findings = _find_duplicates(rows, entity_type="class", limit=100)
        assert len(findings) == 0

    def test_filters_by_entity_type(self):
        """Should filter by entity type."""
        rows = [
            {"name": "Config", "concept": "oo:Class", "file": "a.py"},
            {"name": "Config", "concept": "oo:Function", "file": "b.py"},
        ]
        # Filter for class only
        findings = _find_duplicates(rows, entity_type="class", limit=100)
        assert len(findings) == 0  # Only one class named Config

    def test_all_entity_type_includes_both(self):
        """Should include both types when entity_type is 'all'."""
        rows = [
            {"name": "process", "concept": "oo:Class", "file": "a.py"},
            {"name": "process", "concept": "oo:Function", "file": "b.py"},
        ]
        findings = _find_duplicates(rows, entity_type="all", limit=100)
        assert len(findings) == 1
        assert findings[0]["count"] == 2

    def test_sorts_by_count_descending(self):
        """Should sort results by count descending."""
        rows = [
            {"name": "Foo", "concept": "oo:Class", "file": "a.py"},
            {"name": "Foo", "concept": "oo:Class", "file": "b.py"},
            {"name": "Bar", "concept": "oo:Class", "file": "c.py"},
            {"name": "Bar", "concept": "oo:Class", "file": "d.py"},
            {"name": "Bar", "concept": "oo:Class", "file": "e.py"},
        ]
        findings = _find_duplicates(rows, entity_type="class", limit=100)
        assert len(findings) == 2
        assert findings[0]["name"] == "Bar"  # 3 occurrences
        assert findings[1]["name"] == "Foo"  # 2 occurrences


class TestDuplicateNamesExecution:
    """Test execution with mock context."""

    @pytest.fixture
    def mock_context(self):
        """Create mock context."""
        ctx = Context(
            reter=Mock(),
            params={"entity_type": "class", "limit": 100},
            language="oo",
            instance_name="default"
        )

        mock_table = Mock()
        mock_table.num_rows = 3
        mock_table.to_pylist.return_value = [
            {"?name": "Config", "?type": "class", "?file": "a/config.py"},
            {"?name": "Config", "?type": "class", "?file": "b/config.py"},
            {"?name": "Utils", "?type": "class", "?file": "utils.py"},
        ]
        ctx.reter.reql.return_value = mock_table

        return ctx

    def test_execute_returns_success(self, mock_context):
        """Execution should return success."""
        result = duplicate_names(ctx=mock_context)
        assert "success" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
