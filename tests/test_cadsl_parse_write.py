"""Tests for parse_file source and write_file step in CADSL pipeline."""

import json
import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from reter_code.dsl.core import (
    Context, ParseFileSource, Pipeline,
    pipeline_ok, pipeline_err,
    _get_project_root,
)
from reter_code.cadsl.transformer import WriteFileStep


# ============================================================
# ParseFileSource tests
# ============================================================

class TestParseFileSource:
    def _make_ctx(self, root: str) -> Context:
        """Create a context with a mock reter having project_root."""
        class FakeReter:
            project_root = root
        return Context(reter=FakeReter(), params={})

    def test_parse_csv(self, tmp_path):
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("name,score\nalice,90\nbob,85\n", encoding="utf-8")

        src = ParseFileSource(path="data.csv", format="csv")
        ctx = self._make_ctx(str(tmp_path))
        result = src.execute(ctx)

        assert result.is_ok
        rows = result.value
        assert len(rows) == 2
        assert rows[0]["name"] == "alice"
        assert rows[0]["score"] == 90
        assert rows[1]["name"] == "bob"

    def test_parse_json(self, tmp_path):
        json_file = tmp_path / "data.json"
        data = [{"name": "alice", "score": 90}, {"name": "bob", "score": 85}]
        json_file.write_text(json.dumps(data), encoding="utf-8")

        src = ParseFileSource(path="data.json", format="json")
        ctx = self._make_ctx(str(tmp_path))
        result = src.execute(ctx)

        assert result.is_ok
        rows = result.value
        assert len(rows) == 2
        assert rows[0]["name"] == "alice"

    def test_parse_parquet(self, tmp_path):
        parquet_file = tmp_path / "data.parquet"
        df = pd.DataFrame({"name": ["alice", "bob"], "score": [90, 85]})
        df.to_parquet(parquet_file, index=False)

        src = ParseFileSource(path="data.parquet", format="parquet")
        ctx = self._make_ctx(str(tmp_path))
        result = src.execute(ctx)

        assert result.is_ok
        rows = result.value
        assert len(rows) == 2
        assert rows[0]["name"] == "alice"

    def test_parse_csv_with_columns(self, tmp_path):
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("name,score,grade\nalice,90,A\nbob,85,B\n", encoding="utf-8")

        src = ParseFileSource(path="data.csv", format="csv", columns=["name", "grade"])
        ctx = self._make_ctx(str(tmp_path))
        result = src.execute(ctx)

        assert result.is_ok
        rows = result.value
        assert len(rows) == 2
        assert "name" in rows[0]
        assert "grade" in rows[0]
        assert "score" not in rows[0]

    def test_parse_csv_with_limit(self, tmp_path):
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("name,score\nalice,90\nbob,85\ncharlie,80\n", encoding="utf-8")

        src = ParseFileSource(path="data.csv", format="csv", limit=2)
        ctx = self._make_ctx(str(tmp_path))
        result = src.execute(ctx)

        assert result.is_ok
        rows = result.value
        assert len(rows) == 2

    def test_parse_file_not_found(self, tmp_path):
        src = ParseFileSource(path="nonexistent.csv", format="csv")
        ctx = self._make_ctx(str(tmp_path))
        result = src.execute(ctx)

        assert result.is_err
        assert "File not found" in str(result.error)

    def test_parse_csv_with_separator(self, tmp_path):
        csv_file = tmp_path / "data.tsv"
        csv_file.write_text("name\tscore\nalice\t90\nbob\t85\n", encoding="utf-8")

        src = ParseFileSource(path="data.tsv", format="csv", separator="\t")
        ctx = self._make_ctx(str(tmp_path))
        result = src.execute(ctx)

        assert result.is_ok
        rows = result.value
        assert len(rows) == 2
        assert rows[0]["name"] == "alice"

    def test_parse_unsupported_format(self, tmp_path):
        src = ParseFileSource(path="data.xml", format="xml")
        (tmp_path / "data.xml").write_text("<root/>")
        ctx = self._make_ctx(str(tmp_path))
        result = src.execute(ctx)

        assert result.is_err
        assert "Unsupported format" in str(result.error)


# ============================================================
# WriteFileStep tests
# ============================================================

class TestWriteFileStep:
    def _make_ctx(self, root: str) -> Context:
        class FakeReter:
            project_root = root
        return Context(reter=FakeReter(), params={})

    def test_write_csv(self, tmp_path):
        step = WriteFileStep(path="output.csv", format="csv")
        ctx = self._make_ctx(str(tmp_path))
        data = [{"name": "alice", "score": 90}, {"name": "bob", "score": 85}]

        result = step.execute(data, ctx)

        assert result.is_ok
        assert result.value == data  # pass-through

        out_file = tmp_path / "output.csv"
        assert out_file.exists()
        df = pd.read_csv(out_file)
        assert len(df) == 2
        assert list(df.columns) == ["name", "score"]

    def test_write_json(self, tmp_path):
        step = WriteFileStep(path="output.json", format="json", indent=4)
        ctx = self._make_ctx(str(tmp_path))
        data = [{"name": "alice", "score": 90}]

        result = step.execute(data, ctx)

        assert result.is_ok
        out_file = tmp_path / "output.json"
        assert out_file.exists()
        loaded = json.loads(out_file.read_text(encoding="utf-8"))
        assert len(loaded) == 1
        assert loaded[0]["name"] == "alice"

    def test_write_parquet(self, tmp_path):
        step = WriteFileStep(path="output.parquet", format="parquet")
        ctx = self._make_ctx(str(tmp_path))
        data = [{"name": "alice", "score": 90}]

        result = step.execute(data, ctx)

        assert result.is_ok
        out_file = tmp_path / "output.parquet"
        assert out_file.exists()
        df = pd.read_parquet(out_file)
        assert len(df) == 1

    def test_write_creates_directories(self, tmp_path):
        step = WriteFileStep(path="sub/dir/output.csv", format="csv")
        ctx = self._make_ctx(str(tmp_path))
        data = [{"x": 1}]

        result = step.execute(data, ctx)

        assert result.is_ok
        assert (tmp_path / "sub" / "dir" / "output.csv").exists()

    def test_write_no_overwrite(self, tmp_path):
        out_file = tmp_path / "existing.csv"
        out_file.write_text("old data")

        step = WriteFileStep(path="existing.csv", format="csv", overwrite=False)
        ctx = self._make_ctx(str(tmp_path))

        result = step.execute([{"x": 1}], ctx)

        assert result.is_err
        assert "overwrite=false" in str(result.error)

    def test_write_passthrough(self, tmp_path):
        step = WriteFileStep(path="out.csv", format="csv")
        ctx = self._make_ctx(str(tmp_path))
        data = [{"a": 1}, {"a": 2}]

        result = step.execute(data, ctx)

        assert result.is_ok
        assert result.value is data  # same object

    def test_write_arrow_table(self, tmp_path):
        import pyarrow as pa
        table = pa.table({"name": ["alice", "bob"], "score": [90, 85]})

        step = WriteFileStep(path="out.csv", format="csv")
        ctx = self._make_ctx(str(tmp_path))

        result = step.execute(table, ctx)

        assert result.is_ok
        df = pd.read_csv(tmp_path / "out.csv")
        assert len(df) == 2

    def test_write_auto_generates_temp_file(self, tmp_path):
        """write_file {} with no path auto-generates a file in .reter_code/results/."""
        step = WriteFileStep()  # no path, default format=json
        ctx = self._make_ctx(str(tmp_path))
        data = [{"name": "alice", "score": 90}]

        result = step.execute(data, ctx)

        assert result.is_ok
        assert "_output_file" in result.value
        assert "results" in result.value
        assert result.value["results"] is data

        output_path = Path(result.value["_output_file"])
        assert output_path.exists()
        assert output_path.suffix == ".json"
        assert ".reter_code" in str(output_path)
        assert "results" in str(output_path)

        # Verify content
        loaded = json.loads(output_path.read_text(encoding="utf-8"))
        assert len(loaded) == 1
        assert loaded[0]["name"] == "alice"

    def test_write_auto_generates_csv(self, tmp_path):
        """write_file with format=csv but no path auto-generates a .csv."""
        step = WriteFileStep(format="csv")
        ctx = self._make_ctx(str(tmp_path))
        data = [{"x": 1, "y": 2}]

        result = step.execute(data, ctx)

        assert result.is_ok
        output_path = Path(result.value["_output_file"])
        assert output_path.suffix == ".csv"
        df = pd.read_csv(output_path)
        assert len(df) == 1

    def test_write_auto_creates_results_dir(self, tmp_path):
        """Auto-generated path creates .reter_code/results/ if missing."""
        step = WriteFileStep()
        ctx = self._make_ctx(str(tmp_path))

        result = step.execute([{"a": 1}], ctx)

        assert result.is_ok
        results_dir = tmp_path / ".reter_code" / "results"
        assert results_dir.exists()

    def test_write_auto_unique_filenames(self, tmp_path):
        """Two auto-generated writes produce different filenames."""
        step = WriteFileStep()
        ctx = self._make_ctx(str(tmp_path))

        r1 = step.execute([{"a": 1}], ctx)
        r2 = step.execute([{"b": 2}], ctx)

        assert r1.is_ok and r2.is_ok
        assert r1.value["_output_file"] != r2.value["_output_file"]

    def test_write_explicit_path_no_output_file_key(self, tmp_path):
        """Explicit path does NOT add _output_file — pure pass-through."""
        step = WriteFileStep(path="out.json", format="json")
        ctx = self._make_ctx(str(tmp_path))
        data = [{"a": 1}]

        result = step.execute(data, ctx)

        assert result.is_ok
        assert result.value is data  # raw pass-through, not wrapped

    def test_default_format_is_json(self):
        """Default format should be json."""
        step = WriteFileStep()
        assert step.format == "json"


# ============================================================
# Grammar integration tests
# ============================================================

class TestGrammarIntegration:
    @pytest.fixture
    def parser(self):
        from lark import Lark
        grammar_path = Path(__file__).parent.parent / "src" / "reter_code" / "cadsl" / "grammar.lark"
        return Lark.open(str(grammar_path), start="start", parser="earley")

    def test_parse_file_grammar(self, parser):
        cadsl = '''
        query test_parse() {
            parse_file { path: "data/input.csv", format: csv }
            | emit { results }
        }
        '''
        tree = parser.parse(cadsl)
        assert tree is not None
        # Check parse_file_source appears in tree
        sources = list(tree.find_data("parse_file_source"))
        assert len(sources) == 1

    def test_parse_file_with_options_grammar(self, parser):
        cadsl = '''
        query test_parse() {
            parse_file { path: "data/input.csv", format: csv, encoding: "utf-8", limit: 100 }
            | emit { results }
        }
        '''
        tree = parser.parse(cadsl)
        assert tree is not None

    def test_parse_file_json_grammar(self, parser):
        cadsl = '''
        query test_parse() {
            parse_file { path: "data.json", format: json }
            | emit { results }
        }
        '''
        tree = parser.parse(cadsl)
        sources = list(tree.find_data("parse_file_source"))
        assert len(sources) == 1

    def test_parse_file_with_columns_grammar(self, parser):
        cadsl = '''
        query test_parse() {
            parse_file { path: "data.csv", format: csv, columns: ["name", "score"] }
            | emit { results }
        }
        '''
        tree = parser.parse(cadsl)
        assert tree is not None

    def test_parse_file_with_param_ref_grammar(self, parser):
        cadsl = '''
        query test_parse() {
            param input_path: str;
            parse_file { path: {input_path}, format: csv }
            | emit { results }
        }
        '''
        tree = parser.parse(cadsl)
        assert tree is not None

    def test_write_file_grammar(self, parser):
        cadsl = '''
        query test_write() {
            value { [] }
            | write_file { path: "output.csv", format: csv }
            | emit { results }
        }
        '''
        tree = parser.parse(cadsl)
        steps = list(tree.find_data("write_file_step"))
        assert len(steps) == 1

    def test_write_file_json_grammar(self, parser):
        cadsl = '''
        query test_write() {
            value { [] }
            | write_file { path: "output.json", format: json, indent: 4 }
            | emit { results }
        }
        '''
        tree = parser.parse(cadsl)
        steps = list(tree.find_data("write_file_step"))
        assert len(steps) == 1

    def test_write_file_with_overwrite_grammar(self, parser):
        cadsl = '''
        query test_write() {
            value { [] }
            | write_file { path: "out.csv", format: csv, overwrite: false }
            | emit { results }
        }
        '''
        tree = parser.parse(cadsl)
        assert tree is not None

    def test_write_file_empty_braces_grammar(self, parser):
        """write_file { } with no params should parse successfully."""
        cadsl = '''
        query test_write() {
            value { [] }
            | write_file { }
            | emit { results }
        }
        '''
        tree = parser.parse(cadsl)
        steps = list(tree.find_data("write_file_step"))
        assert len(steps) == 1

    def test_parse_file_in_merge_grammar(self, parser):
        cadsl = '''
        query test_merge() {
            merge {
                parse_file { path: "a.csv", format: csv },
                parse_file { path: "b.csv", format: csv }
            }
            | emit { results }
        }
        '''
        tree = parser.parse(cadsl)
        sources = list(tree.find_data("parse_file_source"))
        assert len(sources) == 2


# ============================================================
# Transformer integration tests
# ============================================================

class TestTransformerIntegration:
    @pytest.fixture
    def transform(self):
        from reter_code.cadsl.transformer import CADSLTransformer
        from lark import Lark
        grammar_path = Path(__file__).parent.parent / "src" / "reter_code" / "cadsl" / "grammar.lark"
        parser = Lark.open(str(grammar_path), start="start", parser="earley")

        def _transform(cadsl_text):
            tree = parser.parse(cadsl_text)
            transformer = CADSLTransformer()
            return transformer.transform(tree)
        return _transform

    def test_parse_file_transformer(self, transform):
        specs = transform('''
        query test() {
            parse_file { path: "data.csv", format: csv }
            | emit { results }
        }
        ''')
        assert len(specs) == 1
        spec = specs[0]
        assert spec.source_type == "parse_file"
        assert spec.rag_params["path"] == "data.csv"
        assert spec.rag_params["format"] == "csv"

    def test_parse_file_with_options_transformer(self, transform):
        specs = transform('''
        query test() {
            parse_file { path: "data.csv", format: json, encoding: "latin-1", limit: 50 }
            | emit { results }
        }
        ''')
        spec = specs[0]
        assert spec.rag_params["path"] == "data.csv"
        assert spec.rag_params["format"] == "json"
        assert spec.rag_params["encoding"] == "latin-1"
        assert spec.rag_params["limit"] == 50

    def test_write_file_transformer(self, transform):
        specs = transform('''
        query test() {
            value { [] }
            | write_file { path: "out.json", format: json, indent: 4 }
            | emit { results }
        }
        ''')
        spec = specs[0]
        # Find write_file step
        write_steps = [s for s in spec.steps if s.get("type") == "write_file"]
        assert len(write_steps) == 1
        ws = write_steps[0]
        assert ws["path"] == "out.json"
        assert ws["format"] == "json"
        assert ws["indent"] == 4

    def test_write_file_overwrite_false_transformer(self, transform):
        specs = transform('''
        query test() {
            value { [] }
            | write_file { path: "out.csv", format: csv, overwrite: false }
            | emit { results }
        }
        ''')
        spec = specs[0]
        write_steps = [s for s in spec.steps if s.get("type") == "write_file"]
        assert len(write_steps) == 1
        assert write_steps[0]["overwrite"] is False

    def test_write_file_empty_braces_transformer(self, transform):
        """write_file { } should produce defaults: empty path, json format."""
        specs = transform('''
        query test() {
            value { [] }
            | write_file { }
            | emit { results }
        }
        ''')
        spec = specs[0]
        write_steps = [s for s in spec.steps if s.get("type") == "write_file"]
        assert len(write_steps) == 1
        ws = write_steps[0]
        assert ws["path"] == ""
        assert ws["format"] == "json"
