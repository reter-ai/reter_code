"""
Unit tests for CADSL (Code Analysis DSL) modules.

Tests cover:
- Core: Pipeline, Step classes, Result types
- Decorators: @query, @detector, @diagram, @param, @meta
- Operators: when, unless, branch, merge, etc.
- Registry: Tool registration and lookup
"""

import pytest
from typing import Dict, Any, List

from src.reter_code.dsl import (
    # Category Theory
    Functor, Applicative, Monad,
    Result, Ok, Err,
    Maybe, Just, Nothing,
    ListF,
    PipelineError, pipeline_ok, pipeline_err,
    # Core
    Pipeline, Query, Detector, Diagram,
    Context, ToolSpec, ParamSpec, ToolType,
    reql, rag, value,
    # Decorators
    query, detector, diagram, param, meta,
    ToolBuilder, query_builder,
    # Operators
    when, unless, identity, tap, compose,
    # Registry
    Registry, namespace
)
from src.reter_code.dsl.core import (
    FilterStep, SelectStep, OrderByStep, LimitStep, MapStep
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(autouse=True)
def clear_registry():
    """Clear registry before each test."""
    Registry.clear()
    yield
    Registry.clear()


@pytest.fixture
def sample_data():
    """Sample data for pipeline tests."""
    return [
        {"name": "Foo", "file": "foo.py", "count": 15},
        {"name": "Bar", "file": "bar.py", "count": 25},
        {"name": "Baz", "file": "test_baz.py", "count": 5},
        {"name": "Qux", "file": "qux.py", "count": 10},
    ]


@pytest.fixture
def mock_context(sample_data):
    """Create a mock context with sample data."""
    class MockReter:
        def __init__(self, data):
            self._data = data

        def query(self, q, query_type="reql"):
            return self._data

    return Context(
        reter=MockReter(sample_data),
        params={"threshold": 10, "exclude_tests": True},
        language="python"
    )


# =============================================================================
# Result Type Tests
# =============================================================================

class TestCatpyFoundations:
    """Tests for catpy category theory foundations."""

    # -------------------------------------------------------------------------
    # Maybe Monad Tests
    # -------------------------------------------------------------------------

    def test_maybe_just_fmap(self):
        result = Just(10).fmap(lambda x: x + 1)
        assert result == Just(11)

    def test_maybe_nothing_fmap(self):
        result = Nothing().fmap(lambda x: x + 1)
        assert isinstance(result, Nothing)

    def test_maybe_just_bind(self):
        result = Just(10).bind(lambda x: Just(x // 2))
        assert result == Just(5)

    def test_maybe_nothing_bind(self):
        result = Nothing().bind(lambda x: Just(x))
        assert isinstance(result, Nothing)

    def test_maybe_get_or_else(self):
        assert Just(42).get_or_else(0) == 42
        assert Nothing().get_or_else(0) == 0

    # -------------------------------------------------------------------------
    # ListF Monad Tests
    # -------------------------------------------------------------------------

    def test_listf_fmap(self):
        xs = ListF.from_iter([1, 2, 3])
        result = xs.fmap(lambda n: n * 2)
        assert result == ListF.from_iter([2, 4, 6])

    def test_listf_bind(self):
        xs = ListF.from_iter([1, 2, 3])
        result = xs.bind(lambda n: ListF.from_iter(range(n)))
        # 1 -> [0], 2 -> [0,1], 3 -> [0,1,2]
        assert result == ListF.from_iter([0, 0, 1, 0, 1, 2])

    def test_listf_filter(self):
        xs = ListF.from_iter([1, 2, 3, 4, 5])
        result = xs.filter(lambda n: n > 2)
        assert result == ListF.from_iter([3, 4, 5])

    def test_listf_head(self):
        assert ListF.from_iter([1, 2, 3]).head() == Just(1)
        assert ListF.empty().head() == Nothing()

    # -------------------------------------------------------------------------
    # Functor Laws
    # -------------------------------------------------------------------------

    def test_functor_identity_law(self):
        """fmap(id) == id"""
        # For Maybe
        just_val = Just(5)
        assert just_val.fmap(lambda x: x) == just_val

        # For ListF
        xs = ListF.from_iter([1, 2, 3])
        assert xs.fmap(lambda x: x) == xs

        # For Result
        ok_val = Ok(10)
        assert ok_val.fmap(lambda x: x) == ok_val

    def test_functor_composition_law(self):
        """fmap(f).fmap(g) == fmap(g . f)"""
        f = lambda x: x + 1
        g = lambda x: x * 2

        # For Maybe
        just_val = Just(5)
        assert just_val.fmap(f).fmap(g) == just_val.fmap(lambda x: g(f(x)))

        # For ListF
        xs = ListF.from_iter([1, 2, 3])
        assert xs.fmap(f).fmap(g) == xs.fmap(lambda x: g(f(x)))

    # -------------------------------------------------------------------------
    # Monad Laws
    # -------------------------------------------------------------------------

    def test_monad_left_identity(self):
        """pure(x).bind(f) == f(x)"""
        f = lambda x: Just(x * 2)
        x = 5
        assert Maybe.pure(x).bind(f) == f(x)

    def test_monad_right_identity(self):
        """m.bind(pure) == m"""
        m = Just(5)
        assert m.bind(Maybe.pure) == m

    def test_monad_associativity(self):
        """m.bind(f).bind(g) == m.bind(lambda x: f(x).bind(g))"""
        f = lambda x: Just(x + 1)
        g = lambda x: Just(x * 2)
        m = Just(5)

        left = m.bind(f).bind(g)
        right = m.bind(lambda x: f(x).bind(g))
        assert left == right


class TestResultTypes:
    """Tests for Ok/Err result types (from catpy)."""

    def test_ok_is_ok(self):
        result = Ok(42)
        assert result.is_ok()
        assert not result.is_err()

    def test_ok_unwrap(self):
        result = Ok("value")
        assert result.unwrap() == "value"

    def test_ok_fmap(self):
        result = Ok(5).fmap(lambda x: x * 2)
        assert result.is_ok()
        assert result.unwrap() == 10

    def test_ok_bind(self):
        result = Ok(5).bind(lambda x: Ok(x * 2))
        assert result.is_ok()
        assert result.unwrap() == 10

    def test_err_is_err(self):
        # catpy Err takes a single error value
        result = Err("error message")
        assert result.is_err()
        assert not result.is_ok()

    def test_err_unwrap_raises(self):
        result = Err("error")
        with pytest.raises(ValueError, match="Cannot unwrap Err"):
            result.unwrap()

    def test_err_fmap_passes_through(self):
        result = Err("error").fmap(lambda x: x * 2)
        assert result.is_err()

    def test_err_bind_passes_through(self):
        result = Err("error").bind(lambda x: Ok(x * 2))
        assert result.is_err()

    def test_pipeline_error_type(self):
        # Test PipelineError for detailed errors
        err = pipeline_err("filter", "Filter failed")
        assert err.is_err()
        assert err.error.step == "filter"
        assert "Filter failed" in err.error.message


# =============================================================================
# Step Tests
# =============================================================================

class TestSteps:
    """Tests for pipeline step classes."""

    def test_filter_step(self, sample_data):
        step = FilterStep(predicate=lambda r: r["count"] > 10)
        result = step.execute(sample_data)
        assert result.is_ok()
        data = result.unwrap()
        assert len(data) == 2
        assert all(r["count"] > 10 for r in data)

    def test_filter_step_with_condition_true(self, sample_data):
        step = FilterStep(
            predicate=lambda r: r["count"] > 10,
            condition=lambda: True
        )
        result = step.execute(sample_data)
        assert result.is_ok()
        assert len(result.unwrap()) == 2

    def test_filter_step_with_condition_false(self, sample_data):
        step = FilterStep(
            predicate=lambda r: r["count"] > 10,
            condition=lambda: False
        )
        result = step.execute(sample_data)
        assert result.is_ok()
        # Filter skipped, all data passes through
        assert len(result.unwrap()) == 4

    def test_select_step(self, sample_data):
        step = SelectStep(fields={"n": "name", "f": "file"})
        result = step.execute(sample_data)
        assert result.is_ok()
        data = result.unwrap()
        assert all("n" in r and "f" in r for r in data)
        assert all("count" not in r for r in data)

    def test_order_by_step_ascending(self, sample_data):
        step = OrderByStep(field_name="count", descending=False)
        result = step.execute(sample_data)
        assert result.is_ok()
        data = result.unwrap()
        assert data[0]["count"] == 5
        assert data[-1]["count"] == 25

    def test_order_by_step_descending(self, sample_data):
        step = OrderByStep(field_name="count", descending=True)
        result = step.execute(sample_data)
        assert result.is_ok()
        data = result.unwrap()
        assert data[0]["count"] == 25
        assert data[-1]["count"] == 5

    def test_limit_step(self, sample_data):
        step = LimitStep(count=2)
        result = step.execute(sample_data)
        assert result.is_ok()
        assert len(result.unwrap()) == 2

    def test_map_step(self, sample_data):
        step = MapStep(transform=lambda r: {"name": r["name"].upper()})
        result = step.execute(sample_data)
        assert result.is_ok()
        data = result.unwrap()
        assert data[0]["name"] == "FOO"


# =============================================================================
# Pipeline Tests
# =============================================================================

class TestPipeline:
    """Tests for Pipeline class."""

    def test_from_value(self):
        pipeline = Pipeline.from_value([1, 2, 3])
        ctx = Context(reter=None, params={})
        result = pipeline.run(ctx)
        assert result.is_ok()
        assert result.unwrap() == [1, 2, 3]

    def test_filter_method(self):
        pipeline = (
            Pipeline.from_value([1, 2, 3, 4, 5])
            .filter(lambda x: x > 2)
        )
        ctx = Context(reter=None, params={})
        result = pipeline.run(ctx)
        assert result.is_ok()
        assert result.unwrap() == [3, 4, 5]

    def test_select_method(self, sample_data):
        pipeline = (
            Pipeline.from_value(sample_data)
            .select("name", "file")
        )
        ctx = Context(reter=None, params={})
        result = pipeline.run(ctx)
        assert result.is_ok()
        data = result.unwrap()
        assert all("name" in r and "file" in r for r in data)

    def test_order_by_method(self, sample_data):
        pipeline = (
            Pipeline.from_value(sample_data)
            .order_by("-count")
        )
        ctx = Context(reter=None, params={})
        result = pipeline.run(ctx)
        assert result.is_ok()
        data = result.unwrap()
        assert data[0]["count"] == 25

    def test_limit_method(self, sample_data):
        pipeline = (
            Pipeline.from_value(sample_data)
            .limit(2)
        )
        ctx = Context(reter=None, params={})
        result = pipeline.run(ctx)
        assert result.is_ok()
        assert len(result.unwrap()) == 2

    def test_chained_operations(self, sample_data):
        pipeline = (
            Pipeline.from_value(sample_data)
            .filter(lambda r: r["count"] > 10)
            .select("name", "count")
            .order_by("-count")
            .limit(1)
        )
        ctx = Context(reter=None, params={})
        result = pipeline.run(ctx)
        assert result.is_ok()
        data = result.unwrap()
        assert len(data) == 1
        assert data[0]["name"] == "Bar"
        assert data[0]["count"] == 25

    def test_emit_method(self, sample_data):
        pipeline = (
            Pipeline.from_value(sample_data)
            .filter(lambda r: r["count"] > 10)
            .emit("results")
        )
        ctx = Context(reter=None, params={})
        output = pipeline.execute(ctx)
        assert output["success"]
        assert "results" in output
        assert output["count"] == 2

    def test_rshift_operator(self, sample_data):
        pipeline = (
            Pipeline.from_value(sample_data)
            >> FilterStep(lambda r: r["count"] > 10)
            >> LimitStep(1)
        )
        ctx = Context(reter=None, params={})
        result = pipeline.run(ctx)
        assert result.is_ok()
        assert len(result.unwrap()) == 1

    def test_or_operator(self, sample_data):
        pipeline = (
            Pipeline.from_value(sample_data)
            | FilterStep(lambda r: r["count"] > 10)
        )
        ctx = Context(reter=None, params={})
        result = pipeline.run(ctx)
        assert result.is_ok()
        assert len(result.unwrap()) == 2


# =============================================================================
# Decorator Tests
# =============================================================================

class TestDecorators:
    """Tests for tool decorators."""

    def test_query_decorator_registers_tool(self):
        @query("test_list_items")
        def test_list_items(p: Pipeline) -> Pipeline:
            """List all items."""
            return p

        assert Registry.get("test_list_items") is not None
        spec = Registry.get("test_list_items")
        assert spec.type == ToolType.QUERY

    def test_detector_decorator_registers_tool(self):
        @detector("test_find_issues")
        def test_find_issues(p: Pipeline) -> Pipeline:
            """Find issues."""
            return p

        assert Registry.get("test_find_issues") is not None
        spec = Registry.get("test_find_issues")
        assert spec.type == ToolType.DETECTOR

    def test_diagram_decorator_registers_tool(self):
        @diagram("test_class_diagram")
        def test_class_diagram(p: Pipeline) -> Pipeline:
            """Generate diagram."""
            return p

        assert Registry.get("test_class_diagram") is not None
        spec = Registry.get("test_class_diagram")
        assert spec.type == ToolType.DIAGRAM

    def test_param_decorator(self):
        @query("test_with_params")
        @param("limit", int, default=100)
        @param("module", str, required=False)
        def test_with_params(p: Pipeline) -> Pipeline:
            return p

        spec = Registry.get("test_with_params")
        assert "limit" in spec.params
        assert "module" in spec.params
        assert spec.params["limit"].default == 100
        assert spec.params["module"].required is False

    def test_meta_decorator(self):
        @detector("test_with_meta")
        @meta(category="code_smell", severity="high", tags=["design"])
        def test_with_meta(p: Pipeline) -> Pipeline:
            return p

        spec = Registry.get("test_with_meta")
        assert spec.meta["category"] == "code_smell"
        assert spec.meta["severity"] == "high"
        assert spec.meta["tags"] == ["design"]


# =============================================================================
# Operator Tests
# =============================================================================

class TestOperators:
    """Tests for pipeline operators."""

    def test_when_operator_executes(self, sample_data):
        condition_result = True
        step = FilterStep(lambda r: r["count"] > 10)
        conditional = when(lambda: condition_result)(step)

        result = conditional.execute(sample_data)
        assert result.is_ok()
        assert len(result.unwrap()) == 2

    def test_when_operator_skips(self, sample_data):
        condition_result = False
        step = FilterStep(lambda r: r["count"] > 10)
        conditional = when(lambda: condition_result)(step)

        result = conditional.execute(sample_data)
        assert result.is_ok()
        assert len(result.unwrap()) == 4  # All data passes through

    def test_unless_operator_executes(self, sample_data):
        condition_result = False  # unless False = execute
        step = FilterStep(lambda r: r["count"] > 10)
        conditional = unless(lambda: condition_result)(step)

        result = conditional.execute(sample_data)
        assert result.is_ok()
        assert len(result.unwrap()) == 2

    def test_unless_operator_skips(self, sample_data):
        condition_result = True  # unless True = skip
        step = FilterStep(lambda r: r["count"] > 10)
        conditional = unless(lambda: condition_result)(step)

        result = conditional.execute(sample_data)
        assert result.is_ok()
        assert len(result.unwrap()) == 4

    def test_identity_operator(self, sample_data):
        step = identity()
        result = step.execute(sample_data)
        assert result.is_ok()
        assert result.unwrap() == sample_data

    def test_tap_operator(self, sample_data):
        captured = []
        step = tap(lambda d: captured.append(len(d)))
        result = step.execute(sample_data)
        assert result.is_ok()
        assert result.unwrap() == sample_data
        assert captured == [4]

    def test_compose_operator(self, sample_data):
        composed = compose(
            FilterStep(lambda r: r["count"] > 5),
            LimitStep(2)
        )
        result = composed.execute(sample_data)
        assert result.is_ok()
        assert len(result.unwrap()) == 2


# =============================================================================
# Registry Tests
# =============================================================================

class TestRegistry:
    """Tests for tool registry."""

    def test_register_and_get(self):
        spec = ToolSpec(
            name="test_tool",
            type=ToolType.QUERY,
            description="A test tool",
            pipeline_factory=lambda ctx: Pipeline.from_value([]),
            params={},
            meta={}
        )
        Registry.register(spec)

        retrieved = Registry.get("test_tool")
        assert retrieved is not None
        assert retrieved.name == "test_tool"

    def test_get_nonexistent(self):
        assert Registry.get("nonexistent") is None

    def test_list_all(self):
        # Register some tools
        for name in ["tool_a", "tool_b", "tool_c"]:
            spec = ToolSpec(
                name=name,
                type=ToolType.QUERY,
                description="",
                pipeline_factory=lambda ctx: Pipeline.from_value([]),
                params={},
                meta={}
            )
            Registry.register(spec)

        names = Registry.list_all()
        assert "tool_a" in names
        assert "tool_b" in names
        assert "tool_c" in names

    def test_get_by_type(self):
        # Register query
        Registry.register(ToolSpec(
            name="query_tool",
            type=ToolType.QUERY,
            description="",
            pipeline_factory=lambda ctx: Pipeline.from_value([]),
            params={},
            meta={}
        ))
        # Register detector
        Registry.register(ToolSpec(
            name="detector_tool",
            type=ToolType.DETECTOR,
            description="",
            pipeline_factory=lambda ctx: Pipeline.from_value([]),
            params={},
            meta={}
        ))

        queries = Registry.get_by_type(ToolType.QUERY)
        assert len(queries) == 1
        assert queries[0].name == "query_tool"

    def test_get_by_category(self):
        Registry.register(ToolSpec(
            name="smell_detector",
            type=ToolType.DETECTOR,
            description="",
            pipeline_factory=lambda ctx: Pipeline.from_value([]),
            params={},
            meta={"category": "code_smell"}
        ))
        Registry.register(ToolSpec(
            name="arch_detector",
            type=ToolType.DETECTOR,
            description="",
            pipeline_factory=lambda ctx: Pipeline.from_value([]),
            params={},
            meta={"category": "architecture"}
        ))

        smells = Registry.get_by_category("code_smell")
        assert len(smells) == 1
        assert smells[0].name == "smell_detector"

    def test_stats(self):
        Registry.register(ToolSpec(
            name="q1", type=ToolType.QUERY, description="",
            pipeline_factory=lambda ctx: Pipeline.from_value([]),
            params={}, meta={"category": "analysis"}
        ))
        Registry.register(ToolSpec(
            name="d1", type=ToolType.DETECTOR, description="",
            pipeline_factory=lambda ctx: Pipeline.from_value([]),
            params={}, meta={"category": "code_smell"}
        ))

        stats = Registry.stats()
        assert stats["total"] == 2
        assert stats["by_type"]["query"] == 1
        assert stats["by_type"]["detector"] == 1


class TestNamespace:
    """Tests for tool namespaces."""

    def test_namespace_creates_prefixed_tools(self):
        ns = namespace("refactoring")

        @ns.query("list_smells")
        def list_smells(p: Pipeline) -> Pipeline:
            return p

        # Tool should be registered with prefixed name
        assert Registry.get("refactoring:list_smells") is not None

    def test_namespace_get_tool(self):
        ns = namespace("test_ns")

        @ns.detector("find_issues")
        def find_issues(p: Pipeline) -> Pipeline:
            return p

        tool = ns.get("find_issues")
        assert tool is not None

    def test_namespace_list_tools(self):
        ns = namespace("my_tools")

        @ns.query("tool_a")
        def tool_a(p): return p

        @ns.query("tool_b")
        def tool_b(p): return p

        tools = ns.list_tools()
        assert "tool_a" in tools
        assert "tool_b" in tools


# =============================================================================
# ParamSpec Validation Tests
# =============================================================================

class TestParamSpec:
    """Tests for parameter validation."""

    def test_validate_required_missing(self):
        spec = ParamSpec(name="limit", type=int, required=True)
        result = spec.validate(None)
        assert result.is_err()

    def test_validate_required_present(self):
        spec = ParamSpec(name="limit", type=int, required=True)
        result = spec.validate(10)
        assert result.is_ok()
        assert result.unwrap() == 10

    def test_validate_optional_missing(self):
        spec = ParamSpec(name="limit", type=int, required=False, default=100)
        result = spec.validate(None)
        assert result.is_ok()
        assert result.unwrap() == 100

    def test_validate_type_coercion_int(self):
        spec = ParamSpec(name="limit", type=int, required=True)
        result = spec.validate("42")
        assert result.is_ok()
        assert result.unwrap() == 42

    def test_validate_type_coercion_bool(self):
        spec = ParamSpec(name="flag", type=bool, required=True)
        result = spec.validate("true")
        assert result.is_ok()
        assert result.unwrap() is True

    def test_validate_choices_valid(self):
        spec = ParamSpec(
            name="format", type=str, required=True,
            choices=["json", "mermaid", "markdown"]
        )
        result = spec.validate("mermaid")
        assert result.is_ok()

    def test_validate_choices_invalid(self):
        spec = ParamSpec(
            name="format", type=str, required=True,
            choices=["json", "mermaid", "markdown"]
        )
        result = spec.validate("xml")
        assert result.is_err()


# =============================================================================
# ToolBuilder Tests
# =============================================================================

class TestToolBuilder:
    """Tests for ToolBuilder fluent API."""

    def test_builder_creates_tool(self):
        tool = (
            query_builder("built_tool")
            .description("A tool built with builder")
            .param("limit", int, default=50)
            .pipeline(lambda p: p.limit(50))
            .build()
        )

        spec = Registry.get("built_tool")
        assert spec is not None
        assert spec.description == "A tool built with builder"
        assert "limit" in spec.params

    def test_builder_requires_pipeline(self):
        builder = query_builder("incomplete_tool")
        with pytest.raises(ValueError, match="Pipeline function not set"):
            builder.build()


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests combining multiple components."""

    def test_complete_query_tool(self, sample_data):
        """Test a complete query tool with all components."""
        # Data must be defined outside the decorated function to be accessible
        test_data = sample_data.copy()

        @query("list_large_classes")
        @param("threshold", int, default=10)
        @param("exclude_tests", bool, default=True)
        def list_large_classes(p: Pipeline) -> Pipeline:
            return (
                Pipeline.from_value(test_data)
                .filter(lambda r: r["count"] > 10)
                .select("name", "file", "count")
                .order_by("-count")
                .emit("classes")
            )

        # Execute the tool
        result = list_large_classes(reter=None, threshold=10)

        assert result["success"]
        assert "classes" in result
        assert len(result["classes"]) == 2

    def test_complete_detector_tool(self, sample_data):
        """Test a complete detector tool."""
        # Data must be defined outside the decorated function to be accessible
        test_data = sample_data.copy()

        @detector("find_large_entities")
        @meta(category="code_smell", severity="medium")
        @param("threshold", int, default=15)
        def find_large_entities(p: Pipeline) -> Pipeline:
            return (
                Pipeline.from_value(test_data)
                .filter(lambda r: r["count"] > 15)
                .emit("findings")
            )

        result = find_large_entities(reter=None)

        assert result["success"]
        assert result["detector"] == "find_large_entities"
        assert result["category"] == "code_smell"
        assert result["severity"] == "medium"

    def test_pipeline_error_handling(self):
        """Test that pipeline errors are properly captured."""

        def bad_filter(r):
            raise ValueError("Intentional error")

        pipeline = (
            Pipeline.from_value([{"x": 1}])
            .filter(bad_filter)
        )

        ctx = Context(reter=None, params={})
        result = pipeline.run(ctx)

        assert result.is_err()
        assert "Intentional error" in str(result)
