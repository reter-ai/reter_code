"""
CADSL Core - Monadic Pipeline and Tool Type Classes

This module provides the core abstractions for the Code Analysis DSL,
built on category theory foundations from catpy:

- Pipeline: A Monad for composable data transformation chains
- Step: A function wrapper that transforms data within the pipeline
- Query/Detector/Diagram: Tool type wrappers

The Pipeline follows monad laws:
  1) Left identity:  Pipeline.pure(x).bind(f) == f(x)
  2) Right identity: m.bind(Pipeline.pure) == m
  3) Associativity:  m.bind(f).bind(g) == m.bind(lambda x: f(x).bind(g))
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    TypeVar, Generic, Callable, List, Dict, Any, Optional,
    Union, Tuple, Iterator, Sequence
)
from enum import Enum
from abc import ABC, abstractmethod

# Import category theory foundations
from .catpy import (
    Functor, Applicative, Monad,
    Result, Ok, Err,
    Maybe, Just, Nothing,
    ListF,
    PipelineError, PipelineResult,
    pipeline_ok, pipeline_err,
    compose, identity
)

# PyArrow for vectorized operations
import pyarrow as pa
import pyarrow.compute as pc

from ..services.utils import matches_pattern


# =============================================================================
# Arrow Utilities
# =============================================================================

def to_arrow(data: Union[pa.Table, List[Dict]]) -> pa.Table:
    """Convert data to PyArrow table."""
    if isinstance(data, pa.Table):
        return data
    if not data:
        return pa.table({})
    return pa.Table.from_pylist(data)


def to_list(data: Union[pa.Table, List[Dict]]) -> List[Dict]:
    """Convert data to list of dicts."""
    if isinstance(data, pa.Table):
        return data.to_pylist()
    return data


def is_arrow(data) -> bool:
    """Check if data is a PyArrow table."""
    return isinstance(data, pa.Table)


def resolve_column(table: pa.Table, name: str) -> Optional[str]:
    """Find column name, handling ? prefix from REQL."""
    if name in table.column_names:
        return name
    if f"?{name}" in table.column_names:
        return f"?{name}"
    return None

T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")


# =============================================================================
# Re-export catpy types for convenience
# =============================================================================

__all__ = [
    # From catpy
    "Functor", "Applicative", "Monad",
    "Result", "Ok", "Err",
    "Maybe", "Just", "Nothing",
    "ListF",
    "PipelineError", "PipelineResult",
    "pipeline_ok", "pipeline_err",
    "compose", "identity",
    # Core types
    "Context", "Source", "Step", "Pipeline",
    "Query", "Detector", "Diagram",
    "ToolSpec", "ParamSpec", "ToolType",
    # Step types
    "TapStep",
    # Source builders
    "reql", "rag", "value",
]


# =============================================================================
# Pipeline Context
# =============================================================================

@dataclass
class Context:
    """Execution context for pipelines.

    ::: This is-in-layer Domain-Specific-Language-Layer.
    ::: This is a context.
    ::: This is-in-process Main-Process.
    ::: This is stateless.
    """
    reter: Any  # RETER instance
    params: Dict[str, Any] = field(default_factory=dict)
    instance_name: str = "default"

    def get_param(self, name: str, default: Any = None) -> Any:
        """Get a parameter value."""
        return self.params.get(name, default)

    def get(self, name: str, default: Any = None) -> Any:
        """Get a value from params (alias for get_param)."""
        return self.params.get(name, default)

    def with_params(self, **kwargs) -> "Context":
        """Create a new context with additional params."""
        new_params = {**self.params, **kwargs}
        return Context(
            reter=self.reter,
            params=new_params,
            instance_name=self.instance_name
        )


# =============================================================================
# Sources (Query Origins) - Functors that produce initial data
# =============================================================================

class Source(ABC, Generic[T]):
    """
    Abstract base for pipeline data sources.

    A Source is a functor that produces data when executed in a context.

    ::: This is-in-layer Domain-Specific-Language-Layer.
    ::: This is a source.
    ::: This is-in-process Main-Process.
    ::: This is stateless.
    """

    @abstractmethod
    def execute(self, ctx: Context) -> PipelineResult[T]:
        """Execute the source and return data wrapped in Result."""
        pass

    def fmap(self, f: Callable[[T], U]) -> "MappedSource[T, U]":
        """Map a function over the source output."""
        return MappedSource(self, f)


@dataclass
class MappedSource(Source[U], Generic[T, U]):
    """A source with a mapped transformation.

    ::: This is-in-layer Domain-Specific-Language-Layer.
    ::: This is a source.
    ::: This is-in-process Main-Process.
    ::: This is stateless.
    """
    inner: Source[T]
    transform: Callable[[T], U]

    def execute(self, ctx: Context) -> PipelineResult[U]:
        result = self.inner.execute(ctx)
        return result.fmap(self.transform)


@dataclass
class REQLSource(Source[pa.Table]):
    """REQL query source - returns PyArrow table for vectorized operations.

    REQL queries use plain type names (CNL naming convention):
    - class, method, function (language-independent)
    - Predicates use hyphenated format: is-in-file, has-name, is-defined-in

    Parameter placeholders like {limit}, {target} are still supported
    and resolved from ctx.params at runtime.

    Timeout can be specified via ctx.params['timeout_ms'] (default: 300000ms = 5 minutes).

    ::: This is-in-layer Domain-Specific-Language-Layer.
    ::: This is a source.
    ::: This is-in-process Main-Process.
    ::: This is stateless.
    """
    query: str

    def execute(self, ctx: Context) -> PipelineResult[pa.Table]:
        """Execute REQL query against RETER - returns PyArrow table."""
        try:
            query = self.query

            # Substitute parameters in query: {target} -> actual value
            for key, value in ctx.params.items():
                placeholder = "{" + key + "}"
                if placeholder in query:
                    query = query.replace(placeholder, str(value))

            # Get timeout from params (default 5 minutes = 300000ms)
            timeout_ms = ctx.params.get('timeout_ms', 300000)

            # Execute query - returns PyArrow table directly
            table = ctx.reter.reql(query, timeout_ms=timeout_ms)

            if table is None or table.num_rows == 0:
                return pipeline_ok(pa.table({}))

            return pipeline_ok(table)
        except Exception as e:
            return pipeline_err("reql", f"Query failed: {e}: {query}", e)


@dataclass
class ValueSource(Source[T], Generic[T]):
    """Literal value source.

    ::: This is-in-layer Domain-Specific-Language-Layer.
    ::: This is a source.
    ::: This is-in-process Main-Process.
    ::: This is stateless.
    """
    value: T

    def execute(self, ctx: Context) -> PipelineResult[T]:
        return pipeline_ok(self.value)


def _get_rag_manager(ctx: Context):
    """Get RAG manager from context or default instance."""
    rag_manager = ctx.get("rag_manager")
    if rag_manager is None:
        try:
            from reter_code.services.default_instance_manager import DefaultInstanceManager
            default_mgr = DefaultInstanceManager.get_instance()
            if default_mgr:
                rag_manager = default_mgr.get_rag_manager()
        except Exception:
            pass
    return rag_manager


@dataclass
class RAGSearchSource(Source[List[Dict[str, Any]]]):
    """RAG semantic search source.

    ::: This is-in-layer Domain-Specific-Language-Layer.
    ::: This is a source.
    ::: This is-in-process Main-Process.
    ::: This is stateless.
    """
    query: str
    top_k: int = 10
    entity_types: Optional[List[str]] = None

    def execute(self, ctx: Context) -> PipelineResult[List[Dict[str, Any]]]:
        """Execute semantic search."""
        try:
            rag_manager = _get_rag_manager(ctx)
            if rag_manager is None:
                return pipeline_err("rag", "RAG manager not available")

            # Call search() method which returns (results, stats)
            results, stats = rag_manager.search(
                query=self.query,
                top_k=self.top_k,
                entity_types=self.entity_types
            )

            # Check for errors in stats
            if stats.get("error"):
                return pipeline_err("rag", stats.get("error", "Search failed"))

            # Convert RAGSearchResult objects to dicts
            result_dicts = [r.to_dict() for r in results]

            return pipeline_ok(result_dicts)
        except Exception as e:
            return pipeline_err("rag", f"Semantic search failed: {e}", e)


@dataclass
class RAGDuplicatesSource(Source[List[Dict[str, Any]]]):
    """RAG duplicate code detection source.

    ::: This is-in-layer Domain-Specific-Language-Layer.
    ::: This is a source.
    ::: This is-in-process Main-Process.
    ::: This is stateless.
    """
    similarity: float = 0.85
    limit: int = 50
    exclude_same_file: bool = True
    exclude_same_class: bool = True
    entity_types: Optional[List[str]] = None

    def execute(self, ctx: Context) -> PipelineResult[List[Dict[str, Any]]]:
        """Find duplicate code using RAG embeddings."""
        try:
            rag_manager = _get_rag_manager(ctx)
            if rag_manager is None:
                return pipeline_err("rag", "RAG manager not available")

            result = rag_manager.find_duplicate_candidates(
                similarity_threshold=self.similarity,
                max_results=self.limit,
                exclude_same_file=self.exclude_same_file,
                exclude_same_class=self.exclude_same_class,
                entity_types=self.entity_types or ["method", "function"]
            )

            if not result.get("success"):
                return pipeline_err("rag", result.get("error", "Duplicate detection failed"))

            # Transform pairs to flat findings
            findings = []
            for pair in result.get("pairs", []):
                e1, e2 = pair["entity1"], pair["entity2"]
                findings.append({
                    "similarity": pair["similarity"],
                    "entity1_name": e1["name"],
                    "entity1_file": e1["file"],
                    "entity1_line": e1["line"],
                    "entity1_class": e1.get("class_name", ""),
                    "entity2_name": e2["name"],
                    "entity2_file": e2["file"],
                    "entity2_line": e2["line"],
                    "entity2_class": e2.get("class_name", ""),
                })

            return pipeline_ok(findings)
        except Exception as e:
            return pipeline_err("rag", f"Duplicate detection failed: {e}", e)


@dataclass
class RAGClustersSource(Source[List[Dict[str, Any]]]):
    """RAG code clustering source using K-means.

    ::: This is-in-layer Domain-Specific-Language-Layer.
    ::: This is a source.
    ::: This is-in-process Main-Process.
    ::: This is stateless.
    """
    n_clusters: int = 50
    min_size: int = 2
    exclude_same_file: bool = True
    exclude_same_class: bool = True
    entity_types: Optional[List[str]] = None

    def execute(self, ctx: Context) -> PipelineResult[List[Dict[str, Any]]]:
        """Find code clusters using K-means on RAG embeddings."""
        try:
            rag_manager = _get_rag_manager(ctx)
            if rag_manager is None:
                return pipeline_err("rag", "RAG manager not available")

            result = rag_manager.find_similar_clusters(
                n_clusters=self.n_clusters,
                min_cluster_size=self.min_size,
                exclude_same_file=self.exclude_same_file,
                exclude_same_class=self.exclude_same_class,
                entity_types=self.entity_types or ["method", "function"]
            )

            if not result.get("success"):
                return pipeline_err("rag", result.get("error", "Clustering failed"))

            # Transform clusters to findings
            findings = []
            for cluster in result.get("clusters", []):
                member_names = [m["name"] for m in cluster["members"]]
                member_files = list(set(m["file"] for m in cluster["members"]))
                findings.append({
                    "cluster_id": cluster["cluster_id"],
                    "member_count": cluster["member_count"],
                    "unique_files": cluster["unique_files"],
                    "members": member_names,
                    "files": member_files,
                    "avg_distance": cluster.get("avg_distance", 0),
                    "details": cluster["members"]
                })

            return pipeline_ok(findings)
        except Exception as e:
            return pipeline_err("rag", f"Clustering failed: {e}", e)


@dataclass
class RAGDBScanSource(Source[List[Dict[str, Any]]]):
    """RAG code clustering source using DBSCAN (Density-Based Spatial Clustering).

    DBSCAN advantages over K-means:
    - No need to specify number of clusters upfront
    - Automatically discovers natural groupings in the data
    - Identifies outliers/noise points (not forced into clusters)
    - Better for finding tight clusters of truly similar code

    ::: This is-in-layer Domain-Specific-Language-Layer.
    ::: This is a source.
    ::: This is-in-process Main-Process.
    ::: This is stateless.
    """
    eps: float = 0.5  # Max distance between neighbors (0.3-0.8 typical)
    min_samples: int = 3  # Min points to form dense region
    min_size: int = 2  # Min cluster size to return
    exclude_same_file: bool = True
    exclude_same_class: bool = True
    entity_types: Optional[List[str]] = None

    def execute(self, ctx: Context) -> PipelineResult[List[Dict[str, Any]]]:
        """Find code clusters using DBSCAN on RAG embeddings."""
        try:
            rag_manager = _get_rag_manager(ctx)
            if rag_manager is None:
                return pipeline_err("rag", "RAG manager not available")

            result = rag_manager.find_similar_clusters_dbscan(
                eps=self.eps,
                min_samples=self.min_samples,
                min_cluster_size=self.min_size,
                exclude_same_file=self.exclude_same_file,
                exclude_same_class=self.exclude_same_class,
                entity_types=self.entity_types or ["method", "function"]
            )

            if not result.get("success"):
                return pipeline_err("rag", result.get("error", "DBSCAN clustering failed"))

            # Transform clusters to findings
            findings = []
            for cluster in result.get("clusters", []):
                member_names = [m["name"] for m in cluster["members"]]
                member_files = list(set(m["file"] for m in cluster["members"]))
                findings.append({
                    "cluster_id": cluster["cluster_id"],
                    "member_count": cluster["member_count"],
                    "unique_files": cluster["unique_files"],
                    "members": member_names,
                    "files": member_files,
                    "avg_distance": cluster.get("avg_distance", 0),
                    "avg_similarity": cluster.get("avg_similarity", 0),
                    "details": cluster["members"]
                })

            return pipeline_ok(findings)
        except Exception as e:
            return pipeline_err("rag", f"DBSCAN clustering failed: {e}", e)


@dataclass
class FileScanSource(Source[List[Dict[str, Any]]]):
    """
    Scan RETER sources for files matching patterns, with optional content search.

    Iterates over RETER's tracked sources (already loaded files) rather than
    scanning the filesystem directly. This ensures consistency with the knowledge
    graph and enables seamless JOINs via the 'file' field.

    Source format: "hash|relative_path" (e.g., "abc123|src/module.py")
    Output 'file' field: normalized path with forward slashes (matches REQL is-in-file)

    ::: This is-in-layer Domain-Specific-Language-Layer.
    ::: This is a source.
    ::: This is-in-process Main-Process.
    ::: This is stateless.
    """
    glob: str = "*"
    exclude: Optional[List[str]] = None
    contains: Optional[str] = None
    not_contains: Optional[str] = None
    case_sensitive: bool = True
    include_matches: bool = False
    context_lines: int = 0
    max_matches_per_file: Optional[int] = None
    include_stats: bool = True

    def execute(self, ctx: Context) -> PipelineResult[List[Dict[str, Any]]]:
        """Scan RETER sources for matching files."""
        import re
        from pathlib import Path
        from datetime import datetime

        # Use shared utility from services.utils
        matches_glob = matches_pattern

        try:
            # Get RETER instance
            reter = ctx.reter
            if reter is None:
                return pipeline_err("file_scan", "RETER instance not available in context")

            # Get all sources from RETER
            sources_result = reter.get_all_sources()
            if isinstance(sources_result, tuple) and len(sources_result) == 2:
                sources = sources_result[0]
            else:
                sources = sources_result if isinstance(sources_result, list) else []

            if not isinstance(sources, list):
                return pipeline_err("file_scan", f"Unexpected sources format: {type(sources)}")

            # Compile regex patterns
            flags = 0 if self.case_sensitive else re.IGNORECASE
            contains_re = re.compile(self.contains, flags) if self.contains else None
            not_contains_re = re.compile(self.not_contains, flags) if self.not_contains else None

            # Get base directory - same pattern as logging_config.py
            # Priority: 1. RETER_PROJECT_ROOT env var, 2. reter.base_directory, 3. Path.cwd()
            import os
            base_dir = os.environ.get('RETER_PROJECT_ROOT')
            if base_dir is None:
                base_dir = getattr(reter, 'base_directory', None)
            if base_dir is None:
                base_dir = getattr(reter, 'working_directory', None)
            if base_dir is None:
                base_dir = Path.cwd()
            else:
                base_dir = Path(base_dir)

            results = []

            # Debug: track why files are filtered
            debug_stats = {
                "total_sources": len(sources),
                "glob_filtered": 0,
                "excluded": 0,
                "not_found": 0,
                "content_filtered": 0,
                "matched": 0,
                "base_dir": str(base_dir),
                "glob_pattern": self.glob,
                "sample_sources": sources[:3] if sources else []
            }

            for source_id in sources:
                # Parse source ID: "hash|relative_path"
                if '|' in source_id:
                    rel_path = source_id.split('|', 1)[1]
                else:
                    rel_path = source_id

                # Normalize path to forward slashes (matches REQL is-in-file format)
                normalized_path = rel_path.replace('\\', '/')

                # Apply glob filter using helper that supports **
                if not matches_glob(normalized_path, self.glob):
                    debug_stats["glob_filtered"] += 1
                    continue

                # Apply exclusion filters
                if self.exclude:
                    excluded = False
                    for pattern in self.exclude:
                        if matches_glob(normalized_path, pattern):
                            excluded = True
                            break
                    if excluded:
                        debug_stats["excluded"] += 1
                        continue

                # Try to read the file for content filtering and stats
                abs_path = base_dir / rel_path.replace('/', '\\')
                if not abs_path.exists():
                    # Try with original path
                    abs_path = base_dir / rel_path
                    if not abs_path.exists():
                        debug_stats["not_found"] += 1
                        continue

                try:
                    content = abs_path.read_text(encoding='utf-8', errors='ignore')
                    lines = content.splitlines()

                    # Apply not_contains filter
                    if not_contains_re and not_contains_re.search(content):
                        debug_stats["content_filtered"] += 1
                        continue

                    # Apply contains filter
                    if contains_re and not contains_re.search(content):
                        debug_stats["content_filtered"] += 1
                        continue

                    debug_stats["matched"] += 1

                    # Build result
                    result = {
                        "file": normalized_path,  # Matches REQL is-in-file format
                    }

                    # Add stats if requested
                    if self.include_stats:
                        stat = abs_path.stat()
                        result["line_count"] = len(lines)
                        result["file_size"] = stat.st_size
                        result["last_modified"] = datetime.fromtimestamp(stat.st_mtime).isoformat()

                    # Find matches if requested
                    if self.include_matches and contains_re:
                        matches = []
                        for i, line in enumerate(lines):
                            if contains_re.search(line):
                                match = {
                                    "line_number": i + 1,
                                    "content": line,
                                }
                                if self.context_lines > 0:
                                    match["context_before"] = lines[max(0, i - self.context_lines):i]
                                    match["context_after"] = lines[i + 1:i + 1 + self.context_lines]
                                matches.append(match)
                                if self.max_matches_per_file and len(matches) >= self.max_matches_per_file:
                                    break
                        result["match_count"] = len(matches)
                        result["matches"] = matches
                    elif contains_re:
                        # Count matches without storing them
                        result["match_count"] = len(contains_re.findall(content))

                    results.append(result)

                except (IOError, UnicodeDecodeError, PermissionError):
                    # Skip files that can't be read
                    debug_stats["not_found"] += 1
                    continue

            # Return results with debug stats if no matches found
            if not results:
                return pipeline_ok({
                    "files": [],
                    "count": 0,
                    "debug": debug_stats
                })

            return pipeline_ok(results)

        except Exception as e:
            return pipeline_err("file_scan", f"File scan failed: {e}", e)


@dataclass
class ParseFileSource(Source[List[Dict[str, Any]]]):
    """
    Parse an external data file (CSV, JSON, Parquet) as table rows.

    ::: This is-in-layer Domain-Specific-Language-Layer.
    ::: This is a source.
    ::: This is-in-process Main-Process.
    ::: This is stateless.
    """
    path: str = ""
    format: str = "csv"
    encoding: str = "utf-8"
    separator: str = ","
    sheet: Optional[str] = None
    columns: Optional[List[str]] = None
    limit: Optional[int] = None

    def execute(self, ctx: Context) -> PipelineResult[List[Dict[str, Any]]]:
        import pandas as pd
        from pathlib import Path

        project_root = _get_project_root(ctx)
        file_path = Path(project_root) / self.path
        if not file_path.exists():
            return pipeline_err("parse_file", f"File not found: {file_path}")

        try:
            if self.format == "csv":
                df = pd.read_csv(file_path, encoding=self.encoding, sep=self.separator,
                                 usecols=self.columns, nrows=self.limit)
            elif self.format == "json":
                df = pd.read_json(file_path, encoding=self.encoding)
                if self.columns:
                    df = df[self.columns]
                if self.limit:
                    df = df.head(self.limit)
            elif self.format == "parquet":
                df = pd.read_parquet(file_path, columns=self.columns)
                if self.limit:
                    df = df.head(self.limit)
            else:
                return pipeline_err("parse_file", f"Unsupported format: {self.format}")

            df = df.where(df.notna(), None)
            results = df.to_dict(orient="records")
            return pipeline_ok(results)
        except Exception as e:
            return pipeline_err("parse_file", f"Failed to parse {self.path}: {e}", e)


def _get_project_root(ctx):
    """Get project root from context or environment."""
    if ctx and hasattr(ctx, 'reter') and ctx.reter:
        root = getattr(ctx.reter, 'project_root', None)
        if root:
            return root
        root = getattr(ctx.reter, 'base_directory', None)
        if root:
            return root
    import os
    return os.environ.get("RETER_PROJECT_ROOT", os.getcwd())


# Backward compatibility alias
RAGSource = RAGSearchSource


# =============================================================================
# Step - Kleisli Arrow for Pipeline
# =============================================================================

class Step(ABC, Generic[T, U]):
    """
    A step in a pipeline - a Kleisli arrow: T -> Result[U, PipelineError]

    Steps are the building blocks of pipelines. Each step transforms
    input data and returns a Result, allowing for error propagation.

    ::: This is-in-layer Domain-Specific-Language-Layer.
    ::: This is a step.
    ::: This is-in-process Main-Process.
    ::: This is stateless.
    """

    @abstractmethod
    def execute(self, data: T, ctx: Optional["Context"] = None) -> PipelineResult[U]:
        """Transform input data and return result."""
        pass

    def _resolve_param(self, value: Any, ctx: Optional["Context"]) -> Any:
        """Resolve {param} placeholders from context."""
        if ctx is None or not isinstance(value, str):
            return value
        if value.startswith("{") and value.endswith("}"):
            param_name = value[1:-1]
            return ctx.params.get(param_name, value)
        return value

    def __rshift__(self, other: "Step[U, V]") -> "ComposedStep[T, U, V]":
        """Compose steps: step1 >> step2"""
        return ComposedStep(self, other)

    def and_then(self, other: "Step[U, V]") -> "ComposedStep[T, U, V]":
        """Compose steps: step1.and_then(step2)"""
        return self >> other


@dataclass
class ComposedStep(Step[T, V], Generic[T, U, V]):
    """Composition of two steps (Kleisli composition).

    ::: This is-in-layer Domain-Specific-Language-Layer.
    ::: This is a step.
    ::: This is-in-process Main-Process.
    ::: This is stateless.
    """
    first: Step[T, U]
    second: Step[U, V]

    def execute(self, data: T, ctx: Optional["Context"] = None) -> PipelineResult[V]:
        result = self.first.execute(data, ctx)
        if result.is_err():
            return result  # type: ignore
        return self.second.execute(result.unwrap(), ctx)


@dataclass
class FilterStep(Step[Union[pa.Table, List[T]], Union[pa.Table, List[T]]], Generic[T]):
    """Filter items based on predicate - Arrow-optimized.

    ::: This is-in-layer Domain-Specific-Language-Layer.
    ::: This is a step.
    ::: This is-in-process Main-Process.
    ::: This is stateless.
    """
    predicate: Callable[[T], bool]
    condition: Optional[Callable[[], bool]] = None  # when/unless condition

    def execute(self, data: Union[pa.Table, List[T]], ctx: Optional["Context"] = None) -> PipelineResult[Union[pa.Table, List[T]]]:
        if self.condition is not None and not self.condition():
            return pipeline_ok(data)  # Skip filter if condition not met
        try:
            # Use Arrow vectorized filter if data is Arrow table
            if is_arrow(data):
                return self._arrow_filter(data, ctx)

            # Fall back to row-by-row for list
            import inspect
            try:
                sig = inspect.signature(self.predicate)
                if len(sig.parameters) >= 2:
                    return pipeline_ok([item for item in data if self.predicate(item, ctx)])
                else:
                    return pipeline_ok([item for item in data if self.predicate(item)])
            except (ValueError, TypeError):
                return pipeline_ok([item for item in data if self.predicate(item)])
        except Exception as e:
            return pipeline_err("filter", f"Filter failed: {e}", e)

    def _arrow_filter(self, table: pa.Table, ctx: Optional["Context"]) -> PipelineResult[pa.Table]:
        """Apply filter using Arrow compute - vectorized."""
        try:
            # Convert to list and filter row-by-row (predicate is Python function)
            # For true vectorization, use ArrowFilterStep with expression parsing
            rows = table.to_pylist()
            import inspect
            try:
                sig = inspect.signature(self.predicate)
                if len(sig.parameters) >= 2:
                    filtered = [item for item in rows if self.predicate(item, ctx)]
                else:
                    filtered = [item for item in rows if self.predicate(item)]
            except (ValueError, TypeError):
                filtered = [item for item in rows if self.predicate(item)]

            if not filtered:
                return pipeline_ok(pa.table({}))
            return pipeline_ok(pa.Table.from_pylist(filtered))
        except Exception as e:
            return pipeline_err("filter", f"Arrow filter failed: {e}", e)


@dataclass
class SelectStep(Step[Union[pa.Table, List[Dict]], Union[pa.Table, List[Dict]]]):
    """Select/rename fields from items - Arrow-optimized.

    ::: This is-in-layer Domain-Specific-Language-Layer.
    ::: This is a step.
    ::: This is-in-process Main-Process.
    ::: This is stateless.
    """
    fields: Dict[str, str]  # output_name -> source_name

    def execute(self, data: Union[pa.Table, List[Dict]], ctx: Optional["Context"] = None) -> PipelineResult[Union[pa.Table, List[Dict]]]:
        try:
            if is_arrow(data):
                return self._arrow_select(data)

            # List of dicts fallback
            result = []
            for item in data:
                new_item = {}
                for out_name, src_name in self.fields.items():
                    clean_out = out_name.lstrip("?")
                    if src_name in item:
                        new_item[clean_out] = item[src_name]
                    elif not src_name.startswith("?") and f"?{src_name}" in item:
                        new_item[clean_out] = item[f"?{src_name}"]
                    elif out_name in item:
                        new_item[clean_out] = item[out_name]
                    elif not out_name.startswith("?") and f"?{out_name}" in item:
                        new_item[clean_out] = item[f"?{out_name}"]
                result.append(new_item)
            return pipeline_ok(result)
        except Exception as e:
            return pipeline_err("select", f"Select failed: {e}", e)

    def _arrow_select(self, table: pa.Table) -> PipelineResult[pa.Table]:
        """Select columns using Arrow - vectorized O(1) per column."""
        try:
            if table.num_rows == 0:
                return pipeline_ok(pa.table({}))

            columns = {}
            for out_name, src_name in self.fields.items():
                clean_out = out_name.lstrip("?")
                col_name = resolve_column(table, src_name)
                if col_name is None:
                    col_name = resolve_column(table, out_name)
                if col_name:
                    columns[clean_out] = table.column(col_name)

            return pipeline_ok(pa.table(columns))
        except Exception as e:
            return pipeline_err("select", f"Arrow select failed: {e}", e)


@dataclass
class OrderByStep(Step[Union[pa.Table, List[Dict]], Union[pa.Table, List[Dict]]]):
    """Sort items by field - Arrow-optimized.

    ::: This is-in-layer Domain-Specific-Language-Layer.
    ::: This is a step.
    ::: This is-in-process Main-Process.
    ::: This is stateless.
    """
    field_name: str
    descending: bool = False

    def execute(self, data: Union[pa.Table, List[Dict]], ctx: Optional["Context"] = None) -> PipelineResult[Union[pa.Table, List[Dict]]]:
        try:
            if is_arrow(data):
                return self._arrow_sort(data)

            # List fallback
            def get_value(x):
                if self.field_name in x:
                    return x.get(self.field_name, "")
                elif not self.field_name.startswith("?") and f"?{self.field_name}" in x:
                    return x.get(f"?{self.field_name}", "")
                return ""
            return pipeline_ok(sorted(data, key=get_value, reverse=self.descending))
        except Exception as e:
            return pipeline_err("order_by", f"Sort failed: {e}", e)

    def _arrow_sort(self, table: pa.Table) -> PipelineResult[pa.Table]:
        """Sort using Arrow compute - vectorized."""
        try:
            if table.num_rows == 0:
                return pipeline_ok(table)

            col_name = resolve_column(table, self.field_name)
            if col_name is None:
                return pipeline_ok(table)

            order = "descending" if self.descending else "ascending"
            indices = pc.sort_indices(table, sort_keys=[(col_name, order)])
            return pipeline_ok(table.take(indices))
        except Exception as e:
            return pipeline_err("order_by", f"Arrow sort failed: {e}", e)


@dataclass
class LimitStep(Step[Union[pa.Table, List[T]], Union[pa.Table, List[T]]], Generic[T]):
    """Limit number of results - Arrow-optimized.

    ::: This is-in-layer Domain-Specific-Language-Layer.
    ::: This is a step.
    ::: This is-in-process Main-Process.
    ::: This is stateless.
    """
    count: int

    def execute(self, data: Union[pa.Table, List[T]], ctx: Optional["Context"] = None) -> PipelineResult[Union[pa.Table, List[T]]]:
        if is_arrow(data):
            return pipeline_ok(data.slice(0, self.count))
        return pipeline_ok(data[:self.count])


@dataclass
class OffsetStep(Step[Union[pa.Table, List[T]], Union[pa.Table, List[T]]], Generic[T]):
    """Skip first N results - Arrow-optimized.

    ::: This is-in-layer Domain-Specific-Language-Layer.
    ::: This is a step.
    ::: This is-in-process Main-Process.
    ::: This is stateless.
    """
    count: int

    def execute(self, data: Union[pa.Table, List[T]], ctx: Optional["Context"] = None) -> PipelineResult[Union[pa.Table, List[T]]]:
        if is_arrow(data):
            return pipeline_ok(data.slice(self.count))
        return pipeline_ok(data[self.count:])


@dataclass
class MapStep(Step[Union[pa.Table, List[T]], Union[pa.Table, List[U]]], Generic[T, U]):
    """Transform each item using fmap semantics - Arrow-aware.

    ::: This is-in-layer Domain-Specific-Language-Layer.
    ::: This is a step.
    ::: This is-in-process Main-Process.
    ::: This is stateless.
    """
    transform: Callable[[T], U]

    def execute(self, data: Union[pa.Table, List[T]], ctx: Optional["Context"] = None) -> PipelineResult[Union[pa.Table, List[U]]]:
        try:
            # For Arrow tables, convert to list, apply transform, convert back
            # This preserves Arrow format while allowing arbitrary transforms
            if is_arrow(data):
                rows = data.to_pylist()
                transformed = [self.transform(item) for item in rows]
                if not transformed:
                    return pipeline_ok(pa.table({}))
                return pipeline_ok(pa.Table.from_pylist(transformed))

            return pipeline_ok([self.transform(item) for item in data])
        except Exception as e:
            return pipeline_err("map", f"Map failed: {e}", e)


@dataclass
class FlatMapStep(Step[Union[pa.Table, List[T]], Union[pa.Table, List[U]]], Generic[T, U]):
    """Transform and flatten using bind semantics - Arrow-aware.

    ::: This is-in-layer Domain-Specific-Language-Layer.
    ::: This is a step.
    ::: This is-in-process Main-Process.
    ::: This is stateless.
    """
    transform: Callable[[T], List[U]]

    def execute(self, data: Union[pa.Table, List[T]], ctx: Optional["Context"] = None) -> PipelineResult[Union[pa.Table, List[U]]]:
        try:
            # Convert Arrow to list for flat_map (variable output per row)
            if is_arrow(data):
                data = data.to_pylist()

            result = []
            for item in data:
                result.extend(self.transform(item))

            # Convert back to Arrow if input was Arrow
            if result:
                return pipeline_ok(pa.Table.from_pylist(result) if isinstance(result[0], dict) else result)
            return pipeline_ok(pa.table({}))
        except Exception as e:
            return pipeline_err("flat_map", f"FlatMap failed: {e}", e)


@dataclass
class GroupByStep(Step[List[Dict], Dict[str, Any]]):
    """Group items by field value or key function.

    ::: This is-in-layer Domain-Specific-Language-Layer.
    ::: This is a step.
    ::: This is-in-process Main-Process.
    ::: This is stateless.
    """
    field_name: Optional[str] = None
    key_fn: Optional[Callable[[Dict], str]] = None
    aggregate_fn: Optional[Callable[[List[Dict]], Any]] = None

    def execute(self, data: Union[pa.Table, List[Dict]], ctx: Optional["Context"] = None) -> PipelineResult[Dict[str, Any]]:
        try:
            # Convert Arrow table to list of dicts
            if is_arrow(data):
                data = data.to_pylist()

            groups: Dict[str, List[Dict]] = {}
            for item in data:
                if self.key_fn:
                    key = str(self.key_fn(item))
                elif self.field_name:
                    key = str(item.get(self.field_name, ""))
                else:
                    key = "default"
                if key not in groups:
                    groups[key] = []
                groups[key].append(item)

            # Apply aggregation if provided
            if self.aggregate_fn:
                result = {}
                for key, items in groups.items():
                    # Try to call with context if the function accepts it
                    try:
                        import inspect
                        sig = inspect.signature(self.aggregate_fn)
                        if len(sig.parameters) >= 2:
                            result[key] = self.aggregate_fn(items, ctx)
                        else:
                            result[key] = self.aggregate_fn(items)
                    except (ValueError, TypeError):
                        result[key] = self.aggregate_fn(items)
                return pipeline_ok({"items": result})

            return pipeline_ok(groups)
        except Exception as e:
            return pipeline_err("group_by", f"Group by failed: {e}", e)


@dataclass
class AggregateStep(Step[Union[pa.Table, List[Dict]], Dict[str, Any]]):
    """Aggregate data with specified functions - Arrow-optimized.

    ::: This is-in-layer Domain-Specific-Language-Layer.
    ::: This is a step.
    ::: This is-in-process Main-Process.
    ::: This is stateless.
    """
    aggregations: Dict[str, Tuple[str, str]]  # output -> (field, func)

    def execute(self, data: Union[pa.Table, List[Dict]], ctx: Optional["Context"] = None) -> PipelineResult[Dict[str, Any]]:
        try:
            if is_arrow(data):
                return self._arrow_aggregate(data)

            # List fallback
            result = {"count": len(data)}
            for out_name, (field_name, func) in self.aggregations.items():
                values = [item.get(field_name) for item in data if item.get(field_name) is not None]
                if func == "count":
                    result[out_name] = len(values)
                elif func == "sum":
                    result[out_name] = sum(values)
                elif func == "avg" or func == "mean":
                    result[out_name] = sum(values) / len(values) if values else 0
                elif func == "min":
                    result[out_name] = min(values) if values else None
                elif func == "max":
                    result[out_name] = max(values) if values else None
            return pipeline_ok(result)
        except Exception as e:
            return pipeline_err("aggregate", f"Aggregation failed: {e}", e)

    def _arrow_aggregate(self, table: pa.Table) -> PipelineResult[Dict[str, Any]]:
        """Aggregate using Arrow compute - vectorized."""
        try:
            result = {"count": table.num_rows}

            for out_name, (field_name, func) in self.aggregations.items():
                col_name = resolve_column(table, field_name)
                if col_name is None:
                    continue

                column = table.column(col_name)

                if func == "count":
                    result[out_name] = pc.count(column).as_py()
                elif func == "sum":
                    result[out_name] = pc.sum(column).as_py()
                elif func == "avg" or func == "mean":
                    result[out_name] = pc.mean(column).as_py()
                elif func == "min":
                    result[out_name] = pc.min(column).as_py()
                elif func == "max":
                    result[out_name] = pc.max(column).as_py()
                elif func == "stddev":
                    result[out_name] = pc.stddev(column).as_py()

            return pipeline_ok(result)
        except Exception as e:
            return pipeline_err("aggregate", f"Arrow aggregate failed: {e}", e)


@dataclass
class FlattenStep(Step[List[List[T]], List[T]], Generic[T]):
    """Flatten nested lists (join in monad terms).

    ::: This is-in-layer Domain-Specific-Language-Layer.
    ::: This is a step.
    ::: This is-in-process Main-Process.
    ::: This is stateless.
    """

    def execute(self, data: List[List[T]], ctx: Optional["Context"] = None) -> PipelineResult[List[T]]:
        try:
            return pipeline_ok([item for sublist in data for item in sublist])
        except Exception as e:
            return pipeline_err("flatten", f"Flatten failed: {e}", e)


@dataclass
class UniqueStep(Step[Union[pa.Table, List[T]], Union[pa.Table, List[T]]], Generic[T]):
    """Remove duplicates based on key - Arrow-optimized.

    ::: This is-in-layer Domain-Specific-Language-Layer.
    ::: This is a step.
    ::: This is-in-process Main-Process.
    ::: This is stateless.
    """
    key: Optional[Callable[[T], Any]] = None
    columns: Optional[List[str]] = None  # For Arrow: columns to dedupe on

    def execute(self, data: Union[pa.Table, List[T]], ctx: Optional["Context"] = None) -> PipelineResult[Union[pa.Table, List[T]]]:
        try:
            if is_arrow(data):
                return self._arrow_unique(data)

            # List fallback
            if self.key:
                seen = set()
                result = []
                for item in data:
                    k = self.key(item)
                    if k not in seen:
                        seen.add(k)
                        result.append(item)
                return pipeline_ok(result)
            else:
                seen = set()
                result = []
                for item in data:
                    if isinstance(item, dict):
                        k = tuple(sorted(item.items()))
                    else:
                        k = item
                    if k not in seen:
                        seen.add(k)
                        result.append(item)
                return pipeline_ok(result)
        except Exception as e:
            return pipeline_err("unique", f"Unique failed: {e}", e)

    def _arrow_unique(self, table: pa.Table) -> PipelineResult[pa.Table]:
        """Remove duplicates using Arrow group_by - vectorized."""
        try:
            if table.num_rows == 0:
                return pipeline_ok(table)

            # Determine columns to dedupe on
            if self.columns:
                cols = [resolve_column(table, c) for c in self.columns if resolve_column(table, c)]
            else:
                cols = table.column_names

            if not cols:
                return pipeline_ok(table)

            # Use group_by with no aggregations to get unique rows
            unique_table = table.group_by(cols).aggregate([])
            return pipeline_ok(unique_table)
        except Exception as e:
            return pipeline_err("unique", f"Arrow unique failed: {e}", e)


@dataclass
class TapStep(Step[T, T], Generic[T]):
    """Execute a side effect function and pass through data unchanged.

    ::: This is-in-layer Domain-Specific-Language-Layer.
    ::: This is a step.
    ::: This is-in-process Main-Process.
    ::: This is stateless.
    """
    fn: Callable[[T], Any]

    def execute(self, data: T, ctx: Optional["Context"] = None) -> PipelineResult[T]:
        try:
            # Execute side effect - result is stored but data passes through
            # Try to call with context if the function accepts it
            try:
                import inspect
                sig = inspect.signature(self.fn)
                if len(sig.parameters) >= 2:
                    result = self.fn(data, ctx)
                else:
                    result = self.fn(data)
            except (ValueError, TypeError):
                result = self.fn(data)
            # If the tap function returns something useful, attach it to the data
            if result is not None and isinstance(data, (list, dict)):
                # Store tap result for later use (e.g., by render step)
                if isinstance(data, list):
                    return pipeline_ok({"items": data, "_tap_result": result})
                elif isinstance(data, dict):
                    data_copy = dict(data)
                    data_copy["_tap_result"] = result
                    return pipeline_ok(data_copy)
            return pipeline_ok(data)
        except Exception as e:
            return pipeline_err("tap", f"Tap failed: {e}", e)


@dataclass
class RenderStep(Step[Any, str]):
    """Render data into a formatted string.

    ::: This is-in-layer Domain-Specific-Language-Layer.
    ::: This is a step.
    ::: This is-in-process Main-Process.
    ::: This is stateless.
    """
    format: str
    renderer: Callable[[Any, str], Any]

    def execute(self, data: Any, ctx: Optional["Context"] = None) -> PipelineResult[str]:
        try:
            # If data has tap result, use that
            if isinstance(data, dict) and "_tap_result" in data:
                render_data = data["_tap_result"]
            else:
                render_data = data
            # Resolve format from context params if it's a placeholder
            fmt = self._resolve_param(self.format, ctx)
            result = self.renderer(render_data, fmt)
            return pipeline_ok(result)
        except Exception as e:
            return pipeline_err("render", f"Render failed: {e}", e)


# =============================================================================
# Pipeline Monad
# =============================================================================

@dataclass
class Pipeline(Monad[T], Generic[T]):
    """
    A monadic, composable, lazy data transformation pipeline.

    Pipeline is a Monad where:
    - pure(x) creates a pipeline that yields x
    - bind(f) chains a function that returns another pipeline
    - fmap(f) transforms the pipeline output

    The pipeline is lazy - it only executes when run() is called.

    Example:
        pipeline = (
            Pipeline.from_reql("SELECT ...")
            .filter(lambda x: x["count"] > 10)
            .fmap(lambda x: x["name"])  # Functor operation
            .order_by("-count")
            .limit(100)
        )
        result = pipeline.run(context)  # Result[T, PipelineError]

    ::: This is-in-layer Domain-Specific-Language-Layer.
    ::: This is a monad.
    ::: This is-in-process Main-Process.
    ::: This is stateless.
    """
    _source: Source[Any]
    _steps: List[Step] = field(default_factory=list)
    _emit_key: Optional[str] = None

    # -------------------------------------------------------------------------
    # Monad Implementation
    # -------------------------------------------------------------------------

    @classmethod
    def pure(cls, value: U) -> "Pipeline[U]":
        """Lift a value into a pipeline (Applicative.pure)."""
        return cls(_source=ValueSource(value))

    def bind(self, f: Callable[[T], "Pipeline[U]"]) -> "Pipeline[U]":
        """
        Chain a function that returns a pipeline (Monad.bind).

        This is the core monadic operation that allows sequencing
        computations where each step can produce a new pipeline.
        """
        # Create a new pipeline that, when executed:
        # 1. Runs this pipeline
        # 2. Applies f to the result to get a new pipeline
        # 3. Runs that new pipeline
        return BoundPipeline(self, f)

    def fmap(self, f: Callable[[T], U]) -> "Pipeline[U]":
        """Map a function over the pipeline output (Functor.fmap)."""
        # For lists, we want to map over each element
        # This creates a new pipeline with a map step
        return self._add_step(MapStep(f))

    def ap(self, x: "Pipeline[T]") -> "Pipeline[U]":
        """Apply a wrapped function to a wrapped value (Applicative.ap)."""
        # Default monad implementation
        return self.bind(lambda f: x.bind(lambda a: Pipeline.pure(f(a))))

    def _add_step(self, step: Step) -> "Pipeline":
        """Add a step to the pipeline and return a new pipeline."""
        return Pipeline(
            _source=self._source,
            _steps=self._steps + [step],
            _emit_key=self._emit_key
        )

    # -------------------------------------------------------------------------
    # Constructors
    # -------------------------------------------------------------------------

    @classmethod
    def from_source(cls, source: Source[T]) -> "Pipeline[T]":
        """Create pipeline from a custom Source."""
        return cls(_source=source)

    @classmethod
    def from_reql(cls, query: str) -> "Pipeline[List[Dict]]":
        """Create pipeline from REQL query."""
        return cls(_source=REQLSource(query))

    @classmethod
    def from_value(cls, value: T) -> "Pipeline[T]":
        """Create pipeline from literal value."""
        return cls(_source=ValueSource(value))

    @classmethod
    def from_rag(cls, query: str, top_k: int = 10,
                 entity_types: Optional[List[str]] = None) -> "Pipeline[List[Dict]]":
        """Create pipeline from RAG semantic search."""
        return cls(_source=RAGSource(query, top_k, entity_types))

    # -------------------------------------------------------------------------
    # Transformation Methods (fluent API using Steps)
    # -------------------------------------------------------------------------

    def filter(self, predicate: Callable[[Any], bool],
               when: Optional[Callable[[], bool]] = None) -> "Pipeline":
        """Filter items matching predicate."""
        return self._add_step(FilterStep(predicate, when))

    def select(self, *fields: str, **renames: str) -> "Pipeline":
        """Select and optionally rename fields."""
        field_map = {f: f for f in fields}
        field_map.update(renames)
        return self._add_step(SelectStep(field_map))

    def order_by(self, field: str) -> "Pipeline":
        """Sort by field. Prefix with - for descending."""
        descending = field.startswith("-")
        if descending:
            field = field[1:]
        return self._add_step(OrderByStep(field, descending))

    def limit(self, count: int) -> "Pipeline":
        """Limit number of results."""
        return self._add_step(LimitStep(count))

    def offset(self, count: int) -> "Pipeline":
        """Skip first N results."""
        return self._add_step(OffsetStep(count))

    def map(self, transform: Callable[[Any], Any]) -> "Pipeline":
        """Transform each item (alias for fmap on list elements)."""
        return self._add_step(MapStep(transform))

    def flat_map(self, transform: Callable[[Any], List]) -> "Pipeline":
        """Transform and flatten (monadic bind for lists)."""
        return self._add_step(FlatMapStep(transform))

    def group_by(self, field: Optional[str] = None, *,
                 key: Optional[Callable[[Any], str]] = None,
                 aggregate: Optional[Callable[[List], Any]] = None) -> "Pipeline":
        """Group items by field or key function, with optional aggregation."""
        return self._add_step(GroupByStep(field_name=field, key_fn=key, aggregate_fn=aggregate))

    def aggregate(self, **aggregations: Tuple[str, str]) -> "Pipeline":
        """Aggregate with functions: field=("source", "sum|avg|min|max|count")"""
        return self._add_step(AggregateStep(aggregations))

    def flatten(self) -> "Pipeline":
        """Flatten nested lists (monad join)."""
        return self._add_step(FlattenStep())

    def unique(self, key: Optional[Callable[[Any], Any]] = None) -> "Pipeline":
        """Remove duplicates."""
        return self._add_step(UniqueStep(key))

    def tap(self, fn: Callable[[Any], Any]) -> "Pipeline":
        """Execute a side effect function and pass through data unchanged."""
        return self._add_step(TapStep(fn))

    def render(self, format: str, renderer: Callable[[Any, str], Any]) -> "Pipeline":
        """Render data into a formatted output."""
        return self._add_step(RenderStep(format, renderer))

    def emit(self, key: str) -> "Pipeline":
        """Set the output key for the result."""
        return Pipeline(_source=self._source, _steps=self._steps, _emit_key=key)

    # -------------------------------------------------------------------------
    # Operator Overloads
    # -------------------------------------------------------------------------

    def __rshift__(self, step: Step) -> "Pipeline":
        """Syntactic sugar: pipeline >> step"""
        return self._add_step(step)

    def __or__(self, step: Step) -> "Pipeline":
        """Alternative syntax: pipeline | step"""
        return self >> step

    # -------------------------------------------------------------------------
    # Execution
    # -------------------------------------------------------------------------

    def run(self, ctx: Context) -> PipelineResult[T]:
        """
        Execute the pipeline and return Result.

        This is where the lazy pipeline is evaluated.
        Errors short-circuit using Result monad semantics.
        """
        # Execute source
        source_result = self._source.execute(ctx)
        if source_result.is_err():
            return source_result

        # Run through steps using monadic bind
        current = source_result.unwrap()
        for step in self._steps:
            result = step.execute(current, ctx)
            if result.is_err():
                return result
            current = result.unwrap()

        return pipeline_ok(current)

    def execute(self, ctx: Context) -> Dict[str, Any]:
        """Execute and return formatted output dict."""
        result = self.run(ctx)

        if result.is_err():
            return {
                "success": False,
                "error": str(result.error) if hasattr(result, 'error') else str(result)
            }

        data = result.unwrap()

        # Convert Arrow table to list at output boundary
        if is_arrow(data):
            data = data.to_pylist()

        output = {"success": True}

        if self._emit_key:
            output[self._emit_key] = data
            if isinstance(data, list):
                output["count"] = len(data)
        else:
            if isinstance(data, dict):
                output.update(data)
            elif isinstance(data, list):
                output["results"] = data
                output["count"] = len(data)
            else:
                output["result"] = data

        return output


class BoundPipeline(Generic[T, U]):
    """
    A pipeline created by monadic bind.

    This represents the composition of a pipeline with a function
    that produces another pipeline.

    ::: This is-in-layer Domain-Specific-Language-Layer.
    ::: This is a monad.
    ::: This is-in-process Main-Process.
    ::: This is stateless.
    """
    def __init__(self, source_pipeline: Pipeline[T], continuation: Callable[[T], Pipeline[U]]):
        self.source_pipeline = source_pipeline
        self.continuation = continuation
        self._emit_key: Optional[str] = None

    @classmethod
    def pure(cls, value: V) -> "Pipeline[V]":
        """Lift a value into a pipeline."""
        return Pipeline.pure(value)

    def bind(self, f: Callable[[U], "Pipeline[V]"]) -> "BoundPipeline[U, V]":
        """Chain another bind."""
        return BoundPipeline(self, f)  # type: ignore

    def run(self, ctx: Context) -> PipelineResult[U]:
        """Execute the bound pipeline."""
        # First run the source pipeline
        result = self.source_pipeline.run(ctx)
        if result.is_err():
            return result

        # Apply continuation to get next pipeline
        next_pipeline = self.continuation(result.unwrap())

        # Run the next pipeline
        return next_pipeline.run(ctx)

    def execute(self, ctx: Context) -> Dict[str, Any]:
        """Execute and return formatted output dict."""
        result = self.run(ctx)

        if result.is_err():
            return {
                "success": False,
                "error": str(result.error) if hasattr(result, 'error') else str(result)
            }

        data = result.unwrap()

        # Convert Arrow table to list at output boundary
        if is_arrow(data):
            data = data.to_pylist()

        output = {"success": True}

        if self._emit_key:
            output[self._emit_key] = data
            if isinstance(data, list):
                output["count"] = len(data)
        else:
            if isinstance(data, dict):
                output.update(data)
            elif isinstance(data, list):
                output["results"] = data
                output["count"] = len(data)
            else:
                output["result"] = data

        return output


# =============================================================================
# Tool Specification Types
# =============================================================================

class ToolType(Enum):
    """Type of CADSL tool.

    ::: This is-in-layer Domain-Specific-Language-Layer.
    ::: This is a value-object.
    ::: This is-in-process Main-Process.
    ::: This is stateless.
    """
    QUERY = "query"
    DETECTOR = "detector"
    DIAGRAM = "diagram"


@dataclass
class ParamSpec:
    """Specification for a tool parameter.

    ::: This is-in-layer Domain-Specific-Language-Layer.
    ::: This is a value-object.
    ::: This is-in-process Main-Process.
    ::: This is stateless.
    """
    name: str
    type: type
    required: bool = True
    default: Any = None
    description: str = ""
    choices: Optional[List[Any]] = None

    def validate(self, value: Any) -> PipelineResult[Any]:
        """Validate a parameter value using Result monad."""
        if value is None:
            # If a default is provided, use it regardless of required flag
            if self.default is not None:
                return pipeline_ok(self.default)
            # Only error if required and no default
            if self.required:
                return pipeline_err("param", f"Required parameter '{self.name}' not provided")
            return pipeline_ok(self.default)

        if self.choices and value not in self.choices:
            return pipeline_err("param", f"Parameter '{self.name}' must be one of {self.choices}")

        try:
            # Type coercion
            if self.type == bool:
                if isinstance(value, str):
                    value = value.lower() in ("true", "1", "yes")
                else:
                    value = bool(value)
            elif self.type == int:
                value = int(value)
            elif self.type == float:
                value = float(value)
            elif self.type == str:
                value = str(value)

            return pipeline_ok(value)
        except (ValueError, TypeError) as e:
            return pipeline_err("param", f"Invalid type for '{self.name}': {e}")


@dataclass
class ToolSpec:
    """Specification for a CADSL tool.

    ::: This is-in-layer Domain-Specific-Language-Layer.
    ::: This is a value-object.
    ::: This is-in-process Main-Process.
    ::: This is stateless.
    """
    name: str
    type: ToolType
    description: str
    pipeline_factory: Callable[[Context], Pipeline]
    params: Dict[str, ParamSpec] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Tool Type Classes
# =============================================================================

class Query:
    """
    A read-only code inspection tool (functor over code data).

    Queries do not modify state - they only retrieve and transform data.

    ::: This is-in-layer Domain-Specific-Language-Layer.
    ::: This is a tool.
    ::: This is-in-process Main-Process.
    ::: This is stateless.
    """

    def __init__(self, spec: ToolSpec):
        self.spec = spec

    def execute(self, ctx: Context) -> Dict[str, Any]:
        """Execute the query and return results."""
        pipeline = self.spec.pipeline_factory(ctx)
        return pipeline.execute(ctx)


class Detector:
    """
    A code smell/pattern detector that creates findings.

    Detectors analyze code and produce findings with severity levels.

    ::: This is-in-layer Domain-Specific-Language-Layer.
    ::: This is a tool.
    ::: This is-in-process Main-Process.
    ::: This is stateless.
    """

    def __init__(self, spec: ToolSpec):
        self.spec = spec

    def execute(self, ctx: Context) -> Dict[str, Any]:
        """Execute the detector and return findings."""
        pipeline = self.spec.pipeline_factory(ctx)
        result = pipeline.execute(ctx)

        # Add detector metadata
        result["detector"] = self.spec.name
        result["category"] = self.spec.meta.get("category", "general")
        result["severity"] = self.spec.meta.get("severity", "medium")

        return result


class Diagram:
    """
    A visualization generator (transforms code structure to visual format).

    ::: This is-in-layer Domain-Specific-Language-Layer.
    ::: This is a tool.
    ::: This is-in-process Main-Process.
    ::: This is stateless.
    """

    def __init__(self, spec: ToolSpec):
        self.spec = spec

    def execute(self, ctx: Context) -> Dict[str, Any]:
        """Execute the diagram generator and return visualization."""
        pipeline = self.spec.pipeline_factory(ctx)
        result = pipeline.execute(ctx)

        result["format"] = ctx.params.get("format", "mermaid")
        return result


# =============================================================================
# Builder Functions (for DSL convenience)
# =============================================================================

def reql(query: str) -> Pipeline:
    """Create a pipeline from REQL query."""
    return Pipeline.from_reql(query)


def rag(query: str, top_k: int = 10,
        entity_types: Optional[List[str]] = None) -> Pipeline:
    """Create a pipeline from RAG semantic search."""
    return Pipeline.from_rag(query, top_k, entity_types)


def value(data: Any) -> Pipeline:
    """Create a pipeline from a literal value."""
    return Pipeline.from_value(data)
