"""
RAG Search Mixin

Contains search and result aggregation methods for the RAG system.
Extracted from RAGIndexManager to reduce file size.
"""

import fnmatch
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

from .rag_types import RAGSearchResult
from .initialization_progress import require_rag_code_index, require_rag_document_index
from ..logging_config import configure_logger_for_debug_trace

logger = configure_logger_for_debug_trace(__name__)


class RAGSearchMixin:
    """
    Mixin providing search methods for RAGIndexManager.

    ::: This is-in-layer Service-Layer.
    ::: This is-part-of-component RAG-Index.
    ::: This is-in-process Main-Process.
    ::: This is stateless.

    These methods require:
    - self._faiss_wrapper: FAISSWrapper instance
    - self._embedding_service: EmbeddingService instance
    - self._metadata: dict mapping vector IDs to entity metadata
    - self._content_extractor: ContentExtractor instance
    """

    def search(
        self,
        query: str,
        top_k: int = 10,
        entity_types: Optional[List[str]] = None,
        file_filter: Optional[str] = None,
        search_scope: str = "all",  # "all", "code" (Python/JS/HTML), "docs"
        include_content: bool = False,
        aggregate_chunks: bool = True
    ) -> Tuple[List[RAGSearchResult], Dict[str, Any]]:
        """
        Semantic search over indexed code and documentation.

        Args:
            query: Natural language search query
            top_k: Maximum number of results
            entity_types: Filter by type (class, method, function, section, etc.)
            file_filter: Glob pattern to filter files
            search_scope: "all", "code" (Python, JavaScript, HTML), "docs" (Markdown only)
            include_content: Include full content in results
            aggregate_chunks: If True (default), deduplicate results by parent entity
                            when chunking is enabled, keeping highest score per entity

        Returns:
            Tuple of (results list, search stats)

        Raises:
            ComponentNotReadyError: If required RAG component is not ready
        """
        # Check appropriate components based on search scope
        if search_scope == "code":
            require_rag_code_index()
        elif search_scope == "docs":
            require_rag_document_index()
        else:  # "all"
            require_rag_code_index()
            require_rag_document_index()

        if not self._enabled:
            return [], {"status": "disabled", "error": "RAG is disabled via configuration"}

        if not self._initialized:
            return [], {
                "status": "not_initialized",
                "error": "RAG index not initialized. Call rag_reindex(force=True) first to build the index."
            }

        start_time = time.time()

        # Generate query embedding
        embed_start = time.time()
        query_embedding = self._embedding_service.generate_embedding(query)
        embed_time_ms = int((time.time() - embed_start) * 1000)

        # Search FAISS (get more results for filtering)
        search_start = time.time()
        search_results = self._faiss_wrapper.search_with_scores(
            query_embedding, top_k=top_k * 3  # Over-fetch for filtering
        )
        search_time_ms = int((time.time() - search_start) * 1000)

        # Filter and enrich results
        results = []
        for sr in search_results:
            if sr.vector_id == -1:
                continue

            meta = self._metadata["vectors"].get(str(sr.vector_id))
            if not meta:
                continue

            # Apply scope filter
            source_type = meta.get("source_type", "python")
            # Code includes: python, javascript, html, csharp, cpp (and their literal variants)
            code_types = (
                "python", "python_literal", "python_comment",
                "javascript", "javascript_literal", "html",
                "java", "csharp", "cpp", "go", "rust", "erlang",
                "php", "objc", "swift", "vb6", "scala", "haskell", "kotlin", "r", "ruby", "dart", "delphi",
                "java_comment", "csharp_comment", "cpp_comment",
                "go_comment", "rust_comment", "erlang_comment",
                "php_comment", "objc_comment", "swift_comment",
                "vb6_comment", "scala_comment", "haskell_comment", "kotlin_comment", "r_comment", "ruby_comment", "dart_comment", "delphi_comment",
                "javascript_comment",
            )
            if search_scope == "code" and source_type not in code_types:
                continue
            if search_scope == "docs" and source_type != "markdown":
                continue

            # Apply entity type filter
            if entity_types and meta.get("entity_type") not in entity_types:
                continue

            # Apply file filter
            if file_filter:
                file_path = meta.get("file", "")
                if not fnmatch.fnmatch(file_path, file_filter):
                    continue

            # Build result
            content = None
            if include_content:
                content = self._get_entity_content(meta)

            # Extract chunk metadata if present
            chunk_index = meta.get("chunk_index")
            total_chunks = meta.get("total_chunks")
            chunk_line_start = meta.get("chunk_line_start")
            chunk_line_end = meta.get("chunk_line_end")
            parent_qualified_name = meta.get("parent_qualified_name")

            result = RAGSearchResult(
                entity_type=meta.get("entity_type", "unknown"),
                name=meta.get("name", ""),
                qualified_name=meta.get("qualified_name", ""),
                file=meta.get("file", ""),
                line=meta.get("line", 0),
                end_line=meta.get("end_line"),
                score=sr.score,
                source_type=source_type,
                docstring=meta.get("docstring_preview"),
                content_preview=meta.get("content_preview"),
                content=content,
                heading=meta.get("heading"),
                language=meta.get("language"),
                class_name=meta.get("class_name"),
                chunk_index=chunk_index,
                total_chunks=total_chunks,
                chunk_line_start=chunk_line_start,
                chunk_line_end=chunk_line_end,
                parent_qualified_name=parent_qualified_name,
            )
            results.append(result)

            if len(results) >= top_k * 3:  # Collect more for aggregation
                break

        # Aggregate chunks: group by parent entity, keep highest score
        if aggregate_chunks:
            results = self._aggregate_chunk_results(results, top_k)
        else:
            results = results[:top_k]

        stats = {
            "query_embedding_time_ms": embed_time_ms,
            "search_time_ms": search_time_ms,
            "total_time_ms": int((time.time() - start_time) * 1000),
            "total_vectors": self._faiss_wrapper.total_vectors,
            "results_before_filter": len(search_results),
        }

        return results, stats

    def _aggregate_chunk_results(
        self,
        results: List[RAGSearchResult],
        top_k: int
    ) -> List[RAGSearchResult]:
        """
        Aggregate chunked results by parent entity, keeping highest score per entity.

        For chunked entities, groups results by parent_qualified_name and keeps
        only the highest-scoring chunk. Non-chunked entities pass through unchanged.

        Args:
            results: List of search results (may include chunks)
            top_k: Maximum results to return

        Returns:
            Deduplicated results list
        """
        # Track best result per entity (by parent_qualified_name or qualified_name)
        best_by_entity: Dict[str, RAGSearchResult] = {}

        for result in results:
            # Determine the grouping key
            if result.parent_qualified_name:
                # This is a chunk - group by parent
                key = result.parent_qualified_name
            else:
                # Non-chunked entity - use its own qualified_name
                key = result.qualified_name

            # Keep only the highest-scoring result per entity
            if key not in best_by_entity or result.score > best_by_entity[key].score:
                best_by_entity[key] = result

        # Sort by score and return top_k
        aggregated = sorted(best_by_entity.values(), key=lambda r: r.score, reverse=True)
        return aggregated[:top_k]

    def _get_entity_content(self, meta: Dict[str, Any]) -> Optional[str]:
        """Get full content for an entity."""
        file_path = meta.get("file")
        line = meta.get("line")
        end_line = meta.get("end_line")

        if not file_path or not line:
            return None

        if self._project_root:
            abs_path = self._project_root / file_path
        else:
            abs_path = Path(file_path)

        try:
            content = abs_path.read_text(encoding='utf-8')
            lines = content.split('\n')

            start_idx = line - 1
            end_idx = (end_line - 1) if end_line else start_idx

            if start_idx < 0 or start_idx >= len(lines):
                return None

            end_idx = min(end_idx, len(lines) - 1)
            return '\n'.join(lines[start_idx:end_idx + 1])
        except Exception as e:
            logger.debug(f"Failed to get content for {file_path}: {e}")
            return None
