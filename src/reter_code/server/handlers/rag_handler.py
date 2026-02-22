"""
RETER RAG Handler.

Handles RAG (Retrieval-Augmented Generation) operations via ZeroMQ.

::: This is-in-layer Handler-Layer.
::: This is-in-component RAG-Handlers.
::: This depends-on reter_code.services.rag_index_manager.
"""

from typing import Any, Dict, List

from . import BaseHandler
from ..protocol import (
    METHOD_RAG_SEARCH,
    METHOD_RAG_REINDEX,
    RAG_ERROR,
)


class RAGHandler(BaseHandler):
    """Handler for RAG operations (search, reindex).

    ::: This is-in-layer Service-Layer.
    ::: This is a handler.
    ::: This is stateful.
    """

    def _register_methods(self) -> None:
        """Register RAG method handlers."""
        self._methods = {
            METHOD_RAG_SEARCH: self._handle_semantic_search,
            METHOD_RAG_REINDEX: self._handle_reindex,
            "rag_duplicates": self._handle_duplicates,
            "rag_clusters": self._handle_clusters,
            "rag_status": self._handle_status,
        }


    def _handle_semantic_search(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute semantic search over code and docs.

        Params:
            query: Natural language search query
            top_k: Maximum number of results (default: 10)
            entity_types: Filter by type (class, method, function, etc.)
            file_filter: Glob pattern to filter files
            include_content: Include source code in results
            search_scope: "all", "code", or "docs"

        Returns:
            Dictionary with search results and metadata
        """
        query = params.get("query", "")
        top_k = params.get("top_k", 10)
        entity_types = params.get("entity_types")
        file_filter = params.get("file_filter")
        include_content = params.get("include_content", False)
        search_scope = params.get("search_scope", "all")

        if not query:
            raise ValueError("Query is required")

        # Execute search via RAG manager
        # Returns (results_list, stats_dict) tuple
        results_list, stats = self.rag_manager.search(
            query=query,
            top_k=top_k,
            entity_types=entity_types,
            file_filter=file_filter,
            include_content=include_content,
            search_scope=search_scope
        )

        # Convert RAGSearchResult objects to dicts
        results_dicts = [
            {
                "entity_type": r.entity_type,
                "name": r.name,
                "qualified_name": r.qualified_name,
                "file": r.file,
                "line": r.line,
                "end_line": r.end_line,
                "score": r.score,
                "source_type": r.source_type,
                "docstring": r.docstring,
                "content_preview": r.content_preview,
                "content": r.content,
                "heading": r.heading,
                "language": r.language,
                "class_name": r.class_name,
            }
            for r in results_list
        ]

        return {
            "success": True,
            "results": results_dicts,
            "count": len(results_dicts),
            "query_embedding_time_ms": stats.get("query_embedding_time_ms", 0),
            "search_time_ms": stats.get("search_time_ms", 0),
            "total_vectors": stats.get("total_vectors", 0)
        }

    def _handle_reindex(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Rebuild RAG index.

        Params:
            force: Force full rebuild even if index exists

        Returns:
            Dictionary with reindex statistics
        """
        project_root = self.rag_manager._project_root
        if not project_root:
            return {
                "success": False,
                "error": "RAG not initialized (no project root)"
            }

        result = self.rag_manager.reindex_all(self.reter, project_root)

        return {
            "success": True,
            "vectors_indexed": result.get("vectors_indexed", result.get("total_vectors", 0)),
            "files_processed": result.get("files_processed", 0),
            "execution_time_ms": result.get("execution_time_ms", 0)
        }

    def _handle_duplicates(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Find duplicate/similar code using embeddings.

        Params:
            threshold: Similarity threshold (default: 0.85)
            entity_types: Filter by type
            limit: Maximum pairs to return

        Returns:
            Dictionary with duplicate pairs
        """
        threshold = params.get("threshold", 0.85)
        entity_types = params.get("entity_types")
        limit = params.get("limit", 100)

        result = self.rag_manager.find_duplicates(
            threshold=threshold,
            entity_types=entity_types,
            limit=limit
        )

        return {
            "success": True,
            "duplicates": result.get("duplicates", []),
            "count": result.get("count", 0),
            "execution_time_ms": result.get("execution_time_ms", 0)
        }

    def _handle_clusters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Cluster code entities by semantic similarity.

        Params:
            n_clusters: Number of clusters (default: 50)
            entity_types: Filter by type

        Returns:
            Dictionary with cluster assignments
        """
        n_clusters = params.get("n_clusters", 50)
        entity_types = params.get("entity_types")

        result = self.rag_manager.cluster(
            n_clusters=n_clusters,
            entity_types=entity_types
        )

        return {
            "success": True,
            "clusters": result.get("clusters", []),
            "count": result.get("count", 0),
            "execution_time_ms": result.get("execution_time_ms", 0)
        }

    def _handle_status(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get RAG index status.

        Returns:
            Dictionary with index statistics
        """
        status = self.rag_manager.get_status()

        return {
            "success": True,
            "initialized": status.get("initialized", False),
            "total_vectors": status.get("total_vectors", 0),
            "index_size_mb": status.get("index_size_mb", 0),
            "last_updated": status.get("last_updated"),
            "embedding_model": status.get("embedding_model", "unknown")
        }


__all__ = ["RAGHandler"]
