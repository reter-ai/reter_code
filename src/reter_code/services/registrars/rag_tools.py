"""
RAG Tools Registrar

Registers the semantic_search MCP tool.
Other RAG-related tools (rag_status, rag_reindex, init_status) moved to system tool.
"""

from typing import Dict, Any, Optional, List
from fastmcp import FastMCP
from .base import ToolRegistrarBase, truncate_mcp_response
from ..initialization_progress import (
    ComponentNotReadyError,
    require_rag_code_index,
    require_rag_document_index,
)
from ...reter_wrapper import DefaultInstanceNotInitialised


class RAGToolsRegistrar(ToolRegistrarBase):
    """Registers RAG tools with FastMCP."""

    def __init__(self, instance_manager, persistence_service, default_manager=None):
        super().__init__(instance_manager, persistence_service)
        self._default_manager = default_manager

    def register(self, app: FastMCP) -> None:
        """Register RAG tools (semantic_search only - others moved to system tool)."""
        self._register_semantic_search(app)

    def _get_rag_manager(self):
        """Get the RAG manager from the default instance manager."""
        if self._default_manager is None:
            return None
        return self._default_manager.get_rag_manager()

    def _register_semantic_search(self, app: FastMCP) -> None:
        """Register the semantic_search tool."""
        registrar = self

        @app.tool()
        @truncate_mcp_response
        def semantic_search(
            query: str,
            top_k: int = 10,
            entity_types: Optional[List[str]] = None,
            file_filter: Optional[str] = None,
            include_content: bool = False,
            search_scope: str = "all"
        ) -> Dict[str, Any]:
            """
            Search code and documentation semantically using natural language.

            Uses FAISS vector similarity to find code entities and documentation
            sections that are semantically similar to your query.

            Args:
                query: Natural language search query (e.g., "function that handles authentication")
                top_k: Maximum number of results (default: 10)
                entity_types: Filter by type: ["class", "method", "function", "section", "document", "code_block"]
                file_filter: Glob pattern to filter files (e.g., "src/**/*.py", "docs/**/*.md")
                include_content: Include source code/content in results (default: False)
                search_scope: "all" (code+docs), "code" (Python only), "docs" (Markdown only)

            Returns:
                {
                    "success": True,
                    "results": [
                        {
                            "entity_type": "method",
                            "name": "authenticate_user",
                            "qualified_name": "auth.service.AuthService.authenticate_user",
                            "file": "src/auth/service.py",
                            "line": 45,
                            "end_line": 78,
                            "score": 0.92,
                            "source_type": "python",
                            "docstring": "Authenticate a user with credentials...",
                            "content": "def authenticate_user(...)..."  # if include_content=True
                        },
                        ...
                    ],
                    "count": 10,
                    "query_embedding_time_ms": 12,
                    "search_time_ms": 3,
                    "total_vectors": 1523
                }

            Examples:
                - semantic_search("authentication and JWT tokens")
                - semantic_search("error handling", entity_types=["method", "function"])
                - semantic_search("installation guide", search_scope="docs")
                - semantic_search("database connection", file_filter="src/db/**")
            """
            # Check RAG component readiness based on search scope
            try:
                if search_scope == "code":
                    require_rag_code_index()
                elif search_scope == "docs":
                    require_rag_document_index()
                else:  # "all"
                    require_rag_code_index()
                    require_rag_document_index()
            except ComponentNotReadyError as e:
                return e.to_response(results=[], count=0)

            rag_manager = registrar._get_rag_manager()

            if rag_manager is None:
                return {
                    "success": False,
                    "error": "RAG is not configured. Set RETER_PROJECT_ROOT to enable.",
                    "results": [],
                    "count": 0,
                }

            if not rag_manager.is_enabled:
                return {
                    "success": False,
                    "error": "RAG is disabled via configuration (rag_enabled=false).",
                    "results": [],
                    "count": 0,
                }

            try:
                # Check sync status and auto-sync if stale
                synced = False
                try:
                    reter = registrar._get_reter("default")
                    sync_status = rag_manager.get_sync_status(reter)
                    if not sync_status.get("is_synced", True):
                        # Auto-sync the index
                        project_root = registrar._default_manager.project_root
                        if project_root:
                            rag_manager.sync_sources(reter, project_root)
                            synced = True
                except Exception:
                    pass  # Don't block search on sync errors

                results, stats = rag_manager.search(
                    query=query,
                    top_k=top_k,
                    entity_types=entity_types,
                    file_filter=file_filter,
                    search_scope=search_scope,
                    include_content=include_content,
                )

                response = {
                    "success": True,
                    "results": [r.to_dict(include_content=include_content) for r in results],
                    "count": len(results),
                    **stats,
                }

                # Indicate if we auto-synced
                if synced:
                    response["auto_synced"] = True

                return response

            except ComponentNotReadyError as e:
                return e.to_response(results=[], count=0)
            except DefaultInstanceNotInitialised as e:
                return {
                    "success": False,
                    "error": str(e),
                    "status": "initializing",
                    "results": [],
                    "count": 0,
                }
            except Exception as e:
                return {
                    "success": False,
                    "error": str(e),
                    "results": [],
                    "count": 0,
                }

    def _register_find_similar_clusters(self, app: FastMCP) -> None:
        """Register the find_similar_clusters tool."""
        registrar = self

        @app.tool()
        @truncate_mcp_response
        def find_similar_clusters(
            n_clusters: int = 50,
            min_cluster_size: int = 2,
            exclude_same_file: bool = True,
            exclude_same_class: bool = True,
            entity_types: Optional[List[str]] = None,
            source_type: Optional[str] = None
        ) -> Dict[str, Any]:
            """
            Find clusters of semantically similar code using K-means clustering.

            Uses FAISS K-means to group code entities by semantic similarity,
            then filters to find potential duplicates (similar code in different locations).

            This is useful for:
            - Finding duplicated code across different files
            - Identifying similar implementations that could be unified
            - Discovering patterns of code reuse

            Args:
                n_clusters: Number of clusters to create (default: 50, auto-adjusted)
                min_cluster_size: Minimum members for a cluster (default: 2)
                exclude_same_file: Exclude clusters where all members are from same file (default: True)
                exclude_same_class: Exclude clusters where all members are from same class (default: True)
                entity_types: Filter by type: ["class", "method", "function"] (default: all)
                source_type: Filter by source: "python" or "markdown" (default: all)

            Returns:
                {
                    "success": True,
                    "total_clusters": 15,
                    "total_vectors_analyzed": 3216,
                    "clusters": [
                        {
                            "cluster_id": 5,
                            "member_count": 3,
                            "unique_files": 2,
                            "unique_classes": 2,
                            "avg_distance": 0.15,
                            "members": [
                                {
                                    "name": "find_large_classes",
                                    "file": "advanced_python_tools.py",
                                    "line": 59,
                                    "entity_type": "method",
                                    "class_name": "AdvancedPythonTools"
                                },
                                {
                                    "name": "find_large_classes",
                                    "file": "code_quality.py",
                                    "line": 15,
                                    "entity_type": "method",
                                    "class_name": "CodeQualityTools"
                                }
                            ]
                        }
                    ],
                    "time_ms": 250
                }
            """
            # Clustering requires RAG code index to be ready
            try:
                require_rag_code_index()
            except ComponentNotReadyError as e:
                return e.to_response(clusters=[])

            rag_manager = registrar._get_rag_manager()

            if rag_manager is None:
                return {
                    "success": False,
                    "error": "RAG is not configured. Set RETER_PROJECT_ROOT to enable.",
                    "clusters": [],
                }

            try:
                # Check sync status and auto-sync if stale
                synced = False
                try:
                    reter = registrar._get_reter("default")
                    sync_status = rag_manager.get_sync_status(reter)
                    if not sync_status.get("is_synced", True):
                        project_root = registrar._default_manager.project_root
                        if project_root:
                            rag_manager.sync_sources(reter, project_root)
                            synced = True
                except Exception:
                    pass

                result = rag_manager.find_similar_clusters(
                    n_clusters=n_clusters,
                    min_cluster_size=min_cluster_size,
                    exclude_same_file=exclude_same_file,
                    exclude_same_class=exclude_same_class,
                    entity_types=entity_types,
                    source_type=source_type,
                )

                if synced:
                    result["auto_synced"] = True

                return result
            except ComponentNotReadyError as e:
                return e.to_response(clusters=[])
            except DefaultInstanceNotInitialised as e:
                return {
                    "success": False,
                    "error": str(e),
                    "status": "initializing",
                    "clusters": [],
                }
            except Exception as e:
                return {
                    "success": False,
                    "error": str(e),
                    "clusters": [],
                }

    def _register_find_duplicate_candidates(self, app: FastMCP) -> None:
        """Register the find_duplicate_candidates tool."""
        registrar = self

        @app.tool()
        @truncate_mcp_response
        def find_duplicate_candidates(
            similarity_threshold: float = 0.85,
            max_results: int = 50,
            exclude_same_file: bool = True,
            exclude_same_class: bool = True,
            entity_types: Optional[List[str]] = None
        ) -> Dict[str, Any]:
            """
            Find pairs of code entities that are highly similar (potential duplicates).

            Uses pairwise similarity search to find code that may be duplicated
            across different files or classes. More precise than clustering for
            finding exact duplicates.

            Args:
                similarity_threshold: Minimum similarity (0-1) to consider (default: 0.85)
                max_results: Maximum pairs to return (default: 50)
                exclude_same_file: Exclude pairs from same file (default: True)
                exclude_same_class: Exclude pairs from same class (default: True)
                entity_types: Filter by type: ["method", "function"] (default: all)

            Returns:
                {
                    "success": True,
                    "total_pairs": 12,
                    "pairs": [
                        {
                            "similarity": 0.95,
                            "entity1": {
                                "name": "find_large_classes",
                                "file": "advanced_python_tools.py",
                                "line": 59,
                                "entity_type": "method"
                            },
                            "entity2": {
                                "name": "find_large_classes",
                                "file": "code_quality.py",
                                "line": 15,
                                "entity_type": "method"
                            }
                        }
                    ],
                    "time_ms": 150
                }
            """
            # Duplicate detection requires RAG code index to be ready
            try:
                require_rag_code_index()
            except ComponentNotReadyError as e:
                return e.to_response(pairs=[])

            rag_manager = registrar._get_rag_manager()

            if rag_manager is None:
                return {
                    "success": False,
                    "error": "RAG is not configured. Set RETER_PROJECT_ROOT to enable.",
                    "pairs": [],
                }

            try:
                # Check sync status and auto-sync if stale
                synced = False
                try:
                    reter = registrar._get_reter("default")
                    sync_status = rag_manager.get_sync_status(reter)
                    if not sync_status.get("is_synced", True):
                        project_root = registrar._default_manager.project_root
                        if project_root:
                            rag_manager.sync_sources(reter, project_root)
                            synced = True
                except Exception:
                    pass

                result = rag_manager.find_duplicate_candidates(
                    similarity_threshold=similarity_threshold,
                    max_results=max_results,
                    exclude_same_file=exclude_same_file,
                    exclude_same_class=exclude_same_class,
                    entity_types=entity_types,
                )

                if synced:
                    result["auto_synced"] = True

                return result
            except ComponentNotReadyError as e:
                return e.to_response(pairs=[])
            except DefaultInstanceNotInitialised as e:
                return {
                    "success": False,
                    "error": str(e),
                    "status": "initializing",
                    "pairs": [],
                }
            except Exception as e:
                return {
                    "success": False,
                    "error": str(e),
                    "pairs": [],
                }

        @app.tool()
        @truncate_mcp_response
        def analyze_documentation_relevance(
            min_relevance: float = 0.5,
            max_results: int = 100,
            doc_types: Optional[List[str]] = None,
            code_types: Optional[List[str]] = None
        ) -> Dict[str, Any]:
            """
            Analyze how relevant documentation is to actual code.

            Uses semantic similarity to match documentation chunks (sections, code blocks)
            to code entities (classes, methods, functions).

            This helps detect:
            - Documentation that doesn't match any code (orphaned/outdated docs)
            - Documentation closely related to specific code entities
            - Overall documentation coverage quality

            Args:
                min_relevance: Minimum similarity score to consider "relevant" (0-1, default: 0.5)
                max_results: Maximum documentation chunks to analyze (default: 100)
                doc_types: Documentation types to analyze (default: ["section", "code_block", "document"])
                code_types: Code types to match against (default: ["class", "method", "function"])

            Returns:
                {
                    "success": True,
                    "relevant_docs": [...],  # Docs matching code (similarity >= min_relevance)
                    "orphaned_docs": [...],  # Docs not matching code (potentially outdated)
                    "stats": {
                        "total_docs_analyzed": int,
                        "relevant_count": int,
                        "orphaned_count": int,
                        "relevance_rate": float,  # % of docs that are relevant
                        "avg_similarity": float
                    }
                }

            Examples:
                - analyze_documentation_relevance()  # Default analysis
                - analyze_documentation_relevance(min_relevance=0.7)  # Stricter threshold
                - analyze_documentation_relevance(doc_types=["section"])  # Only sections
            """
            # Documentation relevance analysis requires both code and docs indexes
            try:
                require_rag_code_index()
                require_rag_document_index()
            except ComponentNotReadyError as e:
                return e.to_response(relevant_docs=[], orphaned_docs=[])

            rag_manager = registrar._get_rag_manager()

            if rag_manager is None:
                return {
                    "success": False,
                    "error": "RAG is not configured. Set RETER_PROJECT_ROOT to enable.",
                    "relevant_docs": [],
                    "orphaned_docs": [],
                }

            try:
                # Check sync status and auto-sync if stale
                synced = False
                try:
                    reter = registrar._get_reter("default")
                    sync_status = rag_manager.get_sync_status(reter)
                    if not sync_status.get("is_synced", True):
                        project_root = registrar._default_manager.project_root
                        if project_root:
                            rag_manager.sync_sources(reter, project_root)
                            synced = True
                except Exception:
                    pass

                result = rag_manager.analyze_documentation_relevance(
                    min_relevance=min_relevance,
                    max_results=max_results,
                    doc_entity_types=doc_types,
                    code_entity_types=code_types,
                )

                if synced:
                    result["auto_synced"] = True

                return result
            except ComponentNotReadyError as e:
                return e.to_response(relevant_docs=[], orphaned_docs=[])
            except DefaultInstanceNotInitialised as e:
                return {
                    "success": False,
                    "error": str(e),
                    "status": "initializing",
                    "relevant_docs": [],
                    "orphaned_docs": [],
                }
            except Exception as e:
                import traceback
                return {
                    "success": False,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                    "relevant_docs": [],
                    "orphaned_docs": [],
                }
