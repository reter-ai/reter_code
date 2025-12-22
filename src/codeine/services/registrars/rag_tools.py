"""
RAG Tools Registrar

Registers the RAG (Retrieval-Augmented Generation) MCP tools:
- semantic_search: Semantic search over code and documentation
- rag_status: Get RAG index status and statistics
- rag_reindex: Trigger RAG index rebuild
- init_status: Get initialization and sync status
"""

from typing import Dict, Any, Optional, List
from fastmcp import FastMCP
from .base import ToolRegistrarBase, truncate_mcp_response
from ..initialization_progress import (
    get_instance_progress,
    InstanceNotReadyError,
    get_initializing_response,
    get_component_readiness,
    ComponentNotReadyError,
    require_rag_code_index,
    require_rag_document_index,
    require_default_instance,
)
from ...reter_wrapper import DefaultInstanceNotInitialised, is_initialization_complete


class RAGToolsRegistrar(ToolRegistrarBase):
    """Registers RAG tools with FastMCP."""

    def __init__(self, instance_manager, persistence_service, default_manager=None):
        super().__init__(instance_manager, persistence_service)
        self._default_manager = default_manager

    def register(self, app: FastMCP) -> None:
        """Register all RAG tools."""
        self._register_init_status(app)  # Always available, even during initialization
        self._register_semantic_search(app)
        self._register_rag_status(app)
        self._register_rag_reindex(app)
        # Moved to recommender tool:
        # - find_similar_clusters, find_duplicate_candidates -> recommender("redundancy_reduction")
        # - analyze_documentation_relevance -> recommender("documentation_maintenance", "relevance_analysis")

    def _get_rag_manager(self):
        """Get the RAG manager from the default instance manager."""
        if self._default_manager is None:
            return None
        return self._default_manager.get_rag_manager()

    def _register_init_status(self, app: FastMCP) -> None:
        """Register the init_status tool."""

        @app.tool()
        def init_status() -> Dict[str, Any]:
            """
            Get initialization and sync status of the default RETER instance.

            Call this to check if the server is ready to handle queries.
            This tool is ALWAYS available, even during initialization.

            Returns status for both:
            - Initial startup (init_*): Loading files for the first time
            - File sync (sync_*): Processing file changes during session

            The instance is ready when:
            - is_ready == True

            Example response when initializing:
            {
                "is_ready": false,
                "blocking_reason": {
                    "reason": "initializing",
                    "phase": "loading_python",
                    "progress": 0.35,
                    "message": "Loaded 45/128 Python files",
                    "elapsed_seconds": 12
                },
                "init": { "status": "initializing", "phase": "loading_python", ... },
                "sync": { "status": "idle", ... }
            }

            Example response when ready:
            {
                "is_ready": true,
                "blocking_reason": null,
                "init": { "status": "ready", "phase": "complete", ... },
                "sync": { "status": "idle", ... }
            }

            Example response when syncing files:
            {
                "is_ready": false,
                "blocking_reason": {
                    "reason": "syncing",
                    "phase": "loading",
                    "files_changed": 12,
                    "files_processed": 5,
                    "elapsed_seconds": 3
                },
                "init": { "status": "ready", ... },
                "sync": { "status": "syncing", "phase": "loading", ... }
            }
            """
            # Get component readiness status
            components = get_component_readiness()
            component_status = components.get_status()

            # Check if ALL components are ready (for is_ready flag)
            is_ready = components.is_fully_ready()

            # Build blocking reason based on which components are not ready
            blocking = None
            if not is_ready:
                # Find first non-ready component
                if not components.sql_ready:
                    blocking = {
                        "reason": "component_not_ready",
                        "component": "sql",
                        "message": "SQLite is not initialized yet.",
                    }
                elif not components.reter_ready:
                    blocking = {
                        "reason": "component_not_ready",
                        "component": "reter",
                        "message": "Default RETER instance is still loading Python files.",
                    }
                elif not components.rag_code_ready:
                    blocking = {
                        "reason": "component_not_ready",
                        "component": "rag_code",
                        "message": "RAG code index is still building.",
                    }
                elif not components.rag_docs_ready:
                    blocking = {
                        "reason": "component_not_ready",
                        "component": "rag_docs",
                        "message": "RAG document index is still indexing.",
                    }
                else:
                    blocking = {
                        "reason": "initializing",
                        "message": "Server is still initializing.",
                    }

            # Also get progress info for additional context
            progress = get_instance_progress()

            return {
                "is_ready": is_ready,
                "blocking_reason": blocking,
                "components": component_status,
                "init": progress.get_init_snapshot(),
                "sync": progress.get_sync_snapshot(),
            }

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
            search_scope: str = "all",
            instance_name: str = "default"
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
                instance_name: RETER instance name (default: "default")

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
                    reter = registrar._get_reter(instance_name)
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

    def _register_rag_status(self, app: FastMCP) -> None:
        """Register the rag_status tool."""
        registrar = self

        @app.tool()
        def rag_status(
            instance_name: str = "default"
        ) -> Dict[str, Any]:
            """
            Get RAG index status and statistics.

            Returns information about the RAG index including:
            - Index status (ready, not_initialized, disabled)
            - Embedding model and provider
            - Total vectors indexed
            - Breakdown by source type (Python vs Markdown)
            - Entity type distribution
            - Cache information

            Args:
                instance_name: RETER instance name (default: "default")

            Returns:
                {
                    "success": True,
                    "status": "ready",
                    "embedding_model": "sentence-transformers/all-mpnet-base-v2",
                    "embedding_provider": "local",
                    "embedding_dimension": 768,
                    "total_vectors": 1823,
                    "index_size_mb": 2.4,
                    "created_at": "2024-01-15T10:30:00Z",
                    "updated_at": "2024-01-15T14:45:00Z",
                    "python_sources": {
                        "files_indexed": 45,
                        "total_vectors": 1523
                    },
                    "markdown_sources": {
                        "files_indexed": 12,
                        "total_vectors": 300
                    },
                    "entity_counts": {
                        "class": 89,
                        "method": 1203,
                        "function": 231,
                        "section": 89,
                        "document": 3
                    },
                    "cache_info": {
                        "cache_size": 234,
                        "max_cache_size": 1000
                    }
                }
            """
            # rag_status doesn't require full initialization - just show current status
            # This allows users to check status even during initialization
            rag_manager = registrar._get_rag_manager()

            if rag_manager is None:
                return {
                    "success": False,
                    "status": "not_configured",
                    "error": "RAG is not configured. Set RETER_PROJECT_ROOT to enable.",
                    "components": get_component_readiness().get_status(),
                }

            try:
                status = rag_manager.get_status()
                return {
                    "success": True,
                    "components": get_component_readiness().get_status(),
                    **status,
                }
            except ComponentNotReadyError as e:
                return e.to_response()
            except DefaultInstanceNotInitialised as e:
                return {
                    "success": False,
                    "error": str(e),
                    "status": "initializing",
                    "components": get_component_readiness().get_status(),
                }
            except Exception as e:
                return {
                    "success": False,
                    "status": "error",
                    "error": str(e),
                    "components": get_component_readiness().get_status(),
                }

    def _register_rag_reindex(self, app: FastMCP) -> None:
        """Register the rag_reindex tool."""
        import asyncio
        registrar = self

        @app.tool()
        async def rag_reindex(
            force: bool = False,
            instance_name: str = "default"
        ) -> Dict[str, Any]:
            """
            Trigger RAG index rebuild.

            Normally not needed as the index syncs automatically with file changes.
            Use force=True to completely rebuild the index from scratch.

            Args:
                force: Force complete rebuild (default: False for incremental)
                instance_name: RETER instance name (default: "default")

            Returns:
                {
                    "success": True,
                    "python_sources": 45,
                    "python_vectors": 1523,
                    "markdown_files": 12,
                    "markdown_vectors": 300,
                    "total_vectors": 1823,
                    "time_ms": 4500,
                    "errors": []
                }
            """
            # Reindexing requires RETER to be ready (to query Python entities)
            try:
                require_default_instance()
            except ComponentNotReadyError as e:
                return e.to_response()

            rag_manager = registrar._get_rag_manager()

            if rag_manager is None:
                return {
                    "success": False,
                    "error": "RAG is not configured. Set RETER_PROJECT_ROOT to enable.",
                }

            if not rag_manager.is_enabled:
                return {
                    "success": False,
                    "error": "RAG is disabled via configuration.",
                }

            try:
                # Get RETER instance for the queries
                reter = registrar._get_reter(instance_name)
                project_root = registrar._default_manager.project_root

                if project_root is None:
                    return {
                        "success": False,
                        "error": "Project root is not set.",
                    }

                # Run blocking reindex in thread pool to avoid asyncio deadlock
                stats = await asyncio.to_thread(rag_manager.reindex_all, reter, project_root)
                return {
                    "success": True,
                    **stats,
                }
            except ComponentNotReadyError as e:
                return e.to_response()
            except DefaultInstanceNotInitialised as e:
                return {
                    "success": False,
                    "error": str(e),
                    "status": "initializing",
                }
            except Exception as e:
                return {
                    "success": False,
                    "error": str(e),
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
            source_type: Optional[str] = None,
            instance_name: str = "default"
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
                instance_name: RETER instance name (default: "default")

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
                    reter = registrar._get_reter(instance_name)
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
            entity_types: Optional[List[str]] = None,
            instance_name: str = "default"
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
                instance_name: RETER instance name (default: "default")

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
                    reter = registrar._get_reter(instance_name)
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
            code_types: Optional[List[str]] = None,
            instance_name: str = "default"
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
                instance_name: RETER instance name (default: "default")

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
                    reter = registrar._get_reter(instance_name)
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
