"""
RAG Index Manager Service

Manages the FAISS RAG index for semantic code and documentation search.
Tightly integrated with DefaultInstanceManager for automatic synchronization
when files change.
"""

import hashlib
import json
import logging
import time
from datetime import datetime, UTC
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Callable, Set, TYPE_CHECKING

import numpy as np

# Import data classes from rag_types (re-export for backward compatibility)
from .rag_types import LanguageSourceChanges, SyncChanges, RAGSearchResult

# Re-export for backward compatibility
__all__ = [
    "LanguageSourceChanges",
    "SyncChanges",
    "RAGSearchResult",
    "RAGIndexManager",
    "ChunkConfig",
]

from .faiss_wrapper import FAISSWrapper, SearchResult, FAISS_AVAILABLE
from .embedding_service import EmbeddingService, get_embedding_service
from .content_extractor import ContentExtractor, CodeEntity, CodeChunk
from .markdown_indexer import MarkdownIndexer, MarkdownChunk
from .initialization_progress import (
    require_default_instance,
    require_rag_code_index,
    require_rag_document_index,
    ComponentNotReadyError,
)

if TYPE_CHECKING:
    from ..reter_wrapper import ReterWrapper
    from .state_persistence import StatePersistenceService

from ..logging_config import configure_logger_for_debug_trace
from .rag_analysis import RAGAnalysisMixin
from .rag_collectors import RAGCollectorMixin, ChunkConfig

logger = configure_logger_for_debug_trace(__name__)


class RAGIndexManager(RAGAnalysisMixin, RAGCollectorMixin):
    """
    Manages the FAISS RAG index for semantic code search.

    Tightly integrated with DefaultInstanceManager for automatic
    synchronization when files change.

    Attributes:
        embedding_model: Name of the embedding model being used
        embedding_dim: Dimension of embeddings
        is_initialized: Whether the index is loaded and ready

    ::: This is-in-layer Service-Layer.
    ::: This is a retrieval-augmented-generation-component.
    ::: This depends-on `reter_code.services.EmbeddingService`.
    ::: This depends-on `reter_code.services.FAISSWrapper`.
    ::: This depends-on `reter_code.services.ContentExtractor`.
    ::: This is-in-process Main-Process.
    ::: This is stateful.
    ::: This holds-expensive-resource "faiss-index".
    ::: This has-startup-order = 3.
    ::: This has-singleton-scope.
    """

    def __init__(
        self,
        persistence: "StatePersistenceService",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the RAG index manager.

        Args:
            persistence: StatePersistenceService for snapshot management
            config: Configuration dictionary (from reter.json)
        """
        self._persistence = persistence
        self._config = config or {}

        # Check if RAG is enabled
        self._enabled = self._config.get("rag_enabled", True)
        if not self._enabled:
            logger.info("RAG indexing is disabled via configuration")
            return

        # Check FAISS availability
        if not FAISS_AVAILABLE:
            logger.warning(
                "FAISS not available, RAG functionality disabled. "
                "Install with: pip install faiss-cpu"
            )
            self._enabled = False
            return

        # Initialize components
        self._embedding_service: Optional[EmbeddingService] = None
        self._faiss_wrapper: Optional[FAISSWrapper] = None
        self._content_extractor: Optional[ContentExtractor] = None
        self._markdown_indexer: Optional[MarkdownIndexer] = None

        # Index paths
        self._index_path: Optional[Path] = None
        self._metadata_path: Optional[Path] = None
        self._rag_files_path: Optional[Path] = None

        # Indexed files tracking: {rel_path: md5_hash}
        # Stored separately in .default.rag_files.json for fast loading
        self._indexed_files: Dict[str, str] = {}

        # Metadata tracking
        self._metadata: Dict[str, Any] = {
            "version": "1.0",
            "embedding_model": None,
            "embedding_dimension": None,
            "created_at": None,
            "updated_at": None,
            "total_vectors": 0,
            "sources": {},  # source_id -> {md5, file_type, indexed_at, vector_ids}
            "vectors": {},  # vector_id -> entity metadata
        }

        self._next_vector_id = 0
        self._initialized = False
        self._project_root: Optional[Path] = None

        # Initialize chunk config from settings
        self._chunk_config: Optional[ChunkConfig] = None

    @property
    def is_enabled(self) -> bool:
        """Check if RAG is enabled."""
        return self._enabled

    @property
    def is_initialized(self) -> bool:
        """Check if index is loaded and ready."""
        return self._initialized and self._faiss_wrapper is not None

    def set_preloaded_model(self, model) -> None:
        """
        Set a pre-loaded embedding model to avoid loading delay during first use.

        The model will be passed to the EmbeddingService when initialize() is called.

        Args:
            model: Pre-loaded SentenceTransformer model instance
        """
        self._preloaded_model = model
        self._model_loaded = True

    @property
    def is_model_loaded(self) -> bool:
        """Check if the embedding model has been loaded (may be loading in background)."""
        return getattr(self, '_model_loaded', False)

    @property
    def embedding_model(self) -> Optional[str]:
        """Get the embedding model name."""
        if self._embedding_service:
            return self._embedding_service.model_name
        return None

    @property
    def embedding_dim(self) -> Optional[int]:
        """Get the embedding dimension."""
        if self._embedding_service:
            return self._embedding_service.embedding_dim
        return None

    def get_sync_status(self, reter: "ReterWrapper") -> Dict[str, Any]:
        """
        Check if RAG index is synced with current RETER sources.

        This is a lightweight check that compares MD5 hashes without re-indexing.
        Use this to detect stale search results.

        Args:
            reter: RETER wrapper instance

        Returns:
            Dict with:
                - is_synced: bool - True if index matches current sources
                - stale_files: List of files that have changed
                - missing_files: List of files in RETER but not indexed
                - extra_files: List of indexed files no longer in RETER
                - warnings: List of human-readable warnings
        """
        if not self._enabled or not self._initialized:
            return {
                "is_synced": True,
                "stale_files": [],
                "missing_files": [],
                "extra_files": [],
                "warnings": []
            }

        try:
            # Get current sources from RETER (all supported code file types)
            all_sources, _ = reter.get_all_sources()
            current_code: Dict[str, str] = {}  # rel_path -> md5
            # Must match extensions from sync_sources: JS_EXTENSIONS, HTML_EXTENSIONS, etc.
            code_extensions = (
                ".py",  # Python
                ".js", ".ts", ".jsx", ".tsx", ".mjs",  # JavaScript/TypeScript
                ".html", ".htm",  # HTML
                ".cs",  # C#
                ".cpp", ".cc", ".cxx", ".hpp", ".h",  # C++
            )
            for source_id in all_sources:
                if "|" in source_id:
                    md5_hash, rel_path = source_id.split("|", 1)
                    if rel_path.endswith(code_extensions):
                        rel_path_normalized = rel_path.replace("\\", "/")
                        current_code[rel_path_normalized] = md5_hash

            # Get indexed code files
            # Must strip prefixes like js:, html:, cs:, cpp: to match RETER source paths
            indexed_code: Dict[str, str] = {}  # rel_path -> md5
            for key, md5 in self._indexed_files.items():
                if key.startswith("md:"):
                    continue  # Skip markdown
                # Strip file type prefixes
                clean_key = key
                for prefix in ("js:", "html:", "cs:", "cpp:"):
                    if key.startswith(prefix):
                        clean_key = key[len(prefix):]
                        break
                indexed_code[clean_key.replace("\\", "/")] = md5

            logger.debug(f"[RAG] get_sync_status: current_code has {len(current_code)} files, indexed_code has {len(indexed_code)} files")
            logger.debug(f"[RAG] get_sync_status: _indexed_files sample keys: {list(self._indexed_files.keys())[:5]}")

            stale_files = []
            missing_files = []
            extra_files = []
            warnings = []

            # Check for stale/missing files
            for rel_path, current_md5 in current_code.items():
                indexed_md5 = indexed_code.get(rel_path)
                if indexed_md5 is None:
                    missing_files.append(rel_path)
                elif current_md5 != indexed_md5:
                    stale_files.append(rel_path)

            # Check for extra indexed files (deleted from RETER)
            for rel_path in indexed_code:
                if rel_path not in current_code:
                    extra_files.append(rel_path)

            # Build warnings
            if stale_files:
                warnings.append(f"{len(stale_files)} file(s) have changed since last index")
            if missing_files:
                warnings.append(f"{len(missing_files)} file(s) not yet indexed")
            if extra_files:
                warnings.append(f"{len(extra_files)} indexed file(s) no longer exist")

            is_synced = len(stale_files) == 0 and len(missing_files) == 0 and len(extra_files) == 0

            return {
                "is_synced": is_synced,
                "stale_files": stale_files[:10],  # Limit to first 10
                "missing_files": missing_files[:10],
                "extra_files": extra_files[:10],
                "warnings": warnings
            }

        except Exception as e:
            logger.debug(f"[RAG] get_sync_status: Error checking sync status: {e}")
            return {
                "is_synced": True,  # Assume synced on error to not block searches
                "stale_files": [],
                "missing_files": [],
                "extra_files": [],
                "warnings": [f"Could not check sync status: {str(e)}"]
            }

    def initialize(self, project_root: Path) -> None:
        """
        Initialize or load existing FAISS index.

        Args:
            project_root: Project root directory
        """
        if not self._enabled:
            logger.debug("[RAG] initialize: RAG is disabled, skipping")
            return

        # Check if embedding model is loaded (may be loading in background)
        if not self.is_model_loaded:
            logger.debug("[RAG] initialize: Embedding model not yet loaded, skipping initialization")
            return

        self._project_root = project_root
        logger.debug(f"[RAG] initialize: Starting initialization for {project_root}")
        logger.info(f"Initializing RAG index for {project_root}")

        # Initialize embedding service
        # If a pre-loaded model was set on RAGIndexManager, pass it to embedding service
        logger.info("[RAG] initialize: Creating embedding service...")
        embed_start = time.time()
        self._embedding_service = get_embedding_service(self._config)

        # Check if we have a pre-loaded model (set by server at startup to avoid async deadlock)
        if hasattr(self, '_preloaded_model') and self._preloaded_model is not None:
            logger.info("[RAG] initialize: Using pre-loaded model")
            self._embedding_service.set_preloaded_model(self._preloaded_model)

        logger.info(f"[RAG] initialize: Loading embedding model '{self._embedding_service.model_name}'...")
        self._embedding_service.initialize()
        logger.info(f"[RAG] initialize: Embedding model loaded in {time.time() - embed_start:.2f}s")

        # Initialize content extractor
        logger.debug("[RAG] initialize: Creating content extractor...")
        max_body_lines = self._config.get("rag_max_body_lines", 50)
        self._content_extractor = ContentExtractor(
            project_root=project_root,
            max_body_lines=max_body_lines
        )

        # Initialize markdown indexer
        logger.debug("[RAG] initialize: Creating markdown indexer...")
        max_chunk_words = self._config.get("rag_markdown_max_chunk_words", 500)
        min_chunk_words = self._config.get("rag_markdown_min_chunk_words", 50)
        self._markdown_indexer = MarkdownIndexer(
            max_chunk_words=max_chunk_words,
            min_chunk_words=min_chunk_words,
            include_code_blocks=True
        )

        # Initialize code chunk config - use config loader for defaults
        from .config_loader import get_config_loader
        rag_config = get_config_loader().get_rag_config()
        chunk_enabled = rag_config.get("rag_code_chunk_enabled", True)
        if chunk_enabled:
            self._chunk_config = ChunkConfig(
                enabled=True,
                chunk_size=rag_config.get("rag_code_chunk_size", 30),
                chunk_overlap=rag_config.get("rag_code_chunk_overlap", 10),
                min_chunk_size=rag_config.get("rag_code_chunk_min_size", 15)
            )
            logger.debug(f"[RAG] initialize: Code chunking enabled (size={self._chunk_config.chunk_size}, overlap={self._chunk_config.chunk_overlap})")
        else:
            self._chunk_config = None
            logger.debug("[RAG] initialize: Code chunking disabled")

        # Set up index paths
        reter_dir = self._persistence.snapshots_dir
        reter_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = reter_dir / ".default.faiss"
        self._metadata_path = reter_dir / ".default.faiss.meta"
        self._rag_files_path = reter_dir / ".default.rag_files.json"
        logger.debug(f"[RAG] initialize: Index path: {self._index_path}")

        # Try to load existing index
        if self._index_path.exists() and self._metadata_path.exists():
            logger.debug("[RAG] initialize: Found existing index, loading...")
            try:
                self._load_index()
                logger.debug(f"[RAG] initialize: Loaded existing index with {self._faiss_wrapper.total_vectors} vectors")
            except Exception as e:
                logger.debug(f"[RAG] initialize: Failed to load existing index: {e}. Will rebuild.")
                logger.warning(f"Failed to load existing index: {e}. Will rebuild.")
                self._create_new_index()
        else:
            logger.debug("[RAG] initialize: No existing index found, creating new...")
            self._create_new_index()

        self._initialized = True
        logger.debug(
            f"[RAG] initialize: READY - {self._faiss_wrapper.total_vectors} vectors, "
            f"model={self._embedding_service.model_name}"
        )
        logger.info(
            f"RAG index ready: {self._faiss_wrapper.total_vectors} vectors, "
            f"model={self._embedding_service.model_name}"
        )

    def _create_new_index(self) -> None:
        """Create a new empty FAISS index."""
        self._faiss_wrapper = FAISSWrapper(
            dimension=self._embedding_service.embedding_dim,
            index_type="flat",  # Use flat for smaller corpora
            metric="ip"  # Inner product (cosine after normalization)
        )
        self._faiss_wrapper.create_index()

        self._metadata = {
            "version": "1.0",
            "embedding_model": self._embedding_service.model_name,
            "embedding_provider": self._embedding_service.provider,
            "embedding_dimension": self._embedding_service.embedding_dim,
            "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "updated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "total_vectors": 0,
            "sources": {},
            "vectors": {},
        }
        self._next_vector_id = 0

        logger.info("Created new FAISS index")

    def _load_index(self) -> None:
        """Load existing FAISS index and metadata."""
        # Load metadata first
        with open(self._metadata_path, 'r', encoding='utf-8') as f:
            self._metadata = json.load(f)

        # Check model compatibility
        saved_model = self._metadata.get("embedding_model")
        current_model = self._embedding_service.model_name
        if saved_model != current_model:
            logger.warning(
                f"Embedding model changed from {saved_model} to {current_model}. "
                f"Index will be rebuilt."
            )
            self._create_new_index()
            return

        # Load FAISS index
        self._faiss_wrapper = FAISSWrapper(
            dimension=self._metadata.get("embedding_dimension", 768),
            index_type="flat",
            metric="ip"
        )
        self._faiss_wrapper.load(str(self._index_path))

        # Set next vector ID
        if self._metadata.get("vectors"):
            max_id = max(int(k) for k in self._metadata["vectors"].keys())
            self._next_vector_id = max_id + 1
        else:
            self._next_vector_id = 0

        logger.info(
            f"Loaded FAISS index: {self._faiss_wrapper.total_vectors} vectors, "
            f"next_id={self._next_vector_id}"
        )

    def _save_index(self) -> None:
        """Save FAISS index and metadata to disk."""
        if not self._initialized or not self._faiss_wrapper:
            return

        # Update metadata
        self._metadata["updated_at"] = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        self._metadata["total_vectors"] = self._faiss_wrapper.total_vectors

        # Save FAISS index
        self._faiss_wrapper.save(str(self._index_path))

        # Save metadata
        with open(self._metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self._metadata, f, indent=2)

        logger.debug(f"Saved RAG index: {self._faiss_wrapper.total_vectors} vectors")

    def _load_rag_files(self) -> Dict[str, str]:
        """
        Load indexed files tracking from .default.rag_files.json.

        Returns:
            Dict mapping rel_path -> md5_hash for all indexed files
            Keys may have prefixes: js:, html:, cs:, cpp:, md:
        """
        if not self._rag_files_path or not self._rag_files_path.exists():
            return {}

        try:
            with open(self._rag_files_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Format: {"python": {...}, "javascript": {...}, "html": {...}, etc.}
                result = {}
                # Python files have no prefix
                result.update(data.get("python", {}))
                # JavaScript files get js: prefix
                result.update({f"js:{k}": v for k, v in data.get("javascript", {}).items()})
                # HTML files get html: prefix
                result.update({f"html:{k}": v for k, v in data.get("html", {}).items()})
                # C# files get cs: prefix
                result.update({f"cs:{k}": v for k, v in data.get("csharp", {}).items()})
                # C++ files get cpp: prefix
                result.update({f"cpp:{k}": v for k, v in data.get("cpp", {}).items()})
                # Markdown files get md: prefix
                result.update({f"md:{k}": v for k, v in data.get("markdown", {}).items()})

                logger.debug(f"[RAG] _load_rag_files: Loaded {len(result)} indexed files "
                         f"(py={len(data.get('python', {}))}, js={len(data.get('javascript', {}))}, "
                         f"html={len(data.get('html', {}))}, cs={len(data.get('csharp', {}))}, "
                         f"cpp={len(data.get('cpp', {}))}, md={len(data.get('markdown', {}))})")
                return result
        except Exception as e:
            logger.debug(f"[RAG] _load_rag_files: Error loading: {e}")
            return {}

    def _save_rag_files(self) -> None:
        """
        Save indexed files tracking to .default.rag_files.json.
        """
        if not self._rag_files_path:
            return

        # Split by type for cleaner JSON
        python_files = {}
        javascript_files = {}
        html_files = {}
        csharp_files = {}
        cpp_files = {}
        markdown_files = {}

        for key, md5 in self._indexed_files.items():
            if key.startswith("md:"):
                markdown_files[key[3:]] = md5  # Remove "md:" prefix
            elif key.startswith("js:"):
                javascript_files[key[3:]] = md5  # Remove "js:" prefix
            elif key.startswith("html:"):
                html_files[key[5:]] = md5  # Remove "html:" prefix
            elif key.startswith("cs:"):
                csharp_files[key[3:]] = md5  # Remove "cs:" prefix
            elif key.startswith("cpp:"):
                cpp_files[key[4:]] = md5  # Remove "cpp:" prefix
            else:
                python_files[key] = md5  # Python has no prefix

        data = {
            "version": "1.1",
            "updated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "python": python_files,
            "javascript": javascript_files,
            "html": html_files,
            "csharp": csharp_files,
            "cpp": cpp_files,
            "markdown": markdown_files,
        }

        logger.debug(f"[RAG] _save_rag_files: Saving py={len(python_files)}, js={len(javascript_files)}, "
                 f"html={len(html_files)}, cs={len(csharp_files)}, cpp={len(cpp_files)}, md={len(markdown_files)}")

        try:
            with open(self._rag_files_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            logger.debug(f"[RAG] _save_rag_files: Saved {len(self._indexed_files)} indexed files")
        except Exception as e:
            logger.debug(f"[RAG] _save_rag_files: Error saving: {e}")

    def sync(
        self,
        reter: "ReterWrapper",
        changed_python_sources: List[str],
        deleted_python_sources: List[str],
        project_root: Path,
        changed_markdown_files: Optional[List[str]] = None,
        deleted_markdown_files: Optional[List[str]] = None,
        changed_javascript_sources: Optional[List[str]] = None,
        deleted_javascript_sources: Optional[List[str]] = None,
        changed_html_sources: Optional[List[str]] = None,
        deleted_html_sources: Optional[List[str]] = None,
        changed_csharp_sources: Optional[List[str]] = None,
        deleted_csharp_sources: Optional[List[str]] = None,
        changed_cpp_sources: Optional[List[str]] = None,
        deleted_cpp_sources: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Synchronize FAISS index after file changes (backward compatible signature).

        This method preserves the original 15-parameter signature for backward
        compatibility. New code should use sync_with_changes() with SyncChanges.

        Args:
            reter: RETER instance for code entity queries
            changed_python_sources: Python source IDs added/modified
            deleted_python_sources: Python source IDs deleted
            project_root: Project root for resolving paths
            changed_markdown_files: Markdown files added/modified (rel paths)
            deleted_markdown_files: Markdown files deleted
            changed_javascript_sources: JavaScript source IDs added/modified
            deleted_javascript_sources: JavaScript source IDs deleted
            changed_html_sources: HTML source IDs added/modified
            deleted_html_sources: HTML source IDs deleted
            changed_csharp_sources: C# source IDs added/modified
            deleted_csharp_sources: C# source IDs deleted
            changed_cpp_sources: C++ source IDs added/modified
            deleted_cpp_sources: C++ source IDs deleted

        Returns:
            Sync statistics
        """
        changes = SyncChanges.from_params(
            changed_python_sources=changed_python_sources,
            deleted_python_sources=deleted_python_sources,
            changed_javascript_sources=changed_javascript_sources,
            deleted_javascript_sources=deleted_javascript_sources,
            changed_html_sources=changed_html_sources,
            deleted_html_sources=deleted_html_sources,
            changed_csharp_sources=changed_csharp_sources,
            deleted_csharp_sources=deleted_csharp_sources,
            changed_cpp_sources=changed_cpp_sources,
            deleted_cpp_sources=deleted_cpp_sources,
            changed_markdown_files=changed_markdown_files,
            deleted_markdown_files=deleted_markdown_files,
        )
        return self.sync_with_changes(reter, project_root, changes)

    def sync_with_changes(
        self,
        reter: "ReterWrapper",
        project_root: Path,
        changes: SyncChanges,
    ) -> Dict[str, Any]:
        """
        Synchronize FAISS index after file changes.

        Called by DefaultInstanceManager after syncing source files.

        Args:
            reter: RETER instance for code entity queries
            project_root: Project root for resolving paths
            changes: SyncChanges object containing all language change lists

        Returns:
            Sync statistics

        Example:
            changes = SyncChanges(
                python=LanguageSourceChanges(changed=["mod1.py"], deleted=[]),
                markdown=LanguageSourceChanges(changed=["README.md"], deleted=[]),
            )
            stats = manager.sync_with_changes(reter, project_root, changes)
        """
        if not self._enabled or not self._initialized:
            return {"status": "disabled"}

        # Check if embedding model is loaded
        if not self.is_model_loaded:
            logger.debug("[RAG] sync: Embedding model not yet loaded, skipping sync")
            return {"status": "model_loading", "error": "Embedding model still loading in background"}

        start_time = time.time()
        stats = {
            "python_vectors_added": 0,
            "python_vectors_removed": 0,
            "javascript_vectors_added": 0,
            "javascript_vectors_removed": 0,
            "html_vectors_added": 0,
            "html_vectors_removed": 0,
            "csharp_vectors_added": 0,
            "csharp_vectors_removed": 0,
            "cpp_vectors_added": 0,
            "cpp_vectors_removed": 0,
            "markdown_vectors_added": 0,
            "markdown_vectors_removed": 0,
            "errors": [],
        }

        # Update content extractor project root
        self._content_extractor.project_root = project_root

        # 1. Remove vectors for deleted sources
        for source_id in changes.python.deleted:
            removed = self._remove_vectors_for_source(source_id)
            stats["python_vectors_removed"] += removed

        for source_id in changes.javascript.deleted:
            removed = self._remove_vectors_for_source(source_id)
            stats["javascript_vectors_removed"] += removed

        for source_id in changes.html.deleted:
            removed = self._remove_vectors_for_source(source_id)
            stats["html_vectors_removed"] += removed

        for source_id in changes.csharp.deleted:
            removed = self._remove_vectors_for_source(source_id)
            stats["csharp_vectors_removed"] += removed

        for source_id in changes.cpp.deleted:
            removed = self._remove_vectors_for_source(source_id)
            stats["cpp_vectors_removed"] += removed

        for rel_path in changes.markdown.deleted:
            removed = self._remove_vectors_for_markdown(rel_path)
            stats["markdown_vectors_removed"] += removed

        # 2. Re-remove vectors for changed sources (they'll be re-indexed)
        for source_id in changes.python.changed:
            self._remove_vectors_for_source(source_id)

        for source_id in changes.javascript.changed:
            self._remove_vectors_for_source(source_id)

        for source_id in changes.html.changed:
            self._remove_vectors_for_source(source_id)

        for source_id in changes.csharp.changed:
            self._remove_vectors_for_source(source_id)

        for source_id in changes.cpp.changed:
            self._remove_vectors_for_source(source_id)

        for rel_path in changes.markdown.changed:
            self._remove_vectors_for_markdown(rel_path)

        # 3. BATCHED INDEXING: Collect all texts first, then generate embeddings in one batch
        all_texts = []
        all_metadata = []
        source_tracking = []  # (source_id, file_type, start_idx, count)

        # 3a. Collect Python entities and comments
        for source_id in changes.python.changed:
            try:
                entities = self._query_entities_for_source(reter, source_id, language="python")
                texts, metadata = self._collect_python_entities(entities, source_id, project_root, self._chunk_config)
                if texts:
                    source_tracking.append((source_id, "python", len(all_texts), len(texts)))
                    all_texts.extend(texts)
                    all_metadata.extend(metadata)
                # Also collect comments
                comment_texts, comment_metadata = self._collect_python_comments(entities, source_id, project_root)
                if comment_texts:
                    source_tracking.append((f"comments:{source_id}", "python_comment", len(all_texts), len(comment_texts)))
                    all_texts.extend(comment_texts)
                    all_metadata.extend(comment_metadata)
            except Exception as e:
                logger.error(f"Error collecting Python {source_id}: {e}")
                stats["errors"].append(f"Python: {source_id}: {e}")

        # 3b. Collect Python literals (only from changed sources)
        if changes.python.changed:
            literal_texts, literal_metadata = self._collect_all_python_literals_bulk(
                reter, project_root, changed_sources=changes.python.changed
            )
            if literal_texts:
                source_tracking.append(("python_literals_bulk", "python_literal", len(all_texts), len(literal_texts)))
                all_texts.extend(literal_texts)
                all_metadata.extend(literal_metadata)

        # 3c. Collect JavaScript entities
        for source_id in changes.javascript.changed:
            try:
                entities = self._query_entities_for_source(reter, source_id, language="javascript")
                texts, metadata = self._collect_javascript_entities(entities, source_id, project_root, self._chunk_config)
                if texts:
                    source_tracking.append((source_id, "javascript", len(all_texts), len(texts)))
                    all_texts.extend(texts)
                    all_metadata.extend(metadata)
            except Exception as e:
                logger.error(f"Error collecting JavaScript {source_id}: {e}")
                stats["errors"].append(f"JavaScript: {source_id}: {e}")

        # 3d. Collect JavaScript literals (bulk query - filtered by changed sources)
        if changes.javascript.changed:
            js_literal_texts, js_literal_metadata = self._collect_all_javascript_literals_bulk(
                reter, project_root, changed_sources=changes.javascript.changed
            )
            if js_literal_texts:
                source_tracking.append(("javascript_literals_bulk", "javascript_literal", len(all_texts), len(js_literal_texts)))
                all_texts.extend(js_literal_texts)
                all_metadata.extend(js_literal_metadata)

        # 3e. Collect HTML entities
        for source_id in changes.html.changed:
            try:
                entities = self._query_html_entities_for_source(reter, source_id)
                texts, metadata = self._collect_html_entities(entities, source_id, project_root)
                if texts:
                    source_tracking.append((source_id, "html", len(all_texts), len(texts)))
                    all_texts.extend(texts)
                    all_metadata.extend(metadata)
            except Exception as e:
                logger.error(f"Error collecting HTML {source_id}: {e}")
                stats["errors"].append(f"HTML: {source_id}: {e}")

        # 3f. Collect C# entities (uses similar pattern to Python)
        for source_id in changes.csharp.changed:
            try:
                entities = self._query_entities_for_source(reter, source_id, language="csharp")
                texts, metadata = self._collect_csharp_entities(entities, source_id, project_root, self._chunk_config)
                if texts:
                    source_tracking.append((source_id, "csharp", len(all_texts), len(texts)))
                    all_texts.extend(texts)
                    all_metadata.extend(metadata)
            except Exception as e:
                logger.error(f"Error collecting C# {source_id}: {e}")
                stats["errors"].append(f"C#: {source_id}: {e}")

        # 3g. Collect C++ entities (uses similar pattern to C#)
        for source_id in changes.cpp.changed:
            try:
                entities = self._query_entities_for_source(reter, source_id, language="cpp")
                texts, metadata = self._collect_cpp_entities(entities, source_id, project_root, self._chunk_config)
                if texts:
                    source_tracking.append((source_id, "cpp", len(all_texts), len(texts)))
                    all_texts.extend(texts)
                    all_metadata.extend(metadata)
            except Exception as e:
                logger.error(f"Error collecting C++ {source_id}: {e}")
                stats["errors"].append(f"C++: {source_id}: {e}")

        # 3h. Collect Markdown chunks
        for rel_path in changes.markdown.changed:
            try:
                abs_path = project_root / rel_path
                chunks = self._markdown_indexer.parse_file(str(abs_path))
                texts, metadata = self._collect_markdown_chunks(chunks, rel_path)
                if texts:
                    source_tracking.append((rel_path, "markdown", len(all_texts), len(texts)))
                    all_texts.extend(texts)
                    all_metadata.extend(metadata)
            except Exception as e:
                logger.error(f"Error collecting markdown {rel_path}: {e}")
                stats["errors"].append(f"Markdown: {rel_path}: {e}")

        # 4. Generate embeddings for ALL texts in one batch
        if all_texts:
            logger.debug(f"[RAG] sync_with_sources: Generating embeddings for {len(all_texts)} texts in one batch")
            batch_size = self._config.get("rag_batch_size", 32)
            embeddings = self._embedding_service.generate_embeddings_batch(all_texts, batch_size=batch_size)

            # Assign vector IDs
            vector_ids = list(range(self._next_vector_id, self._next_vector_id + len(all_texts)))
            self._next_vector_id += len(all_texts)

            # Add to FAISS
            ids_array = np.array(vector_ids, dtype=np.int64)
            self._faiss_wrapper.add_vectors(embeddings, ids_array)

            # Update metadata for all vectors
            for vid, meta in zip(vector_ids, all_metadata):
                self._metadata["vectors"][str(vid)] = meta

            # Track sources
            for source_id, file_type, start_idx, count in source_tracking:
                source_vector_ids = vector_ids[start_idx:start_idx + count]
                if "|" in source_id:
                    md5_hash, rel_path = source_id.split("|", 1)
                else:
                    md5_hash = ""
                    rel_path = source_id

                self._metadata["sources"][source_id] = {
                    "file_type": file_type,
                    "md5": md5_hash,
                    "rel_path": rel_path,
                    "indexed_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                    "vector_ids": source_vector_ids,
                }

                # Update _indexed_files to keep sync status accurate
                # Use prefixes matching sync_sources() convention
                if file_type in ("python", "python_comment", "python_literal") and md5_hash:
                    self._indexed_files[rel_path] = md5_hash
                elif file_type in ("javascript", "javascript_literal") and md5_hash:
                    self._indexed_files[f"js:{rel_path}"] = md5_hash
                elif file_type == "html" and md5_hash:
                    self._indexed_files[f"html:{rel_path}"] = md5_hash
                elif file_type == "csharp" and md5_hash:
                    self._indexed_files[f"cs:{rel_path}"] = md5_hash
                elif file_type == "cpp" and md5_hash:
                    self._indexed_files[f"cpp:{rel_path}"] = md5_hash
                elif file_type == "markdown":
                    # Markdown uses rel_path as key with md: prefix
                    self._indexed_files[f"md:{rel_path}"] = md5_hash

                # Update stats
                if file_type in ("python", "python_comment", "python_literal"):
                    stats["python_vectors_added"] += count
                elif file_type in ("javascript", "javascript_literal"):
                    stats["javascript_vectors_added"] += count
                elif file_type == "html":
                    stats["html_vectors_added"] += count
                elif file_type == "csharp":
                    stats["csharp_vectors_added"] += count
                elif file_type == "cpp":
                    stats["cpp_vectors_added"] += count
                elif file_type == "markdown":
                    stats["markdown_vectors_added"] += count

        # 5. Save index and metadata
        self._save_index()
        self._save_rag_files()  # Also save _indexed_files to keep sync status accurate

        stats["time_ms"] = int((time.time() - start_time) * 1000)
        total_removed = (
            stats['python_vectors_removed'] +
            stats['javascript_vectors_removed'] +
            stats['html_vectors_removed'] +
            stats['csharp_vectors_removed'] +
            stats['cpp_vectors_removed'] +
            stats['markdown_vectors_removed']
        )
        logger.info(
            f"RAG sync: +{stats['python_vectors_added']} Python, "
            f"+{stats['javascript_vectors_added']} JavaScript, "
            f"+{stats['html_vectors_added']} HTML, "
            f"+{stats['csharp_vectors_added']} C#, "
            f"+{stats['cpp_vectors_added']} C++, "
            f"+{stats['markdown_vectors_added']} Markdown, "
            f"-{total_removed} removed "
            f"in {stats['time_ms']}ms"
        )

        return stats

    def sync_sources(
        self,
        reter: "ReterWrapper",
        project_root: Path,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        changed_files: Optional[Set[str]] = None
    ) -> Dict[str, Any]:
        """
        Sync RAG index with current file state using MD5 comparison.

        Key optimization: Only loads the embedding model if files need indexing.
        If no files changed, the model is never loaded.

        Flow:
        1. Set up paths and load .rag_files.json (fast, no model)
        2. Compare MD5 hashes to find changes
        3. If no changes, return immediately (no model loading!)
        4. If changes, initialize fully (load model) and process

        Args:
            reter: RETER instance for Python entity queries
            project_root: Project root for resolving paths
            progress_callback: Optional callback(current, total, phase)
            changed_files: Optional set of rel_paths that changed (targeted sync).
                          If provided, only these files are checked for changes.
                          Files not in this set are assumed unchanged.

        Returns:
            Sync statistics
        """
        if not self._enabled:
            logger.debug("[RAG] sync_sources: RAG is disabled")
            return {"status": "disabled"}

        # Check if embedding model is loaded (may be loading in background)
        if not self.is_model_loaded:
            logger.debug("[RAG] sync_sources: Embedding model not yet loaded, skipping sync")
            return {"status": "model_loading", "error": "Embedding model still loading in background"}

        start_time = time.time()
        self._project_root = project_root

        # Set up paths (no model loading yet)
        reter_dir = self._persistence.snapshots_dir
        reter_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = reter_dir / ".default.faiss"
        self._metadata_path = reter_dir / ".default.faiss.meta"
        self._rag_files_path = reter_dir / ".default.rag_files.json"

        logger.debug(f"[RAG] sync_sources: Starting sync for {project_root}")

        # Load indexed files from JSON (fast, no model needed)
        self._indexed_files = self._load_rag_files()
        # Count by type for debugging
        py_count = sum(1 for k in self._indexed_files if not k.startswith(("js:", "html:", "cs:", "cpp:", "md:")))
        js_count = sum(1 for k in self._indexed_files if k.startswith("js:"))
        html_count = sum(1 for k in self._indexed_files if k.startswith("html:"))
        cs_count = sum(1 for k in self._indexed_files if k.startswith("cs:"))
        cpp_count = sum(1 for k in self._indexed_files if k.startswith("cpp:"))
        md_count = sum(1 for k in self._indexed_files if k.startswith("md:"))
        logger.debug(f"[RAG] sync_sources: Loaded {len(self._indexed_files)} indexed files from JSON "
                 f"(py={py_count}, js={js_count}, html={html_count}, cs={cs_count}, cpp={cpp_count}, md={md_count})")

        stats = {
            "python_added": 0,
            "python_removed": 0,
            "python_unchanged": 0,
            "javascript_added": 0,
            "javascript_removed": 0,
            "javascript_unchanged": 0,
            "html_added": 0,
            "html_removed": 0,
            "html_unchanged": 0,
            "csharp_added": 0,
            "csharp_removed": 0,
            "csharp_unchanged": 0,
            "cpp_added": 0,
            "cpp_removed": 0,
            "cpp_unchanged": 0,
            "markdown_added": 0,
            "markdown_removed": 0,
            "markdown_unchanged": 0,
            "errors": [],
        }

        # File extension patterns
        JS_EXTENSIONS = ('.js', '.ts', '.jsx', '.tsx', '.mjs')
        HTML_EXTENSIONS = ('.html', '.htm')
        CSHARP_EXTENSIONS = ('.cs',)
        CPP_EXTENSIONS = ('.cpp', '.cc', '.cxx', '.hpp', '.h')

        # Get current sources from RETER (already loaded by BackgroundInitializer)
        all_sources, _ = reter.get_all_sources()
        current_python: Dict[str, Tuple[str, str]] = {}  # rel_path -> (source_id, md5)
        current_javascript: Dict[str, Tuple[str, str]] = {}  # rel_path -> (source_id, md5)
        current_html: Dict[str, Tuple[str, str]] = {}  # rel_path -> (source_id, md5)
        current_csharp: Dict[str, Tuple[str, str]] = {}  # rel_path -> (source_id, md5)
        current_cpp: Dict[str, Tuple[str, str]] = {}  # rel_path -> (source_id, md5)

        # Helper to compute actual file MD5 (more reliable than RETER snapshot MD5s)
        def compute_file_md5(abs_path: Path) -> Optional[str]:
            try:
                content = abs_path.read_bytes()
                return hashlib.md5(content).hexdigest()
            except Exception:
                return None

        # If targeted sync, log the count
        if changed_files:
            logger.debug(f"[RAG] sync_sources: Targeted sync for {len(changed_files)} changed files")
            if not progress_callback:
                logger.info(f"[RAG] Targeted sync for {len(changed_files)} changed files")

        for source_id in all_sources:
            if "|" in source_id:
                reter_md5, rel_path = source_id.split("|", 1)
                rel_path_normalized = rel_path.replace("\\", "/")
                rel_path_lower = rel_path.lower()

                # OPTIMIZATION: If targeted sync and file not in changed_files,
                # use cached MD5 instead of computing (assume unchanged)
                if changed_files and rel_path_normalized not in changed_files:
                    # Use cached MD5 if available
                    if rel_path_lower.endswith(".py"):
                        cached_md5 = self._indexed_files.get(rel_path_normalized)
                        if cached_md5:
                            current_python[rel_path_normalized] = (source_id, cached_md5)
                    elif rel_path_lower.endswith(JS_EXTENSIONS):
                        cached_md5 = self._indexed_files.get(f"js:{rel_path_normalized}")
                        if cached_md5:
                            current_javascript[rel_path_normalized] = (source_id, cached_md5)
                    elif rel_path_lower.endswith(HTML_EXTENSIONS):
                        cached_md5 = self._indexed_files.get(f"html:{rel_path_normalized}")
                        if cached_md5:
                            current_html[rel_path_normalized] = (source_id, cached_md5)
                    elif rel_path_lower.endswith(CSHARP_EXTENSIONS):
                        cached_md5 = self._indexed_files.get(f"cs:{rel_path_normalized}")
                        if cached_md5:
                            current_csharp[rel_path_normalized] = (source_id, cached_md5)
                    elif rel_path_lower.endswith(CPP_EXTENSIONS):
                        cached_md5 = self._indexed_files.get(f"cpp:{rel_path_normalized}")
                        if cached_md5:
                            current_cpp[rel_path_normalized] = (source_id, cached_md5)
                    continue

                # Compute actual MD5 for this file
                if rel_path_lower.endswith(".py"):
                    # Use actual file MD5 (RETER's MD5 is from parsed content, not stable across restarts)
                    actual_md5 = compute_file_md5(project_root / rel_path_normalized)
                    if actual_md5:
                        current_python[rel_path_normalized] = (source_id, actual_md5)
                elif rel_path_lower.endswith(JS_EXTENSIONS):
                    # For non-Python files, compute actual file MD5 to avoid stale snapshot MD5s
                    actual_md5 = compute_file_md5(project_root / rel_path_normalized)
                    if actual_md5:
                        current_javascript[rel_path_normalized] = (source_id, actual_md5)
                elif rel_path_lower.endswith(HTML_EXTENSIONS):
                    actual_md5 = compute_file_md5(project_root / rel_path_normalized)
                    if actual_md5:
                        current_html[rel_path_normalized] = (source_id, actual_md5)
                elif rel_path_lower.endswith(CSHARP_EXTENSIONS):
                    actual_md5 = compute_file_md5(project_root / rel_path_normalized)
                    if actual_md5:
                        current_csharp[rel_path_normalized] = (source_id, actual_md5)
                elif rel_path_lower.endswith(CPP_EXTENSIONS):
                    actual_md5 = compute_file_md5(project_root / rel_path_normalized)
                    if actual_md5:
                        current_cpp[rel_path_normalized] = (source_id, actual_md5)

        # Get indexed files from _indexed_files (using prefixes: py:, js:, html:, cs:, cpp:, md:)
        indexed_python: Dict[str, str] = {}  # rel_path -> md5
        indexed_javascript: Dict[str, str] = {}  # rel_path -> md5
        indexed_html: Dict[str, str] = {}  # rel_path -> md5
        indexed_csharp: Dict[str, str] = {}  # rel_path -> md5
        indexed_cpp: Dict[str, str] = {}  # rel_path -> md5
        for key, md5 in self._indexed_files.items():
            if key.startswith("js:"):
                indexed_javascript[key[3:]] = md5
            elif key.startswith("html:"):
                indexed_html[key[5:]] = md5
            elif key.startswith("cs:"):
                indexed_csharp[key[3:]] = md5
            elif key.startswith("cpp:"):
                indexed_cpp[key[4:]] = md5
            elif key.startswith("md:"):
                pass  # Handled separately below
            else:
                # Legacy: Python files without prefix
                indexed_python[key] = md5

        logger.debug(f"[RAG] sync_sources: Python - {len(current_python)} current, {len(indexed_python)} indexed")

        # Find Python changes
        python_to_add: List[Tuple[str, str, str]] = []  # (source_id, rel_path, md5) to add
        python_to_remove: List[str] = []  # rel_paths to remove from _indexed_files

        for rel_path, (source_id, current_md5) in current_python.items():
            indexed_md5 = indexed_python.get(rel_path)
            if indexed_md5 is None:
                # New file
                logger.debug(f"[RAG] sync_sources: Python NEW: {rel_path}")
                python_to_add.append((source_id, rel_path, current_md5))
            elif current_md5 != indexed_md5:
                # Modified file
                logger.debug(f"[RAG] sync_sources: Python MODIFIED: {rel_path} (indexed={indexed_md5[:8]}... vs current={current_md5[:8]}...)")
                python_to_remove.append(rel_path)
                python_to_add.append((source_id, rel_path, current_md5))
            else:
                stats["python_unchanged"] += 1

        for rel_path in indexed_python:
            if rel_path not in current_python:
                # Deleted file
                python_to_remove.append(rel_path)

        logger.debug(f"[RAG] sync_sources: Python changes - +{len(python_to_add)} -{len(python_to_remove)} ={stats['python_unchanged']}")

        # Find JavaScript changes
        logger.debug(f"[RAG] sync_sources: JavaScript - {len(current_javascript)} current, {len(indexed_javascript)} indexed")
        javascript_to_add: List[Tuple[str, str, str]] = []  # (source_id, rel_path, md5) to add
        javascript_to_remove: List[str] = []  # rel_paths to remove from _indexed_files

        for rel_path, (source_id, current_md5) in current_javascript.items():
            indexed_md5 = indexed_javascript.get(rel_path)
            if indexed_md5 is None:
                # New file
                javascript_to_add.append((source_id, rel_path, current_md5))
            elif current_md5 != indexed_md5:
                # Modified file
                javascript_to_remove.append(rel_path)
                javascript_to_add.append((source_id, rel_path, current_md5))
            else:
                stats["javascript_unchanged"] += 1

        for rel_path in indexed_javascript:
            if rel_path not in current_javascript:
                # Deleted file
                javascript_to_remove.append(rel_path)

        logger.debug(f"[RAG] sync_sources: JavaScript changes - +{len(javascript_to_add)} -{len(javascript_to_remove)} ={stats['javascript_unchanged']}")

        # Find HTML changes
        logger.debug(f"[RAG] sync_sources: HTML - {len(current_html)} current, {len(indexed_html)} indexed")
        html_to_add: List[Tuple[str, str, str]] = []  # (source_id, rel_path, md5) to add
        html_to_remove: List[str] = []  # rel_paths to remove from _indexed_files

        for rel_path, (source_id, current_md5) in current_html.items():
            indexed_md5 = indexed_html.get(rel_path)
            if indexed_md5 is None:
                # New file
                html_to_add.append((source_id, rel_path, current_md5))
            elif current_md5 != indexed_md5:
                # Modified file
                html_to_remove.append(rel_path)
                html_to_add.append((source_id, rel_path, current_md5))
            else:
                stats["html_unchanged"] += 1

        for rel_path in indexed_html:
            if rel_path not in current_html:
                # Deleted file
                html_to_remove.append(rel_path)

        logger.debug(f"[RAG] sync_sources: HTML changes - +{len(html_to_add)} -{len(html_to_remove)} ={stats['html_unchanged']}")

        # Find C# changes
        logger.debug(f"[RAG] sync_sources: C# - {len(current_csharp)} current, {len(indexed_csharp)} indexed")
        csharp_to_add: List[Tuple[str, str, str]] = []  # (source_id, rel_path, md5) to add
        csharp_to_remove: List[str] = []  # rel_paths to remove from _indexed_files

        for rel_path, (source_id, current_md5) in current_csharp.items():
            indexed_md5 = indexed_csharp.get(rel_path)
            if indexed_md5 is None:
                # New file
                csharp_to_add.append((source_id, rel_path, current_md5))
            elif current_md5 != indexed_md5:
                # Modified file
                csharp_to_remove.append(rel_path)
                csharp_to_add.append((source_id, rel_path, current_md5))
            else:
                stats["csharp_unchanged"] += 1

        for rel_path in indexed_csharp:
            if rel_path not in current_csharp:
                # Deleted file
                csharp_to_remove.append(rel_path)

        logger.debug(f"[RAG] sync_sources: C# changes - +{len(csharp_to_add)} -{len(csharp_to_remove)} ={stats['csharp_unchanged']}")

        # Find C++ changes
        logger.debug(f"[RAG] sync_sources: C++ - {len(current_cpp)} current, {len(indexed_cpp)} indexed")
        cpp_to_add: List[Tuple[str, str, str]] = []  # (source_id, rel_path, md5) to add
        cpp_to_remove: List[str] = []  # rel_paths to remove from _indexed_files

        for rel_path, (source_id, current_md5) in current_cpp.items():
            indexed_md5 = indexed_cpp.get(rel_path)
            if indexed_md5 is None:
                # New file
                cpp_to_add.append((source_id, rel_path, current_md5))
            elif current_md5 != indexed_md5:
                # Modified file
                cpp_to_remove.append(rel_path)
                cpp_to_add.append((source_id, rel_path, current_md5))
            else:
                stats["cpp_unchanged"] += 1

        for rel_path in indexed_cpp:
            if rel_path not in current_cpp:
                # Deleted file
                cpp_to_remove.append(rel_path)

        logger.debug(f"[RAG] sync_sources: C++ changes - +{len(cpp_to_add)} -{len(cpp_to_remove)} ={stats['cpp_unchanged']}")

        # Get current Markdown files and compute MD5s
        current_markdown: Dict[str, str] = {}  # rel_path -> md5
        md_files = self._scan_markdown_files(project_root)
        for rel_path in md_files:
            rel_path_normalized = rel_path.replace("\\", "/")

            # OPTIMIZATION: If targeted sync and file not in changed_files, use cached MD5
            if changed_files and rel_path_normalized not in changed_files:
                cached_md5 = self._indexed_files.get(f"md:{rel_path_normalized}")
                if cached_md5:
                    current_markdown[rel_path_normalized] = cached_md5
                continue

            try:
                abs_path = project_root / rel_path
                content = abs_path.read_text(encoding='utf-8')
                md5_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
                current_markdown[rel_path_normalized] = md5_hash
            except Exception as e:
                logger.debug(f"[RAG] sync_sources: Error reading {rel_path}: {e}")

        # Get indexed Markdown from _indexed_files
        indexed_markdown: Dict[str, str] = {}  # rel_path -> md5
        for key, md5 in self._indexed_files.items():
            if key.startswith("md:"):
                indexed_markdown[key[3:]] = md5  # Remove "md:" prefix

        logger.debug(f"[RAG] sync_sources: Markdown - {len(current_markdown)} current, {len(indexed_markdown)} indexed")

        # Find Markdown changes
        markdown_to_add: List[Tuple[str, str]] = []  # (rel_path, md5) to add
        markdown_to_remove: List[str] = []  # rel_paths to remove

        for rel_path, current_md5 in current_markdown.items():
            indexed_md5 = indexed_markdown.get(rel_path)
            if indexed_md5 is None:
                # New file
                markdown_to_add.append((rel_path, current_md5))
            elif current_md5 != indexed_md5:
                # Modified file
                markdown_to_remove.append(rel_path)
                markdown_to_add.append((rel_path, current_md5))
            else:
                stats["markdown_unchanged"] += 1

        for rel_path in indexed_markdown:
            if rel_path not in current_markdown:
                # Deleted file
                markdown_to_remove.append(rel_path)

        logger.debug(f"[RAG] sync_sources: Markdown changes - +{len(markdown_to_add)} -{len(markdown_to_remove)} ={stats['markdown_unchanged']}")

        # Check if any changes
        total_changes = (
            len(python_to_add) + len(python_to_remove) +
            len(javascript_to_add) + len(javascript_to_remove) +
            len(html_to_add) + len(html_to_remove) +
            len(csharp_to_add) + len(csharp_to_remove) +
            len(cpp_to_add) + len(cpp_to_remove) +
            len(markdown_to_add) + len(markdown_to_remove)
        )

        logger.debug(f"[RAG] sync_sources: {total_changes} changes detected")

        # Estimate total files to index for progress reporting
        total_files_to_index = (
            len(python_to_add) + len(javascript_to_add) + len(html_to_add) +
            len(csharp_to_add) + len(cpp_to_add) + len(markdown_to_add)
        )
        if progress_callback:
            progress_callback(0, total_files_to_index, "initializing")
        else:
            logger.info(f"[RAG] sync_sources: {total_changes} changes detected, {total_files_to_index} files to index")

        # ALWAYS initialize fully (loads embedding model + FAISS index)
        self.initialize(project_root)

        # If no changes, just return with existing index
        if total_changes == 0:
            stats["time_ms"] = int((time.time() - start_time) * 1000)
            stats["total_vectors"] = self._faiss_wrapper.total_vectors
            logger.debug(f"[RAG] sync_sources: COMPLETE (no changes) - {stats['total_vectors']} vectors in {stats['time_ms']}ms")
            return stats
        logger.debug(f"[RAG] sync_sources: Initialized with {self._faiss_wrapper.total_vectors} existing vectors")

        # Remove vectors for deleted/modified Python files
        for rel_path in python_to_remove:
            # Find the source_id in metadata that matches this rel_path
            for source_id, info in list(self._metadata["sources"].items()):
                if info.get("file_type") == "python" and info.get("rel_path", "").replace("\\", "/") == rel_path:
                    removed = self._remove_vectors_for_source(source_id)
                    stats["python_removed"] += removed
                    break
            # Remove from indexed files
            self._indexed_files.pop(rel_path, None)

        # Remove vectors for deleted/modified JavaScript files
        for rel_path in javascript_to_remove:
            # Find the source_id in metadata that matches this rel_path
            for source_id, info in list(self._metadata["sources"].items()):
                if info.get("file_type") == "javascript" and info.get("rel_path", "").replace("\\", "/") == rel_path:
                    removed = self._remove_vectors_for_source(source_id)
                    stats["javascript_removed"] += removed
                    break
            # Remove from indexed files
            self._indexed_files.pop(f"js:{rel_path}", None)

        # Remove vectors for deleted/modified HTML files
        for rel_path in html_to_remove:
            # Find the source_id in metadata that matches this rel_path
            for source_id, info in list(self._metadata["sources"].items()):
                if info.get("file_type") == "html" and info.get("rel_path", "").replace("\\", "/") == rel_path:
                    removed = self._remove_vectors_for_source(source_id)
                    stats["html_removed"] += removed
                    break
            # Remove from indexed files
            self._indexed_files.pop(f"html:{rel_path}", None)

        # Remove vectors for deleted/modified C# files
        for rel_path in csharp_to_remove:
            # Find the source_id in metadata that matches this rel_path
            for source_id, info in list(self._metadata["sources"].items()):
                if info.get("file_type") == "csharp" and info.get("rel_path", "").replace("\\", "/") == rel_path:
                    removed = self._remove_vectors_for_source(source_id)
                    stats["csharp_removed"] += removed
                    break
            # Remove from indexed files
            self._indexed_files.pop(f"cs:{rel_path}", None)

        # Remove vectors for deleted/modified C++ files
        for rel_path in cpp_to_remove:
            # Find the source_id in metadata that matches this rel_path
            for source_id, info in list(self._metadata["sources"].items()):
                if info.get("file_type") == "cpp" and info.get("rel_path", "").replace("\\", "/") == rel_path:
                    removed = self._remove_vectors_for_source(source_id)
                    stats["cpp_removed"] += removed
                    break
            # Remove from indexed files
            self._indexed_files.pop(f"cpp:{rel_path}", None)

        # Remove vectors for deleted/modified Markdown files
        for rel_path in markdown_to_remove:
            removed = self._remove_vectors_for_markdown(rel_path)
            stats["markdown_removed"] += removed
            # Remove from indexed files
            self._indexed_files.pop(f"md:{rel_path}", None)

        # =============================================================================
        # UNIFIED BATCHED INDEXING: Collect ALL texts first, then embed in ONE batch
        # =============================================================================
        all_texts: List[str] = []
        all_metadata: List[Dict[str, Any]] = []
        source_tracking: List[Tuple[str, str, int, int]] = []  # (source_id, file_type, start_idx, count)

        # --- Phase 1: Collect Python entities, comments, and literals ---
        if python_to_add:
            logger.debug(f"[RAG] sync_sources: Querying entities for {len(python_to_add)} Python files...")
            entities_by_file = self._query_all_entities_bulk(reter)
            total_python = len(python_to_add)

            for i, (source_id, rel_path, md5_hash) in enumerate(python_to_add):
                try:
                    logger.debug(f"[RAG] sync_sources: Collecting Python [{i+1}/{total_python}]: {rel_path}")
                    # Convert rel_path to is-in-file format (keep extension)
                    in_file = rel_path.replace("\\", "/")
                    entities = entities_by_file.get(in_file, [])
                    if entities:
                        # Collect entity texts
                        texts, metadata = self._prepare_python_entities(entities, source_id, project_root)
                        if texts:
                            source_tracking.append((source_id, "python", len(all_texts), len(texts)))
                            all_texts.extend(texts)
                            all_metadata.extend(metadata)

                        # Collect comment texts
                        c_texts, c_meta = self._prepare_python_comments(entities, source_id, project_root)
                        if c_texts:
                            source_tracking.append((f"comments:{source_id}", "python_comment", len(all_texts), len(c_texts)))
                            all_texts.extend(c_texts)
                            all_metadata.extend(c_meta)

                        logger.debug(f"[RAG] sync_sources: Collected {len(texts)} entities + {len(c_texts)} comments for {rel_path}")

                    # Update indexed files tracking
                    self._indexed_files[rel_path] = md5_hash

                    if progress_callback and (i + 1) % 10 == 0:
                        progress_callback(i + 1, total_python, "collecting_python")
                    elif not progress_callback and (i + 1) % 10 == 0:
                        logger.info(f"[RAG] Collecting Python: {i+1}/{total_python}")

                except Exception as e:
                    import traceback
                    logger.debug(f"[RAG] sync_sources: Error collecting {source_id}: {e}")
                    logger.debug(f"[RAG] sync_sources: Traceback: {traceback.format_exc()}")
                    stats["errors"].append(f"Python: {source_id}: {e}")

            # Collect Python literals (only from files being added/modified)
            logger.debug("[RAG] sync_sources: Collecting Python string literals...")
            # Extract source_ids from python_to_add for filtering
            changed_source_ids = [source_id for source_id, _, _ in python_to_add]
            literal_texts, literal_metadata = self._collect_all_python_literals_bulk(
                reter, project_root, changed_sources=changed_source_ids
            )
            if literal_texts:
                source_tracking.append(("python_literals_bulk", "python_literal", len(all_texts), len(literal_texts)))
                all_texts.extend(literal_texts)
                all_metadata.extend(literal_metadata)
                logger.debug(f"[RAG] sync_sources: Collected {len(literal_texts)} Python literals")

        # --- Phase 2: Collect JavaScript entities and literals (bulk query) ---
        if javascript_to_add:
            logger.debug(f"[RAG] sync_sources: Querying entities for {len(javascript_to_add)} JavaScript files...")
            js_entities_by_file = self._query_all_entities_bulk(reter, language="javascript")
            total_javascript = len(javascript_to_add)

            for i, (source_id, rel_path, md5_hash) in enumerate(javascript_to_add):
                try:
                    logger.debug(f"[RAG] sync_sources: Collecting JavaScript [{i+1}/{total_javascript}]: {rel_path}")
                    # Convert rel_path to is-in-file format (JavaScript keeps extension)
                    in_file = rel_path.replace("\\", "/")
                    entities = js_entities_by_file.get(in_file, [])
                    if entities:
                        texts, metadata = self._collect_javascript_entities(entities, source_id, project_root, self._chunk_config)
                        if texts:
                            source_tracking.append((source_id, "javascript", len(all_texts), len(texts)))
                            all_texts.extend(texts)
                            all_metadata.extend(metadata)
                        logger.debug(f"[RAG] sync_sources: Collected {len(texts)} entities for {rel_path}")

                    # Update indexed files tracking
                    self._indexed_files[f"js:{rel_path}"] = md5_hash

                    if progress_callback and (i + 1) % 10 == 0:
                        progress_callback(i + 1, total_javascript, "collecting_javascript")
                    elif not progress_callback and (i + 1) % 10 == 0:
                        logger.info(f"[RAG] Collecting JavaScript: {i+1}/{total_javascript}")

                except Exception as e:
                    import traceback
                    logger.debug(f"[RAG] sync_sources: Error collecting JavaScript {source_id}: {e}")
                    logger.debug(f"[RAG] sync_sources: Traceback: {traceback.format_exc()}")
                    stats["errors"].append(f"JavaScript: {source_id}: {e}")

            # Collect JavaScript literals (bulk query)
            logger.debug("[RAG] sync_sources: Collecting JavaScript string literals...")
            js_literal_texts, js_literal_metadata = self._collect_all_javascript_literals_bulk(reter, project_root)
            if js_literal_texts:
                source_tracking.append(("javascript_literals_bulk", "javascript_literal", len(all_texts), len(js_literal_texts)))
                all_texts.extend(js_literal_texts)
                all_metadata.extend(js_literal_metadata)
                logger.debug(f"[RAG] sync_sources: Collected {len(js_literal_texts)} JavaScript literals")

        # --- Phase 3: Collect HTML entities (bulk query) ---
        if html_to_add:
            logger.debug(f"[RAG] sync_sources: Querying entities for {len(html_to_add)} HTML files...")
            html_entities_by_file = self._query_all_html_entities_bulk(reter)
            total_html = len(html_to_add)

            for i, (source_id, rel_path, md5_hash) in enumerate(html_to_add):
                try:
                    logger.debug(f"[RAG] sync_sources: Collecting HTML [{i+1}/{total_html}]: {rel_path}")
                    # Convert rel_path to is-in-document format (keeps extension)
                    in_doc = rel_path.replace("\\", ".").replace("/", ".")
                    entities = html_entities_by_file.get(in_doc, [])
                    if entities:
                        texts, metadata = self._collect_html_entities(entities, source_id, project_root)
                        if texts:
                            source_tracking.append((source_id, "html", len(all_texts), len(texts)))
                            all_texts.extend(texts)
                            all_metadata.extend(metadata)
                        logger.debug(f"[RAG] sync_sources: Collected {len(texts)} entities for {rel_path}")

                    # Update indexed files tracking
                    self._indexed_files[f"html:{rel_path}"] = md5_hash

                    if progress_callback and (i + 1) % 10 == 0:
                        progress_callback(i + 1, total_html, "collecting_html")
                    elif not progress_callback and (i + 1) % 10 == 0:
                        logger.info(f"[RAG] Collecting HTML: {i+1}/{total_html}")

                except Exception as e:
                    import traceback
                    logger.debug(f"[RAG] sync_sources: Error collecting HTML {source_id}: {e}")
                    logger.debug(f"[RAG] sync_sources: Traceback: {traceback.format_exc()}")
                    stats["errors"].append(f"HTML: {source_id}: {e}")

        # --- Phase 4a: Collect C# entities (bulk query) ---
        if csharp_to_add:
            logger.debug(f"[RAG] sync_sources: Querying entities for {len(csharp_to_add)} C# files...")
            csharp_entities_by_file = self._query_all_entities_bulk(reter, language="csharp")
            total_csharp = len(csharp_to_add)

            for i, (source_id, rel_path, md5_hash) in enumerate(csharp_to_add):
                try:
                    logger.debug(f"[RAG] sync_sources: Collecting C# [{i+1}/{total_csharp}]: {rel_path}")
                    # Convert rel_path to is-in-file format (keep extension)
                    in_file = rel_path.replace("\\", "/")
                    entities = csharp_entities_by_file.get(in_file, [])
                    if entities:
                        texts, metadata = self._collect_csharp_entities(entities, source_id, project_root, self._chunk_config)
                        if texts:
                            source_tracking.append((source_id, "csharp", len(all_texts), len(texts)))
                            all_texts.extend(texts)
                            all_metadata.extend(metadata)
                        logger.debug(f"[RAG] sync_sources: Collected {len(texts)} entities for {rel_path}")

                    # Update indexed files tracking
                    self._indexed_files[f"cs:{rel_path}"] = md5_hash

                    if progress_callback and (i + 1) % 10 == 0:
                        progress_callback(i + 1, total_csharp, "collecting_csharp")
                    elif not progress_callback and (i + 1) % 10 == 0:
                        logger.info(f"[RAG] Collecting C#: {i+1}/{total_csharp}")

                except Exception as e:
                    import traceback
                    logger.debug(f"[RAG] sync_sources: Error collecting C# {source_id}: {e}")
                    logger.debug(f"[RAG] sync_sources: Traceback: {traceback.format_exc()}")
                    stats["errors"].append(f"C#: {source_id}: {e}")

        # --- Phase 4b: Collect C++ entities (bulk query) ---
        if cpp_to_add:
            logger.debug(f"[RAG] sync_sources: Querying entities for {len(cpp_to_add)} C++ files...")
            cpp_entities_by_file = self._query_all_entities_bulk(reter, language="cpp")
            total_cpp = len(cpp_to_add)

            for i, (source_id, rel_path, md5_hash) in enumerate(cpp_to_add):
                try:
                    logger.debug(f"[RAG] sync_sources: Collecting C++ [{i+1}/{total_cpp}]: {rel_path}")
                    # Convert rel_path to is-in-file format (keep extension)
                    in_file = rel_path.replace("\\", "/")
                    entities = cpp_entities_by_file.get(in_file, [])
                    if entities:
                        texts, metadata = self._collect_cpp_entities(entities, source_id, project_root, self._chunk_config)
                        if texts:
                            source_tracking.append((source_id, "cpp", len(all_texts), len(texts)))
                            all_texts.extend(texts)
                            all_metadata.extend(metadata)
                        logger.debug(f"[RAG] sync_sources: Collected {len(texts)} entities for {rel_path}")

                    # Update indexed files tracking
                    self._indexed_files[f"cpp:{rel_path}"] = md5_hash

                    if progress_callback and (i + 1) % 10 == 0:
                        progress_callback(i + 1, total_cpp, "collecting_cpp")
                    elif not progress_callback and (i + 1) % 10 == 0:
                        logger.info(f"[RAG] Collecting C++: {i+1}/{total_cpp}")

                except Exception as e:
                    import traceback
                    logger.debug(f"[RAG] sync_sources: Error collecting C++ {source_id}: {e}")
                    logger.debug(f"[RAG] sync_sources: Traceback: {traceback.format_exc()}")
                    stats["errors"].append(f"C++: {source_id}: {e}")

        # --- Phase 5: Collect Markdown chunks ---
        if markdown_to_add:
            logger.debug(f"[RAG] sync_sources: Collecting {len(markdown_to_add)} Markdown files...")
            total_markdown = len(markdown_to_add)

            for i, (rel_path, md5_hash) in enumerate(markdown_to_add):
                try:
                    abs_path = project_root / rel_path
                    chunks = self._markdown_indexer.parse_file(str(abs_path))
                    texts, metadata = self._collect_markdown_chunks(chunks, rel_path)
                    if texts:
                        source_tracking.append((f"md:{rel_path}", "markdown", len(all_texts), len(texts)))
                        all_texts.extend(texts)
                        all_metadata.extend(metadata)

                    # Update indexed files tracking
                    self._indexed_files[f"md:{rel_path}"] = md5_hash

                    if progress_callback and (i + 1) % 10 == 0:
                        progress_callback(i + 1, total_markdown, "collecting_markdown")
                    elif not progress_callback and (i + 1) % 10 == 0:
                        logger.info(f"[RAG] Collecting Markdown: {i+1}/{total_markdown}")

                except Exception as e:
                    logger.debug(f"[RAG] sync_sources: Error collecting {rel_path}: {e}")
                    stats["errors"].append(f"Markdown: {rel_path}: {e}")

        # --- Phase 5: Generate embeddings for ALL texts in ONE batch ---
        if all_texts:
            logger.debug(f"[RAG] sync_sources: Generating embeddings for {len(all_texts)} texts in ONE batch...")
            if progress_callback:
                progress_callback(0, len(all_texts), "generating_embeddings")
            else:
                logger.info(f"[RAG] Generating embeddings for {len(all_texts)} texts...")

            batch_size = self._config.get("rag_batch_size", 32)
            last_console_percent = 0

            # Create embedding progress callback wrapper
            def embedding_progress(current: int, total: int):
                nonlocal last_console_percent
                if progress_callback:
                    progress_callback(current, total, "generating_embeddings")
                elif total > 0:
                    percent = (current * 100) // total
                    if percent >= last_console_percent + 10:  # Report every 10%
                        last_console_percent = percent
                        logger.info(f"[RAG] Generating embeddings: {percent}%")

            embeddings = self._embedding_service.generate_embeddings_batch(
                all_texts, batch_size=batch_size, progress_callback=embedding_progress
            )

            if progress_callback:
                progress_callback(len(all_texts), len(all_texts), "embeddings_complete")
            else:
                logger.info(f"[RAG] Embeddings complete: {len(all_texts)} vectors")

            # Assign vector IDs
            vector_ids = list(range(self._next_vector_id, self._next_vector_id + len(all_texts)))
            self._next_vector_id += len(all_texts)

            # Add to FAISS
            ids_array = np.array(vector_ids, dtype=np.int64)
            self._faiss_wrapper.add_vectors(embeddings, ids_array)

            # Update metadata for all vectors
            for vid, meta in zip(vector_ids, all_metadata):
                self._metadata["vectors"][str(vid)] = meta

            # Track sources and update stats
            for source_id, file_type, start_idx, count in source_tracking:
                source_vector_ids = vector_ids[start_idx:start_idx + count]

                # Extract md5 and rel_path from source_id if present
                if "|" in source_id:
                    md5_hash, rel_path = source_id.split("|", 1)
                else:
                    md5_hash = ""
                    rel_path = source_id

                self._metadata["sources"][source_id] = {
                    "file_type": file_type,
                    "md5": md5_hash,
                    "rel_path": rel_path,
                    "indexed_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                    "vector_ids": source_vector_ids,
                }

                # Update stats by type
                if file_type in ("python", "python_comment", "python_literal"):
                    stats["python_added"] += count
                elif file_type in ("javascript", "javascript_literal"):
                    stats["javascript_added"] += count
                elif file_type == "html":
                    stats["html_added"] += count
                elif file_type == "csharp":
                    stats["csharp_added"] += count
                elif file_type == "cpp":
                    stats["cpp_added"] += count
                elif file_type == "markdown":
                    stats["markdown_added"] += count

            logger.debug(f"[RAG] sync_sources: Added {len(all_texts)} vectors to index")

        # Save index and rag_files.json
        logger.debug("[RAG] sync_sources: Saving index and rag_files.json...")
        if not progress_callback:
            logger.info("[RAG] Saving index...")
        self._save_index()
        self._save_rag_files()

        stats["time_ms"] = int((time.time() - start_time) * 1000)
        stats["total_vectors"] = self._faiss_wrapper.total_vectors

        logger.debug(
            f"[RAG] sync_sources: COMPLETE - Python(+{stats['python_added']} -{stats['python_removed']} "
            f"={stats['python_unchanged']}) Markdown(+{stats['markdown_added']} -{stats['markdown_removed']} "
            f"={stats['markdown_unchanged']}) total={stats['total_vectors']} in {stats['time_ms']}ms"
        )
        logger.info(
            f"RAG sync: {stats['total_vectors']} vectors "
            f"(+{stats['python_added']+stats['markdown_added']} -{stats['python_removed']+stats['markdown_removed']} "
            f"unchanged={stats['python_unchanged']+stats['markdown_unchanged']}) in {stats['time_ms']}ms"
        )
        if not progress_callback:
            total_added = stats['python_added'] + stats['javascript_added'] + stats['html_added'] + stats['csharp_added'] + stats['cpp_added'] + stats['markdown_added']
            total_removed = stats['python_removed'] + stats['javascript_removed'] + stats['html_removed'] + stats['csharp_removed'] + stats['cpp_removed'] + stats['markdown_removed']
            logger.info(f"[RAG] Sync complete: {stats['total_vectors']} vectors (+{total_added} -{total_removed}) in {stats['time_ms']}ms")

        return stats

    def _remove_vectors_for_source(self, source_id: str) -> int:
        """Remove all vectors associated with a Python source."""
        if source_id not in self._metadata["sources"]:
            return 0

        source_info = self._metadata["sources"][source_id]
        vector_ids = source_info.get("vector_ids", [])

        if vector_ids:
            ids_array = np.array(vector_ids, dtype=np.int64)
            self._faiss_wrapper.remove_vectors(ids_array)

            # Remove vector metadata
            for vid in vector_ids:
                self._metadata["vectors"].pop(str(vid), None)

        del self._metadata["sources"][source_id]
        return len(vector_ids)

    def _remove_vectors_for_markdown(self, rel_path: str) -> int:
        """Remove all vectors associated with a markdown file."""
        # Use rel_path as source key for markdown
        md_key = f"md:{rel_path}"
        if md_key not in self._metadata["sources"]:
            return 0

        source_info = self._metadata["sources"][md_key]
        vector_ids = source_info.get("vector_ids", [])

        if vector_ids:
            ids_array = np.array(vector_ids, dtype=np.int64)
            self._faiss_wrapper.remove_vectors(ids_array)

            # Remove vector metadata
            for vid in vector_ids:
                self._metadata["vectors"].pop(str(vid), None)

        del self._metadata["sources"][md_key]
        return len(vector_ids)

    def _query_entities_for_source(
        self,
        reter: "ReterWrapper",
        source_id: str,
        language: str = "python"
    ) -> List[Dict[str, Any]]:
        """
        Query RETER for all indexable entities in a source.

        Args:
            reter: RETER wrapper instance
            source_id: Source ID (format: "md5|rel_path")
            language: "python", "javascript", "csharp", or "cpp"

        Returns entities with their metadata (type, name, line numbers, docstring).
        """
        entities = []

        # Determine concept prefix based on language
        if language == "python":
            prefix = "py"
        elif language == "javascript":
            prefix = "js"
        elif language == "csharp":
            prefix = "cs"
        elif language == "cpp":
            prefix = "cpp"
        else:
            prefix = "py"  # fallback

        # Convert source_id to is-in-file format
        # source_id format: "md5hash|path\\to\\file.py" or "md5hash|path\\to\\file.js" or "md5hash|path\\to\\file.cs"
        # is-in-file format: "path.to.file.ext" (all languages keep extension)
        if "|" in source_id:
            _, rel_path = source_id.split("|", 1)
        else:
            rel_path = source_id

        # Convert backslashes to dots (keep extension)
        in_file = rel_path.replace("\\", "/")

        logger.debug(f"[RAG] _query_entities: source_id={source_id}, rel_path={rel_path}, in_file={in_file}, language={language}")

        # Query classes
        class_query = f'''
        SELECT DISTINCT ?entity ?name ?line ?endLine ?docstring
        WHERE {{
            ?entity is-in-file "{in_file}" .
            ?entity type class .
            ?entity has-name ?name .
            ?entity is-at-line ?line .
            ?entity has-end-line ?endLine .
            OPTIONAL {{ ?entity has-docstring ?docstring }}
        }}
        '''
        try:
            logger.debug(f"[RAG] _query_entities: Executing class query for in_file={in_file}")
            class_table = reter.reql(class_query)
            if class_table is not None and class_table.num_rows > 0:
                class_results = class_table.to_pylist()  # PyArrow built-in: list of dicts
                logger.debug(f"[RAG] _query_entities: Class query returned {len(class_results)} results")
                for row in class_results:
                    line = int(row.get("?line", 0))
                    end_line_val = row.get("?endLine")
                    end_line = int(end_line_val) if end_line_val else None
                    entities.append({
                        "entity_type": "class",
                        "name": row.get("?name", ""),
                        "qualified_name": row.get("?entity", ""),
                        "line": line,
                        "end_line": end_line,
                        "docstring": row.get("?docstring"),
                    })
            else:
                logger.debug(f"[RAG] _query_entities: Class query returned 0 results")
        except Exception as e:
            logger.debug(f"[RAG] _query_entities: Class query FAILED for {source_id}: {e}")
            logger.debug(f"Class query failed for {source_id}: {e}")

        # Query methods
        # Note: is-defined-in is required (not OPTIONAL) because REQL OPTIONAL doesn't return bound values
        method_query = f'''
        SELECT DISTINCT ?entity ?name ?line ?endLine ?docstring ?className
        WHERE {{
            ?entity is-in-file "{in_file}" .
            ?entity type method .
            ?entity has-name ?name .
            ?entity is-at-line ?line .
            ?entity has-end-line ?endLine .
            ?entity is-defined-in ?className .
            OPTIONAL {{ ?entity has-docstring ?docstring }}
        }}
        '''
        try:
            logger.debug(f"[RAG] _query_entities: Executing method query for in_file={in_file}")
            method_table = reter.reql(method_query)
            if method_table is not None and method_table.num_rows > 0:
                method_results = method_table.to_pylist()
                logger.debug(f"[RAG] _query_entities: Method query returned {len(method_results)} results")
                for row in method_results:
                    line = int(row.get("?line", 0))
                    end_line_val = row.get("?endLine")
                    end_line = int(end_line_val) if end_line_val else None
                    class_name = row.get("?className", "")
                    entities.append({
                        "entity_type": "method",
                        "name": row.get("?name", ""),
                        "qualified_name": row.get("?entity", ""),
                        "line": line,
                        "end_line": end_line,
                        "docstring": row.get("?docstring"),
                        "class_name": class_name,
                    })
            else:
                logger.debug(f"[RAG] _query_entities: Method query returned 0 results")
        except Exception as e:
            logger.debug(f"[RAG] _query_entities: Method query FAILED for {source_id}: {e}")
            logger.debug(f"Method query failed for {source_id}: {e}")

        # Query functions (excluding methods - methods are already indexed separately)
        # Note: method is_subclass_of function, so we need to exclude methods
        func_query = f'''
        SELECT DISTINCT ?entity ?name ?line ?endLine ?docstring
        WHERE {{
            ?entity is-in-file "{in_file}" .
            ?entity type function .
            ?entity has-name ?name .
            ?entity is-at-line ?line .
            ?entity has-end-line ?endLine .
            OPTIONAL {{ ?entity has-docstring ?docstring }}
            FILTER(NOT EXISTS {{ ?entity type method }})
        }}
        '''
        try:
            logger.debug(f"[RAG] _query_entities: Executing function query for in_file={in_file}")
            func_table = reter.reql(func_query)
            if func_table is not None and func_table.num_rows > 0:
                func_results = func_table.to_pylist()
                logger.debug(f"[RAG] _query_entities: Function query returned {len(func_results)} results")
                for row in func_results:
                    line = int(row.get("?line", 0))
                    end_line_val = row.get("?endLine")
                    end_line = int(end_line_val) if end_line_val else None
                    entities.append({
                        "entity_type": "function",
                        "name": row.get("?name", ""),
                        "qualified_name": row.get("?entity", ""),
                        "line": line,
                        "end_line": end_line,
                        "docstring": row.get("?docstring"),
                    })
            else:
                logger.debug(f"[RAG] _query_entities: Function query returned 0 results")
        except Exception as e:
            logger.debug(f"[RAG] _query_entities: Function query FAILED for {source_id}: {e}")
            logger.debug(f"Function query failed for {source_id}: {e}")

        logger.debug(f"[RAG] _query_entities: Total entities found for {source_id}: {len(entities)}")
        return entities

    def _prepare_python_entities(
        self,
        entities: List[Dict[str, Any]],
        source_id: str,
        project_root: Path
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Prepare Python entities for indexing (extract texts and metadata without generating embeddings).

        This allows collecting texts from multiple files for bulk embedding generation.
        Delegates to _collect_python_entities which handles chunking.

        Returns:
            Tuple of (texts, entity_metadata)
        """
        return self._collect_python_entities(entities, source_id, project_root, self._chunk_config)

    def _add_vectors_bulk(
        self,
        texts: List[str],
        metadata_list: List[Dict[str, Any]],
        source_type: str
    ) -> int:
        """
        Generate embeddings and add vectors in bulk.

        Args:
            texts: List of texts to embed
            metadata_list: List of metadata dicts (must include 'source_id')
            source_type: Type for logging ("python", "javascript", etc.)

        Returns:
            Number of vectors added
        """
        if not texts:
            return 0

        logger.debug(f"[RAG] _add_vectors_bulk: Generating {len(texts)} embeddings for {source_type}...")
        batch_size = self._config.get("rag_batch_size", 32)
        embeddings = self._embedding_service.generate_embeddings_batch(
            texts, batch_size=batch_size
        )
        logger.debug(f"[RAG] _add_vectors_bulk: Generated {len(embeddings)} embeddings")

        # Assign vector IDs
        vector_ids = list(range(self._next_vector_id, self._next_vector_id + len(texts)))
        self._next_vector_id += len(texts)

        # Add to FAISS
        ids_array = np.array(vector_ids, dtype=np.int64)
        self._faiss_wrapper.add_vectors(embeddings, ids_array)

        # Update metadata and track by source
        vectors_by_source: Dict[str, List[int]] = {}
        for vid, meta in zip(vector_ids, metadata_list):
            self._metadata["vectors"][str(vid)] = meta
            source_id = meta.get("source_id", "")
            if source_id:
                if source_id not in vectors_by_source:
                    vectors_by_source[source_id] = []
                vectors_by_source[source_id].append(vid)

        # Update source metadata
        for source_id, vids in vectors_by_source.items():
            if "|" in source_id:
                md5_hash, rel_path = source_id.split("|", 1)
            else:
                md5_hash, rel_path = "", source_id

            if source_id not in self._metadata["sources"]:
                self._metadata["sources"][source_id] = {
                    "file_type": source_type,
                    "md5": md5_hash,
                    "rel_path": rel_path,
                    "indexed_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                    "vector_ids": vids,
                }
            else:
                # Append to existing vector_ids
                self._metadata["sources"][source_id]["vector_ids"].extend(vids)

        return len(texts)

    def _prepare_python_comments(
        self,
        entities: List[Dict[str, Any]],
        source_id: str,
        project_root: Path
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Prepare Python comments for indexing (extract texts and metadata without generating embeddings).

        Returns:
            Tuple of (texts, comment_metadata)
        """
        from .content_extractor import CommentBlock

        # Extract file path from source_id
        if "|" in source_id:
            _, rel_path = source_id.split("|", 1)
        else:
            rel_path = source_id

        abs_path = project_root / rel_path

        # Prepare entity locations for context
        entity_locations = [
            {
                'name': e.get('qualified_name') or e.get('name'),
                'line_start': e.get('line', 0),
                'line_end': e.get('end_line') or e.get('line', 0),
                'entity_type': e.get('entity_type'),
            }
            for e in entities
        ]

        # Extract comments
        try:
            comments = self._content_extractor.extract_comments(
                str(abs_path),
                entity_locations=entity_locations
            )
        except Exception:
            return [], []

        if not comments:
            return [], []

        # Only index special comments (TODO, FIXME, etc.)
        SPECIAL_COMMENT_TYPES = {'todo', 'fixme', 'bug', 'hack', 'note', 'warning', 'review', 'optimize', 'xxx'}

        texts = []
        comment_metadata = []

        for comment in comments:
            if comment.comment_type not in SPECIAL_COMMENT_TYPES:
                continue
            text = self._content_extractor.build_comment_indexable_text(comment)
            texts.append(text)
            comment_metadata.append({
                "entity_type": f"comment_{comment.comment_type}",
                "name": comment.comment_type.upper() if comment.comment_type in ('todo', 'fixme', 'bug', 'hack') else "Comment",
                "qualified_name": f"{rel_path}:{comment.line_start}",
                "file": rel_path,
                "line": comment.line_start,
                "end_line": comment.line_end,
                "source_type": "python_comment",
                "comment_type": comment.comment_type,
                "context_entity": comment.context_entity,
                "is_standalone": comment.is_standalone,
                "content_preview": comment.content[:100] if comment.content else "",
                "source_id": f"comments:{source_id}",
            })

        return texts, comment_metadata

    # Alias for batched indexing
    _collect_python_comments = _prepare_python_comments

    def _index_entities(
        self,
        entities: List[Dict[str, Any]],
        source_id: str,
        project_root: Path,
        language: str
    ) -> int:
        """
        Index code entities into FAISS.

        Args:
            entities: List of entity dicts from _query_entities_for_source
            source_id: Source ID (format: "md5|rel_path")
            project_root: Project root directory
            language: Language type (e.g., "python", "javascript")

        Returns number of vectors added.
        """
        log_prefix = f"[RAG] _index_{language}_entities"
        logger.debug(f"{log_prefix}: Starting for {source_id} with {len(entities)} entities")
        if not entities:
            logger.debug(f"{log_prefix}: No entities, returning 0")
            return 0

        # Extract file path from source_id (format: "md5|rel_path")
        if "|" in source_id:
            _, rel_path = source_id.split("|", 1)
        else:
            rel_path = source_id

        abs_path = project_root / rel_path
        logger.debug(f"{log_prefix}: File path: {abs_path}")

        # Build indexable texts
        texts = []
        entity_metadata = []

        for i, entity in enumerate(entities):
            logger.debug(f"{log_prefix}: Processing entity [{i+1}/{len(entities)}]: {entity.get('name')}")
            # Extract content from file
            try:
                code_entity = self._content_extractor.extract_and_build(
                    file_path=str(abs_path),
                    entity_type=entity["entity_type"],
                    name=entity["name"],
                    qualified_name=entity["qualified_name"],
                    start_line=entity["line"],
                    end_line=entity.get("end_line"),
                    docstring=entity.get("docstring"),
                    class_name=entity.get("class_name")
                )
                logger.debug(f"{log_prefix}: extract_and_build returned: {code_entity is not None}")
            except Exception as e:
                import traceback
                logger.debug(f"{log_prefix}: extract_and_build FAILED: {e}")
                logger.debug(f"{log_prefix}: Traceback: {traceback.format_exc()}")
                continue

            if code_entity:
                text = self._content_extractor.build_indexable_text(code_entity)
                texts.append(text)
                entity_metadata.append({
                    "entity_type": entity["entity_type"],
                    "name": entity["name"],
                    "qualified_name": entity["qualified_name"],
                    "file": rel_path,
                    "line": entity["line"],
                    "end_line": entity.get("end_line"),
                    "docstring_preview": (entity.get("docstring") or "")[:100],
                    "source_type": language,
                    "class_name": entity.get("class_name"),
                })

        logger.debug(f"{log_prefix}: Built {len(texts)} texts for embedding")
        if not texts:
            logger.debug(f"{log_prefix}: No texts to embed, returning 0")
            return 0

        # Generate embeddings
        logger.debug(f"{log_prefix}: Generating embeddings...")
        batch_size = self._config.get("rag_batch_size", 32)
        embeddings = self._embedding_service.generate_embeddings_batch(
            texts, batch_size=batch_size
        )
        logger.debug(f"{log_prefix}: Generated {len(embeddings)} embeddings")

        # Assign vector IDs
        vector_ids = list(range(self._next_vector_id, self._next_vector_id + len(texts)))
        self._next_vector_id += len(texts)

        # Add to FAISS
        ids_array = np.array(vector_ids, dtype=np.int64)
        self._faiss_wrapper.add_vectors(embeddings, ids_array)

        # Update metadata
        for vid, meta in zip(vector_ids, entity_metadata):
            self._metadata["vectors"][str(vid)] = meta

        # Track source with MD5 hash
        # source_id format: "md5|rel_path" - extract md5 for tracking
        if "|" in source_id:
            md5_hash = source_id.split("|", 1)[0]
        else:
            md5_hash = ""

        self._metadata["sources"][source_id] = {
            "file_type": language,
            "md5": md5_hash,
            "rel_path": rel_path,
            "indexed_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "vector_ids": vector_ids,
        }

        return len(texts)

    def _index_python_entities(
        self,
        entities: List[Dict[str, Any]],
        source_id: str,
        project_root: Path
    ) -> int:
        """Index Python entities into FAISS."""
        return self._index_entities(entities, source_id, project_root, "python")

    def _index_javascript_entities(
        self,
        entities: List[Dict[str, Any]],
        source_id: str,
        project_root: Path
    ) -> int:
        """Index JavaScript entities into FAISS."""
        return self._index_entities(entities, source_id, project_root, "javascript")

    def _query_html_entities_for_source(
        self,
        reter: "ReterWrapper",
        source_id: str
    ) -> List[Dict[str, Any]]:
        """
        Query RETER for all indexable HTML entities in a source.

        Args:
            reter: RETER wrapper instance
            source_id: Source ID (format: "md5|rel_path")

        Returns entities with their metadata (type, name, line numbers, content).
        """
        entities = []

        # Convert source_id to is-in-file format
        if "|" in source_id:
            _, rel_path = source_id.split("|", 1)
        else:
            rel_path = source_id

        # Convert backslashes to dots
        in_file = rel_path.replace("\\", "/")

        logger.debug(f"[RAG] _query_html_entities: source_id={source_id}, in_file={in_file}")

        # Query scripts (inline JavaScript)
        script_query = f'''
        SELECT DISTINCT ?entity ?name ?line ?content
        WHERE {{
            ?entity is-in-document "{in_file}" .
            ?entity type script .
            OPTIONAL {{ ?entity has-name ?name }}
            OPTIONAL {{ ?entity is-at-line ?line }}
            OPTIONAL {{ ?entity has-content ?content }}
        }}
        '''
        try:
            script_table = reter.reql(script_query)
            if script_table is not None and script_table.num_rows > 0:
                for row in script_table.to_pylist():
                    entities.append({
                        "entity_type": "script",
                        "name": row.get("?name", "inline_script"),
                        "line": int(row.get("?line", 0)),
                        "content": row.get("?content", ""),
                        "source_type": "html",
                    })
        except Exception as e:
            logger.debug(f"[RAG] _query_html_entities: Script query failed: {e}")

        # Query event handlers
        handler_query = f'''
        SELECT DISTINCT ?entity ?event ?handler ?line
        WHERE {{
            ?entity is-in-document "{in_file}" .
            ?entity type event-handler .
            OPTIONAL {{ ?entity has-event-name ?event }}
            OPTIONAL {{ ?entity has-handler-code ?handler }}
            OPTIONAL {{ ?entity is-at-line ?line }}
        }}
        '''
        try:
            handler_table = reter.reql(handler_query)
            if handler_table is not None and handler_table.num_rows > 0:
                for row in handler_table.to_pylist():
                    event = row.get("?event", "unknown")
                    handler = row.get("?handler", "")
                    entities.append({
                        "entity_type": "event_handler",
                        "name": f"on{event}",
                        "line": int(row.get("?line", 0)),
                        "content": handler,
                        "event": event,
                        "source_type": "html",
                    })
        except Exception as e:
            logger.debug(f"[RAG] _query_html_entities: Handler query failed: {e}")

        # Query forms
        form_query = f'''
        SELECT DISTINCT ?entity ?name ?action ?method ?line
        WHERE {{
            ?entity is-in-document "{in_file}" .
            ?entity type form .
            OPTIONAL {{ ?entity has-name ?name }}
            OPTIONAL {{ ?entity has-action ?action }}
            OPTIONAL {{ ?entity has-method ?method }}
            OPTIONAL {{ ?entity is-at-line ?line }}
        }}
        '''
        try:
            form_table = reter.reql(form_query)
            if form_table is not None and form_table.num_rows > 0:
                for row in form_table.to_pylist():
                    name = row.get("?name", "")
                    action = row.get("?action", "")
                    method = row.get("?method", "GET")
                    entities.append({
                        "entity_type": "form",
                        "name": name or f"form_{action}",
                        "line": int(row.get("?line", 0)),
                        "action": action,
                        "method": method,
                        "source_type": "html",
                    })
        except Exception as e:
            logger.debug(f"[RAG] _query_html_entities: Form query failed: {e}")

        # Query framework directives (Vue, Angular, HTMX, Alpine)
        for framework, concept in [
            ("vue", "vue-directive"),
            ("angular", "angular-directive"),
            ("htmx", "htmx-attribute"),
            ("alpine", "alpine-directive"),
        ]:
            directive_query = f'''
            SELECT DISTINCT ?entity ?directive ?value ?line
            WHERE {{
                ?entity is-in-document "{in_file}" .
                ?entity type {concept} .
                OPTIONAL {{ ?entity has-directive-name ?directive }}
                OPTIONAL {{ ?entity has-directive-value ?value }}
                OPTIONAL {{ ?entity is-at-line ?line }}
            }}
            '''
            try:
                directive_table = reter.reql(directive_query)
                if directive_table is not None and directive_table.num_rows > 0:
                    for row in directive_table.to_pylist():
                        directive = row.get("?directive", "")
                        value = row.get("?value", "")
                        entities.append({
                            "entity_type": f"{framework}_directive",
                            "name": directive,
                            "line": int(row.get("?line", 0)),
                            "content": value,
                            "framework": framework,
                            "source_type": "html",
                        })
            except Exception as e:
                logger.debug(f"[RAG] _query_html_entities: {framework} query failed: {e}")

        logger.debug(f"[RAG] _query_html_entities: Found {len(entities)} total entities")
        return entities

    def _index_html_entities(
        self,
        entities: List[Dict[str, Any]],
        source_id: str,
        project_root: Path
    ) -> int:
        """
        Index HTML entities into FAISS.

        Args:
            entities: List of HTML entity dicts from _query_html_entities_for_source
            source_id: Source ID (format: "md5|rel_path")
            project_root: Project root directory

        Returns number of vectors added.
        """
        logger.debug(f"[RAG] _index_html_entities: Starting for {source_id} with {len(entities)} entities")
        if not entities:
            return 0

        # Extract file path from source_id
        if "|" in source_id:
            _, rel_path = source_id.split("|", 1)
        else:
            rel_path = source_id

        # Build indexable texts
        texts = []
        entity_metadata = []

        for entity in entities:
            entity_type = entity.get("entity_type", "")
            name = entity.get("name", "")
            content = entity.get("content", "")
            line = entity.get("line", 0)

            # Build searchable text based on entity type
            if entity_type == "script":
                # For scripts, index the script content
                text = f"HTML inline script: {content[:500]}" if content else f"HTML script at line {line}"
            elif entity_type == "event_handler":
                event = entity.get("event", "")
                text = f"HTML event handler {name} ({event}): {content}"
            elif entity_type == "form":
                action = entity.get("action", "")
                method = entity.get("method", "")
                text = f"HTML form {name} action={action} method={method}"
            elif entity_type.endswith("_directive"):
                framework = entity.get("framework", "")
                text = f"HTML {framework} directive {name}: {content}"
            else:
                text = f"HTML {entity_type} {name}: {content}"

            if text:
                texts.append(text)
                entity_metadata.append({
                    "entity_type": entity_type,
                    "name": name,
                    "qualified_name": f"{rel_path}:{name}",
                    "file": rel_path,
                    "line": line,
                    "end_line": None,
                    "source_type": "html",
                    "content_preview": content[:200] if content else "",
                })

        if not texts:
            return 0

        # Generate embeddings
        batch_size = self._config.get("rag_batch_size", 32)
        embeddings = self._embedding_service.generate_embeddings_batch(
            texts, batch_size=batch_size
        )

        # Assign vector IDs
        vector_ids = list(range(self._next_vector_id, self._next_vector_id + len(texts)))
        self._next_vector_id += len(texts)

        # Add to FAISS
        ids_array = np.array(vector_ids, dtype=np.int64)
        self._faiss_wrapper.add_vectors(embeddings, ids_array)

        # Update metadata
        for vid, meta in zip(vector_ids, entity_metadata):
            self._metadata["vectors"][str(vid)] = meta

        # Track source
        if "|" in source_id:
            md5_hash = source_id.split("|", 1)[0]
        else:
            md5_hash = ""

        self._metadata["sources"][source_id] = {
            "file_type": "html",
            "md5": md5_hash,
            "rel_path": rel_path,
            "indexed_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "vector_ids": vector_ids,
        }

        logger.debug(f"[RAG] _index_html_entities: Indexed {len(texts)} entities for {rel_path}")
        return len(texts)

    def _index_python_comments(
        self,
        entities: List[Dict[str, Any]],
        source_id: str,
        project_root: Path
    ) -> int:
        """
        Index Python comments into FAISS.

        Args:
            entities: List of code entities for context
            source_id: Source ID (format: "md5|rel_path")
            project_root: Project root directory

        Returns number of comment vectors added.
        """
        from .content_extractor import CommentBlock

        # Extract file path from source_id
        if "|" in source_id:
            _, rel_path = source_id.split("|", 1)
        else:
            rel_path = source_id

        abs_path = project_root / rel_path

        # Prepare entity locations for context
        entity_locations = [
            {
                'name': e.get('qualified_name') or e.get('name'),
                'line_start': e.get('line', 0),
                'line_end': e.get('end_line') or e.get('line', 0),
                'entity_type': e.get('entity_type'),
            }
            for e in entities
        ]

        # Extract comments
        comments = self._content_extractor.extract_comments(
            str(abs_path),
            entity_locations=entity_locations
        )

        if not comments:
            return 0

        # Only index special comments (TODO, FIXME, etc.), not regular inline/block comments
        # Regular comments should only be part of entity bodies, not indexed separately
        SPECIAL_COMMENT_TYPES = {'todo', 'fixme', 'bug', 'hack', 'note', 'warning', 'review', 'optimize', 'xxx'}

        # Build indexable texts
        texts = []
        comment_metadata = []

        for comment in comments:
            # Skip regular inline/block comments - only index special markers
            if comment.comment_type not in SPECIAL_COMMENT_TYPES:
                continue
            text = self._content_extractor.build_comment_indexable_text(comment)
            texts.append(text)
            comment_metadata.append({
                "entity_type": f"comment_{comment.comment_type}",
                "name": comment.comment_type.upper() if comment.comment_type in ('todo', 'fixme', 'bug', 'hack') else "Comment",
                "qualified_name": f"{rel_path}:{comment.line_start}",
                "file": rel_path,
                "line": comment.line_start,
                "end_line": comment.line_end,
                "source_type": "python_comment",
                "comment_type": comment.comment_type,
                "context_entity": comment.context_entity,
                "is_standalone": comment.is_standalone,
                "content_preview": comment.content[:100] if comment.content else "",
            })

        if not texts:
            return 0

        # Generate embeddings
        batch_size = self._config.get("rag_batch_size", 32)
        embeddings = self._embedding_service.generate_embeddings_batch(
            texts, batch_size=batch_size
        )

        # Assign vector IDs
        vector_ids = list(range(self._next_vector_id, self._next_vector_id + len(texts)))
        self._next_vector_id += len(texts)

        # Add to FAISS
        ids_array = np.array(vector_ids, dtype=np.int64)
        self._faiss_wrapper.add_vectors(embeddings, ids_array)

        # Update metadata
        for vid, meta in zip(vector_ids, comment_metadata):
            self._metadata["vectors"][str(vid)] = meta

        # Track comment source
        comment_key = f"comments:{source_id}"
        if "|" in source_id:
            md5_hash = source_id.split("|", 1)[0]
        else:
            md5_hash = ""

        self._metadata["sources"][comment_key] = {
            "file_type": "python_comment",
            "md5": md5_hash,
            "rel_path": rel_path,
            "indexed_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "vector_ids": vector_ids,
        }

        return len(texts)

    def _index_python_literals(
        self,
        reter: "ReterWrapper",
        source_id: str,
        project_root: Path,
        min_length: int = 32
    ) -> int:
        """
        Index Python string literals into FAISS for semantic search.

        This allows searching for:
        - Error messages and user-facing strings (i18n candidates)
        - SQL queries embedded in code
        - LLM prompts and templates
        - Configuration strings and paths

        Args:
            reter: ReterWrapper instance to query literals
            source_id: Source ID (format: "md5|rel_path")
            project_root: Project root directory
            min_length: Minimum string length to index (default: 20 chars)

        Returns number of literal vectors added.
        """
        # Extract file path from source_id
        if "|" in source_id:
            md5_hash, rel_path = source_id.split("|", 1)
        else:
            md5_hash = ""
            rel_path = source_id

        # Convert rel_path to module name (e.g., "foo/bar.py" -> "foo.bar")
        module_name = rel_path.replace("\\", ".").replace("/", ".")
        if module_name.endswith(".py"):
            module_name = module_name[:-3]

        # Query string literals for this module
        try:
            query = f"""
                SELECT DISTINCT ?entity ?literal ?entityName
                WHERE {{
                    ?entity has-string-literal ?literal .
                    ?entity is-in-module ?module .
                    ?entity has-name ?entityName .
                    FILTER(CONTAINS(?module, "{module_name}"))
                }}
            """
            result = reter.reql(query, timeout_ms=10000)

            if result.num_rows == 0:
                return 0

            # Convert to list
            columns = [result.column(name).to_pylist() for name in result.column_names]
            rows = list(zip(*columns))
        except Exception as e:
            logger.debug(f"[RAG] _index_python_literals: Query failed for {source_id}: {e}")
            return 0

        # Build indexable texts - filter by length and skip common patterns
        texts = []
        literal_metadata = []

        for entity, literal, entity_name in rows:
            # Skip short strings
            if len(literal) < min_length:
                continue

            # Skip strings that look like code identifiers (all lowercase/uppercase with underscores)
            clean = literal.strip('"\'')
            if clean.replace('_', '').isalnum() and (clean.isupper() or clean.islower()):
                continue

            # Build searchable text: include context and the literal
            text = f"String literal in {entity_name}: {clean}"

            texts.append(text)
            literal_metadata.append({
                "entity_type": "string_literal",
                "name": clean[:50] + "..." if len(clean) > 50 else clean,
                "qualified_name": f"{entity}:literal",
                "file": rel_path,
                "line": 0,  # We don't have line info for literals yet
                "source_type": "python_literal",
                "containing_entity": entity,
                "containing_name": entity_name,
                "content_preview": clean[:200] if clean else "",
                "literal_length": len(clean),
            })

        if not texts:
            return 0

        # Generate embeddings
        batch_size = self._config.get("rag_batch_size", 32)
        embeddings = self._embedding_service.generate_embeddings_batch(
            texts, batch_size=batch_size
        )

        # Assign vector IDs
        vector_ids = list(range(self._next_vector_id, self._next_vector_id + len(texts)))
        self._next_vector_id += len(texts)

        # Add to FAISS
        ids_array = np.array(vector_ids, dtype=np.int64)
        self._faiss_wrapper.add_vectors(embeddings, ids_array)

        # Update metadata
        for vid, meta in zip(vector_ids, literal_metadata):
            self._metadata["vectors"][str(vid)] = meta

        # Track literal source
        literal_key = f"literals:{source_id}"
        self._metadata["sources"][literal_key] = {
            "file_type": "python_literal",
            "md5": md5_hash,
            "rel_path": rel_path,
            "indexed_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "vector_ids": vector_ids,
        }

        logger.debug(f"[RAG] _index_python_literals: Indexed {len(texts)} literals for {rel_path}")
        return len(texts)

    def _index_all_python_literals_bulk(
        self,
        reter: "ReterWrapper",
        project_root: Path,
        min_length: int = 32
    ) -> int:
        """
        Bulk index ALL Python string literals in one query.

        Much faster than per-file indexing for initial load / full reindex.
        Runs ONE REQL query and processes all literals in large batches.

        Args:
            reter: ReterWrapper instance
            project_root: Project root directory
            min_length: Minimum string length to index

        Returns total number of literal vectors added.
        """
        import time
        start_time = time.time()

        # Query ALL string literals in one go
        try:
            query = """
                SELECT DISTINCT ?entity ?literal ?entityName ?module
                WHERE {
                    ?entity has-string-literal ?literal .
                    ?entity is-in-module ?module .
                    ?entity has-name ?entityName .
                }
            """
            result = reter.reql(query, timeout_ms=60000)  # Longer timeout for bulk

            if result.num_rows == 0:
                logger.debug("[RAG] _index_all_python_literals_bulk: No literals found")
                return 0

            # Convert to list
            columns = [result.column(name).to_pylist() for name in result.column_names]
            rows = list(zip(*columns))
            logger.debug(f"[RAG] _index_all_python_literals_bulk: Found {len(rows)} total literals")
        except Exception as e:
            logger.debug(f"[RAG] _index_all_python_literals_bulk: Query failed: {e}")
            return 0

        query_time = time.time() - start_time
        logger.debug(f"[RAG] _index_all_python_literals_bulk: Query took {query_time:.2f}s")

        # Build indexable texts - filter by length and skip common patterns
        texts = []
        literal_metadata = []
        seen_literals: set = set()  # Deduplicate by entity + literal content

        for entity, literal, entity_name, module in rows:
            # Skip short strings
            if len(literal) < min_length:
                continue

            # Skip strings that look like code identifiers
            clean = literal.strip('"\'')
            if clean.replace('_', '').isalnum() and (clean.isupper() or clean.islower()):
                continue

            # Deduplicate by entity + literal content
            dedup_key = f"{entity}:{clean}"
            if dedup_key in seen_literals:
                continue
            seen_literals.add(dedup_key)

            # Derive file path from module
            rel_path = module.replace(".", "/") + ".py" if module else "unknown.py"

            # Build searchable text
            text = f"String literal in {entity_name}: {clean}"

            texts.append(text)
            literal_metadata.append({
                "entity_type": "string_literal",
                "name": clean[:50] + "..." if len(clean) > 50 else clean,
                "qualified_name": f"{entity}:literal",
                "file": rel_path,
                "line": 0,
                "source_type": "python_literal",
                "containing_entity": entity,
                "containing_name": entity_name,
                "content_preview": clean[:200] if clean else "",
                "literal_length": len(clean),
            })

        if not texts:
            logger.debug("[RAG] _index_all_python_literals_bulk: No literals passed filtering")
            return 0

        logger.debug(f"[RAG] _index_all_python_literals_bulk: {len(texts)} literals passed filtering")

        # Generate embeddings in large batches
        batch_size = self._config.get("rag_batch_size", 32)
        embed_start = time.time()
        embeddings = self._embedding_service.generate_embeddings_batch(
            texts, batch_size=batch_size
        )
        embed_time = time.time() - embed_start
        logger.debug(f"[RAG] _index_all_python_literals_bulk: Embeddings took {embed_time:.2f}s")

        # Assign vector IDs
        vector_ids = list(range(self._next_vector_id, self._next_vector_id + len(texts)))
        self._next_vector_id += len(texts)

        # Add to FAISS
        ids_array = np.array(vector_ids, dtype=np.int64)
        self._faiss_wrapper.add_vectors(embeddings, ids_array)

        # Update metadata
        for vid, meta in zip(vector_ids, literal_metadata):
            self._metadata["vectors"][str(vid)] = meta

        # Track as bulk literal source
        self._metadata["sources"]["literals:bulk"] = {
            "file_type": "python_literal_bulk",
            "indexed_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "vector_ids": vector_ids,
            "total_literals": len(texts),
        }

        total_time = time.time() - start_time
        logger.debug(f"[RAG] _index_all_python_literals_bulk: Indexed {len(texts)} literals in {total_time:.2f}s")
        return len(texts)

    def _index_javascript_literals(
        self,
        reter: "ReterWrapper",
        source_id: str,
        project_root: Path,
        min_length: int = 32
    ) -> int:
        """
        Index JavaScript string literals into FAISS for semantic search.

        This allows searching for:
        - Error messages and user-facing strings (i18n candidates)
        - SQL queries embedded in code
        - API endpoints and URLs
        - Configuration strings

        Args:
            reter: ReterWrapper instance to query literals
            source_id: Source ID (format: "md5|rel_path")
            project_root: Project root directory
            min_length: Minimum string length to index (default: 20 chars)

        Returns number of literal vectors added.
        """
        # Extract file path from source_id
        if "|" in source_id:
            md5_hash, rel_path = source_id.split("|", 1)
        else:
            md5_hash = ""
            rel_path = source_id

        # Convert rel_path to module name (e.g., "foo/bar.js" -> "foo.bar")
        module_name = rel_path.replace("\\", ".").replace("/", ".")
        if module_name.endswith(".js"):
            module_name = module_name[:-3]

        # Query string literals for this module
        try:
            query = f"""
                SELECT DISTINCT ?entity ?literal ?entityName
                WHERE {{
                    ?entity has-string-literal ?literal .
                    ?entity is-in-module ?module .
                    ?entity has-name ?entityName .
                    FILTER(CONTAINS(?module, "{module_name}"))
                }}
            """
            result = reter.reql(query, timeout_ms=10000)

            if result.num_rows == 0:
                return 0

            # Convert to list
            columns = [result.column(name).to_pylist() for name in result.column_names]
            rows = list(zip(*columns))
        except Exception as e:
            logger.debug(f"[RAG] _index_javascript_literals: Query failed for {source_id}: {e}")
            return 0

        # Build indexable texts - filter by length and skip common patterns
        texts = []
        literal_metadata = []

        for entity, literal, entity_name in rows:
            # Skip short strings
            if len(literal) < min_length:
                continue

            # Skip strings that look like code identifiers (all lowercase/uppercase with underscores)
            clean = literal.strip('"\'`')  # Also handle backticks for JS
            if clean.replace('_', '').isalnum() and (clean.isupper() or clean.islower()):
                continue

            # Build searchable text: include context and the literal
            text = f"String literal in {entity_name}: {clean}"

            texts.append(text)
            literal_metadata.append({
                "entity_type": "string_literal",
                "name": clean[:50] + "..." if len(clean) > 50 else clean,
                "qualified_name": f"{entity}:literal",
                "file": rel_path,
                "line": 0,  # We don't have line info for literals yet
                "source_type": "javascript_literal",
                "containing_entity": entity,
                "containing_name": entity_name,
                "content_preview": clean[:200] if clean else "",
                "literal_length": len(clean),
            })

        if not texts:
            return 0

        # Generate embeddings
        batch_size = self._config.get("rag_batch_size", 32)
        embeddings = self._embedding_service.generate_embeddings_batch(
            texts, batch_size=batch_size
        )

        # Assign vector IDs
        vector_ids = list(range(self._next_vector_id, self._next_vector_id + len(texts)))
        self._next_vector_id += len(texts)

        # Add to FAISS
        ids_array = np.array(vector_ids, dtype=np.int64)
        self._faiss_wrapper.add_vectors(embeddings, ids_array)

        # Update metadata
        for vid, meta in zip(vector_ids, literal_metadata):
            self._metadata["vectors"][str(vid)] = meta

        # Track literal source
        literal_key = f"js_literals:{source_id}"
        self._metadata["sources"][literal_key] = {
            "file_type": "javascript_literal",
            "md5": md5_hash,
            "rel_path": rel_path,
            "indexed_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "vector_ids": vector_ids,
        }

        logger.debug(f"[RAG] _index_javascript_literals: Indexed {len(texts)} literals for {rel_path}")
        return len(texts)

    def _index_all_javascript_literals_bulk(
        self,
        reter: "ReterWrapper",
        project_root: Path,
        min_length: int = 32
    ) -> int:
        """
        Bulk index ALL JavaScript string literals in one query.

        Much faster than per-file indexing for initial load / full reindex.
        Runs ONE REQL query and processes all literals in large batches.

        Args:
            reter: ReterWrapper instance
            project_root: Project root directory
            min_length: Minimum string length to index

        Returns total number of literal vectors added.
        """
        import time
        start_time = time.time()

        # Query ALL JavaScript string literals in one go
        try:
            query = """
                SELECT DISTINCT ?entity ?literal ?entityName ?module
                WHERE {
                    ?entity has-string-literal ?literal .
                    ?entity is-in-module ?module .
                    ?entity has-name ?entityName .
                    FILTER(CONTAINS(?module, ".js") || CONTAINS(?module, ".mjs") || CONTAINS(?module, ".cjs"))
                }
            """
            result = reter.reql(query, timeout_ms=60000)  # Longer timeout for bulk

            if result.num_rows == 0:
                logger.debug("[RAG] _index_all_javascript_literals_bulk: No literals found")
                return 0

            # Convert to list
            columns = [result.column(name).to_pylist() for name in result.column_names]
            rows = list(zip(*columns))
            logger.debug(f"[RAG] _index_all_javascript_literals_bulk: Found {len(rows)} total literals")
        except Exception as e:
            logger.debug(f"[RAG] _index_all_javascript_literals_bulk: Query failed: {e}")
            return 0

        query_time = time.time() - start_time
        logger.debug(f"[RAG] _index_all_javascript_literals_bulk: Query took {query_time:.2f}s")

        # Build indexable texts - filter by length and skip common patterns
        texts = []
        literal_metadata = []
        seen_literals: set = set()  # Deduplicate by entity + literal content

        for entity, literal, entity_name, module in rows:
            # Skip short strings
            if len(literal) < min_length:
                continue

            # Skip strings that look like code identifiers
            clean = literal.strip('"\'`')  # Also handle backticks for JS
            if clean.replace('_', '').isalnum() and (clean.isupper() or clean.islower()):
                continue

            # Deduplicate by entity + literal content
            dedup_key = f"{entity}:{clean}"
            if dedup_key in seen_literals:
                continue
            seen_literals.add(dedup_key)

            # Derive file path from module
            rel_path = module.replace(".", "/") + ".js" if module else "unknown.js"

            # Build searchable text
            text = f"String literal in {entity_name}: {clean}"

            texts.append(text)
            literal_metadata.append({
                "entity_type": "string_literal",
                "name": clean[:50] + "..." if len(clean) > 50 else clean,
                "qualified_name": f"{entity}:literal",
                "file": rel_path,
                "line": 0,
                "source_type": "javascript_literal",
                "containing_entity": entity,
                "containing_name": entity_name,
                "content_preview": clean[:200] if clean else "",
                "literal_length": len(clean),
            })

        if not texts:
            logger.debug("[RAG] _index_all_javascript_literals_bulk: No literals passed filtering")
            return 0

        logger.debug(f"[RAG] _index_all_javascript_literals_bulk: {len(texts)} literals passed filtering")

        # Generate embeddings in large batches
        batch_size = self._config.get("rag_batch_size", 32)
        embed_start = time.time()
        embeddings = self._embedding_service.generate_embeddings_batch(
            texts, batch_size=batch_size
        )
        embed_time = time.time() - embed_start
        logger.debug(f"[RAG] _index_all_javascript_literals_bulk: Embeddings took {embed_time:.2f}s")

        # Assign vector IDs
        vector_ids = list(range(self._next_vector_id, self._next_vector_id + len(texts)))
        self._next_vector_id += len(texts)

        # Add to FAISS
        ids_array = np.array(vector_ids, dtype=np.int64)
        self._faiss_wrapper.add_vectors(embeddings, ids_array)

        # Update metadata
        for vid, meta in zip(vector_ids, literal_metadata):
            self._metadata["vectors"][str(vid)] = meta

        # Track as bulk literal source
        self._metadata["sources"]["js_literals:bulk"] = {
            "file_type": "javascript_literal_bulk",
            "indexed_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "vector_ids": vector_ids,
            "total_literals": len(texts),
        }

        total_time = time.time() - start_time
        logger.debug(f"[RAG] _index_all_javascript_literals_bulk: Indexed {len(texts)} literals in {total_time:.2f}s")
        return len(texts)

    def _index_markdown_chunks(
        self,
        chunks: List[MarkdownChunk],
        rel_path: str,
        md5_hash: Optional[str] = None
    ) -> int:
        """
        Index markdown chunks into FAISS.

        Args:
            chunks: Parsed markdown chunks
            rel_path: Relative path to markdown file
            md5_hash: Optional MD5 hash of file content for change tracking

        Returns number of vectors added.
        """
        if not chunks:
            return 0

        # Build indexable texts
        texts = []
        chunk_metadata = []

        for chunk in chunks:
            text = self._markdown_indexer.build_indexable_text(chunk)
            texts.append(text)
            chunk_metadata.append({
                "entity_type": chunk.chunk_type,
                "name": chunk.heading or chunk.title or rel_path,
                "qualified_name": f"{rel_path}:{chunk.line_start}",
                "file": rel_path,
                "line": chunk.line_start,
                "end_line": chunk.line_end,
                "heading": chunk.heading,
                "source_type": "markdown",
                "language": chunk.language,
                "content_preview": chunk.content[:100] if chunk.content else "",
            })

        # Generate embeddings
        batch_size = self._config.get("rag_batch_size", 32)
        embeddings = self._embedding_service.generate_embeddings_batch(
            texts, batch_size=batch_size
        )

        # Assign vector IDs
        vector_ids = list(range(self._next_vector_id, self._next_vector_id + len(texts)))
        self._next_vector_id += len(texts)

        # Add to FAISS
        ids_array = np.array(vector_ids, dtype=np.int64)
        self._faiss_wrapper.add_vectors(embeddings, ids_array)

        # Update metadata
        for vid, meta in zip(vector_ids, chunk_metadata):
            self._metadata["vectors"][str(vid)] = meta

        # Track source with MD5 hash
        md_key = f"md:{rel_path}"
        self._metadata["sources"][md_key] = {
            "file_type": "markdown",
            "md5": md5_hash or "",
            "rel_path": rel_path,
            "indexed_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "vector_ids": vector_ids,
        }

        return len(texts)

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
            code_types = ("python", "python_literal", "python_comment",
                          "javascript", "javascript_literal", "html",
                          "csharp", "cpp")
            if search_scope == "code" and source_type not in code_types:
                continue
            if search_scope == "docs" and source_type != "markdown":
                continue

            # Apply entity type filter
            if entity_types and meta.get("entity_type") not in entity_types:
                continue

            # Apply file filter
            if file_filter:
                import fnmatch
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

    def _query_all_entities_bulk(self, reter: "ReterWrapper", language: str = "python") -> Dict[str, List[Dict[str, Any]]]:
        """
        Query ALL entities from RETER in bulk (3 queries instead of 3 per file).

        Args:
            reter: RETER wrapper instance
            language: "python", "javascript", "csharp", or "cpp"

        Returns dict mapping inFile -> list of entities
        """
        entities_by_file: Dict[str, List[Dict[str, Any]]] = {}
        seen_entities: set = set()  # Track seen qualified_names to avoid duplicates

        # Determine concept prefix based on language
        if language == "python":
            prefix = "py"
        elif language == "javascript":
            prefix = "js"
        elif language == "csharp":
            prefix = "cs"
        elif language == "cpp":
            prefix = "cpp"
        else:
            prefix = "py"  # fallback

        # Bulk query all classes
        logger.debug(f"[RAG] _query_all_entities_bulk: Querying all {language} classes...")
        class_query = f'''
        SELECT DISTINCT ?entity ?name ?line ?endLine ?docstring ?inFile
        WHERE {{
            ?entity type class .
            ?entity has-name ?name .
            ?entity is-at-line ?line .
            ?entity is-in-file ?inFile .
            ?entity has-end-line ?endLine .
            OPTIONAL {{ ?entity has-docstring ?docstring }}
        }}
        '''
        try:
            class_table = reter.reql(class_query)
            if class_table is not None and class_table.num_rows > 0:
                class_results = class_table.to_pylist()
                logger.debug(f"[RAG] _query_all_entities_bulk: Found {len(class_results)} classes")
                for row in class_results:
                    qualified_name = row.get("?entity", "")
                    if qualified_name in seen_entities:
                        continue
                    seen_entities.add(qualified_name)
                    in_file = row.get("?inFile", "")
                    if in_file not in entities_by_file:
                        entities_by_file[in_file] = []
                    line = int(row.get("?line", 0))
                    end_line_val = row.get("?endLine")
                    end_line = int(end_line_val) if end_line_val else None
                    entities_by_file[in_file].append({
                        "entity_type": "class",
                        "name": row.get("?name", ""),
                        "qualified_name": qualified_name,
                        "line": line,
                        "end_line": end_line,
                        "docstring": row.get("?docstring"),
                    })
        except Exception as e:
            logger.debug(f"[RAG] _query_all_entities_bulk: Class query FAILED: {e}")

        # Bulk query all methods
        # Note: We query definedIn as required (not OPTIONAL) because REQL OPTIONAL
        # doesn't return bound values. All methods should have a defining class.
        logger.debug(f"[RAG] _query_all_entities_bulk: Querying all {language} methods...")
        method_query = f'''
        SELECT DISTINCT ?entity ?name ?line ?endLine ?docstring ?className ?inFile
        WHERE {{
            ?entity type method .
            ?entity has-name ?name .
            ?entity is-at-line ?line .
            ?entity is-in-file ?inFile .
            ?entity has-end-line ?endLine .
            ?entity is-defined-in ?className .
            OPTIONAL {{ ?entity has-docstring ?docstring }}
        }}
        '''
        try:
            method_table = reter.reql(method_query)
            if method_table is not None and method_table.num_rows > 0:
                method_results = method_table.to_pylist()
                logger.debug(f"[RAG] _query_all_entities_bulk: Found {len(method_results)} methods")
                for row in method_results:
                    qualified_name = row.get("?entity", "")
                    if qualified_name in seen_entities:
                        continue
                    seen_entities.add(qualified_name)
                    in_file = row.get("?inFile", "")
                    if in_file not in entities_by_file:
                        entities_by_file[in_file] = []
                    line = int(row.get("?line", 0))
                    end_line_val = row.get("?endLine")
                    end_line = int(end_line_val) if end_line_val else None
                    # Get class_name from query result
                    class_name = row.get("?className", "")
                    entities_by_file[in_file].append({
                        "entity_type": "method",
                        "name": row.get("?name", ""),
                        "qualified_name": qualified_name,
                        "line": line,
                        "end_line": end_line,
                        "docstring": row.get("?docstring"),
                        "class_name": class_name,
                    })
        except Exception as e:
            logger.debug(f"[RAG] _query_all_entities_bulk: Method query FAILED: {e}")

        # Bulk query all functions (excluding methods - they are already queried above)
        # Note: Method is_subclass_of Function, so we need to exclude methods
        logger.debug(f"[RAG] _query_all_entities_bulk: Querying all {language} functions...")
        func_query = f'''
        SELECT DISTINCT ?entity ?name ?line ?endLine ?docstring ?inFile
        WHERE {{
            ?entity type function .
            ?entity has-name ?name .
            ?entity is-at-line ?line .
            ?entity is-in-file ?inFile .
            ?entity has-end-line ?endLine .
            OPTIONAL {{ ?entity has-docstring ?docstring }}
            FILTER(NOT EXISTS {{ ?entity type method }})
        }}
        '''
        try:
            func_table = reter.reql(func_query)
            if func_table is not None and func_table.num_rows > 0:
                func_results = func_table.to_pylist()
                logger.debug(f"[RAG] _query_all_entities_bulk: Found {len(func_results)} functions")
                for row in func_results:
                    qualified_name = row.get("?entity", "")
                    if qualified_name in seen_entities:
                        continue
                    seen_entities.add(qualified_name)
                    in_file = row.get("?inFile", "")
                    if in_file not in entities_by_file:
                        entities_by_file[in_file] = []
                    line = int(row.get("?line", 0))
                    end_line_val = row.get("?endLine")
                    end_line = int(end_line_val) if end_line_val else None
                    entities_by_file[in_file].append({
                        "entity_type": "function",
                        "name": row.get("?name", ""),
                        "qualified_name": qualified_name,
                        "line": line,
                        "end_line": end_line,
                        "docstring": row.get("?docstring"),
                    })
        except Exception as e:
            logger.debug(f"[RAG] _query_all_entities_bulk: Function query FAILED: {e}")

        total_entities = sum(len(ents) for ents in entities_by_file.values())
        logger.debug(f"[RAG] _query_all_entities_bulk: Total {len(entities_by_file)} files, {total_entities} entities (deduplicated via {len(seen_entities)} unique qualified_names)")
        return entities_by_file

    def _query_all_html_entities_bulk(self, reter: "ReterWrapper") -> Dict[str, List[Dict[str, Any]]]:
        """
        Query ALL HTML entities from RETER in bulk.

        Returns dict mapping is-in-document -> list of entities
        """
        entities_by_file: Dict[str, List[Dict[str, Any]]] = {}

        # Bulk query all scripts
        logger.debug("[RAG] _query_all_html_entities_bulk: Querying all HTML scripts...")
        script_query = '''
        SELECT DISTINCT ?entity ?name ?line ?content ?inDocument
        WHERE {
            ?entity type script .
            ?entity is-in-document ?inDocument .
            OPTIONAL { ?entity has-name ?name }
            OPTIONAL { ?entity is-at-line ?line }
            OPTIONAL { ?entity has-content ?content }
        }
        '''
        try:
            script_table = reter.reql(script_query)
            if script_table is not None and script_table.num_rows > 0:
                script_results = script_table.to_pylist()
                logger.debug(f"[RAG] _query_all_html_entities_bulk: Found {len(script_results)} scripts")
                for row in script_results:
                    in_doc = row.get("?inDocument", "")
                    if in_doc not in entities_by_file:
                        entities_by_file[in_doc] = []
                    entities_by_file[in_doc].append({
                        "entity_type": "script",
                        "name": row.get("?name", "inline_script"),
                        "line": int(row.get("?line", 0)),
                        "content": row.get("?content", ""),
                        "source_type": "html",
                    })
        except Exception as e:
            logger.debug(f"[RAG] _query_all_html_entities_bulk: Script query FAILED: {e}")

        # Bulk query all event handlers
        logger.debug("[RAG] _query_all_html_entities_bulk: Querying all HTML event handlers...")
        handler_query = '''
        SELECT DISTINCT ?entity ?event ?handler ?line ?inDocument
        WHERE {
            ?entity type event-handler .
            ?entity is-in-document ?inDocument .
            OPTIONAL { ?entity has-event-name ?event }
            OPTIONAL { ?entity has-handler-code ?handler }
            OPTIONAL { ?entity is-at-line ?line }
        }
        '''
        try:
            handler_table = reter.reql(handler_query)
            if handler_table is not None and handler_table.num_rows > 0:
                handler_results = handler_table.to_pylist()
                logger.debug(f"[RAG] _query_all_html_entities_bulk: Found {len(handler_results)} event handlers")
                for row in handler_results:
                    in_doc = row.get("?inDocument", "")
                    if in_doc not in entities_by_file:
                        entities_by_file[in_doc] = []
                    event = row.get("?event", "unknown")
                    handler = row.get("?handler", "")
                    entities_by_file[in_doc].append({
                        "entity_type": "event_handler",
                        "name": f"on{event}",
                        "line": int(row.get("?line", 0)),
                        "content": handler,
                        "event": event,
                        "source_type": "html",
                    })
        except Exception as e:
            logger.debug(f"[RAG] _query_all_html_entities_bulk: Handler query FAILED: {e}")

        # Bulk query all forms
        logger.debug("[RAG] _query_all_html_entities_bulk: Querying all HTML forms...")
        form_query = '''
        SELECT DISTINCT ?entity ?name ?action ?method ?line ?inDocument
        WHERE {
            ?entity type form .
            ?entity is-in-document ?inDocument .
            OPTIONAL { ?entity has-name ?name }
            OPTIONAL { ?entity has-action ?action }
            OPTIONAL { ?entity has-method ?method }
            OPTIONAL { ?entity is-at-line ?line }
        }
        '''
        try:
            form_table = reter.reql(form_query)
            if form_table is not None and form_table.num_rows > 0:
                form_results = form_table.to_pylist()
                logger.debug(f"[RAG] _query_all_html_entities_bulk: Found {len(form_results)} forms")
                for row in form_results:
                    in_doc = row.get("?inDocument", "")
                    if in_doc not in entities_by_file:
                        entities_by_file[in_doc] = []
                    name = row.get("?name", "")
                    action = row.get("?action", "")
                    method = row.get("?method", "GET")
                    entities_by_file[in_doc].append({
                        "entity_type": "form",
                        "name": name or f"form_{action}",
                        "line": int(row.get("?line", 0)),
                        "action": action,
                        "method": method,
                        "source_type": "html",
                    })
        except Exception as e:
            logger.debug(f"[RAG] _query_all_html_entities_bulk: Form query FAILED: {e}")

        # Bulk query framework directives
        for framework, concept in [
            ("vue", "vue-directive"),
            ("angular", "angular-directive"),
            ("htmx", "htmx-attribute"),
            ("alpine", "alpine-directive"),
        ]:
            logger.debug(f"[RAG] _query_all_html_entities_bulk: Querying all {framework} directives...")
            directive_query = f'''
            SELECT DISTINCT ?entity ?directive ?value ?line ?inDocument
            WHERE {{
                ?entity type {concept} .
                ?entity is-in-document ?inDocument .
                OPTIONAL {{ ?entity has-directive-name ?directive }}
                OPTIONAL {{ ?entity has-directive-value ?value }}
                OPTIONAL {{ ?entity is-at-line ?line }}
            }}
            '''
            try:
                directive_table = reter.reql(directive_query)
                if directive_table is not None and directive_table.num_rows > 0:
                    directive_results = directive_table.to_pylist()
                    logger.debug(f"[RAG] _query_all_html_entities_bulk: Found {len(directive_results)} {framework} directives")
                    for row in directive_results:
                        in_doc = row.get("?inDocument", "")
                        if in_doc not in entities_by_file:
                            entities_by_file[in_doc] = []
                        directive = row.get("?directive", "")
                        value = row.get("?value", "")
                        entities_by_file[in_doc].append({
                            "entity_type": f"{framework}_directive",
                            "name": directive,
                            "line": int(row.get("?line", 0)),
                            "content": value,
                            "framework": framework,
                            "source_type": "html",
                        })
            except Exception as e:
                logger.debug(f"[RAG] _query_all_html_entities_bulk: {framework} directive query FAILED: {e}")

        logger.debug(f"[RAG] _query_all_html_entities_bulk: Total {len(entities_by_file)} documents with entities")
        return entities_by_file

    def reindex_all(
        self,
        reter: "ReterWrapper",
        project_root: Path,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> Dict[str, Any]:
        """
        Force complete reindex of all entities.

        Args:
            reter: RETER instance
            project_root: Project root directory
            progress_callback: Optional callback(vectors_done, vectors_total, phase)
                Called periodically during indexing to report progress.

        Returns:
            Reindex statistics

        Raises:
            ComponentNotReadyError: If default RETER instance is not ready
        """
        # Reindexing requires RETER to query Python entities, but not RAG (we're building it)
        require_default_instance()

        if not self._enabled:
            logger.debug("[RAG] reindex_all: RAG is disabled")
            return {"status": "disabled"}

        logger.debug(f"[RAG] reindex_all: Starting reindex for {project_root}")

        # Initialize if not already done
        if not self._initialized:
            logger.debug("[RAG] reindex_all: Initializing RAG index manager...")
            init_start = time.time()
            self.initialize(project_root)
            logger.debug(f"[RAG] reindex_all: Initialization complete in {time.time() - init_start:.2f}s")

        start_time = time.time()

        # Clear existing index
        logger.debug("[RAG] reindex_all: Clearing existing index...")
        self._faiss_wrapper.clear()
        self._metadata["sources"] = {}
        self._metadata["vectors"] = {}
        self._next_vector_id = 0

        # Get all Python sources from RETER
        logger.debug("[RAG] reindex_all: Querying Python sources from RETER...")
        all_sources, _ = reter.get_all_sources()
        python_sources = [s for s in all_sources if s.endswith(".py") or "|" in s]
        logger.debug(f"[RAG] reindex_all: Found {len(python_sources)} Python sources")

        stats = {
            "python_sources": len(python_sources),
            "python_vectors": 0,
            "markdown_files": 0,
            "markdown_vectors": 0,
            "errors": [],
        }

        # BULK query all entities (3 queries instead of 3 per file)
        logger.debug("[RAG] reindex_all: Bulk querying all Python entities...")
        bulk_start = time.time()
        entities_by_file = self._query_all_entities_bulk(reter)
        logger.debug(f"[RAG] reindex_all: Bulk query complete in {time.time() - bulk_start:.2f}s")

        # Build mapping from inFile to source_id
        infile_to_source: Dict[str, str] = {}
        for source_id in python_sources:
            if "|" in source_id:
                _, rel_path = source_id.split("|", 1)
            else:
                rel_path = source_id
            in_file = rel_path.replace("\\", "/")
            infile_to_source[in_file] = source_id

        # Index Python entities by file
        logger.debug(f"[RAG] reindex_all: Indexing entities from {len(entities_by_file)} files...")

        # Estimate total vectors for progress tracking
        total_entities = sum(len(entities) for entities in entities_by_file.values())
        total_files = len(entities_by_file)

        py_start = time.time()
        for i, (in_file, entities) in enumerate(entities_by_file.items()):
            try:
                source_id = infile_to_source.get(in_file)
                if not source_id:
                    # Try to find matching source
                    for sid in python_sources:
                        if "|" in sid:
                            _, rel_path = sid.split("|", 1)
                        else:
                            rel_path = sid
                        if rel_path.replace("\\", ".").replace("/", ".") == in_file:
                            source_id = sid
                            break

                if not source_id:
                    logger.debug(f"[RAG] reindex_all: No source_id found for {in_file}, skipping")
                    continue

                # Log progress every 10 files
                if (i + 1) % 10 == 0 or i == 0:
                    logger.debug(f"[RAG] reindex_all: Python [{i+1}/{len(entities_by_file)}] {in_file} ({len(entities)} entities)")

                added = self._index_python_entities(entities, source_id, project_root)
                stats["python_vectors"] += added
                # Also index comments
                comment_added = self._index_python_comments(entities, source_id, project_root)
                stats["python_vectors"] += comment_added
                # Note: literals are indexed in bulk after the loop for better performance

                # Call progress callback
                if progress_callback and (i + 1) % 5 == 0:
                    progress_callback(stats["python_vectors"], total_entities, "python")

            except Exception as e:
                logger.debug(f"[RAG] reindex_all: ERROR indexing {in_file}: {e}")
                stats["errors"].append(f"Python: {in_file}: {e}")

        py_elapsed = time.time() - py_start
        logger.debug(f"[RAG] reindex_all: Python indexing complete: {stats['python_vectors']} vectors in {py_elapsed:.2f}s")

        # Bulk index all string literals (much faster than per-file)
        logger.debug("[RAG] reindex_all: Bulk indexing string literals...")
        literal_added = self._index_all_python_literals_bulk(reter, project_root)
        stats["python_vectors"] += literal_added

        # Index Markdown files if enabled
        if self._config.get("rag_index_markdown", True):
            logger.debug("[RAG] reindex_all: Scanning for Markdown files...")
            md_files = self._scan_markdown_files(project_root)
            stats["markdown_files"] = len(md_files)
            logger.debug(f"[RAG] reindex_all: Found {len(md_files)} Markdown files")

            if md_files:
                logger.debug(f"[RAG] reindex_all: Indexing {len(md_files)} Markdown files...")
                md_start = time.time()
                for i, rel_path in enumerate(md_files):
                    try:
                        # Log progress every 5 files or at key milestones
                        if (i + 1) % 5 == 0 or i == 0 or i == len(md_files) - 1:
                            logger.debug(f"[RAG] reindex_all: Markdown [{i+1}/{len(md_files)}] {rel_path}")

                        abs_path = project_root / rel_path
                        chunks = self._markdown_indexer.parse_file(str(abs_path))
                        added = self._index_markdown_chunks(chunks, rel_path)
                        stats["markdown_vectors"] += added

                        # Call progress callback for markdown
                        total_current = stats["python_vectors"] + stats["markdown_vectors"]
                        if progress_callback and (i + 1) % 3 == 0:
                            progress_callback(total_current, total_entities + len(md_files) * 3, "markdown")

                    except Exception as e:
                        logger.debug(f"[RAG] reindex_all: ERROR indexing {rel_path}: {e}")
                        stats["errors"].append(f"Markdown: {rel_path}: {e}")

                md_elapsed = time.time() - md_start
                logger.debug(f"[RAG] reindex_all: Markdown indexing complete: {stats['markdown_vectors']} vectors in {md_elapsed:.2f}s")

        # Save index
        logger.debug("[RAG] reindex_all: Saving index to disk...")
        save_start = time.time()
        self._save_index()
        logger.debug(f"[RAG] reindex_all: Index saved in {time.time() - save_start:.2f}s")

        stats["time_ms"] = int((time.time() - start_time) * 1000)
        stats["total_vectors"] = self._faiss_wrapper.total_vectors

        logger.debug(
            f"[RAG] reindex_all: COMPLETE - {stats['python_vectors']} Python + "
            f"{stats['markdown_vectors']} Markdown = {stats['total_vectors']} total vectors "
            f"in {stats['time_ms']}ms"
        )

        logger.info(
            f"Reindex complete: {stats['python_vectors']} Python, "
            f"{stats['markdown_vectors']} Markdown vectors in {stats['time_ms']}ms"
        )

        return stats

    def _scan_markdown_files(self, project_root: Path) -> List[str]:
        """Scan for markdown files to index."""
        include_patterns = self._config.get("rag_markdown_include", "**/*.md")
        if isinstance(include_patterns, str):
            include_patterns = [p.strip() for p in include_patterns.split(",")]

        exclude_patterns = self._config.get("rag_markdown_exclude", "")
        if isinstance(exclude_patterns, str):
            exclude_patterns = [p.strip() for p in exclude_patterns.split(",") if p.strip()]

        md_files = []
        processed = set()

        for pattern in include_patterns:
            for md_file in project_root.glob(pattern):
                if not md_file.is_file():
                    continue

                rel_path = str(md_file.relative_to(project_root)).replace('\\', '/')
                if rel_path in processed:
                    continue

                # Check exclusions
                excluded = False
                for exc_pattern in exclude_patterns:
                    import fnmatch
                    if fnmatch.fnmatch(rel_path, exc_pattern):
                        excluded = True
                        break

                if not excluded:
                    processed.add(rel_path)
                    md_files.append(rel_path)

        return md_files

    def get_indexed_markdown_sources(self) -> Dict[str, str]:
        """
        Get indexed markdown sources.

        Returns:
            Dict mapping rel_path to indexed_at timestamp
        """
        result = {}
        for key, info in self._metadata.get("sources", {}).items():
            if key.startswith("md:"):
                rel_path = key[3:]  # Remove "md:" prefix
                result[rel_path] = info.get("indexed_at", "")
        return result

    def get_status(self) -> Dict[str, Any]:
        """Get RAG index status and statistics."""
        if not self._enabled:
            return {
                "status": "disabled",
                "reason": "RAG is disabled via configuration",
            }

        if not self._initialized:
            # Return basic status without loading embedding model
            model_name = self._config.get(
                "rag_embedding_model",
                "flax-sentence-embeddings/st-codesearch-distilroberta-base"
            )
            return {
                "status": "not_initialized",
                "reason": "RAG index not yet initialized. Call rag_reindex(force=True) to build the index.",
                "embedding_model": model_name,
                "note": "Embedding model will be loaded on first reindex/search",
            }

        # Count by type
        python_count = 0
        javascript_count = 0
        html_count = 0
        csharp_count = 0
        cpp_count = 0
        markdown_count = 0
        entity_counts: Dict[str, int] = {}

        for vid, meta in self._metadata.get("vectors", {}).items():
            source_type = meta.get("source_type", "python")
            entity_type = meta.get("entity_type", "unknown")

            if source_type == "python" or source_type == "python_literal":
                python_count += 1
            elif source_type == "javascript" or source_type == "javascript_literal":
                javascript_count += 1
            elif source_type == "html":
                html_count += 1
            elif source_type == "csharp":
                csharp_count += 1
            elif source_type == "cpp":
                cpp_count += 1
            else:
                markdown_count += 1

            entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1

        # Count sources
        python_sources = sum(
            1 for info in self._metadata.get("sources", {}).values()
            if info.get("file_type") == "python"
        )
        javascript_sources = sum(
            1 for info in self._metadata.get("sources", {}).values()
            if info.get("file_type") == "javascript"
        )
        html_sources = sum(
            1 for info in self._metadata.get("sources", {}).values()
            if info.get("file_type") == "html"
        )
        csharp_sources = sum(
            1 for info in self._metadata.get("sources", {}).values()
            if info.get("file_type") == "csharp"
        )
        cpp_sources = sum(
            1 for info in self._metadata.get("sources", {}).values()
            if info.get("file_type") == "cpp"
        )
        markdown_sources = sum(
            1 for info in self._metadata.get("sources", {}).values()
            if info.get("file_type") == "markdown"
        )

        # Get index file size
        index_size_mb = 0
        if self._index_path and self._index_path.exists():
            index_size_mb = self._index_path.stat().st_size / (1024 * 1024)

        return {
            "status": "ready",
            "embedding_model": self._embedding_service.model_name,
            "embedding_provider": self._embedding_service.provider,
            "embedding_dimension": self._embedding_service.embedding_dim,
            "total_vectors": self._faiss_wrapper.total_vectors if self._faiss_wrapper else 0,
            "index_size_mb": round(index_size_mb, 2),
            "created_at": self._metadata.get("created_at"),
            "updated_at": self._metadata.get("updated_at"),
            "python_sources": {
                "files_indexed": python_sources,
                "total_vectors": python_count,
            },
            "javascript_sources": {
                "files_indexed": javascript_sources,
                "total_vectors": javascript_count,
            },
            "html_sources": {
                "files_indexed": html_sources,
                "total_vectors": html_count,
            },
            "csharp_sources": {
                "files_indexed": csharp_sources,
                "total_vectors": csharp_count,
            },
            "cpp_sources": {
                "files_indexed": cpp_sources,
                "total_vectors": cpp_count,
            },
            "markdown_sources": {
                "files_indexed": markdown_sources,
                "total_vectors": markdown_count,
            },
            "entity_counts": entity_counts,
            "cache_info": self._embedding_service.get_info() if self._embedding_service else {},
        }
