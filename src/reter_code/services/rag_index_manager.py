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


# Language configuration for RAG indexing.
# Each entry: (key, prefix, display_name, extensions)
# - key: language key used in SyncChanges, stats, and JSON storage
# - prefix: prefix used in _indexed_files dict (empty string = no prefix, i.e. Python)
# - display_name: human-readable name for logging
# - extensions: tuple of file extensions for this language
RAG_LANGUAGE_CONFIG = [
    ("python",     "",        "Python",       (".py",)),
    ("javascript", "js:",     "JavaScript",   (".js", ".ts", ".jsx", ".tsx", ".mjs")),
    ("html",       "html:",   "HTML",         (".html", ".htm")),
    ("csharp",     "cs:",     "C#",           (".cs",)),
    ("cpp",        "cpp:",    "C++",          (".cpp", ".cc", ".cxx", ".hpp", ".h")),
    ("java",       "java:",   "Java",         (".java",)),
    ("go",         "go:",     "Go",           (".go",)),
    ("rust",       "rust:",   "Rust",         (".rs",)),
    ("erlang",     "erlang:", "Erlang",       (".erl", ".hrl")),
    ("php",        "php:",    "PHP",          (".php",)),
    ("objc",       "objc:",   "Objective-C",  (".m", ".mm")),
    ("swift",      "swift:",  "Swift",        (".swift",)),
    ("vb6",        "vb6:",    "VB6",          (".bas", ".cls", ".frm")),
    ("scala",      "scala:",  "Scala",        (".scala",)),
    ("haskell",    "haskell:","Haskell",      (".hs", ".lhs")),
]

# Build derived data structures from config
RAG_ALL_CODE_EXTENSIONS = tuple(
    ext for _, _, _, exts in RAG_LANGUAGE_CONFIG for ext in exts
)
RAG_ALL_PREFIXES = tuple(
    prefix for _, prefix, _, _ in RAG_LANGUAGE_CONFIG if prefix
)
# Map extension -> (key, prefix) for fast lookup
RAG_EXT_TO_LANG = {}
for _key, _prefix, _, _exts in RAG_LANGUAGE_CONFIG:
    for _ext in _exts:
        RAG_EXT_TO_LANG[_ext] = (_key, _prefix)


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
            for source_id in all_sources:
                if "|" in source_id:
                    md5_hash, rel_path = source_id.split("|", 1)
                    if rel_path.endswith(RAG_ALL_CODE_EXTENSIONS):
                        rel_path_normalized = rel_path.replace("\\", "/")
                        current_code[rel_path_normalized] = md5_hash

            # Get indexed code files
            # Must strip language prefixes to match RETER source paths
            indexed_code: Dict[str, str] = {}  # rel_path -> md5
            for key, md5 in self._indexed_files.items():
                if key.startswith("md:"):
                    continue  # Skip markdown
                # Strip file type prefixes
                clean_key = key
                for prefix in RAG_ALL_PREFIXES:
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
            Keys may have language prefixes (e.g. js:, java:, go:) or md: for markdown.
            Python files have no prefix.
        """
        if not self._rag_files_path or not self._rag_files_path.exists():
            return {}

        try:
            with open(self._rag_files_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                result = {}
                # Load all languages using config
                for key, prefix, _, _ in RAG_LANGUAGE_CONFIG:
                    lang_files = data.get(key, {})
                    if prefix:
                        result.update({f"{prefix}{k}": v for k, v in lang_files.items()})
                    else:
                        # Python has no prefix
                        result.update(lang_files)
                # Markdown files get md: prefix
                result.update({f"md:{k}": v for k, v in data.get("markdown", {}).items()})

                counts = ", ".join(
                    f"{key}={len(data.get(key, {}))}" for key, _, _, _ in RAG_LANGUAGE_CONFIG
                )
                logger.debug(f"[RAG] _load_rag_files: Loaded {len(result)} indexed files "
                         f"({counts}, md={len(data.get('markdown', {}))})")
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
        # Build prefix->key lookup for splitting
        lang_files: Dict[str, Dict[str, str]] = {key: {} for key, _, _, _ in RAG_LANGUAGE_CONFIG}
        markdown_files: Dict[str, str] = {}

        for file_key, md5 in self._indexed_files.items():
            if file_key.startswith("md:"):
                markdown_files[file_key[3:]] = md5
            else:
                matched = False
                for lang_key, prefix, _, _ in RAG_LANGUAGE_CONFIG:
                    if prefix and file_key.startswith(prefix):
                        lang_files[lang_key][file_key[len(prefix):]] = md5
                        matched = True
                        break
                if not matched:
                    # Python has no prefix
                    lang_files["python"][file_key] = md5

        data: Dict[str, Any] = {
            "version": "1.2",
            "updated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        }
        for lang_key, _, _, _ in RAG_LANGUAGE_CONFIG:
            data[lang_key] = lang_files[lang_key]
        data["markdown"] = markdown_files

        counts = ", ".join(f"{k}={len(lang_files[k])}" for k, _, _, _ in RAG_LANGUAGE_CONFIG)
        logger.debug(f"[RAG] _save_rag_files: Saving {counts}, md={len(markdown_files)}")

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
        stats: Dict[str, Any] = {"errors": []}
        for key, _, _, _ in RAG_LANGUAGE_CONFIG:
            stats[f"{key}_vectors_added"] = 0
            stats[f"{key}_vectors_removed"] = 0
        stats["markdown_vectors_added"] = 0
        stats["markdown_vectors_removed"] = 0

        # Update content extractor project root
        self._content_extractor.project_root = project_root

        # 1. Remove vectors for deleted sources (all languages)
        for key, _, _, _ in RAG_LANGUAGE_CONFIG:
            lang_changes = getattr(changes, key, None)
            if lang_changes:
                for source_id in lang_changes.deleted:
                    removed = self._remove_vectors_for_source(source_id)
                    stats[f"{key}_vectors_removed"] += removed

        for rel_path in changes.markdown.deleted:
            removed = self._remove_vectors_for_markdown(rel_path)
            stats["markdown_vectors_removed"] += removed

        # 2. Re-remove vectors for changed sources (they'll be re-indexed)
        for key, _, _, _ in RAG_LANGUAGE_CONFIG:
            lang_changes = getattr(changes, key, None)
            if lang_changes:
                for source_id in lang_changes.changed:
                    self._remove_vectors_for_source(source_id)

        for rel_path in changes.markdown.changed:
            self._remove_vectors_for_markdown(rel_path)

        # 3. BATCHED INDEXING: Collect all texts first, then generate embeddings in one batch
        all_texts = []
        all_metadata = []
        source_tracking = []  # (source_id, file_type, start_idx, count)

        # 3a. Collect Python entities, comments, and literals (special handling)
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

        # 3c. Collect JavaScript entities and literals (special handling)
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

        if changes.javascript.changed:
            js_literal_texts, js_literal_metadata = self._collect_all_javascript_literals_bulk(
                reter, project_root, changed_sources=changes.javascript.changed
            )
            if js_literal_texts:
                source_tracking.append(("javascript_literals_bulk", "javascript_literal", len(all_texts), len(js_literal_texts)))
                all_texts.extend(js_literal_texts)
                all_metadata.extend(js_literal_metadata)

        # 3d. Collect HTML entities (special query)
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

        # 3e. Collect entities for all other code languages (generic pattern)
        _generic_langs = [
            ("csharp", "C#"), ("cpp", "C++"), ("java", "Java"), ("go", "Go"),
            ("rust", "Rust"), ("erlang", "Erlang"), ("php", "PHP"),
            ("objc", "Objective-C"), ("swift", "Swift"), ("vb6", "VB6"),
            ("scala", "Scala"), ("haskell", "Haskell"),
        ]
        for lang_key, lang_display in _generic_langs:
            lang_changes = getattr(changes, lang_key, None)
            if not lang_changes or not lang_changes.changed:
                continue
            collect_fn = getattr(self, f"_collect_{lang_key}_entities", None)
            if not collect_fn:
                continue
            for source_id in lang_changes.changed:
                try:
                    entities = self._query_entities_for_source(reter, source_id, language=lang_key)
                    texts, metadata = collect_fn(entities, source_id, project_root, self._chunk_config)
                    if texts:
                        source_tracking.append((source_id, lang_key, len(all_texts), len(texts)))
                        all_texts.extend(texts)
                        all_metadata.extend(metadata)
                except Exception as e:
                    logger.error(f"Error collecting {lang_display} {source_id}: {e}")
                    stats["errors"].append(f"{lang_display}: {source_id}: {e}")

        # 3f. Collect Markdown chunks
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
                if file_type == "markdown":
                    self._indexed_files[f"md:{rel_path}"] = md5_hash
                elif md5_hash:
                    # Map file_type to its base language key
                    base_type = file_type.split("_")[0]  # "python_comment" -> "python"
                    # Find the prefix for this language
                    matched_prefix = ""
                    for lk, lp, _, _ in RAG_LANGUAGE_CONFIG:
                        if lk == base_type:
                            matched_prefix = lp
                            break
                    self._indexed_files[f"{matched_prefix}{rel_path}"] = md5_hash

                # Update stats
                base_type = file_type.split("_")[0]  # "python_comment" -> "python"
                stat_key = f"{base_type}_vectors_added"
                if stat_key in stats:
                    stats[stat_key] += count
                elif file_type == "markdown":
                    stats["markdown_vectors_added"] += count

        # 5. Save index and metadata
        self._save_index()
        self._save_rag_files()  # Also save _indexed_files to keep sync status accurate

        stats["time_ms"] = int((time.time() - start_time) * 1000)
        total_added = sum(stats.get(f"{k}_vectors_added", 0) for k, _, _, _ in RAG_LANGUAGE_CONFIG) + stats["markdown_vectors_added"]
        total_removed = sum(stats.get(f"{k}_vectors_removed", 0) for k, _, _, _ in RAG_LANGUAGE_CONFIG) + stats["markdown_vectors_removed"]
        # Log per-language stats for languages with activity
        active_parts = []
        for key, _, display, _ in RAG_LANGUAGE_CONFIG:
            added = stats.get(f"{key}_vectors_added", 0)
            if added > 0:
                active_parts.append(f"+{added} {display}")
        if stats["markdown_vectors_added"] > 0:
            active_parts.append(f"+{stats['markdown_vectors_added']} Markdown")
        if total_removed > 0:
            active_parts.append(f"-{total_removed} removed")
        logger.info(
            f"RAG sync: {', '.join(active_parts) if active_parts else 'no changes'} "
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
        all_prefixes_plus_md = tuple(p for _, p, _, _ in RAG_LANGUAGE_CONFIG if p) + ("md:",)
        py_count = sum(1 for k in self._indexed_files if not k.startswith(all_prefixes_plus_md))
        md_count = sum(1 for k in self._indexed_files if k.startswith("md:"))
        logger.debug(f"[RAG] sync_sources: Loaded {len(self._indexed_files)} indexed files from JSON "
                 f"(py={py_count}, other_code={len(self._indexed_files) - py_count - md_count}, md={md_count})")

        stats: Dict[str, Any] = {"errors": []}
        for key, _, _, _ in RAG_LANGUAGE_CONFIG:
            stats[f"{key}_added"] = 0
            stats[f"{key}_removed"] = 0
            stats[f"{key}_unchanged"] = 0
        stats["markdown_added"] = 0
        stats["markdown_removed"] = 0
        stats["markdown_unchanged"] = 0

        # Get current sources from RETER (already loaded by BackgroundInitializer)
        all_sources, _ = reter.get_all_sources()
        # Per-language current file tracking: key -> {rel_path: (source_id, md5)}
        current_by_lang: Dict[str, Dict[str, Tuple[str, str]]] = {
            key: {} for key, _, _, _ in RAG_LANGUAGE_CONFIG
        }

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

                # Determine language from extension
                ext = Path(rel_path_normalized).suffix.lower()
                lang_info = RAG_EXT_TO_LANG.get(ext)
                if not lang_info:
                    continue
                lang_key, lang_prefix = lang_info

                # OPTIMIZATION: If targeted sync and file not in changed_files,
                # use cached MD5 instead of computing (assume unchanged)
                if changed_files and rel_path_normalized not in changed_files:
                    cached_md5 = self._indexed_files.get(f"{lang_prefix}{rel_path_normalized}")
                    if cached_md5:
                        current_by_lang[lang_key][rel_path_normalized] = (source_id, cached_md5)
                    continue

                # Compute actual MD5 for this file
                actual_md5 = compute_file_md5(project_root / rel_path_normalized)
                if actual_md5:
                    current_by_lang[lang_key][rel_path_normalized] = (source_id, actual_md5)

        # Get indexed files from _indexed_files (split by language prefix)
        indexed_by_lang: Dict[str, Dict[str, str]] = {key: {} for key, _, _, _ in RAG_LANGUAGE_CONFIG}
        for file_key, md5 in self._indexed_files.items():
            if file_key.startswith("md:"):
                pass  # Handled separately below
            else:
                matched = False
                for lang_key, prefix, _, _ in RAG_LANGUAGE_CONFIG:
                    if prefix and file_key.startswith(prefix):
                        indexed_by_lang[lang_key][file_key[len(prefix):]] = md5
                        matched = True
                        break
                if not matched:
                    # Legacy: Python files without prefix
                    indexed_by_lang["python"][file_key] = md5

        # Find changes for all code languages
        to_add_by_lang: Dict[str, List[Tuple[str, str, str]]] = {key: [] for key, _, _, _ in RAG_LANGUAGE_CONFIG}
        to_remove_by_lang: Dict[str, List[str]] = {key: [] for key, _, _, _ in RAG_LANGUAGE_CONFIG}

        for lang_key, _, lang_display, _ in RAG_LANGUAGE_CONFIG:
            current_lang = current_by_lang[lang_key]
            indexed_lang = indexed_by_lang[lang_key]
            logger.debug(f"[RAG] sync_sources: {lang_display} - {len(current_lang)} current, {len(indexed_lang)} indexed")

            for rel_path, (source_id, current_md5) in current_lang.items():
                indexed_md5 = indexed_lang.get(rel_path)
                if indexed_md5 is None:
                    to_add_by_lang[lang_key].append((source_id, rel_path, current_md5))
                elif current_md5 != indexed_md5:
                    to_remove_by_lang[lang_key].append(rel_path)
                    to_add_by_lang[lang_key].append((source_id, rel_path, current_md5))
                else:
                    stats[f"{lang_key}_unchanged"] += 1

            for rel_path in indexed_lang:
                if rel_path not in current_lang:
                    to_remove_by_lang[lang_key].append(rel_path)

            if to_add_by_lang[lang_key] or to_remove_by_lang[lang_key]:
                logger.debug(f"[RAG] sync_sources: {lang_display} changes - +{len(to_add_by_lang[lang_key])} -{len(to_remove_by_lang[lang_key])} ={stats[f'{lang_key}_unchanged']}")

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
        total_changes = sum(
            len(to_add_by_lang[k]) + len(to_remove_by_lang[k]) for k, _, _, _ in RAG_LANGUAGE_CONFIG
        ) + len(markdown_to_add) + len(markdown_to_remove)

        logger.debug(f"[RAG] sync_sources: {total_changes} changes detected")

        # Estimate total files to index for progress reporting
        total_files_to_index = sum(
            len(to_add_by_lang[k]) for k, _, _, _ in RAG_LANGUAGE_CONFIG
        ) + len(markdown_to_add)
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

        # Remove vectors for deleted/modified code files (all languages)
        for lang_key, lang_prefix, _, _ in RAG_LANGUAGE_CONFIG:
            for rel_path in to_remove_by_lang[lang_key]:
                for source_id, info in list(self._metadata["sources"].items()):
                    if info.get("file_type") == lang_key and info.get("rel_path", "").replace("\\", "/") == rel_path:
                        removed = self._remove_vectors_for_source(source_id)
                        stats[f"{lang_key}_removed"] += removed
                        break
                self._indexed_files.pop(f"{lang_prefix}{rel_path}", None)

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

        # --- Phase 1: Collect Python entities, comments, and literals (special handling) ---
        python_to_add = to_add_by_lang["python"]
        if python_to_add:
            logger.debug(f"[RAG] sync_sources: Querying entities for {len(python_to_add)} Python files...")
            entities_by_file = self._query_all_entities_bulk(reter)
            total_python = len(python_to_add)

            for i, (source_id, rel_path, md5_hash) in enumerate(python_to_add):
                try:
                    in_file = rel_path.replace("\\", "/")
                    entities = entities_by_file.get(in_file, [])
                    if entities:
                        texts, metadata = self._prepare_python_entities(entities, source_id, project_root)
                        if texts:
                            source_tracking.append((source_id, "python", len(all_texts), len(texts)))
                            all_texts.extend(texts)
                            all_metadata.extend(metadata)

                        c_texts, c_meta = self._prepare_python_comments(entities, source_id, project_root)
                        if c_texts:
                            source_tracking.append((f"comments:{source_id}", "python_comment", len(all_texts), len(c_texts)))
                            all_texts.extend(c_texts)
                            all_metadata.extend(c_meta)

                    self._indexed_files[rel_path] = md5_hash

                    if progress_callback and (i + 1) % 10 == 0:
                        progress_callback(i + 1, total_python, "collecting_python")
                    elif not progress_callback and (i + 1) % 10 == 0:
                        logger.info(f"[RAG] Collecting Python: {i+1}/{total_python}")

                except Exception as e:
                    import traceback
                    logger.debug(f"[RAG] sync_sources: Error collecting {source_id}: {e}\n{traceback.format_exc()}")
                    stats["errors"].append(f"Python: {source_id}: {e}")

            # Collect Python literals
            changed_source_ids = [source_id for source_id, _, _ in python_to_add]
            literal_texts, literal_metadata = self._collect_all_python_literals_bulk(
                reter, project_root, changed_sources=changed_source_ids
            )
            if literal_texts:
                source_tracking.append(("python_literals_bulk", "python_literal", len(all_texts), len(literal_texts)))
                all_texts.extend(literal_texts)
                all_metadata.extend(literal_metadata)

        # --- Phase 2: Collect JavaScript entities and literals (special handling) ---
        javascript_to_add = to_add_by_lang["javascript"]
        if javascript_to_add:
            logger.debug(f"[RAG] sync_sources: Querying entities for {len(javascript_to_add)} JavaScript files...")
            js_entities_by_file = self._query_all_entities_bulk(reter, language="javascript")

            for i, (source_id, rel_path, md5_hash) in enumerate(javascript_to_add):
                try:
                    in_file = rel_path.replace("\\", "/")
                    entities = js_entities_by_file.get(in_file, [])
                    if entities:
                        texts, metadata = self._collect_javascript_entities(entities, source_id, project_root, self._chunk_config)
                        if texts:
                            source_tracking.append((source_id, "javascript", len(all_texts), len(texts)))
                            all_texts.extend(texts)
                            all_metadata.extend(metadata)

                    self._indexed_files[f"js:{rel_path}"] = md5_hash

                    if progress_callback and (i + 1) % 10 == 0:
                        progress_callback(i + 1, len(javascript_to_add), "collecting_javascript")

                except Exception as e:
                    import traceback
                    logger.debug(f"[RAG] sync_sources: Error collecting JavaScript {source_id}: {e}\n{traceback.format_exc()}")
                    stats["errors"].append(f"JavaScript: {source_id}: {e}")

            # Collect JavaScript literals
            js_literal_texts, js_literal_metadata = self._collect_all_javascript_literals_bulk(reter, project_root)
            if js_literal_texts:
                source_tracking.append(("javascript_literals_bulk", "javascript_literal", len(all_texts), len(js_literal_texts)))
                all_texts.extend(js_literal_texts)
                all_metadata.extend(js_literal_metadata)

        # --- Phase 3: Collect HTML entities (special query) ---
        html_to_add = to_add_by_lang["html"]
        if html_to_add:
            logger.debug(f"[RAG] sync_sources: Querying entities for {len(html_to_add)} HTML files...")
            html_entities_by_file = self._query_all_html_entities_bulk(reter)

            for i, (source_id, rel_path, md5_hash) in enumerate(html_to_add):
                try:
                    in_doc = rel_path.replace("\\", ".").replace("/", ".")
                    entities = html_entities_by_file.get(in_doc, [])
                    if entities:
                        texts, metadata = self._collect_html_entities(entities, source_id, project_root)
                        if texts:
                            source_tracking.append((source_id, "html", len(all_texts), len(texts)))
                            all_texts.extend(texts)
                            all_metadata.extend(metadata)

                    self._indexed_files[f"html:{rel_path}"] = md5_hash

                    if progress_callback and (i + 1) % 10 == 0:
                        progress_callback(i + 1, len(html_to_add), "collecting_html")

                except Exception as e:
                    import traceback
                    logger.debug(f"[RAG] sync_sources: Error collecting HTML {source_id}: {e}\n{traceback.format_exc()}")
                    stats["errors"].append(f"HTML: {source_id}: {e}")

        # --- Phase 4: Collect entities for all other code languages (generic pattern) ---
        _generic_sync_langs = [
            ("csharp", "cs:", "C#"), ("cpp", "cpp:", "C++"),
            ("java", "java:", "Java"), ("go", "go:", "Go"),
            ("rust", "rust:", "Rust"), ("erlang", "erlang:", "Erlang"),
            ("php", "php:", "PHP"), ("objc", "objc:", "Objective-C"),
            ("swift", "swift:", "Swift"), ("vb6", "vb6:", "VB6"),
            ("scala", "scala:", "Scala"), ("haskell", "haskell:", "Haskell"),
        ]
        for lang_key, lang_prefix, lang_display in _generic_sync_langs:
            lang_to_add = to_add_by_lang[lang_key]
            if not lang_to_add:
                continue

            logger.debug(f"[RAG] sync_sources: Querying entities for {len(lang_to_add)} {lang_display} files...")
            lang_entities_by_file = self._query_all_entities_bulk(reter, language=lang_key)
            collect_fn = getattr(self, f"_collect_{lang_key}_entities", None)
            if not collect_fn:
                continue

            for i, (source_id, rel_path, md5_hash) in enumerate(lang_to_add):
                try:
                    in_file = rel_path.replace("\\", "/")
                    entities = lang_entities_by_file.get(in_file, [])
                    if entities:
                        texts, metadata = collect_fn(entities, source_id, project_root, self._chunk_config)
                        if texts:
                            source_tracking.append((source_id, lang_key, len(all_texts), len(texts)))
                            all_texts.extend(texts)
                            all_metadata.extend(metadata)

                    self._indexed_files[f"{lang_prefix}{rel_path}"] = md5_hash

                    if progress_callback and (i + 1) % 10 == 0:
                        progress_callback(i + 1, len(lang_to_add), f"collecting_{lang_key}")
                    elif not progress_callback and (i + 1) % 10 == 0:
                        logger.info(f"[RAG] Collecting {lang_display}: {i+1}/{len(lang_to_add)}")

                except Exception as e:
                    import traceback
                    logger.debug(f"[RAG] sync_sources: Error collecting {lang_display} {source_id}: {e}\n{traceback.format_exc()}")
                    stats["errors"].append(f"{lang_display}: {source_id}: {e}")

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
                base_type = file_type.split("_")[0]  # "python_comment" -> "python"
                stat_key = f"{base_type}_added"
                if stat_key in stats:
                    stats[stat_key] += count
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

        total_added = sum(stats.get(f"{k}_added", 0) for k, _, _, _ in RAG_LANGUAGE_CONFIG) + stats["markdown_added"]
        total_removed = sum(stats.get(f"{k}_removed", 0) for k, _, _, _ in RAG_LANGUAGE_CONFIG) + stats["markdown_removed"]
        total_unchanged = sum(stats.get(f"{k}_unchanged", 0) for k, _, _, _ in RAG_LANGUAGE_CONFIG) + stats["markdown_unchanged"]
        logger.debug(
            f"[RAG] sync_sources: COMPLETE - total={stats['total_vectors']} "
            f"(+{total_added} -{total_removed} ={total_unchanged}) in {stats['time_ms']}ms"
        )
        logger.info(
            f"RAG sync: {stats['total_vectors']} vectors "
            f"(+{total_added} -{total_removed} unchanged={total_unchanged}) in {stats['time_ms']}ms"
        )
        if not progress_callback:
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
        _lang_to_prefix = {
            "python": "py", "javascript": "js", "csharp": "cs", "cpp": "cpp",
            "java": "java", "go": "go", "rust": "rust", "erlang": "erlang",
            "php": "php", "objc": "objc", "swift": "swift", "vb6": "vb6",
            "scala": "scala", "haskell": "haskell",
        }
        prefix = _lang_to_prefix.get(language, "py")

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
            language: Language key (e.g. "python", "javascript", "java", "go", "rust", etc.)

        Returns dict mapping inFile -> list of entities
        """
        entities_by_file: Dict[str, List[Dict[str, Any]]] = {}
        seen_entities: set = set()  # Track seen qualified_names to avoid duplicates

        # Determine concept prefix based on language
        _lang_to_prefix = {
            "python": "py", "javascript": "js", "csharp": "cs", "cpp": "cpp",
            "java": "java", "go": "go", "rust": "rust", "erlang": "erlang",
            "php": "php", "objc": "objc", "swift": "swift", "vb6": "vb6",
            "scala": "scala", "haskell": "haskell",
        }
        prefix = _lang_to_prefix.get(language, "py")

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

        # Get all sources from RETER, classify by language
        logger.debug("[RAG] reindex_all: Querying sources from RETER...")
        all_sources, _ = reter.get_all_sources()
        sources_by_lang: Dict[str, List[str]] = {key: [] for key, _, _, _ in RAG_LANGUAGE_CONFIG}

        for source_id in all_sources:
            if "|" in source_id:
                _, rel_path = source_id.split("|", 1)
                ext = Path(rel_path).suffix.lower()
                lang_info = RAG_EXT_TO_LANG.get(ext)
                if lang_info:
                    sources_by_lang[lang_info[0]].append(source_id)

        stats: Dict[str, Any] = {"errors": []}
        for key, _, _, _ in RAG_LANGUAGE_CONFIG:
            stats[f"{key}_sources"] = len(sources_by_lang[key])
            stats[f"{key}_vectors"] = 0
        stats["markdown_files"] = 0
        stats["markdown_vectors"] = 0

        total_vectors = 0

        # --- Index each code language ---
        for lang_key, lang_prefix, lang_display, _ in RAG_LANGUAGE_CONFIG:
            lang_sources = sources_by_lang[lang_key]
            if not lang_sources:
                continue

            logger.debug(f"[RAG] reindex_all: Bulk querying {lang_display} entities ({len(lang_sources)} sources)...")
            bulk_start = time.time()
            entities_by_file = self._query_all_entities_bulk(reter, language=lang_key)
            logger.debug(f"[RAG] reindex_all: {lang_display} bulk query complete in {time.time() - bulk_start:.2f}s")

            # Build mapping from inFile to source_id
            infile_to_source: Dict[str, str] = {}
            for source_id in lang_sources:
                if "|" in source_id:
                    _, rel_path = source_id.split("|", 1)
                else:
                    rel_path = source_id
                in_file = rel_path.replace("\\", "/")
                infile_to_source[in_file] = source_id

            lang_start = time.time()
            for i, (in_file, entities) in enumerate(entities_by_file.items()):
                try:
                    source_id = infile_to_source.get(in_file)
                    if not source_id:
                        for sid in lang_sources:
                            if "|" in sid:
                                _, rel_path = sid.split("|", 1)
                            else:
                                rel_path = sid
                            if rel_path.replace("\\", ".").replace("/", ".") == in_file:
                                source_id = sid
                                break
                    if not source_id:
                        continue

                    added = self._index_entities(entities, source_id, project_root, lang_key)
                    stats[f"{lang_key}_vectors"] += added

                    # Python-specific: also index comments
                    if lang_key == "python":
                        comment_added = self._index_python_comments(entities, source_id, project_root)
                        stats["python_vectors"] += comment_added

                    if progress_callback and (i + 1) % 5 == 0:
                        progress_callback(stats[f"{lang_key}_vectors"], len(entities_by_file), lang_key)

                except Exception as e:
                    stats["errors"].append(f"{lang_display}: {in_file}: {e}")

            lang_elapsed = time.time() - lang_start
            logger.debug(f"[RAG] reindex_all: {lang_display} indexing: {stats[f'{lang_key}_vectors']} vectors in {lang_elapsed:.2f}s")
            total_vectors += stats[f"{lang_key}_vectors"]

        # Bulk index Python string literals
        if sources_by_lang["python"]:
            logger.debug("[RAG] reindex_all: Bulk indexing Python string literals...")
            literal_added = self._index_all_python_literals_bulk(reter, project_root)
            stats["python_vectors"] += literal_added
            total_vectors += literal_added

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

        # Build summary of active languages
        active_parts = []
        for key, _, display, _ in RAG_LANGUAGE_CONFIG:
            v = stats.get(f"{key}_vectors", 0)
            if v > 0:
                active_parts.append(f"{v} {display}")
        if stats["markdown_vectors"] > 0:
            active_parts.append(f"{stats['markdown_vectors']} Markdown")

        logger.debug(
            f"[RAG] reindex_all: COMPLETE - {', '.join(active_parts)} = {stats['total_vectors']} total "
            f"in {stats['time_ms']}ms"
        )
        logger.info(f"Reindex complete: {', '.join(active_parts)} = {stats['total_vectors']} vectors in {stats['time_ms']}ms")

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

        # Count vectors by source type
        vector_counts: Dict[str, int] = {}
        entity_counts: Dict[str, int] = {}

        for vid, meta in self._metadata.get("vectors", {}).items():
            source_type = meta.get("source_type", "python")
            entity_type = meta.get("entity_type", "unknown")
            # Map sub-types to base language: "python_literal" -> "python"
            base_type = source_type.split("_")[0]
            vector_counts[base_type] = vector_counts.get(base_type, 0) + 1
            entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1

        # Count sources by file_type
        source_counts: Dict[str, int] = {}
        for info in self._metadata.get("sources", {}).values():
            ft = info.get("file_type", "")
            base_ft = ft.split("_")[0]
            source_counts[base_ft] = source_counts.get(base_ft, 0) + 1

        # Get index file size
        index_size_mb = 0
        if self._index_path and self._index_path.exists():
            index_size_mb = self._index_path.stat().st_size / (1024 * 1024)

        result: Dict[str, Any] = {
            "status": "ready",
            "embedding_model": self._embedding_service.model_name,
            "embedding_provider": self._embedding_service.provider,
            "embedding_dimension": self._embedding_service.embedding_dim,
            "total_vectors": self._faiss_wrapper.total_vectors if self._faiss_wrapper else 0,
            "index_size_mb": round(index_size_mb, 2),
            "created_at": self._metadata.get("created_at"),
            "updated_at": self._metadata.get("updated_at"),
        }

        # Add per-language stats (only for languages with data)
        for key, _, _, _ in RAG_LANGUAGE_CONFIG:
            vc = vector_counts.get(key, 0)
            sc = source_counts.get(key, 0)
            if vc > 0 or sc > 0:
                result[f"{key}_sources"] = {
                    "files_indexed": sc,
                    "total_vectors": vc,
                }
        # Always include markdown
        result["markdown_sources"] = {
            "files_indexed": source_counts.get("markdown", 0),
            "total_vectors": vector_counts.get("markdown", 0),
        }
        result["entity_counts"] = entity_counts
        result["cache_info"] = self._embedding_service.get_info() if self._embedding_service else {}

        return result
