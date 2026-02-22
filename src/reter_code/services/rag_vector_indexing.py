"""
RAG Vector Indexing Mixin

Contains methods for indexing code entities, comments, and markdown into the FAISS vector store.
Extracted from RAGIndexManager to reduce file size.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, TYPE_CHECKING

import numpy as np

from .markdown_indexer import MarkdownChunk

if TYPE_CHECKING:
    from ..reter_wrapper import ReterWrapper

from ..logging_config import configure_logger_for_debug_trace

logger = configure_logger_for_debug_trace(__name__)


class RAGVectorIndexingMixin:
    """
    Mixin providing vector indexing methods for RAGIndexManager.

    ::: This is-in-layer Service-Layer.
    ::: This is-part-of-component RAG-Index.
    ::: This is-in-process Main-Process.
    ::: This is stateless.

    These methods require:
    - self._reter: ReterWrapper instance
    - self._faiss_wrapper: FAISSWrapper instance
    - self._embedding_service: EmbeddingService instance
    - self._metadata: dict mapping vector IDs to entity metadata
    - self._content_extractor: ContentExtractor instance
    """

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
                    "indexed_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                    "vector_ids": vids,
                }
            else:
                # Append to existing vector_ids
                self._metadata["sources"][source_id]["vector_ids"].extend(vids)

        return len(texts)

    def _prepare_comments(
        self,
        entities: List[Dict[str, Any]],
        source_id: str,
        project_root: Path,
        language: str = "python"
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Prepare comments for indexing (extract texts and metadata without generating embeddings).

        Args:
            entities: List of entity dicts for context
            source_id: Source identifier (format: "md5|rel_path")
            project_root: Project root path
            language: Source language (python, java, cpp, etc.)

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
                entity_locations=entity_locations,
                language=language
            )
        except Exception:
            return [], []

        if not comments:
            return [], []

        # Only index special comments (TODO, FIXME, etc.)
        SPECIAL_COMMENT_TYPES = {'todo', 'fixme', 'bug', 'hack', 'note', 'warning', 'review', 'optimize', 'xxx'}

        source_type = f"{language}_comment"
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
                "source_type": source_type,
                "comment_type": comment.comment_type,
                "context_entity": comment.context_entity,
                "is_standalone": comment.is_standalone,
                "content_preview": comment.content[:100] if comment.content else "",
                "source_id": f"comments:{source_id}",
            })

        return texts, comment_metadata

    # Backward-compatible aliases
    _prepare_python_comments = _prepare_comments
    _collect_python_comments = _prepare_comments
    _collect_comments = _prepare_comments

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
            "indexed_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
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
            "indexed_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "vector_ids": vector_ids,
        }

        logger.debug(f"[RAG] _index_html_entities: Indexed {len(texts)} entities for {rel_path}")
        return len(texts)

    def _index_comments(
        self,
        entities: List[Dict[str, Any]],
        source_id: str,
        project_root: Path,
        language: str = "python"
    ) -> int:
        """
        Index comments into FAISS for any language.

        Args:
            entities: List of code entities for context
            source_id: Source ID (format: "md5|rel_path")
            project_root: Project root directory
            language: Source language (python, java, cpp, etc.)

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
            entity_locations=entity_locations,
            language=language
        )

        if not comments:
            return 0

        # Only index special comments (TODO, FIXME, etc.), not regular inline/block comments
        # Regular comments should only be part of entity bodies, not indexed separately
        SPECIAL_COMMENT_TYPES = {'todo', 'fixme', 'bug', 'hack', 'note', 'warning', 'review', 'optimize', 'xxx'}

        source_type = f"{language}_comment"

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
                "source_type": source_type,
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
            "file_type": source_type,
            "md5": md5_hash,
            "rel_path": rel_path,
            "indexed_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "vector_ids": vector_ids,
        }

        return len(texts)

    # Backward-compatible alias
    _index_python_comments = _index_comments

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
            "indexed_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "vector_ids": vector_ids,
        }

        return len(texts)
