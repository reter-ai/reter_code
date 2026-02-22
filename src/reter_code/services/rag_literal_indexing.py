"""
RAG Literal Indexing Mixin

Contains methods for indexing string literals (Python and JavaScript) in the RAG system.
Extracted from RAGIndexManager to reduce file size.
"""

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..reter_wrapper import ReterWrapper

from ..logging_config import configure_logger_for_debug_trace

logger = configure_logger_for_debug_trace(__name__)


class RAGLiteralIndexingMixin:
    """
    Mixin providing string literal indexing methods for RAGIndexManager.

    ::: This is-in-layer Service-Layer.
    ::: This is-part-of-component RAG-Index.
    ::: This is-in-process Main-Process.
    ::: This is stateless.

    These methods require:
    - self._reter: ReterWrapper instance
    - self._faiss_wrapper: FAISSWrapper instance
    - self._embedding_service: EmbeddingService instance
    - self._metadata: dict mapping vector IDs to entity metadata
    """

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
            "indexed_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
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
            "indexed_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
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
            "indexed_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
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
            "indexed_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "vector_ids": vector_ids,
            "total_literals": len(texts),
        }

        total_time = time.time() - start_time
        logger.debug(f"[RAG] _index_all_javascript_literals_bulk: Indexed {len(texts)} literals in {total_time:.2f}s")
        return len(texts)
