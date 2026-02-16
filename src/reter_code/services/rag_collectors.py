"""
RAG Entity Collector Mixin

Contains methods for collecting entities from different languages
for batched RAG indexing without generating embeddings.

Extracted from RAGIndexManager to reduce file size.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from ..logging_config import configure_logger_for_debug_trace

logger = configure_logger_for_debug_trace(__name__)

if TYPE_CHECKING:
    from ..reter_wrapper import ReterWrapper


@dataclass
class ChunkConfig:
    """
    Configuration for code entity chunking.

    ::: This is-in-layer Utility-Layer.
    ::: This is a value-object.
    """
    enabled: bool = True
    chunk_size: int = 30  # lines
    chunk_overlap: int = 10  # lines
    min_chunk_size: int = 15  # lines


class RAGCollectorMixin:
    """
    Mixin providing entity collection methods for RAG indexing.

    Requires:
        - self._content_extractor: ContentExtractor instance

    ::: This is-in-layer Service-Layer.
    ::: This is a extractor.
    ::: This is-in-process Main-Process.
    ::: This is stateless.
    """

    def _collect_entities(
        self,
        entities: List[Dict[str, Any]],
        source_id: str,
        project_root: Path,
        source_type: str,
        chunk_config: Optional[ChunkConfig] = None
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Collect code entities for batched indexing (without generating embeddings).

        Args:
            entities: List of entity dictionaries from RETER
            source_id: Source identifier (format: "md5|rel_path")
            project_root: Project root path
            source_type: Language type (e.g., "python", "javascript", "csharp", "cpp")
            chunk_config: Optional chunking configuration for long methods

        Returns:
            Tuple of (texts, entity_metadata)
        """
        if not entities:
            return [], []

        # Extract file path from source_id (format: "md5|rel_path")
        if "|" in source_id:
            _, rel_path = source_id.split("|", 1)
        else:
            rel_path = source_id

        # Normalize path separators to forward slashes (consistent with RETER)
        rel_path = rel_path.replace("\\", "/")
        abs_path = project_root / rel_path

        texts = []
        entity_metadata = []

        for entity in entities:
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
            except Exception:
                continue

            if not code_entity:
                continue

            # Check if chunking is enabled and entity should be chunked
            should_chunk = (
                chunk_config is not None
                and chunk_config.enabled
                and code_entity.body
                and entity["entity_type"] in ("method", "function")
                and len(code_entity.body.split('\n')) > chunk_config.chunk_size
            )

            if should_chunk:
                # Chunk the entity body
                chunks = self._content_extractor.chunk_entity_body(
                    code_entity,
                    chunk_size=chunk_config.chunk_size,
                    chunk_overlap=chunk_config.chunk_overlap,
                    min_chunk_size=chunk_config.min_chunk_size
                )

                for chunk in chunks:
                    text = self._content_extractor.build_chunk_indexable_text(chunk)
                    texts.append(text)
                    entity_metadata.append({
                        "entity_type": entity["entity_type"],
                        "name": entity["name"],
                        "qualified_name": f"{entity['qualified_name']}:chunk:{chunk.chunk_index}",
                        "file": rel_path,
                        "line": entity["line"],
                        "end_line": entity.get("end_line"),
                        "docstring_preview": (entity.get("docstring") or "")[:100],
                        "source_type": source_type,
                        "class_name": entity.get("class_name"),
                        "source_id": source_id,
                        # Chunk-specific metadata
                        "is_chunked": True,
                        "chunk_index": chunk.chunk_index,
                        "total_chunks": chunk.total_chunks,
                        "chunk_line_start": chunk.line_start,
                        "chunk_line_end": chunk.line_end,
                        "parent_qualified_name": entity["qualified_name"],
                    })
            else:
                # Standard non-chunked indexing
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
                    "source_type": source_type,
                    "class_name": entity.get("class_name"),
                    "source_id": source_id,
                })

        return texts, entity_metadata

    def _collect_python_entities(
        self,
        entities: List[Dict[str, Any]],
        source_id: str,
        project_root: Path,
        chunk_config: Optional[ChunkConfig] = None
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Collect Python entities for batched indexing."""
        return self._collect_entities(entities, source_id, project_root, "python", chunk_config)

    def _collect_all_python_literals_bulk(
        self,
        reter: "ReterWrapper",
        project_root: Path,
        min_length: int = 32,
        changed_sources: Optional[List[str]] = None
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Collect Python string literals for batched indexing (without generating embeddings).

        Args:
            reter: RETER wrapper instance
            project_root: Project root path
            min_length: Minimum literal length to include
            changed_sources: If provided, only collect literals from these sources (format: md5|rel_path)
                           If None, collect ALL literals (for full reindex)

        Returns:
            Tuple of (texts, literal_metadata)
        """
        # Convert source_ids to module names for filtering
        changed_modules: Optional[set] = None
        if changed_sources:
            changed_modules = set()
            for source_id in changed_sources:
                if "|" in source_id:
                    _, rel_path = source_id.split("|", 1)
                else:
                    rel_path = source_id
                # Convert path to module name: reter_code/src/reter_code/foo/bar.py -> reter_code.foo.bar
                rel_path = rel_path.replace("\\", "/")
                if rel_path.endswith(".py"):
                    rel_path = rel_path[:-3]
                # Remove common prefixes like src/, and handle pkg/src/pkg/ pattern
                if rel_path.startswith("src/"):
                    rel_path = rel_path[4:]
                elif "/src/" in rel_path:
                    # Handle pattern like reter_code/src/reter_code/... -> reter_code/...
                    parts = rel_path.split("/src/", 1)
                    if len(parts) == 2:
                        rel_path = parts[1]  # Take part after /src/
                module_name = rel_path.replace("/", ".")
                changed_modules.add(module_name)
            logger.debug(f"[RAG] _collect_all_python_literals_bulk: filtering by {len(changed_modules)} modules: {list(changed_modules)[:5]}...")

        try:
            query = """
                SELECT DISTINCT ?entity ?literal ?module ?line
                WHERE {
                    ?entity has-string-literal ?literal .
                    ?entity is-in-module ?module .
                    ?entity is-at-line ?line
                }
            """
            result = reter.reql(query, timeout_ms=60000)

            if result.num_rows == 0:
                return [], []

            columns = [result.column(name).to_pylist() for name in result.column_names]
            rows = list(zip(*columns))
        except Exception:
            return [], []

        texts = []
        literal_metadata = []

        for row in rows:
            entity, literal, module, line = row

            # Filter by changed modules if specified
            if changed_modules is not None and module not in changed_modules:
                continue

            # Clean literal
            clean = literal.strip().strip('"\'')
            if len(clean) < min_length:
                continue

            # Skip common patterns
            if clean.startswith(('http://', 'https://', '/', '.', '#')):
                continue

            # Get relative path from module
            rel_path = module.replace(".", "/") + ".py" if module else ""

            texts.append(clean)
            literal_metadata.append({
                "entity_type": "string_literal",
                "name": f"literal@{line}",
                "qualified_name": f"{entity}:literal:{line}",
                "file": rel_path,
                "line": int(line) if line else 0,
                "source_type": "python_literal",
                "content_preview": clean[:200] if clean else "",
                "literal_length": len(clean),
            })

        return texts, literal_metadata

    def _collect_javascript_entities(
        self,
        entities: List[Dict[str, Any]],
        source_id: str,
        project_root: Path,
        chunk_config: Optional[ChunkConfig] = None
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Collect JavaScript entities for batched indexing."""
        return self._collect_entities(entities, source_id, project_root, "javascript", chunk_config)

    def _collect_all_javascript_literals_bulk(
        self,
        reter: "ReterWrapper",
        project_root: Path,
        min_length: int = 32,
        changed_sources: Optional[List[str]] = None
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Collect JavaScript string literals for batched indexing (without generating embeddings).

        Args:
            reter: RETER wrapper instance
            project_root: Project root path
            min_length: Minimum literal length to include
            changed_sources: If provided, only collect literals from these sources (format: md5|rel_path)
                           If None, collect ALL literals (for full reindex)

        Returns:
            Tuple of (texts, literal_metadata)
        """
        # Convert source_ids to module names for filtering
        changed_modules: Optional[set] = None
        if changed_sources:
            changed_modules = set()
            for source_id in changed_sources:
                if "|" in source_id:
                    _, rel_path = source_id.split("|", 1)
                else:
                    rel_path = source_id
                # Convert path to module name
                rel_path = rel_path.replace("\\", "/")
                # For JavaScript, use the file path directly as the "module"
                changed_modules.add(rel_path)
            logger.debug(f"[RAG] _collect_all_javascript_literals_bulk: filtering by {len(changed_modules)} modules: {list(changed_modules)[:5]}...")

        try:
            query = """
                SELECT DISTINCT ?entity ?literal ?file ?line
                WHERE {
                    ?entity has-string-literal ?literal .
                    ?entity is-in-file ?file .
                    ?entity is-at-line ?line
                }
            """
            result = reter.reql(query, timeout_ms=60000)

            if result.num_rows == 0:
                return [], []

            columns = [result.column(name).to_pylist() for name in result.column_names]
            rows = list(zip(*columns))
        except Exception:
            return [], []

        texts = []
        literal_metadata = []

        for row in rows:
            entity, literal, file_path, line = row

            # Filter by changed files if specified
            if changed_modules is not None and file_path not in changed_modules:
                continue

            # Skip non-JavaScript files
            if not file_path.endswith(('.js', '.mjs', '.jsx', '.ts', '.tsx')):
                continue

            # Clean literal
            clean = literal.strip().strip('"\'')
            if len(clean) < min_length:
                continue

            # Skip common patterns
            if clean.startswith(('http://', 'https://', '/', '.', '#')):
                continue

            texts.append(clean)
            literal_metadata.append({
                "entity_type": "string_literal",
                "name": f"literal@{line}",
                "qualified_name": f"{entity}:literal:{line}",
                "file": file_path,
                "line": int(line) if line else 0,
                "source_type": "javascript_literal",
                "content_preview": clean[:200] if clean else "",
                "literal_length": len(clean),
            })

        return texts, literal_metadata

    def _collect_html_entities(
        self,
        entities: List[Dict[str, Any]],
        source_id: str,
        project_root: Path
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Collect HTML entities for batched indexing (without generating embeddings).

        Returns:
            Tuple of (texts, entity_metadata)
        """
        if not entities:
            return [], []

        if "|" in source_id:
            _, rel_path = source_id.split("|", 1)
        else:
            rel_path = source_id

        # Normalize path separators to forward slashes (consistent with RETER)
        rel_path = rel_path.replace("\\", "/")

        texts = []
        entity_metadata = []

        for entity in entities:
            # Build indexable text for HTML entity
            parts = []
            if entity.get("entity_type"):
                parts.append(f"HTML {entity['entity_type']}")
            if entity.get("tag"):
                parts.append(f"Tag: {entity['tag']}")
            if entity.get("id"):
                parts.append(f"ID: {entity['id']}")
            if entity.get("classes"):
                parts.append(f"Classes: {entity['classes']}")
            if entity.get("text_content"):
                parts.append(f"Content: {entity['text_content'][:200]}")

            text = " | ".join(parts) if parts else ""
            if not text:
                continue

            texts.append(text)
            entity_metadata.append({
                "entity_type": entity.get("entity_type", "html_element"),
                "name": entity.get("tag") or entity.get("name") or "element",
                "qualified_name": entity.get("qualified_name", f"{rel_path}:{entity.get('line', 0)}"),
                "file": rel_path,
                "line": entity.get("line", 0),
                "end_line": entity.get("end_line"),
                "source_type": "html",
                "source_id": source_id,
            })

        return texts, entity_metadata

    def _collect_csharp_entities(
        self,
        entities: List[Dict[str, Any]],
        source_id: str,
        project_root: Path,
        chunk_config: Optional[ChunkConfig] = None
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Collect C# entities for batched indexing."""
        return self._collect_entities(entities, source_id, project_root, "csharp", chunk_config)

    def _collect_c_entities(
        self,
        entities: List[Dict[str, Any]],
        source_id: str,
        project_root: Path,
        chunk_config: Optional[ChunkConfig] = None
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Collect C entities for batched indexing."""
        return self._collect_entities(entities, source_id, project_root, "c", chunk_config)

    def _collect_cpp_entities(
        self,
        entities: List[Dict[str, Any]],
        source_id: str,
        project_root: Path,
        chunk_config: Optional[ChunkConfig] = None
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Collect C++ entities for batched indexing."""
        return self._collect_entities(entities, source_id, project_root, "cpp", chunk_config)

    def _collect_java_entities(
        self,
        entities: List[Dict[str, Any]],
        source_id: str,
        project_root: Path,
        chunk_config: Optional[ChunkConfig] = None
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Collect Java entities for batched indexing."""
        return self._collect_entities(entities, source_id, project_root, "java", chunk_config)

    def _collect_go_entities(
        self,
        entities: List[Dict[str, Any]],
        source_id: str,
        project_root: Path,
        chunk_config: Optional[ChunkConfig] = None
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Collect Go entities for batched indexing."""
        return self._collect_entities(entities, source_id, project_root, "go", chunk_config)

    def _collect_rust_entities(
        self,
        entities: List[Dict[str, Any]],
        source_id: str,
        project_root: Path,
        chunk_config: Optional[ChunkConfig] = None
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Collect Rust entities for batched indexing."""
        return self._collect_entities(entities, source_id, project_root, "rust", chunk_config)

    def _collect_erlang_entities(
        self,
        entities: List[Dict[str, Any]],
        source_id: str,
        project_root: Path,
        chunk_config: Optional[ChunkConfig] = None
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Collect Erlang entities for batched indexing."""
        return self._collect_entities(entities, source_id, project_root, "erlang", chunk_config)

    def _collect_php_entities(
        self,
        entities: List[Dict[str, Any]],
        source_id: str,
        project_root: Path,
        chunk_config: Optional[ChunkConfig] = None
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Collect PHP entities for batched indexing."""
        return self._collect_entities(entities, source_id, project_root, "php", chunk_config)

    def _collect_objc_entities(
        self,
        entities: List[Dict[str, Any]],
        source_id: str,
        project_root: Path,
        chunk_config: Optional[ChunkConfig] = None
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Collect Objective-C entities for batched indexing."""
        return self._collect_entities(entities, source_id, project_root, "objc", chunk_config)

    def _collect_swift_entities(
        self,
        entities: List[Dict[str, Any]],
        source_id: str,
        project_root: Path,
        chunk_config: Optional[ChunkConfig] = None
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Collect Swift entities for batched indexing."""
        return self._collect_entities(entities, source_id, project_root, "swift", chunk_config)

    def _collect_vb6_entities(
        self,
        entities: List[Dict[str, Any]],
        source_id: str,
        project_root: Path,
        chunk_config: Optional[ChunkConfig] = None
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Collect VB6 entities for batched indexing."""
        return self._collect_entities(entities, source_id, project_root, "vb6", chunk_config)

    def _collect_scala_entities(
        self,
        entities: List[Dict[str, Any]],
        source_id: str,
        project_root: Path,
        chunk_config: Optional[ChunkConfig] = None
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Collect Scala entities for batched indexing."""
        return self._collect_entities(entities, source_id, project_root, "scala", chunk_config)

    def _collect_haskell_entities(
        self,
        entities: List[Dict[str, Any]],
        source_id: str,
        project_root: Path,
        chunk_config: Optional[ChunkConfig] = None
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Collect Haskell entities for batched indexing."""
        return self._collect_entities(entities, source_id, project_root, "haskell", chunk_config)

    def _collect_kotlin_entities(
        self,
        entities: List[Dict[str, Any]],
        source_id: str,
        project_root: Path,
        chunk_config: Optional[ChunkConfig] = None
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Collect Kotlin entities for batched indexing."""
        return self._collect_entities(entities, source_id, project_root, "kotlin", chunk_config)

    def _collect_r_entities(
        self,
        entities: List[Dict[str, Any]],
        source_id: str,
        project_root: Path,
        chunk_config: Optional[ChunkConfig] = None
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Collect R entities for batched indexing."""
        return self._collect_entities(entities, source_id, project_root, "r", chunk_config)

    def _collect_ruby_entities(
        self,
        entities: List[Dict[str, Any]],
        source_id: str,
        project_root: Path,
        chunk_config: Optional[ChunkConfig] = None
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Collect Ruby entities for batched indexing."""
        return self._collect_entities(entities, source_id, project_root, "ruby", chunk_config)

    def _collect_dart_entities(
        self,
        entities: List[Dict[str, Any]],
        source_id: str,
        project_root: Path,
        chunk_config: Optional[ChunkConfig] = None
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Collect Dart entities for batched indexing."""
        return self._collect_entities(entities, source_id, project_root, "dart", chunk_config)

    def _collect_delphi_entities(
        self,
        entities: List[Dict[str, Any]],
        source_id: str,
        project_root: Path,
        chunk_config: Optional[ChunkConfig] = None
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Collect Delphi entities for batched indexing."""
        return self._collect_entities(entities, source_id, project_root, "delphi", chunk_config)

    def _collect_ada_entities(
        self,
        entities: List[Dict[str, Any]],
        source_id: str,
        project_root: Path,
        chunk_config: Optional[ChunkConfig] = None
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Collect Ada entities for batched indexing."""
        return self._collect_entities(entities, source_id, project_root, "ada", chunk_config)

    def _collect_lua_entities(
        self,
        entities: List[Dict[str, Any]],
        source_id: str,
        project_root: Path,
        chunk_config: Optional[ChunkConfig] = None
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Collect Lua entities for batched indexing."""
        return self._collect_entities(entities, source_id, project_root, "lua", chunk_config)

    def _collect_xaml_entities(
        self,
        entities: List[Dict[str, Any]],
        source_id: str,
        project_root: Path,
        chunk_config: Optional[ChunkConfig] = None
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Collect XAML entities for batched indexing."""
        return self._collect_entities(entities, source_id, project_root, "xaml", chunk_config)

    def _collect_bash_entities(
        self,
        entities: List[Dict[str, Any]],
        source_id: str,
        project_root: Path,
        chunk_config: Optional[ChunkConfig] = None
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Collect Bash entities for batched indexing."""
        return self._collect_entities(entities, source_id, project_root, "bash", chunk_config)

    def _collect_markdown_chunks(
        self,
        chunks: List[Any],
        rel_path: str
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Collect Markdown chunks for batched indexing (without generating embeddings).

        Returns:
            Tuple of (texts, chunk_metadata)
        """
        if not chunks:
            return [], []

        texts = []
        chunk_metadata = []

        for chunk in chunks:
            text = getattr(chunk, 'content', '') or str(chunk)
            if not text or len(text.strip()) < 10:
                continue

            texts.append(text)
            chunk_metadata.append({
                "entity_type": getattr(chunk, 'chunk_type', 'section'),
                "name": getattr(chunk, 'heading', '') or "Section",
                "qualified_name": f"{rel_path}:{getattr(chunk, 'line_start', 0)}",
                "file": rel_path,
                "line": getattr(chunk, 'line_start', 0),
                "end_line": getattr(chunk, 'line_end', 0),
                "source_type": "markdown",
                "content_preview": text[:100] if text else "",
            })

        return texts, chunk_metadata


__all__ = ["RAGCollectorMixin", "ChunkConfig"]
