"""
Content Extractor for RAG functionality.

Extracts source code content from files using line numbers obtained from RETER queries.
Builds indexable text representations of code entities (classes, methods, functions).
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CodeEntity:
    """
    Represents a code entity extracted from source files.

    ::: This is-in-layer Utility-Layer.
    ::: This is a value-object.
    """
    entity_type: str  # "class", "method", "function"
    name: str
    qualified_name: str
    file_path: str
    line_start: int
    line_end: Optional[int]
    docstring: Optional[str]
    signature: Optional[str]
    body: Optional[str]
    module: Optional[str] = None
    class_name: Optional[str] = None  # For methods


@dataclass
class CommentBlock:
    """
    Represents a comment block extracted from source files.

    ::: This is-in-layer Utility-Layer.
    ::: This is a value-object.
    """
    comment_type: str  # "inline", "block", "todo", "fixme", "note", "warning"
    content: str
    file_path: str
    line_start: int
    line_end: int
    context_entity: Optional[str] = None  # Entity the comment is near (class/method/function)
    context_line: Optional[int] = None  # Line of the context entity
    is_standalone: bool = False  # True if not attached to any code entity


@dataclass
class CodeChunk:
    """
    Represents a chunk of a code entity for RAG indexing.

    Used when a method/function body is too long and needs to be split
    into multiple chunks for better semantic search coverage.

    ::: This is-in-layer Utility-Layer.
    ::: This is a value-object.
    """
    parent_entity: CodeEntity
    chunk_index: int
    total_chunks: int
    content: str
    line_start: int
    line_end: int
    includes_header: bool  # True if this chunk includes signature/docstring


class ContentExtractor:
    """
    Extracts source code content from files using line numbers.

    ::: This is-in-layer Service-Layer.
    ::: This is a extractor.
    ::: This is-in-process Main-Process.
    ::: This is stateless.

    Works with RETER query results that provide file paths and line numbers.
    Maintains a file cache to avoid repeated disk reads during batch operations.

    Attributes:
        project_root: Project root directory for resolving relative paths
        max_body_lines: Maximum number of body lines to include
    """

    def __init__(
        self,
        project_root: Optional[Path] = None,
        max_body_lines: int = 50
    ):
        """
        Initialize the content extractor.

        Args:
            project_root: Project root directory (uses CWD if not specified)
            max_body_lines: Maximum number of code body lines to include
        """
        self._project_root = project_root or Path.cwd()
        self._max_body_lines = max_body_lines
        self._file_cache: Dict[str, List[str]] = {}  # path -> lines

    @property
    def project_root(self) -> Path:
        """Get the project root directory."""
        return self._project_root

    @project_root.setter
    def project_root(self, value: Path) -> None:
        """Set project root and clear file cache."""
        self._project_root = value
        self._file_cache.clear()

    def _get_file_lines(self, file_path: str) -> Optional[List[str]]:
        """
        Get file lines with caching.

        Args:
            file_path: Absolute or relative file path

        Returns:
            List of lines (without newlines) or None if file not found
        """
        # Normalize path
        path = Path(file_path)
        if not path.is_absolute():
            path = self._project_root / path

        path_str = str(path)

        # Check cache
        if path_str in self._file_cache:
            return self._file_cache[path_str]

        # Read file
        try:
            content = path.read_text(encoding='utf-8')
            lines = content.splitlines()
            self._file_cache[path_str] = lines
            return lines
        except FileNotFoundError:
            logger.warning(f"File not found: {path}")
            return None
        except UnicodeDecodeError:
            logger.warning(f"Failed to decode file: {path}")
            return None
        except Exception as e:
            logger.warning(f"Error reading file {path}: {e}")
            return None

    def extract_lines(
        self,
        file_path: str,
        start_line: int,
        end_line: Optional[int] = None
    ) -> Optional[str]:
        """
        Extract content between line numbers.

        Args:
            file_path: Path to the source file
            start_line: Starting line number (1-indexed)
            end_line: Ending line number (inclusive), None for single line

        Returns:
            Extracted content as string, or None if extraction fails
        """
        lines = self._get_file_lines(file_path)
        if lines is None:
            return None

        # Convert to 0-indexed
        start_idx = start_line - 1
        end_idx = (end_line - 1) if end_line else start_idx

        # Bounds check
        if start_idx < 0 or start_idx >= len(lines):
            logger.warning(f"Start line {start_line} out of bounds for {file_path}")
            return None

        if end_idx >= len(lines):
            end_idx = len(lines) - 1

        # Extract and join
        extracted_lines = lines[start_idx:end_idx + 1]
        return '\n'.join(extracted_lines)

    def extract_entity_content(
        self,
        file_path: str,
        start_line: int,
        end_line: Optional[int] = None,
        entity_type: str = "unknown"
    ) -> Optional[str]:
        """
        Extract content for a code entity.

        If end_line is not provided, attempts to infer it from indentation
        (useful for Python code where blocks are indentation-based).

        Args:
            file_path: Path to the source file
            start_line: Starting line number (1-indexed)
            end_line: Ending line number (if None, infers from indentation)
            entity_type: Type hint for inference (class, method, function)

        Returns:
            Extracted source code content
        """
        lines = self._get_file_lines(file_path)
        if lines is None:
            return None

        start_idx = start_line - 1
        if start_idx < 0 or start_idx >= len(lines):
            return None

        # If end_line is provided, use it directly
        if end_line is not None:
            end_idx = min(end_line - 1, len(lines) - 1)
            extracted = lines[start_idx:end_idx + 1]

            # Limit body lines
            if len(extracted) > self._max_body_lines:
                extracted = extracted[:self._max_body_lines]
                extracted.append("    # ... (truncated)")

            return '\n'.join(extracted)

        # Infer end line from indentation
        return self._extract_by_indentation(lines, start_idx, entity_type)

    def _extract_by_indentation(
        self,
        lines: List[str],
        start_idx: int,
        entity_type: str
    ) -> str:
        """
        Extract code block by following indentation.

        Continues until we find a line with equal or less indentation
        that isn't empty or a comment.

        Args:
            lines: All file lines
            start_idx: Starting line index (0-indexed)
            entity_type: Entity type for context

        Returns:
            Extracted content
        """
        if start_idx >= len(lines):
            return ""

        first_line = lines[start_idx]
        base_indent = len(first_line) - len(first_line.lstrip())

        extracted = [first_line]
        in_body = False

        for i in range(start_idx + 1, len(lines)):
            line = lines[i]
            stripped = line.strip()

            # Empty lines are included
            if not stripped:
                extracted.append(line)
                continue

            # Get current indentation
            current_indent = len(line) - len(line.lstrip())

            # If we're past the docstring and see same/less indent, we're done
            if in_body and current_indent <= base_indent:
                # Check if this starts a new definition
                if stripped.startswith(('def ', 'class ', '@')):
                    break
                # Non-empty line at base level means we're done
                break

            # First line with content after definition marks body start
            if stripped and not stripped.startswith(('"""', "'''")):
                in_body = True

            extracted.append(line)

            # Limit extraction length
            if len(extracted) > self._max_body_lines:
                extracted.append("    # ... (truncated)")
                break

        return '\n'.join(extracted)

    def extract_signature(
        self,
        file_path: str,
        start_line: int
    ) -> Optional[str]:
        """
        Extract function/method signature from definition line.

        Handles multi-line signatures (parameters spanning multiple lines).

        Args:
            file_path: Path to the source file
            start_line: Line number of the definition

        Returns:
            Complete signature string or None
        """
        lines = self._get_file_lines(file_path)
        if lines is None:
            return None

        start_idx = start_line - 1
        if start_idx < 0 or start_idx >= len(lines):
            return None

        first_line = lines[start_idx]

        # Single-line signature
        if ':' in first_line:
            return first_line.strip()

        # Multi-line signature - collect until we find the colon
        signature_lines = [first_line]

        for i in range(start_idx + 1, min(start_idx + 20, len(lines))):
            line = lines[i]
            signature_lines.append(line)
            if ':' in line:
                break

        return ' '.join(line.strip() for line in signature_lines)

    def extract_docstring(
        self,
        file_path: str,
        start_line: int
    ) -> Optional[str]:
        """
        Extract docstring from a definition.

        Looks for triple-quoted string immediately after the definition line.

        Args:
            file_path: Path to the source file
            start_line: Line number of the definition

        Returns:
            Docstring content (without quotes) or None
        """
        lines = self._get_file_lines(file_path)
        if lines is None:
            return None

        start_idx = start_line - 1
        if start_idx < 0 or start_idx >= len(lines):
            return None

        # Find the line after the definition (after the colon)
        doc_start_idx = start_idx + 1

        # Skip to line after the colon if multi-line signature
        for i in range(start_idx, min(start_idx + 20, len(lines))):
            if ':' in lines[i]:
                doc_start_idx = i + 1
                break

        if doc_start_idx >= len(lines):
            return None

        # Look for docstring start
        for i in range(doc_start_idx, min(doc_start_idx + 3, len(lines))):
            line = lines[i].strip()
            if not line:
                continue

            # Check for docstring start
            if line.startswith(('"""', "'''")):
                return self._extract_docstring_content(lines, i)

            # If we hit non-docstring code, no docstring
            if not line.startswith('#'):
                break

        return None

    def _extract_docstring_content(
        self,
        lines: List[str],
        start_idx: int
    ) -> str:
        """Extract docstring content from triple-quoted string."""
        first_line = lines[start_idx].strip()

        # Determine quote style
        if first_line.startswith('"""'):
            quote = '"""'
        else:
            quote = "'''"

        # Single-line docstring
        if first_line.count(quote) >= 2:
            content = first_line[3:-3].strip()
            return content

        # Multi-line docstring
        docstring_lines = [first_line[3:]]

        for i in range(start_idx + 1, len(lines)):
            line = lines[i]
            if quote in line:
                # Found closing quote
                end_idx = line.find(quote)
                docstring_lines.append(line[:end_idx])
                break
            docstring_lines.append(line)

        return '\n'.join(docstring_lines).strip()

    def build_indexable_text(
        self,
        entity: CodeEntity,
        include_body: bool = True
    ) -> str:
        """
        Build text suitable for embedding from a code entity.

        Returns code body only (no method identity) to enable clustering
        by code similarity rather than name similarity. This supports
        extraction detection - finding similar code fragments across
        methods with different names.

        Args:
            entity: CodeEntity to build text for
            include_body: Ignored - kept for API compatibility

        Returns:
            Code body for embedding (truncated to max_body_lines)
        """
        # Body-only embedding: enables clustering by code similarity
        # Method identity (name, class, signature) is stored in metadata
        # but not included in embedding to avoid polluting similarity
        if entity.body:
            body_lines = entity.body.split('\n')[:self._max_body_lines]
            return '\n'.join(body_lines)
        return ""

    def build_indexable_text_from_parts(
        self,
        entity_name: str,
        entity_type: str,
        docstring: Optional[str] = None,
        signature: Optional[str] = None,
        body: Optional[str] = None,
        class_name: Optional[str] = None,
        qualified_name: Optional[str] = None
    ) -> str:
        """
        Build indexable text from individual parts.

        Returns code body only (no method identity) to enable clustering
        by code similarity rather than name similarity.

        Args:
            entity_name: Ignored - kept for API compatibility
            entity_type: Ignored - kept for API compatibility
            docstring: Ignored - kept for API compatibility
            signature: Ignored - kept for API compatibility
            body: Code body (will be truncated)
            class_name: Ignored - kept for API compatibility
            qualified_name: Ignored - kept for API compatibility

        Returns:
            Code body for embedding (truncated)
        """
        # Body-only embedding for code similarity clustering
        if body:
            body_lines = body.split('\n')[:self._max_body_lines]
            return '\n'.join(body_lines)
        return ""

    def extract_and_build(
        self,
        file_path: str,
        entity_type: str,
        name: str,
        qualified_name: str,
        start_line: int,
        end_line: Optional[int] = None,
        docstring: Optional[str] = None,
        class_name: Optional[str] = None
    ) -> Optional[CodeEntity]:
        """
        Extract content and build a CodeEntity.

        Combines extraction and entity building in one step.

        Args:
            file_path: Path to source file
            entity_type: Type of entity
            name: Entity name
            qualified_name: Fully qualified name
            start_line: Starting line number (1-indexed)
            end_line: Ending line number (optional)
            docstring: Pre-extracted docstring (will extract if None)
            class_name: Parent class name for methods

        Returns:
            CodeEntity with extracted content, or None if extraction fails
        """
        # Extract body
        body = self.extract_entity_content(
            file_path, start_line, end_line, entity_type
        )
        if body is None:
            return None

        # Extract signature
        signature = self.extract_signature(file_path, start_line)

        # Extract docstring if not provided
        if docstring is None:
            docstring = self.extract_docstring(file_path, start_line)

        # Determine module from file path
        module = None
        try:
            rel_path = Path(file_path).relative_to(self._project_root)
            module = str(rel_path).replace('/', '.').replace('\\', '.')
            if module.endswith('.py'):
                module = module[:-3]
        except ValueError:
            pass

        return CodeEntity(
            entity_type=entity_type,
            name=name,
            qualified_name=qualified_name,
            file_path=str(file_path),
            line_start=start_line,
            line_end=end_line,
            docstring=docstring,
            signature=signature,
            body=body,
            module=module,
            class_name=class_name
        )

    def invalidate_cache(self, file_path: Optional[str] = None) -> None:
        """
        Invalidate file cache.

        Args:
            file_path: Specific file to invalidate, or None for all files
        """
        if file_path:
            # Normalize path
            path = Path(file_path)
            if not path.is_absolute():
                path = self._project_root / path
            self._file_cache.pop(str(path), None)
        else:
            self._file_cache.clear()
            logger.debug("Content extractor cache cleared")

    def get_cache_info(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'cached_files': len(self._file_cache),
            'total_lines': sum(len(lines) for lines in self._file_cache.values()),
            'files': list(self._file_cache.keys())
        }

    def extract_comments(
        self,
        file_path: str,
        entity_locations: Optional[List[Dict[str, Any]]] = None
    ) -> List[CommentBlock]:
        """
        Extract all comments from a Python file.

        Identifies inline comments (#), block comments (consecutive # lines),
        and special markers (TODO, FIXME, NOTE, WARNING, HACK, XXX).

        Args:
            file_path: Path to the Python source file
            entity_locations: Optional list of code entity locations for context
                              Each dict should have: name, line_start, line_end, entity_type

        Returns:
            List of CommentBlock objects
        """
        lines = self._get_file_lines(file_path)
        if lines is None:
            return []

        comments = []
        current_block: List[tuple] = []  # [(line_num, content), ...]

        # Special markers to detect
        special_markers = {
            'TODO': 'todo',
            'FIXME': 'fixme',
            'NOTE': 'note',
            'WARNING': 'warning',
            'HACK': 'hack',
            'XXX': 'xxx',
            'BUG': 'bug',
            'OPTIMIZE': 'optimize',
            'REVIEW': 'review',
        }

        for i, line in enumerate(lines):
            line_num = i + 1  # 1-indexed
            stripped = line.strip()

            # Skip docstrings (handled separately)
            if stripped.startswith(('"""', "'''")):
                continue

            # Check for comment
            if '#' in line:
                # Find comment start (not inside string)
                comment_start = self._find_comment_start(line)
                if comment_start >= 0:
                    comment_text = line[comment_start + 1:].strip()
                    if comment_text:  # Skip empty comments
                        current_block.append((line_num, comment_text))
                    continue

            # If we hit a non-comment line and have accumulated comments, flush them
            if current_block:
                comment = self._create_comment_block(
                    current_block, file_path, special_markers, entity_locations
                )
                if comment:
                    comments.append(comment)
                current_block = []

        # Flush any remaining comments
        if current_block:
            comment = self._create_comment_block(
                current_block, file_path, special_markers, entity_locations
            )
            if comment:
                comments.append(comment)

        return comments

    def _find_comment_start(self, line: str) -> int:
        """
        Find the index of # that starts a comment (not inside a string).

        Args:
            line: Line of code

        Returns:
            Index of comment start, or -1 if no comment
        """
        in_string = False
        string_char = None
        escape_next = False

        for i, char in enumerate(line):
            if escape_next:
                escape_next = False
                continue

            if char == '\\':
                escape_next = True
                continue

            if char in ('"', "'"):
                if not in_string:
                    in_string = True
                    string_char = char
                elif char == string_char:
                    in_string = False
                    string_char = None

            if char == '#' and not in_string:
                return i

        return -1

    def _create_comment_block(
        self,
        block: List[tuple],
        file_path: str,
        special_markers: Dict[str, str],
        entity_locations: Optional[List[Dict[str, Any]]]
    ) -> Optional[CommentBlock]:
        """
        Create a CommentBlock from accumulated comment lines.

        Args:
            block: List of (line_num, content) tuples
            file_path: Path to source file
            special_markers: Dict mapping marker text to comment type
            entity_locations: Optional list of entity locations for context

        Returns:
            CommentBlock or None if block is empty/trivial
        """
        if not block:
            return None

        line_start = block[0][0]
        line_end = block[-1][0]
        content = '\n'.join(text for _, text in block)

        # Skip trivial comments (very short, just separators, etc.)
        clean_content = content.replace('#', '').replace('-', '').replace('=', '').strip()
        if len(clean_content) < 5:  # Skip very short/empty comments
            return None

        # Determine comment type
        comment_type = 'block' if len(block) > 1 else 'inline'

        # Check for special markers at the START of the comment
        # Use first line only, strip whitespace, check if starts with marker
        first_line = block[0][1].strip().upper()
        for marker, marker_type in special_markers.items():
            # Must start with marker, optionally followed by : or whitespace
            if first_line.startswith(marker) and (
                len(first_line) == len(marker) or
                first_line[len(marker):].lstrip().startswith((':',)) or
                first_line[len(marker)] in (' ', '\t', ':')
            ):
                comment_type = marker_type
                break

        # Find context entity (nearest code entity)
        context_entity = None
        context_line = None
        is_standalone = True

        if entity_locations:
            # Find the entity that this comment is closest to
            for entity in entity_locations:
                entity_start = entity.get('line_start', 0)
                entity_end = entity.get('line_end', entity_start)

                # Comment immediately before entity
                if line_end == entity_start - 1 or (line_end >= entity_start - 3 and line_end < entity_start):
                    context_entity = entity.get('name')
                    context_line = entity_start
                    is_standalone = False
                    break

                # Comment inside entity
                if entity_start <= line_start <= entity_end:
                    context_entity = entity.get('name')
                    context_line = entity_start
                    is_standalone = False
                    break

        return CommentBlock(
            comment_type=comment_type,
            content=content,
            file_path=file_path,
            line_start=line_start,
            line_end=line_end,
            context_entity=context_entity,
            context_line=context_line,
            is_standalone=is_standalone
        )

    def build_comment_indexable_text(self, comment: CommentBlock) -> str:
        """
        Build text suitable for embedding from a comment block.

        Args:
            comment: CommentBlock to build text for

        Returns:
            Combined text for embedding
        """
        parts = []

        # Type indicator
        type_label = {
            'inline': 'Comment',
            'block': 'Block Comment',
            'todo': 'TODO',
            'fixme': 'FIXME',
            'note': 'Note',
            'warning': 'Warning',
            'hack': 'Hack',
            'xxx': 'XXX',
            'bug': 'Bug',
            'optimize': 'Optimization Note',
            'review': 'Review Note',
        }.get(comment.comment_type, 'Comment')

        parts.append(f"{type_label}:")

        # Context if available
        if comment.context_entity:
            parts.append(f"In: {comment.context_entity}")

        # Content
        parts.append(comment.content)

        return '\n'.join(parts)

    def chunk_entity_body(
        self,
        entity: CodeEntity,
        chunk_size: int = 30,
        chunk_overlap: int = 10,
        min_chunk_size: int = 15
    ) -> List[CodeChunk]:
        """
        Split a code entity's body into overlapping chunks for RAG indexing.

        This enables better semantic search coverage for long methods/functions
        by creating multiple indexed entries with overlapping content.

        Args:
            entity: CodeEntity to chunk
            chunk_size: Maximum number of lines per chunk (default: 30)
            chunk_overlap: Number of overlapping lines between chunks (default: 10)
            min_chunk_size: Minimum lines to avoid tiny trailing chunks (default: 15)

        Returns:
            List of CodeChunk objects. Returns single chunk if body <= chunk_size.
        """
        if not entity.body:
            return []

        body_lines = entity.body.split('\n')
        total_lines = len(body_lines)

        # If body fits in a single chunk, return it as-is
        if total_lines <= chunk_size:
            return [CodeChunk(
                parent_entity=entity,
                chunk_index=0,
                total_chunks=1,
                content=entity.body,
                line_start=entity.line_start,
                line_end=entity.line_end or (entity.line_start + total_lines - 1),
                includes_header=True
            )]

        chunks = []
        start_idx = 0
        chunk_index = 0

        # Calculate step size (chunk_size - overlap)
        step = chunk_size - chunk_overlap
        if step <= 0:
            step = 1  # Ensure we make progress

        # Estimate total chunks for metadata
        estimated_chunks = max(1, (total_lines - chunk_overlap) // step + 1)

        while start_idx < total_lines:
            end_idx = min(start_idx + chunk_size, total_lines)

            # Check if remaining content is smaller than min_chunk_size
            remaining = total_lines - start_idx
            if remaining < min_chunk_size and chunk_index > 0:
                # Merge with previous chunk by extending it
                if chunks:
                    prev_chunk = chunks[-1]
                    # Extend the previous chunk to include remaining content
                    extended_content = '\n'.join(body_lines[
                        prev_chunk.line_start - entity.line_start:total_lines
                    ])
                    chunks[-1] = CodeChunk(
                        parent_entity=entity,
                        chunk_index=prev_chunk.chunk_index,
                        total_chunks=len(chunks),  # Will be updated
                        content=extended_content,
                        line_start=prev_chunk.line_start,
                        line_end=entity.line_start + total_lines - 1,
                        includes_header=prev_chunk.includes_header
                    )
                break

            chunk_content = '\n'.join(body_lines[start_idx:end_idx])

            # Calculate actual line numbers in the file
            chunk_line_start = entity.line_start + start_idx
            chunk_line_end = entity.line_start + end_idx - 1

            chunks.append(CodeChunk(
                parent_entity=entity,
                chunk_index=chunk_index,
                total_chunks=estimated_chunks,  # Will be updated
                content=chunk_content,
                line_start=chunk_line_start,
                line_end=chunk_line_end,
                includes_header=(chunk_index == 0)
            ))

            start_idx += step
            chunk_index += 1

        # Update total_chunks in all chunks now that we know the final count
        total_chunks = len(chunks)
        for i, chunk in enumerate(chunks):
            chunks[i] = CodeChunk(
                parent_entity=chunk.parent_entity,
                chunk_index=chunk.chunk_index,
                total_chunks=total_chunks,
                content=chunk.content,
                line_start=chunk.line_start,
                line_end=chunk.line_end,
                includes_header=chunk.includes_header
            )

        return chunks

    def build_chunk_indexable_text(
        self,
        chunk: CodeChunk,
        include_context: bool = True
    ) -> str:
        """
        Build text suitable for embedding from a code chunk.

        Returns code body only (no method identity) to enable clustering
        by code similarity rather than name similarity. This supports
        extraction detection - finding similar code fragments across
        methods with different names.

        Args:
            chunk: CodeChunk to build text for
            include_context: Ignored - kept for API compatibility

        Returns:
            Code content only for embedding
        """
        # Body-only embedding: enables clustering by code similarity
        # Method identity (name, class, signature) is stored in metadata
        # but not included in embedding to avoid polluting similarity
        return chunk.content
