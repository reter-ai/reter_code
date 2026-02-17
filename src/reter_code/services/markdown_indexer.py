"""
Markdown Indexer for RAG functionality.

Parses markdown files and extracts indexable chunks:
- Document-level (full content, chunked if large)
- Section-level (by headings)
- Code blocks (fenced code)
"""

import re
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)


@dataclass
class MarkdownChunk:
    """
    Represents an indexable chunk from a markdown file.

    ::: This is-in-layer Utility-Layer.
    ::: This is a value-object.
    """
    file: str
    chunk_type: str  # "document", "section", "code_block"
    content: str
    line_start: int
    line_end: int
    heading: Optional[str] = None
    heading_level: Optional[int] = None
    language: Optional[str] = None  # For code blocks
    title: Optional[str] = None  # Document title (first H1)
    word_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Calculate word count after initialization."""
        if self.word_count == 0:
            self.word_count = len(self.content.split())


class MarkdownIndexer:
    """
    Parses markdown files and extracts chunks for embedding.

    ::: This is-in-layer Service-Layer.
    ::: This is a parser.
    ::: This is-in-process Main-Process.
    ::: This is stateless.

    Strategies:
    - Small files (<max_chunk_words): Index as single document
    - Large files: Split by sections (headings)
    - Code blocks: Optionally index separately with context

    Attributes:
        max_chunk_words: Maximum words per chunk before splitting
        min_chunk_words: Minimum words for a chunk to be indexed
        include_code_blocks: Whether to index code blocks separately
    """

    def __init__(
        self,
        max_chunk_words: int = 500,
        min_chunk_words: int = 50,
        include_code_blocks: bool = True
    ):
        """
        Initialize the markdown indexer.

        Args:
            max_chunk_words: Maximum words before splitting document
            min_chunk_words: Minimum words for a chunk to be included
            include_code_blocks: Whether to extract code blocks separately
        """
        self.max_chunk_words = max_chunk_words
        self.min_chunk_words = min_chunk_words
        self.include_code_blocks = include_code_blocks

        # Regex patterns
        self._heading_pattern = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
        self._code_block_pattern = re.compile(
            r'^```(\w*)\n(.*?)^```',
            re.MULTILINE | re.DOTALL
        )

    def parse_file(self, file_path: str) -> List[MarkdownChunk]:
        """
        Parse a markdown file and extract indexable chunks.

        Args:
            file_path: Path to the markdown file

        Returns:
            List of MarkdownChunk objects
        """
        path = Path(file_path)
        if not path.exists():
            logger.warning(f"Markdown file not found: {file_path}")
            return []

        try:
            content = path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            # Try with different encodings
            for encoding in ['latin-1', 'cp1252']:
                try:
                    content = path.read_text(encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                logger.warning(f"Failed to decode markdown file: {file_path}")
                return []
        except Exception as e:
            logger.warning(f"Error reading markdown file {file_path}: {e}")
            return []

        return self.parse_content(content, file_path)

    def parse_content(self, content: str, file_path: str) -> List[MarkdownChunk]:
        """
        Parse markdown content and extract indexable chunks.

        Args:
            content: Markdown content string
            file_path: Path to the file (for metadata)

        Returns:
            List of MarkdownChunk objects
        """
        lines = content.split('\n')
        chunks = []

        # Extract document title (first H1)
        title = self._extract_title(lines)

        # Decide chunking strategy based on document size
        word_count = len(content.split())

        if word_count <= self.max_chunk_words:
            # Small document - index as whole
            if word_count >= self.min_chunk_words:
                chunks.append(MarkdownChunk(
                    file=file_path,
                    chunk_type="document",
                    content=content,
                    line_start=1,
                    line_end=len(lines),
                    title=title,
                    word_count=word_count
                ))
        else:
            # Large document - split by sections
            section_chunks = self._extract_sections(file_path, content, lines, title)
            chunks.extend(section_chunks)

            # If no sections found (no headings), chunk by paragraphs
            if not section_chunks:
                chunks.extend(
                    self._extract_paragraphs(file_path, content, lines, title)
                )

        # Extract code blocks if enabled
        if self.include_code_blocks:
            code_chunks = self._extract_code_blocks(file_path, content, lines)
            chunks.extend(code_chunks)

        return chunks

    def _extract_title(self, lines: List[str]) -> Optional[str]:
        """Extract document title (first H1 heading)."""
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('# ') and not stripped.startswith('##'):
                return stripped[2:].strip()
        return None

    def _extract_sections(
        self,
        file_path: str,
        content: str,
        lines: List[str],
        title: Optional[str]
    ) -> List[MarkdownChunk]:
        """
        Extract sections based on headings.

        Groups content by heading, creating chunks for each section.
        """
        chunks = []
        sections = []

        # Find all headings with their positions
        for i, line in enumerate(lines):
            match = self._heading_pattern.match(line)
            if match:
                level = len(match.group(1))
                heading = match.group(2).strip()
                sections.append({
                    'line': i,
                    'level': level,
                    'heading': heading
                })

        if not sections:
            return []

        # Create chunks between headings
        for idx, section in enumerate(sections):
            line_num = section['line']
            level = section['level']
            heading = section['heading']

            # Determine end of section
            if idx + 1 < len(sections):
                end_line = sections[idx + 1]['line'] - 1
            else:
                end_line = len(lines) - 1

            # Extract section content
            section_lines = lines[line_num:end_line + 1]
            section_content = '\n'.join(section_lines)
            section_words = len(section_content.split())

            # Skip very short sections
            if section_words < self.min_chunk_words:
                continue

            # If section is too large, split further
            if section_words > self.max_chunk_words * 2:
                # Try to split by subsections or paragraphs
                sub_chunks = self._split_large_section(
                    file_path, section_content, section_lines,
                    line_num + 1, heading, level, title
                )
                if sub_chunks:
                    chunks.extend(sub_chunks)
                    continue

            chunks.append(MarkdownChunk(
                file=file_path,
                chunk_type="section",
                content=section_content,
                line_start=line_num + 1,  # 1-indexed
                line_end=end_line + 1,
                heading=heading,
                heading_level=level,
                title=title,
                word_count=section_words
            ))

        return chunks

    def _split_large_section(
        self,
        file_path: str,
        content: str,
        lines: List[str],
        base_line: int,
        heading: str,
        level: int,
        title: Optional[str]
    ) -> List[MarkdownChunk]:
        """Split a large section into smaller chunks."""
        chunks = []

        # Try to find subsections (higher level headings)
        sub_headings = []
        for i, line in enumerate(lines):
            match = self._heading_pattern.match(line)
            if match:
                sub_level = len(match.group(1))
                if sub_level > level:  # Subsection
                    sub_headings.append({
                        'line': i,
                        'level': sub_level,
                        'heading': match.group(2).strip()
                    })

        if sub_headings:
            # Split by subsections
            for idx, sub in enumerate(sub_headings):
                sub_line = sub['line']
                sub_heading = sub['heading']
                sub_level = sub['level']

                if idx + 1 < len(sub_headings):
                    end = sub_headings[idx + 1]['line'] - 1
                else:
                    end = len(lines) - 1

                sub_lines = lines[sub_line:end + 1]
                sub_content = '\n'.join(sub_lines)
                sub_words = len(sub_content.split())

                if sub_words >= self.min_chunk_words:
                    chunks.append(MarkdownChunk(
                        file=file_path,
                        chunk_type="section",
                        content=sub_content,
                        line_start=base_line + sub_line,
                        line_end=base_line + end,
                        heading=sub_heading,
                        heading_level=sub_level,
                        title=title,
                        word_count=sub_words,
                        metadata={'parent_heading': heading}
                    ))
        else:
            # Split by paragraphs (double newlines)
            paragraphs = content.split('\n\n')
            current_chunk = []
            current_words = 0
            current_start = 0

            for para in paragraphs:
                para_words = len(para.split())

                if current_words + para_words > self.max_chunk_words and current_chunk:
                    # Save current chunk
                    chunk_content = '\n\n'.join(current_chunk)
                    if len(chunk_content.split()) >= self.min_chunk_words:
                        chunks.append(MarkdownChunk(
                            file=file_path,
                            chunk_type="section",
                            content=chunk_content,
                            line_start=base_line + current_start,
                            line_end=base_line + current_start + len(chunk_content.split('\n')),
                            heading=heading,
                            heading_level=level,
                            title=title,
                            metadata={'split': True}
                        ))
                    current_chunk = [para]
                    current_words = para_words
                    current_start += len('\n\n'.join(current_chunk[:-1]).split('\n')) + 2
                else:
                    current_chunk.append(para)
                    current_words += para_words

            # Don't forget the last chunk
            if current_chunk:
                chunk_content = '\n\n'.join(current_chunk)
                if len(chunk_content.split()) >= self.min_chunk_words:
                    chunks.append(MarkdownChunk(
                        file=file_path,
                        chunk_type="section",
                        content=chunk_content,
                        line_start=base_line + current_start,
                        line_end=base_line + len(lines) - 1,
                        heading=heading,
                        heading_level=level,
                        title=title,
                        metadata={'split': True}
                    ))

        return chunks

    def _extract_paragraphs(
        self,
        file_path: str,
        content: str,
        lines: List[str],
        title: Optional[str]
    ) -> List[MarkdownChunk]:
        """
        Extract chunks by paragraphs when no headings are found.

        Groups paragraphs until max_chunk_words is reached.
        """
        chunks = []
        paragraphs = content.split('\n\n')

        current_chunk = []
        current_words = 0
        current_line = 1

        for para in paragraphs:
            para_words = len(para.split())
            para_lines = para.count('\n') + 1

            if current_words + para_words > self.max_chunk_words and current_chunk:
                # Save current chunk
                chunk_content = '\n\n'.join(current_chunk)
                chunk_line_count = chunk_content.count('\n') + 1

                if len(chunk_content.split()) >= self.min_chunk_words:
                    chunks.append(MarkdownChunk(
                        file=file_path,
                        chunk_type="document",
                        content=chunk_content,
                        line_start=current_line,
                        line_end=current_line + chunk_line_count - 1,
                        title=title
                    ))

                current_line += chunk_line_count + 1  # +1 for blank line
                current_chunk = [para]
                current_words = para_words
            else:
                current_chunk.append(para)
                current_words += para_words

        # Last chunk
        if current_chunk:
            chunk_content = '\n\n'.join(current_chunk)
            if len(chunk_content.split()) >= self.min_chunk_words:
                chunks.append(MarkdownChunk(
                    file=file_path,
                    chunk_type="document",
                    content=chunk_content,
                    line_start=current_line,
                    line_end=len(lines),
                    title=title
                ))

        return chunks

    def _extract_code_blocks(
        self,
        file_path: str,
        content: str,
        lines: List[str]
    ) -> List[MarkdownChunk]:
        """
        Extract fenced code blocks.

        Tries the grammar-based ``extract_code_fences`` first (handles ``~~~``
        fences and non-word language tags like ``c++``).  Falls back to the
        legacy regex on any failure.
        """
        try:
            from reter_code.mermaid.markdown_validator import extract_code_fences
            _, code_blocks, _ = extract_code_fences(content)

            chunks: List[MarkdownChunk] = []
            for cb in code_blocks:
                code = cb.content.strip()
                if not code or len(code.split()) < 3:
                    continue

                context_heading = None
                for i in range(cb.start_line - 2, -1, -1):
                    if i < len(lines):
                        heading_match = self._heading_pattern.match(lines[i])
                        if heading_match:
                            context_heading = heading_match.group(2).strip()
                            break

                chunks.append(MarkdownChunk(
                    file=file_path,
                    chunk_type="code_block",
                    content=code,
                    line_start=cb.start_line,
                    line_end=cb.end_line,
                    language=cb.language or "text",
                    heading=context_heading,
                ))
            return chunks
        except Exception:
            logger.debug("extract_code_fences failed, falling back to regex")
            return self._extract_code_blocks_regex(file_path, content, lines)

    def _extract_code_blocks_regex(
        self,
        file_path: str,
        content: str,
        lines: List[str]
    ) -> List[MarkdownChunk]:
        """
        Regex fallback for fenced code block extraction.
        """
        chunks = []

        # Find code blocks and their line positions
        for match in self._code_block_pattern.finditer(content):
            language = match.group(1) or "text"
            code = match.group(2).strip()

            # Skip empty code blocks
            if not code or len(code.split()) < 3:
                continue

            # Calculate line numbers
            start_pos = match.start()
            line_start = content[:start_pos].count('\n') + 1
            line_end = line_start + code.count('\n') + 2  # +2 for fence lines

            # Find context (nearest heading above)
            context_heading = None
            for i in range(line_start - 2, -1, -1):
                if i < len(lines):
                    heading_match = self._heading_pattern.match(lines[i])
                    if heading_match:
                        context_heading = heading_match.group(2).strip()
                        break

            chunks.append(MarkdownChunk(
                file=file_path,
                chunk_type="code_block",
                content=code,
                line_start=line_start,
                line_end=line_end,
                language=language,
                heading=context_heading
            ))

        return chunks

    def build_indexable_text(self, chunk: MarkdownChunk) -> str:
        """
        Build text suitable for embedding from a chunk.

        Prepends context information to help with semantic matching.

        Args:
            chunk: MarkdownChunk to process

        Returns:
            Text optimized for semantic search
        """
        parts = []

        if chunk.chunk_type == "document":
            if chunk.title:
                parts.append(f"Document: {chunk.title}")
            parts.append(chunk.content)

        elif chunk.chunk_type == "section":
            if chunk.title:
                parts.append(f"Document: {chunk.title}")
            if chunk.heading:
                level_indicator = '#' * (chunk.heading_level or 2)
                parts.append(f"Section: {level_indicator} {chunk.heading}")
            parts.append(chunk.content)

        elif chunk.chunk_type == "code_block":
            if chunk.heading:
                parts.append(f"Context: {chunk.heading}")
            if chunk.language and chunk.language != "text":
                parts.append(f"Language: {chunk.language}")
            parts.append(f"Code:\n{chunk.content}")

        return '\n\n'.join(parts)

    def get_chunk_id(self, chunk: MarkdownChunk) -> str:
        """
        Generate a unique identifier for a chunk.

        Used for tracking and updating chunks.

        Args:
            chunk: MarkdownChunk to identify

        Returns:
            Unique string identifier
        """
        parts = [
            chunk.file,
            chunk.chunk_type,
            str(chunk.line_start),
            str(chunk.line_end)
        ]

        if chunk.heading:
            parts.append(chunk.heading[:50])

        return '|'.join(parts)

    def parse_directory(
        self,
        directory: str,
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
        recursive: bool = True
    ) -> List[MarkdownChunk]:
        """
        Parse all markdown files in a directory.

        Args:
            directory: Directory path to scan
            include_patterns: Glob patterns to include (default: ["**/*.md"])
            exclude_patterns: Glob patterns to exclude
            recursive: Whether to search recursively

        Returns:
            List of all MarkdownChunk objects from all files
        """
        dir_path = Path(directory)
        if not dir_path.exists():
            logger.warning(f"Directory not found: {directory}")
            return []

        # Default patterns
        if include_patterns is None:
            include_patterns = ["**/*.md"] if recursive else ["*.md"]

        if exclude_patterns is None:
            exclude_patterns = []

        all_chunks = []
        processed_files = set()

        for pattern in include_patterns:
            for md_file in dir_path.glob(pattern):
                if not md_file.is_file():
                    continue

                # Skip if already processed
                file_str = str(md_file)
                if file_str in processed_files:
                    continue

                # Check exclusions
                rel_path = str(md_file.relative_to(dir_path))
                if self._is_excluded(rel_path, exclude_patterns):
                    continue

                processed_files.add(file_str)
                chunks = self.parse_file(file_str)
                all_chunks.extend(chunks)

        logger.info(
            f"Parsed {len(processed_files)} markdown files, "
            f"extracted {len(all_chunks)} chunks"
        )

        return all_chunks

    def _is_excluded(self, path: str, patterns: List[str]) -> bool:
        """Check if path matches any exclusion pattern."""
        import fnmatch

        path_normalized = path.replace('\\', '/')

        for pattern in patterns:
            if fnmatch.fnmatch(path_normalized, pattern):
                return True
            # Also check without leading **/ for simpler patterns
            if pattern.startswith('**/'):
                simple_pattern = pattern[3:]
                if fnmatch.fnmatch(path_normalized, simple_pattern):
                    return True

        return False
