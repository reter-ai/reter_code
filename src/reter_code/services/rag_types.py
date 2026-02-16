"""
RAG Types - Data classes for RAG index operations.

Contains value objects used by RAGIndexManager and related components.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class LanguageSourceChanges:
    """
    Source changes for a single language type.

    ::: This is-in-layer Utility-Layer.
    ::: This is a value-object.

    Groups changed and deleted source IDs for one language.
    """
    changed: List[str] = field(default_factory=list)
    deleted: List[str] = field(default_factory=list)


@dataclass
class SyncChanges:
    """
    All source changes for RAG index synchronization.

    ::: This is-in-layer Utility-Layer.
    ::: This is a value-object.

    Consolidates per-language change lists (19 languages + markdown)
    into a structured object organized by language.
    """
    python: LanguageSourceChanges = field(default_factory=LanguageSourceChanges)
    javascript: LanguageSourceChanges = field(default_factory=LanguageSourceChanges)
    html: LanguageSourceChanges = field(default_factory=LanguageSourceChanges)
    csharp: LanguageSourceChanges = field(default_factory=LanguageSourceChanges)
    c: LanguageSourceChanges = field(default_factory=LanguageSourceChanges)
    cpp: LanguageSourceChanges = field(default_factory=LanguageSourceChanges)
    java: LanguageSourceChanges = field(default_factory=LanguageSourceChanges)
    go: LanguageSourceChanges = field(default_factory=LanguageSourceChanges)
    rust: LanguageSourceChanges = field(default_factory=LanguageSourceChanges)
    erlang: LanguageSourceChanges = field(default_factory=LanguageSourceChanges)
    php: LanguageSourceChanges = field(default_factory=LanguageSourceChanges)
    objc: LanguageSourceChanges = field(default_factory=LanguageSourceChanges)
    swift: LanguageSourceChanges = field(default_factory=LanguageSourceChanges)
    vb6: LanguageSourceChanges = field(default_factory=LanguageSourceChanges)
    scala: LanguageSourceChanges = field(default_factory=LanguageSourceChanges)
    haskell: LanguageSourceChanges = field(default_factory=LanguageSourceChanges)
    kotlin: LanguageSourceChanges = field(default_factory=LanguageSourceChanges)
    r: LanguageSourceChanges = field(default_factory=LanguageSourceChanges)
    ruby: LanguageSourceChanges = field(default_factory=LanguageSourceChanges)
    dart: LanguageSourceChanges = field(default_factory=LanguageSourceChanges)
    delphi: LanguageSourceChanges = field(default_factory=LanguageSourceChanges)
    ada: LanguageSourceChanges = field(default_factory=LanguageSourceChanges)
    lua: LanguageSourceChanges = field(default_factory=LanguageSourceChanges)
    xaml: LanguageSourceChanges = field(default_factory=LanguageSourceChanges)
    bash: LanguageSourceChanges = field(default_factory=LanguageSourceChanges)
    eval: LanguageSourceChanges = field(default_factory=LanguageSourceChanges)
    markdown: LanguageSourceChanges = field(default_factory=LanguageSourceChanges)

    @classmethod
    def from_params(
        cls,
        changed_python_sources: Optional[List[str]] = None,
        deleted_python_sources: Optional[List[str]] = None,
        changed_javascript_sources: Optional[List[str]] = None,
        deleted_javascript_sources: Optional[List[str]] = None,
        changed_html_sources: Optional[List[str]] = None,
        deleted_html_sources: Optional[List[str]] = None,
        changed_csharp_sources: Optional[List[str]] = None,
        deleted_csharp_sources: Optional[List[str]] = None,
        changed_cpp_sources: Optional[List[str]] = None,
        deleted_cpp_sources: Optional[List[str]] = None,
        changed_markdown_files: Optional[List[str]] = None,
        deleted_markdown_files: Optional[List[str]] = None,
    ) -> "SyncChanges":
        """Create SyncChanges from individual parameters for backward compatibility."""
        return cls(
            python=LanguageSourceChanges(
                changed=changed_python_sources or [],
                deleted=deleted_python_sources or [],
            ),
            javascript=LanguageSourceChanges(
                changed=changed_javascript_sources or [],
                deleted=deleted_javascript_sources or [],
            ),
            html=LanguageSourceChanges(
                changed=changed_html_sources or [],
                deleted=deleted_html_sources or [],
            ),
            csharp=LanguageSourceChanges(
                changed=changed_csharp_sources or [],
                deleted=deleted_csharp_sources or [],
            ),
            cpp=LanguageSourceChanges(
                changed=changed_cpp_sources or [],
                deleted=deleted_cpp_sources or [],
            ),
            markdown=LanguageSourceChanges(
                changed=changed_markdown_files or [],
                deleted=deleted_markdown_files or [],
            ),
        )

    def has_changes(self) -> bool:
        """Check if there are any changes to sync."""
        for lang_changes in (
            self.python, self.javascript, self.html, self.csharp, self.c, self.cpp,
            self.java, self.go, self.rust, self.erlang, self.php,
            self.objc, self.swift, self.vb6, self.scala, self.haskell,
            self.kotlin, self.r, self.ruby, self.dart, self.delphi, self.ada, self.lua, self.xaml, self.bash, self.eval, self.markdown,
        ):
            if lang_changes.changed or lang_changes.deleted:
                return True
        return False


class RAGSearchResult:
    """
    Result from a RAG semantic search.

    ::: This is-in-layer Utility-Layer.
    ::: This is a value-object.
    """

    def __init__(
        self,
        entity_type: str,
        name: str,
        qualified_name: str,
        file: str,
        line: int,
        end_line: Optional[int],
        score: float,
        source_type: str,  # "python", "javascript", "html", "csharp", "cpp", "java", "go", "rust", etc.
        docstring: Optional[str] = None,
        content_preview: Optional[str] = None,
        content: Optional[str] = None,
        heading: Optional[str] = None,
        language: Optional[str] = None,
        class_name: Optional[str] = None,
        # Chunk metadata (optional, for chunked method bodies)
        chunk_index: Optional[int] = None,
        total_chunks: Optional[int] = None,
        chunk_line_start: Optional[int] = None,
        chunk_line_end: Optional[int] = None,
        parent_qualified_name: Optional[str] = None
    ):
        self.entity_type = entity_type
        self.name = name
        self.qualified_name = qualified_name
        self.file = file
        self.line = line
        self.end_line = end_line
        self.score = score
        self.source_type = source_type
        self.docstring = docstring
        self.content_preview = content_preview
        self.content = content
        self.heading = heading
        self.language = language
        self.class_name = class_name
        # Chunk fields
        self.chunk_index = chunk_index
        self.total_chunks = total_chunks
        self.chunk_line_start = chunk_line_start
        self.chunk_line_end = chunk_line_end
        self.parent_qualified_name = parent_qualified_name

    def to_dict(self, include_content: bool = False) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "entity_type": self.entity_type,
            "name": self.name,
            "qualified_name": self.qualified_name,
            "file": self.file,
            "line": self.line,
            "end_line": self.end_line,
            "score": round(self.score, 4),
            "similarity": round(self.score, 4),
            "source_type": self.source_type,
        }

        if self.docstring:
            result["docstring"] = self.docstring
        if self.content_preview:
            result["content_preview"] = self.content_preview
        if self.heading:
            result["heading"] = self.heading
        if self.language:
            result["language"] = self.language
        if self.class_name:
            result["class_name"] = self.class_name
        if include_content and self.content:
            result["content"] = self.content
        # Chunk metadata
        if self.chunk_index is not None:
            result["chunk_index"] = self.chunk_index
        if self.total_chunks is not None:
            result["total_chunks"] = self.total_chunks
        if self.chunk_line_start is not None:
            result["chunk_line_start"] = self.chunk_line_start
        if self.chunk_line_end is not None:
            result["chunk_line_end"] = self.chunk_line_end
        if self.parent_qualified_name:
            result["parent_qualified_name"] = self.parent_qualified_name

        return result
