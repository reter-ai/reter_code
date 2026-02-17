"""
Markdown validator with embedded mermaid block validation.

Extracts fenced code blocks via regex, optionally parses remaining
markdown structure with a Lark grammar, and validates each ```mermaid
block using the existing MermaidValidator.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from lark import Lark

from reter_code.mermaid.validator import (
    MermaidValidationError,
    MermaidValidationResult,
    validate_mermaid,
)

logger = logging.getLogger(__name__)

# ============================================================
# DATA CLASSES
# ============================================================


@dataclass
class CodeBlock:
    """A fenced code block extracted from markdown."""

    language: Optional[str]
    content: str
    start_line: int  # 1-indexed, line of opening fence
    end_line: int  # 1-indexed, line of closing fence


@dataclass
class MarkdownValidationResult:
    """Result of validating markdown content (including embedded mermaid)."""

    valid: bool
    code_blocks: list[CodeBlock] = field(default_factory=list)
    mermaid_results: list[MermaidValidationResult] = field(default_factory=list)
    errors: list[MermaidValidationError] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d: dict = {
            "valid": self.valid,
            "code_blocks": len(self.code_blocks),
        }
        if self.mermaid_results:
            d["mermaid_blocks"] = [r.to_dict() for r in self.mermaid_results]
        if self.errors:
            d["errors"] = [e.to_dict() for e in self.errors]
        if self.warnings:
            d["warnings"] = self.warnings
        return d


# ============================================================
# FENCE EXTRACTION
# ============================================================

# Matches opening fence: ``` or ~~~ with optional language tag
_FENCE_OPEN = re.compile(r"^(?P<indent>[ \t]{0,3})(?P<fence>`{3,}|~{3,})(?P<lang>[^\s`]*).*$")


def extract_code_fences(
    content: str,
) -> tuple[str, list[CodeBlock], list[MermaidValidationError]]:
    """Extract fenced code blocks and replace with placeholders.

    Returns:
        (processed_content, code_blocks, errors)
        where processed_content has code blocks replaced with ___CODEBLOCK_N___.
    """
    lines = content.split("\n")
    blocks: list[CodeBlock] = []
    errors: list[MermaidValidationError] = []
    output_lines: list[str] = []

    in_fence = False
    fence_char = ""
    fence_min_len = 0
    fence_lang: Optional[str] = None
    fence_start = 0
    fence_content_lines: list[str] = []

    for i, line in enumerate(lines, 1):
        if not in_fence:
            m = _FENCE_OPEN.match(line)
            if m:
                fence_char = m.group("fence")[0]
                fence_min_len = len(m.group("fence"))
                lang = m.group("lang").strip().lower()
                fence_lang = lang if lang else None
                fence_start = i
                fence_content_lines = []
                in_fence = True
                continue
            output_lines.append(line)
        else:
            # Check for closing fence: same char, at least same length, no info string
            stripped = line.lstrip()
            if stripped.startswith(fence_char * fence_min_len) and stripped.rstrip() == fence_char * len(stripped.rstrip()):
                # Verify it's only fence characters (and optional trailing whitespace)
                close_stripped = line.strip()
                if all(c == fence_char for c in close_stripped) and len(close_stripped) >= fence_min_len:
                    block = CodeBlock(
                        language=fence_lang,
                        content="\n".join(fence_content_lines),
                        start_line=fence_start,
                        end_line=i,
                    )
                    blocks.append(block)
                    placeholder = f"___CODEBLOCK_{len(blocks) - 1}___"
                    output_lines.append(placeholder)
                    in_fence = False
                    continue
            fence_content_lines.append(line)

    if in_fence:
        errors.append(
            MermaidValidationError(
                message=f"Unclosed code fence starting at line {fence_start}",
                line=fence_start,
                context=lines[fence_start - 1] if fence_start <= len(lines) else None,
                suggestion="Add a closing ``` or ~~~ fence",
            )
        )

    processed = "\n".join(output_lines)
    return processed, blocks, errors


# ============================================================
# MARKDOWN VALIDATOR
# ============================================================


class MarkdownValidator:
    """Validates markdown content with embedded mermaid block checking.

    Singleton — grammar loaded once and cached.
    """

    _instance: Optional[MarkdownValidator] = None

    def __new__(cls) -> MarkdownValidator:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._parser = None
            cls._instance._load_grammar()
        return cls._instance

    def _load_grammar(self) -> None:
        grammar_path = Path(__file__).parent / "markdown_grammar.lark"
        try:
            grammar_text = grammar_path.read_text(encoding="utf-8")
            self._parser = Lark(
                grammar_text,
                parser="earley",
                start="start",
                propagate_positions=True,
            )
        except Exception as e:
            logger.warning("Failed to load markdown grammar: %s", e)
            self._parser = None

    def validate(self, content: str) -> MarkdownValidationResult:
        """Validate markdown content including embedded mermaid blocks."""
        if not content or not content.strip():
            return MarkdownValidationResult(
                valid=False,
                errors=[MermaidValidationError(message="Empty content", line=0)],
            )

        # Phase 1: Extract code fences
        processed, blocks, fence_errors = extract_code_fences(content)

        all_errors: list[MermaidValidationError] = list(fence_errors)
        warnings: list[str] = []
        mermaid_results: list[MermaidValidationResult] = []

        # Phase 2: Parse markdown structure (non-blocking)
        if self._parser and processed.strip():
            try:
                text = processed.strip()
                if not text.endswith("\n"):
                    text += "\n"
                self._parser.parse(text)
            except Exception as e:
                warnings.append(f"Markdown structure parse note: {e}")

        # Phase 3: Validate mermaid blocks
        for block in blocks:
            if block.language == "mermaid":
                vr = validate_mermaid(block.content)
                mermaid_results.append(vr)
                if not vr.valid:
                    for err in vr.errors:
                        adjusted_line = (
                            block.start_line + err.line
                            if err.line > 0
                            else block.start_line
                        )
                        all_errors.append(
                            MermaidValidationError(
                                message=err.message,
                                line=adjusted_line,
                                column=err.column,
                                context=err.context,
                                suggestion=err.suggestion,
                            )
                        )
                if vr.warnings:
                    warnings.extend(vr.warnings)

        valid = len(fence_errors) == 0 and all(
            r.valid for r in mermaid_results
        )

        return MarkdownValidationResult(
            valid=valid,
            code_blocks=blocks,
            mermaid_results=mermaid_results,
            errors=all_errors,
            warnings=warnings,
        )


# ============================================================
# CONVENIENCE FUNCTION
# ============================================================


def validate_markdown(content: str) -> MarkdownValidationResult:
    """Validate markdown content. Module-level convenience function."""
    return MarkdownValidator().validate(content)
