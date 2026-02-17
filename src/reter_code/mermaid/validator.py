"""
Mermaid syntax validator using Lark grammar.

Validates mermaid diagram syntax before pushing to the ViewServer.
Supports 13 diagram types covering ~95% of real-world usage.
Unknown diagram types get a lenient pass-through (valid=True + warning).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from lark import Lark
from lark.exceptions import (
    UnexpectedCharacters,
    UnexpectedEOF,
    UnexpectedInput,
    UnexpectedToken,
)

logger = logging.getLogger(__name__)

# ============================================================
# DATA CLASSES
# ============================================================


@dataclass
class MermaidValidationError:
    """A single validation error with location info."""

    message: str
    line: int = 0
    column: int = 0
    context: Optional[str] = None
    suggestion: Optional[str] = None

    def to_dict(self) -> dict:
        d = {"message": self.message, "line": self.line, "column": self.column}
        if self.context:
            d["context"] = self.context
        if self.suggestion:
            d["suggestion"] = self.suggestion
        return d


@dataclass
class MermaidValidationResult:
    """Result of validating mermaid content."""

    valid: bool
    diagram_type: Optional[str] = None
    errors: list[MermaidValidationError] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = {
            "valid": self.valid,
            "diagram_type": self.diagram_type,
        }
        if self.errors:
            d["errors"] = [e.to_dict() for e in self.errors]
        if self.warnings:
            d["warnings"] = self.warnings
        return d


# ============================================================
# TYPE DETECTION MAP
# ============================================================

# Maps first-line keyword (lowercased) to (start_rule, display_name)
_TYPE_MAP: dict[str, tuple[str, str]] = {
    "flowchart": ("flowchart_diagram", "flowchart"),
    "graph": ("flowchart_diagram", "flowchart"),
    "sequencediagram": ("sequence_diagram", "sequence"),
    "classdiagram": ("class_diagram", "class"),
    "statediagram-v2": ("state_diagram", "state"),
    "statediagram": ("state_diagram", "state"),
    "erdiagram": ("er_diagram", "er"),
    "gantt": ("gantt_diagram", "gantt"),
    "pie": ("pie_diagram", "pie"),
    "block-beta": ("block_beta_diagram", "block-beta"),
    "mindmap": ("mindmap_diagram", "mindmap"),
    "timeline": ("timeline_diagram", "timeline"),
    "gitgraph": ("gitgraph_diagram", "gitgraph"),
    "journey": ("journey_diagram", "journey"),
}

# Known but unsupported types — pass-through with warning
_KNOWN_UNSUPPORTED = {
    "quadrantchart",
    "requirementdiagram",
    "c4context",
    "c4container",
    "c4component",
    "c4deployment",
    "sankey-beta",
    "xychart-beta",
    "packet-beta",
    "kanban",
    "architecture-beta",
    "zenuml",
}


# ============================================================
# VALIDATOR
# ============================================================


class MermaidValidator:
    """
    Validates mermaid diagram syntax using a Lark grammar.

    Singleton — the grammar is loaded once and cached.
    Uses Earley parser to handle ambiguous label text gracefully.
    """

    _instance: Optional[MermaidValidator] = None

    def __new__(cls) -> MermaidValidator:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._parsers = {}
            cls._instance._load_grammar()
        return cls._instance

    def _load_grammar(self) -> None:
        """Load the Lark grammar and create parsers for each diagram type."""
        grammar_path = Path(__file__).parent / "mermaid_validator.lark"
        if not grammar_path.exists():
            raise FileNotFoundError(
                f"Mermaid grammar not found: {grammar_path}"
            )
        grammar_text = grammar_path.read_text(encoding="utf-8")

        # Create one parser per start rule
        for keyword, (start_rule, _name) in _TYPE_MAP.items():
            if start_rule not in self._parsers:
                try:
                    self._parsers[start_rule] = Lark(
                        grammar_text,
                        start=start_rule,
                        parser="earley",
                        propagate_positions=True,
                        ambiguity="forest",
                    )
                except Exception as e:
                    logger.warning(
                        "Failed to create parser for %s: %s", start_rule, e
                    )

    def detect_type(self, content: str) -> tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Detect mermaid diagram type from content.

        Returns:
            (start_rule, display_name, keyword) or (None, None, None)
        """
        stripped = content.strip()
        if not stripped:
            return None, None, None

        first_line = stripped.split("\n", 1)[0].strip()
        first_word = re.split(r"[\s\n]", first_line, maxsplit=1)[0].lower()

        # Check supported types
        if first_word in _TYPE_MAP:
            start_rule, display_name = _TYPE_MAP[first_word]
            return start_rule, display_name, first_word

        # Check known-unsupported types
        if first_word in _KNOWN_UNSUPPORTED:
            return None, first_word, first_word

        return None, None, first_word if first_word else None

    def validate(self, content: str) -> MermaidValidationResult:
        """
        Validate mermaid content.

        Returns MermaidValidationResult with valid=True/False plus details.
        Unknown but recognized types pass through with a warning.
        """
        if not content or not content.strip():
            return MermaidValidationResult(
                valid=False,
                errors=[
                    MermaidValidationError(
                        message="Empty content",
                        suggestion="Provide mermaid diagram content",
                    )
                ],
            )

        start_rule, display_name, keyword = self.detect_type(content)

        # Known-unsupported type — lenient pass-through
        if start_rule is None and display_name is not None:
            return MermaidValidationResult(
                valid=True,
                diagram_type=display_name,
                warnings=[
                    f"Diagram type '{display_name}' is not validated "
                    "(grammar not available); syntax errors may occur at render time"
                ],
            )

        # Completely unknown type
        if start_rule is None:
            return MermaidValidationResult(
                valid=False,
                errors=[
                    MermaidValidationError(
                        message=f"Unrecognized diagram type: '{keyword}'",
                        line=1,
                        column=1,
                        context=content.strip().split("\n", 1)[0][:80],
                        suggestion=(
                            "Supported types: flowchart, graph, sequenceDiagram, "
                            "classDiagram, stateDiagram, erDiagram, gantt, pie, "
                            "block-beta, mindmap, timeline, gitgraph, journey"
                        ),
                    )
                ],
            )

        # Parser not available (grammar load failed)
        parser = self._parsers.get(start_rule)
        if parser is None:
            return MermaidValidationResult(
                valid=True,
                diagram_type=display_name,
                warnings=[
                    f"Parser for '{display_name}' not available; skipping validation"
                ],
            )

        # Normalize content for parsing
        normalized = _normalize(content)

        # Parse
        try:
            parser.parse(normalized)
            return MermaidValidationResult(
                valid=True, diagram_type=display_name
            )
        except UnexpectedToken as e:
            return _handle_unexpected_token(e, content, display_name)
        except UnexpectedCharacters as e:
            return _handle_unexpected_chars(e, content, display_name)
        except UnexpectedEOF as e:
            return _handle_unexpected_eof(e, content, display_name)
        except UnexpectedInput as e:
            return _handle_generic_error(e, content, display_name)
        except Exception as e:
            # Unexpected error — don't block rendering
            logger.debug("Mermaid validation internal error: %s", e)
            return MermaidValidationResult(
                valid=True,
                diagram_type=display_name,
                warnings=[f"Validation error (non-blocking): {e}"],
            )


# ============================================================
# HELPERS
# ============================================================


def _normalize(content: str) -> str:
    """Normalize content for parsing."""
    # Strip leading/trailing whitespace, ensure trailing newline
    text = content.strip()
    if not text.endswith("\n"):
        text += "\n"
    return text


def _get_line(content: str, line_no: int) -> Optional[str]:
    """Get a specific line from content (1-indexed)."""
    lines = content.split("\n")
    if 1 <= line_no <= len(lines):
        return lines[line_no - 1]
    return None


def _handle_unexpected_token(
    e: UnexpectedToken, content: str, diagram_type: str
) -> MermaidValidationResult:
    line = getattr(e, "line", 0) or 0
    col = getattr(e, "column", 0) or 0
    ctx = _get_line(content, line)

    expected = getattr(e, "expected", set())
    expected_str = ", ".join(sorted(str(x) for x in expected)[:5]) if expected else ""
    token = getattr(e, "token", None)
    token_str = repr(str(token)) if token else ""

    msg = f"Unexpected token {token_str}"
    suggestion = f"Expected one of: {expected_str}" if expected_str else None

    return MermaidValidationResult(
        valid=False,
        diagram_type=diagram_type,
        errors=[
            MermaidValidationError(
                message=msg,
                line=line,
                column=col,
                context=ctx[:120] if ctx else None,
                suggestion=suggestion,
            )
        ],
    )


def _handle_unexpected_chars(
    e: UnexpectedCharacters, content: str, diagram_type: str
) -> MermaidValidationResult:
    line = getattr(e, "line", 0) or 0
    col = getattr(e, "column", 0) or 0
    ctx = _get_line(content, line)
    char = getattr(e, "char", "")

    return MermaidValidationResult(
        valid=False,
        diagram_type=diagram_type,
        errors=[
            MermaidValidationError(
                message=f"Unexpected character '{char}'",
                line=line,
                column=col,
                context=ctx[:120] if ctx else None,
            )
        ],
    )


def _handle_unexpected_eof(
    e: UnexpectedEOF, content: str, diagram_type: str
) -> MermaidValidationResult:
    lines = content.strip().split("\n")
    line = len(lines)

    expected = getattr(e, "expected", set())
    expected_str = ", ".join(sorted(str(x) for x in expected)[:5]) if expected else ""
    suggestion = None

    # Common case: missing 'end' keyword
    if any("end" in str(x).lower() for x in expected):
        suggestion = "Missing 'end' keyword — check that all blocks are closed"
    elif expected_str:
        suggestion = f"Expected: {expected_str}"

    return MermaidValidationResult(
        valid=False,
        diagram_type=diagram_type,
        errors=[
            MermaidValidationError(
                message="Unexpected end of input",
                line=line,
                column=0,
                context=_get_line(content, line),
                suggestion=suggestion,
            )
        ],
    )


def _handle_generic_error(
    e: UnexpectedInput, content: str, diagram_type: str
) -> MermaidValidationResult:
    line = getattr(e, "line", 0) or 0
    col = getattr(e, "column", 0) or 0
    ctx = _get_line(content, line)

    return MermaidValidationResult(
        valid=False,
        diagram_type=diagram_type,
        errors=[
            MermaidValidationError(
                message=str(e)[:200],
                line=line,
                column=col,
                context=ctx[:120] if ctx else None,
            )
        ],
    )


# ============================================================
# MODULE-LEVEL CONVENIENCE
# ============================================================


def validate_mermaid(content: str) -> MermaidValidationResult:
    """Validate mermaid diagram syntax. Module-level convenience function."""
    return MermaidValidator().validate(content)
