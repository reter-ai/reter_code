"""
CADSL Parser - Lark-based parser for Codeine Analysis DSL.

This module provides the parser infrastructure for parsing CADSL tool definitions
into an AST that can be transformed into executable Pipeline objects.
"""

from pathlib import Path
from typing import Optional, List, Union
from dataclasses import dataclass, field

from lark import Lark, Tree, Token
from lark.exceptions import (
    UnexpectedInput,
    UnexpectedToken,
    UnexpectedCharacters,
    UnexpectedEOF,
)


# ============================================================
# ERROR TYPES
# ============================================================

@dataclass
class ParseError:
    """Represents a parsing error with location information."""
    message: str
    line: int
    column: int
    context: Optional[str] = None
    suggestion: Optional[str] = None

    def __str__(self) -> str:
        loc = f"line {self.line}, column {self.column}"
        msg = f"Parse error at {loc}: {self.message}"
        if self.context:
            msg += f"\n  Context: {self.context}"
        if self.suggestion:
            msg += f"\n  Suggestion: {self.suggestion}"
        return msg


@dataclass
class ParseResult:
    """Result of parsing CADSL source code."""
    success: bool
    tree: Optional[Tree] = None
    errors: List[ParseError] = field(default_factory=list)
    source: Optional[str] = None

    @property
    def error_count(self) -> int:
        return len(self.errors)

    def __bool__(self) -> bool:
        return self.success


# ============================================================
# PARSER
# ============================================================

class CADSLParser:
    """
    Parser for CADSL (Codeine Analysis DSL) tool definitions.

    Usage:
        parser = CADSLParser()
        result = parser.parse(source_code)
        if result.success:
            tree = result.tree
            # Process tree...
        else:
            for error in result.errors:
                print(error)
    """

    _instance: Optional["CADSLParser"] = None
    _parser: Optional[Lark] = None

    def __new__(cls) -> "CADSLParser":
        """Singleton pattern for parser reuse."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize the parser with the grammar file."""
        if CADSLParser._parser is not None:
            return

        grammar_path = Path(__file__).parent / "grammar.lark"

        if not grammar_path.exists():
            raise FileNotFoundError(
                f"Grammar file not found: {grammar_path}\n"
                "Ensure grammar.lark is in the same directory as parser.py"
            )

        with open(grammar_path, "r", encoding="utf-8") as f:
            grammar = f.read()

        CADSLParser._parser = Lark(
            grammar,
            start="start",
            parser="lalr",
            propagate_positions=True,
            maybe_placeholders=False,
        )

    @property
    def parser(self) -> Lark:
        """Get the Lark parser instance."""
        if CADSLParser._parser is None:
            raise RuntimeError("Parser not initialized")
        return CADSLParser._parser

    @classmethod
    def reset(cls) -> None:
        """Reset the parser cache to force grammar reload on next use."""
        cls._parser = None
        cls._instance = None

    def parse(self, source: str) -> ParseResult:
        """
        Parse CADSL source code into an AST.

        Args:
            source: CADSL source code string

        Returns:
            ParseResult containing the tree or errors
        """
        if not source or not source.strip():
            return ParseResult(
                success=False,
                errors=[ParseError(
                    message="Empty source code",
                    line=1,
                    column=1,
                    suggestion="Provide at least one tool definition"
                )],
                source=source
            )

        try:
            tree = self.parser.parse(source)
            return ParseResult(success=True, tree=tree, source=source)

        except UnexpectedToken as e:
            error = self._handle_unexpected_token(e, source)
            return ParseResult(success=False, errors=[error], source=source)

        except UnexpectedCharacters as e:
            error = self._handle_unexpected_characters(e, source)
            return ParseResult(success=False, errors=[error], source=source)

        except UnexpectedEOF as e:
            error = self._handle_unexpected_eof(e, source)
            return ParseResult(success=False, errors=[error], source=source)

        except UnexpectedInput as e:
            error = self._handle_unexpected_input(e, source)
            return ParseResult(success=False, errors=[error], source=source)

        except Exception as e:
            return ParseResult(
                success=False,
                errors=[ParseError(
                    message=f"Unexpected error: {type(e).__name__}: {e}",
                    line=1,
                    column=1
                )],
                source=source
            )

    def parse_file(self, path: Union[str, Path]) -> ParseResult:
        """
        Parse a CADSL file.

        Args:
            path: Path to the .cadsl file

        Returns:
            ParseResult containing the tree or errors
        """
        path = Path(path)

        if not path.exists():
            return ParseResult(
                success=False,
                errors=[ParseError(
                    message=f"File not found: {path}",
                    line=1,
                    column=1
                )]
            )

        try:
            with open(path, "r", encoding="utf-8") as f:
                source = f.read()
            result = self.parse(source)
            return result

        except IOError as e:
            return ParseResult(
                success=False,
                errors=[ParseError(
                    message=f"Error reading file: {e}",
                    line=1,
                    column=1
                )]
            )

    # ============================================================
    # ERROR HANDLERS
    # ============================================================

    def _handle_unexpected_token(
        self, e: UnexpectedToken, source: str
    ) -> ParseError:
        """Handle unexpected token errors with helpful messages."""
        line = e.line or 1
        column = e.column or 1

        # Get the unexpected token
        token = e.token
        token_value = str(token) if token else "unknown"

        # Get expected tokens
        expected = list(e.expected) if e.expected else []
        expected_str = ", ".join(sorted(expected)[:5])
        if len(expected) > 5:
            expected_str += f" (and {len(expected) - 5} more)"

        # Build message
        message = f"Unexpected token '{token_value}'"
        if expected:
            suggestion = f"Expected one of: {expected_str}"
        else:
            suggestion = None

        # Get context line
        context = self._get_context_line(source, line)

        return ParseError(
            message=message,
            line=line,
            column=column,
            context=context,
            suggestion=suggestion
        )

    def _handle_unexpected_characters(
        self, e: UnexpectedCharacters, source: str
    ) -> ParseError:
        """Handle unexpected character errors."""
        line = e.line or 1
        column = e.column or 1

        # Get the problematic character
        char = e.char if hasattr(e, 'char') else "unknown"

        message = f"Unexpected character '{char}'"
        context = self._get_context_line(source, line)

        # Provide suggestions based on common mistakes
        suggestion = self._suggest_for_char(char, context)

        return ParseError(
            message=message,
            line=line,
            column=column,
            context=context,
            suggestion=suggestion
        )

    def _handle_unexpected_eof(
        self, e: UnexpectedEOF, source: str
    ) -> ParseError:
        """Handle unexpected end-of-file errors."""
        lines = source.split('\n')
        line = len(lines)
        column = len(lines[-1]) + 1 if lines else 1

        expected = list(e.expected) if e.expected else []
        expected_str = ", ".join(sorted(expected)[:5])

        message = "Unexpected end of file"
        suggestion = None

        if "RBRACE" in expected or "}" in str(expected):
            suggestion = "Missing closing brace '}'"
        elif "SEMICOLON" in expected or ";" in str(expected):
            suggestion = "Missing semicolon ';' after parameter"
        elif expected_str:
            suggestion = f"Expected: {expected_str}"

        return ParseError(
            message=message,
            line=line,
            column=column,
            suggestion=suggestion
        )

    def _handle_unexpected_input(
        self, e: UnexpectedInput, source: str
    ) -> ParseError:
        """Handle generic unexpected input errors."""
        line = getattr(e, 'line', 1) or 1
        column = getattr(e, 'column', 1) or 1

        return ParseError(
            message=str(e),
            line=line,
            column=column,
            context=self._get_context_line(source, line)
        )

    # ============================================================
    # HELPERS
    # ============================================================

    def _get_context_line(self, source: str, line: int) -> Optional[str]:
        """Get the source line for context."""
        lines = source.split('\n')
        if 1 <= line <= len(lines):
            return lines[line - 1].rstrip()
        return None

    def _suggest_for_char(
        self, char: str, context: Optional[str]
    ) -> Optional[str]:
        """Suggest fixes for common character errors."""
        if char == ':':
            return "Use '=' for metadata values, ':' for type annotations"
        if char == '(':
            return "Parentheses are for metadata, use '{' for blocks"
        if char == '[':
            return "Square brackets are for lists and choices"
        if char == '"' or char == "'":
            return "Check for unclosed string"
        return None


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def parse_cadsl(source: str) -> ParseResult:
    """
    Convenience function to parse CADSL source code.

    Args:
        source: CADSL source code string

    Returns:
        ParseResult containing the tree or errors
    """
    parser = CADSLParser()
    return parser.parse(source)


def parse_cadsl_file(path: Union[str, Path]) -> ParseResult:
    """
    Convenience function to parse a CADSL file.

    Args:
        path: Path to the .cadsl file

    Returns:
        ParseResult containing the tree or errors
    """
    parser = CADSLParser()
    return parser.parse_file(path)


# ============================================================
# AST UTILITIES
# ============================================================

def pretty_print_tree(tree: Tree, indent: int = 0) -> str:
    """
    Pretty print a parse tree for debugging.

    Args:
        tree: Lark parse tree
        indent: Current indentation level

    Returns:
        Formatted string representation of the tree
    """
    lines = []
    prefix = "  " * indent

    if isinstance(tree, Tree):
        lines.append(f"{prefix}{tree.data}")
        for child in tree.children:
            lines.append(pretty_print_tree(child, indent + 1))
    elif isinstance(tree, Token):
        lines.append(f"{prefix}{tree.type}: {tree.value!r}")
    else:
        lines.append(f"{prefix}{tree!r}")

    return "\n".join(lines)


def get_tool_names(tree: Tree) -> List[str]:
    """
    Extract tool names from a parse tree.

    Args:
        tree: Parsed CADSL tree

    Returns:
        List of tool names defined in the source
    """
    names = []
    for child in tree.children:
        if isinstance(child, Tree) and child.data == "tool_def":
            # Tool name is the second child (after tool_type)
            for grandchild in child.children:
                if isinstance(grandchild, Token) and grandchild.type == "NAME":
                    names.append(str(grandchild))
                    break
    return names


def count_tools(tree: Tree) -> dict:
    """
    Count tools by type in a parse tree.

    Args:
        tree: Parsed CADSL tree

    Returns:
        Dictionary with counts: {"query": N, "detector": N, "diagram": N}
    """
    counts = {"query": 0, "detector": 0, "diagram": 0}

    for child in tree.children:
        if isinstance(child, Tree) and child.data == "tool_def":
            tool_type_node = child.children[0]
            if isinstance(tool_type_node, Tree):
                tool_type = tool_type_node.data
                if tool_type == "tool_query":
                    counts["query"] += 1
                elif tool_type == "tool_detector":
                    counts["detector"] += 1
                elif tool_type == "tool_diagram":
                    counts["diagram"] += 1

    return counts
