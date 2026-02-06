"""CADSL utility functions shared across validator, transformer, and other modules."""

from typing import Any, Callable, Optional, Union
from lark import Tree, Token


def unquote(s: str) -> str:
    """Remove surrounding quotes from a string.

    Handles both single and double quotes.

    Args:
        s: String potentially wrapped in quotes

    Returns:
        String with surrounding quotes removed, or original if not quoted
    """
    if len(s) >= 2:
        if (s.startswith('"') and s.endswith('"')) or \
           (s.startswith("'") and s.endswith("'")):
            return s[1:-1]
    return s


def token_to_value(token: Token) -> Any:
    """Convert a Lark Token to a Python value.

    Args:
        token: Lark Token node

    Returns:
        Converted Python value (str, int, float, bool, None, or raw string)
    """
    if token.type == "STRING":
        return unquote(str(token))
    elif token.type in ("SIGNED_INT", "INT"):
        return int(str(token))
    elif token.type in ("SIGNED_FLOAT", "FLOAT"):
        return float(str(token))
    elif token.type == "NAME":
        name = str(token)
        if name == "true":
            return True
        elif name == "false":
            return False
        elif name == "null":
            return None
        return name
    return str(token)


def extract_value(
    node: Union[Tree, Token],
    extract_list_fn: Optional[Callable[[Tree], Any]] = None
) -> Any:
    """Extract a literal value from a Lark AST node.

    Handles both Token and Tree nodes, recursively unwrapping nested structures.

    Args:
        node: Lark Tree or Token node
        extract_list_fn: Optional callback for handling val_list/capability_array nodes

    Returns:
        Extracted Python value (str, int, float, bool, None, or list if extract_list_fn provided)
    """
    if isinstance(node, Token):
        return token_to_value(node)

    if isinstance(node, Tree):
        # Handle known tree types directly
        if node.data == "val_string":
            return unquote(str(node.children[0]))
        elif node.data == "val_int":
            return int(str(node.children[0]))
        elif node.data == "val_float":
            return float(str(node.children[0]))
        elif node.data == "val_true":
            return True
        elif node.data == "val_false":
            return False
        elif node.data == "val_null":
            return None
        elif node.data in ("val_list", "capability_array") and extract_list_fn:
            return extract_list_fn(node)

        # Recurse into children
        for child in node.children:
            if isinstance(child, Token):
                return token_to_value(child)
            elif isinstance(child, Tree):
                return extract_value(child, extract_list_fn)

    return None


def get_tool_name(node: Tree) -> str:
    """Extract tool name from tool_def node.

    Args:
        node: Lark Tree node (tool_def)

    Returns:
        Tool name string, or "unnamed" if not found
    """
    for child in node.children:
        if isinstance(child, Token) and child.type == "NAME":
            return str(child)
    return "unnamed"


def get_tool_type(node: Tree, default: str = "query") -> str:
    """Extract tool type from tool_def node.

    Args:
        node: Lark Tree node (tool_def)
        default: Default type if not found (default: "query")

    Returns:
        Tool type: "query", "detector", "diagram", or default
    """
    for child in node.children:
        if isinstance(child, Tree):
            if child.data == "tool_query":
                return "query"
            elif child.data == "tool_detector":
                return "detector"
            elif child.data == "tool_diagram":
                return "diagram"
    return default
