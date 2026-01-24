"""
CADSL Validator - Parse-time validation for CADSL tool definitions.

This module validates CADSL parse trees for:
- Type correctness (parameter types, default values)
- Semantic correctness (required fields, valid references)
- Security level validation
- Capability format validation
"""

from dataclasses import dataclass, field
from typing import List, Optional, Set, Dict, Any, Union
from enum import Enum

from lark import Tree, Token


# ============================================================
# VALIDATION RESULT TYPES
# ============================================================

class Severity(Enum):
    """Severity level for validation issues."""
    ERROR = "error"      # Must be fixed, blocks compilation
    WARNING = "warning"  # Should be fixed, but allows compilation
    INFO = "info"        # Informational, style suggestions


@dataclass
class ValidationIssue:
    """Represents a validation issue found in CADSL code."""
    severity: Severity
    message: str
    line: Optional[int] = None
    column: Optional[int] = None
    node_type: Optional[str] = None
    suggestion: Optional[str] = None

    def __str__(self) -> str:
        loc = ""
        if self.line:
            loc = f" at line {self.line}"
            if self.column:
                loc += f", column {self.column}"

        msg = f"[{self.severity.value.upper()}]{loc}: {self.message}"
        if self.suggestion:
            msg += f"\n  Suggestion: {self.suggestion}"
        return msg


@dataclass
class ValidationResult:
    """Result of validating a CADSL parse tree."""
    valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    tool_info: Dict[str, Any] = field(default_factory=dict)

    @property
    def errors(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == Severity.ERROR]

    @property
    def warnings(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == Severity.WARNING]

    @property
    def error_count(self) -> int:
        return len(self.errors)

    @property
    def warning_count(self) -> int:
        return len(self.warnings)

    def __bool__(self) -> bool:
        return self.valid


# ============================================================
# TYPE SYSTEM
# ============================================================

# Valid CADSL types
VALID_TYPES = {"int", "str", "float", "bool", "list"}

# Valid security levels
VALID_SECURITY_LEVELS = {"trusted", "standard", "restricted"}

# Valid tool metadata keys
VALID_METADATA_KEYS = {
    "category", "severity", "security", "capabilities",
    "version", "author", "description", "tags"
}

# Valid detector severities
VALID_SEVERITIES = {"info", "low", "medium", "high", "critical"}

# Valid detector categories
VALID_CATEGORIES = {
    "code_smell", "design", "complexity", "dependencies",
    "test_coverage", "security", "performance", "style"
}

# Capability pattern format
CAPABILITY_PREFIXES = {"fs:read:", "fs:write:", "net:http:", "env:", "subprocess:"}


# ============================================================
# VALIDATOR
# ============================================================

class CADSLValidator:
    """
    Validates CADSL parse trees for correctness.

    Usage:
        validator = CADSLValidator()
        result = validator.validate(tree)
        if not result.valid:
            for issue in result.issues:
                print(issue)
    """

    def __init__(self, strict: bool = False):
        """
        Initialize the validator.

        Args:
            strict: If True, treat warnings as errors
        """
        self.strict = strict
        self.issues: List[ValidationIssue] = []
        self.current_tool: Optional[str] = None
        self.defined_params: Set[str] = set()
        self.tool_info: Dict[str, Any] = {}

    def validate(self, tree: Tree) -> ValidationResult:
        """
        Validate a CADSL parse tree.

        Args:
            tree: Lark parse tree from CADSLParser

        Returns:
            ValidationResult with issues and tool info
        """
        self.issues = []
        self.tool_info = {"tools": []}

        # Validate each tool definition
        for child in tree.children:
            if isinstance(child, Tree) and child.data == "tool_def":
                self._validate_tool_def(child)

        # Check for duplicate tool names
        self._check_duplicate_names()

        # Determine validity
        has_errors = any(i.severity == Severity.ERROR for i in self.issues)
        if self.strict:
            has_errors = has_errors or any(
                i.severity == Severity.WARNING for i in self.issues
            )

        return ValidationResult(
            valid=not has_errors,
            issues=self.issues,
            tool_info=self.tool_info
        )

    # ============================================================
    # TOOL DEFINITION VALIDATION
    # ============================================================

    def _validate_tool_def(self, node: Tree) -> None:
        """Validate a tool definition."""
        self.defined_params = set()

        # Extract tool info
        tool_type = self._get_tool_type(node)
        tool_name = self._get_tool_name(node)
        self.current_tool = tool_name

        tool_entry = {
            "name": tool_name,
            "type": tool_type,
            "params": [],
            "metadata": {}
        }

        # Validate metadata
        metadata_node = self._find_child(node, "metadata")
        if metadata_node:
            tool_entry["metadata"] = self._validate_metadata(
                metadata_node, tool_type
            )

        # Validate tool body
        tool_body = self._find_child(node, "tool_body")
        if tool_body:
            # Validate parameters
            for child in tool_body.children:
                if isinstance(child, Tree) and child.data == "param_def":
                    param_info = self._validate_param_def(child)
                    if param_info:
                        tool_entry["params"].append(param_info)

            # Validate pipeline
            pipeline = self._find_child(tool_body, "pipeline")
            if pipeline:
                self._validate_pipeline(pipeline)
            else:
                self._add_issue(
                    Severity.ERROR,
                    f"Tool '{tool_name}' is missing a pipeline",
                    node=node
                )

        self.tool_info["tools"].append(tool_entry)
        self.current_tool = None

    def _get_tool_type(self, node: Tree) -> str:
        """Extract tool type from tool_def node."""
        for child in node.children:
            if isinstance(child, Tree):
                if child.data == "tool_query":
                    return "query"
                elif child.data == "tool_detector":
                    return "detector"
                elif child.data == "tool_diagram":
                    return "diagram"
        return "unknown"

    def _get_tool_name(self, node: Tree) -> str:
        """Extract tool name from tool_def node."""
        for child in node.children:
            if isinstance(child, Token) and child.type == "NAME":
                return str(child)
        return "unnamed"

    # ============================================================
    # METADATA VALIDATION
    # ============================================================

    def _validate_metadata(
        self, node: Tree, tool_type: str
    ) -> Dict[str, Any]:
        """Validate tool metadata."""
        metadata = {}

        for child in node.children:
            if isinstance(child, Tree) and child.data == "meta_item":
                key, value = self._extract_meta_item(child)

                # Validate known keys
                if key not in VALID_METADATA_KEYS:
                    self._add_issue(
                        Severity.WARNING,
                        f"Unknown metadata key '{key}'",
                        node=child,
                        suggestion=f"Valid keys: {', '.join(sorted(VALID_METADATA_KEYS))}"
                    )

                # Validate specific values
                if key == "severity":
                    self._validate_severity_value(value, child)
                elif key == "category":
                    self._validate_category_value(value, child)
                elif key == "security":
                    self._validate_security_value(value, child)
                elif key == "capabilities":
                    self._validate_capabilities(value, child)

                metadata[key] = value

        # Check required metadata for detectors
        if tool_type == "detector":
            if "category" not in metadata:
                self._add_issue(
                    Severity.WARNING,
                    "Detector is missing 'category' metadata",
                    node=node,
                    suggestion="Add category='code_smell' or similar"
                )
            if "severity" not in metadata:
                self._add_issue(
                    Severity.WARNING,
                    "Detector is missing 'severity' metadata",
                    node=node,
                    suggestion="Add severity='medium' or similar"
                )

        return metadata

    def _extract_meta_item(self, node: Tree) -> tuple:
        """Extract key-value pair from meta_item node."""
        key = None
        value = None

        for child in node.children:
            if isinstance(child, Token) and child.type == "NAME":
                key = str(child)
            elif isinstance(child, Tree) and child.data == "meta_value":
                value = self._extract_value(child)
            elif isinstance(child, Token) and child.type == "STRING":
                value = self._unquote(str(child))

        return key, value

    def _validate_severity_value(self, value: Any, node: Tree) -> None:
        """Validate severity metadata value."""
        if isinstance(value, str) and value not in VALID_SEVERITIES:
            self._add_issue(
                Severity.WARNING,
                f"Invalid severity '{value}'",
                node=node,
                suggestion=f"Valid severities: {', '.join(sorted(VALID_SEVERITIES))}"
            )

    def _validate_category_value(self, value: Any, node: Tree) -> None:
        """Validate category metadata value."""
        if isinstance(value, str) and value not in VALID_CATEGORIES:
            self._add_issue(
                Severity.INFO,
                f"Non-standard category '{value}'",
                node=node,
                suggestion=f"Standard categories: {', '.join(sorted(VALID_CATEGORIES))}"
            )

    def _validate_security_value(self, value: Any, node: Tree) -> None:
        """Validate security level metadata value."""
        if isinstance(value, str) and value not in VALID_SECURITY_LEVELS:
            self._add_issue(
                Severity.ERROR,
                f"Invalid security level '{value}'",
                node=node,
                suggestion=f"Valid levels: {', '.join(sorted(VALID_SECURITY_LEVELS))}"
            )

    def _validate_capabilities(self, value: Any, node: Tree) -> None:
        """Validate capability list."""
        if not isinstance(value, list):
            return

        for cap in value:
            if not isinstance(cap, str):
                continue

            # Check capability format
            valid_format = any(cap.startswith(p) for p in CAPABILITY_PREFIXES)
            if not valid_format:
                self._add_issue(
                    Severity.ERROR,
                    f"Invalid capability format: '{cap}'",
                    node=node,
                    suggestion="Capabilities must start with: " +
                               ", ".join(sorted(CAPABILITY_PREFIXES))
                )

    # ============================================================
    # PARAMETER VALIDATION
    # ============================================================

    def _validate_param_def(self, node: Tree) -> Optional[Dict[str, Any]]:
        """Validate a parameter definition."""
        param_name = None
        param_type = None
        default_value = None
        is_required = False
        choices = None

        for child in node.children:
            if isinstance(child, Token) and child.type == "NAME":
                param_name = str(child)
            elif isinstance(child, Tree):
                if child.data.startswith("type_"):
                    param_type = self._extract_type(child)
                elif child.data == "param_modifiers":
                    for mod in child.children:
                        if isinstance(mod, Tree):
                            if mod.data == "param_default":
                                default_value = self._extract_value(mod)
                            elif mod.data == "param_required":
                                is_required = True
                            elif mod.data == "param_choices":
                                choices = self._extract_choices(mod)

        # Check for duplicate parameter names
        if param_name:
            if param_name in self.defined_params:
                self._add_issue(
                    Severity.ERROR,
                    f"Duplicate parameter name '{param_name}'",
                    node=node
                )
            else:
                self.defined_params.add(param_name)

        # Validate type
        if param_type and param_type not in VALID_TYPES:
            if not param_type.startswith("list<"):
                self._add_issue(
                    Severity.ERROR,
                    f"Invalid type '{param_type}' for parameter '{param_name}'",
                    node=node,
                    suggestion=f"Valid types: {', '.join(sorted(VALID_TYPES))}"
                )

        # Validate default value matches type
        if default_value is not None and param_type:
            self._validate_value_type(default_value, param_type, node, param_name)

        # Validate choices
        if choices and param_type:
            for choice in choices:
                self._validate_value_type(choice, param_type, node, param_name)

        # Check for required with default (warning)
        if is_required and default_value is not None:
            self._add_issue(
                Severity.WARNING,
                f"Parameter '{param_name}' is marked required but has default",
                node=node,
                suggestion="Remove 'required' or the default value"
            )

        return {
            "name": param_name,
            "type": param_type,
            "default": default_value,
            "required": is_required,
            "choices": choices
        }

    def _extract_type(self, node: Tree) -> str:
        """Extract type from type_spec node."""
        if node.data == "type_int":
            return "int"
        elif node.data == "type_str":
            return "str"
        elif node.data == "type_float":
            return "float"
        elif node.data == "type_bool":
            return "bool"
        elif node.data == "type_list":
            return "list"
        elif node.data == "type_list_of":
            inner = self._find_child(node, "type_")
            if inner:
                return f"list<{self._extract_type(inner)}>"
            return "list"
        return "unknown"

    def _validate_value_type(
        self,
        value: Any,
        expected_type: str,
        node: Tree,
        param_name: str
    ) -> None:
        """Validate that a value matches the expected type."""
        actual_type = type(value).__name__

        type_matches = {
            "int": isinstance(value, int) and not isinstance(value, bool),
            "float": isinstance(value, (int, float)) and not isinstance(value, bool),
            "str": isinstance(value, str),
            "bool": isinstance(value, bool),
            "list": isinstance(value, list),
        }

        if expected_type in type_matches:
            if not type_matches[expected_type]:
                # Allow null for any type
                if value is None:
                    return
                self._add_issue(
                    Severity.ERROR,
                    f"Default value for '{param_name}' has wrong type: "
                    f"expected {expected_type}, got {actual_type}",
                    node=node
                )

    def _extract_choices(self, node: Tree) -> List[Any]:
        """Extract choices from param_choices node."""
        choices = []
        for child in node.children:
            if isinstance(child, Tree) and child.data == "value_list":
                for value_node in child.children:
                    if isinstance(value_node, Tree):
                        choices.append(self._extract_value(value_node))
        return choices

    # ============================================================
    # PIPELINE VALIDATION
    # ============================================================

    def _validate_pipeline(self, node: Tree) -> None:
        """Validate a pipeline definition."""
        has_source = False
        has_emit = False
        step_count = 0

        for child in node.children:
            if isinstance(child, Tree):
                # Check for source (which contains reql_source, rag_source, etc.)
                if child.data == "source":
                    has_source = True
                    for source_child in child.children:
                        if isinstance(source_child, Tree):
                            self._validate_source(source_child)
                elif child.data in ("reql_source", "rag_source", "value_source"):
                    has_source = True
                    self._validate_source(child)
                # Check for step (which contains the actual step type)
                elif child.data == "step":
                    for step_child in child.children:
                        if isinstance(step_child, Tree):
                            if step_child.data == "emit_step":
                                has_emit = True
                            elif step_child.data.endswith("_step"):
                                step_count += 1
                                self._validate_step(step_child)
                elif child.data == "emit_step":
                    has_emit = True
                elif child.data.endswith("_step"):
                    step_count += 1
                    self._validate_step(child)

        if not has_source:
            self._add_issue(
                Severity.ERROR,
                f"Pipeline in '{self.current_tool}' is missing a source "
                "(reql, rag, or value)",
                node=node
            )

        if not has_emit:
            self._add_issue(
                Severity.WARNING,
                f"Pipeline in '{self.current_tool}' is missing emit step",
                node=node,
                suggestion="Add '| emit { result_key }' at the end"
            )

    def _validate_source(self, node: Tree) -> None:
        """Validate a pipeline source."""
        if node.data == "reql_source":
            # Check REQL content exists (now stored in REQL_BLOCK token)
            has_content = False
            for child in node.children:
                if isinstance(child, Token) and child.type == "REQL_BLOCK":
                    content = str(child).strip()
                    if content and len(content) > 2:  # More than just {}
                        has_content = True
                    break
            if not has_content:
                self._add_issue(
                    Severity.ERROR,
                    "Empty REQL source",
                    node=node
                )

    def _validate_step(self, node: Tree) -> None:
        """Validate a pipeline step."""
        if node.data == "python_step":
            # Python blocks require security validation later
            # Check PYTHON_BLOCK token exists and has content
            has_content = False
            for child in node.children:
                if isinstance(child, Token) and child.type == "PYTHON_BLOCK":
                    content = str(child).strip()
                    if content and len(content) > 2:  # More than just {}
                        has_content = True
                    break
            if not has_content:
                self._add_issue(
                    Severity.ERROR,
                    "Empty Python block",
                    node=node
                )

    # ============================================================
    # DUPLICATE CHECKING
    # ============================================================

    def _check_duplicate_names(self) -> None:
        """Check for duplicate tool names across all definitions."""
        names = [t["name"] for t in self.tool_info.get("tools", [])]
        seen = set()
        for name in names:
            if name in seen:
                self._add_issue(
                    Severity.ERROR,
                    f"Duplicate tool name '{name}'",
                    suggestion="Tool names must be unique within a file"
                )
            seen.add(name)

    # ============================================================
    # HELPERS
    # ============================================================

    def _find_child(
        self, node: Tree, data_prefix: str
    ) -> Optional[Tree]:
        """Find first child with matching data prefix."""
        for child in node.children:
            if isinstance(child, Tree):
                if child.data == data_prefix or child.data.startswith(data_prefix):
                    return child
        return None

    def _extract_value(self, node: Tree) -> Any:
        """Extract a value from a value node."""
        if isinstance(node, Token):
            return self._token_to_value(node)

        for child in node.children:
            if isinstance(child, Token):
                return self._token_to_value(child)
            elif isinstance(child, Tree):
                if child.data == "val_string":
                    return self._unquote(str(child.children[0]))
                elif child.data == "val_int":
                    return int(str(child.children[0]))
                elif child.data == "val_float":
                    return float(str(child.children[0]))
                elif child.data == "val_true":
                    return True
                elif child.data == "val_false":
                    return False
                elif child.data == "val_null":
                    return None
                elif child.data == "val_list":
                    return self._extract_list(child)
                elif child.data == "capability_array":
                    return self._extract_list(child)
                else:
                    return self._extract_value(child)

        return None

    def _extract_list(self, node: Tree) -> List[Any]:
        """Extract a list value."""
        items = []
        for child in node.children:
            if isinstance(child, Tree):
                if child.data == "value_list":
                    for item in child.children:
                        items.append(self._extract_value(item))
                elif child.data == "capability_list":
                    for item in child.children:
                        if isinstance(item, Token) and item.type == "STRING":
                            items.append(self._unquote(str(item)))
                else:
                    items.append(self._extract_value(child))
            elif isinstance(child, Token) and child.type == "STRING":
                items.append(self._unquote(str(child)))
        return items

    def _token_to_value(self, token: Token) -> Any:
        """Convert a token to a Python value."""
        if token.type == "STRING":
            return self._unquote(str(token))
        elif token.type == "SIGNED_INT" or token.type == "INT":
            return int(str(token))
        elif token.type == "SIGNED_FLOAT":
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

    def _unquote(self, s: str) -> str:
        """Remove quotes from a string."""
        if len(s) >= 2:
            if (s.startswith('"') and s.endswith('"')) or \
               (s.startswith("'") and s.endswith("'")):
                return s[1:-1]
        return s

    def _add_issue(
        self,
        severity: Severity,
        message: str,
        node: Optional[Tree] = None,
        suggestion: Optional[str] = None
    ) -> None:
        """Add a validation issue."""
        line = None
        column = None
        node_type = None

        if node is not None:
            if hasattr(node, 'meta') and node.meta:
                line = node.meta.line
                column = node.meta.column
            if isinstance(node, Tree):
                node_type = node.data

        self.issues.append(ValidationIssue(
            severity=severity,
            message=message,
            line=line,
            column=column,
            node_type=node_type,
            suggestion=suggestion
        ))


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def validate_cadsl(tree: Tree, strict: bool = False) -> ValidationResult:
    """
    Validate a CADSL parse tree.

    Args:
        tree: Lark parse tree from CADSLParser
        strict: If True, treat warnings as errors

    Returns:
        ValidationResult with issues and tool info
    """
    validator = CADSLValidator(strict=strict)
    return validator.validate(tree)
