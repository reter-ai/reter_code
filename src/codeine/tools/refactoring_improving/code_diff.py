"""
Code Diff Analyzer for Extraction Opportunities

DEPRECATED: This module's tokenization approach is deprecated.
Use RETER-based analysis in tool.py instead (via _analyze_methods_with_reter).

The dataclasses (ParameterSuggestion, ExtractionAnalysis) have been moved to tool.py.
The tokenization functions remain for backwards compatibility but should not be used.

Old approach problems:
- Python's tokenize module fails on indented code snippets
- Requires file I/O to read source code
- Fragile and slow

New approach (RETER-based):
- Queries RETER for already-parsed method properties
- No re-tokenization needed
- Works on any indentation level
- Faster and more robust

Analyzes two similar code blocks to determine:
1. Whether they can be merged into a common function
2. What parameters would be needed
3. The common code pattern
"""

import tokenize
import io
import re
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from difflib import SequenceMatcher


@dataclass
class TokenInfo:
    """Simplified token representation."""
    type: str  # NAME, NUMBER, STRING, OP, KEYWORD, OTHER
    value: str
    line: int
    col: int

    def __eq__(self, other):
        if not isinstance(other, TokenInfo):
            return False
        return self.type == other.type and self.value == other.value


@dataclass
class ParameterSuggestion:
    """A suggested parameter for the extracted function."""
    name: str
    inferred_type: str
    values: List[str]  # The different values found in the code blocks
    diff_type: str  # "name", "literal", "string"


@dataclass
class ExtractionAnalysis:
    """Result of analyzing two code blocks for extraction."""
    extractable: bool
    reason: str
    similarity_score: float
    parameters: List[ParameterSuggestion]
    common_pattern: str  # Template with placeholders
    suggested_name: str
    diff_count: int
    total_tokens: int


# Python keywords that should not be treated as variable names
PYTHON_KEYWORDS = {
    'False', 'None', 'True', 'and', 'as', 'assert', 'async', 'await',
    'break', 'class', 'continue', 'def', 'del', 'elif', 'else', 'except',
    'finally', 'for', 'from', 'global', 'if', 'import', 'in', 'is',
    'lambda', 'nonlocal', 'not', 'or', 'pass', 'raise', 'return', 'try',
    'while', 'with', 'yield'
}

# Common type hints to infer from variable names
TYPE_HINTS = {
    'id': 'int', 'idx': 'int', 'index': 'int', 'count': 'int', 'num': 'int',
    'size': 'int', 'length': 'int', 'len': 'int', 'n': 'int', 'i': 'int',
    'name': 'str', 'text': 'str', 'msg': 'str', 'message': 'str', 's': 'str',
    'path': 'str', 'file': 'str', 'url': 'str', 'key': 'str', 'value': 'str',
    'flag': 'bool', 'is_': 'bool', 'has_': 'bool', 'can_': 'bool',
    'enabled': 'bool', 'active': 'bool', 'valid': 'bool',
    'items': 'List', 'data': 'Dict', 'result': 'Any', 'obj': 'Any',
    'timeout': 'float', 'delay': 'float', 'rate': 'float',
}


def tokenize_code(code: str) -> List[TokenInfo]:
    """
    Tokenize Python code into simplified tokens.

    Args:
        code: Python source code string

    Returns:
        List of TokenInfo objects
    """
    tokens = []
    try:
        # Handle potential encoding issues
        code_bytes = code.encode('utf-8')
        readline = io.BytesIO(code_bytes).readline

        for tok in tokenize.tokenize(readline):
            tok_type = tokenize.tok_name[tok.type]
            value = tok.string

            # Skip encoding, newlines, endmarker, comments
            if tok_type in ('ENCODING', 'NEWLINE', 'ENDMARKER', 'NL', 'COMMENT', 'INDENT', 'DEDENT'):
                continue

            # Classify token type
            if tok_type == 'NAME':
                if value in PYTHON_KEYWORDS:
                    simplified_type = 'KEYWORD'
                else:
                    simplified_type = 'NAME'
            elif tok_type == 'NUMBER':
                simplified_type = 'NUMBER'
            elif tok_type == 'STRING':
                simplified_type = 'STRING'
            elif tok_type == 'OP':
                simplified_type = 'OP'
            else:
                simplified_type = 'OTHER'

            tokens.append(TokenInfo(
                type=simplified_type,
                value=value,
                line=tok.start[0],
                col=tok.start[1]
            ))
    except tokenize.TokenError:
        # If tokenization fails, return empty list
        pass

    return tokens


def align_tokens(tokens1: List[TokenInfo], tokens2: List[TokenInfo]) -> List[Tuple[Optional[TokenInfo], Optional[TokenInfo]]]:
    """
    Align two token sequences for comparison.

    Uses a hybrid approach:
    1. For tokens of the same type at the same position, pair them (even if values differ)
    2. For structural differences, use LCS to find best alignment

    Returns list of (token1, token2) pairs where None indicates a gap.
    """
    # Create structural keys that ignore specific values but preserve token types
    def structural_key(t: TokenInfo) -> str:
        # For variable names/numbers/strings, use just the type
        # This allows pairing `user` with `client` (both NAME tokens)
        if t.type in ('NAME', 'NUMBER', 'STRING'):
            return t.type
        # For keywords and operators, use the actual value (structure must match)
        return f"{t.type}:{t.value}"

    keys1 = [structural_key(t) for t in tokens1]
    keys2 = [structural_key(t) for t in tokens2]

    matcher = SequenceMatcher(None, keys1, keys2)

    aligned = []
    i, j = 0, 0

    for block in matcher.get_matching_blocks():
        # Handle non-matching region before this block
        # Try to pair up tokens positionally if they're the same type
        while i < block.a and j < block.b:
            t1, t2 = tokens1[i], tokens2[j]
            if t1.type == t2.type:
                # Same type, pair them (even if values differ)
                aligned.append((t1, t2))
                i += 1
                j += 1
            else:
                # Different types - this is a structural diff
                # Add both as gaps
                aligned.append((t1, None))
                aligned.append((None, t2))
                i += 1
                j += 1

        # Handle remaining unmatched tokens on either side
        while i < block.a:
            aligned.append((tokens1[i], None))
            i += 1
        while j < block.b:
            aligned.append((None, tokens2[j]))
            j += 1

        # Add matching tokens (structurally equivalent)
        for k in range(block.size):
            if i + k < len(tokens1) and j + k < len(tokens2):
                aligned.append((tokens1[i + k], tokens2[j + k]))

        i = block.a + block.size
        j = block.b + block.size

    return aligned


def infer_type_from_name(name: str) -> str:
    """Infer a type hint from a variable name."""
    name_lower = name.lower()

    # Check exact matches
    if name_lower in TYPE_HINTS:
        return TYPE_HINTS[name_lower]

    # Check prefixes
    for prefix, hint in TYPE_HINTS.items():
        if prefix.endswith('_') and name_lower.startswith(prefix):
            return hint

    # Check suffixes
    if name_lower.endswith('_id') or name_lower.endswith('id'):
        return 'int'
    if name_lower.endswith('_name') or name_lower.endswith('name'):
        return 'str'
    if name_lower.endswith('_list') or name_lower.endswith('s'):
        return 'List'
    if name_lower.endswith('_dict') or name_lower.endswith('_map'):
        return 'Dict'

    return 'Any'


def suggest_parameter_name(values: List[str]) -> str:
    """Suggest a parameter name based on the differing values."""
    if len(values) == 0:
        return "param"

    # Find common suffix/prefix
    if len(values) >= 2:
        # Try to find what they have in common
        common = []
        for i, chars in enumerate(zip(*values)):
            if len(set(chars)) == 1:
                common.append(chars[0])
            else:
                break

        if common:
            name = ''.join(common).rstrip('_')
            if name and name not in PYTHON_KEYWORDS:
                return name

    # Use the shortest value if it looks like a valid name
    shortest = min(values, key=len)
    if re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', shortest) and shortest not in PYTHON_KEYWORDS:
        return shortest

    return "param"


def suggest_function_name(code: str) -> str:
    """Suggest a function name based on the code content."""
    # Look for common patterns
    patterns = [
        (r'\.save\s*\(', '_save'),
        (r'\.load\s*\(', '_load'),
        (r'\.process\s*\(', '_process'),
        (r'\.validate\s*\(', '_validate'),
        (r'\.handle\s*\(', '_handle'),
        (r'\.create\s*\(', '_create'),
        (r'\.update\s*\(', '_update'),
        (r'\.delete\s*\(', '_delete'),
        (r'\.get\s*\(', '_get'),
        (r'\.set\s*\(', '_set'),
        (r'\.fetch\s*\(', '_fetch'),
        (r'\.send\s*\(', '_send'),
        (r'\.parse\s*\(', '_parse'),
        (r'\.format\s*\(', '_format'),
        (r'\.convert\s*\(', '_convert'),
        (r'\.transform\s*\(', '_transform'),
        (r'\.check\s*\(', '_check'),
        (r'\.init\s*\(', '_init'),
        (r'\.setup\s*\(', '_setup'),
        (r'\.cleanup\s*\(', '_cleanup'),
    ]

    for pattern, name in patterns:
        if re.search(pattern, code):
            return name

    # Default name
    return "_extracted_logic"


def analyze_code_diff(code1: str, code2: str, min_similarity: float = 0.5) -> ExtractionAnalysis:
    """
    Analyze two code blocks to determine if they can be extracted.

    Args:
        code1: First code block
        code2: Second code block
        min_similarity: Minimum token similarity to consider extractable

    Returns:
        ExtractionAnalysis with extraction details
    """
    tokens1 = tokenize_code(code1)
    tokens2 = tokenize_code(code2)

    if not tokens1 or not tokens2:
        return ExtractionAnalysis(
            extractable=False,
            reason="Failed to tokenize one or both code blocks",
            similarity_score=0.0,
            parameters=[],
            common_pattern="",
            suggested_name="_extracted",
            diff_count=0,
            total_tokens=0
        )

    # Align tokens
    aligned = align_tokens(tokens1, tokens2)

    # Analyze differences
    parameters = []
    name_diffs = {}  # Map from position to (value1, value2)
    literal_diffs = {}
    string_diffs = {}
    structural_diffs = 0
    matches = 0

    for i, (t1, t2) in enumerate(aligned):
        if t1 is None or t2 is None:
            # Gap in alignment - structural difference
            structural_diffs += 1
            continue

        if t1.value == t2.value:
            matches += 1
            continue

        # Tokens differ
        if t1.type == 'NAME' and t2.type == 'NAME':
            # Variable names differ - can parameterize
            name_diffs[i] = (t1.value, t2.value)
        elif t1.type == 'NUMBER' and t2.type == 'NUMBER':
            # Numeric literals differ - can parameterize
            literal_diffs[i] = (t1.value, t2.value)
        elif t1.type == 'STRING' and t2.type == 'STRING':
            # String literals differ - can parameterize
            string_diffs[i] = (t1.value, t2.value)
        elif t1.type == t2.type and t1.type in ('OP', 'KEYWORD'):
            # Different operators or keywords - structural diff
            structural_diffs += 1
        else:
            # Type mismatch - structural diff
            structural_diffs += 1

    total_tokens = max(len(tokens1), len(tokens2))
    diff_count = len(name_diffs) + len(literal_diffs) + len(string_diffs) + structural_diffs

    # Calculate similarity
    if total_tokens > 0:
        similarity = matches / total_tokens
    else:
        similarity = 0.0

    # Determine if extractable
    # Too many structural diffs means not extractable
    max_structural_diffs = max(3, total_tokens * 0.1)  # Allow up to 10% structural diffs

    if structural_diffs > max_structural_diffs:
        return ExtractionAnalysis(
            extractable=False,
            reason=f"Too many structural differences ({structural_diffs})",
            similarity_score=similarity,
            parameters=[],
            common_pattern="",
            suggested_name="_extracted",
            diff_count=diff_count,
            total_tokens=total_tokens
        )

    if similarity < min_similarity:
        return ExtractionAnalysis(
            extractable=False,
            reason=f"Similarity too low ({similarity:.2f} < {min_similarity})",
            similarity_score=similarity,
            parameters=[],
            common_pattern="",
            suggested_name="_extracted",
            diff_count=diff_count,
            total_tokens=total_tokens
        )

    # Create parameter suggestions
    param_counter = 0

    # Group name diffs by value pairs
    for i, (v1, v2) in name_diffs.items():
        # Check if we already have this pair
        found = False
        for p in parameters:
            if v1 in p.values and v2 in p.values:
                found = True
                break

        if not found:
            suggested_name = suggest_parameter_name([v1, v2])
            if any(p.name == suggested_name for p in parameters):
                param_counter += 1
                suggested_name = f"{suggested_name}_{param_counter}"

            parameters.append(ParameterSuggestion(
                name=suggested_name,
                inferred_type=infer_type_from_name(suggested_name),
                values=[v1, v2],
                diff_type="name"
            ))

    for i, (v1, v2) in literal_diffs.items():
        param_counter += 1
        # Try to infer meaning from context
        param_name = f"value_{param_counter}"

        # Check if it looks like a timeout, count, etc.
        if '.' in v1 or '.' in v2:
            param_name = f"float_value_{param_counter}"
            inferred_type = "float"
        else:
            inferred_type = "int"

        parameters.append(ParameterSuggestion(
            name=param_name,
            inferred_type=inferred_type,
            values=[v1, v2],
            diff_type="literal"
        ))

    for i, (v1, v2) in string_diffs.items():
        param_counter += 1
        parameters.append(ParameterSuggestion(
            name=f"text_{param_counter}",
            inferred_type="str",
            values=[v1, v2],
            diff_type="string"
        ))

    # Generate common pattern (simplified)
    common_pattern = generate_common_pattern(code1, parameters)
    suggested_name = suggest_function_name(code1)

    return ExtractionAnalysis(
        extractable=True,
        reason="Code blocks can be merged with parameterization",
        similarity_score=similarity,
        parameters=parameters,
        common_pattern=common_pattern,
        suggested_name=suggested_name,
        diff_count=diff_count,
        total_tokens=total_tokens
    )


def generate_common_pattern(code: str, parameters: List[ParameterSuggestion]) -> str:
    """
    Generate a template pattern from code by replacing variable values with parameters.
    """
    pattern = code

    for param in parameters:
        if param.values:
            # Replace the first value with the parameter name
            # This is a simplified approach - a more sophisticated one would use AST
            first_value = param.values[0]
            if param.diff_type == "name":
                # Replace word boundary matches
                pattern = re.sub(
                    rf'\b{re.escape(first_value)}\b',
                    param.name,
                    pattern
                )
            elif param.diff_type == "literal":
                pattern = pattern.replace(first_value, param.name, 1)
            elif param.diff_type == "string":
                pattern = pattern.replace(first_value, param.name, 1)

    return pattern


def get_source_code(file_path: str, start_line: int, end_line: int) -> str:
    """
    Read source code from a file.

    Args:
        file_path: Path to the Python file
        start_line: Starting line number (1-indexed)
        end_line: Ending line number (1-indexed)

    Returns:
        Source code as string
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            # Adjust for 1-indexed line numbers
            start_idx = max(0, start_line - 1)
            end_idx = min(len(lines), end_line)
            return ''.join(lines[start_idx:end_idx])
    except Exception:
        return ""
