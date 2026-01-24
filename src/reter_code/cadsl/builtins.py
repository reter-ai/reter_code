"""
CADSL Builtins - Safe Built-in Functions for CADSL Python Blocks.

This module provides a curated set of safe functions that are available
in CADSL Python blocks. These functions are designed for code analysis
and data transformation tasks.

Categories:
- String operations
- Collection utilities
- Math functions
- Code analysis helpers
- Data transformation
"""

from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Union
import re
import json
from collections import Counter, defaultdict
from functools import reduce


# ============================================================
# STRING OPERATIONS
# ============================================================

def camel_to_snake(name: str) -> str:
    """Convert CamelCase to snake_case."""
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def snake_to_camel(name: str) -> str:
    """Convert snake_case to CamelCase."""
    components = name.split('_')
    return components[0] + ''.join(x.title() for x in components[1:])


def snake_to_pascal(name: str) -> str:
    """Convert snake_case to PascalCase."""
    return ''.join(x.title() for x in name.split('_'))


def extract_prefix(name: str, delimiter: str = "_") -> str:
    """Extract prefix before delimiter."""
    if delimiter in name:
        return name.split(delimiter)[0]
    return name


def extract_suffix(name: str, delimiter: str = "_") -> str:
    """Extract suffix after last delimiter."""
    if delimiter in name:
        return name.rsplit(delimiter, 1)[-1]
    return name


def pluralize(word: str) -> str:
    """Simple English pluralization."""
    if word.endswith('y') and len(word) > 1 and word[-2] not in 'aeiou':
        return word[:-1] + 'ies'
    elif word.endswith(('s', 'x', 'z', 'ch', 'sh')):
        return word + 'es'
    else:
        return word + 's'


def singularize(word: str) -> str:
    """Simple English singularization."""
    if word.endswith('ies') and len(word) > 3:
        return word[:-3] + 'y'
    elif word.endswith('es') and len(word) > 2:
        if word[:-2].endswith(('s', 'x', 'z', 'ch', 'sh')):
            return word[:-2]
    elif word.endswith('s') and len(word) > 1:
        return word[:-1]
    return word


def truncate(text: str, max_length: int, suffix: str = "...") -> str:
    """Truncate text to max length with suffix."""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def indent_text(text: str, spaces: int = 4) -> str:
    """Indent each line of text."""
    prefix = " " * spaces
    return "\n".join(prefix + line for line in text.splitlines())


# ============================================================
# COLLECTION UTILITIES
# ============================================================

def group_by(items: Iterable[Dict], key: str) -> Dict[Any, List[Dict]]:
    """Group items by a key field."""
    groups = defaultdict(list)
    for item in items:
        k = item.get(key) if isinstance(item, dict) else getattr(item, key, None)
        groups[k].append(item)
    return dict(groups)


def unique_by(items: Iterable[Dict], key: str) -> List[Dict]:
    """Remove duplicates based on key field."""
    seen = set()
    result = []
    for item in items:
        k = item.get(key) if isinstance(item, dict) else getattr(item, key, None)
        if k not in seen:
            seen.add(k)
            result.append(item)
    return result


def flatten(nested: Iterable[Iterable]) -> List:
    """Flatten a nested list one level."""
    result = []
    for item in nested:
        if isinstance(item, (list, tuple)):
            result.extend(item)
        else:
            result.append(item)
    return result


def deep_flatten(nested: Iterable) -> List:
    """Recursively flatten all nested lists."""
    result = []
    for item in nested:
        if isinstance(item, (list, tuple)):
            result.extend(deep_flatten(item))
        else:
            result.append(item)
    return result


def chunk(items: Sequence, size: int) -> List[List]:
    """Split items into chunks of specified size."""
    return [list(items[i:i + size]) for i in range(0, len(items), size)]


def partition(items: Iterable, predicate: Callable[[Any], bool]) -> tuple:
    """Partition items into two lists based on predicate."""
    true_items = []
    false_items = []
    for item in items:
        if predicate(item):
            true_items.append(item)
        else:
            false_items.append(item)
    return true_items, false_items


def pluck(items: Iterable[Dict], key: str) -> List:
    """Extract a single field from each item."""
    return [item.get(key) if isinstance(item, dict) else getattr(item, key, None)
            for item in items]


def index_by(items: Iterable[Dict], key: str) -> Dict[Any, Dict]:
    """Create a lookup dict indexed by key field."""
    return {item.get(key) if isinstance(item, dict) else getattr(item, key, None): item
            for item in items}


def frequencies(items: Iterable) -> Dict[Any, int]:
    """Count occurrences of each item."""
    return dict(Counter(items))


def top_n(items: Iterable[Dict], n: int, key: str, reverse: bool = True) -> List[Dict]:
    """Get top N items by a numeric key."""
    return sorted(items, key=lambda x: x.get(key, 0), reverse=reverse)[:n]


# ============================================================
# MATH FUNCTIONS
# ============================================================

def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    """Safe division that returns default on zero denominator."""
    return a / b if b != 0 else default


def percentage(part: float, total: float, decimals: int = 1) -> float:
    """Calculate percentage."""
    if total == 0:
        return 0.0
    return round((part / total) * 100, decimals)


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp value between min and max."""
    return max(min_val, min(max_val, value))


def average(values: Iterable[float]) -> float:
    """Calculate average of values."""
    items = list(values)
    return sum(items) / len(items) if items else 0.0


def median(values: Iterable[float]) -> float:
    """Calculate median of values."""
    items = sorted(values)
    n = len(items)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2 == 0:
        return (items[mid - 1] + items[mid]) / 2
    return items[mid]


def variance(values: Iterable[float]) -> float:
    """Calculate variance of values."""
    items = list(values)
    if len(items) < 2:
        return 0.0
    avg = average(items)
    return sum((x - avg) ** 2 for x in items) / len(items)


def std_dev(values: Iterable[float]) -> float:
    """Calculate standard deviation of values."""
    return variance(values) ** 0.5


# ============================================================
# CODE ANALYSIS HELPERS
# ============================================================

def is_private(name: str) -> bool:
    """Check if name is private (starts with _)."""
    return name.startswith('_') and not name.startswith('__')


def is_dunder(name: str) -> bool:
    """Check if name is a dunder method (__name__)."""
    return name.startswith('__') and name.endswith('__')


def is_test_name(name: str) -> bool:
    """Check if name indicates a test."""
    name_lower = name.lower()
    return (name_lower.startswith('test') or
            name_lower.endswith('test') or
            name_lower.endswith('tests') or
            '_test_' in name_lower)


def is_constant(name: str) -> bool:
    """Check if name follows constant naming (ALL_CAPS)."""
    return name.isupper() and '_' in name or name.isupper()


def extract_class_name(qualified_name: str) -> str:
    """Extract class name from qualified name."""
    if '.' in qualified_name:
        return qualified_name.rsplit('.', 1)[-1]
    return qualified_name


def extract_module_path(file_path: str) -> str:
    """Convert file path to module path."""
    # Remove .py extension and convert slashes to dots
    path = file_path.replace('\\', '/').replace('/', '.')
    if path.endswith('.py'):
        path = path[:-3]
    # Remove leading dots
    return path.lstrip('.')


def count_complexity(value: int, thresholds: Dict[str, int] = None) -> str:
    """Categorize complexity value into low/medium/high/critical."""
    thresholds = thresholds or {"low": 5, "medium": 10, "high": 20}
    if value <= thresholds.get("low", 5):
        return "low"
    elif value <= thresholds.get("medium", 10):
        return "medium"
    elif value <= thresholds.get("high", 20):
        return "high"
    return "critical"


# ============================================================
# DATA TRANSFORMATION
# ============================================================

def pick(d: Dict, keys: List[str]) -> Dict:
    """Pick only specified keys from dict."""
    return {k: v for k, v in d.items() if k in keys}


def omit(d: Dict, keys: List[str]) -> Dict:
    """Omit specified keys from dict."""
    return {k: v for k, v in d.items() if k not in keys}


def rename_keys(d: Dict, mapping: Dict[str, str]) -> Dict:
    """Rename keys in dict according to mapping."""
    return {mapping.get(k, k): v for k, v in d.items()}


def merge_dicts(*dicts: Dict) -> Dict:
    """Merge multiple dicts, later ones override earlier."""
    result = {}
    for d in dicts:
        result.update(d)
    return result


def deep_get(d: Dict, path: str, default: Any = None) -> Any:
    """Get nested value by dot-separated path."""
    keys = path.split('.')
    result = d
    for key in keys:
        if isinstance(result, dict):
            result = result.get(key)
        else:
            return default
        if result is None:
            return default
    return result


def transform_values(d: Dict, fn: Callable[[Any], Any]) -> Dict:
    """Apply function to all values in dict."""
    return {k: fn(v) for k, v in d.items()}


def filter_dict(d: Dict, predicate: Callable[[str, Any], bool]) -> Dict:
    """Filter dict by key-value predicate."""
    return {k: v for k, v in d.items() if predicate(k, v)}


# ============================================================
# FORMATTING
# ============================================================

def format_count(n: int, singular: str, plural: str = None) -> str:
    """Format count with singular/plural noun."""
    plural = plural or pluralize(singular)
    return f"{n} {singular if n == 1 else plural}"


def format_list(items: List[str], conjunction: str = "and",
                oxford_comma: bool = True) -> str:
    """Format list as English sentence."""
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} {conjunction} {items[1]}"
    comma = "," if oxford_comma else ""
    return f"{', '.join(items[:-1])}{comma} {conjunction} {items[-1]}"


def to_json(obj: Any, indent: int = 2) -> str:
    """Convert object to formatted JSON string."""
    return json.dumps(obj, indent=indent, default=str)


def from_json(s: str) -> Any:
    """Parse JSON string."""
    return json.loads(s)


# ============================================================
# CADSL BUILTINS REGISTRY
# ============================================================

CADSL_BUILTINS: Dict[str, Callable] = {
    # String operations
    "camel_to_snake": camel_to_snake,
    "snake_to_camel": snake_to_camel,
    "snake_to_pascal": snake_to_pascal,
    "extract_prefix": extract_prefix,
    "extract_suffix": extract_suffix,
    "pluralize": pluralize,
    "singularize": singularize,
    "truncate": truncate,
    "indent_text": indent_text,

    # Collection utilities
    "group_by": group_by,
    "unique_by": unique_by,
    "flatten": flatten,
    "deep_flatten": deep_flatten,
    "chunk": chunk,
    "partition": partition,
    "pluck": pluck,
    "index_by": index_by,
    "frequencies": frequencies,
    "top_n": top_n,

    # Math functions
    "safe_divide": safe_divide,
    "percentage": percentage,
    "clamp": clamp,
    "average": average,
    "median": median,
    "variance": variance,
    "std_dev": std_dev,

    # Code analysis helpers
    "is_private": is_private,
    "is_dunder": is_dunder,
    "is_test_name": is_test_name,
    "is_constant": is_constant,
    "extract_class_name": extract_class_name,
    "extract_module_path": extract_module_path,
    "count_complexity": count_complexity,

    # Data transformation
    "pick": pick,
    "omit": omit,
    "rename_keys": rename_keys,
    "merge_dicts": merge_dicts,
    "deep_get": deep_get,
    "transform_values": transform_values,
    "filter_dict": filter_dict,

    # Formatting
    "format_count": format_count,
    "format_list": format_list,
    "to_json": to_json,
    "from_json": from_json,
}


def get_cadsl_builtins() -> Dict[str, Callable]:
    """Get all CADSL built-in functions."""
    return CADSL_BUILTINS.copy()
