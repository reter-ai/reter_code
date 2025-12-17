# Python Analysis Tools - Complete Reference

**74 High-Level Tools for Python Code Analysis**

This document provides a complete reference for all Python analysis tools available in the logical-thinking MCP server. These tools make it easy to analyze Python codebases without writing REQL queries directly.

---

## Quick Links

- =� **[Python Query Patterns](PYTHON_QUERY_PATTERNS.md)** - Learn REQL for Python
- =� **[Python Analysis Reference](python/PYTHON_ANALYSIS_REFERENCE.md)** - Understand extracted facts
- >� **[Test Results](../ADVANCED_PYTHON_TOOLS_TESTS_COMPLETE.md)** - Verification & examples

---
---

## 🆕 What's New (2025-11-28)

**Exception Handling Analysis** - New detection capabilities:

### Exception Handling Detectors (6 new tools)
- ✅ **Silent exception swallowing** - Find `except: pass` anti-patterns
- ✅ **Too general exceptions** - Detect `except Exception:` and bare `except:`
- ✅ **General exception raising** - Find `raise Exception(...)` instead of specific types
- ✅ **Error codes over exceptions** - Detect returning `-1`, `None`, `False` for errors
- ✅ **Missing context managers** - Find RAII cleanup in `finally` that should use `with`
- ✅ **Comprehensive analysis** - Run all exception detectors at once

### New Semantic Facts Extracted
- ✅ **py:TryBlock** - try statements with hasFinally, hasElse flags
- ✅ **py:ExceptHandler** - except clauses with isSilentSwallow, isGeneralExcept detection
- ✅ **py:FinallyBlock** - finally blocks with isRAIICleanup detection
- ✅ **py:RaiseStatement** - raise statements with isGeneralException flag
- ✅ **py:ReturnStatement** - return statements with looksLikeErrorCode heuristic
- ✅ **py:WithStatement** - context manager usage with isAsync flag
- ✅ **py:ContextManager** - individual with items with contextType detection

**Query Example**:
```python
# Find all silent exception swallowing
result = reql("""
    SELECT ?handler ?function ?line
    WHERE {
        ?handler concept "py:ExceptHandler" .
        ?handler isSilentSwallow "true" .
        OPTIONAL { ?handler inFunction ?function } .
        OPTIONAL { ?handler atLine ?line }
    }
""")
```

See [Python Query Patterns](PYTHON_QUERY_PATTERNS.md#exception-handling-patterns) for more examples.

---

## 🆕 What's New (2025-11-19)

**Enhanced Python Analysis** - New extraction capabilities:

### Class Attribute Detection
- ✅ **All instance attributes extracted** from `self.x = value` assignments
- ✅ **Type inference** from constructor calls (e.g., `self.manager = PluginManager()`)
- ✅ **Visibility classification**: public, protected (`_prefix`), private (`__prefix`)
- ✅ **Works with or without type annotations**

### Type-Resolved Method Calls
- ✅ **Method calls fully resolved** using type tracking (e.g., `self.manager.load()` → `PluginManager.load`)
- ✅ **Cross-method type persistence** (types from `__init__` visible in all methods)
- ✅ **Out-of-order execution support** (calls resolved even if they appear before type assignments)
- ✅ **Enables accurate sequence diagrams** and call graph analysis

**Query Example**:
```python
# Find all public attributes with their types
result = reql("""
    SELECT ?class ?attr ?type ?visibility
    WHERE {
        ?attr concept "py:Attribute" .
        ?attr ofClass ?class .
        ?attr hasType ?type .
        ?attr visibility ?visibility
    }
""")
```

See [Python Query Patterns](PYTHON_QUERY_PATTERNS.md#new-features-2025-11-19) for more examples.


## Overview

The logical-thinking server provides **74 high-level tools** for Python code analysis, divided into two categories:

### Basic Python Tools (10 tools)
General-purpose tools for exploring code structure, finding relationships, and understanding dependencies.

### Advanced Python Tools (64 tools)
Specialized tools for code quality, refactoring opportunities, testing analysis, and architecture exploration including:
- **7 code smell detectors** (long functions, large classes, data classes, feature envy, refused bequest, etc.)
- **30+ refactoring opportunity identifiers** (extract function/class, inline, move features, encapsulation, etc.)
- **Dependency analysis** (import graphs, circular imports, external dependencies, change impact)
- **Quality metrics** (complexity, documentation coverage, type hints)
- **Test analysis** (test files, fixtures)
- **Architecture exploration** (package structure, public API, magic methods)

**All tools return**:
- Structured results (lists, counts, statistics)
- `queries` field showing REQL queries executed
- `success` field indicating if operation succeeded
- `error` field (if operation failed)

---

## Table of Contents

1. [Basic Python Tools](#basic-python-tools-10-tools)
2. [Advanced Python Tools](#advanced-python-tools-64-tools)
3. [Query Tracking](#query-tracking)
4. [Usage Examples](#usage-examples)
5. [Tool Selection Guide](#tool-selection-guide)

**Note**: This reference documents all 74 Python analysis tools. The Advanced Python Tools section contains 64 tools categorized by function (code smells, refactoring patterns, dependencies, quality, testing, architecture).

---

## Basic Python Tools (10 Tools)

### 1. py_list_modules
**List all Python modules in the codebase**

```python
result = await server.py_list_modules(instance_name)
```

**Returns**:
- `modules`: List of `{qualified_name, name, file}`
- `count`: Number of modules
- `queries`: REQL queries executed

**Use Case**: Get overview of codebase structure

---

### 2. py_list_classes
**List classes in codebase or specific module**

```python
result = await server.py_list_classes(
    instance_name,
    module_name=None  # Optional filter
)
```

**Returns**:
- `classes`: List of `{qualified_name, name, full_qualified_name, line}`
- `count`: Number of classes
- `module_filter`: Filter applied (if any)
- `queries`: REQL queries executed

**Use Case**: Browse classes, filter by module

---

### 3. py_describe_class
**Get detailed class information including methods**

```python
result = await server.py_describe_class(
    instance_name,
    class_name  # Simple or qualified name
)
```

**Returns**:
- `class_info`: `{qualified_name, name, module, line, docstring}`
- `methods`: List with signatures and parameters
- `method_count`: Number of methods
- `queries`: REQL queries executed (3+)

**Use Case**: Deep dive into class structure

---

### 4. py_find_usages
**Find where a class or method is called**

```python
result = await server.py_find_usages(
    instance_name,
    target_name  # Name to find usages of
)
```

**Returns**:
- `usages`: List of `{caller, caller_name, target}`
- `count`: Number of usages
- `queries`: REQL queries executed

**Use Case**: Impact analysis, refactoring

---

### 5. py_find_subclasses
**Find all subclasses of a class**

```python
result = await server.py_find_subclasses(
    instance_name,
    class_name
)
```

**Returns**:
- `subclasses`: List of `{qualified_name, name, parent}`
- `count`: Number of subclasses
- `queries`: REQL queries executed

**Use Case**: Understand inheritance hierarchy

---

### 6. py_get_method_signature
**Get method signature with parameters**

```python
result = await server.py_get_method_signature(
    instance_name,
    method_name
)
```

**Returns**:
- `methods`: List with `{qualified_name, name, return_type, parameters}`
- `count`: Number of matching methods
- `queries`: REQL queries executed (2+ per method)

**Use Case**: Understand method interfaces

---

### 7. py_get_docstring
**Get docstrings for classes or methods**

```python
result = await server.py_get_docstring(
    instance_name,
    name  # Name to search for
)
```

**Returns**:
- `entities`: List of `{qualified_name, name, type, docstring}`
- `count`: Number found
- `queries`: REQL queries executed

**Use Case**: Extract documentation

---

### 8. py_list_functions
**List top-level functions**

```python
result = await server.py_list_functions(
    instance_name,
    module_name=None  # Optional filter
)
```

**Returns**:
- `functions`: List with signatures
- `count`: Number of functions
- `module_filter`: Filter applied (if any)
- `queries`: REQL queries executed

**Use Case**: Find module-level functions

---

### 9. py_get_class_hierarchy
**Get parent and child classes**

```python
result = await server.py_get_class_hierarchy(
    instance_name,
    class_name
)
```

**Returns**:
- `class_name`: Target class
- `qualified_name`: Full name
- `parents`: List of parent classes
- `children`: List of `{qualified_name, name}` for children
- `parent_count`, `child_count`: Counts
- `queries`: REQL queries executed (3)

**Use Case**: Visualize inheritance tree

---

### 10. py_analyze_dependencies
**Analyze codebase dependency graph**

```python
result = await server.py_analyze_dependencies(instance_name)
```

**Returns**:
- `statistics`: `{modules, classes, functions, methods, inheritance_relationships}`
- `call_graph`: List of `{from, to}` call relationships
- `inheritance_graph`: List of `{child, parent}` relationships
- `queries`: REQL queries executed (6)

**Use Case**: Understand overall architecture

---

## Advanced Python Tools (20 Tools)

### Code Quality Metrics (2 tools)

#### 11. py_find_large_classes
**Find "God classes" with too many methods**

```python
result = await server.py_find_large_classes(
    instance_name,
    threshold=20  # Min methods
)
```

**Returns**: `{success, classes, count, threshold, queries, time_ms}`

**Use Case**: Detect SRP violations

---

#### 12. py_find_long_parameter_lists
**Find functions with too many parameters**

```python
result = await server.py_find_long_parameter_lists(
    instance_name,
    threshold=5  # Max params
)
```

**Returns**: `{success, functions, count, threshold, queries, time_ms}`

**Use Case**: Code smell detection

---

### Dependency Analysis (3 tools)

#### 13. py_get_import_graph
**Get module import dependency graph**

```python
result = await server.py_get_import_graph(instance_name)
```

**Returns**: `{success, edges [{from, to}], edge_count, module_count, queries, time_ms}`

**Use Case**: Visualize module dependencies

---

#### 14. py_find_circular_imports P
**Find circular import dependencies using DFS-based cycle detection**

```python
result = await server.py_find_circular_imports(instance_name)
```

**Returns**: `{success, cycles [{modules, type}], count, self_references, direct_cycles, transitive_cycles, queries, time_ms}`

**Detects**:
- Self-references (A→A)
- Direct cycles (A↔B)
- Transitive cycles (A→B→C→A, etc.)

**Use Case**: Identify problematic import cycles

---

#### 15. py_get_external_dependencies
**List external PyPI packages**

```python
result = await server.py_get_external_dependencies(instance_name)
```

**Returns**: `{success, external_modules, count, queries, time_ms}`

**Use Case**: Generate requirements.txt

---

### Pattern Detection (3 tools)

#### 16. py_find_decorators_usage
**Find decorator usages**

```python
result = await server.py_find_decorators_usage(
    instance_name,
    decorator_name=None  # Optional: "@property", "staticmethod"
)
```

**Returns**: `{success, entities, count, decorator_filter, queries, time_ms}`

**Use Case**: Find all `@property`, `@staticmethod`, `@dataclass`

---

#### 17. py_get_magic_methods
**Find magic methods (__init__, __str__, etc.)**

```python
result = await server.py_get_magic_methods(instance_name)
```

**Returns**: `{success, methods [{qualified_name, name, class}], count, queries, time_ms}`

**Use Case**: Analyze special method implementations

---

#### 18. py_get_interface_implementations
**Find ABC implementations**

```python
result = await server.py_get_interface_implementations(
    instance_name,
    interface_name=None  # Optional filter
)
```

**Returns**: `{success, implementations, count, interface_filter, queries, time_ms}`

**Use Case**: Track interface usage

---

### API Analysis (3 tools)

#### 19. py_get_public_api
**Get public classes and functions**

```python
result = await server.py_get_public_api(instance_name)
```

**Returns**: `{success, entities, count, queries, time_ms}`

**Filters**: Excludes names starting with `_`

**Use Case**: Generate API documentation

---

#### 20. py_get_type_hints
**Extract all type annotations**

```python
result = await server.py_get_type_hints(instance_name)
```

**Returns**: `{success, parameter_hints, return_hints, total_hints, queries, time_ms}`

**Executes**: 2 queries (parameters + returns)

**Use Case**: Type coverage analysis

---

#### 21. py_find_untyped_functions
**Find functions without return type hints**

```python
result = await server.py_find_untyped_functions(instance_name)
```

**Returns**: `{success, functions, count, queries, time_ms}`

**Use Case**: Improve type safety

---

### Test Coverage Analysis (2 tools)

#### 22. py_find_test_files
**Find test files by naming convention**

```python
result = await server.py_find_test_files(instance_name)
```

**Returns**: `{success, test_files [{module, file}], count, queries, time_ms}`

**Patterns**: `test_*.py`, `*_test.py`

**Use Case**: Test file discovery

---

#### 23. py_find_test_fixtures
**Find pytest fixtures**

```python
result = await server.py_find_test_fixtures(instance_name)
```

**Returns**: `{success, fixtures, count, queries, time_ms}`

**Decorator**: `@pytest.fixture`

**Use Case**: Test infrastructure analysis

---

### Change Impact Analysis (3 tools) P

#### 24. py_predict_change_impact
**Predict refactoring impact**

```python
result = await server.py_predict_change_impact(
    instance_name,
    entity_name  # Qualified name
)
```

**Returns**: `{success, target_entity, impacted_entities, count, queries, time_ms}`

**Uses**: `callsTransitive` (SWRL-inferred)

**Use Case**: Safe refactoring

---

#### 25. py_find_callers_recursive
**Find all callers (transitive)**

```python
result = await server.py_find_callers_recursive(
    instance_name,
    target_name  # Qualified name
)
```

**Returns**: `{success, target, callers, count, queries, time_ms}`

**Uses**: `callsTransitive` (SWRL-inferred)

**Use Case**: Understand function dependencies

---

#### 26. py_find_callees_recursive
**Find all callees (transitive)**

```python
result = await server.py_find_callees_recursive(
    instance_name,
    source_name  # Qualified name
)
```

**Returns**: `{success, source, callees, count, queries, time_ms}`

**Use Case**: Understand what a function depends on

---

### Documentation Analysis (2 tools)

#### 27. py_find_undocumented_code P
**Find undocumented classes and functions**

```python
result = await server.py_find_undocumented_code(instance_name)
```

**Returns**: `{success, entities, count, queries, time_ms}`

**Uses**: SWRL-inferred `undocumented` predicate

**Use Case**: Documentation coverage

---

#### 28. py_get_api_documentation
**Extract all docstrings**

```python
result = await server.py_get_api_documentation(instance_name)
```

**Returns**: `{success, entities [{name, type, docstring}], count, queries, time_ms}`

**Use Case**: Generate API docs

---

### Architecture Metrics (2 tools)

#### 29. py_get_exception_hierarchy
**Get custom exception class hierarchy**

```python
result = await server.py_get_exception_hierarchy(instance_name)
```

**Returns**: `{success, exceptions [{qualified_name, name, parent}], count, queries, time_ms}`

**Filter**: Classes inheriting from `Exception`

**Use Case**: Error handling architecture

---

#### 30. py_get_package_structure
**Get package/module organization**

```python
result = await server.py_get_package_structure(instance_name)
```

**Returns**: `{success, modules, by_directory, module_count, directory_count, queries, time_ms}`

**Use Case**: Understand codebase structure

---

### Exception Handling Analysis (6 tools)

#### 31. detect_silent_exception_swallowing
**Find except blocks that silently swallow exceptions**

```python
result = await server.refactoring_improving_detector(
    detector_name="detect_silent_exception_swallowing"
)
```

**Detects**:
- Empty except blocks
- `except: pass` patterns
- `except Exception: pass`

**Severity**: CRITICAL

**Returns**: `{success, findings [{handler_id, function, module, exception_type, line, issue, recommendation}], count}`

**Use Case**: Find hidden bugs where exceptions are silently ignored

---

#### 32. detect_too_general_exceptions
**Find except blocks catching overly broad exceptions**

```python
result = await server.refactoring_improving_detector(
    detector_name="detect_too_general_exceptions"
)
```

**Detects**:
- `except Exception:`
- `except BaseException:`
- Bare `except:` (no type)

**Severity**: HIGH

**Returns**: `{success, findings [{handler_id, function, module, exception_type, line, is_bare_except, issue, recommendation}], count}`

**Use Case**: Prevent catching unexpected errors like `KeyboardInterrupt`

---

#### 33. detect_general_exception_raising
**Find raise statements using generic exceptions**

```python
result = await server.refactoring_improving_detector(
    detector_name="detect_general_exception_raising"
)
```

**Detects**:
- `raise Exception(...)`
- `raise BaseException(...)`

**Severity**: MEDIUM

**Returns**: `{success, findings [{raise_id, function, module, exception_type, line, issue, recommendation}], count}`

**Use Case**: Encourage custom exception classes for better error handling

---

#### 34. detect_error_codes_over_exceptions
**Find functions returning error codes instead of raising**

```python
result = await server.refactoring_improving_detector(
    detector_name="detect_error_codes_over_exceptions"
)
```

**Detects**:
- `return -1`
- `return None` (in error contexts)
- `return False` (in error contexts)
- `return {"error": ...}`

**Severity**: MEDIUM

**Returns**: `{success, findings [{return_id, function, module, return_value, line, issue, recommendation}], count}`

**Use Case**: Pythonic error handling prefers exceptions over return codes

---

#### 35. detect_finally_without_context_manager
**Find RAII cleanup in finally blocks that should use 'with'**

```python
result = await server.refactoring_improving_detector(
    detector_name="detect_finally_without_context_manager"
)
```

**Detects**:
- `finally: resource.close()`
- `finally: lock.release()`
- `finally: mutex.unlock()`

**Severity**: MEDIUM

**Returns**: `{success, findings [{finally_id, function, module, line, cleanup_operations, issue, recommendation}], count}`

**Use Case**: Suggest using `with` statement for automatic resource management

---

#### 36. analyze_exception_handling
**Comprehensive exception handling analysis (runs all detectors)**

```python
result = await server.refactoring_improving_detector(
    detector_name="analyze_exception_handling"
)
```

**Returns**:
```python
{
    "success": True,
    "total_issues": 42,
    "by_severity": {"critical": 5, "high": 15, "medium": 22},
    "categories": {
        "silent_swallowing": {...},
        "too_general_catching": {...},
        "too_general_raising": {...},
        "error_codes": {...},
        "missing_context_managers": {...}
    }
}
```

**Use Case**: Complete exception handling audit in one call

---

## Query Tracking

**All tools return the REQL queries they executed** in the `queries` field:

```python
result = await server.py_get_type_hints("main")

# Result includes:
{
    "success": True,
    "total_hints": 241,
    "queries": [
        "SELECT ?param ?param_name ?func ?type\nWHERE {...}",
        "SELECT ?func ?name ?return_type\nWHERE {...}"
    ]
}
```

**Benefits**:
- = **Transparency** - See what queries run
- =� **Learning** - Learn REQL by example
- = **Debugging** - Understand unexpected results
- { **Reuse** - Copy and modify for custom analysis

---

## Usage Examples

### Example 1: Code Quality Audit

```python
# Find code smells
large_classes = await server.py_find_large_classes("main", threshold=10)
long_params = await server.py_find_long_parameter_lists("main", threshold=4)
untyped = await server.py_find_untyped_functions("main")

print(f"Found {large_classes['count']} god classes")
print(f"Found {long_params['count']} functions with too many parameters")
print(f"Found {untyped['count']} untyped functions")
```

### Example 2: Refactoring Planning

```python
# Analyze impact before refactoring
impact = await server.py_predict_change_impact(
    "main",
    "mymodule.MyClass.critical_method"
)

print(f"Changing this method will affect {impact['count']} entities:")
for entity in impact['impacted_entities'][:10]:
    print(f"  - {entity['name']} ({entity['type']})")
```

### Example 3: Documentation Coverage

```python
# Check documentation status
undocumented = await server.py_find_undocumented_code("main")
documented = await server.py_get_api_documentation("main")

total = undocumented['count'] + documented['count']
coverage = (documented['count'] / total * 100) if total > 0 else 0

print(f"Documentation coverage: {coverage:.1f}%")
print(f"Need to document {undocumented['count']} entities")
```

### Example 4: Dependency Analysis

```python
# Understand module structure
imports = await server.py_get_import_graph("main")
circular = await server.py_find_circular_imports("main")
external = await server.py_get_external_dependencies("main")

print(f"Total import relationships: {imports['edge_count']}")
print(f"Modules: {imports['module_count']}")
print(f"Circular dependencies: {circular['count']}")
print(f"External packages: {', '.join(external['external_modules'][:5])}")
```

### Example 5: Analyzing Circular Dependencies

```python
# Find all circular imports with categorization
result = await server.py_find_circular_imports("main")

print(f"Total cycles: {result['count']}")
print(f"  Self-references: {result['self_references']}")
print(f"  Direct cycles (A↔B): {result['direct_cycles']}")
print(f"  Transitive cycles (A→B→C→A): {result['transitive_cycles']}")

# Examine specific cycles
for cycle in result['cycles'][:5]:
    modules = " → ".join(cycle['modules'])
    print(f"  [{cycle['type']}] {modules}")

# The underlying REQL query fetches the import graph:
print("REQL query used:")
print(result['queries'][0])
```

---

## Tool Selection Guide

### When to use Basic Tools:
-  Exploring unfamiliar codebase
-  Finding specific classes/functions
-  Understanding relationships
-  General code navigation

### When to use Advanced Tools:
-  Code quality assessment
-  Refactoring planning
-  Architecture analysis
-  Test coverage tracking
-  Documentation audits

### Quick Decision Matrix:

| Task | Tool(s) |
|------|---------|
| Browse code structure | `py_list_modules`, `py_list_classes`, `py_list_functions` |
| Understand a class | `py_describe_class`, `py_get_class_hierarchy` |
| Find usages before refactoring | `py_find_usages`, `py_predict_change_impact` |
| Check code quality | `py_find_large_classes`, `py_find_long_parameter_lists` |
| Analyze dependencies | `py_get_import_graph`, `py_find_circular_imports` |
| Track tests | `py_find_test_files`, `py_find_test_fixtures` |
| Generate documentation | `py_get_public_api`, `py_get_api_documentation` |
| Check type coverage | `py_get_type_hints`, `py_find_untyped_functions` |
| Understand architecture | `py_get_package_structure`, `py_analyze_dependencies` |

---

## Performance Notes

**Query Execution Times** (typical):
- Simple queries: < 50ms
- Medium queries: 50-100ms
- Complex queries: 100-200ms
- Multi-query tools: 200ms+

**Scalability**: All queries scale O(n) or O(n log n) with codebase size.

**Optimization Tips**:
1. Use module filters when possible
2. Cache results for repeated queries
3. Use snapshot feature for large codebases
4. Consider batching multiple tool calls

---

## Related Documentation

- **[Python Query Patterns](PYTHON_QUERY_PATTERNS.md)** - Learn to write custom REQL queries
- **[Python Analysis Reference](python/PYTHON_ANALYSIS_REFERENCE.md)** - Understand extracted facts
- **[Grammar Reference](GRAMMAR_REFERENCE.md)** - Complete REQL syntax
- **[AI Agent Usage Guide](AI_AGENT_USAGE_GUIDE.md)** - Best practices

---

## Status

 **30 tools available**
 **All tools tested**
 **Query tracking enabled**
 **Production ready**

**Last Updated**: 2025-11-12
