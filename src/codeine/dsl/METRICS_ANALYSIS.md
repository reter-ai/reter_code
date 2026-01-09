# CADSL Pre-computed Metrics Analysis

## Overview

CADSL tools assume certain pre-computed metrics exist as properties in the knowledge base. However, many of these are **NOT emitted by the Python parser** and must be computed dynamically using `COUNT()` aggregations.

This document details which metrics are available, which need to be computed, and the fix status for affected CADSL tools.

**Status: ✅ ALL TOOLS FIXED** (as of 2025-12-25)

---

## Metrics Emitted by Parser

The Python fact extraction (`PythonFactExtractionVisitor.cpp`) emits the following metrics:

| Property | Type | Description | Emitted For |
|----------|------|-------------|-------------|
| `atLine` | int | Start line number | Class, Function, Method |
| `endLine` | int | End line number | Class, Function, Method |
| `lineCount` | int | `endLine - atLine + 1` | Class, Function, Method |

**Usage in REQL:**
```sql
SELECT ?m ?name ?line_count
WHERE {
    ?m type {Method} .
    ?m name ?name .
    ?m lineCount ?line_count .
}
FILTER { ?line_count > 50 }
```

---

## Metrics NOT Emitted (Must Use COUNT())

These metrics are **NOT** pre-computed properties. They must be calculated using `COUNT()` aggregations:

### methodCount

**Wrong (assumes pre-computed property):**
```sql
?c methodCount ?method_count .
FILTER { ?method_count > 15 }
```

**Correct (use COUNT aggregation):**
```sql
SELECT ?c ?name (COUNT(?method) AS ?method_count)
WHERE {
    ?c type {Class} .
    ?c name ?name .
    ?method type {Method} .
    ?method definedIn ?c .
}
GROUP BY ?c ?name
HAVING (?method_count > 15)
```

### parameterCount

**Wrong:**
```sql
?m parameterCount ?param_count .
```

**Correct:**
```sql
SELECT ?m ?name (COUNT(?param) AS ?param_count)
WHERE {
    ?m type {Method} .
    ?m name ?name .
    ?param type {Parameter} .
    ?param ofFunction ?m .
}
GROUP BY ?m ?name
HAVING (?param_count > 5)
```

### attributeCount

**Wrong:**
```sql
?c attributeCount ?attr_count .
```

**Correct:**
```sql
SELECT ?c ?name (COUNT(?attr) AS ?attr_count)
WHERE {
    ?c type {Class} .
    ?c name ?name .
    ?attr type {Field} .
    ?attr definedIn ?c .
}
GROUP BY ?c ?name
```

### callerCount

**Wrong:**
```sql
?m callerCount ?caller_count .
```

**Correct:**
```sql
SELECT ?m ?name (COUNT(?caller) AS ?caller_count)
WHERE {
    ?m type {Method} .
    ?m name ?name .
    ?caller calls ?m .
}
GROUP BY ?m ?name
```

### fanoutCount (callees)

**Wrong:**
```sql
?m fanoutCount ?fanout .
```

**Correct:**
```sql
SELECT ?m ?name (COUNT(?callee) AS ?fanout)
WHERE {
    ?m type {Method} .
    ?m name ?name .
    ?m calls ?callee .
}
GROUP BY ?m ?name
```

### subclassCount

**Wrong:**
```sql
?c subclassCount ?count .
```

**Correct:**
```sql
SELECT ?c ?name (COUNT(?sub) AS ?subclass_count)
WHERE {
    ?c type {Class} .
    ?c name ?name .
    ?sub inheritsFrom ?c .
}
GROUP BY ?c ?name
```

### setterCount (via naming pattern)

**Correct:**
```sql
SELECT ?c ?name (COUNT(?setter) AS ?setter_count)
WHERE {
    ?c type {Class} .
    ?c name ?name .
    ?setter type {Method} .
    ?setter definedIn ?c .
    ?setter name ?setter_name .
    FILTER { STRSTARTS(?setter_name, "set_") || STRSTARTS(?setter_name, "set") }
}
GROUP BY ?c ?name
```

---

## Metrics NOT Available (Require AST Analysis)

These metrics are **NOT emitted by the parser** and cannot be easily computed via REQL:

| Metric | Description | Recommendation |
|--------|-------------|----------------|
| `complexityScore` | Cyclomatic complexity | Requires AST analysis |
| `conditionCount` | Number of conditionals | Requires AST analysis |
| `maxNestingDepth` | Maximum nesting level | Requires AST analysis |
| `accesses` | Variable access tracking | Requires AST analysis |
| `assigns` | Variable assignment tracking | Requires AST analysis |
| `hasLocal` | Local variable detection | Requires AST analysis |
| `isMutable` | Mutability tracking | Requires AST analysis |

**Removed Detectors** (used unavailable properties):
- `deep_nesting.py` - used `maxNestingDepth`
- `mutable_shared_data.py` - used `accesses`, `isGlobal`
- `function_groups.py` - used `accesses`, `isGlobal`
- `split_variable.py` - used `assigns`, `hasLocal`
- `encapsulate_record.py` - used `hasLocal`, Variable type
- `encapsulate_field.py` - used `accesses`

---

## Properties Emitted by Parser

Reference of properties available from `PythonFactExtractionVisitor.cpp`:

Note: Entity IDs ARE the qualified names. For methods with overloads, the ID
includes the parameter signature: `module.Class.method(ParamType1,ParamType2)`.

| Property | Description | Available For |
|----------|-------------|---------------|
| `name` | Entity name | All entities |
| `inFile` | Source file path | All entities |
| `atLine` | Start line | All entities |
| `endLine` | End line | Class, Function, Method |
| `lineCount` | Line count | Class, Function, Method |
| `definedIn` | Parent class | Method, Field |
| `inheritsFrom` | Parent class(es) | Class |
| `hasMethod` | Class has method | Class |
| `hasAttribute` | Class has attribute | Class |
| `hasParameter` | Function has param | Function, Method |
| `calls` | Calls target | Function, Method |
| `imports` | Imports module | Module |
| `returnType` | Return type annotation | Function, Method |
| `typeAnnotation` | Type annotation | Parameter |
| `hasType` | Type annotation | Attribute/Field |
| `hasDecorator` | Decorator name | Function, Method, Class |
| `hasDocstring` | Has docstring | Function, Method, Class |
| `isAsync` | Async function | Function, Method |
| `isAbstract` | Abstract method | Method |

---

## Computable via REQL

Some metrics not directly emitted can be computed via REQL patterns:

### sharedMethodCount (via self-join)

```sql
SELECT ?c1 ?class1 ?c2 ?class2 (COUNT(?shared_name) AS ?shared_methods)
WHERE {
    ?c1 type {Class} . ?c1 name ?class1 .
    ?c2 type {Class} . ?c2 name ?class2 .
    ?m1 type {Method} . ?m1 definedIn ?c1 . ?m1 name ?shared_name .
    ?m2 type {Method} . ?m2 definedIn ?c2 . ?m2 name ?shared_name .
    FILTER { ?c1 != ?c2 }
}
GROUP BY ?c1 ?class1 ?c2 ?class2
HAVING (?shared_methods >= 3)
```

### Global variables (via Assignment without inFunction)

```sql
SELECT ?a ?target ?file ?line
WHERE {
    ?a type {Assignment} .
    ?a target ?target .
    ?a inFile ?file .
    ?a atLine ?line .
    FILTER NOT EXISTS { ?a inFunction ?f }
    FILTER NOT EXISTS { ?a inClass ?c }
}
```

---

## Fixed Files Status

### Smells ✅ ALL FIXED

| File | Original Issue | Status |
|------|----------------|--------|
| `god_class.py` | `methodCount`, `attributeCount` | ✅ Fixed - uses COUNT() |
| `long_parameter_list.py` | `parameterCount` | ✅ Fixed - uses COUNT() |
| `data_class.py` | `attributeCount`, `methodCount` | ✅ Fixed - uses COUNT() |
| `long_methods.py` | `lineCount` | ✅ OK - parser emits this |
| `feature_envy.py` | `externalCallCount`, `internalCallCount` | ✅ Fixed - COUNT() on external calls |
| `trivial_commands.py` | `methodCount` | ✅ Fixed - uses COUNT() |
| `setting_methods.py` | `setterCount` | ✅ Fixed - COUNT() with name pattern |
| `speculative_generality.py` | `subclassCount` | ✅ Fixed - COUNT() on inheritsFrom |
| `shotgun_surgery.py` | `dependentCount` | ✅ Fixed - COUNT() on callers |
| `primitive_obsession.py` | `primitiveParamCount` | ✅ Fixed - COUNT() with typeAnnotation |
| `global_data.py` | Global detection | ✅ Fixed - Assignment without inFunction |

### Refactoring ✅ ALL FIXED

| File | Original Issue | Status |
|------|----------------|--------|
| `extract_class.py` | `methodCount`, `attributeCount` | ✅ Fixed - uses COUNT() |
| `extract_method.py` | `lineCount` | ✅ OK - parser emits this |
| `inline_class.py` | `methodCount`, `attributeCount`, `userCount` | ✅ Fixed - uses COUNT() |
| `inline_method.py` | `lineCount`, `callerCount` | ✅ Fixed - lineCount OK, COUNT() for callers |
| `introduce_parameter_object.py` | `parameterCount` | ✅ Fixed - uses COUNT() |
| `remove_middle_man.py` | `methodCount`, `delegationCount` | ✅ Fixed - COUNT() on delegations |
| `replace_inheritance_with_delegation.py` | `methodCount` | ✅ Fixed - COUNT() on parent calls |
| `push_down_method.py` | `subclassCount`, `subclassUsageCount` | ✅ Fixed - uses COUNT() |
| `hide_delegate.py` | `delegateUsageCount` | ✅ Fixed - uses COUNT() with returnType |
| `duplicate_parameter_lists.py` | Uses `ParameterSignature` type | ✅ Fixed - simplified to COUNT() |
| `attribute_data_clumps.py` | Uses `AttributeClump` type | ✅ Fixed - simplified to COUNT() |

### Inheritance ✅ ALL FIXED

| File | Original Issue | Status |
|------|----------------|--------|
| `extract_superclass.py` | `sharedMethodCount` | ✅ Fixed - self-join COUNT() |
| `remove_subclass.py` | `methodCount`, `overrideCount` | ✅ Fixed - uses COUNT() |
| `replace_with_delegate.py` | `methodCount`, `parentMethodUsage` | ✅ Fixed - uses COUNT() |
| `collapse_hierarchy.py` | `addedMethodCount` | ✅ Fixed - uses COUNT() |

### Testing ✅ ALL FIXED

| File | Original Issue | Status |
|------|----------------|--------|
| `untested_classes.py` | `methodCount` | ✅ Fixed - uses COUNT(), removed Source class |
| `untested_functions.py` | `lineCount` | ✅ OK - parser emits this |
| `untested_methods.py` | `lineCount` | ✅ OK - parser emits this |
| `high_fanout_untested.py` | `fanoutCount` | ✅ Fixed - COUNT() on calls |
| `partial_coverage.py` | `testedMethodCount`, `publicMethodCount` | ✅ Fixed - COUNT() with OPTIONAL |
| `shallow_tests.py` | `testMethodCount` | ✅ Fixed - uses COUNT() |
| `large_untested_modules.py` | `classCount`, `testCount` | ✅ Fixed - uses COUNT() |

### Inspection ✅ ALL FIXED

| File | Original Issue | Status |
|------|----------------|--------|
| `list_classes.py` | Uses COUNT() correctly | ✅ OK - already uses COUNT() |
| `get_complexity.py` | `methodCount`, `attributeCount` | ✅ Fixed - uses COUNT() |
| `get_architecture.py` | `classCount`, `functionCount`, `importCount` | ✅ Fixed - uses COUNT() |
| `untyped_functions.py` | `parameterCount`, `typedParameterCount` | ✅ Fixed - uses COUNT() with typeAnnotation |

---

## Reference Implementation Patterns

### Pattern 1: Single COUNT() in SELECT

```python
reql('''
    SELECT ?c ?name ?file ?line (COUNT(?method) AS ?method_count)
    WHERE {
        ?c type {Class} .
        ?c name ?name .
        ?c inFile ?file .
        ?c atLine ?line .
        ?method type {Method} .
        ?method definedIn ?c .
    }
    GROUP BY ?c ?name ?file ?line
    HAVING (?method_count >= {min_methods})
    ORDER BY DESC(?method_count)
''')
```

### Pattern 2: Multiple COUNT() in SELECT

```python
reql('''
    SELECT ?c ?name ?file ?line (COUNT(?method) AS ?method_count) (COUNT(?attr) AS ?attr_count)
    WHERE {
        ?c type {Class} .
        ?c name ?name .
        ?c inFile ?file .
        ?c atLine ?line .
        OPTIONAL { ?method type {Method} . ?method definedIn ?c }
        OPTIONAL { ?attr type {Field} . ?attr definedIn ?c }
    }
    GROUP BY ?c ?name ?file ?line
    HAVING (?attr_count >= {min_attributes} && ?method_count <= {max_methods})
''')
```

### Pattern 3: COUNT() with FILTER (e.g., setters by name pattern)

```python
reql('''
    SELECT ?c ?name ?file ?line (COUNT(?setter) AS ?setter_count)
    WHERE {
        ?c type {Class} .
        ?c name ?name .
        ?c inFile ?file .
        ?c atLine ?line .
        ?setter type {Method} .
        ?setter definedIn ?c .
        ?setter name ?setter_name .
        FILTER { STRSTARTS(?setter_name, "set_") || STRSTARTS(?setter_name, "set") }
    }
    GROUP BY ?c ?name ?file ?line
    HAVING (?setter_count >= {min_setters})
''')
```

### Pattern 4: COUNT() with Type Filter (e.g., primitive params)

```python
reql('''
    SELECT ?m ?name ?file ?line (COUNT(?primitive_param) AS ?primitive_params)
    WHERE {
        ?m type {Method} .
        ?m name ?name .
        ?m inFile ?file .
        ?m atLine ?line .
        ?primitive_param type {Parameter} .
        ?primitive_param ofFunction ?m .
        ?primitive_param typeAnnotation ?ptype .
        FILTER { REGEX(?ptype, "^(str|int|float|bool|bytes)$") }
    }
    GROUP BY ?m ?name ?file ?line
    HAVING (?primitive_params >= {min_primitives})
''')
```

### Pattern 5: Direct lineCount Query (parser emits this)

```python
reql('''
    SELECT ?m ?name ?file ?line ?line_count
    WHERE {
        ?m type {Method} .
        ?m name ?name .
        ?m inFile ?file .
        ?m atLine ?line .
        ?m lineCount ?line_count .
    }
    FILTER { ?line_count > {max_lines} }
''')
```

---

## Notes

1. **HAVING vs FILTER**: Use `HAVING` for filtering on aggregated values (after GROUP BY), use `FILTER` for filtering on individual rows (before GROUP BY).

2. **GROUP BY requirements**: All non-aggregated SELECT variables must appear in GROUP BY.

3. **OPTIONAL for counts**: Use `OPTIONAL` when an entity might not have any items to count (e.g., a class with no methods).

4. **Performance**: COUNT() aggregations are efficient in RETER. Single queries with multiple COUNTs are preferred over multiple separate queries.

5. **Removed Source classes**: All custom Source classes for metric computation have been removed in favor of pure REQL with COUNT() aggregations.

---

## Python Post-Processing Patterns

When REQL alone can't express the computation, use `.tap()` with Python functions.

### Pattern 6: DFS Graph Traversal (circular imports)

```python
reql('''
    SELECT ?module1 ?module2 ?file1 ?file2
    WHERE { ?m1 imports ?m2 . ... }
''')
.tap(lambda rows: _detect_cycles_dfs(rows))
.emit("findings")

def _detect_cycles_dfs(rows):
    # Build adjacency graph
    graph = defaultdict(list)
    for row in rows:
        graph[row['module1']].append(row['module2'])

    # DFS with recursion stack
    visited, rec_stack = set(), set()
    cycles = []

    def dfs(node, path):
        visited.add(node)
        rec_stack.add(node)
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                dfs(neighbor, path + [node])
            elif neighbor in rec_stack:
                cycles.append(path[path.index(neighbor):])
        rec_stack.remove(node)

    for node in graph:
        if node not in visited:
            dfs(node, [])
    return cycles
```

### Pattern 7: N² Similarity Comparison (alternative interfaces)

```python
reql('''
    SELECT ?c ?class_name ?method_name ...
    WHERE { ?m definedIn ?c . ?m name ?method_name . }
''')
.tap(lambda rows: _find_similar_classes(rows))

def _find_similar_classes(rows):
    # Build class -> methods mapping
    class_methods = defaultdict(set)
    for row in rows:
        class_methods[row['c']].add(row['method_name'])

    # N² comparison with Jaccard similarity
    findings = []
    class_list = list(class_methods.items())
    for i, (c1, methods1) in enumerate(class_list):
        for j in range(i + 1, len(class_list)):
            c2, methods2 = class_list[j]
            intersection = methods1 & methods2
            union = methods1 | methods2
            similarity = len(intersection) / len(union)
            if similarity >= threshold:
                findings.append({...})
    return findings
```

### Pattern 8: Grouping with defaultdict (duplicate names)

```python
reql('''
    SELECT ?name ?type ?file WHERE { ?e name ?name . ?e inFile ?file }
''')
.tap(lambda rows: _group_and_count(rows))

def _group_and_count(rows):
    name_occurrences = defaultdict(lambda: {"files": set(), "types": set()})
    for row in rows:
        name_occurrences[row['name']]["files"].add(row['file'])
    return [
        {"name": n, "count": len(d["files"]), ...}
        for n, d in name_occurrences.items()
        if len(d["files"]) > 1
    ]
```

### Pattern 9: Name Pattern Matching (parallel inheritance)

```python
def _names_are_parallel(name1, name2):
    # Check for common suffix (FooHandler, BarHandler)
    min_len = min(len(name1), len(name2))
    for suffix_len in range(3, min_len):
        if name1[-suffix_len:] == name2[-suffix_len:]:
            return True
    return False
```

---

## Removed Detectors (Second Round)

These detectors were removed because they require AST-level analysis:

| File | Reason |
|------|--------|
| `complex_conditionals.py` | Uses `conditionCount`, `maxNestingDepth` - requires AST |

## Fixed with Python Post-Processing

| File | Fix Applied |
|------|-------------|
| `circular_imports.py` | DFS cycle detection on import graph |
| `alternative_interfaces.py` | N² Jaccard similarity on method sets |
| `duplicate_names.py` | defaultdict grouping |
| `parallel_inheritance.py` | Name pattern matching |
| `complex_untested.py` | Uses `lineCount` as complexity proxy |
