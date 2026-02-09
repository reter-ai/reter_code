# Python Code Analysis - Query Pattern Guide

**Status**: ✅ Python analysis is working correctly (all tests pass)

## Quick Start

After loading Python code:
```python
from reter import Reter

reasoner = Reter()
wme_count = reasoner.load_python_code(source_code, "mymodule")
```

You can query the extracted semantic structure using REQL.

## Query Result Format

**IMPORTANT**: Query results are returned as **PyArrow Tables**, not lists of dictionaries or tuples.

### Accessing Results

```python
# Run query
result = reasoner.reql("""
    SELECT ?method ?name
    WHERE {
        ?method type method .
        ?method has-name ?name
    }
""")

# Access by column name
method_ids = result.column('?method').to_pylist()
method_names = result.column('?name').to_pylist()

# Or convert to list of tuples
def query_to_rows(result):
    if result.num_rows == 0:
        return []
    columns = [result.column(name).to_pylist() for name in result.column_names]
    return list(zip(*columns))

rows = query_to_rows(result)
for method_id, method_name in rows:
    print(f"{method_name}: {method_id}")
```

## Core Concepts Extracted

Note: Entity IDs ARE the qualified names. For methods with overloads, the ID
includes the parameter signature: `module.Class.method(ParamType1,ParamType2)`.

### 1. Classes
- `type: class`
- `has-name: str` - Simple class name
- Entity ID = qualified name (e.g., `mymodule.MyClass`)
- `is-at-line: str` - Line number where defined
- `is-at-column: str` - Column number
- `has-docstring: str` - Class docstring (if present)

### 2. Methods
- `type: method`
- `has-name: str` - Method name
- Entity ID = qualified name with signature (e.g., `mymodule.MyClass.method_name(arg1_type,arg2_type)`)
- `is-defined-in: str` - Parent class qualified name
- `is-at-line: str` - Line number
- `has-return-type: str` - Return type annotation (if present)
- `has-docstring: str` - Method docstring (if present)
- `has-decorator: str` - Decorator name (if present)

### 3. Parameters
- `type: parameter`
- `has-name: str` - Parameter name
- `is-of-function: str` - Method qualified name
- `has-position: str` - Position in parameter list
- `has-type-annotation: str` - Type annotation (if present)
- `has-default-value: str` - Default value (if present)

### 4. Call Relationships
- `calls: str` - Method A calls Method B
- Format: `?caller calls ?callee`

## Common Query Patterns

### Pattern 1: Find All Methods in a Class

```python
result = reasoner.reql("""
    SELECT ?method ?name
    WHERE {
        ?method type method .
        ?method is-defined-in "mymodule.MyClass" .
        ?method has-name ?name
    }
""")

rows = query_to_rows(result)
for method_id, method_name in rows:
    print(f"Method: {method_name}")
```

### Pattern 2: Find Method Parameters with Types

```python
result = reasoner.reql("""
    SELECT ?param ?name ?type ?has-position
    WHERE {
        ?param type parameter .
        ?param is-of-function "mymodule.MyClass.add" .
        ?param has-name ?name .
        ?param has-type-annotation ?type .
        ?param has-position ?has-position
    }
    ORDER BY ?has-position
""")

rows = query_to_rows(result)
for param_id, param_name, param_type, has-position in rows:
    print(f"Parameter {has-position}: {param_name}: {param_type}")
```

### Pattern 3: Find Methods with Specific Return Type

```python
result = reasoner.reql("""
    SELECT ?method ?name
    WHERE {
        ?method type method .
        ?method has-return-type "int" .
        ?method has-name ?name
    }
""")

rows = query_to_rows(result)
for method_id, method_name in rows:
    print(f"Method returning int: {method_name}")
```

### Pattern 4: Find Call Relationships

```python
result = reasoner.reql("""
    SELECT ?caller ?callee
    WHERE {
        ?caller calls ?callee
    }
""")

rows = query_to_rows(result)
for caller, callee in rows:
    print(f"{caller} calls {callee}")
```

### Pattern 5: Find Methods at Specific Line Numbers

```python
result = reasoner.reql("""
    SELECT ?method ?name ?line
    WHERE {
        ?method type method .
        ?method has-name ?name .
        ?method is-at-line ?line
    }
    ORDER BY ?line
""")

rows = query_to_rows(result)
for method_id, method_name, line_num in rows:
    print(f"Line {line_num}: {method_name}")
```

### Pattern 6: Find Methods with Decorators

```python
result = reasoner.reql("""
    SELECT ?method ?name ?decorator
    WHERE {
        ?method type method .
        ?method has-name ?name .
        ?method has-decorator ?decorator
    }
""")

rows = query_to_rows(result)
for method_id, method_name, decorator in rows:
    print(f"{method_name} has decorator: {decorator}")
```

### Pattern 7: Find Methods with Default Parameters

```python
result = reasoner.reql("""
    SELECT ?method ?param ?paramName ?default
    WHERE {
        ?param type parameter .
        ?param is-of-function ?method .
        ?param has-name ?paramName .
        ?param has-default-value ?default
    }
""")

rows = query_to_rows(result)
for method, param_id, param_name, default in rows:
    print(f"{method}.{param_name} = {default}")
```

### Pattern 8: Find Classes with Docstrings

```python
result = reasoner.reql("""
    SELECT ?class ?name ?docstring
    WHERE {
        ?class type class .
        ?class has-name ?name .
        ?class has-docstring ?docstring
    }
""")

rows = query_to_rows(result)
for class_id, class_name, docstring in rows:
    print(f"{class_name}: {docstring}")
```

### Pattern 9: Find All Methods a Method Calls

```python
result = reasoner.reql("""
    SELECT ?callee
    WHERE {
        "mymodule.MyClass.calculate" calls ?callee
    }
""")

rows = query_to_rows(result)
for (callee,) in rows:
    print(f"Calls: {callee}")
```

### Pattern 10: Find Who Calls a Specific Method

```python
result = reasoner.reql("""
    SELECT ?caller
    WHERE {
        ?caller calls "mymodule.MyClass.add"
    }
""")

rows = query_to_rows(result)
for (caller,) in rows:
    print(f"Called by: {caller}")
```

## What IS Extracted (Confirmed by Tests)

✅ **Structural Elements**:
- All methods including `__init__` and magic methods
- Method parameters with names, types, has-positions
- Line and column numbers for all elements
- Return type annotations
- Call relationships between methods
- Decorators (e.g., `@staticmethod`, `@property`)
- Default parameter values
- Class and method docstrings
- Fully qualified names (module.Class.method)

## What is NOT Extracted (By Design)

❌ **Implementation Details**:
- Method bodies/logic
- Variable assignments inside methods
- Control flow (if/while/for statements)
- Expression trees
- Comments (except docstrings)

**Reason**: RETER focuses on **semantic structure** for reasoning, not detailed AST analysis.

## Common Mistakes

### Mistake 1: Wrong Result Type Assumption

**Wrong**:
```python
result = reasoner.reql("SELECT ?method WHERE { ?method type method }")
for row in result:  # ❌ PyArrow Table is not iterable this way
    print(row['?method'])
```

**Correct**:
```python
result = reasoner.reql("SELECT ?method WHERE { ?method type method }")
methods = result.column('?method').to_pylist()  # ✅ Access by column
for method_id in methods:
    print(method_id)
```

### Mistake 2: Incorrect Qualified Name Format

**Wrong**:
```python
# Using simple class name instead of qualified name
result = reasoner.reql("""
    SELECT ?method
    WHERE {
        ?method is-defined-in "MyClass"  # ❌ Missing module prefix
    }
""")
```

**Correct**:
```python
# Use fully qualified name
result = reasoner.reql("""
    SELECT ?method
    WHERE {
        ?method is-defined-in "mymodule.MyClass"  # ✅ Full path
    }
""")
```

### Mistake 3: Querying for Implementation Details

**Wrong**:
```python
# Trying to find variable assignments inside methods
result = reasoner.reql("""
    SELECT ?var ?value
    WHERE {
        ?var type variable .  # ❌ Variables not extracted
        ?var has-assigned-value ?value
    }
""")
```

**Correct**:
```python
# Query for structural elements instead
result = reasoner.reql("""
    SELECT ?param ?default
    WHERE {
        ?param type parameter .  # ✅ Parameters ARE extracted
        ?param has-default-value ?default
    }
""")
```

### Mistake 4: Not Checking for Optional Predicates

**Wrong**:
```python
# Assuming all methods have return types
result = reasoner.reql("""
    SELECT ?method ?has-return-type
    WHERE {
        ?method type method .
        ?method has-return-type ?has-return-type  # ❌ Will miss methods without type hints
    }
""")
```

**Correct**:
```python
# Use OPTIONAL for predicates that may not exist
result = reasoner.reql("""
    SELECT ?method ?has-return-type
    WHERE {
        ?method type method .
        OPTIONAL { ?method has-return-type ?has-return-type }  # ✅ Returns NULL if missing
    }
""")
```

## Troubleshooting

### Issue: Query Returns Empty Results

**Check**:
1. Is the Python code loaded? `wme_count = reasoner.load_python_code(code, "module_name")`
2. Are you using the correct qualified names? Use `"module.Class.method"` format
3. Are you querying for concepts that exist? See "What IS Extracted" section
4. Is your REQL syntax correct? Check for typos in predicate names

**Debug**:
```python
# List all concepts
result = reasoner.reql("SELECT DISTINCT ?concept WHERE { ?x concept ?concept }")
concepts = result.column('?concept').to_pylist()
print("Available concepts:", concepts)

# List all predicates for a specific entity
result = reasoner.reql("""
    SELECT ?predicate ?value
    WHERE {
        "mymodule.MyClass.method" ?predicate ?value
    }
""")
```

### Issue: Can't Find Method by Class

**Problem**: Query for methods in a class returns nothing

**Solution**: Ensure you're using the `is-defined-in` predicate with the fully qualified class name:

```python
# Correct pattern
result = reasoner.reql("""
    SELECT ?method ?name
    WHERE {
        ?method type method .
        ?method is-defined-in "mymodule.MyClass" .  # ✅ Full qualified name
        ?method has-name ?name
    }
""")
```

### Issue: Call Relationships Not Found

**Problem**: Query for `calls` predicate returns empty

**Solution**: Ensure:
1. The methods actually have call statements in the code
2. You're using qualified names for both caller and callee
3. The code was loaded successfully

```python
# Debug: List all call relationships
result = reasoner.reql("""
    SELECT ?caller ?callee
    WHERE {
        ?caller calls ?callee
    }
""")
rows = query_to_rows(result)
print(f"Found {len(rows)} call relationships")
```

## Complete Example

```python
from reter import Reter

# Sample code to analyze
code = """
class Calculator:
    '''A simple calculator'''

    def __init__(self, initial=0):
        self.value = initial

    def add(self, x: int) -> int:
        '''Add x to value'''
        self.value += x
        return self.value

    def calculate(self, a: int, b: int) -> int:
        temp = self.add(a)
        result = self.add(b)
        return result
"""

# Load code
reasoner = Reter()
wme_count = reasoner.load_python_code(code, "calculator")
print(f"Loaded {wme_count} facts")

# Query 1: Find all methods
result = reasoner.reql("""
    SELECT ?method ?name
    WHERE {
        ?method type method .
        ?method has-name ?name
    }
""")

print("\nMethods found:")
names = result.column('?name').to_pylist()
for name in names:
    print(f"  - {name}")

# Query 2: Find parameters of 'add' method
result = reasoner.reql("""
    SELECT ?name ?type ?has-position
    WHERE {
        ?param type parameter .
        ?param is-of-function "calculator.Calculator.add" .
        ?param has-name ?name .
        ?param has-type-annotation ?type .
        ?param has-position ?has-position
    }
    ORDER BY ?has-position
""")

print("\nParameters of 'add' method:")
def query_to_rows(result):
    if result.num_rows == 0:
        return []
    columns = [result.column(name).to_pylist() for name in result.column_names]
    return list(zip(*columns))

rows = query_to_rows(result)
for name, ptype, pos in rows:
    print(f"  {pos}: {name}: {ptype}")

# Query 3: Find what 'calculate' calls
result = reasoner.reql("""
    SELECT ?callee
    WHERE {
        "calculator.Calculator.calculate" calls ?callee
    }
""")

print("\n'calculate' method calls:")
callees = result.column('?callee').to_pylist()
for callee in callees:
    print(f"  - {callee}")
```

## Summary

- **Python analysis works correctly** - All tests pass
- **Use PyArrow Table column access** for query results
- **Use fully qualified names** in queries
- **Query semantic structure**, not implementation details
- **Check for optional predicates** using OPTIONAL clause
- **Refer to test suite** (`tests/test_python_analysis_queries.py`) for working examples

---

**Reference**: `tests/test_python_analysis_queries.py` contains 10 working test examples

## New Features (2025-11-19)

### Pattern 11: Find Class Attributes with Types

**NEW**: Class attributes are now extracted with type information inferred from constructor calls.

```python
result = reasoner.reql("""
    SELECT ?attr ?name ?type ?has-visibility
    WHERE {
        ?attr type field .
        ?attr is-defined-in "mymodule.MyClass" .
        ?attr has-name ?name .
        ?attr has-type ?type .
        ?attr has-visibility ?has-visibility
    }
""")

rows = query_to_rows(result)
for attr_id, attr_name, attr_type, has-visibility in rows:
    print(f"{has-visibility} attribute: {attr_name}: {attr_type}")
```

**Output**:
```
public attribute: manager: mymodule.PluginManager
protected attribute: _state: mymodule.StateManager
private attribute: __secret: mymodule.SecretManager
```

### Pattern 12: Find All Public Attributes

```python
result = reasoner.reql("""
    SELECT ?class ?attr ?name ?type
    WHERE {
        ?attr type field .
        ?attr is-defined-in ?class .
        ?attr has-name ?name .
        ?attr has-visibility "public" .
        OPTIONAL { ?attr has-type ?type }
    }
""")

rows = query_to_rows(result)
for class_name, attr_id, attr_name, attr_type in rows:
    type_str = f": {attr_type}" if attr_type else ""
    print(f"{class_name}.{attr_name}{type_str}")
```

### Pattern 13: Find Attributes By Visibility

```python
# Find protected attributes (single underscore)
result = reasoner.reql("""
    SELECT ?class ?name
    WHERE {
        ?attr type field .
        ?attr is-defined-in ?class .
        ?attr has-name ?name .
        ?attr has-visibility "protected"
    }
""")

# Find private attributes (double underscore)
result = reasoner.reql("""
    SELECT ?class ?name
    WHERE {
        ?attr type field .
        ?attr is-defined-in ?class .
        ?attr has-name ?name .
        ?attr has-visibility "private"
    }
""")
```

### Pattern 14: Find Type-Resolved Method Calls

**NEW**: Method calls like `self.manager.load()` are now resolved using type tracking.

```python
result = reasoner.reql("""
    SELECT ?caller ?caller_name ?callee ?callee_name
    WHERE {
        ?caller calls ?callee .
        ?caller has-name ?caller_name .
        ?callee has-name ?callee_name
    }
""")

rows = query_to_rows(result)
for caller, caller_name, callee, callee_name in rows:
    print(f"{caller_name} calls {callee_name}")
    print(f"  Full: {caller} -> {callee}")
```

**Example output**:
```
initialize calls load_all_plugins
  Full: mymodule.Server.initialize -> mymodule.PluginManager.load_all_plugins
```

### Pattern 15: Find Classes with Specific Attribute Types

```python
# Find classes that have a PluginManager attribute
result = reasoner.reql("""
    SELECT ?class ?attr_name
    WHERE {
        ?attr type field .
        ?attr is-defined-in ?class .
        ?attr has-name ?attr_name .
        ?attr has-type ?type .
        FILTER(REGEX(?type, "PluginManager"))
    }
""")

rows = query_to_rows(result)
for class_name, attr_name in rows:
    print(f"{class_name} has PluginManager attribute: {attr_name}")
```

### Pattern 16: Trace Type-Resolved Call Chains

```python
# Find what methods a method calls (type-resolved)
result = reasoner.reql("""
    SELECT ?method ?calls_what
    WHERE {
        "mymodule.Server.initialize" calls ?calls_what
    }
""")

rows = query_to_rows(result)
print("Server.initialize calls:")
for (callee,) in rows:
    print(f"  - {callee}")
```

## Feature Summary

### What's NEW (2025-11-19):

✅ **Class Attributes**:
- Extracted for all `self.x = value` assignments
- Type inference from constructor calls (`self.manager = PluginManager()`)
- Visibility detection: public, protected (`_`), private (`__`)
- Works with or without type annotations

✅ **Type-Resolved Method Calls**:
- Calls like `self.manager.load()` now resolve to actual methods
- Uses type tracking from constructor assignments
- Handles Python's out-of-order execution (call before assignment)
- Cross-method type persistence (types set in `__init__` visible in other methods)

### Attribute Predicates:

| Predicate | Type | Description |
|-----------|------|-------------|
| `type` | `field` | Identifies a field/attribute |
| `name` | `str` | Attribute name (e.g., `"manager"`) |
| `is-defined-in` | `str` | Qualified class name (standardized with methods) |
| `has-type` | `str` | Inferred type (qualified name) |
| `has-visibility` | `str` | `"public"`, `"protected"`, or `"private"` |

### Type Tracking Example:

```python
code = """
class Server:
    def __init__(self):
        self.manager = PluginManager()  # Type tracked: PluginManager
        self.config = ConfigManager()   # Type tracked: ConfigManager
    
    def start(self):
        self.manager.load_plugins()  # Resolves to PluginManager.load_plugins
        self.config.validate()       # Resolves to ConfigManager.validate
"""
```

After loading, you can query:
1. Attributes: `manager` (type: `PluginManager`), `config` (type: `ConfigManager`)
2. Calls: `Server.start` → `PluginManager.load_plugins`, `ConfigManager.validate`

## Exception Handling Patterns

### What's NEW (2025-11-28):

✅ **Exception Handling Facts**:
- `try-block` - Try statement blocks with metadata
- `catch-clause` - Exception handlers with detection flags
- `try-else-block` - Try-else blocks
- `finally-block` - Finally blocks with RAII detection
- `throw-statement` - Raise/throw statements with exception info
- `return-statement` - Return statements with error code detection
- `with-statement` - Context manager (with) statements
- `context-manager` - Individual context managers in with statements

### Exception Handling Predicates

| Concept | Predicate | Type | Description |
|---------|-----------|------|-------------|
| `try-block` | `is-in-function` | `str` | Function containing the try block |
| | `is-at-line` | `str` | Line number |
| | `has-except-handlers` | `str` | Number of except handlers |
| | `has-else` | `bool` | Has else block |
| | `has-finally` | `bool` | Has finally block |
| `catch-clause` | `is-in-try-block` | `str` | Parent try block |
| | `has-exception-type` | `str` | Caught exception type |
| | `has-alias-name` | `str` | Exception alias (e.g., `as e`) |
| | `is-at-line` | `str` | Line number |
| | `is-silent-swallow` | `bool` | Empty/pass handler (code smell!) |
| | `is-general-except` | `bool` | Catches broad exception type |
| | `is-bare-except` | `bool` | Bare except clause |
| `throw-statement` | `is-in-function` | `str` | Function containing raise |
| | `has-exception-type` | `str` | Raised exception type |
| | `is-at-line` | `str` | Line number |
| | `is-general-exception` | `bool` | Raises broad exception type |
| | `is-reraise` | `bool` | Bare raise (re-raise) |
| `return-statement` | `is-in-function` | `str` | Function containing return |
| | `is-at-line` | `str` | Line number |
| | `has-return-value` | `str` | Returned value |
| | `is-looks-like-error-code` | `bool` | Returns -1, None, False (smell!) |
| `with-statement` | `is-in-function` | `str` | Function containing with |
| | `is-at-line` | `str` | Line number |
| `context-manager` | `is-in-with-statement` | `str` | Parent with statement |
| | `expression` | `str` | Context manager expression |
| | `has-alias-name` | `str` | Alias (e.g., `as f`) |
| `finally-block` | `is-in-try-block` | `str` | Parent try block |
| | `is-at-line` | `str` | Line number |
| | `is-raii-cleanup` | `bool` | Manual cleanup (could use with) |

### Pattern 17: Find Silent Exception Swallowing

Find dangerous `except: pass` patterns that hide errors:

```python
result = reasoner.reql("""
    SELECT ?handler ?try_block ?exception_type ?line
    WHERE {
        ?handler type catch-clause .
        ?handler is-silent-swallow "true" .
        ?handler is-in-try-block ?try_block .
        ?handler is-at-line ?line .
        OPTIONAL { ?handler has-exception-type ?exception_type }
    }
""")

rows = query_to_rows(result)
for handler_id, try_block, exc_type, line in rows:
    exc = exc_type if exc_type else "bare except"
    print(f"Line {line}: Silent swallow of {exc} in {try_block}")
```

### Pattern 18: Find Too General Exception Catching

Find code catching overly broad exceptions:

```python
result = reasoner.reql("""
    SELECT ?handler ?exception_type ?line ?function
    WHERE {
        ?handler type catch-clause .
        ?handler is-general-except "true" .
        ?handler has-exception-type ?exception_type .
        ?handler is-at-line ?line .
        ?handler is-in-try-block ?try .
        ?try is-in-function ?function
    }
""")

rows = query_to_rows(result)
for handler_id, exc_type, line, func in rows:
    print(f"Line {line}: {func} catches too-general '{exc_type}'")
```

### Pattern 19: Find Functions Raising General Exceptions

Find code that raises Exception, BaseException, etc:

```python
result = reasoner.reql("""
    SELECT ?raise ?exception_type ?line ?function
    WHERE {
        ?raise type throw-statement .
        ?raise is-general-exception "true" .
        ?raise has-exception-type ?exception_type .
        ?raise is-in-function ?function .
        ?raise is-at-line ?line
    }
""")

rows = query_to_rows(result)
for raise_id, exc_type, line, func in rows:
    print(f"Line {line}: {func} raises generic '{exc_type}'")
```

### Pattern 20: Find Error Code Returns

Find functions returning error codes instead of raising exceptions:

```python
result = reasoner.reql("""
    SELECT ?return ?value ?line ?function
    WHERE {
        ?return type return-statement .
        ?return is-looks-like-error-code "true" .
        ?return has-return-value ?value .
        ?return is-in-function ?function .
        ?return is-at-line ?line
    }
""")

rows = query_to_rows(result)
for return_id, value, line, func in rows:
    print(f"Line {line}: {func} returns error code '{value}'")
```

### Pattern 21: Find Context Managers (with statements)

Find all with statement usages:

```python
result = reasoner.reql("""
    SELECT ?with ?function ?line
    WHERE {
        ?with type with-statement .
        ?with is-in-function ?function .
        ?with is-at-line ?line
    }
""")

rows = query_to_rows(result)
for with_id, func, line in rows:
    print(f"Line {line}: Context manager in {func}")
```

### Pattern 22: Find Finally Blocks That Could Be Context Managers

Find manual cleanup in finally that should use `with`:

```python
result = reasoner.reql("""
    SELECT ?finally ?try_block ?line ?function
    WHERE {
        ?finally type finally-block .
        ?finally is-raii-cleanup "true" .
        ?finally is-in-try-block ?try_block .
        ?finally is-at-line ?line .
        ?try_block is-in-function ?function
    }
""")

rows = query_to_rows(result)
for finally_id, try_block, line, func in rows:
    print(f"Line {line}: {func} has manual cleanup - consider using 'with'")
```

### Pattern 23: Find Bare Except Clauses

Find `except:` without specifying exception type:

```python
result = reasoner.reql("""
    SELECT ?handler ?line ?function
    WHERE {
        ?handler type catch-clause .
        ?handler is-bare-except "true" .
        ?handler is-at-line ?line .
        ?handler is-in-try-block ?try .
        ?try is-in-function ?function
    }
""")

rows = query_to_rows(result)
for handler_id, line, func in rows:
    print(f"Line {line}: Bare except in {func} - catches everything!")
```

### Pattern 24: Analyze Exception Handling in a Function

Get complete exception handling profile for a function:

```python
function_name = "mymodule.MyClass.process"

# Find try blocks
try_result = reasoner.reql(f"""
    SELECT ?try ?line ?handlers ?has_else ?has_finally
    WHERE {{
        ?try type try-block .
        ?try is-in-function "{function_name}" .
        ?try is-at-line ?line .
        ?try has-except-handlers ?handlers .
        OPTIONAL {{ ?try has-else ?has_else }} .
        OPTIONAL {{ ?try has-finally ?has_finally }}
    }}
""")

# Find raises
raise_result = reasoner.reql(f"""
    SELECT ?raise ?exception_type ?line
    WHERE {{
        ?raise type throw-statement .
        ?raise is-in-function "{function_name}" .
        ?raise is-at-line ?line .
        OPTIONAL {{ ?raise has-exception-type ?exception_type }}
    }}
""")

# Find with statements
with_result = reasoner.reql(f"""
    SELECT ?with ?line
    WHERE {{
        ?with type with-statement .
        ?with is-in-function "{function_name}" .
        ?with is-at-line ?line
    }}
""")
```

### Pattern 25: Find All Re-raises

Find `raise` without arguments (re-raising caught exception):

```python
result = reasoner.reql("""
    SELECT ?raise ?line ?function
    WHERE {
        ?raise type throw-statement .
        ?raise is-reraise "true" .
        ?raise is-in-function ?function .
        ?raise is-at-line ?line
    }
""")

rows = query_to_rows(result)
for raise_id, line, func in rows:
    print(f"Line {line}: Re-raise in {func}")
```

### Complete Exception Handling Example

```python
code = """
class FileProcessor:
    def process_file(self, path):
        try:
            f = open(path)
            data = f.read()
            f.close()  # Manual cleanup - should use 'with'
        except:  # Bare except - too general!
            pass  # Silent swallow - dangerous!

    def better_process(self, path):
        try:
            with open(path) as f:  # Good: context manager
                return f.read()
        except FileNotFoundError as e:  # Good: specific exception
            raise ProcessingError(f"File not found: {path}") from e

    def check_status(self, item):
        if not item:
            return -1  # Error code - should raise exception
        return 0
"""

reasoner = Reter()
reasoner.load_python_code(code, "fileproc")

# Find all issues
silent_swallows = reasoner.reql("""
    SELECT ?line WHERE {
        ?h type catch-clause .
        ?h is-silent-swallow "true" .
        ?h is-at-line ?line
    }
""")

error_codes = reasoner.reql("""
    SELECT ?line ?value WHERE {
        ?r type return-statement .
        ?r is-looks-like-error-code "true" .
        ?r is-at-line ?line .
        ?r has-return-value ?value
    }
""")

bare_excepts = reasoner.reql("""
    SELECT ?line WHERE {
        ?h type catch-clause .
        ?h is-bare-except "true" .
        ?h is-at-line ?line
    }
""")
```
