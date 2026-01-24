# JavaScript Source Code Analysis - Complete Reference

This document describes all facts (triples/WMEs) extracted from JavaScript source code by RETER's JavaScript analyzer.

**For MCP Users**: After loading JavaScript code with `load_javascript_file()`, `load_javascript_directory()`, or `load_javascript_code()`, use this reference to understand how to query the extracted code structure using `query_select_reql()`.

## Quick Links

- **[JavaScript Ontology](js_ontology.reol)** - Inference rules for JS code analysis
- **[Python Analysis Reference](../python/PYTHON_ANALYSIS_REFERENCE.md)** - Compare with Python extraction

---

## Overview

RETER parses JavaScript source code (ES6+) and extracts semantic facts into a knowledge graph. Each code element (module, class, function, etc.) becomes an entity with properties and relationships.

**How to load JavaScript code**:
```python
# Load a single file
result = load_javascript_file(
    filepath="/path/to/mymodule.js",
    module_name="mymodule"
)

# Load entire directory
result = load_javascript_directory(
    directory="/path/to/src",
    recursive=True
)

# Load code from string
result = load_javascript_code(
    javascript_code="class MyClass { }",
    module_name="mymodule"
)
```

**Total Entity Types**: 15
- `js:Module` - JavaScript modules/files
- `js:Class` - Class definitions (ES6+)
- `js:Function` - Function declarations
- `js:Method` - Method definitions (inside classes)
- `js:Constructor` - Constructor methods
- `js:ArrowFunction` - Arrow function expressions
- `js:Parameter` - Function/method parameters
- `js:Variable` - Variable declarations (var, let, const)
- `js:Field` - Class field definitions
- `js:Import` - Import statements
- `js:Export` - Export statements
- `js:TryBlock` - Try/catch/finally blocks
- `js:CatchClause` - Catch clauses
- `js:ThrowStatement` - Throw statements
- `js:ReturnStatement` - Return statements
- `js:Call` - Function/method calls
- `js:Assignment` - Assignment expressions

---

## Complete Predicate Reference

### Core Identity Predicates

Used by ALL entities:

| Predicate | Type | Description | Example Value |
|-----------|------|-------------|---------------|
| `type` | string | Always "instance_of" | `"instance_of"` |
| `individual` | string | Unique identifier | `"class_5"` |
| `concept` | string | Entity type | `"js:Class"`, `"js:Function"` |
| `name` | string | Simple name | `"MyClass"` |

### Module Predicates

For entities of type `js:Module`:

| Predicate | Type | Description | Example |
|-----------|------|-------------|---------|
| `name` | string | Module name | `"mymodule"` |

Note: The entity ID IS the qualified name (e.g., `mymodule`).

**Example Module Entity**:
```
mymodule  type         "instance_of"
mymodule  individual   "mymodule"
mymodule  concept      "js:Module"
mymodule  name         "mymodule"
```

### Class Predicates

For entities of type `js:Class`:

| Predicate | Type | Description | Example |
|-----------|------|-------------|---------|
| `name` | string | Simple class name | `"Calculator"` |
| `inModule` | string | Parent module | `"mymodule"` |
| `atLine` | string | Line number | `"10"` |
| `lineCount` | string | Number of lines | `"25"` |
| `hasDocstring` | string | JSDoc comment | `"Calculator class..."` |
| `noConstructor` | string | Has no constructor? | `"true"` |
| `hasPrivateFields` | string | Has private fields? | `"true"` |

Note: The entity ID IS the qualified name (e.g., `mymodule.Calculator`).

**Inheritance Relationship** (separate fact):
```
rel_fact  type         "role_assertion"
rel_fact  subject      "mymodule.ChildClass"
rel_fact  role         "inheritsFrom"
rel_fact  object       "mymodule.ParentClass"
```

**Example Class Entity**:
```
mymodule.Calculator  type           "instance_of"
mymodule.Calculator  individual     "mymodule.Calculator"
mymodule.Calculator  concept        "js:Class"
mymodule.Calculator  name           "Calculator"
mymodule.Calculator  inModule       "mymodule"
mymodule.Calculator  atLine         "10"
mymodule.Calculator  lineCount      "25"
mymodule.Calculator  hasDocstring   "Calculator for math operations"
```

### Function Predicates

For entities of type `js:Function`:

| Predicate | Type | Description | Example |
|-----------|------|-------------|---------|
| `name` | string | Function name | `"calculate"` |
| `inModule` | string | Parent module | `"mymodule"` |
| `atLine` | string | Line number | `"25"` |
| `lineCount` | string | Number of lines | `"10"` |
| `isAsync` | string | Is async function? | `"true"` / `"false"` |
| `isGenerator` | string | Is generator function? | `"true"` / `"false"` |
| `hasDocstring` | string | JSDoc comment | `"Calculates result"` |
| `parameterCount` | string | Number of parameters | `"3"` |

Note: The entity ID IS the qualified name (e.g., `mymodule.calculate()`).
For overloaded functions, the ID includes parameter types: `module.func(Type1,Type2)`.

**Example Function Entity**:
```
mymodule.calculate()  type           "instance_of"
mymodule.calculate()  individual     "mymodule.calculate()"
mymodule.calculate()  concept        "js:Function"
mymodule.calculate()  name           "calculate"
mymodule.calculate()  inModule       "mymodule"
mymodule.calculate()  atLine         "25"
mymodule.calculate()  lineCount      "10"
mymodule.calculate()  isAsync        "false"
mymodule.calculate()  isGenerator    "false"
```

### Method Predicates

For entities of type `js:Method`:

Same as `js:Function` predicates, PLUS:

| Predicate | Type | Description | Example |
|-----------|------|-------------|---------|
| `isStatic` | string | Is static method? | `"true"` / `"false"` |
| `isGetter` | string | Is getter? | `"true"` / `"false"` |
| `isSetter` | string | Is setter? | `"true"` / `"false"` |
| `isPrivate` | string | Is private (#)? | `"true"` / `"false"` |

**Relationships**:
```
# Class has method
rel_fact  type      "role_assertion"
rel_fact  subject   "class_1"
rel_fact  role      "hasMethod"
rel_fact  object    "method_1"

# Method defined in class
rel_fact  type      "role_assertion"
rel_fact  subject   "method_1"
rel_fact  role      "definedIn"
rel_fact  object    "class_1"
```

**Example Method Entity**:
```
method_1  type           "instance_of"
method_1  individual     "method_1"
method_1  concept        "js:Method"
method_1  name           "add"
method_1  atLine         "15"
method_1  lineCount      "3"
method_1  isAsync        "false"
method_1  isGetter       "false"
method_1  isSetter       "false"
method_1  isStatic       "false"
method_1  isPrivate      "false"
```

### Constructor Predicates

For entities of type `js:Constructor`:

Same predicates as `js:Method`.

**Relationship**:
```
rel_fact  type      "role_assertion"
rel_fact  subject   "class_1"
rel_fact  role      "hasConstructor"
rel_fact  object    "constructor_1"
```

### Arrow Function Predicates

For entities of type `js:ArrowFunction`:

| Predicate | Type | Description | Example |
|-----------|------|-------------|---------|
| `name` | string | Always `"<arrow>"` | `"<arrow>"` |
| `atLine` | string | Line number | `"42"` |
| `isArrow` | string | Always `"true"` | `"true"` |
| `isAsync` | string | Is async? | `"true"` / `"false"` |

### Parameter Predicates

For entities of type `js:Parameter`:

| Predicate | Type | Description | Example |
|-----------|------|-------------|---------|
| `name` | string | Parameter name | `"x"` |
| `index` | string | Parameter index (0-based) | `"0"` |
| `hasDefault` | string | Has default value? | `"true"` |
| `isRest` | string | Is rest parameter? | `"true"` |

**Relationship**:
```
rel_fact  type      "role_assertion"
rel_fact  subject   "function_1"
rel_fact  role      "hasParameter"
rel_fact  object    "parameter_1"
```

### Variable Predicates

For entities of type `js:Variable`:

| Predicate | Type | Description | Example |
|-----------|------|-------------|---------|
| `name` | string | Variable name | `"count"` |
| `declarationType` | string | var/let/const | `"const"` |
| `atLine` | string | Line number | `"5"` |

**Relationship** (to containing scope):
```
rel_fact  type      "role_assertion"
rel_fact  subject   "function_1"
rel_fact  role      "hasVariable"
rel_fact  object    "variable_1"
```

### Field Predicates

For entities of type `js:Field`:

| Predicate | Type | Description | Example |
|-----------|------|-------------|---------|
| `name` | string | Field name | `"#privateField"` |
| `atLine` | string | Line number | `"12"` |
| `isPrivate` | string | Is private (#)? | `"true"` |
| `isStatic` | string | Is static? | `"false"` |

**Relationship**:
```
rel_fact  type      "role_assertion"
rel_fact  subject   "class_1"
rel_fact  role      "hasField"
rel_fact  object    "field_1"
```

### Import Predicates

For entities of type `js:Import`:

| Predicate | Type | Description | Example |
|-----------|------|-------------|---------|
| `modulePath` | string | Import path | `"react"` |
| `atLine` | string | Line number | `"1"` |

**Module imports relationship**:
```
rel_fact  type      "role_assertion"
rel_fact  subject   "mymodule"
rel_fact  role      "imports"
rel_fact  object    "react"
```

### Export Predicates

For entities of type `js:Export`:

| Predicate | Type | Description | Example |
|-----------|------|-------------|---------|
| `atLine` | string | Line number | `"50"` |
| `isDefault` | string | Is default export? | `"true"` |

**Module exports relationship**:
```
rel_fact  type      "role_assertion"
rel_fact  subject   "mymodule"
rel_fact  role      "exports"
rel_fact  object    "export_1"
```

### TryBlock Predicates

For entities of type `js:TryBlock`:

| Predicate | Type | Description | Example |
|-----------|------|-------------|---------|
| `atLine` | string | Line number | `"20"` |
| `hasCatch` | string | Has catch clause? | `"true"` |
| `hasFinally` | string | Has finally clause? | `"true"` |
| `inFunction` | string | Containing function | `"function_1"` |
| `inClass` | string | Containing class | `"class_1"` |
| `inModule` | string | Containing module | `"mymodule"` |

**Relationships**:
```
rel_fact  type      "role_assertion"
rel_fact  subject   "try_1"
rel_fact  role      "hasCatch"
rel_fact  object    "catch_1"

rel_fact  type      "role_assertion"
rel_fact  subject   "try_1"
rel_fact  role      "hasFinally"
rel_fact  object    "finally_1"
```

### CatchClause Predicates

For entities of type `js:CatchClause`:

| Predicate | Type | Description | Example |
|-----------|------|-------------|---------|
| `atLine` | string | Line number | `"25"` |
| `inTryBlock` | string | Parent try block | `"try_1"` |
| `exceptionVar` | string | Exception variable | `"e"` |
| `bodyIsEmpty` | string | Is catch body empty? | `"true"` / `"false"` |

### ThrowStatement Predicates

For entities of type `js:ThrowStatement`:

| Predicate | Type | Description | Example |
|-----------|------|-------------|---------|
| `atLine` | string | Line number | `"30"` |
| `expression` | string | Thrown expression | `"new Error(\"msg\")"` |
| `throwsErrorType` | string | Throws Error type? | `"true"` |
| `throwsNewError` | string | Uses new Error()? | `"true"` |
| `inFunction` | string | Containing function | `"function_1"` |
| `inTryBlock` | string | Inside try block? | `"try_1"` |

### ReturnStatement Predicates

For entities of type `js:ReturnStatement`:

| Predicate | Type | Description | Example |
|-----------|------|-------------|---------|
| `atLine` | string | Line number | `"35"` |
| `hasValue` | string | Has return value? | `"true"` / `"false"` |
| `returnValue` | string | The return value | `"result"` |
| `returnsThis` | string | Returns this? | `"true"` |
| `returnsNullish` | string | Returns null/undefined? | `"true"` |
| `looksLikeStatusCode` | string | Returns -1/0/1/true/false? | `"true"` |
| `inFunction` | string | Containing function | `"function_1"` |

**Function has return relationship**:
```
rel_fact  type      "role_assertion"
rel_fact  subject   "function_1"
rel_fact  role      "hasReturn"
rel_fact  object    "35"
```

### Call Predicates

For entities of type `js:Call`:

| Predicate | Type | Description | Example |
|-----------|------|-------------|---------|
| `callee` | string | Called function/method | `"console.log"` |
| `atLine` | string | Line number | `"40"` |
| `argumentCount` | string | Number of arguments | `"2"` |
| `caller` | string | Calling function | `"function_1"` |
| `isMethodCall` | string | Is method call? | `"true"` |
| `isSuperCall` | string | Calls super()? | `"true"` |
| `isThisCall` | string | Calls this.method()? | `"true"` |

**Calls relationship**:
```
rel_fact  type      "role_assertion"
rel_fact  subject   "function_1"
rel_fact  role      "calls"
rel_fact  object    "console.log"
```

### Assignment Predicates

For entities of type `js:Assignment`:

| Predicate | Type | Description | Example |
|-----------|------|-------------|---------|
| `target` | string | Assignment target | `"this.name"` |
| `value` | string | Assigned value | `"name"` |
| `atLine` | string | Line number | `"15"` |
| `operator` | string | Assignment operator | `"="` / `"+="` |
| `isAugmented` | string | Is augmented (+=)? | `"true"` / `"false"` |
| `isAttributeAssignment` | string | Is this.x = ...? | `"true"` |
| `attributeName` | string | Attribute name | `"name"` |
| `inFunction` | string | Containing function | `"constructor_1"` |

---

## Common Query Patterns

### Find All Classes

```python
result = query_select_reql(
    query="""
    SELECT ?class ?name
    WHERE {
        ?class concept "js:Class" .
        ?class name ?name
    }
    """,
    limit=1000
)
```

### Find All Methods in a Class

```python
result = query_select_reql(
    query="""
    SELECT ?method ?name
    WHERE {
        ?rel subject ?class .
        ?rel role "hasMethod" .
        ?rel object ?method .
        ?class concept "js:Class" .
        ?class name "MyClass" .
        ?method name ?name
    }
    """
)
```

### Find Async Functions

```python
result = query_select_reql(
    query="""
    SELECT ?func ?name
    WHERE {
        ?func concept "js:Function" .
        ?func isAsync "true" .
        ?func name ?name
    }
    """
)
```

### Find Functions With JSDoc

```python
result = query_select_reql(
    query="""
    SELECT ?func ?name ?doc
    WHERE {
        ?func concept "js:Function" .
        ?func hasDocstring ?doc .
        ?func name ?name
    }
    """
)
```

### Find Empty Catch Blocks

```python
result = query_select_reql(
    query="""
    SELECT ?catch ?line
    WHERE {
        ?catch concept "js:CatchClause" .
        ?catch bodyIsEmpty "true" .
        ?catch atLine ?line
    }
    """
)
```

### Find Class Inheritance

```python
result = query_select_reql(
    query="""
    SELECT ?child ?parent
    WHERE {
        ?rel subject ?child .
        ?rel role "inheritsFrom" .
        ?rel object ?parent
    }
    """
)
```

### Find Private Fields

```python
result = query_select_reql(
    query="""
    SELECT ?field ?name ?class
    WHERE {
        ?field concept "js:Field" .
        ?field isPrivate "true" .
        ?field name ?name .
        ?rel subject ?class .
        ?rel role "hasField" .
        ?rel object ?field
    }
    """
)
```

### Find Function Calls

```python
result = query_select_reql(
    query="""
    SELECT ?caller ?callee
    WHERE {
        ?rel role "calls" .
        ?rel subject ?caller .
        ?rel object ?callee
    }
    """
)
```

### Find Large Classes (> 100 lines)

```python
# Note: lineCount is a string, so comparison needs care
result = query_select_reql(
    query="""
    SELECT ?class ?name ?lines
    WHERE {
        ?class concept "js:Class" .
        ?class name ?name .
        ?class lineCount ?lines
    }
    """
)
# Filter in Python: int(lines) > 100
```

---

## JavaScript vs Python: Entity Comparison

| Python Entity | JavaScript Entity | Notes |
|---------------|-------------------|-------|
| `py:Module` | `js:Module` | Same concept |
| `py:Class` | `js:Class` | ES6 classes |
| `py:Function` | `js:Function` | Top-level functions |
| `py:Method` | `js:Method` | Class methods |
| - | `js:Constructor` | Explicit constructor type |
| - | `js:ArrowFunction` | JS-specific |
| `py:Parameter` | `js:Parameter` | Same concept |
| - | `js:Variable` | var/let/const |
| - | `js:Field` | Class fields |
| `py:Import` | `js:Import` | ES6 imports |
| - | `js:Export` | ES6 exports |
| `py:Assignment` | `js:Assignment` | Similar |
| - | `js:TryBlock` | Error handling |
| - | `js:CatchClause` | Error handling |
| - | `js:ThrowStatement` | Error handling |
| - | `js:ReturnStatement` | Control flow |
| - | `js:Call` | Function calls |

---

## Ontology-Based Reasoning

Load the JavaScript ontology for automatic inference:

```python
# Load the ontology
add_knowledge(
    source=open("js_ontology.reol").read(),
    type="ontology",
    source_id="js_ontology"
)

# Load JavaScript code
load_javascript_directory(
    directory="/path/to/your/js/code",
    recursive=True
)

# Query with inferred facts
result = query_select_reql(
    query="""
    SELECT ?class
    WHERE {
        ?class undocumented "true"
    }
    """
)
```

### Available Inference Rules

The JavaScript ontology (`js_ontology.reol`) provides:

**Transitive Relationships**:
- `inheritsFrom` transitivity
- `calls` / `callsTransitive`
- `imports` / `importsTransitive`

**Structural Patterns**:
- Method-Class relationships
- Inherited methods
- Field-Class relationships

**Code Quality Detection**:
- `undocumented` - Missing JSDoc
- `swallowedException` - Empty catch blocks
- `circularDependency` - Circular imports
- `largeClass` - Classes > 200 lines
- `largeFunction` - Functions > 50 lines
- `manyConstructorParams` - Constructor > 5 params

**Pattern Recognition**:
- `asyncFunction` / `asyncMethod` / `asyncArrow`
- `generatorFunction`
- `staticMethod` / `getterMethod` / `setterMethod`
- `privateMethod` / `privateField`
- `fluentMethod` - Returns `this`
- `constVariable` / `letVariable` / `varVariable`
- `defaultExport` / `namedExport`

---

## Complete Example

### Source Code:
```javascript
/**
 * Calculator class for basic math operations.
 * @class
 */
class Calculator {
    #history = [];

    constructor(name) {
        this.name = name;
    }

    /**
     * Add two numbers.
     * @param {number} a - First number
     * @param {number} b - Second number
     * @returns {number} Sum
     */
    add(a, b) {
        const result = a + b;
        this.#history.push(result);
        return result;
    }

    static create(name) {
        return new Calculator(name);
    }
}

async function fetchData(url) {
    try {
        const response = await fetch(url);
        return response.json();
    } catch (e) {
        throw new Error("Fetch failed");
    }
}

export default Calculator;
export { fetchData };
```

### Generated Facts (Selection):

**Module**:
```
module_0  concept       "js:Module"
module_0  name          "calculator"
```

**Class Calculator**:
```
class_1   concept       "js:Class"
class_1   name          "Calculator"
class_1   hasDocstring  "Calculator class for basic math operations.\n@class"
class_1   lineCount     "23"
class_1   hasPrivateFields "true"
```

**Private Field #history**:
```
field_2   concept       "js:Field"
field_2   name          "#history"
field_2   isPrivate     "true"
```

**Constructor**:
```
constructor_3  concept      "js:Constructor"
constructor_3  name         "constructor"
constructor_3  lineCount    "3"
```

**Method add**:
```
method_4   concept       "js:Method"
method_4   name          "add"
method_4   hasDocstring  "Add two numbers.\n@param {number} a..."
method_4   lineCount     "5"
```

**Static Method create**:
```
method_5   concept       "js:Method"
method_5   name          "create"
method_5   isStatic      "true"
```

**Async Function fetchData**:
```
function_6  concept      "js:Function"
function_6  name         "fetchData"
function_6  isAsync      "true"
function_6  lineCount    "8"
```

**TryBlock**:
```
try_7     concept        "js:TryBlock"
try_7     hasCatch       "true"
try_7     hasFinally     "false"
try_7     inFunction     "function_6"
```

**ThrowStatement**:
```
throw_8   concept        "js:ThrowStatement"
throw_8   expression     "new Error(\"Fetch failed\")"
throw_8   throwsErrorType "true"
throw_8   throwsNewError  "true"
```

**Exports**:
```
export_9   concept       "js:Export"
export_9   isDefault     "true"

export_10  concept       "js:Export"
export_10  isDefault     (not set - named export)
```

---

## Summary Statistics

For a typical JavaScript module with:
- 1 module
- 2 classes
- 5 methods
- 2 functions
- 10 parameters
- 5 imports
- 3 exports

**Approximate WMEs Generated**: ~150-200 triples

---

**Document Version**: 1.0
**Last Updated**: 2025-12-05
**JavaScript Parser**: ES6+ (ANTLR4)
**Ontology Support**: Full ontology-based reasoning with `js_ontology.reol`

**MCP Tools Available**:
- **JavaScript Loading**: `load_javascript_file`, `load_javascript_directory`, `load_javascript_code`
- **Queries**: `query_select_reql`, `quick_query`
- **Ontology**: `add_knowledge`
