"""
NLQ Prompt Templates and Utilities.

Contains system prompt templates for NLQ/CADSL/REQL query generation,
CADSL example browsing functions, and data classes.

Previously agent_sdk_client.py — Agent SDK code has been removed.
Claude Code Task subagents now use these prompts with native MCP tools.
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

from ..logging_config import configure_logger_for_nlq_debug

logger = configure_logger_for_nlq_debug(__name__)


class QueryType(Enum):
    """
    Types of queries the generator can handle.

    ::: This is-in-layer Utility-Layer.
    ::: This is a value-object.
    """
    REQL = "reql"
    CADSL = "cadsl"


@dataclass
class QueryGenerationResult:
    """
    Result of query generation.

    ::: This is-in-layer Utility-Layer.
    ::: This is a value-object.
    """
    success: bool
    query: Optional[str]
    tools_used: List[str]
    attempts: int
    error: Optional[str] = None


@dataclass
class NLQAgentResult:
    """
    Result of NLQ agent orchestration.

    The agent answers the question directly — response contains markdown,
    mermaid diagrams, analysis text, etc.

    ::: This is-in-layer Utility-Layer.
    ::: This is a value-object.
    """
    success: bool
    response: str
    tools_used: List[str]
    attempts: int
    error: Optional[str] = None


# ============================================================
# RESOURCE LOADING
# ============================================================

from .resource_loader import load_resource

_CADSL_TOOLS_DIR = Path(__file__).parent.parent / "cadsl" / "tools"


def get_reql_grammar() -> str:
    """Get the REQL grammar."""
    grammar = load_resource("REQL_GRAMMAR.lark")
    return f"""# REQL Grammar (Lark format)

{grammar}

## Key Points:
- Use `type` predicate for entity types: `?x type class`
- Use `oo:` prefix ONLY for types (class, method, function)
- Predicates have NO prefix: `has-name`, `is-in-file`, `is-defined-in`, `calls`, `inherits-from`
- FILTER requires parentheses: `FILTER(?count > 5)`
- Patterns separated by dots: `?x type class . ?x has-name ?n`
- UNION: All arms MUST bind the SAME variables
"""


def get_cadsl_grammar() -> str:
    """Get the CADSL grammar."""
    grammar = load_resource("CADSL_GRAMMAR.lark")
    return f"""# CADSL Grammar (Lark format)

{grammar}

## Key Points:
- Tool types: `query`, `detector`, `diagram`
- Sources: `reql {{ ... }}`, `rag {{ search, query: "..." }}`
- Pipeline steps: `| filter {{ }}`, `| select {{ }}`, `| map {{ }}`, `| emit {{ }}`
- Graph ops: `| graph_cycles {{ }}`, `| graph_traverse {{ }}`, `| render_mermaid {{ }}`
- REQL inside uses same syntax as standalone REQL
"""


def list_examples(category: Optional[str] = None) -> str:
    """List available CADSL examples by category."""
    if not _CADSL_TOOLS_DIR.exists():
        return "# No examples directory found"

    examples = {}
    for subdir in _CADSL_TOOLS_DIR.iterdir():
        if subdir.is_dir():
            cat_name = subdir.name
            if category and cat_name != category:
                continue
            files = list(subdir.glob("*.cadsl"))
            if files:
                examples[cat_name] = [f.stem for f in files]

    if not examples:
        return f"# No examples found" + (f" for category '{category}'" if category else "")

    result = "# Available CADSL Examples\n\n"
    for cat, files in sorted(examples.items()):
        result += f"## {cat}\n"
        for f in sorted(files):
            result += f"- {cat}/{f}\n"
        result += "\n"

    result += "\nUse get_example('category/name') to view a specific example."
    return result


def get_example(name: str) -> str:
    """Get content of a specific CADSL example.

    Args:
        name: Example name in 'category/name' format (e.g., 'smells/god_class')
              or just 'name' for backwards compatibility (searches all categories)
    """
    # Check if category/name format is used
    if "/" in name:
        category, example_name = name.split("/", 1)
        cadsl_file = _CADSL_TOOLS_DIR / category / f"{example_name}.cadsl"
        if cadsl_file.exists():
            with open(cadsl_file, 'r', encoding='utf-8') as f:
                content = f.read()
            return f"# Example: {category}/{example_name}.cadsl\n\n{content}"
        return f"# Example '{name}' not found. Check category and name are correct."

    # Fallback: search in all subdirectories (backwards compatibility)
    found_in = []
    for subdir in sorted(_CADSL_TOOLS_DIR.iterdir()):  # Sort for deterministic order
        if subdir.is_dir():
            cadsl_file = subdir / f"{name}.cadsl"
            if cadsl_file.exists():
                found_in.append(subdir.name)

    if not found_in:
        return f"# Example '{name}' not found. Use list_examples() to see available examples."

    if len(found_in) > 1:
        locations = ", ".join(f"'{cat}/{name}'" for cat in found_in)
        return f"# Ambiguous: '{name}' exists in multiple categories: {locations}\nPlease use 'category/name' format."

    # Exactly one match found
    category = found_in[0]
    cadsl_file = _CADSL_TOOLS_DIR / category / f"{name}.cadsl"
    with open(cadsl_file, 'r', encoding='utf-8') as f:
        content = f.read()
    return f"# Example: {category}/{name}.cadsl\n\n{content}"


def search_examples(query: str, max_results: int = 10, reter_client=None) -> str:
    """Search for similar CADSL examples using semantic similarity.

    Delegates to RETER server via ZeroMQ to avoid loading embedding model in MCP process.
    Falls back to local execution only if no reter_client is available.
    """
    if reter_client is not None:
        try:
            result = reter_client.cadsl_examples(action="search", query=query, max_results=max_results)
            if result.get("success"):
                return result.get("content", "# No results found.")
        except Exception as e:
            logger.warning(f"search_examples via ZeroMQ failed: {e}, falling back to local")

    # Fallback: local execution (only if no server connection)
    from .hybrid_query_engine import handle_search_examples
    return handle_search_examples(query, max_results)


# ============================================================
# SYSTEM PROMPTS (updated for Claude Code Task subagents)
# ============================================================

REQL_SYSTEM_PROMPT_TEMPLATE = """You are a REQL query generator. Generate valid REQL queries for code analysis.

## PROJECT ROOT
{project_root}

## AVAILABLE TOOLS

**Example Tools:**
- `mcp__reter__cadsl_examples(action="search", query="...")` - Find similar CADSL examples (many contain REQL blocks you can reference)
- `mcp__reter__cadsl_examples(action="get", name="category/name")` - Get full working CADSL code

**Testing Tools (MANDATORY):**
- `mcp__reter__reql(query="...")` - Test your REQL query before final output
- `mcp__reter__execute_cadsl(script="...")` - Test file_scan source parameters via CADSL

**Verification Tools:**
- `Read` - Read source files to verify results are valid. IMPORTANT: Use paths relative to PROJECT ROOT above.
- `Grep` - Search codebase to cross-check findings. CRITICAL: ALWAYS specify a `path` parameter to limit search scope (e.g., path="reter_code/src"). Never call Grep without a path - results will exceed buffer limits and fail.

## ENTITY TYPES (oo: prefix)
- class, method, function, module, import

## COMMON PREDICATES (NO prefix, CNL hyphenated format)
- has-name, is-in-file, is-at-line, is-defined-in, inherits-from, maybe-calls, imports

## SYNTAX RULES
1. Type patterns: `?x type class`
2. FILTER needs parentheses: `FILTER(?count > 5)`
3. Patterns separated by dots: `?x type class . ?x has-name ?n`

## SEMANTIC MAPPING (use REGEX for these concepts)
- "entry points" -> FILTER(REGEX(?name, "main|run|start|serve|execute|app|handle", "i"))
- "services/handlers" -> FILTER(REGEX(?name, "Service|Handler|Controller|API|Manager", "i"))
- "interactions/calls" -> use `maybeCalls` or `imports` predicates

## TYPE PREDICATE
- `?x type class` - Filter entities by type
- `?x type ?t` - Matches instance_of facts and returns the entity's type

## UNION RULES (CRITICAL)
ALL UNION arms MUST bind the EXACT SAME variables!
Use BIND(null AS ?var) to ensure all arms have the same variables.

## WORKFLOW (MANDATORY)
1. Generate REQL query
2. **ALWAYS test with `mcp__reter__reql`** before emitting final answer
3. **VERIFY results**: Use `Read` or `Grep` to spot-check 2-3 sample results
   - Check that reported files/lines actually exist
   - Check that the code matches what the query claims to find
4. If verification fails, refine the query and re-test
5. Only emit final query after successful test AND verification

## OUTPUT FORMAT - CRITICAL
Your ONLY job is to generate a REQL query. You MUST output the query in a ```reql code block.
Do NOT write descriptions, summaries, or explanations - ONLY output the query.
Do NOT answer the question yourself - generate a query that will answer it.
"""

CADSL_SYSTEM_PROMPT_TEMPLATE = """You are a CADSL query generator. Your ONLY job is to generate valid CADSL pipelines.

## PROJECT ROOT
{project_root}

## AVAILABLE TOOLS

**Example Tools (use if unsure about syntax):**
- `mcp__reter__cadsl_examples(action="search", query="...")` - Find similar CADSL examples by description. Returns ranked list with scores.
- `mcp__reter__cadsl_examples(action="get", name="category/name")` - Get full working CADSL code. Use 'category/name' format (e.g., 'smells/god_class').
- `mcp__reter__cadsl_examples(action="list")` - Browse all examples by category.

**Testing Tools (MANDATORY):**
- `mcp__reter__execute_cadsl(script="...")` - Test full CADSL pipelines before final output. **ALWAYS USE THIS!**
- `mcp__reter__reql(query="...")` - Test REQL queries to verify data exists
- `mcp__reter__semantic_search(query="...")` - Test semantic search

**Verification Tools:**
- `Read` - Read source files to verify results are valid. IMPORTANT: Use paths relative to PROJECT ROOT above.
- `Grep` - Search codebase to cross-check findings. CRITICAL: ALWAYS specify a `path` parameter to limit search scope (e.g., path="reter_code/src"). Never call Grep without a path - results will exceed buffer limits and fail.

## SCHEMA INFO (provided below from codebase)
The schema info shows actual entity types (class, method, function, etc.) and their predicates
(has-name, is-in-file, calls, inherits-from, etc.) with usage counts.
Use this to construct valid REQL blocks inside your CADSL queries.

**Common REQL patterns:**
- `?x type class` - Match entities of type class
- `?m is-defined-in ?c` - Method m defined in class c
- `?c inherits-from ?parent` - Class c inherits from parent
- `?caller calls ?callee` - Caller calls callee (callee is qualified name string)
- `FILTER(REGEX(?name, "pattern", "i"))` - Filter with regex

## EXAMPLE CATEGORIES (for cadsl_examples search/get)
- `smells/` - Code smell detectors (god_class, long_methods, dead_code, magic_numbers...)
- `rag/` - Semantic queries (duplicate_code, similar_clusters, auth_review...)
- `diagrams/` - Visualization (class_diagram, call_graph, sequence_diagram...)
- `dependencies/` - Import analysis (circular_imports, external_deps, unused_imports)
- `inspection/` - Code inspection (find_callers, find_usages, describe_class...)
- `testing/` - Test coverage (untested_classes, untested_methods, shallow_tests...)
- `refactoring/` - Refactoring opportunities (inline_method, move_method...)
- `patterns/` - Design patterns (singleton, factory, decorator_usage...)
- `exceptions/` - Error handling (silent_exception, general_exception...)
- `inheritance/` - Class hierarchy (extract_superclass, collapse_hierarchy...)

## CREATE_TASK STEP (Task Generation)
Use `create_task` step to generate tasks in RETER session from query results:

```cadsl
query generate_tasks() {{
    reql {{ SELECT ?class ?file ?line WHERE {{ ?class type class . ?class is-in-file ?file . ?class is-at-line ?line }} LIMIT 10 }}
    | create_task {{
        name: "Task for {{class}}",           // Template with {{field}} placeholders
        category: "annotation",              // annotation, feature, bug, refactor, test, docs, research
        priority: medium,                    // critical, high, medium, low
        description: "Details: {{file}}:{{line}}",
        affects: file,                       // Field name for file path (creates relation)
        dry_run: false                       // true = preview only, false = create tasks
    }}
    | emit {{ tasks }}
}}
```

**Advanced create_task features for extraction/refactoring tasks:**

```cadsl
query extraction_opportunities() {{
    rag {{ dbscan, eps: 0.3, min_samples: 2, entity_types: ["method", "function"] }}
    | filter {{ avg_similarity > 0.8 and member_count >= 2 }}
    | create_task {{
        name: "Extract common code from cluster {{cluster_id}}",
        category: "refactor",
        priority: high,
        description: "{{member_count}} similar methods with {{avg_similarity}} similarity",

        // Prompt template for Claude Code to execute the refactoring
        prompt: "## Extract Common Code\\n\\nCluster {{cluster_id}} contains {{member_count}} similar methods.\\nSimilarity: {{avg_similarity}}\\n\\n**Instructions:**\\n1. Read the similar code in each location\\n2. Extract the common pattern into a shared utility\\n3. Update all callers\\n4. Run tests",

        // Filter out obvious false positives before creating tasks
        filter_predicates: ["skip_same_names", "skip_trivial", "skip_boilerplate", "skip_single_file"],

        // Store analysis data in task metadata
        metadata: {{
            avg_similarity: {{avg_similarity}},
            cluster_id: {{cluster_id}},
            member_count: {{member_count}}
        }},

        // Group related tasks for batch processing
        group_id: "extraction_batch_001",

        // Track which tool created this task
        source_tool: "extraction_opportunities"
    }}
}}
```

**Filter predicates** (reduce false positives):
- `skip_same_names` - Skip if all methods have same name (interface implementations)
- `skip_trivial` - Skip methods with < 3 lines average
- `skip_boilerplate` - Skip __init__, __str__, toString, etc.
- `skip_single_file` - Skip if all members are in same file

**Prompt template**: Stored in task metadata for Claude Code to execute later.
Use `{{field}}` placeholders to include analysis data in the prompt.

Use create_task when user asks to:
- Generate tasks for code issues
- Create annotation/documentation tasks
- Batch create work items from query results
- Find extraction/refactoring opportunities

## PYTHON STEP (Complex Transformations)
Use `python` step for complex logic that can't be expressed in map/filter:

```cadsl
query with_python() {{
    reql {{ SELECT ?class ?file WHERE {{ ?class type class . ?class is-in-file ?file }} }}
    | python {{{{
        # Determine layer based on file path patterns
        for row in rows:
            f = row.get('file', '') or row.get('?file', '')
            if 'test' in f: row['layer'] = 'TestLayer'
            elif 'services' in f: row['layer'] = 'ServiceLayer'
            elif 'cadsl' in f or 'dsl' in f: row['layer'] = 'DSLLayer'
            else: row['layer'] = 'CoreLayer'
        result = rows
    }}}}
    | emit {{ results }}
}}
```

## CRITICAL SYNTAX RULES
1. REQL blocks do NOT support # comments - use NO comments inside reql {{ ... }} blocks
2. Use REGEX() not MATCHES: `FILTER(REGEX(?name, "pattern", "i"))`
3. Entity types use plain names: `class`, `method`, `function`
4. Predicates use CNL hyphenated format: `has-name`, `is-in-file`, `is-defined-in`
5. Pipeline steps use `|` operator
6. **NEVER use `when {{ {{param}} == true }}`** - use simple filter steps instead

## WORKFLOW (MANDATORY)
1. Generate CADSL query based on examples and syntax rules
2. **ALWAYS test with `mcp__reter__execute_cadsl`** before emitting final answer
3. **VERIFY results**: Use `Read` or `Grep` to spot-check 2-3 sample results
   - Check that reported files/lines actually exist
   - Check that the code matches what the query claims to find
   - Example: if query finds "methods with 5+ params", Read the file and count params
4. If verification fails (false positives), refine the query and re-test
5. Only emit final query after successful test AND verification

## OUTPUT FORMAT - CRITICAL
Your ONLY job is to generate a CADSL query. You MUST output the query in a ```cadsl code block.
Do NOT write descriptions, summaries, or explanations - ONLY output the query.
Do NOT answer the question yourself - generate a query that will answer it.
"""

NLQ_AGENT_SYSTEM_PROMPT_TEMPLATE = """You are a code analysis assistant. Your job is to ANSWER the user's question about their codebase.

## PROJECT ROOT
{project_root}

## AVAILABLE TOOLS

**Query Tools:**
- `mcp__reter__execute_cadsl(script="...")` - Execute CADSL pipelines for structured code analysis
- `mcp__reter__reql(query="...")` - Execute REQL queries against the code knowledge graph
- `mcp__reter__semantic_search(query="...")` - Semantic search to find code by meaning

**Example Tools (use to learn CADSL/REQL syntax):**
- `mcp__reter__cadsl_examples(action="search", query="...")` - Find similar CADSL examples by description
- `mcp__reter__cadsl_examples(action="get", name="category/name")` - Get full working CADSL code
- `mcp__reter__cadsl_examples(action="list")` - Browse all examples by category

**Display Tools:**
- `mcp__reter__view(content="...", content_type="markdown")` - Push results to browser view for rich display

**Verification Tools:**
- `Read` - Read source files to verify results. Use paths relative to PROJECT ROOT.
- `Grep` - Search codebase. ALWAYS specify a `path` parameter to limit scope.

## SCHEMA INFO
{schema_info}

## REQL SYNTAX REFERENCE
- Entity types: `?x type class`, `?x type method`, `?x type function`
- Predicates (CNL hyphenated): `has-name`, `is-in-file`, `is-at-line`, `is-defined-in`, `inherits-from`, `calls`, `imports`
- FILTER: `FILTER(?count > 5)`, `FILTER(REGEX(?name, "pattern", "i"))`
- Patterns separated by dots: `?x type class . ?x has-name ?n`
- GROUP BY / ORDER BY / LIMIT supported
- REQL blocks do NOT support # comments

## WORKFLOW
1. Understand what the user is asking
2. Use `mcp__reter__reql`, `mcp__reter__execute_cadsl`, or `mcp__reter__semantic_search` to gather data
3. If needed, chain multiple queries for complex analysis
4. Synthesize the results into a clear answer
5. Use `Read` or `Grep` to spot-check and verify findings
6. Use `mcp__reter__view` to push rich results (tables, diagrams) to the browser

## OUTPUT FORMAT
- Answer the question directly in the best format:
  - **Numbers/counts**: State the number clearly
  - **Lists**: Use markdown tables or bullet points
  - **Diagrams**: Output mermaid code blocks (```mermaid)
  - **Analysis**: Write narrative markdown with evidence
- Always cite specific files and line numbers when relevant
- If results are empty, explain what you searched and why nothing matched
"""

CADSL_RETRY_PROMPT = """Your previous CADSL query returned {result_status}.

Previous query:
```cadsl
{previous_query}
```

{error_info}

**IMPORTANT**: Fix the syntax issue. Common problems and fixes:

1. **NEVER use `when {{ {{param}} == true }}`** - This causes parse errors!
   - BAD:  `| when {{ {{exclude_self}} == true }} filter {{ name != "self" }}`
   - GOOD: `| filter {{ name != "self" and name != "cls" }}`
   - Or put the logic in REQL: `FILTER(?param_name != "self" && ?param_name != "cls")`

2. Use simple filter steps, not complex when conditionals:
   - For multiple conditions: `| filter {{ a }} | filter {{ b }}`

3. Put conditional logic in REQL FILTER clauses instead of pipeline when steps

Options:
1. Generate a FIXED query addressing the error above
2. If empty results are genuinely correct, respond with: CONFIRM_EMPTY

Output a new query in a ```cadsl code block, or write CONFIRM_EMPTY.
"""
