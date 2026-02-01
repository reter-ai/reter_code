"""
Agent SDK Client for REQL/CADSL Query Generation.

This module provides a Claude Agent SDK-based query generator where:
1. We call Agent SDK with a prompt
2. Agent tells us what it needs (grammar, examples) via text
3. We provide resources and Agent generates a query
4. We execute and validate the query
5. If errors, we feed back and Agent retries

No MCP tools - just text-based orchestration.
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

from ..logging_config import nlq_debug_logger as debug_log

# Check if Agent SDK is available
_agent_sdk_available = None


def is_agent_sdk_available() -> bool:
    """Check if Claude Agent SDK is available."""
    global _agent_sdk_available
    if _agent_sdk_available is None:
        try:
            from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions
            _agent_sdk_available = True
        except ImportError:
            _agent_sdk_available = False
    return _agent_sdk_available


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


# ============================================================
# RESOURCE LOADING
# ============================================================

_RESOURCES_DIR = Path(__file__).parent.parent / "resources"
_CADSL_TOOLS_DIR = Path(__file__).parent.parent / "cadsl" / "tools"


def _load_resource(filename: str) -> str:
    """Load a resource file from the resources directory."""
    resource_path = _RESOURCES_DIR / filename
    if resource_path.exists():
        with open(resource_path, 'r', encoding='utf-8') as f:
            return f.read()
    return f"# Resource file not found: {filename}"


def get_reql_grammar() -> str:
    """Get the REQL grammar."""
    grammar = _load_resource("REQL_GRAMMAR.lark")
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
    grammar = _load_resource("CADSL_GRAMMAR.lark")
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


def search_examples(query: str, max_results: int = 10) -> str:
    """Search for similar CADSL examples using semantic similarity."""
    from .hybrid_query_engine import handle_search_examples
    return handle_search_examples(query, max_results)


# ============================================================
# SYSTEM PROMPTS
# ============================================================

REQL_SYSTEM_PROMPT_TEMPLATE = """You are a REQL query generator. Generate valid REQL queries for code analysis.

## PROJECT ROOT
{project_root}

## AVAILABLE TOOLS

**Example Tools:**
- `search_examples` - Find similar CADSL examples (many contain REQL blocks you can reference)
- `get_example` - Get full code. Use 'category/name' format (e.g., 'smells/god_class')

**Testing Tools (MANDATORY):**
- `run_reql` - Test your REQL query before final output
- `run_file_scan` - Test file_scan source parameters (glob, contains, exclude, include_matches, context_lines, limit)

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

## TYPE vs CONCEPT
- `?x type class` - Filter with subsumption (matches class, class, etc.)
- `?x type ?t` - Returns ALL types (asserted + inferred) - MULTIPLE rows per entity
- `?x concept ?t` - Returns ONLY asserted type - ONE row (e.g., "method")

## UNION RULES (CRITICAL)
ALL UNION arms MUST bind the EXACT SAME variables!
Use BIND(null AS ?var) to ensure all arms have the same variables.

## WORKFLOW (MANDATORY)
1. Generate REQL query
2. **ALWAYS test with `run_reql`** before emitting final answer
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
- `search_examples` - Find similar CADSL examples by description. Returns ranked list with scores.
- `get_example` - Get full working CADSL code. Use 'category/name' format (e.g., 'smells/god_class').
- `list_examples` - Browse all examples by category.

**Testing Tools (MANDATORY):**
- `run_cadsl` - Test full CADSL pipelines before final output. **ALWAYS USE THIS!**
- `run_reql` - Test REQL queries to verify data exists
- `run_file_scan` - Test file_scan source parameters (glob, contains, exclude, include_matches, context_lines, limit)
- `run_rag_search` - Test semantic search
- `run_rag_duplicates` - Test duplicate detection
- `run_rag_clusters` - Test clustering

**Verification Tools:**
- `Read` - Read source files to verify results are valid. IMPORTANT: Use paths relative to PROJECT ROOT above.
- `Grep` - Search codebase to cross-check findings. CRITICAL: ALWAYS specify a `path` parameter to limit search scope (e.g., path="reter_code/src"). Never call Grep without a path - results will exceed buffer limits and fail.

## EXAMPLE CATEGORIES (for search_examples / get_example)
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

Use create_task when user asks to:
- Generate tasks for code issues
- Create annotation/documentation tasks
- Batch create work items from query results

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
2. **ALWAYS test with `run_cadsl`** before emitting final answer
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

CLASSIFICATION_SYSTEM_PROMPT = """You classify code analysis questions into query types.

Respond with JSON only:
{
  "type": "reql" | "cadsl" | "rag",
  "confidence": 0.0-1.0,
  "reasoning": "brief explanation"
}

Classification rules:
- REQL: Simple structural queries (find classes, methods, inheritance, calls)
- CADSL: Complex pipelines, graph algorithms, diagrams, code smells
- RAG: Semantic similarity, duplicate detection, "find similar code"
"""


# ============================================================
# AGENT SDK ORCHESTRATOR
# ============================================================

def _parse_requests(text: str) -> List[Dict[str, str]]:
    """Parse REQUEST_ commands from agent output."""
    requests = []

    # REQUEST_GRAMMAR: reql/cadsl
    for match in re.finditer(r'REQUEST_GRAMMAR:\s*(\w+)', text, re.IGNORECASE):
        requests.append({"type": "grammar", "value": match.group(1).lower()})

    # REQUEST_EXAMPLE: name
    for match in re.finditer(r'REQUEST_EXAMPLE:\s*(\w+)', text, re.IGNORECASE):
        requests.append({"type": "example", "value": match.group(1)})

    # REQUEST_EXAMPLES_LIST or REQUEST_EXAMPLES_LIST: category
    for match in re.finditer(r'REQUEST_EXAMPLES_LIST(?::\s*(\w+))?', text, re.IGNORECASE):
        category = match.group(1) if match.group(1) else None
        requests.append({"type": "examples_list", "value": category})

    return requests


def _extract_query(text: str, query_type: QueryType) -> Optional[str]:
    """Extract query from code block."""
    lang = "cadsl" if query_type == QueryType.CADSL else "reql|sparql|sql"

    # Try to find query in code blocks
    code_block_match = re.search(
        rf'```(?:{lang})?\s*\n?(.*?)\n?```',
        text,
        re.DOTALL | re.IGNORECASE
    )
    if code_block_match:
        query = code_block_match.group(1).strip()
        if query:
            return query

    return None


def _handle_requests(requests: List[Dict[str, str]]) -> str:
    """Handle resource requests and return combined response."""
    responses = []

    for req in requests:
        if req["type"] == "grammar":
            if req["value"] == "reql":
                responses.append(get_reql_grammar())
            elif req["value"] == "cadsl":
                responses.append(get_cadsl_grammar())
        elif req["type"] == "example":
            responses.append(get_example(req["value"]))
        elif req["type"] == "examples_list":
            responses.append(list_examples(req["value"]))

    return "\n\n".join(responses)


def _create_query_tools(reter_instance=None, rag_manager=None):
    """Create custom MCP tools for query generation assistance."""
    from claude_agent_sdk import tool, create_sdk_mcp_server

    @tool("get_grammar", "Get the formal grammar specification for REQL or CADSL. Use language='reql' for structural queries or language='cadsl' for pipelines.", {"language": str})
    async def get_grammar_tool(args):
        language = args.get("language", "").lower()
        if language == "reql":
            content = get_reql_grammar()
        elif language == "cadsl":
            content = get_cadsl_grammar()
        else:
            content = f"Unknown language: {language}. Use 'reql' or 'cadsl'."
        return {"content": [{"type": "text", "text": content}]}

    @tool("list_examples", "List all CADSL example files organized by category. Categories: smells (code smells), rag (semantic queries), diagrams, dependencies, inheritance, inspection, patterns, refactoring, testing, exceptions. Use category param to filter.", {"category": str})
    async def list_examples_tool(args):
        category = args.get("category") or None
        content = list_examples(category)
        return {"content": [{"type": "text", "text": content}]}

    @tool("get_example", "IMPORTANT: Get full working CADSL code for a specific example. Use 'category/name' format (e.g., 'smells/god_class', 'rag/duplicate_code'). These are production-ready templates you can adapt.", {"name": str})
    async def get_example_tool(args):
        name = args.get("name", "")
        content = get_example(name)
        return {"content": [{"type": "text", "text": content}]}

    @tool("search_examples", "Search for CADSL examples similar to your question using semantic similarity. Returns top 3 matches with FULL CADSL code (use as templates), plus metadata for other matches. This is your primary tool for finding working examples to adapt.", {"query": str, "max_results": int})
    async def search_examples_tool(args):
        query = args.get("query", "")
        max_results = args.get("max_results", 10)
        content = search_examples(query, max_results)
        return {"content": [{"type": "text", "text": content}]}

    @tool("run_reql", "Execute a REQL query and return results (use to test queries)", {"query": str, "limit": int})
    async def run_reql_tool(args):
        debug_log.debug(f"run_reql called with args: {args}")
        if reter_instance is None:
            debug_log.debug("run_reql: RETER instance not available")
            return {"content": [{"type": "text", "text": "Error: RETER instance not available"}], "is_error": True}

        query = args.get("query", "")
        limit = args.get("limit", 10)

        try:
            debug_log.debug(f"run_reql executing: {query[:200]}...")
            result = reter_instance.reql(query)
            rows = result.to_pylist()[:limit]
            row_count = result.num_rows
            debug_log.debug(f"run_reql result: {row_count} rows")

            content = f"Query executed successfully. {row_count} total rows.\n\nFirst {min(limit, row_count)} results:\n"
            for i, row in enumerate(rows):
                content += f"{i+1}. {row}\n"

            return {"content": [{"type": "text", "text": content}]}
        except Exception as e:
            debug_log.debug(f"run_reql error: {e}")
            return {"content": [{"type": "text", "text": f"Query error: {str(e)}"}], "is_error": True}

    @tool("run_rag_search", "Semantic search - find code/docs by meaning (query: str, top_k: int, entity_types: list)", {"query": str, "top_k": int, "entity_types": str})
    async def run_rag_search_tool(args):
        debug_log.debug(f"run_rag_search called with args: {args}")
        if rag_manager is None:
            debug_log.debug("run_rag_search: RAG manager not available")
            return {"content": [{"type": "text", "text": "Error: RAG manager not available"}], "is_error": True}

        query = args.get("query", "")
        top_k = args.get("top_k", 10)
        entity_types_str = args.get("entity_types", "")
        entity_types = [t.strip() for t in entity_types_str.split(",")] if entity_types_str else None

        try:
            debug_log.debug(f"run_rag_search searching: '{query}' (top_k={top_k}, types={entity_types})")
            results, stats = rag_manager.search(
                query=query,
                top_k=top_k,
                entity_types=entity_types
            )
            debug_log.debug(f"run_rag_search result: {len(results)} matches, stats: {stats}")

            content = f"Semantic search for: '{query}'\nFound {len(results)} results:\n\n"
            for i, r in enumerate(results[:top_k]):
                score = getattr(r, 'score', 0)
                name = getattr(r, 'name', getattr(r, 'qualified_name', 'unknown'))
                entity_type = getattr(r, 'entity_type', 'unknown')
                file = getattr(r, 'file', '')
                line = getattr(r, 'line', '')
                content += f"{i+1}. [{score:.3f}] {entity_type}: {name}\n   File: {file}:{line}\n"

            return {"content": [{"type": "text", "text": content}]}
        except Exception as e:
            debug_log.debug(f"run_rag_search error: {e}")
            return {"content": [{"type": "text", "text": f"RAG search error: {str(e)}"}], "is_error": True}

    @tool("run_rag_duplicates", "Find duplicate code pairs using semantic similarity (similarity: float 0-1, limit: int, entity_types: str)", {"similarity": float, "limit": int, "entity_types": str})
    async def run_rag_duplicates_tool(args):
        debug_log.debug(f"run_rag_duplicates called with args: {args}")
        if rag_manager is None:
            debug_log.debug("run_rag_duplicates: RAG manager not available")
            return {"content": [{"type": "text", "text": "Error: RAG manager not available"}], "is_error": True}

        similarity = args.get("similarity", 0.85)
        limit = args.get("limit", 50)
        entity_types_str = args.get("entity_types", "")
        entity_types = [t.strip() for t in entity_types_str.split(",")] if entity_types_str else ["method", "function"]

        try:
            debug_log.debug(f"run_rag_duplicates: similarity={similarity}, limit={limit}, types={entity_types}")
            result = rag_manager.find_duplicate_candidates(
                similarity_threshold=similarity,
                max_results=limit,
                exclude_same_file=True,
                exclude_same_class=True,
                entity_types=entity_types
            )
            debug_log.debug(f"run_rag_duplicates result: {result.get('total_pairs', 0)} pairs found")

            if not result.get("success"):
                return {"content": [{"type": "text", "text": f"Error: {result.get('error', 'Unknown error')}"}], "is_error": True}

            pairs = result.get("pairs", [])
            content = f"Found {len(pairs)} duplicate code pairs (similarity >= {similarity}):\n\n"
            for i, pair in enumerate(pairs[:limit]):
                e1, e2 = pair.get("entity1", {}), pair.get("entity2", {})
                sim = pair.get("similarity", 0)
                content += f"{i+1}. [{sim:.3f}] {e1.get('name', '?')} ({e1.get('file', '?')}:{e1.get('line', '?')})\n"
                content += f"         ↔ {e2.get('name', '?')} ({e2.get('file', '?')}:{e2.get('line', '?')})\n"

            return {"content": [{"type": "text", "text": content}]}
        except Exception as e:
            debug_log.debug(f"run_rag_duplicates error: {e}")
            return {"content": [{"type": "text", "text": f"RAG duplicates error: {str(e)}"}], "is_error": True}

    @tool("run_rag_clusters", "Find clusters of semantically similar code using K-means (n_clusters: int, min_size: int, entity_types: str)", {"n_clusters": int, "min_size": int, "entity_types": str})
    async def run_rag_clusters_tool(args):
        debug_log.debug(f"run_rag_clusters called with args: {args}")
        if rag_manager is None:
            debug_log.debug("run_rag_clusters: RAG manager not available")
            return {"content": [{"type": "text", "text": "Error: RAG manager not available"}], "is_error": True}

        n_clusters = args.get("n_clusters", 50)
        min_size = args.get("min_size", 2)
        entity_types_str = args.get("entity_types", "")
        entity_types = [t.strip() for t in entity_types_str.split(",")] if entity_types_str else ["method", "function"]

        try:
            debug_log.debug(f"run_rag_clusters: n_clusters={n_clusters}, min_size={min_size}, types={entity_types}")
            result = rag_manager.find_similar_clusters(
                n_clusters=n_clusters,
                min_cluster_size=min_size,
                exclude_same_file=True,
                exclude_same_class=True,
                entity_types=entity_types
            )
            debug_log.debug(f"run_rag_clusters result: {result.get('total_clusters', 0)} clusters found")

            if not result.get("success"):
                return {"content": [{"type": "text", "text": f"Error: {result.get('error', 'Unknown error')}"}], "is_error": True}

            clusters = result.get("clusters", [])
            content = f"Found {len(clusters)} code clusters (min_size >= {min_size}):\n\n"
            for cluster in clusters[:20]:  # Show first 20 clusters
                cid = cluster.get("cluster_id", "?")
                count = cluster.get("member_count", 0)
                files = cluster.get("unique_files", 0)
                members = cluster.get("members", [])
                content += f"Cluster {cid}: {count} members across {files} files\n"
                for m in members[:5]:  # Show first 5 members
                    content += f"  - {m.get('name', '?')} ({m.get('file', '?')}:{m.get('line', '?')})\n"
                if len(members) > 5:
                    content += f"  ... and {len(members) - 5} more\n"
                content += "\n"

            return {"content": [{"type": "text", "text": content}]}
        except Exception as e:
            debug_log.debug(f"run_rag_clusters error: {e}")
            return {"content": [{"type": "text", "text": f"RAG clusters error: {str(e)}"}], "is_error": True}

    @tool("run_cadsl", "Execute a CADSL script and return results. Use to test CADSL pipelines before final output. Returns parsed results or error messages.", {"script": str, "limit": int})
    async def run_cadsl_tool(args):
        """Execute a CADSL script and return results."""
        debug_log.debug(f"run_cadsl called with args: {args}")

        script = args.get("script", "")
        limit = args.get("limit", 20)

        if not script.strip():
            return {"content": [{"type": "text", "text": "Error: Empty CADSL script"}], "is_error": True}

        try:
            # Import CADSL components
            from ..cadsl.parser import parse_cadsl
            from ..cadsl.transformer import CADSLTransformer
            from ..cadsl.loader import build_pipeline_factory
            from ..dsl.core import Context as PipelineContext
            from ..dsl.catpy import Ok, Err

            debug_log.debug(f"run_cadsl parsing: {script[:200]}...")

            # Parse CADSL
            parse_result = parse_cadsl(script)
            if not parse_result.success:
                error_msg = f"Parse error: {parse_result.errors}"
                debug_log.debug(f"run_cadsl parse error: {error_msg}")
                return {"content": [{"type": "text", "text": error_msg}], "is_error": True}

            # Transform AST to tool spec
            transformer = CADSLTransformer()
            tool_specs = transformer.transform(parse_result.tree)

            if not tool_specs:
                return {"content": [{"type": "text", "text": "Error: No tool spec generated from CADSL"}], "is_error": True}

            tool_spec = tool_specs[0]
            debug_log.debug(f"run_cadsl tool_spec: {tool_spec.name}")

            # Build pipeline context
            params = {"rag_manager": rag_manager}
            for param in tool_spec.params:
                if param.default is not None:
                    params[param.name] = param.default

            pipeline_ctx = PipelineContext(reter=reter_instance, params=params)

            # Build and execute pipeline
            pipeline_factory = build_pipeline_factory(tool_spec)
            pipeline = pipeline_factory(pipeline_ctx)

            pipeline_result = pipeline.execute(pipeline_ctx)

            # Handle Result monad
            if isinstance(pipeline_result, Err):
                error_msg = f"Pipeline error: {pipeline_result.value}"
                debug_log.debug(f"run_cadsl pipeline error: {error_msg}")
                return {"content": [{"type": "text", "text": error_msg}], "is_error": True}

            if isinstance(pipeline_result, Ok):
                pipeline_result = pipeline_result.value

            # Format results
            if isinstance(pipeline_result, dict):
                results = pipeline_result.get("results", pipeline_result.get("findings", []))
                count = pipeline_result.get("count", len(results) if isinstance(results, list) else 0)
            elif isinstance(pipeline_result, list):
                results = pipeline_result
                count = len(results)
            else:
                results = [pipeline_result]
                count = 1

            debug_log.debug(f"run_cadsl result: {count} items")

            # Build output
            content = f"CADSL executed successfully. {count} total results.\n\n"

            if isinstance(results, list):
                content += f"First {min(limit, len(results))} results:\n"
                for i, row in enumerate(results[:limit]):
                    if isinstance(row, dict):
                        # Format dict nicely
                        row_str = ", ".join(f"{k}={v}" for k, v in list(row.items())[:5])
                        if len(row) > 5:
                            row_str += f", ... ({len(row)} fields)"
                    else:
                        row_str = str(row)
                    content += f"{i+1}. {row_str}\n"

                if len(results) > limit:
                    content += f"\n... and {len(results) - limit} more results"
            else:
                content += f"Result: {results}"

            return {"content": [{"type": "text", "text": content}]}

        except Exception as e:
            error_msg = f"CADSL execution error: {str(e)}"
            debug_log.debug(f"run_cadsl error: {e}", exc_info=True)
            return {"content": [{"type": "text", "text": error_msg}], "is_error": True}

    @tool("run_file_scan", "Test file_scan CADSL source block. Scans RETER-tracked sources with glob/content patterns. Use to verify file_scan parameters before using in CADSL pipelines.", {"glob": str, "contains": str, "exclude": str, "include_matches": bool, "context_lines": int, "limit": int})
    async def run_file_scan_tool(args):
        """Execute a file scan over RETER sources and return results."""
        debug_log.debug(f"run_file_scan called with args: {args}")

        if reter_instance is None:
            debug_log.debug("run_file_scan: RETER instance not available")
            return {"content": [{"type": "text", "text": "Error: RETER instance not available"}], "is_error": True}

        # Extract parameters
        glob_pattern = args.get("glob", "*")
        contains = args.get("contains") or None
        exclude_str = args.get("exclude", "")
        exclude = [p.strip() for p in exclude_str.split(",")] if exclude_str else None
        include_matches = args.get("include_matches", False)
        context_lines = args.get("context_lines", 0)
        limit = args.get("limit", 50)

        try:
            # Import and create FileScanSource
            from ..dsl.core import FileScanSource, Context as PipelineContext
            from ..dsl.catpy import Ok, Err

            debug_log.debug(f"run_file_scan: glob={glob_pattern}, contains={contains}, exclude={exclude}")

            # Create the source
            source = FileScanSource(
                glob=glob_pattern,
                exclude=exclude,
                contains=contains,
                case_sensitive=False,  # Case-insensitive by default for usability
                include_matches=include_matches,
                context_lines=context_lines,
                include_stats=True
            )

            # Build context and execute
            pipeline_ctx = PipelineContext(reter=reter_instance, params={})
            result = source.execute(pipeline_ctx)

            # Handle Result monad
            if isinstance(result, Err):
                error_msg = f"File scan error: {result.value}"
                debug_log.debug(f"run_file_scan error: {error_msg}")
                return {"content": [{"type": "text", "text": error_msg}], "is_error": True}

            if isinstance(result, Ok):
                files = result.value
            else:
                files = result

            debug_log.debug(f"run_file_scan result: {len(files)} files")

            # Build output
            total_files = len(files)
            files = files[:limit]

            content = f"File scan completed. {total_files} files matched.\n\n"

            if contains:
                content += f"Pattern: '{contains}'\n"
            content += f"Glob: '{glob_pattern}'\n\n"

            content += f"Showing {len(files)} of {total_files} files:\n\n"

            for i, f in enumerate(files):
                file_path = f.get("file", "?")
                line_count = f.get("line_count", "?")
                match_count = f.get("match_count", 0)

                if contains:
                    content += f"{i+1}. {file_path} ({line_count} lines, {match_count} matches)\n"
                else:
                    content += f"{i+1}. {file_path} ({line_count} lines)\n"

                # Show matches if requested
                if include_matches and "matches" in f:
                    for m in f["matches"][:3]:  # Show first 3 matches
                        line_num = m.get("line_number", "?")
                        line_content = m.get("content", "").strip()[:80]
                        content += f"   L{line_num}: {line_content}\n"
                    if len(f["matches"]) > 3:
                        content += f"   ... and {len(f['matches']) - 3} more matches\n"

            if total_files > limit:
                content += f"\n... and {total_files - limit} more files"

            return {"content": [{"type": "text", "text": content}]}

        except Exception as e:
            error_msg = f"File scan error: {str(e)}"
            debug_log.debug(f"run_file_scan error: {e}", exc_info=True)
            return {"content": [{"type": "text", "text": error_msg}], "is_error": True}

    tools = [get_grammar_tool, list_examples_tool, search_examples_tool, get_example_tool]
    if reter_instance is not None:
        tools.append(run_reql_tool)
        tools.append(run_file_scan_tool)
    if rag_manager is not None:
        tools.append(run_rag_search_tool)
        tools.append(run_rag_duplicates_tool)
        tools.append(run_rag_clusters_tool)
    # CADSL needs both reter and rag for full functionality
    if reter_instance is not None and rag_manager is not None:
        tools.append(run_cadsl_tool)

    return create_sdk_mcp_server(
        name="query_helpers",
        version="1.0.0",
        tools=tools
    )


def _build_agent_options(system_prompt: str, max_turns: int, reter_instance, rag_manager):
    """Build ClaudeAgentOptions with query helper tools."""
    from claude_agent_sdk import ClaudeAgentOptions

    # Create custom MCP server with query helper tools
    query_tools_server = _create_query_tools(reter_instance, rag_manager)

    # Configure tools - include both built-in and custom tools
    base_tools = [
        "mcp__query_helpers__get_grammar",
        "mcp__query_helpers__list_examples",
        "mcp__query_helpers__search_examples",
        "mcp__query_helpers__get_example"
    ]
    if reter_instance is not None:
        base_tools.append("mcp__query_helpers__run_reql")
        base_tools.append("mcp__query_helpers__run_file_scan")
    if rag_manager is not None:
        base_tools.append("mcp__query_helpers__run_rag_search")
        base_tools.append("mcp__query_helpers__run_rag_duplicates")
        base_tools.append("mcp__query_helpers__run_rag_clusters")
    # CADSL needs both reter and rag for full functionality
    if reter_instance is not None and rag_manager is not None:
        base_tools.append("mcp__query_helpers__run_cadsl")

    # Allow custom MCP tools plus Read and Grep for code exploration
    allowed_tools = base_tools + ["Read", "Grep"]
    disallowed_tools = ["Glob", "Bash", "Edit", "Write", "WebSearch", "WebFetch"]
    permission_mode = "bypassPermissions"

    options = ClaudeAgentOptions(
        system_prompt=system_prompt,
        allowed_tools=allowed_tools,
        disallowed_tools=disallowed_tools,
        permission_mode=permission_mode,
        max_turns=max_turns,
        mcp_servers={"query_helpers": query_tools_server}
    )

    debug_log.debug(f"Built agent options with allowed_tools: {allowed_tools}")
    return options


async def _process_agent_response(client, tools_used: List[str]) -> str:
    """Process agent response and return the final text output."""
    from claude_agent_sdk import AssistantMessage, TextBlock, ToolUseBlock, ToolResultBlock

    result_text = ""
    async for message in client.receive_response():
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    result_text = block.text
                elif isinstance(block, ToolUseBlock):
                    tool_name = block.name
                    tool_input = block.input
                    tools_used.append(tool_name)
                    debug_log.debug(f"TOOL CALL: {tool_name} with input: {str(tool_input)[:200]}")
                elif isinstance(block, ToolResultBlock):
                    result_preview = str(block.content)[:200] if block.content else "(empty)"
                    debug_log.debug(f"TOOL RESULT: {result_preview}...")

    return result_text


async def _call_agent(prompt: str, system_prompt: str, max_turns: int = 15, reter_instance=None, rag_manager=None) -> str:
    """Call Agent SDK using ClaudeSDKClient and return the text response.

    NOTE: This creates a new session. For multi-turn with retries, use the session-based
    generate_reql_query or generate_cadsl_query functions instead.

    Args:
        prompt: The prompt to send to the agent
        system_prompt: System prompt for the agent
        max_turns: Maximum conversation turns (default 15)
        reter_instance: Optional RETER instance for running test REQL queries
        rag_manager: Optional RAG manager for running semantic search
    """
    if not is_agent_sdk_available():
        raise ImportError("Claude Agent SDK not installed. Run: pip install claude-agent-sdk")

    from claude_agent_sdk import ClaudeSDKClient

    tools_used = []
    options = _build_agent_options(system_prompt, max_turns, reter_instance, rag_manager)

    debug_log.debug(f"Starting single-turn agent session")

    async with ClaudeSDKClient(options=options) as client:
        await client.query(prompt)
        result_text = await _process_agent_response(client, tools_used)

    debug_log.debug(f"Agent finished. Tools used: {tools_used}")
    return result_text


async def generate_reql_query(
    question: str,
    schema_info: str,
    reter_instance,
    max_iterations: int = 5,
    similar_tools_context: Optional[str] = None,
    rag_manager=None,
    project_root: Optional[str] = None
) -> QueryGenerationResult:
    """
    Generate and validate a REQL query using Agent SDK.

    Uses a SINGLE agentic session for all retries, maintaining conversation context.

    Orchestration loop (within one session):
    1. Send question to Agent
    2. Agent uses tools to explore grammars/examples
    3. If Agent outputs query, execute and validate
    4. If error, send follow-up message with error feedback
    5. Repeat until success or max iterations
    """
    if not is_agent_sdk_available():
        return QueryGenerationResult(
            success=False,
            query=None,
            tools_used=[],
            attempts=0,
            error="Claude Agent SDK not available"
        )

    from claude_agent_sdk import ClaudeSDKClient

    tools_used = []
    attempts = 0
    last_error = None
    last_query = None

    # Build initial prompt
    initial_prompt = f"{schema_info}\n\nQuestion: {question}"
    if similar_tools_context:
        initial_prompt = f"{initial_prompt}\n\n{similar_tools_context}"

    # Format system prompt with project root
    if project_root is None:
        import os
        project_root = os.environ.get("RETER_PROJECT_ROOT", os.getcwd())
    system_prompt = REQL_SYSTEM_PROMPT_TEMPLATE.format(project_root=project_root)

    # Build agent options once for the session
    options = _build_agent_options(system_prompt, max_iterations * 3, reter_instance, rag_manager)

    debug_log.debug(f"Starting REQL generation session with max_iterations={max_iterations}")

    try:
        async with ClaudeSDKClient(options=options) as client:
            # Send initial query
            await client.query(initial_prompt)
            response_text = await _process_agent_response(client, tools_used)
            attempts += 1
            debug_log.debug(f"\n{'='*60}\nREQL ITERATION {attempts}/{max_iterations}\n{'='*60}")
            debug_log.debug(f"Agent response: {response_text[:500]}...")

            for iteration in range(max_iterations):
                # Try to extract query from response
                query = _extract_query(response_text, QueryType.REQL)

                if not query:
                    # No query found, ask again within same session
                    debug_log.debug("No query found in response, asking again")
                    await client.query("Please output the REQL query in a ```reql code block.")
                    response_text = await _process_agent_response(client, tools_used)
                    attempts += 1
                    debug_log.debug(f"Retry response: {response_text[:500]}...")
                    continue

                last_query = query
                debug_log.debug(f"Generated query: {query}")

                # Execute and validate
                try:
                    result = reter_instance.reql(query)
                    row_count = result.num_rows
                    debug_log.debug(f"Query executed successfully: {row_count} rows")

                    return QueryGenerationResult(
                        success=True,
                        query=query,
                        tools_used=tools_used,
                        attempts=attempts
                    )

                except Exception as e:
                    last_error = str(e)
                    debug_log.debug(f"Query execution error: {last_error}")

                    # Send error feedback within same session
                    error_feedback = f"""Your query failed with error:
{last_error}

Previous query:
```reql
{query}
```

Please fix the query and output the corrected version in a ```reql code block."""

                    await client.query(error_feedback)
                    response_text = await _process_agent_response(client, tools_used)
                    attempts += 1
                    debug_log.debug(f"\n{'='*60}\nREQL RETRY {attempts}/{max_iterations}\n{'='*60}")
                    debug_log.debug(f"Retry response: {response_text[:500]}...")

            # Max iterations reached
            return QueryGenerationResult(
                success=False,
                query=last_query,
                tools_used=tools_used,
                attempts=attempts,
                error=f"Max iterations reached. Last error: {last_error}"
            )

    except Exception as e:
        debug_log.debug(f"Agent SDK error: {e}")
        return QueryGenerationResult(
            success=False,
            query=last_query,
            tools_used=tools_used,
            attempts=attempts,
            error=str(e)
        )


async def generate_cadsl_query(
    question: str,
    schema_info: str,
    max_iterations: int = 5,
    similar_tools_context: Optional[str] = None,
    reter_instance=None,
    rag_manager=None,
    project_root: Optional[str] = None
) -> QueryGenerationResult:
    """
    Generate a CADSL query using Agent SDK.

    Uses a SINGLE agentic session for all retries, maintaining conversation context.

    Orchestration loop (within one session):
    1. Send question to Agent with similar tools context
    2. Agent uses tools to explore grammars/examples
    3. If Agent outputs query, return it for caller to execute
    4. If no query found, send follow-up message asking for cadsl block
    5. Repeat until success or max iterations
    """
    if not is_agent_sdk_available():
        return QueryGenerationResult(
            success=False,
            query=None,
            tools_used=[],
            attempts=0,
            error="Claude Agent SDK not available"
        )

    from claude_agent_sdk import ClaudeSDKClient

    tools_used = []
    attempts = 0

    # Build initial prompt
    initial_prompt = f"Question: {question}"
    if similar_tools_context:
        initial_prompt = f"{initial_prompt}\n\n{similar_tools_context}"

    # Format system prompt with project root
    if project_root is None:
        import os
        project_root = os.environ.get("RETER_PROJECT_ROOT", os.getcwd())
    system_prompt = CADSL_SYSTEM_PROMPT_TEMPLATE.format(project_root=project_root)

    # Build agent options once for the session
    options = _build_agent_options(system_prompt, max_iterations * 3, reter_instance, rag_manager)

    debug_log.debug(f"Starting CADSL generation session with max_iterations={max_iterations}")

    try:
        async with ClaudeSDKClient(options=options) as client:
            # Send initial query
            await client.query(initial_prompt)
            response_text = await _process_agent_response(client, tools_used)
            attempts += 1
            debug_log.debug(f"\n{'='*60}\nCADSL ITERATION {attempts}/{max_iterations}\n{'='*60}")
            debug_log.debug(f"Agent response: {response_text[:500]}...")

            for iteration in range(max_iterations):
                # Check for resource requests (legacy pattern - now handled by tools)
                requests = _parse_requests(response_text)
                if requests:
                    debug_log.debug(f"Agent requested: {requests}")
                    tools_used.extend([r["type"] for r in requests])

                    resources = _handle_requests(requests)
                    await client.query(f"Here are the requested resources:\n\n{resources}\n\nNow generate the CADSL query for: {question}")
                    response_text = await _process_agent_response(client, tools_used)
                    attempts += 1
                    debug_log.debug(f"Retry response: {response_text[:500]}...")
                    continue

                # Try to extract query
                query = _extract_query(response_text, QueryType.CADSL)

                if not query:
                    # No query found, ask again within same session
                    debug_log.debug("No query found in response, asking again")
                    await client.query("Please output the CADSL query in a ```cadsl code block.")
                    response_text = await _process_agent_response(client, tools_used)
                    attempts += 1
                    debug_log.debug(f"Retry response: {response_text[:500]}...")
                    continue

                debug_log.debug(f"Generated query: {query}")

                # For CADSL, we return the query and let the caller execute/validate
                return QueryGenerationResult(
                    success=True,
                    query=query,
                    tools_used=tools_used,
                    attempts=attempts
                )

            # Max iterations reached
            return QueryGenerationResult(
                success=False,
                query=None,
                tools_used=tools_used,
                attempts=attempts,
                error="Max iterations reached"
            )

    except Exception as e:
        debug_log.debug(f"Agent SDK error: {e}")
        return QueryGenerationResult(
            success=False,
            query=None,
            tools_used=tools_used,
            attempts=attempts,
            error=str(e)
        )


async def retry_cadsl_query(
    question: str,
    previous_query: str,
    result_count: int,
    error_message: Optional[str] = None,
    reter_instance=None,
    rag_manager=None
) -> QueryGenerationResult:
    """
    Ask agent to retry a CADSL query after empty results or error.

    Returns:
        QueryGenerationResult with either:
        - A new query to try (success=True, query=<new query>)
        - Confirmation that empty is correct (success=True, query=None, error="CONFIRM_EMPTY")
        - Failure (success=False)
    """
    if not is_agent_sdk_available():
        return QueryGenerationResult(
            success=False,
            query=None,
            tools_used=[],
            attempts=0,
            error="Claude Agent SDK not available"
        )

    # Build retry prompt
    if error_message:
        result_status = f"an ERROR: {error_message}"
        error_info = f"Error details: {error_message}"
    else:
        result_status = "0 results (empty)"
        error_info = "The query executed successfully but found no matching data."

    prompt = CADSL_RETRY_PROMPT.format(
        result_status=result_status,
        previous_query=previous_query,
        error_info=error_info
    )
    prompt = f"Original question: {question}\n\n{prompt}"

    debug_log.debug(f"\n{'='*60}\nCADSL RETRY REQUEST\n{'='*60}")
    debug_log.debug(f"Previous query returned: {result_status}")

    # Format system prompt with project root (same as main function)
    import os
    project_root = os.environ.get("RETER_PROJECT_ROOT", os.getcwd())
    system_prompt = CADSL_SYSTEM_PROMPT_TEMPLATE.format(project_root=project_root)

    try:
        response_text = await _call_agent(prompt, system_prompt, reter_instance=reter_instance, rag_manager=rag_manager)
        debug_log.debug(f"Retry agent response: {response_text[:500]}...")

        # Check if agent confirms empty is correct
        if "CONFIRM_EMPTY" in response_text.upper():
            debug_log.debug("Agent confirmed empty results are correct")
            return QueryGenerationResult(
                success=True,
                query=None,
                tools_used=[],
                attempts=1,
                error="CONFIRM_EMPTY"  # Special marker
            )

        # Try to extract new query
        query = _extract_query(response_text, QueryType.CADSL)
        if query:
            debug_log.debug(f"Agent provided new query: {query[:200]}...")
            return QueryGenerationResult(
                success=True,
                query=query,
                tools_used=[],
                attempts=1
            )

        # No query extracted
        debug_log.debug("Agent did not provide a new query or CONFIRM_EMPTY")
        return QueryGenerationResult(
            success=False,
            query=None,
            tools_used=[],
            attempts=1,
            error="Agent did not provide a retry query"
        )

    except Exception as e:
        debug_log.debug(f"Retry agent error: {e}")
        return QueryGenerationResult(
            success=False,
            query=None,
            tools_used=[],
            attempts=1,
            error=str(e)
        )


async def classify_query(question: str) -> Dict[str, Any]:
    """Classify a question into query type using Agent SDK."""
    if not is_agent_sdk_available():
        return {"type": "reql", "confidence": 0.5, "reasoning": "Agent SDK not available, defaulting to REQL"}

    try:
        response_text = await _call_agent(
            f"Classify this code analysis question:\n\n{question}",
            CLASSIFICATION_SYSTEM_PROMPT
        )

        # Parse JSON response
        import json
        json_match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())

        return {"type": "reql", "confidence": 0.5, "reasoning": "Could not parse classification"}

    except Exception as e:
        debug_log.debug(f"Classification error: {e}")
        return {"type": "reql", "confidence": 0.5, "reasoning": f"Error: {e}"}
