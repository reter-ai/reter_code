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
    """Types of queries the generator can handle."""
    REQL = "reql"
    CADSL = "cadsl"


@dataclass
class QueryGenerationResult:
    """Result of query generation."""
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
- Use `type` predicate for entity types: `?x type oo:Class`
- Use `oo:` prefix ONLY for types (oo:Class, oo:Method, oo:Function)
- Predicates have NO prefix: `name`, `inFile`, `definedIn`, `calls`, `inheritsFrom`
- FILTER requires parentheses: `FILTER(?count > 5)`
- Patterns separated by dots: `?x type oo:Class . ?x name ?n`
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

REQL_SYSTEM_PROMPT = """You are a REQL query generator. Generate valid REQL queries for code analysis.

## AVAILABLE TOOLS
- `search_examples` - Find similar CADSL examples (many contain REQL blocks you can reference)
- `get_example` - Get full code. Use 'category/name' format (e.g., 'smells/god_class')
- `run_reql` - Test your REQL query before final output

## ENTITY TYPES (oo: prefix)
- oo:Class, oo:Method, oo:Function, oo:Module, oo:Import

## COMMON PREDICATES (NO prefix)
- name, inFile, atLine, definedIn, inheritsFrom, maybeCalls, imports

## SYNTAX RULES
1. Type patterns: `?x type oo:Class`
2. FILTER needs parentheses: `FILTER(?count > 5)`
3. Patterns separated by dots: `?x type oo:Class . ?x name ?n`

## SEMANTIC MAPPING (use REGEX for these concepts)
- "entry points" -> FILTER(REGEX(?name, "main|run|start|serve|execute|app|handle", "i"))
- "services/handlers" -> FILTER(REGEX(?name, "Service|Handler|Controller|API|Manager", "i"))
- "interactions/calls" -> use `maybeCalls` or `imports` predicates

## TYPE vs CONCEPT
- `?x type oo:Class` - Filter with subsumption (matches py:Class, cpp:Class, etc.)
- `?x type ?t` - Returns ALL types (asserted + inferred) - MULTIPLE rows per entity
- `?x concept ?t` - Returns ONLY asserted type - ONE row (e.g., "py:Method")

## UNION RULES (CRITICAL)
ALL UNION arms MUST bind the EXACT SAME variables!
Use BIND(null AS ?var) to ensure all arms have the same variables.

## OUTPUT FORMAT - CRITICAL
Your ONLY job is to generate a REQL query. You MUST output the query in a ```reql code block.
Do NOT write descriptions, summaries, or explanations - ONLY output the query.
Do NOT answer the question yourself - generate a query that will answer it.
"""

CADSL_SYSTEM_PROMPT = """You are a CADSL query generator. Your ONLY job is to generate valid CADSL pipelines.

## AVAILABLE TOOLS

**Example Tools (use if unsure about syntax):**
- `search_examples` - Find similar CADSL examples by description. Returns ranked list with scores.
- `get_example` - Get full working CADSL code. Use 'category/name' format (e.g., 'smells/god_class').
- `list_examples` - Browse all examples by category.

**Testing Tools:**
- `run_reql` - Test REQL queries to verify data exists
- `run_rag_search` - Test semantic search
- `run_rag_duplicates` - Test duplicate detection
- `run_rag_clusters` - Test clustering

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

## CRITICAL SYNTAX RULES
1. REQL blocks do NOT support # comments - use NO comments inside reql {}
2. Use REGEX() not MATCHES: `FILTER(REGEX(?name, "pattern", "i"))`
3. Entity types use `oo:` prefix: `oo:Class`, `oo:Method`
4. Predicates have NO prefix: `name`, `inFile`, `definedIn`
5. Pipeline steps use `|` operator
6. **NEVER use `when { {param} == true }`** - use simple filter steps instead

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

    @tool("search_examples", "Search for CADSL examples similar to your question using semantic similarity. Returns ranked list with scores. Use get_example to fetch full code. Helpful if unsure about syntax.", {"query": str, "max_results": int})
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

    tools = [get_grammar_tool, list_examples_tool, search_examples_tool, get_example_tool]
    if reter_instance is not None:
        tools.append(run_reql_tool)
    if rag_manager is not None:
        tools.append(run_rag_search_tool)
        tools.append(run_rag_duplicates_tool)
        tools.append(run_rag_clusters_tool)

    return create_sdk_mcp_server(
        name="query_helpers",
        version="1.0.0",
        tools=tools
    )


async def _call_agent(prompt: str, system_prompt: str, max_turns: int = 15, reter_instance=None, rag_manager=None) -> str:
    """Call Agent SDK using ClaudeSDKClient and return the text response.

    Args:
        prompt: The prompt to send to the agent
        system_prompt: System prompt for the agent
        max_turns: Maximum conversation turns (default 15)
        reter_instance: Optional RETER instance for running test REQL queries
        rag_manager: Optional RAG manager for running semantic search
    """
    if not is_agent_sdk_available():
        raise ImportError("Claude Agent SDK not installed. Run: pip install claude-agent-sdk")

    from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions, AssistantMessage, TextBlock, ToolUseBlock, ToolResultBlock

    result_text = ""
    tools_used = []

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
    if rag_manager is not None:
        base_tools.append("mcp__query_helpers__run_rag_search")
        base_tools.append("mcp__query_helpers__run_rag_duplicates")
        base_tools.append("mcp__query_helpers__run_rag_clusters")

    # Only use custom MCP tools - disable Read, Grep, Glob
    allowed_tools = base_tools
    disallowed_tools = ["Read", "Grep", "Glob", "Bash", "Edit", "Write", "WebSearch", "WebFetch"]
    permission_mode = "bypassPermissions"

    options = ClaudeAgentOptions(
        system_prompt=system_prompt,
        allowed_tools=allowed_tools,
        disallowed_tools=disallowed_tools,
        permission_mode=permission_mode,
        max_turns=max_turns,
        mcp_servers={"query_helpers": query_tools_server}
    )

    debug_log.debug(f"Starting agent with allowed_tools: {allowed_tools}, disallowed_tools: {disallowed_tools}")

    async with ClaudeSDKClient(options=options) as client:
        await client.query(prompt)

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

    debug_log.debug(f"Agent finished. Tools used: {tools_used}")
    return result_text


async def generate_reql_query(
    question: str,
    schema_info: str,
    reter_instance,
    max_iterations: int = 5,
    similar_tools_context: Optional[str] = None,
    rag_manager=None
) -> QueryGenerationResult:
    """
    Generate and validate a REQL query using Agent SDK.

    Orchestration loop:
    1. Send question to Agent
    2. If Agent requests resources, provide them
    3. If Agent outputs query, execute and validate
    4. If error, feed back to Agent
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

    tools_used = []
    attempts = 0
    last_error = None
    last_query = None

    # Build initial prompt
    prompt = f"{schema_info}\n\nQuestion: {question}"
    if similar_tools_context:
        prompt = f"{prompt}\n\n{similar_tools_context}"

    for iteration in range(max_iterations):
        attempts += 1
        debug_log.debug(f"\n{'='*60}\nREQL ITERATION {iteration + 1}/{max_iterations}\n{'='*60}")

        # If we have an error from previous iteration, add it to prompt
        if last_error and last_query:
            prompt = f"""Previous query failed with error:
{last_error}

Previous query:
```reql
{last_query}
```

Please fix the query. Original question: {question}

{schema_info}"""

        try:
            # Call Agent SDK
            response_text = await _call_agent(prompt, REQL_SYSTEM_PROMPT, reter_instance=reter_instance, rag_manager=rag_manager)
            debug_log.debug(f"Agent response: {response_text[:500]}...")

            # Check for resource requests
            requests = _parse_requests(response_text)
            if requests:
                debug_log.debug(f"Agent requested: {requests}")
                tools_used.extend([r["type"] for r in requests])

                # Provide requested resources and continue
                resources = _handle_requests(requests)
                prompt = f"Here are the requested resources:\n\n{resources}\n\nNow generate the REQL query for: {question}"
                continue

            # Try to extract query
            query = _extract_query(response_text, QueryType.REQL)
            if not query:
                debug_log.debug("No query found in response, asking again")
                prompt = f"Please output the REQL query in a ```reql code block. Question: {question}"
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
                # Continue to next iteration with error feedback

        except Exception as e:
            debug_log.debug(f"Agent SDK error: {e}")
            return QueryGenerationResult(
                success=False,
                query=last_query,
                tools_used=tools_used,
                attempts=attempts,
                error=str(e)
            )

    # Max iterations reached
    return QueryGenerationResult(
        success=False,
        query=last_query,
        tools_used=tools_used,
        attempts=attempts,
        error=f"Max iterations reached. Last error: {last_error}"
    )


async def generate_cadsl_query(
    question: str,
    schema_info: str,
    max_iterations: int = 5,
    similar_tools_context: Optional[str] = None,
    reter_instance=None,
    rag_manager=None
) -> QueryGenerationResult:
    """Generate a CADSL query using Agent SDK."""
    if not is_agent_sdk_available():
        return QueryGenerationResult(
            success=False,
            query=None,
            tools_used=[],
            attempts=0,
            error="Claude Agent SDK not available"
        )

    tools_used = []
    attempts = 0

    # Build initial prompt
    prompt = f"Question: {question}"
    if similar_tools_context:
        prompt = f"{prompt}\n\n{similar_tools_context}"

    for iteration in range(max_iterations):
        attempts += 1
        debug_log.debug(f"\n{'='*60}\nCADSL ITERATION {iteration + 1}/{max_iterations}\n{'='*60}")

        try:
            response_text = await _call_agent(prompt, CADSL_SYSTEM_PROMPT, reter_instance=reter_instance, rag_manager=rag_manager)
            debug_log.debug(f"Agent response: {response_text[:500]}...")

            # Check for resource requests
            requests = _parse_requests(response_text)
            if requests:
                debug_log.debug(f"Agent requested: {requests}")
                tools_used.extend([r["type"] for r in requests])

                resources = _handle_requests(requests)
                prompt = f"Here are the requested resources:\n\n{resources}\n\nNow generate the CADSL query for: {question}"
                continue

            # Try to extract query
            query = _extract_query(response_text, QueryType.CADSL)
            if not query:
                prompt = f"Please output the CADSL query in a ```cadsl code block. Question: {question}"
                continue

            debug_log.debug(f"Generated query: {query}")

            # For CADSL, we return the query and let the caller execute/validate
            return QueryGenerationResult(
                success=True,
                query=query,
                tools_used=tools_used,
                attempts=attempts
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

    return QueryGenerationResult(
        success=False,
        query=None,
        tools_used=tools_used,
        attempts=attempts,
        error="Max iterations reached"
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

    try:
        response_text = await _call_agent(prompt, CADSL_SYSTEM_PROMPT, reter_instance=reter_instance, rag_manager=rag_manager)
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
