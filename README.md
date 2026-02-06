# Reter Code

Architectural visibility for AI coding agents.

The name comes from the **RETE algorithm** — the forward-chaining pattern matching engine at its core. RETER *RETEs your code*: it feeds source files through a RETE network of 70+ OWL 2 RL inference rules and what comes out the other end is a formal code ontology.

## The Problem

AI coding agents are **architecturally blind**. They see files and functions, but not the architecture — the class hierarchies, dependency graphs, layer boundaries, and patterns that hold a codebase together. Without that visibility:

- **They duplicate code** that already exists three directories away
- **They reinvent patterns** instead of following the ones already established in the codebase
- **They cross architectural layers** — calling infrastructure from presentation, mixing concerns
- **They create god classes** — adding methods to whatever file is open rather than finding the right home
- **They can't detect redundancy** — similar code accumulates across files with no one noticing

Every task gets solved **locally**. The code works, but the architecture erodes with every change. Over time you get spaghetti — tangled dependencies, scattered duplication, and designs that no one can reason about.

**The root cause is simple: agents operate on text, not on structure.** They have no model of the codebase as a whole.

## The Solution

Reter Code gives AI agents **architectural visibility** by building a **code ontology** — a formal semantic model of the entire codebase — and exposing it through the [Model Context Protocol](https://modelcontextprotocol.io).

The ontology is built with OWL 2 RL Description Logic and enhanced with **RAG embeddings** and **ML-based clustering**, so agents can ask questions like *"does a utility for this already exist?"*, *"what layer is this class in?"*, *"is there similar code elsewhere?"* — before writing a single line.

### What it does

1. **Builds a code ontology** — parses source code into a semantic knowledge graph (classes, methods, calls, inheritance, dependencies) using language-specific C++ parsers and OWL 2 RL reasoning with 70+ inference rules
2. **Enhances with RAG and ML** — indexes code into FAISS vector embeddings for semantic similarity search, pairwise duplicate detection, and DBSCAN density-based clustering to find redundancy that structural analysis alone misses
3. **Pipelines batch analysis** — CADSL pipelines chain ontology queries with ML-powered enrichment steps to detect code smells, find duplicates, identify extraction opportunities, and generate refactoring task lists across the entire codebase
4. **Synthesizes new queries on demand** — a Claude subagent translates natural language questions into CADSL/REQL queries at runtime, with access to the schema, 130 examples, and verification tools
5. **Feeds results** to AI agents for automated or assisted refactoring

```
                    RETER Server (separate process)
                    ┌─────────────────────────────────────────────┐
Source Code ───────►│ Parsers → Code Ontology (OWL 2 RL) ─┐      │
                    │                                      ├→ Results ──► ZeroMQ ──► MCP Client ──► AI Agent
                    │          RAG Embeddings (FAISS/ML) ──┘      │
                    │          CADSL Pipelines                     │
                    └─────────────────────────────────────────────┘
```

## Architecture

Reter Code runs as **two separate processes** connected via ZeroMQ:

```
┌──────────────────────────────────┐     ZeroMQ      ┌──────────────────────────────────┐
│  RETER Server (reter_server)     │   REQ/REP       │  MCP Client (reter_code)         │
│                                  │   tcp://         │                                  │
│  • Holds RETE network + RAG      │◄────────────────►│  • Stateless FastMCP proxy        │
│  • Builds code ontology          │   127.0.0.1:5555 │  • Registers MCP tools            │
│  • Processes all queries         │   msgpack        │  • Runs inside Claude             │
│  • Runs in separate console      │                  │  • Forwards requests to server    │
└──────────────────────────────────┘                  └──────────────────────────────────┘
```

The **server** is stateful — it initializes the C++ RETE engine, loads the codebase into the ontology, builds RAG embeddings, and processes all queries. The **MCP client** is stateless — it registers tools with Claude via FastMCP and forwards every request to the server over ZeroMQ.

This separation means the expensive RETE network and RAG index stay alive across Claude sessions, and multiple MCP clients can connect to the same server.

| Component | Command | Description |
|-----------|---------|-------------|
| RETER Server | `reter_server` | Stateful server: RETE network, RAG, CADSL pipelines, query processing |
| MCP Client | `reter_code` | Stateless FastMCP proxy: tool registration, Claude integration |
| Python API | `reter` | `Reter` class, query result sets, CLI |
| C++ Engine | `reter_core` | RETE network, OWL RL rules, language parsers (closed source) |

## Installation

### Step 1: Install

```bash
pip install reter_code
```

Or with uvx:

```bash
uvx --from git+https://github.com/reter-ai/reter_code --find-links https://raw.githubusercontent.com/reter-ai/reter/main/reter_core/index.html reter_code
```

### Step 2: Start the RETER Server

In a **separate terminal**, start the server on your project:

```bash
reter_server --project /path/to/your/project
```

The server will:
1. Initialize the RETE network and load your codebase into the ontology
2. Build RAG embeddings (FAISS index)
3. Bind ZeroMQ on `tcp://127.0.0.1:5555`
4. Write a discovery file at `.reter_code/server.json`
5. Display a console UI showing status and queries

Keep this terminal open — the server must be running for the MCP client to work.

**Server options:**

```bash
reter_server --project /path/to/project          # Default port 5555
reter_server --project /path/to/project --port 6000  # Custom port
reter_server --project /path/to/project --no-console # No rich UI
reter_server --project /path/to/project --verbose    # Debug logging
```

### Step 3: Add MCP Client to Claude Code

```bash
claude mcp add reter_code -s user -- uvx --from git+https://github.com/reter-ai/reter_code --find-links https://raw.githubusercontent.com/reter-ai/reter/main/reter_core/index.html reter_code
```

The MCP client automatically discovers the server via `.reter_code/server.json`.

---

## Configure with Claude Desktop

**Config file location:**
- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
- **Linux**: `~/.config/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "reter_code": {
      "command": "uvx",
      "args": [
        "--from", "git+https://github.com/reter-ai/reter_code",
        "--find-links", "https://raw.githubusercontent.com/reter-ai/reter/main/reter_core/index.html",
        "reter_code"
      ],
      "timeout": 120000
    }
  }
}
```

**Important:** The RETER server must be running before starting Claude. The MCP client connects to it via ZeroMQ.

---

## Environment Variables

### Server

| Variable | Description | Default |
|----------|-------------|---------|
| `RETER_PROJECT_ROOT` | Path to project for auto-loading code | Auto-detected from CWD |
| `RETER_HOST` | Server bind address | `127.0.0.1` |
| `RETER_QUERY_PORT` | ZeroMQ query port | `5555` |

### MCP Client

| Variable | Description | Default |
|----------|-------------|---------|
| `RETER_PROJECT_ROOT` | Project path for server discovery | Auto-detected from CWD |
| `RETER_SERVER_HOST` | Server host (fallback if no discovery) | `127.0.0.1` |
| `RETER_SERVER_QUERY_PORT` | Server port (fallback if no discovery) | `5555` |

---

## MCP Tools

### Refactoring Pipeline

| Tool | Description |
|------|-------------|
| `recommender` | Generate refactoring and redundancy reduction recommendations |
| `execute_cadsl` | Run CADSL pipelines — the primary tool for batch codebase processing |
| `natural_language_query` | Ask refactoring questions in plain English — translates to CADSL automatically |
| `reql` | Execute REQL queries directly against the knowledge graph |
| `semantic_search` | Find semantically similar code using vector similarity (FAISS) |
| `code_inspection` | Multi-language code analysis (Python, JS, C#, C++) |

### Session and Tracking

| Tool | Description |
|------|-------------|
| `session` | Session lifecycle — **call `context` first to restore state** |
| `thinking` | Record reasoning with design doc sections (context, goals, design, tasks) |
| `items` | Track refactoring tasks, milestones, and progress |
| `diagram` | Visualize class hierarchy, dependencies, call graphs |
| `system` | Initialization status, source management, reindexing |

---

## Refactoring Workflow

Reter Code follows a structured **detect → review → implement** workflow:

### Phase 1: Detect

Run CADSL pipelines to scan the entire codebase and create review tasks:

```python
# Find redundant code across files (dry run first)
execute_cadsl(
    script="cadsl/tools/good/redundant_code_tasks.cadsl",
    params={"min_similarity": 0.75, "limit": 50, "dry_run": true}
)

# Find extraction opportunities using DBSCAN clustering
execute_cadsl(
    script="cadsl/tools/good/extraction_opportunities_dbscan.cadsl",
    params={"eps": 0.5, "min_samples": 2, "file_filter": ".py", "dry_run": false}
)

# Find files too large for AI context windows
execute_cadsl(
    script="cadsl/tools/smells/large_file_tasks.cadsl",
    params={"max_lines": 500, "max_classes": 1, "dry_run": false}
)
```

Each detector creates **review tasks** with detailed descriptions, file locations, and classification guidance.

### Phase 2: Classify

For each task, the AI agent reads the actual code and classifies it:

- **TP-EXTRACT** — Real duplication, extract to shared function
- **TP-PARAMETERIZE** — Similar with small variations, extract with parameters
- **PARTIAL-TP** — Some extractable fragments within larger methods
- **FP-INTERFACE** — Same interface pattern, different implementations (Template Method)
- **FP-LAYERS** — Same params through architectural layers (proper layering)
- **FP-STRUCTURAL** — Similar Python idioms, different logic
- **FP-TRIVIAL** — Too short to extract

True positives automatically generate follow-up implementation tasks via `create_followup=true`.

### Phase 3: Implement

For each TP follow-up task, choose an extraction strategy:

| Situation | Strategy |
|-----------|----------|
| Same class | Extract private helper method |
| Same file | Extract module-level function |
| Related classes | Extract to base class |
| Unrelated classes | Extract to utility module |
| Same structure, different details | Template Method pattern |

---

## CADSL Pipelines (130 built-in)

CADSL (Code Analysis DSL) chains knowledge graph queries with transformation steps. A pipeline can query the graph, filter results, enrich with RAG embeddings, detect duplicates via DBSCAN clustering, fetch actual source code, and create ranked task lists — all in a single pass.

Pipeline steps include: `reql`, `rag` (duplicates/dbscan/enrich), `filter`, `select`, `map`, `join`, `order_by`, `limit`, `unique`, `fetch_content`, `rag_enrich`, `python`, `file_scan`, `render_mermaid`, `create_task`, `emit`.

130 pre-built pipelines across 12 categories:

| Category | Count | Examples |
|----------|-------|---------|
| **Refactoring** | 26 | extract method/class, move field, inline class, pull up/push down, introduce parameter object |
| **Code Smells** | 24 | god class, long method, feature envy, dead code, shotgun surgery, data class |
| **Inspection** | 21 | class hierarchy, callers/callees, API docs, complexity, impact prediction |
| **RAG (Semantic)** | 15 | cross-file duplicates, DBSCAN clusters, orphan helpers, semantic dead code |
| **Testing** | 11 | untested classes/methods, shallow tests, coverage gaps, public API untested |
| **Diagrams** | 6 | call graph, class diagram, dependency graph, sequence diagram |
| **Patterns** | 6 | singleton, factory, decorator, interface implementations |
| **Dependencies** | 3 | circular imports, unused imports, external deps |
| **Exceptions** | 5 | silent exceptions, generic raise, error codes |
| **File Search** | 5 | TODOs, debug statements, hardcoded secrets |
| **Inheritance** | 4 | collapse hierarchy, extract superclass, replace with delegate |
| **Good Practices** | 4 | architecture diagram, redundant code detection, extraction opportunities |

---

## Natural Language Queries and CADSL Synthesis

You don't need to write CADSL or REQL by hand. The `natural_language_query` tool translates plain English questions into executable queries using a **Claude subagent** with access to:

- The **live schema** of entity types and predicates from the current codebase
- The **REQL and CADSL grammars** (formal Lark specifications)
- **130 working examples** searchable by semantic similarity
- **Verification tools** — the subagent tests its generated query against the actual knowledge graph, reads source files to spot-check results, and retries on errors

The subagent workflow:

1. **Classifies** the question as REQL (structural), CADSL (pipeline), or RAG (semantic)
2. **Searches examples** — finds the closest matching CADSL tools via semantic similarity
3. **Generates a query** — synthesizes a new CADSL/REQL query using the grammar, schema, and examples as templates
4. **Tests and verifies** — executes the query via `run_cadsl`/`run_reql`, reads sample files with `Read`/`Grep` to confirm results are valid
5. **Retries on failure** — if the query errors or returns empty results, the subagent gets error feedback and generates a corrected version (up to 5 attempts)

This means any question about the codebase can become a query:

```python
natural_language_query("Find duplicate code that could be extracted into shared functions")
natural_language_query("Which classes have more than 20 methods?")
natural_language_query("Show me the call graph for the authentication module")
natural_language_query("Find untested public methods in the services layer")
```

The `generate_cadsl` tool does the same synthesis but returns the query text without executing it — useful for saving queries as reusable `.cadsl` files.

---

## Supported Languages

Code analysis is available for:

| Language | Extensions | Status |
|----------|-----------|--------|
| Python | `.py` | Full support |
| C# | `.cs` | Full support |
| C++ | `.cpp`, `.hpp`, `.h`, `.cc` | Full support |
| JavaScript | `.js` | Full support |

The C++ engine additionally includes parsers for Java, Go, Rust, Swift, PHP, Erlang, Objective-C, OCL, C, CSS, HTML, and PlantUML.

---

## How It Works

### Code Ontology

Source code is parsed into OWL facts using language-specific C++ parsers, forming a formal **code ontology** — not just a syntax tree, but a semantically reasoned model. The RETE network applies 70+ OWL 2 RL inference rules to derive implicit relationships — if class A calls class B which inherits from C, the ontology knows A depends on C even though that's never stated in source. The ontology is enhanced by RAG vector embeddings (FAISS + sentence-transformers) that capture semantic similarity between code fragments, and ML clustering (DBSCAN) that groups related code across the codebase.

### REQL (Query Language)

REQL queries the knowledge graph using SPARQL-like triple patterns:

```
SELECT ?class ?name ?method_count
WHERE {
    ?class type class .
    ?class has-name ?name .
    ?class has-method-count ?method_count .
    FILTER(?method_count > 20)
}
```

### CADSL (Pipeline Language)

CADSL chains REQL queries with transformation and detection steps. Here's a real pipeline that finds redundant code across files, filters false positives, and creates review tasks:

```
detector redundant_code_tasks(category="smell-review", severity="medium") {
    """Find redundant code and create review tasks."""

    param min_similarity: float = 0.75;
    param limit: int = 50;
    param dry_run: bool = true;

    # Step 1: RAG pairwise duplicate detection across files
    rag { duplicates, similarity: {min_similarity}, limit: 500,
          exclude_same_file: true, exclude_same_class: true }

    # Step 2: Filter known false positives (boilerplate, visitors, tests)
    | python {
        BOILERPLATE = {"__init__", "__str__", "setUp", "tearDown"}
        output = [r for r in rows
                  if r["entity1_name"] not in BOILERPLATE
                  and r["entity2_name"] not in BOILERPLATE]
        result = output
    }

    # Step 3: Reshape and rank by similarity
    | select { source_method: entity1_name, source_file: entity1_file,
               similar_method: entity2_name, similar_file: entity2_file,
               similarity: similarity }
    | order_by { -similarity }
    | limit { {limit} }

    # Step 4: Create review tasks with classification guidance
    | create_task {
        name: "[redundant] {source_method} ~ {similar_method}",
        category: "smell-review",
        description: "...",    # includes file locations, Read commands, TP/FP criteria
        dry_run: {dry_run}
    }
    | emit { tasks }
}
```

### RAG (Semantic Duplicate Detection)

A FAISS vector index built from sentence-transformers embeddings powers two detection modes:

- **Pairwise duplicates** — finds pairs of methods with similar body content across files
- **DBSCAN clustering** — groups methods by code similarity without specifying cluster count upfront, identifying natural clusters of redundant implementations

This catches duplication that structural analysis alone misses — code that does the same thing with different variable names, in different files, with different method names.

### Semantic Annotations

Classes can be annotated with `:::` CNL (Controlled Natural Language) statements that become part of the knowledge graph:

```python
class ReterWrapper:
    """
    ::: This is-in-layer Infrastructure-Layer.
    ::: This is-in-process Main-Process.
    ::: This is stateful.
    ::: This holds-expensive-resource "rete-network".
    ::: This depends-on `reter.Reter`.
    """
```

These annotations enable architectural queries — find all stateful classes, map process boundaries, detect layer violations, generate architecture diagrams.

---

## License

MIT License
