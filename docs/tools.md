# Codeine MCP Tools Reference

> **Website**: https://codeine.ai | **Docs**: https://codeine.ai/docs

Codeine is an AI-powered code reasoning MCP server that provides semantic code analysis, session management, and intelligent recommendations.

---

## Table of Contents

1. [Session & Thinking](#session--thinking)
   - [session](#session)
   - [thinking](#thinking)
   - [items](#items)
   - [project](#project)
2. [Code Analysis](#code-analysis)
   - [code_inspection](#code_inspection)
   - [natural_language_query](#natural_language_query)
   - [recommender](#recommender)
   - [diagram](#diagram)
3. [Semantic Search (RAG)](#semantic-search-rag)
   - [semantic_search](#semantic_search)
   - [find_similar_clusters](#find_similar_clusters)
   - [find_duplicate_candidates](#find_duplicate_candidates)
   - [analyze_documentation_relevance](#analyze_documentation_relevance)
   - [rag_status](#rag_status)
   - [rag_reindex](#rag_reindex)
4. [Knowledge Management](#knowledge-management)
   - [add_knowledge](#add_knowledge)
   - [add_external_directory](#add_external_directory)
   - [quick_query](#quick_query)
   - [instance_manager](#instance_manager)
5. [System](#system)
   - [init_status](#init_status)
   - [initialize_project](#initialize_project)
   - [reter_info](#reter_info)

---

## Session & Thinking

### session

Session lifecycle management for reasoning chains.

**CRITICAL**: Call `action="context"` at the START of every conversation to restore your reasoning state.

#### Actions

| Action | Description |
|--------|-------------|
| `start` | Begin new session with optional goal and timeline |
| `context` | **CRITICAL** - Restore full context after compactification |
| `end` | Archive session (preserves data) |
| `clear` | Reset session (deletes all data) |

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `action` | string | **required** | One of: `start`, `context`, `end`, `clear` |
| `instance_name` | string | `"default"` | Session instance name |
| `goal` | string | `null` | Session goal (for `start` action) |
| `project_start` | string | `null` | Project start date ISO format (for `start`) |
| `project_end` | string | `null` | Project end date ISO format (for `start`) |

#### Returns

- **start**: `{session_id, goal, status, created_at}`
- **context**: `{session, design_doc, tasks, project_health, milestones, suggestions, mcp_guide}`
- **end**: `{session_id, status, summary}`
- **clear**: `{success, items_deleted}`

#### Examples

```python
# Restore context at conversation start
session(action="context")

# Start new session with goal
session(action="start", goal="Implement authentication system", project_end="2024-02-01")

# Archive session when done
session(action="end")
```

---

### thinking

Unified thinking tool with integrated operations for recording reasoning steps.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `thought` | string | **required** | Your current reasoning step |
| `thought_number` | integer | **required** | Current step number (1-indexed) |
| `total_thoughts` | integer | **required** | Estimated total steps |
| `instance_name` | string | `"default"` | Session instance name |
| `thought_type` | string | `"reasoning"` | One of: `reasoning`, `analysis`, `decision`, `planning`, `verification` |
| `section` | string | `null` | Design doc section: `context`, `goals`, `non_goals`, `design`, `alternatives`, `risks`, `implementation`, `tasks` |
| `next_thought_needed` | boolean | `true` | Whether more thoughts are needed |
| `branch_id` | string | `null` | ID for branching |
| `branch_from` | integer | `null` | Thought number to branch from |
| `is_revision` | boolean | `false` | Whether this revises a previous thought |
| `revises_thought` | integer | `null` | Which thought number is being revised |
| `needs_more_thoughts` | boolean | `false` | Signal that more analysis is needed |
| `operations` | object | `null` | Dict of operations to execute |

#### Operations

The `operations` parameter supports:

**Create Items:**
- `requirement`: `{text, risk, priority, ...}`
- `recommendation`: `{text, priority, ...}`
- `task`: `{name, start_date, duration_days, ...}`
- `milestone`: `{name, date, ...}`
- `decision`: `{text, rationale, ...}`

**Create Relations:**
- `traces`: `["REQ-001", ...]`
- `satisfies`: `["REQ-002", ...]`
- `implements`: `["TASK-001", ...]`
- `depends_on`: `["TASK-002", ...]`
- `affects`: `["module.py", ...]`

**Update Items:**
- `update_item`: `{id, fields...}`
- `update_task`: `{id, fields...}`
- `complete_task`: `"TASK-001"`

#### Examples

```python
# Design doc workflow - document context
thinking(
    thought="We need to add dark mode support to the application...",
    thought_number=1,
    total_thoughts=6,
    section="context"
)

# Define goals
thinking(
    thought="Goals: 1) Support system/light/dark themes 2) Persist preference",
    thought_number=2,
    total_thoughts=6,
    section="goals"
)

# Document design decision
thinking(
    thought="Using CSS variables with ThemeProvider context for theming",
    thought_number=3,
    total_thoughts=6,
    section="design",
    thought_type="decision"
)

# Create tasks from design
thinking(
    thought="Breaking down the implementation into tasks...",
    thought_number=4,
    total_thoughts=6,
    section="tasks",
    operations={
        "task": {"name": "Add ThemeProvider", "category": "feature"},
        "task": {"name": "Create dark/light CSS vars", "category": "feature"}
    }
)
```

---

### items

Query and manage items (requirements, tasks, recommendations, etc.).

#### Actions

| Action | Description |
|--------|-------------|
| `list` | Query items with filters |
| `get` | Get single item by ID |
| `delete` | Delete item and relations |
| `update` | Update item fields |
| `clear` | Delete multiple items matching filters |

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `action` | string | `"list"` | One of: `list`, `get`, `delete`, `update`, `clear` |
| `instance_name` | string | `"default"` | Session instance name |
| `item_id` | string | `null` | Item ID (required for `get`/`delete`/`update`) |
| `updates` | object | `null` | Fields to update (for `update` action) |
| `item_type` | string | `null` | Filter by type: `thought`, `requirement`, `task`, etc. |
| `status` | string | `null` | Filter by status: `pending`, `in_progress`, `completed` |
| `priority` | string | `null` | Filter by priority: `critical`, `high`, `medium`, `low` |
| `phase` | string | `null` | Filter by project phase |
| `category` | string | `null` | Filter by category |
| `source_tool` | string | `null` | Filter by source tool |
| `traces_to` | string | `null` | Items that trace to this ID |
| `traced_by` | string | `null` | Items traced by this ID |
| `depends_on` | string | `null` | Items depending on this ID |
| `blocks` | string | `null` | Items blocked by this ID |
| `affects` | string | `null` | Items affecting this file/entity |
| `start_after` | string | `null` | Tasks starting after this date |
| `end_before` | string | `null` | Tasks ending before this date |
| `include_relations` | boolean | `false` | Include related items in response |
| `limit` | integer | `100` | Maximum items to return |
| `offset` | integer | `0` | Pagination offset |

#### Examples

```python
# List all pending tasks
items(action="list", item_type="task", status="pending")

# Get specific item with relations
items(action="get", item_id="REQ-001", include_relations=True)

# Update task status
items(action="update", item_id="TASK-001", updates={"status": "in_progress"})

# Clear all completed recommendations
items(action="clear", item_type="recommendation", status="completed")
```

---

### project

Project management and analytics.

#### Actions

| Action | Description |
|--------|-------------|
| `health` | Overall project status and metrics |
| `critical_path` | Tasks on critical path (zero float) |
| `overdue` | Tasks past end_date |
| `impact` | Impact analysis for task delay |

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `action` | string | **required** | One of: `health`, `critical_path`, `overdue`, `impact` |
| `instance_name` | string | `"default"` | Session instance name |
| `task_id` | string | `null` | Task ID (for `impact` action) |
| `delay_days` | integer | `null` | Number of days delay (for `impact`) |
| `start_date` | string | `null` | Start date for timeline filter |
| `end_date` | string | `null` | End date for timeline filter |

#### Returns

- **health**: `{tasks: {...}, timeline: {...}, milestones: [...], recommendations: {...}}`
- **critical_path**: `{critical_tasks: [...], total_duration}`
- **overdue**: `{overdue_tasks: [...], total_overdue}`
- **impact**: `{delayed_task: {...}, affected_tasks: [...], affected_milestones: [...]}`

---

## Code Analysis

### code_inspection

Unified code inspection tool for multi-language analysis.

#### Languages

| Value | Description |
|-------|-------------|
| `"oo"` | Language-independent (matches all) |
| `"python"` / `"py"` | Python-specific queries |
| `"javascript"` / `"js"` | JavaScript-specific queries |
| `"html"` / `"htm"` | HTML document queries |

#### Actions - Structure/Navigation

| Action | Description | Required Params |
|--------|-------------|-----------------|
| `list_modules` | List all modules in codebase | - |
| `list_classes` | List classes, optionally by module | `module` (optional) |
| `list_functions` | List top-level functions | `module` (optional) |
| `describe_class` | Get class details with methods/attributes | `target` |
| `get_docstring` | Get docstring of entity | `target` |
| `get_method_signature` | Get method signature with params | `target` |
| `get_class_hierarchy` | Get inheritance hierarchy | `target` |
| `get_package_structure` | Get package/module structure | - |

#### Actions - Search/Find

| Action | Description | Required Params |
|--------|-------------|-----------------|
| `find_usages` | Find where entity is called | `target` |
| `find_subclasses` | Find all subclasses | `target` |
| `find_callers` | Find functions that call target | `target` |
| `find_callees` | Find functions called by target | `target` |
| `find_decorators` | Find decorator usages | `target` (optional) |
| `find_tests` | Find tests for entity | `target`, `module` (optional) |

#### Actions - Analysis

| Action | Description | Params |
|--------|-------------|--------|
| `analyze_dependencies` | Module dependency graph | - |
| `get_imports` | Import dependency graph | - |
| `get_external_deps` | External (pip) dependencies | - |
| `predict_impact` | Impact of changing entity | `target` |
| `get_complexity` | Complexity metrics | - |
| `get_magic_methods` | Find dunder methods | - |
| `get_interfaces` | Find ABC implementations | `target` (optional) |
| `get_public_api` | Get public classes/functions | - |
| `get_type_hints` | Extract type annotations | - |
| `get_api_docs` | Extract API documentation | - |
| `get_exceptions` | Exception class hierarchy | - |
| `get_architecture` | High-level architecture overview | `format` |

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `action` | string | **required** | Action to perform |
| `instance_name` | string | `"default"` | RETER instance name |
| `target` | string | `null` | Target entity name |
| `module` | string | `null` | Module name filter |
| `language` | string | `"oo"` | Language filter |
| `limit` | integer | `100` | Maximum results |
| `offset` | integer | `0` | Pagination offset |
| `format` | string | `"json"` | Output format: `json`, `markdown`, `mermaid` |
| `include_methods` | boolean | `true` | Include methods in class descriptions |
| `include_attributes` | boolean | `true` | Include attributes in class descriptions |
| `include_docstrings` | boolean | `true` | Include docstrings |
| `summary_only` | boolean | `false` | Return summary only |
| `params` | object | `null` | Additional action-specific params |

#### Examples

```python
# Get architecture overview
code_inspection(action="get_architecture", format="markdown")

# Describe a class
code_inspection(action="describe_class", target="ReterWrapper")

# Find all callers of a function
code_inspection(action="find_callers", target="authenticate")

# Get complexity metrics
code_inspection(action="get_complexity", language="python")
```

---

### natural_language_query

Query code structure using natural language (translates to REQL).

**PURPOSE**: Ask questions about code structure, relationships, and patterns.

**NOT FOR**: General knowledge questions or semantic code search (use `semantic_search` instead).

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `question` | string | **required** | Natural language question |
| `instance_name` | string | `"default"` | RETER instance to query |
| `max_retries` | integer | `5` | Maximum retry attempts on syntax errors |
| `timeout` | integer | `30` | Query timeout in seconds |
| `max_results` | integer | `500` | Maximum results to return |

#### Examples

```python
# Structure questions
natural_language_query("What classes inherit from BaseTool?")
natural_language_query("Find all methods that call the save function")
natural_language_query("List modules with more than 5 classes")
natural_language_query("Which functions have the most parameters?")
natural_language_query("Find functions with magic number 100")
natural_language_query("Show string literals containing 'error'")
```

---

### recommender

Unified recommender tool for code analysis and recommendations.

#### Recommender Types

| Type | Description |
|------|-------------|
| `"refactoring"` | Code smell detection and refactoring suggestions |
| `"test_coverage"` | Test coverage analysis and gaps |
| `"documentation_maintenance"` | Documentation quality and freshness |

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `recommender_type` | string | **required** | Type of recommender |
| `detector_name` | string | `null` | Specific detector to run (if null, lists available) |
| `instance_name` | string | `"default"` | RETER instance to analyze |
| `session_instance` | string | `"default"` | Session for storing recommendations |
| `categories` | list | `null` | Filter detectors by category |
| `severities` | list | `null` | Filter by severity: `low`, `medium`, `high` |
| `detector_type` | string | `"all"` | `all`, `improving`, or `patterns` |
| `params` | object | `null` | Override detector defaults |
| `create_tasks` | boolean | `false` | Auto-create tasks from findings |
| `link_to_thought` | string | `null` | Link recommendations to thought ID |

#### Examples

```python
# List all refactoring detectors
recommender(recommender_type="refactoring")

# Run specific detector
recommender(recommender_type="refactoring", detector_name="god_class")

# Run test coverage analysis
recommender(recommender_type="test_coverage")
```

---

### diagram

Generate diagrams and visualizations.

#### Diagram Types

**Session/Project Diagrams:**

| Type | Description |
|------|-------------|
| `gantt` | Gantt chart for tasks and milestones |
| `thought_chain` | Reasoning chain with branches |
| `design_doc` | Design doc structure with sections (context, goals, design, etc.) |
| `traceability` | Task traceability matrix |

**UML/Code Diagrams:**

| Type | Description | Key Params |
|------|-------------|------------|
| `class_hierarchy` | Class inheritance hierarchy | `target` (optional root class) |
| `class_diagram` | Class diagram with methods/attributes | `classes` |
| `sequence` | Sequence diagram of method calls | `classes`, `target` (entry point) |
| `dependencies` | Module dependency graph | `target` (optional filter) |
| `call_graph` | Call graph from entry point | `target` (focus function) |
| `coupling` | Coupling/cohesion matrix | `classes` (optional) |

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `diagram_type` | string | **required** | Type of diagram |
| `instance_name` | string | `"default"` | Session/RETER instance |
| `format` | string | `"mermaid"` | Output: `mermaid`, `markdown`, `json` |
| `root_id` | string | `null` | Root item ID for tree diagrams |
| `start_date` | string | `null` | Start date filter for gantt |
| `end_date` | string | `null` | End date filter for gantt |
| `target` | string | `null` | Target entity for UML diagrams |
| `classes` | list | `null` | List of class names |
| `include_methods` | boolean | `true` | Include methods in class diagrams |
| `include_attributes` | boolean | `true` | Include attributes |
| `max_depth` | integer | `10` | Max depth for sequence/call_graph |
| `show_external` | boolean | `false` | Show external deps |
| `params` | object | `null` | Additional diagram params |

#### Examples

```python
# Design doc structure diagram
diagram(diagram_type="design_doc", format="mermaid")

# Class hierarchy diagram
diagram(diagram_type="class_hierarchy", target="BaseTool", format="mermaid")

# Call graph
diagram(diagram_type="call_graph", target="authenticate", max_depth=5)

# Gantt chart
diagram(diagram_type="gantt", start_date="2024-01-01", end_date="2024-03-01")
```

---

## Semantic Search (RAG)

### semantic_search

Search code and documentation semantically using natural language.

Uses FAISS vector similarity to find code entities and documentation sections by meaning.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | string | **required** | Natural language search query |
| `top_k` | integer | `10` | Maximum number of results |
| `entity_types` | list | `null` | Filter: `class`, `method`, `function`, `section`, `document`, `code_block` |
| `file_filter` | string | `null` | Glob pattern (e.g., `"src/**/*.py"`) |
| `include_content` | boolean | `false` | Include source code in results |
| `search_scope` | string | `"all"` | `all`, `code`, or `docs` |
| `instance_name` | string | `"default"` | RETER instance name |

#### Examples

```python
# Search for authentication code
semantic_search("authentication and JWT tokens")

# Search only methods and functions
semantic_search("error handling", entity_types=["method", "function"])

# Search documentation only
semantic_search("installation guide", search_scope="docs")

# Search in specific directory
semantic_search("database connection", file_filter="src/db/**")
```

---

### find_similar_clusters

Find clusters of semantically similar code using K-means clustering.

Useful for finding duplicated code, similar implementations, and patterns of code reuse.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_clusters` | integer | `50` | Number of clusters (auto-adjusted) |
| `min_cluster_size` | integer | `2` | Minimum members per cluster |
| `exclude_same_file` | boolean | `true` | Exclude same-file clusters |
| `exclude_same_class` | boolean | `true` | Exclude same-class clusters |
| `entity_types` | list | `null` | Filter: `class`, `method`, `function` |
| `source_type` | string | `null` | Filter: `python` or `markdown` |
| `instance_name` | string | `"default"` | RETER instance name |

---

### find_duplicate_candidates

Find pairs of code entities that are highly similar (potential duplicates).

More precise than clustering for finding exact duplicates.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `similarity_threshold` | float | `0.85` | Minimum similarity (0-1) |
| `max_results` | integer | `50` | Maximum pairs to return |
| `exclude_same_file` | boolean | `true` | Exclude same-file pairs |
| `exclude_same_class` | boolean | `true` | Exclude same-class pairs |
| `entity_types` | list | `null` | Filter: `method`, `function` |
| `instance_name` | string | `"default"` | RETER instance name |

---

### analyze_documentation_relevance

Analyze how relevant documentation is to actual code.

Detects orphaned/outdated docs and measures documentation coverage.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `min_relevance` | float | `0.5` | Minimum similarity to consider "relevant" |
| `max_results` | integer | `100` | Maximum doc chunks to analyze |
| `doc_types` | list | `null` | Doc types: `section`, `code_block`, `document` |
| `code_types` | list | `null` | Code types: `class`, `method`, `function` |
| `instance_name` | string | `"default"` | RETER instance name |

---

### rag_status

Get RAG index status and statistics.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `instance_name` | string | `"default"` | RETER instance name |

#### Returns

```json
{
    "status": "ready",
    "embedding_model": "sentence-transformers/all-mpnet-base-v2",
    "embedding_dimension": 768,
    "total_vectors": 1823,
    "index_size_mb": 2.4,
    "python_sources": {"files_indexed": 45, "total_vectors": 1523},
    "markdown_sources": {"files_indexed": 12, "total_vectors": 300},
    "entity_counts": {"class": 89, "method": 1203, "function": 231}
}
```

---

### rag_reindex

Trigger RAG index rebuild.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `force` | boolean | `false` | Force complete rebuild |
| `instance_name` | string | `"default"` | RETER instance name |

---

## Knowledge Management

### add_knowledge

Incrementally add knowledge to RETER's semantic memory.

RETER is an incremental forward-chaining reasoner - knowledge accumulates!

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `instance_name` | string | **required** | RETER instance name |
| `source` | string | **required** | Ontology content, file path, or code |
| `type` | string | `"ontology"` | `ontology`, `python`, `javascript`, `html`, `csharp`, `cpp` |
| `source_id` | string | `null` | Identifier for selective forgetting |

#### Special Instance

The `"default"` instance auto-syncs with `RETER_PROJECT_ROOT`. Files are automatically loaded/reloaded/forgotten based on MD5 changes.

---

### add_external_directory

Load external code files from a directory into a named RETER instance.

**IMPORTANT**: Cannot use `"default"` instance - it auto-syncs with project root.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `instance_name` | string | **required** | Instance name (NOT `"default"`) |
| `directory` | string | **required** | Path to external code directory |
| `recursive` | boolean | `true` | Recursively search subdirectories |
| `exclude_patterns` | list | `null` | Glob patterns to exclude |

#### Supported Languages

- Python: `.py`
- JavaScript: `.js`, `.mjs`, `.jsx`, `.ts`, `.tsx`
- HTML: `.html`, `.htm`
- C#: `.cs`
- C++: `.cpp`, `.cc`, `.cxx`, `.hpp`, `.h`

---

### quick_query

Execute a quick REQL query outside of reasoning flow.

**NOTE**: Prefer `natural_language_query` for most use cases!

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `instance_name` | string | **required** | RETER instance name |
| `query` | string | **required** | Query in REQL syntax |
| `type` | string | `"reql"` | `reql`, `dl`, or `pattern` |

---

### instance_manager

Manage RETER instances and sources.

#### Actions

| Action | Description |
|--------|-------------|
| `list` | List all RETER instances |
| `list_sources` | List sources in instance |
| `get_facts` | Get fact IDs for source |
| `forget` | Remove facts from source |
| `reload` | Reload modified sources |
| `check` | Consistency check |

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `action` | string | **required** | Action to perform |
| `instance_name` | string | `"default"` | RETER instance name |
| `source` | string | `null` | Source ID or file path |

---

## System

### init_status

Get initialization and sync status of the default RETER instance.

This tool is ALWAYS available, even during initialization.

#### Returns

```json
{
    "is_ready": true,
    "blocking_reason": null,
    "init": {"status": "ready", "phase": "complete"},
    "sync": {"status": "idle"}
}
```

---

### initialize_project

Initialize or re-initialize the default project instance.

Runs as a background task, loading Python files and building RAG index.

---

### reter_info

Get version and diagnostic information about RETER components.

Returns versions for:
- MCP server
- Python reter package
- C++ RETE binding (owl_rete_cpp)

---

## Quick Reference

### Essential Workflow - Design Docs Approach

```python
# 1. Restore context at start
session(action="context")

# 2. Document context/problem
thinking(thought="We need to...", thought_number=1, total_thoughts=6, section="context")

# 3. Define goals
thinking(thought="Goals: 1) X 2) Y", thought_number=2, total_thoughts=6, section="goals")

# 4. Analyze code
code_inspection(action="get_architecture", format="markdown")

# 5. Document design decision
thinking(thought="Will use pattern X because...", thought_number=3, total_thoughts=6, section="design")

# 6. Create tasks
thinking(thought="Tasks: ...", thought_number=4, total_thoughts=6, section="tasks",
         operations={"task": {"name": "Implement X", "category": "feature"}})

# 7. Visualize design doc
diagram(diagram_type="design_doc", format="mermaid")
```

### Common Queries

```python
# Find all classes
code_inspection(action="list_classes")

# Describe a class
code_inspection(action="describe_class", target="MyClass")

# Find who calls a function
code_inspection(action="find_callers", target="my_function")

# Natural language query
natural_language_query("What classes inherit from ABC?")

# Semantic search
semantic_search("error handling", entity_types=["method"])
```
